from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    cv2.setNumThreads(1)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "yolo_best.pt")
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.30
DEFAULT_MAX_DET = 30
DEFAULT_IMGSZ = 224
DEFAULT_FRAME_SKIP = 2
DEFAULT_PAD_RATIO = 0.04
DEFAULT_BOX_PAD_PIXELS = 1
DEFAULT_MIN_STABLE_FRAMES = 3
DEFAULT_MIN_AREA_RATIO = 0.0005
DEFAULT_TRACKER_NAME = "bytetrack.yaml"


def resolve_local_path(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else SCRIPT_DIR / p)


def clamp_box(box: Iterable[float], frame_shape: tuple[int, ...]) -> list[int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    return None if x2 <= x1 or y2 <= y1 else [x1, y1, x2, y2]


def expand_box(box: list[int], frame_shape: tuple[int, ...], pad_ratio: float, pad_pixels: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    px = max(int(round(max(1, x2 - x1) * pad_ratio)), pad_pixels)
    py = max(int(round(max(1, y2 - y1) * pad_ratio)), pad_pixels)
    return clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], frame_shape)


def crop_from_box(frame: np.ndarray, box: list[int]) -> np.ndarray | None:
    box = clamp_box(box, frame.shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    return None if crop.size == 0 else crop


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def crop_quality(frame: np.ndarray, box: list[int], detector_score: float) -> float:
    crop = crop_from_box(frame, box)
    if crop is None:
        return -1.0
    fh, fw = frame.shape[:2]
    ch, cw = crop.shape[:2]
    if cw < 28 or ch < 28:
        return -1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp_score = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 700.0, 1.0)
    area_score = min((cw * ch) / max(1.0, 0.20 * fw * fh), 1.0)
    aspect_score = 1.0 if 0.22 <= cw / max(1, ch) <= 7.0 else 0.45
    edge_penalty = 0.16 if box[0] <= 2 or box[1] <= 2 or box[2] >= fw - 2 or box[3] >= fh - 2 else 0.0
    return 0.42 * float(detector_score) + 0.33 * sharp_score + 0.18 * area_score + 0.07 * aspect_score - edge_penalty


def detector_preview(frame: np.ndarray, boxes: list["TrackBox"], long_side: int = 480) -> np.ndarray:
    preview = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"ID {box.track_id} {box.score:.2f}"
        cv2.putText(preview, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    h, w = preview.shape[:2]
    longest = max(h, w)
    if longest > long_side:
        scale = long_side / float(longest)
        preview = cv2.resize(preview, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return preview


@dataclass
class TrackBox:
    xyxy: list[int]
    score: float
    label: str
    cls_id: int
    track_id: int
    frame_index: int


@dataclass
class CropRecord:
    object_id: int
    label: str
    cls_id: int
    best_crop: np.ndarray
    best_box: list[int]
    best_quality: float
    best_score: float
    seen_count: int
    version: int
    first_frame: int
    last_seen_frame: int
    ready_for_ocr: bool = False

    def summary(self, include_crop: bool = True) -> dict[str, Any]:
        out = {
            "id": self.object_id,
            "track_id": self.object_id,
            "label": self.label,
            "cls_id": self.cls_id,
            "bbox": list(self.best_box),
            "quality": float(self.best_quality),
            "score": float(self.best_score),
            "seen_count": int(self.seen_count),
            "version": int(self.version),
            "first_frame": int(self.first_frame),
            "last_seen_frame": int(self.last_seen_frame),
            "ready_for_ocr": bool(self.ready_for_ocr),
        }
        if include_crop:
            out["crop"] = self.best_crop.copy()
        return out


class YOLOEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)
    poster_found_record = pyqtSignal(dict)
    finished_records = pyqtSignal(list)
    frame_preview = pyqtSignal(object)

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        box_pad: int = DEFAULT_BOX_PAD_PIXELS,
        pad_ratio: float = DEFAULT_PAD_RATIO,
        max_det: int = DEFAULT_MAX_DET,
        class_names: list[str] | tuple[str, ...] | None = None,
        tracker_name: str = DEFAULT_TRACKER_NAME,
        min_stable_frames: int = DEFAULT_MIN_STABLE_FRAMES,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
        rebuild_tracker_on_start: bool = True,
        require_track_id: bool = True,
    ):
        super().__init__()
        self.model_path = resolve_local_path(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.frame_skip = max(1, int(frame_skip))
        self.box_pad = int(box_pad)
        self.pad_ratio = float(pad_ratio)
        self.max_det = int(max_det)
        self.class_names = list(class_names) if class_names else None
        self.tracker_name = str(tracker_name)
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.min_area_ratio = float(min_area_ratio)
        self.rebuild_tracker_on_start = bool(rebuild_tracker_on_start)
        self.require_track_id = bool(require_track_id)
        self.model = None
        self.loaded_model_path: str | None = None
        self.stop_requested = False
        self.crop_records: dict[int, CropRecord] = {}
        self.detect_fps_smooth = 0.0
        self._fallback_id_next = 10_000

    def request_stop(self):
        self.stop_requested = True

    def reset_tracking(self, rebuild_model: bool = True):
        self.crop_records.clear()
        self.detect_fps_smooth = 0.0
        self._fallback_id_next = 10_000
        if rebuild_model:
            self.model = None
            self.loaded_model_path = None
        elif self.model is not None:
            try:
                setattr(self.model, "predictor", None)
            except Exception as exc:
                self.progress.emit(0, f"Tracker reset warning: {exc}")

    def load_model(self) -> bool:
        if YOLO is None:
            self.error.emit("Ultralytics YOLO is not installed. Install it with: pip install ultralytics")
            return False
        if self.model is not None and self.loaded_model_path == self.model_path:
            return True
        self.progress.emit(5, f"Loading YOLO model: {Path(self.model_path).name}")
        try:
            model = YOLO(self.model_path)
            try:
                model.fuse()
            except Exception:
                pass
        except Exception as exc:
            self.error.emit(f"Failed to load YOLO model: {exc}")
            return False
        self.model = model
        self.loaded_model_path = self.model_path
        self.progress.emit(8, f"Loaded YOLO model: {Path(self.model_path).name}")
        return True

    def allowed_class_ids(self) -> set[int] | None:
        if not self.class_names or self.model is None or not isinstance(getattr(self.model, "names", None), dict):
            return None
        wanted = {name.lower().strip() for name in self.class_names}
        ids = {int(k) for k, v in self.model.names.items() if str(v).lower().strip() in wanted}
        if not ids:
            self.progress.emit(9, f"Warning: class_names {self.class_names} did not match model.names; not filtering.")
        return ids or None

    def process_video(self, video_path: str):
        self.stop_requested = False
        self.reset_tracking(rebuild_model=self.rebuild_tracker_on_start)
        if not self.load_model():
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {video_path}")
            return
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_area = max(1, width * height)
            allowed = self.allowed_class_ids()
            frame_idx = 0
            self.progress.emit(10, "Starting video processing...")
            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.frame_skip == 0:
                    self.process_frame(frame, frame_idx, allowed, frame_area)
                frame_idx += 1
                self.emit_frame_progress(frame_idx, total)
            self.finish()
        except Exception as exc:
            self.error.emit(f"Error during YOLO processing: {exc}")
        finally:
            cap.release()

    def process_frame(self, frame: np.ndarray, frame_idx: int, allowed: set[int] | None, frame_area: int):
        start = time.time()
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_name,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            self.error.emit(f"YOLO tracking failed on frame {frame_idx}: {exc}")
            self.stop_requested = True
            return
        elapsed = time.time() - start
        if elapsed > 0:
            inst = 1.0 / elapsed
            self.detect_fps_smooth = inst if self.detect_fps_smooth == 0 else 0.85 * self.detect_fps_smooth + 0.15 * inst
        boxes = self.parse_results(results, frame_idx, allowed, frame.shape)
        self.frame_preview.emit(detector_preview(frame, boxes))
        for box in boxes:
            if box_area(box.xyxy) >= self.min_area_ratio * frame_area:
                self.update_crop_record(frame, box)

    def parse_results(self, results: Any, frame_idx: int, allowed: set[int] | None, frame_shape: tuple[int, ...]) -> list[TrackBox]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy), -1)
        ids = boxes.id.detach().cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        out: list[TrackBox] = []
        for i, raw in enumerate(xyxy):
            cls_id = int(clss[i])
            if allowed is not None and cls_id not in allowed:
                continue
            fixed = clamp_box(raw[:4], frame_shape)
            if fixed is None:
                continue
            if ids is not None:
                track_id = int(ids[i])
            elif self.require_track_id:
                continue
            else:
                track_id = self._fallback_id_next
                self._fallback_id_next += 1
            label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
            out.append(TrackBox(fixed, float(confs[i]), label, cls_id, track_id, int(frame_idx)))
        return out

    def update_crop_record(self, frame: np.ndarray, box: TrackBox) -> bool:
        expanded = expand_box(box.xyxy, frame.shape, self.pad_ratio, self.box_pad)
        if expanded is None:
            return False
        crop = crop_from_box(frame, expanded)
        if crop is None:
            return False
        quality = crop_quality(frame, expanded, box.score)
        if quality < 0.05:
            return False
        rec = self.crop_records.get(box.track_id)
        if rec is None:
            rec = CropRecord(box.track_id, box.label, box.cls_id, crop.copy(), expanded, quality, box.score, 1, 1, box.frame_index, box.frame_index, self.min_stable_frames <= 1)
            self.crop_records[box.track_id] = rec
            self.emit_record(rec)
            return True
        rec.seen_count += 1
        rec.last_seen_frame = box.frame_index
        became_ready = rec.seen_count >= self.min_stable_frames and not rec.ready_for_ocr
        if became_ready:
            rec.ready_for_ocr = True
        improved = quality > rec.best_quality + 0.035
        if improved:
            rec.best_crop = crop.copy()
            rec.best_box = expanded
            rec.best_quality = quality
            rec.best_score = box.score
            rec.label = box.label
            rec.cls_id = box.cls_id
            rec.version += 1
        if improved or became_ready:
            self.emit_record(rec)
            return True
        return False

    def emit_record(self, rec: CropRecord):
        self.poster_found.emit(int(rec.object_id), rec.best_crop.copy())
        self.poster_found_record.emit(rec.summary(include_crop=True))

    def emit_frame_progress(self, frame_idx: int, total: int):
        if total > 0:
            pct = min(94, int(10 + (frame_idx / total) * 84))
            msg = f"Frame {frame_idx}/{total} | tracks: {len(self.crop_records)} | detect FPS: {self.detect_fps_smooth:.1f}"
        else:
            pct = 50
            msg = f"Frame {frame_idx} | tracks: {len(self.crop_records)} | detect FPS: {self.detect_fps_smooth:.1f}"
        self.progress.emit(pct, msg)

    def finish(self):
        self.progress.emit(95, "Stopping and collecting best tracked crops..." if self.stop_requested else "Finishing up...")
        records = self.get_crop_records(include_crop=True)
        crops = [rec["crop"] for rec in records if isinstance(rec.get("crop"), np.ndarray) and rec["crop"].size > 0]
        self.finished_records.emit(records)
        self.finished.emit(crops)
        self.progress.emit(100, "Stopped" if self.stop_requested else f"Done — {len(crops)} best crop(s)")

    def get_crop_records(self, include_crop: bool = True, only_ready: bool = False) -> list[dict[str, Any]]:
        records = [rec.summary(include_crop) for rec in self.crop_records.values() if not only_ready or rec.ready_for_ocr]
        return sorted(records, key=lambda item: (-float(item["quality"]), int(item["id"])))


class YOLOWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)
    poster_found_record = pyqtSignal(dict)
    finished_records = pyqtSignal(list)
    frame_preview = pyqtSignal(object)

    def __init__(self, video_path: str, **engine_kwargs):
        super().__init__()
        self.video_path = str(video_path)
        self.engine = YOLOEngine(**engine_kwargs)

    def request_stop(self):
        self.engine.request_stop()

    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.poster_found_record.connect(self.poster_found_record)
        self.engine.finished_records.connect(self.finished_records)
        self.engine.frame_preview.connect(self.frame_preview)
        self.engine.process_video(self.video_path)
