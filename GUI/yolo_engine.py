

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    from ultralytics import YOLO
except ImportError:  # Keep importable even before dependencies are installed.
    YOLO = None

# Keep OpenCV from competing with Qt/YOLO for CPU threads.
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
    """Resolve relative model paths next to this engine file, not the terminal CWD."""
    p = Path(path)
    return str(p if p.is_absolute() else SCRIPT_DIR / p)

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

    def to_summary(self, include_crop: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
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


def get_centroid(bbox: list[list[int]] | list[int]) -> tuple[int, int]:
    if len(bbox) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in bbox):
        x1, y1, x2, y2 = bbox  # xyxy
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    x_coords = [p[0] for p in bbox]  # quadrilateral
    y_coords = [p[1] for p in bbox]
    return (int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords)))


def euclidean(c1: tuple[int, int], c2: tuple[int, int]) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def bbox_xyxy(bbox: list[list[int]] | list[int]) -> tuple[int, int, int, int]:
    if len(bbox) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in bbox):
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2), int(y2)
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def bbox_area(bbox: list[list[int]] | list[int]) -> int:
    x1, y1, x2, y2 = bbox_xyxy(bbox)
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(b1: list[list[int]] | list[int], b2: list[list[int]] | list[int]) -> float:
    x1a, y1a, x2a, y2a = bbox_xyxy(b1)
    x1b, y1b, x2b, y2b = bbox_xyxy(b2)

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h

    area_a = max(0, x2a - x1a) * max(0, y2a - y1a)
    area_b = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def clamp_box(box: Iterable[float], frame_shape: tuple[int, ...]) -> list[int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def xyxy_to_quad(box: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def expand_box(
    box: list[int],
    frame_shape: tuple[int, ...],
    pad_ratio: float = DEFAULT_PAD_RATIO,
    pad_pixels: int = DEFAULT_BOX_PAD_PIXELS,
) -> list[int] | None:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = max(int(round(bw * float(pad_ratio))), int(pad_pixels))
    py = max(int(round(bh * float(pad_ratio))), int(pad_pixels))
    return clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], frame_shape)


def crop_from_box(frame_bgr: np.ndarray, box: list[int]) -> np.ndarray | None:
    fixed = clamp_box(box, frame_bgr.shape)
    if fixed is None:
        return None
    x1, y1, x2, y2 = fixed
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def crop_quality_score(frame_bgr: np.ndarray, box: list[int], score: float) -> float:
    """Quality score for choosing the best crop of a tracked poster/flyer."""
    crop = crop_from_box(frame_bgr, box)
    if crop is None:
        return -1.0

    fh, fw = frame_bgr.shape[:2]
    ch, cw = crop.shape[:2]
    if cw < 28 or ch < 28:
        return -1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_score = min(float(sharp) / 700.0, 1.0)

    area_score = min((cw * ch) / max(1.0, 0.20 * fw * fh), 1.0)
    aspect = cw / max(1, ch)
    aspect_score = 1.0 if 0.22 <= aspect <= 7.0 else 0.45

    x1, y1, x2, y2 = box
    edge_penalty = 0.16 if (x1 <= 2 or y1 <= 2 or x2 >= fw - 2 or y2 >= fh - 2) else 0.0

    return 0.42 * float(score) + 0.33 * sharp_score + 0.18 * area_score + 0.07 * aspect_score - edge_penalty


def score_frame(frame: np.ndarray, bbox: list[list[int]] | list[int], detector_score: float = 0.0) -> tuple[float, np.ndarray]:
    """Compatibility wrapper returning (quality, crop) for old callers."""
    box = list(bbox_xyxy(bbox))
    fixed = clamp_box(box, frame.shape)
    if fixed is None:
        return 0.0, np.empty((0, 0, 3), dtype=frame.dtype)
    crop = crop_from_box(frame, fixed)
    if crop is None:
        return 0.0, np.empty((0, 0, 3), dtype=frame.dtype)
    return crop_quality_score(frame, fixed, detector_score), crop


def pad_bbox(bbox: list[list[int]] | list[int], frame_shape: tuple[int, ...], pad: int = 1) -> list[list[int]]:
    box = list(bbox_xyxy(bbox))
    expanded = expand_box(box, frame_shape, pad_ratio=0.0, pad_pixels=pad)
    return xyxy_to_quad(expanded) if expanded is not None else xyxy_to_quad(box)


class YOLOEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)           
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    poster_found_record = pyqtSignal(dict)
    finished_records = pyqtSignal(list)

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
        self.loaded_model_path: str | None = None
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
        self.stop_requested = False
        self.crop_records: dict[int, CropRecord] = {}
        self.detect_fps_smooth = 0.0
        self._fallback_id_next = 10_000

    def request_stop(self):
        self.stop_requested = True

    def set_params(
        self,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        frame_skip: int | None = None,
        max_det: int | None = None,
        box_pad: int | None = None,
        pad_ratio: float | None = None,
    ):
        if conf is not None:
            self.conf = float(conf)
        if iou is not None:
            self.iou = float(iou)
        if imgsz is not None:
            self.imgsz = int(imgsz)
        if frame_skip is not None:
            self.frame_skip = max(1, int(frame_skip))
        if max_det is not None:
            self.max_det = int(max_det)
        if box_pad is not None:
            self.box_pad = int(box_pad)
        if pad_ratio is not None:
            self.pad_ratio = float(pad_ratio)

    def set_model_path(self, model_path: str):
        new_path = resolve_local_path(model_path)
        if new_path != self.model_path:
            self.model_path = new_path
            self.reset_tracking(rebuild_model=True)

    def reset_tracking(self, rebuild_model: bool = True):
        """
        clear crop memory and reset YOLO/tracking.
        """
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

    def _allowed_class_ids(self) -> set[int] | None:
        """Return allowed class IDs if class_names is set; otherwise return None."""
        if not self.class_names or self.model is None:
            return None

        names = getattr(self.model, "names", None)
        if not isinstance(names, dict):
            return None

        wanted = {name.lower().strip() for name in self.class_names}
        allowed = {
            int(class_id)
            for class_id, class_name in names.items()
            if str(class_name).lower().strip() in wanted
        }

        if not allowed:
            self.progress.emit(9, f"Warning: class_names {self.class_names} did not match model.names; not filtering.")
            return None
        return allowed

    def process_video(self, video_path: str):
        self.stop_requested = False
        self.crop_records.clear()

        # fresh tracking session
        if self.rebuild_tracker_on_start:
            self.reset_tracking(rebuild_model=True)
        else:
            self.reset_tracking(rebuild_model=False)

        if not self.load_model():
            return

        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.error.emit(f"Could not open video file: {video_path}")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            allowed_class_ids = self._allowed_class_ids()
            frame_area = max(1, width * height)

            frame_idx = 0
            self.progress.emit(10, "Starting YOLO.track video processing...")

            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_skip == 0:
                    t0 = time.time()
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
                        return

                    elapsed = time.time() - t0
                    if elapsed > 0:
                        inst = 1.0 / elapsed
                        self.detect_fps_smooth = inst if self.detect_fps_smooth == 0 else 0.85 * self.detect_fps_smooth + 0.15 * inst

                    boxes = self._parse_track_results(results, frame_idx, allowed_class_ids)
                    for box in boxes:
                        if bbox_area(box.xyxy) < self.min_area_ratio * frame_area:
                            continue
                        self._update_crop_record(frame, box)

                frame_idx += 1

                if total > 0:
                    pct = min(94, int(10 + (frame_idx / total) * 84))
                    self.progress.emit(
                        pct,
                        f"YOLO.track frame {frame_idx}/{total} | tracks: {len(self.crop_records)} | detect FPS: {self.detect_fps_smooth:.1f}",
                    )
                else:
                    self.progress.emit(
                        50,
                        f"YOLO.track frame {frame_idx} | tracks: {len(self.crop_records)} | detect FPS: {self.detect_fps_smooth:.1f}",
                    )

            self.progress.emit(95, "Stopping and collecting best tracked crops..." if self.stop_requested else "Finishing up...")

            records = self.get_crop_records(include_crop=True, only_ready=False)
            best_crops = [rec["crop"] for rec in records if isinstance(rec.get("crop"), np.ndarray) and rec["crop"].size > 0]

            self.finished_records.emit(records)
            self.finished.emit(best_crops)
            self.progress.emit(100, "Stopped" if self.stop_requested else f"Done — {len(best_crops)} best crop(s)")

        except Exception as exc:
            self.error.emit(f"Error during YOLO processing: {exc}")
        finally:
            if cap is not None:
                cap.release()

    def _parse_track_results(
        self,
        results: Any,
        frame_idx: int,
        allowed_class_ids: set[int] | None,
    ) -> list[TrackBox]:
        boxes_out: list[TrackBox] = []
        if not results:
            return boxes_out

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return boxes_out

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=float)
        cls_arr = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.full((len(xyxy),), -1, dtype=int)
        id_arr = None
        if getattr(boxes, "id", None) is not None:
            id_arr = boxes.id.detach().cpu().numpy().astype(int)

        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}

        for det_idx, raw_box in enumerate(xyxy):
            cls_id = int(cls_arr[det_idx])
            if allowed_class_ids is not None and cls_id not in allowed_class_ids:
                continue

            fixed = clamp_box(raw_box[:4], result.orig_shape if hasattr(result, "orig_shape") else (10**9, 10**9))

            if fixed is None:
                fixed = [int(round(v)) for v in raw_box[:4].tolist()]

            if id_arr is not None:
                track_id = int(id_arr[det_idx])
            elif self.require_track_id:
  
                continue
            else:
  
                track_id = self._fallback_id_next
                self._fallback_id_next += 1

            label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
            boxes_out.append(
                TrackBox(
                    xyxy=[int(round(v)) for v in raw_box[:4].tolist()],
                    score=float(confs[det_idx]),
                    label=label,
                    cls_id=cls_id,
                    track_id=track_id,
                    frame_index=int(frame_idx),
                )
            )

        return boxes_out

    def _update_crop_record(self, frame_bgr: np.ndarray, box: TrackBox) -> bool:
        expanded = expand_box(box.xyxy, frame_bgr.shape, pad_ratio=self.pad_ratio, pad_pixels=self.box_pad)
        if expanded is None:
            return False

        crop = crop_from_box(frame_bgr, expanded)
        if crop is None:
            return False

        quality = crop_quality_score(frame_bgr, expanded, box.score)
        if quality < 0.05:
            return False

        rec = self.crop_records.get(box.track_id)
        if rec is None:
            rec = CropRecord(
                object_id=box.track_id,
                label=box.label,
                cls_id=box.cls_id,
                best_crop=crop.copy(),
                best_box=expanded,
                best_quality=quality,
                best_score=box.score,
                seen_count=1,
                version=1,
                first_frame=box.frame_index,
                last_seen_frame=box.frame_index,
                ready_for_ocr=self.min_stable_frames <= 1,
            )
            self.crop_records[box.track_id] = rec
            self._emit_record_update(rec)
            return True

        rec.seen_count += 1
        rec.last_seen_frame = box.frame_index
        became_ready = False
        if rec.seen_count >= self.min_stable_frames and not rec.ready_for_ocr:
            rec.ready_for_ocr = True
            became_ready = True

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
            self._emit_record_update(rec)
            return True
        return False

    def _emit_record_update(self, rec: CropRecord):
        summary = rec.to_summary(include_crop=True)
        self.poster_found.emit(int(rec.object_id), rec.best_crop.copy())
        self.poster_found_record.emit(summary)

    def get_crop_records(self, include_crop: bool = True, only_ready: bool = False) -> list[dict[str, Any]]:
        records = []
        for rec in self.crop_records.values():
            if only_ready and not rec.ready_for_ocr:
                continue
            records.append(rec.to_summary(include_crop=include_crop))
        records.sort(key=lambda item: (-float(item["quality"]), int(item["id"])))
        return records


class YOLOWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    poster_found_record = pyqtSignal(dict)
    finished_records = pyqtSignal(list)

    def __init__(
        self,
        video_path: str,
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
        self.video_path = str(video_path)
        self.engine = YOLOEngine(
            model_path=model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            frame_skip=frame_skip,
            box_pad=box_pad,
            pad_ratio=pad_ratio,
            max_det=max_det,
            class_names=class_names,
            tracker_name=tracker_name,
            min_stable_frames=min_stable_frames,
            min_area_ratio=min_area_ratio,
            rebuild_tracker_on_start=rebuild_tracker_on_start,
            require_track_id=require_track_id,
        )

    def request_stop(self):
        self.engine.request_stop()

    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.poster_found_record.connect(self.poster_found_record)
        self.engine.finished_records.connect(self.finished_records)
        self.engine.process_video(self.video_path)
