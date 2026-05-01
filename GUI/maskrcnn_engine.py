from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
except Exception:
    maskrcnn_resnet50_fpn = None

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except Exception:
    BYTETracker = None

try:
    cv2.setNumThreads(1)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "maskrcnn_best.pt")
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.30
DEFAULT_MAX_DET = 30
DEFAULT_IMGSZ = 960
DEFAULT_FRAME_SKIP = 2
DEFAULT_PAD_RATIO = 0.04
DEFAULT_BOX_PAD_PIXELS = 1
DEFAULT_MIN_STABLE_FRAMES = 3
DEFAULT_MIN_AREA_RATIO = 0.0005
DEFAULT_TRACK_BUFFER = 30
DEFAULT_TRACKER_NAME = "bytetrack"


def resolve_local_path(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else SCRIPT_DIR / p)


def choose_device(allow_mps: bool = False) -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clamp_box(box: Iterable[float], frame_shape: tuple[int, ...]) -> list[int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(float(x1)))))
    y1 = max(0, min(h - 1, int(round(float(y1)))))
    x2 = max(0, min(w, int(round(float(x2)))))
    y2 = max(0, min(h, int(round(float(y2)))))
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


def box_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = box_area(a)
    area_b = box_area(b)
    return inter / max(1.0, float(area_a + area_b - inter))


def box_center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_distance(a: list[int], b: list[int]) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


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


def resize_for_inference(frame: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if long_side <= 0 or longest <= long_side:
        return frame, 1.0
    scale = float(long_side) / float(longest)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


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
            "id": int(self.object_id),
            "track_id": int(self.object_id),
            "label": self.label,
            "cls_id": int(self.cls_id),
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


class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.30, max_missed: int = DEFAULT_TRACK_BUFFER):
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: list[TrackBox]) -> list[TrackBox]:
        for track in self.tracks.values():
            track["missed"] += 1

        assigned_tracks: set[int] = set()
        output: list[TrackBox] = []
        for det in sorted(detections, key=lambda item: item.score, reverse=True):
            best_id = None
            best_score = 0.0
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks or int(track["cls_id"]) != int(det.cls_id):
                    continue
                overlap = box_iou(det.xyxy, track["box"])
                dist_score = 1.0 / (1.0 + center_distance(det.xyxy, track["box"]) / 120.0)
                score = max(overlap, 0.25 * dist_score)
                if score > best_score:
                    best_score = score
                    best_id = track_id

            if best_id is None or best_score < self.iou_thresh:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {"box": list(det.xyxy), "cls_id": det.cls_id, "score": det.score, "missed": 0}
            assigned_tracks.add(best_id)
            output.append(TrackBox(det.xyxy, det.score, det.label, det.cls_id, best_id, det.frame_index))

        for track_id in [tid for tid, track in self.tracks.items() if track["missed"] > self.max_missed]:
            self.tracks.pop(track_id, None)
        return output


class _UltralyticsBoxes:
    def __init__(self, detections: list[TrackBox]):
        arr = np.array([det.xyxy + [det.score, det.cls_id] for det in detections], dtype=np.float32)
        if arr.size == 0:
            arr = np.empty((0, 6), dtype=np.float32)
        self.xyxy = arr[:, :4]
        self.conf = arr[:, 4]
        self.cls = arr[:, 5]
        x1, y1, x2, y2 = self.xyxy.T if len(arr) else ([], [], [], [])
        if len(arr):
            self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __len__(self):
        return len(self.conf)


class ByteTrackWrapper:
    def __init__(self, high_thresh: float, match_thresh: float, track_buffer: int):
        if BYTETracker is None:
            raise RuntimeError("Ultralytics BYTETracker is unavailable")
        args = SimpleNamespace(
            track_high_thresh=float(high_thresh),
            track_low_thresh=max(0.01, min(0.20, float(high_thresh) * 0.50)),
            new_track_thresh=float(high_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            fuse_score=True,
        )
        self.tracker = BYTETracker(args, frame_rate=30)

    def reset(self):
        try:
            self.tracker.reset()
        except Exception:
            pass

    def update(self, detections: list[TrackBox], frame: np.ndarray) -> list[TrackBox]:
        if not detections:
            try:
                self.tracker.update(_UltralyticsBoxes([]), img=frame)
            except Exception:
                pass
            return []

        tracks = self.tracker.update(_UltralyticsBoxes(detections), img=frame)
        parsed: list[TrackBox] = []
        for item in tracks or []:
            arr = np.asarray(item, dtype=float).reshape(-1)
            if arr.size < 5:
                continue
            box = clamp_box(arr[:4], frame.shape)
            if box is None:
                continue
            track_id = int(arr[4])
            score = float(arr[5]) if arr.size > 5 else 0.0
            cls_id = int(arr[6]) if arr.size > 6 else -1
            label = str(cls_id)
            parsed.append(TrackBox(box, score, label, cls_id, track_id, 0))
        return parsed


def strip_state_dict_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in state.items():
        new_key = str(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


def infer_num_classes(state: dict[str, Any], fallback: int = 2) -> int:
    for key in (
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.mask_predictor.mask_fcn_logits.weight",
        "roi_heads.mask_predictor.mask_fcn_logits.bias",
    ):
        value = state.get(key)
        if hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
    return fallback


def build_maskrcnn_model(num_classes: int):
    if maskrcnn_resnet50_fpn is None:
        raise RuntimeError("torchvision Mask R-CNN is unavailable")
    try:
        return maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=int(num_classes))
    except TypeError:
        return maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=int(num_classes))


class MaskRCNNEngine(QObject):
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
        label_names: dict[int, str] | list[str] | tuple[str, ...] | None = None,
        tracker_name: str = DEFAULT_TRACKER_NAME,
        min_stable_frames: int = DEFAULT_MIN_STABLE_FRAMES,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
        track_buffer: int = DEFAULT_TRACK_BUFFER,
        rebuild_tracker_on_start: bool = True,
        device: str | None = None,
        allow_mps: bool = False,
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
        self.label_names = label_names
        self.tracker_name = str(tracker_name).lower().strip()
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.min_area_ratio = float(min_area_ratio)
        self.track_buffer = int(track_buffer)
        self.rebuild_tracker_on_start = bool(rebuild_tracker_on_start)
        self.device = str(device).lower() if device else choose_device(allow_mps=allow_mps)
        if self.device == "mps" and not allow_mps:
            self.device = "cpu"
        self.model = None
        self.loaded_model_path: str | None = None
        self.stop_requested = False
        self.crop_records: dict[int, CropRecord] = {}
        self.detect_fps_smooth = 0.0
        self.simple_tracker = SimpleTracker(iou_thresh=self.iou, max_missed=self.track_buffer)
        self.byte_tracker = None
        self._byte_tracker_failed = False

    def request_stop(self):
        self.stop_requested = True

    def reset_tracking(self, rebuild_model: bool = False):
        self.crop_records.clear()
        self.detect_fps_smooth = 0.0
        self.simple_tracker.reset()
        if self.byte_tracker is not None:
            self.byte_tracker.reset()
        if rebuild_model:
            self.model = None
            self.loaded_model_path = None

    def load_model(self) -> bool:
        if torch is None:
            self.error.emit("PyTorch is not installed. Install torch and torchvision to use Mask R-CNN.")
            return False
        if self.model is not None and self.loaded_model_path == self.model_path:
            return True

        self.progress.emit(5, f"Loading Mask R-CNN model: {Path(self.model_path).name}")
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            model = self.model_from_checkpoint(checkpoint)
            model.to(self.device).eval()
        except Exception as exc:
            self.error.emit(f"Failed to load Mask R-CNN model: {exc}")
            return False

        self.model = model
        self.loaded_model_path = self.model_path
        note = " CPU is used because TorchVision Mask R-CNN/NMS is not implemented on MPS." if self.device == "cpu" and torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ""
        self.progress.emit(8, f"Loaded Mask R-CNN on {self.device}: {Path(self.model_path).name}.{note}")
        return True

    def model_from_checkpoint(self, checkpoint: Any):
        if nn is not None and isinstance(checkpoint, nn.Module):
            return checkpoint
        if isinstance(checkpoint, dict):
            model_obj = checkpoint.get("model") or checkpoint.get("module")
            if nn is not None and isinstance(model_obj, nn.Module):
                return model_obj
            state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            if not isinstance(state, dict):
                raise RuntimeError("checkpoint dict does not contain a usable state_dict")
            state = strip_state_dict_prefixes(state)
            model = build_maskrcnn_model(infer_num_classes(state))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                self.progress.emit(7, f"Mask R-CNN load warning: {len(unexpected)} unexpected key(s)")
            if missing:
                self.progress.emit(7, f"Mask R-CNN load warning: {len(missing)} missing key(s)")
            return model
        raise RuntimeError("unsupported checkpoint format; expected nn.Module or state_dict checkpoint")

    def load_tracker(self):
        if self.tracker_name != "bytetrack" or self._byte_tracker_failed:
            return
        if self.byte_tracker is not None:
            return
        try:
            self.byte_tracker = ByteTrackWrapper(self.conf, self.iou, self.track_buffer)
            self.progress.emit(9, "Using ByteTrack for Mask R-CNN detections.")
        except Exception as exc:
            self._byte_tracker_failed = True
            self.byte_tracker = None
            self.progress.emit(9, f"ByteTrack unavailable; using built-in IoU tracker. {exc}")

    def label_for(self, cls_id: int) -> str:
        if isinstance(self.label_names, dict):
            return str(self.label_names.get(int(cls_id), cls_id))
        if isinstance(self.label_names, (list, tuple)) and 0 <= int(cls_id) < len(self.label_names):
            return str(self.label_names[int(cls_id)])
        return "poster" if int(cls_id) == 1 else str(cls_id)

    def class_allowed(self, cls_id: int, label: str) -> bool:
        if not self.class_names:
            return True
        wanted = {str(name).lower().strip() for name in self.class_names}
        return str(cls_id) in wanted or label.lower().strip() in wanted

    def process_video(self, video_path: str):
        self.stop_requested = False
        self.reset_tracking(rebuild_model=self.rebuild_tracker_on_start)
        if not self.load_model():
            return
        self.load_tracker()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {video_path}")
            return

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_area = max(1, width * height)
            frame_idx = 0
            self.progress.emit(10, "Starting Mask R-CNN video processing...")
            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.frame_skip == 0:
                    self.process_frame(frame, frame_idx, frame_area)
                frame_idx += 1
                self.emit_frame_progress(frame_idx, total)
            self.finish()
        except Exception as exc:
            self.error.emit(f"Error during Mask R-CNN processing: {exc}")
        finally:
            cap.release()

    def process_frame(self, frame: np.ndarray, frame_idx: int, frame_area: int):
        start = time.time()
        detections = self.detect_frame(frame, frame_idx)
        elapsed = time.time() - start
        if elapsed > 0:
            inst = 1.0 / elapsed
            self.detect_fps_smooth = inst if self.detect_fps_smooth == 0 else 0.85 * self.detect_fps_smooth + 0.15 * inst

        tracks = self.track_detections(detections, frame, frame_idx)
        self.frame_preview.emit(detector_preview(frame, tracks))
        for box in tracks:
            if box_area(box.xyxy) >= self.min_area_ratio * frame_area:
                self.update_crop_record(frame, box)

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[TrackBox]:
        small, scale = resize_for_inference(frame, self.imgsz)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0).to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]

        boxes = output.get("boxes")
        scores = output.get("scores")
        labels = output.get("labels")
        if boxes is None or scores is None or labels is None or len(boxes) == 0:
            return []

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)
        keep = np.where(scores_np >= self.conf)[0]
        if keep.size == 0:
            return []
        keep = keep[np.argsort(scores_np[keep])[::-1]][: self.max_det]

        detections: list[TrackBox] = []
        for idx in keep:
            raw = boxes_np[idx] / max(scale, 1e-9)
            fixed = clamp_box(raw, frame.shape)
            if fixed is None:
                continue
            cls_id = int(labels_np[idx])
            label = self.label_for(cls_id)
            if self.class_allowed(cls_id, label):
                detections.append(TrackBox(fixed, float(scores_np[idx]), label, cls_id, -1, int(frame_idx)))
        return detections

    def track_detections(self, detections: list[TrackBox], frame: np.ndarray, frame_idx: int) -> list[TrackBox]:
        if self.byte_tracker is not None:
            try:
                tracked = self.byte_tracker.update(detections, frame)
                if tracked or not detections:
                    return [TrackBox(t.xyxy, t.score, self.label_for(t.cls_id), t.cls_id, t.track_id, frame_idx) for t in tracked]
            except Exception as exc:
                self._byte_tracker_failed = True
                self.byte_tracker = None
                self.progress.emit(9, f"ByteTrack failed; using built-in IoU tracker. {exc}")
        return self.simple_tracker.update(detections)

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
            msg = f"Mask R-CNN frame {frame_idx}/{total} | tracks: {len(self.crop_records)} | detect FPS: {self.detect_fps_smooth:.1f}"
        else:
            pct = 50
            msg = f"Mask R-CNN frame {frame_idx} | tracks: {len(self.crop_records)} | detect FPS: {self.detect_fps_smooth:.1f}"
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


class MaskRCNNWorker(QThread):
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
        self.engine = MaskRCNNEngine(**engine_kwargs)

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
