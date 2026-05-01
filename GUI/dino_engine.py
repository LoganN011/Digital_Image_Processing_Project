from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    NEW_API = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 50)
except Exception:
    AutoModelForZeroShotObjectDetection = None
    AutoProcessor = None
    NEW_API = False

try:
    cv2.setNumThreads(1)
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "poster . flyer . notice . bulletin"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
MIN_W = 80
MIN_H = 100
MIN_AREA_RATIO = 0.02
MIN_ASPECT = 0.2
MAX_ASPECT = 5.0
EDGE_MARGIN = 8
TRACK_IOU_THRESH = 0.15
TRACK_CENTER_DIST = 200
DEFAULT_FRAME_SKIP = 1
MAX_RECENT_TRACKS = 50


def valid_box(box, shape) -> bool:
    fh, fw = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_W or bh < MIN_H:
        return False
    if (bw * bh) / max(1.0, float(fw * fh)) < MIN_AREA_RATIO:
        return False
    aspect = bw / float(max(1, bh))
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    return not (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN or y2 >= fh - EDGE_MARGIN)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1.0, float(area_a + area_b - inter))


def centroid(box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def center_dist(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def pad_box(box, shape, pad: int = 8) -> tuple[int, int, int, int]:
    fh, fw = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    return max(0, x1 - pad), max(0, y1 - pad), min(fw, x2 + pad), min(fh, y2 + pad)




def detector_preview(frame: np.ndarray, boxes, scores=None, long_side: int = 480) -> np.ndarray:
    """Return a BGR preview frame with DINO detection boxes drawn on it."""
    preview = frame.copy()
    scores = scores if scores is not None else [None] * len(boxes)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        score = scores[i] if i < len(scores) else None
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"DINO {float(score):.2f}" if score is not None else "DINO"
        cv2.putText(
            preview,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    h, w = preview.shape[:2]
    longest = max(h, w)
    if longest > long_side:
        scale = long_side / float(longest)
        preview = cv2.resize(
            preview,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return preview

def crop_score(crop: np.ndarray, box, shape) -> float:
    fh, fw = shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    cx, cy = centroid(box)
    max_center_dist = max(1.0, ((fw / 2) ** 2 + (fh / 2) ** 2) ** 0.5)
    center_penalty = ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5 / max_center_dist
    area = max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))
    return float(sharp) + area * 0.01 - center_penalty * 300.0


class DINOEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)
    frame_preview = pyqtSignal(object)

    def __init__(self, frame_skip: int = DEFAULT_FRAME_SKIP):
        super().__init__()
        self.frame_skip = max(1, int(frame_skip))
        self.model = None
        self.processor = None
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def load_model(self) -> bool:
        if self.model is not None and self.processor is not None:
            return True
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            self.error.emit("Transformers library is not installed.")
            return False
        self.progress.emit(5, "Loading Grounding DINO model...")
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, backend="pil")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
            self.progress.emit(8, f"Grounding DINO ready on {DEVICE}.")
            return True
        except Exception as exc:
            self.error.emit(f"Failed to load DINO: {exc}")
            return False

    def detect_posters(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        inputs = self.processor(images=image, text=PROMPT, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if NEW_API:
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w)],
            )[0]
        else:
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w)],
            )[0]
        return result["boxes"].detach().cpu().numpy().astype(int), result["scores"].detach().cpu().numpy()

    def update_tracks(self, tracks: list[dict[str, Any]], box, crop: np.ndarray, shape):
        c = centroid(box)
        score = crop_score(crop, box, shape)
        start = max(0, len(tracks) - MAX_RECENT_TRACKS)
        for idx in range(start, len(tracks)):
            track = tracks[idx]
            if iou(box, track["box"]) > TRACK_IOU_THRESH or center_dist(c, track["centroid"]) < TRACK_CENTER_DIST:
                track["centroid"] = c
                track["box"] = tuple(int(v) for v in box)
                if score > track["score"]:
                    track["score"] = score
                    track["crop"] = crop.copy()
                    self.poster_found.emit(idx, track["crop"])
                return
        tracks.append({"centroid": c, "box": tuple(int(v) for v in box), "crop": crop.copy(), "score": score})
        self.poster_found.emit(len(tracks) - 1, tracks[-1]["crop"])

    def summarize_tracks(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": idx,
                "bbox": list(track["box"]),
                "score": float(track["score"]),
                "crop": track["crop"].copy(),
            }
            for idx, track in enumerate(tracks)
        ]

    def process_video(self, video_path: str | Path):
        self.stop_requested = False
        if not self.load_model():
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {video_path}")
            return

        tracks: list[dict[str, Any]] = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        processed = 0
        self.progress.emit(10, "Starting DINO session...")

        try:
            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    self._emit_frame_progress(frame_idx, total)
                    continue

                boxes, scores = self.detect_posters(frame)
                processed += 1
                if self.stop_requested:
                    break

                valid_boxes = []
                valid_scores = []
                for box, _score in zip(boxes, scores):
                    if self.stop_requested:
                        break
                    if not valid_box(box, frame.shape):
                        continue
                    valid_boxes.append(box)
                    valid_scores.append(_score)
                    x1, y1, x2, y2 = pad_box(box, frame.shape)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        self.update_tracks(tracks, box, crop, frame.shape)

                self.frame_preview.emit(detector_preview(frame, valid_boxes, valid_scores))
                self._emit_frame_progress(frame_idx, total, processed)

            self.progress.emit(100, "DINO stopped." if self.stop_requested else "DINO done.")
            self.finished.emit(self.summarize_tracks(tracks))
        except Exception as exc:
            self.error.emit(f"Error during DINO processing: {exc}")
        finally:
            cap.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _emit_frame_progress(self, frame_idx: int, total: int, processed: int | None = None):
        pct = int(10 + (frame_idx / total) * 80) if total > 0 else 50
        if total > 0:
            msg = f"DINO frame {frame_idx}/{total}"
        else:
            msg = f"DINO frame {frame_idx}"
        if processed is not None and self.frame_skip > 1:
            msg += f" ({processed} inference frames)"
        self.progress.emit(max(10, min(95, pct)), msg)


class DINOWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)
    frame_preview = pyqtSignal(object)
    frame_preview = pyqtSignal(object)

    def __init__(self, video_path: str | Path, frame_skip: int = DEFAULT_FRAME_SKIP):
        super().__init__()
        self.video_path = str(video_path)
        self.frame_skip = max(1, int(frame_skip))
        self.engine: DINOEngine | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def request_stop(self):
        with self._lock:
            self._stop_requested = True
            engine = self.engine
        if engine is not None:
            engine.request_stop()
        self.requestInterruption()

    def run(self):
        engine = DINOEngine(frame_skip=self.frame_skip)
        with self._lock:
            self.engine = engine
            if self._stop_requested:
                engine.request_stop()

        engine.progress.connect(self.progress)
        engine.finished.connect(self.finished_processing)
        engine.poster_found.connect(self.poster_found)
        engine.error.connect(self.error)
        engine.frame_preview.connect(self.frame_preview)
        try:
            engine.process_video(self.video_path)
        finally:
            with self._lock:
                self.engine = None
