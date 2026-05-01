import gc
import cv2
import numpy as np
import torch
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    from sam3.model_builder import build_sam3_video_predictor
except ImportError:
    build_sam3_video_predictor = None

CENTROID_THRESHOLD = 250
MIN_AREA_RATIO = 0.0005
PROMPT = "flyer, poster"


def centroid(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def xyxy(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def area(box):
    x1, y1, x2, y2 = xyxy(box)
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def pad_box(box, shape, pad=1):
    h, w = shape[:2]
    x1, y1, x2, y2 = xyxy(box)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def crop_box(frame, box):
    x1, y1, x2, y2 = xyxy(box)
    return frame[y1:y2, x1:x2]


def crop_score(frame, box):
    crop = crop_box(frame, box)
    if crop.size == 0:
        return 0.0, crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var()
    contrast = max(gray.std() / 255.0, 1e-6)
    x1, y1, x2, y2 = xyxy(box)
    aspect = min((x2 - x1) / max(1, y2 - y1), 3.0)
    boundary = 0.6 if x1 <= 5 or y1 <= 5 or x2 >= frame.shape[1] - 5 or y2 >= frame.shape[0] - 5 else 1.0
    return float(sharpness * contrast * np.log1p(area(box)) * aspect * boundary), crop


class SAMEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.video_predictor = None
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def load_model(self):
        if self.video_predictor is not None:
            return True
        if build_sam3_video_predictor is None:
            self.error.emit("SAM3 is not installed. Please install it to use SAM.")
            return False
        self.progress.emit(5, "Building SAM3 model...")
        try:
            self.video_predictor = build_sam3_video_predictor()
            return True
        except Exception as exc:
            self.error.emit(f"Failed to load SAM3: {exc}")
            return False

    def process_video(self, video_path):
        self.stop_requested = False
        if not self.load_model():
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {video_path}")
            return
        session_id = None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_area = max(1, width * height)
            tracks = []
            response = self.video_predictor.handle_request(request={"type": "start_session", "resource_path": str(video_path)})
            session_id = response["session_id"]
            frame_idx = 0
            self.progress.emit(10, "Starting SAM3 session...")
            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                response = self.video_predictor.handle_request(request={"type": "add_prompt", "session_id": session_id, "frame_index": frame_idx, "text": PROMPT})
                for x, y, w, h in response.get("outputs", {}).get("out_boxes_xywh", []):
                    box = pad_box([[int(x * width), int(y * height)], [int((x + w) * width), int(y * height)], [int((x + w) * width), int((y + h) * height)], [int(x * width), int((y + h) * height)]], frame.shape)
                    if area(box) >= MIN_AREA_RATIO * frame_area:
                        self.update_tracks(tracks, frame, box, frame_idx)
                frame_idx += 1
                pct = int(10 + (frame_idx / total) * 80) if total > 0 else 50
                self.progress.emit(pct, f"Processing frame {frame_idx}/{total}..." if total > 0 else f"Processing frame {frame_idx}...")
            self.progress.emit(95, "Stopping and collecting current best crops..." if self.stop_requested else "Finishing up...")
            best = [max(track["candidates"], key=lambda item: item["score"])["crop"] for track in tracks if track["candidates"]]
            self.progress.emit(100, "Stopped" if self.stop_requested else "Done")
            self.finished.emit(best)
        except Exception as exc:
            self.error.emit(f"Error during SAM processing: {exc}")
        finally:
            cap.release()
            if self.video_predictor is not None and session_id is not None:
                try:
                    self.video_predictor.handle_request(request={"type": "close_session", "session_id": session_id, "run_gc_collect": True})
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def update_tracks(self, tracks, frame, box, frame_idx):
        c = centroid(box)
        score, crop = crop_score(frame, box)
        if crop.size == 0:
            return
        for idx, track in enumerate(tracks):
            if dist(c, track["centroid"]) < CENTROID_THRESHOLD and iou(box, track["bbox"]) > 0.15:
                old_best = max(track["candidates"], key=lambda item: item["score"])["score"] if track["candidates"] else -1
                track["centroid"] = int(0.7 * track["centroid"][0] + 0.3 * c[0]), int(0.7 * track["centroid"][1] + 0.3 * c[1])
                track["bbox"] = box
                track["candidates"].append({"frame_idx": frame_idx, "score": score, "crop": crop.copy(), "bbox": box})
                if score > old_best:
                    self.poster_found.emit(idx, crop)
                return
        tracks.append({"centroid": c, "bbox": box, "first_frame": frame_idx, "candidates": [{"frame_idx": frame_idx, "score": score, "crop": crop.copy(), "bbox": box}]})
        self.poster_found.emit(len(tracks) - 1, crop)


class SAMWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = str(video_path)
        self.engine = SAMEngine()

    def request_stop(self):
        self.engine.request_stop()

    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.process_video(self.video_path)
