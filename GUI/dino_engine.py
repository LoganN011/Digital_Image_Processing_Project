import gc
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
    AutoProcessor = None
    AutoModelForZeroShotObjectDetection = None
    NEW_API = False

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


def valid_box(box, shape):
    fh, fw = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_W or bh < MIN_H:
        return False
    if (bw * bh) / max(1.0, float(fw * fh)) < MIN_AREA_RATIO:
        return False
    aspect = bw / float(max(1, bh))
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return False
    return not (x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN or y2 >= fh - EDGE_MARGIN)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1.0, float(area_a + area_b - inter))


def centroid(box):
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def center_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def pad_box(box, shape, pad=8):
    fh, fw = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    return max(0, x1 - pad), max(0, y1 - pad), min(fw, x2 + pad), min(fh, y2 + pad)


def crop_score(crop, box, shape):
    fh, fw = shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    cx, cy = centroid(box)
    center = ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5 / max(1.0, ((fw / 2) ** 2 + (fh / 2) ** 2) ** 0.5)
    return float(sharp) + (box[2] - box[0]) * (box[3] - box[1]) * 0.01 - center * 300


class DINOEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def load_model(self):
        if self.model is not None:
            return True
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            self.error.emit("Transformers library not installed.")
            return False
        self.progress.emit(5, "Loading Grounding DINO model...")
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, backend="pil")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
            return True
        except Exception as exc:
            self.error.emit(f"Failed to load DINO: {exc}")
            return False

    def detect_posters(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        inputs = self.processor(images=image, text=PROMPT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if NEW_API:
            results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
        else:
            results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
        return results["boxes"].cpu().numpy().astype(int), results["scores"].cpu().numpy()

    def update_tracks(self, tracks, box, crop, shape):
        c = centroid(box)
        score = crop_score(crop, box, shape)
        for i, track in enumerate(tracks[-50:], max(0, len(tracks) - 50)):
            if iou(box, track["box"]) > TRACK_IOU_THRESH or center_dist(c, track["centroid"]) < TRACK_CENTER_DIST:
                track.update(centroid=c, box=box)
                if score > track["score"]:
                    track.update(score=score, crop=crop.copy())
                    self.poster_found.emit(i, crop)
                return
        tracks.append({"centroid": c, "box": box, "crop": crop.copy(), "score": score})
        self.poster_found.emit(len(tracks) - 1, crop)

    def process_video(self, video_path):
        self.stop_requested = False
        if not self.load_model():
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {video_path}")
            return
        try:
            tracks = []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            self.progress.emit(10, "Starting DINO session...")
            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                boxes, scores = self.detect_posters(frame)
                for box, _score in zip(boxes, scores):
                    if not valid_box(box, frame.shape):
                        continue
                    x1, y1, x2, y2 = pad_box(box, frame.shape)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        self.update_tracks(tracks, box, crop, frame.shape)
                frame_idx += 1
                pct = int(10 + (frame_idx / total) * 80) if total > 0 else 50
                self.progress.emit(pct, f"Processing frame {frame_idx}/{total}..." if total > 0 else f"Processing frame {frame_idx}...")
            self.progress.emit(100, "Stopped" if self.stop_requested else "Done")
            self.finished.emit([])
        except Exception as exc:
            self.error.emit(f"Error during DINO processing: {exc}")
        finally:
            cap.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DINOWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = str(video_path)
        self.engine = DINOEngine()

    def request_stop(self):
        self.engine.request_stop()

    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.process_video(self.video_path)
