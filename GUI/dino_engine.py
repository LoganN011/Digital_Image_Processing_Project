import os
import math
import cv2
import torch
import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal, QThread

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import transformers
    _new_api = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 50)
except ImportError:
    AutoProcessor = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "poster . flyer . notice . bulletin"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

multiplier = 1.0
MIN_W          = 80 * multiplier
MIN_H          = 100 * multiplier
MIN_AREA_RATIO = 0.02 * multiplier
MIN_ASPECT     = 0.2 * multiplier
MAX_ASPECT     = 5.0 * multiplier
EDGE_MARGIN    = 8
TRACK_IOU_THRESH  = 0.15
TRACK_CENTER_DIST = 200


def valid_poster_box(box, frame_shape):
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_W or bh < MIN_H:
        return False, "too small"
    if (bw * bh) / float(fw * fh) < MIN_AREA_RATIO:
        return False, "area too small"
    aspect = bw / float(bh) if bh > 0 else 0
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False, f"bad aspect {aspect:.2f}"
    if x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN or y2 >= fh - EDGE_MARGIN:
        return False, "touches border"
    return True, "ok"

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def centroid(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

def crop_score(crop, box, frame_shape):
    fh, fw = frame_shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    cx, cy = centroid(box)
    center_d = ((cx - fw/2)**2 + (cy - fh/2)**2) ** 0.5 / ((fw/2)**2 + (fh/2)**2) ** 0.5
    area = (box[2]-box[0]) * (box[3]-box[1])
    return sharp + area * 0.01 - center_d * 300

def pad_box(box, frame_shape, pad=8):
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = box
    return max(0, x1-pad), max(0, y1-pad), min(fw, x2+pad), min(fh, y2+pad)


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
            
        if AutoProcessor is None:
            self.error.emit("Transformers library not installed.")
            return False

        self.progress.emit(5, "Loading Grounding DINO Model...")
        try:
            # backend="pil" suppresses the use_fast warning and is more stable here
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, backend="pil")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
            self.model.eval()
            return True
        except Exception as e:
            self.error.emit(f"Failed to load DINO: {str(e)}")
            return False

    def detect_posters(self, frame_bgr):
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h, w = frame_bgr.shape[:2]
        inputs = self.processor(images=image, text=PROMPT, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if _new_api:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w)]
            )[0]
        else:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w)]
            )[0]

        boxes = results["boxes"].cpu().numpy().astype(int)
        scores = results["scores"].cpu().numpy()
        labels = results.get("labels", results.get("text_labels", []))
        return boxes, scores, labels

    def update_tracks(self, tracks, box, crop, frame_shape):
        c = centroid(box)
        # Limit track search to the last 50 tracks to keep it fast
        for track in tracks[-50:]:
            if iou(box, track["box"]) > TRACK_IOU_THRESH or dist(c, track["centroid"]) < TRACK_CENTER_DIST:
                track["centroid"] = c
                track["box"] = box
                s = crop_score(crop, box, frame_shape)
                if s > track["best_score"]:
                    track["best_score"] = s
                    track["best_crop"] = crop.copy()
                    self.poster_found.emit(tracks.index(track), crop)
                return
        new_track = {
            "centroid": c,
            "box": box,
            "best_crop": crop.copy(),
            "best_score": crop_score(crop, box, frame_shape),
        }
        tracks.append(new_track)
        self.poster_found.emit(len(tracks) - 1, crop)

    def process_video(self, video_path):
        self.stop_requested = False
        
        if not self.load_model():
            return

        self.progress.emit(10, "Starting DINO session...")
        cap = None
        try:
            tracks = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.error.emit(f"Could not open video file: {video_path}")
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


            frame_idx = 0
            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                boxes, scores, labels = self.detect_posters(frame)

                for box, score, label in zip(boxes, scores, labels):
                    ok, reason = valid_poster_box(box, frame.shape)
                    if not ok:
                        continue

                    x1, y1, x2, y2 = pad_box(box, frame.shape)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    self.update_tracks(tracks, box, crop, frame.shape)

                frame_idx += 1
                
                if total > 0:
                    pct = int(10 + (frame_idx / total) * 80)
                    self.progress.emit(pct, f"Processing frame {frame_idx}/{total}...")
                else:
                    self.progress.emit(50, f"Processing frame {frame_idx}...")


            if self.stop_requested:
                self.progress.emit(100, "Stopped")
            else:
                self.progress.emit(100, "Done")

            # Don't send the massive list of images through the signal
            # The GUI already has them via poster_found
            self.finished.emit([])

        except Exception as e:
            self.error.emit(f"Error during DINO processing: {str(e)}")

        finally:
            if cap is not None:
                cap.release()
            import gc
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
        self.video_path = video_path
        self.engine = DINOEngine()

    def request_stop(self):
        self.engine.request_stop()
    
    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.process_video(self.video_path)
