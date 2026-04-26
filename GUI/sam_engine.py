import math
import time
import cv2
import torch
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QThread

try:
    from sam3.model_builder import build_sam3_video_predictor
except ImportError:
    build_sam3_video_predictor = None

CENTROID_THRESHOLD = 250

def get_centroid(bbox):
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return (int(sum(x_coords)/len(x_coords)), int(sum(y_coords)/len(y_coords)))

def euclidean(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def bbox_area(bbox):
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return (max(x_coords)-min(x_coords)) * (max(y_coords)-min(y_coords))

def bbox_xyxy(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_iou(b1, b2):
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

def score_frame(frame, bbox):
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return 0, crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpness = cv2.Laplacian(blurred, cv2.CV_64F).var()

    contrast = gray.std() / 255.0
    contrast = max(contrast, 1e-6)

    w = x_max - x_min
    h = y_max - y_min
    area = w * h
    area_score = math.log1p(area)

    aspect = (w / h) if h > 0 else 0
    aspect_bonus = min(aspect, 3.0)

    fh, fw = frame.shape[:2]
    margin = 5
    on_boundary = (
        x_min <= margin or y_min <= margin or
        x_max >= fw - margin or y_max >= fh - margin
    )
    boundary_penalty = 0.6 if on_boundary else 1.0

    combined = sharpness * contrast * area_score * aspect_bonus * boundary_penalty
    return combined, crop

def pad_bbox(bbox, frame_shape, pad=60):
    fh, fw = frame_shape[:2]
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    x_min = max(0,  min(x_coords) - pad)
    x_max = min(fw, max(x_coords) + pad)
    y_min = max(0,  min(y_coords) - pad)
    y_max = min(fh, max(y_coords) + pad)
    return [[x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max]]

class SAMEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self, checkpoint_path="sam3_video_predictor.pt"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
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

        self.progress.emit(5, "Building SAM3 model (this could take a while)...")
        try:
            self.video_predictor = build_sam3_video_predictor()
            return True
        except Exception as e:
            self.error.emit(f"Failed to load SAM3: {str(e)}")
            return False

    def process_video(self, video_path):
        self.stop_requested = False
        
        if not self.load_model():
            return

        self.progress.emit(10, "Starting SAM3 session...")
        session_id = None
        cap = None
        writer = None
        try:
            unique_signs = []
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.error.emit(f"Could not open video file: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            FRAME_SKIP = 1



            response = self.video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            session_id = response["session_id"]
            
            frame_idx = 0

            
            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SKIP == 0:

                    response = self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=frame_idx,
                            text="flyer, poster",
                        )
                    )

                    outputs = response.get("outputs", {})
                    boxes = outputs.get("out_boxes_xywh", [])

                    for box in boxes:
                        x, y, w, h = box
                        x1 = int(x * width)
                        y1 = int(y * height)
                        x2 = int((x + w) * width)
                        y2 = int((y + h) * height)

                        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        BOX_PAD = 1
                        bbox = pad_bbox(bbox, frame.shape, pad=BOX_PAD)

                        if bbox_area(bbox) < 0.0005 * (width * height):
                            continue

                        centroid = get_centroid(bbox)


                        matched = False
                        for sign in unique_signs:
                            dist = euclidean(centroid, sign['centroid'])
                            iou  = bbox_iou(bbox, sign['bbox'])

                            if dist < CENTROID_THRESHOLD and iou > 0.15:
                                matched = True
                                sign['centroid'] = (
                                    int(0.7 * sign['centroid'][0] + 0.3 * centroid[0]),
                                    int(0.7 * sign['centroid'][1] + 0.3 * centroid[1])
                                )
                                sign['bbox'] = bbox

                                score, crop = score_frame(frame, bbox)
                                if crop.size > 0:
                                    # Find if this is the best so far
                                    old_best = max(sign['candidates'], key=lambda x: x['score'])['score'] if sign['candidates'] else -1
                                    
                                    sign['candidates'].append({
                                        'frame_idx': frame_idx,
                                        'score': score,
                                        'crop': crop.copy(),
                                        'bbox': bbox
                                    })
                                    
                                    if score > old_best:
                                        self.poster_found.emit(unique_signs.index(sign), crop)
                                break

                        if not matched:
                            score, crop = score_frame(frame, bbox)
                            if crop.size > 0:
                                new_sign = {
                                    'centroid': centroid,
                                    'bbox': bbox,
                                    'first_frame': frame_idx,
                                    'candidates': [{
                                        'frame_idx': frame_idx,
                                        'score': score,
                                        'crop': crop.copy(),
                                        'bbox': bbox
                                    }]
                                }
                                unique_signs.append(new_sign)
                                self.poster_found.emit(len(unique_signs) - 1, crop)









                frame_idx += 1
                
                if total > 0:
                    pct = int(10 + (frame_idx / total) * 80)
                    self.progress.emit(pct, f"Processing frame {frame_idx}/{total}...")
                else:
                    self.progress.emit(50, f"Processing frame {frame_idx}...")

            if self.stop_requested:
                self.progress.emit(95, "Stopping and collecting current best crops...")
            else:
                self.progress.emit(95, "Finishing up...")
            
            best_crops = []
            for s in unique_signs:
                if s['candidates']:
                    best_cand = max(s['candidates'], key=lambda x: x['score'])
                    best_crops.append(best_cand['crop'])
            
            if self.stop_requested:
                self.progress.emit(100, "Stopped")
            else:
                self.progress.emit(100, "Done")

            self.finished.emit(best_crops)

        except Exception as e:
            self.error.emit(f"Error during SAM processing: {str(e)}")

        finally:
            if cap is not None:
                cap.release()

            if self.video_predictor is not None and session_id is not None:
                try:
                    self.video_predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=session_id,
                            run_gc_collect=True,
                        )
                    )
                except Exception:
                    pass

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SAMWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.engine = SAMEngine()

    def request_stop(self):
        self.engine.request_stop()

    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.process_video(self.video_path)
