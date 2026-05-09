from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import argparse
import io
import os
import time

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
import gdown
import torch
import uvicorn

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT_URL = "https://drive.google.com/file/d/17ftyxnIUuabLsjzQYO3-YmXTcNPnKcup/view?usp=sharing"
LOCAL_CHECKPOINT_PATH = Path(__file__).parent
MIN_CHECKPOINT_BYTES = 2 * 1024 ** 3

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_checkpoint() -> Path:
    if not LOCAL_CHECKPOINT_PATH.exists() or LOCAL_CHECKPOINT_PATH.stat().st_size < MIN_CHECKPOINT_BYTES:
        if LOCAL_CHECKPOINT_PATH.exists():
            LOCAL_CHECKPOINT_PATH.unlink()
        print("Downloading SAM3 checkpoint from Google Drive...", flush=True)
        gdown.download(url=CHECKPOINT_URL, output=str(LOCAL_CHECKPOINT_PATH), quiet=False, fuzzy=True)

    if not LOCAL_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint download failed: {LOCAL_CHECKPOINT_PATH}")

    if LOCAL_CHECKPOINT_PATH.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(f"Downloaded file is too small to be the checkpoint: {LOCAL_CHECKPOINT_PATH.stat().st_size} bytes")

    print(f"Checkpoint ready: {LOCAL_CHECKPOINT_PATH}", flush=True)
    print(f"Size: {LOCAL_CHECKPOINT_PATH.stat().st_size / (1024 ** 3):.2f} GB", flush=True)
    return LOCAL_CHECKPOINT_PATH


def build_local_sam3_image_model():
    checkpoint_path = prepare_checkpoint()
    print("Loading SAM3 image model from checkpoint", flush=True)
    print(f"  {checkpoint_path}", flush=True)

    try:
        model = build_sam3_image_model(
            device=device,
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
        )
    except TypeError:
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
        )

    return model, str(checkpoint_path)


model, sam3_checkpoint_path = build_local_sam3_image_model()
try:
    processor = Sam3Processor(model, device=device)
except TypeError:
    processor = Sam3Processor(model)
lock = Lock()


@dataclass
class Track:
    track_id: int
    box: list
    score: float
    age: int
    hits: int
    misses: int
    frame_index: int


class BoxTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frame_index = 0

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.frame_index = 0

    def iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = aa + ba - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def update(self, detections, track_iou, max_misses):
        self.frame_index += 1
        track_ids = list(self.tracks.keys())
        pairs = []

        for ti in track_ids:
            for di, det in enumerate(detections):
                pairs.append((self.iou(self.tracks[ti].box, det["box"]), ti, di))

        pairs.sort(reverse=True, key=lambda x: x[0])
        used_tracks = set()
        used_detections = set()

        for value, ti, di in pairs:
            if value < track_iou:
                break
            if ti in used_tracks or di in used_detections:
                continue

            det = detections[di]
            track = self.tracks[ti]
            track.box = det["box"]
            track.score = det["score"]
            track.age += 1
            track.hits += 1
            track.misses = 0
            track.frame_index = self.frame_index
            used_tracks.add(ti)
            used_detections.add(di)

        for ti in track_ids:
            if ti not in used_tracks:
                self.tracks[ti].age += 1
                self.tracks[ti].misses += 1

        for di, det in enumerate(detections):
            if di in used_detections:
                continue
            ti = self.next_id
            self.next_id += 1
            self.tracks[ti] = Track(ti, det["box"], det["score"], 1, 1, 0, self.frame_index)

        remove = [ti for ti, track in self.tracks.items() if track.misses > max_misses]
        for ti in remove:
            del self.tracks[ti]

        visible = [track for track in self.tracks.values() if track.misses == 0]
        visible.sort(key=lambda t: t.track_id)

        return [
            {
                "track_id": t.track_id,
                "box": t.box,
                "score": t.score,
                "age": t.age,
                "hits": t.hits,
                "misses": t.misses,
            }
            for t in visible
        ]


tracker = BoxTracker()


def as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def get_output_value(output, names):
    for name in names:
        if isinstance(output, dict) and name in output:
            return output[name]
        if hasattr(output, name):
            return getattr(output, name)
    return None


def normalize_boxes(boxes, width, height):
    normalized = []
    for box in boxes:
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in box[:4]]

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        x1 = min(max(x1, 0.0), float(width - 1))
        x2 = min(max(x2, 0.0), float(width - 1))
        y1 = min(max(y1, 0.0), float(height - 1))
        y2 = min(max(y2, 0.0), float(height - 1))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
            normalized.append([x1, y1, x2, y2])

    return normalized


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def suppress_duplicates(detections, duplicate_iou):
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    for det in ordered:
        if all(iou(det["box"], other["box"]) < duplicate_iou for other in keep):
            keep.append(det)

    return keep


@app.get("/health")
def health():
    gpu_type = None
    if device == "cuda" and torch.cuda.is_available():
        gpu_type = torch.cuda.get_device_name(0)
    return {
        "ok": True,
        "device": device,
        "gpu_type": gpu_type,
        "tracks": len(tracker.tracks),
        "checkpoint_path": sam3_checkpoint_path,
        "huggingface_load": False,
    }


@app.post("/reset")
def reset():
    with lock:
        tracker.reset()
    return {"ok": True}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    prompt: str = Form("poster, flyer"),
    conf: float = Form(0.25),
    track_iou: float = Form(0.30),
    max_misses: int = Form(8),
    duplicate_iou: float = Form(0.65),
):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    t0 = time.time()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    with lock:
        with torch.inference_mode(), autocast_ctx:
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=prompt)

        raw_boxes = as_list(get_output_value(output, ["boxes", "pred_boxes"]))
        raw_scores = as_list(get_output_value(output, ["scores", "pred_scores", "confidence", "confidences"]))
        boxes = normalize_boxes(raw_boxes, width, height)

        if not raw_scores:
            raw_scores = [1.0 for _ in boxes]

        detections = []
        for box, score in zip(boxes, raw_scores):
            score = float(score)
            if score >= conf:
                detections.append({"box": box, "score": score})

        detections = suppress_duplicates(detections, duplicate_iou)
        tracks = tracker.update(detections, track_iou, max_misses)

    server_ms = (time.time() - t0) * 1000.0

    return {
        "ok": True,
        "device": device,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_detections": len(detections),
        "tracks": tracks,
        "server_ms": server_ms,
        "checkpoint_path": sam3_checkpoint_path,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SAM3_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SAM3_SERVER_PORT", "8000")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
