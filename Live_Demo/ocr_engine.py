import argparse
import base64
import io
import json
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class CropMessage:
    track_id: int
    frame_index: int
    score: float
    box: list
    image_jpeg_b64: str
    source: str
    timestamp: float
    frame: int


def recv_exact(conn, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class SocketReceiver(QThread):
    message_ready = pyqtSignal(dict)
    status_ready = pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.running = True
        self.sock = None

    def stop(self):
        self.running = False
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.wait(1500)

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(8)
        self.sock.settimeout(0.5)
        self.status_ready.emit(f"Listening on {self.host}:{self.port}")
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()

    def handle_client(self, conn, addr):
        with conn:
            while self.running:
                header = recv_exact(conn, 4)
                if not header:
                    break
                size = struct.unpack("!I", header)[0]
                if size <= 0 or size > 80_000_000:
                    break
                data = recv_exact(conn, size)
                if not data:
                    break
                try:
                    message = json.loads(data.decode("utf-8"))
                    self.message_ready.emit(message)
                except Exception as e:
                    self.status_ready.emit(str(e))


def as_easyocr_boxes(horizontal, free, width, height):
    boxes = []
    if horizontal is None:
        horizontal = []
    if free is None:
        free = []
    if len(horizontal) == 1 and isinstance(horizontal[0], list) and (not horizontal[0] or isinstance(horizontal[0][0], (list, tuple))):
        horizontal = horizontal[0]
    if len(free) == 1 and isinstance(free[0], list) and (not free[0] or isinstance(free[0][0], (list, tuple))):
        free = free[0]
    for item in horizontal:
        if len(item) < 4:
            continue
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in item[:4]):
            x1, x2, y1, y2 = [float(v) for v in item[:4]]
        else:
            pts = np.array(item, dtype=float).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
        boxes.append([x1, y1, x2, y2])
    for item in free:
        try:
            pts = np.array(item, dtype=float).reshape(-1, 2)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
    clean = []
    for x1, y1, x2, y2 in boxes:
        x1 = int(max(0, min(width - 1, round(x1))))
        x2 = int(max(0, min(width - 1, round(x2))))
        y1 = int(max(0, min(height - 1, round(y1))))
        y2 = int(max(0, min(height - 1, round(y2))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            clean.append([x1, y1, x2, y2])
    clean.sort(key=lambda b: (b[1], b[0]))
    return clean


class OCRWorker(QThread):
    status_ready = pyqtSignal(str)
    result_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.queue = Queue(maxsize=200)
        self.running = True
        self.reader = None
        self.parseq = None
        self.transform = None
        self.device = "cpu"

    def submit(self, message):
        try:
            self.queue.put_nowait(message)
        except Exception:
            self.status_ready.emit("OCR queue full; crop skipped")

    def stop(self):
        self.running = False
        self.wait(2000)

    def run(self):
        try:
            self.load_models()
        except Exception as e:
            self.status_ready.emit(f"Model load failed: {e}")
            return
        while self.running:
            try:
                message = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                result = self.process(message)
                self.result_ready.emit(result)
            except Exception as e:
                self.status_ready.emit(f"OCR failed: {e}")

    def load_models(self):
        self.status_ready.emit("Loading EasyOCR detector...")
        import torch
        import easyocr

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reader = easyocr.Reader(["en"], gpu=self.device == "cuda", detector=True, recognizer=False, verbose=False)
        self.status_ready.emit("Loading PARSeq recognizer...")
        try:
            self.parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(self.device)
        except TypeError:
            self.parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval().to(self.device)
        self.transform = self.make_parseq_transform()
        self.status_ready.emit(f"Models ready on {self.device}")

    def make_parseq_transform(self):
        try:
            from strhub.data.module import SceneTextDataModule
            return SceneTextDataModule.get_transform(self.parseq.hparams.img_size)
        except Exception:
            from torchvision import transforms as T
            img_size = tuple(self.parseq.hparams.img_size)
            return T.Compose([
                T.Resize(img_size, T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ])

    def recognize_line(self, pil_image):
        import torch
        image = pil_image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.parseq(tensor)
            probs = logits.softmax(-1)
            labels, confidences = self.parseq.tokenizer.decode(probs)
        text = labels[0] if labels else ""
        conf = confidences[0]
        if hasattr(conf, "detach"):
            conf = conf.detach().float().cpu()
            confidence = float(conf.mean().item()) if conf.numel() else 0.0
        elif isinstance(conf, (list, tuple)) and conf:
            confidence = float(sum(float(v) for v in conf) / len(conf))
        else:
            confidence = float(conf) if conf is not None else 0.0
        return text, confidence

    def process(self, message):
        raw = base64.b64decode(message["image_jpeg_b64"])
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        np_img = np.array(pil)
        h, w = np_img.shape[:2]
        t0 = time.time()
        horizontal, free = self.reader.detect(
            np_img,
            min_size=0,
            text_threshold=0.36,
            low_text=0.18,
            link_threshold=0.18,
            canvas_size=1280,
            mag_ratio=1.0,
            slope_ths=0.1,
            ycenter_ths=0.5,
            height_ths=0.0,
            width_ths=0.0,
            add_margin=0.12,
            reformat=True,
            threshold=0.0,
            bbox_min_score=0.0,
            bbox_min_size=0,
            max_candidates=0,
        )
        boxes = as_easyocr_boxes(horizontal, free, w, h)
        lines = []
        for box in boxes:
            x1, y1, x2, y2 = box
            pad = max(2, int(0.03 * max(x2 - x1, y2 - y1)))
            crop = pil.crop((max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)))
            text, confidence = self.recognize_line(crop)
            if text.strip():
                lines.append({"box": box, "text": text, "confidence": confidence})
        elapsed = (time.time() - t0) * 1000.0
        return {
            "track_id": message.get("track_id", -1),
            "frame_index": message.get("frame_index", -1),
            "frame_acquired": message.get("frame_acquired", message.get("frame_index", -1)),
            "score": message.get("score", 0.0),
            "box": message.get("box", []),
            "source": message.get("source", "sam_client"),
            "timestamp": message.get("timestamp", time.time()),
            "crop_b64": message["image_jpeg_b64"],
            "line_boxes": boxes,
            "lines": lines,
            "elapsed_ms": elapsed,
            "width": w,
            "height": h,
        }


def pixmap_from_pil(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.results = []
        self.follow_latest = True
        self.displayed_result = None
        self.setWindowTitle("Poster Reader - OCR Viewer")
        self.preview = QLabel("Waiting for crops")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(520, 360)
        self.preview.setStyleSheet("background: black; color: white;")
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Time", "Track", "Score", "Lines", "Text", "ms", "Frame"])
        self.table.cellClicked.connect(self.show_row)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.status = QLabel("Starting...")
        self.status.setWordWrap(True)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_results)
        right = QVBoxLayout()
        right.addWidget(self.preview, 3)
        right.addWidget(QLabel("Recognized text"))
        right.addWidget(self.text, 1)
        left = QVBoxLayout()
        left.addWidget(self.table, 1)
        left.addWidget(self.clear_button)
        left.addWidget(self.status)
        root = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(left_widget, 1)
        root.addWidget(right_widget, 2)
        widget = QWidget()
        widget.setLayout(root)
        self.setCentralWidget(widget)
        self.setStyleSheet("QLabel, QTableWidget, QTextEdit { font-size: 11px; } QPushButton { font-size: 11px; padding: 3px 8px; }")
        self.ocr_worker = OCRWorker()
        self.ocr_worker.status_ready.connect(self.status.setText)
        self.ocr_worker.result_ready.connect(self.add_result)
        self.ocr_worker.start()
        self.receiver = SocketReceiver(args.host, args.port)
        self.receiver.message_ready.connect(self.receive_message)
        self.receiver.status_ready.connect(self.status.setText)
        self.receiver.start()

    def receive_message(self, message):
        self.ocr_worker.submit(message)

    def clear_results(self):
        self.results.clear()
        self.follow_latest = True
        self.displayed_result = None
        self.table.setRowCount(0)
        self.text.clear()
        self.preview.setText("Waiting for crops")

    def add_result(self, result):
        self.results.insert(0, result)
        self.table.insertRow(0)
        recognized = " ".join(line["text"] for line in result["lines"])
        values = [
            time.strftime("%H:%M:%S", time.localtime(result["timestamp"])),
            str(result["track_id"]),
            f"{float(result['score']):.2f}",
            str(len(result["lines"])),
            recognized[:120],
            f"{result['elapsed_ms']:.0f}",
            str(result.get("frame_acquired", result.get("frame_index", -1))),
        ]
        for col, value in enumerate(values):
            self.table.setItem(0, col, QTableWidgetItem(value))
        if self.follow_latest:
            self.table.blockSignals(True)
            self.table.selectRow(0)
            self.table.blockSignals(False)
            self.show_result(result)
        else:
            self.restore_displayed_selection()
        self.status.setText(f"Track {result['track_id']} | frame {result.get('frame_acquired', result.get('frame_index', -1))} | {len(result['line_boxes'])} detected lines | {len(result['lines'])} recognized")

    def show_row(self, row, col):
        if 0 <= row < len(self.results):
            self.follow_latest = False
            self.show_result(self.results[row])

    def restore_displayed_selection(self):
        if self.displayed_result is None:
            return
        for row, result in enumerate(self.results):
            if result is self.displayed_result:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                return

    def show_result(self, result):
        self.displayed_result = result
        raw = base64.b64decode(result["crop_b64"])
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.array(pil)
        for box in result.get("line_boxes", []):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pixmap = pixmap_from_pil(Image.fromarray(img))
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        lines = []
        for item in result.get("lines", []):
            lines.append(f"[{item['confidence']:.2f}] {item['text']}")
        self.text.setPlainText("\n".join(lines) if lines else "No text recognized")

    def closeEvent(self, event):
        self.receiver.stop()
        self.ocr_worker.stop()
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(1050, 640)
    window.show()
    sys.exit(app.exec())
