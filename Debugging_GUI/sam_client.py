import argparse
import base64
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread

import cv2
import numpy as np
import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget

class OCRBridge:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.queue = Queue(maxsize=200)
        self.running = True
        self.enabled = True
        self.host = "127.0.0.1"
        self.port = 8765
        self.script_path = Path(__file__).resolve().parent / "ocr_engine.py"
        self.proc = None
        self.sock = None
        self.lock = Lock()
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def set_config(self, enabled, host, port):
        with self.lock:
            self.enabled = enabled
            self.host = host
            self.port = int(port)

    def status(self, text):
        if self.status_callback:
            self.status_callback(text)

    def close(self):
        self.running = False
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    def ensure_app(self):
        if self.can_connect():
            return True
        if not self.script_path.exists():
            self.status(f"OCR app not found: {self.script_path}")
            return False
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen([sys.executable, str(self.script_path), "--host", self.host, "--port", str(self.port)])
            self.status("OCR app starting")
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if self.can_connect():
                self.status("OCR app connected")
                return True
            time.sleep(0.2)
        self.status("OCR app launch requested; still waiting")
        return False

    def can_connect(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=0.25):
                return True
        except Exception:
            return False

    def connect(self):
        if self.sock:
            return True
        self.sock = socket.create_connection((self.host, self.port), timeout=0.8)
        return True

    def submit(self, message):
        try:
            self.queue.put_nowait(message)
        except Exception:
            self.status("OCR send queue full; crop skipped")

    def run(self):
        while self.running:
            try:
                message = self.queue.get(timeout=0.1)
            except Empty:
                continue
            with self.lock:
                enabled = self.enabled
                self.host = self.host.strip() or "127.0.0.1"
            if not enabled:
                continue
            try:
                if not self.sock:
                    if not self.ensure_app():
                        continue
                    self.connect()
                data = json.dumps(message).encode("utf-8")
                packet = struct.pack("!I", len(data)) + data
                self.sock.sendall(packet)
            except Exception as e:
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self.status(f"OCR send failed: {e}")


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    status_ready = pyqtSignal(str)

    def __init__(self, server_url, camera_index, prompt, conf, track_iou, max_misses, duplicate_iou, jpeg_quality, max_width, target_fps, timeout):
        super().__init__()
        self.lock = Lock()
        self.server_url = server_url.rstrip("/")
        self.camera_index = camera_index
        self.prompt = prompt
        self.conf = conf
        self.track_iou = track_iou
        self.max_misses = max_misses
        self.duplicate_iou = duplicate_iou
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self.target_fps = target_fps
        self.timeout = timeout
        self.running = True
        self.paused = False
        self.frame_acquired = 0

    def update_config(self, server_url, prompt, conf, track_iou, max_misses, duplicate_iou, jpeg_quality, max_width, target_fps, timeout):
        with self.lock:
            self.server_url = server_url.rstrip("/")
            self.prompt = prompt
            self.conf = conf
            self.track_iou = track_iou
            self.max_misses = max_misses
            self.duplicate_iou = duplicate_iou
            self.jpeg_quality = jpeg_quality
            self.max_width = max_width
            self.target_fps = target_fps
            self.timeout = timeout

    def set_paused(self, value):
        self.paused = value

    def stop(self):
        self.running = False
        self.wait(2000)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status_ready.emit("Camera failed to open")
            return
        last_send = 0.0
        while self.running:
            if self.paused:
                self.msleep(40)
                continue
            ok, frame = cap.read()
            if not ok:
                self.status_ready.emit("Camera frame failed")
                self.msleep(100)
                continue
            self.frame_acquired += 1
            with self.lock:
                server_url = self.server_url
                prompt = self.prompt
                conf = self.conf
                track_iou = self.track_iou
                max_misses = self.max_misses
                duplicate_iou = self.duplicate_iou
                jpeg_quality = self.jpeg_quality
                max_width = self.max_width
                target_fps = self.target_fps
                timeout = self.timeout
            if max_width > 0 and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            period = 1.0 / max(1, target_fps)
            now = time.time()
            if now - last_send < period:
                self.msleep(2)
                continue
            last_send = now
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                continue
            try:
                t0 = time.time()
                response = requests.post(
                    f"{server_url}/infer",
                    files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
                    data={
                        "prompt": prompt,
                        "conf": str(conf),
                        "track_iou": str(track_iou),
                        "max_misses": str(max_misses),
                        "duplicate_iou": str(duplicate_iou),
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                payload["frame_acquired"] = self.frame_acquired
                payload["client_ms"] = (time.time() - t0) * 1000.0
                self.frame_ready.emit(frame, payload)
            except Exception as e:
                self.status_ready.emit(str(e))
                self.frame_ready.emit(frame, {"tracks": [], "error": str(e), "frame_acquired": self.frame_acquired})
        cap.release()


def color_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 1009 + 17)
    color = rng.integers(80, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_tracks(frame, payload):
    out = frame.copy()
    for item in payload.get("tracks", []):
        box = item.get("box", [0, 0, 0, 0])
        track_id = item.get("track_id", -1)
        score = item.get("score", 0.0)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = color_for_id(track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"ID {track_id} {score:.2f}"
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - size[1] - 8)), (x1 + size[0] + 8, y1), color, -1)
        cv2.putText(out, text, (x1 + 4, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def frame_to_pixmap(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.worker = None
        self.paused = False
        self.last_ocr_send = {}
        self.ocr_bridge = OCRBridge(status_callback=self.set_status)
        self.setWindowTitle("Poster Reader - Client")
        self.video = QLabel("Camera stopped")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(700, 394)
        self.video.setStyleSheet("background: black; color: white;")
        self.status = QLabel("Idle")
        self.status.setWordWrap(True)
        self.url = QLineEdit(args.server)
        self.prompt = QLineEdit(args.prompt)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)
        self.camera_index.setValue(args.camera)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.0, 1.0)
        self.conf.setSingleStep(0.01)
        self.conf.setDecimals(2)
        self.conf.setValue(args.conf)
        self.track_iou = QDoubleSpinBox()
        self.track_iou.setRange(0.0, 1.0)
        self.track_iou.setSingleStep(0.01)
        self.track_iou.setDecimals(2)
        self.track_iou.setValue(args.track_iou)
        self.duplicate_iou = QDoubleSpinBox()
        self.duplicate_iou.setRange(0.0, 1.0)
        self.duplicate_iou.setSingleStep(0.01)
        self.duplicate_iou.setDecimals(2)
        self.duplicate_iou.setValue(args.duplicate_iou)
        self.max_misses = QSpinBox()
        self.max_misses.setRange(0, 200)
        self.max_misses.setValue(args.max_misses)
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(20, 100)
        self.jpeg_quality.setValue(args.jpeg_quality)
        self.max_width = QSpinBox()
        self.max_width.setRange(0, 4096)
        self.max_width.setValue(args.max_width)
        self.target_fps = QSpinBox()
        self.target_fps.setRange(1, 60)
        self.target_fps.setValue(args.fps)
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(0.2, 60.0)
        self.timeout.setSingleStep(0.2)
        self.timeout.setDecimals(1)
        self.timeout.setValue(args.timeout)
        self.ocr_enabled = QCheckBox()
        self.ocr_enabled.setChecked(args.ocr_enabled)
        self.ocr_host = QLineEdit(args.ocr_host)
        self.ocr_port = QSpinBox()
        self.ocr_port.setRange(1024, 65535)
        self.ocr_port.setValue(args.ocr_port)
        self.ocr_interval = QDoubleSpinBox()
        self.ocr_interval.setRange(0.2, 60.0)
        self.ocr_interval.setSingleStep(0.2)
        self.ocr_interval.setDecimals(1)
        self.ocr_interval.setValue(args.ocr_interval)
        self.ocr_crop_quality = QSpinBox()
        self.ocr_crop_quality.setRange(20, 100)
        self.ocr_crop_quality.setValue(args.ocr_jpeg_quality)
        self.setStyleSheet("""
            QLabel { font-size: 10px; }
            QLineEdit, QSpinBox, QDoubleSpinBox { font-size: 10px; min-height: 18px; max-height: 22px; padding: 1px 3px; }
            QPushButton { font-size: 10px; padding: 2px 6px; min-height: 20px; }
            QCheckBox { font-size: 10px; }
        """)
        self.make_compact(self.url, 300)
        self.make_compact(self.prompt, 150)
        self.make_compact(self.ocr_host, 100)
        for control in [self.camera_index, self.conf, self.track_iou, self.duplicate_iou, self.max_misses, self.jpeg_quality, self.max_width, self.target_fps, self.timeout, self.ocr_port, self.ocr_interval, self.ocr_crop_quality]:
            self.make_compact(control, 82)
        settings = QFormLayout()
        settings.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        settings.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        settings.setHorizontalSpacing(6)
        settings.setVerticalSpacing(2)
        self.add_setting(settings, "Server", "Colab/Cloudflare URL.", self.url)
        self.add_setting(settings, "Prompt", "Text prompt for SAM.", self.prompt)
        self.add_setting(settings, "Camera", "Local camera index, usually 0.", self.camera_index)
        self.add_setting(settings, "Conf", "Minimum detection score.", self.conf)
        self.add_setting(settings, "Track IoU", "Lower keeps IDs through more motion.", self.track_iou)
        self.add_setting(settings, "Duplicate IoU", "Higher suppresses more duplicate boxes.", self.duplicate_iou)
        self.add_setting(settings, "Max misses", "Frames before a missing track is dropped.", self.max_misses)
        self.add_setting(settings, "JPEG quality", "Lower uploads faster; higher preserves detail.", self.jpeg_quality)
        self.add_setting(settings, "Max width", "Resize before sending; lower is smoother.", self.max_width)
        self.add_setting(settings, "FPS", "How often frames are sent to SAM.", self.target_fps)
        self.add_setting(settings, "Timeout", "Seconds before a request fails.", self.timeout)
        self.add_setting(settings, "OCR on", "Start and send crops to the OCR app.", self.ocr_enabled)
        self.add_setting(settings, "OCR host", "OCR app socket host.", self.ocr_host)
        self.add_setting(settings, "OCR port", "OCR app socket port.", self.ocr_port)
        self.add_setting(settings, "OCR interval", "Minimum seconds between OCR sends per track.", self.ocr_interval)
        self.add_setting(settings, "OCR JPEG", "JPEG quality for track crops sent to OCR.", self.ocr_crop_quality)
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")
        self.apply_button = QPushButton("Apply")
        self.ocr_button = QPushButton("Start OCR")
        self.start_button.clicked.connect(self.start_camera)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_camera)
        self.reset_button.clicked.connect(self.reset_tracking)
        self.apply_button.clicked.connect(self.apply_settings)
        self.ocr_button.clicked.connect(self.start_ocr_app)
        buttons = QGridLayout()
        buttons.setHorizontalSpacing(4)
        buttons.setVerticalSpacing(4)
        buttons.addWidget(self.start_button, 0, 0)
        buttons.addWidget(self.pause_button, 0, 1)
        buttons.addWidget(self.stop_button, 0, 2)
        buttons.addWidget(self.reset_button, 1, 0)
        buttons.addWidget(self.apply_button, 1, 1)
        buttons.addWidget(self.ocr_button, 1, 2)
        side = QVBoxLayout()
        side.setContentsMargins(4, 0, 0, 0)
        side.setSpacing(5)
        side.addLayout(settings)
        side.addLayout(buttons)
        side.addWidget(self.status)
        side.addStretch(1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMaximumWidth(390)
        root = QHBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
        root.addWidget(self.video, 1)
        root.addWidget(side_widget, 0)
        widget = QWidget()
        widget.setLayout(root)
        self.setCentralWidget(widget)
        self.apply_ocr_config()

    def make_compact(self, control, width):
        control.setFixedWidth(width)

    def add_setting(self, form, title, description, control):
        label = QLabel(title)
        label.setToolTip(description)
        control.setToolTip(description)
        form.addRow(label, control)

    def set_status(self, text):
        self.status.setText(text)

    def apply_ocr_config(self):
        self.ocr_bridge.set_config(self.ocr_enabled.isChecked(), self.ocr_host.text().strip(), self.ocr_port.value())

    def collect_config(self):
        return dict(
            server_url=self.url.text().strip(),
            prompt=self.prompt.text().strip(),
            conf=self.conf.value(),
            track_iou=self.track_iou.value(),
            max_misses=self.max_misses.value(),
            duplicate_iou=self.duplicate_iou.value(),
            jpeg_quality=self.jpeg_quality.value(),
            max_width=self.max_width.value(),
            target_fps=self.target_fps.value(),
            timeout=self.timeout.value(),
        )

    def start_ocr_app(self):
        self.apply_ocr_config()
        self.ocr_bridge.ensure_app()

    def start_camera(self):
        self.stop_camera()
        self.apply_ocr_config()
        if self.ocr_enabled.isChecked():
            self.ocr_bridge.ensure_app()
        cfg = self.collect_config()
        self.worker = CameraWorker(camera_index=self.camera_index.value(), **cfg)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_ready.connect(self.status.setText)
        self.worker.start()
        self.paused = False
        self.pause_button.setText("Pause")
        self.status.setText("Camera started")

    def apply_settings(self):
        self.apply_ocr_config()
        if self.worker:
            cfg = self.collect_config()
            self.worker.update_config(**cfg)
            self.status.setText("Settings applied")

    def toggle_pause(self):
        if not self.worker:
            return
        self.paused = not self.paused
        self.worker.set_paused(self.paused)
        self.pause_button.setText("Resume" if self.paused else "Pause")
        self.status.setText("Paused" if self.paused else "Running")

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.status.setText("Stopped")

    def reset_tracking(self):
        try:
            response = requests.post(f"{self.url.text().strip().rstrip('/')}/reset", timeout=self.timeout.value())
            response.raise_for_status()
            self.last_ocr_send.clear()
            self.status.setText("Tracking reset")
        except Exception as e:
            self.status.setText(str(e))

    def send_ocr_crops(self, frame, payload):
        if not self.ocr_enabled.isChecked():
            return
        self.apply_ocr_config()
        now = time.time()
        h, w = frame.shape[:2]
        interval = self.ocr_interval.value()
        for item in payload.get("tracks", []):
            track_id = int(item.get("track_id", -1))
            if track_id < 0:
                continue
            if now - self.last_ocr_send.get(track_id, 0.0) < interval:
                continue
            box = item.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            pad = max(4, int(0.04 * max(x2 - x1, y2 - y1)))
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(self.ocr_crop_quality.value())])
            if not ok:
                continue
            message = {
                "type": "crop",
                "source": "sam_client",
                "track_id": track_id,
                "frame_index": payload.get("frame_index", -1),
                "frame_acquired": payload.get("frame_acquired", -1),
                "score": float(item.get("score", 0.0)),
                "box": [x1, y1, x2, y2],
                "source_width": w,
                "source_height": h,
                "timestamp": now,
                "image_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
            self.ocr_bridge.submit(message)
            self.last_ocr_send[track_id] = now

    def on_frame(self, frame, payload):
        drawn = draw_tracks(frame, payload)
        pixmap = frame_to_pixmap(drawn)
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.send_ocr_crops(frame, payload)
        tracks = len(payload.get("tracks", []))
        server_ms = payload.get("server_ms", 0.0)
        client_ms = payload.get("client_ms", 0.0)
        device = payload.get("device", "")
        error = payload.get("error")
        if error:
            self.status.setText(error)
        else:
            self.status.setText(f"frame {payload.get('frame_acquired', -1)} | {tracks} tracks | server {server_ms:.1f} ms | round trip {client_ms:.1f} ms | {device}")

    def closeEvent(self, event):
        self.stop_camera()
        self.ocr_bridge.close()
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="poster, flyer")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--track-iou", type=float, default=0.60)
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument("--max-misses", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--ocr-host", default="127.0.0.1")
    parser.add_argument("--ocr-port", type=int, default=8765)
    parser.add_argument("--ocr-interval", type=float, default=2.0)
    parser.add_argument("--ocr-jpeg-quality", type=int, default=92)
    parser.add_argument("--no-ocr", dest="ocr_enabled", action="store_false")
    parser.set_defaults(ocr_enabled=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(1050, 640)
    window.show()
    sys.exit(app.exec())
