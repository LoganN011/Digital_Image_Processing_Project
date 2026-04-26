# video plays continuously
# every N frames gets submitted
# if Colab is busy, new frames are skipped
# latest boxes stay drawn until fresh boxes arrive

import sys, json, time
import cv2
import requests

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QLineEdit, QHBoxLayout, QSpinBox
)

# Update this to the quick Tunnel URL from cloudflare_colab.ipynb output
COLAB_URL = "https://claims-hanging-ourselves-discussion.trycloudflare.com"

DEFAULT_PROMPT = "flyers"
DEFAULT_EVERY_N = 1
DEFAULT_JPG_QUALITY = 25
DEFAULT_MAX_WIDTH = 400

def cv_to_pixmap(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def draw_boxes(pixmap, boxes):
    boxed = QPixmap(pixmap)
    painter = QPainter(boxed)
    pen = QPen(Qt.GlobalColor.green)
    pen.setWidth(4)
    painter.setPen(pen)

    for box in boxes:
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    painter.end()
    return boxed


def send_frame_to_colab(frame_bgr, prompt, jpg_quality, max_width):
    h, w = frame_bgr.shape[:2]

    scale = 1.0
    if w > max_width:
        scale = max_width / w
        frame_send = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    else:
        frame_send = frame_bgr

    ok, encoded = cv2.imencode(
        ".jpg",
        frame_send,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    )
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG")

    files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
    data = {"prompt": prompt}

    r = requests.post(f"{COLAB_URL}/infer", files=files, data=data, timeout=120)
    r.raise_for_status()
    result = r.json()

    boxes = result.get("boxes", [])

    # Scale boxes back to original frame size if we downscaled before sending
    if scale != 1.0:
        inv = 1.0 / scale
        for b in boxes:
            b["x1"] *= inv
            b["y1"] *= inv
            b["x2"] *= inv
            b["y2"] *= inv

    return result


class ColabInferenceWorker(QThread):
    result_ready = pyqtSignal(object)
    text_ready = pyqtSignal(str)

    def __init__(self, prompt, jpg_quality, max_width):
        super().__init__()
        self.prompt = prompt
        self.jpg_quality = jpg_quality
        self.max_width = max_width
        self.frame = None
        self.running = True
        self.busy = False

    def submit_frame(self, frame):
        if not self.busy:
            self.frame = frame.copy()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue

            frame = self.frame
            self.frame = None
            self.busy = True

            try:
                result = send_frame_to_colab(frame, self.prompt, self.jpg_quality, self.max_width)
                self.result_ready.emit(result)
                self.text_ready.emit(json.dumps(result, indent=2))
            except Exception as e:
                self.text_ready.emit(f"Colab request error:\n{e}")

            self.busy = False

class VideoPlayerWorker(QThread):
    frame_ready = pyqtSignal(object)
    frame_for_inference = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, video_path, send_every_n_frames, loop=True):
        super().__init__()
        self.video_path = video_path
        self.send_every_n_frames = send_every_n_frames
        self.loop = loop
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1.0 / fps
        frame_idx = 0

        while self.running:
            ret, frame = cap.read()

            if not ret:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                break

            self.frame_ready.emit(frame)

            if frame_idx % self.send_every_n_frames == 0:
                self.frame_for_inference.emit(frame)

            frame_idx += 1
            time.sleep(delay)

        cap.release()
        self.finished.emit()

class ColabMediaViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt → Colab SAM Video Viewer")

        self.video_worker = None
        self.infer_worker = None
        self.latest_boxes = []

        self.prompt_box = QLineEdit(DEFAULT_PROMPT)

        self.every_n_box = QSpinBox()
        self.every_n_box.setRange(1, 300)
        self.every_n_box.setValue(DEFAULT_EVERY_N)

        self.jpg_quality_box = QSpinBox()
        self.jpg_quality_box.setRange(1, 100)
        self.jpg_quality_box.setValue(DEFAULT_JPG_QUALITY)

        self.max_width_box = QSpinBox()
        self.max_width_box.setRange(160, 4096)
        self.max_width_box.setSingleStep(80)
        self.max_width_box.setValue(DEFAULT_MAX_WIDTH)
        
        self.current_every_n = self.every_n_box.value()
        self.current_jpg_quality = self.jpg_quality_box.value()
        self.current_max_width = self.max_width_box.value()

        self.every_n_box.valueChanged.connect(self.update_live_params)
        self.jpg_quality_box.valueChanged.connect(self.update_live_params)
        self.max_width_box.valueChanged.connect(self.update_live_params)

        self.open_button = QPushButton("Open Video")
        self.open_button.clicked.connect(self.open_media)

        self.stop_button = QPushButton("Stop Video")
        self.stop_button.clicked.connect(self.stop_video)

        self.image_label = QLabel("No video loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(420)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Prompt:"))
        controls.addWidget(self.prompt_box)
        controls.addWidget(QLabel("Send every N frames:"))
        controls.addWidget(self.every_n_box)
        controls.addWidget(QLabel("JPG quality:"))
        controls.addWidget(self.jpg_quality_box)
        controls.addWidget(QLabel("Max width:"))
        controls.addWidget(self.max_width_box)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(buttons)
        layout.addWidget(self.image_label)
        layout.addWidget(self.output)
        self.setLayout(layout)

    def open_media(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose video",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv)"
        )
        if not path:
            return

        self.process_video(path)

    def update_live_params(self):
        self.current_every_n = self.every_n_box.value()
        self.current_jpg_quality = self.jpg_quality_box.value()
        self.current_max_width = self.max_width_box.value()

        if self.video_worker is not None:
            self.video_worker.send_every_n_frames = self.current_every_n

        if self.infer_worker is not None:
            self.infer_worker.jpg_quality = self.current_jpg_quality
            self.infer_worker.max_width = self.current_max_width

    def process_video(self, path):
        self.stop_video()

        prompt = self.prompt_box.text().strip() or "posters, flyers"
        every_n = self.every_n_box.value()

        self.latest_boxes = []

        self.update_live_params()

        self.infer_worker = ColabInferenceWorker(
            prompt,
            self.current_jpg_quality,
            self.current_max_width,
        )
        self.infer_worker.result_ready.connect(self.handle_inference_result)
        self.infer_worker.text_ready.connect(self.output.setPlainText)
        self.infer_worker.start()

        self.video_worker = VideoPlayerWorker(path, every_n, loop=True)
        self.video_worker.frame_ready.connect(self.update_video_frame)
        self.video_worker.frame_for_inference.connect(self.infer_worker.submit_frame)
        self.video_worker.finished.connect(lambda: self.output.append("\nVideo stopped."))
        self.video_worker.start()

    def handle_inference_result(self, result):
        self.latest_boxes = result.get("boxes", [])

    def update_video_frame(self, frame_bgr):
        pixmap = cv_to_pixmap(frame_bgr)
        boxed = draw_boxes(pixmap, self.latest_boxes)
        self.show_pixmap(boxed)

    def stop_video(self):
        if hasattr(self, "video_worker") and self.video_worker is not None:
            if self.video_worker.isRunning():
                self.video_worker.stop()
                self.video_worker.wait()
            self.video_worker = None

        if hasattr(self, "infer_worker") and self.infer_worker is not None:
            if self.infer_worker.isRunning():
                self.infer_worker.stop()
                self.infer_worker.wait()
            self.infer_worker = None

    def show_pixmap(self, pixmap):
        scaled = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = ColabMediaViewer()
    window.resize(900, 750)
    window.show()
    sys.exit(app.exec())