# UNM CS591 Digital Image Processing Final Project
# Poster Reader GUI

from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import queue
import threading

import cv2
from PIL import Image
from PyQt6.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal, QEvent
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QScrollArea,
    QGridLayout,
    QStackedWidget,
    QDialog,
    QProgressBar,
)

from audio_engine import AudioEngine
from caption_engine import ImageCaptioner
from ocr_engine import OCREngine

try:
    from sam_engine import SAMWorker
except ImportError:
    SAMWorker = None

try:
    from dino_engine import DINOWorker
except ImportError:
    DINOWorker = None

try:
    from yolo_engine import YOLOWorker
except ImportError:
    YOLOWorker = None

# For quick debugging and testing.
LOAD_POSTERS_START_DIR = Path(__file__).resolve().parent    
LOAD_VIDEO_START_DIR = Path(__file__).resolve().parent

# Choose which detector worker to use for full video processing.
# Valid values: "dino" or "sam" or "yolo".
MODEL_BACKEND = "yolo"

GALLERY_CARD_WIDTH = 230
GALLERY_COLS = 4
OCR_WORKER_COUNT = 2


class ZoomableImageLabel(QLabel):
    """A QLabel that supports zoom (scroll wheel) and pan (click + drag)."""

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._drag_start = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self._update_display()

    def _update_display(self):
        if self._original_pixmap.isNull():
            return

        scaled_w = max(1, int(self._original_pixmap.width() * self._zoom))
        scaled_h = max(1, int(self._original_pixmap.height() * self._zoom))
        scaled = self._original_pixmap.scaled(
            scaled_w,
            scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        canvas = QPixmap(self.size())
        canvas.fill(Qt.GlobalColor.black)
        painter = QPainter(canvas)
        x = (self.width() - scaled.width()) // 2 + self._pan_offset.x()
        y = (self.height() - scaled.height()) // 2 + self._pan_offset.y()
        painter.drawPixmap(x, y, scaled)
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * 1.15, 10.0)
        else:
            self._zoom = max(self._zoom / 1.15, 0.1)
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._pan_offset += delta
            self._drag_start = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        self._drag_start = None

    def keyPressEvent(self, event):
        """Handle keyboard zoom (+/-) and pan (arrow keys) in the viewer."""
        key = event.key()
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self._zoom = min(self._zoom * 1.15, 10.0)
            self._update_display()
        elif key == Qt.Key.Key_Minus:
            self._zoom = max(self._zoom / 1.15, 0.1)
            self._update_display()
        elif key == Qt.Key.Key_Left:
            self._pan_offset += QPoint(30, 0)
            self._update_display()
        elif key == Qt.Key.Key_Right:
            self._pan_offset += QPoint(-30, 0)
            self._update_display()
        elif key == Qt.Key.Key_Up:
            self._pan_offset += QPoint(0, 30)
            self._update_display()
        elif key == Qt.Key.Key_Down:
            self._pan_offset += QPoint(0, -30)
            self._update_display()
        elif key == Qt.Key.Key_0:
            self._zoom = 1.0
            self._pan_offset = QPoint(0, 0)
            self._update_display()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class VideoSourceManager:
    def __init__(self):
        self.cap = None

    def select_file(self):
        """Opens a dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Video",
            str(LOAD_VIDEO_START_DIR),
            "Video Files (*.mp4 *.avi *.mov)",
        )
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            return self.cap, file_path
        return None, None

    def start_camera(self):
        """Initializes the webcam for recording/live processing."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(None, "Error", "Could not access the camera.")
            return None
        return self.cap


class OCRWorkerPool(QThread):

    ocr_ready = pyqtSignal(object, str, object)
    ocr_progress = pyqtSignal(object, object)

    def __init__(self, worker_count=2, engine_factory=OCREngine):
        super().__init__()
        self.worker_count = max(1, int(worker_count))
        self.engine_factory = engine_factory
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.pending_jobs = {}
        self.stopping = False
        self.worker_threads = []

    def add_task(self, poster_key, image_input, preprocess_options, generation):
        if self.stopping:
            return

        with self.lock:
            self.pending_jobs[poster_key] = (
                poster_key,
                image_input,
                dict(preprocess_options or {}),
                generation,
            )

        self.ocr_progress.emit(
            poster_key,
            {
                "message": "OCR queued",
                "percent": 0,
                "stage": "queued",
                "generation": generation,
            },
        )
        self.queue.put(poster_key)

    def stop(self):
        self.stopping = True
        for _ in range(self.worker_count):
            self.queue.put(None)

    def run(self):
        self.worker_threads = []
        for worker_index in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_index + 1,),
                daemon=True,
            )
            thread.start()
            self.worker_threads.append(thread)

        for thread in self.worker_threads:
            thread.join()

    def _worker_loop(self, worker_number):
        try:
            engine = self.engine_factory()
        except Exception as exc:
            engine = None
            engine_error = str(exc)
        else:
            engine_error = None

        while True:
            poster_key = self.queue.get()

            if poster_key is None:
                self.queue.task_done()
                break

            with self.lock:
                job = self.pending_jobs.pop(poster_key, None)

            # Duplicate queue token for a job already superseded by a newer crop.
            if job is None:
                self.queue.task_done()
                continue

            poster_key, image_input, preprocess_options, generation = job

            if engine is None:
                self.ocr_progress.emit(
                    poster_key,
                    {
                        "message": f"OCR worker {worker_number} unavailable: {engine_error}",
                        "percent": 100,
                        "stage": "error",
                        "generation": generation,
                    },
                )
                self.ocr_ready.emit(
                    poster_key,
                    f"OCR Error: {engine_error}",
                    (None, generation),
                )
                self.queue.task_done()
                continue

            def progress_callback(info, key=poster_key, gen=generation, num=worker_number):
                payload = dict(info or {})
                payload["generation"] = gen
                message = str(payload.get("message", "OCR processing"))
                payload["message"] = f"W{num}: {message}"
                self.ocr_progress.emit(key, payload)

            preprocess_options["progress_callback"] = progress_callback

            try:
                self.ocr_progress.emit(
                    poster_key,
                    {
                        "message": f"W{worker_number}: OCR starting",
                        "percent": 2,
                        "stage": "starting",
                        "generation": generation,
                    },
                )
                detected_text, avg_conf = engine.get_text(
                    image_input,
                    preprocess_options,
                )
                detected_text = detected_text or "(no text)"
                self.ocr_ready.emit(
                    poster_key,
                    detected_text,
                    (avg_conf, generation),
                )
            except Exception as e:
                self.ocr_progress.emit(
                    poster_key,
                    {
                        "message": f"W{worker_number}: OCR error: {e}",
                        "percent": 100,
                        "stage": "error",
                        "generation": generation,
                    },
                )
                self.ocr_ready.emit(
                    poster_key,
                    f"OCR Error: {e}",
                    (None, generation),
                )
            finally:
                self.queue.task_done()

class CaptionWorker(QThread):
    """Runs BLIP caption generation away from the GUI thread.

    Uses one newest pending caption job per poster key. It also emits a fallback
    caption on errors so cards do not spin forever if BLIP fails on one crop.
    """

    caption_ready = pyqtSignal(object, str)
    caption_progress = pyqtSignal(object, object)

    def __init__(self, captioner):
        super().__init__()
        self.captioner = captioner
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.pending_jobs = {}
        self.stopping = False

    def add_task(self, poster_key, pil_image):
        safe_image = self.prepare_caption_image(pil_image)
        with self.lock:
            self.pending_jobs[poster_key] = (poster_key, safe_image)
        self.caption_progress.emit(
            poster_key,
            {"message": "Caption queued", "percent": 0, "stage": "queued"},
        )
        self.queue.put(poster_key)

    def stop(self):
        self.stopping = True
        self.queue.put(None)

    def prepare_caption_image(self, pil_image):
        """Copy and lightly downscale images before BLIP to avoid huge slow jobs."""
        if pil_image is None:
            return None
        try:
            img = pil_image.convert("RGB").copy()
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            img.thumbnail((768, 768), resampling)
            return img
        except Exception:
            return pil_image

    def run(self):
        while True:
            poster_key = self.queue.get()

            if poster_key is None:
                self.queue.task_done()
                break

            with self.lock:
                job = self.pending_jobs.pop(poster_key, None)

  
            if job is None:
                self.queue.task_done()
                continue

            poster_key, pil_image = job

            try:
                self.caption_progress.emit(
                    poster_key,
                    {"message": "Captioning image", "percent": 35, "stage": "captioning"},
                )
                if pil_image is None:
                    caption = "Description unavailable."
                else:
                    caption = self.captioner.generate_caption(pil_image)
                    caption = str(caption or "").strip() or "(no description)"
                self.caption_progress.emit(
                    poster_key,
                    {"message": "Caption done", "percent": 100, "stage": "done"},
                )
                self.caption_ready.emit(poster_key, caption)
            except Exception as e:
                print(f"Caption error for {poster_key}: {e}")
                self.caption_progress.emit(
                    poster_key,
                    {"message": f"Caption error: {e}", "percent": 100, "stage": "error"},
                )
                self.caption_ready.emit(
                    poster_key,
                    f"Description unavailable: {e}",
                )
            finally:
                self.queue.task_done()


class PosterReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.manager = VideoSourceManager()
        self.current_cap = None
        self.found_posters = []
        self.is_live_camera = False
        self.video_path = None
        self.temp_video_writer = None
        self.model_worker = None

        self.setWindowTitle("UNM Poster Reader")
        self.resize(1000, 700)

        self.stack = QStackedWidget()
        self.init_menu_screen()
        self.init_processing_screen()
        self.init_results_screen()

        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        self.speaker = AudioEngine()
        self.captioner = ImageCaptioner()
        self.ocr_engine = OCREngine()

        self.ocr_generation = 0
        self.ocr_worker = OCRWorkerPool(worker_count=OCR_WORKER_COUNT)
        self.ocr_worker.ocr_ready.connect(self.on_ocr_ready)
        self.ocr_worker.ocr_progress.connect(self.on_ocr_progress)
        self.ocr_worker.start()

        self.poster_id_to_button = {}
        self._selecting_poster = False
        self.caption_worker = CaptionWorker(self.captioner)
        self.caption_worker.caption_ready.connect(self.on_caption_ready)
        self.caption_worker.caption_progress.connect(self.on_caption_progress)
        self.caption_worker.start()

    def init_menu_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        btn_file = QPushButton("Select Video File  [1]")
        btn_file.clicked.connect(self.run_file_input)

        btn_cam = QPushButton("Start Live Record  [2]")
        btn_cam.clicked.connect(self.run_camera_input)

        layout.addWidget(btn_file)
        layout.addWidget(btn_cam)
        self.stack.addWidget(page)

        shortcut1 = QShortcut(QKeySequence("1"), self)
        shortcut1.activated.connect(self.run_file_input)
        shortcut2 = QShortcut(QKeySequence("2"), self)
        shortcut2.activated.connect(self.run_camera_input)

    def run_file_input(self):
        if self.stack.currentIndex() != 0:
            return
        cap, path = self.manager.select_file()
        if cap:
            self.current_cap = cap
            self.is_live_camera = False
            self.video_path = path
            self.start_pipeline()

    def run_camera_input(self):
        if self.stack.currentIndex() != 0:
            return
        cap = self.manager.start_camera()
        if cap:
            self.current_cap = cap
            self.is_live_camera = True
            self.video_path = "temp_record.mp4"

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self.temp_video_writer = cv2.VideoWriter(
                self.video_path, fourcc, fps, (width, height)
            )

            self.start_pipeline()

    def start_pipeline(self):
        self.stack.setCurrentIndex(1)
        self.timer.start(30)

    def init_processing_screen(self):
        page = QWidget()
        self.proc_layout = QVBoxLayout(page)
        self.video_label = QLabel("Loading Video Feed...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_stop = QPushButton("Finish and View Posters  [Space / Esc]")
        btn_stop.clicked.connect(self.stop_processing)

        self.proc_layout.addWidget(self.video_label)
        self.proc_layout.addWidget(btn_stop)
        self.stack.addWidget(page)

        QShortcut(QKeySequence("Space"), self).activated.connect(self.stop_processing)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.stop_processing)

    def process_frame(self):
        ret, frame = self.current_cap.read()
        if not ret:
            self.stop_processing()
            return

        if self.is_live_camera and self.temp_video_writer is not None:
            self.temp_video_writer.write(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                800, 600, Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def stop_processing(self):
        if self.stack.currentIndex() != 1:
            return
        self.timer.stop()

        if self.current_cap:
            self.current_cap.release()
            self.current_cap = None

        if self.is_live_camera and self.temp_video_writer is not None:
            self.temp_video_writer.release()
            self.temp_video_writer = None

        self.start_model_processing()

    def init_results_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.ocr_result_label = QLabel(
            "Press T to load test images. Navigate posters with Arrow Keys. "
            "Press Enter to open."
        )
        self.ocr_result_label.setWordWrap(True)
        layout.addWidget(self.ocr_result_label)

        self.ocr_global_status_label = QLabel(f"OCR/caption status: 0/0 ready | OCR workers: {OCR_WORKER_COUNT}")
        self.ocr_global_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_global_status_label.setWordWrap(True)
        layout.addWidget(self.ocr_global_status_label)

        scroll = QScrollArea()
        self.gallery_scroll = scroll
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self.grid_widget)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)

        btn_test = QPushButton("Test OCR: Load Local Images  [T]")
        btn_test.clicked.connect(self.load_test_posters)
        layout.addWidget(btn_test)

        self.results_status_label = QLabel("")
        self.results_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_status_label.setVisible(False)
        layout.addWidget(self.results_status_label)

        self.results_progress_bar = QProgressBar()
        self.results_progress_bar.setRange(0, 100)
        self.results_progress_bar.setValue(0)
        self.results_progress_bar.setVisible(False)
        layout.addWidget(self.results_progress_bar)

        self.btn_stop_proc = QPushButton("Stop Processing  [X]")
        self.btn_stop_proc.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;"
        )
        self.btn_stop_proc.setVisible(False)
        self.btn_stop_proc.clicked.connect(self.stop_model_processing)
        layout.addWidget(self.btn_stop_proc)

        page.setLayout(layout)
        self.stack.addWidget(page)

        sc = QShortcut(QKeySequence("T"), self)
        sc.activated.connect(self.load_test_posters)
        sc = QShortcut(QKeySequence("X"), self)
        sc.activated.connect(self.stop_model_processing)
        sc = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        sc.activated.connect(lambda: self.move_poster_focus(1))
        sc = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        sc.activated.connect(lambda: self.move_poster_focus(-1))
        sc = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        sc.activated.connect(lambda: self.move_poster_focus(4))
        sc = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        sc.activated.connect(lambda: self.move_poster_focus(-4))
        sc = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        sc.activated.connect(self.open_focused_poster)
        sc = QShortcut(QKeySequence(Qt.Key.Key_Enter), self)
        sc.activated.connect(self.open_focused_poster)

        self.poster_buttons = []
        self.poster_data = []
        self.focused_poster_index = -1
        if hasattr(self, "ocr_global_status_label"):
            self.update_global_processing_status()

    def get_preprocess_options(self):
        """Return fixed OCR options. The GUI no longer exposes preprocessing controls."""
        return {}

    def start_model_processing(self):
        """Run the selected poster detector worker after video capture finishes."""
        self.clear_gallery()

        self.stack.setCurrentIndex(2)
        self.results_progress_bar.setVisible(True)
        self.results_status_label.setVisible(True)
        self.btn_stop_proc.setVisible(True)
        self.results_progress_bar.setValue(0)
        self.results_status_label.setText("Starting model processing...")
        self.ocr_result_label.setText(
            "Processing video. Detected posters will appear below as they are found."
        )

        if not self.video_path:
            self.on_error("No video path is available for model processing.")
            return

        if MODEL_BACKEND.lower() == "sam":
            worker_cls = SAMWorker
            backend_name = "SAM"
        elif MODEL_BACKEND.lower() == "dino":
            worker_cls = DINOWorker
            backend_name = "DINO"
        elif MODEL_BACKEND.lower() == "yolo":
            worker_cls = YOLOWorker
            backend_name = "YOLO"
        else:
            worker_cls = None
            backend_name = MODEL_BACKEND.upper()
  

        if worker_cls is None:
            self.on_error(
                f"{backend_name} worker is not available. Check that the engine file is in the GUI folder."
            )
            return

        self.model_worker = worker_cls(self.video_path)
        self.model_worker.progress.connect(self.on_progress)
        self.model_worker.poster_found.connect(self.on_poster_found)
        self.model_worker.finished_processing.connect(self.on_model_finished)
        self.model_worker.error.connect(self.on_error)
        self.model_worker.start()

    def stop_model_processing(self):
        """Requests the current vision engine worker to stop without freezing the GUI."""
        if hasattr(self, "model_worker") and self.model_worker is not None and self.model_worker.isRunning():
            self.results_status_label.setText("Stopping after current frame...")
            self.btn_stop_proc.setEnabled(False)

            if hasattr(self.model_worker, "request_stop"):
                self.model_worker.request_stop()
            else:
                # Fallback only if the worker has no cooperative stop support.
                self.model_worker.terminate()

        # Do not call wait() here. wait() blocks the GUI thread.

    def on_progress(self, pct, status):
        self.results_progress_bar.setValue(pct)
        self.results_status_label.setText(status)

    def on_error(self, err_msg):
        self.results_progress_bar.setVisible(False)
        self.btn_stop_proc.setVisible(False)
        self.results_status_label.setVisible(True)
        self.results_status_label.setText("Model processing failed.")
        QMessageBox.critical(self, "Model Error", err_msg)

    def on_poster_found(self, poster_id, image_array):
        """Callback for live updates when a new poster or a better crop is found."""
        preprocess_options = self.get_preprocess_options()

        pending_text = "OCR pending..."
        pending_conf = None

        if poster_id in self.poster_id_to_button:
            self.update_poster_in_gallery(
                poster_id,
                image_array,
                pending_text,
                pending_conf
            )
            btn = self.poster_id_to_button[poster_id]
            idx = self.poster_buttons.index(btn)
            poster_key = self.poster_data[idx]["poster_key"]
        else:
            self.add_poster_to_gallery(
                image_array,
                pending_text,
                pending_conf,
                poster_id=poster_id
            )
            btn = self.poster_id_to_button[poster_id]
            idx = self.poster_buttons.index(btn)
            poster_key = self.poster_data[idx]["poster_key"]

        self.ocr_worker.add_task(
            poster_key,
            image_array,
            preprocess_options,
            self.ocr_generation
        )
    def on_model_finished(self, results=None, output_path=None):
        self.results_progress_bar.setVisible(False)
        self.btn_stop_proc.setVisible(False)
        self.btn_stop_proc.setEnabled(True)
        self.results_status_label.setText(
            f"Processing complete/stopped. Found {len(self.poster_data)} posters. "
            "OCR/captions will continue updating until each card is ready."
        )

    def load_test_posters(self):
        """Manually select images to test the gallery and OCR display."""
        if self.stack.currentIndex() != 2:
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Test Posters",
            str(LOAD_POSTERS_START_DIR),
            "Images (*.png *.jpg *.jpeg)",
        )
        if not files:
            return

        preprocess_options = self.get_preprocess_options()

        import time

        for file_path in files:
            t0 = time.time()
            detected_text, avg_conf = self.ocr_engine.get_text(file_path, preprocess_options)
            print("OCR seconds:", time.time() - t0)

            t1 = time.time()
            self.add_poster_to_gallery(file_path, detected_text, avg_conf)
            print("Add/caption seconds:", time.time() - t1)

            QApplication.processEvents()

    def add_poster_to_gallery(
        self,
        image_input,
        detected_text,
        avg_conf=None,
        poster_id=None,
        run_caption_async=True,
    ):
        """Adds a clickable poster thumbnail to the results gallery."""
        btn = QPushButton()
        full_pixmap, pil_image = self.prepare_pixmap_and_pil(image_input)

        if full_pixmap.isNull():
            print("Warning: could not create pixmap for poster.")
            return

        self.set_button_preview(btn, image_input)

        btn.setStyleSheet(
            """
            QPushButton {
                border: 2px solid transparent;
                border-radius: 3px;
                background-color: transparent;
                padding: 2px;
            }
            QPushButton:hover, QPushButton:focus {
                border: 2px solid #0078d7;
                background-color: rgba(0, 120, 215, 20);
            }
            """
        )

        loading_bar = QProgressBar()
        loading_bar.setRange(0, 100)
        loading_bar.setValue(0)
        loading_bar.setTextVisible(True)
        loading_bar.setFormat("Queued")
        loading_bar.setFixedWidth(btn.width())

        card = QWidget()
        card.setFixedWidth(GALLERY_CARD_WIDTH)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(2, 2, 2, 2)
        card_layout.setSpacing(4)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        card_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(loading_bar, alignment=Qt.AlignmentFlag.AlignHCenter)

        if avg_conf is None:
            confidence_text = "Average OCR confidence: N/A"
        else:
            confidence_text = f"Average OCR confidence: {avg_conf:.2%}"

        poster_key = poster_id if poster_id is not None else f"manual_{len(self.poster_buttons)}"
        description = "Generating description..." if run_caption_async else "Processing..."
        btn.setToolTip(f"Description: {description}")

        self.poster_buttons.append(btn)
        self.poster_data.append(
            {
                "poster_key": poster_key,
                "poster_id": poster_id,
                "image_input": image_input,
                "detected_text": detected_text,
                "pixmap": full_pixmap,
                "description": description,
                "confidence": confidence_text,

                "ocr_done": detected_text not in {"OCR pending...", "OCR updating..."},
                "caption_done": not run_caption_async,
                "ocr_progress": 100 if detected_text not in {"OCR pending...", "OCR updating..."} else 0,
                "caption_progress": 0 if run_caption_async else 100,
                "ocr_stage": "OCR done" if detected_text not in {"OCR pending...", "OCR updating..."} else "OCR queued",
                "caption_stage": "Caption queued" if run_caption_async else "Caption done",
                "loading_bar": loading_bar,
                "card": card,
                "button": btn,
            }
        )

        poster = self.poster_data[-1]
        btn.setEnabled(poster["ocr_done"] and poster["caption_done"])
        self.update_poster_ready_state(len(self.poster_data) - 1)
        self.update_global_processing_status()

        btn.clicked.connect(lambda checked=False, b=btn: self.open_poster_by_button(b))
        btn.installEventFilter(self)

        if poster_id is not None:
            self.poster_id_to_button[poster_id] = btn

        count = self.grid_layout.count()
        self.grid_layout.addWidget(card, count // GALLERY_COLS, count % GALLERY_COLS)

        if self.focused_poster_index == -1:
            self.select_poster_index(0, scroll=False, focus=True)

        if run_caption_async:
            self.caption_worker.add_task(poster_key, pil_image)

    def update_poster_ready_state(self, idx):
        """Update per-card progress bar and visibility based on OCR/caption readiness."""
        if not (0 <= idx < len(self.poster_data)):
            return

        poster = self.poster_data[idx]
        ready = poster.get("ocr_done", False) and poster.get("caption_done", False)

        ocr_progress = int(poster.get("ocr_progress", 0))
        caption_progress = int(poster.get("caption_progress", 0))

        if not poster.get("ocr_done", False):
            combined = int(max(0, min(100, ocr_progress)) * 0.70)
            stage = poster.get("ocr_stage", "OCR queued")
        elif not poster.get("caption_done", False):
            combined = 70 + int(max(0, min(100, caption_progress)) * 0.30)
            stage = poster.get("caption_stage", "Caption queued")
        else:
            combined = 100
            stage = "Ready"

        loading_bar = poster.get("loading_bar")
        if loading_bar is not None:
            loading_bar.setRange(0, 100)
            loading_bar.setValue(max(0, min(100, combined)))
            loading_bar.setTextVisible(True)
            loading_bar.setFormat(stage)
            loading_bar.setVisible(not ready)

        btn = self.poster_buttons[idx]
        btn.setEnabled(True)
        self.update_global_processing_status()

    def update_poster_in_gallery(self, poster_id, image_input, detected_text, avg_conf=None):
        """Update an existing poster thumbnail/data when a better crop is found."""
        if avg_conf is None:
            confidence_text = "Average OCR confidence: N/A"
        else:
            confidence_text = f"Average OCR confidence: {avg_conf:.2%}"

        btn = self.poster_id_to_button.get(poster_id)
        if btn is None:
            return

        try:
            idx = self.poster_buttons.index(btn)
        except ValueError:
            return

        full_pixmap, pil_image = self.prepare_pixmap_and_pil(image_input)
        if full_pixmap.isNull():
            return

        poster = self.poster_data[idx]
        poster["image_input"] = image_input
        poster["detected_text"] = detected_text
        poster["pixmap"] = full_pixmap
        poster["confidence"] = confidence_text
        poster["description"] = "Updating description..."

        # Reset readiness because this is a newer/better crop.
        poster["ocr_done"] = detected_text not in {"OCR pending...", "OCR updating..."}
        poster["caption_done"] = False
        poster["ocr_progress"] = 100 if poster["ocr_done"] else 0
        poster["caption_progress"] = 0
        poster["ocr_stage"] = "OCR done" if poster["ocr_done"] else "OCR queued"
        poster["caption_stage"] = "Caption queued"

        self.set_button_preview(btn, image_input)
        btn.setToolTip("Description: Updating description...")
        self.caption_worker.add_task(poster["poster_key"], pil_image)

        self.update_poster_ready_state(idx)

        if self.focused_poster_index == idx:
            self.ocr_result_label.setText("Description/OCR updating...")
    def update_status_panel_for_index(self, idx):
        """Refresh the top text panel for the currently focused poster."""
        if not (0 <= idx < len(self.poster_data)):
            return
        poster = self.poster_data[idx]
        self.ocr_result_label.setText(
            f"Description: {poster['description']}\n\n"
            f"Detected Text: {poster['detected_text']}\n\n"
            f"Confidence: {poster['confidence']}"
        )

    def select_poster_index(self, idx, *, scroll=True, focus=True):
        """Make arrow keys, focus, mouse hover, and mouse click select the same card."""
        if not (0 <= idx < len(self.poster_data)):
            return

        self.focused_poster_index = idx
        btn = self.poster_buttons[idx]

        if focus and not self._selecting_poster:
            self._selecting_poster = True
            try:
                btn.setFocus()
            finally:
                self._selecting_poster = False

        self.update_status_panel_for_index(idx)

        if scroll and hasattr(self, "gallery_scroll"):
            card = self.poster_data[idx].get("card") or btn
            QTimer.singleShot(0, lambda w=card: self.gallery_scroll.ensureWidgetVisible(w, 24, 24))

    def update_global_processing_status(self):
        total = len(self.poster_data)
        if total == 0:
            self.ocr_global_status_label.setText(f"OCR/caption status: 0/0 ready | OCR workers: {OCR_WORKER_COUNT}")
            return

        ocr_done = sum(1 for p in self.poster_data if p.get("ocr_done", False))
        caption_done = sum(1 for p in self.poster_data if p.get("caption_done", False))
        ready = sum(1 for p in self.poster_data if p.get("ocr_done", False) and p.get("caption_done", False))
        ocr_pending = total - ocr_done
        caption_pending = total - caption_done
        self.ocr_global_status_label.setText(
            f"Ready {ready}/{total}  |  OCR {ocr_done}/{total} complete ({ocr_pending} pending)  |  "
            f"Captions {caption_done}/{total} complete ({caption_pending} pending)  |  "
            f"OCR workers: {OCR_WORKER_COUNT}"
        )

    def on_ocr_progress(self, poster_key, info):
        """Update per-card OCR progress from the OCR engine."""
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return

        if isinstance(info, dict):
            generation = info.get("generation")
            if generation is not None and generation != self.ocr_generation:
                return
            percent = int(info.get("percent", 0))
            message = str(info.get("message", "OCR processing"))
        else:
            percent = 0
            message = "OCR processing"

        poster = self.poster_data[idx]
        poster["ocr_progress"] = max(0, min(100, percent))
        poster["ocr_stage"] = f"OCR: {message}"
        self.update_poster_ready_state(idx)

        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def on_caption_progress(self, poster_key, info):
        """Update per-card caption progress."""
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return

        if isinstance(info, dict):
            percent = int(info.get("percent", 0))
            message = str(info.get("message", "Caption processing"))
        else:
            percent = 0
            message = "Caption processing"

        poster = self.poster_data[idx]
        poster["caption_progress"] = max(0, min(100, percent))
        poster["caption_stage"] = message
        self.update_poster_ready_state(idx)

        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def on_caption_ready(self, poster_key, description):
        """Update a gallery item once BLIP finishes captioning it."""
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return

        poster = self.poster_data[idx]
        poster["description"] = description or "(no description)"
        poster["caption_done"] = True
        poster["caption_progress"] = 100
        poster["caption_stage"] = "Caption done"

        btn = self.poster_buttons[idx]
        btn.setToolTip(f"Description: {poster['description']}")

        self.update_poster_ready_state(idx)

        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def find_poster_index_by_key(self, poster_key):
        for idx, poster in enumerate(self.poster_data):
            if poster.get("poster_key") == poster_key:
                return idx
        return None

    def clear_gallery(self):
        """Remove all poster thumbnails and reset gallery state."""
        self.poster_id_to_button = {}
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.poster_buttons = []
        self.poster_data = []
        self.focused_poster_index = -1
        if hasattr(self, "ocr_global_status_label"):
            self.update_global_processing_status()

    def prepare_pixmap_and_pil(self, image_input):
        """Convert a path or OpenCV BGR image into a QPixmap and PIL image."""
        if isinstance(image_input, str):
            full_pixmap = QPixmap(image_input)
            pil_image = Image.open(image_input).convert("RGB")
            return full_pixmap, pil_image

        rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        full_pixmap = QPixmap.fromImage(q_img.copy())
        pil_image = Image.fromarray(rgb_image)
        return full_pixmap, pil_image

    def make_display_pixmap(self, image_input):
        """Create a color QPixmap for thumbnails and the zoom viewer."""
        if isinstance(image_input, str):
            return QPixmap(image_input)

        if image_input is None:
            return QPixmap()

        rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img.copy())

    def make_processed_pixmap(self, image_input):
        """Compatibility wrapper: previews now show the original color crop."""
        return self.make_display_pixmap(image_input)

    def set_button_preview(self, btn, image_input):
        """Update one gallery button to show the original color crop."""
        preview_pixmap = self.make_display_pixmap(image_input)

        if preview_pixmap.isNull():
            if isinstance(image_input, str):
                preview_pixmap = QPixmap(image_input)
            else:
                rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                q_img = QImage(
                    rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888
                )
                preview_pixmap = QPixmap.fromImage(q_img.copy())

        thumb_max = 220
        thumbnail = preview_pixmap.scaled(
            thumb_max,
            thumb_max,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        btn.setIcon(QIcon(thumbnail))
        btn.setIconSize(thumbnail.size())
        btn.setFixedSize(thumbnail.width() + 8, thumbnail.height() + 8)

        for poster in self.poster_data:
            if poster.get("button") is btn and poster.get("loading_bar") is not None:
                poster["loading_bar"].setFixedWidth(btn.width())
                break

    def refresh_gallery_previews(self):
        """Kept for compatibility; preprocessing controls were removed."""
        return

    def on_ocr_ready(self, poster_key, detected_text, payload):
        """Update stored OCR result when background OCR finishes."""
        avg_conf, generation = payload

        # Ignore stale OCR results from an older preprocessing setting.
        if generation != self.ocr_generation:
            return

        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return

        poster = self.poster_data[idx]
        poster["detected_text"] = detected_text
        poster["confidence"] = (
            "Average OCR confidence: N/A"
            if avg_conf is None
            else f"Average OCR confidence: {avg_conf:.2%}"
        )

        poster["ocr_done"] = True
        poster["ocr_progress"] = 100
        poster["ocr_stage"] = "OCR done"

        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

        self.update_poster_ready_state(idx)

    def move_poster_focus(self, delta):
        """Move focus among poster thumbnails on the results screen."""
        if self.stack.currentIndex() != 2 or not self.poster_buttons:
            return

        if self.focused_poster_index < 0:
            idx = 0
        else:
            idx = self.focused_poster_index + delta

        idx = max(0, min(idx, len(self.poster_buttons) - 1))
        self.select_poster_index(idx, scroll=True, focus=True)

    def open_focused_poster(self):
        """Open the currently focused poster and read only detected text aloud."""
        if self.stack.currentIndex() != 2 or not self.poster_buttons:
            return

        if self.focused_poster_index < 0:
            self.focused_poster_index = 0
            self.poster_buttons[0].setFocus()

        self.open_poster_by_index(self.focused_poster_index)

    def open_poster_by_button(self, btn):
        try:
            idx = self.poster_buttons.index(btn)
        except ValueError:
            return
        self.select_poster_index(idx, scroll=False, focus=True)
        self.open_poster_by_index(idx)

    def open_poster_by_index(self, idx):
        if not (0 <= idx < len(self.poster_data)):
            return

        poster = self.poster_data[idx]

        if not (poster.get("ocr_done", False) and poster.get("caption_done", False)):
            self.ocr_result_label.setText("Poster is still processing. Please wait.")
            return
        text = poster["detected_text"]
        pixmap = poster["pixmap"]
        confidence = poster["confidence"]
        image_input = poster["image_input"]

        display_text = f"{text}\n\n{confidence}"
        self.ocr_result_label.setText(display_text)
        self.speaker.speak(text)

        color_pixmap = self.make_display_pixmap(image_input)
        self.show_zoom_dialog(color_pixmap, display_text)

    def eventFilter(self, obj, event):
        """Track which poster button is selected by keyboard focus or mouse hover."""
        if obj in self.poster_buttons:
            etype = event.type()
            if etype in (QEvent.Type.FocusIn, QEvent.Type.Enter, QEvent.Type.MouseButtonPress):
                idx = self.poster_buttons.index(obj)
                self.select_poster_index(idx, scroll=(etype == QEvent.Type.FocusIn), focus=(etype != QEvent.Type.FocusIn))
        return super().eventFilter(obj, event)

    def show_zoom_dialog(self, color_pixmap, text):
        """Open a zoomable viewer showing the original color crop."""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Poster Viewer — Color Image")
            dialog.resize(900, 700)

            main_layout = QVBoxLayout(dialog)

            title = QLabel("Color Image")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_view = ZoomableImageLabel(color_pixmap, dialog)
            image_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            main_layout.addWidget(title)
            main_layout.addWidget(image_view, stretch=1)

            text_label = QLabel(text)
            text_label.setWordWrap(True)
            text_label.setMaximumHeight(120)
            text_label.setStyleSheet(
                "padding: 8px; background-color: #1e1e1e; color: white;"
            )
            main_layout.addWidget(text_label)

            image_view.setFocus()
            dialog.exec()

        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Zoom Viewer Error", str(e))

    def closeEvent(self, event):
        if self.model_worker is not None and self.model_worker.isRunning():
            self.model_worker.terminate()
            self.model_worker.wait()
        
        self.caption_worker.stop()
        self.caption_worker.wait(1000)

        self.ocr_worker.stop()
        self.ocr_worker.wait(1000)
        
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosterReaderApp()
    window.show()
    sys.exit(app.exec())
