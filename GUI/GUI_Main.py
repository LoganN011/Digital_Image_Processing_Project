from __future__ import annotations

import queue
import sys
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
from PIL import Image
from PyQt6.QtCore import QEvent, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QHBoxLayout, QWidget

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

try:
    from maskrcnn_engine import MaskRCNNWorker
except ImportError:
    MaskRCNNWorker = None

SCRIPT_DIR = Path(__file__).resolve().parent
LOAD_POSTERS_START_DIR = SCRIPT_DIR
LOAD_VIDEO_START_DIR = SCRIPT_DIR
MODEL_BACKEND = "yolo"  # "sam", "dino", "yolo", "maskrcnn"
GALLERY_CARD_WIDTH = 230
GALLERY_THUMB_SIZE = 220
GALLERY_BUTTON_SIZE = 228
DETECTOR_PREVIEW_SIZE = (320, 180)
GALLERY_COLS = 4
OCR_WORKER_COUNT = 2
PENDING_TEXTS = {"OCR pending...", "OCR updating..."}


class ZoomableImageLabel(QLabel):
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
        scaled = self._original_pixmap.scaled(
            max(1, int(self._original_pixmap.width() * self._zoom)),
            max(1, int(self._original_pixmap.height() * self._zoom)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QPixmap(self.size())
        canvas.fill(Qt.GlobalColor.black)
        painter = QPainter(canvas)
        painter.drawPixmap((self.width() - scaled.width()) // 2 + self._pan_offset.x(), (self.height() - scaled.height()) // 2 + self._pan_offset.y(), scaled)
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event):
        self._zoom = min(self._zoom * 1.15, 10.0) if event.angleDelta().y() > 0 else max(self._zoom / 1.15, 0.1)
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._pan_offset += event.pos() - self._drag_start
            self._drag_start = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event):
        self._drag_start = None

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom = min(self._zoom * 1.15, 10.0)
        elif key == Qt.Key.Key_Minus:
            self._zoom = max(self._zoom / 1.15, 0.1)
        elif key == Qt.Key.Key_Left:
            self._pan_offset += QPoint(30, 0)
        elif key == Qt.Key.Key_Right:
            self._pan_offset += QPoint(-30, 0)
        elif key == Qt.Key.Key_Up:
            self._pan_offset += QPoint(0, 30)
        elif key == Qt.Key.Key_Down:
            self._pan_offset += QPoint(0, -30)
        elif key == Qt.Key.Key_0:
            self._zoom = 1.0
            self._pan_offset = QPoint(0, 0)
        else:
            super().keyPressEvent(event)
            return
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class VideoSourceManager:
    def __init__(self):
        self.cap = None

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(None, "Select Video", str(LOAD_VIDEO_START_DIR), "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return None, None
        self.cap = cv2.VideoCapture(path)
        return self.cap, path

    def start_camera(self):
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
            self.pending_jobs[poster_key] = (poster_key, image_input, dict(preprocess_options or {}), generation)
        self.ocr_progress.emit(poster_key, {"message": "OCR queued", "percent": 0, "stage": "queued", "generation": generation})
        self.queue.put(poster_key)

    def stop(self):
        self.stopping = True
        for _ in range(self.worker_count):
            self.queue.put(None)

    def run(self):
        self.worker_threads = [threading.Thread(target=self._worker_loop, args=(i + 1,), daemon=True) for i in range(self.worker_count)]
        for thread in self.worker_threads:
            thread.start()
        for thread in self.worker_threads:
            thread.join()

    def _worker_loop(self, worker_number):
        try:
            engine = self.engine_factory()
            engine_error = None
        except Exception as exc:
            engine = None
            engine_error = str(exc)
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
            poster_key, image_input, options, generation = job
            try:
                if engine is None:
                    raise RuntimeError(engine_error or "OCR engine failed to initialize")
                options["progress_callback"] = self._make_progress_callback(poster_key, generation, worker_number)
                self.ocr_progress.emit(poster_key, {"message": f"W{worker_number}: OCR starting", "percent": 2, "stage": "starting", "generation": generation})
                text, conf = engine.get_text(image_input, options)
                self.ocr_ready.emit(poster_key, text or "(no text)", (conf, generation))
            except Exception as exc:
                self.ocr_progress.emit(poster_key, {"message": f"W{worker_number}: OCR error: {exc}", "percent": 100, "stage": "error", "generation": generation})
                self.ocr_ready.emit(poster_key, f"OCR Error: {exc}", (None, generation))
            finally:
                self.queue.task_done()

    def _make_progress_callback(self, poster_key, generation, worker_number):
        def callback(info):
            payload = dict(info or {})
            payload["generation"] = generation
            payload["message"] = f"W{worker_number}: {payload.get('message', 'OCR processing')}"
            self.ocr_progress.emit(poster_key, payload)
        return callback


class CaptionWorker(QThread):
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
        if self.stopping:
            return
        with self.lock:
            self.pending_jobs[poster_key] = (poster_key, self.prepare_image(pil_image))
        self.caption_progress.emit(poster_key, {"message": "Caption queued", "percent": 0, "stage": "queued"})
        self.queue.put(poster_key)

    def stop(self):
        self.stopping = True
        self.queue.put(None)

    def prepare_image(self, pil_image):
        if pil_image is None:
            return None
        try:
            image = pil_image.convert("RGB").copy()
            image.thumbnail((768, 768), getattr(Image, "Resampling", Image).LANCZOS)
            return image
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
                self.caption_progress.emit(poster_key, {"message": "Captioning image", "percent": 35, "stage": "captioning"})
                caption = "Description unavailable." if pil_image is None else str(self.captioner.generate_caption(pil_image) or "").strip() or "(no description)"
                self.caption_progress.emit(poster_key, {"message": "Caption done", "percent": 100, "stage": "done"})
                self.caption_ready.emit(poster_key, caption)
            except Exception as exc:
                self.caption_progress.emit(poster_key, {"message": f"Caption error: {exc}", "percent": 100, "stage": "error"})
                self.caption_ready.emit(poster_key, f"Description unavailable: {exc}")
            finally:
                self.queue.task_done()


class PosterReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.manager = VideoSourceManager()
        self.current_cap = None
        self.is_live_camera = False
        self.video_path = None
        self.temp_video_writer = None
        self.model_worker = None
        self.poster_id_to_button = {}
        self.poster_buttons = []
        self.poster_data = []
        self.focused_poster_index = -1
        self.ocr_generation = 0
        self._selecting_poster = False
        self.last_detector_frame = None
        self.detector_preview_expanded = False

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
        self.ocr_worker = OCRWorkerPool(worker_count=OCR_WORKER_COUNT)
        self.ocr_worker.ocr_ready.connect(self.on_ocr_ready)
        self.ocr_worker.ocr_progress.connect(self.on_ocr_progress)
        self.ocr_worker.start()
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
        QShortcut(QKeySequence("1"), self).activated.connect(self.run_file_input)
        QShortcut(QKeySequence("2"), self).activated.connect(self.run_camera_input)

    def init_processing_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.video_label = QLabel("Loading Video Feed...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_stop = QPushButton("Finish and View Posters  [Space / Esc]")
        btn_stop.clicked.connect(self.stop_processing)
        layout.addWidget(self.video_label)
        layout.addWidget(btn_stop)
        self.stack.addWidget(page)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.stop_processing)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.stop_processing)

    def init_results_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        top_text = QWidget()
        top_text_layout = QVBoxLayout(top_text)
        top_text_layout.setContentsMargins(0, 0, 0, 0)
        top_text_layout.setSpacing(4)

        self.ocr_result_label = QLabel("Press T to load test images. Navigate posters with Arrow Keys. Press Enter to open.")
        self.ocr_result_label.setWordWrap(True)
        self.ocr_global_status_label = QLabel(f"OCR/caption status: 0/0 ready")
        self.ocr_global_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_global_status_label.setWordWrap(True)
        top_text_layout.addWidget(self.ocr_result_label)
        top_text_layout.addWidget(self.ocr_global_status_label)

        self.detector_preview_label = QLabel("Detector preview")
        self.detector_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detector_preview_label.setFixedSize(*DETECTOR_PREVIEW_SIZE)
        self.detector_preview_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #777; font-size: 11px;")
        self.detector_preview_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detector_preview_label.setToolTip("Click to expand detector preview")
        self.detector_preview_label.installEventFilter(self)

        top_layout.addWidget(top_text, stretch=1)
        top_layout.addWidget(self.detector_preview_label, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        layout.addWidget(top_row)

        self.gallery_scroll = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.gallery_scroll.setWidget(self.grid_widget)
        self.gallery_scroll.setWidgetResizable(True)
        layout.addWidget(self.gallery_scroll)

        btn_test = QPushButton("Test OCR: Load Local Images  [T]")
        btn_test.clicked.connect(self.load_test_posters)
        self.results_status_label = QLabel("")
        self.results_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_status_label.setVisible(False)
        self.results_progress_bar = QProgressBar()
        self.results_progress_bar.setRange(0, 100)
        self.results_progress_bar.setVisible(False)
        self.btn_stop_proc = QPushButton("Stop Processing  [X]")
        self.btn_stop_proc.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
        self.btn_stop_proc.setVisible(False)
        self.btn_stop_proc.clicked.connect(self.stop_model_processing)
        layout.addWidget(btn_test)
        layout.addWidget(self.results_status_label)
        layout.addWidget(self.results_progress_bar)
        layout.addWidget(self.btn_stop_proc)
        page.setLayout(layout)
        self.results_page = page
        self.detector_overlay_label = QLabel(page)
        self.detector_overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detector_overlay_label.setStyleSheet("background: #000; color: #ddd; border: 2px solid #444; font-size: 14px;")
        self.detector_overlay_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detector_overlay_label.setToolTip("Click to return detector preview to corner")
        self.detector_overlay_label.installEventFilter(self)
        self.detector_overlay_label.hide()
        self.detector_overlay_label.raise_()
        self.stack.addWidget(page)

        for key, action in [
            ("T", self.load_test_posters),
            ("X", self.stop_model_processing),
            (Qt.Key.Key_Right, lambda: self.move_poster_focus(1)),
            (Qt.Key.Key_Left, lambda: self.move_poster_focus(-1)),
            (Qt.Key.Key_Down, lambda: self.move_poster_focus(GALLERY_COLS)),
            (Qt.Key.Key_Up, lambda: self.move_poster_focus(-GALLERY_COLS)),
            (Qt.Key.Key_Return, self.open_focused_poster),
            (Qt.Key.Key_Enter, self.open_focused_poster),
        ]:
            QShortcut(QKeySequence(key), self).activated.connect(action)

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
        if cap is None:
            return
        self.current_cap = cap
        self.is_live_camera = True
        self.video_path = str(SCRIPT_DIR / "temp_record.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.temp_video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
        self.start_pipeline()

    def start_pipeline(self):
        self.stack.setCurrentIndex(1)
        self.timer.start(30)

    def process_frame(self):
        if self.current_cap is None:
            return
        ok, frame = self.current_cap.read()
        if not ok:
            self.stop_processing()
            return
        if self.is_live_camera and self.temp_video_writer is not None:
            self.temp_video_writer.write(frame)
        self.video_label.setPixmap(self.pixmap_from_bgr(frame).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))

    def stop_processing(self):
        if self.stack.currentIndex() != 1:
            return
        self.timer.stop()
        if self.current_cap:
            self.current_cap.release()
            self.current_cap = None
        if self.temp_video_writer is not None:
            self.temp_video_writer.release()
            self.temp_video_writer = None
        self.start_model_processing()

    def get_preprocess_options(self):
        return {}

    def start_model_processing(self):
        self.clear_gallery()
        self.stack.setCurrentIndex(2)
        self.results_progress_bar.setVisible(True)
        self.results_status_label.setVisible(True)
        self.btn_stop_proc.setVisible(True)
        self.btn_stop_proc.setEnabled(True)
        self.results_progress_bar.setValue(0)
        self.results_status_label.setText("Starting model processing...")
        self.ocr_result_label.setText("Processing video. Detected posters will appear below as they are found.")
        if not self.video_path:
            self.on_error("No video path is available for model processing.")
            return
        worker_cls = {"sam": SAMWorker, "dino": DINOWorker, "yolo": YOLOWorker, "maskrcnn": MaskRCNNWorker}.get(MODEL_BACKEND.lower())
        if worker_cls is None:
            self.on_error(f"{MODEL_BACKEND.upper()} worker is not available. Check that the engine file is in the GUI folder.")
            return
        self.model_worker = worker_cls(self.video_path)
        self.model_worker.progress.connect(self.on_progress)
        self.model_worker.poster_found.connect(self.on_poster_found)
        self.model_worker.finished_processing.connect(self.on_model_finished)
        self.model_worker.error.connect(self.on_error)
        if hasattr(self.model_worker, "poster_found_record"):
            self.model_worker.poster_found_record.connect(self.on_poster_found_record)
        if hasattr(self.model_worker, "frame_preview"):
            self.model_worker.frame_preview.connect(self.on_detector_preview)
        self.detector_preview_label.setText("Detector preview")
        self.detector_preview_label.setPixmap(QPixmap())
        self.model_worker.start()

    def stop_model_processing(self):
        if self.model_worker is not None and self.model_worker.isRunning():
            self.results_status_label.setText("Stopping after current frame...")
            self.btn_stop_proc.setEnabled(False)
            self.model_worker.request_stop() if hasattr(self.model_worker, "request_stop") else self.model_worker.terminate()

    def on_progress(self, pct, status):
        self.results_progress_bar.setValue(int(pct))
        self.results_status_label.setText(str(status))

    def on_detector_preview(self, frame):
        if frame is None:
            return
        try:
            self.last_detector_frame = frame.copy()
        except Exception:
            self.last_detector_frame = frame
        self.update_detector_preview_pixmaps()

    def update_detector_preview_pixmaps(self):
        if self.last_detector_frame is None:
            return
        corner_pixmap = self.pixmap_from_bgr(self.last_detector_frame).scaled(
            self.detector_preview_label.width(),
            self.detector_preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.detector_preview_label.setPixmap(corner_pixmap)

        if getattr(self, "detector_preview_expanded", False) and hasattr(self, "detector_overlay_label"):
            overlay_pixmap = self.pixmap_from_bgr(self.last_detector_frame).scaled(
                self.detector_overlay_label.width(),
                self.detector_overlay_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.detector_overlay_label.setPixmap(overlay_pixmap)

    def toggle_detector_preview_expanded(self):
        if not hasattr(self, "detector_overlay_label"):
            return
        self.detector_preview_expanded = not self.detector_preview_expanded
        if self.detector_preview_expanded:
            self.detector_overlay_label.setGeometry(self.results_page.rect())
            self.detector_overlay_label.show()
            self.detector_overlay_label.raise_()
            self.detector_overlay_label.setFocus()
        else:
            self.detector_overlay_label.hide()
        self.update_detector_preview_pixmaps()

    def on_error(self, message):
        self.results_progress_bar.setVisible(False)
        self.btn_stop_proc.setVisible(False)
        self.results_status_label.setVisible(True)
        self.results_status_label.setText("Model processing failed.")
        QMessageBox.critical(self, "Model Error", str(message))

    def on_model_finished(self, results=None, output_path=None):
        self.results_progress_bar.setVisible(False)
        self.btn_stop_proc.setVisible(False)
        self.btn_stop_proc.setEnabled(True)
        self.results_status_label.setText(f"Processing complete/stopped. Found {len(self.poster_data)} posters. OCR/captions will continue updating until each card is ready.")

    def on_poster_found(self, poster_id, image_array):
        if poster_id in self.poster_id_to_button:
            self.update_poster_in_gallery(poster_id, image_array, "OCR pending...", None)
        else:
            self.add_poster_to_gallery(image_array, "OCR pending...", None, poster_id=poster_id)
        btn = self.poster_id_to_button[poster_id]
        idx = self.poster_buttons.index(btn)
        self.ocr_worker.add_task(self.poster_data[idx]["poster_key"], image_array, self.get_preprocess_options(), self.ocr_generation)

    def on_poster_found_record(self, record):
        if not isinstance(record, dict):
            return
        poster_id = record.get("track_id", record.get("id"))
        btn = self.poster_id_to_button.get(poster_id)
        if btn is None:
            return
        try:
            idx = self.poster_buttons.index(btn)
        except ValueError:
            return

        poster = self.poster_data[idx]
        poster["track_meta"] = dict(record)
        label = poster.get("meta_label")
        if label is not None:
            label.setText(self.track_meta_text(record))
            label.setVisible(True)
            label.adjustSize()

        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

        self.sort_gallery_by_quality()

    def track_meta_text(self, record):
        quality = self.format_score(record.get("quality"))
        score = self.format_score(record.get("score"))
        seen = self.format_int(record.get("seen_count"))
        version = self.format_int(record.get("version"))
        return f"Q: {quality} | Conf: {score} | Seen: {seen} | V: {version}"

    def format_score(self, value):
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "N/A"

    def format_int(self, value):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "N/A"

    def load_test_posters(self):
        if self.stack.currentIndex() != 2:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Posters", str(LOAD_POSTERS_START_DIR), "Images (*.png *.jpg *.jpeg)")
        if not files:
            return

        self.results_status_label.setVisible(True)
        self.results_status_label.setText(f"Queued {len(files)} local image(s) for OCR testing.")

        for file_path in files:
            self.add_poster_to_gallery(file_path, "OCR pending...", None)
            poster = self.poster_data[-1]
            self.ocr_worker.add_task(poster["poster_key"], file_path, self.get_preprocess_options(), self.ocr_generation)
            QApplication.processEvents()

    def add_poster_to_gallery(self, image_input, detected_text, avg_conf=None, poster_id=None, run_caption_async=True):
        btn = QPushButton()
        full_pixmap, pil_image = self.prepare_pixmap_and_pil(image_input)
        if full_pixmap.isNull():
            return
        self.set_button_preview(btn, image_input)
        btn.setStyleSheet("QPushButton { border: 2px solid transparent; border-radius: 3px; background-color: transparent; padding: 2px; } QPushButton:hover, QPushButton:focus { border: 2px solid #0078d7; background-color: rgba(0, 120, 215, 20); }")

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFormat("Queued")
        bar.setFixedWidth(btn.width())

        meta_label = QLabel("")
        meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta_label.setWordWrap(True)
        meta_label.setFixedWidth(GALLERY_BUTTON_SIZE)
        meta_label.setMinimumHeight(28)
        meta_label.setMaximumHeight(40)
        meta_label.setStyleSheet("font-size: 10px; color: #444; padding-top: 4px; background: transparent;")
        meta_label.setVisible(False)

        card = QWidget()
        card.setFixedWidth(GALLERY_CARD_WIDTH)
        card.setMinimumHeight(GALLERY_BUTTON_SIZE + 56)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(2, 2, 2, 2)
        card_layout.setSpacing(6)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(meta_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(bar, alignment=Qt.AlignmentFlag.AlignHCenter)

        poster_key = poster_id if poster_id is not None else f"manual_{len(self.poster_buttons)}"
        ocr_done = detected_text not in PENDING_TEXTS
        poster = {
            "poster_key": poster_key,
            "poster_id": poster_id,
            "image_input": image_input,
            "detected_text": detected_text,
            "pixmap": full_pixmap,
            "description": "Generating description..." if run_caption_async else "Processing...",
            "confidence": self.confidence_text(avg_conf),
            "ocr_done": ocr_done,
            "caption_done": not run_caption_async,
            "ocr_progress": 100 if ocr_done else 0,
            "caption_progress": 0 if run_caption_async else 100,
            "ocr_stage": "OCR done" if ocr_done else "OCR queued",
            "caption_stage": "Caption queued" if run_caption_async else "Caption done",
            "loading_bar": bar,
            "meta_label": meta_label,
            "track_meta": None,
            "card": card,
            "button": btn,
        }
        self.poster_buttons.append(btn)
        self.poster_data.append(poster)
        btn.clicked.connect(lambda checked=False, b=btn: self.open_poster_by_button(b))
        btn.installEventFilter(self)
        if poster_id is not None:
            self.poster_id_to_button[poster_id] = btn
        count = self.grid_layout.count()
        self.grid_layout.addWidget(card, count // GALLERY_COLS, count % GALLERY_COLS)
        self.update_poster_ready_state(len(self.poster_data) - 1)
        if self.focused_poster_index == -1:
            self.select_poster_index(0, scroll=False, focus=True)
        if run_caption_async:
            self.caption_worker.add_task(poster_key, pil_image)

    def update_poster_in_gallery(self, poster_id, image_input, detected_text, avg_conf=None):
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
        poster.update(
            image_input=image_input,
            detected_text=detected_text,
            pixmap=full_pixmap,
            confidence=self.confidence_text(avg_conf),
            description="Updating description...",
            ocr_done=detected_text not in PENDING_TEXTS,
            caption_done=False,
            ocr_progress=100 if detected_text not in PENDING_TEXTS else 0,
            caption_progress=0,
            ocr_stage="OCR done" if detected_text not in PENDING_TEXTS else "OCR queued",
            caption_stage="Caption queued",
        )
        self.set_button_preview(btn, image_input)
        btn.setToolTip("Description: Updating description...")
        self.caption_worker.add_task(poster["poster_key"], pil_image)
        self.update_poster_ready_state(idx)
        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def update_poster_ready_state(self, idx):
        if not (0 <= idx < len(self.poster_data)):
            return
        poster = self.poster_data[idx]
        ready = poster["ocr_done"] and poster["caption_done"]
        if not poster["ocr_done"]:
            value, stage = int(poster["ocr_progress"] * 0.70), poster["ocr_stage"]
        elif not poster["caption_done"]:
            value, stage = 70 + int(poster["caption_progress"] * 0.30), poster["caption_stage"]
        else:
            value, stage = 100, "Ready"
        bar = poster["loading_bar"]
        bar.setValue(max(0, min(100, value)))
        bar.setFormat(stage)
        bar.setVisible(not ready)
        self.poster_buttons[idx].setEnabled(True)
        self.update_global_processing_status()

    def on_ocr_progress(self, poster_key, info):
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return
        if isinstance(info, dict) and info.get("generation") not in (None, self.ocr_generation):
            return
        percent = int(info.get("percent", 0)) if isinstance(info, dict) else 0
        message = str(info.get("message", "OCR processing")) if isinstance(info, dict) else "OCR processing"
        self.poster_data[idx]["ocr_progress"] = max(0, min(100, percent))
        self.poster_data[idx]["ocr_stage"] = f"OCR: {message}"
        self.update_poster_ready_state(idx)
        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def on_ocr_ready(self, poster_key, detected_text, payload):
        avg_conf, generation = payload
        if generation != self.ocr_generation:
            return
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return
        poster = self.poster_data[idx]
        poster["detected_text"] = detected_text
        poster["confidence"] = self.confidence_text(avg_conf)
        poster["ocr_done"] = True
        poster["ocr_progress"] = 100
        poster["ocr_stage"] = "OCR done"
        self.update_poster_ready_state(idx)
        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def on_caption_progress(self, poster_key, info):
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return
        percent = int(info.get("percent", 0)) if isinstance(info, dict) else 0
        message = str(info.get("message", "Caption processing")) if isinstance(info, dict) else "Caption processing"
        self.poster_data[idx]["caption_progress"] = max(0, min(100, percent))
        self.poster_data[idx]["caption_stage"] = message
        self.update_poster_ready_state(idx)
        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def on_caption_ready(self, poster_key, description):
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return
        poster = self.poster_data[idx]
        poster["description"] = description or "(no description)"
        poster["caption_done"] = True
        poster["caption_progress"] = 100
        poster["caption_stage"] = "Caption done"
        self.poster_buttons[idx].setToolTip(f"Description: {poster['description']}")
        self.update_poster_ready_state(idx)
        if self.focused_poster_index == idx:
            self.update_status_panel_for_index(idx)

    def confidence_text(self, avg_conf):
        return "Average OCR confidence: N/A" if avg_conf is None else f"Average OCR confidence: {avg_conf:.2%}"

    def update_status_panel_for_index(self, idx):
        if not (0 <= idx < len(self.poster_data)):
            return
        poster = self.poster_data[idx]
        parts = [
            f"Description: {poster['description']}",
            f"Detected Text: {poster['detected_text']}",
            f"Confidence: {poster['confidence']}",
        ]
        # if poster.get("track_meta"):
        #     parts.append(f"Track: {self.track_meta_text(poster['track_meta'])}")
        self.ocr_result_label.setText("\n\n".join(parts))

    def select_poster_index(self, idx, *, scroll=True, focus=True):
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
        if scroll:
            QTimer.singleShot(0, lambda w=self.poster_data[idx]["card"]: self.gallery_scroll.ensureWidgetVisible(w, 24, 24))

    def update_global_processing_status(self):
        total = len(self.poster_data)
        if total == 0:
            self.ocr_global_status_label.setText(f"OCR/caption status: 0/0 ready")
            return
        ocr_done = sum(1 for p in self.poster_data if p["ocr_done"])
        caption_done = sum(1 for p in self.poster_data if p["caption_done"])
        ready = sum(1 for p in self.poster_data if p["ocr_done"] and p["caption_done"])
        self.ocr_global_status_label.setText(f"Ready {ready}/{total} | OCR {ocr_done}/{total} complete ({total - ocr_done} pending) | Captions {caption_done}/{total} complete ({total - caption_done} pending)")

    def find_poster_index_by_key(self, poster_key):
        for idx, poster in enumerate(self.poster_data):
            if poster["poster_key"] == poster_key:
                return idx
        return None

    def sort_gallery_by_quality(self):
        if not self.poster_data:
            return

        focused_key = None
        if 0 <= self.focused_poster_index < len(self.poster_data):
            focused_key = self.poster_data[self.focused_poster_index].get("poster_key")

        indexed_items = list(enumerate(zip(self.poster_data, self.poster_buttons)))

        def quality_sort_value(item):
            original_index, (poster, _btn) = item
            meta = poster.get("track_meta") or {}
            try:
                quality = float(meta.get("quality"))
            except (TypeError, ValueError):
                quality = float("-inf")
            return (-quality, original_index)

        sorted_items = sorted(indexed_items, key=quality_sort_value)
        if [old_index for old_index, _item in sorted_items] == list(range(len(indexed_items))):
            return

        while self.grid_layout.count():
            self.grid_layout.takeAt(0)

        self.poster_data = [poster for _old_index, (poster, _btn) in sorted_items]
        self.poster_buttons = [btn for _old_index, (_poster, btn) in sorted_items]

        for idx, poster in enumerate(self.poster_data):
            self.grid_layout.addWidget(poster["card"], idx // GALLERY_COLS, idx % GALLERY_COLS)

        if focused_key is not None:
            self.focused_poster_index = self.find_poster_index_by_key(focused_key) or 0
        elif self.poster_data:
            self.focused_poster_index = 0

    def clear_gallery(self):
        self.last_detector_frame = None
        self.detector_preview_expanded = False
        if hasattr(self, "detector_preview_label"):
            self.detector_preview_label.setText("Detector preview")
            self.detector_preview_label.setPixmap(QPixmap())
        if hasattr(self, "detector_overlay_label"):
            self.detector_overlay_label.hide()
            self.detector_overlay_label.setPixmap(QPixmap())
        self.poster_id_to_button = {}
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.poster_buttons = []
        self.poster_data = []
        self.focused_poster_index = -1
        self.update_global_processing_status()

    def prepare_pixmap_and_pil(self, image_input):
        if isinstance(image_input, (str, Path)):
            return QPixmap(str(image_input)), Image.open(image_input).convert("RGB")
        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img.copy()), Image.fromarray(rgb)

    def make_display_pixmap(self, image_input):
        if isinstance(image_input, (str, Path)):
            return QPixmap(str(image_input))
        if image_input is None:
            return QPixmap()
        return self.pixmap_from_bgr(image_input)

    def make_processed_pixmap(self, image_input):
        return self.make_display_pixmap(image_input)

    def set_button_preview(self, btn, image_input):
        preview = self.make_display_pixmap(image_input)
        if preview.isNull():
            return
        thumbnail = preview.scaled(GALLERY_THUMB_SIZE, GALLERY_THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        btn.setIcon(QIcon(thumbnail))
        btn.setIconSize(thumbnail.size())
        btn.setFixedSize(GALLERY_BUTTON_SIZE, GALLERY_BUTTON_SIZE)
        btn.setMinimumSize(GALLERY_BUTTON_SIZE, GALLERY_BUTTON_SIZE)
        for poster in self.poster_data:
            if poster.get("button") is btn:
                poster["loading_bar"].setFixedWidth(GALLERY_BUTTON_SIZE)
                meta_label = poster.get("meta_label")
                if meta_label is not None:
                    meta_label.setFixedWidth(GALLERY_BUTTON_SIZE)
                break

    def pixmap_from_bgr(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img.copy())

    def refresh_gallery_previews(self):
        return

    def move_poster_focus(self, delta):
        if self.stack.currentIndex() == 2 and self.poster_buttons:
            idx = 0 if self.focused_poster_index < 0 else max(0, min(self.focused_poster_index + delta, len(self.poster_buttons) - 1))
            self.select_poster_index(idx, scroll=True, focus=True)

    def open_focused_poster(self):
        if self.stack.currentIndex() != 2 or not self.poster_buttons:
            return
        if self.focused_poster_index < 0:
            self.select_poster_index(0)
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
        if not (poster["ocr_done"] and poster["caption_done"]):
            self.ocr_result_label.setText("Poster is still processing. Please wait.")
            return
        text = poster["detected_text"]
        display_text = f"{text}\n\n{poster['confidence']}"
        self.ocr_result_label.setText(display_text)
        self.speaker.speak(text)
        self.show_zoom_dialog(self.make_display_pixmap(poster["image_input"]), display_text)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "detector_preview_label", None) and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self.toggle_detector_preview_expanded()
                return True
        if obj is getattr(self, "detector_overlay_label", None) and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self.toggle_detector_preview_expanded()
                return True
        if obj in self.poster_buttons and event.type() in (QEvent.Type.FocusIn, QEvent.Type.Enter, QEvent.Type.MouseButtonPress):
            self.select_poster_index(self.poster_buttons.index(obj), scroll=event.type() == QEvent.Type.FocusIn, focus=event.type() != QEvent.Type.FocusIn)
        return super().eventFilter(obj, event)

    def show_zoom_dialog(self, color_pixmap, text):
        dialog = QDialog(self)
        dialog.setWindowTitle("Poster Viewer — Color Image")
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        title = QLabel("Color Image")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_view = ZoomableImageLabel(color_pixmap, dialog)
        image_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setMaximumHeight(120)
        text_label.setStyleSheet("padding: 8px; background-color: #1e1e1e; color: white;")
        layout.addWidget(title)
        layout.addWidget(image_view, stretch=1)
        layout.addWidget(text_label)
        image_view.setFocus()
        dialog.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "detector_preview_expanded", False) and hasattr(self, "detector_overlay_label"):
            self.detector_overlay_label.setGeometry(self.results_page.rect())
            self.update_detector_preview_pixmaps()

    def closeEvent(self, event):
        if self.model_worker is not None and self.model_worker.isRunning():
            if hasattr(self.model_worker, "request_stop"):
                self.model_worker.request_stop()
            self.model_worker.wait(1000)
            if self.model_worker.isRunning():
                self.model_worker.terminate()
                self.model_worker.wait(1000)
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
