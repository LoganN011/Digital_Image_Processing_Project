from __future__ import annotations

import csv
import json
import queue
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import cv2
from PIL import Image
from PyQt6.QtCore import QEvent, QPoint, Qt, QThread, QTimer, pyqtSignal


from PyQt6.QtGui import QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QHBoxLayout, QWidget, QPlainTextEdit

from audio_engine import AudioEngine
from caption_engine import ImageCaptioner
from ocr_engine import OCREngine

try:
    from yolo_engine import YOLOWorker
except ImportError:
    YOLOWorker = None


SCRIPT_DIR = Path(__file__).resolve().parent
LOAD_POSTERS_START_DIR = SCRIPT_DIR
LOAD_VIDEO_START_DIR = SCRIPT_DIR
MODEL_BACKEND = "yolo"
GALLERY_CARD_WIDTH = 230
GALLERY_THUMB_SIZE = 220
GALLERY_BUTTON_SIZE = 228
DETECTOR_PREVIEW_SIZE = (320, 180)
GALLERY_COLS = 4
import os
OCR_WORKER_COUNT = 1 #os.cpu_count() or 2
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


class OCRWarmupWorker(QThread):
    """Load the heavy PARSeq recognizer in the background at app startup.

    The worker intentionally loads only PARSeq. EasyOCR/CRAFT detection stays
    lazy because it is needed only when real OCR jobs are queued. The loaded
    OCREngine instance is later handed to OCRWorkerPool so the first OCR crop
    does not pay the full PARSeq startup cost.
    """

    status_changed = pyqtSignal(str)
    finished_loading = pyqtSignal(bool, str)

    def __init__(self, engine_factory=OCREngine):
        super().__init__()
        self.engine_factory = engine_factory
        self.ready_event = threading.Event()
        self.engine = None
        self.error = None

    def run(self):
        try:
            self.status_changed.emit("OCR/PARSeq: loading in background...")
            engine = self.engine_factory(lazy_load=True)
            ok = bool(engine.load_parseq_if_needed())
            if ok:
                self.engine = engine
                self.error = None
                self.status_changed.emit("OCR/PARSeq: ready")
                self.finished_loading.emit(True, "OCR/PARSeq: ready")
            else:
                self.engine = None
                self.error = engine.parseq_error or "PARSeq failed to load"
                message = f"OCR/PARSeq: failed to load ({self.error})"
                self.status_changed.emit(message)
                self.finished_loading.emit(False, message)
        except Exception as exc:
            self.engine = None
            self.error = str(exc)
            message = f"OCR/PARSeq: failed to load ({exc})"
            self.status_changed.emit(message)
            self.finished_loading.emit(False, message)
        finally:
            self.ready_event.set()




class CaptionWarmupWorker(QThread):
    """Load the BLIP caption model in the background at app startup.

    This only constructs ImageCaptioner early; it does not generate captions
    until final best crops are queued later. The loaded captioner is handed to
    CaptionWorker so the first caption job does not pay the full BLIP startup
    cost.
    """

    status_changed = pyqtSignal(str)
    finished_loading = pyqtSignal(bool, str)

    def __init__(self, captioner_factory=ImageCaptioner):
        super().__init__()
        self.captioner_factory = captioner_factory
        self.ready_event = threading.Event()
        self.captioner = None
        self.error = None

    def run(self):
        try:
            self.status_changed.emit("Caption/BLIP: loading in background...")
            captioner = self.captioner_factory()
            self.captioner = captioner
            self.error = None
            self.status_changed.emit("Caption/BLIP: ready")
            self.finished_loading.emit(True, "Caption/BLIP: ready")
        except Exception as exc:
            self.captioner = None
            self.error = str(exc)
            message = f"Caption/BLIP: failed to load ({exc})"
            self.status_changed.emit(message)
            self.finished_loading.emit(False, message)
        finally:
            self.ready_event.set()


class OCRWorkerPool(QThread):
    ocr_ready = pyqtSignal(object, str, object)
    ocr_progress = pyqtSignal(object, object)

    def __init__(self, worker_count=2, engine_factory=OCREngine, engine_provider=None):
        super().__init__()
        self.worker_count = max(1, int(worker_count))
        self.engine_factory = engine_factory
        self.engine_provider = engine_provider
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

    def clear_pending_tasks(self):
        """Drop queued OCR jobs that have not started yet.

        This is used on restart so the old run cannot keep feeding OCR jobs
        after the gallery has been cleared. Jobs already inside EasyOCR/PARSeq
        cannot be interrupted safely, but their results are ignored by the
        generation check in the GUI.
        """
        with self.lock:
            self.pending_jobs.clear()
        try:
            while True:
                item = self.queue.get_nowait()
                self.queue.task_done()
                if item is None:
                    # Preserve stop sentinels if stop() has already been called.
                    self.queue.put(None)
                    break
        except queue.Empty:
            pass

    def stop(self):
        self.stopping = True
        self.clear_pending_tasks()
        for _ in range(self.worker_count):
            self.queue.put(None)

    def run(self):
        self.worker_threads = [threading.Thread(target=self._worker_loop, args=(i + 1,), daemon=True) for i in range(self.worker_count)]
        for thread in self.worker_threads:
            thread.start()
        for thread in self.worker_threads:
            thread.join()

    def _load_engine_for_worker(self):
        if self.engine_provider is not None:
            return self.engine_provider()
        return self.engine_factory()

    def _worker_loop(self, worker_number):
        engine = None
        engine_error = None
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
                if engine is None and engine_error is None:
                    self.ocr_progress.emit(
                        poster_key,
                        {
                            "message": f"W{worker_number}: OCR engine loading / waiting for PARSeq",
                            "percent": 1,
                            "stage": "loading_models",
                            "generation": generation,
                        },
                    )
                    try:
                        engine = self._load_engine_for_worker()
                    except Exception as exc:
                        engine = None
                        engine_error = str(exc)
                if engine is None:
                    raise RuntimeError(engine_error or "OCR engine failed to initialize")
                options["progress_callback"] = self._make_progress_callback(poster_key, generation, worker_number)
                self.ocr_progress.emit(poster_key, {"message": f"W{worker_number}: OCR starting", "percent": 2, "stage": "starting", "generation": generation})
                if hasattr(engine, "get_text_details"):
                    result = engine.get_text_details(image_input, options)
                    text = getattr(result, "text", "") or "(no text)"
                    conf = getattr(result, "avg_conf", None)
                    details = self._ocr_result_to_details(result)
                else:
                    text, conf = engine.get_text(image_input, options)
                    details = {"text": text or "(no text)", "avg_conf": conf, "method": "get_text", "lines": [], "raw_count": None, "used_inverted": False}
                self.ocr_ready.emit(poster_key, text or "(no text)", (conf, generation, details))
            except Exception as exc:
                self.ocr_progress.emit(poster_key, {"message": f"W{worker_number}: OCR error: {exc}", "percent": 100, "stage": "error", "generation": generation})
                self.ocr_ready.emit(poster_key, f"OCR Error: {exc}", (None, generation, {"error": str(exc), "lines": []}))
            finally:
                self.queue.task_done()

    def _ocr_result_to_details(self, result):
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def safe_box(value):
            if value is None:
                return None
            try:
                if hasattr(value, "tolist"):
                    return value.tolist()
                return [[float(x), float(y)] for x, y in value]
            except Exception:
                try:
                    return str(value)
                except Exception:
                    return None

        lines = []
        for idx, line in enumerate(getattr(result, "lines", []) or [], 1):
            line_text = str(getattr(line, "text", "") or "")
            line_conf = safe_float(getattr(line, "conf", 0.0))
            words = [word for word in line_text.split() if word]
            lines.append({
                "index": idx,
                "text": line_text,
                "conf": line_conf,
                "method": str(getattr(line, "method", "") or ""),
                "box": safe_box(getattr(line, "box", None)),
                "parseq_text": str(getattr(line, "parseq_text", "") or ""),
                "parseq_conf": safe_float(getattr(line, "parseq_conf", 0.0)),
                "easyocr_text": str(getattr(line, "easyocr_text", "") or ""),
                "easyocr_conf": safe_float(getattr(line, "easyocr_conf", 0.0)),
                "row_index": int(getattr(line, "row_index", idx)),
                "word_index": int(getattr(line, "word_index", 0)),
                "line_break_after": bool(getattr(line, "line_break_after", False)),
                "words": [{"text": word, "conf": line_conf} for word in words],
            })
        return {
            "text": str(getattr(result, "text", "") or ""),
            "avg_conf": safe_float(getattr(result, "avg_conf", 0.0)),
            "method": str(getattr(result, "method", "") or ""),
            "used_inverted": bool(getattr(result, "used_inverted", False)),
            "raw_count": getattr(result, "raw_count", None),
            "line_count": len(lines),
            "lines": lines,
        }

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

    def __init__(self, captioner_factory=ImageCaptioner, captioner_provider=None):
        super().__init__()
        # BLIP is warmed at app startup when possible. Caption generation still
        # waits until final crops are queued; this worker simply reuses the
        # background-loaded captioner instead of constructing a cold one.
        self.captioner_factory = captioner_factory
        self.captioner_provider = captioner_provider
        self.captioner = None
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.pending_jobs = {}
        self.stopping = False

    def _ensure_captioner(self):
        if self.captioner is None:
            if self.captioner_provider is not None:
                self.captioner = self.captioner_provider()
            else:
                self.captioner = self.captioner_factory()
        return self.captioner

    def add_task(self, poster_key, pil_image):
        if self.stopping:
            return
        with self.lock:
            self.pending_jobs[poster_key] = (poster_key, self.prepare_image(pil_image))
        self.caption_progress.emit(poster_key, {"message": "Caption queued", "percent": 0, "stage": "queued"})
        self.queue.put(poster_key)

    def clear_pending_tasks(self):
        """Drop queued caption jobs that have not started yet."""
        with self.lock:
            self.pending_jobs.clear()
        try:
            while True:
                item = self.queue.get_nowait()
                self.queue.task_done()
                if item is None:
                    self.queue.put(None)
                    break
        except queue.Empty:
            pass

    def stop(self):
        self.stopping = True
        self.clear_pending_tasks()
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
                self.caption_progress.emit(poster_key, {"message": "Caption engine loading / waiting for BLIP", "percent": 10, "stage": "loading"})
                captioner = self._ensure_captioner()
                self.caption_progress.emit(poster_key, {"message": "Captioning image", "percent": 35, "stage": "captioning"})
                caption = "Description unavailable." if pil_image is None else str(captioner.generate_caption(pil_image) or "").strip() or "(no description)"
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
        self.post_processing_started = False
        self.pending_finished_records = None

        self.setWindowTitle("Debugger - UNM Poster Reader")
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
        # PARSeq and BLIP start warming in the background immediately. OCR and
        # caption jobs are still delayed until final best crops are ready.
        self.ocr_worker = None
        self.caption_worker = None
        self.retired_threads = []
        self.ocr_warmup_worker = None
        self.caption_warmup_worker = None
        self.preloaded_ocr_engine = None
        self.preloaded_ocr_engine_claimed = False
        self.preloaded_captioner = None
        self.preloaded_captioner_claimed = False
        self.ocr_warmup_error = None
        self.caption_warmup_error = None
        self.ocr_warmup_status_message = "OCR/PARSeq: preparing background load..."
        self.caption_warmup_status_message = "Caption/BLIP: preparing background load..."
        self.ocr_warmup_lock = threading.Lock()
        self.caption_warmup_lock = threading.Lock()
        self.start_ocr_background_warmup()
        self.start_caption_background_warmup()

    def start_ocr_background_warmup(self):
        """Start loading PARSeq as soon as the app opens, without blocking the UI."""
        if self.ocr_warmup_worker is not None:
            try:
                if self.ocr_warmup_worker.isRunning():
                    return
            except Exception:
                pass
        self.ocr_warmup_error = None
        self.ocr_warmup_worker = OCRWarmupWorker()
        self.ocr_warmup_worker.status_changed.connect(self.on_ocr_warmup_status)
        self.ocr_warmup_worker.finished_loading.connect(self.on_ocr_warmup_finished)
        self.ocr_warmup_worker.start()

    def on_ocr_warmup_status(self, message):
        self.ocr_warmup_status_message = str(message or "")
        self.update_background_model_status_labels()

    def on_ocr_warmup_finished(self, ok, message):
        worker = self.ocr_warmup_worker
        with self.ocr_warmup_lock:
            if bool(ok) and worker is not None and getattr(worker, "engine", None) is not None:
                # If the OCR worker thread already claimed the engine while the
                # queued Qt signal was waiting to run, do not mark it unclaimed.
                if not self.preloaded_ocr_engine_claimed:
                    self.preloaded_ocr_engine = worker.engine
                self.ocr_warmup_error = None
            else:
                self.preloaded_ocr_engine = None
                self.preloaded_ocr_engine_claimed = False
                self.ocr_warmup_error = str(message or "PARSeq failed to load")
        self.on_ocr_warmup_status(message)


    def start_caption_background_warmup(self):
        """Start loading BLIP as soon as the app opens, without blocking the UI."""
        if self.caption_warmup_worker is not None:
            try:
                if self.caption_warmup_worker.isRunning():
                    return
            except Exception:
                pass
        self.caption_warmup_error = None
        self.caption_warmup_worker = CaptionWarmupWorker()
        self.caption_warmup_worker.status_changed.connect(self.on_caption_warmup_status)
        self.caption_warmup_worker.finished_loading.connect(self.on_caption_warmup_finished)
        self.caption_warmup_worker.start()

    def on_caption_warmup_status(self, message):
        self.caption_warmup_status_message = str(message or "")
        self.update_background_model_status_labels()

    def on_caption_warmup_finished(self, ok, message):
        worker = self.caption_warmup_worker
        with self.caption_warmup_lock:
            if bool(ok) and worker is not None and getattr(worker, "captioner", None) is not None:
                # If the caption worker thread already claimed the captioner
                # while this queued Qt signal was waiting to run, do not mark
                # it unclaimed again.
                if not self.preloaded_captioner_claimed:
                    self.preloaded_captioner = worker.captioner
                self.caption_warmup_error = None
            else:
                self.preloaded_captioner = None
                self.preloaded_captioner_claimed = False
                self.caption_warmup_error = str(message or "Caption model failed to load")
        self.on_caption_warmup_status(message)

    def update_background_model_status_labels(self):
        ocr_msg = str(getattr(self, "ocr_warmup_status_message", "OCR/PARSeq: preparing background load...") or "")
        caption_msg = str(getattr(self, "caption_warmup_status_message", "Caption/BLIP: preparing background load...") or "")
        combined = f"{ocr_msg}\n{caption_msg}"
        if hasattr(self, "ocr_startup_status_label"):
            self.ocr_startup_status_label.setText(ocr_msg)
        if hasattr(self, "caption_startup_status_label"):
            self.caption_startup_status_label.setText(caption_msg)
        if hasattr(self, "ocr_global_status_label") and len(getattr(self, "poster_data", [])) == 0:
            self.ocr_global_status_label.setText(combined)

    def is_ocr_warmup_ready(self):
        with self.ocr_warmup_lock:
            return self.preloaded_ocr_engine is not None and not self.preloaded_ocr_engine_claimed

    def is_caption_warmup_ready(self):
        with self.caption_warmup_lock:
            return self.preloaded_captioner is not None and not self.preloaded_captioner_claimed

    def claim_background_warmed_ocr_engine(self):
        """Return the warmed OCR engine, waiting in the OCR thread if needed.

        This method is called from OCRWorkerPool's worker thread, not the UI
        thread, so waiting here does not block video browsing or playback.
        """
        worker = self.ocr_warmup_worker
        with self.ocr_warmup_lock:
            if self.preloaded_ocr_engine is not None and not self.preloaded_ocr_engine_claimed:
                self.preloaded_ocr_engine_claimed = True
                return self.preloaded_ocr_engine
            warmup_error = self.ocr_warmup_error

        if worker is not None:
            try:
                worker.ready_event.wait()
            except Exception:
                pass
            with self.ocr_warmup_lock:
                if self.preloaded_ocr_engine is not None and not self.preloaded_ocr_engine_claimed:
                    self.preloaded_ocr_engine_claimed = True
                    return self.preloaded_ocr_engine
                if getattr(worker, "engine", None) is not None and not self.preloaded_ocr_engine_claimed:
                    self.preloaded_ocr_engine = worker.engine
                    self.preloaded_ocr_engine_claimed = True
                    return self.preloaded_ocr_engine
                warmup_error = self.ocr_warmup_error or getattr(worker, "error", None)

        if warmup_error:
            raise RuntimeError(warmup_error)
        return OCREngine()


    def claim_background_warmed_captioner(self):
        """Return the warmed BLIP captioner, waiting in the caption thread if needed.

        This method is called from CaptionWorker's worker thread, not the UI
        thread, so waiting here does not block video browsing or playback.
        """
        worker = self.caption_warmup_worker
        with self.caption_warmup_lock:
            if self.preloaded_captioner is not None and not self.preloaded_captioner_claimed:
                self.preloaded_captioner_claimed = True
                return self.preloaded_captioner
            warmup_error = self.caption_warmup_error

        if worker is not None:
            try:
                worker.ready_event.wait()
            except Exception:
                pass
            with self.caption_warmup_lock:
                if self.preloaded_captioner is not None and not self.preloaded_captioner_claimed:
                    self.preloaded_captioner_claimed = True
                    return self.preloaded_captioner
                if getattr(worker, "captioner", None) is not None and not self.preloaded_captioner_claimed:
                    self.preloaded_captioner = worker.captioner
                    self.preloaded_captioner_claimed = True
                    return self.preloaded_captioner
                warmup_error = self.caption_warmup_error or getattr(worker, "error", None)

        if warmup_error:
            raise RuntimeError(warmup_error)
        return ImageCaptioner()

    def start_post_processing_workers(self):
        # Called only when post-processing is actually needed. OCRWorkerPool
        # receives the background-warmed PARSeq engine when available, and
        # CaptionWorker receives the background-warmed BLIP captioner when available.
        self.ocr_worker = OCRWorkerPool(
            worker_count=OCR_WORKER_COUNT,
            engine_provider=self.claim_background_warmed_ocr_engine,
        )
        self.ocr_worker.ocr_ready.connect(self.on_ocr_ready)
        self.ocr_worker.ocr_progress.connect(self.on_ocr_progress)
        self.ocr_worker.start()

        self.caption_worker = CaptionWorker(
            captioner_provider=self.claim_background_warmed_captioner,
        )
        self.caption_worker.caption_ready.connect(self.on_caption_ready)
        self.caption_worker.caption_progress.connect(self.on_caption_progress)
        self.caption_worker.start()

    def retire_thread_if_needed(self, worker):
        if worker is None:
            return
        try:
            if worker.isRunning():
                self.retired_threads.append(worker)
                worker.finished.connect(lambda w=worker: self.cleanup_retired_thread(w))
        except Exception:
            pass

    def cleanup_retired_thread(self, worker):
        try:
            if worker in self.retired_threads:
                self.retired_threads.remove(worker)
        except Exception:
            pass

    def stop_post_processing_workers(self, wait_ms=1000, disconnect=True):
        if self.ocr_worker is not None:
            if disconnect:
                for signal, slot in (
                    (self.ocr_worker.ocr_ready, self.on_ocr_ready),
                    (self.ocr_worker.ocr_progress, self.on_ocr_progress),
                ):
                    try:
                        signal.disconnect(slot)
                    except Exception:
                        pass
            old_worker = self.ocr_worker
            try:
                old_worker.stop()
                old_worker.wait(wait_ms)
            except Exception:
                pass
            self.retire_thread_if_needed(old_worker)
            self.ocr_worker = None

        if self.caption_worker is not None:
            if disconnect:
                for signal, slot in (
                    (self.caption_worker.caption_ready, self.on_caption_ready),
                    (self.caption_worker.caption_progress, self.on_caption_progress),
                ):
                    try:
                        signal.disconnect(slot)
                    except Exception:
                        pass
            old_worker = self.caption_worker
            try:
                old_worker.stop()
                old_worker.wait(wait_ms)
            except Exception:
                pass
            self.retire_thread_if_needed(old_worker)
            self.caption_worker = None

    def clear_post_processing_queues(self):
        """Clear queued OCR/caption work without starting another worker pool.

        Restart should not create a second OCR/PARSeq pool while the previous
        pool may still be finishing a long OCR call. That overlap was the main
        reason restarts could slow the whole app to a crawl.
        """
        for worker in (self.ocr_worker, self.caption_worker):
            if worker is not None and hasattr(worker, "clear_pending_tasks"):
                try:
                    worker.clear_pending_tasks()
                except Exception:
                    pass

    def ensure_post_processing_workers(self):
        need_ocr = self.ocr_worker is None
        need_caption = self.caption_worker is None
        try:
            need_ocr = need_ocr or not self.ocr_worker.isRunning()
        except Exception:
            need_ocr = True
        try:
            need_caption = need_caption or not self.caption_worker.isRunning()
        except Exception:
            need_caption = True
        if need_ocr or need_caption:
            self.stop_post_processing_workers(wait_ms=1500, disconnect=True)
            self.start_post_processing_workers()

    def init_menu_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        btn_file = QPushButton("Select Video File  [1]")
        btn_file.clicked.connect(self.run_file_input)
        btn_cam = QPushButton("Start Live Record  [2]")
        btn_cam.clicked.connect(self.run_camera_input)
        self.ocr_startup_status_label = QLabel("OCR/PARSeq: preparing background load...")
        self.ocr_startup_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_startup_status_label.setWordWrap(True)
        self.ocr_startup_status_label.setStyleSheet("color: #555; padding-top: 8px;")
        self.caption_startup_status_label = QLabel("Caption/BLIP: preparing background load...")
        self.caption_startup_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_startup_status_label.setWordWrap(True)
        self.caption_startup_status_label.setStyleSheet("color: #555; padding-top: 2px;")
        layout.addWidget(btn_file)
        layout.addWidget(btn_cam)
        layout.addWidget(self.ocr_startup_status_label)
        layout.addWidget(self.caption_startup_status_label)
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
        self.btn_restart = QPushButton("↻ Restart  [R]")
        self.btn_restart.setToolTip("Clear results and restart this video from the beginning")
        self.btn_restart.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_restart.clicked.connect(self.restart_video_from_beginning)

        self.btn_stop_proc = QPushButton("Stop Processing  [X]")
        self.btn_stop_proc.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
        self.btn_stop_proc.setVisible(False)
        self.btn_stop_proc.clicked.connect(self.stop_model_processing)

        self.btn_export_ocr = QPushButton("Export OCR Data  [E]")
        self.btn_export_ocr.setToolTip("Export poster-level and line-level OCR fields for ground-truth comparison")
        self.btn_export_ocr.setEnabled(False)
        self.btn_export_ocr.clicked.connect(self.export_ocr_comparison_data)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        button_layout.addWidget(btn_test)
        button_layout.addWidget(self.btn_restart)
        button_layout.addWidget(self.btn_stop_proc)
        button_layout.addWidget(self.btn_export_ocr)

        layout.addWidget(button_row)
        layout.addWidget(self.results_status_label)
        layout.addWidget(self.results_progress_bar)
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
            ("R", self.restart_video_from_beginning),
            ("X", self.stop_model_processing),
            ("E", self.export_ocr_comparison_data),
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
        # Be defensive: never let a previous detector worker keep emitting
        # frames/crops into a fresh run.
        self.stop_current_model_worker(wait_ms=1500, disconnect=True)
        self.clear_gallery()
        self.ocr_generation += 1
        self.post_processing_started = False
        self.pending_finished_records = None
        self.stack.setCurrentIndex(2)
        if hasattr(self, "btn_restart"):
            self.btn_restart.setEnabled(bool(self.video_path))
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
        worker_cls = {"yolo": YOLOWorker}.get(MODEL_BACKEND.lower())
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
        if hasattr(self.model_worker, "finished_records"):
            self.model_worker.finished_records.connect(self.on_finished_records)
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

    def disconnect_model_worker_signals(self, worker):
        if worker is None:
            return
        signal_slots = [
            (getattr(worker, "progress", None), self.on_progress),
            (getattr(worker, "poster_found", None), self.on_poster_found),
            (getattr(worker, "finished_processing", None), self.on_model_finished),
            (getattr(worker, "error", None), self.on_error),
            (getattr(worker, "poster_found_record", None), self.on_poster_found_record),
            (getattr(worker, "finished_records", None), self.on_finished_records),
            (getattr(worker, "frame_preview", None), self.on_detector_preview),
        ]
        for signal, slot in signal_slots:
            if signal is None:
                continue
            try:
                signal.disconnect(slot)
            except Exception:
                pass

    def stop_current_model_worker(self, wait_ms=2500, disconnect=True):
        worker = self.model_worker
        if worker is None:
            return
        if disconnect:
            self.disconnect_model_worker_signals(worker)
        try:
            if worker.isRunning():
                if hasattr(worker, "request_stop"):
                    worker.request_stop()
                else:
                    worker.terminate()
                if not worker.wait(wait_ms):
                    worker.terminate()
                    worker.wait(1000)
        except Exception:
            pass
        self.retire_thread_if_needed(worker)
        if self.model_worker is worker:
            self.model_worker = None

    def restart_video_from_beginning(self):
        if not self.video_path:
            QMessageBox.information(self, "Restart", "No video is available to restart yet.")
            return

        if hasattr(self, "btn_restart"):
            self.btn_restart.setEnabled(False)
        self.results_status_label.setVisible(True)
        self.results_progress_bar.setVisible(True)
        self.results_progress_bar.setValue(0)
        self.results_status_label.setText("Restarting video from the beginning...")
        QApplication.processEvents()

        # Stop any detector run and disconnect it first so stale finished/crop
        # signals cannot repopulate the gallery after it is cleared.
        self.stop_current_model_worker(wait_ms=2500, disconnect=True)

        # Clear pending OCR/caption jobs from the previous run, but do not
        # start a second OCR/PARSeq worker pool here. Any job already inside
        # OCR will finish in the background and be ignored by the generation
        # check; starting another pool immediately is what caused restarts to
        # slow down badly.
        self.ocr_generation += 1
        self.clear_post_processing_queues()

        self.clear_gallery()
        if hasattr(self, "btn_restart"):
            self.btn_restart.setEnabled(True)
        self.start_model_processing()

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

    def on_finished_records(self, records):
        if not isinstance(records, list):
            return
        self.pending_finished_records = records
        self.apply_finished_records(records)
        self.queue_post_processing_for_all()

    def apply_finished_records(self, records):
        # finished_records are the final end-of-video records. Prefer the crop
        # inside each record so OCR/captioning uses the final highest-quality best crop.
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            poster_id = record.get("track_id", record.get("id"))
            if poster_id is None:
                continue
            crop = record.get("crop")
            if crop is not None:
                if poster_id in self.poster_id_to_button:
                    self.update_poster_in_gallery(
                        poster_id, crop, "OCR pending...", None, defer_post_processing=True
                    )
                else:
                    self.add_poster_to_gallery(
                        crop, "OCR pending...", None, poster_id=poster_id, defer_post_processing=True
                    )
            btn = self.poster_id_to_button.get(poster_id)
            if btn is None:
                continue
            try:
                idx = self.poster_buttons.index(btn)
            except ValueError:
                continue
            poster = self.poster_data[idx]
            poster["track_meta"] = dict(record)
            label = poster.get("meta_label")
            if label is not None:
                label.setText(self.track_meta_text(record))
                label.setVisible(True)
                label.adjustSize()
        self.sort_gallery_by_quality()

    def queue_post_processing_for_all(self):
        if self.post_processing_started:
            return
        self.post_processing_started = True
        if not self.poster_data:
            self.update_global_processing_status()
            return
        if self.is_ocr_warmup_ready() and self.is_caption_warmup_ready():
            self.results_status_label.setText(
                f"Detector finished. Starting OCR/captions for {len(self.poster_data)} final best crop(s)..."
            )
        else:
            waiting_for = []
            if not self.is_ocr_warmup_ready():
                waiting_for.append("PARSeq")
            if not self.is_caption_warmup_ready():
                waiting_for.append("BLIP caption model")
            wait_text = " and ".join(waiting_for) if waiting_for else "background models"
            self.results_status_label.setText(
                f"Detector finished. Waiting for {wait_text}, then OCR/captions for {len(self.poster_data)} final best crop(s)..."
            )
        QApplication.processEvents()
        self.ensure_post_processing_workers()
        for idx, poster in enumerate(self.poster_data):
            image_input = poster.get("image_input")
            if image_input is None:
                continue
            poster["detected_text"] = "OCR pending..."
            poster["confidence"] = self.confidence_text(None)
            poster["ocr_details"] = None
            poster["description"] = "Caption queued"
            poster["ocr_done"] = False
            poster["caption_done"] = False
            poster["ocr_progress"] = 0
            poster["caption_progress"] = 0
            poster["ocr_stage"] = "OCR queued"
            poster["caption_stage"] = "Caption queued"
            self.update_poster_ready_state(idx)
            self.ocr_worker.add_task(
                poster["poster_key"], image_input, self.get_preprocess_options(), self.ocr_generation
            )
            try:
                _pixmap, pil_image = self.prepare_pixmap_and_pil(image_input)
            except Exception:
                pil_image = None
            self.caption_worker.add_task(poster["poster_key"], pil_image)
        self.update_global_processing_status()

    def on_model_finished(self, results=None, output_path=None):
        self.results_progress_bar.setVisible(False)
        self.btn_stop_proc.setVisible(False)
        self.btn_stop_proc.setEnabled(True)
        if not self.post_processing_started:
            # Fallback for engines that do not emit finished_records. Use the
            # current displayed crops, which should be the latest best crops.
            self.queue_post_processing_for_all()
        self.results_status_label.setText(
            f"Processing complete/stopped. Queued OCR/captions for {len(self.poster_data)} final best crop(s)."
        )

    def on_poster_found(self, poster_id, image_array):
        # During model processing, only update the gallery preview. Do NOT start
        # OCR or captioning here, because later frames may produce a better crop.
        if poster_id in self.poster_id_to_button:
            self.update_poster_in_gallery(
                poster_id, image_array, "OCR pending...", None, defer_post_processing=True
            )
        else:
            self.add_poster_to_gallery(
                image_array, "OCR pending...", None, poster_id=poster_id, defer_post_processing=True
            )

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
        if self.is_ocr_warmup_ready() and self.is_caption_warmup_ready():
            self.results_status_label.setText(f"Queued {len(files)} local image(s) for OCR/caption testing.")
        else:
            waiting_for = []
            if not self.is_ocr_warmup_ready():
                waiting_for.append("PARSeq")
            if not self.is_caption_warmup_ready():
                waiting_for.append("BLIP caption model")
            wait_text = " and ".join(waiting_for) if waiting_for else "background models"
            self.results_status_label.setText(
                f"Queued {len(files)} local image(s). Waiting for {wait_text} if still loading."
            )
        self.ensure_post_processing_workers()

        for file_path in files:
            self.add_poster_to_gallery(file_path, "OCR pending...", None)
            poster = self.poster_data[-1]
            self.ocr_worker.add_task(poster["poster_key"], file_path, self.get_preprocess_options(), self.ocr_generation)
            QApplication.processEvents()

    def add_poster_to_gallery(self, image_input, detected_text, avg_conf=None, poster_id=None, run_caption_async=True, defer_post_processing=False):
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
        ocr_done = detected_text not in PENDING_TEXTS and not defer_post_processing
        caption_done = (not run_caption_async) and not defer_post_processing
        poster = {
            "poster_key": poster_key,
            "poster_id": poster_id,
            "image_input": image_input,
            "detected_text": detected_text,
            "pixmap": full_pixmap,
            "description": "Waiting for final best crop..." if defer_post_processing else ("Generating description..." if run_caption_async else "Processing..."),
            "confidence": self.confidence_text(avg_conf),
            "ocr_details": None,
            "ocr_done": ocr_done,
            "caption_done": caption_done,
            "ocr_progress": 0 if defer_post_processing else (100 if ocr_done else 0),
            "caption_progress": 0 if (run_caption_async or defer_post_processing) else 100,
            "ocr_stage": "Waiting for video to finish" if defer_post_processing else ("OCR done" if ocr_done else "OCR queued"),
            "caption_stage": "Waiting for video to finish" if defer_post_processing else ("Caption queued" if run_caption_async else "Caption done"),
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
        if run_caption_async and not defer_post_processing:
            self.ensure_post_processing_workers()
            self.caption_worker.add_task(poster_key, pil_image)

    def update_poster_in_gallery(self, poster_id, image_input, detected_text, avg_conf=None, run_caption_async=True, defer_post_processing=False):
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
        if defer_post_processing:
            poster.update(
                image_input=image_input,
                detected_text="OCR pending...",
                pixmap=full_pixmap,
                confidence=self.confidence_text(None),
                ocr_details=None,
                description="Waiting for final best crop...",
                ocr_done=False,
                caption_done=False,
                ocr_progress=0,
                caption_progress=0,
                ocr_stage="Waiting for video to finish",
                caption_stage="Waiting for video to finish",
            )
        else:
            poster.update(
                image_input=image_input,
                detected_text=detected_text,
                pixmap=full_pixmap,
                confidence=self.confidence_text(avg_conf),
                ocr_details=None,
                description="Updating description..." if run_caption_async else poster.get("description", "Processing..."),
                ocr_done=detected_text not in PENDING_TEXTS,
                caption_done=not run_caption_async,
                ocr_progress=100 if detected_text not in PENDING_TEXTS else 0,
                caption_progress=0 if run_caption_async else 100,
                ocr_stage="OCR done" if detected_text not in PENDING_TEXTS else "OCR queued",
                caption_stage="Caption queued" if run_caption_async else "Caption done",
            )
        self.set_button_preview(btn, image_input)
        btn.setToolTip(f"Description: {poster.get('description', '')}")
        if run_caption_async and not defer_post_processing:
            self.ensure_post_processing_workers()
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
        avg_conf = None
        generation = self.ocr_generation
        ocr_details = None
        if isinstance(payload, (tuple, list)):
            if len(payload) >= 1:
                avg_conf = payload[0]
            if len(payload) >= 2:
                generation = payload[1]
            if len(payload) >= 3:
                ocr_details = payload[2]
        if generation != self.ocr_generation:
            return
        idx = self.find_poster_index_by_key(poster_key)
        if idx is None:
            return
        poster = self.poster_data[idx]
        poster["detected_text"] = detected_text
        poster["confidence"] = self.confidence_text(avg_conf)
        poster["ocr_details"] = ocr_details or {}
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

    def format_percent_score(self, value):
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def make_ocr_debug_text(self, poster, selected_candidate=None):
        if not isinstance(poster, dict):
            return "OCR details unavailable."
        sections = []
        if selected_candidate is not None:
            sections.append("SELECTED CROP CANDIDATE")
            sections.append(self.crop_candidate_text(selected_candidate))
            sections.append("")
        sections.append("OCR SUMMARY")
        sections.append(f"Detected text: {poster.get('detected_text', '(no text)')}")
        sections.append(f"{poster.get('confidence', 'Average OCR confidence: N/A')}")
        details = poster.get("ocr_details") or {}
        if not isinstance(details, dict) or not details:
            sections.append("")
            sections.append("No detailed OCR parsing data was stored for this image.")
            return "\n".join(sections)
        if details.get("error"):
            sections.append(f"Error: {details.get('error')}")
        sections.append(f"Method: {details.get('method', 'N/A')}")
        sections.append(f"Used inverted retry: {bool(details.get('used_inverted', False))}")
        sections.append(f"Raw EasyOCR regions: {details.get('raw_count', 'N/A')}")
        lines = details.get("lines") or []
        sections.append(f"Accepted OCR lines/regions: {len(lines)}")
        sections.append("")
        sections.append("LINE / REGION CONFIDENCES")
        if not lines:
            sections.append("No accepted OCR lines/regions.")
            return "\n".join(sections)
        for line in lines:
            idx = line.get("index", "?")
            line_text = line.get("text", "")
            conf = self.format_percent_score(line.get("conf"))
            method = line.get("method", "N/A")
            parseq_text = line.get("parseq_text", "")
            parseq_conf = self.format_percent_score(line.get("parseq_conf"))
            easy_text = line.get("easyocr_text", "")
            easy_conf = self.format_percent_score(line.get("easyocr_conf"))
            box = line.get("box")
            sections.append(f"[{idx}] {line_text}")
            sections.append(f"    final conf: {conf} | method: {method}")
            sections.append(f"    PARSeq: {parseq_conf} | {parseq_text or '(blank)'}")
            sections.append(f"    EasyOCR: {easy_conf} | {easy_text or '(blank)'}")
            if box is not None:
                sections.append(f"    text box: {box}")
            words = line.get("words") or []
            if words:
                word_bits = [f"{item.get('text', '')} ({self.format_percent_score(item.get('conf'))})" for item in words]
                sections.append("    words from this region: " + ", ".join(word_bits))
            sections.append("")
        return "\n".join(sections).rstrip()

    def crop_candidate_text(self, candidate):
        if not isinstance(candidate, dict):
            return "Crop candidate: N/A"
        q = self.format_score(candidate.get("quality"))
        conf = self.format_score(candidate.get("score"))
        frame = self.format_int(candidate.get("frame_index"))
        cand_idx = self.format_int(candidate.get("candidate_index"))
        width = self.format_int(candidate.get("width"))
        height = self.format_int(candidate.get("height"))
        area = self.format_int(candidate.get("area"))
        bbox = candidate.get("bbox")
        best = "BEST | " if candidate.get("is_best") else ""
        return f"{best}#{cand_idx} | Q: {q} | Conf: {conf}\nFrame: {frame} | Size: {width}×{height} | Area: {area}\nBBox: {bbox}"

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
        if hasattr(self, "btn_export_ocr"):
            self.btn_export_ocr.setEnabled(total > 0)
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


    def export_ocr_comparison_data(self):
        """Export only the OCR fields needed for comparison against manual readings.

        The export intentionally omits captions, thumbnails, crop image arrays,
        full crop histories, and other UI/debug fields. It writes:
          - poster_summary.csv: one row per detected poster/crop,
          - ocr_lines.csv: one row per accepted OCR line/region,
          - ocr_export.json: the same comparison data in a structured form.
        """
        if self.stack.currentIndex() != 2 or not self.poster_data:
            QMessageBox.information(self, "Export OCR Data", "No OCR results are available to export yet.")
            return

        total = len(self.poster_data)
        ocr_done = sum(1 for poster in self.poster_data if poster.get("ocr_done"))
        if ocr_done < total:
            reply = QMessageBox.question(
                self,
                "Export OCR Data",
                f"OCR is still pending for {total - ocr_done} of {total} poster(s). Export the partial data anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        export_root = SCRIPT_DIR / "ocr_exports"
        export_root.mkdir(parents=True, exist_ok=True)
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose folder for OCR comparison export",
            str(export_root),
        )
        if not selected_dir:
            return

        video_stem = self._safe_filename(Path(self.video_path).stem if self.video_path else "manual_images")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(selected_dir) / f"ocr_export_{video_stem}_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            poster_rows, line_rows, payload = self.build_ocr_comparison_export()
            self.write_csv(export_dir / "poster_summary.csv", poster_rows, self.poster_summary_export_fields())
            self.write_csv(export_dir / "ocr_lines.csv", line_rows, self.ocr_line_export_fields())
            with open(export_dir / "ocr_export.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Export OCR Data", f"Failed to export OCR data:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Export OCR Data",
            "OCR comparison export complete.\n\n"
            f"Folder:\n{export_dir}\n\n"
            "Files:\nposter_summary.csv\nocr_lines.csv\nocr_export.json",
        )

    def build_ocr_comparison_export(self):
        exported_at = datetime.now().isoformat(timespec="seconds")
        source_path = str(self.video_path or "")
        source_video = Path(source_path).name if source_path else ""
        poster_rows = []
        line_rows = []
        json_posters = []

        for rank, poster in enumerate(self.poster_data, 1):
            details = poster.get("ocr_details") or {}
            if not isinstance(details, dict):
                details = {}
            meta = poster.get("track_meta") or {}
            if not isinstance(meta, dict):
                meta = {}

            lines = details.get("lines") or []
            if not isinstance(lines, list):
                lines = []

            poster_key = poster.get("poster_key", "")
            track_id = meta.get("track_id", meta.get("id", poster.get("poster_id", poster_key)))
            crop_w, crop_h = self.image_input_size(poster.get("image_input"))
            bbox_value = meta.get("bbox", meta.get("box", meta.get("xyxy")))
            safe_meta = self.filtered_track_meta_for_export(meta)

            summary_row = {
                "exported_at": exported_at,
                "source_video": source_video,
                "source_path": source_path,
                "gallery_rank": rank,
                "poster_key": poster_key,
                "track_id": track_id,
                "ocr_done": bool(poster.get("ocr_done", False)),
                "ocr_text": details.get("text") or poster.get("detected_text", ""),
                "avg_conf": details.get("avg_conf", ""),
                "method": details.get("method", ""),
                "line_count": details.get("line_count", len(lines)),
                "raw_region_count": details.get("raw_count", ""),
                "used_inverted": bool(details.get("used_inverted", False)),
                "detector_quality": meta.get("quality", ""),
                "detector_conf": meta.get("score", ""),
                "seen_count": meta.get("seen_count", ""),
                "version": meta.get("version", ""),
                "frame_index": meta.get("frame_index", meta.get("best_frame_index", "")),
                "bbox_json": self.to_json_cell(bbox_value),
                "crop_width": crop_w,
                "crop_height": crop_h,
                "track_meta_json": self.to_json_cell(safe_meta),
            }
            poster_rows.append(summary_row)

            json_line_rows = []
            for line_number, line in enumerate(lines, 1):
                if not isinstance(line, dict):
                    continue
                line_row = {
                    "exported_at": exported_at,
                    "source_video": source_video,
                    "gallery_rank": rank,
                    "poster_key": poster_key,
                    "track_id": track_id,
                    "line_index": line.get("index", line_number),
                    "row_index": line.get("row_index", ""),
                    "word_index": line.get("word_index", ""),
                    "line_text": line.get("text", ""),
                    "line_conf": line.get("conf", ""),
                    "method": line.get("method", ""),
                    "parseq_text": line.get("parseq_text", ""),
                    "parseq_conf": line.get("parseq_conf", ""),
                    "easyocr_text": line.get("easyocr_text", ""),
                    "easyocr_conf": line.get("easyocr_conf", ""),
                    "line_break_after": bool(line.get("line_break_after", False)),
                    "box_json": self.to_json_cell(line.get("box")),
                }
                line_rows.append(line_row)
                json_line_rows.append(dict(line_row))

            json_posters.append({
                "summary": dict(summary_row),
                "lines": json_line_rows,
            })

        payload = {
            "exported_at": exported_at,
            "source_video": source_video,
            "source_path": source_path,
            "poster_count": len(poster_rows),
            "line_count": len(line_rows),
            "notes": "Comparison export only: no captions, thumbnails, crop arrays, or full crop histories included.",
            "posters": json_posters,
        }
        return poster_rows, line_rows, payload

    def poster_summary_export_fields(self):
        return [
            "exported_at",
            "source_video",
            "source_path",
            "gallery_rank",
            "poster_key",
            "track_id",
            "ocr_done",
            "ocr_text",
            "avg_conf",
            "method",
            "line_count",
            "raw_region_count",
            "used_inverted",
            "detector_quality",
            "detector_conf",
            "seen_count",
            "version",
            "frame_index",
            "bbox_json",
            "crop_width",
            "crop_height",
            "track_meta_json",
        ]

    def ocr_line_export_fields(self):
        return [
            "exported_at",
            "source_video",
            "gallery_rank",
            "poster_key",
            "track_id",
            "line_index",
            "row_index",
            "word_index",
            "line_text",
            "line_conf",
            "method",
            "parseq_text",
            "parseq_conf",
            "easyocr_text",
            "easyocr_conf",
            "line_break_after",
            "box_json",
        ]

    def write_csv(self, path, rows, fieldnames):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def image_input_size(self, image_input):
        try:
            if isinstance(image_input, (str, Path)):
                with Image.open(image_input) as img:
                    return int(img.width), int(img.height)
            if image_input is not None and hasattr(image_input, "shape"):
                height, width = image_input.shape[:2]
                return int(width), int(height)
        except Exception:
            pass
        return "", ""

    def filtered_track_meta_for_export(self, meta):
        if not isinstance(meta, dict):
            return {}
        skip = {"crop", "all_crops", "image", "image_array", "frame", "preview"}
        keep = {}
        for key, value in meta.items():
            if key in skip:
                continue
            if isinstance(value, dict) and any(k in value for k in skip):
                value = {k: v for k, v in value.items() if k not in skip}
            keep[key] = self.json_safe(value)
        return keep

    def to_json_cell(self, value):
        if value is None:
            return ""
        if isinstance(value, str) and value == "":
            return ""
        return json.dumps(self.json_safe(value), ensure_ascii=False)

    def json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            try:
                return self.json_safe(value.tolist())
            except Exception:
                pass
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key in {"crop", "all_crops", "image", "image_array", "frame", "preview"}:
                    continue
                out[str(key)] = self.json_safe(item)
            return out
        if isinstance(value, (list, tuple, set)):
            return [self.json_safe(item) for item in value]
        try:
            return value.item()
        except Exception:
            return str(value)

    def _safe_filename(self, value):
        text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "export"))
        text = text.strip("_")
        return text or "export"

    def clear_gallery(self):
        self.last_detector_frame = None
        self.detector_preview_expanded = False
        self.post_processing_started = False
        self.pending_finished_records = None
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
        self.show_zoom_dialog(self.make_display_pixmap(poster["image_input"]), display_text, poster)

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

    def show_zoom_dialog(self, color_pixmap, text, poster=None):
        dialog = QDialog(self)
        dialog.setWindowTitle("Poster Viewer — Color Image")
        dialog.resize(1100, 760)
        layout = QVBoxLayout(dialog)
        title = QLabel("Color Image")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_view = ZoomableImageLabel(color_pixmap, dialog)
        image_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        text_box = QPlainTextEdit(dialog)
        text_box.setReadOnly(True)
        text_box.setPlainText(self.make_ocr_debug_text(poster))
        text_box.setMinimumHeight(150)
        text_box.setMaximumHeight(210)
        text_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        text_box.setStyleSheet("padding: 8px; background-color: #1e1e1e; color: white; font-family: Menlo, Consolas, monospace; font-size: 11px;")

        main_row = QWidget(dialog)
        main_layout = QHBoxLayout(main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        main_layout.addWidget(image_view, stretch=1)

        crop_candidates = []
        if isinstance(poster, dict):
            meta = poster.get("track_meta") or {}
            raw_candidates = meta.get("all_crops") or []
            if isinstance(raw_candidates, list):
                crop_candidates = [item for item in raw_candidates if isinstance(item, dict)]

        if crop_candidates:
            crop_candidates = sorted(
                crop_candidates,
                key=lambda item: (
                    -float(item.get("quality", float("-inf"))) if item.get("quality") is not None else float("inf"),
                    int(item.get("candidate_index", 0)),
                ),
            )

            side_panel = QWidget(dialog)
            side_layout = QVBoxLayout(side_panel)
            side_layout.setContentsMargins(6, 6, 6, 6)
            side_layout.setSpacing(8)

            header = QLabel(f"Stored crop candidates: {len(crop_candidates)}")
            header.setWordWrap(True)
            header.setStyleSheet("font-weight: bold; padding-bottom: 4px;")
            side_layout.addWidget(header)

            def candidate_summary(candidate):
                return self.crop_candidate_text(candidate)

            def set_dialog_image(pixmap, candidate=None):
                if pixmap is None or pixmap.isNull():
                    return
                image_view._original_pixmap = pixmap
                image_view._zoom = 1.0
                image_view._pan_offset = QPoint(0, 0)
                image_view._update_display()
                text_box.setPlainText(self.make_ocr_debug_text(poster, candidate))
                text_box.moveCursor(text_box.textCursor().MoveOperation.Start)

            for candidate in crop_candidates:
                crop = candidate.get("crop")
                if crop is None:
                    continue
                cand_pixmap = self.make_display_pixmap(crop)
                if cand_pixmap.isNull():
                    continue
                thumb = cand_pixmap.scaled(190, 130, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

                card = QWidget(side_panel)
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(4, 4, 4, 4)
                card_layout.setSpacing(4)

                btn = QPushButton()
                btn.setIcon(QIcon(thumb))
                btn.setIconSize(thumb.size())
                btn.setFixedSize(210, max(80, thumb.height() + 14))
                btn.setToolTip("Click to show this stored crop in the main viewer")
                if candidate.get("is_best"):
                    btn.setStyleSheet("border: 2px solid #0078d7; background-color: rgba(0, 120, 215, 18);")

                info = QLabel(candidate_summary(candidate))
                info.setWordWrap(True)
                info.setStyleSheet("font-size: 10px; color: #333;")

                btn.clicked.connect(lambda checked=False, p=cand_pixmap, c=candidate: set_dialog_image(p, c))
                card_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
                card_layout.addWidget(info)
                side_layout.addWidget(card)

            side_layout.addStretch(1)
            crop_scroll = QScrollArea(dialog)
            crop_scroll.setWidget(side_panel)
            crop_scroll.setWidgetResizable(True)
            crop_scroll.setFixedWidth(260)
            crop_scroll.setMinimumHeight(420)
            main_layout.addWidget(crop_scroll)

        layout.addWidget(title)
        layout.addWidget(main_row, stretch=1)
        layout.addWidget(text_box)
        image_view.setFocus()
        dialog.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "detector_preview_expanded", False) and hasattr(self, "detector_overlay_label"):
            self.detector_overlay_label.setGeometry(self.results_page.rect())
            self.update_detector_preview_pixmaps()

    def closeEvent(self, event):
        self.stop_current_model_worker(wait_ms=1000, disconnect=True)
        self.stop_post_processing_workers(wait_ms=1000, disconnect=True)
        if self.ocr_warmup_worker is not None:
            try:
                self.ocr_warmup_worker.status_changed.disconnect(self.on_ocr_warmup_status)
            except Exception:
                pass
            try:
                self.ocr_warmup_worker.finished_loading.disconnect(self.on_ocr_warmup_finished)
            except Exception:
                pass
            try:
                if not self.ocr_warmup_worker.wait(500):
                    self.ocr_warmup_worker.terminate()
                    self.ocr_warmup_worker.wait(500)
            except Exception:
                pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PosterReaderApp()
    window.show()
    sys.exit(app.exec())
