
import sys

import cv2
from PIL import Image
from PyQt6 import QtGui
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QKeySequence, QShortcut
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, \
    QLabel, QWidget, QScrollArea, QGridLayout, QStackedWidget, QDialog, QCheckBox, QComboBox

from audio_engine import AudioEngine
from caption_engine import ImageCaptioner
from ocr_engine import OCREngine

# For quick debugging and testing
LOAD_POSTERS_START_DIR = ""
LOAD_VIDEO_START_DIR = ""

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
        scaled_w = int(self._original_pixmap.width() * self._zoom)
        scaled_h = int(self._original_pixmap.height() * self._zoom)
        scaled = self._original_pixmap.scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
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
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Video", LOAD_VIDEO_START_DIR, "Video Files (*.mp4 *.avi *.mov)")
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


class PosterReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.manager = VideoSourceManager()
        self.current_cap = None
        self.found_posters = []

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

        QShortcut(QKeySequence("1"), self, activated=self.run_file_input)
        QShortcut(QKeySequence("2"), self, activated=self.run_camera_input)

    def run_file_input(self):
        if self.stack.currentIndex() != 0:
            return
        cap, path = self.manager.select_file()
        if cap:
            self.current_cap = cap
            self.start_pipeline()

    def run_camera_input(self):
        if self.stack.currentIndex() != 0:
            return
        cap = self.manager.start_camera()
        if cap:
            self.current_cap = cap
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

        QShortcut(QKeySequence("Space"), self, activated=self.stop_processing)
        QShortcut(QKeySequence("Escape"), self, activated=self.stop_processing)

    def process_frame(self):
        ret, frame = self.current_cap.read()
        if not ret:
            self.stop_processing()
            return

        # PIPELINE HERE
        # here is where we need to add the tracking of the posters to show on the frame

        # Display the frame in the GUI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))

    def stop_processing(self):
        if self.stack.currentIndex() != 1:
            return
        self.timer.stop()
        if self.current_cap:
            self.current_cap.release()
        self.stack.setCurrentIndex(2)

    def init_results_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.ocr_result_label = QLabel(
            "Choose OCR preprocessing options, then press T to load test images. "
            "Navigate posters with Arrow Keys. Press Enter to open."
        )
        self.ocr_result_label.setWordWrap(True)
        layout.addWidget(self.ocr_result_label)

        preprocess_layout = QHBoxLayout()

        self.chk_grayscale = QCheckBox("Grayscale")
        self.chk_threshold = QCheckBox("Adaptive Threshold")
        self.chk_sharpen = QCheckBox("Sharpen")
        self.chk_median = QCheckBox("Median Filter")

        self.scale_combo = QComboBox()
        self.scale_combo.addItem("0.5x", 0.5)
        self.scale_combo.addItem("1x", 1.0)
        self.scale_combo.addItem("2x", 2.0)
        self.scale_combo.addItem("4x", 4.0)
        self.scale_combo.setCurrentIndex(1)

        self.chk_grayscale.stateChanged.connect(self.refresh_gallery_previews)
        self.chk_threshold.stateChanged.connect(self.refresh_gallery_previews)
        self.chk_sharpen.stateChanged.connect(self.refresh_gallery_previews)
        self.chk_median.stateChanged.connect(self.refresh_gallery_previews)
        self.scale_combo.currentIndexChanged.connect(self.refresh_gallery_previews)

        preprocess_layout.addWidget(QLabel("OCR Preprocessing:"))
        preprocess_layout.addWidget(self.chk_grayscale)
        preprocess_layout.addWidget(self.chk_threshold)
        preprocess_layout.addWidget(self.chk_sharpen)
        preprocess_layout.addWidget(self.chk_median)
        preprocess_layout.addWidget(QLabel("Scale:"))
        preprocess_layout.addWidget(self.scale_combo)

        layout.addLayout(preprocess_layout)



        scroll = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self.grid_widget)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)

        # ADDED FOR TESTING: A button to manually load test images
        btn_test = QPushButton("Test OCR: Load Local Images  [T]")
        btn_test.clicked.connect(self.load_test_posters)
        layout.addWidget(btn_test)

        page.setLayout(layout)
        self.stack.addWidget(page)

        QShortcut(QKeySequence("T"), self, activated=self.load_test_posters)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.move_poster_focus(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.move_poster_focus(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, activated=lambda: self.move_poster_focus(4))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, activated=lambda: self.move_poster_focus(-4))
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.open_focused_poster)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self.open_focused_poster)
        self.poster_buttons = []
        self.poster_data = []
        self.focused_poster_index = -1

    def get_preprocess_options(self):
        """Read OCR preprocessing options from the GUI controls."""
        return {
            "grayscale": self.chk_grayscale.isChecked(),
            "threshold": self.chk_threshold.isChecked(),
            "sharpen": self.chk_sharpen.isChecked(),
            "median": self.chk_median.isChecked(),
            "scale": self.scale_combo.currentData(),
        }

    def load_test_posters(self):
        """Manually select images to test the gallery and OCR display."""
        if self.stack.currentIndex() != 2:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Posters", LOAD_POSTERS_START_DIR, "Images (*.png *.jpg *.jpeg)")

        preprocess_options = self.get_preprocess_options()

        for file_path in files:
            detected_text, avg_conf = self.ocr_engine.get_text(file_path, preprocess_options)
            self.add_poster_to_gallery(file_path, detected_text, avg_conf)

    def add_poster_to_gallery(self, image_input, detected_text, avg_conf=None):
        """Adds a clickable thumbnail and handles rotation."""
        btn = QPushButton()

        if isinstance(image_input, str):
            # It's a file path
            full_pixmap = QPixmap(image_input)
            pil_image = Image.open(image_input).convert('RGB')
        else:
            #TODO TEST THIS WHEN OTHER FUNCTIONS ARE DONE

            # It's a NumPy array (OpenCV frame)
            # Convert BGR to RGB for PIL and QImage
            rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            full_pixmap = QPixmap.fromImage(q_img)

            pil_image = Image.fromarray(rgb_image)

        short_description = self.captioner.generate_caption(pil_image)
        btn.setToolTip(f"Description: {short_description}")

        self.set_button_preview(btn, image_input)

        btn.setStyleSheet("""
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
                """)

        self.poster_buttons.append(btn)


        if avg_conf is None:
            confidence_text = "Average OCR confidence: N/A"
        else:
            confidence_text = f"Average OCR confidence: {avg_conf:.2%}"
        
        self.poster_data.append({
            "image_input": image_input,
            "detected_text": detected_text,
            "pixmap": full_pixmap,
            "description": short_description,
            "confidence": confidence_text,
        })

        def on_poster_click(text, pixmap, confidence, image_input):
            display_text = f"{text}\n\n{confidence}"
            self.ocr_result_label.setText(display_text)

            # Only read detected text aloud, not confidence.
            self.speaker.speak(text)

            processed_pixmap = self.make_processed_pixmap(image_input)
            self.show_zoom_dialog(pixmap, processed_pixmap, display_text)

        btn.clicked.connect(
            lambda checked=False, t=detected_text, p=full_pixmap, c=confidence_text, img=image_input:
            on_poster_click(t, p, c, img)
        )
        btn.installEventFilter(self)

        count = self.grid_layout.count()
        self.grid_layout.addWidget(btn, count // 4, count % 4)

        if self.focused_poster_index == -1:
            self.focused_poster_index = 0
            self.poster_buttons[0].setFocus()

    def make_processed_pixmap(self, image_input):
        """Create a QPixmap showing the currently selected OCR preprocessing result."""
        options = self.get_preprocess_options()

        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
        else:
            frame = image_input.copy()

        if frame is None:
            return QPixmap()

        processed = self.ocr_engine.preprocess_for_ocr(frame, options)

        if len(processed.shape) == 2:
            h, w = processed.shape
            q_img = QImage(
                processed.data,
                w,
                h,
                w,
                QImage.Format.Format_Grayscale8
            )
        else:
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(
                rgb.data,
                w,
                h,
                ch * w,
                QImage.Format.Format_RGB888
            )

        return QPixmap.fromImage(q_img.copy())



    def move_poster_focus(self, delta):
        """Move focus among poster thumbnails on the results screen."""
        if self.stack.currentIndex() != 2 or not self.poster_buttons:
            return

        if self.focused_poster_index < 0:
            idx = 0
        else:
            idx = self.focused_poster_index + delta

        idx = max(0, min(idx, len(self.poster_buttons) - 1))

        self.focused_poster_index = idx
        self.poster_buttons[idx].setFocus()

        poster = self.poster_data[idx]
        self.ocr_result_label.setText(f"Description: {poster['description']}")


    def open_focused_poster(self):
        """Open the currently focused poster and read only detected text aloud."""
        if self.stack.currentIndex() != 2 or not self.poster_buttons:
            return

        if self.focused_poster_index < 0:
            self.focused_poster_index = 0
            self.poster_buttons[0].setFocus()

        poster = self.poster_data[self.focused_poster_index]

        text = poster["detected_text"]
        pixmap = poster["pixmap"]
        confidence = poster["confidence"]
        image_input = poster["image_input"]

        display_text = f"{text}\n\n{confidence}"

        self.ocr_result_label.setText(display_text)
        self.speaker.speak(text)

        processed_pixmap = self.make_processed_pixmap(image_input)
        self.show_zoom_dialog(pixmap, processed_pixmap, display_text)

    def eventFilter(self, obj, event):
        """Track which poster button has keyboard focus."""
        from PyQt6.QtCore import QEvent

        if event.type() == QEvent.Type.FocusIn and obj in self.poster_buttons:
            self.focused_poster_index = self.poster_buttons.index(obj)

        return super().eventFilter(obj, event)

    def refresh_gallery_previews(self):
        """Refresh all gallery thumbnails using the current preprocessing options."""
        if not self.poster_buttons or not self.poster_data:
            return

        for btn, poster in zip(self.poster_buttons, self.poster_data):
            self.set_button_preview(btn, poster["image_input"])

        self.ocr_result_label.setText(
            "Gallery previews updated with current preprocessing options. "
            "Navigate posters with Arrow Keys. Press Enter to open."
        )

    def set_button_preview(self, btn, image_input):
        """Update one gallery button to show the current preprocessed preview."""
        preview_pixmap = self.make_processed_pixmap(image_input)

        # Fallback to original if preprocessing failed for some reason.
        if preview_pixmap.isNull():
            if isinstance(image_input, str):
                preview_pixmap = QPixmap(image_input)
            else:
                rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                q_img = QImage(
                    rgb_image.data,
                    w,
                    h,
                    ch * w,
                    QImage.Format.Format_RGB888
                )
                preview_pixmap = QPixmap.fromImage(q_img.copy())

        thumb_max = 220
        thumbnail = preview_pixmap.scaled(
            thumb_max,
            thumb_max,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        btn.setIcon(QIcon(thumbnail))
        btn.setIconSize(thumbnail.size())
        btn.setFixedSize(thumbnail.width() + 8, thumbnail.height() + 8)

    def show_zoom_dialog(self, original_pixmap, processed_pixmap, text):
        """Opens a dialog with original and preprocessed poster views."""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Poster Viewer — Original vs Preprocessed")
            dialog.resize(1200, 700)

            main_layout = QVBoxLayout(dialog)
            image_layout = QHBoxLayout()
            image_layout.setContentsMargins(0, 0, 0, 0)

            left_layout = QVBoxLayout()
            left_title = QLabel("Original")
            left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left_view = ZoomableImageLabel(original_pixmap, dialog)
            left_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            left_layout.addWidget(left_title)
            left_layout.addWidget(left_view, stretch=1)

            right_layout = QVBoxLayout()
            right_title = QLabel("Preprocessed for OCR")
            right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_view = ZoomableImageLabel(processed_pixmap, dialog)
            right_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            right_layout.addWidget(right_title)
            right_layout.addWidget(right_view, stretch=1)

            image_layout.addLayout(left_layout)
            image_layout.addLayout(right_layout)
            main_layout.addLayout(image_layout, stretch=1)

            text_label = QLabel(text)
            text_label.setWordWrap(True)
            text_label.setMaximumHeight(100)
            text_label.setStyleSheet("padding: 8px; background-color: #1e1e1e; color: white;")
            main_layout.addWidget(text_label)

            right_view.setFocus()
            dialog.exec()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Zoom Viewer Error", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PosterReaderApp()
    window.show()
    sys.exit(app.exec())
