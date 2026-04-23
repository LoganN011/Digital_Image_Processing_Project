import sys

import cv2
from PIL import Image
from PyQt6 import QtGui
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QKeySequence, QShortcut
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, \
    QLabel, QWidget, QScrollArea, QGridLayout, QStackedWidget, QDialog

from audio_engine import AudioEngine
from caption_engine import ImageCaptioner


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
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
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

        self.ocr_result_label = QLabel("Navigate posters with Arrow Keys. Press Enter to open. Press T to load test images.")
        self.ocr_result_label.setWordWrap(True)
        layout.addWidget(self.ocr_result_label)

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

        self.poster_buttons = []
        self.poster_data = []
        self.focused_poster_index = -1

    def load_test_posters(self):
        """Manually select images to test the gallery and OCR display."""
        if self.stack.currentIndex() != 2:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Posters", "", "Images (*.png *.jpg *.jpeg)")

        for file_path in files:
            # In the real pipeline, this text would come from the OCR engine
            mock_ocr_text = f"OCR Output for {file_path.split('/')[-1]}: Sample Text Detected"
            self.add_poster_to_gallery(file_path, mock_ocr_text)

    def add_poster_to_gallery(self, image_input, detected_text):
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

        thumb_max = 220
        thumbnail = full_pixmap.scaled(
            thumb_max, thumb_max,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        btn.setIcon(QIcon(thumbnail))
        btn.setIconSize(thumbnail.size())
        btn.setFixedSize(thumbnail.width() + 8, thumbnail.height() + 8)

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
        self.poster_data.append((detected_text, full_pixmap, short_description))

        def on_poster_click(text, pixmap):
            self.ocr_result_label.setText(text)
            self.speaker.speak(text)
            self.show_zoom_dialog(pixmap, text)

        btn.clicked.connect(lambda checked, t=detected_text, p=full_pixmap: on_poster_click(t, p))
        btn.installEventFilter(self)

        count = self.grid_layout.count()
        self.grid_layout.addWidget(btn, count // 4, count % 4)

    def eventFilter(self, obj, event):
        """When a poster button gains focus, show its description in the label."""
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.FocusIn and obj in self.poster_buttons:
            idx = self.poster_buttons.index(obj)
            self.focused_poster_index = idx
            detected_text, _, description = self.poster_data[idx]
            self.ocr_result_label.setText(f"Description: {description}")
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle arrow key navigation and Enter to open poster in results screen."""
        # Only handle when on the results screen (index 2)
        if self.stack.currentIndex() != 2 or not self.poster_buttons:
            super().keyPressEvent(event)
            return

        key = event.key()
        cols = 4
        idx = self.focused_poster_index

        if key == Qt.Key.Key_Right:
            idx = min(idx + 1, len(self.poster_buttons) - 1)
        elif key == Qt.Key.Key_Left:
            idx = max(idx - 1, 0)
        elif key == Qt.Key.Key_Down:
            idx = min(idx + cols, len(self.poster_buttons) - 1)
        elif key == Qt.Key.Key_Up:
            idx = max(idx - cols, 0)
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if 0 <= idx < len(self.poster_data):
                text, pixmap, _ = self.poster_data[idx]
                self.ocr_result_label.setText(text)
                self.speaker.speak(text)
                self.show_zoom_dialog(pixmap, text)
            return
        else:
            super().keyPressEvent(event)
            return

        self.focused_poster_index = idx
        self.poster_buttons[idx].setFocus()

    def show_zoom_dialog(self, pixmap, text):
        """Opens a dialog with the full-resolution poster that supports zoom and pan."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Poster Viewer — +/- to zoom, Arrows to pan, 0 to reset, Esc to close")
        dialog.resize(800, 700)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        zoom_label = ZoomableImageLabel(pixmap, dialog)
        zoom_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        zoom_label.setSizePolicy(zoom_label.sizePolicy())
        layout.addWidget(zoom_label, stretch=1)

        # Text label at the bottom
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setMaximumHeight(80)
        text_label.setStyleSheet("padding: 8px; background-color: #1e1e1e; color: white;")
        layout.addWidget(text_label)
        zoom_label.setFocus()

        dialog.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PosterReaderApp()
    window.show()
    sys.exit(app.exec())
