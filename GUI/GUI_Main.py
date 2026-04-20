import sys

import cv2
from PIL import Image
from PyQt6 import QtGui
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication, QPushButton, QVBoxLayout, QLabel, QWidget, \
    QScrollArea, QGridLayout, QStackedWidget

from GUI.audio_engine import AudioEngine
from GUI.caption_engine import ImageCaptioner


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

        btn_file = QPushButton("Select Video File")
        btn_file.clicked.connect(self.run_file_input)

        btn_cam = QPushButton("Start Live Record")
        btn_cam.clicked.connect(self.run_camera_input)

        layout.addWidget(btn_file)
        layout.addWidget(btn_cam)
        self.stack.addWidget(page)

    def run_file_input(self):
        cap, path = self.manager.select_file()
        if cap:
            self.current_cap = cap
            self.start_pipeline()

    def run_camera_input(self):
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

        btn_stop = QPushButton("Finish and View Posters")
        btn_stop.clicked.connect(self.stop_processing)

        self.proc_layout.addWidget(self.video_label)
        self.proc_layout.addWidget(btn_stop)
        self.stack.addWidget(page)

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
        self.timer.stop()
        if self.current_cap:
            self.current_cap.release()
        self.stack.setCurrentIndex(2)

    def init_results_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.ocr_result_label = QLabel("Click a poster to see its text")
        self.ocr_result_label.setWordWrap(True)
        layout.addWidget(self.ocr_result_label)

        scroll = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        scroll.setWidget(self.grid_widget)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)

        # ADDED FOR TESTING: A button to manually load test images
        btn_test = QPushButton("Test OCR: Load Local Images")
        btn_test.clicked.connect(self.load_test_posters)
        layout.addWidget(btn_test)

        page.setLayout(layout)
        self.stack.addWidget(page)

    def load_test_posters(self):
        """Manually select images to test the gallery and OCR display."""
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
            pixmap = QPixmap(image_input)
            pil_image = Image.open(image_input).convert('RGB')
        else:
            #TODO TEST THIS WHEN OTHER FUNCTIONS ARE DONE

            # It's a NumPy array (OpenCV frame)
            # Convert BGR to RGB for PIL and QImage
            rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # Create QImage then QPixmap
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            pil_image = Image.fromarray(rgb_image)

        short_description = self.captioner.generate_caption(pil_image)
        btn.setToolTip(f"Description: {short_description}")

        if pixmap.width() > pixmap.height():
            transform = QtGui.QTransform().rotate(90)
            pixmap = pixmap.transformed(transform)

        pixmap = pixmap.scaled(
            150, 150,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        btn.setIcon(QIcon(pixmap))
        btn.setIconSize(pixmap.size())
        btn.setFixedSize(160, 160)

        btn.setStyleSheet("""
                    QPushButton {
                        border: 2px solid transparent;
                        border-radius: 5px;
                        background-color: transparent;
                    }
                    QPushButton:hover {
                        border: 2px solid #0078d7;
                        background-color: rgba(0, 120, 215, 20);
                    }
                """)

        btn.clicked.connect(lambda checked, t=detected_text: on_poster_click(t))

        count = self.grid_layout.count()
        self.grid_layout.addWidget(btn, count // 4, count % 4)

        def on_poster_click(text):
            self.ocr_result_label.setText(text)
            self.speaker.speak(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PosterReaderApp()
    window.show()
    sys.exit(app.exec())
