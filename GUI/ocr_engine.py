import re

import cv2
import easyocr
import numpy as np



class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def get_text(self, image_input):
        """
        Takes a path (str) or a NumPy array (OpenCV frame).
        Returns a single string of all detected text.
        """
        try:
            if isinstance(image_input, str):
                frame = cv2.imread(image_input)
            else:
                frame = image_input

            if frame is None:
                return "Error: Image not found or empty."

            processed_frame = self.preprocess_for_ocr(frame)

            results = self.reader.readtext(processed_frame)

            extracted_text = " ".join([res[1] for res in results if res[2] > 0.4]) # need to tune this 
            extracted_text = self.clean_string(extracted_text)
            return extracted_text if extracted_text else "No text detected."

        except Exception as e:
            return f"OCR Error: {str(e)}"

    def clean_string(self, text):
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!]', '', text)
        return " ".join(cleaned.split())

    def preprocess_for_ocr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        return gray


if __name__ == "__main__":
    test_engine = OCREngine()
    print(f"Detected: {test_engine.get_text('..\\poster_downloads\\3.8_52888.jpg')}")