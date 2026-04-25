import re

import cv2
import easyocr
import numpy as np



class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def get_text(self, image_input, preprocess_options=None):
        """
        Takes a path (str) or a NumPy array (OpenCV frame).
        Returns a tuple of the detected text and the average confidence score.

        preprocess_options example:
        {
            "grayscale": True,
            "scale": 2.0,
            "threshold": False,
            "sharpen": False
        }
        """
        try:
            if isinstance(image_input, str):
                frame = cv2.imread(image_input)
            else:
                frame = image_input

            if frame is None:
                return "Error: Image not found or empty.", None

            if preprocess_options is None:
                preprocess_options = {
                    "grayscale": False,
                    "scale": 1.0,
                    "threshold": False,
                    "sharpen": False,
                }

            processed_frame = self.preprocess_for_ocr(frame, preprocess_options)

            results = self.reader.readtext(processed_frame)

            extracted_text = " ".join([res[1] for res in results])
            extracted_text = self.clean_string(extracted_text)

            if results:
                avg_conf = sum([res[2] for res in results]) / len(results)
            else:
                avg_conf = None

            return extracted_text if extracted_text else "No text detected.", avg_conf

        except Exception as e:
            return f"OCR Error: {str(e)}", None

    def clean_string(self, text):
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!]', '', text)
        return " ".join(cleaned.split())

    def preprocess_for_ocr(self, frame, options):
        """
        Applies user-selected preprocessing before EasyOCR.
        """

        processed = frame.copy()

        scale = options.get("scale", 1.0)
        grayscale = options.get("grayscale", False)
        threshold = options.get("threshold", False)
        sharpen = options.get("sharpen", False)
        median = options.get("median", False)

        if scale != 1.0:
            processed = cv2.resize(
                processed,
                None,       # expects dsize or fx/fy, not both
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            )

        if median:
            processed = cv2.medianBlur(processed, 3)

        if grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        if sharpen:
            blurred = cv2.GaussianBlur(processed, (0, 0), 1.2)
            processed = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)

        if threshold:
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            processed = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

        return processed

if __name__ == "__main__":
    test_engine = OCREngine()
    print(f"Detected: {test_engine.get_text('../poster_downloads/3.8_52888.jpg')}")