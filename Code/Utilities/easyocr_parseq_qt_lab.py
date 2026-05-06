#!/usr/bin/env python3
"""
easyocr_parseq_qt_lab.py

  pip install PyQt6 easyocr torch torchvision pillow opencv-python numpy
  pip install strhub-sdk
  pip install paddleocr paddlepaddle    # optional, only needed for PaddleOCR detection
"""

from __future__ import annotations

import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from strhub.data.module import SceneTextDataModule
except Exception:
    SceneTextDataModule = None

try:
    from torchvision import transforms as T
except Exception:
    T = None

try:
    cv2.setNumThreads(1)
except Exception:
    pass

THIS_DIR = Path(__file__).parent    # Start file browse in the directory of this script
DEFAULT_IMAGE_PATH = Path("/Users/macintosh/UNM/CS591 Digital Image Processing/Final_Project/Code/Utilities/a_poster_sent_to_ocr.png")

PARSEQ_MODEL_OPTIONS = [
    "parseq",
    "parseq_tiny",
    "parseq_patch16_224",
    "vitstr",
    "trba",
    "crnn",
    "abinet",
]

EASYOCR_DETECT_OPTION_REFERENCE = [
    ("min_size", "0", "0–10000 (int)", "Minimum text box size"),
    ("text_threshold", "0.36", "0.0–1.0 (float)", "Text confidence threshold"),
    ("low_text", "0.18", "0.0–1.0 (float)", "Lower bound for text confidence"),
    ("link_threshold", "0.18", "0.0–1.0 (float)", "Threshold for linking text boxes"),
    ("canvas_size", "1280", "32–20000 (int)", "Resize image for detection"),
    ("mag_ratio", "1.0", "0.1–10.0 (float)", "Magnification ratio for resizing"),
    ("slope_ths", "0.1", "0.0–10.0 (float)", "Slope threshold for merging boxes"),
    ("ycenter_ths", "0.5", "0.0–10.0 (float)", "Y-center threshold for merging boxes"),
    ("height_ths", "0.0", "0.0–10.0 (float)", "Height threshold for merging boxes"),
    ("width_ths", "0.0", "0.0–20.0 (float)", "Width threshold for merging boxes"),
    ("add_margin", "0.12", "0.0–10.0 (float)", "Add margin to detected boxes"),
    ("optimal_num_chars", "None", "int or blank", "Expected number of chars per box"),
    ("reformat", "True", "bool (checkbox)", "Reformat output"),
    ("threshold", "0.0", "0.0–1.0 (float)", "Threshold for binarization/post-processing"),
    ("bbox_min_score", "0.0", "0.0–1.0 (float)", "Minimum bbox score"),
    ("bbox_min_size", "0", "0–10000 (int)", "Minimum bbox size"),
    ("max_candidates", "0", "0–100000 (int)", "Max candidates; 0 = no limit"),
    ("Extra kwargs", "—", "JSON dict", "Any other supported EasyOCR detect() kwargs"),
]

PADDLEOCR_DETECT_OPTION_REFERENCE = [
    ("lang", "en", "str", "PaddleOCR language code"),
    ("device", "auto", "auto/cpu/gpu", "Device choice for PaddleOCR when supported"),
    ("use_angle_cls", "False", "bool", "Legacy text-angle classifier flag"),
    ("det_limit_side_len", "2560", "32–8192 (int)", "Higher resize limit for better small-text detection"),
    ("det_limit_type", "max", "max/min", "Whether det_limit_side_len limits the long or short side"),
    ("det_db_thresh", "0.20", "0.0–1.0 (float)", "Lower DB text-map threshold to keep faint/small text candidates"),
    ("det_db_box_thresh", "0.20", "0.0–1.0 (float)", "Lower DB box confidence threshold to favor recall"),
    ("det_db_unclip_ratio", "2.0", "0.1–10.0 (float)", "Larger expansion ratio to include more of each text line"),
    ("Extra init kwargs", '{"model_name": "PP-OCRv5_server_det"}', "JSON dict", "Use the higher-accuracy PaddleOCR server detector"),
    ("Extra run kwargs", "—", "JSON dict", "Additional detection predict kwargs; recognition stays disabled"),
]


@dataclass
class DetectedCrop:
    index: int
    kind: str
    box: np.ndarray
    detect_crop: np.ndarray | None
    original_crop: np.ndarray | None


def clean_ocr_text(text: str) -> str:
    text = str(text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def maybe_split_parseq_text(text: str) -> str:
    text = clean_ocr_text(text)
    if not text or " " in text or len(text) <= 6:
        return text
    if any(ch in text for ch in ("@", "/", "\\", "_", ".", ":")):
        return text
    if not re.search(r"[A-Za-z]", text):
        return text
    try:
        import wordninja
    except Exception:
        return text
    try:
        split = wordninja.split(text)
    except Exception:
        return text
    return " ".join(split) if len(split) > 1 else text


def preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    if crop_bgr.ndim == 2:
        gray = crop_bgr.copy()
    elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 4:
        gray = cv2.cvtColor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    long_side = max(gray.shape[:2])
    if long_side < 900:
        scale = min(2.4, 900.0 / max(1, long_side))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif long_side > 1800:
        scale = 1800.0 / long_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray, 1.45, blur, -0.45, 0)



def affine_to_homography(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape == (2, 3):
        return np.vstack([m, np.array([0, 0, 1], dtype=np.float32)])
    if m.shape == (3, 3):
        return m
    raise ValueError(f"Expected 2x3 or 3x3 affine matrix, got {m.shape}")


def compose_affine(next_matrix: np.ndarray, current_matrix: np.ndarray) -> np.ndarray:
    composed = affine_to_homography(next_matrix) @ affine_to_homography(current_matrix)
    return composed[:2].astype(np.float32)


def transform_points_affine(points: Any, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.hstack([pts, ones])
    mapped = hom @ np.asarray(matrix, dtype=np.float32).T
    return mapped.reshape(np.asarray(points, dtype=np.float32).shape)


def resize_with_matrix(image: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return image.copy(), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    matrix = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
    return resized, matrix


def rotate_bound_with_matrix(image: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    angle_degrees = float(angle_degrees)
    if abs(angle_degrees) < 1e-6:
        return image.copy(), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0).astype(np.float32)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round((h * sin) + (w * cos)))
    new_h = int(round((h * cos) + (w * sin)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, matrix


def estimate_text_angle_degrees(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    long_side = max(gray.shape[:2])
    if long_side > 1200:
        scale = 1200.0 / long_side
        gray_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray

    gray_small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_small)
    edges = cv2.Canny(gray_small, 50, 150, apertureSize=3)
    min_len = max(20, int(0.08 * min(gray_small.shape[:2])))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(20, int(0.02 * min(gray_small.shape[:2]))),
        minLineLength=min_len,
        maxLineGap=max(6, int(0.02 * min(gray_small.shape[:2]))),
    )
    angles: list[float] = []
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = line.astype(float)
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            while angle <= -90:
                angle += 180
            while angle > 90:
                angle -= 180
            if -35 <= angle <= 35:
                angles.append(angle)

    if angles:
        return float(np.median(np.array(angles, dtype=np.float32)))

    # Fallback: estimate orientation from connected foreground pixels.
    try:
        blur = cv2.GaussianBlur(gray_small, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = 255 - th
        coords = np.column_stack(np.where(th > 0))
        if coords.shape[0] < 25:
            return 0.0
        rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
        angle = float(rect[-1])
        if angle < -45:
            angle += 90
        if angle > 45:
            angle -= 90
        if -35 <= angle <= 35:
            return angle
    except Exception:
        pass

    return 0.0


def preprocess_for_ocr_with_matrix(crop_bgr: np.ndarray, allow_internal_resize: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    steps: list[str] = []
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, matrix, steps

    if crop_bgr.ndim == 2:
        gray = crop_bgr.copy()
    elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 4:
        gray = cv2.cvtColor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    steps.append("grayscale")

    if allow_internal_resize:
        long_side = max(gray.shape[:2])
        if long_side < 900:
            scale = min(2.4, 900.0 / max(1, long_side))
            gray, sm = resize_with_matrix(gray, scale)
            matrix = compose_affine(sm, matrix)
            steps.append(f"auto resize {scale:.2f}x")
        elif long_side > 1800:
            scale = 1800.0 / long_side
            gray, sm = resize_with_matrix(gray, scale)
            matrix = compose_affine(sm, matrix)
            steps.append(f"auto resize {scale:.2f}x")

    gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(gray, 1.45, blur, -0.45, 0)
    steps.append("CLAHE + sharpen")
    return out, matrix, steps


def parse_scale_choice(text: str) -> float:
    raw = str(text or "1.0x").strip().lower().replace("×", "x").replace("x", "")
    try:
        return max(1.0, float(raw))
    except Exception:
        return 1.0


def build_detection_preprocess_image(
    frame: np.ndarray,
    use_ocr_preprocess: bool,
    upsample_enabled: bool,
    upsample_scale: float,
    horizontalize_enabled: bool,
    horizontalize_auto: bool,
    manual_angle: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if frame is None or frame.size == 0:
        raise ValueError("No source image loaded.")

    image = frame.copy()
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    steps: list[str] = []

    if horizontalize_enabled:
        if horizontalize_auto:
            estimated = estimate_text_angle_degrees(image)
            rotate_angle = -estimated
            steps.append(f"horizontalize auto estimated {estimated:+.2f}° / rotated {rotate_angle:+.2f}°")
        else:
            rotate_angle = float(manual_angle)
            steps.append(f"horizontalize manual rotate {rotate_angle:+.2f}°")
        image, rm = rotate_bound_with_matrix(image, rotate_angle)
        matrix = compose_affine(rm, matrix)

    if upsample_enabled:
        scale = max(1.0, float(upsample_scale))
        if scale > 1.0:
            image, sm = resize_with_matrix(image, scale)
            matrix = compose_affine(sm, matrix)
            steps.append(f"upsample {scale:.2f}x")

    if use_ocr_preprocess:
        # If the user explicitly upsamples, do not undo it with the old 1800-px cap.
        allow_internal_resize = not (upsample_enabled and float(upsample_scale) > 1.0)
        image, pm, pre_steps = preprocess_for_ocr_with_matrix(image, allow_internal_resize=allow_internal_resize)
        matrix = compose_affine(pm, matrix)
        steps.extend(pre_steps)

    return image, matrix, steps

def ensure_paddleocr_compatible_image(image: np.ndarray) -> np.ndarray:
    """Return a contiguous uint8 3-channel image for PaddleOCR detection.

    PaddleOCR's detector expects H x W x C image input. The lab's standalone
    preprocessing intentionally returns a 2D grayscale image, which can trigger
    errors like: "not enough values to unpack (expected 3, got 2)".
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to PaddleOCR detector.")

    arr = image
    if arr.dtype != np.uint8:
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.copy()
    else:
        raise ValueError(f"Unsupported image shape for PaddleOCR detector: {arr.shape}")

    return np.ascontiguousarray(arr)


def crop_easyocr_text_region(image: np.ndarray, pts: Any, pad: int = 3) -> np.ndarray | None:
    try:
        arr = np.array(pts, dtype=np.float32)
        if arr.shape != (4, 2):
            raise ValueError
        w = max(2, int(round(max(np.linalg.norm(arr[1] - arr[0]), np.linalg.norm(arr[2] - arr[3])))))
        h = max(2, int(round(max(np.linalg.norm(arr[3] - arr[0]), np.linalg.norm(arr[2] - arr[1])))))
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        crop = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(arr, dst),
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:
        try:
            arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
            h_img, w_img = image.shape[:2]
            x1 = max(0, int(np.floor(np.min(arr[:, 0]))) - pad)
            y1 = max(0, int(np.floor(np.min(arr[:, 1]))) - pad)
            x2 = min(w_img, int(np.ceil(np.max(arr[:, 0]))) + pad)
            y2 = min(h_img, int(np.ceil(np.max(arr[:, 1]))) + pad)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = image[y1:y2, x1:x2]
        except Exception:
            return None

    if crop is None or crop.size == 0:
        return None
    if pad > 0:
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    if crop.shape[0] < 32:
        scale = min(4.0, 32.0 / max(1, crop.shape[0]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def scale_box_points(box: Any, from_shape, to_shape) -> np.ndarray:
    pts = np.array(box, dtype=np.float32)
    from_h, from_w = from_shape[:2]
    to_h, to_w = to_shape[:2]
    pts[:, 0] *= to_w / max(1, from_w)
    pts[:, 1] *= to_h / max(1, from_h)
    return pts


def horizontal_box_to_points(box: Any) -> np.ndarray:
    arr = np.array(box, dtype=np.float32).reshape(-1)
    x_min, x_max, y_min, y_max = arr[:4]
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )


def normalize_detect_output(out: Any) -> tuple[list[Any], list[Any]]:
    if not isinstance(out, tuple) or len(out) < 2:
        return [], []
    horizontal, free = out[0], out[1]
    if isinstance(horizontal, list) and len(horizontal) == 1 and isinstance(horizontal[0], list):
        horizontal = horizontal[0]
    if isinstance(free, list) and len(free) == 1 and isinstance(free[0], list):
        free = free[0]
    return list(horizontal or []), list(free or [])


def as_quad_points(obj: Any) -> np.ndarray | None:
    try:
        arr = np.array(obj, dtype=np.float32)
    except Exception:
        return None
    if arr.shape == (4, 2) and np.isfinite(arr).all():
        return arr
    if arr.shape == (8,) and np.isfinite(arr).all():
        return arr.reshape(4, 2)
    return None


def normalize_paddleocr_detection_output(out: Any) -> list[np.ndarray]:
    """
    Extract text-detection polygons from PaddleOCR outputs across v2/v3 APIs.

    Supported examples:
      - PaddleOCR 3.x Result objects from ocr.predict(...), whose .json often wraps data under "res"
      - PaddleOCR 3.x dicts containing dt_polys
      - PaddleOCR 2.x ocr(..., det=True, rec=False) nested list output
      - Raw numpy arrays/lists shaped like (N, 4, 2), (4, 2), or (8,)
    """
    if out is None:
        return []

    boxes: list[np.ndarray] = []
    visited: set[int] = set()

    def walk(obj: Any):
        if obj is None:
            return

        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        # PaddleOCR 3.x Result object: result.json is usually a dict like {"res": {...}}
        for attr in ("json", "res", "result"):
            if hasattr(obj, attr):
                try:
                    walk(getattr(obj, attr))
                    return
                except Exception:
                    pass

        if hasattr(obj, "to_dict"):
            try:
                walk(obj.to_dict())
                return
            except Exception:
                pass

        if isinstance(obj, np.ndarray):
            if obj.ndim == 3 and obj.shape[1:] == (4, 2):
                for item in obj:
                    quad = as_quad_points(item)
                    if quad is not None:
                        boxes.append(quad)
                return
            if obj.ndim == 2 and obj.shape[1] == 8:
                for item in obj:
                    quad = as_quad_points(item)
                    if quad is not None:
                        boxes.append(quad)
                return
            quad = as_quad_points(obj)
            if quad is not None:
                boxes.append(quad)
                return

        if isinstance(obj, dict):
            preferred_keys = (
                "dt_polys", "dt_boxes", "det_polys", "text_polys",
                "rec_polys", "polys", "boxes", "points",
            )
            found_preferred = False
            for key in preferred_keys:
                if key in obj:
                    found_preferred = True
                    walk(obj[key])
            if found_preferred:
                return
            # Many PaddleOCR 3.x results are wrapped as {"res": {...}}.
            for value in obj.values():
                walk(value)
            return

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return
            quad = as_quad_points(obj)
            if quad is not None:
                boxes.append(quad)
                return

            # PaddleOCR 2.x detection-only result can be [[box1, box2, ...]].
            # Full OCR result can be [[box, (text, score)], ...]; take the first item.
            if len(obj) == 2:
                first_quad = as_quad_points(obj[0])
                if first_quad is not None:
                    boxes.append(first_quad)
                    return

            for item in obj:
                walk(item)

    walk(out)

    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for box in boxes:
        if box.shape != (4, 2) or not np.isfinite(box).all():
            continue
        key = tuple(np.round(box.reshape(-1), 1).astype(int).tolist())
        if key not in seen:
            seen.add(key)
            unique.append(box.astype(np.float32))
    return unique


def describe_paddle_output(out: Any) -> str:
    try:
        if isinstance(out, list):
            return f"list(len={len(out)}, first={type(out[0]).__name__ if out else 'empty'})"
        if isinstance(out, tuple):
            return f"tuple(len={len(out)}, first={type(out[0]).__name__ if out else 'empty'})"
        if isinstance(out, dict):
            return f"dict(keys={list(out.keys())[:8]})"
        if isinstance(out, np.ndarray):
            return f"ndarray(shape={out.shape}, dtype={out.dtype})"
        if hasattr(out, "json"):
            j = getattr(out, "json")
            if isinstance(j, dict):
                return f"{type(out).__name__}(json_keys={list(j.keys())[:8]})"
            return f"{type(out).__name__}(json={type(j).__name__})"
        return type(out).__name__
    except Exception:
        return type(out).__name__

def cv_to_qpixmap(img: np.ndarray, max_w: int | None = None, max_h: int | None = None) -> QPixmap:
    if img is None or img.size == 0:
        img = np.zeros((32, 128, 3), dtype=np.uint8)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if max_w is not None or max_h is not None:
        pix = pix.scaled(max_w or w, max_h or h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pix


def parse_json_kwargs(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("Extra kwargs JSON must be an object/dict.")
    return obj


def supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class PaddleOCRDetectorRunner:
    def __init__(
        self,
        lang: str,
        device_choice: str,
        use_angle_cls: bool,
        init_kwargs: dict[str, Any],
    ):
        try:
            import paddleocr as paddleocr_pkg
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle"
            ) from exc

        self.ocr = None
        self.text_detector = None
        self.backend = ""
        self.api_mode = ""

        lang = (lang or "en").strip() or "en"
        extra = dict(init_kwargs or {})

        det_limit_side_len = int(extra.pop("det_limit_side_len", extra.pop("text_det_limit_side_len", extra.pop("limit_side_len", 1280))))
        det_limit_type = str(extra.pop("det_limit_type", extra.pop("text_det_limit_type", extra.pop("limit_type", "max"))))
        det_thresh = float(extra.pop("det_db_thresh", extra.pop("text_det_thresh", extra.pop("thresh", 0.3))))
        det_box_thresh = float(extra.pop("det_db_box_thresh", extra.pop("text_det_box_thresh", extra.pop("box_thresh", 0.3))))
        det_unclip_ratio = float(extra.pop("det_db_unclip_ratio", extra.pop("text_det_unclip_ratio", extra.pop("unclip_ratio", 1.5))))

        device_value: str | None = None
        if device_choice == "gpu":
            device_value = "gpu:0"
        elif device_choice == "cpu":
            device_value = "cpu"

        # Preferred PaddleOCR 3.x path: the standalone TextDetection module.
        # This avoids running recognition at all; PARSeq remains the recognizer.
        TextDetection = getattr(paddleocr_pkg, "TextDetection", None)
        if TextDetection is not None:
            td_extra = dict(extra)
            model_name = td_extra.pop("model_name", td_extra.pop("text_detection_model_name", "PP-OCRv5_server_det"))
            td_kwargs: dict[str, Any] = {
                "model_name": model_name,
                "limit_side_len": det_limit_side_len,
                "limit_type": det_limit_type,
                "thresh": det_thresh,
                "box_thresh": det_box_thresh,
                "unclip_ratio": det_unclip_ratio,
            }
            if device_value is not None:
                td_kwargs["device"] = device_value
            td_kwargs.update(td_extra)

            try:
                self.text_detector = TextDetection(**td_kwargs)
                self.init_kwargs = td_kwargs
                self.backend = "TextDetection"
                self.api_mode = "text_detection"
                self.default_predict_kwargs = {
                    "batch_size": 1,
                    "limit_side_len": det_limit_side_len,
                    "limit_type": det_limit_type,
                    "thresh": det_thresh,
                    "box_thresh": det_box_thresh,
                    "unclip_ratio": det_unclip_ratio,
                }
                return
            except Exception as exc:
                self.text_detector = None
                last_text_detection_exc = exc
        else:
            last_text_detection_exc = RuntimeError("paddleocr.TextDetection is unavailable in this PaddleOCR version")

        PaddleOCR = getattr(paddleocr_pkg, "PaddleOCR", None)
        if PaddleOCR is None:
            raise RuntimeError(f"Could not initialize PaddleOCR TextDetection fallback: {last_text_detection_exc}")

        # Fallback: PaddleOCR pipeline API. This is more compatible, but may run extra OCR stages.
        # Keep it because older PaddleOCR wheels do not expose TextDetection.
        pipeline_extra = dict(extra)
        modern_base: dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": bool(use_angle_cls),
            "text_det_limit_side_len": det_limit_side_len,
            "text_det_limit_type": det_limit_type,
            "text_det_thresh": det_thresh,
            "text_det_box_thresh": det_box_thresh,
            "text_det_unclip_ratio": det_unclip_ratio,
        }
        if device_value is not None:
            modern_base["device"] = device_value
        modern_base.update(pipeline_extra)

        legacy_base: dict[str, Any] = {
            "lang": lang,
            "use_angle_cls": bool(use_angle_cls),
            "det_limit_side_len": det_limit_side_len,
            "det_limit_type": det_limit_type,
            "det_db_thresh": det_thresh,
            "det_db_box_thresh": det_box_thresh,
            "det_db_unclip_ratio": det_unclip_ratio,
        }
        if device_choice == "gpu":
            legacy_base["use_gpu"] = True
        elif device_choice == "cpu":
            legacy_base["use_gpu"] = False
        legacy_base.update(pipeline_extra)

        candidates: list[tuple[str, dict[str, Any]]] = [
            ("modern_pipeline", dict(modern_base)),
            ("legacy_pipeline", dict(legacy_base)),
            ("minimal_pipeline", {"lang": lang}),
        ]

        last_exc: Exception | None = last_text_detection_exc
        for api_mode, kwargs in candidates:
            try:
                self.ocr = PaddleOCR(**kwargs)
                self.init_kwargs = kwargs
                self.backend = "PaddleOCR"
                self.api_mode = api_mode
                self.default_predict_kwargs = {
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": bool(use_angle_cls),
                    "text_det_limit_side_len": det_limit_side_len,
                    "text_det_limit_type": det_limit_type,
                    "text_det_thresh": det_thresh,
                    "text_det_box_thresh": det_box_thresh,
                    "text_det_unclip_ratio": det_unclip_ratio,
                }
                return
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Could not initialize PaddleOCR detector. TextDetection error: {last_text_detection_exc}; PaddleOCR error: {last_exc}")

    def detect(self, image: np.ndarray, run_kwargs: dict[str, Any]) -> tuple[list[np.ndarray], str]:
        clean_kwargs = self._clean_run_kwargs(run_kwargs)
        errors: list[str] = []

        if self.text_detector is not None:
            predict_kwargs = dict(self.default_predict_kwargs)
            predict_kwargs.update(self._to_text_detection_kwargs(clean_kwargs))
            try:
                try:
                    out = self.text_detector.predict(input=image, **predict_kwargs)
                except TypeError:
                    out = self.text_detector.predict(image, **predict_kwargs)
                boxes = normalize_paddleocr_detection_output(out)
                return boxes, f"TextDetection.predict output: {describe_paddle_output(out)}"
            except Exception as exc:
                errors.append(f"TextDetection.predict() failed: {exc}")

        # PaddleOCR 3.x pipeline path. predict(input=np.ndarray, ...) returns Result objects
        # containing dt_polys. This avoids the older ocr(..., det=True, rec=False) path that can
        # throw tuple/list index errors with newer PaddleOCR releases.
        if self.ocr is not None and hasattr(self.ocr, "predict"):
            predict_kwargs = dict(self.default_predict_kwargs)
            predict_kwargs.update(self._to_pipeline_predict_kwargs(clean_kwargs))
            try:
                try:
                    out = self.ocr.predict(input=image, **predict_kwargs)
                except TypeError:
                    out = self.ocr.predict(image, **predict_kwargs)
                boxes = normalize_paddleocr_detection_output(out)
                return boxes, f"PaddleOCR.predict output: {describe_paddle_output(out)}"
            except Exception as exc:
                errors.append(f"PaddleOCR.predict() failed: {exc}")

        # Fallback for PaddleOCR 2.x.
        if self.ocr is not None and hasattr(self.ocr, "ocr"):
            old_kwargs = {
                k: v for k, v in clean_kwargs.items()
                if k not in (
                    "use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation",
                    "text_det_limit_side_len", "text_det_limit_type", "text_det_thresh",
                    "text_det_box_thresh", "text_det_unclip_ratio", "text_rec_score_thresh",
                    "limit_side_len", "limit_type", "thresh", "box_thresh", "unclip_ratio",
                )
            }
            try:
                out = self.ocr.ocr(image, det=True, rec=False, cls=False, **old_kwargs)
                boxes = normalize_paddleocr_detection_output(out)
                return boxes, f"ocr(det=True, rec=False) output: {describe_paddle_output(out)}"
            except Exception as exc:
                errors.append(f"ocr(det=True, rec=False) failed: {exc}")

            # Some old builds prefer image paths rather than ndarray input. Use a temporary PNG.
            try:
                import tempfile
                tmp_path = None
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmp_path = f.name
                cv2.imwrite(tmp_path, image)
                try:
                    out = self.ocr.ocr(tmp_path, det=True, rec=False, cls=False, **old_kwargs)
                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                boxes = normalize_paddleocr_detection_output(out)
                return boxes, f"ocr(temp_path, det=True, rec=False) output: {describe_paddle_output(out)}"
            except Exception as exc:
                errors.append(f"ocr(temp_path, det=True, rec=False) failed: {exc}")

        raise RuntimeError("PaddleOCR detection failed:\n" + "\n".join(errors))

    def _clean_run_kwargs(self, run_kwargs: dict[str, Any]) -> dict[str, Any]:
        clean = dict(run_kwargs or {})
        # Recognition is intentionally not run here; PARSeq handles recognition.
        for key in ("det", "rec", "cls"):
            clean.pop(key, None)
        return clean

    def _to_text_detection_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        clean = dict(kwargs)
        translations = {
            "det_limit_side_len": "limit_side_len",
            "text_det_limit_side_len": "limit_side_len",
            "det_limit_type": "limit_type",
            "text_det_limit_type": "limit_type",
            "det_db_thresh": "thresh",
            "text_det_thresh": "thresh",
            "det_db_box_thresh": "box_thresh",
            "text_det_box_thresh": "box_thresh",
            "det_db_unclip_ratio": "unclip_ratio",
            "text_det_unclip_ratio": "unclip_ratio",
        }
        for old, new in translations.items():
            if old in clean and new not in clean:
                clean[new] = clean.pop(old)
        for key in ("use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation", "text_rec_score_thresh"):
            clean.pop(key, None)
        return clean

    def _to_pipeline_predict_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        clean = dict(kwargs)
        translations = {
            "det_limit_side_len": "text_det_limit_side_len",
            "limit_side_len": "text_det_limit_side_len",
            "det_limit_type": "text_det_limit_type",
            "limit_type": "text_det_limit_type",
            "det_db_thresh": "text_det_thresh",
            "thresh": "text_det_thresh",
            "det_db_box_thresh": "text_det_box_thresh",
            "box_thresh": "text_det_box_thresh",
            "det_db_unclip_ratio": "text_det_unclip_ratio",
            "unclip_ratio": "text_det_unclip_ratio",
        }
        for old, new in translations.items():
            if old in clean and new not in clean:
                clean[new] = clean.pop(old)
        return clean

class PARSeqRunner:
    def __init__(
        self,
        repo: str,
        model_name: str,
        device_choice: str,
        force_reload: bool,
        trust_repo: bool,
        hub_extra_kwargs: dict[str, Any],
        use_scene_transform: bool,
        custom_img_h: int,
        custom_img_w: int,
        normalize_mean: float,
        normalize_std: float,
    ):
        self.repo = repo
        self.model_name = model_name
        self.device = self.choose_device(device_choice)
        hub_kwargs = dict(hub_extra_kwargs)
        hub_kwargs.setdefault("pretrained", True)
        if "force_reload" in inspect.signature(torch.hub.load).parameters:
            hub_kwargs.setdefault("force_reload", bool(force_reload))
        if "trust_repo" in inspect.signature(torch.hub.load).parameters:
            hub_kwargs.setdefault("trust_repo", bool(trust_repo))
        self.model = torch.hub.load(repo, model_name, **hub_kwargs).to(self.device).eval()
        self.transform = self.build_transform(
            self.model,
            use_scene_transform=use_scene_transform,
            custom_img_h=custom_img_h,
            custom_img_w=custom_img_w,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    def choose_device(self, choice: str) -> str:
        if choice == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False.")
            return "cuda"
        if choice == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS was selected, but torch.backends.mps.is_available() is False.")
            return "mps"
        if choice == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def build_transform(
        self,
        model,
        use_scene_transform: bool,
        custom_img_h: int,
        custom_img_w: int,
        normalize_mean: float,
        normalize_std: float,
    ):
        if use_scene_transform and SceneTextDataModule is not None:
            return SceneTextDataModule.get_transform(model.hparams.img_size)
        if T is None:
            raise RuntimeError("torchvision transforms unavailable and PARSeq transform import failed.")
        if custom_img_h > 0 and custom_img_w > 0:
            img_size = (custom_img_h, custom_img_w)
        else:
            hparams = getattr(model, "hparams", None)
            if isinstance(hparams, dict):
                img_size = tuple(hparams.get("img_size", (32, 128)))
            else:
                img_size = tuple(getattr(hparams, "img_size", (32, 128)))
        return T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(normalize_mean, normalize_std),
            ]
        )

    @torch.no_grad()
    def __call__(self, crop: np.ndarray) -> tuple[str, float]:
        if crop is None or crop.size == 0:
            return "", 0.0
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            rgb = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        pred = self.model(tensor).softmax(-1)
        labels, conf = self.model.tokenizer.decode(pred)
        text = labels[0] if labels else ""
        return text, self.confidence(conf, text)

    def confidence(self, raw_conf: Any, text: str) -> float:
        try:
            conf0 = raw_conf[0]
            if hasattr(conf0, "detach"):
                conf0 = conf0.detach().cpu()
            arr = np.array(conf0, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return 0.0
            n = min(arr.size, max(1, len(text) + 1))
            return float(np.prod(np.clip(arr[:n], 0.0, 1.0)) ** (1.0 / n))
        except Exception:
            return 0.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Detection + PARSeq Lab")
        self.resize(1500, 920)
        self.frame: np.ndarray | None = None
        self.detect_img: np.ndarray | None = None
        self.frame_to_detect_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.detect_to_frame_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.reader: easyocr.Reader | None = None
        self.paddleocr_detector: PaddleOCRDetectorRunner | None = None
        self.parseq: PARSeqRunner | None = None
        self.crops: list[DetectedCrop] = []
        self.undo_stack: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []
        self.restoring_options = False
        self.option_widgets: dict[str, Any] = {}
        self.build_ui()
        self.register_option_widgets()
        self.apply_detect_tooltips()
        self.snapshot_options(clear_redo=True)
        self.connect_option_undo_signals()
        self.connect_auto_reload_signals()
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo_options)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo_options)
        self.load_default_image_if_available()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([520, 980])

        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        self.browse_btn = QPushButton("Browse Image")
        self.browse_btn.clicked.connect(self.browse_image)
        self.reload_reader_btn = QPushButton("Reload Detector")
        self.reload_reader_btn.clicked.connect(self.reload_detector)
        self.detect_btn = QPushButton("Run Text Detection")
        self.detect_btn.clicked.connect(self.run_detect)
        self.reload_parseq_btn = QPushButton("Reload PARSeq")
        self.reload_parseq_btn.clicked.connect(self.reload_parseq)
        self.parseq_btn = QPushButton("Run PARSeq")
        self.parseq_btn.clicked.connect(self.run_parseq)
        self.undo_btn = QPushButton("Undo Options")
        self.undo_btn.clicked.connect(self.undo_options)
        self.redo_btn = QPushButton("Redo Options")
        self.redo_btn.clicked.connect(self.redo_options)

        actions_layout.addWidget(self.browse_btn, 0, 0)
        actions_layout.addWidget(self.reload_reader_btn, 0, 1)
        actions_layout.addWidget(self.detect_btn, 0, 2)
        actions_layout.addWidget(self.reload_parseq_btn, 1, 0)
        actions_layout.addWidget(self.parseq_btn, 1, 1)
        actions_layout.addWidget(self.undo_btn, 1, 2)
        actions_layout.addWidget(self.redo_btn, 2, 2)
        controls_layout.addWidget(actions_group)

        self.tabs = QTabWidget()
        controls_layout.addWidget(self.tabs, 1)

        self.build_image_reader_tab()
        self.build_easyocr_detect_tab()
        self.build_paddleocr_detect_tab()
        self.build_detect_reference_tab()
        self.build_crop_tab()
        self.build_parseq_tab()

        self.preview_label = QLabel("Load an image to begin.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(260)
        self.preview_label.setStyleSheet("QLabel { border: 1px solid #999; background: #222; color: #ddd; }")
        right_layout.addWidget(self.preview_label)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setFixedHeight(125)
        right_layout.addWidget(self.status)

        crops_group = QGroupBox("Detected Text Crops")
        crops_layout = QVBoxLayout(crops_group)
        self.crop_scroll = QScrollArea()
        self.crop_scroll.setWidgetResizable(True)
        self.crop_container = QWidget()
        self.crop_grid = QGridLayout(self.crop_container)
        self.crop_scroll.setWidget(self.crop_container)
        crops_layout.addWidget(self.crop_scroll)
        right_layout.addWidget(crops_group, 2)

        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["#", "Source", "PARSeq text", "Confidence", "Time"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.result_table, 1)

    def build_image_reader_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        image_group = QGroupBox("Image")
        image_form = QFormLayout(image_group)
        self.image_path = QLineEdit(str(DEFAULT_IMAGE_PATH))
        image_form.addRow("Image path", self.image_path)
        self.preprocess_check = QCheckBox("Use standalone preprocessing before text detection")
        self.preprocess_check.setChecked(True)
        image_form.addRow(self.preprocess_check)

        self.upsample_check = QCheckBox("Upsample before detection")
        self.upsample_check.setChecked(False)
        self.upsample_scale = QComboBox()
        self.upsample_scale.addItems(["1.25x", "1.5x", "2.0x", "3.0x", "4.0x"])
        self.upsample_scale.setCurrentText("2.0x")
        image_form.addRow(self.upsample_check)
        image_form.addRow("Upsample size", self.upsample_scale)

        self.horizontalize_check = QCheckBox("Horizontalize / deskew text before detection")
        self.horizontalize_check.setChecked(False)
        self.horizontalize_auto_check = QCheckBox("Auto-estimate horizontalize angle")
        self.horizontalize_auto_check.setChecked(True)
        self.horizontalize_angle = self.spin_float(-45.0, 45.0, 0.0, 0.25)
        image_form.addRow(self.horizontalize_check)
        image_form.addRow(self.horizontalize_auto_check)
        image_form.addRow("Manual rotate angle", self.horizontalize_angle)

        preview_note = QLabel("When preprocessing/upsampling/horizontalizing is enabled, the top preview shows the processed detection image.")
        preview_note.setWordWrap(True)
        image_form.addRow(preview_note)
        layout.addWidget(image_group)

        detector_group = QGroupBox("Text Detection Engine")
        detector_form = QFormLayout(detector_group)
        self.detector_engine = QComboBox()
        self.detector_engine.addItems(["EasyOCR / CRAFT", "PaddleOCR detector"])
        self.detector_engine.setCurrentText("PaddleOCR detector")
        detector_form.addRow("Detector", self.detector_engine)
        layout.addWidget(detector_group)

        reader_group = QGroupBox("EasyOCR / CRAFT detector")
        reader_form = QFormLayout(reader_group)
        self.langs_edit = QLineEdit("en")
        self.gpu_check = QCheckBox("Use GPU for EasyOCR")
        self.gpu_check.setChecked(False)
        reader_form.addRow("Languages comma-separated", self.langs_edit)
        reader_form.addRow(self.gpu_check)
        layout.addWidget(reader_group)
        layout.addStretch(1)

        self.tabs.addTab(tab, "Image / Detector")

    def build_easyocr_detect_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.min_size = self.spin_int(0, 10000, 0)
        self.text_threshold = self.spin_float(0.0, 1.0, 0.36, 0.01)
        self.low_text = self.spin_float(0.0, 1.0, 0.18, 0.01)
        self.link_threshold = self.spin_float(0.0, 1.0, 0.18, 0.01)
        self.canvas_size = self.spin_int(32, 20000, 1280)
        self.mag_ratio = self.spin_float(0.1, 10.0, 1.0, 0.1)
        self.slope_ths = self.spin_float(0.0, 10.0, 0.1, 0.05)
        self.ycenter_ths = self.spin_float(0.0, 10.0, 0.5, 0.05)
        self.height_ths = self.spin_float(0.0, 10.0, 0.0, 0.05)
        self.width_ths = self.spin_float(0.0, 20.0, 0.0, 0.05)
        self.add_margin = self.spin_float(0.0, 10.0, 0.12, 0.01)
        self.optimal_num_chars = QLineEdit("")
        self.reformat_check = QCheckBox("reformat")
        self.reformat_check.setChecked(True)
        self.threshold = self.spin_float(0.0, 1.0, 0.0, 0.01)
        self.bbox_min_score = self.spin_float(0.0, 1.0, 0.0, 0.01)
        self.bbox_min_size = self.spin_int(0, 10000, 0)
        self.max_candidates = self.spin_int(0, 100000, 0)
        self.detect_extra_json = QTextEdit()
        self.detect_extra_json.setPlaceholderText('Optional JSON, for example: {"poly": false}')
        self.detect_extra_json.setFixedHeight(90)

        form.addRow("min_size", self.min_size)
        form.addRow("text_threshold", self.text_threshold)
        form.addRow("low_text", self.low_text)
        form.addRow("link_threshold", self.link_threshold)
        form.addRow("canvas_size", self.canvas_size)
        form.addRow("mag_ratio", self.mag_ratio)
        form.addRow("slope_ths", self.slope_ths)
        form.addRow("ycenter_ths", self.ycenter_ths)
        form.addRow("height_ths", self.height_ths)
        form.addRow("width_ths", self.width_ths)
        form.addRow("add_margin", self.add_margin)
        form.addRow("optimal_num_chars blank=None", self.optimal_num_chars)
        form.addRow(self.reformat_check)
        form.addRow("threshold", self.threshold)
        form.addRow("bbox_min_score", self.bbox_min_score)
        form.addRow("bbox_min_size", self.bbox_min_size)
        form.addRow("max_candidates", self.max_candidates)
        form.addRow("Extra detect kwargs JSON", self.detect_extra_json)

        self.tabs.addTab(tab, "EasyOCR detect()")

    def build_paddleocr_detect_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.paddle_lang = QLineEdit("en")
        self.paddle_device = QComboBox()
        self.paddle_device.addItems(["auto", "cpu", "gpu"])
        self.paddle_use_angle_cls = QCheckBox("use_angle_cls")
        self.paddle_use_angle_cls.setChecked(False)
        self.paddle_det_limit_side_len = self.spin_int(32, 8192, 2560)
        self.paddle_det_limit_type = QComboBox()
        self.paddle_det_limit_type.addItems(["max", "min"])
        self.paddle_det_db_thresh = self.spin_float(0.0, 1.0, 0.20, 0.01)
        self.paddle_det_db_box_thresh = self.spin_float(0.0, 1.0, 0.20, 0.01)
        self.paddle_det_db_unclip_ratio = self.spin_float(0.1, 10.0, 2.0, 0.1)
        self.paddle_init_extra_json = QTextEdit()
        self.paddle_init_extra_json.setPlainText(json.dumps({"model_name": "PP-OCRv5_server_det"}, indent=2))
        self.paddle_init_extra_json.setPlaceholderText('Optional PaddleOCR/TextDetection constructor JSON, for example: {"model_name": "PP-OCRv5_server_det"}')
        self.paddle_init_extra_json.setFixedHeight(95)
        self.paddle_run_extra_json = QTextEdit()
        self.paddle_run_extra_json.setPlaceholderText('Optional detection predict JSON. Usually leave blank because PARSeq handles recognition.')
        self.paddle_run_extra_json.setFixedHeight(70)

        form.addRow("lang", self.paddle_lang)
        form.addRow("device", self.paddle_device)
        form.addRow(self.paddle_use_angle_cls)
        form.addRow("det_limit_side_len", self.paddle_det_limit_side_len)
        form.addRow("det_limit_type", self.paddle_det_limit_type)
        form.addRow("det_db_thresh", self.paddle_det_db_thresh)
        form.addRow("det_db_box_thresh", self.paddle_det_db_box_thresh)
        form.addRow("det_db_unclip_ratio", self.paddle_det_db_unclip_ratio)
        form.addRow("Extra PaddleOCR init kwargs JSON", self.paddle_init_extra_json)
        form.addRow("Extra PaddleOCR run kwargs JSON", self.paddle_run_extra_json)

        self.tabs.addTab(tab, "PaddleOCR detect")

    def build_detect_reference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        easy_label = QLabel("EasyOCR / CRAFT detect() options")
        easy_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(easy_label)
        self.detect_ref_table = self.make_detect_reference_table()
        layout.addWidget(self.detect_ref_table)

        paddle_label = QLabel("PaddleOCR detector options")
        paddle_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(paddle_label)
        self.paddle_ref_table = self.make_paddle_detect_reference_table()
        layout.addWidget(self.paddle_ref_table)

        note = QLabel(
            "Only the selected detector is used to find text regions. PARSeq still performs recognition on the detected crops."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.tabs.addTab(tab, "Detect Reference")

    def build_crop_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        self.crop_pad = self.spin_int(0, 100, 3)
        self.thumb_width = self.spin_int(80, 600, 220)
        self.thumb_height = self.spin_int(40, 400, 120)
        form.addRow("Crop pad", self.crop_pad)
        form.addRow("Thumbnail max width", self.thumb_width)
        form.addRow("Thumbnail max height", self.thumb_height)
        self.tabs.addTab(tab, "Crops")

    def build_parseq_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.parseq_repo = QLineEdit("baudm/parseq")
        self.parseq_model = QComboBox()
        self.parseq_model.addItems(PARSEQ_MODEL_OPTIONS)
        self.parseq_model.setCurrentText("parseq")
        self.parseq_model.setEditable(True)
        self.parseq_device = QComboBox()
        self.parseq_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.parseq_source = QComboBox()
        self.parseq_source.addItems(["Detector crop", "Original-image scaled crop"])
        self.parseq_force_reload = QCheckBox("force_reload")
        self.parseq_trust_repo = QCheckBox("trust_repo")
        self.parseq_trust_repo.setChecked(True)
        self.use_scene_transform = QCheckBox("Use SceneTextDataModule transform when available")
        self.use_scene_transform.setChecked(True)
        self.custom_img_h = self.spin_int(0, 2048, 0)
        self.custom_img_w = self.spin_int(0, 4096, 0)
        self.norm_mean = self.spin_float(-10.0, 10.0, 0.5, 0.05)
        self.norm_std = self.spin_float(0.01, 10.0, 0.5, 0.05)
        self.split_parseq = QCheckBox("Apply wordninja split to PARSeq text")
        self.split_parseq.setChecked(True)
        self.parseq_extra_json = QTextEdit()
        self.parseq_extra_json.setPlaceholderText('Optional torch.hub.load kwargs JSON, for example: {"pretrained": true}')
        self.parseq_extra_json.setFixedHeight(90)

        form.addRow("torch.hub repo", self.parseq_repo)
        form.addRow("model", self.parseq_model)
        form.addRow("device", self.parseq_device)
        form.addRow("PARSeq crop source", self.parseq_source)
        form.addRow(self.parseq_force_reload)
        form.addRow(self.parseq_trust_repo)
        form.addRow(self.use_scene_transform)
        form.addRow("custom img height, 0=model default", self.custom_img_h)
        form.addRow("custom img width, 0=model default", self.custom_img_w)
        form.addRow("normalize mean", self.norm_mean)
        form.addRow("normalize std", self.norm_std)
        form.addRow(self.split_parseq)
        form.addRow("torch.hub extra kwargs JSON", self.parseq_extra_json)

        self.tabs.addTab(tab, "PARSeq")

    def make_detect_reference_table(self) -> QTableWidget:
        table = QTableWidget(len(EASYOCR_DETECT_OPTION_REFERENCE), 4)
        table.setHorizontalHeaderLabels(["Option", "Default", "Range/Type", "Description"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for row, values in enumerate(EASYOCR_DETECT_OPTION_REFERENCE):
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        table.resizeRowsToContents()
        return table

    def make_paddle_detect_reference_table(self) -> QTableWidget:
        table = QTableWidget(len(PADDLEOCR_DETECT_OPTION_REFERENCE), 4)
        table.setHorizontalHeaderLabels(["Option", "Default", "Range/Type", "Description"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for row, values in enumerate(PADDLEOCR_DETECT_OPTION_REFERENCE):
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        table.resizeRowsToContents()
        return table

    def apply_detect_tooltips(self):
        widgets = {
            "min_size": self.min_size,
            "text_threshold": self.text_threshold,
            "low_text": self.low_text,
            "link_threshold": self.link_threshold,
            "canvas_size": self.canvas_size,
            "mag_ratio": self.mag_ratio,
            "slope_ths": self.slope_ths,
            "ycenter_ths": self.ycenter_ths,
            "height_ths": self.height_ths,
            "width_ths": self.width_ths,
            "add_margin": self.add_margin,
            "optimal_num_chars": self.optimal_num_chars,
            "reformat": self.reformat_check,
            "threshold": self.threshold,
            "bbox_min_score": self.bbox_min_score,
            "bbox_min_size": self.bbox_min_size,
            "max_candidates": self.max_candidates,
        }
        for option, default, range_type, description in EASYOCR_DETECT_OPTION_REFERENCE:
            widget = widgets.get(option)
            if widget is not None:
                widget.setToolTip(f"{description}\nDefault: {default}\nRange/Type: {range_type}")

        paddle_widgets = {
            "lang": self.paddle_lang,
            "device": self.paddle_device,
            "use_angle_cls": self.paddle_use_angle_cls,
            "det_limit_side_len": self.paddle_det_limit_side_len,
            "det_limit_type": self.paddle_det_limit_type,
            "det_db_thresh": self.paddle_det_db_thresh,
            "det_db_box_thresh": self.paddle_det_db_box_thresh,
            "det_db_unclip_ratio": self.paddle_det_db_unclip_ratio,
        }
        for option, default, range_type, description in PADDLEOCR_DETECT_OPTION_REFERENCE:
            widget = paddle_widgets.get(option)
            if widget is not None:
                widget.setToolTip(f"{description}\nDefault: {default}\nRange/Type: {range_type}")

    def spin_int(self, lo: int, hi: int, val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def spin_float(self, lo: float, hi: float, val: float, step: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(4)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def register_option_widgets(self):
        self.option_widgets = {
            "image_path": self.image_path,
            "preprocess": self.preprocess_check,
            "upsample_enabled": self.upsample_check,
            "upsample_scale": self.upsample_scale,
            "horizontalize_enabled": self.horizontalize_check,
            "horizontalize_auto": self.horizontalize_auto_check,
            "horizontalize_angle": self.horizontalize_angle,
            "detector_engine": self.detector_engine,
            "languages": self.langs_edit,
            "easyocr_gpu": self.gpu_check,
            "min_size": self.min_size,
            "text_threshold": self.text_threshold,
            "low_text": self.low_text,
            "link_threshold": self.link_threshold,
            "canvas_size": self.canvas_size,
            "mag_ratio": self.mag_ratio,
            "slope_ths": self.slope_ths,
            "ycenter_ths": self.ycenter_ths,
            "height_ths": self.height_ths,
            "width_ths": self.width_ths,
            "add_margin": self.add_margin,
            "optimal_num_chars": self.optimal_num_chars,
            "reformat": self.reformat_check,
            "threshold": self.threshold,
            "bbox_min_score": self.bbox_min_score,
            "bbox_min_size": self.bbox_min_size,
            "max_candidates": self.max_candidates,
            "detect_extra_json": self.detect_extra_json,
            "paddle_lang": self.paddle_lang,
            "paddle_device": self.paddle_device,
            "paddle_use_angle_cls": self.paddle_use_angle_cls,
            "paddle_det_limit_side_len": self.paddle_det_limit_side_len,
            "paddle_det_limit_type": self.paddle_det_limit_type,
            "paddle_det_db_thresh": self.paddle_det_db_thresh,
            "paddle_det_db_box_thresh": self.paddle_det_db_box_thresh,
            "paddle_det_db_unclip_ratio": self.paddle_det_db_unclip_ratio,
            "paddle_init_extra_json": self.paddle_init_extra_json,
            "paddle_run_extra_json": self.paddle_run_extra_json,
            "crop_pad": self.crop_pad,
            "thumb_width": self.thumb_width,
            "thumb_height": self.thumb_height,
            "parseq_repo": self.parseq_repo,
            "parseq_model": self.parseq_model,
            "parseq_device": self.parseq_device,
            "parseq_source": self.parseq_source,
            "parseq_force_reload": self.parseq_force_reload,
            "parseq_trust_repo": self.parseq_trust_repo,
            "use_scene_transform": self.use_scene_transform,
            "custom_img_h": self.custom_img_h,
            "custom_img_w": self.custom_img_w,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "split_parseq": self.split_parseq,
            "parseq_extra_json": self.parseq_extra_json,
        }

    def widget_value(self, widget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def set_widget_value(self, widget, value):
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
            elif widget.isEditable():
                widget.setEditText(str(value))
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(str(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def current_options_snapshot(self) -> dict[str, Any]:
        return {name: self.widget_value(widget) for name, widget in self.option_widgets.items()}

    def snapshot_options(self, clear_redo: bool):
        if self.restoring_options:
            return
        snap = self.current_options_snapshot()
        if self.undo_stack and self.undo_stack[-1] == snap:
            return
        self.undo_stack.append(snap)
        if len(self.undo_stack) > 200:
            self.undo_stack = self.undo_stack[-200:]
        if clear_redo:
            self.redo_stack.clear()
        self.update_undo_redo_buttons()

    def connect_option_undo_signals(self):
        for widget in self.option_widgets.values():
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
                if widget.isEditable():
                    widget.currentTextChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
            elif isinstance(widget, QTextEdit):
                widget.textChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))

    def connect_auto_reload_signals(self):
        self.detector_engine.currentTextChanged.connect(self.on_detector_engine_changed)
        self.parseq_model.currentTextChanged.connect(self.on_parseq_model_changed)

    def current_parseq_model_name(self) -> str:
        if isinstance(self.parseq_model, QComboBox):
            return self.parseq_model.currentText().strip()
        return self.parseq_model.text().strip()

    def on_detector_engine_changed(self, engine: str):
        if self.restoring_options:
            return
        self.reader = None
        self.paddleocr_detector = None
        self.crops = []
        self.clear_crops()
        self.result_table.setRowCount(0)
        self.log(f"Detector changed to {engine}. Reloading detector...")
        self.reload_detector()

    def on_parseq_model_changed(self, model_name: str):
        if self.restoring_options:
            return
        self.parseq = None
        self.result_table.setRowCount(0)
        model_name = str(model_name or "").strip()
        if not model_name:
            return
        self.log(f"PARSeq model changed to {model_name}. Reloading PARSeq...")
        self.reload_parseq()

    def restore_options(self, snap: dict[str, Any]):
        self.restoring_options = True
        try:
            for name, value in snap.items():
                widget = self.option_widgets.get(name)
                if widget is not None:
                    self.set_widget_value(widget, value)
        finally:
            self.restoring_options = False
        self.update_undo_redo_buttons()

    def undo_options(self):
        if len(self.undo_stack) <= 1:
            return
        current = self.undo_stack.pop()
        self.redo_stack.append(current)
        self.restore_options(self.undo_stack[-1])

    def redo_options(self):
        if not self.redo_stack:
            return
        snap = self.redo_stack.pop()
        self.undo_stack.append(snap)
        self.restore_options(snap)

    def update_undo_redo_buttons(self):
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(len(self.undo_stack) > 1)
        if hasattr(self, "redo_btn"):
            self.redo_btn.setEnabled(bool(self.redo_stack))

    def log(self, msg: str):
        self.status.append(msg)
        QApplication.processEvents()

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image",
            str(THIS_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return
        self.image_path.setText(path)
        self.load_image(path)

    def load_image(self, path: str, show_error: bool = True):
        frame = cv2.imread(path)
        if frame is None or frame.size == 0:
            if show_error:
                QMessageBox.critical(self, "Image error", f"Could not read image:\n{path}")
            else:
                self.log(f"Startup image not loaded: {path}")
            return
        self.frame = frame
        self.detect_img = None
        self.frame_to_detect_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.detect_to_frame_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.crops = []
        self.clear_crops()
        self.result_table.setRowCount(0)
        self.preview_label.setPixmap(cv_to_qpixmap(frame, 920, 260))
        self.log(f"Loaded image: {path} shape={frame.shape}")

    def load_default_image_if_available(self):
        path = self.image_path.text().strip() or str(DEFAULT_IMAGE_PATH)
        if Path(path).exists():
            self.load_image(path, show_error=False)
        else:
            self.log(f"Default image path does not exist on this machine: {path}")

    def reload_detector(self):
        engine = self.detector_engine.currentText()
        if engine.startswith("EasyOCR"):
            self.reload_reader()
        elif engine.startswith("PaddleOCR"):
            self.reload_paddleocr_detector()

    def reload_reader(self):
        langs = [x.strip() for x in self.langs_edit.text().split(",") if x.strip()]
        if not langs:
            QMessageBox.warning(self, "EasyOCR", "Enter at least one language, such as en.")
            return
        self.log(f"Loading EasyOCR/CRAFT detector languages={langs} gpu={self.gpu_check.isChecked()}...")
        t0 = perf_counter()
        try:
            self.reader = easyocr.Reader(langs, gpu=self.gpu_check.isChecked(), detector=True, recognizer=False, verbose=False)
        except Exception as exc:
            self.reader = None
            QMessageBox.critical(self, "EasyOCR detector load failed", str(exc))
            return
        self.log(f"EasyOCR/CRAFT detector loaded in {perf_counter() - t0:.4f}s")

    def paddle_init_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "det_limit_side_len": self.paddle_det_limit_side_len.value(),
            "det_limit_type": self.paddle_det_limit_type.currentText(),
            "det_db_thresh": self.paddle_det_db_thresh.value(),
            "det_db_box_thresh": self.paddle_det_db_box_thresh.value(),
            "det_db_unclip_ratio": self.paddle_det_db_unclip_ratio.value(),
        }
        kwargs.update(parse_json_kwargs(self.paddle_init_extra_json.toPlainText()))
        return kwargs

    def paddle_run_kwargs(self) -> dict[str, Any]:
        return parse_json_kwargs(self.paddle_run_extra_json.toPlainText())

    def reload_paddleocr_detector(self):
        self.log("Loading PaddleOCR detector...")
        t0 = perf_counter()
        try:
            self.paddleocr_detector = PaddleOCRDetectorRunner(
                lang=self.paddle_lang.text().strip(),
                device_choice=self.paddle_device.currentText(),
                use_angle_cls=self.paddle_use_angle_cls.isChecked(),
                init_kwargs=self.paddle_init_kwargs(),
            )
        except Exception as exc:
            self.paddleocr_detector = None
            QMessageBox.critical(self, "PaddleOCR detector load failed", str(exc))
            return
        self.log(
            f"PaddleOCR detector loaded in {perf_counter() - t0:.4f}s "
            f"with init kwargs: {self.paddleocr_detector.init_kwargs}"
        )

    def reload_parseq(self):
        self.log(f"Loading PARSeq model={self.current_parseq_model_name()}...")
        t0 = perf_counter()
        try:
            extra = parse_json_kwargs(self.parseq_extra_json.toPlainText())
            self.parseq = PARSeqRunner(
                repo=self.parseq_repo.text().strip(),
                model_name=self.current_parseq_model_name(),
                device_choice=self.parseq_device.currentText(),
                force_reload=self.parseq_force_reload.isChecked(),
                trust_repo=self.parseq_trust_repo.isChecked(),
                hub_extra_kwargs=extra,
                use_scene_transform=self.use_scene_transform.isChecked(),
                custom_img_h=self.custom_img_h.value(),
                custom_img_w=self.custom_img_w.value(),
                normalize_mean=self.norm_mean.value(),
                normalize_std=self.norm_std.value(),
            )
        except Exception as exc:
            self.parseq = None
            QMessageBox.critical(self, "PARSeq load failed", str(exc))
            return
        self.log(f"PARSeq loaded model={self.parseq.model_name} in {perf_counter() - t0:.4f}s on device={self.parseq.device}")

    def detect_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "min_size": self.min_size.value(),
            "text_threshold": self.text_threshold.value(),
            "low_text": self.low_text.value(),
            "link_threshold": self.link_threshold.value(),
            "canvas_size": self.canvas_size.value(),
            "mag_ratio": self.mag_ratio.value(),
            "slope_ths": self.slope_ths.value(),
            "ycenter_ths": self.ycenter_ths.value(),
            "height_ths": self.height_ths.value(),
            "width_ths": self.width_ths.value(),
            "add_margin": self.add_margin.value(),
            "reformat": self.reformat_check.isChecked(),
            "threshold": self.threshold.value(),
            "bbox_min_score": self.bbox_min_score.value(),
            "bbox_min_size": self.bbox_min_size.value(),
            "max_candidates": self.max_candidates.value(),
        }
        optimal = self.optimal_num_chars.text().strip()
        if optimal:
            kwargs["optimal_num_chars"] = int(optimal)
        kwargs.update(parse_json_kwargs(self.detect_extra_json.toPlainText()))
        return kwargs

    def run_detect(self):
        if self.frame is None:
            path = self.image_path.text().strip()
            if path:
                self.load_image(path)
            if self.frame is None:
                QMessageBox.warning(self, "No image", "Load an image first.")
                return

        engine = self.detector_engine.currentText()
        try:
            upsample_scale = parse_scale_choice(self.upsample_scale.currentText())
            self.detect_img, self.frame_to_detect_matrix, steps = build_detection_preprocess_image(
                self.frame,
                use_ocr_preprocess=self.preprocess_check.isChecked(),
                upsample_enabled=self.upsample_check.isChecked(),
                upsample_scale=upsample_scale,
                horizontalize_enabled=self.horizontalize_check.isChecked(),
                horizontalize_auto=self.horizontalize_auto_check.isChecked(),
                manual_angle=self.horizontalize_angle.value(),
            )
            self.detect_to_frame_matrix = cv2.invertAffineTransform(self.frame_to_detect_matrix)
        except Exception as exc:
            QMessageBox.critical(self, "Preprocessing failed", str(exc))
            return

        if steps:
            self.preview_label.setPixmap(cv_to_qpixmap(self.detect_img, 920, 260))
            self.log(f"Using processed image for {engine}: " + "; ".join(steps))
            self.log(f"Detection image shape={self.detect_img.shape}; original shape={self.frame.shape}")
        else:
            self.preview_label.setPixmap(cv_to_qpixmap(self.frame, 920, 260))
            self.log(f"Using original image for {engine}.")

        if engine.startswith("EasyOCR"):
            self.run_easyocr_detect()
        elif engine.startswith("PaddleOCR"):
            self.run_paddleocr_detect()

    def run_easyocr_detect(self):
        if self.reader is None:
            self.reload_reader()
            if self.reader is None:
                return

        try:
            kwargs = supported_kwargs(self.reader.detect, self.detect_kwargs())
        except Exception:
            kwargs = self.detect_kwargs()

        self.log(f"Running EasyOCR/CRAFT detect() with kwargs: {kwargs}")
        t0 = perf_counter()
        try:
            raw_out = self.reader.detect(self.detect_img, **kwargs)
        except Exception as exc:
            QMessageBox.critical(self, "EasyOCR detect failed", str(exc))
            return
        dt = perf_counter() - t0
        horizontal, free = normalize_detect_output(raw_out)
        self.log(f"EasyOCR/CRAFT detect() finished in {dt:.4f}s. horizontal={len(horizontal)} free={len(free)}")
        self.crops = self.make_crops(horizontal, free)
        self.show_crops()

    def run_paddleocr_detect(self):
        if self.paddleocr_detector is None:
            self.reload_paddleocr_detector()
            if self.paddleocr_detector is None:
                return

        kwargs = self.paddle_run_kwargs()
        try:
            paddle_input = ensure_paddleocr_compatible_image(self.detect_img)
        except Exception as exc:
            QMessageBox.critical(self, "PaddleOCR input error", str(exc))
            return

        if paddle_input.shape[:2] == self.detect_img.shape[:2] and getattr(self.detect_img, "ndim", 0) == 2:
            self.log("Converted preprocessed grayscale image to 3-channel BGR for PaddleOCR.")

        self.log(f"Running PaddleOCR detection-only mode with run kwargs: {kwargs}")
        t0 = perf_counter()
        try:
            boxes, output_desc = self.paddleocr_detector.detect(paddle_input, kwargs)
        except Exception as exc:
            QMessageBox.critical(self, "PaddleOCR detect failed", str(exc))
            return
        dt = perf_counter() - t0
        self.log(f"PaddleOCR detection finished in {dt:.4f}s. boxes={len(boxes)}; {output_desc}")
        self.crops = self.make_crops_from_points(boxes, kind="paddle")
        self.show_crops()

    def map_detect_points_to_original(self, pts: Any) -> np.ndarray:
        try:
            matrix = getattr(self, "detect_to_frame_matrix", None)
            if matrix is not None:
                mapped = transform_points_affine(pts, matrix)
                if np.isfinite(mapped).all():
                    return np.asarray(mapped, dtype=np.float32)
        except Exception:
            pass
        assert self.detect_img is not None
        assert self.frame is not None
        return scale_box_points(pts, self.detect_img.shape, self.frame.shape)

    def make_crops(self, horizontal: list[Any], free: list[Any]) -> list[DetectedCrop]:
        crops: list[DetectedCrop] = []
        idx = 1
        pad = self.crop_pad.value()
        assert self.detect_img is not None
        assert self.frame is not None

        for box in horizontal:
            try:
                pts = horizontal_box_to_points(box)
                detect_crop = crop_easyocr_text_region(self.detect_img, pts, pad=pad)
                original_pts = self.map_detect_points_to_original(pts)
                original_crop = crop_easyocr_text_region(self.frame, original_pts, pad=pad)
                crops.append(DetectedCrop(idx, "horizontal", pts, detect_crop, original_crop))
                idx += 1
            except Exception as exc:
                self.log(f"Skipped horizontal box due to error: {exc}")

        for box in free:
            try:
                pts = np.array(box, dtype=np.float32).reshape(4, 2)
                detect_crop = crop_easyocr_text_region(self.detect_img, pts, pad=pad)
                original_pts = self.map_detect_points_to_original(pts)
                original_crop = crop_easyocr_text_region(self.frame, original_pts, pad=pad)
                crops.append(DetectedCrop(idx, "free", pts, detect_crop, original_crop))
                idx += 1
            except Exception as exc:
                self.log(f"Skipped free box due to error: {exc}")

        return crops

    def make_crops_from_points(self, boxes: list[Any], kind: str) -> list[DetectedCrop]:
        crops: list[DetectedCrop] = []
        idx = 1
        pad = self.crop_pad.value()
        assert self.detect_img is not None
        assert self.frame is not None

        for box in boxes:
            try:
                pts = np.array(box, dtype=np.float32).reshape(4, 2)
                detect_crop = crop_easyocr_text_region(self.detect_img, pts, pad=pad)
                original_pts = self.map_detect_points_to_original(pts)
                original_crop = crop_easyocr_text_region(self.frame, original_pts, pad=pad)
                crops.append(DetectedCrop(idx, kind, pts, detect_crop, original_crop))
                idx += 1
            except Exception as exc:
                self.log(f"Skipped {kind} box due to error: {exc}")

        return crops

    def clear_crops(self):
        while self.crop_grid.count():
            item = self.crop_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def show_crops(self):
        self.clear_crops()
        cols = 3
        tw = self.thumb_width.value()
        th = self.thumb_height.value()
        for n, crop in enumerate(self.crops):
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("QLabel { border: 1px solid #888; background: #111; }")
            label.setPixmap(cv_to_qpixmap(crop.detect_crop, tw, th))
            self.crop_grid.addWidget(label, n // cols, n % cols)
        self.log(f"Displayed {len(self.crops)} detected crop image(s).")

    def run_parseq(self):
        if not self.crops:
            QMessageBox.warning(self, "No crops", "Run text detection first.")
            return
        if self.parseq is None:
            self.reload_parseq()
            if self.parseq is None:
                return

        self.result_table.setRowCount(0)
        source_name = self.parseq_source.currentText()
        model_name = getattr(self.parseq, "model_name", self.current_parseq_model_name())
        use_original = source_name == "Original-image scaled crop"
        self.log(f"Running PARSeq model={model_name} on {len(self.crops)} crop(s), source={source_name}...")

        for crop in self.crops:
            img = crop.original_crop if use_original else crop.detect_crop
            t0 = perf_counter()
            try:
                text, conf = self.parseq(img)
            except Exception as exc:
                text, conf = f"[PARSeq error: {exc}]", 0.0
            dt = perf_counter() - t0
            if self.split_parseq.isChecked():
                text = maybe_split_parseq_text(text)
            else:
                text = clean_ocr_text(text)

            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(str(crop.index)))
            self.result_table.setItem(row, 1, QTableWidgetItem(f"{source_name} | {model_name}"))
            self.result_table.setItem(row, 2, QTableWidgetItem(text))
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{conf:.4f}"))
            self.result_table.setItem(row, 4, QTableWidgetItem(f"{dt:.4f}s"))
            self.log(f"{crop.index}: {text!r} conf={conf:.4f} time={dt:.4f}s")

        self.result_table.resizeColumnsToContents()
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.log("PARSeq finished.")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
