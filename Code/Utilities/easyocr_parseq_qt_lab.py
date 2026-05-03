#!/usr/bin/env python3
"""
easyocr_parseq_qt_lab.py

Standalone PyQt6 tool for experimenting with EasyOCR detect() and PARSeq.
Shows the EasyOCR detect() defaults/ranges in the GUI.

Usage:
  python easyocr_parseq_qt_lab.py

Install:
  pip install PyQt6 easyocr torch torchvision pillow opencv-python numpy
  pip install strhub-sdk
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

EASYOCR_DETECT_OPTION_REFERENCE = [
    ("min_size", "8", "0–10000 (int)", "Minimum text box size"),
    ("text_threshold", "0.36", "0.0–1.0 (float)", "Text confidence threshold"),
    ("low_text", "0.18", "0.0–1.0 (float)", "Lower bound for text confidence"),
    ("link_threshold", "0.18", "0.0–1.0 (float)", "Threshold for linking text boxes"),
    ("canvas_size", "1280", "32–20000 (int)", "Resize image for detection"),
    ("mag_ratio", "1.0", "0.1–10.0 (float)", "Magnification ratio for resizing"),
    ("slope_ths", "0.1", "0.0–10.0 (float)", "Slope threshold for merging boxes"),
    ("ycenter_ths", "0.5", "0.0–10.0 (float)", "Y-center threshold for merging boxes"),
    ("height_ths", "0.5", "0.0–10.0 (float)", "Height threshold for merging boxes"),
    ("width_ths", "0.5", "0.0–20.0 (float)", "Width threshold for merging boxes"),
    ("add_margin", "0.12", "0.0–10.0 (float)", "Add margin to detected boxes"),
    ("optimal_num_chars", "None", "int or blank", "Expected number of chars per box"),
    ("reformat", "True", "bool (checkbox)", "Reformat output"),
    ("threshold", "0.2", "0.0–1.0 (float)", "Threshold for binarization/post-processing"),
    ("bbox_min_score", "0.2", "0.0–1.0 (float)", "Minimum bbox score"),
    ("bbox_min_size", "3", "0–10000 (int)", "Minimum bbox size"),
    ("max_candidates", "0", "0–100000 (int)", "Max candidates; 0 = no limit"),
    ("Extra kwargs", "—", "JSON dict", "Any other supported EasyOCR detect() kwargs"),
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
        self.setWindowTitle("EasyOCR Detect + PARSeq Lab")
        self.resize(1500, 920)
        self.frame: np.ndarray | None = None
        self.detect_img: np.ndarray | None = None
        self.reader: easyocr.Reader | None = None
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
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo_options)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo_options)

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
        self.reload_reader_btn = QPushButton("Reload EasyOCR")
        self.reload_reader_btn.clicked.connect(self.reload_reader)
        self.detect_btn = QPushButton("Run EasyOCR detect()")
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

        crops_group = QGroupBox("EasyOCR detect() Crops")
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
        self.image_path = QLineEdit()
        image_form.addRow("Image path", self.image_path)
        self.preprocess_check = QCheckBox("Use standalone preprocessing before EasyOCR detect")
        self.preprocess_check.setChecked(True)
        image_form.addRow(self.preprocess_check)
        layout.addWidget(image_group)

        reader_group = QGroupBox("EasyOCR Reader")
        reader_form = QFormLayout(reader_group)
        self.langs_edit = QLineEdit("en")
        self.gpu_check = QCheckBox("Use GPU for EasyOCR")
        self.gpu_check.setChecked(False)
        reader_form.addRow("Languages comma-separated", self.langs_edit)
        reader_form.addRow(self.gpu_check)
        layout.addWidget(reader_group)
        layout.addStretch(1)

        self.tabs.addTab(tab, "Image / Reader")

    def build_easyocr_detect_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.min_size = self.spin_int(0, 10000, 8)
        self.text_threshold = self.spin_float(0.0, 1.0, 0.36, 0.01)
        self.low_text = self.spin_float(0.0, 1.0, 0.18, 0.01)
        self.link_threshold = self.spin_float(0.0, 1.0, 0.18, 0.01)
        self.canvas_size = self.spin_int(32, 20000, 1280)
        self.mag_ratio = self.spin_float(0.1, 10.0, 1.0, 0.1)
        self.slope_ths = self.spin_float(0.0, 10.0, 0.1, 0.05)
        self.ycenter_ths = self.spin_float(0.0, 10.0, 0.5, 0.05)
        self.height_ths = self.spin_float(0.0, 10.0, 0.5, 0.05)
        self.width_ths = self.spin_float(0.0, 20.0, 0.5, 0.05)
        self.add_margin = self.spin_float(0.0, 10.0, 0.12, 0.01)
        self.optimal_num_chars = QLineEdit("")
        self.reformat_check = QCheckBox("reformat")
        self.reformat_check.setChecked(True)
        self.threshold = self.spin_float(0.0, 1.0, 0.2, 0.01)
        self.bbox_min_score = self.spin_float(0.0, 1.0, 0.2, 0.01)
        self.bbox_min_size = self.spin_int(0, 10000, 3)
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

    def build_detect_reference_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.detect_ref_table = self.make_detect_reference_table()
        layout.addWidget(self.detect_ref_table)
        note = QLabel(
            "The GUI passes these values to easyocr.Reader.detect(). "
            "Extra JSON is merged into the detect() kwargs."
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
        self.parseq_model = QLineEdit("parseq")
        self.parseq_device = QComboBox()
        self.parseq_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.parseq_source = QComboBox()
        self.parseq_source.addItems(["EasyOCR detect crop", "Original-image scaled crop"])
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
        form.addRow("model name", self.parseq_model)
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
            elif isinstance(widget, QTextEdit):
                widget.textChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(lambda *_: self.snapshot_options(clear_redo=True))

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
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return
        self.image_path.setText(path)
        self.load_image(path)

    def load_image(self, path: str):
        frame = cv2.imread(path)
        if frame is None or frame.size == 0:
            QMessageBox.critical(self, "Image error", f"Could not read image:\n{path}")
            return
        self.frame = frame
        self.detect_img = None
        self.crops = []
        self.clear_crops()
        self.result_table.setRowCount(0)
        self.preview_label.setPixmap(cv_to_qpixmap(frame, 920, 260))
        self.log(f"Loaded image: {path} shape={frame.shape}")

    def reload_reader(self):
        langs = [x.strip() for x in self.langs_edit.text().split(",") if x.strip()]
        if not langs:
            QMessageBox.warning(self, "EasyOCR", "Enter at least one language, such as en.")
            return
        self.log(f"Loading EasyOCR Reader languages={langs} gpu={self.gpu_check.isChecked()}...")
        t0 = perf_counter()
        try:
            self.reader = easyocr.Reader(langs, gpu=self.gpu_check.isChecked(), detector=True, recognizer=False, verbose=False)
        except Exception as exc:
            self.reader = None
            QMessageBox.critical(self, "EasyOCR load failed", str(exc))
            return
        self.log(f"EasyOCR Reader loaded in {perf_counter() - t0:.4f}s")

    def reload_parseq(self):
        self.log("Loading PARSeq...")
        t0 = perf_counter()
        try:
            extra = parse_json_kwargs(self.parseq_extra_json.toPlainText())
            self.parseq = PARSeqRunner(
                repo=self.parseq_repo.text().strip(),
                model_name=self.parseq_model.text().strip(),
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
        self.log(f"PARSeq loaded in {perf_counter() - t0:.4f}s on device={self.parseq.device}")

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
        if self.reader is None:
            self.reload_reader()
            if self.reader is None:
                return

        if self.preprocess_check.isChecked():
            self.detect_img = preprocess_for_ocr(self.frame)
            self.log("Using preprocessed image for EasyOCR detect().")
        else:
            self.detect_img = self.frame.copy()
            self.log("Using original image for EasyOCR detect().")

        try:
            kwargs = supported_kwargs(self.reader.detect, self.detect_kwargs())
        except Exception:
            kwargs = self.detect_kwargs()

        self.log(f"Running EasyOCR detect() with kwargs: {kwargs}")
        t0 = perf_counter()
        try:
            raw_out = self.reader.detect(self.detect_img, **kwargs)
        except Exception as exc:
            QMessageBox.critical(self, "EasyOCR detect failed", str(exc))
            return
        dt = perf_counter() - t0
        horizontal, free = normalize_detect_output(raw_out)
        self.log(f"EasyOCR detect() finished in {dt:.4f}s. horizontal={len(horizontal)} free={len(free)}")
        self.crops = self.make_crops(horizontal, free)
        self.show_crops()

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
                original_pts = scale_box_points(pts, self.detect_img.shape, self.frame.shape)
                original_crop = crop_easyocr_text_region(self.frame, original_pts, pad=pad)
                crops.append(DetectedCrop(idx, "horizontal", pts, detect_crop, original_crop))
                idx += 1
            except Exception as exc:
                self.log(f"Skipped horizontal box due to error: {exc}")

        for box in free:
            try:
                pts = np.array(box, dtype=np.float32).reshape(4, 2)
                detect_crop = crop_easyocr_text_region(self.detect_img, pts, pad=pad)
                original_pts = scale_box_points(pts, self.detect_img.shape, self.frame.shape)
                original_crop = crop_easyocr_text_region(self.frame, original_pts, pad=pad)
                crops.append(DetectedCrop(idx, "free", pts, detect_crop, original_crop))
                idx += 1
            except Exception as exc:
                self.log(f"Skipped free box due to error: {exc}")

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
            QMessageBox.warning(self, "No crops", "Run EasyOCR detect() first.")
            return
        if self.parseq is None:
            self.reload_parseq()
            if self.parseq is None:
                return

        self.result_table.setRowCount(0)
        source_name = self.parseq_source.currentText()
        use_original = source_name == "Original-image scaled crop"
        self.log(f"Running PARSeq on {len(self.crops)} crop(s), source={source_name}...")

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
            self.result_table.setItem(row, 1, QTableWidgetItem(source_name))
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
