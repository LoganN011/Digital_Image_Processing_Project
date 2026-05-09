#!/usr/bin/env python3
                           
from __future__ import annotations

import csv
import json
import math
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import (

    QAbstractTableModel,
    QEventLoop,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QPoint,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStyle,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR

OCR_LAB_STREAM_HOST = "127.0.0.1"
OCR_LAB_STREAM_PORT = 49327
OCR_LAB_CONNECT_TIMEOUT_SEC = 0.35
OCR_LAB_STARTUP_SEND_WINDOW_SEC = 2.5

COMPACT_UI_FONT_POINT_SIZE = 9
COMPACT_UI_STYLE_SHEET = """
QWidget {
    font-size: 9pt;
}
QMenuBar, QMenu, QStatusBar {
    font-size: 9pt;
}
QPushButton, QComboBox, QDoubleSpinBox, QLineEdit {
    font-size: 9pt;
}
QGroupBox {
    font-size: 9pt;
    margin-top: 0.75em;
}
QGroupBox::title {
    font-size: 9pt;
    subcontrol-origin: margin;
    left: 6px;
    padding: 0 3px;
}
QLabel {
    font-size: 9pt;
}
QTabBar::tab {
    font-size: 9pt;
    padding: 3px 7px;
}
QTableView {
    font-size: 8pt;
}
QHeaderView::section {
    font-size: 8pt;
    padding: 2px 4px;
}
QTextEdit {
    font-size: 8pt;
}
QProgressBar {
    font-size: 8pt;
}
"""


def apply_compact_ui_font(app: QApplication) -> None:
    font = app.font()
    font.setPointSize(COMPACT_UI_FONT_POINT_SIZE)
    app.setFont(font)
    app.setStyleSheet(COMPACT_UI_STYLE_SHEET)


DEFAULT_WEIGHTS = {
    "sharpness": 24.0,
    "contrast": 18.0,
    "edge_density": 16.0,
    "foreground": 17.0,
    "components": 0.0,
    "resolution": 50.0,
                         
                         
}

WEIGHT_ORDER = [
    ("sharpness", "Sharpness"),
    ("contrast", "Contrast"),
    ("edge_density", "Edge density"),
    ("foreground", "Text-pixel mass"),
    ("components", "Components"),
    ("resolution", "Resolution"),
]

DEFAULT_OCR_REDUCER_SETTINGS = {
    "border_region_percent": 4.5,
    "border_penalty_max_percent": 28.0,
    "border_ratio_start": 0.80,
    "border_ratio_range": 2.20,
    "saturation_penalty_max_percent": 18.0,
    "saturation_start_percent": 18.0,
    "saturation_range_percent": 62.0,
}

OCR_REDUCER_ORDER = [
    ("border_region_percent", "Border region width", "Percent of crop width/height treated as border when checking for cut-off text."),
    ("border_penalty_max_percent", "Border-cut max penalty", "Maximum reduction applied when too much foreground sits near the crop border."),
    ("border_ratio_start", "Border ratio start", "Penalty begins once border foreground / center foreground exceeds this value."),
    ("border_ratio_range", "Border ratio range", "How quickly the border-cut penalty ramps up after the start threshold."),
    ("saturation_penalty_max_percent", "Saturation max penalty", "Maximum reduction applied when too many pixels are near black or near white."),
    ("saturation_start_percent", "Saturation start", "Penalty begins once the near-black/near-white fraction exceeds this percent."),
    ("saturation_range_percent", "Saturation range", "How quickly the saturation penalty ramps up after the start threshold."),
]

DEFAULT_YOLO_WEIGHTS = {
    "detector": 42.0,
    "sharpness": 33.0,
    "area": 18.0,
    "aspect": 7.0,
    "edge_penalty": 16.0,
}

YOLO_WEIGHT_ORDER = [
    ("detector", "Detector confidence coefficient"),
    ("sharpness", "Sharpness coefficient"),
    ("area", "Area coefficient"),
    ("aspect", "Aspect coefficient"),
    ("edge_penalty", "Edge penalty amount"),
]


DEFAULT_CRAFT_HEATMAP_SETTINGS = {
    "gpu": 0.0,
    "canvas_size": 1280.0,
    "mag_ratio": 1.0,
    "text_threshold": 0.36,
    "low_text": 0.18,
    "link_threshold": 0.18,
    "craft_weight_text_sum_percent": 50.0,
    "craft_weight_affinity_sum_percent": 20.0,
    "craft_weight_weak_area_percent": 15.0,
    "craft_weight_strong_area_percent": 10.0,
    "craft_weight_peak_text_percent": 5.0,
    "craft_weight_peak_affinity_percent": 0.0,
    "text_density_good_percent": 8.0,
    "affinity_density_good_percent": 6.0,
    "weak_text_area_good_percent": 25.0,
    "strong_text_area_good_percent": 8.0,
}

CRAFT_HEATMAP_ORDER = [
    ("gpu", "Use GPU", "0 = CPU, 1 = GPU if available. This changes speed/device only."),
    ("canvas_size", "Canvas size", "Resize limit before CRAFT inference. Larger values preserve small text but run slower."),
    ("mag_ratio", "Magnification", "Extra magnification before CRAFT inference. Increase for tiny text; may add false positives."),
    ("text_threshold", "Text threshold", "CRAFT peak text-region threshold for strong text evidence."),
    ("low_text", "Low text", "CRAFT low text threshold used for weak text-region heatmap energy."),
    ("link_threshold", "Affinity threshold", "CRAFT affinity/link threshold used for link heatmap energy."),
    ("craft_weight_text_sum_percent", "Text Sum weight", "Formula contribution weight for the normalized score_text heatmap sum term. These weights are normalized during scoring, so they do not have to add to exactly 100%."),
    ("craft_weight_affinity_sum_percent", "Affinity Sum weight", "Formula contribution weight for the normalized score_link / affinity heatmap sum term."),
    ("craft_weight_weak_area_percent", "Weak Area weight", "Formula contribution weight for the fraction of heatmap pixels above low_text."),
    ("craft_weight_strong_area_percent", "Strong Area weight", "Formula contribution weight for the fraction of heatmap pixels above text_threshold."),
    ("craft_weight_peak_text_percent", "Peak Text weight", "Formula contribution weight for the maximum score_text value in the crop."),
    ("craft_weight_peak_affinity_percent", "Peak Affinity weight", "Formula contribution weight for the maximum score_link / affinity value in the crop. Default is 0% so the old formula is preserved until changed."),
    ("text_density_good_percent", "Text density good threshold", "Percent of normalized score_text energy considered enough to saturate the text-density subscore."),
    ("affinity_density_good_percent", "Affinity density good threshold", "Percent of normalized score_link energy considered enough to saturate the affinity-density subscore."),
    ("weak_text_area_good_percent", "Weak text area good threshold", "Percent of heatmap pixels above low_text considered enough to saturate the weak-text-area subscore."),
    ("strong_text_area_good_percent", "Strong text area good threshold", "Percent of heatmap pixels above text_threshold considered enough to saturate the strong-text-area subscore."),
]


CRAFT_HEATMAP_WEIGHT_KEYS = [
    ("craft_weight_text_sum_percent", "Text Sum"),
    ("craft_weight_affinity_sum_percent", "Affinity Sum"),
    ("craft_weight_weak_area_percent", "Weak Area"),
    ("craft_weight_strong_area_percent", "Strong Area"),
    ("craft_weight_peak_text_percent", "Peak Text"),
    ("craft_weight_peak_affinity_percent", "Peak Affinity"),
]


def normalized_craft_heatmap_weights(settings: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    raw = {
        key: max(0.0, float(settings.get(key, DEFAULT_CRAFT_HEATMAP_SETTINGS[key])))
        for key, _label in CRAFT_HEATMAP_WEIGHT_KEYS
    }
    total = float(sum(raw.values()))
    if total <= 0.0:
        raw = {
            key: max(0.0, float(DEFAULT_CRAFT_HEATMAP_SETTINGS[key]))
            for key, _label in CRAFT_HEATMAP_WEIGHT_KEYS
        }
        total = float(sum(raw.values()))
    normalized = {key: raw[key] / max(total, 1e-9) for key, _label in CRAFT_HEATMAP_WEIGHT_KEYS}
    return raw, normalized, total


def capped_heatmap_energy_and_area(
    score_map: np.ndarray,
    threshold: float,
    *,
    cap_fraction: float = 0.006,
) -> Tuple[float, float, float, int, int]:
                                                          
                                                         
                                            
    if score_map.size == 0:
        return 0.0, 0.0, 0.0, 0, 0

    score_map = np.nan_to_num(score_map.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    active = score_map > float(threshold)
    raw_area = float(np.mean(active))
    if not np.any(active):
        return 0.0, 0.0, raw_area, 0, 0

    n_pixels = float(score_map.size)
    cap_pixels = max(4.0, cap_fraction * n_pixels)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), connectivity=8)

    capped_energy = 0.0
    capped_area = 0.0
    component_count = 0
    capped_component_count = 0

    denom = max(1e-6, 1.0 - float(threshold))
    for label_id in range(1, n_labels):
        area = float(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0.0:
            continue

        component_count += 1
        component_mask = labels == label_id
        excess = np.maximum(score_map[component_mask] - float(threshold), 0.0) / denom
        mean_excess = float(excess.mean()) if excess.size else 0.0

        capped_pixels = min(area, cap_pixels)
        if area > cap_pixels:
            capped_component_count += 1

        capped_area += capped_pixels
        capped_energy += mean_excess * capped_pixels

    return (
        float(capped_energy / n_pixels),
        float(capped_area / n_pixels),
        raw_area,
        int(component_count),
        int(capped_component_count),
    )


@dataclass(frozen=True)
class CropScoreResult:
    path: Path
    score: float
    width: int
    height: int
    metrics: Dict[str, Any]
    craft_text_heatmap: Optional[np.ndarray] = None
    craft_affinity_heatmap: Optional[np.ndarray] = None
    timing: Optional[Dict[str, Any]] = None


def identity_index(items: Sequence[Any], target: Any) -> int:
    for i, item in enumerate(items):
        if item is target:
            return i
    return -1


class CropTableModel(QAbstractTableModel):
    YOLO_COLUMNS = [
        "rank", "rank_score", "crop_number", "yolo_quality", "yolo_detector", "yolo_sharpness",
        "yolo_area", "yolo_aspect", "yolo_edge", "track", "candidate",
        "frame", "best", "file", "width", "height", "ocr_quality",
        "ocr_sharpness", "ocr_contrast", "ocr_edges", "ocr_foreground",
        "ocr_components", "ocr_resolution",
    ]
    OCR_COLUMNS = [
        "rank", "rank_score", "crop_number", "ocr_quality", "ocr_sharpness", "ocr_contrast",
        "ocr_edges", "ocr_foreground", "ocr_components", "ocr_resolution",
        "file", "width", "height", "yolo_quality", "yolo_detector",
        "track", "candidate", "frame", "best", "yolo_sharpness",
        "yolo_area", "yolo_aspect", "yolo_edge",
    ]
    CRAFT_COLUMNS = [
        "rank", "rank_score", "crop_number", "craft_quality", "craft_text_density",
        "craft_affinity_density", "craft_weak_area", "craft_strong_area", "craft_peak_text",
        "craft_peak_affinity", "craft_heatmap_size", "file", "width", "height",
        "yolo_quality", "yolo_detector", "track", "candidate", "frame", "best",
    ]
    HEADERS_BY_ID = {
        "rank": "Rank",
        "rank_score": "Rank Score",
        "crop_number": "Crop #",
        "yolo_quality": "YOLO Q",
        "yolo_detector": "Det",
        "yolo_sharpness": "Y-Sharp",
        "yolo_area": "Y-Area",
        "yolo_aspect": "Y-Aspect",
        "yolo_edge": "Y-Edge",
        "track": "Track",
        "candidate": "Cand",
        "frame": "Frame",
        "best": "Best",
        "ocr_quality": "OCR",
        "file": "File",
        "width": "W",
        "height": "H",
        "ocr_sharpness": "O-Sharp",
        "ocr_contrast": "Contrast",
        "ocr_edges": "Edges",
        "ocr_foreground": "Text px",
        "ocr_components": "Components",
        "ocr_resolution": "Resolution",
        "craft_quality": "CRAFT Q",
        "craft_text_density": "Text Sum",
        "craft_affinity_density": "Affinity Sum",
        "craft_weak_area": "Weak Area",
        "craft_strong_area": "Strong Area",
        "craft_peak_text": "Text Peak",
        "craft_peak_affinity": "Affinity Peak",
        "craft_heatmap_size": "Heatmap",
    }

    def __init__(self) -> None:
        super().__init__()
        self.results: List[CropScoreResult] = []
        self._default_order: Dict[int, int] = {}
        self.ranking_method = "yolo"

    def set_ranking_method(self, ranking_method: str) -> None:
        ranking_method = ranking_method if ranking_method in {"ocr", "yolo", "craft"} else "ocr"
        if ranking_method == self.ranking_method:
            return
        self.beginResetModel()
        self.ranking_method = ranking_method
        self.endResetModel()

    def columns(self) -> List[str]:
        if self.ranking_method == "craft":
            return self.CRAFT_COLUMNS
        return self.OCR_COLUMNS if self.ranking_method == "ocr" else self.YOLO_COLUMNS

    def set_results(self, results: Sequence[CropScoreResult]) -> None:
        self.beginResetModel()
        self.results = list(results)
        self._default_order = {id(result): i for i, result in enumerate(self.results)}
        self.endResetModel()

    def default_rank_for_result(self, result: CropScoreResult) -> Optional[int]:
        original_row = self._default_order.get(id(result))
        if original_row is not None:
            return int(original_row) + 1
        current_row = identity_index(self.results, result)
        return None if current_row < 0 else current_row + 1

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.results)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns())

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            cols = self.columns()
            if 0 <= section < len(cols):
                return self.HEADERS_BY_ID.get(cols[section], cols[section])
            return None
        return str(section + 1)

    @staticmethod
    def _fmt(value: Any, digits: int = 2) -> str:
        if value is None or value == "":
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        result = self.results[row]
        m = result.metrics
        cols = self.columns()
        if not 0 <= col < len(cols):
            return None
        col_id = cols[col]

        if role == Qt.ItemDataRole.DisplayRole:
            if col_id == "rank":
                rank = self.default_rank_for_result(result)
                return "—" if rank is None else str(rank)
            if col_id == "rank_score":
                return "—" if result.score < 0 else f"{result.score:.2f}"
            if col_id == "crop_number":
                value = m.get("crop_number")
                return "—" if value is None else str(value)
            if col_id == "yolo_quality":
                return self._fmt(m.get("yolo_quality_score_0_to_100"), 2)
            if col_id == "yolo_detector":
                return self._fmt(m.get("yolo_detector_score"), 3)
            if col_id == "yolo_sharpness":
                return self._fmt(m.get("yolo_sharpness_score"), 3)
            if col_id == "yolo_area":
                return self._fmt(m.get("yolo_area_score"), 3)
            if col_id == "yolo_aspect":
                return self._fmt(m.get("yolo_aspect_score"), 3)
            if col_id == "yolo_edge":
                return self._fmt(m.get("yolo_edge_penalty"), 3)
            if col_id == "track":
                value = m.get("yolo_track_id")
                return "—" if value is None else str(value)
            if col_id == "candidate":
                value = m.get("yolo_candidate_index")
                return "—" if value is None else str(value)
            if col_id == "frame":
                value = m.get("yolo_frame_index")
                return "—" if value is None else str(value)
            if col_id == "best":
                return "★" if bool(m.get("yolo_is_best", False)) else ""
            if col_id == "ocr_quality":
                return self._fmt(m.get("ocr_quality_score"), 2)
            if col_id == "file":
                return result.path.name
            if col_id == "width":
                return str(result.width)
            if col_id == "height":
                return str(result.height)
            if col_id == "ocr_sharpness":
                return self._fmt(m.get("sharpness_score"), 2)
            if col_id == "ocr_contrast":
                return self._fmt(m.get("contrast_score"), 2)
            if col_id == "ocr_edges":
                return self._fmt(m.get("edge_density_score"), 2)
            if col_id == "ocr_foreground":
                return self._fmt(m.get("foreground_score"), 2)
            if col_id == "ocr_components":
                return self._fmt(m.get("component_score"), 2)
            if col_id == "ocr_resolution":
                return self._fmt(m.get("resolution_score"), 2)
            if col_id == "craft_quality":
                return self._fmt(m.get("craft_heatmap_score"), 2)
            if col_id == "craft_text_density":
                return self._fmt(m.get("craft_text_density_score"), 3)
            if col_id == "craft_affinity_density":
                return self._fmt(m.get("craft_affinity_density_score"), 3)
            if col_id == "craft_weak_area":
                return self._fmt(m.get("craft_weak_text_area_score"), 3)
            if col_id == "craft_strong_area":
                return self._fmt(m.get("craft_strong_text_area_score"), 3)
            if col_id == "craft_peak_text":
                return self._fmt(m.get("craft_peak_text"), 3)
            if col_id == "craft_peak_affinity":
                return self._fmt(m.get("craft_peak_affinity"), 3)
            if col_id == "craft_heatmap_size":
                hw = m.get("craft_heatmap_width")
                hh = m.get("craft_heatmap_height")
                return "—" if hw is None or hh is None else f"{int(hw)}×{int(hh)}"

        if role == Qt.ItemDataRole.ToolTipRole:
            return str(result.path)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col_id == "file":
                return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            return int(Qt.AlignmentFlag.AlignCenter)

        if role == Qt.ItemDataRole.FontRole and self.default_rank_for_result(result) == 1:
            font = QFont()
            font.setBold(True)
            return font

        if role == Qt.ItemDataRole.BackgroundRole and self.default_rank_for_result(result) == 1:
            return QBrush(QColor(230, 246, 236))

        return None

    def _sort_value_for_column(self, result: CropScoreResult, col_id: str) -> Any:
        m = result.metrics
        if col_id == "rank":
            return self.default_rank_for_result(result)
        if col_id == "rank_score":
            return None if result.score < 0 else float(result.score)
        if col_id == "file":
            return result.path.name
        if col_id == "width":
            return result.width
        if col_id == "height":
            return result.height
        if col_id == "best":
            return 1 if bool(m.get("yolo_is_best", False)) else 0
        if col_id == "craft_heatmap_size":
            hw = optional_int(m.get("craft_heatmap_width"))
            hh = optional_int(m.get("craft_heatmap_height"))
            return None if hw is None or hh is None else (int(hw) * int(hh), int(hw), int(hh))

        metric_key_by_column = {
            "crop_number": "crop_number",
            "yolo_quality": "yolo_quality_score_0_to_100",
            "yolo_detector": "yolo_detector_score",
            "yolo_sharpness": "yolo_sharpness_score",
            "yolo_area": "yolo_area_score",
            "yolo_aspect": "yolo_aspect_score",
            "yolo_edge": "yolo_edge_penalty",
            "track": "yolo_track_id",
            "candidate": "yolo_candidate_index",
            "frame": "yolo_frame_index",
            "ocr_quality": "ocr_quality_score",
            "ocr_sharpness": "sharpness_score",
            "ocr_contrast": "contrast_score",
            "ocr_edges": "edge_density_score",
            "ocr_foreground": "foreground_score",
            "ocr_components": "component_score",
            "ocr_resolution": "resolution_score",
            "craft_quality": "craft_heatmap_score",
            "craft_text_density": "craft_text_density_score",
            "craft_affinity_density": "craft_affinity_density_score",
            "craft_weak_area": "craft_weak_text_area_score",
            "craft_strong_area": "craft_strong_text_area_score",
            "craft_peak_text": "craft_peak_text",
            "craft_peak_affinity": "craft_peak_affinity",
        }
        key = metric_key_by_column.get(col_id)
        return m.get(key) if key else None

    @staticmethod
    def _is_missing_sort_value(value: Any) -> bool:
        if value is None or value == "" or value == "—":
            return True
        if isinstance(value, (float, np.floating)) and (math.isnan(float(value)) or math.isinf(float(value))):
            return True
        return False

    @staticmethod
    def _sort_key(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(CropTableModel._sort_key(part) for part in value)
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip().lower()
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", text)
            if part != ""
        )

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        cols = self.columns()
        if not (0 <= column < len(cols)) or len(self.results) <= 1:
            return

        col_id = cols[column]
        reverse = order == Qt.SortOrder.DescendingOrder
        present: List[Tuple[Any, CropScoreResult]] = []
        missing: List[CropScoreResult] = []

        for result in self.results:
            value = self._sort_value_for_column(result, col_id)
            if self._is_missing_sort_value(value):
                missing.append(result)
            else:
                present.append((self._sort_key(value), result))

        present.sort(key=lambda item: item[0], reverse=reverse)

        self.layoutAboutToBeChanged.emit()
        self.results = [result for _key, result in present] + missing
        self.layoutChanged.emit()
        if self.results:
            last_row = len(self.results) - 1
            last_col = max(0, len(cols) - 1)
            self.headerDataChanged.emit(Qt.Orientation.Vertical, 0, last_row)
            self.dataChanged.emit(self.index(0, 0), self.index(last_row, last_col))

def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def trapezoid_score(value: float, low_bad: float, low_good: float, high_good: float, high_bad: float) -> float:
    if value <= low_bad or value >= high_bad:
        return 0.0
    if low_good <= value <= high_good:
        return 1.0
    if value < low_good:
        return clamp01((value - low_bad) / max(low_good - low_bad, 1e-9))
    return clamp01((high_bad - value) / max(high_bad - high_good, 1e-9))


def normalized_weights(weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    raw = {key: max(0.0, float(weights.get(key, DEFAULT_WEIGHTS[key]))) for key, _ in WEIGHT_ORDER}
    total = float(sum(raw.values()))
    if total <= 0.0:
        raw = dict(DEFAULT_WEIGHTS)
        total = float(sum(raw.values()))
    normalized = {key: raw[key] / total for key, _ in WEIGHT_ORDER}
    return raw, normalized, total


def imread_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def luminance_uint8(image_bgr: np.ndarray) -> np.ndarray:
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(y, 0, 255).astype(np.uint8)


def score_crop(image_bgr: np.ndarray, weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    raw_weights, effective_weights, weight_total = normalized_weights(weights)
    h, w = image_bgr.shape[:2]
    area = float(max(1, w * h))
    gray = luminance_uint8(image_bgr)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = clamp01(
        (math.log1p(lap_var) - math.log1p(18.0)) / (math.log1p(900.0) - math.log1p(18.0))
    )

    p5, p95 = np.percentile(gray, [5, 95])
    contrast_range = float(p95 - p5)
    contrast_std = float(gray.std())
    contrast_score = 0.55 * clamp01((contrast_range - 35.0) / 150.0) + 0.45 * clamp01(
        (contrast_std - 12.0) / 58.0
    )

    median = float(np.median(gray))
    lower = int(max(12, (1.0 - 0.33) * median))
    upper = int(min(255, max(45, (1.0 + 0.33) * median)))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = float(np.count_nonzero(edges) / area)
    edge_density_score = trapezoid_score(edge_density, 0.004, 0.025, 0.145, 0.310)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_fraction = float(np.mean(otsu == 0))
    foreground_mask = (otsu == 0) if dark_fraction <= 0.5 else (otsu == 255)
    foreground = foreground_mask.astype(np.uint8)
    foreground_fraction = float(foreground.mean())
    foreground_score = trapezoid_score(foreground_fraction, 0.008, 0.040, 0.360, 0.670)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    valid_components = 0
    total_valid_area = 0
    min_component_area = max(5, int(area * 0.000006))
    max_component_area = max(min_component_area + 1, int(area * 0.085))

    for i in range(1, n_labels):
        x, y, cw, ch, c_area = stats[i]
        if c_area < min_component_area or c_area > max_component_area:
            continue
        if cw < 2 or ch < 2:
            continue
        if cw > 0.90 * w or ch > 0.45 * h:
            continue
        valid_components += 1
        total_valid_area += int(c_area)

    component_density = valid_components / max(area / 100000.0, 1e-9)
    component_density_score = trapezoid_score(component_density, 0.20, 3.0, 90.0, 230.0)
    component_count_score = clamp01(math.log1p(valid_components) / math.log1p(85.0))
    component_score = 0.55 * component_count_score + 0.45 * component_density_score

    short_side = float(min(w, h))
    long_side = float(max(w, h))
    resolution_score = 0.45 * clamp01((short_side - 90.0) / 310.0) + 0.55 * clamp01(
        (long_side - 260.0) / 740.0
    )

    border_region_fraction = max(0.0, float(weights.get("border_region_percent", DEFAULT_OCR_REDUCER_SETTINGS["border_region_percent"]))) / 100.0
    border_penalty_max = max(0.0, float(weights.get("border_penalty_max_percent", DEFAULT_OCR_REDUCER_SETTINGS["border_penalty_max_percent"]))) / 100.0
    border_ratio_start = float(weights.get("border_ratio_start", DEFAULT_OCR_REDUCER_SETTINGS["border_ratio_start"]))
    border_ratio_range = max(1e-9, float(weights.get("border_ratio_range", DEFAULT_OCR_REDUCER_SETTINGS["border_ratio_range"])))

    border_x = max(1, int(round(w * border_region_fraction)))
    border_y = max(1, int(round(h * border_region_fraction)))
    border_mask = np.zeros_like(foreground, dtype=bool)
    border_mask[:border_y, :] = True
    border_mask[-border_y:, :] = True
    border_mask[:, :border_x] = True
    border_mask[:, -border_x:] = True
    border_foreground = float(foreground[border_mask].mean()) if np.any(border_mask) else 0.0
    center_mask = ~border_mask
    center_foreground = float(foreground[center_mask].mean()) if np.any(center_mask) else 0.0
    border_ratio = border_foreground / max(center_foreground, 1e-6)
    border_cut_penalty = 1.0 - border_penalty_max * clamp01((border_ratio - border_ratio_start) / border_ratio_range)

    saturation_penalty_max = max(0.0, float(weights.get("saturation_penalty_max_percent", DEFAULT_OCR_REDUCER_SETTINGS["saturation_penalty_max_percent"]))) / 100.0
    saturation_start = max(0.0, float(weights.get("saturation_start_percent", DEFAULT_OCR_REDUCER_SETTINGS["saturation_start_percent"]))) / 100.0
    saturation_range = max(1e-9, float(weights.get("saturation_range_percent", DEFAULT_OCR_REDUCER_SETTINGS["saturation_range_percent"]))) / 100.0
    saturation_fraction = float(np.mean((gray <= 2) | (gray >= 253)))
    saturation_penalty = 1.0 - saturation_penalty_max * clamp01((saturation_fraction - saturation_start) / saturation_range)

    weighted_base = (
        effective_weights["sharpness"] * sharpness_score
        + effective_weights["contrast"] * contrast_score
        + effective_weights["edge_density"] * edge_density_score
        + effective_weights["foreground"] * foreground_score
        + effective_weights["components"] * component_score
        + effective_weights["resolution"] * resolution_score
    )
    final_score = 100.0 * weighted_base * border_cut_penalty * saturation_penalty

    metrics = {
        "score_raw_0_to_1": weighted_base,
        "weight_total_raw_percent": weight_total,
        "weight_sharpness_raw_percent": raw_weights["sharpness"],
        "weight_contrast_raw_percent": raw_weights["contrast"],
        "weight_edge_density_raw_percent": raw_weights["edge_density"],
        "weight_foreground_raw_percent": raw_weights["foreground"],
        "weight_components_raw_percent": raw_weights["components"],
        "weight_resolution_raw_percent": raw_weights["resolution"],
        "weight_sharpness_percent": 100.0 * effective_weights["sharpness"],
        "weight_contrast_percent": 100.0 * effective_weights["contrast"],
        "weight_edge_density_percent": 100.0 * effective_weights["edge_density"],
        "weight_foreground_percent": 100.0 * effective_weights["foreground"],
        "weight_components_percent": 100.0 * effective_weights["components"],
        "weight_resolution_percent": 100.0 * effective_weights["resolution"],
        "sharpness_score": sharpness_score,
        "contrast_score": contrast_score,
        "edge_density_score": edge_density_score,
        "foreground_score": foreground_score,
        "component_score": component_score,
        "component_density_score": component_density_score,
        "component_count_score": component_count_score,
        "resolution_score": resolution_score,
        "border_cut_penalty": border_cut_penalty,
        "saturation_penalty": saturation_penalty,
        "border_region_percent": 100.0 * border_region_fraction,
        "border_penalty_max_percent": 100.0 * border_penalty_max,
        "border_ratio_start": border_ratio_start,
        "border_ratio_range": border_ratio_range,
        "saturation_penalty_max_percent": 100.0 * saturation_penalty_max,
        "saturation_start_percent": 100.0 * saturation_start,
        "saturation_range_percent": 100.0 * saturation_range,
        "laplacian_variance": lap_var,
        "contrast_p95_minus_p5": contrast_range,
        "contrast_std": contrast_std,
        "edge_density": edge_density,
        "foreground_fraction": foreground_fraction,
        "valid_components": float(valid_components),
        "valid_component_area_fraction": float(total_valid_area / area),
        "component_density_per_100k_px": component_density,
        "border_to_center_foreground_ratio": border_ratio,
        "saturation_fraction": saturation_fraction,
    }
    return float(final_score), metrics


class CraftHeatmapScorer:
    _reader_cache: Dict[bool, Tuple[Any, Any]] = {}

    def __init__(self, settings: Dict[str, float]) -> None:
        self.settings = dict(settings)
        self.reader = None
        self.device = None
        self._reader_cache_key: Optional[bool] = None
        self.last_reader_load_seconds = 0.0
        self.last_reader_cache_hit = False
        self.last_reader_source = "not loaded"
        self.reader_load_seconds_total = 0.0
        self.reader_load_count = 0
        self.reader_cache_hit_count = 0
        self.last_heatmap_timing: Dict[str, Any] = {}
        self.last_score_timing: Dict[str, Any] = {}

    @staticmethod
    def reader_cache_key(settings: Dict[str, float]) -> bool:
        return bool(round(float(settings.get("gpu", DEFAULT_CRAFT_HEATMAP_SETTINGS["gpu"]))))

    @classmethod
    def clear_reader_cache(cls) -> None:
        cls._reader_cache.clear()

    def update_settings(self, settings: Dict[str, float]) -> None:
        old_key = self.reader_cache_key(self.settings)
        new_key = self.reader_cache_key(settings)
        self.settings = dict(settings)
        if self.reader is not None and old_key != new_key:
            self.reader = None
            self.device = None
            self._reader_cache_key = None

    def _ensure_reader(self):
        use_gpu = self.reader_cache_key(self.settings)
        load_start = time.perf_counter()
        self.last_reader_load_seconds = 0.0
        self.last_reader_cache_hit = False
        self.last_reader_source = "already loaded"

        if self.reader is not None and self._reader_cache_key == use_gpu:
            self.last_reader_cache_hit = True
            self.reader_cache_hit_count += 1
            self.last_reader_load_seconds = elapsed_since(load_start)
            return self.reader

        cached = self._reader_cache.get(use_gpu)
        if cached is not None:
            self.reader, self.device = cached
            self._reader_cache_key = use_gpu
            self.last_reader_cache_hit = True
            self.reader_cache_hit_count += 1
            self.last_reader_source = "class cache"
            self.last_reader_load_seconds = elapsed_since(load_start)
            return self.reader

        self.last_reader_source = "loaded EasyOCR CRAFT"
        import torch
        import easyocr

        reader = easyocr.Reader(["en"], gpu=use_gpu, detector=True, recognizer=False, verbose=False)
        device = getattr(reader, "device", "cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self._reader_cache[use_gpu] = (reader, device)
        self.reader = reader
        self.device = device
        self._reader_cache_key = use_gpu
        self.last_reader_load_seconds = elapsed_since(load_start)
        self.reader_load_seconds_total += self.last_reader_load_seconds
        self.reader_load_count += 1
        return self.reader

    def heatmaps(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        from easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio

        total_start = time.perf_counter()
        ensure_start = time.perf_counter()
        reader = self._ensure_reader()
        ensure_seconds = elapsed_since(ensure_start)

        preprocess_start = time.perf_counter()
        detector = reader.detector
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        canvas_size = int(round(float(self.settings.get("canvas_size", DEFAULT_CRAFT_HEATMAP_SETTINGS["canvas_size"]))))
        mag_ratio = float(self.settings.get("mag_ratio", DEFAULT_CRAFT_HEATMAP_SETTINGS["mag_ratio"]))

        image_resized, _target_ratio, _ = resize_aspect_ratio(
            image_rgb,
            canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=mag_ratio,
        )
        x = normalizeMeanVariance(image_resized)
        x = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(self.device)
        preprocess_seconds = elapsed_since(preprocess_start)

        forward_start = time.perf_counter()
        detector.eval()
        with torch.no_grad():
            y, _feature = detector(x)
        forward_seconds = elapsed_since(forward_start)

        transfer_start = time.perf_counter()
        score_text = y[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
        score_link = y[0, :, :, 1].detach().cpu().numpy().astype(np.float32)
        transfer_seconds = elapsed_since(transfer_start)

        self.last_heatmap_timing = {
            "craft_reader_ensure_seconds": ensure_seconds,
            "craft_reader_load_seconds": self.last_reader_load_seconds,
            "craft_reader_cache_hit": self.last_reader_cache_hit,
            "craft_reader_source": self.last_reader_source,
            "craft_preprocess_seconds": preprocess_seconds,
            "craft_forward_seconds": forward_seconds,
            "craft_transfer_seconds": transfer_seconds,
            "craft_heatmap_total_seconds": elapsed_since(total_start),
            "craft_device": str(self.device),
            "craft_resized_width": int(image_resized.shape[1]) if image_resized.ndim >= 2 else None,
            "craft_resized_height": int(image_resized.shape[0]) if image_resized.ndim >= 2 else None,
        }
        return score_text, score_link


def score_craft_heatmap_sum(
    image_bgr: np.ndarray,
    settings: Dict[str, float],
    scorer: Optional[CraftHeatmapScorer],
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    score_total_start = time.perf_counter()
    if scorer is None:
        scorer = CraftHeatmapScorer(settings)
    heatmap_start = time.perf_counter()
    score_text, score_link = scorer.heatmaps(image_bgr)
    heatmap_seconds = elapsed_since(heatmap_start)
    post_start = time.perf_counter()
    score_text = np.nan_to_num(score_text, nan=0.0, posinf=1.0, neginf=0.0)
    score_link = np.nan_to_num(score_link, nan=0.0, posinf=1.0, neginf=0.0)

    text_threshold = float(settings.get("text_threshold", DEFAULT_CRAFT_HEATMAP_SETTINGS["text_threshold"]))
    low_text = float(settings.get("low_text", DEFAULT_CRAFT_HEATMAP_SETTINGS["low_text"]))
    link_threshold = float(settings.get("link_threshold", DEFAULT_CRAFT_HEATMAP_SETTINGS["link_threshold"]))

    text_density_raw, weak_text_area_raw, weak_text_area_uncapped_raw, weak_component_count, weak_components_capped = capped_heatmap_energy_and_area(
        score_text,
        low_text,
        cap_fraction=0.006,
    )
    _strong_energy_raw, strong_text_area_raw, strong_text_area_uncapped_raw, strong_component_count, strong_components_capped = capped_heatmap_energy_and_area(
        score_text,
        text_threshold,
        cap_fraction=0.004,
    )
    affinity_density_raw, affinity_area_raw, affinity_area_uncapped_raw, affinity_component_count, affinity_components_capped = capped_heatmap_energy_and_area(
        score_link,
        link_threshold,
        cap_fraction=0.006,
    )
    peak_text = float(np.max(score_text)) if score_text.size else 0.0
    peak_affinity = float(np.max(score_link)) if score_link.size else 0.0

    text_density_good = max(1e-6, float(settings.get("text_density_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["text_density_good_percent"]))) / 100.0
    affinity_density_good = max(1e-6, float(settings.get("affinity_density_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["affinity_density_good_percent"]))) / 100.0
    weak_area_good = max(1e-6, float(settings.get("weak_text_area_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["weak_text_area_good_percent"]))) / 100.0
    strong_area_good = max(1e-6, float(settings.get("strong_text_area_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["strong_text_area_good_percent"]))) / 100.0

    text_density_score = clamp01(text_density_raw / text_density_good)
    affinity_density_score = clamp01(affinity_density_raw / affinity_density_good)
    weak_text_area_score = clamp01(weak_text_area_raw / weak_area_good)
    strong_text_area_score = clamp01(strong_text_area_raw / strong_area_good)
    peak_text_score = clamp01(peak_text)
    peak_affinity_score = clamp01(peak_affinity)

    raw_craft_weights, effective_craft_weights, craft_weight_total = normalized_craft_heatmap_weights(settings)

    final_score = 100.0 * (
        effective_craft_weights["craft_weight_text_sum_percent"] * text_density_score
        + effective_craft_weights["craft_weight_affinity_sum_percent"] * affinity_density_score
        + effective_craft_weights["craft_weight_weak_area_percent"] * weak_text_area_score
        + effective_craft_weights["craft_weight_strong_area_percent"] * strong_text_area_score
        + effective_craft_weights["craft_weight_peak_text_percent"] * peak_text_score
        + effective_craft_weights["craft_weight_peak_affinity_percent"] * peak_affinity_score
    )

    metrics: Dict[str, Any] = {
        "craft_heatmap_score": float(final_score),
        "craft_weight_total_raw_percent": craft_weight_total,
        "craft_weight_text_sum_raw_percent": raw_craft_weights["craft_weight_text_sum_percent"],
        "craft_weight_affinity_sum_raw_percent": raw_craft_weights["craft_weight_affinity_sum_percent"],
        "craft_weight_weak_area_raw_percent": raw_craft_weights["craft_weight_weak_area_percent"],
        "craft_weight_strong_area_raw_percent": raw_craft_weights["craft_weight_strong_area_percent"],
        "craft_weight_peak_text_raw_percent": raw_craft_weights["craft_weight_peak_text_percent"],
        "craft_weight_peak_affinity_raw_percent": raw_craft_weights["craft_weight_peak_affinity_percent"],
        "craft_weight_text_sum_percent": 100.0 * effective_craft_weights["craft_weight_text_sum_percent"],
        "craft_weight_affinity_sum_percent": 100.0 * effective_craft_weights["craft_weight_affinity_sum_percent"],
        "craft_weight_weak_area_percent": 100.0 * effective_craft_weights["craft_weight_weak_area_percent"],
        "craft_weight_strong_area_percent": 100.0 * effective_craft_weights["craft_weight_strong_area_percent"],
        "craft_weight_peak_text_percent": 100.0 * effective_craft_weights["craft_weight_peak_text_percent"],
        "craft_weight_peak_affinity_percent": 100.0 * effective_craft_weights["craft_weight_peak_affinity_percent"],
        "craft_text_density_raw": text_density_raw,
        "craft_affinity_density_raw": affinity_density_raw,
        "craft_weak_text_area_raw": weak_text_area_raw,
        "craft_strong_text_area_raw": strong_text_area_raw,
        "craft_affinity_area_raw": affinity_area_raw,
        "craft_weak_text_area_uncapped_raw": weak_text_area_uncapped_raw,
        "craft_strong_text_area_uncapped_raw": strong_text_area_uncapped_raw,
        "craft_affinity_area_uncapped_raw": affinity_area_uncapped_raw,
        "craft_weak_component_count": weak_component_count,
        "craft_strong_component_count": strong_component_count,
        "craft_affinity_component_count": affinity_component_count,
        "craft_weak_components_capped": weak_components_capped,
        "craft_strong_components_capped": strong_components_capped,
        "craft_affinity_components_capped": affinity_components_capped,
        "craft_component_cap_note": "Existing heatmap-sum terms use per-component caps to reduce large-letter area bias.",
        "craft_text_density_score": text_density_score,
        "craft_affinity_density_score": affinity_density_score,
        "craft_weak_text_area_score": weak_text_area_score,
        "craft_strong_text_area_score": strong_text_area_score,
        "craft_peak_text_score": peak_text_score,
        "craft_peak_affinity_score": peak_affinity_score,
        "craft_peak_text": peak_text,
        "craft_peak_affinity": peak_affinity,
        "craft_heatmap_width": int(score_text.shape[1]) if score_text.ndim >= 2 else None,
        "craft_heatmap_height": int(score_text.shape[0]) if score_text.ndim >= 2 else None,
        "craft_text_threshold": text_threshold,
        "craft_low_text": low_text,
        "craft_link_threshold": link_threshold,
        "craft_canvas_size": int(round(float(settings.get("canvas_size", DEFAULT_CRAFT_HEATMAP_SETTINGS["canvas_size"])))),
        "craft_mag_ratio": float(settings.get("mag_ratio", DEFAULT_CRAFT_HEATMAP_SETTINGS["mag_ratio"])),
        "craft_text_density_good_percent": 100.0 * text_density_good,
        "craft_affinity_density_good_percent": 100.0 * affinity_density_good,
        "craft_weak_text_area_good_percent": 100.0 * weak_area_good,
        "craft_strong_text_area_good_percent": 100.0 * strong_area_good,
    }
    scorer.last_score_timing = dict(scorer.last_heatmap_timing)
    scorer.last_score_timing.update({
        "craft_score_postprocess_seconds": elapsed_since(post_start),
        "craft_score_function_seconds": elapsed_since(score_total_start),
        "craft_heatmaps_call_seconds": heatmap_seconds,
    })
    return float(final_score), metrics, score_text.copy(), score_link.copy()



def optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_jsonish_list(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        parts = [part.strip() for part in text.strip("[]()").split(",")]
        try:
            return [float(part) for part in parts if part]
        except ValueError:
            return None
    return None


def infer_crop_number(path: Path, metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if metadata:
        for key in ("candidate_index", "crop_number", "index"):
            value = optional_int(metadata.get(key))
            if value is not None:
                return value

    stem = path.stem
    for pattern in (r"(?:candidate|crop)[_-]?(\d+)", r"(?:cand|c)[_-]?(\d+)"):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return optional_int(match.group(1))

    match = re.search(r"(\d+)", stem)
    return optional_int(match.group(1)) if match else None


def safe_resolve(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)


class YoloMetadataIndex:
    def __init__(self) -> None:
        self.by_exact_path: Dict[str, Dict[str, Any]] = {}
        self.by_track_and_file: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.by_filename: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata_files: set[str] = set()

    @property
    def count(self) -> int:
        seen = set()
        for items in self.by_filename.values():
            for item in items:
                seen.add(id(item))
        return len(seen)

    def add_candidate(self, item: Dict[str, Any], path_options: Sequence[Path], track_dir_name: str, crop_filename: str) -> None:
        if not crop_filename:
            return
        if track_dir_name:
            self.by_track_and_file.setdefault((track_dir_name, crop_filename), []).append(item)
        self.by_filename.setdefault(crop_filename, []).append(item)
        for option in path_options:
            if option:
                self.by_exact_path[safe_resolve(option)] = item

    def lookup(self, image_path: Path) -> Optional[Dict[str, Any]]:
        exact = self.by_exact_path.get(safe_resolve(image_path))
        if exact is not None:
            return exact

        track_names = []
        if image_path.parent.name.lower() == "crops":
            track_names.append(image_path.parent.parent.name)
        track_names.append(image_path.parent.name)

        for track_name in track_names:
            matches = self.by_track_and_file.get((track_name, image_path.name), [])
            unique = unique_metadata_matches(matches)
            if len(unique) == 1:
                return unique[0]
            if unique:
                return unique[0]

        matches = self.by_filename.get(image_path.name, [])
        unique = unique_metadata_matches(matches)
        if len(unique) == 1:
            return unique[0]
        if unique:
            return unique[0]
        return None


def unique_metadata_matches(matches: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for item in matches:
        key = (
            item.get("track_id"),
            item.get("candidate_index"),
            item.get("crop_filename") or Path(str(item.get("crop_path", ""))).name,
        )
        unique[key] = item
    return list(unique.values())


def metadata_search_roots(folder: Path) -> List[Path]:
    roots = [folder]
    name = folder.name.lower()
    if name == "crops":
        roots.append(folder.parent)
        roots.append(folder.parent.parent)
    elif name.startswith("track_"):
        roots.append(folder.parent)
    unique: List[Path] = []
    seen = set()
    for root in roots:
        if root and root.exists():
            key = safe_resolve(root)
            if key not in seen:
                unique.append(root)
                seen.add(key)
    return unique


def resolve_track_dir_from_record(record: Dict[str, Any], root_dir: Path) -> Tuple[Path, str]:
    raw_track_dir = record.get("track_directory")
    track_dir_name = str(record.get("track_directory_name") or "")
    candidates: List[Path] = []
    if raw_track_dir:
        p = Path(str(raw_track_dir))
        if p.is_absolute():
            candidates.append(p)
            if not track_dir_name:
                track_dir_name = p.name
        else:
            candidates.append(root_dir / p)
            if not track_dir_name:
                track_dir_name = p.name
    if track_dir_name:
        candidates.append(root_dir / track_dir_name)
    track_id = record.get("track_id", record.get("id", "unknown"))
    label = str(record.get("label", "object")).strip().replace(" ", "_") or "object"
    fallback_name = f"track_{optional_int(track_id, -1) or -1:05d}_{label}" if optional_int(track_id) is not None else f"track_{track_id}_{label}"
    candidates.append(root_dir / fallback_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.name
    return candidates[0], candidates[0].name


def candidate_path_options(root_dir: Path, track_dir: Path, crop_path_value: Any, crop_filename: str) -> List[Path]:
    options: List[Path] = []
    if crop_path_value:
        p = Path(str(crop_path_value))
        if p.is_absolute():
            options.append(p)
        else:
            options.extend([track_dir / p, root_dir / p])
            if track_dir.name and len(p.parts) >= 2 and p.parts[0] == track_dir.name:
                options.append(root_dir / p)
    if crop_filename:
        options.extend([
            track_dir / "crops" / crop_filename,
            track_dir / crop_filename,
            root_dir / crop_filename,
        ])
    deduped: List[Path] = []
    seen = set()
    for option in options:
        key = str(option)
        if key not in seen:
            deduped.append(option)
            seen.add(key)
    return deduped


def add_yolo_candidate_to_index(
    index: YoloMetadataIndex,
    candidate: Dict[str, Any],
    *,
    metadata_path: Path,
    root_dir: Path,
    track_dir: Path,
    track_dir_name: str,
    frame_width: Optional[int],
    frame_height: Optional[int],
    record: Optional[Dict[str, Any]] = None,
) -> None:
    crop_filename = str(candidate.get("crop_filename") or Path(str(candidate.get("crop_path", ""))).name)
    if not crop_filename or crop_filename == ".":
        return

    item = dict(candidate)
    if record:
        item.setdefault("track_id", record.get("track_id", record.get("id")))
        item.setdefault("object_id", record.get("id", record.get("track_id")))
        item.setdefault("label", record.get("label", ""))
        item.setdefault("cls_id", record.get("cls_id"))
        item.setdefault("ready_for_ocr", record.get("ready_for_ocr"))
        item.setdefault("seen_count", record.get("seen_count"))
        item.setdefault("first_frame", record.get("first_frame"))
        item.setdefault("last_seen_frame", record.get("last_seen_frame"))
    item.setdefault("detector_score", item.get("score"))
    item["crop_filename"] = crop_filename
    item["_metadata_path"] = str(metadata_path)
    item["_root_directory"] = str(root_dir)
    item["_track_directory"] = str(track_dir)
    item["_track_directory_name"] = track_dir_name
    item["_frame_width"] = frame_width
    item["_frame_height"] = frame_height

    path_options = candidate_path_options(root_dir, track_dir, item.get("crop_path"), crop_filename)
    index.add_candidate(item, path_options, track_dir_name, crop_filename)


def load_yolo_metadata_index(folder: Path) -> YoloMetadataIndex:
    index = YoloMetadataIndex()
    json_paths: List[Path] = []
    csv_paths: List[Path] = []
    for root in metadata_search_roots(folder):
        json_paths.extend(root.rglob("detector_score_metadata.json"))
        csv_paths.extend(root.rglob("detector_score_metadata.csv"))

    seen_json = set()
    for metadata_path in sorted(json_paths, key=lambda p: str(p).lower()):
        key = safe_resolve(metadata_path)
        if key in seen_json:
            continue
        seen_json.add(key)
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        index.metadata_files.add(str(metadata_path))
        root_dir = metadata_path.parent
        frame_size = payload.get("frame_size", {}) if isinstance(payload, dict) else {}
        frame_width = optional_int(frame_size.get("width"))
        frame_height = optional_int(frame_size.get("height"))

        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            for record in payload["records"]:
                if not isinstance(record, dict):
                    continue
                track_dir, track_dir_name = resolve_track_dir_from_record(record, root_dir)
                for candidate in record.get("all_crops", []):
                    if isinstance(candidate, dict):
                        add_yolo_candidate_to_index(
                            index,
                            candidate,
                            metadata_path=metadata_path,
                            root_dir=root_dir,
                            track_dir=track_dir,
                            track_dir_name=track_dir_name,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            record=record,
                        )

        if isinstance(payload, dict) and isinstance(payload.get("track"), dict):
            record = payload["track"]
            track_dir = metadata_path.parent
            track_dir_name = track_dir.name
            for candidate in record.get("all_crops", []):
                if isinstance(candidate, dict):
                    add_yolo_candidate_to_index(
                        index,
                        candidate,
                        metadata_path=metadata_path,
                        root_dir=track_dir,
                        track_dir=track_dir,
                        track_dir_name=track_dir_name,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        record=record,
                    )

    seen_csv = set()
    for csv_path in sorted(csv_paths, key=lambda p: str(p).lower()):
        key = safe_resolve(csv_path)
        if key in seen_csv:
            continue
        seen_csv.add(key)
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        index.metadata_files.add(str(csv_path))
        root_dir = csv_path.parent
        for row in rows:
            if not row:
                continue
            crop_filename = row.get("crop_filename") or Path(str(row.get("crop_path", ""))).name
            if not crop_filename:
                continue
            track_dir_value = row.get("track_directory")
            if track_dir_value:
                track_dir = Path(track_dir_value)
                if not track_dir.is_absolute():
                    track_dir = root_dir / track_dir
            else:
                crop_path = Path(str(row.get("crop_path", "")))
                if len(crop_path.parts) >= 2 and crop_path.parts[-2] == "crops":
                    track_dir = root_dir / Path(*crop_path.parts[:-2])
                else:
                    track_dir = root_dir
            track_dir_name = track_dir.name
            add_yolo_candidate_to_index(
                index,
                dict(row),
                metadata_path=csv_path,
                root_dir=root_dir,
                track_dir=track_dir,
                track_dir_name=track_dir_name,
                frame_width=None,
                frame_height=None,
                record=None,
            )
    return index


def yolo_coefficients(weights: Dict[str, float]) -> Dict[str, float]:
    return {
        key: max(0.0, float(weights.get(key, DEFAULT_YOLO_WEIGHTS[key]))) / 100.0
        for key, _ in YOLO_WEIGHT_ORDER
    }


def score_yolo_candidate_from_metadata(
    image_bgr: np.ndarray,
    metadata: Dict[str, Any],
    yolo_weights: Dict[str, float],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Freshly recompute yolo_engine.crop_quality() from crop pixels + saved metadata.

    This intentionally does not use the OCR-proxy score and does not trust the
    saved metadata quality as the ranking value. The saved quality is shown only
    as a comparison/debug field.
    """
    detector_score = optional_float(metadata.get("detector_score", metadata.get("score")))
    bbox = parse_jsonish_list(metadata.get("bbox"))
    raw_bbox = parse_jsonish_list(metadata.get("raw_bbox")) or []
    saved_quality = optional_float(metadata.get("quality"))
    frame_width = optional_int(metadata.get("_frame_width"))
    frame_height = optional_int(metadata.get("_frame_height"))

    h, w = image_bgr.shape[:2]
    if detector_score is None:
        return None, {
            "yolo_metadata_found": True,
            "yolo_score_available": False,
            "yolo_saved_quality_0_to_1": saved_quality,
            "yolo_score_note": "YOLO metadata was found, but detector_score/score was missing.",
        }

    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    else:
        x1, y1, x2, y2 = 0.0, 0.0, float(w), float(h)

    if frame_width is None or frame_width <= 0:
        explicit_width = optional_int(metadata.get("frame_width"))
        frame_width = explicit_width if explicit_width is not None and explicit_width > 0 else max(w, int(math.ceil(x2 + 3)))
    if frame_height is None or frame_height <= 0:
        explicit_height = optional_int(metadata.get("frame_height"))
        frame_height = explicit_height if explicit_height is not None and explicit_height > 0 else max(h, int(math.ceil(y2 + 3)))

                                                                                
                                              
    ch, cw = image_bgr.shape[:2]
    if cw < 28 or ch < 28:
        yolo_quality = -1.0
        lap_var = 0.0
        sharp_score = 0.0
        area_score = 0.0
        aspect_score = 0.0
        edge_penalty = 0.0
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp_score = min(lap_var / 700.0, 1.0)
        area_score = min((cw * ch) / max(1.0, 0.20 * frame_width * frame_height), 1.0)
        aspect_score = 1.0 if 0.22 <= cw / max(1, ch) <= 7.0 else 0.45
        coeff = yolo_coefficients(yolo_weights)
        edge_penalty = coeff["edge_penalty"] if x1 <= 2 or y1 <= 2 or x2 >= frame_width - 2 or y2 >= frame_height - 2 else 0.0
        yolo_quality = (
            coeff["detector"] * float(detector_score)
            + coeff["sharpness"] * sharp_score
            + coeff["area"] * area_score
            + coeff["aspect"] * aspect_score
            - edge_penalty
        )

    details: Dict[str, Any] = {
        "yolo_metadata_found": True,
        "yolo_score_available": True,
        "yolo_quality_score_0_to_1": float(yolo_quality),
        "yolo_quality_score_0_to_100": float(100.0 * yolo_quality),
        "yolo_saved_quality_0_to_1": saved_quality,
        "yolo_saved_quality_0_to_100": None if saved_quality is None else 100.0 * saved_quality,
        "yolo_quality_abs_diff_from_saved": None if saved_quality is None else abs(float(yolo_quality) - float(saved_quality)),
        "yolo_detector_score": float(detector_score),
        "yolo_laplacian_variance": lap_var,
        "yolo_sharpness_score": float(sharp_score),
        "yolo_area_score": float(area_score),
        "yolo_aspect_score": float(aspect_score),
        "yolo_edge_penalty": float(edge_penalty),
        "yolo_crop_width_from_pixels": int(cw),
        "yolo_crop_height_from_pixels": int(ch),
        "yolo_crop_width_from_bbox": float(max(0.0, x2 - x1)),
        "yolo_crop_height_from_bbox": float(max(0.0, y2 - y1)),
        "yolo_frame_width": int(frame_width),
        "yolo_frame_height": int(frame_height),
        "yolo_bbox": json.dumps(bbox if bbox is not None else [x1, y1, x2, y2]),
        "yolo_raw_bbox": json.dumps(raw_bbox),
        "yolo_track_id": optional_int(metadata.get("track_id", metadata.get("object_id"))),
        "yolo_candidate_index": optional_int(metadata.get("candidate_index")),
        "yolo_frame_index": optional_int(metadata.get("frame_index")),
        "yolo_is_best": bool(str(metadata.get("is_best", "")).lower() in {"true", "1", "yes"} or metadata.get("is_best") is True),
        "yolo_label": metadata.get("label", ""),
        "yolo_metadata_path": metadata.get("_metadata_path", ""),
        "yolo_track_directory": metadata.get("_track_directory", ""),
        "yolo_crop_path_in_metadata": metadata.get("crop_path", ""),
        "yolo_weight_detector_percent": float(yolo_weights.get("detector", DEFAULT_YOLO_WEIGHTS["detector"])),
        "yolo_weight_sharpness_percent": float(yolo_weights.get("sharpness", DEFAULT_YOLO_WEIGHTS["sharpness"])),
        "yolo_weight_area_percent": float(yolo_weights.get("area", DEFAULT_YOLO_WEIGHTS["area"])),
        "yolo_weight_aspect_percent": float(yolo_weights.get("aspect", DEFAULT_YOLO_WEIGHTS["aspect"])),
        "yolo_edge_penalty_amount_percent": float(yolo_weights.get("edge_penalty", DEFAULT_YOLO_WEIGHTS["edge_penalty"])),
    }
    return float(yolo_quality), details


def score_path(
    path: Path,
    ocr_weights: Dict[str, float],
    yolo_weights: Dict[str, float],
    craft_settings: Dict[str, float],
    ranking_method: str,
    yolo_metadata: Optional[YoloMetadataIndex] = None,
    craft_scorer: Optional[CraftHeatmapScorer] = None,
) -> Optional[CropScoreResult]:
    total_start = time.perf_counter()
    timing: Dict[str, Any] = {"file": path.name}

    read_start = time.perf_counter()
    image = imread_bgr(path)
    timing["image_read_seconds"] = elapsed_since(read_start)
    if image is None or image.ndim != 3:
        timing["score_path_seconds"] = elapsed_since(total_start)
        return None

    h, w = image.shape[:2]
    metrics: Dict[str, Any] = {"ranking_method": ranking_method, "crop_number": infer_crop_number(path)}
    craft_text_heatmap: Optional[np.ndarray] = None
    craft_affinity_heatmap: Optional[np.ndarray] = None

    if ranking_method == "yolo":
        lookup_start = time.perf_counter()
        yolo_item = yolo_metadata.lookup(path) if yolo_metadata is not None else None
        timing["yolo_metadata_lookup_seconds"] = elapsed_since(lookup_start)
        score_start = time.perf_counter()
        if yolo_item is not None:
            yolo_score, yolo_metrics = score_yolo_candidate_from_metadata(image, yolo_item, yolo_weights)
            metrics.update(yolo_metrics)
            if metrics.get("crop_number") is None:
                metrics["crop_number"] = infer_crop_number(path, yolo_item) or metrics.get("yolo_candidate_index")
        else:
            yolo_score = None
            metrics.update({
                "yolo_metadata_found": False,
                "yolo_score_available": False,
                "yolo_quality_score_0_to_1": None,
                "yolo_quality_score_0_to_100": None,
                "yolo_detector_score": None,
                "yolo_track_id": None,
                "yolo_candidate_index": None,
                "yolo_frame_index": None,
                "yolo_is_best": False,
                "yolo_score_note": "No YOLO metadata matched this crop image.",
            })
        timing["yolo_score_seconds"] = elapsed_since(score_start)
        rank_score = -1.0 if yolo_score is None else 100.0 * float(yolo_score)
    elif ranking_method == "craft":
        craft_start = time.perf_counter()
        craft_score, craft_metrics, craft_text_heatmap, craft_affinity_heatmap = score_craft_heatmap_sum(image, craft_settings, craft_scorer)
        timing["craft_total_seconds"] = elapsed_since(craft_start)
        if craft_scorer is not None:
            timing.update(craft_scorer.last_score_timing)
        metrics.update(craft_metrics)
        rank_score = float(craft_score)
    else:
        ocr_start = time.perf_counter()
        ocr_score, ocr_metrics = score_crop(image, ocr_weights)
        timing["ocr_proxy_score_seconds"] = elapsed_since(ocr_start)
        metrics.update(ocr_metrics)
        metrics["ocr_quality_score"] = float(ocr_score)
        rank_score = float(ocr_score)

    timing["score_path_seconds"] = elapsed_since(total_start)
    timing["track_key"] = track_key_for_path(path, metrics)

    return CropScoreResult(
        path=path,
        score=rank_score,
        width=w,
        height=h,
        metrics=metrics,
        craft_text_heatmap=craft_text_heatmap,
        craft_affinity_heatmap=craft_affinity_heatmap,
        timing=timing,
    )


def infer_frame_index_from_path_name(path: Path) -> Optional[int]:
    stem = path.stem
    patterns = (
        r"(?:frame|frm|f)[_-]?(\d+)",
        r"(?:global)[_-]?(?:frame)?[_-]?(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return optional_int(match.group(1))
    return None


def frame_index_for_crop_path(path: Path, yolo_metadata: Optional[YoloMetadataIndex]) -> Optional[int]:
    if yolo_metadata is not None:
        item = yolo_metadata.lookup(path)
        if item is not None:
            frame_index = optional_int(item.get("frame_index"))
            if frame_index is not None:
                return frame_index
    return infer_frame_index_from_path_name(path)


def filter_paths_by_frame_skip(
    paths: Sequence[Path],
    frame_skip: int,
    yolo_metadata: Optional[YoloMetadataIndex],
) -> Tuple[List[Path], Dict[str, int]]:
    """Keep all crops from every Nth detected frame.

    frame_skip=1 keeps everything. If a crop has no frame index in YOLO
    metadata or filename, it is kept so the filter does not accidentally hide
    crops from non-video/non-metadata folders.
    """
    frame_skip = max(1, int(frame_skip))
    paths = list(paths)
    if frame_skip <= 1 or not paths:
        return paths, {
            "frame_skip": frame_skip,
            "input_paths": len(paths),
            "kept_paths": len(paths),
            "known_frame_paths": 0,
            "unknown_frame_paths": len(paths),
            "unique_frames": 0,
            "kept_unique_frames": 0,
        }

    path_frames: List[Tuple[Path, Optional[int]]] = [
        (path, frame_index_for_crop_path(path, yolo_metadata)) for path in paths
    ]
    known_frames = sorted({frame for _path, frame in path_frames if frame is not None})
    if not known_frames:
        return paths, {
            "frame_skip": frame_skip,
            "input_paths": len(paths),
            "kept_paths": len(paths),
            "known_frame_paths": 0,
            "unknown_frame_paths": len(paths),
            "unique_frames": 0,
            "kept_unique_frames": 0,
        }

    kept_frames = set(known_frames[::frame_skip])
    kept_paths = [
        path for path, frame in path_frames
        if frame is None or frame in kept_frames
    ]
    known_frame_paths = sum(1 for _path, frame in path_frames if frame is not None)
    unknown_frame_paths = len(paths) - known_frame_paths
    return kept_paths, {
        "frame_skip": frame_skip,
        "input_paths": len(paths),
        "kept_paths": len(kept_paths),
        "known_frame_paths": known_frame_paths,
        "unknown_frame_paths": unknown_frame_paths,
        "unique_frames": len(known_frames),
        "kept_unique_frames": len(kept_frames),
    }


def find_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


def is_track_dir(path: Path) -> bool:
    return path.is_dir() and path.name.lower().startswith("track_")


def track_sort_key(path: Path) -> Tuple[int, str]:
    parts = path.name.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1]), path.name.lower()
        except ValueError:
            pass
    return 10**9, path.name.lower()


def find_track_dirs(run_folder: Path) -> List[Path]:
    if not run_folder.exists() or not run_folder.is_dir():
        return []
    return sorted([p for p in run_folder.iterdir() if is_track_dir(p)], key=track_sort_key)


def infer_run_and_selected_track(folder: Path) -> Tuple[Path, List[Path], Optional[Path]]:
    if folder.name.lower() == "crops" and is_track_dir(folder.parent):
        run_folder = folder.parent.parent
        track_dirs = find_track_dirs(run_folder)
        return run_folder, track_dirs, folder.parent
    if is_track_dir(folder):
        run_folder = folder.parent
        track_dirs = find_track_dirs(run_folder)
        return run_folder, track_dirs, folder
    track_dirs = find_track_dirs(folder)
    return folder, track_dirs, track_dirs[0] if track_dirs else None


def folder_for_track(track_dir: Path) -> Path:
    crops_dir = track_dir / "crops"
    return crops_dir if crops_dir.is_dir() else track_dir


def elapsed_since(start: float) -> float:
    return float(time.perf_counter() - start)


def format_seconds(seconds: Any) -> str:
    value = optional_float(seconds, 0.0) or 0.0
    if value < 0.001:
        return f"{value * 1000.0:.2f} ms"
    if value < 1.0:
        return f"{value * 1000.0:.1f} ms"
    if value < 60.0:
        return f"{value:.3f} s"
    minutes = int(value // 60)
    remainder = value - 60 * minutes
    return f"{minutes}m {remainder:.2f}s"


def track_key_for_path(path: Path, metadata: Optional[Dict[str, Any]] = None) -> str:
    if metadata:
        track_id = metadata.get("yolo_track_id", metadata.get("track_id", metadata.get("object_id")))
        if track_id is not None and track_id != "":
            return f"track {track_id}"
    if path.parent.name.lower() == "crops" and is_track_dir(path.parent.parent):
        return path.parent.parent.name
    if is_track_dir(path.parent):
        return path.parent.name
    return path.parent.name or "current directory"


def timing_stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0.0, "total": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(vals)),
        "total": float(sum(vals)),
        "avg": float(sum(vals) / len(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


class ScoreWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list, dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: Sequence[Path],
        ocr_weights: Dict[str, float],
        yolo_weights: Dict[str, float],
        craft_settings: Dict[str, float],
        ranking_method: str,
        yolo_metadata: Optional[YoloMetadataIndex],
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.ocr_weights = dict(ocr_weights)
        self.yolo_weights = dict(yolo_weights)
        self.craft_settings = dict(craft_settings)
        self.ranking_method = ranking_method
        self.yolo_metadata = yolo_metadata
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            run_start = time.perf_counter()
            results: List[CropScoreResult] = []
            total = len(self.paths)
            craft_scorer = None
            craft_setup_seconds = 0.0
            if self.ranking_method == "craft":
                setup_start = time.perf_counter()
                craft_scorer = CraftHeatmapScorer(self.craft_settings)
                craft_scorer.update_settings(self.craft_settings)
                craft_setup_seconds = elapsed_since(setup_start)

            track_totals: Dict[str, Dict[str, Any]] = {}
            image_read_seconds: List[float] = []
            crop_total_seconds: List[float] = []
            craft_total_seconds: List[float] = []
            craft_forward_seconds: List[float] = []
            craft_preprocess_seconds: List[float] = []
            craft_postprocess_seconds: List[float] = []
            ocr_proxy_seconds: List[float] = []
            yolo_score_seconds: List[float] = []

            for i, path in enumerate(self.paths, start=1):
                if self._cancelled:
                    return
                crop_start = time.perf_counter()
                result = score_path(path, self.ocr_weights, self.yolo_weights, self.craft_settings, self.ranking_method, self.yolo_metadata, craft_scorer)
                crop_seconds = elapsed_since(crop_start)
                if result is not None:
                    timing = dict(result.timing or {})
                    timing["crop_worker_total_seconds"] = crop_seconds
                    result = replace(result, timing=timing)
                    results.append(result)

                    track_key = str(timing.get("track_key") or track_key_for_path(path, result.metrics))
                    track = track_totals.setdefault(track_key, {"count": 0, "seconds": 0.0, "readable": 0})
                    track["count"] += 1
                    track["readable"] += 1
                    track["seconds"] += crop_seconds

                    image_read_seconds.append(float(timing.get("image_read_seconds", 0.0)))
                    crop_total_seconds.append(crop_seconds)
                    if "craft_total_seconds" in timing:
                        craft_total_seconds.append(float(timing.get("craft_total_seconds", 0.0)))
                    if "craft_forward_seconds" in timing:
                        craft_forward_seconds.append(float(timing.get("craft_forward_seconds", 0.0)))
                    if "craft_preprocess_seconds" in timing:
                        craft_preprocess_seconds.append(float(timing.get("craft_preprocess_seconds", 0.0)))
                    if "craft_score_postprocess_seconds" in timing:
                        craft_postprocess_seconds.append(float(timing.get("craft_score_postprocess_seconds", 0.0)))
                    if "ocr_proxy_score_seconds" in timing:
                        ocr_proxy_seconds.append(float(timing.get("ocr_proxy_score_seconds", 0.0)))
                    if "yolo_score_seconds" in timing:
                        yolo_score_seconds.append(float(timing.get("yolo_score_seconds", 0.0)))
                else:
                    track_key = track_key_for_path(path)
                    track = track_totals.setdefault(track_key, {"count": 0, "seconds": 0.0, "readable": 0})
                    track["count"] += 1
                    track["seconds"] += crop_seconds

                self.progress.emit(i, total, path.name)

            sort_start = time.perf_counter()
            if self.ranking_method == "yolo":
                results.sort(
                    key=lambda r: (
                        float(r.score),
                        int(bool(r.metrics.get("yolo_is_best", False))),
                        int(r.metrics.get("yolo_candidate_index") or 0),
                    ),
                    reverse=True,
                )
            else:
                results.sort(key=lambda r: float(r.score), reverse=True)
            sort_seconds = elapsed_since(sort_start)

            timing_summary: Dict[str, Any] = {
                "ranking_method": self.ranking_method,
                "requested_crop_count": total,
                "readable_crop_count": len(results),
                "score_worker_total_seconds": elapsed_since(run_start),
                "craft_scorer_setup_seconds": craft_setup_seconds,
                "sort_seconds": sort_seconds,
                "image_read": timing_stats(image_read_seconds),
                "crop_total": timing_stats(crop_total_seconds),
                "craft_total": timing_stats(craft_total_seconds),
                "craft_forward": timing_stats(craft_forward_seconds),
                "craft_preprocess": timing_stats(craft_preprocess_seconds),
                "craft_postprocess": timing_stats(craft_postprocess_seconds),
                "ocr_proxy_score": timing_stats(ocr_proxy_seconds),
                "yolo_score": timing_stats(yolo_score_seconds),
                "tracks": track_totals,
            }
            if craft_scorer is not None:
                timing_summary.update({
                    "craft_reader_load_seconds_total": craft_scorer.reader_load_seconds_total,
                    "craft_reader_load_count": craft_scorer.reader_load_count,
                    "craft_reader_cache_hit_count": craft_scorer.reader_cache_hit_count,
                    "craft_device": str(craft_scorer.device),
                })
            self.finished.emit(results, timing_summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImagePreview(QLabel):
    def __init__(self, min_size: Tuple[int, int] = (500, 380)) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(*min_size)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #000000; color: #d8d8d8; border: 1px solid #2c2c2c; border-radius: 8px; }"
        )
        self._pixmap_original: Optional[QPixmap] = None
        self.setText("Choose a directory of candidate crops")

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap_original = pixmap
        self._update_scaled_pixmap()

    def clear_preview(self, text: str) -> None:
        self._pixmap_original = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap_original is None or self._pixmap_original.isNull():
            return
        available = self.size()
        scaled = self._pixmap_original.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("")
        self.resize(1220, 760)

        self.current_folder: Optional[Path] = None
        self.results: List[CropScoreResult] = []
        self.thread: Optional[QThread] = None
        self.worker: Optional[ScoreWorker] = None
        self.ocr_lab_process: Optional[subprocess.Popen] = None
        self.ocr_lab_processes: List[subprocess.Popen] = []
        self.weight_spins: Dict[str, QDoubleSpinBox] = {}
        self.reducer_spins: Dict[str, QDoubleSpinBox] = {}
        self.weight_total_label = QLabel()
        self.yolo_weight_spins: Dict[str, QDoubleSpinBox] = {}
        self.yolo_weight_total_label = QLabel()
        self.craft_setting_spins: Dict[str, QDoubleSpinBox] = {}
        self.craft_gpu_check = QComboBox()
        self.craft_formula_label = QLabel()
        self.score_panes: Optional[QTabWidget] = None
        self.ranking_combo = QComboBox()
        self.ranking_combo.addItem("OCR-quality proxy score", "ocr")
        self.ranking_combo.addItem("CRAFT heatmap-sum score (slow)", "craft")
        self.ranking_combo.addItem("YOLO detector/crop score (uses saved metadata)", "yolo")
        self.ranking_combo.setCurrentIndex(max(0, self.ranking_combo.findData("craft")))
        self.ranking_combo.currentIndexChanged.connect(self.on_ranking_changed)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItem("Original crop", "original")
        self.preview_mode_combo.addItem("Text heatmap overlay", "text")
        self.preview_mode_combo.addItem("Affinity map overlay", "affinity")
        self.preview_mode_combo.addItem("Text + affinity overlay", "both")
        self.preview_mode_combo.setCurrentIndex(max(0, self.preview_mode_combo.findData("both")))
        self._last_heatmap_preview_mode = "both"
        self.preview_mode_combo.setToolTip(
            "Controls how selected crop previews are displayed. Heatmap overlays are available after scoring in CRAFT heatmap mode."
        )
        self.preview_mode_combo.currentIndexChanged.connect(self.on_preview_mode_changed)
        self.track_dirs: List[Path] = []
        self.current_run_folder: Optional[Path] = None
        self._updating_track_combo = False
        self.track_combo = QComboBox()
        self.track_combo.currentIndexChanged.connect(self.on_track_changed)
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 9999)
        self.frame_skip_spin.setSingleStep(1)
        self.frame_skip_spin.setValue(3)
        self.frame_skip_spin.setToolTip(
            "Load crops from every Nth detected frame. 1 = load every crop. "
            "Crops without a known frame index are always kept."
        )
        self.left_splitter: Optional[QSplitter] = None
        self.table_expand_button: Optional[QPushButton] = None
        self.table_collapse_button: Optional[QPushButton] = None
        self._selection_update_in_progress = False
        self._selected_row_order: List[int] = []
        self._pending_rescore_selected_paths: Optional[List[str]] = None
        self._pending_rescore_selection_keys: Optional[List[Dict[str, Any]]] = None
        self._last_displayed_selection_keys: List[Dict[str, Any]] = []
        self._table_sort_column: Optional[int] = None
        self._table_sort_order = Qt.SortOrder.DescendingOrder
        self._suppress_ranking_changed = False
        self._load_timing_summary: Dict[str, Any] = {}
        self._score_timing_summary: Dict[str, Any] = {}
        self._ocr_lab_timing_summary: Dict[str, Any] = {}
        self._yolo_metadata_cache: Dict[str, YoloMetadataIndex] = {}
        self._yolo_metadata_cache_roots: Dict[str, Path] = {}
        self._scoring_started_monotonic: Optional[float] = None

        self.model = CropTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(1, Qt.SortOrder.DescendingOrder)
        header.sectionClicked.connect(self.on_table_header_clicked)
        self.table.setMinimumHeight(0)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.clicked.connect(self.on_table_clicked)
        if self.table.selectionModel() is not None:
            self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)

        self._create_result_panels()

        self.open_button = QPushButton("Choose Crop Directory")
        self.open_button.clicked.connect(self.choose_directory)
        self.rescore_button = QPushButton("Apply Weights / Rescore")
        self.rescore_button.clicked.connect(self.rescore_current_directory)
        self.rescore_button.setEnabled(False)
        self.export_button = QPushButton("Export Score CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        self.open_ocr_lab_button = QPushButton("Send Selected to OCR Lab")
        self.open_ocr_lab_button.clicked.connect(self.open_selected_in_ocr_lab)
        self.open_ocr_lab_button.setEnabled(False)

        self.status_label = QLabel("No directory loaded.")
        self.status_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.timing_box = QTextEdit()
        self.timing_box.setReadOnly(True)
        self.timing_box.setMinimumHeight(90)
        self.timing_box.setMaximumHeight(160)
        self.timing_box.setPlainText("Timing: no run yet.")
        self.timing_box.setStyleSheet("QTextEdit { font-family: Menlo, Consolas, monospace; font-size: 8px; }")

        self._build_layout()
        self._build_menu()
        self.sync_scoring_pane_to_ranking_method()
        self.model.set_ranking_method(str(self.ranking_combo.currentData() or "craft"))

        if DEFAULT_INPUT_DIR.exists():
            self.load_directory(DEFAULT_INPUT_DIR, update_track_selector=True)

    def _make_details_box(self) -> QTextEdit:
        details = QTextEdit()
        details.setReadOnly(True)
        details.setMinimumHeight(150)
        details.setStyleSheet("QTextEdit { font-family: Menlo, Consolas, monospace; font-size: 9px; }")
        return details

    def _create_result_panels(self) -> None:
        self.selection_title_label = QLabel("Selected OCR crops (up to 4)")
        title_font = self.selection_title_label.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self.selection_title_label.setFont(title_font)

        self.result_cards_widget = QWidget()
        self.result_cards_layout = QGridLayout(self.result_cards_widget)
        self.result_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.result_cards_layout.setSpacing(8)

        self.result_cards_scroll = QScrollArea()
        self.result_cards_scroll.setWidgetResizable(True)
        self.result_cards_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.result_cards_scroll.setWidget(self.result_cards_widget)

        self.result_panels: List[QWidget] = []
        self.result_titles: List[QLabel] = []
        self.result_previews: List[ImagePreview] = []
        self.result_details: List[QTextEdit] = []

        for idx in range(4):
            panel = QWidget()
            panel.setStyleSheet("QWidget { background: #ffffff; }")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(4, 4, 4, 4)
            panel_layout.setSpacing(4)

            title = QLabel(f"Selection {idx + 1}")
            title.setWordWrap(True)
            title.setStyleSheet("QLabel { font-weight: 600; }")

            preview = ImagePreview(min_size=(240, 160))
            details = self._make_details_box()

            card_splitter = QSplitter(Qt.Orientation.Vertical)
            card_splitter.addWidget(preview)
            card_splitter.addWidget(details)
            card_splitter.setSizes([260, 220])

            panel_layout.addWidget(title)
            panel_layout.addWidget(card_splitter, 1)

            row = idx // 2
            col = idx % 2
            self.result_cards_layout.addWidget(panel, row, col)

            self.result_panels.append(panel)
            self.result_titles.append(title)
            self.result_previews.append(preview)
            self.result_details.append(details)

        self.preview = self.result_previews[0]
        self.details = self.result_details[0]
        self.clear_selected_result_panels("Choose a directory of candidate crops")

    def _build_layout(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.rescore_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.open_ocr_lab_button)
        controls_layout.addWidget(self._build_ranking_panel())
        controls_layout.addWidget(self._build_track_panel())
        controls_layout.addWidget(self._build_scoring_panes())
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress)
        controls_layout.addWidget(self.timing_box)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumHeight(180)
        controls_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        table_label = QLabel("Ranked candidate crops")
        table_font = table_label.font()
        table_font.setBold(True)
        table_label.setFont(table_font)

        self.table_expand_button = QPushButton()
        self.table_expand_button.setText("▲")
        self.table_expand_button.setToolTip("Expand table upward")
        self.table_expand_button.setFixedSize(26, 24)
        self.table_expand_button.clicked.connect(self.expand_table_section)

        self.table_collapse_button = QPushButton()
        self.table_collapse_button.setText("▼")
        self.table_collapse_button.setToolTip("Collapse table to header")
        self.table_collapse_button.setFixedSize(26, 24)
        self.table_collapse_button.clicked.connect(self.collapse_table_section)

        table_header = QWidget()
        table_header_layout = QHBoxLayout(table_header)
        table_header_layout.setContentsMargins(0, 0, 0, 0)
        table_header_layout.addWidget(table_label)
        table_header_layout.addStretch(1)
        table_header_layout.addWidget(self.table_collapse_button)
        table_header_layout.addWidget(self.table_expand_button)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(table_header, 0)
        table_layout.addWidget(self.table, 1)
        table_panel.setMinimumHeight(0)
        table_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.addWidget(controls_scroll)
        self.left_splitter.addWidget(table_panel)
        self.left_splitter.setChildrenCollapsible(True)
                                                                           
                                                                                 
        self.table.setVisible(False)
        self.left_splitter.setSizes([1000, 34])

        left_layout.addWidget(self.left_splitter, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.selection_title_label)
        right_layout.addWidget(self.result_cards_scroll, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([650, 700])

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def collapse_table_section(self) -> None:
        if self.left_splitter is None:
            return
        self.table.setVisible(False)
        total = max(1, sum(self.left_splitter.sizes()))
        header_h = 34
        self.left_splitter.setSizes([max(1, total - header_h), header_h])

    def expand_table_section(self) -> None:
        if self.left_splitter is None:
            return
        self.table.setVisible(True)
        total = max(1, sum(self.left_splitter.sizes()))
                                                              
                                                                 
        controls_h = max(120, min(int(total * 0.18), max(1, total - 260)))
        table_h = max(1, total - controls_h)
        self.left_splitter.setSizes([controls_h, table_h])

    def _build_scoring_panes(self) -> QWidget:
        panes = QTabWidget()
        panes.setDocumentMode(True)
        panes.setTabPosition(QTabWidget.TabPosition.North)
        panes.addTab(self._build_yolo_weight_panel(), "YOLO detector/crop")
        panes.addTab(self._build_weight_panel(), "OCR proxy")
        panes.addTab(self._build_craft_heatmap_panel(), "CRAFT heatmap")
        panes.setToolTip(
            "Switch between YOLO coefficients, the fast OCR proxy, and the CRAFT heatmap-sum score. "
            "Only the currently selected ranking method is used for ranking."
        )
        self.score_panes = panes
        self.sync_scoring_pane_to_ranking_method()
        return panes

    def _build_ranking_panel(self) -> QWidget:
        group = QGroupBox("Ranking method")
        layout = QFormLayout(group)
        layout.addRow("Rank by", self.ranking_combo)
        layout.addRow("Preview", self.preview_mode_combo)
        return group

    def _build_track_panel(self) -> QWidget:
        group = QGroupBox("Track selection")
        layout = QFormLayout(group)
        layout.addRow("Current track", self.track_combo)
        layout.addRow("Frame skip", self.frame_skip_spin)
        self.track_combo.addItem("Current directory", "__current__")
        self.track_combo.setEnabled(False)
        return group

    def _build_yolo_weight_panel(self) -> QWidget:
        group = QGroupBox("YOLO detector/crop scoring coefficients")
        group.setToolTip("Defaults exactly match yolo_engine.crop_quality(): 0.42*detector + 0.33*sharp + 0.18*area + 0.07*aspect - 0.16 edge penalty.")
        layout = QFormLayout(group)

        for key, label in YOLO_WEIGHT_ORDER:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix(" %")
            spin.setValue(DEFAULT_YOLO_WEIGHTS[key])
            spin.valueChanged.connect(self.on_yolo_weights_changed)
            self.yolo_weight_spins[key] = spin
            layout.addRow(label, spin)

        reset_button = QPushButton("Reset YOLO Defaults")
        reset_button.clicked.connect(self.reset_default_yolo_weights)
        layout.addRow(reset_button)
        layout.addRow("Current formula", self.yolo_weight_total_label)
        self.update_yolo_weight_label()
        return group

    def _build_weight_panel(self) -> QWidget:
        group = QGroupBox("OCR-proxy scoring weights (OCR mode only)")
        group.setToolTip("Weights are normalized during scoring, so they do not have to add to exactly 100%.")
        layout = QFormLayout(group)

        for key, label in WEIGHT_ORDER:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix(" %")
            spin.setValue(DEFAULT_WEIGHTS[key])
            spin.valueChanged.connect(self.on_weights_changed)
            self.weight_spins[key] = spin
            layout.addRow(label, spin)

        reducers_label = QLabel("Editable score reducers")
        reducers_label.setStyleSheet("QLabel { font-weight: 600; }")
        layout.addRow(reducers_label)

        for key, label, description in OCR_REDUCER_ORDER:
            spin = QDoubleSpinBox()
            if "percent" in key or key.endswith("_max_percent"):
                spin.setRange(0.0, 100.0)
                spin.setDecimals(1)
                spin.setSingleStep(0.5)
                spin.setSuffix(" %")
            else:
                spin.setRange(0.0, 1000.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.05)
            spin.setValue(DEFAULT_OCR_REDUCER_SETTINGS[key])
            spin.setToolTip(description)
            spin.valueChanged.connect(self.on_weights_changed)
            self.reducer_spins[key] = spin
            layout.addRow(label, spin)

        reset_button = QPushButton("Reset Default Weights")
        reset_button.clicked.connect(self.reset_default_weights)
        layout.addRow(reset_button)
        layout.addRow("Total", self.weight_total_label)
        self.update_weight_total_label()
        return group

    def _build_craft_heatmap_panel(self) -> QWidget:
        group = QGroupBox("CRAFT heatmap-sum score (slow)")
        group.setToolTip(
            "Ranks crops by summing raw CRAFT score_text and score_link heatmap evidence. "
            "This uses EasyOCR's CRAFT detector, so it is much slower than the basic OCR proxy."
        )
        layout = QFormLayout(group)

        self.craft_gpu_check.addItem("CPU", 0.0)
        self.craft_gpu_check.addItem("GPU if available", 1.0)
        self.craft_gpu_check.setCurrentIndex(1 if DEFAULT_CRAFT_HEATMAP_SETTINGS["gpu"] >= 0.5 else 0)
        self.craft_gpu_check.currentIndexChanged.connect(self.on_craft_settings_changed)
        layout.addRow("Device", self.craft_gpu_check)

        for key, label, description in CRAFT_HEATMAP_ORDER:
            if key == "gpu":
                continue
            spin = QDoubleSpinBox()
            if key == "canvas_size":
                spin.setRange(320.0, 4096.0)
                spin.setDecimals(0)
                spin.setSingleStep(64.0)
            elif key.endswith("_weight_percent"):
                spin.setRange(0.0, 100.0)
                spin.setDecimals(1)
                spin.setSingleStep(1.0)
                spin.setSuffix(" %")
            elif key.endswith("percent"):
                spin.setRange(0.1, 100.0)
                spin.setDecimals(1)
                spin.setSingleStep(0.5)
                spin.setSuffix(" %")
            elif "threshold" in key or key in {"low_text", "link_threshold"}:
                spin.setRange(0.0, 1.5)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)
            else:
                spin.setRange(0.1, 5.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.05)
            spin.setValue(float(DEFAULT_CRAFT_HEATMAP_SETTINGS[key]))
            spin.setToolTip(description)
            spin.valueChanged.connect(self.on_craft_settings_changed)
            self.craft_setting_spins[key] = spin
            layout.addRow(label, spin)

        reset_button = QPushButton("Reset CRAFT Defaults")
        reset_button.clicked.connect(self.reset_default_craft_settings)
        layout.addRow(reset_button)
        layout.addRow("Current formula", self.craft_formula_label)
        self.update_craft_formula_label()
        return group

    def current_craft_settings(self) -> Dict[str, float]:
        settings = {key: spin.value() for key, spin in self.craft_setting_spins.items()}
        settings["gpu"] = float(self.craft_gpu_check.currentData() or 0.0)
        return settings

    def update_craft_formula_label(self) -> None:
        if not self.craft_setting_spins:
            return
        s = self.current_craft_settings()
        _raw, w, total = normalized_craft_heatmap_weights(s)
        self.craft_formula_label.setWordWrap(True)
        self.craft_formula_label.setText(
            "100 * (component-capped "
            f"{w['craft_weight_text_sum_percent']:.2f}*text_sum + "
            f"{w['craft_weight_affinity_sum_percent']:.2f}*affinity_sum + "
            f"{w['craft_weight_weak_area_percent']:.2f}*weak_area + "
            f"{w['craft_weight_strong_area_percent']:.2f}*strong_area + "
            f"{w['craft_weight_peak_text_percent']:.2f}*peak_text + "
            f"{w['craft_weight_peak_affinity_percent']:.2f}*peak_affinity"
            "), using "
            f"raw weight total={total:.1f}%, "
            f"canvas={int(s['canvas_size'])}, mag={s['mag_ratio']:.2f}, "
            f"low_text={s['low_text']:.2f}, text_th={s['text_threshold']:.2f}, link_th={s['link_threshold']:.2f}"
        )

    def on_craft_settings_changed(self) -> None:
        self.update_craft_formula_label()
        if self.results and self.thread is None:
            self.status_label.setText("CRAFT heatmap settings changed. Click Apply Weights / Rescore to update rankings.")

    def reset_default_craft_settings(self) -> None:
        self.craft_gpu_check.blockSignals(True)
        self.craft_gpu_check.setCurrentIndex(1 if DEFAULT_CRAFT_HEATMAP_SETTINGS["gpu"] >= 0.5 else 0)
        self.craft_gpu_check.blockSignals(False)
        for key, spin in self.craft_setting_spins.items():
            spin.blockSignals(True)
            spin.setValue(float(DEFAULT_CRAFT_HEATMAP_SETTINGS[key]))
            spin.blockSignals(False)
        self.on_craft_settings_changed()

    def set_combo_by_data(self, combo: QComboBox, data: str) -> None:
        index = combo.findData(data)
        if index >= 0 and combo.currentIndex() != index:
            combo.setCurrentIndex(index)

    def sync_scoring_pane_to_ranking_method(self) -> None:
        ranking_method = str(self.ranking_combo.currentData() or "craft")
        target_tab = 2 if ranking_method == "craft" else (1 if ranking_method == "ocr" else 0)

        if self.score_panes is not None:
            for tab_index in range(self.score_panes.count()):
                self.score_panes.setTabEnabled(tab_index, tab_index == target_tab)
            if self.score_panes.currentIndex() != target_tab:
                self.score_panes.setCurrentIndex(target_tab)

        if ranking_method == "craft":
            self.preview_mode_combo.setEnabled(True)
            restore_mode = getattr(self, "_last_heatmap_preview_mode", "both") or "both"
            self.preview_mode_combo.blockSignals(True)
            self.set_combo_by_data(self.preview_mode_combo, restore_mode)
            self.preview_mode_combo.blockSignals(False)
        else:
            current_preview_mode = str(self.preview_mode_combo.currentData() or "original")
            if current_preview_mode != "original":
                self._last_heatmap_preview_mode = current_preview_mode
            self.preview_mode_combo.blockSignals(True)
            self.set_combo_by_data(self.preview_mode_combo, "original")
            self.preview_mode_combo.setEnabled(False)
            self.preview_mode_combo.blockSignals(False)

    def current_yolo_weights(self) -> Dict[str, float]:
        return {key: self.yolo_weight_spins[key].value() for key, _ in YOLO_WEIGHT_ORDER}

    def update_yolo_weight_label(self) -> None:
        if not self.yolo_weight_spins:
            return
        w = self.current_yolo_weights()
        self.yolo_weight_total_label.setText(
            f"{w['detector']:.1f}%*det + {w['sharpness']:.1f}%*sharp + "
            f"{w['area']:.1f}%*area + {w['aspect']:.1f}%*aspect - {w['edge_penalty']:.1f}%*edge"
        )

    def on_yolo_weights_changed(self) -> None:
        self.update_yolo_weight_label()
        if self.results and self.thread is None:
            self.status_label.setText("YOLO coefficients changed. Click Apply Weights / Rescore to update rankings.")

    def reset_default_yolo_weights(self) -> None:
        for key, _ in YOLO_WEIGHT_ORDER:
            self.yolo_weight_spins[key].blockSignals(True)
            self.yolo_weight_spins[key].setValue(DEFAULT_YOLO_WEIGHTS[key])
            self.yolo_weight_spins[key].blockSignals(False)
        self.on_yolo_weights_changed()

    def current_weights(self) -> Dict[str, float]:
        return {key: self.weight_spins[key].value() for key, _ in WEIGHT_ORDER}

    def current_ocr_settings(self) -> Dict[str, float]:
        settings = self.current_weights()
        settings.update({key: self.reducer_spins[key].value() for key, _, _ in OCR_REDUCER_ORDER})
        return settings

    def update_weight_total_label(self) -> None:
        if not self.weight_spins:
            return
        total = sum(self.current_weights().values())
        if abs(total - 100.0) <= 0.05:
            text = f"{total:.1f}%"
        else:
            text = f"{total:.1f}%  (normalized when scoring)"
        self.weight_total_label.setText(text)

    def on_weights_changed(self) -> None:
        self.update_weight_total_label()
        if self.results and self.thread is None:
            self.status_label.setText("Weights changed. Click Apply Weights / Rescore to update rankings.")

    def on_ranking_changed(self) -> None:
        if self._suppress_ranking_changed:
            return
        ranking_method = str(self.ranking_combo.currentData() or "ocr")
        self.sync_scoring_pane_to_ranking_method()
        self.model.set_ranking_method(ranking_method)
        if self.current_folder is not None and self.thread is None:
            self.load_directory(self.current_folder, update_track_selector=False)

    def on_preview_mode_changed(self, *_args) -> None:
        ranking_method = str(self.ranking_combo.currentData() or "craft")
        preview_mode = str(self.preview_mode_combo.currentData() or "original")
        if ranking_method == "craft":
            self._last_heatmap_preview_mode = preview_mode
        if self.results and self.thread is None:
            self.show_selected_results(self.selected_results())

    def reset_default_weights(self) -> None:
        for key, _ in WEIGHT_ORDER:
            self.weight_spins[key].blockSignals(True)
            self.weight_spins[key].setValue(DEFAULT_WEIGHTS[key])
            self.weight_spins[key].blockSignals(False)
        for key, _, _ in OCR_REDUCER_ORDER:
            self.reducer_spins[key].blockSignals(True)
            self.reducer_spins[key].setValue(DEFAULT_OCR_REDUCER_SETTINGS[key])
            self.reducer_spins[key].blockSignals(False)
        self.on_weights_changed()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Choose Crop Directory...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_directory)
        file_menu.addAction(open_action)

        export_action = QAction("Export Score CSV...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def current_frame_skip(self) -> int:
        return max(1, int(self.frame_skip_spin.value()))

    def choose_directory(self) -> None:
        start_dir = str(self.current_run_folder or self.current_folder or DEFAULT_INPUT_DIR.parent)
        folder = QFileDialog.getExistingDirectory(self, "Choose YOLO export folder, track folder, or crop directory", start_dir)
        if folder:
            self.load_directory(Path(folder), update_track_selector=True)

    def selection_key_for_result(self, result: CropScoreResult) -> Dict[str, Any]:
        m = result.metrics
        return {
            "path": safe_resolve(result.path),
            "name": result.path.name,
            "crop_number": optional_int(m.get("crop_number")),
            "track": optional_int(m.get("yolo_track_id")),
            "candidate": optional_int(m.get("yolo_candidate_index")),
            "frame": optional_int(m.get("yolo_frame_index")),
        }

    def selected_results_for_rescore_preserve(self) -> List[CropScoreResult]:
                                                                                
                                                                           
                                                                     
        results: List[CropScoreResult] = []
        seen_paths: set[str] = set()
        used_rows: set[int] = set()

        for key in self._last_displayed_selection_keys:
            row = self.row_for_rescore_selection_key(key, used_rows)
            if row is not None and 0 <= row < len(self.results):
                result = self.results[row]
                path_key = safe_resolve(result.path)
                if path_key not in seen_paths:
                    results.append(result)
                    seen_paths.add(path_key)
                    used_rows.add(row)
            if len(results) >= 4:
                return results

                                                     
        for row in self._selected_row_order:
            if 0 <= row < len(self.results):
                result = self.results[row]
                key = safe_resolve(result.path)
                if key not in seen_paths:
                    results.append(result)
                    seen_paths.add(key)
                    used_rows.add(row)
            if len(results) >= 4:
                return results

                                                
        selection_model = self.table.selectionModel()
        if selection_model is not None:
            for index in selection_model.selectedRows():
                row = index.row()
                if 0 <= row < len(self.results):
                    result = self.results[row]
                    key = safe_resolve(result.path)
                    if key not in seen_paths:
                        results.append(result)
                        seen_paths.add(key)
                        used_rows.add(row)
                if len(results) >= 4:
                    return results

        current_row = self.table.currentIndex().row()
        if not results and 0 <= current_row < len(self.results):
            results.append(self.results[current_row])

        return results[:4]

    def row_for_rescore_selection_key(self, key: Dict[str, Any], used_rows: set[int]) -> Optional[int]:
                                            
        target_path = key.get("path")
        if target_path:
            for row, result in enumerate(self.results):
                if row not in used_rows and safe_resolve(result.path) == target_path:
                    return row

                                                                        
                                               
        target_name = key.get("name")
        if target_name:
            filename_matches = [
                row for row, result in enumerate(self.results)
                if row not in used_rows and result.path.name == target_name
            ]
            if len(filename_matches) == 1:
                return filename_matches[0]

                                                   
        target_tuple = (key.get("track"), key.get("candidate"), key.get("frame"))
        if any(value is not None for value in target_tuple):
            for row, result in enumerate(self.results):
                if row in used_rows:
                    continue
                m = result.metrics
                candidate_tuple = (
                    optional_int(m.get("yolo_track_id")),
                    optional_int(m.get("yolo_candidate_index")),
                    optional_int(m.get("yolo_frame_index")),
                )
                if candidate_tuple == target_tuple:
                    return row

                                  
        target_crop_number = key.get("crop_number")
        if target_crop_number is not None:
            crop_matches = [
                row for row, result in enumerate(self.results)
                if row not in used_rows and optional_int(result.metrics.get("crop_number")) == target_crop_number
            ]
            if len(crop_matches) == 1:
                return crop_matches[0]

        return None

    def rescore_current_directory(self) -> None:
        if self.current_folder is not None:
            if self._last_displayed_selection_keys:
                self._pending_rescore_selection_keys = list(self._last_displayed_selection_keys)[:4]
            else:
                selected_before = self.selected_results_for_rescore_preserve()
                self._pending_rescore_selection_keys = [
                    self.selection_key_for_result(result) for result in selected_before
                ][:4]

            self._pending_rescore_selected_paths = [
                key["path"] for key in (self._pending_rescore_selection_keys or []) if key.get("path")
            ]
            self.load_directory(self.current_folder, update_track_selector=False)

    def yolo_metadata_cache_root_for_folder(self, folder: Path) -> Path:
        try:
            folder_resolved = folder.expanduser().resolve()
        except Exception:
            folder_resolved = folder

        if self.current_run_folder is not None:
            try:
                run_resolved = self.current_run_folder.expanduser().resolve()
                if folder_resolved == run_resolved or run_resolved in folder_resolved.parents:
                    return self.current_run_folder
            except Exception:
                pass

        run_folder, _track_dirs, _selected_track = infer_run_and_selected_track(folder)
        return run_folder

    def get_cached_yolo_metadata(self, folder: Path) -> Tuple[YoloMetadataIndex, Dict[str, Any]]:
        lookup_start = time.perf_counter()
        cache_root = self.yolo_metadata_cache_root_for_folder(folder)
        cache_key = safe_resolve(cache_root)
        cached = self._yolo_metadata_cache.get(cache_key)
        if cached is not None:
            return cached, {
                "metadata_cache_hit": True,
                "metadata_cache_status": "cache hit",
                "metadata_cache_key": cache_key,
                "metadata_cache_root": str(cache_root),
                "metadata_cache_lookup_seconds": elapsed_since(lookup_start),
                "metadata_load_seconds": 0.0,
                "yolo_metadata_rows": cached.count,
            }

        load_start = time.perf_counter()
        metadata = load_yolo_metadata_index(cache_root)
        metadata_load_seconds = elapsed_since(load_start)
        self._yolo_metadata_cache[cache_key] = metadata
        self._yolo_metadata_cache_roots[cache_key] = cache_root
        return metadata, {
            "metadata_cache_hit": False,
            "metadata_cache_status": "cache miss - loaded",
            "metadata_cache_key": cache_key,
            "metadata_cache_root": str(cache_root),
            "metadata_cache_lookup_seconds": elapsed_since(lookup_start),
            "metadata_load_seconds": metadata_load_seconds,
            "yolo_metadata_rows": metadata.count,
        }

    def format_timing_summary_text(self) -> str:
        lines: List[str] = ["Timing display (not exported):"]
        load = dict(self._load_timing_summary or {})
        score = dict(self._score_timing_summary or {})
        lab = dict(self._ocr_lab_timing_summary or {})

        if load:
            lines.append("Load / setup:")
            if "image_path_scan_seconds" in load:
                lines.append(f"  Image path scan: {format_seconds(load.get('image_path_scan_seconds'))} ({load.get('all_image_count', 0)} found)")
            if "metadata_load_seconds" in load:
                cache_status = load.get("metadata_cache_status") or ("cache hit" if load.get("metadata_cache_hit") else "loaded")
                lines.append(
                    f"  YOLO metadata: {format_seconds(load.get('metadata_load_seconds'))} "
                    f"({load.get('yolo_metadata_rows', 0)} rows, {cache_status})"
                )
                if "metadata_cache_lookup_seconds" in load:
                    lines.append(f"  YOLO metadata cache lookup: {format_seconds(load.get('metadata_cache_lookup_seconds'))}")
            if "frame_filter_seconds" in load:
                lines.append(f"  Frame-skip filter: {format_seconds(load.get('frame_filter_seconds'))} ({load.get('kept_image_count', 0)} kept)")
            if "load_setup_total_seconds" in load:
                lines.append(f"  Load/setup total: {format_seconds(load.get('load_setup_total_seconds'))}")

        if score:
            lines.append("Scoring:")
            lines.append(f"  Worker total: {format_seconds(score.get('score_worker_total_seconds'))} ({score.get('readable_crop_count', 0)}/{score.get('requested_crop_count', 0)} readable)")
            crop = score.get("crop_total", {}) or {}
            if crop.get("count", 0):
                lines.append(f"  Per crop total: avg {format_seconds(crop.get('avg'))}, min {format_seconds(crop.get('min'))}, max {format_seconds(crop.get('max'))}")
            image = score.get("image_read", {}) or {}
            if image.get("count", 0):
                lines.append(f"  Image file reads: total {format_seconds(image.get('total'))}, avg {format_seconds(image.get('avg'))}")
            if score.get("ranking_method") == "craft":
                lines.append(f"  CRAFT reader load: {format_seconds(score.get('craft_reader_load_seconds_total', 0.0))} ({int(score.get('craft_reader_load_count', 0))} load, {int(score.get('craft_reader_cache_hit_count', 0))} cache hits)")
                lines.append(f"  CRAFT device: {score.get('craft_device') or '—'}")
                for key, label in (
                    ("craft_total", "CRAFT heatmap + scoring"),
                    ("craft_preprocess", "CRAFT preprocess"),
                    ("craft_forward", "CRAFT forward pass"),
                    ("craft_postprocess", "CRAFT scoring/postprocess"),
                ):
                    stats = score.get(key, {}) or {}
                    if stats.get("count", 0):
                        lines.append(f"  {label}: total {format_seconds(stats.get('total'))}, avg {format_seconds(stats.get('avg'))}")
            elif score.get("ranking_method") == "ocr":
                stats = score.get("ocr_proxy_score", {}) or {}
                if stats.get("count", 0):
                    lines.append(f"  OCR-proxy scoring: total {format_seconds(stats.get('total'))}, avg {format_seconds(stats.get('avg'))}")
            elif score.get("ranking_method") == "yolo":
                stats = score.get("yolo_score", {}) or {}
                if stats.get("count", 0):
                    lines.append(f"  YOLO scoring: total {format_seconds(stats.get('total'))}, avg {format_seconds(stats.get('avg'))}")
            lines.append(f"  Sorting: {format_seconds(score.get('sort_seconds'))}")

            tracks = score.get("tracks", {}) or {}
            if tracks:
                lines.append("Track processing:")
                ordered_tracks = sorted(
                    tracks.items(),
                    key=lambda item: float((item[1] or {}).get("seconds", 0.0)),
                    reverse=True,
                )
                for name, info in ordered_tracks[:8]:
                    seconds = float((info or {}).get("seconds", 0.0))
                    count = int((info or {}).get("count", 0))
                    avg = seconds / max(1, count)
                    lines.append(f"  {name}: {format_seconds(seconds)} total, {count} crop(s), avg {format_seconds(avg)}")
                if len(ordered_tracks) > 8:
                    lines.append(f"  ... {len(ordered_tracks) - 8} more track(s)")

        if lab:
            lines.append("OCR Lab send:")
            lines.append(f"  Last image: {lab.get('image_name', '—')}")
            lines.append(f"  Total button action: {format_seconds(lab.get('total_seconds'))}")
            if "first_send_seconds" in lab:
                lines.append(f"  First socket send attempt: {format_seconds(lab.get('first_send_seconds'))} ({'success' if lab.get('first_send_success') else 'not connected'})")
            if "process_launch_seconds" in lab:
                lines.append(f"  OCR lab launch call: {format_seconds(lab.get('process_launch_seconds'))}")
            if "startup_stream_seconds" in lab:
                lines.append(f"  Startup stream wait/send: {format_seconds(lab.get('startup_stream_seconds'))} ({int(lab.get('startup_send_attempts', 0))} attempt(s))")

        return "\n".join(lines) if len(lines) > 1 else "Timing: no run yet."

    def update_timing_display(self) -> None:
        self.timing_box.setPlainText(self.format_timing_summary_text())

    def format_result_timing_block(self, r: CropScoreResult) -> str:
        t = dict(r.timing or {})
        if not t:
            return ""
        lines = ["Timing for this crop:"]
        if "image_read_seconds" in t:
            lines.append(f"  Image file read: {format_seconds(t.get('image_read_seconds'))}")
        if "score_path_seconds" in t:
            lines.append(f"  score_path total: {format_seconds(t.get('score_path_seconds'))}")
        if "crop_worker_total_seconds" in t:
            lines.append(f"  Worker crop total: {format_seconds(t.get('crop_worker_total_seconds'))}")
        if "ocr_proxy_score_seconds" in t:
            lines.append(f"  OCR-proxy score: {format_seconds(t.get('ocr_proxy_score_seconds'))}")
        if "yolo_metadata_lookup_seconds" in t:
            lines.append(f"  YOLO metadata lookup: {format_seconds(t.get('yolo_metadata_lookup_seconds'))}")
        if "yolo_score_seconds" in t:
            lines.append(f"  YOLO score: {format_seconds(t.get('yolo_score_seconds'))}")
        if "craft_total_seconds" in t:
            lines.append(f"  CRAFT heatmap + score: {format_seconds(t.get('craft_total_seconds'))}")
            lines.append(f"  CRAFT reader ensure/load: {format_seconds(t.get('craft_reader_ensure_seconds', 0.0))} / {format_seconds(t.get('craft_reader_load_seconds', 0.0))} ({t.get('craft_reader_source', '—')})")
            lines.append(f"  CRAFT preprocess: {format_seconds(t.get('craft_preprocess_seconds', 0.0))}")
            lines.append(f"  CRAFT forward pass: {format_seconds(t.get('craft_forward_seconds', 0.0))}")
            lines.append(f"  CRAFT tensor transfer: {format_seconds(t.get('craft_transfer_seconds', 0.0))}")
            lines.append(f"  CRAFT scoring/postprocess: {format_seconds(t.get('craft_score_postprocess_seconds', 0.0))}")
            if t.get("craft_resized_width") and t.get("craft_resized_height"):
                lines.append(f"  CRAFT resized input: {t.get('craft_resized_width')} x {t.get('craft_resized_height')}")
        lines.append("")
        return "\n".join(lines)

    def configure_track_selector(self, folder: Path) -> Path:
        run_folder, track_dirs, selected_track = infer_run_and_selected_track(folder)
        self._updating_track_combo = True
        self.track_combo.clear()
        self.current_run_folder = run_folder
        self.track_dirs = list(track_dirs)
        try:
            if track_dirs:
                chosen_track = selected_track if selected_track in track_dirs else track_dirs[0]
                self.track_combo.addItem("All tracks", "__all__")
                for track_dir in track_dirs:
                    crop_count = len(find_images(folder_for_track(track_dir)))
                    label = f"{track_dir.name}  ({crop_count} crop{'s' if crop_count != 1 else ''})"
                    self.track_combo.addItem(label, str(track_dir))
                selected_index = 1 + (track_dirs.index(chosen_track) if chosen_track in track_dirs else 0)
                self.track_combo.setCurrentIndex(selected_index)
                self.track_combo.setEnabled(True)
                return folder_for_track(chosen_track)
            self.track_combo.addItem("Current directory (no track subfolders found)", "__current__")
            self.track_combo.setEnabled(False)
            return folder
        finally:
            self._updating_track_combo = False

    def on_track_changed(self) -> None:
        if self._updating_track_combo or self.thread is not None:
            return
        data = self.track_combo.currentData()
        if data == "__current__":
            return
        if data == "__all__":
            if self.current_run_folder is not None:
                self.load_directory(self.current_run_folder, update_track_selector=False)
            return
        if data:
            track_dir = Path(str(data))
            self.load_directory(folder_for_track(track_dir), update_track_selector=False)

    def load_directory(self, folder: Path, update_track_selector: bool = True) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "Scoring in progress", "Please wait for the current scoring pass to finish.")
            return

        load_setup_start = time.perf_counter()
        self._load_timing_summary = {}
        self._score_timing_summary = {}
        self._ocr_lab_timing_summary = dict(self._ocr_lab_timing_summary or {})
        self.update_timing_display()

        if update_track_selector:
            folder = self.configure_track_selector(folder)

        ocr_weights = self.current_ocr_settings()
        yolo_weights = self.current_yolo_weights()
        craft_settings = self.current_craft_settings()
        ranking_method = str(self.ranking_combo.currentData() or "ocr")
        self.model.set_ranking_method(ranking_method)
        if ranking_method == "ocr" and sum(self.current_weights().values()) <= 0.0:
            QMessageBox.warning(self, "Invalid OCR weights", "At least one OCR-proxy scoring weight must be greater than zero.")
            return
        if ranking_method == "yolo" and sum(yolo_weights[k] for k in ("detector", "sharpness", "area", "aspect")) <= 0.0:
            QMessageBox.warning(self, "Invalid YOLO coefficients", "At least one additive YOLO coefficient must be greater than zero.")
            return

        scan_start = time.perf_counter()
        all_paths = find_images(folder)
        image_path_scan_seconds = elapsed_since(scan_start)

        yolo_metadata, metadata_timing = self.get_cached_yolo_metadata(folder)

        frame_skip = self.current_frame_skip()
        filter_start = time.perf_counter()
        paths, frame_skip_summary = filter_paths_by_frame_skip(all_paths, frame_skip, yolo_metadata)
        frame_filter_seconds = elapsed_since(filter_start)

        self._load_timing_summary = {
            "image_path_scan_seconds": image_path_scan_seconds,
            **metadata_timing,
            "frame_filter_seconds": frame_filter_seconds,
            "all_image_count": len(all_paths),
            "kept_image_count": len(paths),
            "yolo_metadata_rows": yolo_metadata.count,
            "load_setup_total_seconds": elapsed_since(load_setup_start),
        }
        self.update_timing_display()
        self.current_folder = folder
        self.results = []
        self._selected_row_order = []
        self._table_sort_column = 1
        self._table_sort_order = Qt.SortOrder.DescendingOrder
        self.table.horizontalHeader().setSortIndicator(1, Qt.SortOrder.DescendingOrder)
        self.model.set_results([])
        self.clear_selected_result_panels("Scoring crops...")
        self.export_button.setEnabled(False)
        self.open_ocr_lab_button.setEnabled(False)
        self.rescore_button.setEnabled(False)

        if not all_paths:
            self.status_label.setText(f"No image files found in:\n{folder}")
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.clear_selected_result_panels("No candidate crops found")
            self.open_ocr_lab_button.setEnabled(False)
            self.rescore_button.setEnabled(True)
            return

        if not paths:
            self.status_label.setText(
                f"No crop images matched frame skip={frame_skip}.\n"
                f"Found {len(all_paths)} image file(s) in:\n{folder}"
            )
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.clear_selected_result_panels("No crops matched the current frame skip")
            self.open_ocr_lab_button.setEnabled(False)
            self.rescore_button.setEnabled(True)
            return

        metadata_cache_note = "cached" if metadata_timing.get("metadata_cache_hit") else "loaded"
        metadata_note = f" | YOLO metadata rows: {yolo_metadata.count} ({metadata_cache_note})" if yolo_metadata.count else f" | no YOLO metadata found ({metadata_cache_note})"
        if frame_skip > 1:
            frame_note = (
                f" | frame skip: every {frame_skip}th frame "
                f"({frame_skip_summary['kept_paths']}/{frame_skip_summary['input_paths']} crops, "
                f"{frame_skip_summary['kept_unique_frames']}/{frame_skip_summary['unique_frames']} known frames)"
            )
        else:
            frame_note = " | frame skip: 1 (all crops)"
        self.status_label.setText(f"Scoring {len(paths)} crop(s) from:\n{folder}{metadata_note}{frame_note}")
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)

        self.thread = QThread(self)
        self.worker = ScoreWorker(paths, ocr_weights, yolo_weights, craft_settings, ranking_method, yolo_metadata)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)
        self._scoring_started_monotonic = time.perf_counter()
        self.thread.start()

    def on_progress(self, current: int, total: int, filename: str) -> None:
        self.progress.setValue(current)
        frame_skip = self.current_frame_skip()
        skip_text = "" if frame_skip <= 1 else f" | frame skip: every {frame_skip}th frame"
        elapsed_text = ""
        if self._scoring_started_monotonic is not None:
            elapsed_text = f" | elapsed: {format_seconds(elapsed_since(self._scoring_started_monotonic))}"
        self.status_label.setText(
            f"Scoring {current}/{total}: {filename}{skip_text}{elapsed_text}\n{self.current_folder if self.current_folder else ''}"
        )

    def restore_pending_rescore_selection(self) -> bool:
        pending_keys = list(self._pending_rescore_selection_keys or [])

                                                                             
        if not pending_keys and self._pending_rescore_selected_paths:
            pending_keys = [{"path": path_key} for path_key in self._pending_rescore_selected_paths]

        self._pending_rescore_selection_keys = None
        self._pending_rescore_selected_paths = None

        if not pending_keys or not self.results:
            return False

        selected_rows: List[int] = []
        used_rows: set[int] = set()
        for key in pending_keys:
            row = self.row_for_rescore_selection_key(key, used_rows)
            if row is not None:
                selected_rows.append(row)
                used_rows.add(row)
            if len(selected_rows) >= 4:
                break

        if not selected_rows:
            return False

        selection_model = self.table.selectionModel()
        if selection_model is None:
            return False

        self._selection_update_in_progress = True
        try:
            selection_model.clearSelection()
            flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
            for row in selected_rows:
                selection_model.select(self.model.index(row, 0), flags)
            self._selected_row_order = selected_rows
            self.table.setCurrentIndex(self.model.index(selected_rows[0], 0))
        finally:
            self._selection_update_in_progress = False

        self.show_selected_results([self.results[row] for row in selected_rows if 0 <= row < len(self.results)])
        return True

    def on_finished(self, results: List[CropScoreResult], timing_summary: Optional[Dict[str, Any]] = None) -> None:
        self._score_timing_summary = dict(timing_summary or {})
        self.update_timing_display()
        self._scoring_started_monotonic = None
        self.results = results
        self.model.set_results(results)
        self._table_sort_column = 1
        self._table_sort_order = Qt.SortOrder.DescendingOrder
        self.table.horizontalHeader().setSortIndicator(1, Qt.SortOrder.DescendingOrder)
        self._resize_table_columns()
        self.rescore_button.setEnabled(True)
        self.export_button.setEnabled(bool(results))
        self.open_ocr_lab_button.setEnabled(bool(results))

        if not results:
            self.status_label.setText("Images were found, but none could be read successfully.")
            self.clear_selected_result_panels("No readable crop images")
            self.open_ocr_lab_button.setEnabled(False)
            return

        best = results[0]
        yolo_rows = sum(1 for r in results if r.metrics.get("yolo_score_available"))
        if best.metrics.get("ranking_method") == "yolo":
            rank_label = "YOLO score"
            score_text = "not available" if best.score < 0 else f"{best.score:.2f}/100"
        elif best.metrics.get("ranking_method") == "craft":
            rank_label = "CRAFT heatmap score"
            score_text = f"{best.score:.2f}/100"
        else:
            rank_label = "OCR score"
            score_text = f"{best.score:.2f}/100"
        total_time = self._score_timing_summary.get("score_worker_total_seconds")
        time_note = f" | scoring time {format_seconds(total_time)}" if total_time is not None else ""
        self.status_label.setText(
            f"Done. Best crop: {best.path.name} | {rank_label} {score_text} | "
            f"{len(results)} readable crop(s) | {yolo_rows} with YOLO metadata{time_note}."
        )

        if self.restore_pending_rescore_selection():
            return

        self._selected_row_order = [0]
        self.table.selectRow(0)
        self.show_selected_results([best])

    def on_failed(self, message: str) -> None:
        self._pending_rescore_selected_paths = None
        self._pending_rescore_selection_keys = None
        self._scoring_started_monotonic = None
        QMessageBox.critical(self, "Scoring failed", message)
        self.status_label.setText(f"Scoring failed: {message}")
        self.rescore_button.setEnabled(True)

    def cleanup_thread(self) -> None:
        self.worker = None
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None

    def _resize_table_columns(self) -> None:
        self.table.resizeColumnsToContents()
        if self.table.model() is not None and hasattr(self.model, "columns"):
            try:
                file_col = self.model.columns().index("file")
                self.table.setColumnWidth(file_col, max(260, self.table.columnWidth(file_col)))
            except ValueError:
                pass

    def send_image_path_to_ocr_lab(self, image_path: Path) -> Tuple[bool, Dict[str, Any]]:
        total_start = time.perf_counter()
        payload = {"command": "load_image", "image_path": str(image_path)}
        timing: Dict[str, Any] = {"image_name": image_path.name}
        try:
            connect_start = time.perf_counter()
            with socket.create_connection(
                (OCR_LAB_STREAM_HOST, OCR_LAB_STREAM_PORT),
                timeout=OCR_LAB_CONNECT_TIMEOUT_SEC,
            ) as sock:
                timing["socket_connect_seconds"] = elapsed_since(connect_start)
                send_start = time.perf_counter()
                sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                timing["socket_send_seconds"] = elapsed_since(send_start)
            timing["socket_total_seconds"] = elapsed_since(total_start)
            timing["success"] = True
            return True, timing
        except OSError as exc:
            timing["socket_total_seconds"] = elapsed_since(total_start)
            timing["success"] = False
            timing["error"] = str(exc)
            return False, timing

    def ensure_ocr_lab_process(self, initial_image_path: Path) -> bool:
        lab_script = SCRIPT_DIR / "ocr_detect_recognize_lab.py"
        if not lab_script.exists():
            QMessageBox.critical(
                self,
                "OCR lab not found",
                f"Could not find ocr_detect_recognize_lab.py next to this script:\n{lab_script}",
            )
            return False

        if self.ocr_lab_process is not None and self.ocr_lab_process.poll() is None:
            return True

        try:
            self.ocr_lab_processes = [p for p in self.ocr_lab_processes if p.poll() is None]
            self.ocr_lab_process = subprocess.Popen(
                [
                    sys.executable,
                    str(lab_script),
                    "--ipc-port",
                    str(OCR_LAB_STREAM_PORT),
                    "--image",
                    str(initial_image_path),
                ],
                cwd=str(SCRIPT_DIR),
            )
            self.ocr_lab_processes.append(self.ocr_lab_process)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Could not open OCR lab", str(exc))
            return False

    def send_or_open_selected_in_ocr_lab(self, image_path: Path) -> None:
        total_start = time.perf_counter()
        summary: Dict[str, Any] = {"image_name": image_path.name}

        sent, first_timing = self.send_image_path_to_ocr_lab(image_path)
        summary["first_send_seconds"] = first_timing.get("socket_total_seconds", 0.0)
        summary["first_send_success"] = sent
        if sent:
            summary["total_seconds"] = elapsed_since(total_start)
            self._ocr_lab_timing_summary = summary
            self.update_timing_display()
            self.status_label.setText(f"Sent selected crop to existing OCR lab window:\n{image_path}")
            return

        launch_start = time.perf_counter()
        opened = self.ensure_ocr_lab_process(image_path)
        summary["process_launch_seconds"] = elapsed_since(launch_start)
        if not opened:
            summary["total_seconds"] = elapsed_since(total_start)
            self._ocr_lab_timing_summary = summary
            self.update_timing_display()
            return

        startup_start = time.perf_counter()
        attempts = 0
        deadline = time.monotonic() + OCR_LAB_STARTUP_SEND_WINDOW_SEC
        while time.monotonic() < deadline:
            attempts += 1
            sent, attempt_timing = self.send_image_path_to_ocr_lab(image_path)
            if sent:
                summary["startup_stream_seconds"] = elapsed_since(startup_start)
                summary["startup_send_attempts"] = attempts
                summary["last_send_seconds"] = attempt_timing.get("socket_total_seconds", 0.0)
                summary["total_seconds"] = elapsed_since(total_start)
                self._ocr_lab_timing_summary = summary
                self.update_timing_display()
                self.status_label.setText(f"Opened OCR lab and streamed selected crop:\n{image_path}")
                return
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            time.sleep(0.08)

        summary["startup_stream_seconds"] = elapsed_since(startup_start)
        summary["startup_send_attempts"] = attempts
        summary["total_seconds"] = elapsed_since(total_start)
        self._ocr_lab_timing_summary = summary
        self.update_timing_display()
        self.status_label.setText(
            f"Opened OCR lab for selected crop. Future crops will stream to the same window:\n{image_path}"
        )

    def open_selected_in_ocr_lab(self) -> None:
        result = self.selected_result()
        if result is None:
            QMessageBox.information(self, "No crop selected", "Select a crop row first.")
            return

        image_path = result.path.expanduser().resolve()
        if not image_path.exists():
            QMessageBox.critical(self, "Image missing", f"Could not find crop image:\n{image_path}")
            return

                                                                                 
                                                                                
                                                      
        ranking_before = str(self.ranking_combo.currentData() or "ocr")
        ranking_index_before = self.ranking_combo.currentIndex()
        score_pane_before = self.score_panes.currentIndex() if self.score_panes is not None else None

        self._suppress_ranking_changed = True
        try:
            self.send_or_open_selected_in_ocr_lab(image_path)
        finally:
            if str(self.ranking_combo.currentData() or "ocr") != ranking_before:
                restore_index = self.ranking_combo.findData(ranking_before)
                if restore_index < 0:
                    restore_index = ranking_index_before
                self.ranking_combo.blockSignals(True)
                self.ranking_combo.setCurrentIndex(restore_index)
                self.ranking_combo.blockSignals(False)

            self.model.set_ranking_method(ranking_before)
            if score_pane_before is not None and self.score_panes is not None:
                self.score_panes.setCurrentIndex(score_pane_before)

            self._suppress_ranking_changed = False

    def selected_rows_ordered(self) -> List[int]:
        if not self.results:
            return []
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return []
        selected_now = {
            index.row()
            for index in selection_model.selectedRows()
            if 0 <= index.row() < len(self.results)
        }
        ordered = [row for row in self._selected_row_order if row in selected_now]
        for row in sorted(selected_now):
            if row not in ordered:
                ordered.append(row)
        return ordered[-4:]

    def selected_results(self) -> List[CropScoreResult]:
        rows = self.selected_rows_ordered()
        if rows:
            return [self.results[row] for row in rows if 0 <= row < len(self.results)]
        current_row = self.table.currentIndex().row()
        if 0 <= current_row < len(self.results):
            return [self.results[current_row]]
        return [self.results[0]] if self.results else []

    def selected_result(self) -> Optional[CropScoreResult]:
        if not self.results:
            return None

                                                                               
                                                                               
                                      
        rows = self.selected_rows_ordered()
        if rows:
            row = rows[0]
            if 0 <= row < len(self.results):
                return self.results[row]

        current_row = self.table.currentIndex().row()
        if 0 <= current_row < len(self.results):
            return self.results[current_row]
        return self.results[0] if self.results else None

    def enforce_selection_limit(self, rows: List[int]) -> List[int]:
        rows = [row for row in rows if 0 <= row < len(self.results)]
        deduped: List[int] = []
        for row in rows:
            if row in deduped:
                deduped.remove(row)
            deduped.append(row)
        rows = deduped[-4:]

        selection_model = self.table.selectionModel()
        if selection_model is None:
            return rows

        selected_now = {
            index.row()
            for index in selection_model.selectedRows()
            if 0 <= index.row() < len(self.results)
        }
        if selected_now == set(rows) and len(selected_now) <= 4:
            return rows

        self._selection_update_in_progress = True
        try:
            selection_model.clearSelection()
            flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
            for row in rows:
                selection_model.select(self.model.index(row, 0), flags)
            if rows:
                self.table.setCurrentIndex(self.model.index(rows[-1], 0))
        finally:
            self._selection_update_in_progress = False
        return rows

    def on_table_selection_changed(self, selected, deselected) -> None:
        if self._selection_update_in_progress:
            return
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return

        selected_now = {
            index.row()
            for index in selection_model.selectedRows()
            if 0 <= index.row() < len(self.results)
        }
        deselected_rows = {index.row() for index in deselected.indexes()}
        newly_selected_rows = {index.row() for index in selected.indexes()}

        order = [row for row in self._selected_row_order if row in selected_now and row not in deselected_rows]
        for row in sorted(newly_selected_rows):
            if 0 <= row < len(self.results) and row in selected_now:
                if row in order:
                    order.remove(row)
                order.append(row)

        current_row = self.table.currentIndex().row()
        if current_row in selected_now:
            if current_row in order:
                order.remove(current_row)
            order.append(current_row)

        for row in sorted(selected_now):
            if row not in order:
                order.append(row)

        self._selected_row_order = self.enforce_selection_limit(order)
        self.show_selected_results(self.selected_results())

    def on_table_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        row = index.row()
        if 0 <= row < len(self.results):
            if row not in self._selected_row_order:
                self._selected_row_order.append(row)
            self._selected_row_order = self.enforce_selection_limit(self._selected_row_order)
            self.show_selected_results(self.selected_results())

    def default_sort_order_for_column(self, col_id: str) -> Qt.SortOrder:
        ascending_columns = {"rank", "crop_number", "track", "candidate", "frame", "file", "width", "height"}
        return Qt.SortOrder.AscendingOrder if col_id in ascending_columns else Qt.SortOrder.DescendingOrder

    def selected_results_from_table_selection(self) -> List[CropScoreResult]:
        if not self.results:
            return []
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return []
        selected: List[CropScoreResult] = []
        for index in selection_model.selectedRows():
            row = index.row()
            if 0 <= row < len(self.results):
                selected.append(self.results[row])
        return selected

    def current_result_from_table(self) -> Optional[CropScoreResult]:
        if not self.results:
            return None
        row = self.table.currentIndex().row()
        if 0 <= row < len(self.results):
            return self.results[row]
        return None

    def restore_selection_after_sort(
        self,
        selected_before: Sequence[CropScoreResult],
        current_before: Optional[CropScoreResult],
    ) -> None:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            self.show_selected_results(self.selected_results())
            return

        selected_rows: List[int] = []
        for result in selected_before:
            row = identity_index(self.results, result)
            if row >= 0:
                selected_rows.append(row)

        self._selection_update_in_progress = True
        try:
            selection_model.clearSelection()
            self._selected_row_order = []
            if selected_rows:
                flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
                for row in selected_rows[-4:]:
                    selection_model.select(self.model.index(row, 0), flags)
                self._selected_row_order = selected_rows[-4:]
                self.table.setCurrentIndex(self.model.index(selected_rows[-1], 0))
            elif current_before is not None:
                current_row = identity_index(self.results, current_before)
                if current_row >= 0:
                    self.table.setCurrentIndex(self.model.index(current_row, 0))
        finally:
            self._selection_update_in_progress = False

        self.show_selected_results(self.selected_results())

    def on_table_header_clicked(self, section: int) -> None:
        cols = self.model.columns()
        if not (0 <= section < len(cols)) or not self.results:
            return

        selected_before = self.selected_results_from_table_selection()
        current_before = self.current_result_from_table()

        if self._table_sort_column == section:
            self._table_sort_order = (
                Qt.SortOrder.DescendingOrder
                if self._table_sort_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )
        else:
            self._table_sort_column = section
            self._table_sort_order = self.default_sort_order_for_column(cols[section])

        self.table.horizontalHeader().setSortIndicator(section, self._table_sort_order)
        self.model.sort(section, self._table_sort_order)
        self.results = list(self.model.results)
        self.restore_selection_after_sort(selected_before, current_before)
        self._resize_table_columns()

    def clear_selected_result_panels(self, message: str = "") -> None:
        if "No crop selected" in message or "No candidate crops" in message or "No readable crop images" in message:
            self._last_displayed_selection_keys = []
        self.selection_title_label.setText("Selected OCR crops (up to 4)")
        for idx, panel in enumerate(self.result_panels):
            self.result_titles[idx].setText(f"Selection {idx + 1}")
            self.result_previews[idx].clear_preview(message if idx == 0 else "")
            self.result_details[idx].clear()
            panel.setVisible(idx == 0)

    def crop_display_label(self, result: CropScoreResult) -> str:
        m = result.metrics
        crop_number = m.get("crop_number")
        if crop_number is None:
            crop_number = m.get("yolo_candidate_index")
        if crop_number is None:
            return "Crop # —"
        try:
            return f"Crop # {int(crop_number)}"
        except (TypeError, ValueError):
            return f"Crop # {crop_number}"

    def copied_heatmap_evidence_terms(self, result: CropScoreResult) -> str:
        m = result.metrics
        required_keys = (
            "craft_text_density_score",
            "craft_text_density_raw",
            "craft_text_density_good_percent",
            "craft_affinity_density_score",
            "craft_affinity_density_raw",
            "craft_affinity_density_good_percent",
            "craft_weak_text_area_score",
            "craft_weak_text_area_raw",
            "craft_weak_text_area_good_percent",
            "craft_strong_text_area_score",
            "craft_strong_text_area_raw",
            "craft_strong_text_area_good_percent",
            "craft_peak_text",
            "craft_peak_affinity",
            "craft_affinity_area_raw",
        )
        if not all(key in m and m.get(key) is not None for key in required_keys):
            return ""
        return (
            "Heatmap evidence terms:\n"
            + "Formula weights used:\n"
            + f"  Text Sum weight:      {float(m.get('craft_weight_text_sum_percent', 50.0)):.1f}%  "
            + f"(raw: {float(m.get('craft_weight_text_sum_raw_percent', 50.0)):.1f}%)\n"
            + f"  Affinity Sum weight:  {float(m.get('craft_weight_affinity_sum_percent', 20.0)):.1f}%  "
            + f"(raw: {float(m.get('craft_weight_affinity_sum_raw_percent', 20.0)):.1f}%)\n"
            + f"  Weak Area weight:     {float(m.get('craft_weight_weak_area_percent', 15.0)):.1f}%  "
            + f"(raw: {float(m.get('craft_weight_weak_area_raw_percent', 15.0)):.1f}%)\n"
            + f"  Strong Area weight:   {float(m.get('craft_weight_strong_area_percent', 10.0)):.1f}%  "
            + f"(raw: {float(m.get('craft_weight_strong_area_raw_percent', 10.0)):.1f}%)\n"
            + f"  Peak Text weight:     {float(m.get('craft_weight_peak_text_percent', 5.0)):.1f}%  "
            + f"(raw: {float(m.get('craft_weight_peak_text_raw_percent', 5.0)):.1f}%)\n"
            + f"  Peak Affinity weight: {float(m.get('craft_weight_peak_affinity_percent', 0.0)):.1f}%  "
            + f"(raw: {float(m.get('craft_weight_peak_affinity_raw_percent', 0.0)):.1f}%)\n\n"
            + "Existing terms use per-component caps before counting/summing, so one large word cannot dominate only by area.\n"
            + f"  Text Sum score:      {float(m['craft_text_density_score']):.3f}  "
            + f"(capped normalized sum: {100.0*float(m['craft_text_density_raw']):.3f}%, good at {float(m['craft_text_density_good_percent']):.1f}%)\n"
            + f"  Affinity Sum score:  {float(m['craft_affinity_density_score']):.3f}  "
            + f"(capped normalized sum: {100.0*float(m['craft_affinity_density_raw']):.3f}%, good at {float(m['craft_affinity_density_good_percent']):.1f}%)\n"
            + f"  Weak Area score:     {float(m['craft_weak_text_area_score']):.3f}  "
            + f"(capped area: {100.0*float(m['craft_weak_text_area_raw']):.3f}%, uncapped: {100.0*float(m.get('craft_weak_text_area_uncapped_raw', m['craft_weak_text_area_raw'])):.3f}%, good at {float(m['craft_weak_text_area_good_percent']):.1f}%)\n"
            + f"  Strong Area score:   {float(m['craft_strong_text_area_score']):.3f}  "
            + f"(capped area: {100.0*float(m['craft_strong_text_area_raw']):.3f}%, uncapped: {100.0*float(m.get('craft_strong_text_area_uncapped_raw', m['craft_strong_text_area_raw'])):.3f}%, good at {float(m['craft_strong_text_area_good_percent']):.1f}%)\n"
            + f"  Peak text:           {float(m['craft_peak_text']):.3f}  "
            + f"(score: {float(m.get('craft_peak_text_score', clamp01(float(m['craft_peak_text'])))):.3f})\n"
            + f"  Peak affinity:       {float(m['craft_peak_affinity']):.3f}  "
            + f"(score: {float(m.get('craft_peak_affinity_score', clamp01(float(m['craft_peak_affinity'])))):.3f})\n"
            + f"  Affinity area:       {100.0*float(m['craft_affinity_area_raw']):.3f}% of heatmap pixels > link_threshold\n\n"
        )

    def format_details_for_selected_panel(self, result: CropScoreResult) -> str:
        return self.copied_heatmap_evidence_terms(result) + self.format_details(result)

    def heatmap_to_color_bgr(self, heatmap: Optional[np.ndarray], output_size: Tuple[int, int]) -> Optional[np.ndarray]:
        if heatmap is None:
            return None
        arr = np.asarray(heatmap, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return None
        width, height = output_size
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        heat_u8 = (arr * 255.0).astype(np.uint8)
        heat_u8 = cv2.resize(heat_u8, (width, height), interpolation=cv2.INTER_CUBIC)
        return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    def overlay_heatmap_like_demo(
        self,
        image_bgr: np.ndarray,
        heatmap: Optional[np.ndarray],
        alpha: float = 0.45,
    ) -> Optional[np.ndarray]:
        if heatmap is None:
            return None
        image_bgr = np.ascontiguousarray(image_bgr)
        h, w = image_bgr.shape[:2]
        heatmap_bgr = self.heatmap_to_color_bgr(heatmap, (w, h))
        if heatmap_bgr is None:
            return None
        alpha = max(0.0, min(1.0, float(alpha)))
        return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)

    def make_preview_rgb(self, image_bgr: np.ndarray, result: CropScoreResult) -> np.ndarray:
        mode = str(self.preview_mode_combo.currentData() or "original")
        if mode == "original":
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        has_text = result.craft_text_heatmap is not None
        has_affinity = result.craft_affinity_heatmap is not None
        if not has_text and not has_affinity:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if mode == "text":
            overlay_bgr = self.overlay_heatmap_like_demo(image_bgr, result.craft_text_heatmap, alpha=0.45)
        elif mode == "affinity":
            overlay_bgr = self.overlay_heatmap_like_demo(image_bgr, result.craft_affinity_heatmap, alpha=0.45)
        elif mode == "both":
            if has_text and has_affinity:
                combined = np.maximum(
                    np.asarray(result.craft_text_heatmap, dtype=np.float32),
                    np.asarray(result.craft_affinity_heatmap, dtype=np.float32),
                )
            else:
                combined = result.craft_text_heatmap if has_text else result.craft_affinity_heatmap
            overlay_bgr = self.overlay_heatmap_like_demo(image_bgr, combined, alpha=0.45)
        else:
            overlay_bgr = image_bgr

        if overlay_bgr is None:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    def preview_mode_suffix(self, result: CropScoreResult) -> str:
        mode = str(self.preview_mode_combo.currentData() or "original")
        if mode == "original":
            return ""
        if result.craft_text_heatmap is None and result.craft_affinity_heatmap is None:
            return "  [overlay unavailable: rescore in CRAFT heatmap mode]"
        label = self.preview_mode_combo.currentText()
        return f"  [{label}]"

    def show_selected_results(self, results: Sequence[CropScoreResult]) -> None:
        results = list(results)[:4]
        self._last_displayed_selection_keys = [
            self.selection_key_for_result(result) for result in results
        ]
        if not results:
            self.clear_selected_result_panels("No crop selected")
            return

        if len(results) == 1:
            self.selection_title_label.setText("Selected OCR crop")
        else:
            self.selection_title_label.setText(f"Selected OCR crops ({len(results)}/4)")

        for idx in range(4):
            if idx >= len(results):
                self.result_titles[idx].setText(f"Selection {idx + 1}")
                self.result_previews[idx].clear_preview("")
                self.result_details[idx].clear()
                self.result_panels[idx].setVisible(False)
                continue

            result = results[idx]
            rank = self.model.default_rank_for_result(result) or (idx + 1)
            self.result_titles[idx].setText(f"Rank {rank}: {self.crop_display_label(result)}{self.preview_mode_suffix(result)}")
            self.result_panels[idx].setVisible(True)

            image_bgr = imread_bgr(result.path)
            if image_bgr is None:
                self.result_previews[idx].clear_preview("Could not read selected crop")
                self.result_details[idx].setPlainText(self.format_details_for_selected_panel(result))
                continue

            image_rgb = self.make_preview_rgb(image_bgr, result)
            h, w = image_rgb.shape[:2]
            bytes_per_line = 3 * w
            qimage = QImage(
                image_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()
            self.result_previews[idx].set_preview_pixmap(QPixmap.fromImage(qimage))
            self.result_details[idx].setPlainText(self.format_details_for_selected_panel(result))

    def show_result(self, result: CropScoreResult) -> None:
        self.show_selected_results([result])

    def yolo_factor_descriptions(self) -> str:
        return (
            "YOLO detector/crop terms:\n"
            "  Detector: saved YOLO confidence for this crop.\n"
            "  Sharpness: crop Laplacian variance divided by 700 and clipped to 1.\n"
            "  Area: crop area compared with 20% of the source frame area.\n"
            "  Aspect: 1.0 for aspect ratio 0.22 to 7.0; otherwise 0.45.\n"
            "  Edge penalty: subtracts the editable penalty if the saved box touches the frame edge.\n"
            "  Saved quality: shown only for comparison; the table ranking is recomputed from the current settings.\n"
        )

    def ocr_factor_descriptions(self) -> str:
        return (
            "OCR-proxy terms:\n"
            "  Sharpness: edge crispness from Laplacian variance.\n"
            "  Contrast: grayscale p95-p5 range plus standard deviation.\n"
            "  Edge density: amount of Canny edge structure; moderate text-like density scores best.\n"
            "  Text-pixel mass: Otsu foreground/stroke coverage.\n"
            "  Components: plausible connected text-like components.\n"
            "  Resolution: crop size, with emphasis on having enough short-side and long-side pixels.\n"
            "  Border cut penalty: reduces the score when foreground is concentrated near crop borders.\n"
            "  Saturation penalty: reduces the score when too many pixels are near pure black or white.\n"
        )

    def craft_factor_descriptions(self) -> str:
        return (
            "CRAFT heatmap terms:\n"
            "  Text Sum: component-capped score_text energy above low_text.\n"
            "  Affinity Sum: component-capped score_link energy above link_threshold.\n"
            "  Weak Area: component-capped area above low_text.\n"
            "  Strong Area: component-capped area above text_threshold.\n"
            "  Peak Text: maximum score_text value in the crop.\n"
            "  Peak Affinity: maximum score_link value in the crop.\n"
            "  Formula weights: editable contribution weights; they are normalized during scoring.\n"
        )

    def selected_crop_table_values(self, r: CropScoreResult) -> str:
        m = r.metrics
        ranking_method = str(m.get("ranking_method", "yolo"))
        rank = self.model.default_rank_for_result(r)
        rank_text = "—" if rank is None else str(rank)
        crop_number = m.get("crop_number") if m.get("crop_number") is not None else "—"

        if ranking_method == "craft":
            required_keys = (
                "craft_text_density_score",
                "craft_text_density_raw",
                "craft_text_density_good_percent",
                "craft_affinity_density_score",
                "craft_affinity_density_raw",
                "craft_affinity_density_good_percent",
                "craft_weak_text_area_score",
                "craft_weak_text_area_raw",
                "craft_weak_text_area_good_percent",
                "craft_strong_text_area_score",
                "craft_strong_text_area_raw",
                "craft_strong_text_area_good_percent",
                "craft_peak_text",
                "craft_peak_affinity",
                "craft_affinity_area_raw",
            )
            if not all(key in m and m.get(key) is not None for key in required_keys):
                return (
                    "Table values (CRAFT heatmap):\n"
                    + f"  Rank: {rank_text} | Crop #: {crop_number}\n"
                )

            return (
                "Existing terms use per-component caps before counting/summing, so one large word cannot dominate only by area.\n"
                + f"  Rank: {rank_text} | Crop #: {crop_number}\n"
                + f"  Text Sum:      {float(m['craft_text_density_score']):.3f}  "
                + f"(capped sum: {100.0*float(m['craft_text_density_raw']):.3f}%, good at {float(m['craft_text_density_good_percent']):.1f}%)\n"
                + f"  Affinity Sum:  {float(m['craft_affinity_density_score']):.3f}  "
                + f"(capped sum: {100.0*float(m['craft_affinity_density_raw']):.3f}%, good at {float(m['craft_affinity_density_good_percent']):.1f}%)\n"
                + f"  Weak Area:     {float(m['craft_weak_text_area_score']):.3f}  "
                + f"(capped: {100.0*float(m['craft_weak_text_area_raw']):.3f}%, uncapped: {100.0*float(m.get('craft_weak_text_area_uncapped_raw', m['craft_weak_text_area_raw'])):.3f}%)\n"
                + f"  Strong Area:   {float(m['craft_strong_text_area_score']):.3f}  "
                + f"(capped: {100.0*float(m['craft_strong_text_area_raw']):.3f}%, uncapped: {100.0*float(m.get('craft_strong_text_area_uncapped_raw', m['craft_strong_text_area_raw'])):.3f}%)\n"
                + f"  Text Peak:     {float(m['craft_peak_text']):.3f}  "
                + f"(score: {float(m.get('craft_peak_text_score', clamp01(float(m['craft_peak_text'])))):.3f})\n"
                + f"  Affinity Peak: {float(m['craft_peak_affinity']):.3f}  "
                + f"(score: {float(m.get('craft_peak_affinity_score', clamp01(float(m['craft_peak_affinity'])))):.3f})\n"
                + f"  Heatmap:       {m.get('craft_heatmap_width')}x{m.get('craft_heatmap_height')}\n"
            )

        if ranking_method == "yolo":
            return (
                "Table values (YOLO detector/crop):\n"
                + f"  Rank: {rank_text} | Crop #: {crop_number} | Rank Score: {'not available' if r.score < 0 else f'{r.score:.2f}'}\n"
                + f"  YOLO Q: {self.model._fmt(m.get('yolo_quality_score_0_to_100'), 2)} | "
                + f"Det: {self.model._fmt(m.get('yolo_detector_score'), 3)} | "
                + f"Y-Sharp: {self.model._fmt(m.get('yolo_sharpness_score'), 3)} | "
                + f"Y-Area: {self.model._fmt(m.get('yolo_area_score'), 3)}\n"
                + f"  Y-Aspect: {self.model._fmt(m.get('yolo_aspect_score'), 3)} | "
                + f"Y-Edge: {self.model._fmt(m.get('yolo_edge_penalty'), 3)} | "
                + f"Track: {m.get('yolo_track_id', '—')} | Cand: {m.get('yolo_candidate_index', '—')} | "
                + f"Frame: {m.get('yolo_frame_index', '—')}\n"
            )

        return (
            "Table values (OCR proxy):\n"
            + f"  Rank: {rank_text} | Crop #: {crop_number} | Rank Score: {'not available' if r.score < 0 else f'{r.score:.2f}'}\n"
            + f"  OCR: {self.model._fmt(m.get('ocr_quality_score'), 2)} | "
            + f"O-Sharp: {self.model._fmt(m.get('sharpness_score'), 2)} | "
            + f"Contrast: {self.model._fmt(m.get('contrast_score'), 2)} | "
            + f"Edges: {self.model._fmt(m.get('edge_density_score'), 2)}\n"
            + f"  Text px: {self.model._fmt(m.get('foreground_score'), 2)} | "
            + f"Components: {self.model._fmt(m.get('component_score'), 2)} | "
            + f"Resolution: {self.model._fmt(m.get('resolution_score'), 2)}\n"
        )

    def selected_crop_overall_score(self, r: CropScoreResult) -> str:
        m = r.metrics
        ranking_method = str(m.get("ranking_method", "yolo"))
        if r.score < 0:
            return "Overall score: not available\n\n"
        if ranking_method == "craft":
            return f"Overall CRAFT heatmap score: {float(m.get('craft_heatmap_score', r.score)):.2f} / 100\n\n"
        if ranking_method == "yolo":
            return f"Overall YOLO detector/crop score: {r.score:.2f} / 100\n\n"
        return f"Overall OCR-proxy score: {float(m.get('ocr_quality_score', r.score)):.2f} / 100\n\n"

    def crop_info_block(self, r: CropScoreResult) -> str:
        m = r.metrics
        return (
            "Crop info:\n"
            + f"  File: {r.path.name}\n"
            + f"  Path: {r.path}\n"
            + f"  Size: {r.width} x {r.height}\n"
            + f"  Crop #: {m.get('crop_number') if m.get('crop_number') is not None else '—'}\n"
            + f"  Ranking method: {m.get('ranking_method', '—')}\n\n"
        )

    def format_details(self, r: CropScoreResult) -> str:
        m = r.metrics
        ranking_method = str(m.get("ranking_method", "yolo"))
        yolo_available = bool(m.get("yolo_score_available"))
        top = self.selected_crop_table_values(r) + self.selected_crop_overall_score(r) + self.crop_info_block(r) + self.format_result_timing_block(r)

        if ranking_method == "yolo":
            if not m.get("yolo_metadata_found"):
                return (
                    top
                    + "YOLO metadata was not found for this image.\n"
                    + "Choose the YOLO run folder, a track folder, or a crops folder that still has detector_score_metadata.json/csv nearby.\n\n"
                    + self.yolo_factor_descriptions()
                )
            if not yolo_available:
                return (
                    top
                    + "YOLO metadata was found, but detector_score/score is missing.\n"
                    + f"Metadata: {m.get('yolo_metadata_path', '')}\n\n"
                    + self.yolo_factor_descriptions()
                )

            saved = m.get("yolo_saved_quality_0_to_1")
            saved_text = "not used / not present" if saved is None else f"{float(saved):.6f} ({100.0*float(saved):.2f}/100)"
            diff = m.get("yolo_quality_abs_diff_from_saved")
            diff_text = "—" if diff is None else f"{float(diff):.9f}"
            return (
                top
                + "YOLO formula:\n"
                + f"  {float(m['yolo_weight_detector_percent']):.1f}% * detector_score\n"
                + f"+ {float(m['yolo_weight_sharpness_percent']):.1f}% * sharpness_score\n"
                + f"+ {float(m['yolo_weight_area_percent']):.1f}% * area_score\n"
                + f"+ {float(m['yolo_weight_aspect_percent']):.1f}% * aspect_score\n"
                + f"- {float(m['yolo_edge_penalty_amount_percent']):.1f}% edge penalty when crop touches frame edge\n\n"
                + "YOLO details:\n"
                + f"  Final YOLO quality: {float(m['yolo_quality_score_0_to_1']):.6f} "
                + f"({float(m['yolo_quality_score_0_to_100']):.2f}/100)\n"
                + f"  Saved metadata quality: {saved_text}\n"
                + f"  Absolute difference from saved value: {diff_text}\n"
                + f"  Detector score: {float(m['yolo_detector_score']):.6f}\n"
                + f"  Sharpness score: {float(m['yolo_sharpness_score']):.6f} "
                + f"(raw Laplacian variance: {float(m['yolo_laplacian_variance']):.2f})\n"
                + f"  Area score: {float(m['yolo_area_score']):.6f} "
                + f"(crop pixels {int(m['yolo_crop_width_from_pixels'])}x{int(m['yolo_crop_height_from_pixels'])}; "
                + f"frame {int(m['yolo_frame_width'])}x{int(m['yolo_frame_height'])})\n"
                + f"  Aspect score: {float(m['yolo_aspect_score']):.6f}\n"
                + f"  Edge penalty applied: {float(m['yolo_edge_penalty']):.6f}\n\n"
                + "Metadata identity:\n"
                + f"  Track: {m.get('yolo_track_id')} | Candidate: {m.get('yolo_candidate_index')} | "
                + f"Frame: {m.get('yolo_frame_index')} | Best in track: {bool(m.get('yolo_is_best'))}\n"
                + f"  BBox: {m.get('yolo_bbox', '')}\n"
                + f"  Raw bbox: {m.get('yolo_raw_bbox', '')}\n"
                + f"  Metadata: {m.get('yolo_metadata_path', '')}\n\n"
                + self.yolo_factor_descriptions()
            )

        if ranking_method == "craft":
            return (
                top
                + "CRAFT formula:\n"
                + f"  100 * ({float(m.get('craft_weight_text_sum_percent', 50.0))/100.0:.3f}*Text Sum "
                + f"+ {float(m.get('craft_weight_affinity_sum_percent', 20.0))/100.0:.3f}*Affinity Sum "
                + f"+ {float(m.get('craft_weight_weak_area_percent', 15.0))/100.0:.3f}*Weak Area "
                + f"+ {float(m.get('craft_weight_strong_area_percent', 10.0))/100.0:.3f}*Strong Area "
                + f"+ {float(m.get('craft_weight_peak_text_percent', 5.0))/100.0:.3f}*Peak Text "
                + f"+ {float(m.get('craft_weight_peak_affinity_percent', 0.0))/100.0:.3f}*Peak Affinity)\n"
                + f"  Raw editable weight total: {float(m.get('craft_weight_total_raw_percent', 100.0)):.1f}% "
                + "(weights are normalized during scoring)\n\n"
                + "CRAFT settings:\n"
                + f"  canvas_size: {int(m['craft_canvas_size'])}\n"
                + f"  mag_ratio: {float(m['craft_mag_ratio']):.3f}\n"
                + f"  text_threshold: {float(m['craft_text_threshold']):.3f}\n"
                + f"  low_text: {float(m['craft_low_text']):.3f}\n"
                + f"  link_threshold: {float(m['craft_link_threshold']):.3f}\n\n"
                + self.craft_factor_descriptions()
            )

        return (
            top
            + "OCR-proxy formula:\n"
            + f"  Base = {m['weight_sharpness_percent']:.1f}%*sharpness "
            + f"+ {m['weight_contrast_percent']:.1f}%*contrast "
            + f"+ {m['weight_edge_density_percent']:.1f}%*edge_density\n"
            + f"       + {m['weight_foreground_percent']:.1f}%*text_pixel_mass "
            + f"+ {m['weight_components_percent']:.1f}%*components "
            + f"+ {m['weight_resolution_percent']:.1f}%*resolution\n"
            + "  Final = 100 * Base * border_cut_penalty * saturation_penalty\n\n"
            + "OCR-proxy terms:\n"
            + f"  Selected weight total: {m['weight_total_raw_percent']:.1f}% "
            + f"(effective weights above sum to 100%)\n"
            + f"  Sharpness:       {m['sharpness_score']:.3f}  "
            + f"(Laplacian variance: {m['laplacian_variance']:.2f})\n"
            + f"  Contrast:        {m['contrast_score']:.3f}  "
            + f"(p95-p5: {m['contrast_p95_minus_p5']:.2f}, std: {m['contrast_std']:.2f})\n"
            + f"  Edge density:    {m['edge_density_score']:.3f}  "
            + f"(raw edge density: {100*m['edge_density']:.2f}%)\n"
            + f"  Text-pixel mass: {m['foreground_score']:.3f}  "
            + f"(foreground: {100*m['foreground_fraction']:.2f}%)\n"
            + f"  Components:      {m['component_score']:.3f}  "
            + f"(valid components: {int(m['valid_components'])}, "
            + f"density: {m['component_density_per_100k_px']:.2f}/100k px)\n"
            + f"  Resolution:      {m['resolution_score']:.3f}\n\n"
            + "OCR-proxy penalties:\n"
            + f"  Border cut penalty: {m['border_cut_penalty']:.3f}  "
            + f"(border/center foreground ratio: {m['border_to_center_foreground_ratio']:.2f})\n"
            + f"    Settings: border width {m['border_region_percent']:.1f}%, max penalty {m['border_penalty_max_percent']:.1f}%, "
            + f"start {m['border_ratio_start']:.3f}, range {m['border_ratio_range']:.3f}\n"
            + f"  Saturation penalty: {m['saturation_penalty']:.3f}  "
            + f"(near-black/near-white pixels: {100*m['saturation_fraction']:.2f}%)\n"
            + f"    Settings: max penalty {m['saturation_penalty_max_percent']:.1f}%, start {m['saturation_start_percent']:.1f}%, "
            + f"range {m['saturation_range_percent']:.1f}%\n\n"
            + self.ocr_factor_descriptions()
        )

    def copied_heatmap_evidence_terms(self, result: CropScoreResult) -> str:
        return self.selected_crop_table_values(result)

    def format_details_for_selected_panel(self, result: CropScoreResult) -> str:
        return self.format_details(result)

    def export_csv(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Nothing to export", "No crop scores are available yet.")
            return

        default_path = self.current_folder / "crop_quality_scores.csv" if self.current_folder else Path("crop_quality_scores.csv")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export score CSV",
            str(default_path),
            "CSV files (*.csv)",
        )
        if not filename:
            return

        out_path = Path(filename)
        metric_keys = sorted(self.results[0].metrics.keys())

        def csv_value(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float, np.integer, np.floating)):
                return f"{float(value):.9f}"
            return str(value)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "ranking_score", "file", "path", "width", "height", *metric_keys])
            for r in self.results:
                rank = self.model.default_rank_for_result(r) or ""
                writer.writerow(
                    [
                        rank,
                        f"{r.score:.6f}",
                        r.path.name,
                        str(r.path),
                        r.width,
                        r.height,
                        *[csv_value(r.metrics.get(k)) for k in metric_keys],
                    ]
                )

        QMessageBox.information(self, "Export complete", f"Saved:\n{out_path}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.cancel()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(1500)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("OCR Crop Quality Lab")
    app.setStyle("Fusion")
    apply_compact_ui_font(app)
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
