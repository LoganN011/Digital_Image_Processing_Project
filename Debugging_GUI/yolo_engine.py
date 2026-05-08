from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    cv2.setNumThreads(1)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "yolo_best.pt")
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.30
DEFAULT_MAX_DET = 30
DEFAULT_IMGSZ = 224
DEFAULT_FRAME_SKIP = 3
DEFAULT_PAD_RATIO = 0.04
DEFAULT_BOX_PAD_PIXELS = 1
DEFAULT_MIN_STABLE_FRAMES = 3
DEFAULT_MIN_AREA_RATIO = 0.0005
DEFAULT_TRACKER_NAME = "bytetrack.yaml"
MAX_STORED_CROPS_PER_TRACK = 0  # 0 means keep every accepted crop

SCORING_METHOD_HEATMAP = "heatmap"
SCORING_METHOD_OCR_PROXY = "ocr_proxy"
SCORING_METHOD_DETECTOR_GEOM = "detector_geom"
SCORING_METHOD_LEGACY_SHARP_AREA = "legacy_sharp_area"

DEFAULT_CROP_SCORING_METHOD = SCORING_METHOD_LEGACY_SHARP_AREA
DEFAULT_MIN_CROP_QUALITY = 0.05

DEFAULT_OCR_PROXY_WEIGHTS = {
    "sharpness": 24.0,
    "contrast": 18.0,
    "edge_density": 16.0,
    "foreground": 17.0,
    "components": 0.0,
    "resolution": 50.0,
}
OCR_PROXY_WEIGHT_ORDER = ("sharpness", "contrast", "edge_density", "foreground", "components", "resolution")

DEFAULT_OCR_PROXY_REDUCERS = {
    "border_region_percent": 4.5,
    "border_penalty_max_percent": 28.0,
    "border_ratio_start": 0.80,
    "border_ratio_range": 2.20,
    "saturation_penalty_max_percent": 18.0,
    "saturation_start_percent": 18.0,
    "saturation_range_percent": 62.0,
}

DEFAULT_DETECTOR_GEOM_SCORE_WEIGHTS = {
    "detector": 42.0,
    "sharpness": 33.0,
    "area": 18.0,
    "aspect": 7.0,
    "edge_penalty": 16.0,
}
DETECTOR_GEOM_SCORE_WEIGHT_ORDER = ("detector", "sharpness", "area", "aspect", "edge_penalty")
# Deprecated names kept for external code that may import them.
DEFAULT_YOLO_SCORE_WEIGHTS = DEFAULT_DETECTOR_GEOM_SCORE_WEIGHTS
YOLO_SCORE_WEIGHT_ORDER = DETECTOR_GEOM_SCORE_WEIGHT_ORDER

DEFAULT_CRAFT_HEATMAP_SETTINGS = {
    # 1.0 means "try CUDA if available, otherwise safely fall back to CPU".
    "gpu": 1.0,
    "canvas_size": 768.0,
    "mag_ratio": 1.0,
    "batch_size": 16.0,
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
CRAFT_HEATMAP_WEIGHT_KEYS = (
    "craft_weight_text_sum_percent",
    "craft_weight_affinity_sum_percent",
    "craft_weight_weak_area_percent",
    "craft_weight_strong_area_percent",
    "craft_weight_peak_text_percent",
    "craft_weight_peak_affinity_percent",
)

SCORING_METHOD_ALIASES = {
    "craft": SCORING_METHOD_HEATMAP,
    "craft_heatmap": SCORING_METHOD_HEATMAP,
    "heatmap": SCORING_METHOD_HEATMAP,
    "ocr": SCORING_METHOD_OCR_PROXY,
    "ocr_proxy": SCORING_METHOD_OCR_PROXY,
    "proxy": SCORING_METHOD_OCR_PROXY,
    "detector_geom": SCORING_METHOD_DETECTOR_GEOM,
    "detector_geometry": SCORING_METHOD_DETECTOR_GEOM,
    "detector": SCORING_METHOD_DETECTOR_GEOM,
    # Backward-compatible old names.
    "yolo": SCORING_METHOD_DETECTOR_GEOM,
    "yolo_detector": SCORING_METHOD_DETECTOR_GEOM,
    "legacy_sharp_area": SCORING_METHOD_LEGACY_SHARP_AREA,
    "legacy_sharpness_area": SCORING_METHOD_LEGACY_SHARP_AREA,
    "sharp_area": SCORING_METHOD_LEGACY_SHARP_AREA,
    # Backward-compatible old names.
    "dino": SCORING_METHOD_LEGACY_SHARP_AREA,
    "dino_engine": SCORING_METHOD_LEGACY_SHARP_AREA,
}


def resolve_local_path(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else SCRIPT_DIR / p)


def clamp_box(box: Iterable[float], frame_shape: tuple[int, ...]) -> list[int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    return None if x2 <= x1 or y2 <= y1 else [x1, y1, x2, y2]


def expand_box(box: list[int], frame_shape: tuple[int, ...], pad_ratio: float, pad_pixels: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    px = max(int(round(max(1, x2 - x1) * pad_ratio)), pad_pixels)
    py = max(int(round(max(1, y2 - y1) * pad_ratio)), pad_pixels)
    return clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], frame_shape)


def crop_from_box(frame: np.ndarray, box: list[int]) -> np.ndarray | None:
    box = clamp_box(box, frame.shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    return None if crop.size == 0 else crop


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])



def normalize_scoring_method(method: str | None) -> str:
    return SCORING_METHOD_ALIASES.get(str(method or DEFAULT_CROP_SCORING_METHOD).strip().lower(), DEFAULT_CROP_SCORING_METHOD)


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


def normalized_ocr_proxy_weights(weights: Optional[dict[str, float]]) -> tuple[dict[str, float], dict[str, float], float]:
    merged = dict(DEFAULT_OCR_PROXY_WEIGHTS)
    if weights:
        merged.update(weights)
    raw = {key: max(0.0, float(merged.get(key, DEFAULT_OCR_PROXY_WEIGHTS[key]))) for key in OCR_PROXY_WEIGHT_ORDER}
    total = float(sum(raw.values()))
    if total <= 0.0:
        raw = dict(DEFAULT_OCR_PROXY_WEIGHTS)
        total = float(sum(raw.values()))
    normalized = {key: raw[key] / total for key in OCR_PROXY_WEIGHT_ORDER}
    return raw, normalized, total


def normalized_craft_heatmap_weights(settings: dict[str, float]) -> tuple[dict[str, float], dict[str, float], float]:
    raw = {key: max(0.0, float(settings.get(key, DEFAULT_CRAFT_HEATMAP_SETTINGS[key]))) for key in CRAFT_HEATMAP_WEIGHT_KEYS}
    total = float(sum(raw.values()))
    if total <= 0.0:
        raw = {key: max(0.0, float(DEFAULT_CRAFT_HEATMAP_SETTINGS[key])) for key in CRAFT_HEATMAP_WEIGHT_KEYS}
        total = float(sum(raw.values()))
    normalized = {key: raw[key] / max(total, 1e-9) for key in CRAFT_HEATMAP_WEIGHT_KEYS}
    return raw, normalized, total


def luminance_uint8(image_bgr: np.ndarray) -> np.ndarray:
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(y, 0, 255).astype(np.uint8)


def score_ocr_proxy_crop(image_bgr: np.ndarray, weights: Optional[dict[str, float]] = None) -> tuple[float, dict[str, Any]]:
    merged_weights = dict(DEFAULT_OCR_PROXY_REDUCERS)
    if weights:
        merged_weights.update(weights)
    raw_weights, effective_weights, weight_total = normalized_ocr_proxy_weights(merged_weights)
    h, w = image_bgr.shape[:2]
    area = float(max(1, w * h))
    gray = luminance_uint8(image_bgr)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = clamp01((math.log1p(lap_var) - math.log1p(18.0)) / (math.log1p(900.0) - math.log1p(18.0)))

    p5, p95 = np.percentile(gray, [5, 95])
    contrast_range = float(p95 - p5)
    contrast_std = float(gray.std())
    contrast_score = 0.55 * clamp01((contrast_range - 35.0) / 150.0) + 0.45 * clamp01((contrast_std - 12.0) / 58.0)

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

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    valid_components = 0
    total_valid_area = 0
    min_component_area = max(5, int(area * 0.000006))
    max_component_area = max(min_component_area + 1, int(area * 0.085))
    for i in range(1, n_labels):
        _x, _y, cw, ch, c_area = stats[i]
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
    resolution_score = 0.45 * clamp01((short_side - 90.0) / 310.0) + 0.55 * clamp01((long_side - 260.0) / 740.0)

    border_region_fraction = max(0.0, float(merged_weights.get("border_region_percent", DEFAULT_OCR_PROXY_REDUCERS["border_region_percent"]))) / 100.0
    border_penalty_max = max(0.0, float(merged_weights.get("border_penalty_max_percent", DEFAULT_OCR_PROXY_REDUCERS["border_penalty_max_percent"]))) / 100.0
    border_ratio_start = float(merged_weights.get("border_ratio_start", DEFAULT_OCR_PROXY_REDUCERS["border_ratio_start"]))
    border_ratio_range = max(1e-9, float(merged_weights.get("border_ratio_range", DEFAULT_OCR_PROXY_REDUCERS["border_ratio_range"])))

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

    saturation_penalty_max = max(0.0, float(merged_weights.get("saturation_penalty_max_percent", DEFAULT_OCR_PROXY_REDUCERS["saturation_penalty_max_percent"]))) / 100.0
    saturation_start = max(0.0, float(merged_weights.get("saturation_start_percent", DEFAULT_OCR_PROXY_REDUCERS["saturation_start_percent"]))) / 100.0
    saturation_range = max(1e-9, float(merged_weights.get("saturation_range_percent", DEFAULT_OCR_PROXY_REDUCERS["saturation_range_percent"]))) / 100.0
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
    return float(final_score), {
        "method": SCORING_METHOD_OCR_PROXY,
        "score_raw_0_to_1": float(weighted_base),
        "weight_total_raw_percent": float(weight_total),
        "weight_sharpness_raw_percent": raw_weights["sharpness"],
        "weight_contrast_raw_percent": raw_weights["contrast"],
        "weight_edge_density_raw_percent": raw_weights["edge_density"],
        "weight_foreground_raw_percent": raw_weights["foreground"],
        "weight_components_raw_percent": raw_weights["components"],
        "weight_resolution_raw_percent": raw_weights["resolution"],
        "sharpness_score": float(sharpness_score),
        "contrast_score": float(contrast_score),
        "edge_density_score": float(edge_density_score),
        "foreground_score": float(foreground_score),
        "component_score": float(component_score),
        "resolution_score": float(resolution_score),
        "border_cut_penalty": float(border_cut_penalty),
        "saturation_penalty": float(saturation_penalty),
        "laplacian_variance": float(lap_var),
        "contrast_p95_minus_p5": float(contrast_range),
        "contrast_std": float(contrast_std),
        "edge_density": float(edge_density),
        "foreground_fraction": float(foreground_fraction),
        "valid_components": int(valid_components),
        "valid_component_area_fraction": float(total_valid_area / area),
        "component_density_per_100k_px": float(component_density),
        "border_to_center_foreground_ratio": float(border_ratio),
        "saturation_fraction": float(saturation_fraction),
    }


def detector_geom_score_coefficients(weights: Optional[dict[str, float]] = None) -> dict[str, float]:
    merged = dict(DEFAULT_DETECTOR_GEOM_SCORE_WEIGHTS)
    if weights:
        merged.update(weights)
    return {
        key: max(0.0, float(merged.get(key, DEFAULT_DETECTOR_GEOM_SCORE_WEIGHTS[key]))) / 100.0
        for key in DETECTOR_GEOM_SCORE_WEIGHT_ORDER
    }


# Deprecated compatibility wrapper.
def yolo_score_coefficients(weights: Optional[dict[str, float]] = None) -> dict[str, float]:
    return detector_geom_score_coefficients(weights)


def score_detector_geom_crop(frame: np.ndarray, box: list[int], detector_score: float, weights: Optional[dict[str, float]] = None) -> tuple[float, dict[str, Any]]:
    crop = crop_from_box(frame, box)
    if crop is None:
        return -1.0, {"method": SCORING_METHOD_DETECTOR_GEOM, "score_available": False}
    fh, fw = frame.shape[:2]
    ch, cw = crop.shape[:2]
    if cw < 28 or ch < 28:
        return -1.0, {"method": SCORING_METHOD_DETECTOR_GEOM, "score_available": False, "too_small": True, "width": int(cw), "height": int(ch)}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_score = min(lap_var / 700.0, 1.0)
    area_score = min((cw * ch) / max(1.0, 0.20 * fw * fh), 1.0)
    aspect_score = 1.0 if 0.22 <= cw / max(1, ch) <= 7.0 else 0.45
    coeff = detector_geom_score_coefficients(weights)
    edge_penalty = coeff["edge_penalty"] if box[0] <= 2 or box[1] <= 2 or box[2] >= fw - 2 or box[3] >= fh - 2 else 0.0
    score_0_to_1 = (
        coeff["detector"] * float(detector_score)
        + coeff["sharpness"] * sharp_score
        + coeff["area"] * area_score
        + coeff["aspect"] * aspect_score
        - edge_penalty
    )
    return float(100.0 * score_0_to_1), {
        "method": SCORING_METHOD_DETECTOR_GEOM,
        "score_0_to_1": float(score_0_to_1),
        "detector_score": float(detector_score),
        "laplacian_variance": float(lap_var),
        "sharpness_score": float(sharp_score),
        "area_score": float(area_score),
        "aspect_score": float(aspect_score),
        "edge_penalty": float(edge_penalty),
        "crop_width": int(cw),
        "crop_height": int(ch),
        "frame_width": int(fw),
        "frame_height": int(fh),
    }


# Deprecated compatibility wrapper.
def score_yolo_detector_crop(frame: np.ndarray, box: list[int], detector_score: float, weights: Optional[dict[str, float]] = None) -> tuple[float, dict[str, Any]]:
    return score_detector_geom_crop(frame, box, detector_score, weights)


def capped_heatmap_energy_and_area(score_map: np.ndarray, threshold: float, *, cap_fraction: float = 0.006) -> tuple[float, float, float, int, int]:
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
    return float(capped_energy / n_pixels), float(capped_area / n_pixels), raw_area, int(component_count), int(capped_component_count)


class CraftHeatmapScorer:
    _reader_cache: dict[bool, tuple[Any, Any]] = {}

    def __init__(self, settings: Optional[dict[str, float]] = None) -> None:
        self.settings = dict(DEFAULT_CRAFT_HEATMAP_SETTINGS)
        if settings:
            self.settings.update(settings)
        self.reader = None
        self.device = None
        self._reader_cache_key: Optional[bool] = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @classmethod
    def reader_cache_key(cls, settings: dict[str, float]) -> bool:
        """Return True only when CRAFT should actually run on CUDA.

        The settings value still behaves like a user-facing switch. A positive
        value requests GPU, but this resolves to False on machines without CUDA
        so EasyOCR/CRAFT does not waste time trying an unsupported device.
        """
        try:
            requested_gpu = float(settings.get("gpu", DEFAULT_CRAFT_HEATMAP_SETTINGS["gpu"])) > 0.0
        except Exception:
            requested_gpu = False
        return bool(requested_gpu and cls._cuda_available())

    @classmethod
    def clear_reader_cache(cls) -> None:
        cls._reader_cache.clear()

    def update_settings(self, settings: Optional[dict[str, float]]) -> None:
        old_key = self.reader_cache_key(self.settings)
        self.settings = dict(DEFAULT_CRAFT_HEATMAP_SETTINGS)
        if settings:
            self.settings.update(settings)
        new_key = self.reader_cache_key(self.settings)
        if self.reader is not None and old_key != new_key:
            self.reader = None
            self.device = None
            self._reader_cache_key = None

    def _ensure_reader(self):
        use_gpu = self.reader_cache_key(self.settings)
        if self.reader is not None and self._reader_cache_key == use_gpu:
            return self.reader
        cached = self._reader_cache.get(use_gpu)
        if cached is not None:
            self.reader, self.device = cached
            self._reader_cache_key = use_gpu
            return self.reader
        import easyocr
        reader = easyocr.Reader(["en"], gpu=use_gpu, detector=True, recognizer=False, verbose=False)
        device = getattr(reader, "device", "cuda" if use_gpu else "cpu")
        self._reader_cache[use_gpu] = (reader, device)
        self.reader = reader
        self.device = device
        self._reader_cache_key = use_gpu
        return self.reader

    def _preprocess_for_craft(self, image_bgr: np.ndarray) -> np.ndarray:
        from easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        canvas_size = int(round(float(self.settings.get("canvas_size", DEFAULT_CRAFT_HEATMAP_SETTINGS["canvas_size"]))))
        mag_ratio = float(self.settings.get("mag_ratio", DEFAULT_CRAFT_HEATMAP_SETTINGS["mag_ratio"]))
        image_resized, _target_ratio, _ = resize_aspect_ratio(
            image_rgb,
            canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=mag_ratio,
        )
        x = normalizeMeanVariance(image_resized).astype(np.float32, copy=False)
        return np.transpose(x, (2, 0, 1))

    def heatmaps(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.heatmaps_batch([image_bgr])[0]

    def heatmaps_batch(self, images_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run CRAFT on a batch of crops and return text/affinity maps.

        The crops are resized individually using EasyOCR's normal CRAFT
        preprocessing, then padded inside each batch chunk so the network sees
        one tensor instead of one forward pass per crop. The output maps are
        cropped back to each crop's unpadded map size before scoring.
        """
        if not images_bgr:
            return []

        import torch

        reader = self._ensure_reader()
        detector = reader.detector
        detector.eval()

        prepared: list[tuple[int, np.ndarray, int, int]] = []
        for idx, image_bgr in enumerate(images_bgr):
            x_chw = self._preprocess_for_craft(image_bgr)
            prepared.append((idx, x_chw, int(x_chw.shape[1]), int(x_chw.shape[2])))

        try:
            batch_size = int(round(float(self.settings.get("batch_size", DEFAULT_CRAFT_HEATMAP_SETTINGS.get("batch_size", 16.0)))))
        except Exception:
            batch_size = 16
        batch_size = max(1, batch_size)

        outputs: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(images_bgr)
        with torch.no_grad():
            for start in range(0, len(prepared), batch_size):
                chunk = prepared[start : start + batch_size]
                max_h = max(item[2] for item in chunk)
                max_w = max(item[3] for item in chunk)
                batch_np = np.zeros((len(chunk), 3, max_h, max_w), dtype=np.float32)
                for local_idx, (_orig_idx, x_chw, h, w) in enumerate(chunk):
                    batch_np[local_idx, :, :h, :w] = x_chw

                x = torch.from_numpy(batch_np).to(self.device)
                y, _feature = detector(x)
                y_np = y.detach().cpu().numpy().astype(np.float32, copy=False)

                out_h_full = int(y_np.shape[1])
                out_w_full = int(y_np.shape[2])
                for local_idx, (orig_idx, _x_chw, h, w) in enumerate(chunk):
                    out_h = max(1, int(round(out_h_full * (h / max_h))))
                    out_w = max(1, int(round(out_w_full * (w / max_w))))
                    score_text = y_np[local_idx, :out_h, :out_w, 0].copy()
                    score_link = y_np[local_idx, :out_h, :out_w, 1].copy()
                    outputs[orig_idx] = (score_text, score_link)

        return [item if item is not None else (np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)) for item in outputs]


def _score_craft_heatmap_maps(
    score_text: np.ndarray,
    score_link: np.ndarray,
    merged_settings: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    score_text = np.nan_to_num(score_text, nan=0.0, posinf=1.0, neginf=0.0)
    score_link = np.nan_to_num(score_link, nan=0.0, posinf=1.0, neginf=0.0)

    text_threshold = float(merged_settings.get("text_threshold", DEFAULT_CRAFT_HEATMAP_SETTINGS["text_threshold"]))
    low_text = float(merged_settings.get("low_text", DEFAULT_CRAFT_HEATMAP_SETTINGS["low_text"]))
    link_threshold = float(merged_settings.get("link_threshold", DEFAULT_CRAFT_HEATMAP_SETTINGS["link_threshold"]))

    text_density_raw, weak_text_area_raw, weak_text_area_uncapped_raw, weak_component_count, weak_components_capped = capped_heatmap_energy_and_area(score_text, low_text, cap_fraction=0.006)
    _strong_energy_raw, strong_text_area_raw, strong_text_area_uncapped_raw, strong_component_count, strong_components_capped = capped_heatmap_energy_and_area(score_text, text_threshold, cap_fraction=0.004)
    affinity_density_raw, affinity_area_raw, affinity_area_uncapped_raw, affinity_component_count, affinity_components_capped = capped_heatmap_energy_and_area(score_link, link_threshold, cap_fraction=0.006)
    peak_text = float(np.max(score_text)) if score_text.size else 0.0
    peak_affinity = float(np.max(score_link)) if score_link.size else 0.0

    text_density_good = max(1e-6, float(merged_settings.get("text_density_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["text_density_good_percent"]))) / 100.0
    affinity_density_good = max(1e-6, float(merged_settings.get("affinity_density_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["affinity_density_good_percent"]))) / 100.0
    weak_area_good = max(1e-6, float(merged_settings.get("weak_text_area_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["weak_text_area_good_percent"]))) / 100.0
    strong_area_good = max(1e-6, float(merged_settings.get("strong_text_area_good_percent", DEFAULT_CRAFT_HEATMAP_SETTINGS["strong_text_area_good_percent"]))) / 100.0

    text_density_score = clamp01(text_density_raw / text_density_good)
    affinity_density_score = clamp01(affinity_density_raw / affinity_density_good)
    weak_text_area_score = clamp01(weak_text_area_raw / weak_area_good)
    strong_text_area_score = clamp01(strong_text_area_raw / strong_area_good)
    peak_text_score = clamp01(peak_text)
    peak_affinity_score = clamp01(peak_affinity)
    raw_craft_weights, effective_craft_weights, craft_weight_total = normalized_craft_heatmap_weights(merged_settings)

    final_score = 100.0 * (
        effective_craft_weights["craft_weight_text_sum_percent"] * text_density_score
        + effective_craft_weights["craft_weight_affinity_sum_percent"] * affinity_density_score
        + effective_craft_weights["craft_weight_weak_area_percent"] * weak_text_area_score
        + effective_craft_weights["craft_weight_strong_area_percent"] * strong_text_area_score
        + effective_craft_weights["craft_weight_peak_text_percent"] * peak_text_score
        + effective_craft_weights["craft_weight_peak_affinity_percent"] * peak_affinity_score
    )
    return float(final_score), {
        "method": SCORING_METHOD_HEATMAP,
        "craft_weight_total_raw_percent": float(craft_weight_total),
        "craft_weight_text_sum_raw_percent": raw_craft_weights["craft_weight_text_sum_percent"],
        "craft_weight_affinity_sum_raw_percent": raw_craft_weights["craft_weight_affinity_sum_percent"],
        "craft_weight_weak_area_raw_percent": raw_craft_weights["craft_weight_weak_area_percent"],
        "craft_weight_strong_area_raw_percent": raw_craft_weights["craft_weight_strong_area_percent"],
        "craft_weight_peak_text_raw_percent": raw_craft_weights["craft_weight_peak_text_percent"],
        "craft_weight_peak_affinity_raw_percent": raw_craft_weights["craft_weight_peak_affinity_percent"],
        "craft_text_density_raw": float(text_density_raw),
        "craft_affinity_density_raw": float(affinity_density_raw),
        "craft_weak_text_area_raw": float(weak_text_area_raw),
        "craft_strong_text_area_raw": float(strong_text_area_raw),
        "craft_affinity_area_raw": float(affinity_area_raw),
        "craft_weak_text_area_uncapped_raw": float(weak_text_area_uncapped_raw),
        "craft_strong_text_area_uncapped_raw": float(strong_text_area_uncapped_raw),
        "craft_affinity_area_uncapped_raw": float(affinity_area_uncapped_raw),
        "craft_weak_component_count": int(weak_component_count),
        "craft_strong_component_count": int(strong_component_count),
        "craft_affinity_component_count": int(affinity_component_count),
        "craft_weak_components_capped": int(weak_components_capped),
        "craft_strong_components_capped": int(strong_components_capped),
        "craft_affinity_components_capped": int(affinity_components_capped),
        "craft_text_density_score": float(text_density_score),
        "craft_affinity_density_score": float(affinity_density_score),
        "craft_weak_text_area_score": float(weak_text_area_score),
        "craft_strong_text_area_score": float(strong_text_area_score),
        "craft_peak_text_score": float(peak_text_score),
        "craft_peak_affinity_score": float(peak_affinity_score),
        "craft_peak_text": float(peak_text),
        "craft_peak_affinity": float(peak_affinity),
        "craft_heatmap_width": int(score_text.shape[1]) if score_text.ndim >= 2 else None,
        "craft_heatmap_height": int(score_text.shape[0]) if score_text.ndim >= 2 else None,
        "craft_text_threshold": float(text_threshold),
        "craft_low_text": float(low_text),
        "craft_link_threshold": float(link_threshold),
        "craft_canvas_size": int(round(float(merged_settings.get("canvas_size", DEFAULT_CRAFT_HEATMAP_SETTINGS["canvas_size"])))),
        "craft_mag_ratio": float(merged_settings.get("mag_ratio", DEFAULT_CRAFT_HEATMAP_SETTINGS["mag_ratio"])),
        "craft_batch_size": int(round(float(merged_settings.get("batch_size", DEFAULT_CRAFT_HEATMAP_SETTINGS.get("batch_size", 16.0))))),
        "craft_requested_gpu": float(merged_settings.get("gpu", DEFAULT_CRAFT_HEATMAP_SETTINGS["gpu"])),
        "craft_text_density_good_percent": float(100.0 * text_density_good),
        "craft_affinity_density_good_percent": float(100.0 * affinity_density_good),
        "craft_weak_text_area_good_percent": float(100.0 * weak_area_good),
        "craft_strong_text_area_good_percent": float(100.0 * strong_area_good),
    }


def score_craft_heatmap_crop(image_bgr: np.ndarray, settings: Optional[dict[str, float]] = None, scorer: Optional[CraftHeatmapScorer] = None) -> tuple[float, dict[str, Any]]:
    merged_settings = dict(DEFAULT_CRAFT_HEATMAP_SETTINGS)
    if settings:
        merged_settings.update(settings)
    if scorer is None:
        scorer = CraftHeatmapScorer(merged_settings)
    score_text, score_link = scorer.heatmaps(image_bgr)
    return _score_craft_heatmap_maps(score_text, score_link, merged_settings)


def score_craft_heatmap_crops_batch(
    images_bgr: list[np.ndarray],
    settings: Optional[dict[str, float]] = None,
    scorer: Optional[CraftHeatmapScorer] = None,
) -> list[tuple[float, dict[str, Any]]]:
    merged_settings = dict(DEFAULT_CRAFT_HEATMAP_SETTINGS)
    if settings:
        merged_settings.update(settings)
    if scorer is None:
        scorer = CraftHeatmapScorer(merged_settings)
    heatmap_pairs = scorer.heatmaps_batch(images_bgr)
    return [_score_craft_heatmap_maps(score_text, score_link, merged_settings) for score_text, score_link in heatmap_pairs]


def score_legacy_sharp_area_crop(crop: np.ndarray, box: list[int], frame_shape: tuple[int, ...]) -> tuple[float, dict[str, Any]]:
    fh, fw = frame_shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    max_center_dist = max(1.0, ((fw / 2) ** 2 + (fh / 2) ** 2) ** 0.5)
    center_penalty = ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5 / max_center_dist
    area = max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))
    score = float(sharp) + area * 0.01 - center_penalty * 300.0
    return float(score), {
        "method": SCORING_METHOD_LEGACY_SHARP_AREA,
        "laplacian_variance": float(sharp),
        "area": int(area),
        "center_penalty": float(center_penalty),
        "center_x": float(cx),
        "center_y": float(cy),
        "frame_width": int(fw),
        "frame_height": int(fh),
    }


# Deprecated compatibility wrapper.
def score_dino_crop(crop: np.ndarray, box: list[int], frame_shape: tuple[int, ...]) -> tuple[float, dict[str, Any]]:
    return score_legacy_sharp_area_crop(crop, box, frame_shape)


def score_crop_candidate(
    *,
    frame: np.ndarray,
    raw_box: list[int],
    crop_box: list[int],
    crop: np.ndarray,
    detector_score: float,
    method: str = DEFAULT_CROP_SCORING_METHOD,
    ocr_proxy_weights: Optional[dict[str, float]] = None,
    yolo_score_weights: Optional[dict[str, float]] = None,
    detector_geom_score_weights: Optional[dict[str, float]] = None,
    craft_heatmap_settings: Optional[dict[str, float]] = None,
    craft_scorer: Optional[CraftHeatmapScorer] = None,
) -> tuple[float, dict[str, Any]]:
    method = normalize_scoring_method(method)
    if method == SCORING_METHOD_DETECTOR_GEOM:
        score, details = score_detector_geom_crop(
            frame,
            crop_box,
            detector_score,
            detector_geom_score_weights if detector_geom_score_weights is not None else yolo_score_weights,
        )
    elif method == SCORING_METHOD_OCR_PROXY:
        score, details = score_ocr_proxy_crop(crop, ocr_proxy_weights)
    elif method == SCORING_METHOD_LEGACY_SHARP_AREA:
        score, details = score_legacy_sharp_area_crop(crop, raw_box, frame.shape)
    else:
        score, details = score_craft_heatmap_crop(crop, craft_heatmap_settings, craft_scorer)
    details = dict(details)
    details["method"] = method
    details["score"] = float(score)
    return float(score), details

def make_crop_candidate(
    *,
    candidate_index: int,
    frame_idx: int,
    raw_box: list[int],
    crop_box: list[int],
    crop: np.ndarray,
    quality: float,
    detector_score: float,
    quality_method: str = DEFAULT_CROP_SCORING_METHOD,
    quality_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = crop_box
    return {
        "candidate_index": int(candidate_index),
        "frame_index": int(frame_idx),
        "bbox": list(crop_box),
        "raw_bbox": list(raw_box),
        "quality": float(quality),
        "quality_method": normalize_scoring_method(quality_method),
        "quality_details": dict(quality_details or {}),
        "score": float(detector_score),
        "area": int(box_area(crop_box)),
        "width": int(max(0, x2 - x1)),
        "height": int(max(0, y2 - y1)),
        "is_best": False,
        "crop": crop.copy(),
    }


def refresh_candidate_flags(rec: "CropRecord") -> None:
    for candidate in rec.all_crops:
        candidate["is_best"] = int(candidate.get("candidate_index", -1)) == int(rec.best_candidate_index)


def trim_stored_candidates(rec: "CropRecord") -> None:
    if MAX_STORED_CROPS_PER_TRACK and len(rec.all_crops) > MAX_STORED_CROPS_PER_TRACK:
        rec.all_crops = rec.all_crops[-MAX_STORED_CROPS_PER_TRACK:]


def sync_best_to_highest_quality_candidate(rec: "CropRecord") -> bool:
    if not rec.all_crops:
        return False

    old_best_index = int(rec.best_candidate_index)
    best = max(
        rec.all_crops,
        key=lambda item: (
            float(item.get("quality", float("-inf"))),
            int(item.get("candidate_index", 0)),
        ),
    )

    rec.best_candidate_index = int(best.get("candidate_index", old_best_index))
    rec.best_quality = float(best.get("quality", rec.best_quality))
    rec.best_quality_method = str(best.get("quality_method", rec.best_quality_method))
    rec.best_quality_details = dict(best.get("quality_details", rec.best_quality_details) or {})
    rec.best_score = float(best.get("score", rec.best_score))
    rec.best_box = list(best.get("bbox", rec.best_box))
    if isinstance(best.get("crop"), np.ndarray):
        rec.best_crop = best["crop"].copy()
    refresh_candidate_flags(rec)
    return rec.best_candidate_index != old_best_index


def detector_preview(frame: np.ndarray, boxes: list["TrackBox"], long_side: int = 480) -> np.ndarray:
    preview = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"ID {box.track_id} {box.score:.2f}"
        cv2.putText(preview, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    h, w = preview.shape[:2]
    longest = max(h, w)
    if longest > long_side:
        scale = long_side / float(longest)
        preview = cv2.resize(preview, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return preview


@dataclass
class TrackBox:
    xyxy: list[int]
    score: float
    label: str
    cls_id: int
    track_id: int
    frame_index: int


@dataclass
class CropRecord:
    object_id: int
    label: str
    cls_id: int
    best_crop: np.ndarray
    best_box: list[int]
    best_quality: float
    best_score: float
    seen_count: int
    version: int
    first_frame: int
    last_seen_frame: int
    ready_for_ocr: bool = False
    best_quality_method: str = DEFAULT_CROP_SCORING_METHOD
    best_quality_details: dict[str, Any] = field(default_factory=dict)
    all_crops: list[dict[str, Any]] = field(default_factory=list)
    best_candidate_index: int = 1

    def summary(self, include_crop: bool = True, include_crop_history: bool = True) -> dict[str, Any]:
        refresh_candidate_flags(self)
        out = {
            "id": self.object_id,
            "track_id": self.object_id,
            "label": self.label,
            "cls_id": self.cls_id,
            "bbox": list(self.best_box),
            "quality": float(self.best_quality),
            "quality_method": str(self.best_quality_method),
            "quality_details": dict(self.best_quality_details),
            "score": float(self.best_score),
            "seen_count": int(self.seen_count),
            "version": int(self.version),
            "first_frame": int(self.first_frame),
            "last_seen_frame": int(self.last_seen_frame),
            "ready_for_ocr": bool(self.ready_for_ocr),
            "best_candidate_index": int(self.best_candidate_index),
            "num_crop_candidates": int(len(self.all_crops)),
        }
        if include_crop:
            out["crop"] = self.best_crop.copy()

        if include_crop_history:
            crop_history = []
            for candidate in self.all_crops:
                item = {k: v for k, v in candidate.items() if k != "crop"}
                if include_crop and isinstance(candidate.get("crop"), np.ndarray):
                    item["crop"] = candidate["crop"].copy()
                crop_history.append(item)
            out["all_crops"] = crop_history
        else:
            out["all_crops"] = []
            out["crop_history_deferred"] = True
        return out


class YOLOEngine(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)
    poster_found_record = pyqtSignal(dict)
    finished_records = pyqtSignal(list)
    frame_preview = pyqtSignal(object)

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        box_pad: int = DEFAULT_BOX_PAD_PIXELS,
        pad_ratio: float = DEFAULT_PAD_RATIO,
        max_det: int = DEFAULT_MAX_DET,
        class_names: list[str] | tuple[str, ...] | None = None,
        tracker_name: str = DEFAULT_TRACKER_NAME,
        min_stable_frames: int = DEFAULT_MIN_STABLE_FRAMES,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
        rebuild_tracker_on_start: bool = True,
        require_track_id: bool = True,
        crop_scoring_method: str = DEFAULT_CROP_SCORING_METHOD,
        min_crop_quality: float = DEFAULT_MIN_CROP_QUALITY,
        ocr_proxy_weights: Optional[dict[str, float]] = None,
        yolo_score_weights: Optional[dict[str, float]] = None,
        detector_geom_score_weights: Optional[dict[str, float]] = None,
        craft_heatmap_settings: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        self.model_path = resolve_local_path(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.frame_skip = max(1, int(frame_skip))
        self.box_pad = int(box_pad)
        self.pad_ratio = float(pad_ratio)
        self.max_det = int(max_det)
        self.class_names = list(class_names) if class_names else None
        self.tracker_name = str(tracker_name)
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.min_area_ratio = float(min_area_ratio)
        self.rebuild_tracker_on_start = bool(rebuild_tracker_on_start)
        self.require_track_id = bool(require_track_id)
        self.crop_scoring_method = normalize_scoring_method(crop_scoring_method)
        self.min_crop_quality = float(min_crop_quality)
        self.ocr_proxy_weights = dict(DEFAULT_OCR_PROXY_WEIGHTS)
        self.ocr_proxy_weights.update(DEFAULT_OCR_PROXY_REDUCERS)
        if ocr_proxy_weights:
            self.ocr_proxy_weights.update(ocr_proxy_weights)
        self.detector_geom_score_weights = dict(DEFAULT_DETECTOR_GEOM_SCORE_WEIGHTS)
        # yolo_score_weights is accepted as a deprecated alias for older callers.
        if yolo_score_weights:
            self.detector_geom_score_weights.update(yolo_score_weights)
        if detector_geom_score_weights:
            self.detector_geom_score_weights.update(detector_geom_score_weights)
        self.yolo_score_weights = self.detector_geom_score_weights
        self.craft_heatmap_settings = dict(DEFAULT_CRAFT_HEATMAP_SETTINGS)
        if craft_heatmap_settings:
            self.craft_heatmap_settings.update(craft_heatmap_settings)
        self.craft_heatmap_scorer = CraftHeatmapScorer(self.craft_heatmap_settings) if self.crop_scoring_method == SCORING_METHOD_HEATMAP else None
        self._scoring_error_reported = False
        self.model = None
        self.loaded_model_path: str | None = None
        self.stop_requested = False
        self.crop_records: dict[int, CropRecord] = {}
        self.detect_fps_smooth = 0.0
        self._fallback_id_next = 10_000

    def request_stop(self):
        self.stop_requested = True

    def set_crop_scoring_method(self, method: str) -> None:
        self.crop_scoring_method = normalize_scoring_method(method)
        if self.crop_scoring_method == SCORING_METHOD_HEATMAP and self.craft_heatmap_scorer is None:
            self.craft_heatmap_scorer = CraftHeatmapScorer(self.craft_heatmap_settings)

    def reset_tracking(self, rebuild_model: bool = True):
        self.crop_records.clear()
        self.detect_fps_smooth = 0.0
        self._fallback_id_next = 10_000
        if rebuild_model:
            self.model = None
            self.loaded_model_path = None
        elif self.model is not None:
            try:
                setattr(self.model, "predictor", None)
            except Exception as exc:
                self.progress.emit(0, f"Tracker reset warning: {exc}")

    def load_model(self) -> bool:
        if YOLO is None:
            self.error.emit("Ultralytics YOLO is not installed. Install it with: pip install ultralytics")
            return False
        if self.model is not None and self.loaded_model_path == self.model_path:
            return True
        self.progress.emit(5, f"Loading YOLO model: {Path(self.model_path).name}")
        try:
            model = YOLO(self.model_path)
            try:
                model.fuse()
            except Exception:
                pass
        except Exception as exc:
            self.error.emit(f"Failed to load YOLO model: {exc}")
            return False
        self.model = model
        self.loaded_model_path = self.model_path
        self.progress.emit(8, f"Loaded YOLO model: {Path(self.model_path).name}")
        return True

    def allowed_class_ids(self) -> set[int] | None:
        if not self.class_names or self.model is None or not isinstance(getattr(self.model, "names", None), dict):
            return None
        wanted = {name.lower().strip() for name in self.class_names}
        ids = {int(k) for k, v in self.model.names.items() if str(v).lower().strip() in wanted}
        if not ids:
            self.progress.emit(9, f"Warning: class_names {self.class_names} did not match model.names; not filtering.")
        return ids or None

    def process_video(self, video_path: str):
        self.stop_requested = False
        self.reset_tracking(rebuild_model=self.rebuild_tracker_on_start)
        if not self.load_model():
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video file: {video_path}")
            return
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_area = max(1, width * height)
            allowed = self.allowed_class_ids()
            frame_idx = 0
            self.progress.emit(10, "Starting video processing...")
            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.frame_skip == 0:
                    self.process_frame(frame, frame_idx, allowed, frame_area)
                frame_idx += 1
                self.emit_frame_progress(frame_idx, total)
            self.finish()
        except Exception as exc:
            self.error.emit(f"Error during YOLO processing: {exc}")
        finally:
            cap.release()

    def process_frame(self, frame: np.ndarray, frame_idx: int, allowed: set[int] | None, frame_area: int):
        start = time.time()
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_name,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            self.error.emit(f"YOLO tracking failed on frame {frame_idx}: {exc}")
            self.stop_requested = True
            return
        elapsed = time.time() - start
        if elapsed > 0:
            inst = 1.0 / elapsed
            self.detect_fps_smooth = inst if self.detect_fps_smooth == 0 else 0.85 * self.detect_fps_smooth + 0.15 * inst
        boxes = self.parse_results(results, frame_idx, allowed, frame.shape)
        self.frame_preview.emit(detector_preview(frame, boxes))
        self.update_crop_records_for_frame(frame, boxes, frame_area)

    def parse_results(self, results: Any, frame_idx: int, allowed: set[int] | None, frame_shape: tuple[int, ...]) -> list[TrackBox]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.full(len(xyxy), -1)
        ids = boxes.id.detach().cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        out: list[TrackBox] = []
        for i, raw in enumerate(xyxy):
            cls_id = int(clss[i])
            if allowed is not None and cls_id not in allowed:
                continue
            fixed = clamp_box(raw[:4], frame_shape)
            if fixed is None:
                continue
            if ids is not None:
                track_id = int(ids[i])
            elif self.require_track_id:
                continue
            else:
                track_id = self._fallback_id_next
                self._fallback_id_next += 1
            label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
            out.append(TrackBox(fixed, float(confs[i]), label, cls_id, track_id, int(frame_idx)))
        return out

    def prepare_crop_for_box(self, frame: np.ndarray, box: TrackBox) -> tuple[list[int], np.ndarray] | None:
        expanded = expand_box(box.xyxy, frame.shape, self.pad_ratio, self.box_pad)
        if expanded is None:
            return None
        crop = crop_from_box(frame, expanded)
        if crop is None:
            return None
        return expanded, crop

    def update_crop_records_for_frame(self, frame: np.ndarray, boxes: list[TrackBox], frame_area: int) -> None:
        prepared: list[tuple[TrackBox, list[int], np.ndarray]] = []
        for box in boxes:
            if box_area(box.xyxy) < self.min_area_ratio * frame_area:
                continue
            item = self.prepare_crop_for_box(frame, box)
            if item is None:
                continue
            expanded, crop = item
            prepared.append((box, expanded, crop))

        if not prepared:
            return

        if self.crop_scoring_method == SCORING_METHOD_HEATMAP:
            if self.craft_heatmap_scorer is None:
                self.craft_heatmap_scorer = CraftHeatmapScorer(self.craft_heatmap_settings)
            try:
                scored = score_craft_heatmap_crops_batch(
                    [crop for _box, _expanded, crop in prepared],
                    self.craft_heatmap_settings,
                    self.craft_heatmap_scorer,
                )
            except Exception as exc:
                if not self._scoring_error_reported:
                    self._scoring_error_reported = True
                    self.error.emit(f"Crop scoring failed ({self.crop_scoring_method}): {exc}")
                return

            for (box, expanded, crop), (quality, quality_details) in zip(prepared, scored):
                self._apply_scored_crop_candidate(frame, box, expanded, crop, quality, quality_details)
            return

        for box, expanded, crop in prepared:
            self.update_crop_record(frame, box, expanded=expanded, crop=crop)

    def update_crop_record(self, frame: np.ndarray, box: TrackBox, expanded: Optional[list[int]] = None, crop: Optional[np.ndarray] = None) -> bool:
        if expanded is None or crop is None:
            item = self.prepare_crop_for_box(frame, box)
            if item is None:
                return False
            expanded, crop = item
        try:
            quality, quality_details = score_crop_candidate(
                frame=frame,
                raw_box=list(box.xyxy),
                crop_box=list(expanded),
                crop=crop,
                detector_score=box.score,
                method=self.crop_scoring_method,
                ocr_proxy_weights=self.ocr_proxy_weights,
                detector_geom_score_weights=self.detector_geom_score_weights,
                craft_heatmap_settings=self.craft_heatmap_settings,
                craft_scorer=self.craft_heatmap_scorer,
            )
        except Exception as exc:
            if not self._scoring_error_reported:
                self._scoring_error_reported = True
                self.error.emit(f"Crop scoring failed ({self.crop_scoring_method}): {exc}")
            return False
        return self._apply_scored_crop_candidate(frame, box, expanded, crop, quality, quality_details)

    def _apply_scored_crop_candidate(
        self,
        frame: np.ndarray,
        box: TrackBox,
        expanded: list[int],
        crop: np.ndarray,
        quality: float,
        quality_details: dict[str, Any],
    ) -> bool:
        if quality < self.min_crop_quality:
            return False

        rec = self.crop_records.get(box.track_id)
        candidate_index = 1 if rec is None else len(rec.all_crops) + 1
        candidate = make_crop_candidate(
            candidate_index=candidate_index,
            frame_idx=box.frame_index,
            raw_box=list(box.xyxy),
            crop_box=list(expanded),
            crop=crop,
            quality=quality,
            detector_score=box.score,
            quality_method=self.crop_scoring_method,
            quality_details=quality_details,
        )

        if rec is None:
            candidate["is_best"] = True
            rec = CropRecord(
                box.track_id,
                box.label,
                box.cls_id,
                crop.copy(),
                expanded,
                quality,
                box.score,
                1,
                1,
                box.frame_index,
                box.frame_index,
                self.min_stable_frames <= 1,
                best_quality_method=self.crop_scoring_method,
                best_quality_details=dict(quality_details),
                all_crops=[candidate],
                best_candidate_index=candidate_index,
            )
            self.crop_records[box.track_id] = rec
            self.emit_record(rec)
            return True

        rec.all_crops.append(candidate)
        trim_stored_candidates(rec)
        rec.seen_count += 1
        rec.last_seen_frame = box.frame_index
        became_ready = rec.seen_count >= self.min_stable_frames and not rec.ready_for_ocr
        if became_ready:
            rec.ready_for_ocr = True

        best_changed = sync_best_to_highest_quality_candidate(rec)
        if best_changed:
            rec.label = box.label
            rec.cls_id = box.cls_id
            rec.version += 1

        if best_changed or became_ready:
            self.emit_record(rec)
            return True

        # Live metadata-only update: do not emit all crop images/history every frame.
        self.poster_found_record.emit(rec.summary(include_crop=False, include_crop_history=False))
        return False

    def emit_record(self, rec: CropRecord):
        self.poster_found.emit(int(rec.object_id), rec.best_crop.copy())
        self.poster_found_record.emit(rec.summary(include_crop=True, include_crop_history=False))

    def emit_frame_progress(self, frame_idx: int, total: int):
        if total > 0:
            pct = min(94, int(10 + (frame_idx / total) * 84))
            msg = f"frame {frame_idx}/{total} | tracks: {len(self.crop_records)} | score: {self.crop_scoring_method} | detect FPS: {self.detect_fps_smooth:.1f}"
        else:
            pct = 50
            msg = f"frame {frame_idx} | tracks: {len(self.crop_records)} | score: {self.crop_scoring_method} | detect FPS: {self.detect_fps_smooth:.1f}"
        self.progress.emit(pct, msg)

    def finish(self):
        self.progress.emit(95, "Stopping and collecting best tracked crops..." if self.stop_requested else "Finishing up...")
        records = self.get_crop_records(include_crop=True, include_crop_history=True)
        crops = [rec["crop"] for rec in records if isinstance(rec.get("crop"), np.ndarray) and rec["crop"].size > 0]
        self.finished_records.emit(records)
        self.finished.emit(crops)
        self.progress.emit(100, "Stopped" if self.stop_requested else f"Done — {len(crops)} best crop(s)")

    def get_crop_records(self, include_crop: bool = True, only_ready: bool = False, include_crop_history: bool = False) -> list[dict[str, Any]]:
        records = [rec.summary(include_crop, include_crop_history) for rec in self.crop_records.values() if not only_ready or rec.ready_for_ocr]
        return sorted(records, key=lambda item: (-float(item["quality"]), int(item["id"])))


class YOLOWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_processing = pyqtSignal(list)
    poster_found = pyqtSignal(int, object)
    error = pyqtSignal(str)
    poster_found_record = pyqtSignal(dict)
    finished_records = pyqtSignal(list)
    frame_preview = pyqtSignal(object)

    def __init__(self, video_path: str, **engine_kwargs):
        super().__init__()
        self.video_path = str(video_path)
        self.engine = YOLOEngine(**engine_kwargs)

    def request_stop(self):
        self.engine.request_stop()

    def run(self):
        self.engine.progress.connect(self.progress)
        self.engine.finished.connect(self.finished_processing)
        self.engine.poster_found.connect(self.poster_found)
        self.engine.error.connect(self.error)
        self.engine.poster_found_record.connect(self.poster_found_record)
        self.engine.finished_records.connect(self.finished_records)
        self.engine.frame_preview.connect(self.frame_preview)
        self.engine.process_video(self.video_path)
