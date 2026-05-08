from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import easyocr
import numpy as np
from PIL import Image

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

DEFAULT_PARSEQ_REPO = "baudm/parseq"
DEFAULT_PARSEQ_MODEL = "parseq"
DEFAULT_WORD_JOINER = " "
DEFAULT_ROW_JOINER = "\n"
DEFAULT_PRESERVE_LAYOUT = True
DEFAULT_RETRY_INVERT_IF_BLANK = True
DEFAULT_PARSEQ_BATCH_SIZE = 16
DEFAULT_EASYOCR_DETECT_OPTIONS: dict[str, Any] = {
    "min_size": 0,
    "text_threshold": 0.36,
    "low_text": 0.18,
    "link_threshold": 0.18,
    "canvas_size": 1280,
    "mag_ratio": 1.0,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.0,
    "width_ths": 0.0,
    "add_margin": 0.12,
    "reformat": True,
    "threshold": 0.0,
    "bbox_min_score": 0.0,
    "bbox_min_size": 0,
    "max_candidates": 0,
}


@dataclass
class OCRLine:
    text: str
    conf: float
    method: str
    box: Any | None = None
    parseq_text: str = ""
    parseq_conf: float = 0.0
    easyocr_text: str = ""
    easyocr_conf: float = 0.0
    row_index: int = 0
    word_index: int = 0
    line_break_after: bool = False


@dataclass
class OCRResult:
    text: str
    avg_conf: float
    method: str
    lines: list[OCRLine]
    used_inverted: bool = False
    raw_count: int = 0


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


def _safe_easyocr_box_points(item: Any) -> np.ndarray | None:
    try:
        pts = np.array(item[0], dtype=float).reshape(-1, 2)
    except Exception:
        return None
    if pts.size == 0 or pts.shape[0] < 2:
        return None
    if not np.all(np.isfinite(pts)):
        return None
    return pts


def _easyocr_layout_box(item: Any, original_index: int) -> dict[str, Any]:
    pts = _safe_easyocr_box_points(item)
    if pts is None:
        return {
            "item": item,
            "index": int(original_index),
            "x_min": 0.0,
            "x_ctr": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_ctr": 0.0,
            "y_max": 1.0,
            "height": 1.0,
            "width": 1.0,
            "baseline": 1.0,
        }

    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    height = max(1.0, y_max - y_min)
    width = max(1.0, x_max - x_min)

    # For four-point boxes, estimate the text baseline from the lower two
    # vertices. This is more stable than mean-y for mixed font sizes.
    lower_count = min(2, pts.shape[0])
    lower_y = np.sort(pts[:, 1])[-lower_count:]
    baseline = float(np.mean(lower_y)) if lower_y.size else y_max

    return {
        "item": item,
        "index": int(original_index),
        "x_min": x_min,
        "x_ctr": float(np.mean(pts[:, 0])),
        "x_max": x_max,
        "y_min": y_min,
        "y_ctr": float(np.mean(pts[:, 1])),
        "y_max": y_max,
        "height": height,
        "width": width,
        "baseline": baseline,
    }


def _median(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values]
    return float(np.median(vals)) if vals else float(default)


def _layout_row_stats(row: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "x_min": _median((box["x_min"] for box in row)),
        "y_min": _median((box["y_min"] for box in row)),
        "y_ctr": _median((box["y_ctr"] for box in row)),
        "y_max": _median((box["y_max"] for box in row)),
        "height": max(1.0, _median((box["height"] for box in row), 1.0)),
        "baseline": _median((box["baseline"] for box in row), 0.0),
    }


def _same_visual_text_row(box: dict[str, Any], row: list[dict[str, Any]]) -> tuple[bool, float]:
    stats = _layout_row_stats(row)
    row_h = max(1.0, stats["height"])
    box_h = max(1.0, float(box["height"]))
    max_h = max(row_h, box_h)
    min_h = max(1.0, min(row_h, box_h))

    overlap = min(float(box["y_max"]), stats["y_max"]) - max(float(box["y_min"]), stats["y_min"])
    overlap_ratio = max(0.0, overlap) / min_h
    center_dist = abs(float(box["y_ctr"]) - stats["y_ctr"])
    baseline_dist = abs(float(box["baseline"]) - stats["baseline"])
    top_dist = abs(float(box["y_min"]) - stats["y_min"])
    bottom_dist = abs(float(box["y_max"]) - stats["y_max"])

    same_by_overlap_and_baseline = (
        overlap_ratio >= 0.58
        and baseline_dist <= max(5.0, 0.42 * max_h)
        and center_dist <= max(6.0, 0.70 * max_h)
    )

    same_by_edges = (
        overlap_ratio >= 0.45
        and top_dist <= max(5.0, 0.38 * max_h)
        and bottom_dist <= max(6.0, 0.50 * max_h)
    )

    same_by_center = (
        overlap_ratio >= 0.38
        and center_dist <= max(4.0, 0.30 * max_h)
        and baseline_dist <= max(6.0, 0.50 * max_h)
    )

    same = same_by_overlap_and_baseline or same_by_edges or same_by_center
    cost = baseline_dist + 0.35 * center_dist + 0.15 * top_dist
    return bool(same), float(cost)


def group_easyocr_results_into_layout_rows(results: list[Any]) -> list[list[Any]]:
    if not results:
        return []

    boxes = [_easyocr_layout_box(item, i) for i, item in enumerate(results)]

    boxes.sort(key=lambda box: (box["y_min"], box["baseline"], box["x_min"], box["index"]))

    rows: list[list[dict[str, Any]]] = []
    for box in boxes:
        best_row: list[dict[str, Any]] | None = None
        best_cost = float("inf")

        for row in rows:
            same_row, cost = _same_visual_text_row(box, row)
            if same_row and cost < best_cost:
                best_row = row
                best_cost = cost

        if best_row is None:
            rows.append([box])
        else:
            best_row.append(box)

    # Re-sort after grouping because a later box can slightly adjust the row.
    rows.sort(key=lambda row: (_median((box["y_min"] for box in row)), _median((box["baseline"] for box in row))))

    out_rows: list[list[Any]] = []
    for row in rows:
        row.sort(key=lambda box: (box["x_min"], box["y_min"], box["index"]))
        out_rows.append([box["item"] for box in row])
    return out_rows


def sort_easyocr_results(results: list[Any]) -> list[Any]:
    """Return EasyOCR boxes in layout-aware reading order.

    Kept as a compatibility wrapper for older callers. The old mean-y/mean-x
    ordering has been removed; this now flattens layout-aware text rows.
    """
    return [item for row in group_easyocr_results_into_layout_rows(results) for item in row]


def format_ocr_lines_with_layout(lines: list[OCRLine], options: dict[str, Any] | None = None) -> str:
    options = dict(options or {})
    preserve_layout = bool(options.get("preserve_layout", DEFAULT_PRESERVE_LAYOUT))
    if not preserve_layout:
        return str(options.get("joiner", DEFAULT_JOINER)).join(line.text for line in lines)

    word_joiner = str(options.get("word_joiner", DEFAULT_WORD_JOINER))
    row_joiner = str(options.get("row_joiner", DEFAULT_ROW_JOINER))

    rows: list[list[OCRLine]] = []
    row_lookup: dict[int, list[OCRLine]] = {}
    for line in lines:
        row_index = int(getattr(line, "row_index", 0))
        if row_index not in row_lookup:
            row_lookup[row_index] = []
            rows.append(row_lookup[row_index])
        row_lookup[row_index].append(line)

    rendered_rows: list[str] = []
    for row in rows:
        row.sort(key=lambda line: int(getattr(line, "word_index", 0)))
        row_text = clean_ocr_text(word_joiner.join(line.text for line in row if line.text))
        if row_text:
            rendered_rows.append(row_text)

    return row_joiner.join(rendered_rows) if rendered_rows else ""


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


class OCREngine:
    def __init__(
        self,
        languages: Iterable[str] | None = None,
        gpu: bool = False,
        verbose: bool = False,
        parseq_repo: str = DEFAULT_PARSEQ_REPO,
        parseq_model_name: str = DEFAULT_PARSEQ_MODEL,
        lazy_load: bool = False,
    ):
        self.languages = list(languages or ["en"])
        self.gpu = bool(gpu)
        self.verbose = bool(verbose)
        self.parseq_repo = parseq_repo
        self.parseq_model_name = parseq_model_name
        self.reader = None
        self.reader_error: str | None = None
        self.torch = None
        self.parseq_model = None
        self.parseq_transform = None
        self.device = "cpu"
        self.parseq_error: str | None = None
        self.parseq_load_attempted = False
        if not lazy_load:
            self.load_models_if_needed()

    def get_text(self, image_input, options: dict[str, Any] | None = None):
        result = self.get_text_details(image_input, options)
        return result.text, result.avg_conf

    def get_text_details(self, image_input, options: dict[str, Any] | None = None) -> OCRResult:
        options = dict(options or {})
        callback = options.get("progress_callback")
        try:
            self._progress(callback, "Reading image", 3, stage="read_image")
            frame = self._read_image(image_input)
            if frame is None or frame.size == 0:
                self._progress(callback, "OCR error: image not found", 100, stage="error")
                return OCRResult("OCR Error: Image not found or empty.", 0.0, "error", [])

            self._progress(callback, "Loading OCR models", 8, stage="loading_models")
            if not self.load_models_if_needed():
                msg = self.reader_error or self.parseq_error or "OCR models failed to load"
                self._progress(callback, f"OCR unavailable: {msg}", 100, stage="error")
                return OCRResult(f"OCR unavailable: {msg}", 0.0, "error", [])

            self._progress(callback, "Preprocessing crop", 15, stage="preprocess")
            ocr_img = preprocess_for_ocr(frame)
            raw, lines = self._read_and_parse(ocr_img, callback, 25, 95, "parseq", options)
            used_inverted = False

            if options.get("retry_invert", DEFAULT_RETRY_INVERT_IF_BLANK) and not lines:
                self._progress(callback, "Retrying OCR with inverted contrast", 70, stage="invert_retry")
                raw2, lines2 = self._read_and_parse(cv2.bitwise_not(ocr_img), callback, 75, 95, "parseq_invert", options)
                if lines2:
                    raw, lines, used_inverted = raw2, lines2, True

            if not lines:
                self._progress(callback, "OCR done: no text", 100, stage="done", done=0, total=len(raw))
                return OCRResult("(no text)", 0.0, "PARSeq+EasyOCR.detect", [], used_inverted, len(raw))

            text = format_ocr_lines_with_layout(lines, options)
            avg_conf = float(np.mean([line.conf for line in lines]))
            method = self._summarize_methods(lines) + ("+invert" if used_inverted else "")
            self._progress(callback, f"OCR done: {len(lines)} line(s)", 100, stage="done", done=len(lines), total=len(raw))
            return OCRResult(text, avg_conf, method, lines, used_inverted, len(raw))
        except Exception as exc:
            self._progress(callback, f"OCR error: {exc}", 100, stage="error")
            return OCRResult(f"OCR Error: {exc}", 0.0, "error", [])

    def _read_and_parse(self, image: np.ndarray, callback, start: int, end: int, stage: str, options: dict[str, Any] | None = None):
        self._progress(callback, "EasyOCR detect: finding line boxes", start, stage="easyocr_detect")
        raw = self.run_easyocr_detect(image)
        total = len(raw or [])
        self._progress(callback, f"PARSeq second pass: 0/{total} lines", start + 10 if total else end, stage=stage, done=0, total=total)
        lines = self.extract_parseq_second_pass_text(raw, image, callback, start + 10, end, stage, options)
        return raw, lines

    def _progress(self, callback, message: str, percent: int, **extra):
        if callback is None:
            return
        try:
            payload = {"message": message, "percent": int(max(0, min(100, percent)))}
            payload.update(extra)
            callback(payload)
        except Exception:
            pass

    def load_models_if_needed(self) -> bool:
        return self.load_easyocr_if_needed() and self.load_parseq_if_needed()

    def load_easyocr_if_needed(self) -> bool:
        if self.reader is not None:
            return True
        if self.reader_error is not None:
            return False
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=self.verbose)
            return True
        except Exception as exc:
            self.reader_error = str(exc)
            return False

    def load_parseq_if_needed(self) -> bool:
        if self.parseq_model is not None and self.parseq_transform is not None:
            return True
        if self.parseq_load_attempted and self.parseq_error is not None:
            return False
        self.parseq_load_attempted = True
        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            model = torch.hub.load(self.parseq_repo, self.parseq_model_name, pretrained=True)
            self.parseq_model = model.to(self.device).eval()
            self.parseq_transform = self._build_parseq_transform(model)
            return True
        except Exception as exc:
            self.parseq_error = str(exc)
            self.parseq_model = None
            self.parseq_transform = None
            self.torch = None
            return False

    def _build_parseq_transform(self, model):
        if SceneTextDataModule is not None:
            return SceneTextDataModule.get_transform(model.hparams.img_size)
        if T is None:
            raise RuntimeError("torchvision transforms unavailable and PARSeq transform import failed")
        hparams = getattr(model, "hparams", None)
        img_size = tuple(hparams.get("img_size", (32, 128))) if isinstance(hparams, dict) else tuple(getattr(hparams, "img_size", (32, 128)))
        return T.Compose([T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def run_easyocr_detect(self, ocr_img: np.ndarray) -> list[Any]:
        if self.reader is None:
            return []
        horizontal_list, free_list = self.reader.detect(ocr_img, **DEFAULT_EASYOCR_DETECT_OPTIONS)
        results: list[Any] = []
        for box in self._unwrap_easyocr_detection_list(horizontal_list):
            quad = self._horizontal_box_to_quad(box)
            if quad is not None:
                results.append([quad, "", 0.0])
        for box in self._unwrap_easyocr_detection_list(free_list):
            quad = self._free_box_to_quad(box)
            if quad is not None:
                results.append([quad, "", 0.0])
        return results

    def run_easyocr_readtext(self, ocr_img: np.ndarray) -> list[Any]:
        return self.run_easyocr_detect(ocr_img)

    def _unwrap_easyocr_detection_list(self, boxes: Any) -> list[Any]:
        if boxes is None:
            return []
        try:
            if len(boxes) == 0:
                return []
        except Exception:
            return []
        if self._looks_like_single_detection_box(boxes):
            return [boxes]
        first = boxes[0]
        if self._looks_like_single_detection_box(first):
            return list(boxes)
        try:
            return list(first)
        except Exception:
            return []

    def _looks_like_single_detection_box(self, box: Any) -> bool:
        try:
            arr = np.array(box, dtype=np.float32)
        except Exception:
            return False
        if arr.shape == (4, 2):
            return True
        return arr.reshape(-1).size == 4

    def _horizontal_box_to_quad(self, box: Any) -> list[list[float]] | None:
        try:
            x1, x2, y1, y2 = [float(v) for v in np.array(box, dtype=np.float32).reshape(-1)[:4]]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def _free_box_to_quad(self, box: Any) -> list[list[float]] | None:
        try:
            arr = np.array(box, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if arr.shape[0] < 4:
            return None
        arr = arr[:4]
        if np.ptp(arr[:, 0]) <= 1 or np.ptp(arr[:, 1]) <= 1:
            return None
        return arr.astype(float).tolist()

    def extract_parseq_second_pass_text(
        self,
        raw: list[Any],
        ocr_img: np.ndarray,
        progress_callback=None,
        progress_start: int = 35,
        progress_end: int = 95,
        stage: str = "parseq",
        options: dict[str, Any] | None = None,
    ) -> list[OCRLine]:
        """Recognize EasyOCR text-region crops with PARSeq in batches.

        EasyOCR/CRAFT is still used only to locate text boxes and preserve the
        visual reading order. The expensive PARSeq recognizer now receives a
        stack of text-line crops per chunk instead of one crop per forward pass.
        """
        options = dict(options or {})
        lines: list[OCRLine] = []
        rows = group_easyocr_results_into_layout_rows(raw)
        ordered = [item for row in rows for item in row]
        total = len(ordered)
        if total == 0:
            self._progress(progress_callback, "No EasyOCR text boxes found", progress_end, stage=stage, done=0, total=0)
            return lines

        try:
            batch_size = int(options.get("parseq_batch_size", DEFAULT_PARSEQ_BATCH_SIZE))
        except Exception:
            batch_size = DEFAULT_PARSEQ_BATCH_SIZE
        batch_size = max(1, batch_size)

        span = max(1, progress_end - progress_start)
        jobs: list[dict[str, Any]] = []
        processed = 0

        for row_index, row in enumerate(rows):
            for word_index, item in enumerate(row):
                if len(item) < 3:
                    processed += 1
                    continue
                easy_text = clean_ocr_text(str(item[1]))
                try:
                    easy_conf = float(item[2])
                except Exception:
                    easy_conf = 0.0
                crop = crop_easyocr_text_region(ocr_img, item[0])
                jobs.append({
                    "item": item,
                    "crop": crop,
                    "easy_text": easy_text,
                    "easy_conf": easy_conf,
                    "row_index": int(row_index),
                    "word_index": int(word_index),
                    "line_break_after": bool(word_index == len(row) - 1),
                    "ordered_index": int(processed),
                })
                processed += 1

        if not jobs:
            self._progress(progress_callback, "No usable text boxes found", progress_end, stage=stage, done=0, total=total)
            return lines

        crop_jobs = [job for job in jobs if job.get("crop") is not None and getattr(job.get("crop"), "size", 0) > 0]
        crop_count = len(crop_jobs)
        self._progress(
            progress_callback,
            f"PARSeq batched pass: 0/{crop_count} crops",
            progress_start,
            stage=stage,
            done=0,
            total=crop_count,
            batch_size=batch_size,
        )

        recognized: dict[int, tuple[str, float]] = {}
        done_crops = 0
        for batch_start in range(0, crop_count, batch_size):
            batch_jobs = crop_jobs[batch_start : batch_start + batch_size]
            batch_crops = [job["crop"] for job in batch_jobs]
            self._progress(
                progress_callback,
                f"PARSeq batch {batch_start // batch_size + 1}: {done_crops}/{crop_count} crops",
                progress_start + span * done_crops // max(1, crop_count),
                stage=stage,
                done=done_crops,
                total=crop_count,
                batch_size=batch_size,
            )
            batch_results = self.run_parseq_on_text_crops(batch_crops, batch_size=batch_size)
            for job, (parseq_text, parseq_conf) in zip(batch_jobs, batch_results):
                recognized[id(job)] = (maybe_split_parseq_text(parseq_text), float(parseq_conf))
            done_crops += len(batch_jobs)
            self._progress(
                progress_callback,
                f"PARSeq batched pass: {done_crops}/{crop_count} crops",
                progress_start + span * done_crops // max(1, crop_count),
                stage=stage,
                done=done_crops,
                total=crop_count,
                batch_size=batch_size,
            )

        for job in jobs:
            item = job["item"]
            parseq_text, parseq_conf = recognized.get(id(job), ("", 0.0))
            easy_text = str(job["easy_text"] or "")
            easy_conf = float(job["easy_conf"] or 0.0)

            line: OCRLine | None = None
            if parseq_text:
                line = OCRLine(parseq_text, float(parseq_conf), "PARSeq", item[0], parseq_text, float(parseq_conf), easy_text, easy_conf)
            elif easy_text:
                line = OCRLine(easy_text, easy_conf, "EasyOCR fallback", item[0], parseq_text, float(parseq_conf), easy_text, easy_conf)

            if line is not None:
                line.row_index = int(job["row_index"])
                line.word_index = int(job["word_index"])
                line.line_break_after = bool(job["line_break_after"])
                lines.append(line)

        return lines

    def _parse_line(self, item: Any, image: np.ndarray) -> OCRLine | None:
        """Compatibility wrapper for older callers; normal OCR uses batching."""
        return self._parse_line_with_parseq_result(item, image, "", 0.0)

    def _parse_line_with_parseq_result(self, item: Any, image: np.ndarray, parseq_text: str, parseq_conf: float) -> OCRLine | None:
        if len(item) < 3:
            return None
        easy_text = clean_ocr_text(str(item[1]))
        try:
            easy_conf = float(item[2])
        except Exception:
            easy_conf = 0.0
        parseq_text = maybe_split_parseq_text(parseq_text)
        if parseq_text:
            return OCRLine(parseq_text, float(parseq_conf), "PARSeq", item[0], parseq_text, float(parseq_conf), easy_text, float(easy_conf))
        if easy_text:
            return OCRLine(easy_text, float(easy_conf), "EasyOCR fallback", item[0], parseq_text, float(parseq_conf), easy_text, float(easy_conf))
        return None

    def _prepare_parseq_tensor(self, text_crop: np.ndarray):
        if text_crop.ndim == 2:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_GRAY2RGB)
        elif text_crop.ndim == 3 and text_crop.shape[2] == 4:
            rgb = cv2.cvtColor(cv2.cvtColor(text_crop, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_BGR2RGB)
        return self.parseq_transform(Image.fromarray(rgb))

    def run_parseq_on_text_crops(self, text_crops: list[np.ndarray], batch_size: int | None = None) -> list[tuple[str, float]]:
        """Run PARSeq on many text crops using batched forward passes."""
        if not text_crops:
            return []
        if self.parseq_model is None or self.parseq_transform is None or self.torch is None:
            return [("", 0.0) for _ in text_crops]
        try:
            batch_size = int(batch_size or DEFAULT_PARSEQ_BATCH_SIZE)
        except Exception:
            batch_size = DEFAULT_PARSEQ_BATCH_SIZE
        batch_size = max(1, batch_size)

        results: list[tuple[str, float]] = []
        for start in range(0, len(text_crops), batch_size):
            batch = text_crops[start : start + batch_size]
            try:
                tensors = [self._prepare_parseq_tensor(crop) for crop in batch]
                tensor = self.torch.stack(tensors, dim=0).to(self.device)
                with self.torch.no_grad():
                    pred = self.parseq_model(tensor).softmax(-1)
                    labels, conf = self.parseq_model.tokenizer.decode(pred)
                for local_idx in range(len(batch)):
                    text = labels[local_idx] if labels and local_idx < len(labels) else ""
                    results.append((text, self._parseq_confidence(conf, text, local_idx)))
            except Exception:
                if len(batch) > 1:
                    for crop in batch:
                        results.extend(self.run_parseq_on_text_crops([crop], batch_size=1))
                else:
                    results.append(("", 0.0))
        return results

    def run_parseq_on_text_crop(self, text_crop: np.ndarray) -> tuple[str, float]:
        return self.run_parseq_on_text_crops([text_crop], batch_size=1)[0]

    def _parseq_confidence(self, raw_conf: Any, text: str, index: int = 0) -> float:
        try:
            if isinstance(raw_conf, (list, tuple)):
                conf0 = raw_conf[index] if index < len(raw_conf) else raw_conf[0]
            else:
                conf0 = raw_conf[index] if getattr(raw_conf, "ndim", 0) > 1 else raw_conf
            if hasattr(conf0, "detach"):
                conf0 = conf0.detach().cpu()
            arr = np.array(conf0, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return 0.0
            n = min(arr.size, max(1, len(text) + 1))
            return float(np.prod(np.clip(arr[:n], 0.0, 1.0)) ** (1.0 / n))
        except Exception:
            return 0.0

    def _summarize_methods(self, lines: list[OCRLine]) -> str:
        methods = []
        for line in lines:
            if line.method not in methods:
                methods.append(line.method)
        return "+".join(methods) if methods else "PARSeq+EasyOCR.detect"

    def preprocess_for_ocr(self, frame: np.ndarray, options: dict[str, Any] | None = None) -> np.ndarray:
        return preprocess_for_ocr(frame)

    def clean_string(self, text: str) -> str:
        return clean_ocr_text(text)

    def _read_image(self, image_input) -> np.ndarray | None:
        if isinstance(image_input, (str, Path)):
            return cv2.imread(str(image_input))
        if isinstance(image_input, np.ndarray):
            return image_input
        return None
