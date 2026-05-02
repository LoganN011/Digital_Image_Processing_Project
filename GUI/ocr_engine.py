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
DEFAULT_JOINER = " | "
DEFAULT_RETRY_INVERT_IF_BLANK = True


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
    """Add likely missing spaces in PARSeq output, without touching URLs/emails/etc.

    PARSeq often recognizes a whole cropped text line as one continuous token
    even when EasyOCR detected it as a line containing multiple words. This
    helper is intentionally applied only to PARSeq text, before OCRLine is
    created, so OCRLine.text and OCRLine.parseq_text stay consistent.
    """
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


def sort_easyocr_results(results: list[Any]) -> list[Any]:
    def key(item: Any) -> tuple[float, float]:
        try:
            pts = np.array(item[0], dtype=float)
            return float(np.mean(pts[:, 1])), float(np.mean(pts[:, 0]))
        except Exception:
            return 0.0, 0.0
    return sorted(results or [], key=key)


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
            raw, lines = self._read_and_parse(ocr_img, callback, 25, 95, "parseq")
            used_inverted = False

            if options.get("retry_invert", DEFAULT_RETRY_INVERT_IF_BLANK) and not lines:
                self._progress(callback, "Retrying OCR with inverted contrast", 70, stage="invert_retry")
                raw2, lines2 = self._read_and_parse(cv2.bitwise_not(ocr_img), callback, 75, 95, "parseq_invert")
                if lines2:
                    raw, lines, used_inverted = raw2, lines2, True

            if not lines:
                self._progress(callback, "OCR done: no text", 100, stage="done", done=0, total=len(raw))
                return OCRResult("(no text)", 0.0, "PARSeq+EasyOCR.readtext", [], used_inverted, len(raw))

            text = str(options.get("joiner", DEFAULT_JOINER)).join(line.text for line in lines)
            avg_conf = float(np.mean([line.conf for line in lines]))
            method = self._summarize_methods(lines) + ("+invert" if used_inverted else "")
            self._progress(callback, f"OCR done: {len(lines)} line(s)", 100, stage="done", done=len(lines), total=len(raw))
            return OCRResult(text, avg_conf, method, lines, used_inverted, len(raw))
        except Exception as exc:
            self._progress(callback, f"OCR error: {exc}", 100, stage="error")
            return OCRResult(f"OCR Error: {exc}", 0.0, "error", [])

    def _read_and_parse(self, image: np.ndarray, callback, start: int, end: int, stage: str):
        self._progress(callback, "EasyOCR readtext: detecting lines", start, stage="easyocr_readtext")
        raw = self.run_easyocr_readtext(image)
        total = len(raw or [])
        self._progress(callback, f"PARSeq second pass: 0/{total} lines", start + 10 if total else end, stage=stage, done=0, total=total)
        lines = self.extract_parseq_second_pass_text(raw, image, callback, start + 10, end, stage)
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

    def run_easyocr_readtext(self, ocr_img: np.ndarray) -> list[Any]:
        if self.reader is None:
            return []
        return self.reader.readtext(
            ocr_img,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
            workers=0,
            min_size=8,
            contrast_ths=0.04,
            adjust_contrast=0.75,
            text_threshold=0.36,
            low_text=0.18,
            link_threshold=0.18,
            canvas_size=1280,
            mag_ratio=1.0,
            add_margin=0.12,
        )

    def extract_parseq_second_pass_text(
        self,
        raw: list[Any],
        ocr_img: np.ndarray,
        progress_callback=None,
        progress_start: int = 35,
        progress_end: int = 95,
        stage: str = "parseq",
    ) -> list[OCRLine]:
        lines: list[OCRLine] = []
        ordered = sort_easyocr_results(raw)
        total = len(ordered)
        if total == 0:
            self._progress(progress_callback, "No EasyOCR text boxes found", progress_end, stage=stage, done=0, total=0)
            return lines

        span = max(1, progress_end - progress_start)
        for idx, item in enumerate(ordered, 1):
            self._progress(progress_callback, f"PARSeq line {idx}/{total}", progress_start + span * (idx - 1) // total, stage=stage, done=idx - 1, total=total, line=idx)
            line = self._parse_line(item, ocr_img)
            if line is not None:
                lines.append(line)
            self._progress(progress_callback, f"PARSeq line {idx}/{total}", progress_start + span * idx // total, stage=stage, done=idx, total=total, line=idx)
        return lines

    def _parse_line(self, item: Any, image: np.ndarray) -> OCRLine | None:
        if len(item) < 3:
            return None
        easy_text = clean_ocr_text(str(item[1]))
        try:
            easy_conf = float(item[2])
        except Exception:
            easy_conf = 0.0
        parseq_text, parseq_conf = "", 0.0
        crop = crop_easyocr_text_region(image, item[0])
        if crop is not None and crop.size > 0:
            parseq_text, parseq_conf = self.run_parseq_on_text_crop(crop)
            parseq_text = maybe_split_parseq_text(parseq_text)
        if parseq_text:
            return OCRLine(parseq_text, float(parseq_conf), "PARSeq", item[0], parseq_text, float(parseq_conf), easy_text, float(easy_conf))
        if easy_text:
            return OCRLine(easy_text, float(easy_conf), "EasyOCR fallback", item[0], parseq_text, float(parseq_conf), easy_text, float(easy_conf))
        return None

    def run_parseq_on_text_crop(self, text_crop: np.ndarray) -> tuple[str, float]:
        if self.parseq_model is None or self.parseq_transform is None or self.torch is None:
            return "", 0.0
        if text_crop.ndim == 2:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_GRAY2RGB)
        elif text_crop.ndim == 3 and text_crop.shape[2] == 4:
            rgb = cv2.cvtColor(cv2.cvtColor(text_crop, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_BGR2RGB)
        tensor = self.parseq_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            pred = self.parseq_model(tensor).softmax(-1)
            labels, conf = self.parseq_model.tokenizer.decode(pred)
        text = labels[0] if labels else ""
        return text, self._parseq_confidence(conf, text)

    def _parseq_confidence(self, raw_conf: Any, text: str) -> float:
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

    def _summarize_methods(self, lines: list[OCRLine]) -> str:
        methods = []
        for line in lines:
            if line.method not in methods:
                methods.append(line.method)
        return "+".join(methods) if methods else "PARSeq+EasyOCR.readtext"

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
