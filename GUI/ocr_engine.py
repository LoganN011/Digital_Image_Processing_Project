
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import easyocr
import numpy as np

try:
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    # PARSeq
    from strhub.data.module import SceneTextDataModule
except Exception: 
    SceneTextDataModule = None

try:
    from torchvision import transforms as T
except Exception:
    T = None

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


def preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """Fast flyer/poster OCR preprocessing from the v11 workflow."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    if crop_bgr.ndim == 2:
        gray = crop_bgr.copy()
    elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 4:
        bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side < 900:
        scale = min(2.4, 900.0 / max(1, long_side))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif long_side > 1800:
        scale = 1800.0 / long_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # mild unsharp mask
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray, 1.45, blur, -0.45, 0)


def clean_ocr_text(text: str) -> str:
    text = str(text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def sort_easyocr_results(results: list[Any]) -> list[Any]:
    """Sort EasyOCR readtext() results top-to-bottom, then left-to-right."""

    def key(item: Any) -> tuple[float, float]:
        try:
            pts = np.array(item[0], dtype=float)
            return (float(np.mean(pts[:, 1])), float(np.mean(pts[:, 0])))
        except Exception:
            return (0.0, 0.0)

    return sorted(results, key=key)


def crop_easyocr_text_region(image: np.ndarray, pts: Any, pad: int = 3) -> np.ndarray | None:
    """Perspective-crop one EasyOCR quadrilateral from the preprocessed image."""
    try:
        arr = np.array(pts, dtype=np.float32)
        if arr.shape != (4, 2):
            raise ValueError("not a 4-point box")

        width_top = np.linalg.norm(arr[1] - arr[0])
        width_bottom = np.linalg.norm(arr[2] - arr[3])
        height_left = np.linalg.norm(arr[3] - arr[0])
        height_right = np.linalg.norm(arr[2] - arr[1])
        out_w = max(2, int(round(max(width_top, width_bottom))))
        out_h = max(2, int(round(max(height_left, height_right))))

        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(arr, dst)
        crop = cv2.warpPerspective(
            image,
            matrix,
            (out_w, out_h),
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

    # PARSeq best when very small text lines are enlarged before transform
    h, _w = crop.shape[:2]
    if h < 32:
        scale = min(4.0, 32.0 / max(1, h))
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
        lazy_load: bool = True,
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
        """Return (text, avg_conf), matching the original OCREngine API."""
        result = self.get_text_details(image_input, options)
        return result.text, result.avg_conf

    def get_text_details(self, image_input, options: dict[str, Any] | None = None) -> OCRResult:
        try:
            frame = self._read_image(image_input)
            if frame is None or frame.size == 0:
                return OCRResult("OCR Error: Image not found or empty.", 0.0, "error", [])

            if not self.load_models_if_needed():
                msg = self.reader_error or self.parseq_error or "OCR models failed to load"
                return OCRResult(f"OCR unavailable: {msg}", 0.0, "error", [])

            options = dict(options or {})
            joiner = str(options.get("joiner", DEFAULT_JOINER))
            retry_invert = bool(options.get("retry_invert", DEFAULT_RETRY_INVERT_IF_BLANK))

            ocr_img = preprocess_for_ocr(frame)
            raw = self.run_easyocr_readtext(ocr_img)
            lines = self.extract_parseq_second_pass_text(raw, ocr_img)
            used_inverted = False

            # Second try: invert colors
            if retry_invert and not lines:
                inv = cv2.bitwise_not(ocr_img)
                raw2 = self.run_easyocr_readtext(inv)
                lines2 = self.extract_parseq_second_pass_text(raw2, inv)
                if lines2:
                    raw = raw2
                    lines = lines2
                    used_inverted = True

            if not lines:
                return OCRResult("(no text)", 0.0, "PARSeq+EasyOCR.readtext", [], used_inverted, raw_count=len(raw))

            text = joiner.join(line.text for line in lines)
            avg_conf = float(np.mean([line.conf for line in lines])) if lines else 0.0
            method = self._summarize_methods(lines)
            if used_inverted:
                method += "+invert"

            return OCRResult(text, avg_conf, method, lines, used_inverted, raw_count=len(raw))

        except Exception as exc:
            return OCRResult(f"OCR Error: {exc}", 0.0, "error", [])

    def load_models_if_needed(self) -> bool:
        return self.load_easyocr_if_needed() and self.load_parseq_if_needed()

    def load_easyocr_if_needed(self) -> bool:
        if self.reader is not None:
            return True
        if self.reader_error is not None:
            return False

        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=self.verbose)
        except Exception as exc:
            self.reader_error = str(exc)
            return False

        return True

    def load_parseq_if_needed(self) -> bool:
        if self.parseq_model is not None and self.parseq_transform is not None:
            return True
        if self.parseq_load_attempted and self.parseq_error is not None:
            return False

        self.parseq_load_attempted = True
        try:
            import torch
            from PIL import Image  # noqa: F401 - verify Pillow is available

            self.torch = torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            model = torch.hub.load(self.parseq_repo, self.parseq_model_name, pretrained=True)
            model = model.to(self.device).eval()
            self.parseq_model = model
            self.parseq_transform = self._build_parseq_transform(model)
        except Exception as exc:
            self.parseq_error = str(exc)
            self.parseq_model = None
            self.parseq_transform = None
            self.torch = None
            return False

        return True

    def _build_parseq_transform(self, model):
        """Build a PARSeq image transform without importing strhub inside methods."""
        if SceneTextDataModule is not None:
            return SceneTextDataModule.get_transform(model.hparams.img_size)

        if T is None:
            raise RuntimeError("torchvision transforms unavailable and strhub SceneTextDataModule could not be imported")

        hparams = getattr(model, "hparams", None)
        if isinstance(hparams, dict):
            img_size = tuple(hparams.get("img_size", (32, 128)))
        else:
            img_size = tuple(getattr(hparams, "img_size", (32, 128)))

        return T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

    def run_easyocr_readtext(self, ocr_img: np.ndarray) -> list[Any]:
        """v11-style EasyOCR readtext() pass: detect boxes and get fallback text."""
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

    def extract_parseq_second_pass_text(self, raw: list[Any], ocr_img: np.ndarray) -> list[OCRLine]:
        """Use PARSeq to re-recognize EasyOCR-detected crops; EasyOCR text is fallback."""
        lines: list[OCRLine] = []

        for item in sort_easyocr_results(raw):
            if len(item) < 3:
                continue

            easy_text = clean_ocr_text(str(item[1]))
            try:
                easy_conf = float(item[2])
            except Exception:
                easy_conf = 0.0

            parseq_text = ""
            parseq_conf = 0.0
            text_crop = crop_easyocr_text_region(ocr_img, item[0])
            if text_crop is not None and text_crop.size > 0:
                parseq_text, parseq_conf = self.run_parseq_on_text_crop(text_crop)
                parseq_text = clean_ocr_text(parseq_text)

            if parseq_text:
                lines.append(
                    OCRLine(
                        text=parseq_text,
                        conf=float(parseq_conf),
                        method="PARSeq",
                        box=item[0],
                        parseq_text=parseq_text,
                        parseq_conf=float(parseq_conf),
                        easyocr_text=easy_text,
                        easyocr_conf=float(easy_conf),
                    )
                )
            elif easy_text:
                lines.append(
                    OCRLine(
                        text=easy_text,
                        conf=float(easy_conf),
                        method="EasyOCR fallback",
                        box=item[0],
                        parseq_text=parseq_text,
                        parseq_conf=float(parseq_conf),
                        easyocr_text=easy_text,
                        easyocr_conf=float(easy_conf),
                    )
                )

        return lines

    def run_parseq_on_text_crop(self, text_crop: np.ndarray) -> tuple[str, float]:
        """Recognize a single line/word crop with PARSeq."""
        if self.parseq_model is None or self.parseq_transform is None or self.torch is None:
            return "", 0.0

        from PIL import Image

        if text_crop.ndim == 2:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_GRAY2RGB)
        elif text_crop.ndim == 3 and text_crop.shape[2] == 4:
            bgr = cv2.cvtColor(text_crop, cv2.COLOR_BGRA2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        tensor = self.parseq_transform(image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.parseq_model(tensor)
            pred = logits.softmax(-1)
            labels, raw_conf = self.parseq_model.tokenizer.decode(pred)

        text = labels[0] if labels else ""
        return text, self._parseq_confidence(raw_conf, text)

    def _parseq_confidence(self, raw_conf: Any, text: str) -> float:
        """Convert PARSeq token probabilities into one line-level confidence."""
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
        methods: list[str] = []
        for line in lines:
            if line.method not in methods:
                methods.append(line.method)
        return "+".join(methods) if methods else "PARSeq+EasyOCR.readtext"

    def preprocess_for_ocr(self, frame: np.ndarray, options: dict[str, Any] | None = None) -> np.ndarray:
        """Compatibility wrapper for GUI preview code.

        Older GUI_Main versions call self.ocr_engine.preprocess_for_ocr(frame, options).
        This engine intentionally ignores old manual preview options so OCR stays
        aligned with the fixed v11-style poster/flyer preprocessing path.
        """
        _ = options
        return preprocess_for_ocr(frame)

    def clean_string(self, text: str) -> str:
        return clean_ocr_text(text)

    def _read_image(self, image_input) -> np.ndarray | None:
        if isinstance(image_input, (str, Path)):
            return cv2.imread(str(image_input))
        if isinstance(image_input, np.ndarray):
            return image_input
        return None


if __name__ == "__main__":
    test_engine = OCREngine()
    text, conf = test_engine.get_text("../poster_downloads/3.8_52888.jpg")
    print(f"Detected: {text}")
    print(f"Average confidence: {conf:.4f}")
