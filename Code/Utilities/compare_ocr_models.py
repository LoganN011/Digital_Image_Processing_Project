#!/usr/bin/env python3
"""
compare_ocr_models_standalone.py

Standalone OCR comparison script for a cropped poster/flyer image.

Pipeline:
  1. Load cropped poster image.
  2. Optionally preprocess using the same preprocessing logic from ocr_engine.py.
  3. Use EasyOCR readtext() to detect text regions.
  4. Crop each detected text line using the same EasyOCR-region crop logic from ocr_engine.py.
  5. Run each line crop through:
       - PARSeq
       - EasyOCR recognizer-only
       - TrOCR
       - PaddleOCR
  6. Print text, confidence, and timing for each model.

Usage:
  python compare_ocr_models_standalone.py /path/to/poster_crop.jpg

No preprocessing for all OCR models:
  python compare_ocr_models_standalone.py /path/to/poster_crop.jpg --no-preprocess

No preprocessing for PARSeq only:
  python compare_ocr_models_standalone.py /path/to/poster_crop.jpg --parseq-no-preprocess

Faster debug:
  python compare_ocr_models_standalone.py /path/to/poster_crop.jpg --no-paddle --no-trocr

Optional:
  python compare_ocr_models_standalone.py /path/to/poster_crop.jpg --gpu
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import easyocr
import numpy as np
import torch
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


def run_easyocr_readtext(reader, ocr_img: np.ndarray) -> list[Any]:
    return reader.readtext(
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


def norm_conf(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def fmt_time(seconds: float) -> str:
    return f"{seconds:.4f}s"


def scale_box_points(box, from_shape, to_shape):
    pts = np.array(box, dtype=np.float32)
    from_h, from_w = from_shape[:2]
    to_h, to_w = to_shape[:2]
    sx = to_w / max(1, from_w)
    sy = to_h / max(1, from_h)
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts


def run_easyocr_recognizer_only(reader, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr

    h, w = gray.shape[:2]
    horizontal_list = [[0, w, 0, h]]
    free_list = []

    try:
        out = reader.recognize(
            gray,
            horizontal_list,
            free_list,
            decoder="greedy",
            batch_size=1,
            workers=0,
            detail=1,
            paragraph=False,
            contrast_ths=0.04,
            adjust_contrast=0.75,
        )

        if not out:
            return "", 0.0

        item = out[0]

        if len(item) >= 3:
            return clean_ocr_text(item[1]), norm_conf(item[2])

        if len(item) >= 2:
            return clean_ocr_text(item[0]), norm_conf(item[1])

    except Exception as exc:
        return f"[EasyOCR recognizer error: {exc}]", 0.0

    return "", 0.0


class PARSeqRunner:
    def __init__(self, repo: str = DEFAULT_PARSEQ_REPO, model_name: str = DEFAULT_PARSEQ_MODEL):
        self.repo = repo
        self.model_name = model_name
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        model = torch.hub.load(repo, model_name, pretrained=True)
        self.model = model.to(self.device).eval()
        self.transform = self._build_transform(model)

    def _build_transform(self, model):
        if SceneTextDataModule is not None:
            return SceneTextDataModule.get_transform(model.hparams.img_size)
        if T is None:
            raise RuntimeError("torchvision transforms unavailable and PARSeq transform import failed")
        hparams = getattr(model, "hparams", None)
        img_size = tuple(hparams.get("img_size", (32, 128))) if isinstance(hparams, dict) else tuple(getattr(hparams, "img_size", (32, 128)))
        return T.Compose([T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])

    @torch.no_grad()
    def __call__(self, text_crop: np.ndarray) -> tuple[str, float]:
        if text_crop is None or text_crop.size == 0:
            return "", 0.0
        if text_crop.ndim == 2:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_GRAY2RGB)
        elif text_crop.ndim == 3 and text_crop.shape[2] == 4:
            rgb = cv2.cvtColor(cv2.cvtColor(text_crop, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(text_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        pred = self.model(tensor).softmax(-1)
        labels, conf = self.model.tokenizer.decode(pred)
        text = labels[0] if labels else ""
        return text, self._confidence(conf, text)

    def _confidence(self, raw_conf: Any, text: str) -> float:
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


class TrOCRRunner:
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        if crop_bgr.ndim == 2:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
        elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        gen = self.model.generate(
            pixel_values,
            max_new_tokens=64,
            return_dict_in_generate=True,
            output_scores=True,
        )

        text = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)[0]
        text = clean_ocr_text(text)

        confs = []
        seq = gen.sequences[0]

        for i, scores in enumerate(gen.scores):
            token_pos = i + 1
            if token_pos >= len(seq):
                break

            token_id = int(seq[token_pos].item())
            probs = torch.softmax(scores[0], dim=-1)
            confs.append(float(probs[token_id].detach().cpu()))

        if confs:
            conf = math.exp(sum(math.log(max(c, 1e-8)) for c in confs) / len(confs))
        else:
            conf = 0.0

        return text, conf


class PaddleRunner:
    def __init__(self):
        from paddleocr import PaddleOCR

        try:
            self.ocr = PaddleOCR(use_angle_cls=False, lang="en")
        except TypeError:
            self.ocr = PaddleOCR(lang="en")

    def __call__(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        if crop_bgr.ndim == 2:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
        elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        try:
            out = self.ocr.ocr(rgb)
        except Exception as exc:
            return f"[PaddleOCR error: {exc}]", 0.0

        return parse_paddle_result(out)


def parse_paddle_result(out):
    if isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict):
        d = out[0]
        texts = d.get("rec_texts", [])
        confs = d.get("rec_scores", [])
        if texts:
            joined = " ".join(t for t in texts if t)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return joined, avg_conf
        return "", 0.0

    if not out:
        return "", 0.0

    cur = out

    while isinstance(cur, list) and len(cur) == 1:
        cur = cur[0]

    if isinstance(cur, tuple) and len(cur) >= 2:
        return clean_ocr_text(cur[0]), norm_conf(cur[1])

    if isinstance(cur, list) and len(cur) >= 2 and isinstance(cur[0], str):
        return clean_ocr_text(cur[0]), norm_conf(cur[1])

    if isinstance(cur, list):
        texts = []
        confs = []

        for item in cur:
            if isinstance(item, tuple) and len(item) >= 2:
                texts.append(clean_ocr_text(item[0]))
                confs.append(norm_conf(item[1]))
            elif isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str):
                texts.append(clean_ocr_text(item[0]))
                confs.append(norm_conf(item[1]))

        if texts:
            joined = " ".join(t for t in texts if t)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return joined, avg_conf

    return clean_ocr_text(str(out)), 0.0


def load_optional(name: str, ctor):
    t0 = perf_counter()
    try:
        obj = ctor()
        dt = perf_counter() - t0
        print(f"{name} load time: {fmt_time(dt)}")
        return obj
    except Exception as exc:
        dt = perf_counter() - t0
        print(f"[WARN] {name} unavailable after {fmt_time(dt)}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Cropped poster/flyer image")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR if available")
    ap.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing for all OCR models")
    ap.add_argument("--parseq-no-preprocess", action="store_true", help="Run PARSeq on crops from the original image while other OCR models use preprocessed crops")
    ap.add_argument("--no-parseq", action="store_true", help="Skip PARSeq")
    ap.add_argument("--no-trocr", action="store_true", help="Skip TrOCR")
    ap.add_argument("--no-paddle", action="store_true", help="Skip PaddleOCR")
    

    args = ap.parse_args()

    total_t0 = perf_counter()

    read_t0 = perf_counter()
    frame = cv2.imread(str(args.image))
    read_time = perf_counter() - read_t0

    if frame is None or frame.size == 0:
        raise SystemExit(f"Could not read image: {args.image}")

    print()
    print("=" * 100)
    print(f"IMAGE: {args.image}")
    print(f"Image shape: {frame.shape}")
    print(f"Image read time: {fmt_time(read_time)}")
    print("=" * 100)

    if args.no_preprocess:
        print("\nSkipping preprocessing because --no-preprocess was used.")
        t0 = perf_counter()
        ocr_img = frame.copy()
        preprocess_time = perf_counter() - t0
    else:
        print("\nPreprocessing with standalone preprocess_for_ocr()...")
        t0 = perf_counter()
        ocr_img = preprocess_for_ocr(frame)
        preprocess_time = perf_counter() - t0

    print(f"Preprocess step time: {fmt_time(preprocess_time)}")

    if args.parseq_no_preprocess and args.no_preprocess:
        print("PARSeq no-preprocess flag is redundant because --no-preprocess already skips preprocessing for everything.")
    elif args.parseq_no_preprocess:
        print("PARSeq will use original-image line crops while the other OCR models use preprocessed line crops.")

    print("\nLoading EasyOCR...")

    t0 = perf_counter()
    try:
        reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)
    except Exception as exc:
        raise SystemExit(f"EasyOCR failed to load: {exc}")
    easy_load_time = perf_counter() - t0
    print(f"EasyOCR load time: {fmt_time(easy_load_time)}")

    print("\nLoading optional OCR models...")
    parseq = None if args.no_parseq else load_optional("PARSeq", PARSeqRunner)
    trocr = None if args.no_trocr else load_optional("TrOCR", TrOCRRunner)
    paddle = None if args.no_paddle else load_optional("PaddleOCR", PaddleRunner)

    print("\nRunning EasyOCR detector/readtext on the full poster crop...")
    t0 = perf_counter()
    raw = run_easyocr_readtext(reader, ocr_img)
    detect_time = perf_counter() - t0

    t0 = perf_counter()
    raw = sort_easyocr_results(raw)
    sort_time = perf_counter() - t0

    print(f"EasyOCR detection/readtext time: {fmt_time(detect_time)}")
    print(f"Sort detected lines time:        {fmt_time(sort_time)}")

    if not raw:
        total_time = perf_counter() - total_t0
        print("\nNo EasyOCR text boxes found.")
        print(f"Total script time: {fmt_time(total_time)}")
        return

    print(f"\nDetected {len(raw)} text region(s).")

    per_model_times = {
        "crop": [],
        "parseq_raw_crop": [],
        "parseq": [],
        "easyocr_recognizer": [],
        "trocr": [],
        "paddle": [],
    }

    for i, item in enumerate(raw, 1):
        if len(item) < 3:
            continue

        box = item[0]
        easy_det_text = clean_ocr_text(item[1])
        easy_det_conf = norm_conf(item[2])

        t0 = perf_counter()
        line_crop = crop_easyocr_text_region(ocr_img, box)
        crop_time = perf_counter() - t0
        per_model_times["crop"].append(crop_time)

        parseq_line_crop = line_crop
        parseq_raw_crop_time = 0.0

        if args.parseq_no_preprocess and not args.no_preprocess:
            t0 = perf_counter()
            original_box = scale_box_points(box, ocr_img.shape, frame.shape)
            parseq_line_crop = crop_easyocr_text_region(frame, original_box)
            parseq_raw_crop_time = perf_counter() - t0
            per_model_times["parseq_raw_crop"].append(parseq_raw_crop_time)

        print()
        print("=" * 100)
        print(f"LINE {i}")
        print(f"Line crop time:            {fmt_time(crop_time)}")
        if args.parseq_no_preprocess and not args.no_preprocess:
            print(f"PARSeq raw crop time:      {fmt_time(parseq_raw_crop_time)}")
        print(
            f"EasyOCR detector/readtext: {easy_det_text!r} "
            f"| conf={easy_det_conf:.4f} "
            f"| time=in full-image detect step"
        )

        if parseq is not None and parseq_line_crop is not None:
            t0 = perf_counter()
            parseq_text, parseq_conf = parseq(parseq_line_crop)
            parseq_text = maybe_split_parseq_text(parseq_text)
            parseq_time = perf_counter() - t0
            per_model_times["parseq"].append(parseq_time)

            print(
                f"PARSeq:                    {parseq_text!r} "
                f"| conf={parseq_conf:.4f} "
                f"| time={fmt_time(parseq_time)}"
            )
        else:
            print("PARSeq:                    [unavailable] | conf=0.0000 | time=0.0000s")

        t0 = perf_counter()
        easy_rec_text, easy_rec_conf = run_easyocr_recognizer_only(reader, line_crop)
        easy_rec_time = perf_counter() - t0
        per_model_times["easyocr_recognizer"].append(easy_rec_time)

        print(
            f"EasyOCR recognizer:        {easy_rec_text!r} "
            f"| conf={easy_rec_conf:.4f} "
            f"| time={fmt_time(easy_rec_time)}"
        )

        if trocr is not None:
            t0 = perf_counter()
            trocr_text, trocr_conf = trocr(line_crop)
            trocr_time = perf_counter() - t0
            per_model_times["trocr"].append(trocr_time)

            print(
                f"TrOCR:                     {trocr_text!r} "
                f"| conf={trocr_conf:.4f} "
                f"| time={fmt_time(trocr_time)}"
            )
        else:
            print("TrOCR:                     [unavailable] | conf=0.0000 | time=0.0000s")

        if paddle is not None:
            t0 = perf_counter()
            paddle_text, paddle_conf = paddle(line_crop)
            paddle_time = perf_counter() - t0
            per_model_times["paddle"].append(paddle_time)

            print(
                f"PaddleOCR:                 {paddle_text!r} "
                f"| conf={paddle_conf:.4f} "
                f"| time={fmt_time(paddle_time)}"
            )
        else:
            print("PaddleOCR:                 [unavailable] | conf=0.0000 | time=0.0000s")

    total_time = perf_counter() - total_t0

    print()
    print("=" * 100)
    print("TIMING SUMMARY")
    print("=" * 100)
    print(f"Image read:                    {fmt_time(read_time)}")
    print(f"Preprocess step:               {fmt_time(preprocess_time)}")
    print(f"EasyOCR load:                  {fmt_time(easy_load_time)}")
    print(f"EasyOCR full-image detection:  {fmt_time(detect_time)}")
    print(f"Sort detected lines:           {fmt_time(sort_time)}")

    for name, values in per_model_times.items():
        if values:
            print(
                f"{name:28s} total={fmt_time(sum(values))} "
                f"| avg={fmt_time(float(np.mean(values)))} "
                f"| n={len(values)}"
            )

    print(f"Total script time:             {fmt_time(total_time)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
