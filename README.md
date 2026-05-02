# Digital Image Processing Project
Authors: Tauha Khan, Logan Nunno, Dawud Shakir

# Needed Packages

The GUI requires the following Python packages:

| Package | Used In | Purpose |
|---------|---------|---------|
| `PyQt6` | `GUI_Main.py` | GUI framework |
| `opencv-python` | `GUI_Main.py`, `ocr_engine.py` | Image/video processing |
| `Pillow` | `GUI_Main.py`, `caption_engine.py` | Image handling |
| `pyttsx3` | `audio_engine.py` | Text-to-speech |
| `transformers` | `caption_engine.py` | BLIP image captioning model |
| `torch` | `caption_engine.py`, `sam_engine.py` | ML backend |
| `torchvision` | `ocr_engine.py` | Image transformations for OCR |
| `easyocr` | `ocr_engine.py` | Optical character recognition |
| `numpy` | `ocr_engine.py` | Array operations |
| `timm` | `ocr_engine.py` | Image models for PARSeq |
| `nltk` | `ocr_engine.py` | Text tokenization for PARSeq |
| `wordninja` | `ocr_engine.py` | Word splitting for OCR |
| `pytorch-lightning` | `ocr_engine.py` | Required by PARSeq |
| `huggingface_hub` | `sam_engine.py` | Model weight downloading |
| `sam3` | `sam_engine.py` | Segment Anything Model 3 |
| `groundingdino` | `dino_engine.py` | Object detection model |

### Install All at Once (Main GUI)

```bash
pip install PyQt6 opencv-python Pillow pyttsx3 transformers torch torchvision easyocr numpy huggingface_hub pycocotools timm einops ftfy pywin32 wordninja nltk pytorch-lightning
pip install git+https://github.com/facebookresearch/sam3.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### Install All at Once (Debugging GUI)

The `Debugging_GUI` uses YOLO for rapid video processing instead of SAM3 or Grounding DINO.

```bash
pip install PyQt6 opencv-python Pillow pyttsx3 transformers torch torchvision easyocr numpy timm einops ftfy pywin32 wordninja nltk pytorch-lightning ultralytics
```

# Computer Vision Engines

The project supports two swappable engines for object detection and extraction:

1. **SAM3 (Segment Anything Model 3):** High-precision segmentation and tracking.
2. **Grounding DINO:** Language-guided object detection.

To switch between models, uncomment the desired model worker in `GUI/GUI_Main.py` inside the `start_model_processing` method.

### Live Results Gallery
The application now features a **Live Results Gallery**. As the video is processed, discovered posters appear in the grid immediately. If a higher-quality crop of a poster is found later in the video, the thumbnail and description will automatically update in real-time.

### Hugging Face Access (SAM3)

SAM3 weights are gated on Hugging Face. To use SAM3:
1. Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3).
2. Create a "Read" token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Log in via your terminal:
   ```bash
   huggingface-cli login
   ```
   Or set the `HF_TOKEN` environment variable on your system.

# Accessibility & Screen Reader

The application includes a built-in **Screen Reader** for visually impaired users. When enabled:
- **Voice Guidance**: All focused elements (buttons, checkboxes, gallery items) are narrated aloud.
- **AI Descriptions**: Narrates visual descriptions of posters and detected text.
- **Contextual Alerts**: Announces screen transitions and dialog states.

**Toggle**: Use the checkbox at the bottom of the window or the keyboard shortcut.

# Keyboard Shortcuts

The GUI is fully navigable without a mouse.

### Global Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl + Alt + S` | **Toggle Screen Reader** on/off |

### Menu Screen

| Key | Action |
|-----|--------|
| `1` | Select Video File |
| `2` | Start Live Record |

### Processing Screen

| Key | Action |
|-----|--------|
| `Space` | Finish and view posters |
| `Escape` | Finish and view posters |

### Results Screen

| Key | Action |
|-----|--------|
| `←` `→` | Move between posters (left/right) |
| `↑` `↓` | Move between poster rows (up/down) |
| `Enter` | Open focused poster in zoom view + TTS |
| `Tab` | Cycle through posters |
| `T` | Load test images |
| `X` | **Stop Processing early** (keeps current posters) |

When a poster is focused via keyboard, its AI-generated description is shown at the top of the screen.

### Zoom / Poster Viewer

| Key | Action |
|-----|--------|
| `R` | **Read OCR Text aloud** |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `←` `→` `↑` `↓` | Pan the image |
| `0` | Reset zoom and pan |
| `Escape` | Close viewer |
