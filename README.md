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
| `torch` | `caption_engine.py` | ML backend for transformers |
| `easyocr` | `ocr_engine.py` | Optical character recognition |
| `numpy` | `ocr_engine.py` | Array operations for image preprocessing |

### Install All at Once

```bash
pip install PyQt6 opencv-python Pillow pyttsx3 transformers torch easyocr numpy
```

# Keyboard Shortcuts

The GUI is fully navigable without a mouse.

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

When a poster is focused via keyboard, its AI-generated description is shown at the top of the screen.

### Zoom / Poster Viewer

| Key | Action |
|-----|--------|
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `←` `→` `↑` `↓` | Pan the image |
| `0` | Reset zoom and pan |
| `Escape` | Close viewer |
