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
