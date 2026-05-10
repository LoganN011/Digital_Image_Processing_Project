# Poster Reader SAM3 Demo

This demo combines a **Colab-hosted SAM3 detection server**, a **local PyQt camera client**, and a **local OCR viewer/engine** for live poster and flyer reading.

The system works in three stages:

1. `sam_client.py` captures live camera frames and sends them to the SAM3 server.
2. `sam_server.ipynb` runs SAM3 inference in Colab, tracks detections across frames, and returns poster/flyer boxes with track IDs.
3. `sam_client.py` crops tracked poster regions and sends them to `ocr_engine.py`, which detects text lines with EasyOCR and recognizes them with PARSeq.

```text
Local camera
    |
    v
sam_client.py  -- HTTP frames -->  SAM3 server in Colab
    |                                  |
    |<-------- tracked boxes ----------|
    |
    +-- local socket crops -->  ocr_engine.py
                                  |
                                  +-- EasyOCR text detection
                                  +-- PARSeq text recognition
```

## Files

### `sam_server.ipynb`

Colab notebook that sets up and runs the remote SAM3 server.

It:

- installs the server and SAM3 dependencies
- downloads the SAM3 checkpoint from the configured Google Drive link
- writes `sam_server.py` inside Colab
- loads the SAM3 image model from the local checkpoint rather than from Hugging Face
- exposes three FastAPI endpoints:
  - `GET /health` — reports server status, device, track count, and checkpoint path
  - `POST /infer` — runs SAM3 detection, duplicate suppression, and lightweight IoU-based tracking
  - `POST /reset` — clears the tracker state
- starts the FastAPI app with Uvicorn
- opens a Cloudflare tunnel so the local client can reach the Colab server

Leave the Cloudflare tunnel cell running and copy the printed `https://...trycloudflare.com` URL into the client.

### `sam_client.py`

Local PyQt application for live camera input and SAM3 interaction.

It:

- captures frames from a local webcam
- resizes and JPEG-encodes frames before upload
- sends frames to the remote `/infer` endpoint
- draws tracked boxes and IDs on the live preview
- exposes controls for prompt, confidence threshold, tracking IoU, duplicate IoU, max misses, JPEG quality, max width, target FPS, and timeout
- can pause/resume capture, reset remote tracking, and apply updated settings while running
- launches `ocr_engine.py` automatically when OCR is enabled and the OCR app is not already running
- crops each tracked poster region and sends it to the OCR engine over a local TCP socket

### `ocr_engine.py`

Local PyQt OCR viewer and worker process.

It:

- listens for poster crops on a local socket, by default `127.0.0.1:8765`
- loads:
  - **EasyOCR** as the text-line detector
  - **PARSeq** as the text recognizer
- detects text regions within each poster crop
- recognizes each detected line and stores confidence values
- displays:
  - a table of OCR results
  - the selected crop with detected text boxes overlaid
  - the recognized lines and confidences
- keeps following the newest OCR result until the user clicks an older row

## Requirements

### Colab server

The notebook installs the required server-side packages itself, including:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `pillow`
- `requests`
- `opencv-python-headless`
- `gdown`
- `sam3`

A CUDA-enabled Colab runtime (e.g., GP4) is recommended for practical inference speed.

### Local client and OCR engine

Install the local dependencies in the Python environment used to run the client:

```bash
pip install pyqt6 opencv-python numpy requests pillow torch torchvision easyocr
```

`ocr_engine.py` loads PARSeq through `torch.hub` the first time it starts, so the first OCR launch may take longer while model files are downloaded and initialized.

## Setup

### 1. Start the SAM3 server in Colab

Open `sam_server.ipynb` in Colab and run the cells from top to bottom.

The notebook will:

1. install dependencies
2. verify PyTorch/CUDA
3. download the SAM3 checkpoint
4. write and start `sam_server.py`
5. confirm that `/health` is responding
6. start a Cloudflare tunnel

When the tunnel starts, copy the printed URL, for example:

```text
https://example-name.trycloudflare.com
```

Keep that tunnel cell running while using the local client.

### 2. Place the local files together

Keep these two files in the same local directory:

```text
sam_client.py
ocr_engine.py
```

`sam_client.py` looks for `ocr_engine.py` beside itself when it needs to launch the OCR app.

### 3. Run the local client

```bash
python sam_client.py --server https://example-name.trycloudflare.com
```

The default prompt is:

```text
poster, flyer
```

If OCR is enabled, the client will start the OCR viewer automatically when needed. You can also launch it manually:

```bash
python ocr_engine.py
```

## Basic Use

1. Start the Colab server and copy the tunnel URL.
2. Launch `sam_client.py` with that URL.
3. Confirm the camera index and prompt.
4. Press **Start**.
5. The live view will show tracked poster/flyer boxes.
6. When OCR is enabled, crop results will appear in the separate OCR viewer.

## Client Controls

| Control | Purpose |
| --- | --- |
| **Server** | Colab/Cloudflare server URL |
| **Prompt** | Text prompt sent to SAM3, such as `poster, flyer` |
| **Camera** | Local camera index, usually `0` |
| **Conf** | Minimum detection score kept from SAM3 |
| **Track IoU** | IoU threshold used to associate detections with existing tracks |
| **Duplicate IoU** | IoU threshold used to suppress overlapping duplicate detections |
| **Max misses** | Number of missed frames before a track is dropped |
| **JPEG quality** | Compression quality for frames sent to the server |
| **Max width** | Resize width before upload; lower values can improve responsiveness |
| **FPS** | Target rate for frames sent to the SAM3 server |
| **Timeout** | Request timeout in seconds |
| **OCR on** | Enables crop sending to the OCR app |
| **OCR host / port** | Local socket address for the OCR app |
| **OCR interval** | Minimum seconds between OCR sends for the same track |
| **OCR JPEG** | JPEG quality for crops sent to OCR |

Buttons:

| Button | Purpose |
| --- | --- |
| **Start** | Start camera capture and server requests |
| **Pause / Resume** | Pause or resume live processing |
| **Stop** | Stop the camera worker |
| **Reset** | Clear server-side tracking state |
| **Apply** | Apply updated settings while running |
| **Start OCR** | Start or connect to the OCR app manually |

## Default Local Settings

| Setting | Default |
| --- | --- |
| Prompt | `poster, flyer` |
| Confidence threshold | `0.25` |
| Track IoU | `0.60` |
| Duplicate IoU | `0.80` |
| Max misses | `8` |
| Upload JPEG quality | `80` |
| Max frame width | `480` |
| Target FPS | `30` |
| Request timeout | `30.0` s |
| OCR socket | `127.0.0.1:8765` |
| OCR interval | `2.0` s per track |
| OCR crop JPEG quality | `92` |

## OCR Output

The OCR viewer shows one row per processed crop with:

- receive time
- track ID
- detector score
- number of recognized lines
- recognized text preview
- OCR processing time in milliseconds
- frame number where the crop was acquired

Selecting a row shows the crop with detected text boxes and the recognized text lines with confidence values.

## Useful Endpoints

After the server is running:

```bash
curl http://127.0.0.1:8000/health
```

or, through the tunnel:

```bash
curl https://example-name.trycloudflare.com/health
```

To reset tracks manually:

```bash
curl -X POST https://example-name.trycloudflare.com/reset
```

## Troubleshooting
### NumPy / torchvision binary compatibility error in Colab

If the notebook shows:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

restart the Colab session and run all cells again.

### The local client cannot reach the server

- Make sure the Cloudflare tunnel cell is still running.
- Confirm that the URL copied into **Server** begins with `https://`.
- Check `/health` in a browser or with `curl`.

### OCR does not open

- Make sure `sam_client.py` and `ocr_engine.py` are in the same folder.
- Confirm OCR is enabled in the client.
- You can start the OCR viewer manually with:

```bash
python ocr_engine.py
```

### OCR starts slowly

The OCR engine loads EasyOCR and PARSeq at startup. The first run can take longer because PARSeq may need to download through `torch.hub`.

### Tracking IDs become stale or incorrect

Press **Reset** in the client to clear the server-side tracker and start new track IDs.

## Notes

- The server uses a lightweight IoU-based box tracker on top of SAM3 detections.
- Duplicate suppression is applied before tracking to reduce repeated overlapping detections.
- The OCR engine processes crops locally; full camera frames are only sent to the SAM3 server.
- The SAM3 checkpoint is loaded from the local Colab file downloaded by the notebook, with Hugging Face loading disabled.
