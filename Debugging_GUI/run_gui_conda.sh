#!/usr/bin/env bash
set -euo pipefail

# Poster Reader GUI setup + launch script for macOS/Linux.
# Put this file in the same folder as GUI_Main.py and requirements.txt.
#
# Normal run:
#   chmod +x run_gui_conda.sh
#   ./run_gui_conda.sh
#
# Optional recreate:
#   FORCE_RECREATE=1 ./run_gui_conda.sh

ENV_NAME="${ENV_NAME:-poster-reader}"
PY_VER="${PY_VER:-3.11}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH."
    echo "Open a terminal where conda/miniforge is initialized, then run this again."
    exit 1
fi

echo
echo "=== Using project folder ==="
echo "$SCRIPT_DIR"

if [[ "${FORCE_RECREATE:-0}" == "1" ]]; then
    echo
    echo "=== Removing old conda environment: $ENV_NAME ==="
    conda env remove -y -n "$ENV_NAME" || true
fi

echo
echo "=== Checking conda environment: $ENV_NAME ==="
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME' with Python $PY_VER..."
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
else
    echo "Environment already exists."
fi

echo
echo "=== Installing/updating pip tools ==="
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel

echo
echo "=== Installing requirements ==="
if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found in:"
    echo "$SCRIPT_DIR"
    exit 1
fi

conda run -n "$ENV_NAME" python -m pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

echo
echo "=== Checking required project files ==="
missing=0
for file in GUI_Main.py yolo_engine.py ocr_engine.py caption_engine.py audio_engine.py; do
    if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
        echo "MISSING: $file"
        missing=1
    fi
done

if [[ "$missing" == "1" ]]; then
    echo
    echo "ERROR: One or more required project files are missing."
    exit 1
fi

echo
echo "=== Starting Poster Reader GUI ==="
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/GUI_Main.py"
