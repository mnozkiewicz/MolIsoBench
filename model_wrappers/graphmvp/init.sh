#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    echo "Downloading GraphMVP pretrained weights..."
    "$PYTHON" -m gdown --folder "1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6" -O "$WEIGHTS_DIR" -q
fi

ZIP="$WEIGHTS_DIR/GraphMVP_simple_features_for_classification.zip"
if [[ -f "$ZIP" && ! -d "$WEIGHTS_DIR/GraphMVP_simple" ]]; then
    echo "Unzipping GraphMVP weights..."
    unzip -q "$ZIP" -d "$WEIGHTS_DIR/GraphMVP_simple"
fi
