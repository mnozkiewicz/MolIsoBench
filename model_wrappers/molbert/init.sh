#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Install molbert package into the venv
if ! "$PYTHON" -c "import molbert" 2>/dev/null; then
    echo "Installing molbert..."
    "$PYTHON" -m pip install -e "$SCRIPT_DIR/molbert_repo" -q
fi

# Download pretrained weights
WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [[ ! -f "$WEIGHTS_DIR/molbert_100epochs/checkpoints/last.ckpt" ]]; then
    echo "Downloading MolBERT weights..."
    mkdir -p "$WEIGHTS_DIR"
    wget -q -O "$WEIGHTS_DIR/weights.zip" https://ndownloader.figshare.com/files/25611290
    unzip -q "$WEIGHTS_DIR/weights.zip" -d "$WEIGHTS_DIR"
    rm "$WEIGHTS_DIR/weights.zip"
fi
