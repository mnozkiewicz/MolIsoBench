#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

# Download pretrained weights (no pip/python logic needed here anymore)
if [[ ! -f "$WEIGHTS_DIR/molbert_100epochs/checkpoints/last.ckpt" ]]; then
    echo "Downloading MolBERT weights..."
    mkdir -p "$WEIGHTS_DIR"
    wget -q -O "$WEIGHTS_DIR/weights.zip" https://ndownloader.figshare.com/files/25611290
    unzip -q "$WEIGHTS_DIR/weights.zip" -d "$WEIGHTS_DIR"
    rm "$WEIGHTS_DIR/weights.zip"
    echo "Weights downloaded successfully."
else
    echo "Weights already exist. Skipping download."
fi