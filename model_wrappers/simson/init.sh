#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT="$SCRIPT_DIR/weights/pretrained_best_model.pth"

if [[ ! -f "$CKPT" ]]; then
    mkdir -p "$SCRIPT_DIR/weights"
    echo "Downloading SimSon weights from Google Drive..."
    uv run --project "$SCRIPT_DIR" python -m gdown \
        --folder "1EZhgntGlPzP9bmUMymaMVYO-DhBWJrxR" \
        -O "$SCRIPT_DIR/weights" -q
    PTH="$(find "$SCRIPT_DIR/weights" -name "pretrained_best_model.pth" | head -1)"
    if [[ -z "$PTH" ]]; then
        echo "ERROR: pretrained_best_model.pth not found after download."
        echo "Place it manually at: $CKPT"
        exit 1
    fi
    [[ "$PTH" != "$CKPT" ]] && mv "$PTH" "$CKPT"
    # Remove per-epoch checkpoints — only the best model is needed
    find "$SCRIPT_DIR/weights" -name "pre_model_epoch_*.pth" -delete
fi
