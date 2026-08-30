#!/usr/bin/env bash
set -euo pipefail

# CheMeleonFingerprint auto-downloads the pretrained message-passing weights to
# ~/.chemprop/chemeleon_mp.pt on first use. Pre-fetch them here so parallel
# workers don't race on the download.
CKPT_DIR="$HOME/.chemprop"
CKPT="$CKPT_DIR/chemeleon_mp.pt"
mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT" ]]; then
    echo "Downloading CheMeleon weights..."
    curl -fsSL -o "$CKPT" "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
fi
