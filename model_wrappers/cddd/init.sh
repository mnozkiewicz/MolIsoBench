#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if ! "$PYTHON" -c "import cddd" 2>/dev/null; then
    "$PYTHON" -m pip install -e "$SCRIPT_DIR/cddd_repo" --no-deps -q
fi

if [[ ! -d "$SCRIPT_DIR/default_model" ]]; then
    echo "Downloading CDDD pretrained weights..."
    FILEID="1oyknOulq_j0w9kzOKKIHdTLo5HphT99h"
    DEST="$SCRIPT_DIR/default_model.zip"
    curl -c /tmp/cddd_cookie -s -L \
        "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
    curl -Lb /tmp/cddd_cookie \
        "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' /tmp/cddd_cookie)&id=${FILEID}" \
        -o "$DEST"
    rm -f /tmp/cddd_cookie
    unzip -q "$DEST" -d "$SCRIPT_DIR"
    rm "$DEST"
fi
