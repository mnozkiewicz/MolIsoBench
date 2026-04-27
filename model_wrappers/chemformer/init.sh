#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

uv pip install --python "$SCRIPT_DIR/.venv/bin/python" \
    -e "$SCRIPT_DIR/Chemformer" --no-deps

if [[ ! -d "$SCRIPT_DIR/weights" ]]; then
    echo ""
    echo "WARNING: No weights found at $SCRIPT_DIR/weights/"
    echo "Download from: https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq"
    echo "and place the .ckpt files under $SCRIPT_DIR/weights/"
    echo ""
fi
