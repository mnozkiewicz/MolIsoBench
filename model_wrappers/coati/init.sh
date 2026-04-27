#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
uv pip install --python "$SCRIPT_DIR/.venv/bin/python" -e "$SCRIPT_DIR/COATI" --no-deps
