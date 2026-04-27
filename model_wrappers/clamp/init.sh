#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# remove unused import that breaks modern setuptools (safe to run repeatedly)
sed -i '' 's/import pkg_resources//' "$SCRIPT_DIR/clamp_repo/setup.py" 2>/dev/null || true

uv pip install --python "$SCRIPT_DIR/.venv/bin/python" \
    -e "$SCRIPT_DIR/clamp_repo" --no-deps
