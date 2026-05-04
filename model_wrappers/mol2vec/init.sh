#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if ! "$PYTHON" -c "import mol2vec" 2>/dev/null; then
    echo "Installing mol2vec from git..."
    "$PYTHON" -m pip install "git+https://github.com/samoturk/mol2vec" -q
fi
