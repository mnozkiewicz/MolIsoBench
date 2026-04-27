#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

TORCH_VERSION="1.13.1"
CUDA_TAG="cpu"  # change to cu116 if running on CUDA GPU

if ! "$PYTHON" -c "import torch_scatter" 2>/dev/null; then
    "$PYTHON" -m pip install \
        "torch-scatter==2.1.0+pt113${CUDA_TAG}" \
        "torch-sparse==0.6.15+pt113${CUDA_TAG}" \
        "torch-geometric==1.7.2" \
        --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html" \
        -q
fi
