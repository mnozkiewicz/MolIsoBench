#!/usr/bin/env bash
# Run a model wrapper on a single input file inside its own isolated uv-managed venv.
#
# Usage:
#   ./embed_model.sh <model_dir> <input.parquet> [options...]
#   ./embed_model.sh --reinstall <model_dir> <input.parquet> [options...]
#
# For running on all datasets in parallel use run_embedding.py instead.
#
# Examples:
#   ./embed_model.sh model_wrappers/chembert datasets/smiles/graph8c.parquet
#   ./embed_model.sh model_wrappers/chembert datasets/smiles/graph8c.parquet --model-kwargs '{"pooling":"mean"}'

set -euo pipefail

REINSTALL=false
if [[ "${1:-}" == "--reinstall" ]]; then
    REINSTALL=true
    shift
fi

MODEL_DIR="${1:?Usage: $0 [--reinstall] <model_dir> <input.parquet> [args...]}"
INPUT="${2:?Usage: $0 [--reinstall] <model_dir> <input.parquet> [args...]}"
shift 2

PROJECT_ROOT="$(pwd)"

if [[ "$REINSTALL" == "true" ]]; then
    rm -rf "$MODEL_DIR/.venv"
fi

echo "Syncing venv for $MODEL_DIR..."
uv sync --project "$MODEL_DIR"

echo "Running init.sh..."
bash "$MODEL_DIR/init.sh"

echo "Running embeddings..."
PYTHONPATH="$PROJECT_ROOT" uv run --project "$MODEL_DIR" python "$PROJECT_ROOT/model_wrappers/run.py" "$MODEL_DIR" "$INPUT" "$@"
