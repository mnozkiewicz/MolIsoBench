#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/GraphFP/mol/vocab.txt" "$SCRIPT_DIR/"
