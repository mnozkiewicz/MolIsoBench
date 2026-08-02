#!/usr/bin/env bash
# CLAMP now runs through scikit-fingerprints (installed via pyproject.toml on
# `uv sync`), so there is no repo to build or patch. This file is intentionally
# a no-op, kept only because embed_model.sh always runs it.
set -euo pipefail
