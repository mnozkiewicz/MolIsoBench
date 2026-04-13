#!/usr/bin/env bash

set -euo pipefail

# Directory to store datasets
OUT_DIR="datasets/g6/"
BASE_URL="https://users.cecs.anu.edu.au/~bdm/data"

mkdir -p "$OUT_DIR"

URLS=()

# graph2c -> graph9c
for i in {2..9}; do
  URLS+=("$BASE_URL/graph${i}c.g6")
done

# special case
URLS+=("$BASE_URL/graph10c.g6.gz")

# ge1c -> ge16c
for i in {1..16}; do
  URLS+=("$BASE_URL/ge${i}c.g6")
done

URLS+=("$BASE_URL/ge17c.g6.gz")
URLS+=("$BASE_URL/ge18c.g6.gz")


# Eulerian graphs
for i in {3..10}; do
  URLS+=("$BASE_URL/eul${i}c.g6")
done

URLS+=("$BASE_URL/eul11c.g6.gz")

# Chordal graphs
for i in {4..11}; do
  URLS+=("$BASE_URL/chordal${i}.g6")
done

URLS+=("$BASE_URL/chordal12.g6.gz")
URLS+=("$BASE_URL/chordal13.g6.gz")


# Perfect graphs
for i in {5..10}; do
  URLS+=("$BASE_URL/perfect${i}.g6")
done

URLS+=("$BASE_URL/perfect11.g6.gz")


# Strongly regular graphs
URLS+=(
    "$BASE_URL/sr291467.g6"
    "$BASE_URL/sr351668.g6"
    "$BASE_URL/sr351899.g6"
    "$BASE_URL/sr361446.g6"
    "$BASE_URL/sr371889some.g6"
    "$BASE_URL/sr361566.g6.gz"
)


# Planar graphs
for i in {4..9}; do
  URLS+=("$BASE_URL/planar_conn.${i}.g6")
done

URLS+=("$BASE_URL/planar_conn.10.g6.gz")


# Irregular graphs
for i in {8..15}; do
  URLS+=("$BASE_URL/highlyirregular${i}.g6")
done

for URL in "${URLS[@]}"; do
  FILENAME=$(basename "$URL")
  OUTPUT_PATH="$OUT_DIR/$FILENAME"

  if [[ -f "$OUTPUT_PATH" ]]; then
    echo "Skipping $OUTPUT_PATH (already exists)"
    continue
  fi

  echo "Downloading $FILENAME -> $OUTPUT_PATH ..."
  curl -L -o "$OUTPUT_PATH" "$URL"
done

echo "Done downloading."

gunzip datasets/g6/*.gz