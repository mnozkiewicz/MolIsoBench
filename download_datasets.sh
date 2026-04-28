#!/usr/bin/env bash

set -euo pipefail

DOWNLOAD_ALL=false
if [[ "${1:-}" == "--download-all" ]]; then
  DOWNLOAD_ALL=true
fi

# Directory to store datasets
OUT_DIR="datasets/g6/"
BASE_URL="https://users.cecs.anu.edu.au/~bdm/data"

mkdir -p "$OUT_DIR"

SMALL_FILES=()
MEDIUM_FILES=()
LARGE_FILES=()

# Sets of all connected non-isomorphic graphs with a given number of vertices 
for i in {2..6}; do
  SMALL_FILES+=("$BASE_URL/graph${i}c.g6")
done

for i in {7..9}; do
  MEDIUM_FILES+=("$BASE_URL/graph${i}c.g6")
done

LARGE_FILES+=("$BASE_URL/graph10c.g6.gz")

# # Sets of all connected non-isomorphic graphs with a given number of edges
for i in {1..12}; do
  SMALL_FILES+=("$BASE_URL/ge${i}c.g6")
done

for i in {13..14}; do
  MEDIUM_FILES+=("$BASE_URL/ge${i}c.g6")
done

LARGE_FILES+=("$BASE_URL/ge15c.g6" "$BASE_URL/ge16c.g6" "$BASE_URL/ge17c.g6.gz" "$BASE_URL/ge18c.g6.gz")


# Eulerian graphs
for i in {3..9}; do
  SMALL_FILES+=("$BASE_URL/eul${i}c.g6")
done

MEDIUM_FILES+=("$BASE_URL/eul11c.g6.gz")

# Chordal graphs
for i in {4..9}; do
  SMALL_FILES+=("$BASE_URL/chordal${i}.g6")
done

MEDIUM_FILES+=("$BASE_URL/chordal10.g6")
LARGE_FILES+=("$BASE_URL/chordal11.g6" "$BASE_URL/chordal12.g6.gz" "$BASE_URL/chordal13.g6.gz")


# Perfect graphs
for i in {5..8}; do
  SMALL_FILES+=("$BASE_URL/perfect${i}.g6")
done

MEDIUM_FILES+=("$BASE_URL/perfect9.g6" "$BASE_URL/perfect10.g6")
LARGE_FILES+=("$BASE_URL/perfect11.g6.gz")


# Strongly regular graphs
SMALL_FILES+=(
    "$BASE_URL/sr291467.g6"
    "$BASE_URL/sr351668.g6"
    "$BASE_URL/sr351899.g6"
    "$BASE_URL/sr361446.g6"
    "$BASE_URL/sr371889some.g6"
)
MEDIUM_FILES+=("$BASE_URL/sr361566.g6.gz")


# Planar graphs
for i in {4..8}; do
  SMALL_FILES+=("$BASE_URL/planar_conn.${i}.g6")
done

MEDIUM_FILES+=("$BASE_URL/planar_conn.9.g6")
LARGE_FILES+=("$BASE_URL/planar_conn.10.g6.gz")


# Irregular graphs
for i in {8..15}; do
  SMALL_FILES+=("$BASE_URL/highlyirregular${i}.g6")
done

URLS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --small|-s)
      URLS+=("${SMALL_FILES[@]}")
      shift
      ;;
    --medium|-m)
      URLS+=("${MEDIUM_FILES[@]}")
      shift
      ;;
    --large|-l)
      URLS+=("${LARGE_FILES[@]}")
      shift
      ;;
    --all|-a)
      URLS+=("${SMALL_FILES[@]}")
      URLS+=("${MEDIUM_FILES[@]}")
      URLS+=("${LARGE_FILES[@]}")
      shift
      ;;
    --)
      shift
      EXTRA_CMD=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done


# If no size option was selected, default to --small
if [[ ${#URLS[@]} -eq 0 ]]; then
  echo "No option specified, defaulting to --small"
  URLS=("${SMALL_FILES[@]}")
fi


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