#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/release"
OUT_FILE="$OUT_DIR/AI-Police.zip"

mkdir -p "$OUT_DIR"

# Package tracked files from HEAD (excludes .git and local runtime artifacts)
git -C "$ROOT_DIR" archive --format=zip --output "$OUT_FILE" HEAD

echo "Created: $OUT_FILE"
