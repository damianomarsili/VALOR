#!/usr/bin/env bash
set -euo pipefail

REPO_ID="glab-caltech/VALOR-GroundingDINO"
FILENAME_IN_REPO="VALOR-GroundingDINO.pth"
TARGET_PATH="modules/GroundingDINO/VALOR-checkpoints/VALOR-GroundingDINO.pth"

echo "Downloading $FILENAME_IN_REPO from $REPO_ID ..."

# Make sure target directory exists
mkdir -p "$(dirname "$TARGET_PATH")"

# Use the Hugging Face CLI to download the file
huggingface-cli download \
  "$REPO_ID" \
  "$FILENAME_IN_REPO" \
  --local-dir "$(dirname "$TARGET_PATH")" \
  --local-dir-use-symlinks False

# If the filename in the repo differs, you can rename it explicitly:
if [ "$FILENAME_IN_REPO" != "$(basename "$TARGET_PATH")" ]; then
  mv "$(dirname "$TARGET_PATH")/$FILENAME_IN_REPO" "$TARGET_PATH"
fi

echo "Checkpoint saved to: $TARGET_PATH"