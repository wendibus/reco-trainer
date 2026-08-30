#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$(cd "$PROJECT_DIR/.." && pwd)}"
ML_WORKER="$PROJECT_DIR/../reco-trainer-mac/Sources/RecoTrainerMac/Resources/ml_worker.py"
VERSION="0.9.0"
STAGE_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

copy_common() {
  local destination="$1"
  mkdir -p "$destination"
  rsync -a \
    --exclude '.DS_Store' \
    --exclude '.git' \
    --exclude '.next' \
    --exclude '.vinext' \
    --exclude '.wrangler' \
    --exclude 'dist' \
    --exclude 'node_modules' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$PROJECT_DIR/" "$destination/"
  cp "$ML_WORKER" "$destination/ml_worker.py"
}

WINDOWS_DIR="$STAGE_DIR/Reco Trainer Windows"
copy_common "$WINDOWS_DIR"
mv "$WINDOWS_DIR/Start Reco Preview Windows.bat" "$WINDOWS_DIR/Start Reco Trainer Windows.bat"
rm -f "$WINDOWS_DIR/Start Reco Preview.command" "$WINDOWS_DIR/Start Reco Trainer.command" "$WINDOWS_DIR/Stop Reco Preview.command" "$WINDOWS_DIR/start-preview-linux.sh"
rm -rf "$WINDOWS_DIR/docker" "$WINDOWS_DIR/scripts" "$WINDOWS_DIR/.openai"
rm -f "$WINDOWS_DIR/Dockerfile" "$WINDOWS_DIR/.dockerignore" "$WINDOWS_DIR/docker-compose.yml" "$WINDOWS_DIR/frame_extractor.swift"
xattr -cr "$WINDOWS_DIR"
rm -f "$OUTPUT_DIR/Reco Trainer Windows 0.9.zip"
(cd "$STAGE_DIR" && zip -q -r "$OUTPUT_DIR/Reco Trainer Windows 0.9.zip" "Reco Trainer Windows")

LINUX_DIR="$STAGE_DIR/Reco Trainer Linux"
copy_common "$LINUX_DIR"
mv "$LINUX_DIR/start-preview-linux.sh" "$LINUX_DIR/Start Reco Trainer Linux.sh"
chmod +x "$LINUX_DIR/Start Reco Trainer Linux.sh" "$LINUX_DIR/docker/start-container.sh"
rm -f "$LINUX_DIR/Start Reco Preview.command" "$LINUX_DIR/Start Reco Trainer.command" "$LINUX_DIR/Stop Reco Preview.command" "$LINUX_DIR/Start Reco Preview Windows.bat"
rm -rf "$LINUX_DIR/docker" "$LINUX_DIR/scripts" "$LINUX_DIR/.openai"
rm -f "$LINUX_DIR/Dockerfile" "$LINUX_DIR/.dockerignore" "$LINUX_DIR/docker-compose.yml" "$LINUX_DIR/frame_extractor.swift"
xattr -cr "$LINUX_DIR"
rm -f "$OUTPUT_DIR/Reco Trainer Linux 0.9.zip"
(cd "$STAGE_DIR" && zip -q -r "$OUTPUT_DIR/Reco Trainer Linux 0.9.zip" "Reco Trainer Linux")

DOCKER_DIR="$STAGE_DIR/Reco Trainer Docker"
copy_common "$DOCKER_DIR"
rm -f "$DOCKER_DIR/Start Reco Preview.command" "$DOCKER_DIR/Start Reco Trainer.command" "$DOCKER_DIR/Stop Reco Preview.command" "$DOCKER_DIR/Start Reco Preview Windows.bat" "$DOCKER_DIR/start-preview-linux.sh"
rm -rf "$DOCKER_DIR/scripts" "$DOCKER_DIR/.openai"
rm -f "$DOCKER_DIR/frame_extractor.swift"
xattr -cr "$DOCKER_DIR"
rm -f "$OUTPUT_DIR/Reco Trainer Docker 0.9.zip"
(cd "$STAGE_DIR" && zip -q -r "$OUTPUT_DIR/Reco Trainer Docker 0.9.zip" "Reco Trainer Docker")

printf '%s\n' \
  "$OUTPUT_DIR/Reco Trainer Windows 0.9.zip" \
  "$OUTPUT_DIR/Reco Trainer Linux 0.9.zip" \
  "$OUTPUT_DIR/Reco Trainer Docker 0.9.zip"
