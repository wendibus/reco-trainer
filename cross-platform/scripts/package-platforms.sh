#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$(cd "$PROJECT_DIR/.." && pwd)}"
ML_WORKER="$PROJECT_DIR/ml_worker.py"
VERSION="0.22.0"
STAGE_DIR="$(mktemp -d)"

# Build the web interface once; Windows/Linux packages ship it prebuilt in ui/,
# so end users need Python only (no Node.js, no npm). The Docker package builds
# it inside the image instead.
(cd "$PROJECT_DIR" && npm ci --no-audit --no-fund && npm run build)
[ -f "$PROJECT_DIR/ui/index.html" ] || { echo "ui/ build is missing" >&2; exit 1; }

# Source-only files that end users of the Windows/Linux packages do not need.
DEV_FILES=(app public index.html package.json package-lock.json tsconfig.json vite.config.ts tests)

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
    --exclude '/ui' \
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
rm -rf "$WINDOWS_DIR/ui" && cp -R "$PROJECT_DIR/ui" "$WINDOWS_DIR/ui"
(cd "$WINDOWS_DIR" && rm -rf "${DEV_FILES[@]}")
rm -rf "$WINDOWS_DIR/docker" "$WINDOWS_DIR/scripts" "$WINDOWS_DIR/.openai"
rm -f "$WINDOWS_DIR/Dockerfile" "$WINDOWS_DIR/.dockerignore" "$WINDOWS_DIR/docker-compose.yml" "$WINDOWS_DIR/frame_extractor.swift"
xattr -cr "$WINDOWS_DIR"
rm -f "$OUTPUT_DIR/Reco.Trainer.Windows.$VERSION.zip"
(cd "$STAGE_DIR" && zip -q -r "$OUTPUT_DIR/Reco.Trainer.Windows.$VERSION.zip" "Reco Trainer Windows")

LINUX_DIR="$STAGE_DIR/Reco Trainer Linux"
copy_common "$LINUX_DIR"
mv "$LINUX_DIR/start-preview-linux.sh" "$LINUX_DIR/Start Reco Trainer Linux.sh"
chmod +x "$LINUX_DIR/Start Reco Trainer Linux.sh" "$LINUX_DIR/docker/start-container.sh"
rm -f "$LINUX_DIR/Start Reco Preview.command" "$LINUX_DIR/Start Reco Trainer.command" "$LINUX_DIR/Stop Reco Preview.command" "$LINUX_DIR/Start Reco Preview Windows.bat"
rm -rf "$LINUX_DIR/ui" && cp -R "$PROJECT_DIR/ui" "$LINUX_DIR/ui"
(cd "$LINUX_DIR" && rm -rf "${DEV_FILES[@]}")
rm -rf "$LINUX_DIR/docker" "$LINUX_DIR/scripts" "$LINUX_DIR/.openai"
rm -f "$LINUX_DIR/Dockerfile" "$LINUX_DIR/.dockerignore" "$LINUX_DIR/docker-compose.yml" "$LINUX_DIR/frame_extractor.swift"
xattr -cr "$LINUX_DIR"
rm -f "$OUTPUT_DIR/Reco.Trainer.Linux.$VERSION.zip"
(cd "$STAGE_DIR" && zip -q -r "$OUTPUT_DIR/Reco.Trainer.Linux.$VERSION.zip" "Reco Trainer Linux")

DOCKER_DIR="$STAGE_DIR/Reco Trainer Docker"
copy_common "$DOCKER_DIR"
rm -f "$DOCKER_DIR/Start Reco Preview.command" "$DOCKER_DIR/Start Reco Trainer.command" "$DOCKER_DIR/Stop Reco Preview.command" "$DOCKER_DIR/Start Reco Preview Windows.bat" "$DOCKER_DIR/start-preview-linux.sh"
rm -rf "$DOCKER_DIR/scripts" "$DOCKER_DIR/.openai"
rm -f "$DOCKER_DIR/frame_extractor.swift"
xattr -cr "$DOCKER_DIR"
rm -f "$OUTPUT_DIR/Reco.Trainer.Docker.$VERSION.zip"
(cd "$STAGE_DIR" && zip -q -r "$OUTPUT_DIR/Reco.Trainer.Docker.$VERSION.zip" "Reco Trainer Docker")

printf '%s\n' \
  "$OUTPUT_DIR/Reco.Trainer.Windows.$VERSION.zip" \
  "$OUTPUT_DIR/Reco.Trainer.Linux.$VERSION.zip" \
  "$OUTPUT_DIR/Reco.Trainer.Docker.$VERSION.zip"
