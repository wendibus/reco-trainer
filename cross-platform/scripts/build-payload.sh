#!/usr/bin/env bash
# Builds the Reco Trainer "payload": the part of the Mac app that updates without a new
# app download - the web interface (ui/) plus the Python workers. The Mac launcher
# runs it from ~/Library/Application Support/Reco Trainer/payloads/<version>/.
#
#   scripts/build-payload.sh [output-dir]
#
# Writes <output-dir>/payload/ (a ready folder) and Reco.Trainer.Payload.<version>.zip.
# Environment: RECO_UI_SOURCE (use an existing built ui/ instead of running npm),
#              RECO_PAYLOAD_MIN_LAUNCHER (oldest launcher that may run this payload).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_DIR/dist-payload}"
VERSION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['version'])" "$PROJECT_DIR/package.json")"
MIN_LAUNCHER="${RECO_PAYLOAD_MIN_LAUNCHER:-0.20.0}"
UI_SOURCE="${RECO_UI_SOURCE:-}"

if [ -z "$UI_SOURCE" ]; then
  (cd "$PROJECT_DIR" && npm ci --no-audit --no-fund && npm run build)
  UI_SOURCE="$PROJECT_DIR/ui"
fi
[ -f "$UI_SOURCE/index.html" ] || { echo "Built interface not found: $UI_SOURCE" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
PAYLOAD_DIR="$OUTPUT_DIR/payload"
ZIP_PATH="$OUTPUT_DIR/Reco.Trainer.Payload.$VERSION.zip"
rm -rf "$PAYLOAD_DIR" "$ZIP_PATH" "$ZIP_PATH.sig"
mkdir -p "$PAYLOAD_DIR"

cp -R "$UI_SOURCE" "$PAYLOAD_DIR/ui"
# Python modules and data next to the workers; keep in sync with the Dockerfile COPY line.
cp "$PROJECT_DIR/local_worker.py" "$PROJECT_DIR/ml_worker.py" "$PROJECT_DIR/anonymize.py" "$PROJECT_DIR/teachers.py" "$PROJECT_DIR/vlm.py" "$PROJECT_DIR/model_registry.json" "$PROJECT_DIR/frame_extractor.swift" "$PAYLOAD_DIR/"
printf '{\n  "version": "%s",\n  "minLauncher": "%s"\n}\n' "$VERSION" "$MIN_LAUNCHER" > "$PAYLOAD_DIR/payload.json"
find "$PAYLOAD_DIR" -name '.DS_Store' -delete

# Zip the folder's contents (payload.json at the archive root), deterministic order.
(cd "$PAYLOAD_DIR" && find . -type f | sed 's#^\./##' | LC_ALL=C sort | zip -q -X "$ZIP_PATH" -@)
echo "$ZIP_PATH"
