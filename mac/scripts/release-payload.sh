#!/bin/zsh
# Builds and signs the update payload for a release (no new app/DMG needed):
#   mac/scripts/release-payload.sh
# Output in mac/dist/: Reco.Trainer.Payload.<version>.zip and .zip.sig - attach both to the GitHub release.
# Needs the signing key from `swift mac/tools/payload-signing.swift keygen`.
set -euo pipefail

ROOT_DIR="${0:A:h}/.."
OUT_DIR="${RECO_DIST_DIR:-$ROOT_DIR/dist}"
mkdir -p "$OUT_DIR"

ZIP_PATH="$("$ROOT_DIR/../cross-platform/scripts/build-payload.sh" "$OUT_DIR/payload-build" | tail -1)"
FINAL="$OUT_DIR/${ZIP_PATH:t}"
mv -f "$ZIP_PATH" "$FINAL"
rm -rf "$OUT_DIR/payload-build"
swift "$ROOT_DIR/tools/payload-signing.swift" sign "$FINAL" ${RECO_SIGNING_KEY:+"$RECO_SIGNING_KEY"} >/dev/null
echo "$FINAL"
echo "$FINAL.sig"
shasum -a 256 "$FINAL" "$FINAL.sig"
