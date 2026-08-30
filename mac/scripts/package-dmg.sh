#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h}/.."
DIST_DIR="${RECO_DIST_DIR:-$ROOT_DIR/dist}"
APP_DIR="$DIST_DIR/Reco Trainer.app"
STAGING_DIR="$DIST_DIR/dmg-stage"
DMG_PATH="$DIST_DIR/Reco-Trainer-Mac-0.9.0.dmg"

"$ROOT_DIR/scripts/package-app.sh"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
ditto "$APP_DIR" "$STAGING_DIR/Reco Trainer.app"
ln -s /Applications "$STAGING_DIR/Applications"
rm -f "$DMG_PATH"
hdiutil create -volname "Reco Trainer" -srcfolder "$STAGING_DIR" -ov -format UDZO "$DMG_PATH"
codesign --force --sign "${RECO_SIGN_IDENTITY:--}" "$DMG_PATH"
echo "$DMG_PATH"
