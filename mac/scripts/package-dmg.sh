#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h}/.."
DIST_DIR="${RECO_DIST_DIR:-$ROOT_DIR/dist}"
APP_DIR="$DIST_DIR/Reco Trainer.app"
STAGING_DIR="$DIST_DIR/dmg-stage"
DMG_PATH="$DIST_DIR/Reco-Trainer-Mac-0.11.0.dmg"
RW_DMG_PATH="$DIST_DIR/Reco-Trainer-Mac-0.11.0-rw.dmg"
MOUNT_DIR="$DIST_DIR/dmg-mount"

cleanup() {
  hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1 || true
  rm -f "$RW_DMG_PATH"
  rm -rf "$MOUNT_DIR"
}
trap cleanup EXIT

"$ROOT_DIR/scripts/package-app.sh"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
ditto --noextattr --noqtn "$APP_DIR" "$STAGING_DIR/Reco Trainer.app"
xattr -cr "$STAGING_DIR/Reco Trainer.app"
codesign --verify --deep --strict --verbose=2 "$STAGING_DIR/Reco Trainer.app"
ln -s /Applications "$STAGING_DIR/Applications"
rm -f "$DMG_PATH" "$RW_DMG_PATH"
rm -rf "$MOUNT_DIR"
mkdir -p "$MOUNT_DIR"
hdiutil create -volname "Reco Trainer" -srcfolder "$STAGING_DIR" -ov -format UDRW "$RW_DMG_PATH"
hdiutil attach "$RW_DMG_PATH" -nobrowse -mountpoint "$MOUNT_DIR" >/dev/null
xattr -cr "$MOUNT_DIR/Reco Trainer.app"
if [[ "${RECO_SIGN_IDENTITY:--}" == "-" ]]; then
  codesign --force --deep --sign - "$MOUNT_DIR/Reco Trainer.app"
else
  codesign --force --deep --options runtime --timestamp --sign "$RECO_SIGN_IDENTITY" "$MOUNT_DIR/Reco Trainer.app"
fi
codesign --verify --deep --strict --verbose=2 "$MOUNT_DIR/Reco Trainer.app"
hdiutil detach "$MOUNT_DIR" >/dev/null
hdiutil convert "$RW_DMG_PATH" -format UDZO -o "$DMG_PATH"
codesign --force --sign "${RECO_SIGN_IDENTITY:--}" "$DMG_PATH"
echo "$DMG_PATH"
