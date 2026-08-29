#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h}/.."
DIST_DIR="${RECO_DIST_DIR:-$ROOT_DIR/dist}"
APP_DIR="$DIST_DIR/Reco Trainer.app"
BUILD_TRIPLE="arm64-apple-macosx"
BUILD_DIR="${RECO_BUILD_DIR:-/tmp/reco-trainer-mac-release-build}"

cd "$ROOT_DIR"
swift build -c release --disable-sandbox --scratch-path "$BUILD_DIR"

mkdir -p "$DIST_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
cp "$BUILD_DIR/$BUILD_TRIPLE/release/RecoTrainerMac" "$APP_DIR/Contents/MacOS/RecoTrainerMac"
cp "$ROOT_DIR/Packaging/Info.plist" "$APP_DIR/Contents/Info.plist"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$BUILD_DIR/$BUILD_TRIPLE/release/RecoTrainerMac_RecoTrainerMac.bundle/" \
  "$APP_DIR/Contents/Resources/RecoTrainerMac_RecoTrainerMac.bundle/"
xattr -cr "$APP_DIR"
SIGN_IDENTITY="${RECO_SIGN_IDENTITY:--}"
if [[ "$SIGN_IDENTITY" == "-" ]]; then
  codesign --force --deep --sign - "$APP_DIR"
  echo "Hinweis: lokal/ad-hoc signiert; für eine notarierte Veröffentlichung RECO_SIGN_IDENTITY setzen."
else
  codesign --force --deep --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP_DIR"
fi
codesign --verify --deep --strict --verbose=2 "$APP_DIR"

echo "$APP_DIR"
