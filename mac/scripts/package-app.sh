#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h}/.."
DIST_DIR="${RECO_DIST_DIR:-$ROOT_DIR/dist}"
APP_DIR="$DIST_DIR/Reco Trainer.app"
BUILD_DIR="${RECO_BUILD_DIR:-/tmp/reco-trainer-mac-release-build}"

cd "$ROOT_DIR"
swift build -c release --disable-sandbox --scratch-path "$BUILD_DIR"
# Ask the toolchain for its own bin path rather than hardcoding a layout:
# newer Swift toolchains build via XCBuild ($BUILD_DIR/out/Products/Release)
# instead of the classic SwiftPM layout ($BUILD_DIR/<triple>/release).
BIN_DIR="$(swift build -c release --disable-sandbox --scratch-path "$BUILD_DIR" --show-bin-path)"

mkdir -p "$DIST_DIR"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
cp "$BIN_DIR/RecoTrainerMac" "$APP_DIR/Contents/MacOS/RecoTrainerMac"
cp "$ROOT_DIR/Packaging/Info.plist" "$APP_DIR/Contents/Info.plist"
ASSET_BUILD_DIR="$BUILD_DIR/asset-catalog"
mkdir -p "$ASSET_BUILD_DIR"
xcrun actool "$ROOT_DIR/Assets.xcassets" \
  --compile "$ASSET_BUILD_DIR" \
  --platform macosx \
  --minimum-deployment-target 14.0 \
  --app-icon AppIcon \
  --output-partial-info-plist "$ASSET_BUILD_DIR/asset-info.plist"
cp "$ASSET_BUILD_DIR/Assets.car" "$APP_DIR/Contents/Resources/Assets.car"
cp "$ASSET_BUILD_DIR/AppIcon.icns" "$APP_DIR/Contents/Resources/AppIcon.icns"
# The payload (web interface + Python workers) ships inside the app and is copied to
# ~/Library/Application Support/Reco Trainer/payloads on first start; later versions
# arrive through the in-app updater without a new app download.
PAYLOAD_BUILD_DIR="$BUILD_DIR/payload-build"
rm -rf "$PAYLOAD_BUILD_DIR"
"$ROOT_DIR/../cross-platform/scripts/build-payload.sh" "$PAYLOAD_BUILD_DIR" >/dev/null
rm -rf "$APP_DIR/Contents/Resources/payload"
cp -R "$PAYLOAD_BUILD_DIR/payload" "$APP_DIR/Contents/Resources/payload"
# Precompiled Apple frame extractor (AVFoundation), so no Xcode tools are needed on the user's Mac.
# Without -target swiftc builds for the macOS version of the build machine; match the app (macOS 14, Apple Silicon).
swiftc -O -target arm64-apple-macos14.0 "$ROOT_DIR/../cross-platform/frame_extractor.swift" -o "$APP_DIR/Contents/MacOS/reco-frame-extractor"
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
