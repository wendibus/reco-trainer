#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h}/.."
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT_DIR/Packaging/Info.plist")"
DMG_PATH="$ROOT_DIR/dist/Reco-Trainer-Mac-$VERSION.dmg"

if [[ -z "${RECO_SIGN_IDENTITY:-}" || "$RECO_SIGN_IDENTITY" == "-" ]]; then
  echo "RECO_SIGN_IDENTITY muss eine gültige 'Developer ID Application'-Identität enthalten." >&2
  exit 2
fi
if [[ -z "${RECO_NOTARY_PROFILE:-}" ]]; then
  echo "RECO_NOTARY_PROFILE muss den Namen eines mit notarytool gespeicherten Schlüsselbundprofils enthalten." >&2
  exit 2
fi

# package-dmg.sh must write to the default dist folder used for DMG_PATH.
unset RECO_DIST_DIR
"$ROOT_DIR/scripts/package-dmg.sh"
xcrun notarytool submit "$DMG_PATH" --keychain-profile "$RECO_NOTARY_PROFILE" --wait
xcrun stapler staple "$DMG_PATH"
spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG_PATH"
echo "$DMG_PATH"
