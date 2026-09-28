# Reco Trainer 0.15.1 — Onboarding Fixes

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Fixed:** the video walkthrough introduced in 0.15.0 could show "Video not found" in the packaged macOS app. The resource lookup checked the wrong path inside the app bundle. Fixed to match the same lookup already used for the app's ML worker script.
- **Fixed:** the "What's new in Reco Trainer" dialog shown after an update listed the entire change history from many past versions instead of only the changes in the version just installed.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.15.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.15.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.15.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.15.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
