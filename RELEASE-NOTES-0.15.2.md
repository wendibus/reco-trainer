# Reco Trainer 0.15.2 — Video Walkthrough Crash Fix

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Fixed:** the video walkthrough could crash the app as soon as the video started (SIGABRT inside Apple's private `_AVKit_SwiftUI` framework, a known issue with SwiftUI's `VideoPlayer` on newer macOS versions). Playback now goes through AVKit's standalone `AVPlayerView` instead of SwiftUI's `VideoPlayer`, avoiding the affected framework entirely.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.15.2.dmg`
- **Windows:** `Reco.Trainer.Windows.0.15.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.15.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.15.2.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
