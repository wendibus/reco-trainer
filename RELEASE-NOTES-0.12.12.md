# Reco Trainer 0.12.12 — Multi-Select Deletion, Category-Restricted Training

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- The "Model" dropdown in "4 · Improve model" was truncating its shown value in a too-narrow fixed width. Fixed.
- The training-image list now supports Cmd-click and Shift-click for multi-select, with its own "Remove" button for the selection, in addition to the existing single-image trash button.
- The confirmation shown when removing a training image gained a "Don't ask again" checkbox.
- New class chips in "4 · Improve model" let you exclude individual categories from a training run (e.g. train only "ball" and "referee" this time). This does not freeze or otherwise guarantee an excluded category's existing detection quality — it simply isn't part of that run's dataset.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.12.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.12.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.12.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.12.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
