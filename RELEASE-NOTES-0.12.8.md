# Reco Trainer 0.12.8 — Fix Auto-Label Wasting Time on Unknown Classes

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Fixed auto-label wastefully running a full detection pass for categories the currently activated model was never trained on (e.g. a model activated back when the project only had "ball" annotated), which previously ended in a single misleading "0 boxes" message with no indication why. Reco Trainer now checks the activated model's actual manifest classes and skips whatever it provably doesn't know, with a clear per-class explanation instead.
- This also fixes the auto-label chip defaults introduced in 0.12.7, which could preselect categories an active model can't detect at all.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.8.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.8.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.8.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.8.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
