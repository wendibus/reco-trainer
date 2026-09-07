# Reco Trainer 0.12.2 — Benchmark and Setup Fixes

Reco Trainer 0.12.2 is a bug-fix release. Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Fixed a name collision in the local benchmark's IoU calculation that caused the model comparison to score box overlap incorrectly, affecting quality score, mAP, precision, recall, and F1 for every ranked model.
- Fixed the "Freeze reviewed answers" step being permanently blocked by pending active-learning review candidates, which have nothing to do with the frozen ground truth. This could make the entire model comparison workflow appear non-functional.
- Both benchmark actions now show a clear hint explaining exactly which precondition is still missing, instead of a silently disabled button.
- The Windows and Linux launcher scripts now clearly warn when the detected Python is not 3.11 or 3.12, the version required by the RF-DETR/PyTorch setup step. Video browsing and upload are unaffected either way.
- Removed internal code duplication in checkpoint lookup and frame extraction; both Python worker test suites now run automatically in CI on every push and pull request.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.2.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.2.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
