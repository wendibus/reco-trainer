# Reco Trainer 0.12.1 — More Reliable Local Training

Reco Trainer 0.12.1 improves iterative RF-DETR training while keeping videos, frames, labels, checkpoints, and metrics entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Interrupted runs resume from the full Lightning checkpoint, including optimizer, learning-rate scheduler, EMA, epoch, and callback state.
- A new fine-tuning cycle starts from the best completed model with a reduced learning rate and cosine decay to retain previously learned examples.
- Early stopping is more patient, reducing premature termination when a small validation set produces noisy metrics.
- Existing metric and configuration logs are archived before RF-DETR refreshes the active run directory.
- Reco Trainer warns when validation or test results are based on fewer than 100 annotations and recommends 300 per split for stronger comparisons.
- TensorBoard is disabled by default when it is not used, removing an unnecessary dependency warning.
- The new status messages and the in-app “What’s New” panel are available in German, English, Spanish, and French.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
