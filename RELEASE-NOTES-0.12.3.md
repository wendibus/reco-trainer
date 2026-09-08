# Reco Trainer 0.12.3 — Windows Launcher and Annotation Fixes

Reco Trainer 0.12.3 is a bug-fix release. Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Fixed `Start Reco Trainer Windows.bat` opening a console window that closed again within seconds instead of starting, a regression introduced by 0.12.2's Python-version check. The check now runs inside Reco Trainer itself (shown on its own status line when the local worker starts) instead of inline in the batch script.
- Added "Accept auto" / "Reject auto" buttons to the Mac app's annotation editor. Previously there was no way to mark an automatically suggested box as reviewed on a regular training frame — only deleting and manually redrawing it worked — which permanently blocked "Freeze reviewed answers" (step 1 of the model comparison) after running auto-labeling.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.3.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.3.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.3.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.3.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
