# Reco Trainer 0.12.10 — Field Boundaries for Auto-Label

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- New "Set field boundaries" button: enter the field's real width/length and click its four corners (top left, top right, bottom right, bottom left) on a reference frame. Afterwards, "Auto-label" for player/referee/goalkeeper only considers people whose feet are actually standing on the marked field — spectators, bench, and staff nearby are ignored instead of being picked up as players.
- A dismissible banner suggests setting this up as soon as a project has frames. It's entirely optional: existing projects without a marked field keep working exactly as before.
- Implemented with a perspective transform (OpenCV) from the four marked corners to real-world meters; each detected person's foot point (bottom-center of the box) is tested against the field rectangle with a small margin for boundary imprecision.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.10.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.10.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.10.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.10.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
