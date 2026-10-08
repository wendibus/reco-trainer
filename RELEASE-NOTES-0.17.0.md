# Reco Trainer 0.17.0 — Field Boundaries With Any Number of Points

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Added (Mac app and Windows/Linux/Docker interface):** the field-boundaries editor is no longer limited to four corners.
  - Keep clicking to add more points around the edge of the field, up to 32 - for curved courts or L-shaped pitches.
  - Drag any point to move it.
  - **Remove last point** deletes the most recently added point (**Reset** still clears all).
  - With exactly four points nothing changes: they are the corners of a rectangular field and still need its real width and length. With more points the real size is optional, and auto-label checks whether a person's feet are inside the outline in the image.
- Outlines saved with more than four points are read by this version and newer; older versions ignore them (no filtering).

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.17.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.17.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.17.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.17.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
