# Reco Trainer 0.12.9 — Update Check, OpenCV on Manual Boxes, Player Fallback

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Reco Trainer now checks once at launch whether a newer version has been published on GitHub, showing a dismissible banner with a download link if so. Only GitHub's public releases API is queried — no project data (videos, frames, annotations, models) is ever part of this request.
- "Refine boxes with OpenCV" (previously "Review ball boxes with OpenCV") now also tightens manually drawn boxes on request, not just unreviewed automatic suggestions. Only plausible, closely-matching adjustments are applied; each box's original coordinates remain stored.
- "Player" stays selectable for auto-label even when the active custom model was only ever trained on other classes (e.g. just "ball"): Reco Trainer now runs a second pass with the generic base model for "player" specifically, since unlike other sport-specific classes it has a COCO equivalent that needs no custom training.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.9.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.9.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.9.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.9.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. The new update check contacts only GitHub's public releases API. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
