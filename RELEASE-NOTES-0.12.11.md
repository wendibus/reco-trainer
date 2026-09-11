# Reco Trainer 0.12.11 — Detect Referees by Typical Clothing

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- The "referee" chip for auto-label now works for every supported sport, even without a custom-trained model. People wearing attire typical for that sport's referees are automatically suggested as "referee":
  - Solid black: football, futsal, handball
  - Grey top + black pants: basketball
  - A small set of solid colors (black, yellow, orange, red): rugby
  - Black-and-white stripes: ice hockey, American football, lacrosse
- This is a best-effort clothing heuristic, not a guarantee — team colors, lighting, and camera quality can all produce a false match. A matching detection still carries the same automatic-suggestion status as any other auto-labeled box, so it goes through the normal review workflow before it counts as confirmed. Requires "player" to also be selected, since the heuristic reclassifies generic person detections rather than running its own detection pass.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.11.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.11.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.11.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.11.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
