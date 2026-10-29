# Reco Trainer 0.22.0 — Better Auto-Labeling: Teachers, Second Opinion, Hardware Tiers

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Hardware tiers

- Reco Trainer now picks a **hardware tier** - strong, medium or weak - from the NVIDIA GPU memory or the Apple chip's memory (CPU only = weak) and uses the models that fit. You can override it under **Advanced options** in the training panel.

| Tier | Typical machine | Suggestion model | Second opinion |
|---|---|---|---|
| strong | NVIDIA with 16 GB, Apple with 32 GB or more | RF-DETR Large | Florence-2 Large |
| medium | NVIDIA with 8 GB, Apple with 16 GB | RF-DETR Medium | Florence-2 Base |
| weak | CPU only | RF-DETR Small | Florence-2 Base |

## Auto-labeling

- **Suggestion model (teacher):** while no trained model knows a class, the largest RF-DETR base model your tier allows suggests people and the ball (all Apache-2.0).
- **Second opinion with Florence-2 (MIT):** "Use a second model to flag uncertain frames" now also works without a second trained model - Florence-2 checks people and the ball and flags frames where it disagrees. It never writes boxes itself.
- **Ball interpolation:** a missing ball between two nearby frames of one video (at most 1 second apart) is filled in as a suggestion to review.
- **SAM 2 (Apache-2.0)** can refine boxes as an experimental option; it only takes over small corrections and keeps the original coordinates. OpenCV stays the default - in a first comparison SAM 2 was not measurably better.

## Models and licenses

- **Advanced options** lists every model with its purpose, license and whether it is downloaded. SAM 2 and Florence-2 are downloaded when first needed, pinned to one exact version. All models are Apache-2.0 or MIT.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.22.0.dmg` (signed and notarized)
- **Mac update package:** `Reco.Trainer.Payload.0.22.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.22.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.22.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.22.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. Model downloads come only from the sources listed in `model_registry.json`.
