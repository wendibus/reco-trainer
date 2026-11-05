# Reco Trainer 0.27.0 — Review consistency results image by image, compare two models in the tracking simulation

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Step-by-step review of consistency-check results:** above the image list, **Review N flagged images** walks through every image the consistency check flagged (**Show flagged only** narrows the list; the check's result window has **Review one by one**). Below the image you see the reason, the VLM's comment and a concrete suggestion. In the image, **red dashed** boxes are measurable duplicates (two boxes of the same class overlapping by at least 60 %) and **green dashed** boxes are people or balls the second detector found without a box. For each image choose **Accept suggestion**, **Fix by hand**, **Reject, keep boxes** or **Delete image**; every decision moves to the next flagged image. When the VLM reports duplicates that cannot be measured, the panel says it is probably a false alarm.
- **Suggestions from the second detector:** consistency checks now store which boxes the second detector would add. For checks run with an earlier version, **Compute suggestions** runs only the second detector over the flagged images, without changing any flag.
- **Compare two models in the ball-tracking simulation:** **Compare two models** shows two players, A and B, each with its own video and its own model (the active model or any library model with a ball class, including imported ONNX YOLO models). Both play on one timeline, each side shows how often the ball was tracked, detected, interpolated, held or lost and the average confidence. **Same video as A** runs both models on identical frames and counts where both, only A, or only B found the ball. Choosing a model here does not change the active model.

## Fixes

- The ball-tracking simulation always used the "nano" model size, even when another model size was selected.

## Good to know

- **Compute suggestions** looks at every flagged image with the second detector; on a few thousand images this takes a while and shares the GPU with any training that is running.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.27.0 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.27.0 update. This release updates the interface and workers; the launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.27.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.27.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.27.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.27.0.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
