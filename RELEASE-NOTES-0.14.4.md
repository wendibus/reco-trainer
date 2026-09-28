# Reco Trainer 0.14.4 — Epochs Field & Translation Fixes (Windows/Linux/Docker)

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Added:** the Windows/Linux/Docker web interface had no field at all to set the number of training epochs (always fixed at 20). It now has the same field as the Mac app.
- **Changed:** the default number of epochs is now 100 on both platforms (up from 20). Early stopping already ends a run once it plateaus (patience scales with the requested epoch count, capped at 20 epochs with no improvement), so a higher ceiling only gives a run room to keep improving instead of being cut off arbitrarily at epoch 20.
- **Fixed:** many status and error messages in the Windows/Linux/Docker interface were hardwired to German regardless of the selected UI language - among them the hardware/accelerator display, a model's test/validation score labels, an active model's package description, and practically every status line shown during an operation (preparing videos, auto-labeling, training, reviewing candidates, importing/exporting models, and more). All of these are now correctly localized in German, English, Spanish, and French.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.14.4.dmg`
- **Windows:** `Reco.Trainer.Windows.0.14.4.zip`
- **Linux:** `Reco.Trainer.Linux.0.14.4.zip`
- **Docker:** `Reco.Trainer.Docker.0.14.4.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
