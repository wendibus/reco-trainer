# Reco Trainer 0.14.3 — Mandatory Model Code-Safety Check

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Hardened:** the code-safety check introduced in 0.14.2 (PyTorch's `weights_only=True` safe loading mode, which rejects a weights file containing anything beyond plain model data) now also runs unconditionally immediately before every actual use of a weights file - training, auto-labeling, the model comparison, and export - not only at import. Previously, if the local ML environment wasn't set up yet at import time, that check was skipped and a weights file could end up in the model library unscanned. Since every one of these real-use call sites already requires the ML environment to exist, there is no longer any point at which an untrusted weights file can reach PyTorch's own unsafe loader unchecked.
- **Added:** zip entries marked as symlinks are now rejected outright when validating a model package, closing off that path even though it was not exploitable with the current extraction method.
- Audited every subprocess/eval/exec call in the local ML worker: all use array-form arguments with no shell interpolation, and none are ever constructed from package-supplied data.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.14.3.dmg`
- **Windows:** `Reco.Trainer.Windows.0.14.3.zip`
- **Linux:** `Reco.Trainer.Linux.0.14.3.zip`
- **Docker:** `Reco.Trainer.Docker.0.14.3.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
