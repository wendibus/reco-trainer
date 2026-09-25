# Reco Trainer 0.14.2 — Model Package Code-Safety Check

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Added:** model packages (`.recomodel`) are now checked, on import and (for defense in depth) also when creating or combining one, with PyTorch's own safe loading mode (`weights_only=True`). A checksum alone only proves a weights file matches its own manifest - it does not prove the file is safe, since whoever builds a package also controls its checksum. The new check rejects any weights file that contains more than plain model tensors, such as executable code that would run automatically when the file is loaded.
- This is not a general-purpose malware scanner; it is PyTorch's own documented, maintained mitigation for the specific, real risk in this file format (arbitrary code execution via a crafted pickle payload disguised as model weights).
- If the local ML environment is not set up yet, the checksum is still verified exactly as before; only this additional check is skipped until it is.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.14.2.dmg`
- **Windows:** `Reco.Trainer.Windows.0.14.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.14.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.14.2.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
