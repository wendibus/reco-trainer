# Reco Trainer 0.17.1 — Automatic Python Detection for "Set up ML"

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Fixed (all platforms):** "Set up ML" no longer fails with `No matching distribution found for rfdetr[coreml,onnx,train]>=1.9.0` on computers where only an old Python (such as the 3.9 that ships with macOS) was picked. Reco Trainer now looks for a Python 3.10 or newer itself (3.12 and 3.11 first, including Homebrew and python.org installs), checks its version, and creates the private ML environment with it.
- **Fixed:** an ML environment created earlier with an old Python (the `.runtime/venv` folder in the training folder) is detected and rebuilt automatically - nothing to delete by hand.
- If no suitable Python is installed, Reco Trainer now says so clearly and suggests `brew install python@3.12` or python.org instead of showing the pip error.
- Windows/Linux/Docker: the Python warning at start-up only appears when no Python 3.10+ is available at all.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.17.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.17.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.17.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.17.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
