# Reco Trainer 0.19.2 — Visible Import Buttons and Direct ONNX Import

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Fixed (Mac):** all buttons below the image were in one row, so at normal window widths the last ones - including **Import model** and **Import labels** - were cut off and invisible. They are now in two rows (work · models and data) and wrap in a narrow window.
- **Added (Mac app and web UI):** YOLO models in **ONNX** format (exported with NMS built in, output 1 × N × 6) can be imported directly - on the Mac with **Import model** (the picker now also lists `.onnx` files), in the web UI with **Import ONNX model**. A form asks for the name, the classes in class-id order, the input size and the license; Reco Trainer builds the package (kept in `inbox/`), installs and activates it. Common class names such as `person` are mapped. No ML environment is needed. The model is used for auto-labeling and the model comparison; it cannot be trained.
- **Added:** worker command `ml_worker.py install-onnx`.
- The "What's new" window on the Mac now shows the 0.19.x changes.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.19.2.dmg` (signed and notarized)
- **Windows:** `Reco.Trainer.Windows.0.19.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.19.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.19.2.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
