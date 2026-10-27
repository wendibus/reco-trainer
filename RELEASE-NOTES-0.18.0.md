# Reco Trainer 0.18.0 — Import External ONNX Detectors

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Added (all platforms):** external **ONNX detectors** (YOLO exports with NMS built in) can now be imported as `.recomodel` packages with **Import model**. They are used for automatic labeling and for the model comparison (benchmark). They cannot be trained or fine-tuned, and they cannot be combined into a combined model.
- **Added:** `tools/make_onnx_package.py` builds such a package from an `.onnx` file (name, sport, classes in class-id order, input size). `tools/validate_model_package.py` validates it.
- An imported ONNX model works with either RF-DETR size selected. Check the license of the original weights before sharing a package (Ultralytics YOLO weights are AGPL-3.0).

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.18.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.18.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.18.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.18.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
