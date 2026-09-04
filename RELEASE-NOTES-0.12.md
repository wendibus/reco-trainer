# Reco Trainer 0.12 — Safer Editing and Clearer Model Comparison

Reco Trainer 0.12 improves the daily annotation and model-exchange workflow while keeping videos, frames, labels and training runs fully local. This remains work-in-progress alpha software.

## Changes

- Exactly ten annotation changes per image can be undone and redone.
- Keyboard shortcuts are supported: `Cmd/Ctrl+Z` for undo and `Cmd/Ctrl+Shift+Z` for redo.
- Reco Trainer asks for a readable model-package name before creating a `.recomodel` file.
- The chosen name is stored in the package manifest and used to create a portable file name.
- The model benchmark contains an expandable plain-language glossary for quality score, mAP@0.50, precision, recall, F1, FP/FN, inference time and confidence threshold.
- All new interface text is available in German, English, Spanish and French.

## Downloads

- **macOS:** `Reco-Trainer-Mac-0.12.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

No media is uploaded. A `.recomodel` package contains model weights, checksums and aggregate metadata, but no videos, frames, local paths or video file names. Verify labels and benchmark results before real-world use, and keep a backup of the selected `.reco-training` directory.
