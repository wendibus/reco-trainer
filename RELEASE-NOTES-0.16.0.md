# Reco Trainer 0.16.0 — Same Features on Every Platform

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Added (Windows/Linux/Docker interface):** the features that previously existed only in the Mac app.
  - **Refine boxes with OpenCV** - locally tightens ball, player, referee and other supported boxes, including hand-drawn ones; only plausible, closely matching adjustments are applied.
  - **Set / edit field boundaries** - type the field's real size and click its four corners on a reference frame, so auto-label only considers people standing on the field. A one-time hint suggests it as soon as frames are ready.
  - **CPU model (ONNX)** export. The Core ML export is now labeled "Apple model (Core ML)" and is only available when the worker runs on a Mac.
  - **Show training folder** - opens the project's training folder in the system file manager (not available inside Docker).
- **Changed:** the default number of images per video is now **60** (was 240) in both the Mac app and the web interface. It can still be set anywhere from 4 up to 5000; existing projects keep the value they were created with. The web input now steps in 20s, like the Mac stepper.
- **Fixed:** buttons in the web dialogs that use the primary style (for example "Get started" in the video walkthrough and "Save" in the new field editor) were invisible because their colour was only defined inside the main window.
- **Improved:** the training action buttons in the web interface now wrap into their own row, and the remaining Mac training-panel button labels are available in Spanish and French (they previously fell back to English).

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.16.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.16.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.16.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.16.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
