# Reco Trainer 0.30.6 — AMD training test fix

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **Test AMD Training.bat** (Windows) stopped after training with a `UnicodeEncodeError`: RF-DETR prints its metric tables with box-drawing characters that the Windows console code page cannot encode. Training itself already worked on the graphics card. The test now runs with UTF-8 output and writes its full report. Reco Trainer itself was not affected.
