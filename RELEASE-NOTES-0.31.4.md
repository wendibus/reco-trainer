# Reco Trainer 0.31.4 — YOLO on the AMD card

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **YOLO models train on the AMD card.** With "Set up AMD training" done and "Train on the AMD card" switched on, a batch training with YOLO now runs completely on the card, RF-DETR and YOLO alike. The first such batch also installs YOLO (Ultralytics, same version as before) into the AMD environment, a few hundred MB. AMD's PyTorch stays as it is.
- If YOLO cannot be set up there, nothing is lost: YOLO trains on the CPU as in 0.31.3, RF-DETR still on the card, and the log says so.

Windows with an AMD Radeon RX 7700–7900, RX 9060/9070, Radeon PRO W7700–W7900 or AI PRO R9700. Nothing changes on other computers.
