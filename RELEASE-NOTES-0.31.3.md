# Reco Trainer 0.31.3 — AMD card in batch training with YOLO

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **Batch training with RF-DETR and YOLO now trains RF-DETR on the AMD card.** Until now a batch that included a YOLO model ran entirely in the normal ML environment, which on Windows with an AMD card means the CPU, even with "Train on the AMD card" switched on. The RF-DETR models of such a batch now use the AMD training environment (ROCm); YOLO models keep training on the CPU, because YOLO is only installed in the normal environment.

Windows with an AMD Radeon RX 7700–7900, RX 9060/9070, Radeon PRO W7700–W7900 or AI PRO R9700 and "Set up AMD training" completed. Nothing changes on other computers.
