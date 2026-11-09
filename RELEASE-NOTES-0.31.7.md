# Reco Trainer 0.31.7 — batch training continues after the confidence check

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **Batch training no longer stops after the first model's confidence check** (seen on an AMD card on Windows). The batch now reads each step's output without waiting forever: a step that has delivered its result but does not end is stopped after a minute, and a confidence check that reports nothing for ten minutes is skipped and noted in the batch report. The trained model is already saved at that point.
- The confidence check now writes "Best confidence: … (F1 …)" to the log before it finishes.
