# Reco Trainer 0.31.6 — AMD batch training no longer gets stuck

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **Batch training on the AMD card could stop after a finished model.** It stayed at the end of one model's confidence check, and the next model never started. Processes in the AMD training environment now end right away once their work is saved, instead of waiting for the graphics driver to shut down.
- The batch window now shows "<model>: starting …" as soon as each model begins, so a step that stops early is visible.

Only affects Windows with "Set up AMD training" done. Nothing changes on other computers.
