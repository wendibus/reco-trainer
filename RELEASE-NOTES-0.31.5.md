# Reco Trainer 0.31.5 — cleaner AMD training log

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **No more "Unknown command line argument … offload-arch.exe" lines when training on the AMD card.** They came from a hardware query in AMD's ROCm package that fails when a folder name contains spaces (as "Reco Trainer" always does). They were harmless; the query is now skipped, and training uses the same settings as before.

Only affects Windows with "Set up AMD training" done. Nothing changes on other computers.
