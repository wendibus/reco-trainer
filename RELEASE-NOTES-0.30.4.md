# Reco Trainer 0.30.4 — Augmentation Studio hint

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **Select source folder** in the Augmentation Studio opened no dialog when no working folder was prepared or ML was not set up; only a small error line appeared at the top. The button is now disabled in that case, and the text next to it says what to do first. The studio also reloads when another working folder is opened or ML is set up.
