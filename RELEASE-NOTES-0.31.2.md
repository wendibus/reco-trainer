# Reco Trainer 0.31.2 — Augmentation Studio: venues in exported datasets

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **The Augmentation Studio now finds venues in datasets from "Export dataset".** The export deliberately leaves video names out of the file names, so every image became a session of its own: no empty-hall plate from several images, and no split by venue. The studio now groups such images by image size and venue background (Roboflow file names too).
- If one venue was spread over training, validation and test in the exported dataset, the studio now splits it again as a whole, so validation and test never show a venue that synthetic images were made from.
- With many sessions the studio shows the 12 biggest in full and lists the rest, so the page stays fast.

**If you already analyzed an exported dataset, click "Analyze" once more.** Finished masks are kept, so this is much faster than the first time.
