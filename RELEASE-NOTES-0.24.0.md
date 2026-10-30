# Reco Trainer 0.24.0 — Working Folder, Hover Hints and Augmentation Studio (experimental)

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Working folder

- The video folder is now the **working folder**. Videos are handled as before, images become frames you can label, and labels lying next to them (YOLO or COCO) come along.
- An empty folder gives a hint to add videos, images or datasets.

## Hover hints

- Moving the mouse over a button shows a short hint in the selected language (DE, EN, ES, FR). The **Hints** button next to the systems turns them off.

## Augmentation Studio (experimental)

- **✦ Augmentation Studio** generates a labelled synthetic training set from images with COCO annotations: SAM 2 masks, an empty-hall plate, people placed with the size and lean of the new spot, labels from the visible mask.
- Fixed rules with an automatic validator: at most two teams, 5 players per team, 3 referees, 1 ball, nobody twice. Validation and test stay real and are split by session. The same seed gives bit-identical images.
- **Not yet proven:** it has not been shown that synthetic images improve a model. Generated images can still contain body remnants and people without a label, so look at the preview gallery. With a moving camera, people stay close to where they stood.

## Fixes

- Imported images are no longer split by file order. Next to real videos they are used for training only, so validation and test come from the videos.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.24.0.dmg` (signed and notarized)
- **Mac update package:** `Reco.Trainer.Payload.0.24.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.24.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.24.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.24.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training and the studio remain local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. Model and runtime downloads come only from pinned sources.
