# Reco Trainer 0.25.0 — Batch Training, Automatic Epochs and the Model Hub (groundwork)

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Batch training

- **Train locally** now asks which models to train: the current one, or also Nano, Small, Medium and Large. They run one after another, smallest first. If one fails, for example for lack of memory, the next one still runs.
- **Automatic epochs:** training stops when the score has not improved for a set number of epochs (default 10). The learning rate halves when the score stalls.
- After training, the best confidence per model and class is measured on the validation images without training again. A table compares all models, and the confidence can be applied.
- RF-DETR Medium and Large can be trained. The computer stays awake during training.

## Model Hub (groundwork)

- The model library has a **Model Hub** button that lists, downloads and activates community models. It only goes online when you click it.
- **The hub starts empty, and sharing is not switched on yet.** The **Share in hub** button appears once the GitHub login is set up. The GitHub side has not run in production yet, so expect rough edges at first.
- When you continue training from a hub model, the trainer measures before and after on your own validation images and keeps the hub model active if the result is worse.

## Privacy

- **`.recomodel` packages used to contain the checkpoint's training arguments with local paths, including the user name.** New packages hold only weights and class names. Packages you already passed on still contain those paths, so replace them if that matters to you.

## Fixes

- The confidence measurement now uses the Augmentation Studio's validation images when training with synthetic data.
- Studio: the train, valid and test split of the source folder is kept, goalkeepers no longer distort team detection and stay in position, goals keep their label from the box, and feet below boxes that are too short are now removed too.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.25.0.dmg` (signed and notarized)
- **Mac update package:** `Reco.Trainer.Payload.0.25.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.25.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.25.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.25.0.zip`
