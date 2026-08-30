# Reco Trainer 0.8 — Per-video Frame Preparation

Reco Trainer 0.8 improves local dataset preparation while keeping the complete privacy model intact. This remains **work in progress / alpha software** intended for testing and feedback.

## What is new

- Choose the number of extracted images per video (4–5,000; default: 240).
- Additional videos now contribute additional images instead of sharing a fixed 240-frame project limit.
- Remove an unsuitable extracted image directly from the training set.
- Never modify or delete the source video when removing a training image.
- Remove stale derived train/validation/test copies automatically.
- Invalidate a frozen benchmark reference when its image set changes, preventing misleading rankings.
- Use the complete workflow in German, English, Spanish and French.

## Installation files

- **macOS:** `Reco.Trainer.Mac.0.8.dmg` — open it, drag Reco Trainer to Applications and start it. This alpha is locally/ad-hoc signed and not Apple-notarized.
- **Windows:** `Reco.Trainer.Windows.0.8.zip` — extract it and run `Start Reco Trainer Windows.bat`.
- **Linux:** `Reco.Trainer.Linux.0.8.zip` — extract it and run `Start Reco Trainer Linux.sh`.
- **Docker:** `Reco.Trainer.Docker.0.8.zip` — extract it and follow `START-HERE.md`.

## Updated workflow

1. Select the sport and local video folder.
2. Set the desired number of images per video.
3. Prepare the videos locally.
4. Remove blurred, obstructed, duplicated or otherwise unsuitable images.
5. Annotate and review every remaining image.
6. Train locally or freeze a separate reviewed set for model benchmarking.

Removing an image affects only local derived training data. The original recording stays unchanged. Existing trained models and run history are retained, but a frozen benchmark reference is cleared because it no longer matches the current image set.

## SHA-256

```text
f951104f388cbe2208633186416dbfc75e44e64954c92bf53355a91da027fdda  Reco.Trainer.Mac.0.8.dmg
e7edbc144311e49088da28a2cdbd38a5cc7f687ad02ac2714c5ccdbe169c5f0e  Reco.Trainer.Windows.0.8.zip
0fd62955426ae602d14a50a6d33e8d7e83341c9deacbeb00f00798e10783d985  Reco.Trainer.Linux.0.8.zip
20a074c70ba0ee2c014f8aedc9e805733fce3d69a5a60c8d9327111052b86bdc  Reco.Trainer.Docker.0.8.zip
```

Please do not attach private footage, extracted frames or `.reco-training` folders to bug reports. Prefer synthetic or redacted examples.
