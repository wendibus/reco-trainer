# Reco Trainer 0.9 — Reviewed Active Learning

Reco Trainer 0.9 adds a privacy-safe active-learning workflow for expanding a local sports dataset with new videos. This remains **work in progress / alpha software** intended for testing and feedback.

## What is new

- Choose **Review new videos** and select a separate local folder.
- Extract up to the selected image count per video, capped at 500 for each expansion run.
- Run the current local model with a recall-oriented 12% confidence threshold.
- Review candidates ordered toward uncertain detections.
- Confirm **Ball** or **No ball**, correct the box, or skip the image.
- Add only deliberately reviewed images to training.
- Keep pending candidates out of training datasets, model packages, statistics, and benchmarks.
- Skip videos the project has already used.
- Preserve source videos unchanged and keep all images and predictions local.
- Explain the workflow in German, English, Spanish, and French.

## Recommended workflow

1. Train an initial model on approximately 1,000 carefully reviewed images.
2. Keep validation and benchmark videos separate and never use them for expansion.
3. Choose **Review new videos** and select different, previously unused recordings.
4. Review Ball / No ball decisions and correct imprecise boxes.
5. Retrain after each batch of approximately 1,000–2,000 confirmed images.
6. Stop adding near-duplicates when independent validation no longer improves.

The full frame remains visible during review so that lamps, heads, logos, and other ball-like objects can be judged in context. No automatic proposal is accepted without a human decision.

## Installation files

- **macOS:** `Reco.Trainer.Mac.0.9.dmg`
- **Windows:** `Reco.Trainer.Windows.0.9.zip`
- **Linux:** `Reco.Trainer.Linux.0.9.zip`
- **Docker:** `Reco.Trainer.Docker.0.9.zip`

## SHA-256

```text
438e26225a9a851508473c480f138b2141dba872bc6e9f1500b0ca4ac46d9160  Reco.Trainer.Mac.0.9.dmg
d814a34ab8c5230c464047b2b7a7e173fdeb6012023f0a6d48617e00cb02ac4d  Reco.Trainer.Windows.0.9.zip
d860ae34d775a62dbe077e24afa55e5a45dbeca4b5c5408a78d51eccdf4100e9  Reco.Trainer.Linux.0.9.zip
e2f7c2a1193da090a5e6ad287199c43742ce5dc8b4263e47c9c24e03d4d74e4a  Reco.Trainer.Docker.0.9.zip
```

Please do not attach private footage, extracted images, or `.reco-training` folders to bug reports.
