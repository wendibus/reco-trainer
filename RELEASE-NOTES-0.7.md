# Reco Trainer 0.7 — Local Model Benchmark Alpha

Reco Trainer 0.7 adds a fully local test platform for comparing downloaded detection models against the same reviewed image set. This remains a **work in progress** intended for testing and feedback.

## What is new

- Freeze the current reviewed annotations as a local ground-truth test set.
- Run every compatible imported model against exactly the same images.
- Automatically rank models using a quality score based on 70% mAP@0.50 and 30% F1.
- Compare precision, recall, false positives, false negatives, mean IoU and inference time per image.
- Keep model predictions and reports locally under `.reco-training/benchmarks/`.
- Use explicit negative images: a reviewed frame without a box means that no target object is present.
- Prevent accidental evaluation against unreviewed automatic suggestions.
- Use the benchmark workflow in German, English, Spanish and French.
- Updated the local web/build toolchain; `npm audit` reports zero known vulnerabilities at release time.

The benchmark runs models one after another. This makes comparisons more reproducible and avoids multiple models competing for the same GPU, Apple Neural Engine or system memory.

## Privacy

Images and videos are processed on the user's own computer. Reco Trainer does not upload media to GitHub or an external training service. Exported model packages and benchmark reports contain model data and numerical results, not the source images or videos.

## Installation files

- **macOS:** `Reco.Trainer.Mac.0.7.dmg` — open the disk image, drag Reco Trainer to Applications and start it. This test build is locally/ad-hoc signed and is not Apple-notarized.
- **Windows:** `Reco.Trainer.Windows.0.7.zip` — extract the ZIP and run `Start Reco Trainer Windows.bat`.
- **Linux:** `Reco.Trainer.Linux.0.7.zip` — extract the ZIP and run `Start Reco Trainer Linux.sh`.
- **Docker:** `Reco.Trainer.Docker.0.7.zip` — extract it and follow `START-HERE.md`; Docker Desktop or a compatible Docker installation is required.

## Benchmark workflow

1. Select or create a project and prepare frames locally.
2. Correct the boxes and remove wrong automatic labels.
3. Import the model packages you want to compare.
4. Open **Test models**.
5. Freeze the reviewed answers.
6. Choose the confidence threshold and start the benchmark.
7. Review the ranking and the detailed error counts.

Do not use the benchmark images for training the compared models. A separate, previously unseen test set gives a meaningful ranking; otherwise the results can be overly optimistic.

## SHA-256

```text
0dee008b71809c40436a9ddeb4d3304731746a6b90d45d3a5605da050aa2cf4f  Reco.Trainer.Mac.0.7.dmg
f9ec0aa76343f5b63c2bda7ae45071f1f2757f2f2b35cc2a0568bf0231fe7955  Reco.Trainer.Windows.0.7.zip
024148c341e10ca74c5e1a3b33a5c2c594acd994c27f8c48ed13212a01cfe1b0  Reco.Trainer.Linux.0.7.zip
6f0986c57c67e672058cf7d71059adf6fb0d0e66e237ec145a51522e5653ff7f  Reco.Trainer.Docker.0.7.zip
```

Please test this alpha release with representative, privately held footage and report reproducible issues without attaching sensitive media.
