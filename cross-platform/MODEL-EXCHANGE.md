# Reco Model Exchange

Reco Trainer 0.7 creates and imports `.recomodel` exchange packages. An imported package is validated, copied into the local model library, and activated only when its sport, RF-DETR size, classes, paths, and checksum are valid. The package contains only:

- one RF-DETR checkpoint;
- `manifest.json` with an English package description, sport, classes, aggregate counts, validation results, hashes, framework version, and license notice.

It never contains source videos, extracted frames, source paths, or video file names. Run the local validator before sharing:

```bash
python3 tools/validate_model_package.py /path/to/model.recomodel
```

## Suggested GitHub workflow

1. Create a GitHub Release for the sport and model version.
2. Attach the `.recomodel` file as a release asset. Checkpoints are often too large for normal Git commits.
3. Copy the printed manifest into the release notes.
4. State which sport, classes, RF-DETR version, and license apply.
5. Never attach `.reco-training/`, videos, frames, datasets, or logs containing private paths.
6. Recipients validate the package and compare it against their own private validation videos before adoption.

Models are not combined by averaging weights. Candidate A and candidate B should be tested on the same private reference set; the better model can then be selected or later distilled into a new model.

The **Import model** button opens a native file picker. The selected package never leaves the computer. Installed candidates are stored below `.reco-training/models/library/`; `.reco-training/models/active.json` records which compatible model is used for the next automatic labeling or fine-tuning run.

Only import packages from a trusted publisher. The checksum detects accidental changes but does not prove who created the checkpoint. RF-DETR/PyTorch checkpoints should be treated like executable software when they are loaded for inference or training.

## GitHub CLI example

Uploading is deliberately a separate, explicit user action:

```bash
gh release create basketball-nano-2026-08 model.recomodel --title "Basketball Nano" --notes-file release-notes.md
```

Reco Trainer never runs this command automatically.
