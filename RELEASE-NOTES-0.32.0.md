# Reco Trainer 0.32.0 — export for the Reco app

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Export for Reco.** A new button next to "Train locally" saves a library model as an ONNX file the Reco camera app loads directly ("AI model (ONNX)" field or `--model`). It works for RF-DETR models, YOLO26 models trained in Reco Trainer, and combined models: all members run inside the one file.
  - Only **Ball** is preselected. Tick **Player** if Reco should follow players too; they are named `person` in the file, the way Reco looks for them. With the ball only, use Reco's tracking mode `ball`.
  - The file has the layout Reco reads: one square input with grey borders, one output `[1, 300, 6]` (box, confidence, class), class names in the `names` metadata.
  - After writing, a test run checks the file the way Reco runs it on up to eight labelled project images and reports on how many the labelled objects were found.
  - The file goes into the project under `exports/reco`.

## Why

The "CPU model (ONNX)" export writes RF-DETR's own layout with two outputs and no class names. Reco reads a file with more than one output as a different kind of detector, so it found nothing. That export also ignored combined models and took a single checkpoint instead. It stays for your own programs; its hover hint now points to "Export for Reco".

## Notes

- Checked against real RF-DETR models (single and combined) and the Reco source (v0.5.4): boxes match the model's own predictions; confidences differ slightly because Reco adds grey borders where training squeezed the picture. Not yet tested inside the Reco app itself.
