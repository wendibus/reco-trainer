# Reco Trainer 0.30.0 — New look (light and dark) and optional YOLO training

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **A Mac-like look, light and dark:** system font at readable sizes, translucent toolbar and sidebar, macOS push buttons and segmented controls, calm cards, dialogs as sheets, hints and the job indicator as glass overlays. The frame sits on a dark graphite stage with a glass bar for navigation and zoom that stays on one line in every language. Ball orange marks the app icon, review candidates and the ball. Dark mode follows the system setting. Layout, buttons and workflows are unchanged.
- **Optional YOLO training:** **Train locally** now offers **YOLO26 Nano, Small and Medium** (Ultralytics) next to RF-DETR, also both families in one comparison run. The trained model is exported to ONNX with NMS built in and filed in the model library: it auto-labels, runs in **Test models** and in the ball-tracking simulation, and the next YOLO run of the same size continues from it. Choose the image size (640, 960 or 1280 px): larger finds small balls better but takes longer. Batch training and the confidence sweep handle YOLO.

## License note for YOLO

Ultralytics and its YOLO26 weights are licensed under **AGPL-3.0**, and so are models trained with them. Reco Trainer installs Ultralytics (pinned to 8.4.180) only after you tick the license box in the training dialog; the base weights are pinned by SHA-256 and are not part of the Apache-2.0/MIT model registry. Ultralytics' usage statistics and automatic package installs are switched off. RF-DETR (Apache-2.0) stays the default; Core ML export, combined models and `.recomodel` sharing stay RF-DETR only.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.30.0 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.30.0 update. This release updates the interface and workers; the launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.30.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.30.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.30.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.30.0.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
