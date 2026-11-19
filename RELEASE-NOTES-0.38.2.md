# Reco Trainer 0.38.2 — fix YOLO base model download on Windows, clearer batch report

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **YOLO base model download failed with `CERTIFICATE_VERIFY_FAILED`** ("unable to get issuer certificate") on some Windows computers, typically with antivirus HTTPS scanning or a proxy, so no model of a batch training finished. Downloads now retry with the Windows certificate store and `certifi`. Certificate checks stay on, and the SHA-256 check of every file is unchanged.

## Changed

- **Batch report columns.** "Best confidence" is now **Best threshold** and "Per class" is **Threshold per class**. A hint on both headings explains that this is the minimum confidence at which the model works best, not an accuracy: "ball 70 %" means keeping only detections of 70 % confidence or more. How good the model is shows in **mAP50**.

## Notes

- Tested with the type check and the existing tests plus new tests for the download fallback; the interface was not opened in a browser. The Windows certificate store path was not run on a Windows computer.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.38.2 through the in-app updater. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.38.2.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.38.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.38.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.38.2.zip`

`SHA256SUMS-0.38.2.txt` contains checksums for all attached update and platform packages.
