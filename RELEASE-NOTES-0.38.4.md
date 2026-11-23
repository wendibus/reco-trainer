# Reco Trainer 0.38.4 — bring images back into training, larger image list

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Use all candidates with boxes.** With the **Not in training** chip active, a button puts every review candidate that already has boxes into training at once, after a confirmation. Their boxes count as confirmed afterwards. The independent test and images without boxes are left alone.
- **Use for training** now also sits next to **Image is empty**, directly above the image, so it is no longer hidden below the visible area.

## Changed

- **Image list.** The sidebar always scrolls, the image list keeps at least 240 px of height, and the thumbnails are larger (72×45 instead of 54×34).
- Search, box count, class and sort filters are folded into **Search and sort**, which shows a dot while a filter is active.

## Notes

- Tested with the type check and the Python tests, including a new one for the bulk action; the interface was not opened in a browser or on Windows. The video strip above the editor was not changed.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.38.4 through the in-app updater. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.38.4.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.38.4.zip`
- **Linux:** `Reco.Trainer.Linux.0.38.4.zip`
- **Docker:** `Reco.Trainer.Docker.0.38.4.zip`

`SHA256SUMS-0.38.4.txt` contains checksums for all attached update and platform packages.
