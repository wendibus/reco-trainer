# Reco Trainer 0.38.5 — find the field automatically

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Find automatically.** In the field dialog, a button proposes the outline of the playing field (at most 10 points). It looks only at the floor surface: no lines, no sport, no lens model. Drag points to adjust, then save as before.
- **Hide areas.** Drag rectangles over anything that gets in the way (nets, neighbouring court, spectators, bench); the search ignores them. Outside the outline the preview is dimmed.

## Notes

- Tested with the type check and the Python tests, including new ones for the detection. On one indoor-hall frame the outline was correct with two hidden areas; without them it reached into the bottom corners. Floors that look like the surroundings may need hidden areas or hand-placed points. The interface was not opened in a browser or on Windows.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.38.5 through the in-app updater. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.38.5.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.38.5.zip`
- **Linux:** `Reco.Trainer.Linux.0.38.5.zip`
- **Docker:** `Reco.Trainer.Docker.0.38.5.zip`

`SHA256SUMS-0.38.5.txt` contains checksums for all attached update and platform packages.
