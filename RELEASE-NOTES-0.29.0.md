# Reco Trainer 0.29.0 — Annotating small balls: larger viewer, shortcuts, zoom to the mouse up to 20×, field boundary per video

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Larger frame viewer on laptops:** the viewer now takes almost the full window height (1366×768: 537 px instead of 204 px high; 1440×900: 669 px instead of 277 px). Sidebar and header no longer squeeze it; the training section sits below and the page scrolls to it. Portrait monitors get a matching layout.
- **Keyboard shortcuts:** hold **Space** to pan, release it and the previous tool (for example **New box**) is active again. **B** = New box, **1**–**9** = class 1–9 in the order of the class list, **←/→** switch frames as before. No shortcut fires while a text field, number field, list or dialog has the focus. The shortcuts appear on the tool buttons, in the class list ("Ball (1)"), in the hover hints and below the frame.
- **Zoom to the mouse:** the mouse wheel zooms to the point under the mouse instead of the centre; that image point stays where it is. This works in every tool, also while **New box** is active, and the wheel no longer scrolls the page.
- **Up to 20× zoom:** the maximum rises from 5× to 20×, in even, multiplicative steps (three clicks double the zoom). From about 3× the image shows its real pixels.
- **Thin lines at every zoom:** box outlines (1.5 px, selected 2.5 px), handles and labels keep the same size on screen. Frame and boxes now live in one SVG whose viewBox is the visible part of the image. Stored box coordinates do not change; automated tests check that the same box gets the same coordinates at every zoom.
- **Field boundary per video / camera position:** the field editor names the video it edits and can **use the outline for all videos**. Existing projects copy their single outline to all their videos the first time they are opened (after a backup in `backups/`; the old value stays in `project.json` as `fieldGeometryBeforePerVideo`). Auto-label checks every frame against the outline of its own video. Tested on a copy of a real project (2,493 frames, 41 videos): frames unchanged, all 49,860 sampled filter decisions identical before and after.
- **"Test models" explains itself:** when the comparison cannot run yet, a box lists what is missing (no images, unreviewed auto suggestions, no confirmed test set, no trained model for the sport, another job running) and what to do next. Disabled buttons show the reason as a tooltip.
- **AMD and Intel graphics on Windows (experimental), now with a diagnosis report:** without an NVIDIA GPU, Reco Trainer asks whether to try the graphics card; until you say yes, everything runs on the CPU. After a yes, a quick test installs `onnxruntime-directml` and checks the card. If the card fails (a fallback to the CPU or a crashed job), a report opens with system, driver, DirectX, packages, a minimal DirectML test, the RF-DETR model on card vs. CPU and Vulkan, ready to copy or save.
- **What's new** opens once after an update; click the version number in the header to see it again.

## Fixed

- "The local ML worker exited with code 1" now names the last error the worker printed (for example a missing DLL).

## For AMD/Intel testers

1. Start Reco Trainer 0.29.0 on the Windows PC and open a working folder. Answer **Yes** when asked whether to try the AMD/Intel graphics card (or click **Test graphics card (AMD/Intel)** next to **Set up ML**).
2. Wait for the test. If a report opens, click **Copy report** or **Save as file** and send it back. If no report opens, run **auto-label** once; a report opens by itself if the card fails.

## Good to know

- The field boundary is still used only by auto-label (people outside the field are skipped); training, benchmarks and inference do not use it.
- A video added after the migration has no field boundary until you mark one (the hint banner asks for it).

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.29.0 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.29.0 update. This release updates the interface and workers; the launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.29.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.29.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.29.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.29.0.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
