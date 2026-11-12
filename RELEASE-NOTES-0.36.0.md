# Reco Trainer 0.36.0 — free length for the simulation, how many new images are test, label the rest

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Any length for the simulation.** Type the length of the section in seconds, or click **¼**, **½**, **¾** or **all** to take that share of the video from the start. The limit of 2,160 images per run is gone; only a run of more than 20,000 images is refused, with a suggestion for a sparser density or a shorter section. The panel shows how many images that gives once the length of the video is known.
- **Changed the section? It asks.** If you change start, length or density while a simulation is shown, the panel asks whether to analyse the new section now. The video does not have to be chosen again.
- **How many new images are test?** The **Train locally** dialog now asks how many of the images you reviewed since the last training should serve as test and not be trained on: none, 10, 20, 30 or 50 percent. The chosen images stay test in later trainings, so the test does not change from run to run.
- **Label the rest.** The same dialog offers **Label the N images without boxes**: it closes the dialog and shows the images that are not in training yet, so you can label them.

## Notes

- Images from the same video look alike, so test scores from images of a video you also train on come out a little too good. For an honest test use a separate video under **Independent model test**.
- Images you confirmed before this version do not count as "new".
- Checked with a real model: a quarter of a one-hour video at every 12th image (905 images in 48 s) and a new section without choosing the video again. The test share is covered by automatic tests; a real training with it was not started. The interface was built and type-checked, but the end-to-end browser test was not run.
