# Reco Trainer 0.31.8 — refresh keeps your images, training/validation/test at a glance

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **"Safely refresh frames" no longer drops the images of videos that left the working folder.** Their images and boxes stay, so videos can move to an external drive once their frames are extracted. Images added with "Extend dataset" or the independent test stay too. Videos prepared before are not extracted again; only new videos are.

## New

- **Training, validation and test at a glance.** Chips above the image list show how many images train, validate and test, and which are not in training yet; a click filters the list. Every image shows its role. The "Train locally" dialog shows the split of the next run and warns when there are no test images or very few validation images.
- **Importing a Reco Trainer dataset keeps its roles.** An "Export dataset" folder or a copy of a project's `dataset` folder keeps its train/valid/test split on "Import labels" (both were split by video). Importing it again sets the roles of images imported before. Other datasets still import as training images.
