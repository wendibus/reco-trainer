# Reco Trainer 0.37.0 — a role for each video, YOLO26 models in the Model Hub

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Choose the role of each video.** Under the chips above the image list, **Roles of the videos** lists every video with its role in the next training and a choice: Automatic, Training, Validation or Test. A video always stays whole in its role. If validation or test is not given to any video, the app picks it from the videos left over, as long as one video stays for training. The choice is saved in the project and used by every training and export.
- **YOLO26 models in the Model Hub.** Models you trained with YOLO (Nano, Small, Medium) can now be shared and loaded like RF-DETR models, with the same safety concept: only the cleaned weights travel, as a safetensors file that cannot run code, no pictures and no local paths, and a maintainer reviews every upload. Whoever loads a YOLO model rebuilds it locally from the weights, which must fit a YOLO26 model of that size exactly, and exports it as ONNX; no Ultralytics checkpoint is ever shared or opened. A model you load can be the starting point of your next YOLO training.
- **License shown.** YOLO26 is licensed under the AGPL-3.0, and so are models trained with it. The hub shows the license, the uploader confirms it, and you agree to it before a YOLO model is loaded. Ultralytics is installed for that if it is missing.

## Notes

- Only models trained in Reco Trainer can be shared; imported ONNX models cannot.
- Older versions of Reco Trainer do not list YOLO models.
- The hub repository has to accept YOLO26 first (pull request wendibus/reco-models#3). Until it is merged, a YOLO upload is rejected by the hub's check.
- Checked with real Ultralytics for Nano, Small and Medium: clean the weights, check them with the hub's code, rebuild locally, export ONNX and run it. A real upload or download through GitHub, a YOLO model trained on real data and the interface in a browser were not tried.
