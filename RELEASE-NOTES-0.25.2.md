# Reco Trainer 0.25.2 — PyTorch that cannot load no longer blocks training

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fix

- **"Set up ML" now checks that PyTorch actually loads.** On a Windows PC with an NVIDIA GPU, the CUDA build of PyTorch could install cleanly and still fail to load (`WinError 127` on `c10_cuda.dll`, caused for example by an outdated NVIDIA driver or a missing Visual C++ runtime). Training then crashed with a confusing error. Set up ML now detects this and automatically switches to the CPU build of PyTorch, which always works (just slower). If even that cannot load, it says so clearly and points to the Microsoft Visual C++ Redistributable.
- **Already affected?** Click **Set up ML** again. The broken environment is repaired without deleting anything. Your videos, labels and models are not touched.
- If training itself hits a PyTorch loading error, the message now says what to do instead of showing a raw traceback.
- **Quality scores of trained models:** after training, the score is now measured on the best saved checkpoint instead of the last epoch. A model that peaked early was previously rated too low, which also affected the automatic "best model" decision.

Thanks to everyone who posts logs in the forum: this fix exists because of such a report from a Windows user.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.25.2.dmg` (signed and notarized)
- **Mac update package:** `Reco.Trainer.Payload.0.25.2.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.25.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.25.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.25.2.zip`
