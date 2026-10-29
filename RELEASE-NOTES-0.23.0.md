# Reco Trainer 0.23.0 — Consistency Check with a Built-in Local VLM

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Consistency check

- **Check consistency** compares your reviewed labels with a second detector (match from an adjustable IoU, default 0.5) and with the timeline of each video: players who suddenly vanish and jumping boxes are flagged as well.
- Flagged frames end up in a review list you can open frame by frame.

## Local vision-language model (VLM)

- Optionally, **Qwen3.5-4B or 9B** (Apache-2.0) looks at the flagged frames and a sample of unremarkable ones. Confirmed problems become **hard examples**, which are used twice in training; unfounded warnings are cleared.
- The anonymized dataset export can let the VLM check a sample of enlarged head crops for faces that are still recognizable; such frames go to the review list instead of the export.
- **Built into the app on every platform** - no Ollama or other separate installation. On first use Reco Trainer downloads a pinned llama.cpp `llama-server` build (b11985, MIT) for your system and the model, verifies their checksums through the job system and keeps them once per computer. The server listens on `127.0.0.1` only, runs just for the job and is stopped afterwards, freeing the GPU memory.

| Platform | llama.cpp build |
|---|---|
| macOS Apple silicon | Metal |
| Windows x64 | CUDA 12.4 with NVIDIA, otherwise CPU |
| Linux x64 | CUDA 12.8 with NVIDIA (glibc 2.38+), otherwise CPU (glibc 2.34+) |
| Linux arm64 | CPU (glibc 2.38+) |
| Docker | Linux build inside the container (image now on Debian 13) |

- Hardware tiers: the VLM is on by default on **strong** (9B selectable), optional on **medium**, and not offered on **weak** (CPU only) - choose medium under **Advanced options** to use it anyway. Downloads: about 3.4 GB (4B) or 6.6 GB (9B).

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.23.0.dmg` (signed and notarized)
- **Mac update package:** `Reco.Trainer.Payload.0.23.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.23.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.23.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.23.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training and the VLM remain local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. Model and runtime downloads come only from the sources listed in `model_registry.json`.
