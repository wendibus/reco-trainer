"""Test: can an AMD graphics card train RF-DETR on Windows with AMD's ROCm build of PyTorch?

Run it with "Test AMD Training.bat" (Windows 11, AMD Radeon RX 7000/9000). It does not touch
Reco Trainer's own ML environment: everything goes into a separate folder,
%LOCALAPPDATA%\\Reco Trainer\\amd-training-probe (delete it afterwards to free about 6 GB).

Steps: find Python 3.12 (AMD's PyTorch wheels exist for 3.12 only) -> create a venv ->
install AMD's ROCm 7.2.1 PyTorch 2.9.1 from repo.radeon.com and RF-DETR -> train RF-DETR Nano
for one epoch on 64 generated images, once on the graphics card and once on the CPU ->
write amd-training-report.txt (also copied to the desktop). The report contains hardware,
versions, timings and errors only - no images from your projects.
Standard library only; the second half (--bench) runs inside the new venv.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROCM = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
SDK_WHEELS = [ROCM + name for name in (
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "rocm-7.2.1.tar.gz")]
TORCH_WHEELS = [ROCM + name for name in (
    "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl")]
# Keep pip from replacing AMD's torch with a CPU build while it installs RF-DETR.
CONSTRAINTS = "torch==2.9.1+rocm7.2.1\ntorchvision==0.24.1+rocm7.2.1\ntorchaudio==2.9.1+rocm7.2.1\n"
RFDETR = "rfdetr[train]==1.11.2"
FOLDER = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Reco Trainer" / "amd-training-probe"


def say(text: str) -> None:
    print(text, flush=True)


def run(command: list[str], log: list[str], timeout: int = 3600) -> int:
    say("> " + " ".join(command)[:200])
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    log.append(f"$ {' '.join(command)}\nexit {result.returncode}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    if result.returncode:
        say((result.stderr or result.stdout)[-1500:])
    return result.returncode


def find_python312() -> str | None:
    for command in (["py", "-3.12"], ["python3.12"], [sys.executable]):
        try:
            out = subprocess.run(command + ["-c", "import sys; print(sys.version_info[:2] == (3, 12), sys.executable)"],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        ok, _, path = out.stdout.strip().partition(" ")
        if out.returncode == 0 and ok == "True":
            return path
    return None


def graphics_cards() -> list[dict]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import gpu_diagnosis
        return [{key: card.get(key) for key in ("DriverDesc", "DriverVersion", "DriverDate", "recognisedAs")} for card in gpu_diagnosis.registry_graphics()]
    except Exception as error:  # noqa: BLE001
        return [{"error": str(error)}]


def setup_and_run() -> int:
    report: dict = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": platform.platform(), "graphics": graphics_cards()}
    log: list[str] = []
    if os.name != "nt":
        say("This test is for Windows only.")
        return 1
    python = find_python312()
    if not python:
        say("Python 3.12 is missing. AMD's PyTorch for Windows exists for Python 3.12 only.\n"
            "Install it from https://www.python.org/downloads/ (Windows installer 64-bit, 3.12.x, tick \"py launcher\"),\n"
            "then start this test again. Your other Python versions stay as they are.")
        return 1
    FOLDER.mkdir(parents=True, exist_ok=True)
    venv = FOLDER / "venv"
    venv_python = venv / "Scripts" / "python.exe"
    started = time.time()
    if not venv_python.exists() and run([python, "-m", "venv", str(venv)], log):
        return finish(report, log, "creating the venv failed")
    pip = [str(venv_python), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check"]
    (FOLDER / "constraints.txt").write_text(CONSTRAINTS, encoding="utf-8")
    say("Installing AMD ROCm 7.2.1 + PyTorch 2.9.1 (about 4 GB, several minutes) ...")
    for name, args in (("ROCm SDK", SDK_WHEELS), ("PyTorch (ROCm)", TORCH_WHEELS),
                       ("RF-DETR", ["-c", str(FOLDER / "constraints.txt"), RFDETR])):
        if run(pip + args, log):
            return finish(report, log, f"installing {name} failed")
    report["installSeconds"] = round(time.time() - started)
    out = FOLDER / "bench.json"
    out.unlink(missing_ok=True)
    say("Training RF-DETR Nano for one epoch on the graphics card and on the CPU ...")
    run([str(venv_python), str(Path(__file__).resolve()), "--bench", str(out)], log, timeout=7200)
    if out.exists():
        report["bench"] = json.loads(out.read_text(encoding="utf-8"))
    return finish(report, log, None)


def finish(report: dict, log: list[str], failure: str | None) -> int:
    report["failure"] = failure
    bench = report.get("bench") or {}
    gpu, cpu = bench.get("train", {}).get("gpu", {}), bench.get("train", {}).get("cpu", {})
    if failure:
        verdict = "FAILED: " + failure
    elif not bench.get("torch", {}).get("cudaAvailable"):
        verdict = "FAILED: PyTorch does not see the graphics card"
    elif gpu.get("seconds") and cpu.get("seconds"):
        verdict = f"OK: one epoch {gpu['seconds']} s on the graphics card, {cpu['seconds']} s on the CPU ({cpu['seconds'] / gpu['seconds']:.1f}x)"
    else:
        verdict = "FAILED: training on the graphics card did not finish" if not gpu.get("seconds") else "PARTLY: the CPU run failed"
    text = "Reco Trainer - AMD training test (ROCm on Windows)\n" + verdict + "\n\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n\n== Log ==\n" + "\n".join(log)[-30000:]
    target = FOLDER / "amd-training-report.txt"
    FOLDER.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    desktop = Path.home() / "Desktop"
    if desktop.is_dir():
        shutil.copy2(target, desktop / "amd-training-report.txt")
    say("\n" + verdict + f"\nReport: {target}" + ("\nA copy is on the desktop: amd-training-report.txt" if desktop.is_dir() else "")
        + "\nPlease send that file. Delete the folder afterwards to free disk space: " + str(FOLDER))
    return 0 if verdict.startswith("OK") else 1


# ---------- inside the ROCm venv ----------

def make_dataset(root: Path) -> None:
    import random
    from PIL import Image, ImageDraw
    random.seed(1)
    for split, count in (("train", 64), ("valid", 16), ("test", 16)):
        (root / split).mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for number in range(count):
            image = Image.new("RGB", (640, 360), (random.randint(150, 200),) * 3)
            draw = ImageDraw.Draw(image)
            for _ in range(random.randint(1, 3)):
                x, y, w, h = random.randint(20, 560), random.randint(20, 240), random.randint(30, 50), random.randint(70, 110)
                draw.rectangle([x, y, x + w, y + h], fill=(30, 60, 160))
                annotations.append({"id": len(annotations) + 1, "image_id": number + 1, "category_id": 2, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            r, x, y = random.randint(6, 14), random.randint(20, 600), random.randint(20, 320)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(230, 120, 30))
            annotations.append({"id": len(annotations) + 1, "image_id": number + 1, "category_id": 1, "bbox": [x - r, y - r, 2 * r, 2 * r], "area": 4 * r * r, "iscrowd": 0})
            image.save(root / split / f"img{number}.jpg")
            images.append({"id": number + 1, "file_name": f"img{number}.jpg", "width": 640, "height": 360})
        (root / split / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "ball"}, {"id": 2, "name": "player"}]}))


def bench(out: Path) -> None:
    result: dict = {}
    try:
        import torch
        info = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None), "cudaAvailable": torch.cuda.is_available()}
        if info["cudaAvailable"]:
            props = torch.cuda.get_device_properties(0)
            info.update(device=torch.cuda.get_device_name(0), memoryGB=round(props.total_memory / 1024 ** 3, 1), arch=getattr(props, "gcnArchName", None))
            a = torch.randn(2048, 2048)
            t = time.time(); gpu = (a.cuda() @ a.cuda()).cpu(); torch.cuda.synchronize(); info["matmulGpuSeconds"] = round(time.time() - t, 3)
            info["matmulMaxDifference"] = float((gpu - a @ a).abs().max())
        result["torch"] = info
    except Exception:  # noqa: BLE001
        result["torchError"] = traceback.format_exc()[-3000:]
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    root = out.parent / "synthetic"
    shutil.rmtree(root, ignore_errors=True)
    make_dataset(root)
    result["train"] = {}
    for name, accelerator in (("gpu", "gpu"), ("cpu", "cpu")):
        if name == "gpu" and not result.get("torch", {}).get("cudaAvailable"):
            continue
        try:
            from rfdetr import RFDETRNano
            model = RFDETRNano()
            t = time.time()
            model.train(dataset_dir=str(root), output_dir=str(out.parent / f"run-{name}"), epochs=1, batch_size=4, grad_accum_steps=1,
                        num_workers=0, accelerator=accelerator, devices=1, early_stopping=False, run_test=False, tensorboard=False)
            result["train"][name] = {"seconds": round(time.time() - t, 1)}
            if name == "gpu":
                result["train"][name]["peakMemoryGB"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        except Exception:  # noqa: BLE001
            result["train"][name] = {"error": traceback.format_exc()[-4000:]}
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--bench":
        bench(Path(sys.argv[2]))
    else:
        sys.exit(setup_and_run())
