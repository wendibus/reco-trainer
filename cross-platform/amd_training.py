"""Training RF-DETR on an AMD graphics card on Windows (ROCm). Standard library only.

AMD publishes PyTorch for Windows with ROCm only for Python 3.12 and only for its own index
(repo.radeon.com), so this is a separate environment next to the normal ML environment:

  <app data>/rocm-env/venv     AMD's ROCm 7.2.1 + PyTorch 2.9.1, RF-DETR in the same version as
                               the normal ML environment, onnxruntime (CPU) for the checks
  <app data>/rocm-env/state.json  {"verified": {...}, "enabled": bool}

Only "Train locally" with RF-DETR runs there; auto-labeling, benchmark, export and YOLO stay in the
normal environment (detection on AMD keeps using DirectML). The weights are plain PyTorch
checkpoints and load in both. In PyTorch the card appears as "cuda", so ml_worker trains on it
without changes. Tested on an RX 7800 XT (gfx1101): one epoch 16 s on the card vs. 711 s on the CPU.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

ROCM = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
SDK_WHEELS = [ROCM + name for name in (
    "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "rocm-7.2.1.tar.gz")]
TORCH_WHEELS = [ROCM + name for name in (
    "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", "torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl")]
# Keeps pip from replacing AMD's torch with a CPU build while it installs RF-DETR.
CONSTRAINTS = "torch==2.9.1+rocm7.2.1\ntorchvision==0.24.1+rocm7.2.1\ntorchaudio==2.9.1+rocm7.2.1\n"
FALLBACK_RFDETR = "1.11.2"
PYTHON = (3, 12)

# Cards on AMD's list for PyTorch on Windows with ROCm 7.2.x: gfx1100/gfx1101 (RX 7700-7900,
# PRO W7700-W7900) and gfx1200/gfx1201 (RX 9060/9070, AI PRO R9700). RX 7600 (gfx1102) and the
# Ryzen graphics are not on it. The check after the install decides; others are not offered.
SUPPORTED_CARD = re.compile(r"RX\s*(7[789]\d\d|9\d\d\d)|PRO\s*W7[789]\d\d|PRO\s*R9\d\d\d", re.IGNORECASE)

# Run inside the new environment: does PyTorch see the card, and does a small training step work?
VERIFY_SCRIPT = """
import json, torch
info = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "available": torch.cuda.is_available()}
if info["available"]:
    props = torch.cuda.get_device_properties(0)
    info.update(device=torch.cuda.get_device_name(0), memoryGB=round(props.total_memory / 1024 ** 3, 1), arch=getattr(props, "gcnArchName", None))
    a = torch.randn(512, 512)
    info["matmulDifference"] = float(((a.cuda() @ a.cuda()).cpu() - a @ a).abs().max())
    layer = torch.nn.Conv2d(3, 8, 3).cuda()
    loss = layer(torch.randn(2, 3, 64, 64, device="cuda")).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    info["backward"] = bool(torch.isfinite(layer.weight.grad).all())
print(json.dumps(info))
"""


def env_dir(app_data: Path | None) -> Path | None:
    return app_data / "rocm-env" if app_data is not None else None


def venv_python(folder: Path) -> Path:
    return folder / "venv" / "Scripts" / "python.exe"


def card_supported(card: dict[str, Any] | None) -> bool:
    return bool(card) and card.get("vendor") == "amd" and bool(SUPPORTED_CARD.search(str(card.get("name", ""))))


def is_windows_x64() -> bool:
    return os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"}


def read_state(folder: Path | None) -> dict[str, Any]:
    if folder is None:
        return {}
    try:
        value = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(folder: Path, state: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, folder / "state.json")


def verified(result: dict[str, Any] | None) -> bool:
    """The check after the install passed: card seen, matrix product right, gradient finite."""
    return bool(result) and bool(result.get("available")) and bool(result.get("backward")) \
        and float(result.get("matmulDifference", 1.0)) < 0.01


def ready(folder: Path | None) -> bool:
    """Installed and checked - whether it is switched on is enabled()."""
    return folder is not None and venv_python(folder).is_file() and verified(read_state(folder).get("verified"))


def enabled(folder: Path | None) -> bool:
    return ready(folder) and read_state(folder).get("enabled", True) is not False


def training_python(folder: Path | None) -> Path | None:
    """The interpreter for "Train locally" with RF-DETR, or None for the normal ML environment."""
    return venv_python(folder) if folder is not None and enabled(folder) else None


def find_python312() -> str | None:
    """AMD's wheels are cp312 only. The py launcher finds a 3.12 next to other versions."""
    for command in (["py", "-3.12"], ["python3.12"], ["python"]):
        try:
            out = subprocess.run(command + ["-c", "import sys; print(sys.version_info[:2] == (3, 12), sys.executable)"],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        ok, _, path = out.stdout.strip().partition(" ")
        if out.returncode == 0 and ok == "True" and path:
            return path
    return None


def install_steps(python: Path, rfdetr_version: str, constraints: Path) -> list[tuple[str, list[str]]]:
    """(name, pip arguments) in order. RF-DETR in the same version as the normal environment, so
    checkpoints and the evaluation match whichever environment reads them."""
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    version = rfdetr_version if re.fullmatch(r"\d+(\.\d+){1,3}", rfdetr_version or "") else FALLBACK_RFDETR
    return [
        ("pip", [str(python), "-m", "pip", "install", "--upgrade", "pip"]),
        ("ROCm SDK", pip + SDK_WHEELS),
        ("PyTorch (ROCm)", pip + TORCH_WHEELS),
        ("RF-DETR", pip + ["-c", str(constraints), f"rfdetr[train,onnx]=={version}", "onnxruntime"]),
    ]


def status(folder: Path | None, card: dict[str, Any] | None) -> dict[str, Any] | None:
    """For the app: None when AMD training is not possible on this computer."""
    if folder is None or (not is_windows_x64() and os.environ.get("RECO_SIMULATE_AMD") != "1"):
        return None
    if not card_supported(card):
        return None
    state = read_state(folder)
    check = state.get("verified") if isinstance(state.get("verified"), dict) else None
    return {
        "card": card.get("name") if card else None,
        "installed": venv_python(folder).is_file(),
        "ready": ready(folder),
        "enabled": enabled(folder),
        "device": (check or {}).get("device"),
        "problem": state.get("problem"),
    }
