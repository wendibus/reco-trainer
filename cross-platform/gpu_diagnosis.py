"""GPU diagnosis for AMD and Intel graphics (DirectML for detection, Vulkan for the VLM).

Runs every check on its own, records what it found, how long it took and the full traceback of
anything that failed, and writes a plain-text report the user can send back. Each step is
independent: a missing package or a crashing driver only fails that step. Standard library at
module level, so the report still comes out when the ML environment itself is broken.

The report contains hardware, driver, package and log information only - no images, videos or
labels."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PACKAGES = ("torch", "torchvision", "onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnx", "rfdetr", "numpy", "pillow")
ONNXRUNTIME_DISTRIBUTIONS = ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu")
# Lines of dxdiag's display section worth reading (the full file is several hundred lines).
# The provider under test; tests swap in CPUExecutionProvider to run every step on any computer.
GPU_PROVIDER = "DmlExecutionProvider"
DXDIAG_KEYS = (
    "Card name", "Manufacturer", "Chip type", "Device Type", "Device Problem Code", "Driver Problem Code",
    "Display Memory", "Dedicated Memory", "Shared Memory", "Driver Version", "Driver Date/Size", "Driver Model",
    "Feature Levels", "DDI Version", "WDDM", "Hardware Scheduling", "Vendor ID", "Device ID", "SubSys ID",
    "Operating System", "DirectX Version", "DxDiag Version",
)
# dxdiag prints "Unknown" when the driver reports no code at all (seen on a working RX 7800 XT).
DXDIAG_OK_CODES = ("No Problem", "Kein Problem", ": 0", "Unknown", "Unbekannt")


class Step:
    def __init__(self, key: str, title: str) -> None:
        self.key, self.title = key, title
        self.status = "ok"  # ok | failed | warning | skipped - a step that finds nothing wrong stays "ok"
        self.summary = ""
        self.details: dict[str, Any] = {}
        self.error: str | None = None
        self.traceback: str | None = None
        self.seconds = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "status": self.status, "summary": self.summary,
            "details": self.details, "error": self.error, "traceback": self.traceback, "seconds": round(self.seconds, 2),
        }


class Diagnosis:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.steps: list[Step] = []

    def run(self, key: str, title: str, function: Callable[[Step], None]) -> Step:
        step = Step(key, title)
        self.steps.append(step)
        self.emit(f"GPU diagnosis: {title} …")
        started = time.perf_counter()
        try:
            function(step)
        except Exception as error:  # noqa: BLE001 - every failure is what the report is for
            step.status = "failed"
            step.error = f"{type(error).__name__}: {error}".strip()
            step.traceback = traceback.format_exc()
        step.seconds = time.perf_counter() - started
        self.emit(f"GPU diagnosis: {title}: {step.status}{(' - ' + step.error) if step.error else ''}")
        return step


def run_command(command: list[str], timeout: float = 60.0, cwd: str | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=cwd, check=False)
        return {"command": command, "exitCode": result.returncode, "output": (result.stdout + result.stderr)[-6000:]}
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + (error.stderr or "") if isinstance(error.stdout, str) else ""
        return {"command": command, "exitCode": None, "output": f"timed out after {timeout:.0f} s\n{output[-3000:]}"}
    except OSError as error:
        return {"command": command, "exitCode": None, "output": f"{type(error).__name__}: {error}"}


# --- steps ---------------------------------------------------------------------------------------

def step_system(step: Step, app_version: str) -> None:
    memory = 0
    try:
        memory = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        if os.name == "nt":
            memory = windows_memory_bytes()
    step.details = {
        "recoTrainer": app_version,
        "platform": platform.platform(),
        "windowsVersion": list(sys.getwindowsversion())[:4] if hasattr(sys, "getwindowsversion") else None,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpuCores": os.cpu_count(),
        "memoryGB": round(memory / 1024**3, 1) if memory else None,
        "python": sys.version,
        "pythonExecutable": sys.executable,
        "environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith(("RECO_", "ORT_", "GGML_", "PYTORCH_", "CUDA_"))},
        "nvidiaSmi": shutil.which("nvidia-smi"),
    }
    step.summary = f"{step.details['platform']} · Python {platform.python_version()} · {step.details['memoryGB']} GB"


def windows_memory_bytes() -> int:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return int(status.ullTotalPhys)
    except Exception:  # noqa: BLE001
        return 0


def registry_graphics() -> list[dict[str, Any]]:
    """Every display adapter in the registry with its driver details (Microsoft's included)."""
    try:
        import winreg
    except ImportError:
        return []
    import teachers
    cards: list[dict[str, Any]] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, teachers.DISPLAY_CLASS_KEY)
    except OSError as error:
        return [{"error": f"registry not readable: {error}"}]
    with root:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            if not name.isdigit():
                continue
            values: dict[str, Any] = {"key": name}
            try:
                with winreg.OpenKey(root, name) as key:
                    for value in ("DriverDesc", "ProviderName", "DriverVersion", "DriverDate", "MatchingDeviceId",
                                  "HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize", "HardwareInformation.ChipType",
                                  "HardwareInformation.AdapterString", "FeatureScore"):
                        try:
                            raw = winreg.QueryValueEx(key, value)[0]
                        except OSError:
                            continue
                        if isinstance(raw, (bytes, bytearray)):
                            raw = int.from_bytes(bytes(raw)[:8], "little") if "MemorySize" in value else bytes(raw).decode("utf-16-le", errors="replace").strip("\x00")
                        values[value] = raw
            except OSError as error:
                values["error"] = str(error)
            entry = teachers.windows_graphics_entry(values)
            values["recognisedAs"] = entry
            cards.append(values)
    return cards


def step_graphics(step: Step) -> None:
    import teachers
    if os.name != "nt":
        step.status, step.summary = "skipped", "not Windows"
        return
    cards = registry_graphics()
    chosen = teachers.directml_graphics()
    step.details = {"adapters": cards, "chosenForDirectML": chosen}
    if not cards:
        step.status, step.summary = "failed", "no display adapter found in the registry"
        return
    if chosen is None:
        step.status = "warning"
        step.summary = "no AMD or Intel card recognised: " + ", ".join(str(card.get("DriverDesc")) for card in cards)
        return
    step.status = "ok"
    step.summary = f"{chosen['name']} ({chosen['vendor']}, {chosen['vramGB']:g} GB)"


def step_dxdiag(step: Step) -> None:
    if os.name != "nt" or not shutil.which("dxdiag"):
        step.status, step.summary = "skipped", "dxdiag not available"
        return
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "dxdiag.txt"
        result = run_command(["dxdiag", "/whql:off", "/t", str(target)], timeout=120)
        deadline = time.time() + 60
        while not target.is_file() and time.time() < deadline:  # dxdiag may return before the file is written
            time.sleep(1)
        if not target.is_file():
            step.status, step.summary, step.details = "warning", "dxdiag wrote no report", result
            return
        text = target.read_text(encoding="utf-8", errors="replace")
    step.status, step.summary, step.details = read_dxdiag(text)


def read_dxdiag(text: str) -> tuple[str, str, dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if any(line.strip().startswith(key + ":") for key in DXDIAG_KEYS)]
    problems = [line for line in lines if re.match(r"(Device|Driver) Problem Code:", line) and not line.endswith(DXDIAG_OK_CODES)]
    summary = "; ".join(problems) if problems else next((line for line in lines if line.startswith("Feature Levels:")), "read")
    return ("warning" if problems else "ok"), summary, {"lines": lines[:80]}


def step_packages(step: Step) -> None:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    step.details = {"versions": versions}
    runtimes = [name for name in ONNXRUNTIME_DISTRIBUTIONS if versions.get(name)]
    if not runtimes:
        step.status, step.summary = "failed", "no ONNX Runtime installed"
    elif len(runtimes) > 1:
        # They share the module "onnxruntime": whichever was installed last owns the files.
        step.status, step.summary = "failed", "several ONNX Runtime packages installed at once: " + ", ".join(runtimes)
    elif os.name == "nt" and runtimes != ["onnxruntime-directml"]:
        step.status, step.summary = "failed", f"{runtimes[0]} installed instead of onnxruntime-directml"
    else:
        step.status = "ok"
        step.summary = ", ".join(f"{name} {version}" for name, version in versions.items() if version)


def step_torch(step: Step) -> None:
    import torch
    step.details = {
        "version": torch.__version__, "cudaBuild": getattr(torch.version, "cuda", None),
        "cudaAvailable": bool(torch.cuda.is_available()), "threads": torch.get_num_threads(),
    }
    step.status = "ok"
    step.summary = f"PyTorch {torch.__version__} · CUDA {'yes' if step.details['cudaAvailable'] else 'no'}"


def step_onnxruntime(step: Step) -> None:
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    package = Path(onnxruntime.__file__).parent
    dlls = sorted(str(path.relative_to(package)) + f" ({path.stat().st_size} bytes)" for path in package.rglob("*.dll") if "directml" in path.name.lower() or "onnxruntime" in path.name.lower())
    step.details = {"version": onnxruntime.__version__, "device": onnxruntime.get_device(), "providers": providers, "folder": str(package), "dlls": dlls}
    if GPU_PROVIDER in providers:
        step.status = "ok"
    else:
        step.status = "failed"
        step.error = f"{GPU_PROVIDER} is not among the available providers"
    step.summary = f"ONNX Runtime {onnxruntime.__version__}: {', '.join(providers)}"


def step_dml_minimal(step: Step) -> None:
    """A two-operation graph (Conv + Relu) on DirectML: separates "DirectML does not start at all"
    from "RF-DETR's graph has an operation the driver cannot run"."""
    import numpy as np
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper, numpy_helper
    weights = numpy_helper.from_array(np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), helper.make_node("Relu", ["c"], ["y"])],
        "reco-dml-check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 64, 64])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 9)
    data = np.random.default_rng(1).standard_normal((1, 3, 64, 64)).astype(np.float32)
    expected = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": data})[0]
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=[GPU_PROVIDER, "CPUExecutionProvider"])
    actual = session.run(None, {"x": data})[0]
    difference = float(np.abs(actual - expected).max())
    step.details = {"sessionProviders": session.get_providers(), "maxDifference": difference, "onnx": onnx.__version__}
    if session.get_providers()[0] != GPU_PROVIDER:
        step.status, step.error = "failed", f"session fell back to {session.get_providers()[0]}"
    elif difference > 1e-2:
        step.status, step.error = "failed", f"DirectML result differs from the CPU by {difference:.4f}"
    else:
        step.status = "ok"
    step.summary = f"Conv+Relu on {session.get_providers()[0]}, max. difference {difference:.2e}"


def step_rfdetr(step: Step, ml_worker: Any) -> None:
    """The real path: RF-DETR Nano exported to ONNX, run on DirectML and on the CPU, compared."""
    import numpy as np
    import onnxruntime
    from PIL import Image
    model_class = ml_worker.import_model_class("nano", "en")
    model = model_class(device="cpu")
    started = time.perf_counter()
    # Same export and cache as accelerated_model(), forced onto DirectML.
    previous = os.environ.get("RECO_ONNX_INFERENCE")
    os.environ["RECO_ONNX_INFERENCE"] = "on"
    try:
        wrapped = ml_worker.accelerated_model(model, None, "nano", "en")
    finally:
        if previous is None:
            os.environ.pop("RECO_ONNX_INFERENCE", None)
        else:
            os.environ["RECO_ONNX_INFERENCE"] = previous
    export_seconds = time.perf_counter() - started
    if not isinstance(wrapped, ml_worker.OnnxRfdetrModel):
        raise RuntimeError("the ONNX export or the DirectML session failed (see the job log line before this step)")
    image = Image.fromarray((np.random.default_rng(2).random((480, 854, 3)) * 255).astype(np.uint8))
    tensor, _width, _height = wrapped._tensor(image)
    onnx_path = Path(wrapped.onnx_path)
    timings: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for label, chosen in (("directml", None), ("cpu", ["CPUExecutionProvider"])):
        session = wrapped.session if label == "directml" else onnxruntime.InferenceSession(str(onnx_path), providers=chosen)
        session.run(None, {wrapped.input_name: tensor})  # warm-up (DirectML compiles shaders on the first run)
        runs = []
        for _ in range(3):
            run_started = time.perf_counter()
            result = session.run(None, {wrapped.input_name: tensor})
            runs.append(time.perf_counter() - run_started)
        timings[label] = {"providers": session.get_providers(), "msPerImage": round(1000 * min(runs), 1)}
        outputs[label] = result
    boxes = float(np.abs(outputs["directml"][wrapped.boxes_index] - outputs["cpu"][wrapped.boxes_index]).max())
    logits = float(np.abs(outputs["directml"][wrapped.logits_index] - outputs["cpu"][wrapped.logits_index]).max())
    step.details = {
        "onnxFile": str(onnx_path), "inputSize": [wrapped.height, wrapped.width], "exportAndLoadSeconds": round(export_seconds, 1),
        "timings": timings, "maxBoxDifference": boxes, "maxLogitDifference": logits,
    }
    if wrapped.providers[0] != GPU_PROVIDER:
        step.status, step.error = "failed", f"RF-DETR session runs on {wrapped.providers[0]}, not DirectML"
    elif boxes > 0.01 or logits > 0.5:
        step.status, step.error = "failed", f"DirectML results differ from the CPU (boxes {boxes:.4f}, logits {logits:.3f})"
    else:
        step.status = "ok"
    step.summary = f"RF-DETR Nano: DirectML {timings['directml']['msPerImage']} ms, CPU {timings['cpu']['msPerImage']} ms per image"


def step_vulkan(step: Step, ml_worker: Any, project_root: Path | None) -> None:
    import vlm
    details: dict[str, Any] = {"vulkaninfo": None, "llamaServer": None}
    vulkaninfo = shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK")
    if vulkaninfo:
        details["vulkaninfo"] = run_command([vulkaninfo, "--summary"], timeout=60)
    system_dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll"
    details["vulkanLoader"] = str(system_dll) if system_dll.is_file() else None
    binary = None
    if project_root is not None:
        try:
            folder = ml_worker.registry_model_path(project_root, "llama-server-windows-vulkan", "en")
            binary = vlm.find_server_binary(folder) if folder.is_dir() else None
            if binary is None and os.name == "nt":
                # Only the server build (a few MB), not the VLM itself: enough to list the devices.
                ml_worker.download_registry_model(project_root, "llama-server-windows-vulkan", "en")
                binary = vlm.find_server_binary(folder)
            details["llamaServerFolder"] = str(folder)
        except SystemExit as error:
            details["llamaServerFolder"] = f"unknown: {error}"
    if binary is not None:
        details["llamaServer"] = run_command([str(binary), "--list-devices"], timeout=60, cwd=str(binary.parent))
        details["llamaServer"]["version"] = run_command([str(binary), "--version"], timeout=30, cwd=str(binary.parent))["output"][-400:]
    step.details = details
    if os.name == "nt" and details["vulkanLoader"] is None:
        step.status, step.error = "failed", "vulkan-1.dll is missing - the graphics driver installs no Vulkan support"
    elif binary is None:
        step.status = "warning"
        step.summary = "Vulkan llama-server not downloaded yet (it comes with the first VLM check)"
        return
    else:
        listed = details["llamaServer"]["output"]
        gpus = [line.strip() for line in listed.splitlines() if re.match(r"\s*Vulkan\d+", line)]
        step.status = "ok" if gpus else "failed"
        if not gpus:
            step.error = "llama-server lists no Vulkan device"
        step.summary = "; ".join(gpus) if gpus else (step.error or "")
        return
    step.summary = step.error or ""


# --- report --------------------------------------------------------------------------------------

def run_diagnosis(ml_worker: Any, out_dir: Path, project_root: Path | None, context: dict[str, Any], app_version: str) -> dict[str, Any]:
    diagnosis = Diagnosis(ml_worker.emit)
    diagnosis.run("system", "System", lambda step: step_system(step, app_version))
    diagnosis.run("graphics", "Graphics cards (registry)", step_graphics)
    diagnosis.run("dxdiag", "DirectX (dxdiag)", step_dxdiag)
    packages = diagnosis.run("packages", "Python packages", step_packages)
    diagnosis.run("torch", "PyTorch", step_torch)
    runtime = diagnosis.run("onnxruntime", "ONNX Runtime providers", step_onnxruntime)
    if runtime.status == "ok":
        minimal = diagnosis.run("dml-minimal", "DirectML: small test graph", step_dml_minimal)
        if minimal.status == "ok":
            diagnosis.run("dml-rfdetr", "DirectML: RF-DETR Nano", lambda step: step_rfdetr(step, ml_worker))
    elif packages.status != "ok":
        runtime.summary += " (see Python packages)"
    diagnosis.run("vulkan", "Vulkan (VLM)", lambda step: step_vulkan(step, ml_worker, project_root))

    steps = [step.as_dict() for step in diagnosis.steps]
    failed = [step for step in steps if step["status"] == "failed"]
    created = datetime.now(timezone.utc)
    identifier = created.strftime("%Y%m%d-%H%M%S")
    report = {"id": identifier, "createdAt": created.isoformat(), "ok": not failed, "trigger": context.get("trigger"), "steps": steps, "context": context}
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / f"gpu-diagnosis-{identifier}.txt"
    text_path.write_text(report_text(report), encoding="utf-8")
    (out_dir / f"gpu-diagnosis-{identifier}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for old in sorted(out_dir.glob("gpu-diagnosis-*.txt"), reverse=True)[10:]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
    return {
        "id": identifier, "ok": not failed, "reportPath": str(text_path), "trigger": context.get("trigger"),
        "steps": [{"key": step["key"], "title": step["title"], "status": step["status"], "summary": step["summary"], "error": step["error"]} for step in steps],
    }


def report_text(report: dict[str, Any]) -> str:
    marks = {"ok": "OK  ", "failed": "FAIL", "warning": "WARN", "skipped": "SKIP"}
    lines = [
        "Reco Trainer - GPU diagnosis (AMD/Intel: DirectML + Vulkan)",
        f"Created: {report['createdAt']}",
        f"Result: {'all checks passed' if report['ok'] else 'checks failed'}",
        "Contains hardware, driver, package and log information only - no images, videos or labels.",
        "",
        "== Overview ==",
    ]
    for step in report["steps"]:
        lines.append(f"[{marks.get(step['status'], step['status'])}] {step['title']}: {step['summary'] or ''}{(' | ' + step['error']) if step['error'] else ''}")
    context = report.get("context") or {}
    if context.get("problem"):
        problem = context["problem"]
        lines += ["", "== What failed in the app ==", f"Operation: {problem.get('operation')}", f"Message: {problem.get('reason')}"]
    for step in report["steps"]:
        lines += ["", f"== {step['title']} ({step['status']}, {step['seconds']} s) =="]
        if step["details"]:
            lines.append(json.dumps(step["details"], indent=2, ensure_ascii=False, default=str))
        if step["traceback"]:
            lines += ["Traceback:", step["traceback"].rstrip()]
    for title, key in (("Log of the failed operation", "problemLog"), ("Setup log", "setupLog"), ("Recent app log", "log")):
        if context.get(key):
            lines += ["", f"== {title} ==", *[str(line) for line in context[key]][-150:]]
    return "\n".join(lines) + "\n"
