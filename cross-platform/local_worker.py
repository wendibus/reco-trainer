#!/usr/bin/env python3
"""Loopback-only bridge between the Reco web UI and the local Mac ML worker.

The service binds exclusively to 127.0.0.1. It has no upload endpoint and no
outbound networking code. A folder can only be selected through the native
macOS folder dialog (or RECO_TEST_FOLDER during automated tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import amd_training  # standard library only: RF-DETR training on AMD cards with ROCm (Windows)
import dataset_split  # standard library only: which images train, validate and test
import field_geometry  # standard library only: field outline per video
import model_hub  # standard library only at module level
import teachers  # standard library only at module level: hardware tiers
import vlm  # standard library only: the app-managed VLM (llama-server + Qwen3.5)
import yolo_training  # standard library only at import; installs Ultralytics on request


HOST = os.environ.get("RECO_BIND_HOST", "127.0.0.1")
DEFAULT_PORT = 8766
BASE_DIR = Path(__file__).resolve().parent
# The interface is served by this worker itself (same origin, see serve_static);
# 8765 stays allowed for the Vite dev server (npm run dev).
ALLOWED_ORIGINS = {"http://localhost:8765", "http://127.0.0.1:8765"}
UI_DIR = Path(os.environ.get("RECO_UI_DIR") or (BASE_DIR / "ui")).resolve()
MODEL_REGISTRY = Path(os.environ.get("RECO_MODEL_REGISTRY") or (BASE_DIR / "model_registry.json"))
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon", ".woff2": "font/woff2", ".mp4": "video/mp4",
    ".txt": "text/plain; charset=utf-8", ".map": "application/json",
}
configured_origin = os.environ.get("RECO_ALLOWED_ORIGIN")
if configured_origin and re.fullmatch(r"http://(?:localhost|127\.0\.0\.1):\d{2,5}", configured_origin):
    ALLOWED_ORIGINS.add(configured_origin)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # same as ml_worker.IMPORT_IMAGE_SUFFIXES
DEFAULT_FRAMES_PER_VIDEO = 60
MIN_FRAMES_PER_VIDEO = 4
MAX_FRAMES_PER_VIDEO = 5000
SPORT_CATEGORIES = {
    "football": ["ball", "player", "goalkeeper", "referee", "goal"],
    "futsal": ["ball", "player", "goalkeeper", "referee", "goal"],
    "basketball": ["ball", "player", "referee", "hoop"],
    "handball": ["ball", "player", "goalkeeper", "referee", "goal"],
    "hockey": ["puck", "player", "goalkeeper", "referee", "goal"],
    "rugby": ["ball", "player", "referee", "goalpost"],
    "lacrosse": ["ball", "player", "goalkeeper", "referee", "goal"],
    "american_football": ["ball", "player", "referee", "goalpost"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def localized(german: str, english: str, spanish: str | None = None, french: str | None = None) -> str:
    """Pick the right translation for STATE's current language (see
    LocalState.language, set from each request's "language" field in
    do_POST()). Mirrors ml_worker.py's localized() helper, which every
    subprocess action already uses via its own --language argument - this is
    the equivalent for status/error text generated directly in this file
    rather than by a ml_worker.py subprocess.
    """
    language = STATE.language
    if language == "en":
        return english
    if language == "es":
        return spanish or english
    if language == "fr":
        return french or english
    return german


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def backup_project(root: Path, reason: str) -> Path | None:
    """Keep small, local JSON snapshots before a project-changing operation."""
    source = root / "project.json"
    if not source.is_file():
        return None
    backups = root / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    target = backups / f"project-{stamp}-{safe_name(reason)}.json"
    shutil.copy2(source, target)
    for old in sorted(backups.glob("project-*.json"), reverse=True)[30:]:
        old.unlink(missing_ok=True)
    return target


_ACCELERATOR_CACHE: dict[str, dict] = {}


def hardware_summary() -> dict:
    memory = 0
    try:
        memory = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        if os.name == "nt":  # no sysconf on Windows: the hardware line showed "? GB"
            import gpu_diagnosis  # standard library only

            memory = gpu_diagnosis.windows_memory_bytes()
    machine = platform.machine()
    return {
        "machine": machine,
        "cpuCores": os.cpu_count() or 1,
        "memoryGB": round(memory / (1024 ** 3)) if memory else None,
        # A structured {"kind": ..., "gpuName": ...} rather than a pre-rendered
        # label - this used to be a hardcoded German (or English, for the Apple
        # case) string baked in here regardless of the UI's selected language,
        # since this function has no idea which language the client is using.
        # The frontend renders the actual localized label from "kind".
        "accelerator": {"kind": "apple-gpu"} if sys.platform == "darwin" and machine == "arm64" else non_apple_accelerator(),
    }


def detect_nvidia_gpu_name() -> str | None:
    """Raw GPU presence via nvidia-smi, independent of whether PyTorch itself
    was built with CUDA support - used both for the pre-setup hardware label
    and to decide whether "ML einrichten" should install a CUDA build of
    torch (see ml_action's "setup" branch).
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None


def nvidia_vram_gb() -> float | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run([nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
        return float(result.stdout.strip().splitlines()[0]) / 1024 if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


_TIER_CACHE: dict[str, str] = {}


def detected_tier() -> str:
    """strong / medium / weak for this machine (see teachers.choose_tier), without PyTorch."""
    if "tier" not in _TIER_CACHE:
        memory_gb = 0.0
        try:
            memory_gb = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE")) / 1024**3
        except (AttributeError, OSError, TypeError, ValueError):
            pass
        if sys.platform == "darwin" and platform.machine() == "arm64":
            _TIER_CACHE["tier"] = teachers.choose_tier("mps", memory_gb=memory_gb)
        else:
            vram = nvidia_vram_gb()
            graphics = teachers.directml_graphics() if (not vram and os.name == "nt" and amd_choice() == "yes") else None
            _TIER_CACHE["tier"] = (
                teachers.choose_tier("cuda", vram_gb=vram) if vram
                else teachers.choose_tier("directml", vram_gb=graphics["vramGB"]) if graphics
                else teachers.choose_tier("cpu")
            )
    return _TIER_CACHE["tier"]


def tier_argument(payload: dict) -> str:
    tier = str(payload.get("tier") or "auto")
    return tier if tier in teachers.TIERS else "auto"


def non_apple_accelerator() -> dict:
    """Best-effort accelerator info for Windows/Linux, as {"kind": ..., "gpuName": ...}
    rather than a rendered label (see hardware_summary()) - "kind" is one of
    "cuda", "cpu-no-cuda", "cpu-directml", "cuda-pending", "cpu".
    "cpu-directml": PyTorch trains on the CPU, detection runs on the GPU through ONNX Runtime
    with DirectML (AMD or Intel graphics on Windows, see accelerated_model in ml_worker).

    Previously this branch was hardcoded to "CPU" unconditionally, so Windows
    users with a working NVIDIA GPU had no way to tell from the app whether
    it was actually being used - the /api/status poll (every 900ms, see
    page.tsx) meant this couldn't just shell out to the venv's torch on every
    call, so the result is cached by venv path once computed (recomputing on
    a fresh venv path covers "just finished ML setup" / "reinstalled with a
    different torch build" without re-probing every second).
    """
    rocm = rocm_status()
    if rocm and rocm.get("enabled"):
        return {"kind": "rocm-directml", "gpuName": rocm.get("device") or rocm.get("card")}
    root = STATE.project_root
    if root is not None:
        venv_python = venv_python_path(ml_venv(root))
        if venv_python.is_file():
            cache_key = str(venv_python)
            cached = _ACCELERATOR_CACHE.get(cache_key)
            if cached is not None:
                return cached
            try:
                probe = (
                    "import json, torch\n"
                    "available = torch.cuda.is_available()\n"
                    "try:\n"
                    "    import onnxruntime\n"
                    "    directml = 'DmlExecutionProvider' in onnxruntime.get_available_providers()\n"
                    "except Exception:\n"
                    "    directml = False\n"
                    "print(json.dumps({'cuda': available, 'name': torch.cuda.get_device_name(0) if available else None, 'directml': directml}))"
                )
                result = subprocess.run(
                    [str(venv_python), "-c", probe],
                    capture_output=True, text=True, timeout=8, check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    payload = json.loads(result.stdout.strip().splitlines()[-1])
                    info = (
                        {"kind": "cuda", "gpuName": payload["name"]} if payload.get("cuda") and payload.get("name")
                        else {"kind": "cuda"} if payload.get("cuda")
                        else {"kind": "cpu-directml", "gpuName": (teachers.directml_graphics() or {}).get("name")} if payload.get("directml") and amd_choice() == "yes"
                        else {"kind": "cpu-no-cuda"}
                    )
                    _ACCELERATOR_CACHE[cache_key] = info
                    return info
            except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError, KeyError):
                pass
    # No venv yet (or its torch probe failed) - fall back to raw GPU presence
    # via nvidia-smi so the hint isn't just "CPU" before ML setup has even
    # run. Cached under a fixed key since this doesn't depend on the venv and
    # hardware presence doesn't change mid-session.
    pre_setup_cached = _ACCELERATOR_CACHE.get("pre-setup")
    if pre_setup_cached is not None:
        return pre_setup_cached
    name = detect_nvidia_gpu_name()
    info = {"kind": "cuda-pending", "gpuName": name} if name else {"kind": "cpu"}
    _ACCELERATOR_CACHE["pre-setup"] = info
    return info


def app_version() -> str:
    """Version of this worker: payload.json (Mac payload), version.json (Windows/Linux packages)
    or package.json (Docker, source checkout)."""
    for name in ("payload.json", "version.json", "package.json"):
        try:
            return str(json.loads((BASE_DIR / name).read_text(encoding="utf-8"))["version"])
        except (OSError, ValueError, KeyError):
            continue
    return "unknown"


def model_registry_snapshot(root: Path | None) -> list[dict]:
    """Registry models with their license (shown in the UI) and whether they are downloaded."""
    try:
        models = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        return []
    result = []
    for entry in models:
        installed = None
        if entry.get("file") and root is not None:
            installed = (root / ".runtime" / "models" / Path(entry["file"]).name).is_file()
        elif entry.get("huggingFace") and root is not None:
            installed = (root / ".runtime" / "models" / "hf" / str(entry["id"]) / ".reco-complete").is_file()
        elif entry.get("shared") and not entry.get("installInto"):
            shared = vlm.shared_models_dir()
            folder = (shared if shared is not None else (root / ".runtime" / "models" if root else None))
            installed = bool(folder and (folder / str(entry["id"]) / f".reco-complete-{entry['id']}").is_file())
        result.append({key: entry.get(key) for key in ("id", "name", "purpose", "license", "licenseUrl", "source", "bytes", "managedBy")} | {"installed": installed})
    return result


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in " ._-" else "_" for character in value).strip() or "video"


class LocalState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.selected_folder: Path | None = None
        self.folder_empty = False
        self.project_root: Path | None = None
        self.project: dict | None = None
        self.operation = "idle"
        self.progress = 0.0
        # Empty, not a hardcoded-German default: STATE exists before any
        # request (and its "language" field) has ever been seen, so there is
        # no language to render this in yet. page.tsx's existing
        # `worker?.message || t.localReady` fallback already shows a properly
        # localized placeholder for an empty message.
        self.message = ""
        self.log: list[str] = []
        self.busy = False
        self.error: str | None = None
        self.simulation: dict | None = None
        # Second simulation for the side-by-side model comparison (slot "b").
        self.simulation_b: dict | None = None
        # Source video per simulation slot, kept out of the status so "same video
        # as A" can re-run without another file dialog.
        self.simulation_videos: dict[str, Path] = {}
        # The section (start and length in seconds, every n-th image) of that video per slot, so "same video as A" takes the same one.
        self.simulation_sections: dict[str, tuple[float, float, int]] = {}
        # Extraction progress for the progress dialog: video n of m, frames done/target, start time.
        self.detail: dict | None = None
        # Multi-step jobs: {"index": 2, "count": 5, "label": "..."} ("Step 2/5: ...").
        self.stage: dict | None = None
        # Summary of the last finished job (for example a dataset export), shown in its dialog.
        self.result: dict | None = None
        # AMD/Intel graphics: the last fallback to the CPU reported by ml_worker (RECO_GPU_FALLBACK)
        # and the summary of the last diagnosis report (see gpu_diagnosis.py).
        self.gpu_problem: dict | None = None
        self.gpu_diagnosis: dict | None = None
        self.gpu_auto_diagnosed = False
        self.gpu_diagnosing = False
        # Updated from the "language" field of every request body (see
        # do_POST()) - every status/error message this file itself generates
        # (as opposed to ml_worker.py subprocess output, which already gets an
        # explicit --language argument per call) reads this via localized()
        # rather than being threaded through as an explicit parameter.
        self.language = "de"

    def update(self, **values) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)
        notify_change()

    def append_log(self, line: str) -> None:
        with self.lock:
            clean = line.rstrip()
            if clean:
                self.log.append(clean)
                self.log = self.log[-250:]

    def _snapshot(self) -> dict:
        with self.lock:
            frames = (self.project or {}).get("frames", [])
            training_frames = [frame for frame in frames if frame.get("reviewStatus") != "candidate" and not frame.get("heldOut")]
            candidates = [frame for frame in frames if frame.get("reviewStatus") == "candidate"]
            annotations = [annotation for frame in training_frames for annotation in frame.get("annotations", [])]
            classes = sorted({str(annotation.get("category")) for annotation in annotations})
            return {
                "connected": True,
                "platform": platform.system(),
                "selectedFolder": str(self.selected_folder) if self.selected_folder else None,
                "folderName": self.selected_folder.name if self.selected_folder else None,
                "folderEmpty": self.folder_empty,
                "projectRoot": str(self.project_root) if self.project_root else None,
                "operation": self.operation,
                "progress": self.progress,
                "message": self.message,
                "busy": self.busy,
                "error": self.error,
                "sport": (self.project or {}).get("sport"),
                "framesPerVideo": (self.project or {}).get("framesPerVideo", DEFAULT_FRAMES_PER_VIDEO),
                # per video since 0.29; "fieldGeometry" only for a project without videos yet
                "fieldGeometries": field_geometry.geometries(self.project),
                "fieldGeometry": (self.project or {}).get("fieldGeometry"),
                "stats": {
                    "frameCount": len(training_frames),
                    "candidateCount": len(candidates),
                    "annotatedFrames": sum(bool(frame.get("annotations")) for frame in training_frames),
                    "unconfirmedEmpty": sum(1 for frame in training_frames if not frame.get("annotations") and frame.get("reviewStatus") != "reviewed" and not frame.get("emptyConfirmed")),
                    "annotationCount": len(annotations),
                    "automaticCount": sum(annotation.get("source") == "auto" for annotation in annotations),
                    "classes": classes,
                },
                "hardware": hardware_summary(),
                "gpu": gpu_status(),
                "rocm": rocm_status(),
                "lastTraining": (self.project or {}).get("lastTraining"),
                "trainingHistory": (self.project or {}).get("trainingHistory", []),
                "modelLibrary": model_library_snapshot(self.project_root),
                "modelRegistry": model_registry_snapshot(self.project_root),
                "hardwareTier": {"detected": detected_tier(), "plans": teachers.TIER_PLANS},
                "mlReady": bool(self.project_root and venv_python_path(ml_venv(self.project_root)).is_file()),
                "benchmark": benchmark_snapshot(self.project_root),
                "frames": frames,
                # The split the next training uses, so the image list can show each image's role.
                "splits": dataset_split.preview(frames, self.language) if self.project and ((self.project.get("augmentation") or {}).get("mode") not in {"mixed", "real-only"}) else None,
                "log": "\n".join(self.log[-80:]),
                # only for the project it was made in: its images live in that project's folder
                "simulation": self.simulation if (self.simulation or {}).get("root", str(self.project_root)) == str(self.project_root) else None,
                "simulationB": self.simulation_b if (self.simulation_b or {}).get("root", str(self.project_root)) == str(self.project_root) else None,
                "detail": self.detail,
                "stage": self.stage,
            }

    def snapshot(self) -> dict:
        data = self._snapshot()
        jobs = JOBS.snapshot()
        data["jobs"] = jobs
        data["hubUploadReady"] = bool(model_hub.HUB_CLIENT_ID)
        current = jobs["current"]
        data["cancellable"] = bool(current and current["cancellable"] and current["status"] == "running")
        return data


# Every change to STATE or the job list wakes up /api/events (server-sent events).
CHANGED = threading.Condition()
_CHANGE_COUNTER = 0


def notify_change() -> None:
    global _CHANGE_COUNTER
    with CHANGED:
        _CHANGE_COUNTER += 1
        CHANGED.notify_all()


STATE = LocalState()

# Cooperative cancel for the running job: /api/cancel (or /api/jobs/<id>/cancel) sets this
# event and stops every tracked subprocess with its whole process tree (ml_worker.py with
# its data loaders, pip, ffmpeg, the frame extractor). Each job then tidies up after itself.
CANCEL_REQUESTED = threading.Event()


class OperationCancelled(Exception):
    pass


class WorkerUserError(RuntimeError):
    """ml_worker stopped with a message for the user (exit code 3), not a crash."""


# The last such message, so record_job_failure does not treat it as a graphics card failure.
LAST_USER_ERROR: dict[str, str | None] = {"message": None}


def raise_if_cancelled() -> None:
    if CANCEL_REQUESTED.is_set():
        raise OperationCancelled()


ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_PROCESS_LOCK = threading.Lock()


def popen_tracked(command: list[str], **kwargs) -> subprocess.Popen:
    """Popen in its own process group/session so a cancel can stop the whole tree."""
    if os.name == "nt":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    with _PROCESS_LOCK:
        ACTIVE_PROCESSES.add(process)
    return process


def untrack(process: subprocess.Popen) -> None:
    with _PROCESS_LOCK:
        ACTIVE_PROCESSES.discard(process)


def terminate_process_tree(process: subprocess.Popen, grace: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        process.wait()


def terminate_active_processes() -> None:
    with _PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        terminate_process_tree(process)


def run_extraction_command(command: list[str], on_tick: Callable[[], None] | None = None) -> tuple[int, str]:
    """Run ffmpeg or the Apple extractor; stop it as soon as a cancel is requested."""
    process = popen_tracked(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    try:
        while True:
            try:
                _, stderr = process.communicate(timeout=0.4)
                if CANCEL_REQUESTED.is_set():
                    raise OperationCancelled()
                return process.returncode, stderr or ""
            except subprocess.TimeoutExpired:
                if CANCEL_REQUESTED.is_set():
                    terminate_process_tree(process, grace=3.0)
                    if process.stderr:
                        process.stderr.close()
                    raise OperationCancelled() from None
                if on_tick:
                    on_tick()
    finally:
        untrack(process)


def output_snapshot(directories: list[Path]) -> dict[Path, float]:
    """mtimes of the top-level entries of a job's output folders, taken when it starts."""
    entries: dict[Path, float] = {}
    for directory in directories:
        if directory.is_dir():
            for entry in directory.iterdir():
                try:
                    entries[entry] = entry.stat().st_mtime
                except OSError:
                    pass
    return entries


def remove_job_outputs(directories: list[Path], before: dict[Path, float]) -> None:
    """After a cancel: delete what the job created or rewrote, so no half-written export remains."""
    for directory in directories:
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            try:
                previous = before.get(entry)
                if previous is not None and entry.stat().st_mtime <= previous:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
            except OSError:
                pass


class JobManager:
    """One job runs at a time; further jobs wait in a queue (two GPU jobs would only
    push each other out of memory). Each job is a function that reports through STATE
    (progress, message, detail, stage) and ends by setting STATE.operation to
    "ready", "error" or "cancelled" - the job's final status is derived from that."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.current: dict | None = None
        self.queue: list[tuple[dict, Callable, tuple, dict]] = []
        self.finished: list[dict] = []

    def submit(self, kind: str, target: Callable, *args, cancellable: bool = True, **kwargs) -> dict:
        job = {
            "id": uuid.uuid4().hex[:12], "type": kind, "status": "queued", "cancellable": cancellable,
            "createdAt": time.time(), "startedAt": None, "finishedAt": None, "error": None, "result": None,
        }
        with self.lock:
            self.queue.append((job, target, args, kwargs))
        notify_change()
        self._start_next()
        return dict(job)

    def _start_next(self) -> None:
        with self.lock:
            if self.current is not None or not self.queue:
                return
            job, target, args, kwargs = self.queue.pop(0)
            job.update(status="running", startedAt=time.time())
            self.current = job
        CANCEL_REQUESTED.clear()
        LAST_USER_ERROR["message"] = None
        STATE.update(busy=True, error=None, stage=None, detail=None, progress=0.0, result=None)
        threading.Thread(target=self._run, args=(job, target, args, kwargs), daemon=True).start()

    def _run(self, job: dict, target: Callable, args: tuple, kwargs: dict) -> None:
        try:
            target(*args, **kwargs)
        except OperationCancelled:
            STATE.update(operation="cancelled", busy=False)
        except Exception as error:  # a job function should handle its own errors; this is the safety net
            STATE.append_log(str(error))
            STATE.update(operation="error", busy=False, error=str(error))
        finally:
            with STATE.lock:
                operation, error, message, result = STATE.operation, STATE.error, STATE.message, getattr(STATE, "result", None)
            status = "cancelled" if operation == "cancelled" else "failed" if operation == "error" else "done"
            with self.lock:
                job.update(status=status, finishedAt=time.time(), error=error if status == "failed" else None, message=message, result=result)
                self.current = None
                self.finished.insert(0, job)
                del self.finished[10:]
            CANCEL_REQUESTED.clear()
            STATE.update(busy=False, stage=None, detail=None)
            if status == "failed":
                record_job_failure(str(job.get("type") or ""), error)
            self._start_next()

    def has_pending(self) -> bool:
        with self.lock:
            return self.current is not None or bool(self.queue)

    def cancel(self, job_id: str | None = None) -> dict:
        with self.lock:
            for index, (queued, *_rest) in enumerate(self.queue):
                if queued["id"] == job_id:
                    del self.queue[index]
                    queued.update(status="cancelled", finishedAt=time.time())
                    self.finished.insert(0, queued)
                    notify_change()
                    return dict(queued)
            current = self.current
            if current is None or (job_id is not None and current["id"] != job_id):
                raise RuntimeError(localized("Dieser Vorgang läuft nicht mehr.", "This operation is no longer running.", "Esta operación ya no está en curso.", "Cette opération n’est plus en cours."))
            if not current["cancellable"]:
                raise RuntimeError(localized("Dieser Vorgang lässt sich nicht abbrechen.", "This operation cannot be cancelled.", "Esta operación no se puede cancelar.", "Cette opération ne peut pas être annulée."))
            current["status"] = "cancelling"
        CANCEL_REQUESTED.set()
        STATE.update(message=localized("Wird abgebrochen …", "Cancelling …", "Cancelando …", "Annulation …"))
        threading.Thread(target=terminate_active_processes, daemon=True).start()
        return dict(current)

    def snapshot(self) -> dict:
        with self.lock:
            current = dict(self.current) if self.current else None
            queue = [dict(job) for job, *_rest in self.queue]
            recent = [dict(job) for job in self.finished[:5]]
        if current:
            with STATE.lock:
                progress = float(STATE.progress or 0.0)
                current.update(message=STATE.message, detail=STATE.detail, stage=STATE.stage, operation=STATE.operation)
            # 0 means "no measurable progress yet" (pip, model loading): the dialog shows an indeterminate bar.
            current["progress"] = progress if progress > 0 else None
            elapsed = time.time() - (current["startedAt"] or time.time())
            current["elapsedSeconds"] = elapsed
            current["etaSeconds"] = elapsed * (1 - progress) / progress if 0.03 < progress < 1 else None
        return {"current": current, "queue": queue, "recent": recent}


JOBS = JobManager()
# Finish in a moment; not worth a Cancel button.
QUICK_ACTIONS = {"activate-model", "rename-model", "delete-model"}


def project_root_for(folder: Path) -> Path:
    root = folder / ".reco-training"
    # Dot-prefixed folders are hidden on macOS and Linux. Keep the stable
    # internal path, but expose the same project permanently under a visible
    # name. Windows Explorer already shows dot-prefixed directories and often
    # requires elevated privileges for symlinks, so it is intentionally skipped.
    if os.name != "nt":
        visible = folder / "Reco Training"
        if not os.path.lexists(visible):
            try:
                visible.symlink_to(Path(".reco-training"), target_is_directory=True)
            except OSError:
                pass
    return root


def start_empty_project(folder: Path, root: Path, sport: str) -> None:
    """A project without frames, as "Prepare working folder" creates it for a folder of images."""
    if sport not in SPORT_CATEGORIES:
        raise RuntimeError(localized("Unbekannte Sportart.", "Unknown sport.", "Deporte desconocido.", "Sport inconnu."))
    now = utc_now()
    for directory in ("frames", "dataset", "runs", "exports"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    atomic_json(root / "project.json", {
        "schemaVersion": 2, "name": folder.name, "sourceFolder": str(folder), "createdAt": now, "updatedAt": now,
        "framesPerVideo": DEFAULT_FRAMES_PER_VIDEO, "frames": [], "sport": sport,
    })
    STATE.update(project=load_project(root))


def load_project(root: Path) -> dict | None:
    path = root / "project.json"
    if not path.is_file():
        return None
    project = json.loads(path.read_text(encoding="utf-8"))
    if field_geometry.needs_migration(project):
        # 0.29: one field outline per video. The old single outline is copied to every video
        # (and kept as fieldGeometryBeforePerVideo); a backup of project.json comes first.
        backup_project(root, "feldgrenze-pro-video")
        field_geometry.migrate(project)
        atomic_json(path, project)
    return project


def model_library_snapshot(root: Path | None) -> dict:
    if root is None:
        return {"packages": [], "activePackageID": None}
    packages = []
    library = root / "models" / "library"
    if library.is_dir():
        for manifest_path in sorted(library.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                packages.append({
                    "packageID": manifest.get("packageID"),
                    "displayName": manifest.get("displayName") or manifest.get("packageID"),
                    "createdAt": manifest.get("createdAt"),
                    "sport": manifest.get("sport"),
                    "modelSize": manifest.get("modelSize"),
                    "classes": manifest.get("classes", []),
                    "description": manifest.get("description"),
                    "source": manifest.get("source"),
                    "statistics": manifest.get("statistics", {}),
                    "validationMetrics": manifest.get("validationMetrics") or (manifest.get("trainingSummary") or {}).get("validationMetrics", {}),
                    "testMetrics": manifest.get("testMetrics") or (manifest.get("trainingSummary") or {}).get("testMetrics", {}),
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    active_id = None
    active_path = root / "models" / "active.json"
    if active_path.is_file():
        try:
            active_id = json.loads(active_path.read_text(encoding="utf-8")).get("packageID")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    def score_metrics(metrics: dict) -> float | None:
        for key, value in metrics.items():
            normalized = str(key).lower().replace("val/", "").replace("val_", "")
            normalized = normalized.replace("test/", "").replace("test_", "")
            if normalized in {"map_50_95", "map50_95", "ap/ball", "ap_ball"}:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None
    def score(item: dict) -> float | None:
        test_score = score_metrics(item.get("testMetrics") or {})
        return test_score if test_score is not None else score_metrics(item.get("validationMetrics") or {})
    scored = [(score(item), item.get("packageID")) for item in packages]
    scored = [item for item in scored if item[0] is not None]
    best_id = max(scored, default=(None, None), key=lambda item: item[0])[1]
    for item in packages:
        item["isActive"] = item.get("packageID") == active_id
        item["isBest"] = item.get("packageID") == best_id
        item["validationScore"] = score_metrics(item.get("validationMetrics") or {})
        item["testScore"] = score_metrics(item.get("testMetrics") or {})
    packages.sort(key=lambda item: (not item["isActive"], not item["isBest"], str(item.get("createdAt") or "")))
    return {"packages": packages, "activePackageID": active_id, "bestPackageID": best_id}


def benchmark_snapshot(root: Path | None) -> dict:
    if root is None:
        return {"groundTruth": None, "latest": None}
    benchmark_root = root / "benchmarks"
    ground_truth = None
    latest = None
    try:
        document = json.loads((benchmark_root / "ground-truth.json").read_text(encoding="utf-8"))
        ground_truth = {
            "createdAt": document.get("createdAt"),
            "datasetID": document.get("datasetID"),
            "sport": document.get("sport"),
            "frameCount": len(document.get("frames", [])),
            "annotationCount": sum(len(frame.get("annotations", [])) for frame in document.get("frames", [])),
            "classes": sorted({annotation.get("category") for frame in document.get("frames", []) for annotation in frame.get("annotations", []) if annotation.get("category")}),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    try:
        latest = json.loads((benchmark_root / "latest.json").read_text(encoding="utf-8"))
        if isinstance(latest, dict):
            latest["matchesGroundTruth"] = bool(ground_truth and latest.get("datasetID") == ground_truth.get("datasetID"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        latest = None
    return {"groundTruth": ground_truth, "latest": latest}


def freeze_ground_truth() -> dict:
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError(localized("Kein lokales Projekt geöffnet.", "No local project is open.", "No hay ningún proyecto local abierto.", "Aucun projet local n’est ouvert."))
    frames = [frame for frame in project.get("frames", []) if frame.get("reviewStatus") != "candidate" and frame.get("heldOut")]
    if not frames:
        raise RuntimeError(localized(
            "Das Projekt enthält keine unabhängigen Testbilder. Zuerst unter „Unabhängiger Modelltest“ einen Videoordner prüfen, der nicht zum Training verwendet wurde.",
            "The project contains no independent test images yet. First review a video folder that was never used for training under “Independent model test”.",
            "El proyecto todavía no tiene imágenes de prueba independientes. Primero revisa una carpeta de vídeos nunca usada para entrenar en “Prueba de modelo independiente”.",
            "Le projet ne contient pas encore d’images de test indépendantes. Vérifiez d’abord un dossier vidéo jamais utilisé pour l’entraînement sous « Test de modèle indépendant ».",
        ))
    pending = sum(annotation.get("source") == "auto" for frame in frames for annotation in frame.get("annotations", []))
    if pending:
        raise RuntimeError(localized(
            "Vor dem Modelltest alle automatischen Vorschläge übernehmen, korrigieren oder verwerfen.",
            "Accept, correct, or reject every automatic suggestion before the model test.",
            "Antes de la prueba del modelo, acepta, corrige o rechaza cada sugerencia automática.",
            "Avant le test du modèle, acceptez, corrigez ou rejetez chaque suggestion automatique.",
        ))
    allowed = set(SPORT_CATEGORIES.get(project.get("sport"), []))
    reference_frames = []
    for frame in frames:
        annotations = []
        for annotation in frame.get("annotations", []):
            category = str(annotation.get("category", ""))
            if category not in allowed:
                continue
            annotations.append({
                "category": category,
                "x": max(0.0, float(annotation.get("x", 0))),
                "y": max(0.0, float(annotation.get("y", 0))),
                "width": max(1.0, float(annotation.get("width", 1))),
                "height": max(1.0, float(annotation.get("height", 1))),
            })
        reference_frames.append({
            "id": str(frame.get("id")),
            "relativePath": str(frame.get("relativePath")),
            "width": int(frame.get("width", 0)),
            "height": int(frame.get("height", 0)),
            "annotations": annotations,
        })
    annotation_count = sum(len(frame["annotations"]) for frame in reference_frames)
    if annotation_count == 0:
        raise RuntimeError(localized(
            "Mindestens eine richtige Box muss vor dem Modelltest festgelegt sein.", "At least one correct box must be set before the model test.",
            "Debe haber al menos un cuadro correcto antes de la prueba del modelo.", "Au moins une boîte correcte doit être définie avant le test du modèle.",
        ))
    canonical = json.dumps({"sport": project.get("sport"), "frames": reference_frames}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    reference = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "sport": project.get("sport"),
        "datasetID": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "frames": reference_frames,
        "reviewStatement": "Every frame was explicitly frozen as ground truth; frames without boxes are intentional negatives.",
    }
    atomic_json(root / "benchmarks" / "ground-truth.json", reference)
    STATE.update(message=localized(
        f"Referenz mit {len(reference_frames)} Bildern und {annotation_count} Boxen festgelegt.", f"Reference frozen with {len(reference_frames)} images and {annotation_count} boxes.",
        f"Referencia fijada con {len(reference_frames)} imágenes y {annotation_count} cuadros.", f"Référence figée avec {len(reference_frames)} images et {annotation_count} boîtes.",
    ))
    return benchmark_snapshot(root)


def choose_video_folder(*, expansion: bool = False, labels: bool = False) -> Path:
    """Native folder picker. labels=True picks a folder of images with YOLO-txt
    or COCO labels for "Import labels" instead of a video folder."""
    if labels:
        configured_folder = os.environ.get("RECO_LABEL_FOLDER")
    elif expansion:
        configured_folder = os.environ.get("RECO_EXPANSION_FOLDER") or os.environ.get("RECO_TEST_EXPANSION_FOLDER")
    else:
        configured_folder = os.environ.get("RECO_VIDEO_FOLDER") or os.environ.get("RECO_TEST_FOLDER")
    dialog_title = "Select folder with images and YOLO or COCO labels" if labels else "Select folder with new sports videos" if expansion else "Select working folder"
    prompt = localized(
        "Ordner mit Bildern und YOLO- oder COCO-Labels auswählen",
        "Select a folder with images and YOLO or COCO labels",
        "Selecciona una carpeta con imágenes y etiquetas YOLO o COCO",
        "Sélectionnez un dossier avec des images et des annotations YOLO ou COCO",
    ) if labels else localized(
        "Ordner mit neuen, noch nicht verwendeten Sportvideos auswählen" if expansion else "Arbeitsordner mit Videos, Bildern oder Datensätzen auswählen",
        "Select a folder with new, not yet used sports videos" if expansion else "Select a working folder with videos, images or datasets",
        "Selecciona una carpeta con vídeos deportivos nuevos aún no usados" if expansion else "Selecciona una carpeta con vídeos deportivos",
        "Sélectionnez un dossier contenant de nouvelles vidéos sportives pas encore utilisées" if expansion else "Sélectionnez un dossier contenant des vidéos sportives",
    )
    if configured_folder:
        selected = Path(configured_folder).expanduser().resolve()
    elif sys.platform == "darwin":
        script = f'POSIX path of (choose folder with prompt "{prompt}")'
        result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(localized("Ordnerauswahl wurde abgebrochen.", "Folder selection was cancelled.", "Se canceló la selección de carpeta.", "La sélection du dossier a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    elif sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            raise RuntimeError(localized("PowerShell wurde für die native Ordnerauswahl nicht gefunden.", "PowerShell was not found for the native folder picker.", "No se encontró PowerShell para el selector de carpetas nativo.", "PowerShell est introuvable pour le sélecteur de dossier natif."))
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$dialog.Description = '{dialog_title}'; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.SelectedPath }"
        )
        result = subprocess.run([powershell, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(localized("Ordnerauswahl wurde abgebrochen.", "Folder selection was cancelled.", "Se canceló la selección de carpeta.", "La sélection du dossier a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    else:
        dialog = shutil.which("zenity")
        command = [dialog, "--file-selection", "--directory", f"--title={dialog_title}"] if dialog else None
        if command is None and shutil.which("kdialog"):
            command = [shutil.which("kdialog"), "--getexistingdirectory", str(Path.home())]
        if command is None:
            raise RuntimeError(localized("Für die Ordnerauswahl bitte Zenity oder KDialog installieren oder RECO_VIDEO_FOLDER setzen.", "Please install Zenity or KDialog for folder selection, or set RECO_VIDEO_FOLDER.", "Instala Zenity o KDialog para seleccionar carpetas, o configura RECO_VIDEO_FOLDER.", "Veuillez installer Zenity ou KDialog pour la sélection de dossier, ou définir RECO_VIDEO_FOLDER."))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(localized("Ordnerauswahl wurde abgebrochen.", "Folder selection was cancelled.", "Se canceló la selección de carpeta.", "La sélection du dossier a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    if not selected.is_dir():
        raise RuntimeError(localized("Der ausgewählte Ordner ist nicht verfügbar.", "The selected folder is not available.", "La carpeta seleccionada no está disponible.", "Le dossier sélectionné n’est pas disponible."))
    return selected


def select_folder() -> Path:
    selected = choose_video_folder()
    root = project_root_for(selected)
    project = load_project(root)
    STATE.update(
        selected_folder=selected,
        folder_empty=folder_is_empty(selected),
        project_root=root,
        project=project,
        operation="selected",
        progress=0.0,
        message=(
            localized(
                f"Vorhandenes Projekt mit {len(project.get('frames', []))} Frames geöffnet.",
                f"Existing project with {len(project.get('frames', []))} frames opened.",
                f"Proyecto existente abierto con {len(project.get('frames', []))} fotogramas.",
                f"Projet existant ouvert avec {len(project.get('frames', []))} images.",
            ) if project else
            localized(
                "Arbeitsordner gewählt. Er kann jetzt lokal vorbereitet werden.",
                "Working folder selected. It can now be prepared locally.",
                "Carpeta de trabajo seleccionada. Ya puedes prepararla localmente.",
                "Dossier de travail sélectionné. Il peut maintenant être préparé localement.",
            )
        ),
        error=None,
    )
    return selected


def select_model_package(suffix: str = ".recomodel") -> Path:
    """Native file picker for a .recomodel package, or with suffix=".onnx" for a
    YOLO ONNX export that "Import ONNX model" wraps into a package."""
    onnx = suffix == ".onnx"
    configured = os.environ.get("RECO_ONNX_MODEL") if onnx else (os.environ.get("RECO_MODEL_PACKAGE") or os.environ.get("RECO_TEST_MODEL_PACKAGE"))
    if configured:
        selected = Path(configured).expanduser().resolve()
    elif sys.platform == "darwin":
        prompt = localized("ONNX-Modell auswählen", "Select ONNX model", "Selecciona un modelo ONNX", "Sélectionnez un modèle ONNX") if onnx else localized("Reco-Modellpaket auswählen", "Select Reco model package", "Selecciona un paquete de modelo de Reco", "Sélectionnez un paquet de modèle Reco")
        script = f'POSIX path of (choose file with prompt "{prompt}")'
        result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(localized("Modellauswahl wurde abgebrochen.", "Model selection was cancelled.", "Se canceló la selección del modelo.", "La sélection du modèle a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    elif sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            raise RuntimeError(localized("PowerShell wurde für die Modellauswahl nicht gefunden.", "PowerShell was not found for model selection.", "No se encontró PowerShell para seleccionar el modelo.", "PowerShell est introuvable pour la sélection du modèle."))
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
            + ("$dialog.Filter = 'ONNX model (*.onnx)|*.onnx'; " if onnx else "$dialog.Filter = 'Reco model package (*.recomodel)|*.recomodel'; ")
            + "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileName }"
        )
        result = subprocess.run([powershell, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(localized("Modellauswahl wurde abgebrochen.", "Model selection was cancelled.", "Se canceló la selección del modelo.", "La sélection du modèle a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    else:
        dialog = shutil.which("zenity")
        command = [dialog, "--file-selection", "--title=Select ONNX model" if onnx else "--title=Select Reco model package", f"--file-filter={'ONNX models' if onnx else 'Reco models'} | *{suffix}"] if dialog else None
        if command is None and shutil.which("kdialog"):
            command = [shutil.which("kdialog"), "--getopenfilename", str(Path.home()), f"*{suffix}"]
        if command is None:
            inboxes = []
            if STATE.selected_folder:
                inboxes.extend(STATE.selected_folder.glob(f"*{suffix}"))
            if STATE.project_root:
                inboxes.extend((STATE.project_root / "inbox").glob(f"*{suffix}"))
            candidates = sorted({path.resolve() for path in inboxes if path.is_file()})
            if len(candidates) == 1:
                return candidates[0]
            raise RuntimeError(localized(
                "Für die Modellauswahl bitte genau ein .recomodel-Paket in den Arbeitsordner oder .reco-training/inbox legen, Zenity/KDialog installieren oder RECO_MODEL_PACKAGE setzen.",
                "For model selection, place exactly one .recomodel package in the working folder or .reco-training/inbox, install Zenity/KDialog, or set RECO_MODEL_PACKAGE.",
                "Para seleccionar el modelo, coloca exactamente un paquete .recomodel en la carpeta de trabajo o en .reco-training/inbox, instala Zenity/KDialog, o configura RECO_MODEL_PACKAGE.",
                "Pour la sélection du modèle, placez exactement un paquet .recomodel dans le dossier de travail ou .reco-training/inbox, installez Zenity/KDialog, ou définissez RECO_MODEL_PACKAGE.",
            ))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(localized("Modellauswahl wurde abgebrochen.", "Model selection was cancelled.", "Se canceló la selección del modelo.", "La sélection du modèle a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    if not selected.is_file() or selected.suffix.lower() != suffix:
        raise RuntimeError(localized(f"Bitte eine vorhandene {suffix}-Datei auswählen.", f"Please select an existing {suffix} file.", f"Selecciona un archivo {suffix} existente.", f"Veuillez sélectionner un fichier {suffix} existant."))
    return selected


def select_video_file() -> Path:
    configured = os.environ.get("RECO_SIMULATION_VIDEO") or os.environ.get("RECO_TEST_SIMULATION_VIDEO")
    if configured:
        selected = Path(configured).expanduser().resolve()
    elif sys.platform == "darwin":
        prompt = localized(
            "Video für die Balltracking-Simulation auswählen", "Select a video for the ball-tracking simulation",
            "Selecciona un vídeo para la simulación de seguimiento del balón", "Sélectionnez une vidéo pour la simulation de suivi du ballon",
        )
        script = f'POSIX path of (choose file with prompt "{prompt}")'
        result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(localized("Videoauswahl wurde abgebrochen.", "Video selection was cancelled.", "Se canceló la selección del vídeo.", "La sélection de la vidéo a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    elif sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            raise RuntimeError(localized("PowerShell wurde für die Videoauswahl nicht gefunden.", "PowerShell was not found for video selection.", "No se encontró PowerShell para seleccionar el vídeo.", "PowerShell est introuvable pour la sélection de la vidéo."))
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
            "$dialog.Filter = 'Videos (*.mp4;*.mov;*.m4v)|*.mp4;*.mov;*.m4v'; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileName }"
        )
        result = subprocess.run([powershell, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(localized("Videoauswahl wurde abgebrochen.", "Video selection was cancelled.", "Se canceló la selección del vídeo.", "La sélection de la vidéo a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    else:
        dialog = shutil.which("zenity")
        command = [dialog, "--file-selection", "--title=Select a short clip for the ball-tracking simulation", "--file-filter=Videos | *.mp4 *.mov *.m4v"] if dialog else None
        if command is None and shutil.which("kdialog"):
            command = [shutil.which("kdialog"), "--getopenfilename", str(Path.home()), "*.mp4 *.mov *.m4v"]
        if command is None:
            raise RuntimeError(localized("Für die Videoauswahl bitte Zenity/KDialog installieren oder RECO_SIMULATION_VIDEO setzen.", "Please install Zenity/KDialog for video selection, or set RECO_SIMULATION_VIDEO.", "Instala Zenity/KDialog para seleccionar el vídeo, o configura RECO_SIMULATION_VIDEO.", "Veuillez installer Zenity/KDialog pour la sélection de la vidéo, ou définir RECO_SIMULATION_VIDEO."))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(localized("Videoauswahl wurde abgebrochen.", "Video selection was cancelled.", "Se canceló la selección del vídeo.", "La sélection de la vidéo a été annulée."))
        selected = Path(result.stdout.strip()).resolve()
    if not selected.is_file() or selected.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError(localized("Bitte eine vorhandene Videodatei auswählen.", "Please select an existing video file.", "Selecciona un archivo de vídeo existente.", "Veuillez sélectionner un fichier vidéo existant."))
    return selected


def discover_videos(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and ".reco-training" not in path.parts),
        key=lambda path: str(path).casefold(),
    )


def folder_has_images(folder: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") or part == "Reco Training" for part in path.relative_to(folder).parts)
        for path in folder.rglob("*")
    )


def folder_is_empty(folder: Path) -> bool:
    return not discover_videos(folder) and not folder_has_images(folder)


def empty_folder_message() -> str:
    return localized(
        "Der Arbeitsordner ist noch leer. Lege Videos, Bilder oder Datensätze hinein und wähle dann „Arbeitsordner vorbereiten“.",
        "The working folder is still empty. Add videos, images or datasets to it, then choose “Prepare working folder”.",
        "La carpeta de trabajo aún está vacía. Añade vídeos, imágenes o conjuntos de datos y elige «Preparar carpeta de trabajo».",
        "Le dossier de travail est encore vide. Ajoutez-y des vidéos, des images ou des jeux de données, puis choisissez « Préparer le dossier de travail ».",
    )


def import_folder_images(folder: Path, root: Path) -> dict:
    """Images in the working folder become frames; labels lying next to them (a YOLO or
    COCO dataset) come along. Runs after project.json is written, so a cancel leaves a valid project."""
    def report(done: int, total: int) -> None:
        STATE.update(progress=min(done / max(total, 1), 0.99))

    STATE.update(operation="extracting", progress=0.0, detail=None, message=localized("Übernehme Bilder aus dem Arbeitsordner …", "Adding images from the working folder …", "Añadiendo imágenes de la carpeta de trabajo …", "Ajout des images du dossier de travail …"))
    # Standard library only - the system Python is enough, no ML environment needed.
    return run_json([
        system_python(), str(find_ml_worker()), "import-labels", "--project", str(root), "--source", str(folder),
        "--format", "auto", "--include-unlabelled", "--language", STATE.language,
    ], on_progress=report)


def working_folder_images_message(summary: dict) -> str:
    images, unlabelled = int(summary.get("imported", 0)), int(summary.get("unlabelled", 0))
    labelled, boxes = images - unlabelled, int(summary.get("annotations", 0))
    parts = [localized(
        f"{images} Bilder übernommen: {labelled} mit Labels ({boxes} Boxen), {unlabelled} zum Beschriften.",
        f"{images} images added: {labelled} with labels ({boxes} boxes), {unlabelled} to label.",
        f"{images} imágenes añadidas: {labelled} con etiquetas ({boxes} cuadros), {unlabelled} por etiquetar.",
        f"{images} images ajoutées : {labelled} avec annotations ({boxes} boîtes), {unlabelled} à annoter.",
    )]
    skipped = summary.get("skippedClasses") or {}
    if skipped:
        classes = ", ".join(f"{name} ({count})" for name, count in sorted(skipped.items()))
        parts.append(localized(f"Nicht übernommene Klassen: {classes}.", f"Classes not imported: {classes}.", f"Clases no importadas: {classes}.", f"Classes non importées : {classes}."))
    return " ".join(parts)


def media_info(path: Path, apple_extractor: Path | None = None) -> tuple[float, int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        if apple_extractor is None:
            raise RuntimeError(localized("Video-Metadaten können lokal nicht gelesen werden.", "Video metadata could not be read locally.", "No se pudieron leer los metadatos del vídeo localmente.", "Les métadonnées vidéo n’ont pas pu être lues localement."))
        result = subprocess.run([str(apple_extractor), "--probe", str(path)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(localized(f"Video konnte nicht gelesen werden: {path.name}", f"Video could not be read: {path.name}", f"No se pudo leer el vídeo: {path.name}", f"La vidéo n’a pas pu être lue : {path.name}") + f"\n{result.stderr.strip()}")
        payload = json.loads(result.stdout)
        return float(payload["duration"]), int(payload["width"]), int(payload["height"])
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(localized(f"Video konnte nicht gelesen werden: {path.name}", f"Video could not be read: {path.name}", f"No se pudo leer el vídeo: {path.name}", f"La vidéo n’a pas pu être lue : {path.name}") + f"\n{result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if not streams or duration <= 0:
        raise RuntimeError(localized(f"Video enthält keine lesbare Bildspur: {path.name}", f"Video has no readable video track: {path.name}", f"El vídeo no tiene una pista de vídeo legible: {path.name}", f"La vidéo ne contient aucune piste vidéo lisible : {path.name}"))
    return duration, int(streams[0].get("width") or 0), int(streams[0].get("height") or 0)


def native_frame_extractor(root: Path) -> Path:
    # The Mac app ships this extractor precompiled (no Xcode tools needed on the user's Mac).
    bundled = os.environ.get("RECO_FRAME_EXTRACTOR")
    if bundled and Path(bundled).is_file() and os.access(bundled, os.X_OK):
        return Path(bundled)
    source = BASE_DIR / "frame_extractor.swift"
    if sys.platform != "darwin" or not source.is_file():
        raise RuntimeError(localized("FFmpeg wurde nicht gefunden.", "FFmpeg was not found.", "No se encontró FFmpeg.", "FFmpeg est introuvable."))
    compiler = shutil.which("swiftc") or "/usr/bin/swiftc"
    if not Path(compiler).exists():
        raise RuntimeError(localized("Weder FFmpeg noch die Xcode Command Line Tools wurden gefunden.", "Neither FFmpeg nor the Xcode Command Line Tools were found.", "No se encontraron ni FFmpeg ni las Herramientas de línea de comandos de Xcode.", "Ni FFmpeg ni les outils en ligne de commande Xcode n’ont été trouvés."))
    binary = root / ".runtime" / "bin" / "reco-frame-extractor"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        module_cache = root / ".runtime" / "swift-module-cache"
        module_cache.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([compiler, "-module-cache-path", str(module_cache), str(source), "-o", str(binary)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(localized("Lokaler Apple-Frame-Extraktor konnte nicht erstellt werden.", "The local Apple frame extractor could not be built.", "No se pudo compilar el extractor de fotogramas local de Apple.", "L’extracteur d’images Apple local n’a pas pu être compilé.") + f"\n{result.stderr.strip()}")
    return binary


def frames_per_video(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = DEFAULT_FRAMES_PER_VIDEO
    return min(MAX_FRAMES_PER_VIDEO, max(MIN_FRAMES_PER_VIDEO, count))


def extract_frames_for_video(
    video: Path,
    duration: float,
    width: int,
    height: int,
    target_count: int,
    output_dir: Path,
    prefix: str,
    ffmpeg: str | None,
    apple_extractor: Path | None,
    failure_message: str,
    on_frames: Callable[[int], None] | None = None,
    section: tuple[float, float, float] | None = None,
) -> tuple[list[Path], float, int, int]:
    """Extract up to target_count evenly spaced frames from one video.

    With section=(start, length, rate) it instead takes target_count frames at a fixed rate
    (frames per second) from start to start + length seconds, as the tracking simulation needs.

    Shared by initial project preparation and active-learning dataset expansion,
    which previously built the same ffmpeg/Apple-extractor command and the same
    even-height output scaling twice with only cosmetic differences.

    Returns the generated frame paths (sorted), the sampling rate actually used
    (frames per second of source video, needed by callers to compute each
    frame's timestamp), and the output width/height. Frames keep the video's
    native resolution (no downscaling): small objects such as players in the far
    corner of a full-field panorama need every pixel.
    """
    pattern = output_dir / f"{prefix}-%06d.jpg"
    if section:
        start, length, rate = section
        command = (
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(video),
             "-vf", f"fps={rate:.8f}",
             "-frames:v", str(target_count), "-q:v", "2", str(pattern)]
            if ffmpeg else
            [str(apple_extractor), str(video), str(output_dir), prefix, str(target_count), f"{start:.3f}", f"{1 / rate:.6f}"]
        )
    else:
        rate = max(target_count / duration, 1 / max(duration, 1.0))
        command = (
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
             "-vf", f"fps={rate:.8f}",
             "-frames:v", str(target_count), "-q:v", "2", str(pattern)]
            if ffmpeg else
            [str(apple_extractor), str(video), str(output_dir), prefix, str(target_count)]
        )
    # Both extractors write the frames one by one, so counting them gives the progress.
    tick = (lambda: on_frames(sum(1 for _ in output_dir.glob(f"{prefix}-*.jpg")))) if on_frames else None
    returncode, stderr = run_extraction_command(command, tick)
    if returncode != 0:
        raise RuntimeError(f"{failure_message}: {video.name}\n{stderr.strip()}")
    generated = sorted(output_dir.glob(f"{prefix}-*.jpg"))
    return generated, rate, max(1, width), max(1, height)


def remove_extracted_frames(frames_dir: Path, prefixes: list[str]) -> None:
    for prefix in prefixes:
        for path in frames_dir.glob(f"{prefix}-*.jpg"):
            path.unlink(missing_ok=True)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def carry_over_frames(root: Path, existing_project: dict | None, frames_dir: Path, sport: str, rebuilt_video_ids: set[str]) -> list[dict]:
    """Keep every earlier frame that this run does not extract again: imported images, frames
    added with "Extend dataset" or the independent test, and frames of videos that are no longer
    in the working folder. Once extracted, the images and their boxes no longer need the video
    (a tester moved the videos to an external drive and lost a few hundred labelled images)."""
    frames_root = (root / "frames").resolve()
    kept: list[dict] = []
    for frame in (existing_project or {}).get("frames", []):
        if str(frame.get("videoID", "")) in rebuilt_video_ids:
            continue
        source = (root / str(frame.get("relativePath", ""))).resolve()
        if frames_root not in source.parents or not source.is_file():
            continue
        link_or_copy(source, frames_dir / source.relative_to(frames_root))
        kept.append({
            **frame,
            "annotations": [item for item in frame.get("annotations", []) if item.get("category") in SPORT_CATEGORIES[sport]],
        })
    return kept


def reusable_video_frames(root: Path, existing_project: dict | None, video_id: str, count_per_video: int) -> list[dict] | None:
    """The frames of a video prepared before with the same number of images, if all are still on
    disk - so refreshing extracts only new videos and keeps everything else as it is."""
    if not existing_project or int(existing_project.get("framesPerVideo") or 0) != count_per_video:
        return None
    frames = [frame for frame in existing_project.get("frames", []) if str(frame.get("videoID")) == video_id and not frame.get("heldOut") and frame.get("reviewStatus") != "candidate"]
    if not frames or not all((root / str(frame.get("relativePath", ""))).is_file() for frame in frames):
        return None
    return frames


def extract_project(sport: str, requested_frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO) -> None:
    try:
        folder = STATE.selected_folder
        root = STATE.project_root
        if folder is None or root is None:
            raise RuntimeError(localized("Zuerst einen Arbeitsordner auswählen.", "Select a working folder first.", "Primero selecciona una carpeta de trabajo.", "Sélectionnez d’abord un dossier de travail."))
        if sport not in SPORT_CATEGORIES:
            raise RuntimeError(localized("Unbekannte Sportart.", "Unknown sport.", "Deporte desconocido.", "Sport inconnu."))
        count_per_video = frames_per_video(requested_frames_per_video)
        existing_project = load_project(root)
        existing_annotations = {
            str(frame.get("id")): [
                annotation for annotation in frame.get("annotations", [])
                if annotation.get("category") in SPORT_CATEGORIES[sport]
            ]
            for frame in (existing_project or {}).get("frames", [])
        }
        backup_project(root, "vorbereitung")
        ffmpeg = shutil.which("ffmpeg")
        apple_extractor = native_frame_extractor(root) if not ffmpeg or not shutil.which("ffprobe") else None
        STATE.update(operation="scanning", busy=True, progress=0.01, detail=None, message=localized("Suche lokale Videos …", "Searching for local videos …", "Buscando vídeos locales …", "Recherche de vidéos locales …"), error=None, log=[])
        videos = discover_videos(folder)
        has_images = folder_has_images(folder)
        if not videos and not has_images:
            STATE.update(folder_empty=True)
            raise RuntimeError(empty_folder_message())
        if not videos:
            # Only images: frames of videos that were prepared earlier stay, nothing is rebuilt.
            now = utc_now()
            project = {
                **(existing_project or {"schemaVersion": 2, "name": folder.name, "sourceFolder": str(folder), "createdAt": now, "framesPerVideo": count_per_video, "frames": []}),
                "sport": sport, "updatedAt": now,
            }
            project["frames"] = [
                {**frame, "annotations": [item for item in frame.get("annotations", []) if item.get("category") in SPORT_CATEGORIES[sport]]}
                for frame in project["frames"]
            ]
            for directory in ("frames", "dataset", "runs", "exports"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            atomic_json(root / "project.json", project)
            summary = import_folder_images(folder, root)
            STATE.update(project=load_project(root), operation="ready", busy=False, progress=1.0, detail=None, message=working_folder_images_message(summary))
            return
        # Reading each video's metadata (ffprobe) is a separate subprocess call
        # per file, done before any extraction starts - with many files (e.g.
        # a GoPro game split into a dozen+ chapter files), this alone can take
        # a while with no visible movement otherwise, looking like a hang.
        infos = []
        started = time.time()
        for index, video in enumerate(videos, start=1):
            raise_if_cancelled()
            STATE.update(
                message=localized(f"Prüfe Video {index}/{len(videos)}: {video.name}", f"Checking video {index}/{len(videos)}: {video.name}", f"Comprobando vídeo {index}/{len(videos)}: {video.name}", f"Vérification de la vidéo {index}/{len(videos)} : {video.name}"),
                detail={"phase": "probe", "videoIndex": index, "videoCount": len(videos), "videoName": video.name, "framesDone": 0, "framesTarget": len(videos) * count_per_video, "startedAt": started},
            )
            infos.append((video, *media_info(video, apple_extractor)))
        total_duration = max(sum(item[1] for item in infos), 1.0)
        temporary_frames = root / f"frames.next-{uuid.uuid4().hex[:8]}"
        temporary_frames.mkdir(parents=True, exist_ok=False)
        frames: list[dict] = []
        completed = 0.0
        frames_target = len(infos) * count_per_video
        rebuilt_video_ids: set[str] = set()
        reused_videos = 0
        try:
            for video_index, (video, duration, width, height) in enumerate(infos, start=1):
                raise_if_cancelled()
                target = count_per_video
                video_id = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:12]
                done_before = len(frames)
                reused = reusable_video_frames(root, existing_project, video_id, count_per_video)
                if reused is not None:
                    frames_root = (root / "frames").resolve()
                    for frame in reused:
                        source = (root / str(frame["relativePath"])).resolve()
                        link_or_copy(source, temporary_frames / source.relative_to(frames_root))
                        frames.append({**frame, "videoName": video.name, "annotations": existing_annotations.get(str(frame.get("id")), [])})
                    rebuilt_video_ids.add(video_id)
                    reused_videos += 1
                    completed += duration
                    STATE.update(progress=min(completed / total_duration, 0.98))
                    continue
                STATE.update(
                    operation="extracting",
                    message=localized(f"Extrahiere Trainingsbilder: {video.name}", f"Extracting training images: {video.name}", f"Extrayendo imágenes de entrenamiento: {video.name}", f"Extraction des images d’entraînement : {video.name}"),
                    detail={"phase": "extract", "videoIndex": video_index, "videoCount": len(infos), "videoName": video.name, "framesDone": done_before, "framesTarget": frames_target, "startedAt": started},
                )

                def report(written: int, video=video, video_index=video_index, duration=duration, done_before=done_before) -> None:
                    share = min(written / max(count_per_video, 1), 1.0)
                    STATE.update(
                        progress=min((completed + duration * share) / total_duration, 0.98),
                        detail={"phase": "extract", "videoIndex": video_index, "videoCount": len(infos), "videoName": video.name, "framesDone": done_before + written, "framesTarget": frames_target, "startedAt": started},
                    )

                generated, rate, output_width, output_height = extract_frames_for_video(
                    video, duration, width, height, target, temporary_frames, video_id,
                    ffmpeg, apple_extractor,
                    localized("Frame-Extraktion fehlgeschlagen", "Frame extraction failed", "La extracción de fotogramas falló", "L’extraction des images a échoué"),
                    on_frames=report,
                )
                rebuilt_video_ids.add(video_id)
                for index, image_path in enumerate(generated, start=1):
                    frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{index}"))
                    frames.append({
                        "id": frame_id,
                        "relativePath": f"frames/{image_path.name}",
                        "videoID": video_id,
                        "videoName": video.name,
                        "timestamp": min((index - 1) / rate, duration),
                        "width": output_width,
                        "height": output_height,
                        "annotations": existing_annotations.get(frame_id, []),
                    })
                completed += duration
                STATE.update(progress=min(completed / total_duration, 0.98))
            carried = carry_over_frames(root, existing_project, temporary_frames, sport, rebuilt_video_ids)
            missing_videos = {str(frame.get("videoID")) for frame in carried if not str(frame.get("videoID", "")).startswith("import-") and not frame.get("heldOut") and not str(frame.get("relativePath", "")).startswith("frames/al-")}
            frames.extend(carried)
            root.mkdir(parents=True, exist_ok=True)
            frames_dir = root / "frames"
            previous = root / "frames.previous"
            if previous.exists(): shutil.rmtree(previous)
            if frames_dir.exists(): frames_dir.replace(previous)
            temporary_frames.replace(frames_dir)
            if previous.exists(): shutil.rmtree(previous)
            for directory in ("dataset", "runs", "exports"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            now = utc_now()
            project = {
                "schemaVersion": 2,
                "name": folder.name,
                "sport": sport,
                "sourceFolder": str(folder),
                "createdAt": (existing_project or {}).get("createdAt", now),
                "updatedAt": now,
                "framesPerVideo": count_per_video,
                "frames": frames,
            }
            if existing_project and existing_project.get("lastTraining"):
                project["lastTraining"] = existing_project["lastTraining"]
            if existing_project and existing_project.get("trainingHistory"):
                project["trainingHistory"] = existing_project["trainingHistory"]
            atomic_json(root / "project.json", project)
            created = localized(
                f"{len(frames)} echte Frames lokal erstellt.", f"{len(frames)} real frames created locally.",
                f"{len(frames)} fotogramas reales creados localmente.", f"{len(frames)} images réelles créées localement.",
            )
            if reused_videos:
                created += " " + localized(
                    f"{reused_videos} bereits vorbereitete Videos übernommen, nicht neu extrahiert.", f"{reused_videos} videos prepared earlier were kept, not extracted again.",
                    f"{reused_videos} vídeos ya preparados se conservaron sin volver a extraerlos.", f"{reused_videos} vidéos déjà préparées ont été conservées sans nouvelle extraction.",
                )
            if missing_videos:
                created += " " + localized(
                    f"Bilder und Markierungen von {len(missing_videos)} Videos, die nicht mehr im Ordner liegen, bleiben erhalten.",
                    f"Images and boxes of {len(missing_videos)} videos that are no longer in the folder were kept.",
                    f"Se conservaron las imágenes y marcas de {len(missing_videos)} vídeos que ya no están en la carpeta.",
                    f"Les images et annotations de {len(missing_videos)} vidéos qui ne sont plus dans le dossier ont été conservées.",
                )
            if has_images:
                summary = import_folder_images(folder, root)
                project = load_project(root)
                created += " " + working_folder_images_message(summary)
            STATE.update(project=project, operation="ready", busy=False, progress=1.0, detail=None, message=created)
        except BaseException:
            if temporary_frames.exists(): shutil.rmtree(temporary_frames)
            raise
    except OperationCancelled:
        STATE.update(operation="cancelled", busy=False, progress=0.0, detail=None, error=None, message=localized(
            "Vorbereitung abgebrochen. Die bisherigen Frames und Markierungen sind unverändert.",
            "Preparation cancelled. The previous frames and annotations are unchanged.",
            "Preparación cancelada. Los fotogramas y marcas anteriores no han cambiado.",
            "Préparation annulée. Les images et annotations précédentes sont inchangées.",
        ))
    except Exception as error:
        STATE.update(operation="error", busy=False, detail=None, error=str(error), message=localized("Lokale Vorbereitung fehlgeschlagen.", "Local preparation failed.", "La preparación local falló.", "La préparation locale a échoué."))
        STATE.append_log(str(error))


def expand_dataset(folder: Path, payload: dict, held_out: bool = False) -> None:
    """Extract a review inbox from separate videos, then pre-label only that inbox.

    held_out marks the new frames so build_dataset() (training) and
    freeze_ground_truth() (benchmarking) treat them as an independent test
    set: never trainable on, and the only source a benchmark ground truth
    may be frozen from. When held_out, every one of the sport's categories
    is pre-labeled for review (not just the caller's single/selected
    category), since an independent validation set needs correct boxes
    across all categories to produce a meaningful per-category comparison.
    """
    try:
        root = STATE.project_root
        project = STATE.project
        if root is None or project is None:
            raise RuntimeError(localized("Zuerst ein bestehendes Trainingsprojekt öffnen.", "Open an existing training project first.", "Primero abre un proyecto de entrenamiento existente.", "Ouvrez d’abord un projet d’entraînement existant."))
        if STATE.selected_folder and folder.resolve() == STATE.selected_folder.resolve():
            raise RuntimeError(localized("Für die Erweiterung bitte einen anderen Videoordner wählen.", "Please choose a different video folder for the expansion.", "Elige una carpeta de vídeos distinta para la ampliación.", "Veuillez choisir un autre dossier vidéo pour l’extension."))
        count_per_video = min(500, frames_per_video(payload.get("framesPerVideo", 100)))
        backup_project(root, "active-learning")
        ffmpeg = shutil.which("ffmpeg")
        apple_extractor = native_frame_extractor(root) if not ffmpeg or not shutil.which("ffprobe") else None
        # busy must be true before this point, not only once extraction itself
        # starts below - otherwise the UI doesn't show anything is happening
        # while videos are being discovered/probed, which for many files (a
        # GoPro game split into a dozen+ chapter files) can itself take a
        # visible moment.
        STATE.update(stage={"index": 1, "count": 2, "label": localized("Bilder extrahieren", "Extract images", "Extraer imágenes", "Extraire les images")})
        STATE.update(operation="active-learning-scan", busy=True, progress=0.01, detail=None, error=None, log=[], message=localized("Suche lokale Videos …", "Searching for local videos …", "Buscando vídeos locales …", "Recherche de vidéos locales …"))
        videos = discover_videos(folder)
        if not videos:
            raise RuntimeError(localized("Im gewählten Erweiterungsordner wurden keine Videos gefunden.", "No videos were found in the selected expansion folder.", "No se encontraron vídeos en la carpeta de ampliación seleccionada.", "Aucune vidéo n’a été trouvée dans le dossier d’extension sélectionné."))
        infos = []
        started = time.time()
        for index, video in enumerate(videos, start=1):
            raise_if_cancelled()
            STATE.update(
                message=localized(f"Prüfe Video {index}/{len(videos)}: {video.name}", f"Checking video {index}/{len(videos)}: {video.name}", f"Comprobando vídeo {index}/{len(videos)}: {video.name}", f"Vérification de la vidéo {index}/{len(videos)} : {video.name}"),
                detail={"phase": "probe", "videoIndex": index, "videoCount": len(videos), "videoName": video.name, "framesDone": 0, "framesTarget": len(videos) * count_per_video, "startedAt": started},
            )
            infos.append((video, *media_info(video, apple_extractor)))
        total_duration = max(sum(item[1] for item in infos), 1.0)
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        new_frames: list[dict] = []
        known_video_ids = {str(frame.get("videoID")) for frame in project.get("frames", [])}
        completed = 0.0
        STATE.update(operation="active-learning-extract", busy=True, progress=0.01, error=None, log=[], message=localized("Extrahiere lokale Prüfkandidaten …", "Extracting local review candidates …", "Extrayendo candidatos de revisión locales …", "Extraction des candidats de révision locaux …"))
        frames_target = len(infos) * count_per_video
        run_prefixes: list[str] = []
        for video_index, (video, duration, width, height) in enumerate(infos, start=1):
            if CANCEL_REQUESTED.is_set():
                remove_extracted_frames(frames_dir, run_prefixes)
                raise OperationCancelled()
            target = count_per_video
            video_id = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:12]
            if video_id in known_video_ids:
                completed += duration
                STATE.update(progress=min(completed / total_duration * 0.45, 0.45), message=localized(f"Bereits verwendetes Video übersprungen: {video.name}", f"Already-used video skipped: {video.name}", f"Vídeo ya usado omitido: {video.name}", f"Vidéo déjà utilisée ignorée : {video.name}"))
                continue
            prefix = f"al-{video_id}"
            run_prefixes.append(prefix)
            done_before = len(new_frames)

            def report(written: int, video=video, video_index=video_index, duration=duration, done_before=done_before) -> None:
                share = min(written / max(count_per_video, 1), 1.0)
                STATE.update(
                    progress=min((completed + duration * share) / total_duration * 0.45, 0.45),
                    detail={"phase": "extract", "videoIndex": video_index, "videoCount": len(infos), "videoName": video.name, "framesDone": done_before + written, "framesTarget": frames_target, "startedAt": started},
                )

            try:
                generated, rate, output_width, output_height = extract_frames_for_video(
                    video, duration, width, height, target, frames_dir, prefix,
                    ffmpeg, apple_extractor,
                    localized("Kandidaten-Extraktion fehlgeschlagen", "Candidate extraction failed", "La extracción de candidatos falló", "L’extraction des candidats a échoué"),
                    on_frames=report,
                )
            except OperationCancelled:
                # Frames of this run go straight into frames/; nothing is in project.json yet.
                remove_extracted_frames(frames_dir, run_prefixes)
                raise
            for index, image_path in enumerate(generated, start=1):
                new_frame = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"active-learning:{video_id}:{index}")),
                    "relativePath": f"frames/{image_path.name}", "videoID": video_id,
                    "videoName": video.name, "timestamp": min((index - 1) / rate, duration),
                    "width": output_width, "height": output_height, "annotations": [],
                    "reviewStatus": "candidate",
                }
                if held_out:
                    new_frame["heldOut"] = True
                new_frames.append(new_frame)
            completed += duration
            STATE.update(progress=min(completed / total_duration * 0.45, 0.45), message=localized(f"Neue Videos werden geprüft: {video.name}", f"Checking new videos: {video.name}", f"Comprobando vídeos nuevos: {video.name}", f"Vérification des nouvelles vidéos : {video.name}"))

        if not new_frames:
            raise RuntimeError(localized("Alle gewählten Videos wurden bereits verwendet. Bitte neue Videos auswählen.", "All selected videos were already used. Please choose new videos.", "Todos los vídeos seleccionados ya se usaron. Elige vídeos nuevos.", "Toutes les vidéos sélectionnées ont déjà été utilisées. Veuillez choisir de nouvelles vidéos."))
        project["frames"] = [*project.get("frames", []), *new_frames]
        project["updatedAt"] = utc_now()
        atomic_json(root / "project.json", project)
        STATE.update(project=project, busy=False, progress=0.5, detail=None, message=localized(
            f"{len(new_frames)} Kandidaten extrahiert. Lokale Erkennung startet …", f"{len(new_frames)} candidates extracted. Local detection starting …",
            f"{len(new_frames)} candidatos extraídos. Iniciando la detección local …", f"{len(new_frames)} candidats extraits. Démarrage de la détection locale …",
        ))
        STATE.update(detail=None, stage={"index": 2, "count": 2, "label": localized("Automatisch markieren", "Auto-label", "Marcado automático", "Marquage automatique")})
        autolabel_payload = {**payload, "candidateOnly": True, "threshold": payload.get("threshold", 0.12)}
        if held_out:
            autolabel_payload["categories"] = list(SPORT_CATEGORIES.get(project.get("sport"), []))
        ml_action("autolabel", autolabel_payload)
    except OperationCancelled:
        STATE.update(operation="cancelled", busy=False, progress=0.0, detail=None, error=None, message=localized(
            "Abgebrochen. Es wurden keine neuen Bilder übernommen.",
            "Cancelled. No new images were added.",
            "Cancelado. No se añadieron imágenes nuevas.",
            "Annulé. Aucune nouvelle image n’a été ajoutée.",
        ))
    except Exception as error:
        STATE.append_log(str(error))
        STATE.update(operation="error", busy=False, detail=None, error=str(error), message=localized("Datensatzerweiterung fehlgeschlagen.", "Dataset expansion failed.", "La ampliación del conjunto de datos falló.", "L’extension du jeu de données a échoué."))


SIMULATION_SLOTS = {"a": "simulation", "b": "simulation-b"}


# The simulation looks at a video at 12 images per second, or at every n-th of them (n = 1..12).
# A long video is not thinned out unevenly: the user picks a section (start and length) and how
# dense it is, because the tracking states of the player and the scene search
# (teachers.tracking_scenes) need neighbouring images that are close in time. There is no fixed
# cap on the number of images; the length of a section is limited so that one run holds at most
# SIMULATION_MAX_IMAGES images (180 seconds at 12 per second).
SIMULATION_FPS = 12.0
SIMULATION_DEFAULT_LENGTH = 25.0
SIMULATION_MAX_IMAGES = 2160
SIMULATION_MAX_EVERY = 12


def simulation_section(payload: dict, duration: float, same: tuple[float, float, int] | None = None) -> tuple[float, float, int, int]:
    """(start, length, every, image count) of the part of a video the simulation extracts: the
    images are 1/(12/every) seconds apart. `same` is the section of slot A, used as it is so
    that both sides show identical images."""
    if same is not None:
        start, length, every = same
    else:
        try:
            start = float(payload.get("start") or 0.0)
            length = float(payload.get("length") or SIMULATION_DEFAULT_LENGTH)
            every = int(payload.get("every") or 1)
        except (TypeError, ValueError):
            start, length, every = 0.0, SIMULATION_DEFAULT_LENGTH, 1
        start = max(0.0, start)
        every = min(SIMULATION_MAX_EVERY, max(1, every))
        length = min(SIMULATION_MAX_IMAGES / (SIMULATION_FPS / every), max(1.0, length))
    if start >= duration:
        raise RuntimeError(localized(
            f"Der Start ({start:g} s) liegt hinter dem Ende des Videos ({duration:.1f} s).",
            f"The start ({start:g} s) is after the end of the video ({duration:.1f} s).",
            f"El inicio ({start:g} s) está después del final del vídeo ({duration:.1f} s).",
            f"Le début ({start:g} s) est après la fin de la vidéo ({duration:.1f} s).",
        ))
    length = min(length, duration - start)
    return start, length, every, max(1, round(length * SIMULATION_FPS / every))


# A Full HD image takes about 450 KB as a JPEG (measured: 2,160 images = 973 MB).
SIMULATION_BYTES_PER_PIXEL = 0.22


def check_simulation_space(folder: Path, width: int, height: int, count: int, length: float, every: int) -> None:
    """Refuse a run whose images do not fit on the disk, and say what would: fewer images
    (every n-th) or a shorter section. The slot's old images are already gone when this runs."""
    per_image = max(width, 1) * max(height, 1) * SIMULATION_BYTES_PER_PIXEL
    needed = per_image * count * 1.1
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return
    if needed <= free:
        return
    fits = int(free * 0.9 / per_image)
    sparser = next((n for n in (1, 2, 3, 4, 6, 12) if n >= every and round(length * SIMULATION_FPS / n) <= fits), None)
    shorter = int(fits / (SIMULATION_FPS / every))
    gb = lambda value: f"{value / 1024 ** 3:.1f}"
    if sparser is not None and sparser > every:
        advice = localized(f"Wähle jedes {sparser}. Bild oder einen kürzeren Abschnitt.", f"Choose one image in {sparser} or a shorter section.", f"Elige una imagen de cada {sparser} o un tramo más corto.", f"Choisissez une image sur {sparser} ou un passage plus court.")
    else:
        advice = localized(f"Wähle einen kürzeren Abschnitt (höchstens etwa {shorter} s).", f"Choose a shorter section (at most about {shorter} s).", f"Elige un tramo más corto (como máximo unos {shorter} s).", f"Choisissez un passage plus court (au plus {shorter} s environ).")
    raise RuntimeError(localized(
        f"Zu wenig freier Speicher für die Simulation: {gb(needed)} GB nötig, {gb(free)} GB frei. ",
        f"Not enough free space for the simulation: {gb(needed)} GB needed, {gb(free)} GB free. ",
        f"No hay espacio libre suficiente para la simulación: hacen falta {gb(needed)} GB, hay {gb(free)} GB libres. ",
        f"Pas assez d’espace libre pour la simulation : {gb(needed)} Go nécessaires, {gb(free)} Go libres. ",
    ) + advice)


def simulation_slot(payload: dict) -> str:
    return "b" if str(payload.get("slot", "a")).lower() == "b" else "a"


def simulation_frames_dir(root: Path, slot: str) -> Path:
    return root / ".runtime" / SIMULATION_SLOTS.get(slot, "simulation")


def simulation_video(payload: dict) -> Path:
    """The clip for a simulation run: the one slot A used when slot B asks for the
    same video (so two models can be compared on identical frames), otherwise a
    file dialog."""
    if simulation_slot(payload) == "b" and payload.get("sameVideo"):
        video = STATE.simulation_videos.get("a")
        if video is None or not video.is_file():
            raise RuntimeError(localized(
                "Zuerst links (A) ein Video simulieren.", "Simulate a video on the left (A) first.",
                "Primero simula un vídeo a la izquierda (A).", "Simulez d’abord une vidéo à gauche (A).",
            ))
        return video
    return select_video_file()


# What a cancelled simulation does with the frames it has already analysed. /api/simulation-cancel
# sets it before it cancels the job; a plain cancel leaves it clear and the frames are deleted.
SIMULATION_KEEP_PARTIAL = threading.Event()


def read_partial_detections(path: Path) -> tuple[str | None, list[dict]]:
    """The frames a cancelled detection run finished (ml_worker --partial): the model's name
    and one {"file", "ball"} per frame. A last line cut off by the cancel is ignored."""
    model: str | None = None
    frames: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return model, frames
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            break
        if isinstance(item, dict) and "file" in item:
            frames.append(item)
        elif isinstance(item, dict) and item.get("model"):
            model = str(item["model"])
    return model, frames


def simulate_ball_tracking(video: Path, payload: dict) -> None:
    """Extract a short clip at near-native frame rate into a scratch directory
    (never added to project.json - this is a read-only diagnostic, not data
    collection) and run the active model's ball detection over every frame.

    Unlike expand_dataset()/extract_project(), this reuses the same scratch
    directory every time (root/.runtime/simulation) rather than one per run,
    since only one simulation is ever shown at a time and there is no reason
    to accumulate throwaway extractions.

    The run has three steps (extract, load the model, detect) and reports them as
    such. A cancel either keeps the frames analysed so far as a shorter simulation
    (SIMULATION_KEEP_PARTIAL) or removes everything; it never leaves a simulation
    behind whose images are gone.
    """
    slot = simulation_slot(payload)
    frames_dir: Path | None = None
    partial_path: Path | None = None
    rate: float | None = None
    duration = 0.0
    start = 0.0
    length = 0.0
    every = 1
    output_width = output_height = 0
    model = str(payload.get("model", "nano"))
    package = str(payload.get("package") or "").strip()
    threshold = 0.25

    def stage(index: int, label: str) -> dict:
        return {"index": index, "count": 3, "label": label}

    def result_for(detected: list[dict], model_label: str | None, total: int | None = None) -> dict:
        frames = [
            {
                "file": item["file"],
                "timestamp": min(start + index / (rate or SIMULATION_FPS), duration),
                "width": output_width,
                "height": output_height,
                "ball": item.get("ball"),
            }
            for index, item in enumerate(detected)
        ]
        scenes, scenes_total = teachers.tracking_scenes_with_total(frames)
        return {
            "id": str(time.time_ns()), "fps": rate, "frames": frames, "slot": slot,
            "start": start, "length": length, "every": every,
            "video": video.name, "model": model_label or model, "package": package or None,
            "threshold": threshold, "root": str(STATE.project_root),
            "scenes": scenes, "scenesTotal": scenes_total,
            **({"partial": {"done": len(frames), "total": total}} if total is not None else {}),
        }

    def publish(result: dict | None, **status) -> None:
        STATE.update(**({"simulation_b": result} if slot == "b" else {"simulation": result}), **status)

    try:
        SIMULATION_KEEP_PARTIAL.clear()
        root = STATE.project_root
        project = STATE.project
        if root is None or project is None:
            raise RuntimeError(localized("Zuerst ein bestehendes Trainingsprojekt öffnen.", "Open an existing training project first.", "Primero abre un proyecto de entrenamiento existente.", "Ouvrez d’abord un projet d’entraînement existant."))
        language = payload.get("language", "de")
        if language not in {"de", "en", "es", "fr"}:
            language = "de"
        threshold = min(0.95, max(0.01, float(payload.get("threshold", 0.25))))
        if package and (Path(package).name != package or package in {".", ".."}):
            raise RuntimeError(localized("Ungültige Modell-ID.", "Invalid model ID.", "ID de modelo no válido.", "ID de modèle invalide."))
        venv_python = venv_python_path(ml_venv(root))
        if not venv_python.is_file():
            raise RuntimeError(localized("ML-Umgebung fehlt. Zuerst „ML einrichten“ anklicken.", "The ML environment is missing. Click “Set up ML” first.", "Falta el entorno de ML. Primero haz clic en “Configurar ML”.", "L’environnement ML est manquant. Cliquez d’abord sur « Configurer le ML »."))

        extract_label = localized("Bilder extrahieren", "Extract images", "Extraer imágenes", "Extraire les images")
        # This slot's old images are replaced now, so its old simulation must not be shown any more.
        publish(None, operation="simulation-extract", busy=True, progress=0.01, error=None, log=[], stage=stage(1, extract_label), message=f"{extract_label} …")
        ffmpeg = shutil.which("ffmpeg")
        apple_extractor = native_frame_extractor(root) if not ffmpeg or not shutil.which("ffprobe") else None
        duration, width, height = media_info(video, apple_extractor)
        same = STATE.simulation_sections.get("a") if slot == "b" and payload.get("sameVideo") else None
        start, length, every, target_count = simulation_section(payload, duration, same)
        frames_dir = simulation_frames_dir(root, slot)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        check_simulation_space(frames_dir, width, height, target_count, length, every)
        # Neither extractor reports progress, but both write frames as they go.
        extraction_done = threading.Event()

        def watch_extraction() -> None:
            while not extraction_done.wait(0.4):
                written = min(target_count, sum(1 for _ in frames_dir.glob("sim-*.jpg")))
                STATE.update(progress=0.01 + 0.29 * written / target_count, message=f"{extract_label} … {written}/{target_count}")

        threading.Thread(target=watch_extraction, daemon=True).start()
        try:
            generated, rate, output_width, output_height = extract_frames_for_video(
                video, duration, width, height, target_count, frames_dir, "sim",
                ffmpeg, apple_extractor,
                localized("Extraktion für die Simulation fehlgeschlagen", "Extraction for the simulation failed", "La extracción para la simulación falló", "L’extraction pour la simulation a échoué"),
                section=(start, length, SIMULATION_FPS / every),
            )
        finally:
            extraction_done.set()
        if not generated:
            raise RuntimeError(localized("Es konnten keine Bilder aus dem Video extrahiert werden.", "No images could be extracted from the video.", "No se pudieron extraer imágenes del vídeo.", "Aucune image n’a pu être extraite de la vidéo."))

        load_label = localized("Modell laden", "Load the model", "Cargar el modelo", "Charger le modèle")
        detect_label = localized("Ball suchen", "Find the ball", "Buscar el balón", "Chercher le ballon")
        STATE.update(operation="simulation-detect", progress=0.3, stage=stage(2, load_label), message=f"{load_label} …")
        worker = find_ml_worker()
        partial_path = frames_dir / "detections.jsonl"
        def detection_progress(done: int, total: int) -> None:
            STATE.update(progress=0.35 + 0.65 * done / max(total, 1), stage=stage(3, detect_label), message=f"{detect_label} … {done}/{total}")

        detections = run_json([
            str(venv_python), str(worker), "simulate-ball-tracking",
            "--project", str(root), "--frames-dir", str(frames_dir),
            "--model", model, "--threshold", str(threshold), "--language", language,
            "--partial", str(partial_path),
            *(["--package", package] if package else []),
        ], on_progress=detection_progress)
        partial_path.unlink(missing_ok=True)
        result = result_for(detections.get("frames", []), detections.get("model"))
        STATE.simulation_videos[slot] = video
        STATE.simulation_sections[slot] = (start, length, every)
        count, scenes = len(result["frames"]), len(result["scenes"])
        publish(result, operation="ready", busy=False, progress=1.0, message=localized(
            f"{count} Bilder simuliert, {scenes} auffällige Szenen gefunden.", f"{count} images simulated, {scenes} notable scenes found.",
            f"{count} imágenes simuladas, {scenes} escenas llamativas encontradas.", f"{count} images simulées, {scenes} scènes notables trouvées.",
        ))
    except OperationCancelled:
        model_label, detected = read_partial_detections(partial_path) if partial_path else (None, [])
        total = sum(1 for _ in frames_dir.glob("sim-*.jpg")) if frames_dir and frames_dir.is_dir() else 0
        if SIMULATION_KEEP_PARTIAL.is_set() and detected and rate and frames_dir:
            kept = {str(item["file"]) for item in detected}
            for path in frames_dir.glob("sim-*.jpg"):
                if path.name not in kept:
                    path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)  # type: ignore[union-attr]
            STATE.simulation_videos[slot] = video
            STATE.simulation_sections[slot] = (start, length, every)
            publish(result_for(detected, model_label, total), operation="cancelled", busy=False, progress=0.0, error=None, message=localized(
                f"Simulation abgebrochen. {len(detected)} von {total} Bildern sind ausgewertet und bleiben erhalten.",
                f"Simulation cancelled. {len(detected)} of {total} images are analysed and kept.",
                f"Simulación cancelada. {len(detected)} de {total} imágenes están analizadas y se conservan.",
                f"Simulation annulée. {len(detected)} images sur {total} sont analysées et conservées.",
            ))
        else:
            wanted = SIMULATION_KEEP_PARTIAL.is_set()
            if frames_dir is not None:
                shutil.rmtree(frames_dir, ignore_errors=True)
            STATE.simulation_videos.pop(slot, None)
            STATE.simulation_sections.pop(slot, None)
            publish(None, operation="cancelled", busy=False, progress=0.0, error=None, message=localized(
                "Simulation abgebrochen. Es war noch kein Bild ausgewertet, es gibt nichts zu behalten.",
                "Simulation cancelled. No image had been analysed yet, so there is nothing to keep.",
                "Simulación cancelada. Aún no se había analizado ninguna imagen; no hay nada que conservar.",
                "Simulation annulée. Aucune image n’était encore analysée, il n’y a rien à conserver.",
            ) if wanted else localized(
                "Simulation abgebrochen. Die Bilder dieses Laufs wurden gelöscht.", "Simulation cancelled. The images of this run were deleted.",
                "Simulación cancelada. Se han eliminado las imágenes de esta ejecución.", "Simulation annulée. Les images de cette exécution ont été supprimées.",
            ))
    except Exception as error:
        if frames_dir is not None:
            shutil.rmtree(frames_dir, ignore_errors=True)
        STATE.simulation_videos.pop(slot, None)
        STATE.simulation_sections.pop(slot, None)
        STATE.append_log(str(error))
        publish(None, operation="error", busy=False, error=str(error), message=localized("Balltracking-Simulation fehlgeschlagen.", "Ball-tracking simulation failed.", "La simulación de seguimiento del balón falló.", "La simulation de suivi du ballon a échoué."))
    finally:
        SIMULATION_KEEP_PARTIAL.clear()


def cancel_simulation(payload: dict) -> dict:
    """Cancel the running simulation; `keep` decides what happens to the frames analysed so far."""
    current = JOBS.snapshot()["current"]
    if not current or current.get("type") != "simulate-ball-tracking":
        raise RuntimeError(localized("Es läuft keine Simulation.", "No simulation is running.", "No hay ninguna simulación en curso.", "Aucune simulation n’est en cours."))
    if payload.get("keep"):
        SIMULATION_KEEP_PARTIAL.set()
    else:
        SIMULATION_KEEP_PARTIAL.clear()
    return JOBS.cancel(str(current["id"]))


def extra_ball_boxes(existing: list[dict], found: list[dict]) -> list[dict]:
    """The balls from a second full-frame look (ml_worker find-extra-balls) that are new: not
    the box the image already carries (or one that mostly covers it) and not twice the same.
    Boxes are {"x", "y", "width", "height", "confidence"} (top left corner and size)."""
    def overlap(a: dict, b: dict) -> float:
        left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
        right, bottom = min(a["x"] + a["width"], b["x"] + b["width"]), min(a["y"] + a["height"], b["y"] + b["height"])
        inside = max(right - left, 0.0) * max(bottom - top, 0.0)
        smaller = min(a["width"] * a["height"], b["width"] * b["height"])
        return inside / smaller if smaller > 0 else 0.0

    kept: list[dict] = []
    for box in sorted(found, key=lambda item: -float(item.get("confidence") or 0.0)):
        if all(overlap(box, other) < 0.3 for other in [*existing, *kept]):
            kept.append(box)
    return kept


def simulation_gaps_to_candidates(payload: dict) -> None:
    """Turn scenes of the last tracking simulation (teachers.tracking_scenes) into review
    candidates. payload["scenes"] lists the scene ids the user chose; without it all gaps
    are taken. A few frames of every chosen scene are copied into the project as candidates:
    - gap: looked at again in a window around the expected position, with a suggested ball
      box where that second look found one, empty otherwise;
    - jump, weak: with the box the detector reported, for the user to confirm or remove;
    - lost: empty.

    This is the only way simulation frames reach the project, and only on request. A video
    that already supplies independent test images is refused: its frames must neither be
    trained on nor change the test."""
    scratch: Path | None = None
    try:
        root = STATE.project_root
        project = STATE.project
        if root is None or project is None:
            raise RuntimeError(localized("Zuerst ein bestehendes Trainingsprojekt öffnen.", "Open an existing training project first.", "Primero abre un proyecto de entrenamiento existente.", "Ouvrez d’abord un projet d’entraînement existant."))
        language = payload.get("language", "de")
        if language not in {"de", "en", "es", "fr"}:
            language = "de"
        slot = simulation_slot(payload)
        simulation = STATE.simulation_b if slot == "b" else STATE.simulation
        video = STATE.simulation_videos.get(slot)
        frames_dir = simulation_frames_dir(root, slot)
        if not simulation or video is None or not frames_dir.is_dir():
            raise RuntimeError(localized("Zuerst ein Video simulieren.", "Simulate a video first.", "Primero simula un vídeo.", "Simulez d’abord une vidéo."))
        video_id = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:12]
        if any(str(frame.get("videoID")) == video_id and frame.get("heldOut") for frame in project.get("frames", [])):
            raise RuntimeError(localized(
                "Dieses Video liefert bereits unabhängige Testbilder. Seine Bilder dürfen nicht ins Training.",
                "This video already supplies independent test images. Its images must not be trained on.",
                "Este vídeo ya aporta imágenes de prueba independientes. Sus imágenes no deben entrenarse.",
                "Cette vidéo fournit déjà des images de test indépendantes. Ses images ne doivent pas être entraînées.",
            ))
        sim_frames = simulation.get("frames", [])
        scenes = simulation.get("scenes")
        if scenes is None:
            scenes = teachers.tracking_scenes_with_total(sim_frames)[0]
        chosen = payload.get("scenes")
        if chosen is None:
            scenes = [scene for scene in scenes if scene["kind"] == "gap"]
        else:
            wanted = {str(item) for item in chosen} if isinstance(chosen, list) else set()
            scenes = [scene for scene in scenes if scene["id"] in wanted]
        picks = []
        taken: set[int] = set()
        for scene in scenes:
            for pick in scene["picks"]:
                if pick["index"] not in taken and len(picks) < 60:
                    taken.add(pick["index"])
                    picks.append({**pick, "kind": scene["kind"]})
        known_ids = {str(frame.get("id")) for frame in project.get("frames", [])}

        def frame_id(pick: dict) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ball-gap:{video_id}:{round(float(sim_frames[pick['index']]['timestamp']) * 1000)}"))

        picks = [pick for pick in picks if frame_id(pick) not in known_ids and (frames_dir / Path(pick["file"]).name).is_file()]
        if not picks:
            STATE.update(operation="ready", busy=False, progress=1.0, message=localized(
                "Keine neuen Lücken gefunden, in denen der Ball zwischen zwei Erkennungen fehlt.",
                "No new gaps found where the ball is missing between two detections.",
                "No se encontraron huecos nuevos en los que falte el balón entre dos detecciones.",
                "Aucune nouvelle lacune où le ballon manque entre deux détections.",
            ) if chosen is None else localized(
                "Die gewählten Szenen enthalten keine neuen Bilder; sie sind schon im Projekt.",
                "The chosen scenes contain no new images; they are already in the project.",
                "Las escenas elegidas no contienen imágenes nuevas; ya están en el proyecto.",
                "Les scènes choisies ne contiennent aucune nouvelle image ; elles sont déjà dans le projet.",
            ))
            return
        searches = [pick for pick in picks if pick["kind"] == "gap"]
        venv_python = venv_python_path(ml_venv(root))
        if searches and not venv_python.is_file():
            raise RuntimeError(localized("ML-Umgebung fehlt. Zuerst „ML einrichten“ anklicken.", "The ML environment is missing. Click “Set up ML” first.", "Falta el entorno de ML. Primero haz clic en “Configurar ML”.", "L’environnement ML est manquant. Cliquez d’abord sur « Configurer le ML »."))

        STATE.update(operation="simulation-gaps", busy=True, progress=0.02, error=None, log=[], message=localized(
            f"Suche den Ball in {len(searches)} Lücken-Bildern …", f"Looking for the ball in {len(searches)} gap images …",
            f"Buscando el balón en {len(searches)} imágenes con huecos …", f"Recherche du ballon dans {len(searches)} images de lacunes …",
        ))
        scratch = root / ".runtime" / f"ball-gaps-{slot}.json"
        atomic_json(scratch, [{key: pick[key] for key in ("file", "x", "y", "radius")} for pick in searches])  # type: ignore[arg-type]
        package = str(simulation.get("package") or "").strip()
        found = {"frames": []} if not searches else run_json([
            str(venv_python), str(find_ml_worker()), "recover-ball-gaps",
            "--project", str(root), "--frames-dir", str(frames_dir), "--requests", str(scratch),
            "--model", str(payload.get("model", "nano")), "--language", language,
            *(["--package", package] if package else []),
        ], on_progress=lambda done, total: STATE.update(progress=0.02 + 0.9 * done / max(total, 1)))
        balls = {str(item.get("file")): item.get("ball") for item in found.get("frames", [])}

        # A frame that becomes a training image should show every ball, not only the one the
        # simulation followed: look at each whole frame again for further balls (edge, floor).
        STATE.update(progress=0.5, message=localized(
            f"Prüfe {len(picks)} Bilder auf weitere Bälle …", f"Checking {len(picks)} images for further balls …",
            f"Comprobando {len(picks)} imágenes en busca de más balones …", f"Recherche d’autres ballons dans {len(picks)} images …",
        ))
        extras: dict[str, list[dict]] = {}
        try:
            scratch_all = root / ".runtime" / f"ball-extra-{slot}.json"
            atomic_json(scratch_all, [{"file": pick["file"]} for pick in picks])
            more = run_json([
                str(venv_python), str(find_ml_worker()), "find-extra-balls",
                "--project", str(root), "--frames-dir", str(frames_dir), "--requests", str(scratch_all),
                "--model", str(payload.get("model", "nano")), "--language", language,
                *(["--package", package] if package else []),
            ], on_progress=lambda done, total: STATE.update(progress=0.5 + 0.42 * done / max(total, 1)))
            extras = {str(item.get("file")): item.get("balls") or [] for item in more.get("frames", [])}
        except OperationCancelled:
            raise
        except Exception as error:  # the candidates are still useful without the extra boxes
            STATE.append_log(f"find-extra-balls: {error}")

        backup_project(root, "ball-gaps")
        target_dir = root / "frames"
        target_dir.mkdir(parents=True, exist_ok=True)
        ball_category = "puck" if project.get("sport") == "hockey" else "ball"
        new_frames: list[dict] = []
        recovered = 0
        further = 0
        for pick in picks:
            source = sim_frames[pick["index"]]
            name = Path(pick["file"]).name
            timestamp = float(source.get("timestamp") or 0.0)
            target = target_dir / f"gap-{video_id}-{round(timestamp * 1000):08d}.jpg"
            shutil.copyfile(frames_dir / name, target)
            annotations = []
            ball = balls.get(name)
            reported = pick.get("ball") if pick["kind"] in {"jump", "weak"} else None
            if reported and reported.get("width") and reported.get("height"):
                # the detector's own box (centre and size), for the user to confirm or remove
                ball = {
                    "x": float(reported["x"]) - float(reported["width"]) / 2, "y": float(reported["y"]) - float(reported["height"]) / 2,
                    "width": float(reported["width"]), "height": float(reported["height"]), "confidence": float(reported.get("confidence") or 0.0),
                }
            elif ball:
                recovered += 1
            if ball:
                annotations.append({
                    "id": str(uuid.uuid4()), "category": ball_category,
                    "x": max(float(ball["x"]), 0.0), "y": max(float(ball["y"]), 0.0),
                    "width": max(float(ball["width"]), 1.0), "height": max(float(ball["height"]), 1.0),
                    "confidence": float(ball["confidence"]), "source": "auto",
                })
            for extra in extra_ball_boxes([{key: float(item[key]) for key in ("x", "y", "width", "height")} for item in annotations], extras.get(name, [])):
                annotations.append({
                    "id": str(uuid.uuid4()), "category": ball_category,
                    "x": max(float(extra["x"]), 0.0), "y": max(float(extra["y"]), 0.0),
                    "width": max(float(extra["width"]), 1.0), "height": max(float(extra["height"]), 1.0),
                    "confidence": float(extra["confidence"]), "source": "auto",
                })
                further += 1
            new_frames.append({
                "id": frame_id(pick), "relativePath": f"frames/{target.name}", "videoID": video_id,
                "videoName": video.name, "timestamp": timestamp,
                "width": source.get("width"), "height": source.get("height"),
                "annotations": annotations, "reviewStatus": "candidate", "origin": f"ball-{pick['kind']}",
            })
        project["frames"] = [*project.get("frames", []), *new_frames]
        project["updatedAt"] = utc_now()
        atomic_json(root / "project.json", project)
        if chosen is not None:
            STATE.update(project=project, operation="ready", busy=False, progress=1.0, message=localized(
                f"{len(new_frames)} Bilder aus {len(scenes)} Szenen als Prüfkandidaten übernommen ({len(searches)} aus Lücken, in {recovered} davon wurde der Ball im Ausschnitt gefunden). Sie stehen jetzt in der Bilderliste zur Prüfung." + (f" Dazu {further} weitere Bälle vorgemerkt." if further else ""),
                f"{len(new_frames)} images from {len(scenes)} scenes added as review candidates ({len(searches)} from gaps; the ball was found in the close-up in {recovered} of them). They are now in the image list for review." + (f" {further} further balls were marked as suggestions." if further else ""),
                f"{len(new_frames)} imágenes de {len(scenes)} escenas añadidas como candidatas ({len(searches)} de huecos; el balón se encontró en el recorte en {recovered} de ellas). Ya están en la lista de imágenes para revisarlas." + (f" Se propusieron {further} balones más." if further else ""),
                f"{len(new_frames)} images de {len(scenes)} scènes ajoutées comme candidates ({len(searches)} issues de lacunes ; le ballon a été trouvé dans le recadrage pour {recovered} d’entre elles). Elles sont maintenant dans la liste d’images à vérifier." + (f" {further} autres ballons ont été proposés." if further else ""),
            ))
            return
        STATE.update(project=project, operation="ready", busy=False, progress=1.0, message=localized(
            f"{len(new_frames)} Lücken-Bilder als Prüfkandidaten übernommen, in {recovered} davon wurde der Ball im Ausschnitt gefunden." + (f" Dazu {further} weitere Bälle vorgemerkt." if further else ""),
            f"{len(new_frames)} gap images added as review candidates; the ball was found in the close-up in {recovered} of them." + (f" {further} further balls were marked as suggestions." if further else ""),
            f"{len(new_frames)} imágenes con huecos añadidas como candidatas; el balón se encontró en el recorte en {recovered} de ellas." + (f" Se propusieron {further} balones más." if further else ""),
            f"{len(new_frames)} images de lacunes ajoutées comme candidates ; le ballon a été trouvé dans le recadrage pour {recovered} d’entre elles." + (f" {further} autres ballons ont été proposés." if further else ""),
        ))
    except OperationCancelled:
        STATE.update(operation="cancelled", busy=False, progress=0.0, error=None, message=localized("Abgebrochen. Es wurden keine neuen Bilder übernommen.", "Cancelled. No new images were added.", "Cancelado. No se añadieron imágenes nuevas.", "Annulé. Aucune nouvelle image n’a été ajoutée."))
    except Exception as error:
        STATE.append_log(str(error))
        STATE.update(operation="error", busy=False, error=str(error), message=localized("Lücken konnten nicht übernommen werden.", "The gaps could not be added.", "No se pudieron añadir los huecos.", "Les lacunes n’ont pas pu être ajoutées."))
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def find_ml_worker() -> Path:
    candidates = [
        BASE_DIR / "ml_worker.py",
        BASE_DIR.parent / "reco-trainer-mac" / "Sources" / "RecoTrainerMac" / "Resources" / "ml_worker.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(localized("Der lokale RF-DETR-Worker wurde nicht gefunden.", "The local RF-DETR worker was not found.", "No se encontró el proceso local de RF-DETR.", "Le processus RF-DETR local est introuvable."))


# The concrete path class of this system, fixed at import (tests fake os.name for Windows).
SYSTEM_PATH = type(Path(__file__))


def shared_ml_env_dir() -> Path | None:
    """The ML environment is set up once per computer, not once per project. None inside
    Docker: there it stays in the project's .runtime folder on the mounted volume."""
    configured = os.environ.get("RECO_ML_ENV")
    if configured:
        return SYSTEM_PATH(configured).expanduser()
    base = vlm.app_data_dir()
    return base / "ml-env" if base is not None else None


def remembered_ml_venv(shared: Path) -> Path | None:
    try:
        value = json.loads((shared / "remembered.json").read_text(encoding="utf-8")).get("venv")
    except (OSError, ValueError, AttributeError):
        return None
    return SYSTEM_PATH(value) if isinstance(value, str) and value else None


def remember_ml_venv(shared: Path, venv: Path) -> None:
    if remembered_ml_venv(shared) == venv:
        return
    try:
        shared.mkdir(parents=True, exist_ok=True)
        atomic_json(shared / "remembered.json", {"venv": str(venv)})
    except OSError:
        pass  # Remembering is a convenience: the project's own environment still works.


def ml_venv(root: Path | None) -> Path:
    """Which ML environment to use: the computer's shared one, else one set up in this project
    before 0.26 (remembered for other projects), else the last remembered one. Without any of
    them this is where "Set up ML" creates the shared one."""
    shared = shared_ml_env_dir()
    own = root / ".runtime" / "venv" if root is not None else None
    if shared is None:
        return own if own is not None else Path(".runtime") / "venv"
    shared_venv = shared / "venv"
    if venv_python_path(shared_venv).is_file():
        return shared_venv
    if own is not None and venv_python_path(own).is_file():
        remember_ml_venv(shared, own)
        return own
    remembered = remembered_ml_venv(shared)
    if remembered is not None and venv_python_path(remembered).is_file():
        return remembered
    return shared_venv


def ml_setup_target(root: Path, venv: Path) -> Path:
    """"Set up ML" repairs the environment in use. One remembered from another project is
    only reused while it works; a too old one is replaced by the shared one and never
    deleted from the other project."""
    shared = shared_ml_env_dir()
    own = root / ".runtime" / "venv"
    if shared is None or venv in {shared / "venv", own}:
        return venv
    existing = python_version_of(venv_python_path(venv))
    return venv if existing is not None and existing >= MIN_PYTHON else shared / "venv"


def venv_python_path(venv: Path) -> Path:
    """Where `python -m venv <venv>` puts its interpreter - venv/bin/python3 on
    macOS/Linux, venv/Scripts/python.exe on Windows. Hardcoding the POSIX layout
    here previously made every local ML action after venv creation fail on
    Windows with "[WinError 2] The system cannot find the file specified",
    since that path never exists there.
    """
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python3"


MIN_PYTHON = (3, 10)
# Validated versions first, then newer/older supported ones.
PYTHON_PREFERENCE = (12, 11, 13, 10)


def python_version_of(executable: Path) -> tuple[int, int] | None:
    """Return (major, minor) of the given interpreter, or None if it cannot be run."""
    try:
        result = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=20,
        )
        major, minor = result.stdout.split()[:2]
        return int(major), int(minor)
    except Exception:
        return None


def python_candidates() -> list[Path]:
    paths: list[Path] = []
    for minor in PYTHON_PREFERENCE:
        found = shutil.which(f"python3.{minor}")
        if found:
            paths.append(Path(found))
        paths += [Path(f"/opt/homebrew/bin/python3.{minor}"), Path(f"/usr/local/bin/python3.{minor}")]
    paths.append(Path(sys.executable))
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            paths.append(Path(found))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        if str(path) not in seen:
            seen.add(str(path))
            unique.append(path)
    return unique


def system_python() -> Path:
    """First installed Python >= 3.10 (3.12/3.11 preferred) - rfdetr>=1.9 cannot be installed with older ones."""
    for candidate in python_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            version = python_version_of(candidate)
            if version is not None and version >= MIN_PYTHON:
                return candidate
    raise RuntimeError(localized(
        "Es wurde kein Python 3.10 oder neuer gefunden (empfohlen: 3.12). Bitte installieren (z. B. python.org oder „brew install python@3.12“) und „ML einrichten“ erneut starten.",
        "No Python 3.10 or newer was found (recommended: 3.12). Please install it (e.g. python.org or `brew install python@3.12`) and run \u201cSet up ML\u201d again.",
        "No se encontró Python 3.10 o posterior (recomendado: 3.12). Instálalo (p. ej. python.org o «brew install python@3.12») y vuelve a ejecutar «Configurar ML».",
        "Aucun Python 3.10 ou plus récent n’a été trouvé (recommandé : 3.12). Installez-le (p. ex. python.org ou « brew install python@3.12 ») puis relancez « Configurer ML ».",
    ))


def yolo_installed(venv_python: Path) -> bool:
    try:
        return subprocess.run([str(venv_python), "-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('ultralytics') else 1)"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_yolo(venv_python: Path, license_accepted: bool) -> None:
    """Install Ultralytics into the ML environment, but only after the user accepted AGPL-3.0."""
    if yolo_installed(venv_python):
        return
    if not license_accepted:
        raise RuntimeError(localized(
            "Für YOLO zuerst der AGPL-3.0-Lizenz von Ultralytics zustimmen.",
            "Accept the Ultralytics AGPL-3.0 license first to use YOLO.",
            "Acepta primero la licencia AGPL-3.0 de Ultralytics para usar YOLO.",
            "Acceptez d’abord la licence AGPL-3.0 d’Ultralytics pour utiliser YOLO.",
        ))
    STATE.update(message=localized(
        f"Installiere YOLO (Ultralytics {yolo_training.ULTRALYTICS_VERSION}, AGPL-3.0) …",
        f"Installing YOLO (Ultralytics {yolo_training.ULTRALYTICS_VERSION}, AGPL-3.0) …",
        f"Instalando YOLO (Ultralytics {yolo_training.ULTRALYTICS_VERSION}, AGPL-3.0) …",
        f"Installation de YOLO (Ultralytics {yolo_training.ULTRALYTICS_VERSION}, AGPL-3.0) …",
    ))
    for arguments in yolo_training.INSTALL_ARGS:
        run_logged([str(venv_python), "-m", "pip", "install", *arguments])


def run_logged(command: list[str], extra_env: dict[str, str] | None = None) -> dict | None:
    # Without an explicit UTF-8 encoding on both ends, a subprocess whose stdout
    # is piped (not a real terminal) falls back to the OS's locale-preferred
    # encoding to decide how to write text - on Windows that's typically a
    # legacy codepage like cp1252, not UTF-8. Training output that includes any
    # non-ASCII character (e.g. RF-DETR's rich-rendered metrics tables use
    # box-drawing characters) then crashes the child with a raw
    # UnicodeEncodeError instead of completing. PYTHONIOENCODING fixes the
    # child's own writes; encoding="utf-8" here fixes how this parent process
    # decodes what it reads back.
    process = popen_tracked(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1, env={**os.environ, **gpu_environment(), "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONIOENCODING": "utf-8", **(extra_env or {})},
    )
    assert process.stdout is not None
    last_json: dict | None = None
    error_line: str | None = None
    user_error: str | None = None
    try:
        for line in process.stdout:
            if EXCEPTION_LINE.match(line.strip()):
                error_line = line.strip()[:400]
            if line.startswith("{"):
                # A command's result (for example a dataset export summary) - not log text.
                try:
                    last_json = json.loads(line)
                    continue
                except ValueError:
                    pass
            if line.startswith("RECO_USER_ERROR "):
                user_error = line.removeprefix("RECO_USER_ERROR ").strip()
                STATE.append_log(user_error)
                continue
            if line.startswith("RECO_GPU_FALLBACK "):
                record_gpu_problem(line.removeprefix("RECO_GPU_FALLBACK ").strip())
                continue
            stage = STAGE_LINE.match(line.strip())
            if stage:
                STATE.update(stage={"index": int(stage.group(1)), "count": int(stage.group(2)), "label": stage.group(3).strip()}, progress=0.0)
                continue
            progress = PROGRESS_LINE.match(line.strip())
            if progress:
                if int(progress.group(2)) > 0:
                    STATE.update(progress=min(0.99, int(progress.group(1)) / int(progress.group(2))))
                continue
            STATE.append_log(line)
            clean = line.strip()
            match = re.search(r"(?:^|\s)(\d+)\s*/\s*(\d+)(?:\s|$)", clean)
            # Library warnings, indented continuation lines and timestamped library logs stay in
            # the log but do not replace the status line shown in the progress window.
            noise = (
                not clean or line[:1].isspace() or "warn" in clean.lower() or "site-packages" in clean
                or clean.startswith(("[20", "Traceback", "W1", "I1")) or clean[0] in "│┃└┌├┏┡┗┣━─╭╰═║╘╞╔╚"
            )
            values = {} if noise else {"message": clean}
            if match and int(match.group(2)) > 0 and not noise:
                values["progress"] = min(0.99, int(match.group(1)) / int(match.group(2)))
            if values:
                STATE.update(**values)
        code = process.wait()
    finally:
        untrack(process)
        if hasattr(process.stdout, "close"):
            process.stdout.close()
    if CANCEL_REQUESTED.is_set():
        raise OperationCancelled()
    if code == 3 and user_error:
        LAST_USER_ERROR["message"] = user_error
        raise WorkerUserError(user_error)
    if code != 0:
        # The bare exit code tells a remote tester nothing; name the last exception the
        # worker printed so the error dialog already points at the cause.
        cause = f" ({error_line})" if error_line else ""
        raise RuntimeError(localized(f"Lokaler ML-Worker wurde mit Code {code} beendet.{cause}", f"The local ML worker exited with code {code}.{cause}", f"El proceso local de ML terminó con el código {code}.{cause}", f"Le processus ML local s’est terminé avec le code {code}.{cause}"))
    return last_json


# "ValueError: ...", "onnxruntime.capi...Fail: ...", "OSError: [WinError 126] ..."
EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Fail|Failure)\b:?")


PROGRESS_LINE = re.compile(r"^RECO_PROGRESS (\d+) (\d+)\s*$")
# Multi-step ml_worker commands announce "RECO_STAGE <index> <count> <label>".
STAGE_LINE = re.compile(r"^RECO_STAGE (\d+) (\d+) (.*)$")


def run_json(command: list[str], on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Like run_logged, but for single-JSON actions (ml_worker.py subcommands
    that print exactly one JSON blob and nothing else instead of a progress
    stream) - captures stdout and parses it instead of logging it line by
    line. Same UTF-8 fix as run_logged (see its comment) applies here too.

    Such actions may report progress as "RECO_PROGRESS <done> <total>" lines on
    stderr (stdout stays reserved for the result); those go to on_progress and
    are kept out of any error message.
    """
    process = popen_tracked(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONIOENCODING": "utf-8"},
    )
    stderr_lines: list[str] = []

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            match = PROGRESS_LINE.match(line)
            if match:
                if on_progress:
                    on_progress(int(match.group(1)), int(match.group(2)))
            else:
                stderr_lines.append(line)

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    assert process.stdout is not None
    try:
        stdout = process.stdout.read()
        returncode = process.wait()
        reader.join()
    finally:
        untrack(process)
        if hasattr(process.stdout, "close"):
            process.stdout.close()
    if CANCEL_REQUESTED.is_set():
        raise OperationCancelled()
    user_error = next((line.removeprefix("RECO_USER_ERROR ").strip() for line in stdout.splitlines() if line.startswith("RECO_USER_ERROR ")), None)
    if returncode == 3 and user_error:
        LAST_USER_ERROR["message"] = user_error
        raise WorkerUserError(user_error)
    if returncode != 0:
        raise RuntimeError("".join(stderr_lines).strip() or stdout.strip() or localized(f"Lokaler ML-Worker wurde mit Code {returncode} beendet.", f"The local ML worker exited with code {returncode}.", f"El proceso local de ML terminó con el código {returncode}.", f"Le processus ML local s’est terminé avec le code {returncode}."))
    # RF-DETR/PyTorch can print their own notices to stdout while loading the
    # model, ahead of the worker's final JSON print - only that last non-empty
    # line is the result (same rule as decodeLastJSONLine in the Mac app).
    last_line = next((line for line in reversed(stdout.splitlines()) if line.strip()), "")
    return json.loads(last_line)


def label_import_message(summary: dict, candidates: bool) -> str:
    """One status line for the final JSON of `ml_worker.py import-labels`."""
    images = int(summary.get("imported", 0)) + int(summary.get("matched", 0))
    boxes = int(summary.get("annotations", 0))
    duplicates = int(summary.get("duplicates", 0))
    label_format = str(summary.get("format", "")).upper()
    parts = [localized(
        f"Labels importiert ({label_format}): {images} Bilder, {boxes} Boxen.",
        f"Labels imported ({label_format}): {images} images, {boxes} boxes.",
        f"Etiquetas importadas ({label_format}): {images} imágenes, {boxes} cuadros.",
        f"Annotations importées ({label_format}) : {images} images, {boxes} boîtes.",
    )]
    roles = summary.get("roles") or {}
    if roles:
        parts.append(localized(
            f"Rollen aus dem Reco-Trainer-Datensatz übernommen: {roles.get('train', 0)} Training, {roles.get('valid', 0)} Validierung, {roles.get('test', 0)} Test.",
            f"Roles kept from the Reco Trainer dataset: {roles.get('train', 0)} training, {roles.get('valid', 0)} validation, {roles.get('test', 0)} test.",
            f"Roles conservados del conjunto de Reco Trainer: {roles.get('train', 0)} entrenamiento, {roles.get('valid', 0)} validación, {roles.get('test', 0)} prueba.",
            f"Rôles conservés du jeu Reco Trainer : {roles.get('train', 0)} entraînement, {roles.get('valid', 0)} validation, {roles.get('test', 0)} test.",
        ))
    if candidates and images:
        parts.append(localized("Als Vorschläge in der Prüfwarteschlange.", "Added as suggestions to the review queue.", "Añadidas como sugerencias a la cola de revisión.", "Ajoutées comme suggestions à la file de vérification."))
    if duplicates:
        parts.append(localized(f"{duplicates} bereits vorhanden, übersprungen.", f"{duplicates} already present, skipped.", f"{duplicates} ya presentes, omitidas.", f"{duplicates} déjà présentes, ignorées."))
    skipped = summary.get("skippedClasses") or {}
    if skipped:
        classes = ", ".join(f"{name} ({count})" for name, count in sorted(skipped.items()))
        parts.append(localized(f"Nicht übernommene Klassen: {classes}.", f"Classes not imported: {classes}.", f"Clases no importadas: {classes}.", f"Classes non importées : {classes}."))
    return " ".join(parts)


def augment_paths(root: Path) -> dict | None:
    """The source the Augmentation Studio works on and its work folder inside the project."""
    try:
        current = json.loads((root / "augment" / "current.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return current if current.get("source") and current.get("work") else None


def augment_read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def augment_select_source(root: Path, language: str) -> dict:
    venv_python = venv_python_path(ml_venv(root))
    if not venv_python.is_file():
        raise RuntimeError(localized("Für das Augmentation Studio zuerst „ML einrichten“ anklicken.", "For the Augmentation Studio, click “Set up ML” first.", "Para Augmentation Studio, pulsa primero «Configurar ML».", "Pour Augmentation Studio, cliquez d’abord sur « Configurer le ML »."))
    source = choose_video_folder(labels=True).resolve()
    scan = run_json([str(venv_python), str(find_ml_worker()), "augment-scan", "--source", str(source), "--language", language])
    work = root / "augment" / hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "augment" / "current.json", {"source": str(source), "work": str(work)})
    config = augment_read_json(work / "config.json") or {}
    config.setdefault("roles", scan["roles"])
    atomic_json(work / "config.json", config)
    return scan


def augment_state(root: Path) -> dict:
    """Everything the studio page shows. Instances only with what the gallery needs."""
    studio = augment_paths(root)
    project = load_project(root) or {}
    state: dict = {"source": None, "training": project.get("augmentation")}
    # What the studio needs before "Select source folder" can open a dialog (shown next to the button).
    if not venv_python_path(ml_venv(root)).is_file():
        state["needs"] = "ml"
    if studio is None:
        return state
    work = Path(studio["work"])
    analysis = augment_read_json(work / "analysis.json")
    if analysis:
        analysis["instances"] = [
            {key: item.get(key) for key in ("id", "session", "role", "team", "uncertain", "complete", "image")}
            for item in analysis["instances"] if item["role"] in {"player", "referee"}
        ]
        for session in analysis["sessions"]:
            plate = session["plate"]["files"]
            session["plateFile"] = plate.get("*") or next(iter(plate.values()), None)
            session.pop("plate", None)
            session.pop("sizeMap", None)
            session.pop("tiltMap", None)
    output = work / "output"
    previews = sorted(path.name for path in (output / "preview").glob("*.jpg")) if (output / "preview").is_dir() else []
    done = 0
    if (output / "manifest.jsonl").is_file():
        with (output / "manifest.jsonl").open(encoding="utf-8") as lines:
            done = sum(1 for _ in lines)
    state.update(source=studio["source"], config=augment_read_json(work / "config.json") or {}, analysis=analysis, report=augment_read_json(output / "report.json"), previews=previews, generated=done)
    return state


def augment_file(root: Path, kind: str, name: str) -> Path | None:
    studio = augment_paths(root)
    folder = {"thumb": "cache/thumbs", "plate": "cache/plates", "preview": "output/preview"}.get(kind)
    if studio is None or folder is None or not name:
        return None
    base = (Path(studio["work"]) / folder).resolve()
    target = (base / name).resolve()
    return target if base in target.parents and target.is_file() else None


AUGMENT_CONFIG_KEYS = {
    "variations", "cropShare", "zoom", "outputSize", "seed", "roles", "limits", "jerseyVariation", "yolo", "anonymize",
    "minVisible", "cropMinVisible", "sessionOverrides", "sessionLinks", "teamOverrides", "polygons", "splitOverrides",
}


def augment_save_config(root: Path, changes: dict) -> None:
    studio = augment_paths(root)
    if studio is None:
        raise RuntimeError(localized("Zuerst einen Quellordner wählen.", "Select a source folder first.", "Primero selecciona una carpeta de origen.", "Sélectionnez d’abord un dossier source."))
    path = Path(studio["work"]) / "config.json"
    config = augment_read_json(path) or {}
    config.update({key: value for key, value in changes.items() if key in AUGMENT_CONFIG_KEYS})
    atomic_json(path, config)


def augment_set_training(root: Path, payload: dict) -> None:
    studio = augment_paths(root)
    project = load_project(root)
    if project is None or studio is None:
        raise RuntimeError(localized("Zuerst einen Quellordner wählen.", "Select a source folder first.", "Primero selecciona una carpeta de origen.", "Sélectionnez d’abord un dossier source."))
    mode = payload.get("mode")
    if mode in {"mixed", "real-only"}:
        project["augmentation"] = {"mode": mode, "work": studio["work"], "realWeight": max(1, min(20, int(payload.get("realWeight", 3))))}
    else:
        project.pop("augmentation", None)
    atomic_json(root / "project.json", project)
    STATE.update(project=project)


def job_output_dirs(root: Path, action: str) -> list[Path]:
    """Folders a job writes into; after a cancel, entries it created or rewrote there are removed.
    Training is left out on purpose: its checkpoints are what a later run resumes from."""
    return {
        "export-coreml": [root / "exports" / "coreml"],
        "export-onnx": [root / "exports" / "onnx"],
        "export-reco": [root / "exports" / "reco"],
        "package-model": [root / "exports" / "share"],
        "combine-models": [root / "exports" / "share", root / "models" / "library"],
        "benchmark": [root / "benchmarks" / "runs"],
        "import-labels": [root / "frames" / "imported"],
        "import-model": [root / "models" / "library"],
        "import-onnx": [root / "models" / "library", root / "inbox"],
        "export-dataset": [root / "exports" / "datasets"],
        "download-models": [root / ".runtime" / "models"],
    }.get(action, [])


def export_result_for_ui(root: Path, summary: dict) -> dict:
    """What the export's result window needs; paths are shown, frames held back can be opened."""
    return {
        "kind": "dataset-export", "name": summary.get("name"), "folder": summary.get("folder"), "zip": summary.get("zip"),
        "images": summary.get("images"), "annotations": summary.get("annotations"), "anonymized": summary.get("anonymized"),
        "excludedFrames": [item.get("frameID") for item in summary.get("excludedForReview") or []],
        "models": summary.get("models") or [],
    }


def consistency_message(summary: dict) -> str:
    flagged = len(summary.get("review") or [])
    return localized(
        f"Konsistenzprüfung fertig: {flagged} von {summary.get('checked')} Bildern zur Prüfung markiert.",
        f"Consistency check finished: {flagged} of {summary.get('checked')} images flagged for review.",
        f"Comprobación de coherencia terminada: {flagged} de {summary.get('checked')} imágenes marcadas para revisar.",
        f"Contrôle de cohérence terminé : {flagged} images sur {summary.get('checked')} signalées pour vérification.",
    )


def export_message(summary: dict) -> str:
    held = len(summary.get("excludedForReview") or [])
    return localized(
        f"Datensatz exportiert: {summary.get('images')} Bilder, {summary.get('annotations')} Boxen." + (f" {held} Bilder wurden zur Prüfung zurückgehalten." if held else ""),
        f"Dataset exported: {summary.get('images')} images, {summary.get('annotations')} boxes." + (f" {held} images were held back for review." if held else ""),
        f"Conjunto exportado: {summary.get('images')} imágenes, {summary.get('annotations')} cuadros." + (f" {held} imágenes quedaron retenidas para revisión." if held else ""),
        f"Jeu de données exporté : {summary.get('images')} images, {summary.get('annotations')} boîtes." + (f" {held} images ont été retenues pour vérification." if held else ""),
    )


def cancelled_message(action: str) -> str:
    if action == "hub-share":
        return localized(
            "Hochladen abgebrochen. Release und Branch wurden aus deinem Fork entfernt.",
            "Upload cancelled. The release and the branch were removed from your fork.",
            "Subida cancelada. La versión y la rama se eliminaron de tu fork.",
            "Téléversement annulé. La version et la branche ont été retirées de votre fork.",
        )
    if action == "train-batch":
        return localized(
            "Batch-Training abgebrochen. Fertige Modelle und ihre Ergebnisse bleiben erhalten.",
            "Batch training cancelled. Finished models and their results are kept.",
            "Entrenamiento por lotes cancelado. Los modelos terminados y sus resultados se conservan.",
            "Entraînement par lots annulé. Les modèles terminés et leurs résultats sont conservés.",
        )
    if action == "train":
        return localized(
            "Training abgebrochen. Der zuletzt gespeicherte Checkpoint bleibt erhalten; „Lokal trainieren“ setzt dort fort.",
            "Training cancelled. The last saved checkpoint is kept; “Train locally” continues from there.",
            "Entrenamiento cancelado. Se conserva el último checkpoint guardado; «Entrenar localmente» continúa desde ahí.",
            "Entraînement annulé. Le dernier checkpoint enregistré est conservé ; « Entraîner localement » reprend à partir de là.",
        )
    if action == "setup":
        return localized(
            "Einrichtung abgebrochen. „ML einrichten“ erneut starten, um sie abzuschließen.",
            "Setup cancelled. Start “Set up ML” again to complete it.",
            "Configuración cancelada. Vuelve a iniciar «Configurar ML» para completarla.",
            "Configuration annulée. Relancez « Configurer le ML » pour la terminer.",
        )
    return localized(
        "Abgebrochen. Das Projekt ist unverändert; halbfertige Dateien wurden entfernt.",
        "Cancelled. The project is unchanged; half-written files were removed.",
        "Cancelado. El proyecto no ha cambiado; se eliminaron los archivos a medio escribir.",
        "Annulé. Le projet est inchangé ; les fichiers incomplets ont été supprimés.",
    )


CPU_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def torch_import_error(venv_python: Path) -> str | None:
    """None when `import torch` works inside the ML environment, else the last line of the error.

    A CUDA build of PyTorch can install cleanly and still fail to load (WinError 127 on
    c10_cuda.dll: NVIDIA driver too old for the CUDA version, missing Visual C++ runtime,
    a clashing DLL). Only importing it shows that.
    """
    try:
        result = subprocess.run(
            [str(venv_python), "-c", "import torch"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)
    if result.returncode == 0:
        return None
    lines = [line.strip() for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else f"exit code {result.returncode}"


def ensure_torch_importable(venv_python: Path) -> None:
    """After installing the ML packages: make sure PyTorch actually loads, else switch to the CPU build.

    The CPU build always works, just slower. If even that does not load, raise a clear error
    instead of leaving an environment that only fails later, in the middle of training.
    """
    error = torch_import_error(venv_python)
    if error is None:
        return
    STATE.append_log(localized(
        f"PyTorch lässt sich nicht laden ({error}). Wechsle zur CPU-Version von PyTorch …",
        f"PyTorch cannot be loaded ({error}). Switching to the CPU build of PyTorch …",
        f"No se puede cargar PyTorch ({error}). Cambiando a la versión de CPU de PyTorch …",
        f"PyTorch ne peut pas être chargé ({error}). Passage à la version CPU de PyTorch …",
    ))
    STATE.update(message=localized(
        "PyTorch mit CUDA lässt sich nicht laden - richte die CPU-Version ein …",
        "PyTorch with CUDA cannot be loaded - setting up the CPU build …",
        "No se puede cargar PyTorch con CUDA - configurando la versión de CPU …",
        "PyTorch avec CUDA ne peut pas être chargé - configuration de la version CPU …",
    ))
    run_logged([str(venv_python), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    install = [str(venv_python), "-m", "pip", "install", "torch", "torchvision"]
    if sys.platform != "darwin":
        install += ["--index-url", CPU_TORCH_INDEX_URL]
    run_logged(install)
    error = torch_import_error(venv_python)
    if error is not None:
        raise RuntimeError(localized(
            f"PyTorch lässt sich auch in der CPU-Version nicht laden: {error}\n"
            "Unter Windows hilft meist das „Microsoft Visual C++ Redistributable (x64)“ von Microsoft und ein Neustart.",
            f"PyTorch cannot be loaded even as the CPU build: {error}\n"
            "On Windows, installing the \u201cMicrosoft Visual C++ Redistributable (x64)\u201d from Microsoft and restarting usually helps.",
            f"PyTorch no se puede cargar ni siquiera en la versión de CPU: {error}\n"
            "En Windows suele ayudar instalar el \u201cMicrosoft Visual C++ Redistributable (x64)\u201d y reiniciar.",
            f"PyTorch ne peut pas être chargé même en version CPU : {error}\n"
            "Sous Windows, l’installation de « Microsoft Visual C++ Redistributable (x64) » puis un redémarrage aident généralement.",
        ))
    STATE.append_log(localized(
        "PyTorch (CPU) wurde eingerichtet. Das Training läuft damit, aber langsamer als mit einer funktionierenden GPU.",
        "PyTorch (CPU) is set up. Training works, but more slowly than with a working GPU.",
        "PyTorch (CPU) está configurado. El entrenamiento funciona, pero más lento que con una GPU operativa.",
        "PyTorch (CPU) est configuré. L’entraînement fonctionne, mais plus lentement qu’avec un GPU opérationnel.",
    ))


def wants_directml(nvidia_gpu: str | None) -> bool:
    """Windows x64 without a usable NVIDIA GPU: PyTorch only has the CPU there, but ONNX Runtime
    with DirectML reaches any DirectX 12 GPU (AMD, Intel), see accelerated_model in ml_worker."""
    return os.name == "nt" and not nvidia_gpu and platform.machine().lower() in {"amd64", "x86_64"} and amd_choice() == "yes"


def install_directml_runtime(venv_python: Path) -> None:
    """Swap onnxruntime for onnxruntime-directml (same module name, so only one may be installed).

    --force-reinstall: after an earlier setup both packages can share the files, and uninstalling
    one removes them for the other. Failure is not fatal - detection then simply stays on the CPU."""
    STATE.update(message=localized(
        "Richte ONNX Runtime mit DirectML ein (AMD- oder Intel-Grafikkarte) …",
        "Setting up ONNX Runtime with DirectML (AMD or Intel graphics) …",
        "Configurando ONNX Runtime con DirectML (gráfica AMD o Intel) …",
        "Configuration d’ONNX Runtime avec DirectML (carte graphique AMD ou Intel) …",
    ))
    try:
        run_logged([str(venv_python), "-m", "pip", "uninstall", "-y", "onnxruntime"])
        run_logged([str(venv_python), "-m", "pip", "install", "--force-reinstall", "--no-deps", "onnxruntime-directml"])
    except RuntimeError as error:
        STATE.append_log(localized(
            f"DirectML ließ sich nicht einrichten, die Erkennung läuft auf der CPU: {error}",
            f"DirectML could not be set up, detection runs on the CPU: {error}",
            f"No se pudo configurar DirectML, la detección se ejecuta en la CPU: {error}",
            f"DirectML n’a pas pu être configuré, la détection s’exécute sur le CPU : {error}",
        ))
        run_logged([str(venv_python), "-m", "pip", "install", "--force-reinstall", "--no-deps", "onnxruntime"])



# --- AMD / Intel graphics on Windows: ask once, test, report ------------------------------------

_GPU_CHOICE: dict = {}
ONNXRUNTIME_DISTRIBUTIONS = ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu")


def gpu_choice_file() -> Path | None:
    base = vlm.app_data_dir()
    return base / "gpu-choice.json" if base is not None else None


def amd_choice() -> str | None:
    """"yes" (use the AMD/Intel card: DirectML + Vulkan), "no" (CPU only) or None (not asked yet).
    Kept per computer, like the shared ML environment."""
    if "choice" not in _GPU_CHOICE:
        value = None
        path = gpu_choice_file()
        try:
            if path is not None and path.is_file():
                value = json.loads(path.read_text(encoding="utf-8")).get("amd")
        except (OSError, ValueError, AttributeError):
            value = None
        _GPU_CHOICE["choice"] = value if value in {"yes", "no"} else None
    return _GPU_CHOICE["choice"]


def save_amd_choice(choice: str) -> None:
    if choice not in {"yes", "no"}:
        raise RuntimeError(localized("Ungültige Auswahl.", "Invalid choice.", "Selección no válida.", "Choix non valide."))
    _GPU_CHOICE["choice"] = choice
    path = gpu_choice_file()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        card = (amd_offer() or {}).get("card")
        path.write_text(json.dumps({"amd": choice, "card": card, "decidedAt": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    # The hardware line and the tier depend on the choice.
    _ACCELERATOR_CACHE.clear()
    _TIER_CACHE.clear()


def amd_offer() -> dict | None:
    """{"card", "choice"} on Windows x64 without an NVIDIA GPU (so no CUDA) but with an AMD or
    Intel graphics card, else None. RECO_SIMULATE_AMD=1 pretends such a card (for developing the
    dialogs on another computer)."""
    if "card" not in _GPU_CHOICE:
        card = None
        if os.environ.get("RECO_SIMULATE_AMD") == "1":
            card = {"name": "AMD Radeon RX 7800 XT (simulated)", "vendor": "amd", "vramGB": 16.0}
        elif os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"} and not detect_nvidia_gpu_name():
            card = teachers.directml_graphics()
        _GPU_CHOICE["card"] = card
    card = _GPU_CHOICE["card"]
    return {"card": card, "choice": amd_choice()} if card else None


def gpu_environment() -> dict[str, str]:
    """Without a "yes", ml_worker leaves the AMD/Intel card alone: detection stays on PyTorch
    (CPU) and the VLM uses the CPU build. Variables the user set themselves win."""
    if amd_offer() is None or amd_choice() == "yes":
        return {}
    wanted = {"RECO_ONNX_INFERENCE": "off", "RECO_VLM_RUNTIME": "cpu"}
    return {key: value for key, value in wanted.items() if key not in os.environ}


# --- AMD training with ROCm on Windows (see amd_training.py) ----------------------------------

# ROCM_SDK_TARGET_FAMILY: AMD's rocm_sdk otherwise runs offload-arch in every process (data loaders
# too) to pick the target family, which fails on paths with spaces ("Reco Trainer") and fills the log
# with "Unknown command line argument". AMD's Windows build only has "custom", its fallback anyway.
ROCM_TRAINING_ENV = {"RECO_ONNX_INFERENCE": "off", "PYTHONUTF8": "1", "ROCM_SDK_TARGET_FAMILY": "custom"}


def rocm_folder() -> Path | None:
    return amd_training.env_dir(vlm.app_data_dir())


def rocm_status() -> dict | None:
    return amd_training.status(rocm_folder(), (amd_offer() or {}).get("card"))


def rocm_training_executable(default: Path) -> tuple[Path, bool]:
    """The ROCm environment's Python when AMD training is set up and switched on."""
    python = amd_training.training_python(rocm_folder()) if rocm_status() else None
    if python is None:
        return default, False
    STATE.append_log(localized(
        "Training auf der AMD-Karte (ROCm).", "Training on the AMD card (ROCm).",
        "Entrenamiento en la tarjeta AMD (ROCm).", "Entraînement sur la carte AMD (ROCm).",
    ))
    return python, True


def ensure_yolo_rocm(python: Path) -> bool:
    """YOLO in the AMD environment as well. False (and YOLO stays on the CPU) when that fails."""
    def check(code: str) -> bool:
        try:
            return subprocess.run([str(python), "-c", code], capture_output=True, timeout=300, env={**os.environ, **ROCM_TRAINING_ENV}).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    if check(amd_training.YOLO_CHECK):
        return True
    STATE.update(message=localized(
        "Installiere YOLO auch für das Training auf der AMD-Karte …", "Installing YOLO for training on the AMD card as well …",
        "Instalando YOLO también para entrenar en la tarjeta AMD …", "Installation de YOLO aussi pour l’entraînement sur la carte AMD …",
    ))
    folder = rocm_folder()
    try:
        if folder is None:
            raise RuntimeError("no AMD training folder")
        constraints = folder / "constraints.txt"
        constraints.write_text(amd_training.CONSTRAINTS, encoding="utf-8")
        has_cv2 = check("import cv2")
        for command in amd_training.yolo_install_steps(python, yolo_training.ULTRALYTICS_VERSION, constraints, has_cv2):
            run_logged(command)
    except Exception as error:  # noqa: BLE001 - the batch still runs, with YOLO on the CPU
        STATE.append_log(str(error))
    if check(amd_training.YOLO_CHECK):
        return True
    STATE.append_log(localized(
        "YOLO ließ sich für die AMD-Karte nicht einrichten. YOLO trainiert deshalb auf der CPU, RF-DETR weiter auf der Karte.",
        "YOLO could not be set up for the AMD card, so YOLO trains on the CPU; RF-DETR still trains on the card.",
        "No se pudo configurar YOLO para la tarjeta AMD, así que YOLO se entrena en la CPU; RF-DETR sigue en la tarjeta.",
        "YOLO n’a pas pu être configuré pour la carte AMD : YOLO s’entraîne donc sur le CPU, RF-DETR toujours sur la carte.",
    ))
    return False


def remember_rocm_problem(problem: str | None) -> None:
    folder = rocm_folder()
    if folder is None or not folder.is_dir():
        return
    state = amd_training.read_state(folder)
    if problem is None:
        state.pop("problem", None)
    else:
        state["problem"] = {"message": problem[-400:], "at": datetime.now(timezone.utc).isoformat()}
    try:
        amd_training.write_state(folder, state)
    except OSError:
        pass


def set_rocm_enabled(enabled: bool) -> None:
    folder = rocm_folder()
    if folder is None or not amd_training.ready(folder):
        raise RuntimeError(localized("AMD-Training ist nicht eingerichtet.", "AMD training is not set up.", "El entrenamiento AMD no está configurado.", "L’entraînement AMD n’est pas configuré."))
    state = amd_training.read_state(folder)
    state["enabled"] = bool(enabled)
    amd_training.write_state(folder, state)
    _ACCELERATOR_CACHE.clear()


def rocm_setup_action(payload: dict) -> None:
    """Python 3.12 venv with AMD's ROCm PyTorch and RF-DETR, then a check on the card."""
    STATE.update(operation="rocm-setup", busy=True, error=None, log=[], progress=0.0, message=localized(
        "Richte AMD-Training ein …", "Setting up AMD training …", "Configurando el entrenamiento AMD …", "Configuration de l’entraînement AMD …"))
    try:
        folder = rocm_folder()
        if folder is None or rocm_status() is None:
            raise RuntimeError(localized(
                "Auf diesem Rechner wurde keine AMD-Karte gefunden, die mit ROCm trainieren kann (Radeon RX 7000/9000, Windows).",
                "No AMD card that can train with ROCm was found on this computer (Radeon RX 7000/9000, Windows).",
                "No se encontró en este equipo ninguna tarjeta AMD que pueda entrenar con ROCm (Radeon RX 7000/9000, Windows).",
                "Aucune carte AMD capable d’entraîner avec ROCm n’a été trouvée sur cet ordinateur (Radeon RX 7000/9000, Windows).",
            ))
        main_python = venv_python_path(ml_venv(STATE.project_root))
        if not main_python.is_file():
            raise RuntimeError(localized("Zuerst „ML einrichten“ anklicken.", "Click “Set up ML” first.", "Primero pulsa «Configurar ML».", "Cliquez d’abord sur « Configurer le ML »."))
        python = amd_training.find_python312()
        if not python:
            raise RuntimeError(localized(
                "AMD-Training braucht Python 3.12 (AMD liefert PyTorch für Windows nur dafür). Bitte von python.org „Python 3.12“ (Windows 64-bit) installieren, „py launcher“ angehakt lassen, dann erneut einrichten. Andere Python-Versionen bleiben unverändert.",
                "AMD training needs Python 3.12 (AMD ships PyTorch for Windows for that version only). Please install “Python 3.12” (Windows 64-bit) from python.org, keep “py launcher” ticked, then set up again. Other Python versions stay as they are.",
                "El entrenamiento AMD necesita Python 3.12 (AMD solo ofrece PyTorch para Windows en esa versión). Instala «Python 3.12» (Windows 64 bits) desde python.org, deja marcado «py launcher» y vuelve a configurar. Las demás versiones de Python no cambian.",
                "L’entraînement AMD nécessite Python 3.12 (AMD ne fournit PyTorch pour Windows que pour cette version). Installez « Python 3.12 » (Windows 64 bits) depuis python.org, laissez « py launcher » coché, puis relancez la configuration. Les autres versions de Python restent inchangées.",
            ))
        rfdetr_version = subprocess.run(
            [str(main_python), "-c", "import importlib.metadata as m; print(m.version('rfdetr'))"],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout.strip()
        if not amd_training.venv_python(folder).is_file():
            folder.mkdir(parents=True, exist_ok=True)
            run_logged([python, "-m", "venv", str(folder / "venv")])
        constraints = folder / "constraints.txt"
        constraints.write_text(amd_training.CONSTRAINTS, encoding="utf-8")
        steps = amd_training.install_steps(amd_training.venv_python(folder), rfdetr_version, constraints)
        for number, (name, command) in enumerate(steps):
            STATE.update(progress=number / (len(steps) + 1), message=localized(
                f"AMD-Training: installiere {name} (insgesamt etwa 4 GB) …", f"AMD training: installing {name} (about 4 GB in total) …",
                f"Entrenamiento AMD: instalando {name} (unos 4 GB en total) …", f"Entraînement AMD : installation de {name} (environ 4 Go au total) …",
            ))
            run_logged(command)
        STATE.update(progress=len(steps) / (len(steps) + 1), message=localized(
            "AMD-Training: prüfe die Karte …", "AMD training: checking the card …", "Entrenamiento AMD: comprobando la tarjeta …", "Entraînement AMD : vérification de la carte …"))
        check = subprocess.run([str(amd_training.venv_python(folder)), "-c", amd_training.VERIFY_SCRIPT], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=600, check=False, env={**os.environ, **ROCM_TRAINING_ENV, "PYTHONIOENCODING": "utf-8"})
        try:
            result = json.loads(check.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result = {"available": False, "error": (check.stderr or check.stdout)[-1500:]}
        state = {"verified": result, "enabled": True, "rfdetr": rfdetr_version or amd_training.FALLBACK_RFDETR,
                 "installedAt": datetime.now(timezone.utc).isoformat()}
        amd_training.write_state(folder, state)
        STATE.append_log(json.dumps(result, ensure_ascii=False))
        if not amd_training.verified(result):
            raise RuntimeError(localized(
                "AMD-Training ist installiert, aber die Karte hat die Prüfung nicht bestanden. Training läuft weiter auf der CPU. Details stehen im Protokoll.",
                "AMD training is installed, but the card failed the check. Training stays on the CPU. Details are in the log.",
                "El entrenamiento AMD está instalado, pero la tarjeta no superó la comprobación. El entrenamiento sigue en la CPU. Los detalles están en el registro.",
                "L’entraînement AMD est installé, mais la carte n’a pas réussi la vérification. L’entraînement reste sur le CPU. Les détails sont dans le journal.",
            ))
        _ACCELERATOR_CACHE.clear()
        device = result.get("device") or ""
        STATE.update(operation="ready", busy=False, progress=1.0, message=localized(
            f"AMD-Training eingerichtet: „Lokal trainieren“ nutzt jetzt {device}.", f"AMD training is set up: “Train locally” now uses {device}.",
            f"Entrenamiento AMD configurado: «Entrenar localmente» usa ahora {device}.", f"Entraînement AMD configuré : « Entraîner localement » utilise maintenant {device}.",
        ))
    except OperationCancelled:
        message = cancelled_message("rocm-setup")
        STATE.append_log(message)
        STATE.update(operation="cancelled", busy=False, progress=0.0, error=None, message=message)
    except Exception as error:  # noqa: BLE001
        STATE.append_log(str(error))
        STATE.update(operation="error", busy=False, error=str(error), message=localized(
            "AMD-Training konnte nicht eingerichtet werden.", "AMD training could not be set up.",
            "No se pudo configurar el entrenamiento AMD.", "L’entraînement AMD n’a pas pu être configuré."))


def gpu_status() -> dict:
    problem = STATE.gpu_problem
    return {
        "offer": amd_offer(),
        "problem": {key: problem[key] for key in ("id", "reason", "operation", "at")} if problem else None,
        "diagnosis": STATE.gpu_diagnosis,
    }


GPU_JOB_KINDS = {"gpu-test", "gpu-diagnose"}


def record_job_failure(kind: str, error: str | None) -> None:
    """A job crashed outright (for example "exited with code 1") while the user had chosen to try
    the AMD/Intel graphics card. A hard crash never reaches the RECO_GPU_FALLBACK signal, so the
    diagnosis report is started from here instead."""
    if kind in GPU_JOB_KINDS or STATE.gpu_diagnosing or amd_choice() != "yes":
        return
    if kind in {"train", "train-batch", "rocm-setup"} and (rocm_status() or {}).get("enabled"):
        return  # ROCm training, not DirectML: the DirectML report would test the wrong thing
    if LAST_USER_ERROR["message"] and error and LAST_USER_ERROR["message"] in error:
        return  # a message for the user (for example "no flagged images"), not a crash
    record_gpu_problem(json.dumps({"reason": error or kind, "traceback": ""}), crashed=kind)


def record_gpu_problem(raw: str, crashed: str | None = None) -> None:
    """ml_worker fell back from the graphics card to the CPU. The first time per session (after a
    "yes"), a diagnosis report is queued right behind the running job."""
    try:
        data = json.loads(raw)
    except ValueError:
        data = {"reason": raw}
    if STATE.gpu_diagnosing:
        return  # the diagnosis reports its own failures
    with STATE.lock:
        problem = {
            "id": uuid.uuid4().hex[:12], "reason": str(data.get("reason") or "")[:500], "traceback": str(data.get("traceback") or ""),
            "operation": crashed or STATE.operation, "at": time.time(), "log": list(STATE.log[-150:]),
        }
        STATE.gpu_problem = problem
        start = amd_choice() == "yes" and not STATE.gpu_auto_diagnosed
        if start:
            STATE.gpu_auto_diagnosed = True
    if crashed:
        STATE.append_log(localized(
            "Der Job ist mit eingeschalteter Grafikkarte abgebrochen; ein Diagnosebericht wird erstellt.",
            "The job failed with the graphics card switched on; a diagnosis report is being created.",
            "La tarea falló con la gráfica activada; se está creando un informe de diagnóstico.",
            "La tâche a échoué avec la carte graphique activée ; un rapport de diagnostic est en cours.",
        ) if start else "")
    else:
        STATE.append_log(localized(
            f"Grafikkarte ausgefallen, weiter auf der CPU: {problem['reason']}",
            f"Graphics card failed, continuing on the CPU: {problem['reason']}",
            f"La gráfica falló, se continúa en la CPU: {problem['reason']}",
            f"La carte graphique a échoué, poursuite sur le CPU : {problem['reason']}",
        ))
    if start:
        JOBS.submit("gpu-diagnose", gpu_diagnose_action, {"trigger": "problem"}, cancellable=True)


def onnxruntime_distributions(venv_python: Path) -> list[str] | None:
    probe = (
        "import importlib.metadata as m, json\n"
        "found = []\n"
        f"for name in {list(ONNXRUNTIME_DISTRIBUTIONS)!r}:\n"
        "    try:\n"
        "        m.version(name); found.append(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        pass\n"
        "print(json.dumps(found))"
    )
    try:
        result = subprocess.run([str(venv_python), "-c", probe], capture_output=True, text=True, timeout=30, check=False)
        return json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def diagnostics_dir() -> Path:
    import tempfile
    base = vlm.app_data_dir()
    if base is None:
        base = (STATE.project_root / ".runtime") if STATE.project_root is not None else Path(tempfile.gettempdir()) / "reco-trainer"
    folder = base / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_gpu_diagnosis(trigger: str, extra: dict | None = None) -> dict:
    """Runs ml_worker gpu-diagnose (in the ML environment when there is one, so ONNX Runtime and
    RF-DETR can be tested) and keeps the summary for the status. A crash of the diagnosis itself
    (a graphics driver can take the process down) still produces a report from the log."""
    root = STATE.project_root
    venv_python = venv_python_path(ml_venv(root))
    executable = venv_python if venv_python.is_file() else system_python()
    folder = diagnostics_dir()
    with STATE.lock:
        problem = dict(STATE.gpu_problem) if STATE.gpu_problem else None
        log = list(STATE.log[-150:])
    context = {
        "trigger": trigger, "log": log, "choice": amd_choice(), "offer": amd_offer(),
        "accelerator": hardware_summary().get("accelerator"), "mlEnvironment": str(venv_python), "mlEnvironmentExists": venv_python.is_file(),
        **(extra or {}),
    }
    if problem:
        context["problem"] = {key: problem.get(key) for key in ("reason", "operation", "at", "traceback")}
        context["problemLog"] = problem.get("log") or []
    context_file = folder / f"context-{uuid.uuid4().hex[:8]}.json"
    context_file.write_text(json.dumps(context, ensure_ascii=False, default=str), encoding="utf-8")
    STATE.update(message=localized("Prüfe die Grafikkarte …", "Checking the graphics card …", "Comprobando la gráfica …", "Vérification de la carte graphique …"))
    command = [str(executable), str(find_ml_worker()), "gpu-diagnose", "--out", str(folder), "--context", str(context_file), "--app-version", app_version(), "--language", STATE.language]
    if root is not None:
        command += ["--project", str(root)]
    STATE.gpu_diagnosing = True
    try:
        summary = run_logged(command)
        if not summary:
            raise RuntimeError("the diagnosis returned no result")
    except OperationCancelled:
        raise
    except Exception as error:  # noqa: BLE001 - the report is most needed exactly when this fails
        identifier = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = folder / f"gpu-diagnosis-{identifier}.txt"
        with STATE.lock:
            lines = list(STATE.log[-250:])
        path.write_text("\n".join([
            "Reco Trainer - GPU diagnosis (AMD/Intel: DirectML + Vulkan)",
            f"The diagnosis process itself failed: {error}",
            "The last 'GPU diagnosis: ...' line below names the check that was running.", "",
            "== Context ==", json.dumps({key: value for key, value in context.items() if key not in {"log", "problemLog"}}, indent=2, ensure_ascii=False, default=str), "",
            "== Diagnosis log ==", *lines, "", "== Log before the diagnosis ==", *[str(line) for line in context.get("problemLog") or log],
        ]) + "\n", encoding="utf-8")
        summary = {"id": identifier, "ok": False, "reportPath": str(path), "trigger": trigger, "steps": [
            {"key": "process", "title": "Diagnosis process", "status": "failed", "summary": "", "error": str(error)},
        ]}
    finally:
        STATE.gpu_diagnosing = False
        context_file.unlink(missing_ok=True)
    STATE.update(gpu_diagnosis=summary)
    return summary


def gpu_test_action(payload: dict) -> None:
    """After "yes, try the AMD card": set up ML if needed (installs onnxruntime-directml), make sure
    exactly onnxruntime-directml is installed, then run the diagnosis. Its result opens the report
    popup in the app."""
    STATE.update(operation="gpu-test", busy=True, error=None, log=[], progress=0.0, message=localized(
        "Teste die Grafikkarte …", "Testing the graphics card …", "Probando la gráfica …", "Test de la carte graphique …"))
    try:
        root = STATE.project_root
        venv_python = venv_python_path(ml_venv(root))
        if root is not None and (root / "project.json").is_file() and not venv_python.is_file():
            ml_action("setup", payload)  # wants_directml() is true now that the answer is "yes"
            venv_python = venv_python_path(ml_venv(root))
            STATE.update(operation="gpu-test", busy=True, error=None)
        elif venv_python.is_file() and os.name == "nt" and onnxruntime_distributions(venv_python) != ["onnxruntime-directml"]:
            install_directml_runtime(venv_python)
        setup_log = list(STATE.log)
        summary = run_gpu_diagnosis(str(payload.get("trigger") or "test"), {"setupLog": setup_log})
    except OperationCancelled:
        raise
    except Exception as error:  # noqa: BLE001
        STATE.append_log(str(error))
        summary = run_gpu_diagnosis("test", {"setupLog": list(STATE.log), "setupError": str(error)})
    _ACCELERATOR_CACHE.clear()
    _TIER_CACHE.clear()
    STATE.update(operation="ready", busy=False, progress=1.0, message=localized(
        "Grafikkarte getestet: funktioniert." if summary.get("ok") else "Grafikkarte getestet: Probleme gefunden, Bericht ist bereit.",
        "Graphics card tested: it works." if summary.get("ok") else "Graphics card tested: problems found, the report is ready.",
        "Gráfica probada: funciona." if summary.get("ok") else "Gráfica probada: se encontraron problemas, el informe está listo.",
        "Carte graphique testée : elle fonctionne." if summary.get("ok") else "Carte graphique testée : problèmes détectés, le rapport est prêt.",
    ))


def gpu_diagnose_action(payload: dict) -> None:
    STATE.update(operation="gpu-diagnose", busy=True, error=None, progress=0.0)
    run_gpu_diagnosis(str(payload.get("trigger") or "manual"))
    STATE.update(operation="ready", busy=False, progress=1.0, message=localized(
        "Diagnosebericht der Grafikkarte ist bereit.", "The graphics card report is ready.",
        "El informe de la gráfica está listo.", "Le rapport de la carte graphique est prêt."))


def gpu_report_text() -> dict:
    """The text of the last report - only that file, never a path from the request."""
    summary = STATE.gpu_diagnosis or {}
    path = Path(str(summary.get("reportPath") or ""))
    if not summary or path.parent.resolve() != diagnostics_dir().resolve() or not path.is_file():
        raise RuntimeError(localized("Kein Diagnosebericht vorhanden.", "No diagnosis report available.", "No hay ningún informe de diagnóstico.", "Aucun rapport de diagnostic disponible."))
    return {"id": summary.get("id"), "path": str(path), "text": path.read_text(encoding="utf-8", errors="replace")}


def ml_action(action: str, payload: dict) -> None:
    output_dirs: list[Path] = []
    before: dict[Path, float] = {}
    rocm_used = False
    try:
        root = STATE.project_root
        if root is None or not (root / "project.json").is_file():
            raise RuntimeError(localized("Zuerst Videos lokal vorbereiten.", "Prepare videos locally first.", "Primero prepara los vídeos localmente.", "Préparez d’abord les vidéos localement."))
        output_dirs = job_output_dirs(root, action)
        before = output_snapshot(output_dirs)
        language = payload.get("language", "de")
        if language not in {"de", "en", "es", "fr"}:
            language = "de"
        model = payload.get("model", "nano")
        if yolo_training.is_variant(model):
            # "model" is the RF-DETR size; an active YOLO model is found through models/active.json.
            model = "nano"
        STATE.update(operation=action, busy=True, progress=0.0, error=None, log=[], message=localized("Lokaler ML-Vorgang startet …", "Local ML operation starting …", "Iniciando la operación local de ML …", "Démarrage de l’opération ML locale …"))
        venv = ml_venv(root)
        venv_python = venv_python_path(venv)
        if action == "setup":
            venv = ml_setup_target(root, venv)
            venv_python = venv_python_path(venv)
            # A venv built earlier with a too-old Python (e.g. macOS's 3.9) can never
            # install rfdetr>=1.9 ("No matching distribution found"): rebuild it.
            if venv_python.is_file():
                existing = python_version_of(venv_python)
                if existing is not None and existing < MIN_PYTHON:
                    STATE.append_log(localized(
                        "Die vorhandene ML-Umgebung nutzt ein zu altes Python und wird neu erstellt.\n",
                        "The existing ML environment uses a Python that is too old and will be recreated.\n",
                        "El entorno ML existente usa un Python demasiado antiguo y se volverá a crear.\n",
                        "L’environnement ML existant utilise un Python trop ancien et va être recréé.\n",
                    ))
                    shutil.rmtree(venv, ignore_errors=True)
            if not venv_python.is_file():
                venv.parent.mkdir(parents=True, exist_ok=True)
                run_logged([str(system_python()), "-m", "venv", str(venv)])
            # Plain `pip install torch` gives a CPU-only wheel on Windows -
            # PyPI only hosts CUDA-enabled torch builds for Linux, Windows
            # needs the dedicated index (pytorch.org/get-started/locally).
            # detect_device()/training already handled CUDA correctly *if*
            # torch itself had it; this is what makes sure it actually does.
            # Guarded by payload["useCuda"] (UI checkbox, default on) so a
            # user who hits trouble with the CUDA install (blocked network,
            # unsupported driver, ...) can opt back into the plain install
            # that always works, just slower.
            use_cuda = bool(payload.get("useCuda", True))
            cuda_index_url = "https://download.pytorch.org/whl/cu121"
            gpu_name = detect_nvidia_gpu_name() if (os.name == "nt" and use_cuda) else None
            if gpu_name:
                STATE.update(message=localized(
                    f"NVIDIA-GPU erkannt ({gpu_name}) - installiere PyTorch mit CUDA-Unterstützung …", f"NVIDIA GPU detected ({gpu_name}) - installing PyTorch with CUDA support …",
                    f"GPU NVIDIA detectada ({gpu_name}) - instalando PyTorch con soporte CUDA …", f"GPU NVIDIA détectée ({gpu_name}) - installation de PyTorch avec support CUDA …",
                ))
                try:
                    run_logged([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
                    run_logged([str(venv_python), "-m", "pip", "install", "torch", "--index-url", cuda_index_url])
                except RuntimeError as error:
                    # Not fatal - fall through to the normal install below,
                    # which still produces a working (CPU) setup.
                    STATE.append_log(localized(f"CUDA-PyTorch-Installation fehlgeschlagen, verwende CPU-Version: {error}", f"CUDA PyTorch install failed, using the CPU version instead: {error}", f"Falló la instalación de PyTorch con CUDA, usando la versión de CPU: {error}", f"Échec de l’installation de PyTorch CUDA, utilisation de la version CPU : {error}"))
                    STATE.update(message=localized("CUDA-Installation fehlgeschlagen, richte CPU-Version ein …", "CUDA install failed, setting up the CPU version …", "La instalación de CUDA falló, configurando la versión de CPU …", "L’installation CUDA a échoué, configuration de la version CPU …"))
            run_logged([
                str(venv_python), "-m", "pip", "install", "--upgrade", "pip",
                "rfdetr[train,onnx,coreml]>=1.9.0", "onnxruntime",
                *(["--extra-index-url", cuda_index_url] if gpu_name else []),
            ])
            if wants_directml(gpu_name):
                install_directml_runtime(venv_python)
            ensure_torch_importable(venv_python)
            # The accelerator label (see non_apple_accelerator()) is cached
            # per venv path - a re-run of setup (e.g. retrying with CUDA
            # switched off, or after fixing a failed CUDA install) just
            # reinstalled the same venv's torch, so any previously cached
            # label for it is now stale.
            _ACCELERATOR_CACHE.pop(str(venv_python), None)
            _ACCELERATOR_CACHE.pop("pre-setup", None)
        else:
            worker = find_ml_worker()
            executable = venv_python
            if action == "import-model":
                package_file = Path(str(payload.get("file", ""))).resolve()
                if not package_file.is_file() or package_file.suffix.lower() != ".recomodel":
                    raise RuntimeError(localized("Das ausgewählte Modellpaket ist nicht verfügbar.", "The selected model package is not available.", "El paquete de modelo seleccionado no está disponible.", "Le paquet de modèle sélectionné n’est pas disponible."))
                # Prefer the venv (not required, unlike autolabel/train/benchmark
                # below) so validate_model_package_file()'s torch.load(weights_only=True)
                # code-safety scan can actually run when a venv exists; it degrades
                # to checksum-only validation via the system Python otherwise.
                executable = venv_python if venv_python.is_file() else system_python()
                args = ["install-package", "--project", str(root), "--file", str(package_file), "--language", language]
            elif action == "import-onnx":
                onnx_file = Path(str(payload.get("file", ""))).resolve()
                if not onnx_file.is_file() or onnx_file.suffix.lower() != ".onnx":
                    raise RuntimeError(localized("Die ausgewählte ONNX-Datei ist nicht verfügbar.", "The selected ONNX file is not available.", "El archivo ONNX seleccionado no está disponible.", "Le fichier ONNX sélectionné n’est pas disponible."))
                try:
                    input_size = int(payload.get("inputSize") or 640)
                except (TypeError, ValueError):
                    input_size = 640
                # Packaging and installing an ONNX model needs no ML libraries.
                executable = system_python()
                args = [
                    "install-onnx", "--project", str(root), "--file", str(onnx_file),
                    "--name", str(payload.get("name", "")), "--class-names", str(payload.get("classNames", "")),
                    "--input-size", str(input_size), "--license", str(payload.get("license", "")), "--language", language,
                ]
            elif action in {"augment-analyze", "augment-generate"}:
                if not venv_python.is_file():
                    raise RuntimeError(localized("Für das Augmentation Studio zuerst „ML einrichten“ anklicken.", "For the Augmentation Studio, click “Set up ML” first.", "Para Augmentation Studio, pulsa primero «Configurar ML».", "Pour Augmentation Studio, cliquez d’abord sur « Configurer le ML »."))
                studio = augment_paths(root)
                if studio is None:
                    raise RuntimeError(localized("Zuerst einen Quellordner wählen.", "Select a source folder first.", "Primero selecciona una carpeta de origen.", "Sélectionnez d’abord un dossier source."))
                args = [action, "--project", str(root), "--source", studio["source"], "--work", studio["work"], "--language", language]
            elif action == "export-dataset":
                anonymized = bool(payload.get("anonymize"))
                if anonymized and not venv_python.is_file():
                    raise RuntimeError(localized("Für einen anonymisierten Export zuerst „ML einrichten“ anklicken.", "For an anonymized export, click “Set up ML” first.", "Para una exportación anonimizada, primero haz clic en «Configurar ML».", "Pour un export anonymisé, cliquez d’abord sur « Configurer le ML »."))
                # A plain export only copies files and needs no ML libraries.
                executable = venv_python if venv_python.is_file() else system_python()
                formats = [item for item in payload.get("formats") or ["yolo", "coco"] if item in {"yolo", "coco"}] or ["yolo", "coco"]
                args = ["export-dataset", "--project", str(root), "--formats", ",".join(formats), "--app-version", app_version(), "--language", language]
                if anonymized:
                    args += ["--anonymize", "--method", "pixelate" if payload.get("method") == "pixelate" else "gaussian"]
                    if payload.get("includeText"):
                        args.append("--include-text")
                    if payload.get("detectPeople", True):
                        args.append("--detect-people")
                if payload.get("zip", True):
                    args.append("--zip")
                vlm_sample = float(payload.get("vlmSample") or 0)
                if anonymized and 0 < vlm_sample <= 1:
                    args += ["--vlm-sample", f"{vlm_sample:.2f}", "--vlm-model", "qwen3.5-9b-vlm" if payload.get("vlmModel") == "qwen3.5-9b-vlm" else "qwen3.5-4b-vlm"]
            elif action == "download-models":
                ids = [str(item) for item in payload.get("ids") or [] if re.fullmatch(r"[a-z0-9-]{1,60}", str(item))]
                if not ids:
                    raise RuntimeError(localized("Keine Modelle ausgewählt.", "No models selected.", "No se seleccionó ningún modelo.", "Aucun modèle sélectionné."))
                # Plain downloads need only the standard library; Hugging Face models need
                # huggingface_hub from the ML environment.
                registry = {entry["id"]: entry for entry in model_registry_snapshot(None)}
                needs_hub = any(registry.get(item, {}).get("source", "").startswith("https://huggingface.co/") for item in ids)
                executable = venv_python if needs_hub and venv_python.is_file() else system_python()
                args = ["download-models", "--project", str(root), "--ids", ",".join(ids), "--language", language]
            elif action == "import-labels":
                source = Path(str(payload.get("source", ""))).resolve()
                if not source.is_dir():
                    raise RuntimeError(localized("Der ausgewählte Ordner ist nicht verfügbar.", "The selected folder is not available.", "La carpeta seleccionada no está disponible.", "Le dossier sélectionné n’est pas disponible."))
                # Standard library only - the system Python is enough, no ML environment needed.
                executable = system_python()
                args = ["import-labels", "--project", str(root), "--source", str(source), "--format", "auto", "--language", language]
                if payload.get("candidates"):
                    args.append("--candidates")
            elif action == "hub-install":
                executable = venv_python if venv_python.is_file() else system_python()
                args = ["hub-install", "--project", str(root), "--size", str(payload.get("size", "")), "--id", str(payload.get("id", "")), "--language", language]
            elif action == "hub-share":
                executable = venv_python if venv_python.is_file() else system_python()
                args = ["hub-share", "--project", str(root), "--package", str(payload.get("packageID", "")), "--name", str(payload.get("name", "")), "--description", str(payload.get("description", "")), "--version", str(payload.get("version", "1.0.0")), "--camera", str(payload.get("camera", "")), "--venue", str(payload.get("venue", "")), "--games", str(int(payload.get("games", 0))), "--trainer-version", str(payload.get("trainerVersion", "")), "--language", language]
                if payload.get("rightsConfirmed") is True:
                    args.append("--rights-confirmed")
            elif action == "package-model":
                executable = venv_python if venv_python.is_file() else system_python()
                args = ["package", "--project", str(root), "--model", model, "--name", str(payload.get("name", "")), "--language", language]
            elif action == "combine-models":
                executable = venv_python if venv_python.is_file() else system_python()
                members = payload.get("members") or []
                if not isinstance(members, list) or not members:
                    raise RuntimeError(localized("Mindestens zwei Modelle für ein kombiniertes Modell auswählen.", "Select at least two models for a combined model.", "Selecciona al menos dos modelos para un modelo combinado.", "Sélectionnez au moins deux modèles pour un modèle combiné."))
                member_args = []
                for member in members:
                    package_id = str((member or {}).get("packageID", ""))
                    categories = (member or {}).get("categories") or []
                    if not package_id or not isinstance(categories, list) or not categories:
                        raise RuntimeError(localized("Ungültige Kategorie-Zuordnung für ein kombiniertes Modell.", "Invalid category assignment for a combined model.", "Asignación de categorías no válida para un modelo combinado.", "Attribution de catégories non valide pour un modèle combiné."))
                    member_args.extend(["--member", f"{package_id}:{','.join(str(category) for category in categories)}"])
                args = ["combine-models", "--project", str(root), *member_args, "--name", str(payload.get("name", "")), "--language", language]
            elif action in {"activate-model", "rename-model", "delete-model"}:
                executable = system_python()
                package_id = str(payload.get("packageID", ""))
                args = [action, "--project", str(root), "--package-id", package_id, "--language", language]
                if action == "rename-model":
                    args.extend(["--name", str(payload.get("name", ""))])
            elif not venv_python.is_file():
                raise RuntimeError(localized("ML-Umgebung fehlt. Zuerst „ML einrichten“ anklicken.", "The ML environment is missing. Click “Set up ML” first.", "Falta el entorno de ML. Primero haz clic en “Configurar ML”.", "L’environnement ML est manquant. Cliquez d’abord sur « Configurer le ML »."))
            elif action == "autolabel":
                backup_project(root, "automatisch")
                threshold = min(0.95, max(0.05, float(payload.get("threshold", 0.25))))
                categories = payload.get("categories") or [payload.get("category", "ball")]
                if not isinstance(categories, list) or not categories:
                    raise RuntimeError(localized("Mindestens eine Klasse für die automatische Markierung auswählen.", "Select at least one class for auto-labeling.", "Selecciona al menos una clase para el marcado automático.", "Sélectionnez au moins une classe pour le marquage automatique."))
                args = ["autolabel", "--project", str(root), "--model", model, "--category", *[str(item) for item in categories], "--threshold", str(threshold), "--language", language]
                if payload.get("candidateOnly"):
                    args.append("--candidate-only")
                if payload.get("compareModels"):
                    args.append("--compare-models")
                args += ["--tier", tier_argument(payload)]
                if payload.get("teacher") in {"nano", "small", "medium", "large"}:
                    args += ["--teacher", str(payload["teacher"])]
                if payload.get("refineMethod") in {"sam2", "opencv"}:
                    args += ["--refine-method", str(payload["refineMethod"])]
            elif action == "benchmark":
                threshold = min(0.95, max(0.01, float(payload.get("threshold", 0.05))))
                args = ["benchmark", "--project", str(root), "--threshold", str(threshold), "--language", language]
            elif action == "train":
                backup_project(root, "training")
                args = ["train", "--project", str(root), "--model", model, "--epochs", str(int(payload.get("epochs", 20))), "--language", language]
                executable, rocm_used = rocm_training_executable(executable)
                if payload.get("freshStart"):
                    args.append("--fresh-start")
                fraction = payload.get("anonymizeFraction")
                if isinstance(fraction, (int, float)) and 0 < fraction <= 0.5:
                    args += ["--anonymize-fraction", f"{float(fraction):.2f}"]
            elif action == "train-batch":
                models = [item for item in payload.get("models") or [] if item in {"nano", "small", "medium", "large"} or yolo_training.is_variant(item)]
                if not models:
                    raise RuntimeError(localized("Mindestens ein Modell auswählen.", "Select at least one model.", "Selecciona al menos un modelo.", "Sélectionnez au moins un modèle."))
                rocm_python = None
                if any(yolo_training.is_variant(item) for item in models):
                    ensure_yolo(venv_python, bool(payload.get("yoloLicenseAccepted")))
                    rocm_python = amd_training.training_python(rocm_folder()) if rocm_status() else None
                    if rocm_python is not None and ensure_yolo_rocm(rocm_python):
                        # YOLO is in the AMD environment too: the whole batch trains on the card.
                        rocm_python = None
                        executable, rocm_used = rocm_training_executable(executable)
                    elif all(yolo_training.is_variant(item) for item in models):
                        rocm_python = None
                    # Otherwise the batch runs in the normal environment and starts just the
                    # RF-DETR models with the AMD environment's Python.
                else:
                    executable, rocm_used = rocm_training_executable(executable)
                backup_project(root, "training")
                args = ["train-batch", "--project", str(root), "--models", ",".join(models), "--epochs", str(max(1, min(500, int(payload.get("epochs", 100))))), "--patience", str(max(3, min(50, int(payload.get("patience", 10))))), "--language", language]
                if rocm_python is not None:
                    args += ["--rfdetr-python", str(rocm_python)]
                if payload.get("autoEpochs"):
                    args.append("--auto-epochs")
                if payload.get("freshStart"):
                    args.append("--fresh-start")
                if payload.get("sweep"):
                    args.append("--sweep")
                if any(yolo_training.is_variant(item) for item in models):
                    image_size = int(payload.get("imageSize") or 640)
                    args += ["--image-size", str(image_size if image_size in yolo_training.IMAGE_SIZES else 640)]
                fraction = payload.get("anonymizeFraction")
                if isinstance(fraction, (int, float)) and 0 < fraction <= 0.5:
                    args += ["--anonymize-fraction", f"{float(fraction):.2f}"]
            elif action == "export-coreml":
                args = ["export", "--project", str(root), "--model", model, "--format", "coreml", "--language", language]
            elif action == "export-onnx":
                args = ["export", "--project", str(root), "--model", model, "--format", "onnx", "--language", language]
            elif action == "export-reco":
                classes = [str(item) for item in (payload.get("classes") or ["ball"]) if str(item).strip()]
                args = ["export-reco", "--project", str(root), "--package", str(payload.get("packageID", "")), "--classes", ",".join(classes) or "ball", "--language", language]
            elif action == "consistency-check":
                iou = min(0.9, max(0.1, float(payload.get("iou", 0.5))))
                sample = min(0.5, max(0.0, float(payload.get("sample", 0.05))))
                args = ["consistency-check", "--project", str(root), "--iou", f"{iou:.2f}", "--sample", f"{sample:.2f}", "--tier", tier_argument(payload), "--language", language]
                if payload.get("vlm"):
                    args += ["--vlm", "--vlm-model", "qwen3.5-9b-vlm" if payload.get("vlmModel") == "qwen3.5-9b-vlm" else "qwen3.5-4b-vlm"]
            elif action == "consistency-suggest":
                backup_project(root, "konsistenz-vorschlaege")
                args = ["consistency-suggest", "--project", str(root), "--tier", tier_argument(payload), "--language", language]
            elif action == "refine-boxes":
                backup_project(root, "boxen")
                args = ["refine-boxes", "--project", str(root), "--language", language, "--tier", tier_argument(payload)]
                if payload.get("refineMethod") in {"sam2", "opencv"}:
                    args += ["--method", str(payload["refineMethod"])]
            else:
                raise RuntimeError(localized("Unbekannter ML-Vorgang.", "Unknown ML operation.", "Operación de ML desconocida.", "Opération ML inconnue."))
            if action == "import-labels":
                summary = run_json([str(executable), str(worker), *args])
                message = label_import_message(summary, bool(payload.get("candidates")))
                STATE.append_log(message)
                STATE.update(project=load_project(root), folder_empty=False, operation="ready", busy=False, progress=1.0, message=message)
                return
            command = [str(executable), str(worker), *args]
            result = run_logged(command, ROCM_TRAINING_ENV) if rocm_used else run_logged(command)
            if rocm_used:
                remember_rocm_problem(None)
            if action == "consistency-suggest" and result:
                STATE.update(project=load_project(root), operation="ready", busy=False, progress=1.0, message=localized(
                    f"Vorschläge für {result.get('suggestions')} von {result.get('checked')} markierten Bildern berechnet.",
                    f"Suggestions computed for {result.get('suggestions')} of {result.get('checked')} flagged images.",
                    f"Sugerencias calculadas para {result.get('suggestions')} de {result.get('checked')} imágenes marcadas.",
                    f"Suggestions calculées pour {result.get('suggestions')} images signalées sur {result.get('checked')}.",
                ))
                return
            if action == "consistency-check" and result:
                STATE.update(result={"kind": "consistency", **result}, project=load_project(root), operation="ready", busy=False, progress=1.0, message=consistency_message(result))
                return
            if action == "export-dataset" and result:
                STATE.update(result=export_result_for_ui(root, result), message=export_message(result))
                STATE.update(project=load_project(root), operation="ready", busy=False, progress=1.0)
                return
        project = load_project(root)
        STATE.update(project=project, operation="ready", busy=False, progress=1.0, message=localized("Lokaler Vorgang abgeschlossen.", "Local operation completed.", "Operación local completada.", "Opération locale terminée."))
    except OperationCancelled:
        remove_job_outputs(output_dirs, before)
        message = cancelled_message(action)
        STATE.append_log(message)
        root = STATE.project_root
        STATE.update(project=load_project(root) if root else STATE.project, operation="cancelled", busy=False, progress=0.0, error=None, message=message)
    except Exception as error:
        STATE.append_log(str(error))
        if rocm_used and not (LAST_USER_ERROR["message"] and LAST_USER_ERROR["message"] in str(error)):
            remember_rocm_problem(str(error))
            error = RuntimeError(f"{error}\n\n" + localized(
                "Das Training auf der AMD-Karte ist abgebrochen. Zum Trainieren auf der CPU „Auf der AMD-Karte trainieren“ ausschalten und erneut starten.",
                "Training on the AMD card stopped. To train on the CPU, switch off “Train on the AMD card” and start again.",
                "El entrenamiento en la tarjeta AMD se detuvo. Para entrenar en la CPU, desactiva «Entrenar en la tarjeta AMD» y vuelve a empezar.",
                "L’entraînement sur la carte AMD s’est arrêté. Pour entraîner sur le CPU, désactivez « Entraîner sur la carte AMD » et relancez.",
            ))
        STATE.update(operation="error", busy=False, error=str(error), message=localized("Lokaler ML-Vorgang fehlgeschlagen.", "Local ML operation failed.", "La operación local de ML falló.", "L’opération ML locale a échoué."))


MAX_FIELD_POINTS = 32


def outline_self_intersects(points: list[list[float]]) -> bool:
    """True when two non-neighbouring edges of the closed outline cross (points clicked
    out of order); such an outline has no clear inside."""
    def orientation(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    count = len(points)
    edges = [(points[index], points[(index + 1) % count]) for index in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            if second == first + 1 or (first == 0 and second == count - 1):
                continue  # neighbours share a point
            (a, b), (c, d) = edges[first], edges[second]
            d1, d2 = orientation(c, d, a), orientation(c, d, b)
            d3, d4 = orientation(a, b, c), orientation(a, b, d)
            if d1 * d2 < 0 and d3 * d4 < 0:
                return True
    return False


def save_field_geometry(payload: dict) -> None:
    """Store the marked field outline in project.json, in the shape
    ml_worker.field_membership_checker reads: at least four fractional (0..1) image
    points joined by straight lines. No real sizes are needed.

    payload["video"] names the video (camera position) the outline belongs to;
    payload["applyToAll"] stores it for every video of the project. Without a video
    (an older client) it applies to all videos, as the single outline used to."""
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError(localized("Kein lokales Projekt geöffnet.", "No local project is open.", "No hay ningún proyecto local abierto.", "Aucun projet local n’est ouvert."))
    if payload.get("remove"):
        remove_field_geometry(root, project, payload)
        return
    invalid = localized("Ungültige Spielfeld-Angaben.", "Invalid field boundaries.", "Límites del campo no válidos.", "Limites du terrain non valides.")
    try:
        corners = [[float(point[0]), float(point[1])] for point in payload.get("corners") or []]
    except (TypeError, ValueError, IndexError):
        raise RuntimeError(invalid) from None
    finite = all(math.isfinite(value) for point in corners for value in point)
    if not 4 <= len(corners) <= MAX_FIELD_POINTS or not finite or any(not 0.0 <= value <= 1.0 for point in corners for value in point):
        raise RuntimeError(invalid)
    if outline_self_intersects(corners):
        raise RuntimeError(localized(
            "Die Kanten des Spielfelds kreuzen sich. Bitte die Punkte der Reihe nach am Rand entlang setzen.",
            "The field's edges cross each other. Please place the points in order along the edge.",
            "Los bordes del campo se cruzan. Coloca los puntos en orden a lo largo del borde.",
            "Les bords du terrain se croisent. Placez les points dans l’ordre le long du bord.",
        ))
    backup_project(root, "spielfeld")
    videos = field_geometry.project_videos(project)
    video = payload.get("video")
    if not videos:
        project["fieldGeometry"] = {"corners": corners}  # no video yet: the default for the first ones
    elif payload.get("applyToAll") or video is None:
        field_geometry.set_geometry(project, videos, {"corners": corners})
    else:
        if str(video) not in videos:
            raise RuntimeError(localized("Unbekanntes Video.", "Unknown video.", "Vídeo desconocido.", "Vidéo inconnue."))
        field_geometry.set_geometry(project, [str(video)], {"corners": corners})
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    STATE.update(project=project, message=localized("Spielfeld lokal gespeichert.", "Field boundaries saved locally.", "Límites del campo guardados localmente.", "Limites du terrain enregistrées localement."))


def remove_field_geometry(root: Path, project: dict, payload: dict) -> None:
    """Delete the field outline of one video, or of all with applyToAll (backup first)."""
    videos = field_geometry.project_videos(project)
    video = payload.get("video")
    if videos and not payload.get("applyToAll") and video is not None and str(video) not in videos:
        raise RuntimeError(localized("Unbekanntes Video.", "Unknown video.", "Vídeo desconocido.", "Vidéo inconnue."))
    backup_project(root, "spielfeld-entfernt")
    field_geometry.clear_geometry(project, None if not videos or payload.get("applyToAll") or video is None else [str(video)])
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    STATE.update(project=project, message=localized("Spielfeldgrenze entfernt.", "Field boundary removed.", "Límite del campo eliminado.", "Limite du terrain supprimée."))


def reveal_training_folder(export_name: str | None = None) -> None:
    """Open the project's visible training folder in the system file manager
    (the web counterpart of the Mac app's "Show training folder"), or one dataset export."""
    root = STATE.project_root
    if root is None:
        raise RuntimeError(localized("Zuerst einen Arbeitsordner auswählen.", "Select a working folder first.", "Primero selecciona una carpeta de trabajo.", "Sélectionnez d’abord un dossier de travail."))
    if Path("/.dockerenv").exists():
        raise RuntimeError(localized(
            "Im Docker-Container kann kein Ordner geöffnet werden. Der Ordner „Reco Training“ liegt im eingebundenen Arbeitsordner.",
            "A folder cannot be opened from inside the Docker container. The “Reco Training” folder is inside the mounted working folder.",
            "No se puede abrir una carpeta desde el contenedor de Docker. La carpeta «Reco Training» está dentro de la carpeta de trabajo montada.",
            "Impossible d’ouvrir un dossier depuis le conteneur Docker. Le dossier « Reco Training » se trouve dans le dossier de travail monté.",
        ))
    visible = root.parent / "Reco Training"
    target = visible if visible.exists() else root
    if export_name is not None:
        exports = (root / "exports" / "datasets").resolve()
        target = (exports / export_name).resolve()
        if exports not in target.parents or not target.is_dir():
            raise RuntimeError(localized("Der Export wurde nicht gefunden.", "The export was not found.", "No se encontró la exportación.", "L’export est introuvable."))
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else shutil.which("xdg-open")
    if not opener:
        raise RuntimeError(localized(
            "Kein Dateimanager gefunden (xdg-open fehlt).", "No file manager found (xdg-open is missing).",
            "No se encontró un gestor de archivos (falta xdg-open).", "Aucun gestionnaire de fichiers trouvé (xdg-open manquant).",
        ))
    subprocess.Popen([opener, str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def clean_annotations(project: dict, annotations: list) -> list[dict]:
    allowed = set(SPORT_CATEGORIES.get(project.get("sport"), []))
    clean = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        category = str(annotation.get("category", ""))
        if category not in allowed:
            continue
        clean.append({
            "id": str(annotation.get("id") or uuid.uuid4()), "category": category,
            "x": max(0.0, float(annotation.get("x", 0))), "y": max(0.0, float(annotation.get("y", 0))),
            "width": max(1.0, float(annotation.get("width", 1))), "height": max(1.0, float(annotation.get("height", 1))),
            "confidence": annotation.get("confidence"), "source": str(annotation.get("source", "manual")),
        })
    return clean


CONSISTENCY_DECISIONS = {"accept", "reject", "edited"}


def review_consistency_frame(payload: dict) -> None:
    """One decision in the step-by-step review of consistency-check flags: "accept" stores
    the suggested boxes the UI sends, "edited" keeps the boxes as corrected by hand, "reject"
    keeps the boxes unchanged. In every case the frame's flags and suggestion are cleared and
    the decision is recorded, so the frame leaves the review list. Deleting an image goes
    through /api/remove-frame."""
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError(localized("Kein lokales Projekt geöffnet.", "No local project is open.", "No hay ningún proyecto local abierto.", "Aucun projet local n’est ouvert."))
    if STATE.busy:
        raise RuntimeError(localized(
            "Bitte warten, bis der laufende Vorgang fertig ist.", "Please wait until the running operation has finished.",
            "Espera a que termine la operación en curso.", "Veuillez attendre la fin de l’opération en cours.",
        ))
    decision = str(payload.get("decision", ""))
    if decision not in CONSISTENCY_DECISIONS:
        raise RuntimeError(localized("Unbekannte Entscheidung.", "Unknown decision.", "Decisión desconocida.", "Décision inconnue."))
    frame_id = str(payload.get("frameId", ""))
    frame = next((item for item in project.get("frames", []) if item.get("id") == frame_id), None)
    if frame is None:
        raise RuntimeError(localized("Frame wurde nicht gefunden.", "Frame was not found.", "No se encontró el fotograma.", "L’image est introuvable."))
    annotations = payload.get("annotations")
    changed = decision == "accept" and isinstance(annotations, list)
    if changed:
        backup_project(root, "konsistenz")
        frame["annotations"] = clean_annotations(project, annotations)
    flags = frame.pop("reviewFlags", None) or []
    frame.pop("consistencySuggestion", None)
    frame["consistencyReview"] = {"decision": decision, "flags": flags, "at": utc_now()}
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    remaining = sum(1 for item in project.get("frames", []) if item.get("reviewFlags"))
    STATE.update(project=project, message=localized(
        f"Entscheidung gespeichert. Noch {remaining} Bilder zu prüfen.", f"Decision saved. {remaining} images left to review.",
        f"Decisión guardada. Quedan {remaining} imágenes por revisar.", f"Décision enregistrée. Encore {remaining} images à vérifier.",
    ))


def save_annotations(payload: dict) -> None:
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError(localized("Kein lokales Projekt geöffnet.", "No local project is open.", "No hay ningún proyecto local abierto.", "Aucun projet local n’est ouvert."))
    frame_id = str(payload.get("frameId", ""))
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise RuntimeError(localized("Ungültige Markierungen.", "Invalid annotations.", "Anotaciones no válidas.", "Annotations non valides."))
    frame = next((item for item in project.get("frames", []) if item.get("id") == frame_id), None)
    if frame is None:
        raise RuntimeError(localized("Frame wurde nicht gefunden.", "Frame was not found.", "No se encontró el fotograma.", "L’image est introuvable."))
    backup_project(root, "markierung")
    clean = clean_annotations(project, annotations)
    frame["annotations"] = clean
    # "Image is empty": a frame without boxes only trains when this was said (see ml_worker.trains_on).
    if clean or payload.get("emptyConfirmed") is False:
        frame.pop("emptyConfirmed", None)
    elif payload.get("emptyConfirmed") is True:
        frame["emptyConfirmed"] = True
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    STATE.update(project=project, message=localized(
        f"{len(clean)} Markierungen lokal gespeichert.", f"{len(clean)} annotations saved locally.",
        f"{len(clean)} anotaciones guardadas localmente.", f"{len(clean)} annotations enregistrées localement.",
    ))


def remove_training_frame(payload: dict) -> None:
    """Remove one derived training image without ever touching a source video."""
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError(localized("Kein lokales Projekt geöffnet.", "No local project is open.", "No hay ningún proyecto local abierto.", "Aucun projet local n’est ouvert."))
    frame_id = str(payload.get("frameId", ""))
    frames = project.get("frames", [])
    index = next((number for number, item in enumerate(frames) if str(item.get("id")) == frame_id), None)
    if index is None:
        raise RuntimeError(localized("Frame wurde nicht gefunden.", "Frame was not found.", "No se encontró el fotograma.", "L’image est introuvable."))
    frame = frames[index]
    is_candidate = frame.get("reviewStatus") == "candidate"
    target = (root / str(frame.get("relativePath", ""))).resolve()
    frames_root = (root / "frames").resolve()
    if frames_root not in target.parents:
        raise RuntimeError(localized("Ungültiger Frame-Pfad.", "Invalid frame path.", "Ruta de fotograma no válida.", "Chemin d’image non valide."))

    backup_project(root, "bild-entfernt")
    target.unlink(missing_ok=True)
    for split in ("train", "valid", "test"):
        (root / "dataset" / split / target.name).unlink(missing_ok=True)
    frames.pop(index)
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)

    # A frozen benchmark no longer represents the current image set.
    benchmark_invalidated = False
    if not is_candidate:
        for name in ("ground-truth.json", "latest.json"):
            benchmark_file = root / "benchmarks" / name
            if benchmark_file.is_file():
                benchmark_file.unlink()
                benchmark_invalidated = True
    message = localized(
        "Trainingsbild lokal entfernt. Das Quellvideo bleibt unverändert.", "Training image removed locally. The source video remains unchanged.",
        "Imagen de entrenamiento eliminada localmente. El vídeo de origen no cambia.", "Image d’entraînement supprimée localement. La vidéo source reste inchangée.",
    )
    if benchmark_invalidated:
        message += " " + localized(
            "Die Benchmark-Referenz muss neu festgelegt werden.", "The benchmark reference must be frozen again.",
            "Hay que volver a fijar la referencia de comparación.", "La référence de comparaison doit être refigée.",
        )
    STATE.update(project=project, message=message)


def review_candidate(payload: dict) -> None:
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError(localized("Kein lokales Projekt geöffnet.", "No local project is open.", "No hay ningún proyecto local abierto.", "Aucun projet local n’est ouvert."))
    frame_id = str(payload.get("frameId", ""))
    decision = str(payload.get("decision", ""))
    frame = next((item for item in project.get("frames", []) if str(item.get("id")) == frame_id), None)
    if frame is None or frame.get("reviewStatus") != "candidate":
        raise RuntimeError(localized("Prüfkandidat wurde nicht gefunden.", "Review candidate was not found.", "No se encontró el candidato de revisión.", "Le candidat de révision est introuvable."))
    target = str(payload.get("category") or "ball")
    if decision == "ball":
        matching = [item for item in frame.get("annotations", []) if item.get("category") == target]
        if not matching:
            raise RuntimeError(localized("Zuerst eine passende Ball-Box einzeichnen oder korrigieren.", "Draw or correct a matching ball box first.", "Primero dibuja o corrige un cuadro de balón adecuado.", "Dessinez ou corrigez d’abord une boîte de ballon adaptée."))
        for annotation in matching:
            annotation["source"] = "manual"
    elif decision == "no-ball":
        frame["annotations"] = [item for item in frame.get("annotations", []) if item.get("category") != target]
    elif decision == "confirm":
        # Held-out/independent-validation frames can carry several categories'
        # boxes, already corrected directly in the box editor - unlike "ball"/
        # "no-ball", this just accepts whatever is there instead of filtering
        # by a single target category.
        for annotation in frame.get("annotations", []):
            annotation["source"] = "manual"
    else:
        raise RuntimeError(localized("Unbekannte Prüfentscheidung.", "Unknown review decision.", "Decisión de revisión desconocida.", "Décision de révision inconnue."))
    backup_project(root, "kandidat-geprueft")
    frame["reviewStatus"] = "reviewed"
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    STATE.update(project=project, message=localized(
        "Geprüftes Bild wurde in den Trainingssatz übernommen.", "Reviewed image was added to the training set.",
        "La imagen revisada se añadió al conjunto de entrenamiento.", "L’image vérifiée a été ajoutée au jeu d’entraînement.",
    ))


class Handler(BaseHTTPRequestHandler):
    def handle(self) -> None:
        # A browser closing a keep-alive connection is normal, not an error worth a traceback.
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    server_version = "RecoLocalWorker/0.1"

    def log_message(self, format: str, *args) -> None:
        return

    def allowed_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if origin in ALLOWED_ORIGINS else None

    def add_cors(self) -> None:
        origin = self.allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def json_response(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.add_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise RuntimeError(localized("Anfrage ist zu groß.", "The request is too large.", "La solicitud es demasiado grande.", "La requête est trop volumineuse."))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_OPTIONS(self) -> None:
        if not self.allowed_origin():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.add_cors()
        self.end_headers()

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/api/status":
            self.json_response(STATE.snapshot())
            return
        if route.path == "/api/gpu-report":
            try:
                self.json_response(gpu_report_text())
            except RuntimeError as error:
                self.json_response({"error": str(error)}, HTTPStatus.NOT_FOUND)
            return
        if route.path == "/api/jobs":
            self.json_response(JOBS.snapshot())
            return
        if route.path == "/api/events":
            self.stream_events()
            return
        if route.path == "/api/frame":
            root = STATE.project_root
            project = STATE.project or {}
            frame_id = (parse_qs(route.query).get("id") or [""])[0]
            frame = next((item for item in project.get("frames", []) if item.get("id") == frame_id), None)
            if root is None or frame is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target = (root / frame["relativePath"]).resolve()
            frames_root = (root / "frames").resolve()
            if frames_root not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.add_cors()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if route.path == "/api/hub/models":
            # Only ever on request: the hub is the one place where the app goes online by itself.
            root = STATE.project_root
            document = load_project(root) if root else None
            if document is None:
                self.json_response({"models": []})
                return
            query = parse_qs(route.query)
            sport = str(document.get("sport", ""))
            classes = sorted({annotation.get("category") for frame in document.get("frames", []) for annotation in frame.get("annotations", []) if annotation.get("category")})
            index = model_hub.fetch_index(root / "hub" / "index.json")
            self.json_response({
                "models": model_hub.filter_models(index, sport, classes, query.get("version", ["0"])[0]),
                "offline": bool(index.get("offline")), "notYetPublished": bool(index.get("notYetPublished")), "sport": sport,
                "comparison": document.get("hubComparison"),
            })
            return
        if route.path == "/api/batch-report":
            root = STATE.project_root
            path = root / "batch-training" / "latest.json" if root else None
            self.json_response(json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else {})
            return
        if route.path == "/api/augment/state":
            root = STATE.project_root
            self.json_response(augment_state(root) if root and (root / "project.json").is_file() else {"source": None, "needs": "project"})
            return
        if route.path == "/api/augment/file":
            root = STATE.project_root
            query = parse_qs(route.query)
            target = augment_file(root, (query.get("kind") or [""])[0], (query.get("name") or [""])[0]) if root else None
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.add_cors()
            self.send_header("Content-Type", "image/png" if target.suffix == ".png" else "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if route.path == "/api/simulation-frame":
            root = STATE.project_root
            file_name = (parse_qs(route.query).get("file") or [""])[0]
            if root is None or not file_name:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            slot = "b" if (parse_qs(route.query).get("slot") or ["a"])[0] == "b" else "a"
            frames_dir = simulation_frames_dir(root, slot).resolve()
            target = (frames_dir / file_name).resolve()
            if frames_dir not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.add_cors()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.serve_static(route.path)

    def stream_events(self) -> None:
        """Server-sent events: the job list (current job with progress, queue, recent results)
        whenever it changes, at most ~4 times a second, plus a keep-alive comment. The full
        /api/status stays the fallback and the source for project data."""
        self.send_response(HTTPStatus.OK)
        self.add_cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        seen = -1
        last_sent = 0.0
        try:
            while True:
                with CHANGED:
                    if _CHANGE_COUNTER == seen:
                        CHANGED.wait(timeout=15)
                    counter = _CHANGE_COUNTER
                if counter == seen:
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    pause = 0.25 - (time.time() - last_sent)
                    if pause > 0:
                        time.sleep(pause)
                    seen = counter
                    last_sent = time.time()
                    payload = json.dumps(JOBS.snapshot(), ensure_ascii=False)
                    self.wfile.write(f"event: jobs\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def serve_static(self, request_path: str) -> None:
        """Serve the built interface (ui/) so end users need neither Node.js nor npm.

        Unknown paths without a file extension fall back to index.html; anything
        that resolves outside ui/ is refused. Range requests are supported because
        browsers need them to play and seek the walkthrough videos.
        """
        if request_path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not (UI_DIR / "index.html").is_file():
            body = (
                "Reco Trainer: Die Oberfläche (Ordner ui/) fehlt. / The interface (ui/ folder) is missing. "
                "Run `npm install && npm run build` or download a release package."
            ).encode("utf-8")
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        relative = unquote(request_path).lstrip("/")
        target = (UI_DIR / relative).resolve() if relative else UI_DIR / "index.html"
        if target.is_dir():
            target = target / "index.html"
        inside = UI_DIR == target or UI_DIR in target.parents
        if not inside:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            if Path(relative).suffix:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target = UI_DIR / "index.html"
        size = target.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        range_header = self.headers.get("Range", "")
        if match := re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip()):
            first, last = match.groups()
            if first == "" and last == "":
                match = None
            elif first == "":
                start, end = max(0, size - int(last)), size - 1
            else:
                start, end = int(first), min(int(last) if last else size - 1, size - 1)
            if match is not None:
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        immutable = "assets" in target.relative_to(UI_DIR).parts[:1]
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if immutable else "no-cache")
        self.end_headers()
        with target.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_POST(self) -> None:
        if not self.allowed_origin():
            self.json_response({"error": localized(
                "Nur die lokale Reco-Oberfläche darf diesen Worker aufrufen.",
                "Only the local Reco interface is allowed to call this worker.",
                "Solo la interfaz local de Reco puede llamar a este proceso.",
                "Seule l’interface Reco locale est autorisée à appeler ce processus.",
            )}, HTTPStatus.FORBIDDEN)
            return
        try:
            payload = self.read_json()
            if language := payload.get("language"):
                if language in {"de", "en", "es", "fr"}:
                    STATE.language = language
            if self.path == "/api/select-folder":
                if JOBS.has_pending():
                    raise RuntimeError(localized("Bitte warten, bis die laufenden Vorgänge beendet sind.", "Please wait until the running operations have finished.", "Espera a que terminen las operaciones en curso.", "Veuillez attendre la fin des opérations en cours."))
                folder = select_folder()
                self.json_response({"ok": True, "folder": str(folder), "status": STATE.snapshot()})
            elif self.path == "/api/cancel":
                job = JOBS.cancel(payload.get("jobId") or None)
                self.json_response({"ok": True, "job": job})
            elif self.path == "/api/simulation-cancel":
                job = cancel_simulation(payload)
                self.json_response({"ok": True, "job": job})
            elif match := re.fullmatch(r"/api/jobs/([0-9a-f]{12})/cancel", self.path):
                job = JOBS.cancel(match.group(1))
                self.json_response({"ok": True, "job": job})
            elif self.path in {"/api/augment/select-source", "/api/augment/config", "/api/augment/training"}:
                root = STATE.project_root
                if root is None or not (root / "project.json").is_file():
                    raise RuntimeError(localized("Zuerst den Arbeitsordner vorbereiten.", "Prepare the working folder first.", "Primero prepara la carpeta de trabajo.", "Préparez d’abord le dossier de travail."))
                if self.path == "/api/augment/select-source":
                    augment_select_source(root, STATE.language)
                elif self.path == "/api/augment/config":
                    augment_save_config(root, payload.get("config") or {})
                else:
                    augment_set_training(root, payload)
                self.json_response({"ok": True, "augment": augment_state(root)})
            elif self.path in {"/api/augment/analyze", "/api/augment/generate"}:
                action = "augment-" + self.path.rsplit("/", 1)[1]
                job = JOBS.submit(action, ml_action, action, payload)
                self.json_response({"ok": True, "job": job})
            elif self.path == "/api/prepare":
                count = frames_per_video(payload.get("framesPerVideo"))
                if STATE.selected_folder:
                    # Files may have been added since selecting; an empty folder is a hint, not a failed job.
                    STATE.update(folder_empty=folder_is_empty(STATE.selected_folder))
                    if STATE.folder_empty:
                        self.json_response({"ok": True, "job": None, "status": STATE.snapshot()})
                        return
                job = JOBS.submit("prepare", extract_project, payload.get("sport", "basketball"), count)
                self.json_response({"ok": True, "job": job})
            elif self.path == "/api/active-learning":
                folder = choose_video_folder(expansion=True)
                job = JOBS.submit("active-learning", expand_dataset, folder, payload)
                self.json_response({"ok": True, "folder": str(folder), "job": job})
            elif self.path == "/api/independent-validation":
                folder = choose_video_folder(expansion=True)
                job = JOBS.submit("independent-validation", expand_dataset, folder, payload, held_out=True)
                self.json_response({"ok": True, "folder": str(folder), "job": job})
            elif self.path == "/api/simulate-ball-tracking":
                video = simulation_video(payload)
                job = JOBS.submit("simulate-ball-tracking", simulate_ball_tracking, video, payload)
                self.json_response({"ok": True, "video": str(video), "job": job})
            elif self.path == "/api/simulation-gaps":
                job = JOBS.submit("simulation-gaps", simulation_gaps_to_candidates, payload)
                self.json_response({"ok": True, "job": job})
            elif self.path == "/api/consistency-review":
                review_consistency_frame(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/annotations":
                save_annotations(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/field-geometry":
                save_field_geometry(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/reveal-training-folder":
                reveal_training_folder(str(payload["exportName"]) if payload.get("exportName") else None)
                self.json_response({"ok": True})
            elif self.path == "/api/remove-frame":
                if STATE.busy:
                    raise RuntimeError(localized("Ein lokaler Vorgang läuft bereits.", "A local operation is already running.", "Ya hay una operación local en curso.", "Une opération locale est déjà en cours."))
                remove_training_frame(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/review-candidate":
                if STATE.busy:
                    raise RuntimeError(localized("Ein lokaler Vorgang läuft bereits.", "A local operation is already running.", "Ya hay una operación local en curso.", "Une opération locale est déjà en cours."))
                review_candidate(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/benchmark-ground-truth":
                if STATE.busy:
                    raise RuntimeError(localized("Ein lokaler Vorgang läuft bereits.", "A local operation is already running.", "Ya hay una operación local en curso.", "Une opération locale est déjà en cours."))
                self.json_response({"ok": True, "benchmark": freeze_ground_truth()})
            elif self.path == "/api/import-model":
                package_file = select_model_package()
                job = JOBS.submit("import-model", ml_action, "import-model", {**payload, "file": str(package_file)})
                self.json_response({"ok": True, "fileName": package_file.name, "job": job})
            elif self.path == "/api/import-onnx":
                if not str(payload.get("classNames", "")).strip():
                    raise RuntimeError(localized("Bitte die Klassen des Modells angeben.", "Please enter the model's classes.", "Indica las clases del modelo.", "Veuillez indiquer les classes du modèle."))
                onnx_file = select_model_package(".onnx")
                job = JOBS.submit("import-onnx", ml_action, "import-onnx", {**payload, "file": str(onnx_file)})
                self.json_response({"ok": True, "fileName": onnx_file.name, "job": job})
            elif self.path == "/api/import-labels":
                if STATE.selected_folder is None or STATE.project_root is None:
                    raise RuntimeError(localized("Zuerst einen Arbeitsordner auswählen.", "Select a working folder first.", "Primero selecciona una carpeta de trabajo.", "Sélectionnez d’abord un dossier de travail."))
                label_folder = choose_video_folder(labels=True)
                # A dataset can be the first thing in a working folder: the project starts with it.
                if not (STATE.project_root / "project.json").is_file():
                    start_empty_project(STATE.selected_folder, STATE.project_root, str(payload.get("sport", "")))
                job = JOBS.submit("import-labels", ml_action, "import-labels", {**payload, "source": str(label_folder)})
                self.json_response({"ok": True, "folder": str(label_folder), "job": job})
            elif self.path == "/api/gpu-choice":
                choice = str(payload.get("choice", ""))
                save_amd_choice(choice)
                job = JOBS.submit("gpu-test", gpu_test_action, payload) if choice == "yes" else None
                self.json_response({"ok": True, "job": job, "status": STATE.snapshot()})
            elif self.path == "/api/rocm-setup":
                job = JOBS.submit("rocm-setup", rocm_setup_action, payload)
                self.json_response({"ok": True, "job": job})
            elif self.path == "/api/rocm-enabled":
                set_rocm_enabled(bool(payload.get("enabled")))
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/gpu-diagnose":
                job = JOBS.submit("gpu-diagnose", gpu_diagnose_action, {"trigger": "manual"})
                self.json_response({"ok": True, "job": job})
            elif self.path in {"/api/setup", "/api/autolabel", "/api/train", "/api/train-batch", "/api/hub-install", "/api/hub-share", "/api/export-coreml", "/api/export-onnx", "/api/export-reco", "/api/refine-boxes", "/api/package-model", "/api/combine-models", "/api/benchmark", "/api/activate-model", "/api/rename-model", "/api/delete-model", "/api/download-models", "/api/export-dataset", "/api/consistency-check", "/api/consistency-suggest"}:
                action = self.path.removeprefix("/api/")
                job = JOBS.submit(action, ml_action, action, payload, cancellable=action not in QUICK_ACTIONS)
                self.json_response({"ok": True, "job": job})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def python_version_warning() -> str | None:
    """Flag a Python older than 3.10 without touching platform launcher scripts.

    Video browsing/upload works on any Python 3, and "Set up ML" picks a suitable
    Python (>= 3.10, 3.12/3.11 preferred) on its own - it only fails when none is
    installed at all.
    """
    if sys.version_info[:2] >= MIN_PYTHON:
        return None
    try:
        system_python()
        return None
    except RuntimeError:
        pass
    return (
        f"Python {sys.version_info[0]}.{sys.version_info[1]} erkannt und kein Python 3.10 oder neuer gefunden "
        "(empfohlen: 3.12). Video-Vorschau und -Upload funktionieren, aber „ML einrichten“ benötigt Python 3.10+."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reco Trainer loopback worker")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if HOST not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        raise SystemExit("RECO_BIND_HOST muss 127.0.0.1, localhost oder 0.0.0.0 sein.")
    # Terminal-only (not STATE.message): at process start no request has
    # happened yet, so STATE.language is still its "de" default - showing
    # this hardcoded-German text as the initial UI status regardless of the
    # user's actual language would be exactly the kind of hardwired-language
    # bug fixed elsewhere in this file (see localized()). system_python()
    # already raises a properly localized error later if the version
    # actually turns out to matter for "Set up ML".
    warning = python_version_warning()
    if warning:
        print(f"WARNUNG: {warning}", flush=True)
    # The interface is served from this very port, so its own origin must be allowed.
    ALLOWED_ORIGINS.update({f"http://localhost:{args.port}", f"http://127.0.0.1:{args.port}"})
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    server.daemon_threads = True

    def shut_down(_signum, _frame) -> None:
        # Jobs run in their own process groups; stop them too, or a training run would
        # outlive the app (the launcher only signals this worker's own group).
        CANCEL_REQUESTED.set()
        terminate_active_processes()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shut_down)
    print(f"Reco Local Worker: http://{HOST}:{args.port}", flush=True)
    print(f"Reco Trainer interface: http://localhost:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
