#!/usr/bin/env python3
"""Local-only ML worker for Reco Trainer.

The worker deliberately has no upload command. It reads and writes only below the
project directory selected by the user.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import struct
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sibling modules (teachers.py, anonymize.py) must import however this file is loaded -
# as a script, from the payload, or by tests from another folder.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
}


def emit(message: str) -> None:
    print(message, flush=True)


def localized(
    language: str,
    german: str,
    english: str,
    spanish: str | None = None,
    french: str | None = None,
) -> str:
    if language == "de":
        return german
    if language == "es":
        return spanish or english
    if language == "fr":
        return french or english
    return english


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def project_file(project_root: Path) -> Path:
    return project_root / "project.json"


def require_project(project_root: Path, language: str = "de") -> dict[str, Any]:
    path = project_file(project_root)
    if not path.is_file():
        raise SystemExit(localized(language, f"Projektdatei fehlt: {path}", f"Project file is missing: {path}"))
    return load_json(path)


def import_model_class(size: str, language: str = "de"):
    try:
        import rfdetr
    except ImportError as error:
        raise SystemExit(localized(
            language,
            "RF-DETR ist nicht installiert. In der App zuerst „ML einrichten“ wählen.",
            "RF-DETR is not installed. Select “Set up ML” in the app first.",
        )) from error
    try:
        return getattr(rfdetr, {**MODEL_CLASSES, **BASE_DETECTOR_CLASSES}[size])
    except AttributeError as error:
        raise SystemExit(localized(
            language,
            f"Die installierte RF-DETR-Version unterstützt {size!r} nicht.",
            f"The installed RF-DETR version does not support {size!r}.",
        )) from error


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def total_memory_bytes() -> int:
    """Return physical memory without reading project or media files."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def training_profile(
    model_size: str,
    device: str,
    memory_bytes: int | None = None,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    """Choose one fast, stable training job instead of competing GPU jobs.

    The MPS memory breakpoints (12/20/36/72 GB) correspond to common Apple
    unified-memory tiers (8/16 GB, 18/24 GB, 32/48 GB, 64/96+ GB machines);
    batch sizes were chosen empirically to stay well under each tier's actual
    usable memory rather than from a formula, since RF-DETR's real memory use
    also depends on image resolution and model size in ways not worth modeling
    exactly here. Gradient checkpointing trades compute for memory on tiers
    where it doesn't fit otherwise.
    """
    memory_bytes = total_memory_bytes() if memory_bytes is None else memory_bytes
    cpu_count = (os.cpu_count() or 1) if cpu_count is None else max(cpu_count, 1)
    memory_gb = max(1, round(memory_bytes / (1024**3))) if memory_bytes else 16

    if device == "mps":
        if memory_gb <= 12:
            batch_size = 1
        elif memory_gb <= 20:
            batch_size = 2 if model_size == "nano" else 1
        elif memory_gb <= 36:
            batch_size = 4 if model_size == "nano" else 2
        elif memory_gb <= 72:
            batch_size = 8 if model_size == "nano" else 4
        else:
            batch_size = 16 if model_size == "nano" else 8
        num_workers = min(8, max(2, cpu_count // 2))
        gradient_checkpointing = (
            memory_gb <= 12 or (model_size == "small" and memory_gb < 32)
        )
    elif device == "cuda":
        batch_size = 4 if model_size == "nano" else 2
        num_workers = min(8, max(2, cpu_count // 2))
        gradient_checkpointing = model_size == "small"
    else:
        batch_size = 1
        num_workers = min(4, max(1, cpu_count // 2))
        gradient_checkpointing = model_size == "small"

    return {
        "batch_size": batch_size,
        "grad_accum_steps": max(1, math.ceil(16 / batch_size)),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 3 if num_workers > 0 else None,
        "pin_memory": device == "cuda",
        "gradient_checkpointing": gradient_checkpointing,
        "memory_gb": memory_gb,
        "cpu_count": cpu_count,
    }


class EpochProgressWatcher:
    """Reports "epoch n of m" while RF-DETR trains, read from the metrics.csv its CSV logger
    writes into the run folder (its own progress bar uses carriage returns, which a piped
    log never sees). Only a file written after training started counts."""

    def __init__(self, metrics_file: Path, target_epochs: int, language: str) -> None:
        import threading
        self.metrics_file = metrics_file
        self.target = max(int(target_epochs), 1)
        self.language = language
        self.started = time.time()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.reported = -1

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=2)

    def latest_epoch(self) -> int | None:
        try:
            if self.metrics_file.stat().st_mtime < self.started:
                return None
            with self.metrics_file.open(encoding="utf-8", errors="replace") as handle:
                rows = csv.reader(handle)
                header = next(rows, [])
                if "epoch" not in header:
                    return None
                column = header.index("epoch")
                epochs = [int(float(row[column])) for row in rows if len(row) > column and row[column]]
            return max(epochs) if epochs else None
        except (OSError, ValueError):
            return None

    def run(self) -> None:
        while not self.stopped.wait(3.0):
            epoch = self.latest_epoch()
            if epoch is None or epoch == self.reported:
                continue
            self.reported = epoch
            current = min(epoch + 1, self.target)
            emit(localized(self.language, f"Epoche {current} von {self.target}", f"Epoch {current} of {self.target}", f"Época {current} de {self.target}", f"Époque {current} sur {self.target}"))
            print(f"RECO_PROGRESS {min(epoch, self.target)} {self.target}", file=sys.stderr, flush=True)


def call_with_supported_kwargs(function, kwargs: dict[str, Any]) -> tuple[Any, list[str]]:
    """Use newer RF-DETR performance switches while staying compatible with older installs."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs), []
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**kwargs), []
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    ignored = sorted(set(kwargs) - set(supported))
    return function(**supported), ignored


def numeric_metrics(value: Any) -> dict[str, float]:
    """Flatten numeric evaluation output without depending on one RF-DETR release."""
    output: dict[str, float] = {}
    if not isinstance(value, dict):
        return output
    for key, item in value.items():
        try:
            if hasattr(item, "item"):
                item = item.item()
            number = float(item)
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(number):
            output[str(key)] = number
    return output


def explicitly_supported_kwargs(function, candidates: dict[str, Any]) -> dict[str, Any]:
    """Return optional arguments named by the public callable signature.

    Some RF-DETR releases expose ``**kwargs`` but forward them to an older inner
    exporter. Treating arbitrary keywords as supported therefore is not safe for
    export. Required, long-standing arguments are supplied separately.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    return {key: value for key, value in candidates.items() if key in parameters}


def doctor(_: argparse.Namespace) -> None:
    torch_installed = importlib.util.find_spec("torch") is not None
    rfdetr_installed = importlib.util.find_spec("rfdetr") is not None
    mps_available = False
    cuda_available = False
    gpu_name: str | None = None
    if torch_installed:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
        # Previously only checked MPS, so Windows/Linux always reported
        # "cpu" here regardless of an available NVIDIA GPU - detect_device()
        # (the function actually used to pick the training/inference device)
        # already checked CUDA correctly; this diagnostic just hadn't caught
        # up with it, leaving Windows users with no way to see their GPU was
        # (or wasn't) actually usable.
        try:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError):
            cuda_available = False
    device = "mps" if mps_available else "cuda" if cuda_available else "cpu"
    profile = training_profile("nano", device)
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torchInstalled": torch_installed,
        "rfdetrInstalled": rfdetr_installed,
        "mpsAvailable": mps_available,
        "cudaAvailable": cuda_available,
        "gpuName": gpu_name,
        "recommendedDevice": device,
        "cpuCores": profile["cpu_count"],
        "memoryGB": profile["memory_gb"],
        "trainingBatchSize": profile["batch_size"],
        "dataWorkers": profile["num_workers"],
    }
    print(json.dumps(payload, ensure_ascii=False))


def preferred_run_checkpoint(run_dir: Path) -> Path | None:
    """Pick the single best checkpoint out of one RF-DETR run directory.

    RF-DETR names its best/last checkpoints consistently; only fall back to the
    newest file by mtime if none of those conventional names exist (e.g. a run
    that used a custom callback or an older RF-DETR release).
    """
    if not run_dir.is_dir():
        return None
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint.pth", "last.ckpt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    candidates = [*run_dir.rglob("*.pth"), *run_dir.rglob("*.ckpt")]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def newest_checkpoint(project_root: Path, size: str) -> Path | None:
    """Return the checkpoint that autolabel/benchmark/train should treat as current.

    The activated library revision (see activate_library_model) wins whenever it
    matches the requested model size; otherwise fall back to whatever the local
    ``runs/<size>`` directory holds, which is the mutable in-progress training run.
    """
    active_file = project_root / "models" / "active.json"
    library_root = (project_root / "models" / "library").resolve()
    if active_file.is_file():
        try:
            active = load_json(active_file)
            active_path = (project_root / str(active.get("weights", ""))).resolve()
            if (
                active.get("modelSize") == size
                and library_root in active_path.parents
                and active_path.is_file()
                and active_path.suffix in {".pth", ".ckpt"}
            ):
                return active_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return preferred_run_checkpoint(project_root / "runs" / size)


def active_model_classes(project_root: Path, size: str) -> set[str] | None:
    """Classes the currently activated library model for this size was actually trained on.

    None means "unknown" rather than "supports nothing" - either there is no activated
    library model for this size (e.g. only a raw, not-yet-packaged runs/<size> checkpoint
    exists), or its manifest could not be read. Callers should fall back to their own
    default assumption in that case instead of treating every category as unsupported.
    """
    try:
        active = load_json(project_root / "models" / "active.json")
        if active.get("modelSize") != size and active.get("format") != ONNX_FORMAT:
            return None
        library_root = (project_root / "models" / "library").resolve()
        manifest_path = (library_root / str(active["packageID"]) / "manifest.json").resolve()
        if library_root not in manifest_path.parents or not manifest_path.is_file():
            return None
        manifest = load_json(manifest_path)
        return {str(item) for item in manifest.get("classes", [])}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def active_model_manifest(project_root: Path, size: str) -> dict[str, Any] | None:
    """The full manifest of the currently activated library model for this size,
    or None if there is no such activation (see active_model_classes, which
    returns just the classes - auto_label()/benchmark() need the full manifest
    to detect an "ensemble" combined model, which isn't backed by a single
    weights file the way a normal package is).
    """
    try:
        active = load_json(project_root / "models" / "active.json")
        if active.get("modelSize") != size and active.get("format") != ONNX_FORMAT:
            return None
        library_root = (project_root / "models" / "library").resolve()
        manifest_path = (library_root / str(active["packageID"]) / "manifest.json").resolve()
        if library_root not in manifest_path.parents or not manifest_path.is_file():
            return None
        return load_json(manifest_path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def ensemble_member_model(package_dir: Path, member: dict[str, Any], language: str = "de"):
    """Load one ensemble member's own checkpoint.

    A combined/ensemble package's manifest.json's weightsFile is relative to
    that combined package's own library directory, not the original source
    model's - combine_models() copies every member's weights into the new
    package at build time (see combine_models), so a combined package is
    self-contained like any other .recomodel.
    """
    model_class = import_model_class(str(member["modelSize"]), language)
    weight_path = (package_dir / str(member["weightsFile"])).resolve()
    if package_dir.resolve() not in weight_path.parents or not weight_path.is_file():
        raise SystemExit(localized(
            language,
            f"Gewichtsdatei für kombiniertes Modell fehlt: {member.get('packageID')}",
            f"Weights file for combined model is missing: {member.get('packageID')}",
        ))
    # Unconditional, regardless of whether this exact file was already scanned
    # at install time (it may not have been, if torch wasn't set up yet back
    # then, or the file reached the library some other way) - this is the
    # actual gate that stands between an untrusted weights file and RF-DETR's
    # own unsafe (non-weights_only) torch.load() below. torch is guaranteed
    # importable here already (model_class itself required it), so this never
    # silently degrades at this call site the way it can at install time.
    verify_weights_pickle_safety(weight_path, language, str(member.get("packageID") or weight_path.name))
    return model_class(pretrain_weights=str(weight_path), device=detect_device())


ONNX_FORMAT = "onnx-yolo"


def is_onnx_manifest(manifest: dict[str, Any] | None) -> bool:
    return isinstance(manifest, dict) and manifest.get("format") == ONNX_FORMAT


class OnnxDetections:
    """Minimal stand-in for the detections object RF-DETR returns (xyxy,
    confidence, class_id, data["class_name"]), so every pass that consumes
    model.predict() works unchanged with an imported ONNX detector."""

    def __init__(self, xyxy: Any, confidence: Any, class_id: Any, class_name: list[str]) -> None:
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.data = {"class_name": class_name}


class OnnxYoloModel:
    """Inference-only detector for an exported Ultralytics-style YOLO ONNX model with
    NMS built into the graph (output 1 x N x 6: x1, y1, x2, y2, confidence, class id).

    Used for imported packages with manifest format "onnx-yolo". It can pre-label and be
    benchmarked, but it cannot be trained or fine-tuned (no RF-DETR checkpoint).
    """

    def __init__(self, weights_path: Path, class_names: list[str], input_size: int, language: str = "de") -> None:
        try:
            import numpy
            import onnxruntime
        except ImportError as error:
            raise SystemExit(localized(
                language,
                "onnxruntime ist nicht installiert. In der App zuerst „ML einrichten“ wählen.",
                "onnxruntime is not installed. Select “Set up ML” in the app first.",
            )) from error
        self.np = numpy
        available = onnxruntime.get_available_providers()
        providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available] or available
        self.session = onnxruntime.InferenceSession(str(weights_path), providers=providers)
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        fixed = [value for value in shape[2:4] if isinstance(value, int) and value > 0]
        self.input_size = fixed[0] if len(fixed) == 2 and fixed[0] == fixed[1] else int(input_size)
        output_shape = self.session.get_outputs()[0].shape
        if len(output_shape) != 3 or output_shape[-1] != 6:
            raise SystemExit(localized(
                language,
                "Das ONNX-Modell muss mit eingebautem NMS exportiert sein (Ausgabe 1 × N × 6).",
                "The ONNX model must be exported with NMS built in (output 1 × N × 6).",
            ))

    def _letterbox(self, path: str) -> tuple[Any, float, int, int, int, int]:
        from PIL import Image

        np = self.np
        size = self.input_size
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = min(size / width, size / height)
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = image.resize((new_width, new_height), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        left = int(round((size - new_width) / 2 - 0.1))
        top = int(round((size - new_height) / 2 - 0.1))
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(array), scale, left, top, width, height

    def predict(self, paths: list[str], threshold: float = 0.25) -> list[OnnxDetections]:
        np = self.np
        results: list[OnnxDetections] = []
        for path in paths:
            tensor, scale, left, top, width, height = self._letterbox(path)
            output = self.session.run(None, {self.input_name: tensor})[0][0]
            rows = output[output[:, 4] >= threshold]
            boxes = rows[:, :4].astype(np.float64).copy()
            boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / scale, 0, width)
            boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / scale, 0, height)
            class_ids = rows[:, 5].astype(int)
            names = [self.class_names[value] if 0 <= value < len(self.class_names) else "" for value in class_ids]
            results.append(OnnxDetections(boxes, rows[:, 4].astype(np.float64), class_ids, names))
        return results


def load_onnx_package_model(package_dir: Path, manifest: dict[str, Any], language: str = "de") -> OnnxYoloModel:
    weight_name = Path(str(manifest.get("weights", {}).get("file", ""))).name
    weight_path = (package_dir / "weights" / weight_name).resolve()
    if package_dir.resolve() not in weight_path.parents or not weight_path.is_file() or weight_path.suffix != ".onnx":
        raise SystemExit(localized(language, "ONNX-Gewichtsdatei fehlt.", "The ONNX weights file is missing."))
    class_names = [str(item) for item in manifest.get("classNames", [])]
    return OnnxYoloModel(weight_path, class_names, int(manifest.get("inputSize") or 640), language)


def active_library_manifest(project_root: Path) -> tuple[Path, dict[str, Any]] | None:
    """(package directory, manifest) of the activated library model, whatever its size."""
    try:
        active = load_json(project_root / "models" / "active.json")
        library_root = (project_root / "models" / "library").resolve()
        manifest_path = (library_root / str(active["packageID"]) / "manifest.json").resolve()
        if library_root not in manifest_path.parents or not manifest_path.is_file():
            return None
        return manifest_path.parent, load_json(manifest_path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


BASE_DETECTOR_CLASSES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def hardware_tier(override: str | None = None) -> str:
    """strong / medium / weak (see teachers.choose_tier); an explicit choice from the app wins."""
    import teachers
    if override in teachers.TIERS:
        return str(override)
    device = detect_device()
    vram_gb = None
    if device == "cuda":
        try:
            import torch
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            vram_gb = None
    return teachers.choose_tier(device, vram_gb, total_memory_bytes() / 1024**3)


def base_detector(size: str, language: str = "de"):
    """Apache-2.0 RF-DETR COCO model of any size N/S/M/L (XL/2XL are PML and not used)."""
    model_class = import_model_class(size, language)
    emit(localized(language, f"Lade Apache-2.0-Basismodell RF-DETR {size} (Teacher) …", f"Loading Apache-2.0 RF-DETR {size} base model (teacher) …"))
    return model_class(device=detect_device())


def teacher_size(choice: str | None, tier: str) -> str:
    import teachers
    if choice in BASE_DETECTOR_CLASSES:
        return str(choice)
    return teachers.TEACHER_SIZES[teachers.tier_plan(tier)["teacher"]]


def model_instance(project_root: Path, size: str, trained: bool, language: str = "de"):
    if trained:
        found = active_library_manifest(project_root)
        if found is not None and is_onnx_manifest(found[1]):
            emit(localized(
                language,
                f"Lade importiertes ONNX-Modell: {found[1].get('displayName') or found[1].get('packageID')}",
                f"Loading imported ONNX model: {found[1].get('displayName') or found[1].get('packageID')}",
            ))
            return load_onnx_package_model(found[0], found[1], language)
    if not trained:
        return base_detector(size, language)
    model_class = import_model_class(size, language)
    checkpoint = newest_checkpoint(project_root, size) if trained else None
    kwargs: dict[str, Any] = {"device": detect_device()}
    if checkpoint:
        verify_weights_pickle_safety(checkpoint, language, checkpoint.name)
        kwargs["pretrain_weights"] = str(checkpoint)
        emit(localized(language, f"Lade trainiertes Modell: {checkpoint.name}", f"Loading trained model: {checkpoint.name}"))
    else:
        emit(localized(
            language,
            f"Lade Apache-2.0-Basismodell RF-DETR {size} …",
            f"Loading Apache-2.0 RF-DETR {size} base model …",
        ))
    return model_class(**kwargs)


def detection_category_map(target_categories: list[str]) -> dict[str, str]:
    """Merge the base-model COCO alias map for every selected target category.

    ball/puck and player never co-occur in the same sport's category list
    (see sport_categories), so "sports ball"/"person" can only ever resolve
    to one project category per project in practice.
    """
    mapping = {category: category for category in target_categories}
    for category in target_categories:
        if category in {"ball", "puck"}:
            mapping["sports ball"] = category
        elif category == "player":
            mapping["person"] = "player"
    return mapping


def box_iou(first: list[float], second: list[float]) -> float:
    """Intersection over union for two [x1, y1, x2, y2] boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    """All-points interpolated AP used for the transparent local benchmark."""
    padded_recalls = [0.0, *recalls, 1.0]
    padded_precisions = [0.0, *precisions, 0.0]
    for index in range(len(padded_precisions) - 2, -1, -1):
        padded_precisions[index] = max(padded_precisions[index], padded_precisions[index + 1])
    return sum(
        (padded_recalls[index] - padded_recalls[index - 1]) * padded_precisions[index]
        for index in range(1, len(padded_recalls))
        if padded_recalls[index] != padded_recalls[index - 1]
    )


def evaluate_predictions(
    ground_truth_frames: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    classes: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate class-aware detections without uploading images or using COCO tooling."""
    per_class: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    aps: list[float] = []
    matched_ious: list[float] = []
    for category in classes:
        truth_by_frame: dict[str, list[list[float]]] = defaultdict(list)
        for frame in ground_truth_frames:
            for annotation in frame.get("annotations", []):
                if annotation.get("category") != category:
                    continue
                x = float(annotation.get("x", 0))
                y = float(annotation.get("y", 0))
                width = max(0.0, float(annotation.get("width", 0)))
                height = max(0.0, float(annotation.get("height", 0)))
                truth_by_frame[str(frame.get("id"))].append([x, y, x + width, y + height])
        candidates = sorted(
            (item for item in predictions if item.get("category") == category),
            key=lambda item: float(item.get("confidence", 0)),
            reverse=True,
        )
        used: dict[str, set[int]] = defaultdict(set)
        cumulative_tp = cumulative_fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        class_ious: list[float] = []
        truth_count = sum(len(items) for items in truth_by_frame.values())
        for prediction in candidates:
            frame_id = str(prediction.get("frameID"))
            box = [float(value) for value in prediction.get("box", [0, 0, 0, 0])]
            best_index = -1
            best_iou = 0.0
            for index, truth in enumerate(truth_by_frame.get(frame_id, [])):
                if index in used[frame_id]:
                    continue
                overlap = box_iou(box, truth)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= iou_threshold:
                used[frame_id].add(best_index)
                cumulative_tp += 1
                class_ious.append(best_iou)
            else:
                cumulative_fp += 1
            recalls.append(cumulative_tp / truth_count if truth_count else 0.0)
            precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))
        false_negatives = max(0, truth_count - cumulative_tp)
        precision = cumulative_tp / max(cumulative_tp + cumulative_fp, 1)
        recall = cumulative_tp / truth_count if truth_count else (1.0 if not candidates else 0.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ap50 = average_precision(recalls, precisions) if truth_count else 0.0
        if truth_count:
            aps.append(ap50)
        total_tp += cumulative_tp
        total_fp += cumulative_fp
        total_fn += false_negatives
        matched_ious.extend(class_ious)
        per_class[category] = {
            "groundTruth": truth_count,
            "predictions": len(candidates),
            "truePositives": cumulative_tp,
            "falsePositives": cumulative_fp,
            "falseNegatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap50,
            "meanIoU": sum(class_ious) / len(class_ious) if class_ious else 0.0,
        }
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    map50 = sum(aps) / len(aps) if aps else 0.0
    return {
        "iouThreshold": iou_threshold,
        "truePositives": total_tp,
        "falsePositives": total_fp,
        "falseNegatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "meanIoU": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "qualityScore": 100.0 * (0.7 * map50 + 0.3 * f1),
        "perClass": per_class,
    }


def benchmark_detection_category(detected_name: str, class_id: int | None, sport: str, package_classes: set[str]) -> str | None:
    categories = sport_categories(sport)
    normalized = detected_name.strip().lower()
    if normalized in package_classes:
        return normalized
    if normalized == "sports ball":
        target = "puck" if sport == "hockey" else "ball"
        return target if target in package_classes else None
    if normalized == "person":
        return "player" if "player" in package_classes else None
    if class_id is not None and 0 <= class_id < len(categories):
        indexed = categories[class_id]
        if indexed in package_classes:
            return indexed
    return None


def normalize_detection_batch(predictions: Any, expected_count: int) -> list[Any]:
    """Normalize RF-DETR single/list/nested-list prediction return shapes."""
    if expected_count < 1:
        return []
    if hasattr(predictions, "xyxy"):
        items = [predictions]
    elif isinstance(predictions, (list, tuple)):
        items = list(predictions)
    else:
        raise RuntimeError(f"RF-DETR returned an unsupported prediction type: {type(predictions).__name__}")

    if (
        len(items) == 1
        and expected_count > 1
        and isinstance(items[0], (list, tuple))
        and not hasattr(items[0], "xyxy")
    ):
        items = list(items[0])

    normalized: list[Any] = []
    for index, item in enumerate(items):
        while isinstance(item, (list, tuple)) and len(item) == 1 and not hasattr(item, "xyxy"):
            item = item[0]
        if not hasattr(item, "xyxy"):
            raise RuntimeError(
                f"RF-DETR prediction {index + 1} has no detection boxes "
                f"(type: {type(item).__name__})."
            )
        normalized.append(item)

    if len(normalized) != expected_count:
        raise RuntimeError(
            f"RF-DETR returned {len(normalized)} prediction results for "
            f"{expected_count} input images."
        )
    return normalized


def run_benchmark_pass(
    model: Any,
    frames: list[dict[str, Any]],
    package_classes: set[str],
    sport: str,
    threshold: float,
    batch_size: int,
    project_root: Path,
    progress_label: str,
    language: str,
) -> tuple[list[dict[str, Any]], float]:
    """Run one model over every benchmark frame, returning (predictions,
    inference_seconds). Shared by benchmark()'s single-model candidates and
    each member of a combined/ensemble candidate, which calls this once per
    member restricted to that member's own category subset (package_classes),
    so evaluate_predictions() sees one merged prediction list either way.
    """
    predictions: list[dict[str, Any]] = []
    inference_seconds = 0.0
    frames_root = (project_root / "frames").resolve()
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        paths: list[str] = []
        for frame in batch:
            candidate_path = (project_root / str(frame.get("relativePath", ""))).resolve()
            if frames_root not in candidate_path.parents or not candidate_path.is_file():
                raise RuntimeError(f"Benchmark frame is missing: {frame.get('id')}")
            paths.append(str(candidate_path))
        inference_started = time.perf_counter()
        detected_batch = normalize_detection_batch(model.predict(paths, threshold=threshold), len(batch))
        inference_seconds += time.perf_counter() - inference_started
        for frame, detections in zip(batch, detected_batch):
            detection_data = (getattr(detections, "data", {}) or {})
            names = detection_data.get("class_name")
            for index, box in enumerate(detections.xyxy):
                class_id = int(detections.class_id[index]) if getattr(detections, "class_id", None) is not None else None
                detected_name = str(names[index]) if names is not None else ""
                category = benchmark_detection_category(detected_name, class_id, sport, package_classes)
                if not category:
                    continue
                predictions.append({"frameID": str(frame.get("id")), "category": category, "confidence": float(detections.confidence[index]), "box": [float(value) for value in box]})
        emit(localized(language, f"{progress_label}: {min(start + batch_size, len(frames))}/{len(frames)}", f"{progress_label}: {min(start + batch_size, len(frames))}/{len(frames)}"))
    return predictions, inference_seconds


def benchmark(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    ground_truth_path = project_root / "benchmarks" / "ground-truth.json"
    if not ground_truth_path.is_file():
        raise SystemExit(localized(args.language, "Zuerst die geprüften Antworten als Referenz festlegen.", "Freeze the reviewed answers as ground truth first.", "Primero fija las respuestas revisadas como referencia.", "Définissez d’abord les réponses vérifiées comme référence."))
    ground_truth = load_json(ground_truth_path)
    if ground_truth.get("sport") != document.get("sport"):
        raise SystemExit(localized(args.language, "Die Referenz gehört zu einer anderen Sportart.", "The ground truth belongs to a different sport."))
    frames = ground_truth.get("frames", [])
    classes = sorted({annotation.get("category") for frame in frames for annotation in frame.get("annotations", []) if annotation.get("category")})
    if not frames or not classes:
        raise SystemExit(localized(args.language, "Die Referenz enthält keine auswertbaren Markierungen.", "The ground truth contains no evaluable annotations."))
    library_root = (project_root / "models" / "library").resolve()
    candidates: list[tuple[Path, dict[str, Any], Path | None]] = []
    for manifest_path in sorted(library_root.glob("*/manifest.json")) if library_root.is_dir() else []:
        try:
            manifest = load_json(manifest_path)
            if manifest.get("sport") != document.get("sport"):
                continue
            ensemble = manifest.get("ensemble")
            if ensemble:
                members = ensemble.get("members") or []
                member_paths = [(manifest_path.parent / str(member.get("weightsFile", ""))).resolve() for member in members]
                if members and all(member.get("modelSize") in MODEL_CLASSES for member in members) and all(library_root in path.parents and path.is_file() for path in member_paths):
                    candidates.append((manifest_path.parent, manifest, None))
                continue
            weight_path = (manifest_path.parent / "weights" / Path(str(manifest.get("weights", {}).get("file", ""))).name).resolve()
            if manifest.get("modelSize") in MODEL_CLASSES and library_root in weight_path.parents and weight_path.is_file():
                candidates.append((manifest_path.parent, manifest, weight_path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        raise SystemExit(localized(args.language, "Keine kompatiblen Modelle in der lokalen Bibliothek gefunden. Zuerst .recomodel-Pakete importieren.", "No compatible models were found in the local library. Import .recomodel packages first."))
    run_id = f"benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = project_root / "benchmarks" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    device = detect_device()
    results: list[dict[str, Any]] = []
    threshold = min(0.95, max(0.01, float(args.threshold)))
    for model_index, (package_dir, manifest, weight_path) in enumerate(candidates, start=1):
        package_id = str(manifest.get("packageID") or (weight_path.parent.parent.name if weight_path else package_dir.name))
        emit(localized(args.language, f"Modelltest {model_index}/{len(candidates)}: {package_id}", f"Model test {model_index}/{len(candidates)}: {package_id}"))
        started = time.perf_counter()
        predictions: list[dict[str, Any]] = []
        loaded_models: list[Any] = []
        try:
            package_classes = {str(item) for item in manifest.get("classes", [])}
            inference_seconds = 0.0
            ensemble = manifest.get("ensemble")
            if ensemble:
                # A combined/"baked" model: run each member only for the
                # categories it was assigned, and merge every member's
                # predictions into one list before evaluate_predictions() -
                # which needs no changes, since it just sees predictions.
                for member in ensemble.get("members", []):
                    member_classes = {str(category) for category in member.get("categories", [])}
                    member_model = ensemble_member_model(package_dir, member, args.language)
                    loaded_models.append(member_model)
                    member_profile = training_profile(str(member["modelSize"]), device)
                    member_batch_size = member_profile["batch_size"] if device != "cpu" else 1
                    member_predictions, member_seconds = run_benchmark_pass(
                        member_model, frames, member_classes, str(document["sport"]), threshold, member_batch_size,
                        project_root, package_id, args.language,
                    )
                    predictions.extend(member_predictions)
                    inference_seconds += member_seconds
            elif is_onnx_manifest(manifest):
                model = load_onnx_package_model(package_dir, manifest, args.language)
                loaded_models.append(model)
                predictions, inference_seconds = run_benchmark_pass(
                    model, frames, package_classes, str(document["sport"]), threshold, 1,
                    project_root, package_id, args.language,
                )
            else:
                model_class = import_model_class(str(manifest["modelSize"]), args.language)
                verify_weights_pickle_safety(weight_path, args.language, package_id)
                model = model_class(pretrain_weights=str(weight_path), device=device)
                loaded_models.append(model)
                profile = training_profile(str(manifest["modelSize"]), device)
                batch_size = profile["batch_size"] if device != "cpu" else 1
                predictions, inference_seconds = run_benchmark_pass(
                    model, frames, package_classes, str(document["sport"]), threshold, batch_size,
                    project_root, package_id, args.language,
                )
            metrics = evaluate_predictions(frames, predictions, classes)
            result = {"packageID": package_id, "modelSize": manifest.get("modelSize"), "classes": sorted(package_classes), "status": "completed", "metrics": metrics, "predictionCount": len(predictions), "totalSeconds": time.perf_counter() - started, "meanLatencyMs": 1000.0 * inference_seconds / max(len(frames), 1)}
            atomic_json(run_dir / f"{package_id}.predictions.json", {"schemaVersion": 1, "packageID": package_id, "datasetID": ground_truth.get("datasetID"), "threshold": threshold, "predictions": predictions})
        except Exception as error:
            result = {"packageID": package_id, "modelSize": manifest.get("modelSize"), "classes": sorted(str(item) for item in manifest.get("classes", [])), "status": "failed", "error": str(error), "totalSeconds": time.perf_counter() - started}
        results.append(result)
        try:
            del loaded_models
            import gc
            gc.collect()
            if device == "mps":
                import torch
                torch.mps.empty_cache()
            elif device == "cuda":
                import torch
                torch.cuda.empty_cache()
        except (NameError, ImportError, AttributeError):
            pass
    successful = [item for item in results if item.get("status") == "completed"]
    successful.sort(key=lambda item: (-float(item["metrics"].get("qualityScore", 0)), -float(item["metrics"].get("mAP50", 0)), float(item.get("meanLatencyMs", math.inf)), str(item.get("packageID"))))
    ranks = {item["packageID"]: index for index, item in enumerate(successful, start=1)}
    for item in results:
        item["rank"] = ranks.get(item.get("packageID"))
    results.sort(key=lambda item: (item.get("rank") is None, item.get("rank") or math.inf, str(item.get("packageID"))))
    report = {"schemaVersion": 1, "runID": run_id, "createdAt": datetime.now(timezone.utc).isoformat(), "sport": document.get("sport"), "datasetID": ground_truth.get("datasetID"), "groundTruthCreatedAt": ground_truth.get("createdAt"), "frameCount": len(frames), "annotationCount": sum(len(frame.get("annotations", [])) for frame in frames), "classes": classes, "modelCount": len(candidates), "successfulModelCount": len(successful), "threshold": threshold, "device": device, "rankingMethod": "70% mAP@0.50 + 30% F1; mean latency is the tie-breaker", "results": results, "privacy": "Images and predictions remained local. This report contains no image data or source video paths."}
    atomic_json(run_dir / "report.json", report)
    atomic_json(project_root / "benchmarks" / "latest.json", report)
    emit(localized(args.language, "Modellvergleich abgeschlossen.", "Model comparison completed.", "Comparación de modelos finalizada.", "Comparaison des modèles terminée."))


def simulate_ball_tracking(args: argparse.Namespace) -> None:
    """Run the active model's ball/puck detection over an already-extracted
    sequence of frame images, for the ball-tracking simulation player.

    Deliberately does not do extraction itself (that's platform-specific glue
    - FrameExtractor.swift on Mac, extract_frames_for_video() in
    local_worker.py on the web) and does not touch project.json - the caller
    extracts a short clip into its own scratch directory first and passes it
    here via --frames-dir, which does not need to be inside the project.

    Prints exactly one JSON blob (the single-JSON action contract
    MLWorker.swift already uses for installModelPackage/activateModel, not
    the streaming-progress contract autoLabel/train use) so a UI can resolve
    tracking/coasting/lost client-side from the raw per-frame detections
    without another subprocess round trip per adjustment.

    Coordinates are raw pixel positions, not normalized fractions: the caller
    already knows each extracted frame's width/height from its own
    extraction step, so there is no need to read image dimensions here.
    """
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    frames_dir = Path(args.frames_dir).resolve()
    if not frames_dir.is_dir():
        raise SystemExit(localized(args.language, "Bildordner für die Simulation wurde nicht gefunden.", "Simulation image folder was not found."))
    image_paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
    if not image_paths:
        raise SystemExit(localized(args.language, "Keine Bilder im Simulationsordner gefunden.", "No images found in the simulation folder."))

    sport = str(document["sport"])
    ball_category = "puck" if sport == "hockey" else "ball"
    category_map = detection_category_map([ball_category])
    device = detect_device()

    # model_instance()/ensemble_member_model() call emit() internally (e.g.
    # "Loading trained model: ...") - fine for every other action, which
    # streams progress, but this action's contract is exactly one JSON blob
    # and nothing else on stdout (see docstring). Suppress emit() for the
    # duration of model loading and inference, restoring it before returning
    # so a caught SystemExit's localized() text still displays normally.
    global emit
    original_emit = emit
    emit = lambda _message: None
    try:
        active_manifest = active_model_manifest(project_root, args.model)
        ensemble = (active_manifest or {}).get("ensemble") if active_manifest else None
        if ensemble:
            package_dir = (project_root / "models" / "library" / str(active_manifest["packageID"])).resolve()
            member = next((item for item in ensemble.get("members", []) if ball_category in set(item.get("categories", []))), None)
            if member is None:
                raise SystemExit(localized(
                    args.language,
                    f"Kein Modell im aktiven kombinierten Modell kennt „{ball_category}“.",
                    f"No model in the active combined model knows “{ball_category}”.",
                ))
            model = ensemble_member_model(package_dir, member, args.language)
            model_size = str(member["modelSize"])
        else:
            model = model_instance(project_root, args.model, trained=True, language=args.language)
            model_size = args.model
    finally:
        emit = original_emit

    profile = training_profile(model_size, device)
    batch_size = profile["batch_size"] if device != "cpu" else 1
    threshold = min(0.95, max(0.01, float(args.threshold)))
    results: list[dict[str, Any]] = []
    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start : start + batch_size]
        detected_batch = normalize_detection_batch(model.predict([str(path) for path in batch], threshold=threshold), len(batch))
        for path, detections in zip(batch, detected_batch):
            detection_data = (getattr(detections, "data", {}) or {})
            names = detection_data.get("class_name")
            best: dict[str, float] | None = None
            for index, box in enumerate(detections.xyxy):
                if names is not None:
                    detected_name = str(names[index])
                else:
                    class_id = int(detections.class_id[index]) if getattr(detections, "class_id", None) is not None else None
                    detected_name = "sports ball" if class_id == 32 else ""
                if category_map.get(detected_name) != ball_category:
                    continue
                confidence = float(detections.confidence[index])
                if best is not None and confidence <= best["confidence"]:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                best = {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "confidence": confidence}
            results.append({"file": path.name, "ball": best})
        # stderr, not stdout: stdout must stay exactly one JSON blob (see docstring).
        print(f"RECO_PROGRESS {len(results)} {len(image_paths)}", file=sys.stderr, flush=True)

    print(json.dumps({"frames": results}, ensure_ascii=False))


def frame_has_category(frame: dict[str, Any], target_category: str) -> bool:
    return any(
        annotation.get("category") == target_category
        for annotation in frame.get("annotations", [])
    )


def box_iou_xywh(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Intersection over union for x/y/width/height boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


PERSON_SHAPED_CATEGORIES: frozenset[str] = frozenset({"player", "referee", "goalkeeper"})


def field_membership_checker(
    field_geometry: dict[str, Any] | None,
) -> Callable[[str, float, float, float, float, float, float], bool] | None:
    """Builds a per-run function deciding whether a detected person-shaped box's
    feet fall inside the project's marked field outline - by request, so
    auto-label only adds "player"/"referee"/"goalkeeper" boxes for people
    actually standing on the field, not spectators, bench, or staff nearby.

    field_geometry is the project's optional "fieldGeometry": at least four image
    points as fractional (0..1) coordinates (so they apply however a given frame was
    extracted/scaled), joined by straight lines into a closed outline in the image.
    Four points can be the corners of a fully visible field; when the field leaves the
    frame, points on the image edge give any other shape. Real sizes in meters are not
    used (older projects may still store realWidth/realLength; they are ignored).
    Returns None - "don't filter anything" - whenever geometry is absent, incomplete,
    or OpenCV isn't installed, so a project without an outline behaves as before.
    """
    if not field_geometry:
        return None
    corners = field_geometry.get("corners")
    if not corners or len(corners) < 4:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    contour_cache: dict[tuple[float, float], Any] = {}

    def is_inside_outline(
        category: str, x1: float, y1: float, x2: float, y2: float, frame_width: float, frame_height: float
    ) -> bool:
        if category not in PERSON_SHAPED_CATEGORIES:
            return True
        key = (frame_width, frame_height)
        if key not in contour_cache:
            try:
                contour_cache[key] = np.array(
                    [[float(fx) * frame_width, float(fy) * frame_height] for fx, fy in corners], dtype=np.float32
                ).reshape(-1, 1, 2)
            except (TypeError, ValueError):
                contour_cache[key] = None
        contour = contour_cache[key]
        if contour is None:
            return True
        # Pixel margin, about 0.5% of the frame diagonal - forgives a foot on the
        # line or an imprecise click without letting in the bench or spectators.
        margin_px = 0.005 * (frame_width ** 2 + frame_height ** 2) ** 0.5
        distance = cv2.pointPolygonTest(contour, (float((x1 + x2) / 2.0), float(y2)), True)
        return distance >= -margin_px

    return is_inside_outline


# Typical referee attire per supported sport, used only as a best-effort hint (see
# referee_clothing_checker) - actual colors vary by league/level and this is not a
# substitute for a custom-trained model. "solid": one flat color, top and bottom
# alike. "solid_two_tone": a distinct top color and bottom color (basketball's
# grey shirt / black pants is very consistent). "stripes_black_white": the
# classic "zebra" officiating shirt used across several sports.
REFEREE_CLOTHING_PROFILES: dict[str, dict[str, Any]] = {
    "football": {"style": "solid", "colors": ["black"]},
    "futsal": {"style": "solid", "colors": ["black"]},
    "basketball": {"style": "solid_two_tone", "top": "grey", "bottom": "black"},
    "handball": {"style": "solid", "colors": ["black"]},
    "hockey": {"style": "stripes_black_white"},
    "rugby": {"style": "solid", "colors": ["black", "yellow", "orange", "red"]},
    "american_football": {"style": "stripes_black_white"},
    "lacrosse": {"style": "stripes_black_white"},
}


def _clothing_color_fraction(hsv_region: Any, color: str) -> float:
    """Fraction of pixels in an HSV image region within a named clothing-color band."""
    if hsv_region.size == 0:
        return 0.0
    hue, saturation, value = hsv_region[..., 0], hsv_region[..., 1], hsv_region[..., 2]
    if color == "black":
        mask = value < 70
    elif color == "white":
        mask = (saturation < 40) & (value > 190)
    elif color == "grey":
        mask = (saturation < 45) & (value >= 70) & (value <= 190)
    elif color == "yellow":
        mask = (hue >= 20) & (hue <= 34) & (saturation > 90) & (value > 90)
    elif color == "orange":
        mask = (hue >= 8) & (hue < 20) & (saturation > 90) & (value > 90)
    elif color == "red":
        mask = ((hue < 8) | (hue > 172)) & (saturation > 90) & (value > 90)
    else:
        return 0.0
    return float(mask.mean())


def referee_clothing_checker(
    sport: str,
) -> Callable[[Any, tuple[float, float, float, float]], bool] | None:
    """Best-effort clothing heuristic: does a detected person's crop match this
    sport's typical referee uniform (see REFEREE_CLOTHING_PROFILES)?

    This is a hint, not a verdict - a matching box still carries source="auto"
    and needs the same manual review as any other automatic suggestion, since
    team colors, lighting, and camera quality can all produce a false match.
    Returns None when OpenCV isn't installed or the sport has no documented
    profile - callers should then skip the heuristic entirely.
    """
    profile = REFEREE_CLOTHING_PROFILES.get(sport)
    if profile is None:
        return None
    try:
        import cv2
    except ImportError:
        return None

    style = profile.get("style")
    solid_threshold = 0.45
    stripe_threshold = 0.20

    def matches(image: Any, box: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = box
        height, width = image.shape[:2]
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(x2)), min(height, int(y2))
        if right - left < 6 or bottom - top < 10:
            return False
        crop = image[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # A crude standing-person proxy (no pose estimation): the top ~55% of the
        # box is torso/arms, the rest legs - close enough to tell a grey shirt
        # from black pants, or check the torso for zebra stripes.
        split = int(hsv.shape[0] * 0.55)
        torso, legs = hsv[:split], hsv[split:]

        if style == "stripes_black_white":
            black = _clothing_color_fraction(torso, "black")
            white = _clothing_color_fraction(torso, "white")
            return black >= stripe_threshold and white >= stripe_threshold and (black + white) >= 0.5
        if style == "solid_two_tone":
            return (
                _clothing_color_fraction(torso, profile["top"]) >= solid_threshold
                and _clothing_color_fraction(legs, profile["bottom"]) >= solid_threshold
            )
        if style == "solid":
            colors = profile.get("colors", [])
            return (
                any(_clothing_color_fraction(torso, color) >= solid_threshold for color in colors)
                and any(_clothing_color_fraction(legs, color) >= solid_threshold for color in colors)
            )
        return False

    return matches


CATEGORY_ASPECT_BOUNDS: dict[str, tuple[float, float]] = {
    # (min width/height, max width/height). ball/puck are the original,
    # empirically-tuned values. The rest are a first documented estimate,
    # not validated against real footage: player/referee/goalkeeper assume a
    # roughly upright person (generous on the wide side for crouching/arms-out
    # poses); hoop is a compact, slightly-wide structure; goal is wide
    # (mouth width dominates); goalpost is tall and narrow (the uprights).
    # These are the first thing to retune if refinement looks wrong for a class.
    "ball": (0.32, 3.1),
    "puck": (0.14, 4.8),
    "player": (0.20, 1.3),
    "referee": (0.20, 1.3),
    "goalkeeper": (0.20, 1.3),
    "hoop": (0.5, 2.5),
    "goal": (1.5, 6.0),
    "goalpost": (0.05, 0.6),
}


def plausible_refined_box(
    original: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
    category: str,
) -> bool:
    """Reject aggressive OpenCV changes before they can become suggestions.

    grabCut can return a contour that has drifted onto a nearby object,
    shadow, or advertising board instead of the annotated one - a real risk
    for person-shaped categories in particular, since two overlapping
    players can merge into a single foreground blob. These thresholds are a
    cheap plausibility filter, not a learned model: a refined box must stay
    close in size (0.20x-2.20x area), overlap the original detection
    meaningfully (IoU >= 0.15), stay centered near it (shift <= ~half the
    original box diagonal), and keep a plausible aspect ratio for its
    category (see CATEGORY_ASPECT_BOUNDS) - this is what should catch a
    blob that merged two overlapping players, since it would come out much
    wider than a single person. Area-ratio and center-shift stay the same
    across every category; only the aspect-ratio range is category-specific.
    """
    ox, oy, ow, oh = original
    cx, cy, cw, ch = candidate
    if min(ow, oh, cw, ch) < 3.0:
        return False
    area_ratio = (cw * ch) / max(ow * oh, 1.0)
    if not 0.20 <= area_ratio <= 2.20 or box_iou_xywh(original, candidate) < 0.15:
        return False
    original_center = (ox + ow / 2.0, oy + oh / 2.0)
    candidate_center = (cx + cw / 2.0, cy + ch / 2.0)
    center_shift = math.hypot(candidate_center[0] - original_center[0], candidate_center[1] - original_center[1])
    if center_shift > math.hypot(ow, oh) * 0.48:
        return False
    aspect = cw / max(ch, 1.0)
    low, high = CATEGORY_ASPECT_BOUNDS.get(category, CATEGORY_ASPECT_BOUNDS["ball"])
    return low <= aspect <= high


def opencv_refined_box(image: Any, annotation: dict[str, Any], cv2: Any, np: Any) -> tuple[tuple[float, float, float, float], float] | None:
    """Propose a tighter foreground box inside a detector-provided region."""
    image_height, image_width = image.shape[:2]
    original = (
        float(annotation.get("x", 0)), float(annotation.get("y", 0)),
        float(annotation.get("width", 0)), float(annotation.get("height", 0)),
    )
    x, y, width, height = original
    if min(width, height) < 5 or image_width < 2 or image_height < 2:
        return None
    margin = max(4, int(round(max(width, height) * 0.65)))
    left, top = max(0, int(math.floor(x)) - margin), max(0, int(math.floor(y)) - margin)
    right = min(image_width, int(math.ceil(x + width)) + margin)
    bottom = min(image_height, int(math.ceil(y + height)) + margin)
    roi = image[top:bottom, left:right]
    if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6:
        return None
    rectangle = (
        max(1, int(round(x - left))), max(1, int(round(y - top))),
        min(roi.shape[1] - 2, max(3, int(round(width)))),
        min(roi.shape[0] - 2, max(3, int(round(height)))),
    )
    if rectangle[0] + rectangle[2] >= roi.shape[1]:
        rectangle = (rectangle[0], rectangle[1], roi.shape[1] - rectangle[0] - 1, rectangle[3])
    if rectangle[1] + rectangle[3] >= roi.shape[0]:
        rectangle = (rectangle[0], rectangle[1], rectangle[2], roi.shape[0] - rectangle[1] - 1)
    if rectangle[2] < 3 or rectangle[3] < 3:
        return None
    mask = np.zeros(roi.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, rectangle, background, foreground, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    foreground_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[tuple[float, float, float, float], float] | None = None
    original_area = max(width * height, 1.0)
    original_center = (x + width / 2.0, y + height / 2.0)
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        padding = max(1.0, round(max(rw, rh) * 0.08))
        candidate = (
            max(0.0, left + rx - padding), max(0.0, top + ry - padding),
            min(float(image_width), left + rx + rw + padding) - max(0.0, left + rx - padding),
            min(float(image_height), top + ry + rh + padding) - max(0.0, top + ry - padding),
        )
        if not plausible_refined_box(original, candidate, str(annotation.get("category", "ball"))):
            continue
        candidate_area = max(candidate[2] * candidate[3], 1.0)
        fill = min(1.0, float(cv2.contourArea(contour)) / max(rw * rh, 1))
        center = (candidate[0] + candidate[2] / 2.0, candidate[1] + candidate[3] / 2.0)
        center_score = max(0.0, 1.0 - math.hypot(center[0] - original_center[0], center[1] - original_center[1]) / max(math.hypot(width, height), 1.0))
        area_score = max(0.0, 1.0 - abs(math.log(candidate_area / original_area)) / math.log(5.0))
        # Staying centered on the original detection matters most (a grabCut contour
        # that drifts to a neighboring object is the main failure mode this guards
        # against); overlap, fill, and area agreement matter less but still count.
        score = 0.40 * center_score + 0.30 * box_iou_xywh(original, candidate) + 0.15 * fill + 0.15 * area_score
        if score >= 0.46 and (best is None or score > best[1]):
            best = (candidate, score)
    return best


SAM_MIN_IOU = 0.75


def sam2_refiner(project_root: Path, method: str, tier: str | None, language: str):
    """SAM 2 only when explicitly chosen (method "sam2"); "auto" and "opencv" keep OpenCV.
    A first comparison against existing labels showed no measurable gain from SAM 2 (its masks
    drop arms and legs of moving players), so it stays opt-in until a hand-checked set proves
    otherwise. The model size follows the hardware tier (small on a weak tier)."""
    import teachers
    if method != "sam2":
        return None, None
    model_id = teachers.tier_plan(tier or hardware_tier())["refiner"] or ("sam2.1-hiera-small" if method == "sam2" else None)
    if model_id is None:
        return None, None
    try:
        model_dir = download_registry_model(project_root, model_id, language)
        return teachers.Sam2Refiner(model_dir, detect_device()), model_id
    except Exception as error:
        if method == "sam2":
            raise
        emit(localized(language, f"SAM 2 nicht verfügbar ({error}); verfeinere mit OpenCV.", f"SAM 2 not available ({error}); refining with OpenCV."))
        return None, None


def refinable_annotations(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Boxes eligible for a one-time OpenCV tightening pass.

    Applies to both automatic and manually-drawn boxes of a category
    plausible_refined_box knows how to judge (see CATEGORY_ASPECT_BOUNDS) - by
    explicit user request, since manual boxes were previously always left
    untouched on principle. Each refined box keeps its original coordinates
    under "opencvRefinement" and its own plausibility checks (area ratio, IoU,
    center shift, aspect ratio) still apply regardless of source, so this
    can't silently drift a box onto the wrong object. Already-refined boxes
    are skipped so a repeated run doesn't keep nudging the same box.
    """
    return [
        item for item in frame.get("annotations", [])
        if item.get("category") in CATEGORY_ASPECT_BOUNDS
        and not item.get("opencvRefinement") and not item.get("refinement")
    ]


def refine_boxes(args: argparse.Namespace) -> None:
    """Refine boxes of a refinable category (automatic and manual); see refinable_annotations()."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise SystemExit(localized(
            args.language,
            "OpenCV fehlt. Bitte zuerst „ML einrichten“ ausführen.",
            "OpenCV is missing. Run “Set up ML” first.",
            "Falta OpenCV. Ejecuta primero «Configurar ML».",
            "OpenCV manque. Lancez d’abord « Configurer le ML ».",
        )) from error
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    refiner, refiner_id = sam2_refiner(project_root, getattr(args, "method", "auto"), getattr(args, "tier", None), args.language)
    checked = refined = unchanged = unreadable = 0
    work = [frame for frame in document.get("frames", []) if refinable_annotations(frame)]
    for done, frame in enumerate(work, start=1):
        print(f"RECO_PROGRESS {done} {len(work)}", file=sys.stderr, flush=True)
        eligible = refinable_annotations(frame)
        image = cv2.imread(str(project_root / frame["relativePath"]), cv2.IMREAD_COLOR)
        if image is None:
            unreadable += 1
            continue
        if refiner is not None:
            from PIL import Image
            proposals = refiner.refine(Image.fromarray(image[:, :, ::-1]), [(float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"])) for a in eligible])
            for annotation, candidate in zip(eligible, proposals):
                checked += 1
                original = {key: float(annotation.get(key, 0)) for key in ("x", "y", "width", "height")}
                box = (original["x"], original["y"], original["width"], original["height"])
                # SAM 2 masks tend to drop arms and legs of moving players; only small, confident
                # corrections are taken over (IoU >= 0.75 with the box it started from).
                if candidate is None or not plausible_refined_box(box, candidate, str(annotation.get("category", "ball"))) or box_iou_xywh(box, candidate) < SAM_MIN_IOU:
                    unchanged += 1
                    continue
                annotation.update({"x": candidate[0], "y": candidate[1], "width": candidate[2], "height": candidate[3]})
                annotation["refinement"] = {"method": refiner_id, "original": original}
                refined += 1
            continue
        for annotation in eligible:
            checked += 1
            original = {key: float(annotation.get(key, 0)) for key in ("x", "y", "width", "height")}
            result = opencv_refined_box(image, annotation, cv2, np)
            if result is None:
                unchanged += 1
                continue
            candidate, score = result
            annotation.update({"x": candidate[0], "y": candidate[1], "width": candidate[2], "height": candidate[3]})
            annotation["opencvRefinement"] = {"method": "grabcut-contour-v1", "score": round(score, 4), "original": original}
            refined += 1
    document["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    atomic_json(project_file(project_root), document)
    if refiner is not None:
        emit(localized(
            args.language,
            f"SAM-2-Verfeinerung abgeschlossen: {refined} von {checked} Boxen plausibel angepasst, {unchanged} sicherheitshalber unverändert, {unreadable} Bilder nicht lesbar. Die ursprünglichen Koordinaten bleiben gespeichert.",
            f"SAM 2 refinement finished: plausibly adjusted {refined} of {checked} boxes, kept {unchanged} unchanged for safety, {unreadable} images unreadable. The original coordinates remain stored.",
            f"Refinado con SAM 2 terminado: {refined} de {checked} cuadros ajustados; {unchanged} sin cambios por seguridad y {unreadable} imágenes ilegibles. Las coordenadas originales se conservan.",
            f"Affinage SAM 2 terminé : {refined} boîtes ajustées sur {checked} ; {unchanged} conservées par sécurité et {unreadable} images illisibles. Les coordonnées d’origine restent enregistrées.",
        ))
        return
    emit(localized(
        args.language,
        f"OpenCV-Prüfung abgeschlossen: {refined} von {checked} Boxen (automatisch und manuell) plausibel verfeinert; {unchanged} sicherheitshalber unverändert, {unreadable} Bilder nicht lesbar. Die ursprünglichen Koordinaten jeder verfeinerten Box bleiben gespeichert.",
        f"OpenCV review completed: plausibly refined {refined} of {checked} boxes (automatic and manual); kept {unchanged} unchanged for safety, {unreadable} images unreadable. Each refined box's original coordinates remain stored.",
        f"Revisión OpenCV finalizada: {refined} de {checked} cuadros (automáticos y manuales) se refinaron; {unchanged} quedaron sin cambios por seguridad y {unreadable} imágenes no se pudieron leer. Las coordenadas originales de cada cuadro refinado permanecen guardadas.",
        f"Vérification OpenCV terminée : {refined} boîtes (automatiques et manuelles) affinées sur {checked} ; {unchanged} conservées par sécurité et {unreadable} images illisibles. Les coordonnées d’origine de chaque boîte affinée restent enregistrées.",
    ))


def predict_boxes(
    model: Any,
    project_root: Path,
    frames: list[dict[str, Any]],
    target_categories: list[str],
    category_map: dict[str, str],
    threshold: float,
    batch_size: int,
) -> dict[str, list[dict[str, Any]]]:
    """Run model over frames' images, returning each frame's raw detected boxes
    for target_categories (category/x/y/width/height/confidence, keyed by frame
    id) without writing anything to the frame itself. Used by the
    ensemble-disagreement comparison pass (see flag_ensemble_disagreement),
    which only compares a second model's boxes against the ones the primary
    pass already wrote and never persists them as annotations.
    """
    wanted = set(target_categories)
    boxes_by_frame: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        paths = [str(project_root / frame["relativePath"]) for frame in batch]
        detections_batch = normalize_detection_batch(model.predict(paths, threshold=threshold), len(batch))
        for frame, detections in zip(batch, detections_batch):
            names = detections.data.get("class_name") if hasattr(detections, "data") else None
            found: list[dict[str, Any]] = []
            for index, box in enumerate(detections.xyxy):
                if names is not None:
                    detected_name = str(names[index])
                else:
                    class_id = int(detections.class_id[index])
                    detected_name = "person" if class_id == 0 else "sports ball" if class_id == 32 else ""
                category = category_map.get(detected_name)
                if not category or category not in wanted:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                found.append({
                    "category": category,
                    "x": max(x1, 0.0),
                    "y": max(y1, 0.0),
                    "width": max(x2 - x1, 1.0),
                    "height": max(y2 - y1, 1.0),
                    "confidence": float(detections.confidence[index]),
                })
            boxes_by_frame[frame["id"]] = found
    return boxes_by_frame


def compute_disagreement(
    primary_boxes: list[dict[str, Any]],
    secondary_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> float:
    """Fraction of boxes (across both sets) that have no matching counterpart in
    the other set, greedily matched per category by best IoU. 0.0 means full
    agreement (or both sets empty); 1.0 means total disagreement.
    """
    total = len(primary_boxes) + len(secondary_boxes)
    if total == 0:
        return 0.0
    matched_secondary: set[int] = set()
    unmatched_primary = 0
    for primary in primary_boxes:
        primary_box = [primary["x"], primary["y"], primary["x"] + primary["width"], primary["y"] + primary["height"]]
        best_index = -1
        best_iou = 0.0
        for index, secondary in enumerate(secondary_boxes):
            if index in matched_secondary or secondary["category"] != primary["category"]:
                continue
            secondary_box = [secondary["x"], secondary["y"], secondary["x"] + secondary["width"], secondary["y"] + secondary["height"]]
            overlap = box_iou(primary_box, secondary_box)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_secondary.add(best_index)
        else:
            unmatched_primary += 1
    unmatched_secondary = len(secondary_boxes) - len(matched_secondary)
    return (unmatched_primary + unmatched_secondary) / total


def flag_disagreement_with_florence(
    project_root: Path,
    frames: list[dict[str, Any]],
    target_categories: list[str],
    category_map: dict[str, str],
    threshold: float,
    language: str,
    tier: str | None,
) -> None:
    """No second trained model installed: Florence-2 (MIT) is the second opinion for people and
    the ball. Like the model comparison, it only flags frames, it never writes boxes."""
    import teachers
    compared = [category for category in target_categories if category in {"player", "ball", "puck"}]
    if not compared:
        return
    model_id = teachers.tier_plan(tier or hardware_tier())["second"]
    try:
        model_dir = download_registry_model(project_root, model_id, language)
        detector = teachers.Florence2Detector(model_dir, detect_device())
        secondary_boxes = predict_boxes(detector, project_root, frames, compared, category_map, threshold, 1)
    except Exception as error:  # the comparison is optional; auto-label itself already succeeded
        emit(localized(language, f"Zweite Meinung (Florence-2) übersprungen: {error}", f"Second opinion (Florence-2) skipped: {error}"))
        return
    flagged = 0
    for frame in frames:
        primary = [box for box in frame.get("annotations", []) if box.get("category") in compared and box.get("source") == "auto"]
        secondary = secondary_boxes.get(frame["id"], [])
        if (primary or secondary) and compute_disagreement(primary, secondary) >= 0.3:
            flags = frame.setdefault("reviewFlags", [])
            if "ensemble-disagreement" not in flags:
                flags.append("ensemble-disagreement")
                flagged += 1
    emit(localized(language, f"Zweite Meinung (Florence-2): {flagged} Bilder mit Abweichungen zur Prüfung markiert.", f"Second opinion (Florence-2): {flagged} frames with differences flagged for review."))


def flag_ensemble_disagreement(
    project_root: Path,
    frames: list[dict[str, Any]],
    target_categories: list[str],
    category_map: dict[str, str],
    active_manifest: dict[str, Any] | None,
    threshold: float,
    batch_size: int,
    sport: str,
    language: str,
    tier: str | None = None,
) -> None:
    """Run a second, comparison-only model over the same frames/categories the
    primary pass just auto-labeled, and flag frames where the two disagree
    (compute_disagreement). Never writes the secondary model's boxes as
    annotations. No-op when no second compatible model is installed. Prefers
    the second-ranked model from the last benchmark run when one exists (see
    benchmark()), otherwise the first other compatible library model found -
    same library-glob discovery pattern benchmark() itself uses.
    """
    library_root = (project_root / "models" / "library").resolve()
    if not library_root.is_dir():
        flag_disagreement_with_florence(project_root, frames, target_categories, category_map, threshold, language, tier)
        return
    active_package_id = str((active_manifest or {}).get("packageID") or "")
    ranked_ids: list[str] = []
    latest_report_path = project_root / "benchmarks" / "latest.json"
    if latest_report_path.is_file():
        try:
            report = load_json(latest_report_path)
            if report.get("sport") == sport:
                ranked_ids = [
                    str(item.get("packageID"))
                    for item in sorted(
                        (item for item in report.get("results", []) if item.get("status") == "completed"),
                        key=lambda item: item.get("rank") or math.inf,
                    )
                ]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            ranked_ids = []

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(library_root.glob("*/manifest.json")):
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        package_id = str(manifest.get("packageID") or manifest_path.parent.name)
        if manifest.get("sport") != sport or package_id == active_package_id:
            continue
        if not set(manifest.get("classes", [])).intersection(target_categories):
            continue
        candidates.append((manifest_path.parent, manifest))
    if not candidates:
        flag_disagreement_with_florence(project_root, frames, target_categories, category_map, threshold, language, tier)
        return
    candidates.sort(key=lambda item: ranked_ids.index(str(item[1].get("packageID"))) if str(item[1].get("packageID")) in ranked_ids else math.inf)
    package_dir, manifest = candidates[0]
    package_classes = [category for category in target_categories if category in set(manifest.get("classes", []))]
    if not package_classes:
        return

    device = detect_device()
    secondary_boxes: dict[str, list[dict[str, Any]]] = {}
    try:
        ensemble = manifest.get("ensemble")
        if ensemble:
            for member in ensemble.get("members", []):
                member_categories = [category for category in package_classes if category in set(member.get("categories", []))]
                if not member_categories:
                    continue
                member_model = ensemble_member_model(package_dir, member, language)
                member_profile = training_profile(str(member["modelSize"]), device)
                member_batch_size = member_profile["batch_size"] if device != "cpu" else 1
                member_boxes = predict_boxes(member_model, project_root, frames, member_categories, category_map, threshold, member_batch_size)
                for frame_id, boxes in member_boxes.items():
                    secondary_boxes.setdefault(frame_id, []).extend(boxes)
                del member_model
        elif is_onnx_manifest(manifest):
            model = load_onnx_package_model(package_dir, manifest, language)
            secondary_boxes = predict_boxes(model, project_root, frames, package_classes, category_map, threshold, 1)
            del model
        else:
            model_class = import_model_class(str(manifest["modelSize"]), language)
            weight_path = (package_dir / "weights" / Path(str(manifest.get("weights", {}).get("file", ""))).name).resolve()
            verify_weights_pickle_safety(weight_path, language, str(manifest.get("packageID") or weight_path.name))
            model = model_class(pretrain_weights=str(weight_path), device=device)
            secondary_boxes = predict_boxes(model, project_root, frames, package_classes, category_map, threshold, batch_size)
            del model
    except Exception:
        return

    for frame in frames:
        primary = [box for box in frame.get("annotations", []) if box.get("category") in package_classes and box.get("source") == "auto"]
        secondary = secondary_boxes.get(frame["id"], [])
        if not primary and not secondary:
            continue
        if compute_disagreement(primary, secondary) >= 0.3:
            flags = frame.setdefault("reviewFlags", [])
            if "ensemble-disagreement" not in flags:
                flags.append("ensemble-disagreement")


def flag_temporal_outliers(
    frames: list[dict[str, Any]],
    max_gap_seconds: float = 5.0,
    deviation_fraction: float = 0.15,
) -> None:
    """Flag candidate frames whose single-instance category box (e.g. the ball)
    sits far from where linear interpolation between its temporal neighbors in
    the same video would put it - reusing the neighbor-interpolation idea
    behind BallTrackingResolver.swift's trajectory reasoning, adapted for
    sparse/unevenly-sampled training frames rather than a dense video. Mutates
    frames in place, appending "temporal-outlier" to reviewFlags. A frame with
    zero or multiple boxes of a category is skipped for that category
    (ambiguous which instance to track) - mirrors how the ball-tracking
    simulation already reduces to one detection per frame.
    """
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if "videoID" not in frame or "timestamp" not in frame:
            continue
        by_video[frame["videoID"]].append(frame)
    for video_frames in by_video.values():
        ordered = sorted(video_frames, key=lambda frame: frame["timestamp"])
        categories = {
            annotation.get("category")
            for frame in ordered
            for annotation in frame.get("annotations", [])
            if annotation.get("category")
        }
        for category in categories:
            points: list[tuple[int, float, float, float]] = []
            for index, frame in enumerate(ordered):
                boxes = [item for item in frame.get("annotations", []) if item.get("category") == category]
                if len(boxes) != 1:
                    continue
                box = boxes[0]
                points.append((index, float(frame["timestamp"]), box["x"] + box["width"] / 2, box["y"] + box["height"] / 2))
            for point_index in range(1, len(points) - 1):
                frame_index, timestamp, cx, cy = points[point_index]
                frame = ordered[frame_index]
                if frame.get("reviewStatus") != "candidate":
                    continue
                _, prev_time, prev_x, prev_y = points[point_index - 1]
                _, next_time, next_x, next_y = points[point_index + 1]
                span = next_time - prev_time
                if span <= 0 or span > max_gap_seconds:
                    continue
                weight = (timestamp - prev_time) / span
                expected_x = prev_x + weight * (next_x - prev_x)
                expected_y = prev_y + weight * (next_y - prev_y)
                distance = math.hypot(cx - expected_x, cy - expected_y)
                diagonal = math.hypot(float(frame.get("width", 0)), float(frame.get("height", 0)))
                if diagonal > 0 and distance / diagonal > deviation_fraction:
                    flags = frame.setdefault("reviewFlags", [])
                    if "temporal-outlier" not in flags:
                        flags.append("temporal-outlier")


def run_autolabel_pass(
    model: Any,
    frames: list[dict[str, Any]],
    target_categories: list[str],
    category_map: dict[str, str],
    threshold: float,
    batch_size: int,
    project_root: Path,
    document: dict[str, Any],
    language: str,
    is_on_field: Callable[[str, float, float, float, float, float, float], bool] | None = None,
    referee_clothing_match: Callable[[Any, tuple[float, float, float, float]], bool] | None = None,
) -> tuple[int, int, int]:
    """Run one model's detection pass over frames for target_categories, adding
    boxes in place and persisting after every batch. Returns (processed_frames,
    frames_with_detections, boxes_added) for this pass alone - auto_label() may
    call this twice (once per model) when a category needs the base model as a
    fallback for a class the active custom model doesn't know.

    is_on_field, when given, additionally requires a person-shaped detection's
    feet to fall inside the project's marked field boundaries (see
    field_membership_checker) - other categories (ball, hoop, ...) are unaffected.

    referee_clothing_match, when given, is checked against every newly added
    "player" box on a frame that doesn't already have a "referee" annotation - a
    match reclassifies that specific box to "referee" before it's saved, still
    with source="auto" so it goes through the same manual review as any other
    automatic suggestion. The caller is responsible for only passing this when
    "referee" was actually requested for the overall run.
    """
    processed_frames = frames_with_detections = boxes_added = 0
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        # Each frame may already have some of the selected categories (e.g. from
        # an earlier single-category run) - only request/keep the ones it's still
        # missing, so a frame with "ball" already present but not "player" gets
        # just the player boxes added, not a duplicate set of ball boxes.
        wanted_by_frame_id = {
            frame["id"]: {category for category in target_categories if not frame_has_category(frame, category)}
            for frame in batch_frames
        }
        pending = [frame for frame in batch_frames if wanted_by_frame_id[frame["id"]]]
        if not pending:
            continue
        paths = [str(project_root / frame["relativePath"]) for frame in pending]
        detections_batch = normalize_detection_batch(
            model.predict(paths, threshold=threshold), len(pending)
        )

        for frame, detections in zip(pending, detections_batch):
            wanted = wanted_by_frame_id[frame["id"]]
            new_annotations = []
            names = detections.data.get("class_name") if hasattr(detections, "data") else None
            for index, box in enumerate(detections.xyxy):
                if names is not None:
                    detected_name = str(names[index])
                else:
                    class_id = int(detections.class_id[index])
                    detected_name = "person" if class_id == 0 else "sports ball" if class_id == 32 else ""
                category = category_map.get(detected_name)
                if not category or category not in wanted:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                if is_on_field is not None and not is_on_field(
                    category, x1, y1, x2, y2, float(frame.get("width", 0)), float(frame.get("height", 0))
                ):
                    continue
                confidence = float(detections.confidence[index])
                new_annotations.append(
                    {
                        "id": str(uuid.uuid4()),
                        "category": category,
                        "x": max(x1, 0.0),
                        "y": max(y1, 0.0),
                        "width": max(x2 - x1, 1.0),
                        "height": max(y2 - y1, 1.0),
                        "confidence": confidence,
                        "source": "auto",
                    }
                )
            if (
                referee_clothing_match is not None
                and not frame_has_category(frame, "referee")
                and any(item["category"] == "player" for item in new_annotations)
            ):
                # cv2 is guaranteed importable here - referee_clothing_checker()
                # only ever returns a non-None matcher once it has confirmed that.
                import cv2
                image = cv2.imread(str(project_root / frame["relativePath"]), cv2.IMREAD_COLOR)
                if image is not None:
                    for item in new_annotations:
                        if item["category"] != "player":
                            continue
                        box = (item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"])
                        if referee_clothing_match(image, box):
                            item["category"] = "referee"
            frame["annotations"] = [*frame.get("annotations", []), *new_annotations]
            processed_frames += 1
            boxes_added += len(new_annotations)
            if new_annotations:
                frames_with_detections += 1
        # Persist every completed batch so a later device/library error can be
        # resumed without repeating the entire local inference run.
        document["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        atomic_json(project_file(project_root), document)
        emit(localized(
            language,
            f"Automatisch geprüft: {min(start + batch_size, len(frames))}/{len(frames)}",
            f"Automatically checked: {min(start + batch_size, len(frames))}/{len(frames)}",
        ))
    return processed_frames, frames_with_detections, boxes_added


def auto_label(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    frames = document.get("frames", [])
    if getattr(args, "candidate_only", False):
        frames = [frame for frame in frames if frame.get("reviewStatus") == "candidate"]
    if not frames:
        raise SystemExit(localized(
            args.language,
            "Das Projekt enthält noch keine Frames.",
            "The project does not contain any frames yet.",
        ))

    target_categories = list(dict.fromkeys(args.category))  # de-duplicate, keep order
    category_map = detection_category_map(target_categories)
    has_checkpoint = newest_checkpoint(project_root, args.model) is not None
    # An activated library model's own manifest is authoritative about which classes it
    # can actually detect (e.g. a model trained back when only "ball" was annotated still
    # has just one class, even though the project has since grown "referee"/"player" too).
    # Falling back to "any checkpoint => every category is fair game" here used to run a
    # full inference pass for classes the active model provably cannot produce, wasting
    # time and reporting a single misleading "0 boxes" that didn't distinguish "the model
    # doesn't know this class yet" from "the model just didn't find anything this time".
    known_classes = active_model_classes(project_root, args.model)
    sport = str(document.get("sport", ""))
    # Whether "referee" can be satisfied via the clothing heuristic instead of a
    # dedicated trained class - it only has something to check once "player" is
    # also being detected in this same run (the heuristic reclassifies generic
    # person detections, it doesn't run its own detection pass).
    clothing_checker = referee_clothing_checker(sport)
    clothing_possible = clothing_checker is not None and "player" in target_categories

    def category_is_supported(category: str) -> bool:
        if known_classes is not None:
            return category in known_classes
        return has_checkpoint or category in {"ball", "puck", "player"}

    unsupported: list[str] = []
    base_fallback_categories: list[str] = []
    clothing_fallback_categories: list[str] = []
    for category in target_categories:
        if category_is_supported(category):
            continue
        if category == "player":
            # No trained "player" class (base model or a custom model missing it),
            # but the generic Apache-2.0 base model can still find people via
            # COCO's "person" class - run a second pass with the base model for
            # just this category instead of skipping it outright.
            base_fallback_categories.append(category)
        elif category == "referee" and clothing_possible:
            clothing_fallback_categories.append(category)
        else:
            unsupported.append(category)
    for category in unsupported:
        emit(localized(
            args.language,
            f"Das aktuell aktive Modell kennt die Klasse „{category}“ noch nicht. "
            "Diese Klasse zunächst manuell markieren und ein eigenes Modell trainieren.",
            f"The currently active model does not know the “{category}” class yet. "
            "Annotate this class manually first and train a custom model.",
        ))
    if "referee" in target_categories and not clothing_possible and clothing_checker is not None and "player" not in target_categories:
        emit(localized(
            args.language,
            "Für die Kleidungs-Erkennung von Schiedsrichtern muss „Spieler“ ebenfalls ausgewählt sein "
            "(die Personenerkennung läuft darüber).",
            "Detecting referees by clothing also requires “player” to be selected (person detection runs through it).",
        ))
    target_categories = [category for category in target_categories if category not in unsupported]
    if not target_categories:
        return
    custom_categories = [
        category for category in target_categories
        if category not in base_fallback_categories and category not in clothing_fallback_categories
    ]

    if base_fallback_categories:
        fallback_list = ", ".join(base_fallback_categories)
        emit(localized(
            args.language,
            f"Für „{fallback_list}“ kennt das aktive Modell keine eigene Klasse; das allgemeine "
            "Basismodell wird zusätzlich genutzt. Es erkennt Personen nur allgemein, nicht speziell "
            "als Spieler - Zuschauer und Schiedsrichter können deshalb fälschlich als Spieler erscheinen.",
            f"The active model has no dedicated class for “{fallback_list}”; the generic base model is "
            "used in addition for it. It only detects people generically, not specifically as players - "
            "spectators and referees may therefore be incorrectly labeled as players.",
        ))
    elif "player" in target_categories and not has_checkpoint:
        emit(localized(
            args.language,
            "Hinweis: Das Basismodell liefert nur die Klasse „Person“. Zuschauer und Schiedsrichter "
            "können deshalb fälschlich als Spieler erscheinen.",
            "Note: The base model only provides the “person” class. Spectators and referees may "
            "therefore be incorrectly labeled as players.",
        ))

    field_geometry = document.get("fieldGeometry")
    is_on_field = field_membership_checker(field_geometry)
    if field_geometry and is_on_field is not None:
        emit(localized(
            args.language,
            "Spielfeldgrenzen aktiv: Nur Personen, die mit den Füßen auf dem markierten Feld stehen, werden berücksichtigt.",
            "Field boundaries active: only people standing with their feet on the marked field are considered.",
        ))
    elif field_geometry and is_on_field is None:
        emit(localized(
            args.language,
            "Spielfeldgrenzen konnten nicht angewendet werden (OpenCV fehlt oder die Markierung ist unvollständig).",
            "Field boundaries could not be applied (OpenCV is missing or the marking is incomplete).",
        ))

    referee_clothing_match = clothing_checker if clothing_fallback_categories else None
    if referee_clothing_match is not None:
        emit(localized(
            args.language,
            "Kleidungs-Heuristik aktiv: Personen mit typischer Schiedsrichter-Kleidung werden als "
            "„Schiedsrichter“ vorgeschlagen (weiterhin zur Prüfung markiert, keine Garantie).",
            "Clothing heuristic active: people wearing typical referee attire are suggested as "
            "“referee” (still flagged for review, not a guarantee).",
        ))

    already_present = {
        category: sum(1 for frame in frames if frame_has_category(frame, category))
        for category in target_categories
    }
    device = detect_device()
    profile = training_profile(args.model, device)
    batch_size = profile["batch_size"] if device != "cpu" else 1
    category_list = ", ".join(target_categories)
    tier = hardware_tier(getattr(args, "tier", None))
    base_size = teacher_size(getattr(args, "teacher", None), tier)
    emit(localized(
        args.language,
        f"Vorbeschriftung für „{category_list}“ nutzt {device.upper()} mit Batch {batch_size}.",
        f"Pre-labeling for “{category_list}” uses {device.upper()} with batch {batch_size}.",
    ))

    processed_frames = frames_with_detections = boxes_added = 0

    if custom_categories:
        active_manifest = active_model_manifest(project_root, args.model)
        ensemble = (active_manifest or {}).get("ensemble")
        if ensemble:
            # A combined/"baked" model: no single checkpoint knows every
            # category, so run each member for only the categories it was
            # assigned when the ensemble was built - the same two-pass
            # accumulation pattern as the base-model fallback below, just
            # generalized to N passes instead of a fixed two.
            package_dir = (project_root / "models" / "library" / str(active_manifest["packageID"])).resolve()
            for member in ensemble.get("members", []):
                member_categories = [category for category in custom_categories if category in set(member.get("categories", []))]
                if not member_categories:
                    continue
                member_model = ensemble_member_model(package_dir, member, args.language)
                pass_processed, pass_with_detections, pass_boxes = run_autolabel_pass(
                    member_model, frames, member_categories, category_map, args.threshold, batch_size,
                    project_root, document, args.language, is_on_field, referee_clothing_match,
                )
                processed_frames += pass_processed
                frames_with_detections += pass_with_detections
                boxes_added += pass_boxes
                del member_model
        else:
            # Without a trained model (or an imported ONNX one) the teacher - the largest base
            # model the hardware tier allows - suggests people and the ball.
            if has_checkpoint or active_library_manifest(project_root) is not None:
                model = model_instance(project_root, args.model, trained=True, language=args.language)
            else:
                model = model_instance(project_root, base_size, trained=False, language=args.language)
            pass_processed, pass_with_detections, pass_boxes = run_autolabel_pass(
                model, frames, custom_categories, category_map, args.threshold, batch_size,
                project_root, document, args.language, is_on_field, referee_clothing_match,
            )
            processed_frames += pass_processed
            frames_with_detections += pass_with_detections
            boxes_added += pass_boxes
            del model

    if base_fallback_categories:
        base_model = model_instance(project_root, base_size, trained=False, language=args.language)
        pass_processed, pass_with_detections, pass_boxes = run_autolabel_pass(
            base_model, frames, base_fallback_categories, category_map, args.threshold, batch_size,
            project_root, document, args.language, is_on_field, referee_clothing_match,
        )
        processed_frames += pass_processed
        frames_with_detections += pass_with_detections
        boxes_added += pass_boxes
        del base_model

    if getattr(args, "compare_models", False) and custom_categories:
        flag_ensemble_disagreement(
            project_root, frames, custom_categories, category_map,
            active_model_manifest(project_root, args.model), args.threshold, batch_size,
            sport, args.language, tier=tier,
        )
    if any(category in ("ball", "puck") for category in target_categories):
        import teachers
        interpolated = teachers.interpolate_ball(document.get("frames", []))
        if interpolated:
            emit(localized(args.language, f"{interpolated} Ball-Positionen zwischen nahen Bildern interpoliert (zur Prüfung markiert).", f"{interpolated} ball positions interpolated between nearby frames (marked for review)."))
    flag_temporal_outliers(document.get("frames", []))

    document["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    atomic_json(project_file(project_root), document)
    skipped_existing = sum(already_present.values())
    emit(localized(
        args.language,
        f"Vorbeschriftung für „{category_list}“ abgeschlossen: {boxes_added} Boxen in "
        f"{frames_with_detections} von {processed_frames} geprüften Frames. "
        f"{skipped_existing} Frame/Klasse-Kombinationen hatten bereits eine Markierung und wurden nicht überschrieben.",
        f"Pre-labeling for “{category_list}” finished: {boxes_added} boxes in "
        f"{frames_with_detections} of {processed_frames} checked frames. "
        f"{skipped_existing} frame/class combinations already had an annotation and were not overwritten.",
    ))
    if boxes_added == 0:
        emit(localized(
            args.language,
            f"Keine Treffer für „{category_list}“ oberhalb der Schwelle. Einige Beispiele manuell "
            "markieren und danach trainieren; anschließend erneut automatisch markieren.",
            f"No “{category_list}” detections above the threshold. Annotate a few examples manually, "
            "train the model, and then run automatic detection again.",
        ))
    elif any(category in CATEGORY_ASPECT_BOUNDS for category in target_categories):
        try:
            refine_boxes(argparse.Namespace(project=str(project_root), language=args.language, method=getattr(args, "refine_method", "auto"), tier=tier))
        except SystemExit as error:
            emit(localized(
                args.language,
                f"OpenCV-Nachprüfung übersprungen: {error}",
                f"Skipped OpenCV post-review: {error}",
                f"Se omitió la revisión posterior con OpenCV: {error}",
                f"Post-vérification OpenCV ignorée : {error}",
            ))


# --- Consistency check: two detectors, time, and the app-managed VLM -----------------------------

PERSON_GROUP = {"player", "goalkeeper", "referee"}
CONSISTENCY_FLAGS = ("detector-disagreement", "players-vanished", "temporal-outlier")


def comparable(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A generic detector only knows "person" and "ball": compare on that level, otherwise
    every referee would count as a disagreement."""
    result = []
    for box in boxes:
        category = box.get("category")
        group = "person" if category in PERSON_GROUP else "ball" if category in {"ball", "puck"} else None
        if group:
            result.append({**box, "category": group})
    return result


def flag_vanished_players(frames: list[dict[str, Any]], max_gap_seconds: float = 1.0) -> int:
    """Within one video, at least two and at least half of the people disappear between two
    frames at most a second apart - usually missing boxes, not people leaving."""
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_video[str(frame.get("videoID"))].append(frame)
    flagged = 0
    for items in by_video.values():
        items.sort(key=lambda frame: float(frame.get("timestamp") or 0.0))
        for previous, current in zip(items, items[1:]):
            if float(current.get("timestamp") or 0.0) - float(previous.get("timestamp") or 0.0) > max_gap_seconds:
                continue
            before = sum(1 for box in previous.get("annotations", []) if box.get("category") in PERSON_GROUP)
            after = sum(1 for box in current.get("annotations", []) if box.get("category") in PERSON_GROUP)
            if before - after >= 2 and after <= before / 2:
                flags = current.setdefault("reviewFlags", [])
                if "players-vanished" not in flags:
                    flags.append("players-vanished")
                    flagged += 1
    return flagged


def prepare_vlm(project_root: Path, model_id: str, language: str):
    """Download (once per computer) the llama-server build for this platform and the VLM;
    returns what vlm.LlamaServer needs plus the model's coordinate format."""
    import vlm
    nvidia = bool(shutil.which("nvidia-smi")) or detect_device() == "cuda"
    try:
        runtime = vlm.runtime_ids(nvidia=nvidia)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    for runtime_id in runtime:
        download_registry_model(project_root, runtime_id, language)
    binary = vlm.find_server_binary(registry_model_path(project_root, runtime[0], language))
    if binary is None:
        raise SystemExit(localized(language, "Der VLM-Server wurde im Download nicht gefunden.", "The VLM server was not found in the download."))
    entry = registry_model(model_id, language)
    folder = download_registry_model(project_root, model_id, language)
    return binary, folder / entry["model"], folder / entry["mmproj"], entry.get("coordinates", "relative-1000")


def consistency_check(args: argparse.Namespace) -> None:
    import random
    import vlm
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    frames = [frame for frame in document.get("frames", []) if not frame.get("heldOut") and frame.get("annotations")]
    if not frames:
        raise SystemExit(localized(args.language, "Keine markierten Bilder zum Prüfen.", "No labelled images to check."))
    for frame in frames:  # a new check replaces the previous one's results
        frame["reviewFlags"] = [flag for flag in frame.get("reviewFlags", []) if flag not in CONSISTENCY_FLAGS + ("vlm-issue",)]
    steps = ["detector", "time"] + (["vlm"] if args.vlm else [])
    step = lambda key, label: stage(steps.index(key) + 1, len(steps), label)  # noqa: E731

    step("detector", localized(args.language, "Zweiter Detektor", "Second detector"))
    tier = hardware_tier(getattr(args, "tier", None))
    size = teacher_size(getattr(args, "teacher", None), tier)
    detector = model_instance(project_root, size, trained=False, language=args.language)
    category_map = detection_category_map(["player", "ball"])
    device = detect_device()
    batch = training_profile("nano", device)["batch_size"] if device != "cpu" else 1
    second = predict_boxes(detector, project_root, frames, ["player", "ball"], category_map, args.threshold, batch)
    del detector
    disagreement = 0
    for frame in frames:
        theirs = comparable(second.get(frame["id"], []))
        ours = comparable(frame.get("annotations", []))
        if (ours or theirs) and compute_disagreement(ours, theirs, iou_threshold=args.iou) >= 0.3:
            frame["reviewFlags"].append("detector-disagreement")
            disagreement += 1

    step("time", localized(args.language, "Zeitlicher Verlauf", "Timeline"))
    vanished = flag_vanished_players(frames)
    flag_temporal_outliers(frames)

    vlm_checked = vlm_issues = vlm_resolved = 0
    if args.vlm:
        step("vlm", localized(args.language, "VLM-Prüfung", "VLM check"))
        flagged = [frame for frame in frames if frame.get("reviewFlags")]
        rest = [frame for frame in frames if not frame.get("reviewFlags")]
        sample = random.Random(42).sample(rest, min(len(rest), round(len(rest) * args.sample))) if rest and args.sample > 0 else []
        targets = flagged + sample
        binary, model, mmproj, coordinates = prepare_vlm(project_root, args.vlm_model, args.language)
        emit(localized(args.language, f"Starte den lokalen VLM-Server ({args.vlm_model}) …", f"Starting the local VLM server ({args.vlm_model}) …"))
        with vlm.LlamaServer(binary, model, mmproj) as server:
            for index, frame in enumerate(targets, start=1):
                print(f"RECO_PROGRESS {index} {len(targets)}", file=sys.stderr, flush=True)
                image = project_root / frame["relativePath"]
                boxes = [box for box in frame.get("annotations", []) if box.get("category")]
                try:
                    answer = vlm.parse_json_answer(vlm.ask_image(server.url, image, vlm.frame_check_prompt(boxes, int(frame.get("width") or 0), int(frame.get("height") or 0))))
                except (OSError, ValueError, KeyError) as error:
                    emit(localized(args.language, f"VLM-Antwort für ein Bild unbrauchbar: {error}", f"Unusable VLM answer for one image: {error}"))
                    continue
                vlm_checked += 1
                missing, duplicates = vlm.as_count(answer.get("missing")), vlm.as_count(answer.get("duplicates"))
                frame["vlmCheck"] = {"model": args.vlm_model, "missing": missing, "duplicates": duplicates, "comment": str(answer.get("comment", ""))[:200]}
                if missing or duplicates:
                    frame["reviewFlags"].append("vlm-issue")
                    frame["hardExample"] = True
                    vlm_issues += 1
                elif frame.get("reviewFlags"):
                    # The VLM sees nothing wrong: the detector/time warning is resolved.
                    frame["reviewFlags"] = [flag for flag in frame["reviewFlags"] if flag not in CONSISTENCY_FLAGS]
                    vlm_resolved += 1

    for frame in frames:
        if not frame.get("reviewFlags"):
            frame.pop("reviewFlags", None)
    document["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    atomic_json(project_file(project_root), document)
    review = [frame["id"] for frame in frames if frame.get("reviewFlags")]
    summary = {
        "checked": len(frames), "detectorDisagreement": disagreement, "playersVanished": vanished,
        "vlmChecked": vlm_checked, "vlmIssues": vlm_issues, "vlmResolved": vlm_resolved,
        "hardExamples": sum(1 for frame in frames if frame.get("hardExample")), "review": review,
    }
    emit(localized(
        args.language,
        f"Konsistenzprüfung: {len(frames)} Bilder, {disagreement} mit abweichendem zweiten Detektor, {vanished} mit plötzlich fehlenden Spielern"
        + (f"; VLM: {vlm_checked} geprüft, {vlm_issues} beanstandet, {vlm_resolved} entwarnt" if args.vlm else "") + f". {len(review)} Bilder zur Prüfung markiert.",
        f"Consistency check: {len(frames)} images, {disagreement} where the second detector disagrees, {vanished} with players suddenly missing"
        + (f"; VLM: {vlm_checked} checked, {vlm_issues} with issues, {vlm_resolved} cleared" if args.vlm else "") + f". {len(review)} images flagged for review.",
    ))
    print(json.dumps(summary, ensure_ascii=False))


def is_imported_frame(frame: dict[str, Any]) -> bool:
    return str(frame.get("videoID", "")).startswith("import-")


def stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def split_frames(
    frames: list[dict[str, Any]], language: str = "de"
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    # Imported images carry no scene information (file names and simple image signatures do not
    # reveal it), so they only train next to real videos and never count as an independent test.
    imported = [frame for frame in frames if is_imported_frame(frame)]
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if not is_imported_frame(frame):
            by_video[frame["videoID"]].append(frame)
    video_ids = sorted(by_video, key=stable_bucket)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    warning = None

    if imported and not video_ids:
        # File order would put whole dataset versions into one split: mix by a stable hash.
        ordered = sorted(imported, key=lambda frame: stable_bucket(str(frame["id"])))
        train_end = max(1, int(len(ordered) * 0.7))
        valid_end = max(train_end + 1, int(len(ordered) * 0.9))
        output["train"] = ordered[:train_end]
        output["valid"] = ordered[train_end:valid_end]
        output["test"] = ordered[valid_end:]
        warning = localized(
            language,
            "Nur importierte Bilder: Der Split ist gemischt, gleiche Szenen können in Training und Test liegen; Qualitätswerte sind nicht unabhängig.",
            "Imported images only: the split is mixed, the same scenes can be in training and test; quality metrics are not independent.",
        )
        return output, warning
    output["train"].extend(imported)

    if len(video_ids) >= 3:
        test_ids = {video_ids[-1]}
        valid_ids = {video_ids[-2]}
        for video_id, items in by_video.items():
            split = "test" if video_id in test_ids else "valid" if video_id in valid_ids else "train"
            output[split].extend(items)
    elif len(video_ids) == 2:
        output["train"].extend(by_video[video_ids[0]])
        output["valid"].extend(by_video[video_ids[1]])
        warning = localized(
            language,
            "Nur zwei Videos: Es wird noch kein unabhängiger Test-Split erzeugt.",
            "Only two videos: no independent test split is created yet.",
        )
    else:
        ordered = sorted((frame for frame in frames if not is_imported_frame(frame)), key=lambda frame: frame["timestamp"])
        train_end = max(1, int(len(ordered) * 0.7))
        valid_end = max(train_end + 1, int(len(ordered) * 0.9))
        output["train"].extend(ordered[:train_end])
        output["valid"] = ordered[train_end:valid_end]
        output["test"] = ordered[valid_end:]
        warning = localized(
            language,
            "Nur ein Video: temporaler Split als Notlösung; Qualitätswerte sind nicht unabhängig.",
            "Only one video: a temporal fallback split is used; quality metrics are not independent.",
        )
    return output, warning


def write_anonymized_copy(anonymizer, source: Path, destination: Path, frame: dict[str, Any]) -> bool:
    """Write an anonymized copy as a new file. The dataset normally hard-links the frames, so
    writing into the existing link would change the original frame - unlink first."""
    import cv2
    image = cv2.imread(str(source))
    if image is None:
        return False
    sx = image.shape[1] / max(float(frame.get("width") or image.shape[1]), 1.0)
    sy = image.shape[0] / max(float(frame.get("height") or image.shape[0]), 1.0)
    scaled = [{**a, "x": a["x"] * sx, "y": a["y"] * sy, "width": a["width"] * sx, "height": a["height"] * sy} for a in frame.get("annotations", [])]
    result, _ = anonymizer.anonymize(image, scaled)
    ok, encoded = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.write_bytes(encoded.tobytes())
    return True


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def training_anonymizer(project_root: Path, fraction: float, language: str):
    """Anonymizer for "anonymization augmentation": None when off or when the face model is not
    available (then training simply runs without it)."""
    if fraction <= 0:
        return None
    try:
        import anonymize as anonymize_module
        face_model = download_registry_model(project_root, "yunet-face", language)
        return anonymize_module.Anonymizer(face_model)
    except (ImportError, OSError, SystemExit) as error:
        emit(localized(language, f"Hinweis: Anonymisierte Trainingsbilder übersprungen ({error}).", f"Note: anonymized training images skipped ({error})."))
        return None


def anonymize_for_training(frame: dict[str, Any], fraction: float) -> bool:
    """Stable choice: the same frames are anonymized in every run with the same share."""
    bucket = int(hashlib.sha1(f"anon:{frame.get('id')}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def build_augmented_dataset(project_root: Path, document: dict[str, Any], augmentation: dict[str, Any], categories: list[str] | None, language: str) -> Path:
    """Training data from the Augmentation Studio: real validation and test images by session,
    synthetic images only in training. "real-only" trains the same real split alone, which is
    the fair comparison run for the benchmark."""
    import augment_studio
    work = Path(str(augmentation.get("work", "")))
    if not (work / "output" / "split.json").is_file():
        raise SystemExit(localized(language, "Der erzeugte Datensatz wurde nicht gefunden. Im Augmentation Studio neu erzeugen oder dort abschalten.", "The generated dataset was not found. Generate it again in the Augmentation Studio or switch it off there.", "No se encontró el conjunto generado. Vuelve a generarlo en Augmentation Studio o desactívalo allí.", "Le jeu de données généré est introuvable. Générez-le à nouveau dans Augmentation Studio ou désactivez-le."))
    allowed = sport_categories(str(document.get("sport", "")))
    dataset_root = project_root / "dataset"
    counts = augment_studio.build_training_dataset(
        work, dataset_root, int(augmentation.get("realWeight", 3)), augmentation["mode"] == "mixed", categories,
        lambda name: normalized_label_class(name, allowed),
    )
    emit(localized(
        language,
        f"Augmentation Studio: Training {counts['train']}, Validierung {counts['valid']}, Test {counts['test']} Bilder (Validierung und Test nur echt).",
        f"Augmentation Studio: training {counts['train']}, validation {counts['valid']}, test {counts['test']} images (validation and test real only).",
    ))
    return dataset_root


def build_dataset(
    project_root: Path,
    language: str = "de",
    categories: list[str] | None = None,
    anonymize_fraction: float = 0.0,
) -> Path:
    """Builds the local COCO-format train/valid/test dataset.

    categories, when given, restricts the dataset to that subset of already-
    annotated categories (e.g. train only "ball" and "referee", leaving "player"
    out of this run entirely) - both the dataset's category list and every box
    of an excluded category are dropped, not just hidden. None (the default)
    keeps the existing behavior: every category that has at least one annotation.
    """
    document = require_project(project_root, language)
    augmentation = document.get("augmentation") or {}
    if augmentation.get("mode") in {"mixed", "real-only"}:
        return build_augmented_dataset(project_root, document, augmentation, categories, language)
    frames = [
        frame for frame in document.get("frames", [])
        if frame.get("reviewStatus") != "candidate" and not frame.get("heldOut")
    ]
    if not frames:
        raise SystemExit(localized(language, "Keine Trainingsframes vorhanden.", "No training frames are available."))
    category_names = {
        annotation["category"]
        for frame in frames
        for annotation in frame.get("annotations", [])
    }
    if not category_names:
        raise SystemExit(localized(
            language,
            "Noch keine Objekte markiert. Zuerst automatisch oder manuell markieren.",
            "No objects have been annotated yet. Add automatic or manual annotations first.",
        ))
    if categories is not None:
        category_names &= set(categories)
        if not category_names:
            raise SystemExit(localized(
                language,
                "Keine der ausgewählten Klassen ist im Projekt markiert.",
                "None of the selected classes are annotated in this project.",
            ))

    ordered_categories = [
        name for name in sport_categories(document["sport"]) if name in category_names
    ]
    category_id = {name: index + 1 for index, name in enumerate(ordered_categories)}
    anonymize_fraction = min(max(anonymize_fraction, 0.0), 0.5)
    anonymizer = training_anonymizer(project_root, anonymize_fraction, language)
    anonymized_count = 0
    splits, warning = split_frames(frames, language)
    if warning:
        emit(localized(language, f"Hinweis: {warning}", f"Note: {warning}"))

    dataset_root = project_root / "dataset"
    for split_name, split_frames_list in splits.items():
        split_dir = dataset_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        # Rebuild each derived split cleanly so removed training images cannot
        # remain as stale hard links or copies.
        for stale in split_dir.iterdir():
            if stale.is_file():
                stale.unlink()
        images = []
        annotations = []
        annotation_number = 1

        for image_number, frame in enumerate(split_frames_list, start=1):
            source = project_root / frame["relativePath"]
            file_name = source.name
            if split_name == "train" and frame.get("hardExample") and source.is_file():
                # Hard examples (the VLM found a problem there) go into training twice.
                hard_name = f"hard-{file_name}"
                link_or_copy(source, split_dir / hard_name)
                hard_number = len(split_frames_list) + image_number
                images.append({"id": hard_number, "file_name": hard_name, "width": frame["width"], "height": frame["height"]})
                for box in frame.get("annotations", []):
                    if box["category"] in category_id:
                        width, height = max(float(box["width"]), 1.0), max(float(box["height"]), 1.0)
                        annotations.append({"id": annotation_number, "image_id": hard_number, "category_id": category_id[box["category"]], "bbox": [float(box["x"]), float(box["y"]), width, height], "area": width * height, "iscrowd": 0})
                        annotation_number += 1
            if anonymizer is not None and split_name == "train" and anonymize_for_training(frame, anonymize_fraction) and write_anonymized_copy(anonymizer, source, split_dir / file_name, frame):
                anonymized_count += 1
            else:
                link_or_copy(source, split_dir / file_name)
            images.append(
                {
                    "id": image_number,
                    "file_name": file_name,
                    "width": frame["width"],
                    "height": frame["height"],
                }
            )
            for box in frame.get("annotations", []):
                if box["category"] not in category_id:
                    continue
                width = max(float(box["width"]), 1.0)
                height = max(float(box["height"]), 1.0)
                annotations.append(
                    {
                        "id": annotation_number,
                        "image_id": image_number,
                        "category_id": category_id[box["category"]],
                        "bbox": [float(box["x"]), float(box["y"]), width, height],
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )
                annotation_number += 1

        coco = {
            "info": {
                "description": f"Local Reco Trainer dataset: {document['sport']}",
                "version": "1",
                "date_created": datetime.now(timezone.utc).isoformat(),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": identifier, "name": name, "supercategory": "sport"}
                for name, identifier in category_id.items()
            ],
        }
        atomic_json(split_dir / "_annotations.coco.json", coco)
        if split_name == "train" and anonymizer is not None:
            emit(localized(
                language,
                f"{anonymized_count} Trainingsbilder anonymisiert (Köpfe weichgezeichnet), damit das Modell auch solche Bilder kennt.",
                f"{anonymized_count} training images anonymized (heads blurred) so the model also knows such images.",
            ))
        emit(localized(
            language,
            f"{split_name}: {len(images)} Bilder, {len(annotations)} Markierungen",
            f"{split_name}: {len(images)} images, {len(annotations)} annotations",
        ))
    return dataset_root


# --- Dataset export (YOLO / COCO), optionally anonymized ----------------------------------------

YOLO_SPLITS = {"train": "train", "valid": "val", "test": "test"}  # Reco split -> Ultralytics folder
PEOPLE_DETECTOR_IDS = {"nano": "rf-detr-nano", "small": "rf-detr-small", "medium": "rf-detr-medium", "large": "rf-detr-large"}


def export_frame_name(frame: dict[str, Any], suffix: str) -> str:
    """Neutral file name: no video name, no path, stable for the same frame."""
    return "frame-" + hashlib.sha1(str(frame.get("id")).encode("utf-8")).hexdigest()[:12] + suffix


def people_detector(project_root: Path, size: str, language: str):
    """RF-DETR COCO base model (Apache-2.0) for people without a label: spectators, bench."""
    try:
        import rfdetr
    except ImportError as error:
        raise SystemExit(localized(language, "RF-DETR ist nicht installiert. Zuerst „ML einrichten“ wählen.", "RF-DETR is not installed. Select “Set up ML” first.")) from error
    model_class = getattr(rfdetr, {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}[size])
    emit(localized(language, f"Lade Personendetektor RF-DETR {size} (COCO, Apache-2.0) …", f"Loading people detector RF-DETR {size} (COCO, Apache-2.0) …"))
    model = model_class(device=detect_device())

    def detect(image_bgr) -> list[tuple[float, float, float, float]]:
        prediction = normalize_detection_batch(model.predict(image_bgr[:, :, ::-1].copy(), threshold=0.35), 1)[0]
        boxes = []
        for box, class_id in zip(prediction.xyxy, prediction.class_id):
            if int(class_id) == 1:  # COCO "person"
                x1, y1, x2, y2 = (float(value) for value in box)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    return detect


def head_mosaic(image: Any, people: list[tuple[float, float, float, float]], tiles: int = 6, tile: int = 256) -> Any | None:
    """Enlarged head-and-shoulders crops of the largest people, side by side in one image, so the
    VLM can judge faces that are only a few pixels tall in a wide shot. None without people."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    chosen = sorted(people, key=lambda box: box[3], reverse=True)[:tiles]
    if not chosen:
        return None
    crops = []
    for x, y, w, h in chosen:
        x0, y0 = int(max(0, x - w * 0.25)), int(max(0, y - h * 0.08))
        x1, y1 = int(min(width, x + w * 1.25)), int(min(height, y + h * 0.42))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((tile, tile, 3), 127, dtype=image.dtype)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crops.append(canvas)
    if not crops:
        return None
    columns = min(3, len(crops))
    rows = [np.hstack(crops[index:index + columns] + [np.full((tile, tile, 3), 127, dtype=image.dtype)] * (columns - len(crops[index:index + columns]))) for index in range(0, len(crops), columns)]
    return np.vstack(rows)


def export_dataset(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    sport = str(document.get("sport", ""))
    formats = [item for item in str(args.formats).split(",") if item in {"yolo", "coco"}] or ["yolo", "coco"]
    frames = [frame for frame in document.get("frames", []) if frame.get("reviewStatus") != "candidate" and not frame.get("heldOut")]
    if not frames:
        raise SystemExit(localized(args.language, "Keine geprüften Bilder zum Exportieren.", "No reviewed images to export."))
    present = {annotation["category"] for frame in frames for annotation in frame.get("annotations", [])}
    categories = [name for name in sport_categories(sport) if name in present]
    if not categories:
        raise SystemExit(localized(args.language, "Noch keine Objekte markiert.", "No objects have been annotated yet."))
    class_index = {name: index for index, name in enumerate(categories)}
    splits, warning = split_frames(frames, args.language)
    if warning:
        emit(localized(args.language, f"Hinweis: {warning}", f"Note: {warning}"))

    anonymized = bool(args.anonymize)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"reco-{sport}-dataset-{stamp}" + ("-anonymized" if anonymized else "")
    out_root = project_root / "exports" / "datasets"
    final = out_root / name
    partial = out_root / f"{name}.partial"
    zip_path = out_root / f"{name}.zip"
    zip_partial = out_root / f"{name}.zip.part"
    review_path = out_root / f"{name}.review.json"
    steps = (["models"] if anonymized else []) + (["people"] if anonymized and args.detect_people else []) + ["write"] + (["zip"] if args.zip else [])
    step = lambda key, label: stage(steps.index(key) + 1, len(steps), label)  # noqa: E731

    used_models: list[dict[str, Any]] = []
    anonymizer = None
    detect_people = None
    vlm_server = None
    try:
        partial.mkdir(parents=True, exist_ok=False)
        if anonymized:
            import cv2  # noqa: F401  (needs the ML environment)
            import anonymize as anonymize_module
            step("models", localized(args.language, "Modelle prüfen", "Check models"))
            face_model = download_registry_model(project_root, "yunet-face", args.language)
            used_models.append(registry_model("yunet-face"))
            text_model = None
            if args.include_text:
                text_model = download_registry_model(project_root, "ppocr-text-detector", args.language)
                used_models.append(registry_model("ppocr-text-detector"))
            anonymizer = anonymize_module.Anonymizer(face_model, args.method, text_model)
            if getattr(args, "vlm_sample", 0) > 0:
                import vlm
                binary, model, mmproj, _coordinates = prepare_vlm(project_root, args.vlm_model, args.language)
                vlm_server = vlm.LlamaServer(binary, model, mmproj).start()
                used_models.append(registry_model(args.vlm_model))
            if args.detect_people:
                step("people", localized(args.language, "Personendetektor laden", "Load people detector"))
                size = args.detector_size if args.detector_size != "auto" else ("nano" if detect_device() == "cpu" else "medium")
                detect_people = people_detector(project_root, size, args.language)
                used_models.append(registry_model(PEOPLE_DETECTOR_IDS[size]))

        step("write", localized(args.language, "Bilder und Labels schreiben", "Write images and labels"))
        total = sum(len(items) for items in splits.values())
        done = 0
        counts: dict[str, dict[str, int]] = {}
        coco: dict[str, dict[str, Any]] = {}
        review: list[dict[str, Any]] = []
        for split, items in splits.items():
            yolo_split = YOLO_SPLITS[split]
            counts[split] = {"images": 0, "annotations": 0}
            coco[split] = {"images": [], "annotations": []}
            for frame in items:
                done += 1
                print(f"RECO_PROGRESS {done} {total}", file=sys.stderr, flush=True)
                source = project_root / str(frame["relativePath"])
                if not source.is_file():
                    continue
                boxes = [a for a in frame.get("annotations", []) if a.get("category") in class_index]
                frame_width, frame_height = float(frame.get("width") or 0), float(frame.get("height") or 0)
                suffix = ".jpg" if anonymized else source.suffix.lower()
                file_name = export_frame_name(frame, suffix)
                image_bytes: bytes | None = None
                image_width, image_height = frame_width, frame_height
                if anonymized:
                    import cv2
                    image = cv2.imread(str(source))
                    if image is None:
                        continue
                    image_height, image_width = image.shape[:2]
                    sx, sy = image_width / max(frame_width, 1.0), image_height / max(frame_height, 1.0)
                    scaled = [{**a, "x": a["x"] * sx, "y": a["y"] * sy, "width": a["width"] * sx, "height": a["height"] * sy} for a in frame.get("annotations", [])]
                    extra = detect_people(image) if detect_people else []
                    result, report = anonymizer.anonymize(image, scaled, extra, include_text=args.include_text)
                    if not report.safe:
                        review.append({"frameID": frame.get("id"), "split": split, "reason": "face-still-detected", "faces": len(report.residual_faces)})
                        continue
                    ok, encoded = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    if not ok:
                        continue
                    image_bytes = encoded.tobytes()
                    mosaic = head_mosaic(result, [(a["x"], a["y"], a["width"], a["height"]) for a in scaled if a.get("category") in {"player", "goalkeeper", "referee"}] + list(extra)) if vlm_server is not None else None
                    if mosaic is not None and stable_bucket(f"vlm:{frame.get('id')}") / 0xFFFFFFFF < args.vlm_sample:
                        # A sample of the anonymized images goes to the local VLM as a second check,
                        # as enlarged head crops (faces in a wide shot are too small for it otherwise).
                        import tempfile
                        import vlm
                        with tempfile.NamedTemporaryFile(suffix=".jpg") as probe:
                            probe.write(cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tobytes())
                            probe.flush()
                            try:
                                answer = vlm.parse_json_answer(vlm.ask_image(vlm_server.url, Path(probe.name), vlm.ANONYMIZATION_PROMPT))
                                recognizable = vlm.as_count(answer.get("recognizable_faces"))
                            except (OSError, ValueError, KeyError, AttributeError):
                                recognizable = 0
                        if recognizable:
                            review.append({"frameID": frame.get("id"), "split": split, "reason": "vlm-face-recognizable", "faces": recognizable})
                            continue
                image_width = image_width or frame_width
                image_height = image_height or frame_height
                targets = []
                if "yolo" in formats:
                    targets.append(partial / ("yolo" if len(formats) > 1 else "") / "images" / yolo_split / file_name)
                if "coco" in formats:
                    targets.append(partial / ("coco" if len(formats) > 1 else "") / split / file_name)
                for target in targets:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if image_bytes is not None:
                        target.write_bytes(image_bytes)
                    else:
                        shutil.copyfile(source, target)
                if "yolo" in formats:
                    lines = []
                    for a in boxes:
                        cx = (a["x"] + a["width"] / 2) / max(frame_width, 1.0)
                        cy = (a["y"] + a["height"] / 2) / max(frame_height, 1.0)
                        lines.append(f"{class_index[a['category']]} {cx:.6f} {cy:.6f} {a['width'] / max(frame_width, 1.0):.6f} {a['height'] / max(frame_height, 1.0):.6f}")
                    label = partial / ("yolo" if len(formats) > 1 else "") / "labels" / yolo_split / (Path(file_name).stem + ".txt")
                    label.parent.mkdir(parents=True, exist_ok=True)
                    label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                image_id = counts[split]["images"] + 1
                coco[split]["images"].append({"id": image_id, "file_name": file_name, "width": int(image_width), "height": int(image_height)})
                sx, sy = image_width / max(frame_width, 1.0), image_height / max(frame_height, 1.0)
                for a in boxes:
                    w, h = a["width"] * sx, a["height"] * sy
                    coco[split]["annotations"].append({
                        "id": len(coco[split]["annotations"]) + 1, "image_id": image_id, "category_id": class_index[a["category"]] + 1,
                        "bbox": [round(a["x"] * sx, 2), round(a["y"] * sy, 2), round(w, 2), round(h, 2)], "area": round(w * h, 2), "iscrowd": 0,
                    })
                counts[split]["images"] += 1
                counts[split]["annotations"] += len(boxes)

        version = args.app_version or "unknown"
        if "yolo" in formats:
            base = partial / ("yolo" if len(formats) > 1 else "")
            base.mkdir(parents=True, exist_ok=True)
            names = "\n".join(f"  {index}: {name}" for index, name in enumerate(categories))
            (base / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
        if "coco" in formats:
            base = partial / ("coco" if len(formats) > 1 else "")
            for split in splits:
                (base / split).mkdir(parents=True, exist_ok=True)
                atomic_json(base / split / "_annotations.coco.json", {
                    "info": {"description": f"Reco Trainer dataset: {sport}", "version": version, "date_created": datetime.now(timezone.utc).isoformat()},
                    "licenses": [], "images": coco[split]["images"], "annotations": coco[split]["annotations"],
                    "categories": [{"id": index + 1, "name": name, "supercategory": "sport"} for index, name in enumerate(categories)],
                })
        metadata = {
            "generator": "Reco Trainer", "version": version, "createdAt": datetime.now(timezone.utc).isoformat(),
            "sport": sport, "classes": categories, "formats": formats, "splits": counts,
            "splitMethod": "per video (frames of one video never share a split)",
            "anonymization": {
                "enabled": anonymized,
                "method": args.method if anonymized else None,
                "regions": ["head of every labelled person", "faces (YuNet)"] + (["people found by the detector"] if detect_people else []) + (["text (jersey numbers, scoreboard)"] if anonymized and args.include_text else []),
                "excludedForReview": len(review),
                "note": "Heads are blurred, not covered; labels are unchanged." if anonymized else None,
            },
            "models": [{key: entry.get(key) for key in ("id", "name", "license", "licenseUrl", "source")} for entry in used_models],
            "privacy": "Contains only the exported images and their labels; no video names, source paths or project data.",
        }
        atomic_json(partial / "metadata.json", metadata)
        (partial / "README.txt").write_text(
            f"Reco Trainer dataset ({sport}), version {version}\n"
            f"Classes: {', '.join(categories)}\n"
            + ("YOLO (Ultralytics): yolo/data.yaml, images/<split>, labels/<split>\n" if "yolo" in formats else "")
            + ("COCO (Roboflow / RF-DETR): coco/<train|valid|test>/_annotations.coco.json\n" if "coco" in formats else "")
            + ("Faces and heads are blurred; see metadata.json for the method and the models (with licenses) used.\n" if anonymized else ""),
            encoding="utf-8",
        )
        partial.replace(final)

        if args.zip:
            step("zip", localized(args.language, "ZIP-Datei schreiben", "Write ZIP file"))
            files = sorted(path for path in final.rglob("*") if path.is_file())
            with zipfile.ZipFile(zip_partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for index, path in enumerate(files, start=1):
                    archive.write(path, f"{name}/{path.relative_to(final).as_posix()}")
                    print(f"RECO_PROGRESS {index} {len(files)}", file=sys.stderr, flush=True)
            zip_partial.replace(zip_path)
        if review:
            # Next to the export, not inside it: frame IDs are project data and must not be shared.
            atomic_json(review_path, {"export": name, "frames": review})
        summary = {
            "folder": str(final), "zip": str(zip_path) if args.zip else None, "name": name,
            "images": sum(item["images"] for item in counts.values()), "annotations": sum(item["annotations"] for item in counts.values()),
            "splits": counts, "anonymized": anonymized, "excludedForReview": review, "reviewFile": str(review_path) if review else None,
            "models": metadata["models"],
        }
        emit(localized(
            args.language,
            f"Datensatz exportiert: {summary['images']} Bilder, {summary['annotations']} Boxen" + (f", {len(review)} Bilder zur Prüfung zurückgehalten." if review else "."),
            f"Dataset exported: {summary['images']} images, {summary['annotations']} boxes" + (f", {len(review)} images held back for review." if review else "."),
        ))
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        if vlm_server is not None:
            vlm_server.stop()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        zip_partial.unlink(missing_ok=True)


def sport_categories(sport: str) -> list[str]:
    if sport == "basketball":
        return ["ball", "player", "referee", "hoop"]
    if sport == "hockey":
        return ["puck", "player", "goalkeeper", "referee", "goal"]
    if sport in {"rugby", "american_football"}:
        return ["ball", "player", "referee", "goalpost"]
    return ["ball", "player", "goalkeeper", "referee", "goal"]


def package_description(sport: str, model_size: str, classes: list[str]) -> str:
    """Return the canonical English-only description stored in exchange packages."""
    sport_name = {
        "football": "football",
        "futsal": "futsal",
        "basketball": "basketball",
        "handball": "handball",
        "hockey": "hockey",
        "rugby": "rugby",
        "lacrosse": "lacrosse",
        "american_football": "American football",
    }.get(sport, sport.replace("_", " "))
    objects = ", ".join(classes) if classes else "sport objects"
    return (
        f"Locally fine-tuned RF-DETR {model_size.capitalize()} model for {sport_name} "
        f"{objects} detection. Contains model weights and aggregate training metadata only; "
        "no videos or frames."
    )


def validation_score(metrics: dict[str, Any] | None) -> float | None:
    """Return the strict-box validation score used to compare local runs."""
    if not isinstance(metrics, dict):
        return None
    for key, value in metrics.items():
        normalized = str(key).lower().replace("val/", "").replace("val_", "").replace("test/", "").replace("test_", "")
        if normalized in {"map_50_95", "map50_95", "ap/ball", "ap_ball"}:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def manifest_validation_score(manifest: dict[str, Any] | None) -> float | None:
    if not isinstance(manifest, dict):
        return None
    direct = validation_score(manifest.get("validationMetrics"))
    if direct is not None:
        return direct
    summary = manifest.get("trainingSummary")
    return validation_score(summary.get("validationMetrics")) if isinstance(summary, dict) else None


def manifest_test_score(manifest: dict[str, Any] | None) -> float | None:
    if not isinstance(manifest, dict):
        return None
    direct = validation_score(manifest.get("testMetrics"))
    if direct is not None:
        return direct
    summary = manifest.get("trainingSummary")
    return validation_score(summary.get("testMetrics")) if isinstance(summary, dict) else None


def manifest_comparison_score(manifest: dict[str, Any] | None) -> float | None:
    test_score = manifest_test_score(manifest)
    return test_score if test_score is not None else manifest_validation_score(manifest)


def latest_test_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        return {}
    latest: dict[str, float] = {}
    try:
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("test/mAP_50_95"):
                    continue
                parsed: dict[str, float] = {}
                for key, value in row.items():
                    if key.startswith("test/") and value not in {None, ""}:
                        try:
                            parsed[key] = float(value)
                        except (TypeError, ValueError):
                            pass
                if parsed:
                    latest = parsed
    except (OSError, csv.Error):
        return {}
    return latest


def library_manifests(project_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    library_root = project_root / "models" / "library"
    records: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(library_root.glob("*/manifest.json")) if library_root.is_dir() else []:
        try:
            manifest = load_json(manifest_path)
            if isinstance(manifest, dict) and manifest.get("packageID"):
                records.append((manifest_path, manifest))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return records


def activate_library_model(project_root: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ensemble = manifest.get("ensemble")
    weights_field = ""
    if ensemble:
        # A combined model has no single checkpoint - auto_label()/benchmark()
        # look up its members via active_model_manifest() instead of this
        # weights path, which stays blank (newest_checkpoint() already treats
        # a blank/invalid weights path as "no single checkpoint", the same
        # safe fallback as an activation with no weights at all).
        for member in ensemble.get("members", []):
            member_path = manifest_path.parent / str(member.get("weightsFile", ""))
            if not member.get("weightsFile") or not member_path.is_file():
                raise ValueError("Combined model weights are missing")
    else:
        weight_name = Path(str(manifest.get("weights", {}).get("file", ""))).name
        weight_path = manifest_path.parent / "weights" / weight_name
        if not weight_name or not weight_path.is_file():
            raise ValueError("Model weights are missing")
        weights_field = str(weight_path.relative_to(project_root))
    active = {
        "packageID": str(manifest["packageID"]),
        "displayName": manifest.get("displayName") or manifest["packageID"],
        "modelSize": manifest.get("modelSize"),
        "sport": manifest.get("sport"),
        "description": manifest.get("description"),
        "format": manifest.get("format"),
        "weights": weights_field,
        "activatedAt": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(project_root / "models" / "active.json", active)
    return active


def archive_local_checkpoint(
    project_root: Path,
    document: dict[str, Any],
    model_size: str,
    checkpoint: Path,
    training_summary: dict[str, Any] | None,
    source: str,
) -> tuple[Path, dict[str, Any], bool]:
    """Copy one immutable local model revision into the model library."""
    digest = sha256_file(checkpoint)
    for manifest_path, manifest in library_manifests(project_root):
        if manifest.get("weights", {}).get("sha256") == digest:
            return manifest_path, manifest, False

    now = datetime.now(timezone.utc)
    package_id = f"reco-{document['sport']}-{model_size}-{now.strftime('%Y%m%d-%H%M%S-%f')}"
    sport_title = str(document["sport"]).replace("_", " ").title()
    display_name = f"{sport_title} · {model_size.capitalize()} · {now.strftime('%Y-%m-%d %H:%M UTC')}"
    classes = sorted({
        str(annotation.get("category"))
        for frame in document.get("frames", [])
        if frame.get("reviewStatus") != "candidate"
        for annotation in frame.get("annotations", [])
        if annotation.get("category")
    })
    metrics = (training_summary or {}).get("validationMetrics", {})
    test_metrics = (training_summary or {}).get("testMetrics", {})
    manifest = {
        "schemaVersion": 1,
        "packageID": package_id,
        "displayName": display_name,
        "createdAt": now.isoformat(),
        "sport": document["sport"],
        "modelSize": model_size,
        "classes": classes,
        "description": package_description(document["sport"], model_size, classes),
        "source": source,
        "trainingSummary": training_summary,
        "validationMetrics": metrics,
        "testMetrics": test_metrics,
        "statistics": {
            "frameCount": len([frame for frame in document.get("frames", []) if frame.get("reviewStatus") != "candidate"]),
            "annotationCount": sum(len(frame.get("annotations", [])) for frame in document.get("frames", []) if frame.get("reviewStatus") != "candidate"),
        },
        "weights": {"file": f"weights/{checkpoint.name}", "sha256": digest},
        "privacy": "This library entry contains model weights and aggregate metadata only; no videos or frames.",
    }
    final = project_root / "models" / "library" / package_id
    staging = project_root / "models" / f".archive-{uuid.uuid4().hex}"
    try:
        (staging / "weights").mkdir(parents=True, exist_ok=False)
        shutil.copy2(checkpoint, staging / "weights" / checkpoint.name)
        atomic_json(staging / "manifest.json", manifest)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final / "manifest.json", manifest, True


def interrupted_run_checkpoint(
    project_root: Path, size: str, document: dict[str, Any]
) -> Path | None:
    """Return a full Lightning checkpoint only when the previous run did not finish.

    A completed Reco Trainer run writes ``lastTraining.completedAt`` after RF-DETR
    has produced its checkpoints and completed evaluation. A newer ``last.ckpt``
    therefore belongs to an interrupted run and can safely restore optimizer,
    scheduler, EMA, and callback state.
    """
    checkpoint = project_root / "runs" / size / "last.ckpt"
    if not checkpoint.is_file():
        return None
    completed_at = (document.get("lastTraining") or {}).get("completedAt")
    if not completed_at:
        return checkpoint
    try:
        completed = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        return checkpoint if checkpoint.stat().st_mtime > completed.timestamp() + 1.0 else None
    except (OSError, TypeError, ValueError):
        return None


def checkpoint_completed_epochs(checkpoint: Path) -> int | None:
    """Read the zero-based Lightning epoch without allowing arbitrary pickle code."""
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        epoch = int(payload.get("epoch", -1))
        return epoch + 1 if epoch >= 0 else None
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return None


def checkpoint_class_count(checkpoint: Path) -> int | None:
    """Read how many classes a full Lightning checkpoint's detection head was built for.

    Guards against Lightning's strict trainer.fit(ckpt_path=...) resume hard-crashing
    with a size-mismatch RuntimeError when the project's class set changed since an
    interrupted run last saved last.ckpt (e.g. a class was added via auto-labeling
    after training started). Returns None if this can't be determined, in which case
    the caller should still allow the resume rather than block a previously working path.
    """
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weight = payload["state_dict"]["model.class_embed.weight"]
        return int(weight.shape[0])
    except (ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError, AttributeError):
        return None


def dataset_class_count(dataset_root: Path, split: str) -> int | None:
    try:
        document = load_json(dataset_root / split / "_annotations.coco.json")
        return len(document.get("categories", []))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def evaluate_trained_model(
    model: Any,
    model_class: Any,
    best_checkpoint: Path | None,
    device: str,
    dataset_root: Path,
    language: str = "de",
) -> dict[str, float]:
    """Validation metrics of the checkpoint that will actually be archived.

    After model.train() the in-memory model holds the weights of the LAST epoch, but the
    archived (and compared) checkpoint is the BEST one. Evaluating the in-memory model
    would store a quality score for different, usually worse, weights - so load the best
    checkpoint into a fresh model and evaluate that. If it cannot be loaded, fall back to
    the in-memory model and say so.
    """
    evaluated = model
    if best_checkpoint is not None:
        try:
            evaluated = model_class(pretrain_weights=str(best_checkpoint), device=device)
        except Exception as error:
            evaluated = model
            emit(localized(
                language,
                f"Hinweis: Der beste Checkpoint konnte für die Bewertung nicht geladen werden ({error}); bewertet wird der letzte Trainingsstand.",
                f"Note: the best checkpoint could not be loaded for evaluation ({error}); the final training state is evaluated instead.",
            ))
    if not hasattr(evaluated, "evaluate"):
        return {}
    try:
        emit(localized(language, "Bewerte das Modell auf dem Validierungssatz …", "Evaluating the model on the validation split …"))
        return numeric_metrics(evaluated.evaluate(dataset_dir=str(dataset_root), split="val"))
    except Exception as error:
        emit(localized(
            language,
            f"Hinweis: Die Qualitätsbewertung war mit dieser RF-DETR-Version nicht verfügbar: {error}",
            f"Note: quality evaluation was unavailable with this RF-DETR version: {error}",
        ))
        return {}


def validate_resume_checkpoint(
    resume_checkpoint: Path | None, dataset_root: Path, language: str = "de"
) -> Path | None:
    """Reject a full-resume checkpoint whose class count no longer matches the project.

    A strict Lightning resume (trainer.fit(ckpt_path=...)) hard-crashes with a
    RuntimeError if the checkpoint's detection head doesn't match the current class
    count - e.g. a class was added via auto-labeling after the run was interrupted.
    Returning None here makes train() fall back to a normal (non-strict) fine-tuning
    cycle instead of letting the whole run abort on a size mismatch. If either class
    count can't be determined, the checkpoint is allowed through unchanged rather than
    blocking an otherwise-working resume.
    """
    if resume_checkpoint is None:
        return None
    current_classes = dataset_class_count(dataset_root, "train")
    checkpoint_classes = checkpoint_class_count(resume_checkpoint)
    if checkpoint_classes is None or current_classes is None or checkpoint_classes == current_classes:
        return resume_checkpoint
    emit(localized(
        language,
        f"Der abgebrochene Trainingslauf wurde für {checkpoint_classes} Klassen begonnen, das Projekt hat jetzt {current_classes}. "
        "Vollständige Fortsetzung übersprungen; es wird stattdessen ein neuer Feinabstimmungslauf gestartet.",
        f"The interrupted run was started with {checkpoint_classes} classes, the project now has {current_classes}. "
        "Skipping the full resume; starting a new fine-tuning cycle instead.",
        f"El entrenamiento interrumpido se inició con {checkpoint_classes} clases; el proyecto ahora tiene {current_classes}. "
        "Se omite la reanudación completa; se inicia un nuevo ciclo de ajuste fino.",
        f"L’entraînement interrompu avait commencé avec {checkpoint_classes} classes ; le projet en compte désormais {current_classes}. "
        "Reprise complète ignorée ; un nouveau cycle d’ajustement fin démarre à la place.",
    ))
    return None


def archive_run_logs(run_dir: Path) -> Path | None:
    """Preserve small text logs before RF-DETR refreshes its mutable run folder."""
    names = ("metrics.csv", "training_config.json")
    existing = [run_dir / name for name in names if (run_dir / name).is_file()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    destination = run_dir / "history" / f"{stamp}-{uuid.uuid4().hex[:6]}"
    destination.mkdir(parents=True, exist_ok=False)
    for source in existing:
        shutil.copy2(source, destination / source.name)
    return destination


def dataset_annotation_count(dataset_root: Path, split: str) -> int:
    try:
        document = load_json(dataset_root / split / "_annotations.coco.json")
        return len(document.get("annotations", []))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def train(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    training_categories = list(dict.fromkeys(args.category)) if getattr(args, "category", None) else None
    stage(1, 3, localized(args.language, "Datensatz vorbereiten", "Prepare dataset", "Preparar el conjunto de datos", "Préparer le jeu de données"))
    dataset_root = build_dataset(project_root, args.language, categories=training_categories, anonymize_fraction=float(getattr(args, "anonymize_fraction", 0.0) or 0.0))
    if training_categories is not None:
        emit(localized(
            args.language,
            f"Training beschränkt auf: {', '.join(training_categories)}.",
            f"Training restricted to: {', '.join(training_categories)}.",
            f"Entrenamiento restringido a: {', '.join(training_categories)}.",
            f"Entraînement limité à : {', '.join(training_categories)}.",
        ))
    model_class = import_model_class(args.model, args.language)
    device = detect_device()
    profile = training_profile(args.model, device)
    output_dir = project_root / "runs" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    # fresh_start means "guaranteed clean slate": ignore both an interrupted run's
    # own last.ckpt and any completed continuation checkpoint, always starting
    # from the Apache-2.0 base model. continuation_checkpoint itself is still
    # computed below (unaffected) since the legacy-checkpoint archiving safety
    # net just below needs to know whether one exists on disk regardless of
    # whether this run actually starts from it - only use_continuation_checkpoint
    # (and resume_checkpoint) reflect the fresh_start override.
    fresh_start = bool(getattr(args, "fresh_start", False))
    continuation_checkpoint = newest_checkpoint(project_root, args.model)
    resume_checkpoint = None if fresh_start else validate_resume_checkpoint(
        interrupted_run_checkpoint(project_root, args.model, document), dataset_root, args.language
    )
    resume_completed_epochs = checkpoint_completed_epochs(resume_checkpoint) if resume_checkpoint else None
    use_continuation_checkpoint = None if fresh_start else continuation_checkpoint
    archived_logs = archive_run_logs(output_dir)

    if archived_logs is not None:
        emit(localized(
            args.language,
            f"Vorherige Messprotokolle gesichert: {archived_logs}",
            f"Previous metric logs preserved: {archived_logs}",
            f"Registros de métricas anteriores conservados: {archived_logs}",
            f"Journaux de mesures précédents conservés : {archived_logs}",
        ))

    validation_annotations = dataset_annotation_count(dataset_root, "valid")
    test_annotations = dataset_annotation_count(dataset_root, "test")
    if validation_annotations < 100 or (test_annotations and test_annotations < 100):
        emit(localized(
            args.language,
            f"Hinweis: Die Qualitätswerte sind noch vorläufig (Validierung: "
            f"{validation_annotations}, Test: {test_annotations} Markierungen). "
            "Für belastbare Modellvergleiche werden je Split mindestens 100, besser 300 Markierungen empfohlen.",
            f"Note: quality metrics are still preliminary (validation: "
            f"{validation_annotations}, test: {test_annotations} annotations). "
            "At least 100, preferably 300 annotations per split are recommended for reliable model comparisons.",
            f"Nota: las métricas de calidad aún son preliminares (validación: "
            f"{validation_annotations}, prueba: {test_annotations} anotaciones). "
            "Para comparar modelos de forma fiable se recomiendan al menos 100, preferiblemente 300 anotaciones por conjunto.",
            f"Remarque : les mesures de qualité sont encore provisoires (validation : "
            f"{validation_annotations}, test : {test_annotations} annotations). "
            "Pour comparer les modèles de manière fiable, au moins 100, idéalement 300 annotations par jeu sont recommandées.",
        ))

    # Older Reco Trainer versions kept only one mutable run checkpoint. Preserve
    # it before RF-DETR writes the next run into the same directory.
    if continuation_checkpoint is not None and (project_root / "runs") in continuation_checkpoint.parents:
        legacy_summary = dict(document.get("lastTraining") or {})
        if not legacy_summary.get("testMetrics"):
            legacy_summary["testMetrics"] = latest_test_metrics(output_dir)
        parent_manifest_path, parent_manifest, parent_created = archive_local_checkpoint(
            project_root,
            document,
            args.model,
            continuation_checkpoint,
            legacy_summary,
            "legacy-local-training",
        )
        active_path = project_root / "models" / "active.json"
        if parent_created or not active_path.is_file():
            activate_library_model(project_root, parent_manifest_path, parent_manifest)
        emit(localized(
            args.language,
            f"Vorherigen Modellstand sicher archiviert: {parent_manifest.get('displayName', parent_manifest['packageID'])}",
            f"Previous model revision safely archived: {parent_manifest.get('displayName', parent_manifest['packageID'])}",
        ))

    emit(localized(
        args.language,
        f"Training startet auf {device.upper()} mit RF-DETR {args.model}. ",
        f"Training starts on {device.upper()} with RF-DETR {args.model}. ",
    ))
    emit(localized(
        args.language,
        f"Leistungsprofil: {profile['memory_gb']} GB gemeinsamer Speicher, "
        f"Batch {profile['batch_size']}, {profile['num_workers']} parallele Datenlader, "
        f"Gradienten-Akkumulation {profile['grad_accum_steps']}×.",
        f"Performance profile: {profile['memory_gb']} GB unified memory, "
        f"batch {profile['batch_size']}, {profile['num_workers']} parallel data loaders, "
        f"gradient accumulation {profile['grad_accum_steps']}×.",
    ))
    if device == "mps":
        emit(localized(
            args.language,
            "Apple-GPU-Kerne werden über Metal/MPS genutzt. Die Neural Engine wird beim "
            "Core-ML-Modell für die spätere Erkennung genutzt, nicht für dieses PyTorch-Training.",
            "Apple GPU cores are used through Metal/MPS. The Neural Engine is used by the "
            "Core ML model for later detection, not by this PyTorch training run.",
        ))
    model_kwargs: dict[str, Any] = {
        "device": device,
        "gradient_checkpointing": profile["gradient_checkpointing"],
    }
    if resume_checkpoint is not None:
        emit(localized(
            args.language,
            f"Abgebrochenen Trainingslauf vollständig fortsetzen: {resume_checkpoint.name}"
            + (f" (nach {resume_completed_epochs} Epochen)" if resume_completed_epochs is not None else ""),
            f"Fully resuming interrupted training run: {resume_checkpoint.name}"
            + (f" (after {resume_completed_epochs} epochs)" if resume_completed_epochs is not None else ""),
            f"Reanudando por completo el entrenamiento interrumpido: {resume_checkpoint.name}"
            + (f" (después de {resume_completed_epochs} épocas)" if resume_completed_epochs is not None else ""),
            f"Reprise complète de l’entraînement interrompu : {resume_checkpoint.name}"
            + (f" (après {resume_completed_epochs} époques)" if resume_completed_epochs is not None else ""),
        ))
    elif fresh_start:
        emit(localized(
            args.language,
            "Von Grund auf neu trainieren angefordert: Training startet mit dem Apache-2.0-Basismodell, ohne einen vorhandenen Checkpoint fortzusetzen.",
            "Training from scratch requested: training starts from the Apache-2.0 base model, without continuing from any existing checkpoint.",
            "Se solicitó entrenar desde cero: el entrenamiento comienza con el modelo base Apache-2.0, sin continuar desde ningún checkpoint existente.",
            "Entraînement à partir de zéro demandé : l’entraînement démarre avec le modèle de base Apache-2.0, sans reprendre un checkpoint existant.",
        ))
    elif use_continuation_checkpoint is not None:
        model_kwargs["pretrain_weights"] = str(use_continuation_checkpoint)
        emit(localized(
            args.language,
            f"Neuer Feinabstimmungslauf mit dem besten Modell: {use_continuation_checkpoint.name}",
            f"Starting a new fine-tuning cycle from the best model: {use_continuation_checkpoint.name}",
            f"Iniciando un nuevo ajuste fino desde el mejor modelo: {use_continuation_checkpoint.name}",
            f"Nouveau cycle d’ajustement à partir du meilleur modèle : {use_continuation_checkpoint.name}",
        ))
    else:
        emit(localized(
            args.language,
            "Noch kein kompatibler Checkpoint vorhanden; Training startet mit dem Basismodell.",
            "No compatible checkpoint exists yet; training starts from the base model.",
        ))
    # Without an explicit num_classes, RF-DETR infers the detection head's size from
    # whichever checkpoint pretrain_weights points to (falling back to 90 COCO classes
    # for the base model) rather than from the dataset it's about to train on. If that
    # checkpoint was saved with fewer classes than the project now has - the same class
    # of problem validate_resume_checkpoint() guards against for a full resume, but here
    # for the weights-only continuation path - training would silently run with too few
    # classes and crash later when a training batch has a target class index that
    # doesn't fit, deep inside loss matching rather than at a clear startup check.
    current_class_count = dataset_class_count(dataset_root, "train")
    if current_class_count is not None:
        model_kwargs["num_classes"] = current_class_count
    model, ignored_model_kwargs = call_with_supported_kwargs(model_class, model_kwargs)
    if ignored_model_kwargs:
        emit(localized(
            args.language,
            f"Die installierte RF-DETR-Version kennt diese Modell-Optionen noch nicht: {', '.join(ignored_model_kwargs)}.",
            f"The installed RF-DETR version does not yet support these model options: {', '.join(ignored_model_kwargs)}.",
        ))
    target_epochs = max(args.epochs, (resume_completed_epochs or 0) + 1) if resume_checkpoint else args.epochs
    patience = max(8, min(20, math.ceil(args.epochs * 0.5)))
    train_kwargs = {
        "dataset_dir": str(dataset_root),
        "output_dir": str(output_dir),
        "epochs": target_epochs,
        "batch_size": profile["batch_size"],
        "grad_accum_steps": profile["grad_accum_steps"],
        "device": device,
        "num_workers": profile["num_workers"],
        "persistent_workers": profile["persistent_workers"],
        "prefetch_factor": profile["prefetch_factor"],
        "pin_memory": profile["pin_memory"],
        "early_stopping": True,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": 0.001,
        "eval_ema_only": True,
        "tensorboard": False,
        "seed": 42,
        "run_test": bool(list((dataset_root / "test").glob("*.jpg"))),
        "notes": {
            "sport": document["sport"],
            "privacy": "trained locally; source media not uploaded",
            "reco_trainer_schema": document.get("schemaVersion", 1),
            "performance_profile": profile,
        },
    }
    if resume_checkpoint is not None:
        train_kwargs["resume"] = str(resume_checkpoint)
    else:
        # Short local cycles benefit from cosine decay. For repeated fine-tuning,
        # use half the base learning rate to retain previously learned examples.
        train_kwargs.update({
            "lr_scheduler": "cosine",
            "lr_scheduler_kwargs": {"min_factor": 0.1},
            "warmup_epochs": 0.5,
        })
        if use_continuation_checkpoint is not None:
            train_kwargs.update({"lr": 0.00005, "lr_encoder": 0.000075})
    stage(2, 3, localized(args.language, "Trainieren", "Train", "Entrenar", "Entraîner"))
    watcher = EpochProgressWatcher(output_dir / "metrics.csv", target_epochs, args.language)
    watcher.start()
    try:
        _, ignored = call_with_supported_kwargs(model.train, train_kwargs)
    finally:
        watcher.stop()
    if ignored:
        emit(localized(
            args.language,
            f"Die installierte RF-DETR-Version kennt diese optionalen Optimierungen noch nicht: "
            f"{', '.join(ignored)}.",
            f"The installed RF-DETR version does not yet support these optional optimizations: "
            f"{', '.join(ignored)}.",
        ))
    stage(3, 3, localized(args.language, "Bestes Modell bewerten", "Evaluate the best model", "Evaluar el mejor modelo", "Évaluer le meilleur modèle"))
    validation_metrics = evaluate_trained_model(
        model, model_class, preferred_run_checkpoint(output_dir), device, dataset_root, args.language
    )

    refreshed = require_project(project_root, args.language)
    refreshed_frames = [
        frame for frame in refreshed.get("frames", [])
        if frame.get("reviewStatus") != "candidate"
    ]
    refreshed_splits, _ = split_frames(refreshed_frames, args.language)
    all_annotations = [
        annotation
        for frame in refreshed_frames
        for annotation in frame.get("annotations", [])
    ]
    trained_checkpoint = preferred_run_checkpoint(output_dir)
    test_metrics = latest_test_metrics(output_dir)
    training_result = {
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": device,
        "epochsRequested": args.epochs,
        "frameCount": len(refreshed_frames),
        "annotatedFrames": sum(bool(frame.get("annotations")) for frame in refreshed_frames),
        "annotationCount": len(all_annotations),
        "classes": sorted({annotation.get("category") for annotation in all_annotations}),
        "splits": {name: len(items) for name, items in refreshed_splits.items()},
        "independentTest": len({frame.get("videoID") for frame in refreshed_frames if not is_imported_frame(frame)}) >= 3,
        "checkpoint": trained_checkpoint.name if trained_checkpoint else None,
        "continuedFrom": (
            resume_checkpoint.name if resume_checkpoint else
            use_continuation_checkpoint.name if use_continuation_checkpoint else None
        ),
        "resumeMode": "full" if resume_checkpoint else "weights" if use_continuation_checkpoint else "base",
        "performanceProfile": profile,
        "validationMetrics": validation_metrics,
        "testMetrics": test_metrics,
    }
    refreshed["lastTraining"] = training_result
    refreshed["trainingHistory"] = [*refreshed.get("trainingHistory", []), training_result][-12:]
    refreshed["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    atomic_json(project_file(project_root), refreshed)

    if trained_checkpoint is not None:
        manifest_path, manifest, _ = archive_local_checkpoint(
            project_root, refreshed, args.model, trained_checkpoint, training_result, "local-training"
        )
        active_path = project_root / "models" / "active.json"
        current_manifest: dict[str, Any] | None = None
        if active_path.is_file():
            try:
                active_id = str(load_json(active_path).get("packageID", ""))
                current_manifest = next(
                    (item for _, item in library_manifests(project_root) if item.get("packageID") == active_id),
                    None,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                current_manifest = None
        new_score = manifest_comparison_score(manifest)
        current_score = manifest_comparison_score(current_manifest)
        # Aggregate quality scores average over a model's own classes, so a ball-only
        # model and a ball+player+referee model are not comparable - never let the
        # score alone swap between them; per-class comparison (Modellvergleich) decides.
        comparable = current_manifest is None or (
            {str(item) for item in manifest.get("classes", [])}
            == {str(item) for item in current_manifest.get("classes", [])}
        )
        promote = current_manifest is None or (
            comparable and new_score is not None and (current_score is None or new_score > current_score)
        )
        if promote:
            activate_library_model(project_root, manifest_path, manifest)
            emit(localized(
                args.language,
                f"Neuer bester Modellstand aktiviert: {manifest['displayName']}",
                f"New best model revision activated: {manifest['displayName']}",
            ))
        elif not comparable:
            current_classes = ", ".join(sorted(str(item) for item in (current_manifest or {}).get("classes", [])))
            new_classes = ", ".join(sorted(str(item) for item in manifest.get("classes", [])))
            emit(localized(
                args.language,
                f"Modellstand archiviert, aber nicht automatisch aktiviert: Das aktive Modell kennt andere Klassen ({current_classes}) als das neue ({new_classes}); die Qualitätswerte sind nicht vergleichbar. Im Modellvergleich pro Kategorie prüfen und ggf. ein kombiniertes Modell erstellen.",
                f"Model revision archived but not automatically activated: the active model knows different classes ({current_classes}) than the new one ({new_classes}), so their quality scores are not comparable. Compare per category in the model comparison and, if useful, create a combined model.",
                f"Revisión del modelo archivada pero no activada automáticamente: el modelo activo conoce otras clases ({current_classes}) que el nuevo ({new_classes}), por lo que sus puntuaciones no son comparables. Compáralos por categoría en la comparación de modelos y, si conviene, crea un modelo combinado.",
                f"Révision du modèle archivée mais non activée automatiquement : le modèle actif connaît d’autres classes ({current_classes}) que le nouveau ({new_classes}), leurs scores ne sont donc pas comparables. Comparez par catégorie dans la comparaison de modèles et créez au besoin un modèle combiné.",
            ))
        else:
            emit(localized(
                args.language,
                f"Modellstand archiviert, aber nicht aktiviert (Qualität {new_score or 0:.4f}; bisher {current_score or 0:.4f}).",
                f"Model revision archived but not activated (quality {new_score or 0:.4f}; current {current_score or 0:.4f}).",
            ))
    emit(localized(
        args.language,
        f"Training abgeschlossen. Modell liegt in {output_dir}",
        f"Training completed. The model is stored in {output_dir}",
    ))


def combine_models(args: argparse.Namespace) -> None:
    """Build a combined ("baked") model from already-installed library models,
    each contributing only the categories it was assigned - e.g. one model's
    referee detection plus another's ball detection, as one deliverable.

    This is an ensemble at inference time (auto_label()/benchmark() run each
    member for its own categories and merge predictions), not real weight
    merging - RF-DETR's joint classification head makes true weight merging
    infeasible. Installs directly into this project's model library, like any
    other model, since every source model is already local: there's no reason
    to force a package/export + re-import round trip for the common case of
    "use the better combination right here". Exporting a combined model as a
    shareable .recomodel isn't built yet (package_model() still only knows
    how to package a single fresh checkpoint) - install_model_package() and
    validate_model_package_file() already accept a multi-weights ensemble
    manifest so that export can be added later without another schema change.
    """
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    sport = str(document["sport"])
    assignments: list[tuple[str, list[str]]] = []
    for raw in args.member:
        package_id, _, category_list = str(raw).partition(":")
        categories = [category.strip() for category in category_list.split(",") if category.strip()]
        if not package_id or not categories:
            raise SystemExit(localized(args.language, f"Ungültige Modellzuordnung: {raw}", f"Invalid model assignment: {raw}"))
        assignments.append((package_id, categories))
    if len(assignments) < 2:
        raise SystemExit(localized(
            args.language,
            "Mindestens zwei Modelle mit je mindestens einer Kategorie werden für ein kombiniertes Modell benötigt.",
            "A combined model needs at least two models, each with at least one category.",
        ))
    seen_categories: set[str] = set()
    for _, categories in assignments:
        duplicate = seen_categories.intersection(categories)
        if duplicate:
            raise SystemExit(localized(
                args.language,
                f"Kategorie(n) mehrfach zugeordnet: {', '.join(sorted(duplicate))}",
                f"Categories assigned more than once: {', '.join(sorted(duplicate))}",
            ))
        seen_categories.update(categories)

    manifests_by_id = {str(manifest.get("packageID")): (path, manifest) for path, manifest in library_manifests(project_root)}
    members: list[dict[str, Any]] = []
    model_sizes: set[str] = set()
    for package_id, categories in assignments:
        found = manifests_by_id.get(package_id)
        if found is None:
            raise SystemExit(localized(args.language, f"Modell nicht in der Bibliothek gefunden: {package_id}", f"Model not found in the library: {package_id}"))
        source_path, source_manifest = found
        if source_manifest.get("ensemble"):
            raise SystemExit(localized(args.language, "Ein kombiniertes Modell kann nicht selbst wieder kombiniert werden.", "A combined model cannot itself be combined again."))
        if is_onnx_manifest(source_manifest):
            raise SystemExit(localized(args.language, f"Importierte ONNX-Modelle können nicht kombiniert werden: {package_id}", f"Imported ONNX models cannot be combined: {package_id}"))
        if source_manifest.get("sport") != sport:
            raise SystemExit(localized(args.language, f"Modell {package_id} gehört zu einer anderen Sportart.", f"Model {package_id} belongs to a different sport."))
        unsupported = set(categories) - {str(item) for item in source_manifest.get("classes", [])}
        if unsupported:
            raise SystemExit(localized(
                args.language,
                f"Modell {package_id} kennt folgende Kategorien nicht: {', '.join(sorted(unsupported))}",
                f"Model {package_id} does not know these categories: {', '.join(sorted(unsupported))}",
            ))
        model_size = str(source_manifest.get("modelSize"))
        model_sizes.add(model_size)
        weight_name = Path(str(source_manifest.get("weights", {}).get("file", ""))).name
        weight_path = source_path.parent / "weights" / weight_name
        if not weight_name or not weight_path.is_file():
            raise SystemExit(localized(args.language, f"Gewichtsdatei für {package_id} fehlt.", f"Weights file for {package_id} is missing."))
        members.append({
            "packageID": package_id,
            "modelSize": model_size,
            "categories": sorted(categories),
            "sourceWeightPath": weight_path,
        })
    if len(model_sizes) > 1:
        raise SystemExit(localized(
            args.language,
            f"Alle Modelle müssen dieselbe Modellgröße verwenden (gefunden: {', '.join(sorted(model_sizes))}).",
            f"All models must use the same model size (found: {', '.join(sorted(model_sizes))}).",
        ))
    combined_model_size = next(iter(model_sizes))

    created = datetime.now(timezone.utc)
    package_name = str(getattr(args, "name", "") or "").strip()
    if len(package_name) > 80:
        raise SystemExit(localized(
            args.language,
            "Der Paketname darf höchstens 80 Zeichen lang sein.",
            "The package name must not exceed 80 characters.",
        ))
    if not package_name:
        package_name = f"{sport.replace('_', ' ').title()} · {combined_model_size.capitalize()} · {localized(args.language, 'Kombiniert', 'Combined')}"
    package_id = f"reco-ensemble-{sport}-{combined_model_size}-{created.strftime('%Y%m%d-%H%M%S')}"
    combined_classes = sorted({category for _, categories in assignments for category in categories})

    library_root = project_root / "models" / "library"
    final = library_root / package_id
    staging = project_root / "models" / f".combine-{uuid.uuid4().hex}"
    try:
        (staging / "weights").mkdir(parents=True, exist_ok=False)
        manifest_members = []
        for member in members:
            destination_name = f"{package_file_slug(member['packageID'])}{member['sourceWeightPath'].suffix}"
            destination = staging / "weights" / destination_name
            shutil.copyfile(member["sourceWeightPath"], destination)
            verify_weights_pickle_safety(destination, args.language, member["packageID"])
            manifest_members.append({
                "packageID": member["packageID"],
                "modelSize": member["modelSize"],
                "categories": member["categories"],
                "weightsFile": f"weights/{destination_name}",
                "sha256": sha256_file(destination),
            })
        manifest = {
            "schemaVersion": 2,
            "packageID": package_id,
            "displayName": package_name,
            "createdAt": created.isoformat(),
            "sport": sport,
            "modelSize": combined_model_size,
            "classes": combined_classes,
            "description": localized(
                args.language,
                f"Kombiniertes Modell aus {len(members)} Modellen, je eigene Kategorien: " + ", ".join(f"{m['packageID']} ({', '.join(m['categories'])})" for m in members),
                f"Combined model from {len(members)} models, each contributing its own categories: " + ", ".join(f"{m['packageID']} ({', '.join(m['categories'])})" for m in members),
            ),
            "ensemble": {"members": manifest_members},
            "privacy": "This package contains model weights and aggregate metadata only. It contains no videos, frames, source paths, or video file names.",
            "license": "Apache-2.0 RF-DETR base; the publisher must confirm rights to redistribute fine-tuned weights.",
        }
        atomic_json(staging / "manifest.json", manifest)
        library_root.mkdir(parents=True, exist_ok=True)
        staging.replace(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    active = activate_library_model(project_root, final / "manifest.json", manifest)
    emit(localized(
        args.language,
        f"Kombiniertes Modell erstellt und aktiviert: {package_name}",
        f"Combined model created and activated: {package_name}",
    ))
    print(json.dumps({"installed": True, "active": active, "manifest": manifest}, ensure_ascii=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Model registry: every model Reco Trainer downloads, with source, license and checksum ---

REGISTRY_PATH = Path(os.environ.get("RECO_MODEL_REGISTRY") or Path(__file__).resolve().with_name("model_registry.json"))


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def registry_model(model_id: str, language: str = "de") -> dict[str, Any]:
    for entry in load_registry()["models"]:
        if entry["id"] == model_id:
            return entry
    raise SystemExit(localized(language, f"Unbekanntes Modell: {model_id}", f"Unknown model: {model_id}"))


def registry_model_path(project_root: Path, model_id: str, language: str = "de") -> Path:
    entry = registry_model(model_id, language)
    if entry.get("installInto"):
        return registry_model_path(project_root, str(entry["installInto"]), language)
    if entry.get("shared"):
        import vlm
        shared = vlm.shared_models_dir()
        return (shared if shared is not None else project_root / ".runtime" / "models") / model_id
    if entry.get("huggingFace"):
        return project_root / ".runtime" / "models" / "hf" / model_id
    if not entry.get("file"):
        raise SystemExit(localized(language, f"{model_id} wird nicht einzeln heruntergeladen.", f"{model_id} is not downloaded on its own."))
    return project_root / ".runtime" / "models" / Path(entry["file"]).name


def download_registry_model(project_root: Path, model_id: str, language: str = "de") -> Path:
    """Download one registry model into .runtime/models/ and check its SHA-256.

    The file is written as <name>.part and only renamed once the checksum matches, so a
    cancel (SIGTERM -> SystemExit) or a broken download never leaves a usable-looking file.
    """
    entry = registry_model(model_id, language)
    target = registry_model_path(project_root, model_id, language)
    if entry.get("huggingFace"):
        return download_hugging_face_model(entry, target, language)
    if entry.get("files"):
        return download_model_files(entry, target, language)
    if entry.get("archive"):
        return download_archive(entry, target, language)
    if target.is_file() and sha256_file(target) == entry["sha256"]:
        return target
    fetch_verified(entry, entry["url"], entry["sha256"], int(entry.get("bytes") or 0), target, language)
    return target


def fetch_verified(entry: dict[str, Any], url: str, sha256: str, size: int, target: Path, language: str, progress_base: int = 0, progress_total: int = 0) -> None:
    """Download url to target via <target>.part and keep it only when the SHA-256 matches."""
    if not str(url).startswith("https://"):
        raise SystemExit(localized(language, f"Für {entry['id']} ist keine sichere Download-Adresse hinterlegt.", f"No secure download address is registered for {entry['id']}."))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "RecoTrainer"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as sink:
            total = progress_total or int(response.headers.get("Content-Length") or size or 0)
            done = progress_base
            while chunk := response.read(1024 * 1024):
                sink.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    print(f"RECO_PROGRESS {min(done, total)} {total}", file=sys.stderr, flush=True)
        if digest.hexdigest() != sha256:
            raise SystemExit(localized(
                language,
                f"Die Prüfsumme von {entry['name']} stimmt nicht. Die Datei wurde verworfen.",
                f"The checksum of {entry['name']} does not match. The file was discarded.",
            ))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def completion_marker(folder: Path, model_id: str) -> Path:
    return folder / f".reco-complete-{model_id}"


def download_model_files(entry: dict[str, Any], target: Path, language: str) -> Path:
    """Several files (a GGUF model and its vision projector) into one folder, each checked."""
    marker = completion_marker(target, entry["id"])
    if marker.is_file():
        return target
    total = sum(int(item["bytes"]) for item in entry["files"])
    done = 0
    for item in entry["files"]:
        destination = target / item["name"]
        if not (destination.is_file() and destination.stat().st_size == int(item["bytes"]) and sha256_file(destination) == item["sha256"]):
            fetch_verified(entry, item["url"], item["sha256"], int(item["bytes"]), destination, language, done, total)
        done += int(item["bytes"])
    marker.write_text(json.dumps({"id": entry["id"], "files": [item["sha256"] for item in entry["files"]]}) + "\n", encoding="utf-8")
    return target


def safe_extract(archive_path: Path, kind: str, destination: Path) -> None:
    """Unpack a runtime archive; entries that would land outside the folder, links and devices are refused."""
    import tarfile
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if kind == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                target = (destination / info.filename).resolve()
                if root != target and root not in target.parents:
                    raise SystemExit(f"unsafe path in archive: {info.filename}")
                if (info.external_attr >> 16) and stat.S_ISLNK(info.external_attr >> 16):
                    raise SystemExit(f"link in archive: {info.filename}")
            archive.extractall(destination)
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        members = []
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise SystemExit(f"unsafe path in archive: {member.name}")
            if member.issym() or member.islnk():
                link = (target.parent / member.linkname).resolve()
                if root not in link.parents:
                    raise SystemExit(f"link leaving the archive: {member.name}")
            elif not (member.isfile() or member.isdir()):
                raise SystemExit(f"unsupported entry in archive: {member.name}")
            members.append(member)
        try:
            archive.extractall(destination, members=members, filter="data")
        except TypeError:  # Python without extraction filters; the checks above already ran
            archive.extractall(destination, members=members)


def download_archive(entry: dict[str, Any], target: Path, language: str) -> Path:
    """A runtime archive (llama-server build or its CUDA runtime), checked, unpacked into target."""
    marker = completion_marker(target, entry["id"])
    if marker.is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / f".{entry['id']}.{entry['archive']}"
    try:
        fetch_verified(entry, entry["url"], entry["sha256"], int(entry.get("bytes") or 0), archive_path, language)
        safe_extract(archive_path, entry["archive"], target)
        if os.name != "nt":
            for path in target.rglob("llama-*"):
                if path.is_file():
                    path.chmod(path.stat().st_mode | 0o111)
        marker.write_text(entry["sha256"] + "\n", encoding="utf-8")
    finally:
        archive_path.unlink(missing_ok=True)
    return target


def download_hugging_face_model(entry: dict[str, Any], target: Path, language: str) -> Path:
    """Download a Hugging Face model pinned to one commit (the registry's revision) into
    .runtime/models/hf/<id>/. huggingface_hub checks every file it fetches; the folder is
    built as <id>.partial and only renamed once complete, so a cancel leaves nothing usable-looking."""
    spec = entry["huggingFace"]
    marker = target / ".reco-complete"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == spec["revision"]:
        return target
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(localized(language, "Für dieses Modell zuerst „ML einrichten“ ausführen.", "Run “Set up ML” first for this model.")) from error
    import threading
    partial = target.with_name(target.name + ".partial")
    finished = threading.Event()
    expected = int(entry.get("bytes") or 0)

    def report_progress() -> None:
        while not finished.wait(1.0):
            if expected and partial.is_dir():
                done = sum(path.stat().st_size for path in partial.rglob("*") if path.is_file())
                print(f"RECO_PROGRESS {min(done, expected)} {expected}", file=sys.stderr, flush=True)

    watcher = threading.Thread(target=report_progress, daemon=True)
    watcher.start()
    try:
        if partial.exists():
            shutil.rmtree(partial)
        snapshot_download(spec["repo"], revision=spec["revision"], local_dir=str(partial), allow_patterns=spec.get("files"))
        (partial / ".reco-complete").write_text(spec["revision"] + "\n", encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        partial.replace(target)
        return target
    finally:
        finished.set()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)


def download_models(args: argparse.Namespace) -> None:
    ids = [item for item in str(args.ids).split(",") if item]
    paths = {}
    for index, model_id in enumerate(ids, start=1):
        entry = registry_model(model_id, args.language)
        stage(index, len(ids), f"{entry['name']} ({entry['license']})")
        paths[model_id] = str(download_registry_model(Path(args.project).resolve(), model_id, args.language))
        emit(localized(args.language, f"{entry['name']} geladen und geprüft.", f"{entry['name']} downloaded and verified."))
    print(json.dumps({"downloaded": paths}, ensure_ascii=False))


def verify_weights_pickle_safety(weights_path: Path, language: str, label: str) -> None:
    """Reject a weights file whose pickle stream references anything beyond
    plain tensors and basic containers - the actual protection against a
    maliciously crafted .pth/.ckpt executing arbitrary code on load, which a
    checksum alone cannot catch (a bad actor controls both the file and its
    own checksum, so a matching checksum only proves the file wasn't corrupted
    in transit, not that it's safe).

    torch.load(..., weights_only=True) enforces PyTorch's own allowlist of
    safe global references during unpickling and raises for anything outside
    it (custom classes with a __reduce__/__setstate__ payload, arbitrary
    callables, etc.) - this is the standard, PyTorch-maintained mitigation for
    untrusted checkpoints, not a bespoke scanner.

    Silently skipped (checksum verification alone still applies) when torch
    is not importable in the current interpreter - this can run from a system
    Python that has no ML packages installed yet (e.g. importing a model
    package before "Set up ML" has ever run). Callers should prefer the
    project's ML venv when one exists so this actually executes; see
    ml_action()'s venv-preferring dispatch for install-package/package/
    combine-models.
    """
    try:
        import torch
    except ImportError:
        emit(localized(
            language,
            f"Hinweis: Code-Sicherheitsprüfung für {label} übersprungen (ML-Umgebung noch nicht eingerichtet). Die Prüfsumme wurde trotzdem verifiziert.",
            f"Note: code-safety check skipped for {label} (ML environment not set up yet). The checksum was still verified.",
            f"Nota: la comprobación de seguridad del código para {label} se omitió (el entorno de ML aún no está configurado). La suma de verificación sí se comprobó.",
            f"Remarque : la vérification de sécurité du code pour {label} a été ignorée (l’environnement ML n’est pas encore configuré). La somme de contrôle a tout de même été vérifiée.",
        ))
        return
    try:
        torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise SystemExit(localized(
            language,
            f"Gewichtsdatei enthält nicht zulässigen Code und wurde abgelehnt: {label}",
            f"Weights file contains disallowed code and was rejected: {label}",
            f"El archivo de pesos contiene código no permitido y fue rechazado: {label}",
            f"Le fichier de poids contient du code non autorisé et a été rejeté : {label}",
        )) from error


def verify_zip_weight_entry(archive: "zipfile.ZipFile", weight_name: str, expected_digest: str, language: str, label: str) -> None:
    """Checksum-verify one weight entry from an opened .recomodel zip AND scan
    its pickle content for anything beyond safe tensor data (see
    verify_weights_pickle_safety), in a single read pass. Used by
    validate_model_package_file() for both plain and ensemble packages.
    """
    checksum = hashlib.sha256()
    scratch_fd, scratch_name = tempfile.mkstemp(prefix="reco-weights-scan-")
    scratch_path = Path(scratch_name)
    try:
        with os.fdopen(scratch_fd, "wb") as scratch, archive.open(weight_name) as weights:
            for chunk in iter(lambda: weights.read(1024 * 1024), b""):
                checksum.update(chunk)
                scratch.write(chunk)
        if checksum.hexdigest() != expected_digest:
            message_label = f": {label}" if label else "."
            raise SystemExit(localized(
                language,
                f"Prüfsumme stimmt nicht{message_label}",
                f"Checksum does not match{message_label}",
                f"La suma de verificación no coincide{message_label}",
                f"La somme de contrôle ne correspond pas{message_label}",
            ))
        if not weight_name.lower().endswith(".onnx"):
            verify_weights_pickle_safety(scratch_path, language, label or weight_name)
    finally:
        scratch_path.unlink(missing_ok=True)


def package_file_slug(name: str) -> str:
    """Create a portable file-name component without exposing local paths."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")
    return slug[:60] or "reco-model"


def package_model(args: argparse.Namespace) -> None:
    """Create a shareable weights package that contains no media or file names."""
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    checkpoint = newest_checkpoint(project_root, args.model)
    if checkpoint is None:
        raise SystemExit(localized(
            args.language,
            "Kein trainiertes Modell gefunden. Zuerst lokal trainieren.",
            "No trained model was found. Train locally first.",
        ))
    verify_weights_pickle_safety(checkpoint, args.language, checkpoint.name)
    reviewed_frames = [frame for frame in document.get("frames", []) if frame.get("reviewStatus") != "candidate"]
    annotations = [annotation for frame in reviewed_frames for annotation in frame.get("annotations", [])]
    created = datetime.now(timezone.utc)
    package_name = str(getattr(args, "name", "") or "").strip()
    if len(package_name) > 80:
        raise SystemExit(localized(
            args.language,
            "Der Paketname darf höchstens 80 Zeichen lang sein.",
            "The package name must not exceed 80 characters.",
            "El nombre del paquete no puede superar los 80 caracteres.",
            "Le nom du paquet ne doit pas dépasser 80 caractères.",
        ))
    if not package_name:
        package_name = f"{str(document['sport']).replace('_', ' ').title()} · {args.model.capitalize()}"
    package_id = f"reco-{document['sport']}-{args.model}-{created.strftime('%Y%m%d-%H%M%S')}"
    output_dir = project_root / "exports" / "share"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{package_file_slug(package_name)}-{created.strftime('%Y%m%d-%H%M%S')}.recomodel"
    try:
        rfdetr_version = importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError:
        rfdetr_version = "unknown"
    package_classes = sorted({annotation.get("category") for annotation in annotations})
    manifest = {
        "schemaVersion": 1,
        "packageID": package_id,
        "displayName": package_name,
        "createdAt": created.isoformat(),
        "sport": document["sport"],
        "modelSize": args.model,
        "classes": package_classes,
        "description": package_description(document["sport"], args.model, package_classes),
        "trainingSummary": document.get("lastTraining"),
        "statistics": {
            "frameCount": len(reviewed_frames),
            "annotatedFrameCount": sum(bool(frame.get("annotations")) for frame in reviewed_frames),
            "annotationCount": len(annotations),
        },
        "framework": {"name": "RF-DETR", "version": rfdetr_version},
        "weights": {"file": f"weights/{checkpoint.name}", "sha256": sha256_file(checkpoint)},
        "privacy": "This package contains model weights and aggregate metadata only. It contains no videos, frames, source paths, or video file names.",
        "license": "Apache-2.0 RF-DETR base; the publisher must confirm rights to redistribute fine-tuned weights.",
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        archive.write(checkpoint, f"weights/{checkpoint.name}")
    emit(localized(
        args.language,
        f"Datenschutzsicheres Austauschpaket erstellt: {output}",
        f"Privacy-safe exchange package created: {output}",
    ))


def install_onnx_model(args: argparse.Namespace) -> None:
    """Wrap an exported YOLO ONNX detector (NMS built in) in a .recomodel package
    for the project's sport and install it like "Import model" does.

    Same manifest as tools/make_onnx_package.py. Class names are given in model
    class-id order; common external names are mapped (person -> player). The
    package is kept in the project's inbox/ so it can be shared later.
    """
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    sport = str(document.get("sport", ""))
    onnx_file = Path(args.file).expanduser().resolve()
    if not onnx_file.is_file() or onnx_file.suffix.lower() != ".onnx":
        raise SystemExit(localized(args.language, "Die ONNX-Datei wurde nicht gefunden.", "The ONNX file was not found.", "No se encontró el archivo ONNX.", "Le fichier ONNX est introuvable."))
    name = package_file_slug(args.name or onnx_file.stem)
    allowed = sport_categories(sport)
    raw_names = [item.strip() for item in str(args.class_names).split(",") if item.strip()]
    class_names = [normalized_label_class(item, allowed) for item in raw_names]
    unsupported = [raw for raw, mapped in zip(raw_names, class_names) if mapped is None]
    if not raw_names or unsupported:
        raise SystemExit(localized(
            args.language,
            f"Klassen in der Reihenfolge der Klassen-IDs angeben, erlaubt: {', '.join(allowed)}" + (f" (nicht erkannt: {', '.join(unsupported)})" if unsupported else ""),
            f"Give the classes in class-id order, allowed: {', '.join(allowed)}" + (f" (not recognised: {', '.join(unsupported)})" if unsupported else ""),
            f"Indica las clases en el orden de sus ID, permitidas: {', '.join(allowed)}" + (f" (no reconocidas: {', '.join(unsupported)})" if unsupported else ""),
            f"Indiquez les classes dans l’ordre des ID, autorisées : {', '.join(allowed)}" + (f" (non reconnues : {', '.join(unsupported)})" if unsupported else ""),
        ))
    if not 64 <= int(args.input_size) <= 4096:
        raise SystemExit(localized(args.language, "Die Eingabegröße muss zwischen 64 und 4096 liegen.", "The input size must be between 64 and 4096.", "El tamaño de entrada debe estar entre 64 y 4096.", "La taille d’entrée doit être comprise entre 64 et 4096."))
    resolved = [str(item) for item in class_names]
    weight_name = f"weights/{name}.onnx"
    manifest = {
        "schemaVersion": 1,
        "packageID": name,
        "displayName": str(args.name).strip()[:120] or name,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sport": sport,
        "modelSize": "small",
        "classes": sorted(set(resolved)),
        "classNames": resolved,
        "format": ONNX_FORMAT,
        "inputSize": int(args.input_size),
        "source": "external",
        "description": (
            f"Imported {sport} object detector (ONNX, input {int(args.input_size)} px) for "
            f"{', '.join(resolved)}. Pre-labeling and benchmarking only; it cannot be fine-tuned."
        )[:500],
        "framework": {"name": "ONNX (YOLO, NMS built in)"},
        "weights": {"file": weight_name, "sha256": sha256_file(onnx_file)},
        "privacy": "This package contains model weights and aggregate metadata only. It contains no videos, frames, source paths, or video file names.",
        "license": str(args.license).strip()[:300] or "Unknown - the publisher must confirm the license of the imported weights.",
    }
    inbox = project_root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    package = inbox / f"{name}.recomodel"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        archive.write(onnx_file, weight_name)
    emit(localized(
        args.language,
        f"ONNX-Modellpaket erstellt: inbox/{package.name} (Klassen: {', '.join(resolved)})",
        f"ONNX model package created: inbox/{package.name} (classes: {', '.join(resolved)})",
    ))
    install_model_package(argparse.Namespace(project=str(project_root), file=str(package), language=args.language))


def verify_model_package(args: argparse.Namespace) -> None:
    package = Path(args.file).expanduser().resolve()
    manifest = validate_model_package_file(package, args.language)
    print(json.dumps({"valid": True, "manifest": manifest}, ensure_ascii=False))


def validate_model_package_file(package: Path, language: str = "de") -> dict[str, Any]:
    if not package.is_file():
        raise SystemExit(localized(language, "Modellpaket wurde nicht gefunden.", "Model package was not found.", "No se encontró el paquete del modelo.", "Le paquet du modèle est introuvable."))
    if package.stat().st_size > 4 * 1024**3:
        raise SystemExit(localized(language, "Modellpaket ist zu groß.", "Model package is too large.", "El paquete del modelo es demasiado grande.", "Le paquet du modèle est trop volumineux."))
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if "manifest.json" not in names:
            raise SystemExit(localized(language, "manifest.json fehlt.", "manifest.json is missing."))
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.file_size > 1024 * 1024:
            raise SystemExit(localized(language, "Manifest ist zu groß.", "Manifest is too large."))
        manifest = json.loads(archive.read("manifest.json"))
        schema_version = manifest.get("schemaVersion")
        if schema_version not in (1, 2):
            raise SystemExit(localized(language, "Unbekannte Paketversion.", "Unsupported package version."))
        description = manifest.get("description")
        if description is not None and (not isinstance(description, str) or not description.strip() or len(description) > 500 or any(ord(character) < 32 and character not in "\t\n\r" for character in description)):
            raise SystemExit(localized(language, "Ungültige Paketbeschreibung.", "Invalid package description."))
        onnx_package = is_onnx_manifest(manifest)
        if onnx_package:
            class_names = manifest.get("classNames")
            input_size = manifest.get("inputSize")
            if (
                schema_version != 1
                or manifest.get("ensemble")
                or not isinstance(class_names, list) or not class_names
                or not all(isinstance(item, str) and item for item in class_names)
                or not set(class_names).issubset({str(item) for item in manifest.get("classes") or []})
                or not isinstance(input_size, int) or isinstance(input_size, bool) or not 64 <= input_size <= 4096
            ):
                raise SystemExit(localized(language, "Ungültige Angaben zum ONNX-Modell im Manifest.", "Invalid ONNX model details in the manifest."))
        weight_suffixes = {".onnx"} if onnx_package else {".pth", ".ckpt"}
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(localized(language, "Unsicherer Pfad im Modellpaket.", "Unsafe path in model package.", "Ruta no segura en el paquete.", "Chemin non sécurisé dans le paquet."))
            if name != "manifest.json" and not (name.startswith("weights/") and path.suffix in weight_suffixes):
                raise SystemExit(localized(language, f"Unzulässige Datei im Modellpaket: {name}", f"Disallowed file in model package: {name}"))
            # Defense in depth: reject anything that isn't a plain regular file
            # entry (e.g. a symlink) even though the extraction path below reads
            # entry bytes directly rather than following filesystem links, so
            # this isn't a live traversal exploit today - it closes the door on
            # that ever becoming one if the extraction method changes later.
            unix_mode = archive.getinfo(name).external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise SystemExit(localized(language, f"Unzulässiger Symlink im Modellpaket: {name}", f"Disallowed symlink in model package: {name}"))

        ensemble = manifest.get("ensemble") if schema_version == 2 else None
        if ensemble:
            # A combined model bundles one weights file per member instead of
            # the usual single manifest + weights pair - verify each member's
            # own checksum instead of one top-level "weights" entry.
            members = ensemble.get("members") or []
            if not members or len(names) != 1 + len(members):
                raise SystemExit(localized(language, "Kombiniertes Modellpaket muss Manifest und genau eine Gewichtsdatei je Mitglied enthalten.", "A combined model package must contain the manifest and exactly one weights file per member."))
            for member in members:
                weight_name = member.get("weightsFile")
                if weight_name not in names:
                    raise SystemExit(localized(language, f"Gewichtsdatei fehlt: {member.get('packageID')}", f"Weights file is missing: {member.get('packageID')}"))
                weight_info = archive.getinfo(weight_name)
                if weight_info.file_size > 4 * 1024**3:
                    raise SystemExit(localized(language, "Gewichtsdatei ist zu groß.", "Weights file is too large."))
                verify_zip_weight_entry(archive, weight_name, str(member.get("sha256")), language, str(member.get("packageID") or weight_name))
        else:
            if len(names) != 2:
                raise SystemExit(localized(language, "Modellpaket muss genau Manifest und Gewichte enthalten.", "The model package must contain exactly a manifest and weights.", "El paquete debe contener exactamente el manifiesto y los pesos.", "Le paquet doit contenir exactement le manifeste et les poids."))
            weight_name = manifest.get("weights", {}).get("file")
            if weight_name not in names:
                raise SystemExit(localized(language, "Gewichtsdatei fehlt.", "Weights file is missing."))
            weight_info = archive.getinfo(weight_name)
            if weight_info.file_size > 4 * 1024**3:
                raise SystemExit(localized(language, "Gewichtsdatei ist zu groß.", "Weights file is too large."))
            verify_zip_weight_entry(archive, weight_name, str(manifest.get("weights", {}).get("sha256")), language, "")
    return manifest


def install_model_package(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    package = Path(args.file).expanduser().resolve()
    manifest = validate_model_package_file(package, args.language)
    package_id = str(manifest.get("packageID", ""))
    if not package_id or len(package_id) > 120 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in package_id):
        raise SystemExit(localized(args.language, "Ungültige Paket-ID.", "Invalid package ID.", "ID de paquete no válido.", "Identifiant de paquet non valide."))
    sport = manifest.get("sport")
    if sport != document.get("sport"):
        raise SystemExit(localized(
            args.language,
            f"Das Modell ist für {sport}, das Projekt aber für {document.get('sport')}.",
            f"The model is for {sport}, but the project is for {document.get('sport')}.",
            f"El modelo es para {sport}, pero el proyecto es para {document.get('sport')}.",
            f"Le modèle est prévu pour {sport}, mais le projet pour {document.get('sport')}.",
        ))
    model_size = manifest.get("modelSize")
    if model_size not in MODEL_CLASSES:
        raise SystemExit(localized(args.language, "Unbekannte Modellgröße.", "Unknown model size."))
    allowed_classes = {
        "football": {"ball", "player", "goalkeeper", "referee", "goal"},
        "futsal": {"ball", "player", "goalkeeper", "referee", "goal"},
        "basketball": {"ball", "player", "referee", "hoop"},
        "handball": {"ball", "player", "goalkeeper", "referee", "goal"},
        "hockey": {"puck", "player", "goalkeeper", "referee", "goal"},
        "rugby": {"ball", "player", "referee", "goalpost"},
        "lacrosse": {"ball", "player", "goalkeeper", "referee", "goal"},
        "american_football": {"ball", "player", "referee", "goalpost"},
    }.get(str(sport), set())
    if not set(manifest.get("classes") or []).issubset(allowed_classes):
        raise SystemExit(localized(args.language, "Das Paket enthält unzulässige Klassen.", "The package contains unsupported classes."))

    ensemble = manifest.get("ensemble")
    weight_entries = (
        [(str(member["weightsFile"]), str(member["sha256"]), str(member.get("packageID"))) for member in ensemble.get("members", [])]
        if ensemble else
        [(str(manifest["weights"]["file"]), str(manifest["weights"]["sha256"]), None)]
    )

    library_root = project_root / "models" / "library"
    final = library_root / package_id
    if final.exists():
        existing_manifest = load_json(final / "manifest.json")
        existing_hashes = (
            {member.get("weightsFile"): member.get("sha256") for member in (existing_manifest.get("ensemble") or {}).get("members", [])}
            if existing_manifest.get("ensemble") else
            {existing_manifest.get("weights", {}).get("file"): existing_manifest.get("weights", {}).get("sha256")}
        )
        if any(existing_hashes.get(weights_file) != digest for weights_file, digest, _ in weight_entries):
            raise SystemExit(localized(args.language, "Paket-ID ist bereits mit anderen Gewichten installiert.", "This package ID is already installed with different weights."))
    else:
        staging = project_root / "models" / f".install-{uuid.uuid4().hex}"
        try:
            (staging / "weights").mkdir(parents=True, exist_ok=False)
            atomic_json(staging / "manifest.json", manifest)
            with zipfile.ZipFile(package) as archive:
                for weights_file, expected_digest, member_id in weight_entries:
                    destination = staging / weights_file
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(weights_file) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=1024 * 1024)
                    if sha256_file(destination) != expected_digest:
                        label = f" ({member_id})" if member_id else ""
                        raise SystemExit(localized(args.language, f"Prüfsumme nach dem Import ungültig{label}.", f"Checksum is invalid after import{label}."))
            library_root.mkdir(parents=True, exist_ok=True)
            staging.replace(final)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    active = activate_library_model(project_root, final / "manifest.json", manifest)
    print(json.dumps({"installed": True, "active": active, "manifest": manifest}, ensure_ascii=False))


def safe_package_id(value: str) -> str:
    if not value or len(value) > 120 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value):
        raise ValueError("Invalid package ID")
    return value


def managed_model(project_root: Path, package_id: str) -> tuple[Path, dict[str, Any]]:
    package_id = safe_package_id(package_id)
    library_root = (project_root / "models" / "library").resolve()
    manifest_path = (library_root / package_id / "manifest.json").resolve()
    if library_root not in manifest_path.parents or not manifest_path.is_file():
        raise ValueError("Model does not exist")
    manifest = load_json(manifest_path)
    if manifest.get("packageID") != package_id:
        raise ValueError("Model manifest does not match its directory")
    return manifest_path, manifest


def list_models(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    require_project(project_root, args.language)
    active_id = None
    active_path = project_root / "models" / "active.json"
    if active_path.is_file():
        try:
            active_id = load_json(active_path).get("packageID")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    entries = library_manifests(project_root)
    scored = [(manifest_comparison_score(manifest), manifest.get("packageID")) for _, manifest in entries]
    valid_scores = [item for item in scored if item[0] is not None]
    best_id = max(valid_scores, default=(None, None), key=lambda item: item[0] if item[0] is not None else -1)[1]
    models = []
    for _, manifest in entries:
        item = dict(manifest)
        item["isActive"] = manifest.get("packageID") == active_id
        item["isBest"] = manifest.get("packageID") == best_id
        item["validationScore"] = manifest_validation_score(manifest)
        item["testScore"] = manifest_test_score(manifest)
        models.append(item)
    models.sort(key=lambda item: (not item["isActive"], not item["isBest"], str(item.get("createdAt", ""))), reverse=False)
    print(json.dumps({"models": models, "activePackageID": active_id, "bestPackageID": best_id}, ensure_ascii=False))


def activate_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    manifest_path, manifest = managed_model(project_root, args.package_id)
    if manifest.get("sport") != document.get("sport"):
        raise SystemExit(localized(args.language, "Das Modell gehört zu einer anderen Sportart.", "The model belongs to a different sport."))
    active = activate_library_model(project_root, manifest_path, manifest)
    print(json.dumps({"active": active}, ensure_ascii=False))


def rename_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    require_project(project_root, args.language)
    name = " ".join(str(args.name).split()).strip()
    if not name or len(name) > 80 or any(ord(character) < 32 for character in name):
        raise SystemExit(localized(args.language, "Der Modellname ist ungültig.", "The model name is invalid."))
    manifest_path, manifest = managed_model(project_root, args.package_id)
    manifest["displayName"] = name
    atomic_json(manifest_path, manifest)
    active_path = project_root / "models" / "active.json"
    if active_path.is_file():
        active = load_json(active_path)
        if active.get("packageID") == args.package_id:
            active["displayName"] = name
            atomic_json(active_path, active)
    print(json.dumps({"renamed": True, "packageID": args.package_id, "displayName": name}, ensure_ascii=False))


def delete_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    require_project(project_root, args.language)
    manifest_path, _ = managed_model(project_root, args.package_id)
    active_path = project_root / "models" / "active.json"
    if active_path.is_file() and load_json(active_path).get("packageID") == args.package_id:
        raise SystemExit(localized(args.language, "Das aktive Modell kann nicht gelöscht werden. Zuerst ein anderes aktivieren.", "The active model cannot be deleted. Activate another model first."))
    shutil.rmtree(manifest_path.parent)
    print(json.dumps({"deleted": True, "packageID": args.package_id}, ensure_ascii=False))


def export_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    checkpoint = newest_checkpoint(project_root, args.model)
    if checkpoint is None:
        raise SystemExit(localized(
            args.language,
            "Kein trainiertes Modell gefunden. Zuerst lokal trainieren.",
            "No trained model was found. Train locally first.",
        ))

    model_class = import_model_class(args.model, args.language)
    verify_weights_pickle_safety(checkpoint, args.language, checkpoint.name)
    model = model_class(pretrain_weights=str(checkpoint), device="cpu")
    export_dir = project_root / "exports" / args.format
    export_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"reco-{document['sport']}-{args.model}"
    emit(localized(
        args.language,
        f"Exportiere {args.format.upper()} auf CPU-Basis …",
        f"Exporting {args.format.upper()} using the CPU path …",
    ))
    export_kwargs = {
        "format": args.format,
        "output_dir": str(export_dir),
    }
    optional_export_kwargs = {
        "notes": {
            "sport": document["sport"],
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "license": "Apache-2.0 base; verify training-data rights before redistribution",
        },
    }
    if args.format == "coreml":
        optional_export_kwargs["coreml_precision"] = "float16"
    supported_optional = explicitly_supported_kwargs(model.export, optional_export_kwargs)
    export_kwargs.update(supported_optional)
    exported = model.export(**export_kwargs)
    if args.format == "coreml" and "coreml_precision" not in supported_optional:
        emit(localized(
            args.language,
            "Diese RF-DETR-Version exportiert Core ML mit ihrer sicheren Standardpräzision.",
            "This RF-DETR version exports Core ML with its safe default precision.",
        ))

    artifacts = [path for path in export_dir.rglob("*") if path.is_file()]
    card = {
        "name": output_name,
        "sport": document["sport"],
        "model_size": args.model,
        "format": args.format,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": checkpoint.name,
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "artifacts": [
            {"file": str(path.relative_to(export_dir)), "sha256": sha256_file(path)}
            for path in artifacts
        ],
        "privacy": "No source videos or extracted frames are included.",
        "redistribution_warning": "Confirm rights to the local training material before sharing weights.",
    }
    atomic_json(export_dir / "model-card.json", card)
    emit(localized(
        args.language,
        f"Export abgeschlossen: {exported or export_dir}",
        f"Export completed: {exported or export_dir}",
    ))


IMPORT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dataset_rglob(source: Path, pattern: str):
    """rglob without hidden entries and the project Reco Trainer keeps inside a working folder."""
    for path in source.rglob(pattern):
        parts = path.relative_to(source).parts
        if not any(part.startswith(".") or part == "Reco Training" for part in parts):
            yield path
LABEL_CLASS_ALIASES = {
    "player": "player", "players": "player", "person": "player", "people": "player",
    "ball": "ball", "soccer ball": "ball", "soccerball": "ball", "sports ball": "ball", "basketball": "ball",
    "referee": "referee", "ref": "referee", "referees": "referee",
    "goalkeeper": "goalkeeper", "keeper": "goalkeeper", "goalie": "goalkeeper",
    "goal": "goal", "goalpost": "goalpost", "hoop": "hoop", "basket": "hoop", "rim": "hoop", "puck": "puck",
}


def normalized_label_class(name: str, allowed: list[str]) -> str | None:
    """Map an external class name onto one of the sport's Reco categories, or None."""
    key = re.sub(r"[\s_\-]+", " ", str(name).strip().lower())
    if key in allowed:
        return key
    mapped = LABEL_CLASS_ALIASES.get(key)
    return mapped if mapped in allowed else None


def image_size(path: Path) -> tuple[int, int] | None:
    """Pixel size of a JPEG/PNG read from its header (no image library needed)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] != b"\xff\xd8":
                return None
            handle.seek(2)
            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                while marker[1] == 0xFF:
                    marker = marker[1:] + handle.read(1)
                code = marker[1]
                if code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:
                    continue
                length = struct.unpack(">H", handle.read(2))[0]
                if code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    _, height, width = struct.unpack(">BHH", handle.read(5))
                    return width, height
                handle.seek(length - 2, 1)
    except (OSError, struct.error):
        return None


def parse_class_map(text: str | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for part in str(text or "").split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            if key.strip().isdigit() and value.strip():
                mapping[int(key.strip())] = value.strip()
    return mapping


def yolo_class_names(source: Path) -> dict[int, str]:
    """Class id -> name from data.yaml (names: list / inline list / id: name block) or classes.txt."""
    for yaml_path in sorted(dataset_rglob(source, "*.yaml"))[:20]:
        try:
            lines = yaml_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for index, line in enumerate(lines):
            if not line.startswith("names:"):
                continue
            rest = line[len("names:"):].strip()
            if rest.startswith("[") and rest.endswith("]"):
                names = [item.strip().strip("'\"") for item in rest[1:-1].split(",") if item.strip()]
                return dict(enumerate(names))
            if rest.startswith("{") and rest.endswith("}"):
                result = {}
                for item in rest[1:-1].split(","):
                    key, _, value = item.partition(":")
                    if key.strip().isdigit() and value.strip():
                        result[int(key.strip())] = value.strip().strip("'\"")
                return result
            result = {}
            list_names: list[str] = []
            for block_line in lines[index + 1:]:
                if not block_line.strip() or block_line.lstrip().startswith("#"):
                    continue
                if not block_line.startswith((" ", "\t", "-")):
                    break
                stripped = block_line.strip()
                if stripped.startswith("- "):
                    list_names.append(stripped[2:].strip().strip("'\""))
                else:
                    key, _, value = stripped.partition(":")
                    if key.strip().isdigit() and value.strip():
                        result[int(key.strip())] = value.strip().strip("'\"")
            return result or dict(enumerate(list_names))
    for name_file in ("classes.txt", "obj.names", "labels.txt"):
        for candidate in sorted(dataset_rglob(source, name_file))[:1]:
            try:
                names = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, UnicodeDecodeError):
                continue
            return dict(enumerate(names))
    return {}


def yolo_label_path(image: Path) -> Path | None:
    candidate = image.with_suffix(".txt")
    if candidate.is_file():
        return candidate
    parts = list(image.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            swapped = Path(*parts[:index], "labels", *parts[index + 1:]).with_suffix(".txt")
            if swapped.is_file():
                return swapped
    return None


def parse_yolo_labels(source: Path, class_names: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Images with YOLO txt labels -> items with pixel boxes; counts what was left out."""
    items: list[dict[str, Any]] = []
    stats = {"imagesWithoutLabels": 0, "invalidLines": 0, "unreadableImages": 0}
    for image in sorted(path for path in dataset_rglob(source, "*") if path.suffix.lower() in IMPORT_IMAGE_SUFFIXES and path.is_file()):
        label_file = yolo_label_path(image)
        if label_file is None:
            stats["imagesWithoutLabels"] += 1
            continue
        size = image_size(image)
        if size is None:
            stats["unreadableImages"] += 1
            continue
        width, height = size
        boxes: list[dict[str, Any]] = []
        for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
            values = line.split()
            if not values:
                continue
            try:
                class_id = int(values[0])
                numbers = [float(value) for value in values[1:]]
            except ValueError:
                stats["invalidLines"] += 1
                continue
            if len(numbers) in (4, 5):
                center_x, center_y, box_w, box_h = numbers[:4]
                x1, y1 = (center_x - box_w / 2) * width, (center_y - box_h / 2) * height
                x2, y2 = (center_x + box_w / 2) * width, (center_y + box_h / 2) * height
                confidence = numbers[4] if len(numbers) == 5 else None
            elif len(numbers) >= 6 and len(numbers) % 2 == 0:
                xs, ys = numbers[0::2], numbers[1::2]
                x1, x2, y1, y2 = min(xs) * width, max(xs) * width, min(ys) * height, max(ys) * height
                confidence = None
            else:
                stats["invalidLines"] += 1
                continue
            boxes.append({"class": class_names.get(class_id, f"class {class_id}"), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence})
        items.append({"image": image, "width": width, "height": height, "boxes": boxes})
    return items, stats


def parse_coco_labels(source: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    stats = {"imagesWithoutLabels": 0, "invalidLines": 0, "unreadableImages": 0}
    for json_path in sorted(dataset_rglob(source, "*.json")):
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not (isinstance(document, dict) and isinstance(document.get("images"), list)
                and isinstance(document.get("annotations"), list) and isinstance(document.get("categories"), list)):
            continue
        names = {category.get("id"): str(category.get("name", "")) for category in document["categories"] if isinstance(category, dict)}
        boxes_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in document["annotations"]:
            bbox = annotation.get("bbox") if isinstance(annotation, dict) else None
            if not (isinstance(bbox, list) and len(bbox) == 4):
                stats["invalidLines"] += 1
                continue
            x, y, box_w, box_h = (float(value) for value in bbox)
            boxes_by_image[annotation.get("image_id")].append({
                "class": names.get(annotation.get("category_id"), f"class {annotation.get('category_id')}"),
                "x1": x, "y1": y, "x2": x + box_w, "y2": y + box_h, "confidence": annotation.get("score"),
            })
        for entry in document["images"]:
            file_name = str(entry.get("file_name", ""))
            image = next((c for c in (json_path.parent / file_name, source / file_name) if c.is_file()), None)
            if image is None:
                matches = [p for p in dataset_rglob(source, Path(file_name).name) if p.is_file()] if file_name else []
                image = matches[0] if matches else None
            if image is None or image.suffix.lower() not in IMPORT_IMAGE_SUFFIXES:
                stats["unreadableImages"] += 1
                continue
            size = image_size(image)
            if size is None:
                stats["unreadableImages"] += 1
                continue
            items.append({"image": image, "width": size[0], "height": size[1], "boxes": boxes_by_image.get(entry.get("id"), [])})
    return items, stats


def import_labels(args: argparse.Namespace) -> None:
    """Import YOLO-txt or COCO labelled images from a folder into the project.

    Images are copied to frames/imported/ and added as frames of one pseudo-video per
    import (split_frames treats these separately from real videos). A frame already in the project with the
    same file name only gets the labels. Re-importing the same image is skipped.
    With include_unlabelled (working folder) images without labels become frames to label.
    """
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    sport = str(document.get("sport", ""))
    allowed = sport_categories(sport)
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(localized(args.language, "Der Quellordner wurde nicht gefunden.", "The source folder was not found."))

    coco_present = any(
        path.suffix.lower() == ".json" and '"annotations"' in path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
        for path in sorted(dataset_rglob(source, "*.json"))[:50]
    )
    chosen = args.format if args.format != "auto" else ("coco" if coco_present else "yolo")
    include_unlabelled = bool(getattr(args, "include_unlabelled", False))
    if chosen == "coco":
        items, stats = parse_coco_labels(source)
    else:
        class_names = {**yolo_class_names(source), **parse_class_map(args.class_map)}
        if not class_names and not include_unlabelled:
            raise SystemExit(localized(
                args.language,
                "Keine Klassennamen gefunden. Eine data.yaml/classes.txt in den Ordner legen oder --class-map \"0=player,1=ball,2=referee\" angeben.",
                "No class names found. Put a data.yaml/classes.txt in the folder or pass --class-map \"0=player,1=ball,2=referee\".",
            ))
        items, stats = parse_yolo_labels(source, class_names)
    if include_unlabelled:
        labelled = {item["image"].resolve() for item in items}
        for image in sorted(path for path in dataset_rglob(source, "*") if path.suffix.lower() in IMPORT_IMAGE_SUFFIXES and path.is_file()):
            if image.resolve() in labelled:
                continue
            size = image_size(image)
            if size is None:
                stats["unreadableImages"] += 1
                continue
            items.append({"image": image, "width": size[0], "height": size[1], "boxes": [], "unlabelled": True})
        stats["imagesWithoutLabels"] = 0
    if not items:
        raise SystemExit(localized(args.language, "Keine Bilder mit Labels gefunden.", "No labelled images were found."))

    as_candidates = bool(args.candidates)
    batch_id = "import-" + hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:8]
    frames = document.setdefault("frames", [])
    by_name = {Path(str(frame.get("relativePath", ""))).name: frame for frame in frames}
    known_ids = {str(frame.get("id")) for frame in frames}
    target_dir = project_root / "frames" / "imported"
    skipped_classes: dict[str, int] = defaultdict(int)
    summary = {"imported": 0, "unlabelled": 0, "matched": 0, "duplicates": 0, "annotations": 0, "format": chosen}
    pending_copies: list[tuple[Path, Path]] = []

    for order, item in enumerate(items):
        if include_unlabelled and order % 50 == 0:
            print(f"RECO_PROGRESS {order} {len(items) * 2}", file=sys.stderr, flush=True)
        unlabelled = bool(item.get("unlabelled"))
        annotations: list[dict[str, Any]] = []
        for box in item["boxes"]:
            category = normalized_label_class(box["class"], allowed)
            if category is None:
                skipped_classes[str(box["class"])] += 1
                continue
            x1, y1 = max(0.0, box["x1"]), max(0.0, box["y1"])
            x2, y2 = min(float(item["width"]), box["x2"]), min(float(item["height"]), box["y2"])
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                stats["invalidLines"] += 1
                continue
            confidence = box.get("confidence")
            annotations.append({
                "id": str(uuid.uuid4()), "category": category,
                "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                "confidence": float(confidence) if confidence is not None else (None if as_candidates else 1.0),
                "source": "auto" if as_candidates else "manual",
            })
        existing = by_name.get(item["image"].name)
        if existing is not None and not is_imported_frame(existing):
            if unlabelled or (existing.get("annotations") and not args.overwrite):
                summary["duplicates"] += 1
                continue
            existing["annotations"] = annotations
            existing["reviewStatus"] = "candidate" if as_candidates else "reviewed"
            summary["matched"] += 1
            summary["annotations"] += len(annotations)
            continue
        digest = sha256_file(item["image"])
        frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"imported:{digest}"))
        if frame_id in known_ids:
            summary["duplicates"] += 1
            continue
        known_ids.add(frame_id)
        safe_name = package_file_slug(item["image"].stem)[:60] + item["image"].suffix.lower()
        destination = target_dir / f"{digest[:10]}-{safe_name}"
        pending_copies.append((item["image"], destination))
        frames.append({
            "id": frame_id,
            "relativePath": f"frames/imported/{destination.name}",
            "videoID": batch_id,
            "videoName": source.name,
            "timestamp": float(order),
            "width": item["width"],
            "height": item["height"],
            "annotations": annotations,
            **({} if unlabelled else {"reviewStatus": "candidate" if as_candidates else "reviewed"}),
        })
        summary["imported"] += 1
        summary["unlabelled"] += int(unlabelled)
        summary["annotations"] += len(annotations)

    if summary["imported"] or summary["matched"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        for copied, (source_file, destination) in enumerate(pending_copies):
            if include_unlabelled and copied % 50 == 0:
                print(f"RECO_PROGRESS {len(items) + copied * len(items) // len(pending_copies)} {len(items) * 2}", file=sys.stderr, flush=True)
            shutil.copy2(source_file, destination)
        backup = project_file(project_root).with_name("project.json.before-import")
        shutil.copy2(project_file(project_root), backup)
        document["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        atomic_json(project_file(project_root), document)
    summary.update({
        "skippedClasses": dict(skipped_classes),
        "imagesWithoutLabels": stats["imagesWithoutLabels"],
        "invalidLines": stats["invalidLines"],
        "unreadableImages": stats["unreadableImages"],
    })
    emit(localized(
        args.language,
        f"Label-Import ({chosen}): {summary['imported']} neue Bilder, {summary['matched']} bestehende Frames beschriftet, "
        f"{summary['annotations']} Boxen, {summary['duplicates']} übersprungen (bereits vorhanden).",
        f"Label import ({chosen}): {summary['imported']} new images, {summary['matched']} existing frames labelled, "
        f"{summary['annotations']} boxes, {summary['duplicates']} skipped (already present).",
    ))
    if skipped_classes:
        emit(localized(
            args.language,
            "Nicht übernommene Klassen: " + ", ".join(f"{name} ({count})" for name, count in sorted(skipped_classes.items())),
            "Classes not imported: " + ", ".join(f"{name} ({count})" for name, count in sorted(skipped_classes.items())),
        ))
    print(json.dumps(summary, ensure_ascii=False))



def augment_scan(args: argparse.Namespace) -> None:
    """What the source folder holds; stops right away when annotations are missing."""
    import augment_studio
    try:
        images, classes = augment_studio.load_source(Path(args.source), args.language)
    except augment_studio.StudioError as error:
        raise SystemExit(str(error))
    print(json.dumps({"images": len(images), "classes": classes, "roles": augment_studio.default_roles(classes)}, ensure_ascii=False))


def augment_config(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads((Path(args.work) / "config.json").read_text(encoding="utf-8"))


def augment_analyze(args: argparse.Namespace) -> None:
    import augment_studio
    project_root = Path(args.project).resolve()

    def make_masks():
        import teachers
        model_id = teachers.tier_plan(hardware_tier())["refiner"] or "sam2.1-hiera-small"
        return augment_studio.Sam2Masks(download_registry_model(project_root, model_id, args.language), detect_device())

    def make_inpaint():
        try:
            folder = download_registry_model(project_root, "lama-inpainting", args.language)
            lama = augment_studio.LamaInpaint(next(Path(folder).rglob("*.onnx")))
        except (ImportError, OSError, StopIteration, SystemExit) as error:
            emit(localized(args.language, f"Hinweis: LaMa nicht verfügbar ({error}), es wird mit OpenCV gefüllt.", f"Note: LaMa is not available ({error}), OpenCV fills the gaps instead."))
            return augment_studio.opencv_inpaint
        return lama

    try:
        analysis = augment_studio.analyze(Path(args.source), Path(args.work), augment_config(args), args.language, make_masks, make_inpaint)
    except augment_studio.StudioError as error:
        raise SystemExit(str(error))
    print(json.dumps({"sessions": len(analysis["sessions"]), "instances": len(analysis["instances"]), "dropped": analysis["dropped"]}, ensure_ascii=False))


def augment_generate(args: argparse.Namespace) -> None:
    import augment_studio
    config = augment_config(args)
    face_model = download_registry_model(Path(args.project).resolve(), "yunet-face", args.language) if config.get("anonymize") else None
    try:
        report = augment_studio.generate(Path(args.work), config, args.language, face_model=face_model)
    except augment_studio.StudioError as error:
        raise SystemExit(str(error))
    print(json.dumps(report, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reco Trainer local ML worker")
    commands = parser.add_subparsers(dest="command", required=True)

    scan_parser = commands.add_parser("augment-scan")
    scan_parser.add_argument("--source", required=True)
    scan_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    scan_parser.set_defaults(func=augment_scan)

    for name, handler in (("augment-analyze", augment_analyze), ("augment-generate", augment_generate)):
        studio_parser = commands.add_parser(name)
        studio_parser.add_argument("--project", required=True)
        studio_parser.add_argument("--source", required=True)
        studio_parser.add_argument("--work", required=True)
        studio_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
        studio_parser.set_defaults(func=handler)

    doctor_parser = commands.add_parser("doctor")
    doctor_parser.set_defaults(func=doctor)

    label_parser = commands.add_parser("autolabel")
    label_parser.add_argument("--project", required=True)
    label_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    label_parser.add_argument("--category", nargs="+", required=True)
    label_parser.add_argument("--threshold", type=float, default=0.25)
    label_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    label_parser.add_argument("--candidate-only", action="store_true")
    label_parser.add_argument("--compare-models", action="store_true")
    label_parser.add_argument("--teacher", choices=["auto", "nano", "small", "medium", "large"], default="auto",
                              help="RF-DETR base model used while no trained model knows a class (auto: by hardware tier)")
    label_parser.add_argument("--tier", choices=["auto", "strong", "medium", "weak"], default="auto")
    label_parser.add_argument("--refine-method", choices=["auto", "sam2", "opencv"], default="auto")
    label_parser.set_defaults(func=auto_label)

    refine_parser = commands.add_parser("refine-boxes")
    refine_parser.add_argument("--method", choices=["auto", "sam2", "opencv"], default="auto")
    refine_parser.add_argument("--tier", choices=["auto", "strong", "medium", "weak"], default="auto")
    refine_parser.add_argument("--project", required=True)
    refine_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    refine_parser.set_defaults(func=refine_boxes)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--project", required=True)
    train_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--category", nargs="+")
    train_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    train_parser.add_argument("--fresh-start", action="store_true")
    train_parser.add_argument("--anonymize-fraction", type=float, default=0.0, help="share of training images whose heads are blurred (0-0.5)")
    train_parser.set_defaults(func=train)

    export_parser = commands.add_parser("export")
    export_parser.add_argument("--project", required=True)
    export_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    export_parser.add_argument("--format", choices=["onnx", "coreml"], required=True)
    export_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    export_parser.set_defaults(func=export_model)

    package_parser = commands.add_parser("package")
    package_parser.add_argument("--project", required=True)
    package_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    package_parser.add_argument("--name", default="")
    package_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    package_parser.set_defaults(func=package_model)

    verify_parser = commands.add_parser("verify-package")
    verify_parser.add_argument("--file", required=True)
    verify_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    verify_parser.set_defaults(func=verify_model_package)

    install_parser = commands.add_parser("install-package")
    install_parser.add_argument("--project", required=True)
    install_parser.add_argument("--file", required=True)
    install_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    install_parser.set_defaults(func=install_model_package)

    install_onnx_parser = commands.add_parser("install-onnx")
    install_onnx_parser.add_argument("--project", required=True)
    install_onnx_parser.add_argument("--file", required=True, help="YOLO ONNX export with NMS built in (output 1 x N x 6)")
    install_onnx_parser.add_argument("--name", default="")
    install_onnx_parser.add_argument("--class-names", required=True, help='Reco class names in class-id order, e.g. "player,ball,referee"')
    install_onnx_parser.add_argument("--input-size", type=int, default=640, help="used only when the model has no fixed square input")
    install_onnx_parser.add_argument("--license", default="")
    install_onnx_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    install_onnx_parser.set_defaults(func=install_onnx_model)

    combine_parser = commands.add_parser("combine-models")
    combine_parser.add_argument("--project", required=True)
    combine_parser.add_argument("--member", action="append", required=True, help="packageID:category1,category2 - repeatable")
    combine_parser.add_argument("--name", default="")
    combine_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    combine_parser.set_defaults(func=combine_models)

    simulate_parser = commands.add_parser("simulate-ball-tracking")
    simulate_parser.add_argument("--project", required=True)
    simulate_parser.add_argument("--frames-dir", required=True)
    simulate_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    simulate_parser.add_argument("--threshold", type=float, default=0.25)
    simulate_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    simulate_parser.set_defaults(func=simulate_ball_tracking)

    list_models_parser = commands.add_parser("list-models")
    list_models_parser.add_argument("--project", required=True)
    list_models_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    list_models_parser.set_defaults(func=list_models)

    activate_model_parser = commands.add_parser("activate-model")
    activate_model_parser.add_argument("--project", required=True)
    activate_model_parser.add_argument("--package-id", required=True)
    activate_model_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    activate_model_parser.set_defaults(func=activate_model)

    rename_model_parser = commands.add_parser("rename-model")
    rename_model_parser.add_argument("--project", required=True)
    rename_model_parser.add_argument("--package-id", required=True)
    rename_model_parser.add_argument("--name", required=True)
    rename_model_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    rename_model_parser.set_defaults(func=rename_model)

    delete_model_parser = commands.add_parser("delete-model")
    delete_model_parser.add_argument("--project", required=True)
    delete_model_parser.add_argument("--package-id", required=True)
    delete_model_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    delete_model_parser.set_defaults(func=delete_model)

    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--project", required=True)
    benchmark_parser.add_argument("--threshold", type=float, default=0.05)
    benchmark_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    benchmark_parser.set_defaults(func=benchmark)

    consistency_parser = commands.add_parser("consistency-check")
    consistency_parser.add_argument("--project", required=True)
    consistency_parser.add_argument("--iou", type=float, default=0.5, help="boxes of both detectors match from this IoU on")
    consistency_parser.add_argument("--threshold", type=float, default=0.35, help="confidence of the second detector")
    consistency_parser.add_argument("--vlm", action="store_true", help="let the local VLM look at flagged images and a sample")
    consistency_parser.add_argument("--vlm-model", default="qwen3.5-4b-vlm", choices=["qwen3.5-4b-vlm", "qwen3.5-9b-vlm"])
    consistency_parser.add_argument("--sample", type=float, default=0.05, help="share of unflagged images the VLM also checks")
    consistency_parser.add_argument("--teacher", choices=["auto", "nano", "small", "medium", "large"], default="auto")
    consistency_parser.add_argument("--tier", choices=["auto", "strong", "medium", "weak"], default="auto")
    consistency_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    consistency_parser.set_defaults(func=consistency_check)

    export_dataset_parser = commands.add_parser("export-dataset")
    export_dataset_parser.add_argument("--project", required=True)
    export_dataset_parser.add_argument("--formats", default="yolo,coco", help="yolo, coco or both (comma-separated)")
    export_dataset_parser.add_argument("--anonymize", action="store_true")
    export_dataset_parser.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    export_dataset_parser.add_argument("--include-text", action="store_true", help="also blur jersey numbers, scoreboard and other text")
    export_dataset_parser.add_argument("--detect-people", action="store_true", help="find unlabelled people (spectators, bench) with RF-DETR")
    export_dataset_parser.add_argument("--detector-size", choices=["auto", "nano", "small", "medium", "large"], default="auto")
    export_dataset_parser.add_argument("--zip", action="store_true")
    export_dataset_parser.add_argument("--vlm-sample", type=float, default=0.0, help="share of anonymized images the local VLM double-checks for recognizable faces")
    export_dataset_parser.add_argument("--vlm-model", default="qwen3.5-4b-vlm", choices=["qwen3.5-4b-vlm", "qwen3.5-9b-vlm"])
    export_dataset_parser.add_argument("--app-version", default="")
    export_dataset_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    export_dataset_parser.set_defaults(func=export_dataset)

    download_parser = commands.add_parser("download-models")
    download_parser.add_argument("--project", required=True)
    download_parser.add_argument("--ids", required=True, help="comma-separated ids from model_registry.json")
    download_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    download_parser.set_defaults(func=download_models)

    import_labels_parser = commands.add_parser("import-labels")
    import_labels_parser.add_argument("--project", required=True)
    import_labels_parser.add_argument("--source", required=True, help="folder with images and YOLO-txt or COCO labels")
    import_labels_parser.add_argument("--format", choices=["auto", "yolo", "coco"], default="auto")
    import_labels_parser.add_argument("--class-map", default="", help='YOLO class ids, e.g. "0=player,1=ball,2=referee"')
    import_labels_parser.add_argument("--candidates", action="store_true", help="import as unreviewed suggestions instead of reviewed labels")
    import_labels_parser.add_argument("--overwrite", action="store_true", help="replace labels of frames that already have some")
    import_labels_parser.add_argument("--include-unlabelled", action="store_true", help="also add images without labels as frames to label (working folder)")
    import_labels_parser.add_argument("--language", default="de", choices=["de", "en", "es", "fr"])
    import_labels_parser.set_defaults(func=import_labels)
    return parser


def stage(index: int, count: int, label: str) -> None:
    """Announce a step of a multi-step command; local_worker.py shows "Step 2/5: label"."""
    print(f"RECO_STAGE {index} {count} {label}", flush=True)


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # A cancel from the app sends SIGTERM to this process group. Turning it into SystemExit
    # lets every finally/except block run, so staging folders and partial files are removed.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
