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
import shutil
import sys
import time
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        return getattr(rfdetr, MODEL_CLASSES[size])
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
    """Choose one fast, stable training job instead of competing GPU jobs."""
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
    if torch_installed:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
    device = "mps" if mps_available else "cpu"
    profile = training_profile("nano", device)
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torchInstalled": torch_installed,
        "rfdetrInstalled": rfdetr_installed,
        "mpsAvailable": mps_available,
        "recommendedDevice": device,
        "cpuCores": profile["cpu_count"],
        "memoryGB": profile["memory_gb"],
        "trainingBatchSize": profile["batch_size"],
        "dataWorkers": profile["num_workers"],
    }
    print(json.dumps(payload, ensure_ascii=False))


def newest_checkpoint(project_root: Path, size: str) -> Path | None:
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
    run_dir = project_root / "runs" / size
    if not run_dir.exists():
        return None
    preferred = [
        run_dir / "checkpoint_best_total.pth",
        run_dir / "checkpoint_best_ema.pth",
        run_dir / "checkpoint.pth",
        run_dir / "last.ckpt",
    ]
    for checkpoint in preferred:
        if checkpoint.is_file():
            return checkpoint
    candidates = [*run_dir.rglob("*.pth"), *run_dir.rglob("*.ckpt")]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def model_instance(project_root: Path, size: str, trained: bool, language: str = "de"):
    model_class = import_model_class(size, language)
    checkpoint = newest_checkpoint(project_root, size) if trained else None
    kwargs: dict[str, Any] = {"device": detect_device()}
    if checkpoint:
        kwargs["pretrain_weights"] = str(checkpoint)
        emit(localized(language, f"Lade trainiertes Modell: {checkpoint.name}", f"Loading trained model: {checkpoint.name}"))
    else:
        emit(localized(
            language,
            f"Lade Apache-2.0-Basismodell RF-DETR {size} …",
            f"Loading Apache-2.0 RF-DETR {size} base model …",
        ))
    return model_class(**kwargs)


def detection_category_map(target_category: str) -> dict[str, str]:
    mapping = {target_category: target_category}
    if target_category in {"ball", "puck"}:
        mapping["sports ball"] = target_category
    elif target_category == "player":
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
    candidates: list[tuple[Path, dict[str, Any], Path]] = []
    for manifest_path in sorted(library_root.glob("*/manifest.json")) if library_root.is_dir() else []:
        try:
            manifest = load_json(manifest_path)
            weight_path = (manifest_path.parent / "weights" / Path(str(manifest.get("weights", {}).get("file", ""))).name).resolve()
            if manifest.get("sport") == document.get("sport") and manifest.get("modelSize") in MODEL_CLASSES and library_root in weight_path.parents and weight_path.is_file():
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
    for model_index, (_, manifest, weight_path) in enumerate(candidates, start=1):
        package_id = str(manifest.get("packageID") or weight_path.parent.parent.name)
        emit(localized(args.language, f"Modelltest {model_index}/{len(candidates)}: {package_id}", f"Model test {model_index}/{len(candidates)}: {package_id}"))
        started = time.perf_counter()
        predictions: list[dict[str, Any]] = []
        try:
            model_class = import_model_class(str(manifest["modelSize"]), args.language)
            model = model_class(pretrain_weights=str(weight_path), device=device)
            package_classes = {str(item) for item in manifest.get("classes", [])}
            profile = training_profile(str(manifest["modelSize"]), device)
            batch_size = profile["batch_size"] if device != "cpu" else 1
            inference_seconds = 0.0
            for start in range(0, len(frames), batch_size):
                batch = frames[start : start + batch_size]
                paths: list[str] = []
                for frame in batch:
                    candidate_path = (project_root / str(frame.get("relativePath", ""))).resolve()
                    frames_root = (project_root / "frames").resolve()
                    if frames_root not in candidate_path.parents or not candidate_path.is_file():
                        raise RuntimeError(f"Benchmark frame is missing: {frame.get('id')}")
                    paths.append(str(candidate_path))
                inference_started = time.perf_counter()
                detected_batch = normalize_detection_batch(
                    model.predict(paths, threshold=threshold), len(batch)
                )
                inference_seconds += time.perf_counter() - inference_started
                for frame, detections in zip(batch, detected_batch):
                    detection_data = (getattr(detections, "data", {}) or {})
                    names = detection_data.get("class_name")
                    for index, box in enumerate(detections.xyxy):
                        class_id = int(detections.class_id[index]) if getattr(detections, "class_id", None) is not None else None
                        detected_name = str(names[index]) if names is not None else ""
                        category = benchmark_detection_category(detected_name, class_id, str(document["sport"]), package_classes)
                        if not category:
                            continue
                        predictions.append({"frameID": str(frame.get("id")), "category": category, "confidence": float(detections.confidence[index]), "box": [float(value) for value in box]})
                emit(localized(args.language, f"{package_id}: {min(start + batch_size, len(frames))}/{len(frames)}", f"{package_id}: {min(start + batch_size, len(frames))}/{len(frames)}"))
            metrics = evaluate_predictions(frames, predictions, classes)
            result = {"packageID": package_id, "modelSize": manifest.get("modelSize"), "classes": sorted(package_classes), "status": "completed", "metrics": metrics, "predictionCount": len(predictions), "totalSeconds": time.perf_counter() - started, "meanLatencyMs": 1000.0 * inference_seconds / max(len(frames), 1)}
            atomic_json(run_dir / f"{package_id}.predictions.json", {"schemaVersion": 1, "packageID": package_id, "datasetID": ground_truth.get("datasetID"), "threshold": threshold, "predictions": predictions})
        except Exception as error:
            result = {"packageID": package_id, "modelSize": manifest.get("modelSize"), "classes": sorted(str(item) for item in manifest.get("classes", [])), "status": "failed", "error": str(error), "totalSeconds": time.perf_counter() - started}
        results.append(result)
        try:
            del model
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


def frame_has_category(frame: dict[str, Any], target_category: str) -> bool:
    return any(
        annotation.get("category") == target_category
        for annotation in frame.get("annotations", [])
    )


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Intersection over union for x/y/width/height boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def plausible_refined_box(
    original: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
    category: str,
) -> bool:
    """Reject aggressive OpenCV changes before they can become suggestions."""
    ox, oy, ow, oh = original
    cx, cy, cw, ch = candidate
    if min(ow, oh, cw, ch) < 3.0:
        return False
    area_ratio = (cw * ch) / max(ow * oh, 1.0)
    if not 0.20 <= area_ratio <= 2.20 or box_iou(original, candidate) < 0.15:
        return False
    original_center = (ox + ow / 2.0, oy + oh / 2.0)
    candidate_center = (cx + cw / 2.0, cy + ch / 2.0)
    center_shift = math.hypot(candidate_center[0] - original_center[0], candidate_center[1] - original_center[1])
    if center_shift > math.hypot(ow, oh) * 0.48:
        return False
    aspect = cw / max(ch, 1.0)
    return (0.14 <= aspect <= 4.8) if category == "puck" else (0.32 <= aspect <= 3.1)


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
        score = 0.40 * center_score + 0.30 * box_iou(original, candidate) + 0.15 * fill + 0.15 * area_score
        if score >= 0.46 and (best is None or score > best[1]):
            best = (candidate, score)
    return best


def refinable_annotations(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in frame.get("annotations", [])
        if item.get("source") == "auto"
        and item.get("category") in {"ball", "puck"}
        and not item.get("opencvRefinement")
    ]


def refine_boxes(args: argparse.Namespace) -> None:
    """Refine only unreviewed automatic ball/puck boxes; manual labels are immutable."""
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
    checked = refined = unchanged = unreadable = 0
    for frame in document.get("frames", []):
        eligible = refinable_annotations(frame)
        if not eligible:
            continue
        image = cv2.imread(str(project_root / frame["relativePath"]), cv2.IMREAD_COLOR)
        if image is None:
            unreadable += 1
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
    emit(localized(
        args.language,
        f"OpenCV-Prüfung abgeschlossen: {refined} von {checked} automatischen Ball-/Puck-Boxen plausibel verfeinert; {unchanged} sicherheitshalber unverändert, {unreadable} Bilder nicht lesbar. Manuelle Boxen wurden nicht verändert.",
        f"OpenCV review completed: plausibly refined {refined} of {checked} automatic ball/puck boxes; kept {unchanged} unchanged for safety, {unreadable} images unreadable. Manual boxes were not changed.",
        f"Revisión OpenCV finalizada: {refined} de {checked} cuadros automáticos de balón/disco se refinaron; {unchanged} quedaron sin cambios por seguridad y {unreadable} imágenes no se pudieron leer. Los cuadros manuales no cambiaron.",
        f"Vérification OpenCV terminée : {refined} boîtes automatiques ballon/palet affinées sur {checked} ; {unchanged} conservées par sécurité et {unreadable} images illisibles. Les boîtes manuelles n’ont pas été modifiées.",
    ))


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

    target_category = args.category
    category_map = detection_category_map(target_category)

    if newest_checkpoint(project_root, args.model) is None and target_category not in {"ball", "puck", "player"}:
        emit(localized(
            args.language,
            f"Das allgemeine Basismodell kennt die Klasse „{target_category}“ nicht. "
            "Diese Klasse zunächst manuell markieren und ein eigenes Modell trainieren.",
            f"The generic base model does not know the “{target_category}” class. "
            "Annotate this class manually first and train a custom model.",
        ))
        return

    if target_category == "player" and newest_checkpoint(project_root, args.model) is None:
        emit(localized(
            args.language,
            "Hinweis: Das Basismodell liefert nur die Klasse „Person“. Zuschauer und Schiedsrichter "
            "können deshalb fälschlich als Spieler erscheinen.",
            "Note: The base model only provides the “person” class. Spectators and referees may "
            "therefore be incorrectly labeled as players.",
        ))

    model = model_instance(project_root, args.model, trained=True, language=args.language)
    processed_frames = 0
    frames_with_detections = 0
    boxes_added = 0
    skipped_existing = sum(
        1
        for frame in frames
        if frame_has_category(frame, target_category)
    )
    device = detect_device()
    profile = training_profile(args.model, device)
    batch_size = profile["batch_size"] if device != "cpu" else 1
    emit(localized(
        args.language,
        f"Vorbeschriftung nutzt {device.upper()} mit Batch {batch_size}.",
        f"Pre-labeling uses {device.upper()} with batch {batch_size}.",
    ))

    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        pending = [
            frame
            for frame in batch_frames
            if not frame_has_category(frame, target_category)
        ]
        if not pending:
            continue
        paths = [str(project_root / frame["relativePath"]) for frame in pending]
        detections_batch = normalize_detection_batch(
            model.predict(paths, threshold=args.threshold), len(pending)
        )

        for frame, detections in zip(pending, detections_batch):
            new_annotations = []
            names = detections.data.get("class_name") if hasattr(detections, "data") else None
            for index, box in enumerate(detections.xyxy):
                if names is not None:
                    detected_name = str(names[index])
                else:
                    class_id = int(detections.class_id[index])
                    detected_name = "person" if class_id == 0 else "sports ball" if class_id == 32 else ""
                category = category_map.get(detected_name)
                if not category:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
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
            args.language,
            f"Automatisch geprüft: {min(start + batch_size, len(frames))}/{len(frames)}",
            f"Automatically checked: {min(start + batch_size, len(frames))}/{len(frames)}",
        ))

    document["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    atomic_json(project_file(project_root), document)
    emit(localized(
        args.language,
        f"Vorbeschriftung für „{target_category}“ abgeschlossen: {boxes_added} Boxen in "
        f"{frames_with_detections} von {processed_frames} geprüften Frames. "
        f"{skipped_existing} Frames hatten bereits diese Klasse und wurden nicht überschrieben.",
        f"Pre-labeling for “{target_category}” finished: {boxes_added} boxes in "
        f"{frames_with_detections} of {processed_frames} checked frames. "
        f"{skipped_existing} frames already contained this class and were not overwritten.",
    ))
    if boxes_added == 0:
        emit(localized(
            args.language,
            f"Keine Treffer für „{target_category}“ oberhalb der Schwelle. Einige Beispiele manuell "
            "markieren und danach trainieren; anschließend erneut automatisch markieren.",
            f"No “{target_category}” detections above the threshold. Annotate a few examples manually, "
            "train the model, and then run automatic detection again.",
        ))
    elif target_category in {"ball", "puck"}:
        try:
            refine_boxes(argparse.Namespace(project=str(project_root), language=args.language))
        except SystemExit as error:
            emit(localized(
                args.language,
                f"OpenCV-Nachprüfung übersprungen: {error}",
                f"Skipped OpenCV post-review: {error}",
                f"Se omitió la revisión posterior con OpenCV: {error}",
                f"Post-vérification OpenCV ignorée : {error}",
            ))


def stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def split_frames(
    frames: list[dict[str, Any]], language: str = "de"
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_video[frame["videoID"]].append(frame)
    video_ids = sorted(by_video, key=stable_bucket)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    warning = None

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
        ordered = sorted(frames, key=lambda frame: frame["timestamp"])
        train_end = max(1, int(len(ordered) * 0.7))
        valid_end = max(train_end + 1, int(len(ordered) * 0.9))
        output["train"] = ordered[:train_end]
        output["valid"] = ordered[train_end:valid_end]
        output["test"] = ordered[valid_end:]
        warning = localized(
            language,
            "Nur ein Video: temporaler Split als Notlösung; Qualitätswerte sind nicht unabhängig.",
            "Only one video: a temporal fallback split is used; quality metrics are not independent.",
        )
    return output, warning


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_dataset(project_root: Path, language: str = "de") -> Path:
    document = require_project(project_root, language)
    frames = [
        frame for frame in document.get("frames", [])
        if frame.get("reviewStatus") != "candidate"
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

    ordered_categories = [
        name for name in sport_categories(document["sport"]) if name in category_names
    ]
    category_id = {name: index + 1 for index, name in enumerate(ordered_categories)}
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
        emit(localized(
            language,
            f"{split_name}: {len(images)} Bilder, {len(annotations)} Markierungen",
            f"{split_name}: {len(images)} images, {len(annotations)} annotations",
        ))
    return dataset_root


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
    weight_name = Path(str(manifest.get("weights", {}).get("file", ""))).name
    weight_path = manifest_path.parent / "weights" / weight_name
    if not weight_name or not weight_path.is_file():
        raise ValueError("Model weights are missing")
    active = {
        "packageID": str(manifest["packageID"]),
        "displayName": manifest.get("displayName") or manifest["packageID"],
        "modelSize": manifest.get("modelSize"),
        "sport": manifest.get("sport"),
        "description": manifest.get("description"),
        "weights": str(weight_path.relative_to(project_root)),
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


def preferred_run_checkpoint(run_dir: Path) -> Path | None:
    for name in ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint.pth", "last.ckpt"]:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    candidates = [*run_dir.rglob("*.pth"), *run_dir.rglob("*.ckpt")] if run_dir.is_dir() else []
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def train(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    document = require_project(project_root, args.language)
    dataset_root = build_dataset(project_root, args.language)
    model_class = import_model_class(args.model, args.language)
    device = detect_device()
    profile = training_profile(args.model, device)
    output_dir = project_root / "runs" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    continuation_checkpoint = newest_checkpoint(project_root, args.model)

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
    if continuation_checkpoint is not None:
        model_kwargs["pretrain_weights"] = str(continuation_checkpoint)
        emit(localized(
            args.language,
            f"Setze das zuletzt trainierte Modell fort: {continuation_checkpoint.name}",
            f"Continuing from the latest trained model: {continuation_checkpoint.name}",
        ))
    else:
        emit(localized(
            args.language,
            "Noch kein kompatibler Checkpoint vorhanden; Training startet mit dem Basismodell.",
            "No compatible checkpoint exists yet; training starts from the base model.",
        ))
    model = model_class(**model_kwargs)
    train_kwargs = {
        "dataset_dir": str(dataset_root),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": profile["batch_size"],
        "grad_accum_steps": profile["grad_accum_steps"],
        "device": device,
        "num_workers": profile["num_workers"],
        "persistent_workers": profile["persistent_workers"],
        "prefetch_factor": profile["prefetch_factor"],
        "pin_memory": profile["pin_memory"],
        "early_stopping": True,
        "early_stopping_patience": max(5, min(12, args.epochs // 3)),
        "eval_ema_only": True,
        "run_test": bool(list((dataset_root / "test").glob("*.jpg"))),
        "notes": {
            "sport": document["sport"],
            "privacy": "trained locally; source media not uploaded",
            "reco_trainer_schema": document.get("schemaVersion", 1),
            "performance_profile": profile,
        },
    }
    _, ignored = call_with_supported_kwargs(model.train, train_kwargs)
    if ignored:
        emit(localized(
            args.language,
            f"Die installierte RF-DETR-Version kennt diese optionalen Optimierungen noch nicht: "
            f"{', '.join(ignored)}.",
            f"The installed RF-DETR version does not yet support these optional optimizations: "
            f"{', '.join(ignored)}.",
        ))
    validation_metrics: dict[str, float] = {}
    if hasattr(model, "evaluate"):
        try:
            emit(localized(args.language, "Bewerte das Modell auf dem Validierungssatz …", "Evaluating the model on the validation split …"))
            evaluation = model.evaluate(dataset_dir=str(dataset_root), split="val")
            validation_metrics = numeric_metrics(evaluation)
        except Exception as error:
            emit(localized(
                args.language,
                f"Hinweis: Die Qualitätsbewertung war mit dieser RF-DETR-Version nicht verfügbar: {error}",
                f"Note: quality evaluation was unavailable with this RF-DETR version: {error}",
            ))

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
        "independentTest": len({frame.get("videoID") for frame in refreshed_frames}) >= 3,
        "checkpoint": trained_checkpoint.name if trained_checkpoint else None,
        "continuedFrom": continuation_checkpoint.name if continuation_checkpoint else None,
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
        promote = current_manifest is None or (
            new_score is not None and (current_score is None or new_score > current_score)
        )
        if promote:
            activate_library_model(project_root, manifest_path, manifest)
            emit(localized(
                args.language,
                f"Neuer bester Modellstand aktiviert: {manifest['displayName']}",
                f"New best model revision activated: {manifest['displayName']}",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    reviewed_frames = [frame for frame in document.get("frames", []) if frame.get("reviewStatus") != "candidate"]
    annotations = [annotation for frame in reviewed_frames for annotation in frame.get("annotations", [])]
    created = datetime.now(timezone.utc)
    package_id = f"reco-{document['sport']}-{args.model}-{created.strftime('%Y%m%d-%H%M%S')}"
    output_dir = project_root / "exports" / "share"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{package_id}.recomodel"
    try:
        rfdetr_version = importlib.metadata.version("rfdetr")
    except importlib.metadata.PackageNotFoundError:
        rfdetr_version = "unknown"
    package_classes = sorted({annotation.get("category") for annotation in annotations})
    manifest = {
        "schemaVersion": 1,
        "packageID": package_id,
        "displayName": f"{str(document['sport']).replace('_', ' ').title()} · {args.model.capitalize()} · {created.strftime('%Y-%m-%d %H:%M UTC')}",
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
        if len(names) != 2:
            raise SystemExit(localized(language, "Modellpaket muss genau Manifest und Gewichte enthalten.", "The model package must contain exactly a manifest and weights.", "El paquete debe contener exactamente el manifiesto y los pesos.", "Le paquet doit contenir exactement le manifeste et les poids."))
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(localized(language, "Unsicherer Pfad im Modellpaket.", "Unsafe path in model package.", "Ruta no segura en el paquete.", "Chemin non sécurisé dans le paquet."))
            if name != "manifest.json" and not (name.startswith("weights/") and path.suffix in {".pth", ".ckpt"}):
                raise SystemExit(localized(language, f"Unzulässige Datei im Modellpaket: {name}", f"Disallowed file in model package: {name}"))
        if "manifest.json" not in names:
            raise SystemExit(localized(language, "manifest.json fehlt.", "manifest.json is missing."))
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.file_size > 1024 * 1024:
            raise SystemExit(localized(language, "Manifest ist zu groß.", "Manifest is too large."))
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schemaVersion") != 1:
            raise SystemExit(localized(language, "Unbekannte Paketversion.", "Unsupported package version."))
        description = manifest.get("description")
        if description is not None and (not isinstance(description, str) or not description.strip() or len(description) > 500 or any(ord(character) < 32 and character not in "\t\n\r" for character in description)):
            raise SystemExit(localized(language, "Ungültige Paketbeschreibung.", "Invalid package description."))
        weight_name = manifest.get("weights", {}).get("file")
        if weight_name not in names:
            raise SystemExit(localized(language, "Gewichtsdatei fehlt.", "Weights file is missing."))
        weight_info = archive.getinfo(weight_name)
        if weight_info.file_size > 4 * 1024**3:
            raise SystemExit(localized(language, "Gewichtsdatei ist zu groß.", "Weights file is too large."))
        checksum = hashlib.sha256()
        with archive.open(weight_name) as weights:
            for chunk in iter(lambda: weights.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if digest != manifest.get("weights", {}).get("sha256"):
            raise SystemExit(localized(language, "Prüfsumme stimmt nicht.", "Checksum does not match.", "La suma de verificación no coincide.", "La somme de contrôle ne correspond pas."))
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

    library_root = project_root / "models" / "library"
    final = library_root / package_id
    weights_name = Path(str(manifest["weights"]["file"])).name
    if final.exists():
        existing_manifest = load_json(final / "manifest.json")
        if existing_manifest.get("weights", {}).get("sha256") != manifest["weights"]["sha256"]:
            raise SystemExit(localized(args.language, "Paket-ID ist bereits mit anderen Gewichten installiert.", "This package ID is already installed with different weights."))
    else:
        staging = project_root / "models" / f".install-{uuid.uuid4().hex}"
        try:
            (staging / "weights").mkdir(parents=True, exist_ok=False)
            atomic_json(staging / "manifest.json", manifest)
            with zipfile.ZipFile(package) as archive, archive.open(manifest["weights"]["file"]) as source, (staging / "weights" / weights_name).open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if sha256_file(staging / "weights" / weights_name) != manifest["weights"]["sha256"]:
                raise SystemExit(localized(args.language, "Prüfsumme nach dem Import ungültig.", "Checksum is invalid after import."))
            library_root.mkdir(parents=True, exist_ok=True)
            staging.replace(final)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    active = {
        "packageID": package_id,
        "displayName": manifest.get("displayName") or package_id,
        "modelSize": model_size,
        "sport": sport,
        "description": manifest.get("description"),
        "weights": str((final / "weights" / weights_name).relative_to(project_root)),
        "activatedAt": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(project_root / "models" / "active.json", active)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reco Trainer local ML worker")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor_parser = commands.add_parser("doctor")
    doctor_parser.set_defaults(func=doctor)

    label_parser = commands.add_parser("autolabel")
    label_parser.add_argument("--project", required=True)
    label_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    label_parser.add_argument("--category", required=True)
    label_parser.add_argument("--threshold", type=float, default=0.25)
    label_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    label_parser.add_argument("--candidate-only", action="store_true")
    label_parser.set_defaults(func=auto_label)

    refine_parser = commands.add_parser("refine-boxes")
    refine_parser.add_argument("--project", required=True)
    refine_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
    refine_parser.set_defaults(func=refine_boxes)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--project", required=True)
    train_parser.add_argument("--model", choices=MODEL_CLASSES, default="nano")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--language", choices=["de", "en", "es", "fr"], default="de")
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
    return parser


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
