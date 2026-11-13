"""Optional YOLO training with Ultralytics (AGPL-3.0). Standard library at import time.

RF-DETR (Apache-2.0) stays the default. Ultralytics is installed into the ML environment only
after the user picks YOLO in the training dialog and accepts its license; the app's model
registry (Apache-2.0/MIT only) is not involved. The trained model is exported to ONNX with
NMS built in (YOLO26 is end-to-end, output 1 x 300 x 6) and stored as an "onnx-yolo" library
package, so auto-labeling, the benchmark and the tracking simulation use it like an imported
YOLO model. The PyTorch weights stay in the package so the next YOLO run can continue from them.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

# Pinned: Ultralytics changes quickly; 8.4.180 was released on 2026-10-17.
ULTRALYTICS_VERSION = "8.4.180"
# opencv-python is left out on purpose: the ML environment already has opencv-python-headless,
# and both packages write the same cv2 module (removing one later breaks the other).
INSTALL_ARGS = [
    ["--no-deps", f"ultralytics=={ULTRALYTICS_VERSION}"],
    ["ultralytics-thop>=2.2.4", "polars>=0.20.0"],
]
LICENSE = "AGPL-3.0 (Ultralytics YOLO26)"
LICENSE_URL = "https://github.com/ultralytics/ultralytics/blob/main/LICENSE"
FRAMEWORK = "Ultralytics YOLO26"

# Base weights, COCO-pretrained, pinned by checksum (github.com/ultralytics/assets, release v8.4.0).
VARIANTS: dict[str, dict[str, Any]] = {
    "yolo-nano": {
        "title": "YOLO26 Nano", "file": "yolo26n.pt", "bytes": 5544453,
        "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    },
    "yolo-small": {
        "title": "YOLO26 Small", "file": "yolo26s.pt", "bytes": 20422725,
        "sha256": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
    },
    "yolo-medium": {
        "title": "YOLO26 Medium", "file": "yolo26m.pt", "bytes": 44255705,
        "sha256": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    },
}
for _name, _entry in VARIANTS.items():
    _entry["id"] = _name
    _entry["name"] = _entry["title"]
    _entry["url"] = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{_entry['file']}"

IMAGE_SIZES = (640, 960, 1280)
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def is_variant(name: str | None) -> bool:
    return str(name) in VARIANTS


def installed() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def configure_environment(config_dir: Path) -> None:
    """No telemetry, no automatic pip installs, settings kept next to the app's other data."""
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ.setdefault("YOLO_VERBOSE", "False")


def weights_folder() -> Path:
    """Where Ultralytics keeps downloaded weights (the base models, the check of mixed precision):
    next to the other app data, never in the folder the app happened to be started from."""
    return Path(sys.prefix).parent / "yolo-weights"


def is_miopen_failure(error: BaseException) -> bool:
    """AMD's MIOpen compiles a kernel for BatchNorm at run time; on some cards and Windows set-ups
    that fails ("'type_traits' file not found", miopenStatusUnknownError)."""
    text = str(error).lower()
    return "miopen" in text or "type_traits" in text or "hiprtc" in text


def without_miopen() -> None:
    """Let PyTorch use its own kernels instead of MIOpen (slower, but they need no run-time compile)."""
    import torch

    torch.backends.cudnn.enabled = False


def disable_sync() -> None:
    try:
        from ultralytics import settings

        settings.update({"weights_dir": str(weights_folder()), "sync": False, "hub": False, "tensorboard": False, "wandb": False, "mlflow": False, "clearml": False, "comet": False, "dvc": False, "neptune": False, "raytune": False})
    except Exception:  # noqa: BLE001 - older/newer settings schemas: the env variables above still apply
        pass


def coco_to_yolo(dataset_root: Path, out_root: Path) -> tuple[Path, list[str]]:
    """Write an Ultralytics view (images linked, labels as txt, data.yaml) of a COCO dataset
    with train/valid/test folders. Class ids follow the COCO category ids in ascending order."""
    if out_root.exists():
        shutil.rmtree(out_root)
    names: list[str] | None = None
    present: list[str] = []
    for split, target in SPLITS.items():
        annotations_file = dataset_root / split / "_annotations.coco.json"
        if not annotations_file.is_file():
            continue
        document = json.loads(annotations_file.read_text(encoding="utf-8"))
        categories = sorted(document.get("categories", []), key=lambda item: int(item["id"]))
        split_names = [str(item["name"]) for item in categories]
        if names is None:
            names = split_names
        index = {int(item["id"]): names.index(str(item["name"])) for item in categories if str(item["name"]) in names}
        images = {int(item["id"]): item for item in document.get("images", [])}
        lines: dict[int, list[str]] = {image_id: [] for image_id in images}
        for annotation in document.get("annotations", []):
            image = images.get(int(annotation["image_id"]))
            class_id = index.get(int(annotation["category_id"]))
            if image is None or class_id is None:
                continue
            x, y, width, height = (float(value) for value in annotation["bbox"])
            image_width, image_height = float(image["width"]), float(image["height"])
            if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
                continue
            centre_x = min(max((x + width / 2) / image_width, 0.0), 1.0)
            centre_y = min(max((y + height / 2) / image_height, 0.0), 1.0)
            lines[int(image["id"])].append(
                f"{class_id} {centre_x:.6f} {centre_y:.6f} {min(width / image_width, 1.0):.6f} {min(height / image_height, 1.0):.6f}"
            )
        image_dir = out_root / "images" / target
        label_dir = out_root / "labels" / target
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id, image in images.items():
            source = (dataset_root / split / str(image["file_name"])).resolve()
            if not source.is_file():
                continue
            link = image_dir / source.name
            try:
                link.symlink_to(source)
            except OSError:
                shutil.copy2(source, link)
            (label_dir / f"{source.stem}.txt").write_text("\n".join(lines[image_id]) + ("\n" if lines[image_id] else ""), encoding="utf-8")
        if images:
            present.append(target)
    if not names or "train" not in present or "val" not in present:
        raise ValueError("dataset needs train and valid images")
    yaml_lines = [f"path: {json.dumps(str(out_root.resolve()))}", "train: images/train", "val: images/val"]
    if "test" in present:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines.extend(f"  {number}: {json.dumps(name)}" for number, name in enumerate(names))
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return data_yaml, names


def batch_size(variant: str, device: str, image_size: int) -> int:
    base = {"yolo-nano": 32, "yolo-small": 16, "yolo-medium": 8}[variant]
    if device == "cpu":
        base = max(4, base // 4)
    return max(2, int(base * (640 / image_size) ** 2))


def metrics_from(values: dict[str, Any], prefix: str) -> dict[str, float]:
    """Ultralytics metric names -> the names Reco Trainer compares (val/mAP_50, val/mAP_50_95, ...)."""
    mapping = {
        "metrics/mAP50(B)": "mAP_50", "metrics/mAP50-95(B)": "mAP_50_95",
        "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    }
    result: dict[str, float] = {}
    for key, value in (values or {}).items():
        name = mapping.get(str(key))
        if name is None:
            continue
        try:
            result[f"{prefix}/{name}"] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def eval_folder(run_dir: Path, split: str) -> dict[str, Any]:
    """Ultralytics writes val() output to ./runs/detect/val* unless told otherwise; keep it in the run."""
    return {"project": str(run_dir), "name": f"eval-{split}", "exist_ok": True}


def export_onnx(model: Any, image_size: int) -> Path:
    """ONNX with NMS inside the graph -> output 1 x 300 x 6 (x1, y1, x2, y2, confidence, class id),
    what OnnxYoloModel reads. A low conf keeps weak boxes for the confidence sweep and auto-label."""
    return Path(str(model.export(format="onnx", imgsz=int(image_size), simplify=False, dynamic=False, device="cpu", nms=True, conf=0.01, iou=0.7, max_det=300)))


def train(
    variant: str,
    start_weights: Path,
    data_yaml: Path,
    run_dir: Path,
    device: str,
    epochs: int,
    patience: int,
    image_size: int,
    fine_tune: bool,
    on_epoch: Callable[[int, int, dict[str, float]], None],
    notice: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train, evaluate on the test split and export ONNX. Returns paths and metrics.
    `notice("miopen")` is called when training starts over without MIOpen (see is_miopen_failure)."""
    disable_sync()
    from ultralytics import YOLO

    if os.environ.get("RECO_NO_MIOPEN") == "1":
        without_miopen()

    def epoch_end(trainer: Any) -> None:
        # Also fires once more for the final validation; the caller sees the last epoch twice.
        total = int(trainer.epochs)
        on_epoch(min(int(trainer.epoch) + 1, total), total, metrics_from(getattr(trainer, "metrics", {}) or {}, "val"))

    def fresh_model() -> Any:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        model = YOLO(str(start_weights))
        model.add_callback("on_fit_epoch_end", epoch_end)
        return model

    model = fresh_model()
    options: dict[str, Any] = {
        "data": str(data_yaml), "epochs": int(epochs), "patience": int(patience), "imgsz": int(image_size),
        "batch": batch_size(variant, device, image_size), "device": device, "project": str(run_dir.parent),
        "name": run_dir.name, "exist_ok": True, "seed": 42, "deterministic": False, "plots": False,
        "verbose": False, "workers": min(8, os.cpu_count() or 4), "amp": device == "cuda",
    }
    if fine_tune:
        # Continuing from an own model: a smaller learning rate keeps what it learned.
        options.update({"lr0": 0.002, "warmup_epochs": 0.5})
    try:
        model.train(**options)
    except RuntimeError as error:
        # BatchNorm of the AMD card (MIOpen) failed: start over once with PyTorch's own kernels.
        if device != "cuda" or not is_miopen_failure(error):
            raise
        without_miopen()
        if notice:
            notice("miopen")
        model = fresh_model()
        model.train(**options)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        best = run_dir / "weights" / "last.pt"
    trained = YOLO(str(best))
    validation = trained.val(data=str(data_yaml), split="val", imgsz=int(image_size), device=device, plots=False, verbose=False, **eval_folder(run_dir, "val"))
    validation_metrics = metrics_from(getattr(validation, "results_dict", {}) or {}, "val")
    test_metrics: dict[str, float] = {}
    if "test: images/test" in data_yaml.read_text(encoding="utf-8"):
        test = trained.val(data=str(data_yaml), split="test", imgsz=int(image_size), device=device, plots=False, verbose=False, **eval_folder(run_dir, "test"))
        test_metrics = metrics_from(getattr(test, "results_dict", {}) or {}, "test")
    exported = export_onnx(trained, image_size)
    epochs_completed = None
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        rows = [line for line in results_csv.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
        epochs_completed = len(rows)
    return {
        "best": best, "onnx": exported, "validationMetrics": validation_metrics, "testMetrics": test_metrics,
        "epochsCompleted": epochs_completed, "imageSize": int(image_size),
    }
