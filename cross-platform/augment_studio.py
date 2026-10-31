"""Augmentation Studio: turns labelled full-court images into a large synthetic training set.

Two jobs, both resumable because every intermediate result is cached under the work folder:

  * analyze()  - source -> sessions, instance masks, empty-hall plates, court geometry, teams
  * generate() - analysis -> output/images, annotations.coco.json, manifest.jsonl, report.json

Every image is first *planned* (which instance goes where; JSON only) and then *rendered*, each
from its own random stream derived from (seed, index, attempt). Results therefore do not depend
on the number of processes or on a resume, and the manifest alone reproduces an image.

Labels are always computed from the visible mask after compositing, never from a moved box.
Needs numpy and OpenCV; torch only for SAM 2 and onnxruntime only for LaMa, imported on demand.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROLES = ("player", "referee", "ball", "static", "ignore")
MIN_MASK_SCORE = 0.6
MIN_MASK_FILL = 0.25
MIN_FRAMES_FOR_MEDIAN = 15
MAX_MEDIAN_FRAMES = 61
PERSON_HEIGHT_M = 1.9
MIN_FOOT_DISTANCE_M = 0.5
MAX_TILT_CHANGE_DEG = 10.0
MAX_BALL_TO_PERSON = 0.5
MIN_COMPLETE_FILL = 0.35
MIN_TEAM_SHARE = 0.2
KEEPER_NAMES = {"goalkeeper", "goalie", "keeper", "torwart", "torhüter"}
PLATE_BOX_MARGIN = 0.12
PLATE_BOX_BELOW = 0.3
MASK_PROMPT_BELOW = 0.2
BYTES_PER_IMAGE = 400_000
MIN_SCALE, MAX_SCALE = 0.7, 1.35
MIN_JERSEY_DELTA_E = 25.0
MAX_NEGATIVE_SHARE = 0.05
RANDOM_CROP_SHARE = 0.10
MAX_ATTEMPTS = 25
MIN_LABEL_PIXELS = 1.5
SPLIT_FOLDERS = {"train", "valid", "val", "test", "images", "labels", "coco", "yolo", "annotations"}

DEFAULT_CONFIG: dict[str, Any] = {
    "variations": 0, "cropShare": 0.6, "zoom": [0.2, 0.7], "outputSize": 560, "seed": 1,
    "roles": {}, "limits": {"players": 5, "referees": 3, "balls": 1},
    "jerseyVariation": False, "yolo": False, "anonymize": False,
    "minVisible": 0.30, "cropMinVisible": 0.40,
    "sessionOverrides": {}, "sessionLinks": [], "teamOverrides": {}, "polygons": {}, "splitOverrides": {},
}


class StudioError(RuntimeError):
    """A problem the user can solve; the message is shown as it is."""


def text(language: str, german: str, english: str, spanish: str | None = None, french: str | None = None) -> str:
    return {"de": german, "es": spanish or english, "fr": french or english}.get(language, english)


def progress(done: int, total: int) -> None:
    print(f"RECO_PROGRESS {done} {total}", file=sys.stderr, flush=True)


def stage(index: int, count: int, label: str) -> None:
    print(f"RECO_STAGE {index} {count} {label}", flush=True)


def require_space(folder: Path, needed: int, language: str) -> None:
    free = shutil.disk_usage(folder).free
    if free < needed:
        gigabytes = f"{needed / 1e9:.1f}"
        raise StudioError(text(
            language,
            f"Zu wenig Speicherplatz: Es werden etwa {gigabytes} GB gebraucht, frei sind {free / 1e9:.1f} GB.",
            f"Not enough disk space: about {gigabytes} GB are needed, {free / 1e9:.1f} GB are free.",
            f"Espacio insuficiente: se necesitan unos {gigabytes} GB, hay {free / 1e9:.1f} GB libres.",
            f"Espace disque insuffisant : environ {gigabytes} Go sont nécessaires, {free / 1e9:.1f} Go sont libres.",
        ))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8"))


def stable_hash(*values: Any) -> str:
    return hashlib.sha256("\x00".join(str(value) for value in values).encode("utf-8")).hexdigest()


# --- COCO run-length encoding ---------------------------------------------------------------

def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Compressed COCO RLE (the string form pycocotools reads and writes)."""
    height, width = mask.shape
    flat = mask.astype(bool).ravel(order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    edges = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(edges).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    out: list[str] = []
    for index, value in enumerate(counts):
        number = int(value) - (int(counts[index - 2]) if index > 2 else 0)
        more = True
        while more:
            chunk = number & 0x1F
            number >>= 5
            more = number != -1 if chunk & 0x10 else number != 0
            if more:
                chunk |= 0x20
            out.append(chr(chunk + 48))
    return {"size": [int(height), int(width)], "counts": "".join(out)}


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    height, width = rle["size"]
    raw = rle["counts"]
    counts: list[int] = []
    if isinstance(raw, list):
        counts = [int(value) for value in raw]
    else:
        position = 0
        data = raw.encode("ascii") if isinstance(raw, str) else bytes(raw)
        while position < len(data):
            number, shift, more = 0, 0, True
            while more:
                chunk = data[position] - 48
                number |= (chunk & 0x1F) << shift
                more = bool(chunk & 0x20)
                position += 1
                shift += 5
                if not more and chunk & 0x10:
                    number |= -1 << shift
            if len(counts) > 2:
                number += counts[-2]
            counts.append(number)
    flat = np.zeros(height * width, dtype=bool)
    position, value = 0, False
    for count in counts:
        if value:
            flat[position:position + count] = True
        position += count
        value = not value
    return flat.reshape((height, width), order="F")


def segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray | None:
    """A mask from COCO polygons or RLE; None when the annotation has no usable one."""
    if isinstance(segmentation, dict) and "counts" in segmentation:
        return rle_to_mask(segmentation)
    if isinstance(segmentation, list) and segmentation:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
        return canvas.astype(bool) if canvas.any() else None
    return None


# --- source ---------------------------------------------------------------------------------

def visible_files(source: Path, pattern: str) -> Iterable[Path]:
    for path in source.rglob(pattern):
        parts = path.relative_to(source).parts
        if not any(part.startswith(".") or part == "Reco Training" for part in parts) and path.is_file():
            yield path


def load_source(source: Path, language: str = "de") -> tuple[list[dict[str, Any]], list[str]]:
    """Images and annotations of every COCO file below source, whatever its layout."""
    images: dict[str, dict[str, Any]] = {}
    classes: list[str] = []
    by_name: dict[str, Path] | None = None
    for json_path in sorted(visible_files(source, "*.json")):
        try:
            document = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not (isinstance(document, dict) and isinstance(document.get("images"), list)
                and isinstance(document.get("annotations"), list) and isinstance(document.get("categories"), list)):
            continue
        names = {category.get("id"): str(category.get("name", "")) for category in document["categories"] if isinstance(category, dict)}
        for name in names.values():
            if name and name not in classes:
                classes.append(name)
        boxes: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in document["annotations"]:
            if isinstance(annotation, dict) and isinstance(annotation.get("bbox"), list) and len(annotation["bbox"]) == 4 and annotation.get("category_id") in names:
                boxes[annotation.get("image_id")].append(annotation)
        for entry in document["images"]:
            file_name = str(entry.get("file_name", ""))
            candidates = [json_path.parent / file_name, source / file_name, json_path.parent.parent / "images" / json_path.parent.name / file_name]
            path = next((item for item in candidates if item.is_file()), None)
            if path is None:
                if by_name is None:
                    by_name = {}
                    for item in sorted(visible_files(source, "*")):
                        if item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            by_name.setdefault(item.name, item)
                path = by_name.get(Path(file_name).name)
            if path is None:
                continue
            relative = path.relative_to(source).as_posix()
            record = images.setdefault(relative, {"rel": relative, "width": int(entry.get("width") or 0), "height": int(entry.get("height") or 0), "instances": []})
            for number, annotation in enumerate(boxes.get(entry.get("id"), []), start=len(record["instances"])):
                x, y, w, h = (float(value) for value in annotation["bbox"])
                if w < 1 or h < 1:
                    continue
                record["instances"].append({
                    "id": stable_hash(relative, number, x, y, w, h)[:16], "class": names[annotation["category_id"]],
                    "bbox": [x, y, w, h], "segmentation": annotation.get("segmentation"),
                })
    labelled = [images[key] for key in sorted(images) if images[key]["instances"]]
    if not labelled:
        raise StudioError(text(
            language,
            "Im Quellordner wurden keine Bilder mit COCO-Annotationen gefunden. Lege Bilder und eine COCO-Datei hinein, zum Beispiel aus „Datensatz exportieren“.",
            "No images with COCO annotations were found in the source folder. Add images and a COCO file, for example from “Export dataset”.",
            "No se encontraron imágenes con anotaciones COCO en la carpeta de origen. Añade imágenes y un archivo COCO, por ejemplo de «Exportar conjunto de datos».",
            "Aucune image avec annotations COCO n’a été trouvée dans le dossier source. Ajoutez des images et un fichier COCO, par exemple via « Exporter le jeu de données ».",
        ))
    return labelled, classes


# --- sessions -------------------------------------------------------------------------------

def session_hint(relative: str) -> str:
    """Subfolder (without dataset split folders) plus the file name without its running number."""
    parts = Path(relative).parts
    folders = [part for part in parts[:-1] if part.lower() not in SPLIT_FOLDERS]
    stem = re.sub(r"[-_ .]*\d+$", "", Path(relative).stem)
    return "/".join([*folders, stem])


def background_signature(image: np.ndarray, boxes: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Small grey picture plus which of its pixels are not covered by a labelled object."""
    height, width = image.shape[:2]
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    free = np.ones((36, 64), dtype=bool)
    for x, y, w, h in boxes:
        free[max(0, int(y / height * 36)):min(36, int(math.ceil((y + h) / height * 36))), max(0, int(x / width * 64)):min(64, int(math.ceil((x + w) / width * 64)))] = False
    return small, free


def group_sessions(images: list[dict[str, Any]], signatures: dict[str, tuple[np.ndarray, np.ndarray]], overrides: dict[str, str], threshold: float = 14.0) -> dict[str, list[str]]:
    """Same subfolder/prefix hint, same size and a similar background form a session."""
    leaders: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
    groups: dict[str, list[str]] = defaultdict(list)
    for image in images:
        relative = image["rel"]
        if relative in overrides:
            groups[overrides[relative]].append(relative)
            continue
        key = f"{session_hint(relative)}|{image['width']}x{image['height']}"
        signature = signatures[relative]
        chosen = None
        for leader_id, leader in leaders:
            if not leader_id.startswith(key + "#"):
                continue
            both = signature[1] & leader[1]
            if both.sum() >= 0.2 * both.size and float(np.mean(np.abs(signature[0][both] - leader[0][both]))) < threshold:
                chosen = leader_id
                break
        if chosen is None:
            chosen = f"{key}#{sum(1 for leader_id, _ in leaders if leader_id.startswith(key + '#')) + 1}"
            leaders.append((chosen, signature))
        groups[chosen].append(relative)
    return dict(sorted(groups.items()))


def splits_from_folders(sessions: dict[str, list[str]]) -> dict[str, str]:
    """A dataset that is already split keeps its split: its test images must not feed synthetic ones."""
    names = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    result = {}
    for name, items in sessions.items():
        found = {names.get(item.split("/")[0]) for item in items}
        if len(found) == 1 and None not in found:
            result[name] = found.pop()
    held_out = {"valid", "test"} & set(result.values())
    return result if len(result) == len(sessions) and held_out and "train" in result.values() else {}


def choose_split(sessions: dict[str, list[str]], overrides: dict[str, str]) -> tuple[dict[str, str], str | None]:
    """Whole sessions go to train, valid or test, so nothing leaks into the evaluation.
    Held-out sessions never feed synthetic images."""
    ordered = sorted(sessions, key=lambda name: stable_hash("split", name))
    total = sum(len(items) for items in sessions.values())
    result: dict[str, str] = {}
    warning = None
    if len(ordered) >= 3:
        # The largest session always trains; only what remains is held out.
        ordered.sort(key=lambda name: name == max(sessions, key=lambda item: (len(sessions[item]), item)))
        test_goal, valid_goal = 0.1 * total, 0.2 * total
        test_count = valid_count = 0
        for name in ordered:
            size = len(sessions[name])
            if (test_count == 0 or test_count + size <= test_goal) and len(result) < len(ordered) - 1:
                result[name] = "test"
                test_count += size
            elif (valid_count == 0 or valid_count + size <= valid_goal) and len(result) < len(ordered) - 1:
                result[name] = "valid"
                valid_count += size
            else:
                result[name] = "train"
        if "train" not in result.values():
            result[ordered[-1]] = "train"
    elif len(ordered) == 2:
        result = {ordered[0]: "train", ordered[1]: "valid"}
        warning = "two-sessions"
    else:
        result = {name: "train" for name in ordered}
        warning = "one-session"
    result.update({name: value for name, value in overrides.items() if name in sessions and value in {"train", "valid", "test"}})
    return result, warning


# --- masks ----------------------------------------------------------------------------------

class Sam2Masks:
    """SAM 2 with a box prompt on a crop around the box, because a small player in a panorama
    would otherwise be only a few pixels at SAM's input size."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from PIL import Image
        from transformers import Sam2Model, Sam2Processor
        self.torch, self.pil, self.device = torch, Image, device
        self.model = Sam2Model.from_pretrained(str(model_dir)).to(device).eval()
        self.processor = Sam2Processor.from_pretrained(str(model_dir))

    def __call__(self, image_bgr: np.ndarray, box: list[float]) -> tuple[np.ndarray, tuple[int, int], float]:
        height, width = image_bgr.shape[:2]
        x, y, w, h = box
        margin = max(w, h, 64.0) * 0.75
        x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
        x1, y1 = min(width, int(math.ceil(x + w + margin))), min(height, int(math.ceil(y + h + margin)))
        crop = self.pil.fromarray(cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=crop, input_boxes=[[[x - x0, y - y0, x + w - x0, y + h - y0]]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        mask = masks[0][0].numpy() if masks[0].ndim == 3 else masks[0].numpy()
        return mask.astype(bool), (x0, y0), float(output.iou_scores.flatten()[0].cpu())


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """The largest connected part: SAM sometimes adds specks of the neighbouring player."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def mask_quality_problem(mask: np.ndarray, offset: tuple[int, int], box: list[float], score: float) -> str | None:
    x, y, w, h = box
    area = int(mask.sum())
    if score < MIN_MASK_SCORE:
        return f"score {score:.2f}"
    if area == 0:
        return "empty"
    x0, y0 = int(max(0, x - offset[0])), int(max(0, y - offset[1]))
    inside = int(mask[y0:int(y0 + h + 1), x0:int(x0 + w + 1)].sum())
    if inside / max(w * h, 1.0) < MIN_MASK_FILL:
        return f"fills {inside / max(w * h, 1.0):.0%} of the box"
    if area - inside > 0.5 * area:
        return "mostly outside the box"
    return None


# --- inpainting -----------------------------------------------------------------------------

def opencv_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.inpaint(image, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


class LamaInpaint:
    """LaMa (Apache-2.0, ONNX, fixed 512 px input), one region at a time."""

    def __init__(self, model_file: Path) -> None:
        import onnxruntime
        self.session = onnxruntime.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)), connectivity=8)
        height, width = image.shape[:2]
        for label in range(1, count):
            x, y, w, h = stats[label][:4]
            side = min(max(int(max(w, h) * 1.6), 256), min(height, width))
            x0 = int(min(max(0, x + w / 2 - side / 2), width - side))
            y0 = int(min(max(0, y + h / 2 - side / 2), height - side))
            part = mask[y0:y0 + side, x0:x0 + side] & (labels[y0:y0 + side, x0:x0 + side] == label)
            crop = cv2.cvtColor(result[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB)
            tensor = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            holes = (cv2.resize(part.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
            names = [item.name for item in self.session.get_inputs()]
            output = self.session.run(None, {names[0]: tensor, names[1]: holes})[0][0].transpose(1, 2, 0)
            output = np.clip(output if output.max() > 2 else output * 255.0, 0, 255).astype(np.uint8)
            restored = cv2.cvtColor(cv2.resize(output, (side, side), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)
            region = result[y0:y0 + side, x0:x0 + side]
            region[part] = restored[part]
        return result


# --- plates ---------------------------------------------------------------------------------

def median_plate(frames: Iterable[tuple[np.ndarray, np.ndarray]], shape: tuple[int, int], count: int, scratch: Path, on_row: Callable[[float], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pixel median over the frames, ignoring labelled people and balls. Frames wait on disk so
    a panorama session does not have to fit into memory. Returns the plate and its holes."""
    height, width = shape
    scratch.mkdir(parents=True, exist_ok=True)
    pixels = np.memmap(scratch / "pixels.bin", dtype=np.uint8, mode="w+", shape=(count, height, width, 3))
    covered = np.memmap(scratch / "covered.bin", dtype=bool, mode="w+", shape=(count, height, width))
    used = 0
    for image, mask in frames:
        pixels[used] = image
        covered[used] = mask
        used += 1
        if on_row:
            on_row(used / count * 0.5)
    plate = np.zeros((height, width, 3), dtype=np.uint8)
    holes = np.zeros((height, width), dtype=bool)
    step = max(16, min(height, 128))
    for top in range(0, height, step):
        bottom = min(height, top + step)
        values = np.array(pixels[:used, top:bottom], dtype=np.uint16)
        hidden = np.array(covered[:used, top:bottom])
        valid = used - hidden.sum(axis=0)
        values[hidden] = 65535  # sorts behind every real value
        values.sort(axis=0)
        middle = np.clip((valid - 1) // 2, 0, used - 1)
        chosen = np.take_along_axis(values, middle[None, :, :, None].repeat(3, axis=3), axis=0)[0]
        plate[top:bottom] = np.where(chosen == 65535, 0, chosen).astype(np.uint8)
        holes[top:bottom] = valid < max(2, min(3, used))
        if on_row:
            on_row(0.5 + bottom / height * 0.5)
    del pixels, covered
    shutil.rmtree(scratch, ignore_errors=True)
    return plate, holes


# --- court geometry -------------------------------------------------------------------------

def poly_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones_like(x)]
    if degree >= 1:
        columns += [x, y]
    if degree >= 2:
        columns += [x * x, x * y, y * y]
    return np.stack(columns, axis=-1)


def fit_field(points: np.ndarray, values: np.ndarray, width: int, height: int) -> dict[str, Any]:
    """Robust 2D polynomial over normalized image positions; a degree must stay plausible across the image.

    A few samples can be fitted exactly and then explode between them, so a degree needs clearly more
    samples than terms, and a fit that leaves the range of the samples falls back to the next lower degree.
    """
    samples = len(values)
    if samples:
        low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
    else:
        low, high = 0.0, 0.0
    margin = max(0.25 * (high - low), 0.15 * abs(high), 1.0)
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 9))
    x, y = (points[:, 0] / width, points[:, 1] / height) if samples else (np.zeros(0), np.zeros(0))
    for degree in (2, 1):
        terms = poly_terms(x, y, degree)
        if samples < 4 * terms.shape[1]:
            continue
        keep = np.ones(samples, dtype=bool)
        coefficients = None
        for _ in range(4):
            if keep.sum() < terms.shape[1]:
                break
            coefficients = np.linalg.lstsq(terms[keep], values[keep], rcond=None)[0]
            residual = values - terms @ coefficients
            spread = 1.4826 * float(np.median(np.abs(residual[keep] - np.median(residual[keep])))) + 1e-6
            keep = np.abs(residual) <= 2.5 * spread
        if coefficients is None:
            continue
        # Only the area the samples cover has to be plausible: nobody is placed far outside it.
        covered = (grid_x >= x.min()) & (grid_x <= x.max()) & (grid_y >= y.min()) & (grid_y <= y.max())
        surface = poly_terms(grid_x[covered], grid_y[covered], degree) @ coefficients if covered.any() else np.zeros(0)
        if surface.size and (surface.min() < low - margin or surface.max() > high + margin):
            continue
        return {"degree": degree, "coefficients": [float(item) for item in coefficients], "width": width, "height": height, "samples": samples, "low": low - margin, "high": high + margin}
    constant = float(np.median(values)) if samples else 0.0
    return {"degree": 0, "coefficients": [constant], "width": width, "height": height, "samples": samples, "low": constant, "high": constant}


def fit_size_field(points: np.ndarray, heights: np.ndarray, width: int, height: int) -> dict[str, Any]:
    """Person height over the image. Few people still give the perspective: height grows towards the camera."""
    general = fit_field(points, heights, width, height)
    if general["degree"] > 0 or len(heights) < 3:
        return general
    y = points[:, 1] / height
    pairs = [(heights[j] - heights[i]) / (y[j] - y[i]) for i in range(len(y)) for j in range(i + 1, len(y)) if abs(y[j] - y[i]) > 0.02]
    if not pairs or float(y.max() - y.min()) < 0.1:
        return general
    slope = float(np.median(pairs))
    offset = float(np.median(heights - slope * y))
    if slope <= 0.0 or offset + slope * float(y.min()) <= 0.0:
        return general
    margin = max(0.25 * float(heights.max() - heights.min()), 0.15 * float(heights.max()), 1.0)
    return {**general, "degree": 1, "coefficients": [offset, 0.0, slope], "low": float(heights.min()) - margin, "high": float(heights.max()) + margin}


def field_at(model: dict[str, Any], x: float, y: float) -> float:
    terms = poly_terms(np.array([x / model["width"]]), np.array([y / model["height"]]), model["degree"])
    value = float((terms @ np.array(model["coefficients"]))[0])
    # Outside the sampled area a polynomial keeps running; clip it to what was actually measured.
    return float(np.clip(value, model["low"], model["high"])) if "low" in model else value


def image_to_court(homography: list[list[float]], x: float, y: float) -> tuple[float, float]:
    point = np.array(homography, dtype=np.float64) @ np.array([x, y, 1.0])
    return float(point[0] / point[2]), float(point[1] / point[2])


class Geometry:
    """Height of a person and "up" at an image position, and distances on the court."""

    def __init__(self, session: dict[str, Any]) -> None:
        self.size = session["sizeMap"]
        self.tilt = session["tiltMap"]
        self.homography = session.get("homography")
        self.polygon = np.array(session["polygon"], dtype=np.float32)
        # A moving camera without calibration knows the floor only near where people actually stood.
        self.local = not self.homography and session.get("polygonSource") == "estimated" and len(session["images"]) < MIN_FRAMES_FOR_MEDIAN

    def height_at(self, x: float, y: float) -> float:
        if self.homography:
            # Pixels per metre along the image horizontal: both it and a person's height scale with 1/depth.
            ax, ay = image_to_court(self.homography, x, y)
            bx, by = image_to_court(self.homography, x + 1.0, y)
            return PERSON_HEIGHT_M / max(math.hypot(bx - ax, by - ay), 1e-6)
        return max(field_at(self.size, x, y), 1.0)

    def tilt_at(self, x: float, y: float) -> float:
        return float(np.clip(field_at(self.tilt, x, y), -60.0, 60.0))

    def distance_m(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        if self.homography:
            ax, ay = image_to_court(self.homography, *a)
            bx, by = image_to_court(self.homography, *b)
            return math.hypot(ax - bx, ay - by)
        per_metre = max(self.height_at((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), 1.0) / PERSON_HEIGHT_M
        # Without calibration: depth is foreshortened on the image, roughly by a broadcast camera's angle.
        return math.hypot((a[0] - b[0]) / per_metre, (a[1] - b[1]) / (per_metre * 0.35))

    def on_field(self, x: float, y: float) -> bool:
        return point_in_polygon(self.polygon, x, y)


def point_in_polygon(polygon: Any, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), (float(x), float(y)), False) >= 0


def mask_tilt(mask: np.ndarray) -> float:
    """Lean of a person in degrees: direction from the centre of the lower to the upper third."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 6:
        return 0.0
    top, bottom = rows[0], rows[-1] + 1
    third = max(1, (bottom - top) // 3)
    upper = np.nonzero(mask[top:top + third])
    lower = np.nonzero(mask[bottom - third:bottom])
    if not len(upper[0]) or not len(lower[0]):
        return 0.0
    dx = float(upper[1].mean() - lower[1].mean())
    dy = float((bottom - third + lower[0].mean()) - (top + upper[0].mean()))
    return math.degrees(math.atan2(dx, max(dy, 1.0)))


def polygon_of_feet(feet: np.ndarray, width: int, height: int, margin: float = 0.0) -> list[list[float]]:
    if len(feet) < 3:
        return [[0.0, height * 0.3], [float(width), height * 0.3], [float(width), float(height)], [0.0, float(height)]]
    # Sideways only: above the highest foot there may already be the wall.
    wider = np.vstack((feet + [margin, 0.0], feet - [margin, 0.0]))
    wider[:, 0] = np.clip(wider[:, 0], 0.0, float(width))
    hull = cv2.convexHull(wider.astype(np.float32)).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in hull]


# --- teams ----------------------------------------------------------------------------------

def jersey_feature(patch: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """HSV histogram of the middle third of the mask: 16 hue bins plus dark, grey and white,
    so white and black jerseys are told apart as well."""
    rows = np.flatnonzero(alpha.any(axis=1))
    top, bottom = rows[0], rows[-1] + 1
    third = max(1, (bottom - top) // 3)
    torso = np.zeros_like(alpha)
    torso[top + third:bottom - third] = alpha[top + third:bottom - third]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[torso]
    histogram = np.zeros(19, dtype=np.float64)
    if not len(hsv):
        return histogram
    hue, saturation, value = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    colourful = (saturation > 50) & (value > 40)
    np.add.at(histogram, hue[colourful] * 16 // 180, 1.0)
    histogram[16] = float(((~colourful) & (value <= 70)).sum())
    histogram[17] = float(((~colourful) & (value > 70) & (value <= 170)).sum())
    histogram[18] = float(((~colourful) & (value > 170)).sum())
    return histogram / max(histogram.sum(), 1.0)


def cluster_teams(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k=2 plus an outlier class (-1). Returns team per instance, uncertain flags and centroids.

    A goalkeeper's jersey is the farthest from everything else and would become a "team" of its own,
    with both real teams merged into the other. So a tiny cluster counts as outliers and the rest is clustered again."""
    count = len(features)
    if count == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool), np.zeros((2, 19))
    active = np.arange(count)
    for _ in range(4):
        teams, uncertain, centroids = cluster_two(features[active])
        sizes = [(teams == team).sum() for team in (0, 1)]
        small = int(np.argmin(sizes))
        if len(active) < 6 or sizes[small] >= MIN_TEAM_SHARE * len(active):
            break
        active = active[teams != small]
    result, flags = np.full(count, -1, dtype=int), np.ones(count, dtype=bool)
    result[active], flags[active] = teams, uncertain
    return result, flags, centroids


def cluster_two(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain k=2. Deterministic: starts from the two instances farthest apart."""
    count = len(features)
    first = int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1)))
    second = int(np.argmax(np.linalg.norm(features - features[first], axis=1)))
    centroids = np.stack([features[first], features[second]])
    labels = np.zeros(count, dtype=int)
    for round_number in range(25):
        distances = np.linalg.norm(features[:, None] - centroids[None], axis=2)
        updated = distances.argmin(axis=1)
        if round_number and (updated == labels).all():
            break
        labels = updated
        for team in (0, 1):
            if (labels == team).any():
                centroids[team] = features[labels == team].mean(axis=0)
    distances = np.linalg.norm(features[:, None] - centroids[None], axis=2)
    nearest = distances.min(axis=1)
    margin = np.abs(distances[:, 0] - distances[:, 1]) / np.maximum(distances.sum(axis=1), 1e-9)
    limit = float(np.median(nearest) + 3.0 * 1.4826 * np.median(np.abs(nearest - np.median(nearest))) + 0.05)
    teams = labels.copy()
    teams[nearest > limit] = -1
    return teams, (margin < 0.2) | (teams == -1), centroids


# --- analysis -------------------------------------------------------------------------------

def merged_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_CONFIG, **{key: value for key, value in config.items() if value is not None}}
    merged["limits"] = {**DEFAULT_CONFIG["limits"], **(config.get("limits") or {})}
    return merged


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise StudioError(f"Unreadable image: {path.name}")
    return image


def default_roles(classes: list[str]) -> dict[str, str]:
    roles = {}
    for name in classes:
        lowered = name.lower()
        roles[name] = ("referee" if "ref" in lowered or "schieds" in lowered else
                       "static" if lowered in {"hoop", "goal", "goalpost", "basket", "korb", "tor", "rim", "net"} else
                       "player" if "player" in lowered or "spieler" in lowered or "keeper" in lowered else
                       "ball" if "ball" in lowered or lowered == "puck" else "player")
    return roles


def load_masks_index(path: Path) -> dict[str, dict[str, Any]]:
    index = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                index[entry["id"]] = entry
            except (ValueError, KeyError):
                continue  # a line cut off by a cancel: that mask is simply made again
    return index


def analyze(
    source: Path, work: Path, config: dict[str, Any], language: str = "de",
    make_masks: Callable[[], Callable[[np.ndarray, list[float]], tuple[np.ndarray, tuple[int, int], float]]] | None = None,
    make_inpaint: Callable[[], Callable[[np.ndarray, np.ndarray], np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Masks, sessions, plates, geometry and teams. Everything finished stays cached."""
    config = merged_config(config)
    cache = work / "cache"
    for folder in ("patches", "thumbs", "plates"):
        (cache / folder).mkdir(parents=True, exist_ok=True)
    stage(1, 4, text(language, "Bilder und Sessions lesen", "Read images and sessions", "Leer imágenes y sesiones", "Lire images et sessions"))
    images, classes = load_source(source, language)
    roles = {**default_roles(classes), **{key: value for key, value in (config.get("roles") or {}).items() if value in ROLES}}
    log: list[str] = []
    signatures = {}
    for index, image in enumerate(images):
        picture = read_image(source / image["rel"])
        image["height"], image["width"] = picture.shape[:2]
        signatures[image["rel"]] = background_signature(picture, [item["bbox"] for item in image["instances"] if roles.get(item["class"]) != "static"])
        progress(index + 1, len(images))
    sessions = group_sessions(images, signatures, config.get("sessionOverrides") or {})
    split, split_warning = choose_split(sessions, {**splits_from_folders(sessions), **(config.get("splitOverrides") or {})})
    by_rel = {image["rel"]: image for image in images}
    session_of = {relative: name for name, items in sessions.items() for relative in items}

    stage(2, 4, text(language, "Masken erzeugen", "Create masks", "Crear máscaras", "Créer les masques"))
    index_path = cache / "masks.jsonl"
    known = load_masks_index(index_path)
    provider = None
    instances: list[dict[str, Any]] = []
    total = sum(len(image["instances"]) for image in images)
    done = 0
    unmasked: dict[str, list[list[float]]] = defaultdict(list)
    with index_path.open("a", encoding="utf-8") as index_file:
        for image in images:
            picture = None
            people_heights = [float(item["bbox"][3]) for item in image["instances"] if roles.get(item["class"], "player") in {"player", "referee"}]
            person_height = float(np.median(people_heights)) if people_heights else None
            for item in image["instances"]:
                done += 1
                role = roles.get(item["class"], "player")
                if role == "ignore":
                    continue
                # A "ball" as big as a person is a logo or a wrong label, and it is no ball in the output either.
                if role == "ball" and person_height and max(item["bbox"][2:]) > MAX_BALL_TO_PERSON * person_height:
                    log.append(f"{image['rel']}: {item['class']} ignored (larger than half a person)")
                    continue
                entry = known.get(item["id"])
                if entry is None or (role == "static" and entry.get("origin") == "sam2"):
                    if picture is None:
                        picture = read_image(source / image["rel"])
                    existing = segmentation_to_mask(item.get("segmentation"), image["height"], image["width"])
                    if existing is not None:
                        ys, xs = np.nonzero(existing)
                        offset, mask, score, origin = (int(xs.min()), int(ys.min())), existing[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 1.0, "coco"
                    elif role == "static":
                        # A goal net or a hoop has no closed outline, and it does not move: the box is the label.
                        x0, y0 = max(int(item["bbox"][0]), 0), max(int(item["bbox"][1]), 0)
                        x1, y1 = min(int(math.ceil(item["bbox"][0] + item["bbox"][2])), image["width"]), min(int(math.ceil(item["bbox"][1] + item["bbox"][3])), image["height"])
                        offset, mask, score, origin = (x0, y0), np.ones((max(y1 - y0, 1), max(x1 - x0, 1)), dtype=bool), 1.0, "box"
                    else:
                        if provider is None:
                            if make_masks is None:
                                raise StudioError(text(language, "SAM 2 ist nicht verfügbar. Zuerst „ML einrichten“ ausführen.", "SAM 2 is not available. Run “Set up ML” first.", "SAM 2 no está disponible. Ejecuta primero «Configurar ML».", "SAM 2 n’est pas disponible. Lancez d’abord « Configurer le ML »."))
                            provider = make_masks()
                        # Boxes often end above the shoes, and SAM stays inside its prompt: give it room to find the feet.
                        prompt = item["bbox"]
                        if role in {"player", "referee"}:
                            prompt = [prompt[0], prompt[1], prompt[2], min(prompt[3] * (1.0 + MASK_PROMPT_BELOW), image["height"] - prompt[1])]
                        mask, offset, score = provider(picture, prompt)
                        mask, origin = clean_mask(mask), "sam2"
                    problem = mask_quality_problem(mask, offset, item["bbox"], score)
                    entry = {"id": item["id"], "score": round(score, 4), "origin": origin, "problem": problem}
                    if problem is None:
                        ys, xs = np.nonzero(mask)
                        top, left, bottom, right = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
                        x0, y0 = offset[0] + left, offset[1] + top
                        alpha = (mask[top:bottom, left:right] * 255).astype(np.uint8)
                        patch = np.dstack((picture[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]], alpha))
                        ok, encoded = cv2.imencode(".png", patch)
                        if not ok:
                            raise StudioError(f"Could not store a mask: {image['rel']}")
                        atomic_write_bytes(cache / "patches" / f"{item['id']}.png", encoded.tobytes())
                        entry.update(x=x0, y=y0, w=alpha.shape[1], h=alpha.shape[0], area=int(mask.sum()))
                    index_file.write(json.dumps(entry) + "\n")
                    index_file.flush()
                if entry.get("problem"):
                    log.append(f"{image['rel']}: {item['class']} dropped ({entry['problem']})")
                    if role != "static":
                        unmasked[image["rel"]].append(item["bbox"])
                else:
                    instances.append({
                        "id": item["id"], "image": image["rel"], "class": item["class"], "role": role, "bbox": item["bbox"],
                        "session": session_of[image["rel"]], "x": entry["x"], "y": entry["y"], "w": entry["w"], "h": entry["h"], "area": entry["area"],
                    })
                progress(done, total)

    stage(3, 4, text(language, "Leere Hallenplatten berechnen", "Compute empty-hall plates", "Calcular placas de pabellón vacío", "Calculer les plaques de salle vide"))
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        by_image[instance["image"]].append(instance)

    def covered_mask(relative: str, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)

        def erase_box(box: list[float]) -> None:
            # Labelled boxes are often tight: heads, arms and shoes stick out and must go with the person.
            bx, by, bw, bh = box
            margin = PLATE_BOX_MARGIN * bh
            below = PLATE_BOX_BELOW * bh  # most often the feet are what a box cuts off
            mask[max(int(by - margin), 0):int(math.ceil(by + bh + below)) + 1, max(int(bx - margin), 0):int(math.ceil(bx + bw + margin)) + 1] = 1

        for instance in by_image.get(relative, []):
            if instance["role"] == "static":
                continue
            with_alpha = cv2.imdecode(np.fromfile(str(cache / "patches" / f"{instance['id']}.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            y0, x0 = instance["y"], instance["x"]
            mask[y0:y0 + instance["h"], x0:x0 + instance["w"]] |= (with_alpha[:, :, 3] > 127).astype(np.uint8)
            if instance["role"] in {"player", "referee"}:
                # Masks often miss legs or shoes: the whole box goes, or body parts stay behind as ghosts.
                erase_box(instance["bbox"])
        # A person without a usable mask must not stay in the picture unlabelled: the whole box goes.
        for box in unmasked.get(relative, []):
            erase_box(box)
        # A little wider than the mask: the edges and contact shadows must not stay in the plate.
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)

    inpainter = None

    def inpaint(picture: np.ndarray, mask: np.ndarray) -> np.ndarray:
        nonlocal inpainter
        if inpainter is None:
            inpainter = make_inpaint() if make_inpaint else opencv_inpaint
        return inpainter(picture, mask)

    plates: dict[str, dict[str, Any]] = {}
    for number, (name, members) in enumerate(sessions.items()):
        slug = stable_hash(name)[:12]
        if split[name] != "train":
            plates[name] = {"mode": "none", "files": {}}
            continue
        height, width = by_rel[members[0]]["height"], by_rel[members[0]]["width"]
        key = stable_hash("plate-5", *members, *[instance["id"] for relative in members for instance in by_image.get(relative, [])], *[box for relative in members for box in unmasked.get(relative, [])])[:16]
        if len(members) >= MIN_FRAMES_FOR_MEDIAN:
            target = cache / "plates" / f"{slug}-{key}.png"
            if not target.is_file():
                chosen = [members[int(round(i))] for i in np.linspace(0, len(members) - 1, min(len(members), MAX_MEDIAN_FRAMES))]
                frames = ((read_image(source / relative), covered_mask(relative, (height, width))) for relative in chosen)
                plate, holes = median_plate(frames, (height, width), len(chosen), cache / "plates" / f"scratch-{slug}", lambda share, n=number: progress(int((n + share) * 1000), len(sessions) * 1000))
                if holes.any():
                    log.append(f"{name}: {int(holes.sum())} plate pixels were never free and were inpainted")
                    plate = inpaint(plate, holes)
                ok, encoded = cv2.imencode(".png", plate)
                atomic_write_bytes(target, encoded.tobytes())
            plates[name] = {"mode": "median", "files": {"*": target.name}}
        else:
            files = {}
            for relative in members:
                target = cache / "plates" / f"{slug}-{stable_hash(relative, key)[:12]}.png"
                if not target.is_file():
                    picture = read_image(source / relative)
                    ok, encoded = cv2.imencode(".png", inpaint(picture, covered_mask(relative, picture.shape[:2])))
                    atomic_write_bytes(target, encoded.tobytes())
                files[relative] = target.name
            plates[name] = {"mode": "inpaint", "files": files}
        progress(number + 1, len(sessions))

    stage(4, 4, text(language, "Spielfeld und Teams bestimmen", "Determine court and teams", "Determinar campo y equipos", "Déterminer terrain et équipes"))
    calibration = load_calibration(source)
    overrides = config.get("teamOverrides") or {}
    records = []
    previous_centroids: dict[str, np.ndarray] = {}
    link_of = link_groups(list(sessions), config.get("sessionLinks") or [])
    for number, (name, members) in enumerate(sessions.items()):
        people = [instance for relative in members for instance in by_image.get(relative, []) if instance["role"] in {"player", "referee"}]
        height, width = by_rel[members[0]]["height"], by_rel[members[0]]["width"]
        patches = {}
        for instance in [item for relative in members for item in by_image.get(relative, [])]:
            patches[instance["id"]] = cv2.imdecode(np.fromfile(str(cache / "patches" / f"{instance['id']}.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            thumb = cache / "thumbs" / f"{instance['id']}.jpg"
            if not thumb.is_file():
                piece = patches[instance["id"]]
                scale = min(1.0, 96.0 / max(piece.shape[:2]))
                alpha = (piece[:, :, 3:] > 127)
                shown = np.where(alpha, piece[:, :, :3], 235).astype(np.uint8)
                ok, encoded = cv2.imencode(".jpg", cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
                atomic_write_bytes(thumb, encoded.tobytes())

        def complete(instance: dict[str, Any]) -> bool:
            """Not cut off by the image edge or hidden behind someone: only such a person can be moved."""
            if instance["x"] <= 1 or instance["y"] <= 1 or instance["x"] + instance["w"] >= width - 1 or instance["y"] + instance["h"] >= height - 1:
                return False
            ax, ay, aw, ah = instance["bbox"]
            # Feet or head missing from the mask: it would float or lose its head at a new place.
            if instance["h"] < 0.85 * ah or instance["y"] + instance["h"] < ay + 0.92 * ah or instance["area"] < MIN_COMPLETE_FILL * aw * ah:
                return False
            for other in by_image[instance["image"]]:
                if other is instance or other["role"] in {"static", "ball"}:
                    continue
                bx, by, bw, bh = other["bbox"]
                overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
                if overlap > 0.15 * aw * ah:
                    return False
            return True

        feet, heights, tilts, candidates = [], [], [], []
        for instance in people:
            instance["foot"] = [instance["x"] + instance["w"] / 2.0, float(instance["y"] + instance["h"])]
            instance["tilt"] = mask_tilt(patches[instance["id"]][:, :, 3] > 127)
            instance["complete"] = complete(instance)
            if instance["complete"]:
                feet.append(instance["foot"])
                heights.append(float(instance["h"]))
                tilts.append(instance["tilt"])
                candidates.append(instance)
        feet_array = np.array(feet, dtype=np.float64).reshape(-1, 2)
        size_map = fit_size_field(feet_array, np.array(heights, dtype=np.float64), width, height)
        tilt_map = fit_field(feet_array, np.array(tilts, dtype=np.float64), width, height)
        # Bent over or half hidden despite passing the box test: much smaller than people there are.
        for instance in candidates:
            expected = max(field_at(size_map, *instance["foot"]), 1.0)
            ratio = instance["h"] / expected
            instance["complete"] = 0.7 <= ratio <= 1.4 and abs(instance["tilt"] - field_at(tilt_map, *instance["foot"])) <= 20.0
        info = calibration.get(name) or calibration.get("*") or {}
        polygon = config["polygons"].get(name) or info.get("polygon")
        polygon_source = "clicked" if config["polygons"].get(name) else "calibration" if info.get("polygon") else "estimated"
        if not polygon:
            polygon = polygon_of_feet(feet_array, width, height, float(np.median(heights)) if heights else 0.0)
        for item in people:
            item["keeper"] = str(item["class"]).casefold() in KEEPER_NAMES
        # Goalkeepers stay on their line and wear neither team's colours.
        players = [item for item in people if item["role"] == "player" and not item["keeper"]]
        features = np.array([jersey_feature(patches[item["id"]][:, :, :3], patches[item["id"]][:, :, 3] > 127) for item in players]).reshape(-1, 19)
        teams, uncertain, centroids = cluster_teams(features)
        group = link_of.get(name, name)
        if group in previous_centroids and len(players) and np.linalg.norm(centroids[0] - previous_centroids[group][1]) + np.linalg.norm(centroids[1] - previous_centroids[group][0]) < np.linalg.norm(centroids - previous_centroids[group], axis=1).sum():
            # Linked sessions must count the same team as 0 or 1.
            teams = np.where(teams == -1, -1, 1 - teams)
            centroids = centroids[::-1]
        previous_centroids.setdefault(group, centroids)
        for item, team, flag in zip(players, teams, uncertain):
            item["team"] = int(overrides.get(item["id"], team))
            item["uncertain"] = bool(flag) and item["id"] not in overrides
        counts = Counter(item["team"] for item in players)
        balls = [item for item in instances if item["session"] == name and item["role"] == "ball"]
        for ball in balls:
            bx, by, bw, bh = ball["bbox"]
            best, share = None, 0.3
            for person in by_image[ball["image"]]:
                if person["role"] not in {"player", "referee"}:
                    continue
                px, py, pw, ph = person["bbox"]
                covered = max(0.0, min(bx + bw, px + pw) - max(bx, px)) * max(0.0, min(by + bh, py + ph) - max(by, py)) / max(bw * bh, 1.0)
                if covered > share:
                    best, share = person, covered
            if best is not None:
                ball["holder"] = best["id"]
                best["ball"] = ball["id"]
        records.append({
            "id": name, "slug": stable_hash(name)[:12], "images": members, "split": split[name], "width": width, "height": height,
            "plate": plates[name], "polygon": polygon, "polygonSource": polygon_source,
            "homography": info.get("homography"), "sizeMap": size_map, "tiltMap": tilt_map,
            "teamCounts": {str(key): value for key, value in sorted(counts.items())},
            "link": group, "teamColours": team_colours(players, patches),
        })
        progress(number + 1, len(sessions))
    analysis = {
        "source": str(source), "classes": classes, "roles": roles, "sessions": records, "instances": instances,
        "splitWarning": split_warning, "log": log,
        "dropped": sum(1 for line in log if " dropped (" in line),
        "key": stable_hash(json.dumps([records, instances], sort_keys=True))[:16],
    }
    atomic_json(work / "analysis.json", analysis)
    return analysis


def load_calibration(source: Path) -> dict[str, dict[str, Any]]:
    """Optional calibration.json: {"homography": 3x3 image->court in metres, "polygon": [[x, y], ...]},
    either directly or per session id under "sessions"."""
    path = source / "calibration.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(document.get("sessions"), dict):
        return document["sessions"]
    return {"*": document}


def link_groups(names: list[str], links: list[list[str]]) -> dict[str, str]:
    """Session -> name of its group; linked sessions show the same two teams."""
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            name = parent[name]
        return name

    for link in links:
        members = [name for name in link if name in parent]
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    return {name: find(name) for name in names}


def team_colours(players: list[dict[str, Any]], patches: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Typical jersey colour of each team (BGR): drawn in the gallery and the base of the jersey variation."""
    result = {}
    for team in (0, 1):
        pixels = []
        for player in players:
            if player.get("team") != team:
                continue
            patch = patches[player["id"]]
            top, bottom = patch.shape[0] // 3, 2 * patch.shape[0] // 3
            pixels.append(patch[top:bottom, :, :3][patch[top:bottom, :, 3] > 127])
        joined = np.concatenate(pixels) if pixels else np.zeros((0, 3), dtype=np.uint8)
        result[str(team)] = [int(item) for item in np.median(joined, axis=0)] if len(joined) else [128, 128, 128]
    return result


# --- planning -------------------------------------------------------------------------------

class Context:
    """Everything generate() needs, read-only, shared by the worker processes."""

    def __init__(self, work: Path, config: dict[str, Any]) -> None:
        self.work = work
        self.config = merged_config(config)
        self.analysis = json.loads((work / "analysis.json").read_text(encoding="utf-8"))
        self.instances = {item["id"]: item for item in self.analysis["instances"]}
        self.sessions = {item["id"]: item for item in self.analysis["sessions"]}
        self.geometry = {name: Geometry(item) for name, item in self.sessions.items()}
        self.class_id = {name: index + 1 for index, name in enumerate(self.output_classes())}
        self.groups = defaultdict(list)
        for session in self.sessions.values():
            if session["split"] == "train":
                self.groups[session["link"]].append(session["id"])
        self._patches: dict[str, np.ndarray] = {}
        self._plates: dict[str, np.ndarray] = {}
        self.anonymizer = None

    def output_classes(self) -> list[str]:
        roles = self.analysis["roles"]
        return [name for name in self.analysis["classes"] if roles.get(name) in {"player", "referee", "ball", "static"}]

    def patch(self, instance_id: str) -> np.ndarray:
        patch = self._patches.get(instance_id)
        if patch is None:
            if len(self._patches) > 400:
                self._patches.clear()
            patch = cv2.imdecode(np.fromfile(str(self.work / "cache" / "patches" / f"{instance_id}.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self._patches[instance_id] = patch
        return patch

    def plate_behind(self, instance: dict[str, Any]) -> np.ndarray:
        """The plate of the session the instance comes from."""
        plate = self.sessions[instance["session"]]["plate"]
        return self.plate(plate["files"].get("*") or plate["files"][instance["image"]])

    def plate(self, name: str) -> np.ndarray:
        plate = self._plates.get(name)
        if plate is None:
            if len(self._plates) > 3:
                self._plates.clear()
            plate = read_image(self.work / "cache" / "plates" / name)
            self._plates[name] = plate
        return plate


def usable_groups(context: Context) -> dict[str, dict[str, Any]]:
    """Groups of training sessions that can show two teams, with their pools of people."""
    usable = {}
    for group, names in context.groups.items():
        pool: dict[int, list[str]] = {0: [], 1: []}
        referees: list[str] = []
        for instance in context.analysis["instances"]:
            if instance["session"] not in names or not instance.get("complete") or not context.geometry[instance["session"]].on_field(*instance["foot"]):
                continue
            if instance["role"] == "referee":
                referees.append(instance["id"])
            elif instance["role"] == "player" and instance.get("team") in (0, 1):
                pool[instance["team"]].append(instance["id"])
        if len(pool[0]) >= 2 and len(pool[1]) >= 2:
            usable[group] = {"sessions": sorted(names), "teams": pool, "referees": referees}
    return usable


def image_rng(seed: int, index: int, attempt: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(index), int(attempt)])))


def find_foot(context: Context, geometry: Geometry, instance: dict[str, Any], taken: list[tuple[float, float]], rng: np.random.Generator) -> dict[str, Any] | None:
    """A new foot point: inside the field, not on someone's feet, and (without calibration) with
    a lean close enough to the person's own that turning stays invisible."""
    polygon = geometry.polygon
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    old = instance["foot"]
    source_geometry = context.geometry[instance["session"]]
    old_height = source_geometry.height_at(*old)
    old_tilt = source_geometry.tilt_at(*old)
    if geometry.local:
        reach_x, reach_y = 1.5 * old_height, (0.35 if geometry.size["degree"] else 0.1) * old_height
        x0, x1 = max(x0, old[0] - reach_x), min(x1, old[0] + reach_x)
        y0, y1 = max(y0, old[1] - reach_y), min(y1, old[1] + reach_y)
        if x1 <= x0 or y1 <= y0:
            return None
    for _ in range(40):
        x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        if not geometry.on_field(x, y):
            continue
        if any(geometry.distance_m((x, y), other) < MIN_FOOT_DISTANCE_M for other in taken):
            continue
        tilt = geometry.tilt_at(x, y)
        flip = bool(rng.random() < 0.5)
        tilt_change = tilt - (-old_tilt if flip else old_tilt)
        if flip and abs(tilt_change) > MAX_TILT_CHANGE_DEG and abs(tilt - old_tilt) <= MAX_TILT_CHANGE_DEG:
            flip, tilt_change = False, tilt - old_tilt
        if not geometry.homography and abs(tilt_change) > MAX_TILT_CHANGE_DEG:
            continue
        scale = geometry.height_at(x, y) / max(old_height, 1.0)
        if not MIN_SCALE <= scale <= MAX_SCALE:
            continue
        return {"new": [x, y], "scale": scale, "rotate": tilt_change, "flip": flip}
    return None


def plan_image(context: Context, index: int, attempt: int) -> dict[str, Any]:
    """Which instances go where, and which part of the court is shown. Only JSON values."""
    config = context.config
    seed = int(config["seed"])
    rng = image_rng(seed, index, attempt)
    groups = usable_groups_cached(context)
    names = sorted(groups)
    group = groups[names[int(rng.integers(len(names)))]]
    session_name = group["sessions"][int(rng.integers(len(group["sessions"])))]
    session = context.sessions[session_name]
    geometry = context.geometry[session_name]
    plate_info = session["plate"]
    frame = None
    if plate_info["mode"] == "inpaint":
        options = sorted(plate_info["files"])
        frame = options[int(rng.integers(len(options)))]
        plate = plate_info["files"][frame]
    else:
        plate = plate_info["files"]["*"]
    limits = config["limits"]
    placements: list[dict[str, Any]] = []
    taken: list[tuple[float, float]] = []
    used: set[str] = set()
    ball_used = False
    max_balls = int(limits["balls"])

    def place(instance_id: str) -> None:
        nonlocal ball_used
        instance = context.instances[instance_id]
        ball_id = instance.get("ball")
        carries = ball_id in context.instances and context.instances[ball_id]["session"] == instance["session"]
        spot = find_foot(context, geometry, instance, taken, rng)
        if spot is None:
            return
        taken.append(tuple(spot["new"]))
        used.add(instance_id)
        placements.append({"id": instance_id, "old": instance["foot"], **spot})
        if carries and not ball_used and max_balls > 0:
            # The ball moves with the player holding it, by the same transform.
            placements.append({"id": ball_id, "old": instance["foot"], **spot, "with": instance_id})
            used.add(ball_id)
            ball_used = True

    for team in (0, 1):
        pool = group["teams"][team]
        most = int(limits["players"])
        wanted = int(rng.integers(min(2, most), most + 1))
        for position in rng.permutation(len(pool))[:wanted]:
            # A second ball is not allowed, so anyone holding one sits out once the ball is taken.
            instance = context.instances[pool[int(position)]]
            if ball_used and instance.get("ball"):
                continue
            place(pool[int(position)])
    if group["referees"] and int(limits["referees"]) > 0:
        wanted = int(rng.integers(0, min(int(limits["referees"]), len(group["referees"])) + 1))
        for position in rng.permutation(len(group["referees"]))[:wanted]:
            place(group["referees"][int(position)])

    members = set(session["images"])
    anchor = frame or sorted(members)[int(rng.integers(len(members)))]
    # A free ball stays exactly where it was, in the session the plate belongs to.
    if not ball_used and max_balls > 0:
        free = [item["id"] for item in context.analysis["instances"] if item["role"] == "ball" and item["session"] == session_name and not item.get("holder")]
        if free and rng.random() < 0.5:
            chosen = free[int(rng.integers(len(free)))]
            foot = [context.instances[chosen]["x"] + context.instances[chosen]["w"] / 2.0, float(context.instances[chosen]["y"] + context.instances[chosen]["h"])]
            placements.append({"id": chosen, "old": foot, "new": foot, "scale": 1.0, "rotate": 0.0, "flip": False})
            used.add(chosen)
    # People beside the court keep their place and do not count towards the limits.
    for item in context.analysis["instances"]:
        if item["image"] == anchor and item["role"] in {"player", "referee"} and item["id"] not in used and (not geometry.on_field(*item["foot"]) or item.get("keeper")) and item.get("team") in (0, 1, None):
            placements.append({"id": item["id"], "old": item["foot"], "new": item["foot"], "scale": 1.0, "rotate": 0.0, "flip": False, "fixed": True})
    statics = [item["id"] for item in context.analysis["instances"] if item["image"] == anchor and item["role"] == "static"]
    if not statics:
        # The anchor frame may lack a label for the hoop: take the frame of this session with the most.
        counts = Counter(item["image"] for item in context.analysis["instances"] if item["session"] == session_name and item["role"] == "static")
        if counts:
            best = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
            statics = [item["id"] for item in context.analysis["instances"] if item["image"] == best and item["role"] == "static"]
    width, height = session["width"], session["height"]
    use_crop = ((index + seed % 1000) * 0.6180339887498949) % 1.0 < float(config["cropShare"])
    window = None
    if use_crop:
        low, high = config["zoom"]
        window_width = float(rng.uniform(low, high)) * width
        window_height = min(float(height), window_width)  # square output: the panorama's height is the limit
        window_width = min(window_width, float(width))
        feet = [item["new"] for item in placements if not item.get("fixed") and not item.get("with")]
        if feet and rng.random() >= RANDOM_CROP_SHARE:
            cx, cy = feet[int(rng.integers(len(feet)))]
            cx += float(rng.normal(0.0, window_width * 0.15))
            cy += float(rng.normal(0.0, window_height * 0.15))
        else:
            cx, cy = float(rng.uniform(0, width)), float(rng.uniform(0, height))
        x0 = float(np.clip(cx - window_width / 2, 0, width - window_width))
        y0 = float(np.clip(cy - window_height / 2, 0, height - window_height))
        window = [int(round(x0)), int(round(y0)), int(round(window_width)), int(round(window_height))]
    photometric = {
        "brightness": float(rng.uniform(-18, 18)), "contrast": float(rng.uniform(0.88, 1.12)),
        "white": [float(rng.uniform(0.95, 1.05)) for _ in range(3)], "noise": float(rng.uniform(0, 4)),
        "blur": bool(rng.random() < 0.2), "jpeg": int(rng.integers(60, 96)),
        "ballBlur": int(rng.choice([3, 5, 7, 9])) if rng.random() < 0.3 else 0, "ballBlurAngle": float(rng.uniform(0, 180)),
        "noiseSeed": int(rng.integers(0, 2**31)),
    }
    jersey = jersey_plan(context, session, rng) if config.get("jerseyVariation") else None
    return {
        "index": index, "attempt": attempt, "seed": seed, "session": session_name, "group": session["link"], "plate": plate,
        "placements": placements, "statics": statics, "window": window, "photometric": photometric, "jersey": jersey,
    }


def usable_groups_cached(context: Context) -> dict[str, dict[str, Any]]:
    cached = getattr(context, "_usable", None)
    if cached is None:
        cached = usable_groups(context)
        context._usable = cached
    if not cached:
        raise StudioError("No session has at least two complete players per team on the court.")
    return cached


def bgr_to_lab(colour: Iterable[float]) -> np.ndarray:
    pixel = np.array([[list(colour)]], dtype=np.float32) / 255.0
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0, 0]


def shifted_colour(colour: list[int], shift: int) -> list[float]:
    pixel = np.array([[colour]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    hsv[0, 0, 0] = (int(hsv[0, 0, 0]) + shift) % 180
    return [float(item) for item in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]]


def jersey_plan(context: Context, session: dict[str, Any], rng: np.random.Generator) -> dict[str, Any] | None:
    """One hue shift per team, or none: both teams must stay clearly apart (ΔE > 25)."""
    colours = session["teamColours"]
    for _ in range(12):
        shifts = {team: int(rng.integers(-60, 61)) for team in ("0", "1")}
        lab = [bgr_to_lab(shifted_colour(colours[team], shifts[team])) for team in ("0", "1")]
        if float(np.linalg.norm(lab[0] - lab[1])) > MIN_JERSEY_DELTA_E:
            return {"shifts": shifts, "hues": {team: int(cv2.cvtColor(np.array([[colours[team]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0, 0]) for team in ("0", "1")}}
    return None


def apply_jersey(patch: np.ndarray, plan: dict[str, Any], team: int) -> np.ndarray:
    """Hue shift of the pixels in the team's colour cluster only; skin and shoes stay as they are."""
    if team not in (0, 1):
        return patch
    key = str(team)
    hsv = cv2.cvtColor(patch[:, :, :3], cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    distance = np.abs(hue - plan["hues"][key])
    close = np.minimum(distance, 180 - distance) <= 12
    jersey = close & (hsv[:, :, 1] > 60) & (patch[:, :, 3] > 127)
    hsv[:, :, 0] = np.where(jersey, (hue + plan["shifts"][key]) % 180, hue).astype(np.uint8)
    result = patch.copy()
    result[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result


# --- rendering ------------------------------------------------------------------------------

def placement_matrix(placement: dict[str, Any], origin: tuple[float, float]) -> np.ndarray:
    """Patch pixels -> canvas: uniform scale, turn and optional mirror about the foot point."""
    old_x, old_y = placement["old"][0] - origin[0], placement["old"][1] - origin[1]
    new_x, new_y = placement["new"]
    scale = float(placement["scale"])
    angle = math.radians(float(placement["rotate"]))
    # "rotate" is the change of lean: positive moves the head to the right (y points down).
    cosine, sine = math.cos(angle) * scale, math.sin(angle) * scale
    mirror = -1.0 if placement.get("flip") else 1.0
    a, b, c, d = cosine * mirror, -sine, sine * mirror, cosine
    return np.array([
        [a, b, new_x - a * old_x - b * old_y],
        [c, d, new_y - c * old_x - d * old_y],
    ], dtype=np.float64)


def warp_patch(patch: np.ndarray, matrix: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, int, int, float] | None:
    """The patch on the canvas, only in its bounding region. None when nothing of it is inside."""
    h, w = patch.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float64) @ matrix.T
    x0, y0 = max(0, int(math.floor(corners[:, 0].min())) - 2), max(0, int(math.floor(corners[:, 1].min())) - 2)
    x1, y1 = min(width, int(math.ceil(corners[:, 0].max())) + 2), min(height, int(math.ceil(corners[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    shifted = matrix.copy()
    shifted[0, 2] -= x0
    shifted[1, 2] -= y0
    colour = cv2.warpAffine(patch[:, :, :3], shifted, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine((patch[:, :, 3] > 127).astype(np.float32), shifted, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return colour, alpha, x0, y0, float(abs(np.linalg.det(matrix[:2, :2])))


def resolve_visibility(items: list[dict[str, Any]], shape: tuple[int, int], min_visible: float) -> list[dict[str, Any]]:
    """items are ordered far to near, with "mask" (bool), "x", "y" and "full" (area before anything
    hid it). Sets "visible" (bool mask) and "visibleShare"; drops whoever is hidden too much.
    Dropping someone can uncover the ones behind, so this repeats until it is stable."""
    current = list(items)
    while True:
        occupied = np.zeros(shape, dtype=bool)
        worst = None
        for item in reversed(current):  # near to far
            mask = item["mask"]
            h, w = mask.shape
            region = occupied[item["y"]:item["y"] + h, item["x"]:item["x"] + w]
            visible = mask & ~region
            item["visible"] = visible
            item["visibleShare"] = float(visible.sum()) / max(float(item["full"]), 1.0)
            region |= mask
            if item.get("keep") or item["visibleShare"] >= min_visible:
                continue
            if worst is None or item["visibleShare"] < worst["visibleShare"]:
                worst = item
        if worst is None:
            return current
        current = [item for item in current if item is not worst]


def tight_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def render_canvas(context: Context, plan: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The full-court composition and its instances with their visible masks."""
    plate = context.plate(plan["plate"])
    canvas = plate.copy()
    height, width = canvas.shape[:2]
    # Far away first; a ball is in front of the player holding it.
    ordered = sorted(plan["placements"], key=lambda item: (item["new"][1], 1 if item.get("with") else 0, item["id"]))
    items = []
    for placement in ordered:
        instance = context.instances[placement["id"]]
        patch = context.patch(placement["id"])
        if plan.get("jersey") and instance["role"] == "player":
            patch = apply_jersey(patch, plan["jersey"], instance.get("team", -1))
        origin = (float(instance["x"]), float(instance["y"]))
        matrix = placement_matrix(placement, origin)
        warped = warp_patch(patch, matrix, width, height)
        if warped is None:
            continue
        colour, alpha, x, y, area_scale = warped
        mask = alpha > 0.5
        if not mask.any():
            continue
        items.append({
            "placement": placement, "instance": instance, "colour": colour, "mask": mask, "x": x, "y": y,
            "full": float(instance["area"]) * area_scale, "keep": bool(placement.get("fixed")),
        })
    items = resolve_visibility(items, (height, width), float(context.config["minVisible"]))
    grey = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for item in items:
        mask, colour, x, y = item["mask"], item["colour"], item["x"], item["y"]
        h, w = mask.shape
        # Soft 1-2 px edge, and a slight adjustment to the brightness of the hall at the new place.
        sigma = 0.7 if item["instance"]["role"] == "ball" else 1.2
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)[:, :, None]
        instance = item["instance"]
        behind = context.plate_behind(instance)[instance["y"]:instance["y"] + instance["h"], instance["x"]:instance["x"] + instance["w"]]
        gain = float(np.clip(float(grey[y:y + h, x:x + w][mask].mean()) / max(float(cv2.cvtColor(behind, cv2.COLOR_BGR2GRAY).mean()), 1.0), 0.85, 1.15))
        adjusted = np.clip(colour.astype(np.float32) * gain, 0, 255)
        part = canvas[y:y + h, x:x + w].astype(np.float32)
        canvas[y:y + h, x:x + w] = np.round(part * (1.0 - soft) + adjusted * soft).astype(np.uint8)
    labels = [
        {"id": item["instance"]["id"], "instance": item["instance"], "mask": item["visible"], "x": item["x"], "y": item["y"],
         "occluded": 1.0 - item["visibleShare"], "foot": item["placement"]["new"]}
        for item in items
    ]
    for instance_id in plan["statics"]:
        instance = context.instances[instance_id]
        alpha = context.patch(instance_id)[:, :, 3] > 127
        labels.append({"id": instance_id, "instance": instance, "mask": alpha, "x": instance["x"], "y": instance["y"], "occluded": 0.0, "foot": None})
    return canvas, labels


def framed(image: np.ndarray, window: list[int] | None, size: int) -> tuple[np.ndarray, list[int], float, tuple[int, int]]:
    """Uniform scale into the square output with padding; returns scale and padding too."""
    height, width = image.shape[:2]
    x, y, w, h = window if window else [0, 0, width, height]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image[y:y + h, x:x + w], (new_w, new_h), interpolation=interpolation)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out, [x, y, w, h], scale, (pad_x, pad_y)


def output_labels(labels: list[dict[str, Any]], window: list[int], scale: float, padding: tuple[int, int], size: int, context: Context) -> tuple[list[dict[str, Any]], int]:
    """Labels in the output image: clipped to the window, from the mask, or dropped."""
    wx, wy, ww, wh = window
    result = []
    too_small = 0
    pad_x, pad_y = padding
    minimum = float(context.config["cropMinVisible"])
    for label in labels:
        mask = label["mask"]
        total = int(mask.sum())
        if total == 0:
            continue
        h, w = mask.shape
        x0, y0 = max(label["x"], wx), max(label["y"], wy)
        x1, y1 = min(label["x"] + w, wx + ww), min(label["y"] + h, wy + wh)
        if x1 <= x0 or y1 <= y0:
            continue
        part = mask[y0 - label["y"]:y1 - label["y"], x0 - label["x"]:x1 - label["x"]]
        inside = int(part.sum())
        if inside == 0 or inside / total < minimum:
            continue
        box = tight_box(part)
        left, top, right, bottom = box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0
        out_x0, out_y0 = (left - wx) * scale + pad_x, (top - wy) * scale + pad_y
        out_x1, out_y1 = (right - wx) * scale + pad_x, (bottom - wy) * scale + pad_y
        if out_x1 - out_x0 < MIN_LABEL_PIXELS or out_y1 - out_y0 < MIN_LABEL_PIXELS:
            too_small += 1
            continue
        out_mask = np.zeros((size, size), dtype=bool)
        tx0, ty0 = int(round((x0 - wx) * scale + pad_x)), int(round((y0 - wy) * scale + pad_y))
        tw, th = max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))
        tw, th = min(tw, size - tx0), min(th, size - ty0)
        if tw > 0 and th > 0:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            out_mask[ty0:ty0 + th, tx0:tx0 + tw] = cv2.resize(part.astype(np.float32), (tw, th), interpolation=interpolation) >= 0.5
        # The box is what can be seen in the output: a one-pixel sliver of a hidden leg disappears when
        # scaling down and must not stretch the box. Only a very small object keeps its exact box.
        drawn = tight_box(out_mask)
        if drawn is not None:
            out_x0, out_y0, out_x1, out_y1 = float(drawn[0]), float(drawn[1]), float(drawn[2]), float(drawn[3])
        else:
            out_mask[min(size - 1, int((out_y0 + out_y1) / 2)), min(size - 1, int((out_x0 + out_x1) / 2))] = True
        instance = label["instance"]
        out_x0, out_y0 = max(0.0, out_x0), max(0.0, out_y0)
        out_x1, out_y1 = min(float(size), out_x1), min(float(size), out_y1)
        annotation = {
            "category_id": context.class_id[instance["class"]],
            "bbox": [round(out_x0, 2), round(out_y0, 2), round(out_x1 - out_x0, 2), round(out_y1 - out_y0, 2)],
            "area": int(out_mask.sum()), "iscrowd": 0, "segmentation": mask_to_rle(out_mask),
            "instance_id": label["id"], "session_id": instance["session"], "role": instance["role"],
            "team_id": instance.get("team") if instance["role"] == "player" else None,
            "occluded_ratio": round(float(label["occluded"]), 4),
            "truncated": inside < total,
        }
        if label["foot"] is not None:
            annotation["foot"] = [round(label["foot"][0], 2), round(label["foot"][1], 2)]
        result.append(annotation)
    return result, too_small


def apply_photometric(image: np.ndarray, settings: dict[str, Any], annotations: list[dict[str, Any]]) -> np.ndarray:
    """Whole-image changes; nothing here moves a pixel to another place, apart from the ball's short blur."""
    if settings["ballBlur"]:
        for annotation in annotations:
            if annotation["role"] != "ball":
                continue
            x, y, w, h = annotation["bbox"]
            margin = settings["ballBlur"]
            x0, y0 = max(0, int(x) - margin), max(0, int(y) - margin)
            x1, y1 = min(image.shape[1], int(x + w) + margin + 1), min(image.shape[0], int(y + h) + margin + 1)
            size = settings["ballBlur"]
            kernel = np.zeros((size, size), dtype=np.float32)
            angle = math.radians(settings["ballBlurAngle"])
            middle = (size - 1) / 2.0
            cv2.line(kernel, (int(round(middle - math.cos(angle) * middle)), int(round(middle - math.sin(angle) * middle))), (int(round(middle + math.cos(angle) * middle)), int(round(middle + math.sin(angle) * middle))), 1.0, 1)
            kernel /= max(float(kernel.sum()), 1.0)
            image[y0:y1, x0:x1] = cv2.filter2D(image[y0:y1, x0:x1], -1, kernel)
    result = image.astype(np.float32)
    result = (result - 127.5) * settings["contrast"] + 127.5 + settings["brightness"]
    result *= np.array(settings["white"], dtype=np.float32)
    if settings["noise"] > 0:
        result += np.random.Generator(np.random.PCG64(settings["noiseSeed"])).normal(0.0, settings["noise"], result.shape).astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if settings["blur"]:
        result = cv2.GaussianBlur(result, (3, 3), 0.6)
    return result


def render_plan(context: Context, plan: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]], int]:
    canvas, labels = render_canvas(context, plan)
    size = int(context.config["outputSize"])
    image, window, scale, padding = framed(canvas, plan["window"], size)
    annotations, too_small = output_labels(labels, window, scale, padding, size, context)
    image = apply_photometric(image, plan["photometric"], annotations)
    if context.anonymizer is not None:
        reco = [{"category": "ball" if item["role"] == "ball" else "player" if item["role"] in {"player", "referee"} else "static", "x": item["bbox"][0], "y": item["bbox"][1], "width": item["bbox"][2], "height": item["bbox"][3]} for item in annotations]
        image, _ = context.anonymizer.anonymize(image, reco)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(plan["photometric"]["jpeg"])])
    if not ok:
        raise StudioError("Could not encode an image.")
    return encoded.tobytes(), annotations, too_small


# --- validator ------------------------------------------------------------------------------

def check_annotations(annotations: list[dict[str, Any]], size: int, limits: dict[str, Any], valid_classes: set[int], polygon_of: Callable[[str], Any], group_of: Callable[[str], str]) -> list[str]:
    """Hard rules for one image. Returns what is wrong; empty means valid."""
    problems = []
    seen: set[str] = set()
    teams: set[int] = set()
    per_team: Counter = Counter()
    referees = balls = 0
    groups = set()
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > size + 1e-6 or y + h > size + 1e-6:
            problems.append("bbox")
        if annotation["category_id"] not in valid_classes:
            problems.append("class")
        if annotation["instance_id"] in seen:
            problems.append("duplicate")
        seen.add(annotation["instance_id"])
        role = annotation["role"]
        if role == "static":
            continue
        groups.add(group_of(annotation["session_id"]))
        foot = annotation.get("foot")
        on_field = foot is not None and point_in_polygon(polygon_of(annotation["session_id"]), foot[0], foot[1])
        if role == "ball":
            balls += 1
        elif role == "player":
            team = annotation.get("team_id")
            if team is not None:
                teams.add(team)
            if on_field:
                per_team[team] += 1
        elif role == "referee" and on_field:
            referees += 1
    if len(teams) > 2 or teams - {0, 1}:
        problems.append("teams")
    if len(groups) > 1:
        problems.append("sessions")
    if any(count > int(limits["players"]) for count in per_team.values()):
        problems.append("players")
    if referees > int(limits["referees"]):
        problems.append("referees")
    if balls > int(limits["balls"]):
        problems.append("balls")
    return problems


def make_image(context: Context, index: int, first_attempt: int = 0) -> dict[str, Any]:
    """Plans and renders image `index`; a plan breaking a hard rule is thrown away and redone."""
    rejected: Counter = Counter()
    polygon_of = lambda name: context.sessions[name]["polygon"]  # noqa: E731
    group_of = lambda name: context.sessions[name]["link"]  # noqa: E731
    valid = set(context.class_id.values())
    size = int(context.config["outputSize"])
    for attempt in range(first_attempt, first_attempt + MAX_ATTEMPTS):
        plan = plan_image(context, index, attempt)
        data, annotations, too_small = render_plan(context, plan)
        problems = check_annotations(annotations, size, context.config["limits"], valid, polygon_of, group_of)
        has_person = any(item["role"] in {"player", "referee"} for item in annotations)
        if not has_person and index % int(1 / MAX_NEGATIVE_SHARE) != 0:
            problems.append("negative")
        if problems:
            rejected.update(problems)
            continue
        return {"index": index, "attempt": attempt, "data": data, "annotations": annotations, "plan": plan, "tooSmall": too_small, "rejected": dict(rejected)}
    raise StudioError(f"Image {index}: no valid composition in {MAX_ATTEMPTS} attempts ({dict(rejected)}).")


_worker_context: Context | None = None


def _init_worker(work: str, config: dict[str, Any], face_model: str | None) -> None:
    global _worker_context
    cv2.setNumThreads(1)  # one process per core already, and results stay identical
    _worker_context = Context(Path(work), config)
    if face_model:
        import anonymize
        _worker_context.anonymizer = anonymize.Anonymizer(Path(face_model))


def _make_in_worker(job: tuple[int, int]) -> dict[str, Any]:
    assert _worker_context is not None
    return make_image(_worker_context, job[0], job[1])


# --- generation -----------------------------------------------------------------------------

def run_key(config: dict[str, Any], analysis_key: str) -> str:
    relevant = {key: config[key] for key in ("cropShare", "zoom", "outputSize", "seed", "limits", "jerseyVariation", "anonymize", "minVisible", "cropMinVisible")}
    return stable_hash(json.dumps(relevant, sort_keys=True), analysis_key)[:16]


def read_finished(output: Path) -> list[dict[str, Any]]:
    """The finished images of an earlier run. Lines cut off by a cancel, and images without a
    complete line, are removed, so what remains is always a consistent partial dataset."""
    manifest = output / "manifest.jsonl"
    labels = output / "labels.jsonl"
    entries: list[dict[str, Any]] = []
    if not manifest.is_file() or not labels.is_file():
        return entries
    manifest_lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
    label_lines = labels.read_text(encoding="utf-8").splitlines(keepends=True)
    keep = 0
    for manifest_line, label_line in zip(manifest_lines, label_lines):
        if not manifest_line.endswith("\n") or not label_line.endswith("\n"):
            break
        try:
            entry = json.loads(manifest_line)
            json.loads(label_line)
        except ValueError:
            break
        if entry.get("index") != keep or not (output / "images" / entry["file"]).is_file():
            break
        entries.append(entry)
        keep += 1
    manifest.write_text("".join(manifest_lines[:keep]), encoding="utf-8")
    labels.write_text("".join(label_lines[:keep]), encoding="utf-8")
    known = {entry["file"] for entry in entries}
    for path in (output / "images").glob("*"):
        if path.name not in known:
            path.unlink()
    return entries


def image_name(index: int) -> str:
    return f"syn-{index:07d}.jpg"


def write_coco(output: Path, context: Context, count: int) -> None:
    """annotations.coco.json without holding the dataset in memory."""
    size = int(context.config["outputSize"])
    temporary = output / "annotations.coco.json.part"
    annotation_id = 0
    with temporary.open("w", encoding="utf-8") as target, (output / "labels.jsonl").open(encoding="utf-8") as lines:
        target.write('{"info": {"description": "Reco Trainer Augmentation Studio"}, "licenses": [], "categories": ')
        target.write(json.dumps([{"id": number, "name": name, "supercategory": "sport"} for name, number in context.class_id.items()], ensure_ascii=False))
        target.write(', "images": [')
        target.write(", ".join(json.dumps({"id": index + 1, "file_name": image_name(index), "width": size, "height": size}) for index in range(count)))
        target.write('], "annotations": [')
        first = True
        for index, line in enumerate(lines):
            for annotation in json.loads(line)["annotations"]:
                annotation_id += 1
                target.write(("" if first else ", ") + json.dumps({"id": annotation_id, "image_id": index + 1, **annotation}, ensure_ascii=False, sort_keys=True))
                first = False
        target.write("]}\n")
    os.replace(temporary, output / "annotations.coco.json")


def write_yolo(output: Path, context: Context) -> None:
    size = float(context.config["outputSize"])
    folder = output / "yolo"
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    with (output / "labels.jsonl").open(encoding="utf-8") as lines:
        for index, line in enumerate(lines):
            rows = []
            for annotation in json.loads(line)["annotations"]:
                x, y, w, h = annotation["bbox"]
                rows.append(f"{annotation['category_id'] - 1} {(x + w / 2) / size:.6f} {(y + h / 2) / size:.6f} {w / size:.6f} {h / size:.6f}")
            (folder / "labels" / f"{Path(image_name(index)).stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    names = "\n".join(f"  {number - 1}: {name}" for name, number in context.class_id.items())
    (folder / "data.yaml").write_text(f"path: ..\ntrain: images\nnames:\n{names}\n", encoding="utf-8")


def write_split(output: Path, context: Context) -> dict[str, list[str]]:
    """Validation and test hold only real images, by whole session; synthetic images only train."""
    split: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for session in context.analysis["sessions"]:
        split[session["split"]].extend(session["images"])
    atomic_json(output / "split.json", {"source": context.analysis["source"], "real": split, "synthetic": "train"})
    return split


def validate_output(output: Path, context: Context) -> tuple[Counter, list[int]]:
    """Independent check of the written dataset. Returns the problems and the images concerned."""
    problems: Counter = Counter()
    bad: set[int] = set()
    size = int(context.config["outputSize"])
    polygon_of = lambda name: context.sessions[name]["polygon"]  # noqa: E731
    group_of = lambda name: context.sessions[name]["link"]  # noqa: E731
    valid = set(context.class_id.values())
    negatives = 0
    with (output / "labels.jsonl").open(encoding="utf-8") as lines:
        for index, line in enumerate(lines):
            annotations = json.loads(line)["annotations"]
            found = check_annotations(annotations, size, context.config["limits"], valid, polygon_of, group_of)
            negatives += not any(item["role"] in {"player", "referee"} for item in annotations)
            if not (output / "images" / image_name(index)).is_file():
                found.append("missing-image")
            if found:
                problems.update(found)
                bad.add(index)
            if index % 200 == 0:
                progress(index, int(context.config["variations"]))
    if negatives > MAX_NEGATIVE_SHARE * int(context.config["variations"]) + 1:
        problems["negatives"] += negatives
    return problems, sorted(bad)


def write_previews(output: Path, context: Context, count: int, wanted: int = 50) -> None:
    """A random sample with labels and team colours drawn in, for a look before training."""
    folder = output / "preview"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    chosen = set(image_rng(int(context.config["seed"]), 0, 987654).permutation(count)[:wanted].tolist())
    colours = {0: (40, 90, 240), 1: (240, 130, 30)}
    with (output / "labels.jsonl").open(encoding="utf-8") as lines:
        for index, line in enumerate(lines):
            if index not in chosen:
                continue
            image = read_image(output / "images" / image_name(index))
            for annotation in json.loads(line)["annotations"]:
                x, y, w, h = annotation["bbox"]
                role = annotation["role"]
                colour = colours.get(annotation.get("team_id"), (0, 200, 255) if role == "ball" else (160, 60, 200) if role == "static" else (40, 40, 40))
                mask = rle_to_mask(annotation["segmentation"])
                image[mask] = np.round(image[mask] * 0.7 + np.array(colour) * 0.3).astype(np.uint8)
                cv2.rectangle(image, (int(x), int(y)), (int(math.ceil(x + w)), int(math.ceil(y + h))), colour, 1)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            atomic_write_bytes(folder / image_name(index), encoded.tobytes())


def generate(work: Path, config: dict[str, Any], language: str = "de", processes: int | None = None, face_model: Path | None = None) -> dict[str, Any]:
    """Generates the dataset; continues an interrupted run with the same settings."""
    config = merged_config(config)
    count = int(config["variations"])
    if count <= 0:
        raise StudioError(text(language, "Bitte die Anzahl der Variationen angeben.", "Please enter the number of variations.", "Indica el número de variaciones.", "Veuillez indiquer le nombre de variations."))
    if not (work / "analysis.json").is_file():
        raise StudioError(text(language, "Zuerst „Analysieren“ ausführen.", "Run “Analyze” first.", "Ejecuta primero «Analizar».", "Lancez d’abord « Analyser »."))
    context = Context(work, config)
    if not usable_groups(context):
        raise StudioError(text(
            language,
            "Keine Trainings-Session hat je Team mindestens zwei vollständig sichtbare Spieler auf dem Feld. Prüfe Sessions, Teams und Spielfeld.",
            "No training session has at least two fully visible players per team on the court. Check sessions, teams and the court.",
            "Ninguna sesión de entrenamiento tiene al menos dos jugadores completamente visibles por equipo en el campo. Revisa sesiones, equipos y campo.",
            "Aucune session d’entraînement n’a au moins deux joueurs entièrement visibles par équipe sur le terrain. Vérifiez sessions, équipes et terrain.",
        ))
    output = work / "output"
    key = run_key(config, context.analysis["key"])
    state_file = output / "run.json"
    if state_file.is_file() and json.loads(state_file.read_text(encoding="utf-8")).get("key") != key:
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    require_space(output, count * BYTES_PER_IMAGE, language)
    atomic_json(state_file, {"key": key, "config": config})

    stage(1, 2, text(language, "Bilder erzeugen", "Generate images", "Generar imágenes", "Générer les images"))
    finished = read_finished(output)
    if len(finished) > count:
        with (output / "manifest.jsonl").open("r+", encoding="utf-8") as manifest, (output / "labels.jsonl").open("r+", encoding="utf-8") as labels:
            for file in (manifest, labels):
                kept = file.readlines()[:count]
                file.seek(0)
                file.truncate()
                file.writelines(kept)
        for entry in finished[count:]:
            (output / "images" / entry["file"]).unlink(missing_ok=True)
        finished = finished[:count]
    rejected: Counter = Counter()
    too_small = 0
    for entry in finished:
        rejected.update(entry.get("rejected", {}))
        too_small += entry.get("tooSmall", 0)

    def consume(results: Iterable[dict[str, Any]], manifest: Any, labels: Any) -> None:
        nonlocal too_small
        for done, result in enumerate(results, start=len(finished) + 1):
            atomic_write_bytes(output / "images" / image_name(result["index"]), result["data"])
            labels.write(json.dumps({"index": result["index"], "annotations": result["annotations"]}, ensure_ascii=False, sort_keys=True) + "\n")
            manifest.write(json.dumps({
                "index": result["index"], "file": image_name(result["index"]), "seed": config["seed"], "attempt": result["attempt"],
                "plan": result["plan"], "rejected": result["rejected"], "tooSmall": result["tooSmall"],
            }, ensure_ascii=False, sort_keys=True) + "\n")
            labels.flush()
            manifest.flush()
            rejected.update(result["rejected"])
            too_small += result["tooSmall"]
            if done % 5 == 0 or done == count:
                progress(done, count)

    jobs = [(index, 0) for index in range(len(finished), count)]
    worker_count = processes if processes is not None else max(1, min((os.cpu_count() or 2) - 1, 8))
    face = str(face_model) if face_model else None
    if jobs:
        with (output / "manifest.jsonl").open("a", encoding="utf-8") as manifest, (output / "labels.jsonl").open("a", encoding="utf-8") as labels:
            if worker_count <= 1:
                _init_worker(str(work), config, face)
                consume((_make_in_worker(job) for job in jobs), manifest, labels)
            else:
                # imap keeps the order, so the files are the same however many processes there are.
                with multiprocessing.get_context("spawn").Pool(worker_count, _init_worker, (str(work), config, face)) as pool:
                    consume(pool.imap(_make_in_worker, jobs, chunksize=2), manifest, labels)

    stage(2, 2, text(language, "Datensatz prüfen", "Validate dataset", "Validar el conjunto de datos", "Valider le jeu de données"))
    problems, bad = validate_output(output, context)
    found_first = Counter(problems)
    repairs = 0
    while bad and repairs < 3:
        repairs += 1
        replaced: dict[int, dict[str, Any]] = {}
        for index in bad:
            replaced[index] = make_image(_context_with(context, face), index, 1000 * repairs)
        rewrite_lines(output, replaced, config)
        problems, bad = validate_output(output, context)
    write_coco(output, context, count)
    if config.get("yolo"):
        write_yolo(output, context)
    split = write_split(output, context)
    write_previews(output, context, count)
    report = {
        "images": count, "violations": dict(problems), "foundThenRegenerated": dict(found_first), "rejectedWhileGenerating": dict(rejected),
        "labelsTooSmall": too_small, "real": {name: len(items) for name, items in split.items()},
        "splitWarning": context.analysis.get("splitWarning"), "sessionsUsed": sorted(usable_groups(context)),
    }
    atomic_json(output / "report.json", report)
    return report


def _context_with(context: Context, face: str | None) -> Context:
    if face and context.anonymizer is None:
        import anonymize
        context.anonymizer = anonymize.Anonymizer(Path(face))
    return context


def rewrite_lines(output: Path, replaced: dict[int, dict[str, Any]], config: dict[str, Any]) -> None:
    for name, build in (
        ("labels.jsonl", lambda result: {"index": result["index"], "annotations": result["annotations"]}),
        ("manifest.jsonl", lambda result: {"index": result["index"], "file": image_name(result["index"]), "seed": config["seed"], "attempt": result["attempt"], "plan": result["plan"], "rejected": result["rejected"], "tooSmall": result["tooSmall"]}),
    ):
        path = output / name
        temporary = path.with_name(name + ".part")
        with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for index, line in enumerate(source):
                target.write(json.dumps(build(replaced[index]), ensure_ascii=False, sort_keys=True) + "\n" if index in replaced else line)
        os.replace(temporary, path)
    for index, result in replaced.items():
        atomic_write_bytes(output / "images" / image_name(index), result["data"])


# --- training ------------------------------------------------------------------------------

def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_training_dataset(work: Path, dataset_root: Path, real_weight: int = 3, use_synthetic: bool = True, categories: list[str] | None = None, normalize: Callable[[str], str | None] | None = None) -> dict[str, int]:
    """dataset/train, valid and test for RF-DETR: real images by session, real training images
    repeated real_weight times, synthetic images in training only. With use_synthetic off the
    same real split trains alone: that is the comparison run for the benchmark."""
    output = work / "output"
    split = json.loads((output / "split.json").read_text(encoding="utf-8"))
    source = Path(split["source"])
    images = {image["rel"]: image for image in load_source(source)[0]}
    analysis = json.loads((work / "analysis.json").read_text(encoding="utf-8"))
    roles = analysis["roles"]
    normalize = normalize or (lambda name: name)
    names: list[str] = []
    for name in analysis["classes"]:
        mapped = normalize(name) if roles.get(name) in {"player", "referee", "ball", "static"} else None
        if mapped and mapped not in names and (categories is None or mapped in categories):
            names.append(mapped)
    class_id = {name: index + 1 for index, name in enumerate(names)}
    counts: dict[str, int] = {}
    for split_name in ("train", "valid", "test"):
        folder = dataset_root / split_name
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.iterdir():
            if stale.is_file():
                stale.unlink()
        out_images, out_annotations = [], []
        repeats = max(1, int(real_weight)) if split_name == "train" else 1
        for relative in split["real"][split_name]:
            image = images.get(relative)
            if image is None:
                continue
            for repeat in range(repeats):
                file_name = f"real{repeat}-{stable_hash(relative)[:12]}{Path(relative).suffix.lower()}"
                _link(source / relative, folder / file_name)
                image_id = len(out_images) + 1
                out_images.append({"id": image_id, "file_name": file_name, "width": image["width"], "height": image["height"]})
                for item in image["instances"]:
                    mapped = normalize(item["class"]) if roles.get(item["class"]) in {"player", "referee", "ball", "static"} else None
                    if mapped in class_id:
                        x, y, w, h = item["bbox"]
                        out_annotations.append({"id": len(out_annotations) + 1, "image_id": image_id, "category_id": class_id[mapped], "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        if split_name == "train" and use_synthetic:
            coco = json.loads((output / "annotations.coco.json").read_text(encoding="utf-8"))
            old_names = {entry["id"]: entry["name"] for entry in coco["categories"]}
            offset = len(out_images)
            for entry in coco["images"]:
                _link(output / "images" / entry["file_name"], folder / entry["file_name"])
                out_images.append({**entry, "id": entry["id"] + offset})
            for annotation in coco["annotations"]:
                mapped = normalize(old_names[annotation["category_id"]])
                if mapped in class_id:
                    out_annotations.append({"id": len(out_annotations) + 1, "image_id": annotation["image_id"] + offset, "category_id": class_id[mapped], "bbox": annotation["bbox"], "area": annotation["bbox"][2] * annotation["bbox"][3], "iscrowd": 0})
        atomic_json(folder / "_annotations.coco.json", {
            "info": {"description": "Reco Trainer: real images and Augmentation Studio"}, "licenses": [], "images": out_images, "annotations": out_annotations,
            "categories": [{"id": number, "name": name, "supercategory": "sport"} for name, number in class_id.items()],
        })
        counts[split_name] = len(out_images)
    return counts
