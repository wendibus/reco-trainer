"""Auto-labeling helpers: hardware tiers and the "teacher" models behind auto-label.

Standard library only at module level, so local_worker.py can import the tier logic without
the ML environment; torch/transformers are imported inside the functions that need them.

  * choose_tier()       - strong / medium / weak from the device and its memory
  * TIER_PLANS          - which models each tier uses
  * Sam2Refiner         - tightens boxes with SAM 2 (Apache-2.0), box prompt -> mask -> box
  * Florence2Detector   - a weak second teacher (MIT) with the RF-DETR predict() interface
  * interpolate_ball()  - fills a missing ball between two nearby detections of one video
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIERS = ("strong", "medium", "weak")

# Models per tier (ids from model_registry.json). The teacher detects people and the ball
# when no trained model knows a class yet; SAM 2 tightens boxes; Florence-2 is the second
# opinion for "models disagree"; the VLM check runs by default on strong machines, can be
# switched on for medium ones and is not offered on weak (CPU-only) ones.
TIER_PLANS: dict[str, dict[str, Any]] = {
    "strong": {"teacher": "rf-detr-large", "refiner": "sam2.1-hiera-base-plus", "second": "florence-2-large", "vlm": True},
    "medium": {"teacher": "rf-detr-medium", "refiner": "sam2.1-hiera-small", "second": "florence-2-base", "vlm": "optional"},
    "weak": {"teacher": "rf-detr-small", "refiner": None, "second": "florence-2-base", "vlm": False},
}
TEACHER_SIZES = {"rf-detr-nano": "nano", "rf-detr-small": "small", "rf-detr-medium": "medium", "rf-detr-large": "large"}


def choose_tier(device: str, vram_gb: float | None = None, memory_gb: float | None = None) -> str:
    """CUDA by VRAM (16 GB is the comfortable target), Apple silicon by unified memory,
    everything on the CPU is "weak" (no SAM 2, no VLM - the export still works)."""
    if device == "cuda":
        vram = vram_gb or 0.0
        return "strong" if vram >= 14 else "medium" if vram >= 8 else "weak"
    if device == "mps":
        memory = memory_gb or 0.0
        return "strong" if memory >= 32 else "medium" if memory >= 16 else "weak"
    return "weak"


def tier_plan(tier: str | None) -> dict[str, Any]:
    return TIER_PLANS.get(tier or "", TIER_PLANS["weak"])


# --- SAM 2 box refinement --------------------------------------------------------------------

class Sam2Refiner:
    """Box prompt -> SAM 2 mask -> the mask's bounding box. The caller decides with
    plausible_refined_box() whether to accept it."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.device = device
        self.model = Sam2Model.from_pretrained(str(model_dir)).to(device).eval()
        self.processor = Sam2Processor.from_pretrained(str(model_dir))

    def refine(self, image: Any, boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float] | None]:
        """image: PIL RGB; boxes: (x, y, w, h) in its pixels."""
        if not boxes:
            return []
        import numpy as np
        prompts = [[[x, y, x + w, y + h] for x, y, w, h in boxes]]
        inputs = self.processor(images=image, input_boxes=prompts, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(output.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        result: list[tuple[float, float, float, float] | None] = []
        for mask in masks:
            array = mask[0].numpy() if mask.ndim == 3 else mask.numpy()
            ys, xs = np.nonzero(array)
            if len(xs) == 0:
                result.append(None)
                continue
            result.append((float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)))
        return result


# --- Florence-2 as a second teacher ----------------------------------------------------------

PERSON_WORDS = {"person", "people", "man", "woman", "boy", "girl", "player", "referee", "athlete", "human", "child", "men", "women"}
BALL_WORDS = {"ball", "sports ball", "basketball", "football", "soccer ball", "volleyball", "handball", "rugby ball", "puck"}


def florence_label(label: str) -> str:
    """Map Florence-2's free-text labels onto the COCO names RF-DETR uses ("person", "sports ball")."""
    text = label.strip().lower()
    if text in PERSON_WORDS:
        return "person"
    if text in BALL_WORDS or text.endswith(" ball"):
        return "sports ball"
    return text


@dataclass
class Detections:
    """Same shape as RF-DETR's detections (xyxy, confidence, class_id, data["class_name"])."""
    xyxy: Any
    confidence: Any
    class_id: Any
    data: dict[str, Any]


class Florence2Detector:
    """Florence-2 <OD> object detection behind RF-DETR's predict(paths, threshold) interface.
    It has no scores; every box counts as confidence 1.0."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.torch = torch
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype
        self.model = Florence2ForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_dir))

    def predict(self, paths: list[str], threshold: float = 0.0) -> list[Detections]:
        import numpy as np
        from PIL import Image
        results = []
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            inputs = self.processor(text="<OD>", images=image, return_tensors="pt").to(self.device, self.dtype)
            with self.torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
            text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(text, task="<OD>", image_size=image.size).get("<OD>", {})
            boxes = np.asarray(parsed.get("bboxes") or [], dtype=float).reshape(-1, 4)
            names = [florence_label(str(label)) for label in parsed.get("labels") or []]
            results.append(Detections(boxes, np.ones(len(boxes)), np.zeros(len(boxes), dtype=int), {"class_name": names}))
        return results


# --- Ball interpolation ----------------------------------------------------------------------

BALL_CATEGORIES = ("ball", "puck")


def interpolate_ball(frames: list[dict[str, Any]], max_gap_seconds: float = 1.0, max_jump_ratio: float = 0.25) -> int:
    """Within each video, a frame without a ball that lies between two frames with exactly one
    ball each (at most max_gap_seconds apart) gets an interpolated ball box as an automatic,
    unreviewed suggestion. Skipped when the ball moved further than max_jump_ratio of the frame
    width between the two - then a straight line is a guess, not an interpolation.
    Frames prepared from long videos are usually seconds apart, so this mostly helps densely
    sampled clips. Returns the number of boxes added."""
    by_video: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        if frame.get("heldOut"):
            continue
        by_video.setdefault(str(frame.get("videoID")), []).append(frame)

    def single_ball(frame: dict[str, Any]) -> dict[str, Any] | None:
        balls = [box for box in frame.get("annotations", []) if box.get("category") in BALL_CATEGORIES]
        return balls[0] if len(balls) == 1 else None

    added = 0
    for items in by_video.values():
        items.sort(key=lambda frame: float(frame.get("timestamp") or 0.0))
        anchors = [(index, single_ball(frame)) for index, frame in enumerate(items)]
        anchors = [(index, ball) for index, ball in anchors if ball is not None]
        for (start, first), (end, second) in zip(anchors, anchors[1:]):
            if end - start < 2 or first["category"] != second["category"]:
                continue
            t0, t1 = float(items[start].get("timestamp") or 0.0), float(items[end].get("timestamp") or 0.0)
            width = float(items[start].get("width") or 1.0)
            if t1 - t0 <= 0 or t1 - t0 > max_gap_seconds:
                continue
            c0 = (first["x"] + first["width"] / 2, first["y"] + first["height"] / 2)
            c1 = (second["x"] + second["width"] / 2, second["y"] + second["height"] / 2)
            if ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5 > max_jump_ratio * width:
                continue
            for frame in items[start + 1:end]:
                if any(box.get("category") in BALL_CATEGORIES for box in frame.get("annotations", [])):
                    continue
                share = (float(frame.get("timestamp") or 0.0) - t0) / (t1 - t0)
                w = first["width"] + (second["width"] - first["width"]) * share
                h = first["height"] + (second["height"] - first["height"]) * share
                cx, cy = c0[0] + (c1[0] - c0[0]) * share, c0[1] + (c1[1] - c0[1]) * share
                frame.setdefault("annotations", []).append({
                    "id": str(uuid.uuid4()), "category": first["category"],
                    "x": cx - w / 2, "y": cy - h / 2, "width": w, "height": h,
                    "confidence": 0.3, "source": "auto", "interpolated": True,
                })
                added += 1
    return added
