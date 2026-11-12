"""Auto-labeling helpers: hardware tiers and the "teacher" models behind auto-label.

Standard library only at module level, so local_worker.py can import the tier logic without
the ML environment; torch/transformers are imported inside the functions that need them.

  * choose_tier()       - strong / medium / weak from the device and its memory
  * TIER_PLANS          - which models each tier uses
  * Sam2Refiner         - tightens boxes with SAM 2 (Apache-2.0), box prompt -> mask -> box
  * Florence2Detector   - a weak second teacher (MIT) with the RF-DETR predict() interface
  * interpolate_ball()  - fills a missing ball between two nearby detections of one video
  * ball_gaps()         - frames of a densely sampled clip where the ball was missed between two detections
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
    everything on the CPU is "weak" (no SAM 2, no VLM - the export still works).

    "directml" (AMD or Intel graphics on Windows): detection and the VLM run on the GPU, but
    training, SAM 2 and Florence-2 stay on the CPU, so at most "medium" - from 8 GB of graphics
    memory; integrated graphics with little memory of their own stay "weak"."""
    if device == "cuda":
        vram = vram_gb or 0.0
        return "strong" if vram >= 14 else "medium" if vram >= 8 else "weak"
    if device == "directml":
        return "medium" if (vram_gb or 0.0) >= 8 else "weak"
    if device == "mps":
        memory = memory_gb or 0.0
        return "strong" if memory >= 32 else "medium" if memory >= 16 else "weak"
    return "weak"


# Display adapters under Windows (the class key of all graphics drivers).
DISPLAY_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_graphics() -> list[dict[str, Any]]:
    """[{"name", "vendor", "vramGB"}] of the graphics cards, from the registry (no PyTorch needed).

    HardwareInformation.qwMemorySize is the 64-bit dedicated memory; WMI's AdapterRAM stops at 4 GB.
    Microsoft's own adapters (Basic Display, Remote Display) are left out."""
    try:
        import winreg
    except ImportError:
        return []
    found: list[dict[str, Any]] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DISPLAY_CLASS_KEY)
    except OSError:
        return []
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
            try:
                with winreg.OpenKey(root, name) as key:
                    values: dict[str, Any] = {}
                    for value in ("DriverDesc", "ProviderName", "HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            values[value] = winreg.QueryValueEx(key, value)[0]
                        except OSError:
                            pass
            except OSError:
                continue
            found_card = windows_graphics_entry(values)
            if found_card is not None:
                found.append(found_card)
    return found


def windows_graphics_entry(values: dict[str, Any]) -> dict[str, Any] | None:
    description = str(values.get("DriverDesc") or "").strip()
    provider = str(values.get("ProviderName") or "").strip()
    if not description or provider.lower() == "microsoft":
        return None
    memory = values.get("HardwareInformation.qwMemorySize", values.get("HardwareInformation.MemorySize"))
    if isinstance(memory, (bytes, bytearray)):
        memory = int.from_bytes(bytes(memory)[:8], "little")
    try:
        vram_gb = round(int(memory) / 1024**3, 1)
    except (TypeError, ValueError):
        vram_gb = 0.0
    lowered = f"{description} {provider}".lower()
    vendor = "nvidia" if "nvidia" in lowered else "amd" if ("amd" in lowered or "radeon" in lowered or "advanced micro" in lowered) else "intel" if "intel" in lowered else "other"
    return {"name": description, "vendor": vendor, "vramGB": vram_gb}


def directml_graphics() -> dict[str, Any] | None:
    """The AMD or Intel card with the most memory of its own, or None (no card, or NVIDIA present -
    an NVIDIA GPU uses CUDA instead)."""
    cards = windows_graphics()
    if any(card["vendor"] == "nvidia" for card in cards):
        return None
    candidates = [card for card in cards if card["vendor"] in {"amd", "intel"}]
    return max(candidates, key=lambda card: card["vramGB"]) if candidates else None


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


def ball_gaps(
    frames: list[dict[str, Any]],
    max_gap_seconds: float = 1.5,
    max_jump_ratio: float = 0.25,
    per_gap: int = 3,
    limit: int = 60,
) -> list[dict[str, Any]]:
    """Frames of one densely sampled clip (the tracking simulation) where the detector missed
    the ball between two frames in which it found it. These are the model's own failures, and
    the two detections around them say roughly where the ball must be.

    frames: [{"file", "timestamp", "width", "ball": {"x", "y"} | None}, ...] in playback order.
    Returns at most `limit` picks, at most `per_gap` evenly spread over each gap (neighbouring
    frames of a dense clip are near-duplicates): {"index", "file", "x", "y", "radius"} with the
    straight-line position and how far around it to search. A gap is skipped when it lasts
    longer than max_gap_seconds or the ball moved further than max_jump_ratio of the frame
    width - then a cut, a wrong detection or a ball out of view is more likely than a miss."""
    anchors = [index for index, frame in enumerate(frames) if frame.get("ball")]
    picks: list[dict[str, Any]] = []
    for start, end in zip(anchors, anchors[1:]):
        missing = end - start - 1
        if missing < 1:
            continue
        first, second = frames[start], frames[end]
        t0, t1 = float(first.get("timestamp") or 0.0), float(second.get("timestamp") or 0.0)
        if t1 - t0 <= 0 or t1 - t0 > max_gap_seconds:
            continue
        width = float(first.get("width") or 0.0)
        if width <= 0:
            continue
        x0, y0 = float(first["ball"]["x"]), float(first["ball"]["y"])
        x1, y1 = float(second["ball"]["x"]), float(second["ball"]["y"])
        jump = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if jump > max_jump_ratio * width:
            continue
        count = min(max(per_gap, 1), missing)
        chosen = sorted({start + round((step + 1) * (missing + 1) / (count + 1)) for step in range(count)})
        for index in chosen:
            if not start < index < end:
                continue
            share = (float(frames[index].get("timestamp") or 0.0) - t0) / (t1 - t0)
            share = min(max(share, 0.0), 1.0)
            picks.append({
                "index": index, "file": str(frames[index].get("file", "")),
                "x": x0 + (x1 - x0) * share, "y": y0 + (y1 - y0) * share,
                # A thrown ball leaves the straight line, so search well beyond it.
                "radius": max(jump / 2, 0.08 * width),
            })
    if len(picks) > limit > 0:
        stride = len(picks) / limit
        picks = [picks[int(position * stride)] for position in range(limit)]
    return picks


SCENE_KINDS = ("gap", "jump", "weak", "lost")


def tracking_scenes(
    frames: list[dict[str, Any]],
    max_gap_seconds: float = 1.5,
    max_jump_ratio: float = 0.25,
    jump_base_ratio: float = 0.12,
    jump_ratio_per_second: float = 0.6,
    weak_confidence: float = 0.4,
    min_weak_frames: int = 3,
    per_scene: int = 3,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Stretches of one densely sampled clip (the tracking simulation) that are worth a look,
    because the detector most likely got something wrong there:

    - "gap":  the ball is missing between two detections that lie close together in time and
              place - a miss, and the two detections say roughly where the ball must be.
    - "jump": two detections lie further apart than a ball travels in the time between them
              (jump_base_ratio of the frame width plus jump_ratio_per_second for every second) -
              at least one of them is something else (a head, a lamp, a second ball).
    - "weak": the ball is found several frames in a row, but only with low confidence.
    - "lost": no ball for longer than max_gap_seconds - out of view, or missed for long.

    frames: [{"file", "timestamp", "width", "ball": {"x", "y", "confidence", ...} | None}, ...]
    in playback order. Returns at most `limit` scenes in playback order (when there are more,
    gaps and jumps are kept first):
    {"id", "kind", "start", "end", "startTime", "endTime", "picks": [...]} with frame indices
    `start`..`end` inclusive. `picks` are at most `per_scene` frames to put before the user:
    {"index", "file"} plus "x", "y", "radius" (where to search again) for a gap, or "ball" (what
    the detector reported) for a jump or a weak stretch."""
    def time_of(index: int) -> float:
        return float(frames[index].get("timestamp") or 0.0)

    def spread(first: int, last: int) -> list[int]:
        count = min(max(per_scene, 1), last - first + 1)
        if count <= 0:
            return []
        return sorted({first + round((step + 1) * (last - first + 2) / (count + 1)) - 1 for step in range(count)})

    def scene(kind: str, start: int, end: int, picks: list[dict[str, Any]]) -> dict[str, Any]:
        return {"id": f"{kind}-{start}", "kind": kind, "start": start, "end": end,
                "startTime": time_of(start), "endTime": time_of(end), "picks": picks}

    def plain(indices: list[int]) -> list[dict[str, Any]]:
        return [{"index": index, "file": str(frames[index].get("file", ""))} for index in indices]

    def reported(indices: list[int]) -> list[dict[str, Any]]:
        return [{"index": index, "file": str(frames[index].get("file", "")), "ball": frames[index]["ball"]} for index in indices]

    scenes: list[dict[str, Any]] = []
    anchors = [index for index, frame in enumerate(frames) if frame.get("ball")]
    if not frames:
        return scenes

    # no ball at all before the first or after the last detection
    edges = [(0, (anchors[0] if anchors else len(frames)) - 1)]
    if anchors:
        edges.append((anchors[-1] + 1, len(frames) - 1))
    for first, last in edges:
        if last >= first and time_of(last) - time_of(first) >= max_gap_seconds:
            scenes.append(scene("lost", first, last, plain(spread(first, last))))

    jumps: list[list[int]] = []
    for start, end in zip(anchors, anchors[1:]):
        first, second = frames[start], frames[end]
        elapsed = time_of(end) - time_of(start)
        width = float(first.get("width") or 0.0)
        if elapsed <= 0 or width <= 0:
            continue
        x0, y0 = float(first["ball"]["x"]), float(first["ball"]["y"])
        x1, y1 = float(second["ball"]["x"]), float(second["ball"]["y"])
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        missing = end - start - 1
        if elapsed > max_gap_seconds:
            if missing >= 1:
                scenes.append(scene("lost", start + 1, end - 1, plain(spread(start + 1, end - 1))))
        elif distance > (jump_base_ratio + jump_ratio_per_second * elapsed) * width:
            # Further than a ball travels in that time. Neighbouring jumps (there and back, or
            # several within half a second) are one scene around the odd detections.
            if jumps and time_of(start) - time_of(jumps[-1][1]) <= 0.5:
                jumps[-1][1] = end
            else:
                jumps.append([start, end])
        elif missing >= 1 and distance <= max_jump_ratio * width:
            picks = []
            for index in spread(start + 1, end - 1):
                share = min(max((time_of(index) - time_of(start)) / elapsed, 0.0), 1.0)
                picks.append({
                    "index": index, "file": str(frames[index].get("file", "")),
                    "x": x0 + (x1 - x0) * share, "y": y0 + (y1 - y0) * share,
                    # A thrown ball leaves the straight line, so search well beyond it.
                    "radius": max(distance / 2, 0.08 * width),
                })
            scenes.append(scene("gap", start + 1, end - 1, picks))
    for start, end in jumps:
        involved = [index for index in anchors if start <= index <= end]
        if len(involved) > max(per_scene, 1):
            stride = (len(involved) - 1) / (max(per_scene, 1) - 1) if per_scene > 1 else 0
            involved = sorted({involved[round(position * stride)] for position in range(max(per_scene, 1))})
        scenes.append(scene("jump", start, end, reported(involved)))

    run: list[int] = []
    for index in [*range(len(frames)), None]:  # type: ignore[list-item]
        ball = frames[index].get("ball") if index is not None else None
        if ball and float(ball.get("confidence") or 0.0) < weak_confidence:
            run.append(index)
            continue
        if len(run) >= max(min_weak_frames, 1):
            scenes.append(scene("weak", run[0], run[-1], reported(spread(run[0], run[-1]))))
        run = []

    if len(scenes) > limit > 0:
        rank = {kind: position for position, kind in enumerate(SCENE_KINDS)}
        scenes = sorted(scenes, key=lambda item: (rank[item["kind"]], item["start"]))[:limit]
    return sorted(scenes, key=lambda item: (item["start"], SCENE_KINDS.index(item["kind"])))


def scene_limit(seconds: float, per_minute: int = 40, minimum: int = 40) -> int:
    """How many scenes a clip of this length may list: 40 for the first minute, 40 more for every
    further minute (a longer section has proportionally more to show; a list longer than that
    is not worked through anyway)."""
    return max(minimum, int(-(-max(seconds, 0.0) * per_minute // 60)))


def tracking_scenes_with_total(frames: list[dict[str, Any]], **options: Any) -> tuple[list[dict[str, Any]], int]:
    """tracking_scenes for a clip, with the limit growing with its length, and the number of
    scenes there were before the limit cut some (gaps and jumps are kept first)."""
    times = [float(frame.get("timestamp") or 0.0) for frame in frames]
    seconds = (max(times) - min(times)) if times else 0.0
    everything = tracking_scenes(frames, **{**options, "limit": 0})
    return tracking_scenes(frames, **{**options, "limit": scene_limit(seconds)}), len(everything)
