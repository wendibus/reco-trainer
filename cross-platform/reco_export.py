"""Export a library model as an ONNX file the Reco camera app can run.

Reco (the stitcher/autocam) reads every detector as a YOLO end-to-end export:
- one input ``[1, 3, S, S]``: RGB, letterboxed with grey 114 and centred, divided by 255;
- exactly one output ``[1, N, 6]`` = x1, y1, x2, y2 in input pixels, confidence, class id
  (with more than one output Reco switches to a different decoder and finds nothing);
- the class names in the ``names`` metadata as a Python dict string, found by name:
  ``ball`` (or ``sports ball``) and ``person``.

RF-DETR models (also combined ones) are wrapped in a small module that undoes the
letterbox, normalizes like RF-DETR, runs every member and writes that layout. YOLO
models trained in Reco Trainer already have it; only the classes are filtered and renamed.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

# Reco looks for players as "person" (COCO name); everything else keeps its Reco Trainer name.
RECO_NAMES = {"player": "person"}
MAX_DETECTIONS = 300
GREY = 114.0 / 255.0


def reco_name(category: str) -> str:
    return RECO_NAMES.get(category, category)


def names_metadata(categories: list[str]) -> str:
    """Ultralytics-style names, the only form Reco parses: {0: 'ball', 1: 'person'}."""
    return "{" + ", ".join(f"{index}: '{reco_name(category)}'" for index, category in enumerate(categories)) + "}"


def letterbox_geometry(width: int, height: int, size: int) -> tuple[float, float, float, int, int]:
    """Same arithmetic as Reco's preprocess: (scale, pad_x, pad_y, content width, content height)."""
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    return scale, (size - new_w) / 2.0, (size - new_h) / 2.0, new_w, new_h


def content_rows(aspect: float, size: int) -> tuple[int, int]:
    """Rows of the letterboxed input that hold the picture for a camera of this width/height ratio."""
    if aspect <= 1.0:
        return 0, size
    _, _, pad_y, _, new_h = letterbox_geometry(1000 * aspect, 1000, size)
    return int(pad_y), int(new_h)


def input_size_for(resolutions: list[int]) -> int:
    """Square input Reco letterboxes into. Larger than the model so the picture keeps its detail
    before the wrapper resizes it to the model's own resolution."""
    return 960 if max(resolutions) <= 640 else 1280


def project_aspect(project_root: Path, frames: list[dict[str, Any]], image_size) -> float:
    """Width/height of the project's images (median of up to 40), the shape the model was trained on."""
    ratios = []
    for frame in frames:
        relative = frame.get("relativePath")
        if not relative:
            continue
        size = image_size(project_root / str(relative))
        if size and size[1]:
            ratios.append(size[0] / size[1])
        if len(ratios) >= 40:
            break
    return float(statistics.median(ratios)) if ratios else 16 / 9


def letterbox_image(image: Any, size: int):
    """A PIL image -> the float32 [1, 3, S, S] tensor Reco feeds the model, plus the geometry."""
    import numpy as np
    width, height = image.size
    scale, pad_x, pad_y, new_w, new_h = letterbox_geometry(width, height, size)
    canvas = np.full((size, size, 3), GREY, dtype=np.float32)
    resized = np.asarray(image.convert("RGB").resize((new_w, new_h)), dtype=np.float32) / 255.0
    canvas[int(pad_y):int(pad_y) + new_h, int(pad_x):int(pad_x) + new_w] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), (scale, pad_x, pad_y)


def write_metadata(path: Path, categories: list[str], size: int, extra: dict[str, str]) -> None:
    import onnx
    model = onnx.load(str(path))
    values = {
        "names": names_metadata(categories),
        "imgsz": f"[{size}, {size}]",
        "task": "detect",
        "end2end": "True",
        "stride": "32",
        "batch": "1",
        **extra,
    }
    kept = [prop for prop in model.metadata_props if prop.key not in values]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    for key, value in values.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.save(model, str(path))


def filter_yolo_onnx(source: Path, target: Path, class_names: list[str], categories: list[str]) -> int:
    """Copy a YOLO end-to-end ONNX ([1, N, 6]) and append nodes that keep only `categories`:
    their class ids become 0..k-1 in the chosen order, all other detections get confidence 0."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError(f"expected one output, found {len(graph.output)}")
    output = graph.output[0]
    dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    if len(dims) != 3 or dims[2] != 6:
        raise ValueError(f"expected an output [1, N, 6], found {dims}")
    raw = output.name + "_all_classes"
    for node in graph.node:
        node.output[:] = [raw if name == output.name else name for name in node.output]
        node.input[:] = [raw if name == output.name else name for name in node.input]
    table = np.full(len(class_names), -1.0, dtype=np.float32)
    for new_id, category in enumerate(categories):
        table[class_names.index(category)] = float(new_id)
    initializers = {
        "reco_starts_box": [0], "reco_ends_box": [4], "reco_starts_score": [4], "reco_ends_score": [5],
        "reco_starts_class": [5], "reco_ends_class": [6], "reco_axes": [2],
    }
    for name, value in initializers.items():
        graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name))
    graph.initializer.append(numpy_helper.from_array(table, "reco_class_table"))
    graph.initializer.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), "reco_zero"))
    graph.node.extend([
        helper.make_node("Slice", [raw, "reco_starts_box", "reco_ends_box", "reco_axes"], ["reco_boxes"]),
        helper.make_node("Slice", [raw, "reco_starts_score", "reco_ends_score", "reco_axes"], ["reco_scores"]),
        helper.make_node("Slice", [raw, "reco_starts_class", "reco_ends_class", "reco_axes"], ["reco_class_float"]),
        helper.make_node("Cast", ["reco_class_float"], ["reco_class_int"], to=onnx.TensorProto.INT64),
        helper.make_node("Gather", ["reco_class_table", "reco_class_int"], ["reco_new_class"], axis=0),
        helper.make_node("GreaterOrEqual", ["reco_new_class", "reco_zero"], ["reco_keep"]),
        helper.make_node("Cast", ["reco_keep"], ["reco_keep_float"], to=onnx.TensorProto.FLOAT),
        helper.make_node("Mul", ["reco_scores", "reco_keep_float"], ["reco_kept_scores"]),
        helper.make_node("Max", ["reco_new_class", "reco_zero"], ["reco_class_out"]),
        helper.make_node("Concat", ["reco_boxes", "reco_kept_scores", "reco_class_out"], [output.name], axis=2),
    ])
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    inputs = model.graph.input[0].type.tensor_type.shape.dim
    return int(inputs[2].dim_value or 640)


def rfdetr_wrapper(members: list[dict[str, Any]], size: int, aspect: float):
    """members: [{"net": export-mode RF-DETR module, "resolution": int, "means", "stds",
    "columns": logit columns, "ids": output class ids}] -> torch module for Reco's layout."""
    import torch
    import torch.nn.functional as F

    top, rows = content_rows(aspect, size)

    class RecoRfdetr(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nets = torch.nn.ModuleList([member["net"] for member in members])
            self.resolutions = [int(member["resolution"]) for member in members]
            for index, member in enumerate(members):
                self.register_buffer(f"mean_{index}", torch.tensor(member["means"], dtype=torch.float32).view(1, 3, 1, 1))
                self.register_buffer(f"std_{index}", torch.tensor(member["stds"], dtype=torch.float32).view(1, 3, 1, 1))
                self.register_buffer(f"columns_{index}", torch.tensor(member["columns"], dtype=torch.long))
                self.register_buffer(f"ids_{index}", torch.tensor(member["ids"], dtype=torch.float32))

        def forward(self, images):
            picture = images[:, :, top:top + rows, :]
            outputs = []
            for index, net in enumerate(self.nets):
                resolution = self.resolutions[index]
                # RF-DETR was trained on the whole picture squeezed to a square: do the same.
                x = F.interpolate(picture, size=(resolution, resolution), mode="bilinear", align_corners=False)
                x = (x - getattr(self, f"mean_{index}")) / getattr(self, f"std_{index}")
                result = net(x)
                boxes, logits = result[0], result[1]
                probabilities = logits.index_select(2, getattr(self, f"columns_{index}")).sigmoid()
                scores, which = probabilities.max(dim=2)
                classes = getattr(self, f"ids_{index}")[which]
                cx, cy, w, h = boxes.unbind(dim=2)
                x1 = ((cx - w / 2) * size).clamp(0, size)
                x2 = ((cx + w / 2) * size).clamp(0, size)
                y1 = (top + (cy - h / 2) * rows).clamp(top, top + rows)
                y2 = (top + (cy + h / 2) * rows).clamp(top, top + rows)
                outputs.append(torch.stack([x1, y1, x2, y2, scores, classes], dim=2))
            merged = torch.cat(outputs, dim=1)
            count = min(MAX_DETECTIONS, int(merged.shape[1]))
            best = merged[:, :, 4].topk(count, dim=1).indices
            return torch.gather(merged, 1, best.unsqueeze(2).expand(-1, -1, 6))

    return RecoRfdetr().eval()


def export_ready_rfdetr(model: Any):
    """A copy of an RFDETR model's network, switched to its ONNX-friendly forward at its own
    resolution (backbone position embeddings frozen, export forward like RF-DETR's exporter)."""
    import copy
    network = copy.deepcopy(model.model.model).cpu().eval()
    resolution = int(model.model.resolution)
    for module in network.modules():
        if type(module).__name__ == "DinoV2" and hasattr(module, "export"):
            module.shape = (resolution, resolution)
            module.export()
    if not getattr(network, "_export", False) and callable(getattr(network, "export", None)):
        network.export()
    return network, resolution


def export_onnx(module: Any, size: int, target: Path) -> None:
    import inspect
    import torch
    options: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    dummy = torch.full((1, 3, size, size), GREY, dtype=torch.float32)
    with torch.no_grad():
        module(dummy)
        torch.onnx.export(module, (dummy,), str(target), input_names=["images"], output_names=["output0"],
                          opset_version=17, do_constant_folding=True, export_params=True, **options)


def check_file(path: Path, samples: list[tuple[Any, list[tuple[float, float, float, float]]]], threshold: float = 0.25) -> dict[str, Any]:
    """Run the file like Reco does: one output [1, N, 6], square input, names readable. samples =
    [(PIL image, labelled boxes x, y, w, h)]: counts images where a detection's centre lies in a box."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.get_outputs()
    if len(outputs) != 1:
        raise ValueError(f"{len(outputs)} outputs instead of one")
    shape = session.get_inputs()[0].shape
    if shape[2] != shape[3]:
        raise ValueError(f"input is not square: {shape}")
    size = int(shape[2])
    names = session.get_modelmeta().custom_metadata_map.get("names", "")
    if not names.startswith("{"):
        raise ValueError("names metadata missing")
    found = 0
    for image, boxes in samples:
        tensor, (scale, pad_x, pad_y) = letterbox_image(image, size)
        data = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if data.ndim != 3 or data.shape[2] != 6:
            raise ValueError(f"output shape {data.shape} instead of [1, N, 6]")
        hits = data[0][data[0, :, 4] >= threshold]
        centres = [(((row[0] + row[2]) / 2 - pad_x) / scale, ((row[1] + row[3]) / 2 - pad_y) / scale) for row in hits]
        if any(x <= cx <= x + w and y <= cy <= y + h for cx, cy in centres for x, y, w, h in boxes):
            found += 1
    return {"inputSize": size, "names": names, "found": found, "samples": len(samples)}
