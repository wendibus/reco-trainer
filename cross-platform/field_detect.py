"""Light field detection: finds only the outline of the playing floor, nothing else.

No lines, no lens model, no sport: the floor is the large, fairly uniform surface in the
lower part of the frame. The frame is shrunk, its colours are grouped (k-means), the group
that covers the most floor-like area is kept, gaps made by court lines are closed, and the
outline of the biggest connected area is simplified to a few points. Areas the user hides
("ignore") are left out of the colour grouping and of the result. numpy and Pillow only."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

WORK_WIDTH = 256
CLUSTERS = 6
MIN_AREA = 0.06          # smaller than this share of the frame is not a field
MAX_POINTS = 10
MERGE_DISTANCE = 22.0
CLOSE_RADIUS = 3         # bridges court lines at WORK_WIDTH


class FieldNotFound(RuntimeError):
    pass


def _shift_any(mask: np.ndarray, radius: int, grow: bool) -> np.ndarray:
    work = mask if grow else ~mask
    padded = np.pad(work, radius, constant_values=False)
    result = np.zeros_like(work)
    height, width = work.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result |= padded[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
    return result if grow else ~result


def _largest_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[int] = []
    flat = mask.ravel()
    for start in np.flatnonzero(flat):
        if seen.ravel()[start]:
            continue
        stack, members = [int(start)], []
        seen.ravel()[start] = True
        while stack:
            index = stack.pop()
            members.append(index)
            y, x = divmod(index, width)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width:
                    neighbour = ny * width + nx
                    if flat[neighbour] and not seen.ravel()[neighbour]:
                        seen.ravel()[neighbour] = True
                        stack.append(neighbour)
        if len(members) > len(best):
            best = members
    result = np.zeros(height * width, dtype=bool)
    result[best] = True
    return result.reshape(height, width)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    outside = np.zeros_like(mask)
    stack = [(y, x) for y in range(height) for x in (0, width - 1)] + [(y, x) for x in range(width) for y in (0, height - 1)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < height and 0 <= x < width and not mask[y, x] and not outside[y, x]:
            outside[y, x] = True
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return ~outside


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _simplify(hull: np.ndarray, limit: int) -> np.ndarray:
    """Drop the hull corner whose removal loses the least area until at most limit remain."""
    points = [tuple(p) for p in hull]
    while len(points) > max(4, limit):
        best, best_loss = 0, math.inf
        for index in range(len(points)):
            a, b, c = points[index - 1], points[index], points[(index + 1) % len(points)]
            loss = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
            if loss < best_loss:
                best, best_loss = index, loss
        points.pop(best)
    return np.array(points, dtype=float)


def _kmeans(samples: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(samples))
    centres = samples[rng.choice(len(samples), k, replace=False)].astype(float)
    labels = np.zeros(len(samples), dtype=int)
    for _ in range(12):
        distance = ((samples[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = distance.argmin(axis=1)
        for index in range(k):
            members = samples[labels == index]
            if len(members):
                centres[index] = members.mean(axis=0)
    return centres, labels


def detect_field(image_path: Path, ignore: list[list[float]] | None = None) -> dict[str, Any]:
    """Outline of the floor as fractional corners, plus the share of the frame it covers.
    ignore: rectangles [x0, y0, x1, y1] (fractions) that are hidden from the detection."""
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        width = WORK_WIDTH
        height = max(16, round(source.height * width / max(1, source.width)))
        small = np.asarray(source.resize((width, height), Image.BILINEAR), dtype=np.float32)
    hidden = np.zeros((height, width), dtype=bool)
    for rect in ignore or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in rect)
        except (TypeError, ValueError):
            continue
        x0, x1 = sorted((min(1, max(0, x0)), min(1, max(0, x1))))
        y0, y1 = sorted((min(1, max(0, y0)), min(1, max(0, y1))))
        hidden[int(y0 * height):math.ceil(y1 * height), int(x0 * width):math.ceil(x1 * width)] = True
    usable = ~hidden
    if usable.sum() < 0.1 * usable.size:
        raise FieldNotFound("Too little of the frame is left after hiding areas.")

    # Chroma separates floor from walls and lines better than brightness, which lighting changes.
    luma = small.mean(axis=2, keepdims=True)
    features = np.concatenate([small - luma, luma * 0.5], axis=2)
    samples = features[usable]
    rng = np.random.default_rng(7)
    centres, labels = _kmeans(samples[::2] if len(samples) > 20000 else samples, CLUSTERS, rng)
    distance = ((features.reshape(-1, 1, features.shape[2]) - centres[None, :, :]) ** 2).sum(axis=2)
    label_map = distance.argmin(axis=1).reshape(height, width)

    best_mask, best_score = None, 0.0
    for index in range(len(centres)):
        # Lighting often splits one floor into several close colours: join those.
        close = [other for other in range(len(centres)) if np.linalg.norm(centres[other] - centres[index]) < MERGE_DISTANCE]
        mask = np.isin(label_map, close) & usable
        area = mask.mean()
        if area < MIN_AREA:
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        lower = mask[height // 3:].mean()
        score = 0.4 * area + 0.6 * lower if rows.size else 0.0
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None:
        raise FieldNotFound("No large, even surface found.")

    closed = _shift_any(_shift_any(best_mask, CLOSE_RADIUS, True), CLOSE_RADIUS, False) & usable
    component = _fill_holes(_largest_component(closed)) & usable
    coverage = float(component.mean())
    if coverage < MIN_AREA:
        raise FieldNotFound("The surface found is too small to be the field.")
    ys_idx, xs_idx = np.nonzero(component)
    cloud = np.stack([xs_idx + 0.5, ys_idx + 0.5], axis=1)
    hull = _simplify(_convex_hull(cloud), MAX_POINTS)
    corners = [[round(float(min(1, max(0, x / width))), 4), round(float(min(1, max(0, y / height))), 4)] for x, y in hull]
    return {"corners": corners, "coverage": round(coverage, 3)}
