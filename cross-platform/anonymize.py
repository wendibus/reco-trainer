"""Anonymize sports frames for sharing or training while keeping them useful for training.

Faces and heads are blurred (or pixelated) inside soft-edged ellipses, never covered with
opaque boxes, so a head keeps its shape and colours and a model does not learn black
rectangles as part of a person. Optionally, text (jersey numbers, scoreboard, ads) is
blurred too. Ball/puck boxes are restored afterwards, labels are never changed.

Regions come from three sources:
  * the head region of every labelled person (player, goalkeeper, referee) - works for
    tiny and back-facing heads where a face detector finds nothing,
  * extra people found by a detector (spectators, bench), passed in by the caller,
  * faces found by YuNet (OpenCV, MIT) on the whole frame and in enlarged head crops.

After blurring, YuNet runs again; a face it still finds outside the blurred areas makes the
frame "unsafe" - the exporter then leaves it out and lists it for manual review.

Needs numpy and OpenCV (both are in the ML environment); imported lazily by ml_worker.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PERSON_CATEGORIES = {"player", "goalkeeper", "referee"}
PROTECTED_CATEGORIES = {"ball", "puck"}
METHODS = {"gaussian", "pixelate"}


@dataclass
class Region:
    x: float
    y: float
    width: float
    height: float
    kind: str  # "head" | "face" | "text"
    ellipse: bool = True

    def clipped(self, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
        x0, y0 = max(0, int(np.floor(self.x))), max(0, int(np.floor(self.y)))
        x1, y1 = min(image_width, int(np.ceil(self.x + self.width))), min(image_height, int(np.ceil(self.y + self.height)))
        return (x0, y0, x1, y1) if x1 - x0 >= 2 and y1 - y0 >= 2 else None


@dataclass
class Report:
    heads: int = 0
    faces: int = 0
    texts: int = 0
    residual_faces: list[list[float]] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.residual_faces

    def as_dict(self) -> dict[str, Any]:
        return {"heads": self.heads, "faces": self.faces, "texts": self.texts, "residualFaces": len(self.residual_faces), "safe": self.safe}


def head_region(box: tuple[float, float, float, float]) -> Region:
    """Estimated head of a person box (x, y, w, h): the top ~17 % for an upright person,
    a larger share for crouching/sitting people (wide boxes)."""
    x, y, width, height = box
    upright = width / max(height, 1.0) < 0.75
    head_height = height * 0.17 if upright else min(height * 0.32, width * 0.45)
    head_width = max(width * (0.55 if upright else 0.45), head_height * 0.9)
    center_x = x + width / 2
    top = y - height * 0.02
    return Region(center_x - head_width / 2, top, head_width, head_height * 1.12, "head")


def expand(region: Region, factor: float) -> Region:
    extra_w, extra_h = region.width * (factor - 1) / 2, region.height * (factor - 1) / 2
    return Region(region.x - extra_w, region.y - extra_h, region.width * factor, region.height * factor, region.kind, region.ellipse)


def overlap_share(inner: tuple[float, float, float, float], outer: Region) -> float:
    """Share of `inner` (x, y, w, h) that lies inside `outer`."""
    x, y, w, h = inner
    ix = max(0.0, min(x + w, outer.x + outer.width) - max(x, outer.x))
    iy = max(0.0, min(y + h, outer.y + outer.height) - max(y, outer.y))
    return (ix * iy) / max(w * h, 1e-6)


class Anonymizer:
    def __init__(
        self,
        face_model: Path | None,
        method: str = "gaussian",
        text_model: Path | None = None,
        face_threshold: float = 0.6,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.face_threshold = face_threshold
        self.face_detector = cv2.FaceDetectorYN.create(str(face_model), "", (320, 320), face_threshold, 0.3, 5000) if face_model else None
        self.text_detector = None
        if text_model:
            detector = cv2.dnn.TextDetectionModel_DB(str(text_model))
            detector.setBinaryThreshold(0.3)
            detector.setPolygonThreshold(0.5)
            detector.setMaxCandidates(400)
            detector.setUnclipRatio(2.0)
            detector.setInputParams(1.0 / 255.0, (736, 736), (122.67891434, 116.66876762, 104.00698793), True)
            self.text_detector = detector

    # --- detection ---------------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray, threshold: float | None = None) -> list[tuple[float, float, float, float, float]]:
        if self.face_detector is None:
            return []
        height, width = image.shape[:2]
        self.face_detector.setScoreThreshold(threshold if threshold is not None else self.face_threshold)
        self.face_detector.setInputSize((width, height))
        _, found = self.face_detector.detect(image)
        return [] if found is None else [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in found]

    def detect_faces_in_heads(self, image: np.ndarray, heads: Iterable[Region]) -> list[tuple[float, float, float, float, float]]:
        """YuNet misses faces of ~10 px; enlarged crops around each head find most of them."""
        height, width = image.shape[:2]
        found = []
        for head in heads:
            area = expand(head, 2.2).clipped(width, height)
            if area is None:
                continue
            x0, y0, x1, y1 = area
            crop = image[y0:y1, x0:x1]
            scale = max(1.0, 160.0 / max(crop.shape[0], 1))
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for fx, fy, fw, fh, score in self.detect_faces(crop, max(self.face_threshold - 0.1, 0.3)):
                found.append((x0 + fx / scale, y0 + fy / scale, fw / scale, fh / scale, score))
        return found

    def detect_text(self, image: np.ndarray, people: list[tuple[float, float, float, float]]) -> list[Region]:
        """Scoreboards, ads, watermarks on the whole frame plus jersey numbers/names in person
        crops. Huge regions and regions covering most of a person are dropped - the text
        detector sometimes merges a group of players into one block."""
        if self.text_detector is None:
            return []
        height, width = image.shape[:2]
        regions: list[Region] = []

        def add(polygons, offset_x: float = 0.0, offset_y: float = 0.0, scale: float = 1.0) -> None:
            for polygon in polygons:
                points = np.asarray(polygon, dtype=np.float32) / scale
                x, y, w, h = cv2.boundingRect(points)
                region = Region(offset_x + x - 2, offset_y + y - 2, w + 4, h + 4, "text", ellipse=False)
                if region.width * region.height > 0.03 * width * height:
                    continue
                if any(overlap_share(person, region) > 0.5 for person in people):
                    continue
                regions.append(region)

        polygons, _ = self.text_detector.detect(image)
        add(polygons)
        for x, y, w, h in people:
            torso = Region(x, y + h * 0.12, w, h * 0.5, "text", ellipse=False).clipped(width, height)
            if torso is None:
                continue
            x0, y0, x1, y1 = torso
            crop = image[y0:y1, x0:x1]
            scale = max(1.0, 192.0 / max(crop.shape[0], 1))
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            polygons, _ = self.text_detector.detect(crop)
            add(polygons, x0, y0, scale)
        return regions

    # --- blurring ----------------------------------------------------------------------------

    def _blurred(self, patch: np.ndarray) -> np.ndarray:
        height, width = patch.shape[:2]
        if self.method == "pixelate":
            block = max(2, min(height, width) // 6)
            small = cv2.resize(patch, (max(1, width // block), max(1, height // block)), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        # Down- and upsampling removes identifying detail; the Gaussian then smooths it so the
        # head keeps its shape and colours (no blocks, no boxes).
        small = cv2.resize(patch, (max(1, width // 8), max(1, height // 8)), interpolation=cv2.INTER_AREA)
        coarse = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        sigma = max(1.5, min(height, width) * 0.18)
        return cv2.GaussianBlur(coarse, (0, 0), sigma)

    def apply(self, image: np.ndarray, regions: Iterable[Region]) -> np.ndarray:
        result = image.copy()
        height, width = image.shape[:2]
        for region in regions:
            area = expand(region, 1.15).clipped(width, height)
            if area is None:
                continue
            x0, y0, x1, y1 = area
            patch = result[y0:y1, x0:x1]
            blurred = self._blurred(patch)
            mask = np.zeros(patch.shape[:2], dtype=np.float32)
            if region.ellipse:
                center = ((x1 - x0) / 2, (y1 - y0) / 2)
                cv2.ellipse(mask, (int(center[0]), int(center[1])), (max(1, int((x1 - x0) / 2)), max(1, int((y1 - y0) / 2))), 0, 0, 360, 1.0, -1)
            else:
                mask[:, :] = 1.0
            feather = max(1.0, min(x1 - x0, y1 - y0) * 0.08)
            mask = cv2.GaussianBlur(mask, (0, 0), feather)[..., None]
            result[y0:y1, x0:x1] = (blurred * mask + patch * (1 - mask)).astype(image.dtype)
        return result

    # --- whole frame -------------------------------------------------------------------------

    def anonymize(
        self,
        image: np.ndarray,
        annotations: list[dict],
        extra_people: Iterable[tuple[float, float, float, float]] = (),
        include_text: bool = False,
    ) -> tuple[np.ndarray, Report]:
        """Return the anonymized frame and a report. `annotations` are Reco boxes in this image's
        pixels ({"category", "x", "y", "width", "height"}); they are only read, never changed."""
        report = Report()
        people = [(float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"])) for a in annotations if a.get("category") in PERSON_CATEGORIES]
        people += [tuple(map(float, box)) for box in extra_people]
        heads = [head_region(box) for box in people]
        faces = self.detect_faces(image) + self.detect_faces_in_heads(image, heads)
        face_regions = [expand(Region(x, y, w, h, "face"), 1.4) for x, y, w, h, _ in faces]
        text_regions = self.detect_text(image, people) if include_text else []
        regions = heads + face_regions + text_regions
        report.heads, report.faces, report.texts = len(heads), len(face_regions), len(text_regions)

        result = self.apply(image, regions)
        for annotation in annotations:  # the ball must stay exactly as it was
            if annotation.get("category") in PROTECTED_CATEGORIES:
                box = Region(float(annotation["x"]), float(annotation["y"]), float(annotation["width"]), float(annotation["height"]), "ball").clipped(image.shape[1], image.shape[0])
                if box:
                    x0, y0, x1, y1 = box
                    result[y0:y1, x0:x1] = image[y0:y1, x0:x1]

        # Verification: a face YuNet still finds away from every blurred region was missed and
        # makes the frame unsafe. YuNet also "finds" many blurred heads (the blob keeps a face's
        # shape), so a hit that overlaps a blurred region counts as covered.
        for x, y, w, h, score in self.detect_faces(result, 0.75) + self.detect_faces_in_heads(result, heads):
            if score < 0.75:
                continue
            if not any(overlap_share((x, y, w, h), expand(region, 1.3)) >= 0.3 for region in regions):
                report.residual_faces.append([x, y, w, h, score])
        return result, report
