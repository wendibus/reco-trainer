"""Anonymization keeps labels and the ball untouched, blurs heads without opaque boxes,
and a dataset export (plain or anonymized) has consistent YOLO and COCO labels."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
    import anonymize
    HAS_CV = True
except ImportError:
    HAS_CV = False


def load_ml_worker():
    spec = importlib.util.spec_from_file_location("anonymize_ml_worker", ROOT / "ml_worker.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml_worker = load_ml_worker()

PLAYER = {"id": "p1", "category": "player", "x": 100.0, "y": 60.0, "width": 60.0, "height": 180.0}
BALL_ON_HEAD = {"id": "b1", "category": "ball", "x": 120.0, "y": 58.0, "width": 18.0, "height": 18.0}


def textured_image(width: int = 320, height: int = 280):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@unittest.skipUnless(HAS_CV, "OpenCV / numpy are not installed")
class AnonymizerTests(unittest.TestCase):
    def setUp(self):
        self.anonymizer = anonymize.Anonymizer(face_model=None)

    def test_labels_are_not_changed(self):
        annotations = [dict(PLAYER), dict(BALL_ON_HEAD)]
        before = json.dumps(annotations, sort_keys=True)
        _, report = self.anonymizer.anonymize(textured_image(), annotations)
        self.assertEqual(json.dumps(annotations, sort_keys=True), before)
        self.assertEqual(report.heads, 1)
        self.assertTrue(report.safe)

    def test_head_is_blurred_but_not_covered_and_the_rest_stays(self):
        image = textured_image()
        result, _ = self.anonymizer.anonymize(image, [dict(PLAYER)])
        head = anonymize.head_region((PLAYER["x"], PLAYER["y"], PLAYER["width"], PLAYER["height"]))
        x0, y0, x1, y1 = head.clipped(image.shape[1], image.shape[0])
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        core = (slice(cy - 4, cy + 4), slice(cx - 6, cx + 6))
        # Blurred: far less detail than the random texture there before ...
        self.assertLess(result[core].astype(float).std(), image[core].astype(float).std() * 0.5)
        # ... but no opaque box: the colours stay close to the original average.
        self.assertLess(abs(result[core].mean() - image[core].mean()), 25)
        # Legs and background are untouched.
        legs = (slice(200, 240), slice(105, 155))
        self.assertTrue(np.array_equal(result[legs], image[legs]))
        self.assertTrue(np.array_equal(result[:, 250:], image[:, 250:]))

    def test_ball_pixels_are_restored_even_inside_a_head(self):
        image = textured_image()
        result, _ = self.anonymizer.anonymize(image, [dict(PLAYER), dict(BALL_ON_HEAD)])
        ball = (slice(58, 76), slice(120, 138))
        self.assertTrue(np.array_equal(result[ball], image[ball]))

    def test_pixelate_makes_blocks(self):
        anonymizer = anonymize.Anonymizer(face_model=None, method="pixelate")
        patch = textured_image(60, 60)
        blocks = anonymizer._blurred(patch)
        self.assertEqual(len(np.unique(blocks[0:10, 0:10].reshape(-1, 3), axis=0)), 1)

    def test_extra_people_are_anonymized_too(self):
        image = textured_image()
        spectator = (230.0, 40.0, 50.0, 150.0)
        result, report = self.anonymizer.anonymize(image, [], extra_people=[spectator])
        self.assertEqual(report.heads, 1)
        self.assertFalse(np.array_equal(result[45:60, 240:270], image[45:60, 240:270]))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            anonymize.Anonymizer(face_model=None, method="black-box")

    def test_text_regions_covering_players_are_dropped(self):
        class FakeTextDetector:
            def detect(self, image):
                height, width = image.shape[:2]
                if width > 200:  # whole frame: one block over the player, one small scoreboard
                    return [np.array([[90, 50], [170, 50], [170, 250], [90, 250]]), np.array([[10, 10], [60, 10], [60, 25], [10, 25]])], []
                return [], []

        self.anonymizer.text_detector = FakeTextDetector()
        regions = self.anonymizer.detect_text(textured_image(), [(PLAYER["x"], PLAYER["y"], PLAYER["width"], PLAYER["height"])])
        self.assertEqual(len(regions), 1)
        self.assertLess(regions[0].x, 20)


class ExportDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        frames = []
        for video in ("v1", "v2", "v3"):
            for index in range(2):
                frame_id = f"{video}-{index}"
                path = self.project / "frames" / f"{frame_id}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                if HAS_CV:
                    cv2.imwrite(str(path), textured_image())
                else:
                    path.write_bytes(b"\xff\xd8 fake jpeg")
                frames.append({
                    "id": frame_id, "relativePath": f"frames/{frame_id}.jpg", "videoID": video, "videoName": f"Secret match {video}.mp4",
                    "timestamp": index, "width": 320, "height": 280, "reviewStatus": "reviewed",
                    "annotations": [dict(PLAYER), dict(BALL_ON_HEAD)],
                })
        frames.append({**frames[0], "id": "candidate", "reviewStatus": "candidate"})
        ml_worker.atomic_json(self.project / "project.json", {"sport": "basketball", "frames": frames})

    def tearDown(self):
        self.temporary.cleanup()

    def run_export(self, **overrides) -> dict:
        values = dict(project=str(self.project), formats="yolo,coco", anonymize=False, method="gaussian", include_text=False,
                      detect_people=False, detector_size="auto", zip=True, app_version="9.9.9", language="en")
        values.update(overrides)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            ml_worker.export_dataset(argparse.Namespace(**values))
        return json.loads(output.getvalue().strip().splitlines()[-1])

    def test_plain_export_has_matching_yolo_and_coco_labels_per_video_split(self):
        summary = self.run_export()
        folder = Path(summary["folder"])
        self.assertEqual(summary["images"], 6)  # the candidate frame is not exported
        data_yaml = (folder / "yolo" / "data.yaml").read_text()
        self.assertIn("0: ball", data_yaml)
        self.assertIn("1: player", data_yaml)
        coco_files = {split: json.loads((folder / "coco" / split / "_annotations.coco.json").read_text()) for split in ("train", "valid", "test")}
        self.assertEqual(sum(len(c["images"]) for c in coco_files.values()), 6)
        for split, yolo_split in (("train", "train"), ("valid", "val"), ("test", "test")):
            coco = coco_files[split]
            # One video per split here, so both frames of a video share it.
            for image in coco["images"]:
                label = folder / "yolo" / "labels" / yolo_split / (Path(image["file_name"]).stem + ".txt")
                lines = [line.split() for line in label.read_text().splitlines()]
                self.assertEqual(len(lines), 2)
                player = next(line for line in lines if line[0] == "1")
                self.assertAlmostEqual(float(player[1]), (100 + 30) / 320, places=5)
                self.assertAlmostEqual(float(player[4]), 180 / 280, places=5)
                boxes = [a["bbox"] for a in coco["annotations"] if a["image_id"] == image["id"]]
                self.assertIn([100.0, 60.0, 60.0, 180.0], boxes)
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(metadata["version"], "9.9.9")
        self.assertFalse(metadata["anonymization"]["enabled"])
        # Privacy: no video names or source paths anywhere in the export.
        with zipfile.ZipFile(summary["zip"]) as archive:
            names = archive.namelist()
            text = "".join(archive.read(name).decode("utf-8", "ignore") for name in names if name.endswith((".json", ".txt", ".yaml")))
        self.assertTrue(all(Path(name).name.startswith("frame-") or not name.endswith(".jpg") for name in names))
        self.assertNotIn("Secret match", text)
        self.assertNotIn(str(self.project), text)

    def test_cancelled_export_leaves_no_partial_folder(self):
        with mock.patch.object(ml_worker, "atomic_json", side_effect=SystemExit(143)):
            with self.assertRaises(SystemExit):
                self.run_export()
        leftovers = sorted(path.name for path in (self.project / "exports" / "datasets").iterdir())
        self.assertEqual(leftovers, [])

    @unittest.skipUnless(HAS_CV, "OpenCV / numpy are not installed")
    def test_anonymized_export_blurs_heads_and_keeps_labels(self):
        with mock.patch.object(ml_worker, "download_registry_model", lambda *_args, **_kwargs: None):
            summary = self.run_export(anonymize=True, formats="coco", zip=False)
        folder = Path(summary["folder"])
        coco = json.loads((folder / "train" / "_annotations.coco.json").read_text())
        self.assertEqual(len(coco["annotations"]), 2 * len(coco["images"]))
        exported = cv2.imread(str(folder / "train" / coco["images"][0]["file_name"]))
        original = textured_image()
        self.assertFalse(np.array_equal(exported[60:75, 110:150], original[60:75, 110:150]))
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertTrue(metadata["anonymization"]["enabled"])
        self.assertEqual(metadata["anonymization"]["method"], "gaussian")


    @unittest.skipUnless(HAS_CV, "OpenCV / numpy are not installed")
    def test_training_augmentation_anonymizes_copies_never_the_original_frames(self):
        originals = {path.name: path.read_bytes() for path in (self.project / "frames").iterdir()}
        with mock.patch.object(ml_worker, "download_registry_model", lambda *_args, **_kwargs: None), \
                contextlib.redirect_stdout(io.StringIO()):
            dataset = ml_worker.build_dataset(self.project, "en", anonymize_fraction=0.5)
        for name, data in originals.items():
            self.assertEqual((self.project / "frames" / name).read_bytes(), data, f"original {name} was changed")
        train = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
        changed = [image["file_name"] for image in train["images"] if (dataset / "train" / image["file_name"]).read_bytes() != originals[image["file_name"]]]
        chosen = [frame for frame in json.loads((self.project / "project.json").read_text())["frames"]
                  if frame["reviewStatus"] != "candidate" and ml_worker.anonymize_for_training(frame, 0.5) and Path(frame["relativePath"]).name in {i["file_name"] for i in train["images"]}]
        self.assertEqual(len(changed), len(chosen))
        self.assertEqual(len(train["annotations"]), 2 * len(train["images"]))

    def test_training_anonymization_choice_is_stable(self):
        frame = {"id": "abc"}
        self.assertEqual(ml_worker.anonymize_for_training(frame, 0.3), ml_worker.anonymize_for_training(frame, 0.3))
        self.assertFalse(ml_worker.anonymize_for_training(frame, 0.0))


if __name__ == "__main__":
    unittest.main()
