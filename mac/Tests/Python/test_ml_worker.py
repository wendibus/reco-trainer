from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path


WORKER_PATH = (
    Path(__file__).parents[2]
    / "Sources"
    / "RecoTrainerMac"
    / "Resources"
    / "ml_worker.py"
)
SPEC = importlib.util.spec_from_file_location("ml_worker", WORKER_PATH)
assert SPEC and SPEC.loader
ml_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ml_worker)

try:
    import cv2  # noqa: F401
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class DatasetTests(unittest.TestCase):
    def make_frame(self, video_id: str, number: int, root: Path, annotated: bool = True):
        relative = f"frames/{video_id}-{number}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg")
        annotations = []
        if annotated:
            annotations.append(
                {
                    "id": str(uuid.uuid4()),
                    "category": "ball",
                    "x": 10.0,
                    "y": 20.0,
                    "width": 8.0,
                    "height": 8.0,
                    "source": "manual",
                }
            )
        return {
            "id": str(uuid.uuid4()),
            "relativePath": relative,
            "videoID": video_id,
            "videoName": f"{video_id}.mov",
            "timestamp": float(number),
            "width": 1920,
            "height": 1080,
            "annotations": annotations,
        }

    def test_builds_coco_and_keeps_videos_in_separate_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = [
                self.make_frame(video, number, root)
                for video in ("a", "b", "c")
                for number in range(2)
            ]
            project = {
                "schemaVersion": 1,
                "name": "Test",
                "sport": "football",
                "sourceFolder": str(root),
                "createdAt": "2026-08-28T10:00:00Z",
                "updatedAt": "2026-08-28T10:00:00Z",
                "frames": frames,
            }
            ml_worker.atomic_json(root / "project.json", project)

            dataset = ml_worker.build_dataset(root)

            seen_videos = {}
            for split in ("train", "valid", "test"):
                coco = json.loads((dataset / split / "_annotations.coco.json").read_text())
                self.assertEqual(coco["categories"][0]["name"], "ball")
                for image in coco["images"]:
                    video = image["file_name"].split("-")[0]
                    if video in seen_videos:
                        self.assertEqual(seen_videos[video], split)
                    else:
                        seen_videos[video] = split
            self.assertEqual(set(seen_videos.values()), {"train", "valid", "test"})

    def test_unreviewed_candidates_are_excluded_from_training_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewed = self.make_frame("reviewed", 1, root)
            candidate = self.make_frame("candidate", 1, root)
            candidate["reviewStatus"] = "candidate"
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Active learning", "sport": "basketball",
                "sourceFolder": str(root), "frames": [reviewed, candidate],
            })

            dataset = ml_worker.build_dataset(root)
            names = []
            for split in ("train", "valid", "test"):
                coco = json.loads((dataset / split / "_annotations.coco.json").read_text())
                names.extend(image["file_name"] for image in coco["images"])

            self.assertIn(Path(reviewed["relativePath"]).name, names)
            self.assertNotIn(Path(candidate["relativePath"]).name, names)

    def test_build_dataset_restricts_to_the_given_categories(self):
        # Regression/feature coverage: by request, a training run can be
        # restricted to a subset of already-annotated categories (e.g. "only
        # ball and referee this time"). build_dataset() must drop both the
        # excluded category from the dataset's category list AND every
        # individual box of that category - not just hide it from the list.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self.make_frame("one", 1, root)
            frame["annotations"].append({
                "id": str(uuid.uuid4()), "category": "referee",
                "x": 30.0, "y": 40.0, "width": 6.0, "height": 6.0, "source": "manual",
            })
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Test", "sport": "basketball",
                "sourceFolder": str(root), "frames": [frame],
            })

            dataset = ml_worker.build_dataset(root, categories=["ball"])

            coco = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
            self.assertEqual([category["name"] for category in coco["categories"]], ["ball"])
            self.assertEqual(len(coco["annotations"]), 1)

    def test_build_dataset_rejects_a_category_subset_with_no_matching_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self.make_frame("one", 1, root)  # only annotated with "ball"
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Test", "sport": "basketball",
                "sourceFolder": str(root), "frames": [frame],
            })

            with self.assertRaises(SystemExit):
                ml_worker.build_dataset(root, categories=["referee"])

    def test_single_video_temporal_split(self):
        frames = [
            {
                "videoID": "one",
                "timestamp": float(number),
                "annotations": [],
            }
            for number in range(20)
        ]
        splits, warning = ml_worker.split_frames(frames)
        self.assertIsNotNone(warning)
        self.assertEqual(len(splits["train"]), 14)
        self.assertEqual(len(splits["valid"]), 4)
        self.assertEqual(len(splits["test"]), 2)

    def test_ball_focus_ignores_all_people(self):
        mapping = ml_worker.detection_category_map(["ball"])
        self.assertEqual(mapping["sports ball"], "ball")
        self.assertNotIn("person", mapping)

    def test_detection_category_map_merges_multiple_selected_categories(self):
        mapping = ml_worker.detection_category_map(["ball", "player"])
        self.assertEqual(mapping["sports ball"], "ball")
        self.assertEqual(mapping["person"], "player")
        # A category with no base-model alias (e.g. hoop) still maps to itself,
        # so a custom-trained model's own class name passes through unchanged.
        mapping_custom = ml_worker.detection_category_map(["hoop"])
        self.assertEqual(mapping_custom["hoop"], "hoop")
        self.assertNotIn("sports ball", mapping_custom)

    def test_autolabel_adds_multiple_categories_in_one_run(self):
        """A single auto_label() call with multiple --category values should add
        every still-missing selected category from one model pass per frame,
        without duplicating a category a frame already has."""
        class FakeDetections:
            data = {"class_name": ["sports ball", "person"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0], [40.0, 10.0, 60.0, 70.0]]
            class_id = [32, 0]
            confidence = [.9, .8]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            (root / "frames" / "two.jpg").write_bytes(b"synthetic")
            project = {
                "sport": "basketball",
                "frames": [
                    {"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": []},
                    {"id": "two", "relativePath": "frames/two.jpg", "width": 100, "height": 80, "annotations": [
                        {"id": "existing", "category": "ball", "x": 1, "y": 1, "width": 5, "height": 5, "source": "manual"},
                    ]},
                ],
            }
            ml_worker.atomic_json(root / "project.json", project)
            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["ball", "player"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device

            document = ml_worker.load_json(root / "project.json")
            frame_one = next(frame for frame in document["frames"] if frame["id"] == "one")
            frame_two = next(frame for frame in document["frames"] if frame["id"] == "two")

            self.assertEqual({item["category"] for item in frame_one["annotations"]}, {"ball", "player"})
            self.assertTrue(all(item["source"] == "auto" for item in frame_one["annotations"]))
            # frame two already had a manual "ball" box, so autolabel should only add "player".
            self.assertEqual(sorted(item["category"] for item in frame_two["annotations"]), ["ball", "player"])
            self.assertEqual(sum(item["category"] == "ball" for item in frame_two["annotations"]), 1)

    def test_active_model_classes_reads_the_activated_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ml_worker.atomic_json(root / "models" / "active.json", {"packageID": "pkg-1", "modelSize": "nano"})
            ml_worker.atomic_json(root / "models" / "library" / "pkg-1" / "manifest.json", {"classes": ["ball"]})

            self.assertEqual(ml_worker.active_model_classes(root, "nano"), {"ball"})
            # A different, non-activated model size is "unknown", not "unsupported".
            self.assertIsNone(ml_worker.active_model_classes(root, "small"))
            # No active.json at all (e.g. no model ever installed/trained).
            self.assertIsNone(ml_worker.active_model_classes(root / "missing", "nano"))

    def test_auto_label_skips_categories_the_active_model_was_not_trained_on(self):
        """Regression test: an activated model that was only ever trained on "ball"
        (e.g. before "referee" was added to the project) used to still have a full
        inference pass run for "referee" too, since the old check only asked "does
        any checkpoint exist" rather than "does this specific model's own manifest
        cover this class" - wasting time and reporting a single misleading "0 boxes"
        instead of a clear per-category explanation.
        """
        class FakeDetections:
            data = {"class_name": ["ball", "referee"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0], [40.0, 10.0, 60.0, 70.0]]
            class_id = [0, 1]
            confidence = [.9, .8]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {
                "sport": "basketball",
                "frames": [
                    {"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": []},
                ],
            })
            ml_worker.atomic_json(root / "models" / "active.json", {"packageID": "pkg-1", "modelSize": "nano"})
            ml_worker.atomic_json(root / "models" / "library" / "pkg-1" / "manifest.json", {"classes": ["ball"]})

            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            original_emit = ml_worker.emit
            messages: list[str] = []
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.emit = lambda message: messages.append(message)
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["ball", "referee"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device
                ml_worker.emit = original_emit

            self.assertTrue(any("referee" in message and "does not know" in message for message in messages))
            document = ml_worker.load_json(root / "project.json")
            categories = {item["category"] for item in document["frames"][0]["annotations"]}
            self.assertEqual(categories, {"ball"})

    def test_auto_label_falls_back_to_base_model_for_player_when_active_model_lacks_it(self):
        """Feature test: unlike "referee" (no COCO equivalent), "player" can still be
        detected generically via the base model's "person" class even while a
        specialized "ball"-only model is active. auto_label() should run a second
        pass with the base model for "player" instead of skipping it, so the user
        doesn't have to draw every player by hand just because their model was
        trained on ball alone so far.
        """
        class CustomDetections:
            data = {"class_name": ["ball"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0]]
            class_id = [0]
            confidence = [.9]

        class BaseDetections:
            data = {"class_name": ["person"]}
            xyxy = [[40.0, 10.0, 60.0, 70.0]]
            class_id = [0]
            confidence = [.8]

        class DualFakeModel:
            def __init__(self, **kwargs):
                self.is_custom = "pretrain_weights" in kwargs

            def predict(self, paths, threshold):
                detections = CustomDetections() if self.is_custom else BaseDetections()
                return [detections for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {
                "sport": "basketball",
                "frames": [
                    {"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": []},
                ],
            })
            weights_dir = root / "models" / "library" / "pkg-1" / "weights"
            weights_dir.mkdir(parents=True)
            weights_file = weights_dir / "checkpoint_best_total.pth"
            weights_file.write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "models" / "active.json", {
                "packageID": "pkg-1", "modelSize": "nano",
                "weights": str(weights_file.relative_to(root)),
            })
            ml_worker.atomic_json(root / "models" / "library" / "pkg-1" / "manifest.json", {"classes": ["ball"]})

            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            original_emit = ml_worker.emit
            messages: list[str] = []
            try:
                ml_worker.import_model_class = lambda *_: DualFakeModel
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.emit = lambda message: messages.append(message)
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["ball", "player"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device
                ml_worker.emit = original_emit

            self.assertTrue(any("player" in message and "generic base model is used in addition" in message for message in messages))
            document = ml_worker.load_json(root / "project.json")
            categories = {item["category"] for item in document["frames"][0]["annotations"]}
            self.assertEqual(categories, {"ball", "player"})

    # Coverage for the field-boundary filter: by request, auto-label should only
    # add "player"/"referee"/"goalkeeper" boxes for people whose feet fall inside
    # the project's marked field (four image corners + real dimensions), so a
    # spectator or bench player standing nearby doesn't silently become "player".
    #
    # The test geometry is deliberately axis-aligned (corners form a square, not a
    # perspective-distorted quad) so the expected real-world coordinates can be
    # worked out by hand: frame pixels (100,100)-(900,900) map to field meters
    # (0,0)-(20,20), a uniform scale of 0.025 m/px with a (-100,-100)px offset.
    def _square_field_geometry(self) -> dict:
        return {
            "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "realWidth": 20.0,
            "realLength": 20.0,
        }

    def test_field_membership_checker_returns_none_without_geometry(self):
        self.assertIsNone(ml_worker.field_membership_checker(None))
        self.assertIsNone(ml_worker.field_membership_checker({}))

    def test_field_membership_checker_returns_none_for_incomplete_or_invalid_geometry(self):
        self.assertIsNone(ml_worker.field_membership_checker({
            "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],  # only 3 corners
            "realWidth": 20.0, "realLength": 20.0,
        }))
        self.assertIsNone(ml_worker.field_membership_checker({
            "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "realWidth": 0.0, "realLength": 20.0,  # zero real width
        }))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_field_membership_checker_accepts_inside_and_rejects_outside(self):
        is_on_field = ml_worker.field_membership_checker(self._square_field_geometry())
        self.assertIsNotNone(is_on_field)
        # Frame center (500, 500) -> field (10, 10): well inside the 20x20 field.
        self.assertTrue(is_on_field("player", 480.0, 400.0, 520.0, 500.0, 1000.0, 1000.0))
        # Near the frame's top-left corner (50, 50), well outside the marked
        # square (which starts at pixel 100,100) -> negative field coordinates.
        self.assertFalse(is_on_field("player", 30.0, 20.0, 70.0, 50.0, 1000.0, 1000.0))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_field_membership_checker_only_filters_person_shaped_categories(self):
        is_on_field = ml_worker.field_membership_checker(self._square_field_geometry())
        self.assertIsNotNone(is_on_field)
        # A "ball" detection far outside the marked field is not filtered - the
        # field boundary only applies to people (see PERSON_SHAPED_CATEGORIES).
        self.assertTrue(is_on_field("ball", 30.0, 20.0, 70.0, 50.0, 1000.0, 1000.0))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_auto_label_skips_off_field_people_and_keeps_on_field_ones(self):
        class FakeDetections:
            data = {"class_name": ["person", "person"]}
            # Box 1: feet (bottom-center) at pixel (500, 500) -> inside the field.
            # Box 2: feet at pixel (50, 50) -> outside the marked field.
            xyxy = [[480.0, 400.0, 520.0, 500.0], [30.0, 20.0, 70.0, 50.0]]
            class_id = [0, 0]
            confidence = [.9, .8]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {
                "sport": "basketball",
                "fieldGeometry": {
                    "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    "realWidth": 20.0, "realLength": 20.0,
                },
                "frames": [
                    {
                        "id": "one", "relativePath": "frames/one.jpg",
                        "width": 1000, "height": 1000, "annotations": [],
                    },
                ],
            })

            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            original_emit = ml_worker.emit
            messages: list[str] = []
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.emit = lambda message: messages.append(message)
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["player"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device
                ml_worker.emit = original_emit

            self.assertTrue(any("Field boundaries active" in message for message in messages))
            document = ml_worker.load_json(root / "project.json")
            annotations = document["frames"][0]["annotations"]
            self.assertEqual(len(annotations), 1)
            self.assertAlmostEqual(annotations[0]["x"], 480.0)

    # Coverage for the clothing-based referee heuristic: by request, a generic
    # "player" detection whose crop matches the sport's typical referee attire
    # (see REFEREE_CLOTHING_PROFILES) gets reclassified to "referee" - still
    # source="auto", so it goes through the same manual review as any other
    # automatic suggestion rather than being trusted blindly.
    def test_referee_clothing_checker_returns_none_for_an_unknown_sport(self):
        self.assertIsNone(ml_worker.referee_clothing_checker("curling"))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_basketball_matches_grey_top_black_bottom(self):
        matches = ml_worker.referee_clothing_checker("basketball")
        self.assertIsNotNone(matches)
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        image[:110] = (128, 128, 128)  # grey torso (BGR)
        image[110:] = (10, 10, 10)  # black legs
        self.assertTrue(matches(image, (0.0, 0.0, 100.0, 200.0)))

        # A solid red kit is not a basketball referee.
        red_image = np.zeros((200, 100, 3), dtype=np.uint8)
        red_image[:] = (0, 0, 220)
        self.assertFalse(matches(red_image, (0.0, 0.0, 100.0, 200.0)))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_hockey_matches_black_and_white_stripes(self):
        matches = ml_worker.referee_clothing_checker("hockey")
        self.assertIsNotNone(matches)
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        image[:55] = (10, 10, 10)  # black band, upper torso
        image[55:110] = (250, 250, 250)  # white band, lower torso
        image[110:] = (10, 10, 10)  # black legs (irrelevant to the stripe check)
        self.assertTrue(matches(image, (0.0, 0.0, 100.0, 200.0)))

        solid_image = np.zeros((200, 100, 3), dtype=np.uint8)
        solid_image[:] = (0, 0, 220)
        self.assertFalse(matches(solid_image, (0.0, 0.0, 100.0, 200.0)))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_auto_label_reclassifies_a_matching_player_to_referee(self):
        class FakeDetections:
            data = {"class_name": ["person"]}
            xyxy = [[0.0, 0.0, 100.0, 200.0]]
            class_id = [0]
            confidence = [.9]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            image = np.zeros((200, 100, 3), dtype=np.uint8)
            image[:110] = (128, 128, 128)
            image[110:] = (10, 10, 10)
            cv2.imwrite(str(root / "frames" / "one.jpg"), image)
            ml_worker.atomic_json(root / "project.json", {
                "sport": "basketball",
                "frames": [
                    {
                        "id": "one", "relativePath": "frames/one.jpg",
                        "width": 100, "height": 200, "annotations": [],
                    },
                ],
            })

            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            original_emit = ml_worker.emit
            messages: list[str] = []
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.emit = lambda message: messages.append(message)
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["player", "referee"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device
                ml_worker.emit = original_emit

            self.assertTrue(any("Clothing heuristic active" in message for message in messages))
            document = ml_worker.load_json(root / "project.json")
            annotations = document["frames"][0]["annotations"]
            self.assertEqual(len(annotations), 1)
            self.assertEqual(annotations[0]["category"], "referee")
            self.assertEqual(annotations[0]["source"], "auto")

    def test_referee_alone_without_player_is_reported_as_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {
                "sport": "basketball",
                "frames": [
                    {"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 200, "annotations": []},
                ],
            })
            messages: list[str] = []
            original_emit = ml_worker.emit
            try:
                ml_worker.emit = lambda message: messages.append(message)
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["referee"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.emit = original_emit

            self.assertTrue(any("does not know" in message for message in messages))
            if HAS_OPENCV:
                self.assertTrue(any("also requires" in message for message in messages))
            document = ml_worker.load_json(root / "project.json")
            self.assertEqual(document["frames"][0]["annotations"], [])

    def test_category_specific_skip_preserves_other_classes(self):
        frame = {"annotations": [{"category": "player"}]}
        self.assertFalse(ml_worker.frame_has_category(frame, "ball"))
        self.assertTrue(ml_worker.frame_has_category(frame, "player"))

    def test_refinable_annotations_includes_manual_boxes_by_request(self):
        # By explicit user request, OpenCV refinement now also considers manually
        # drawn boxes eligible (previously only source == "auto" boxes were, on the
        # principle that a manual box is already human-confirmed ground truth).
        # plausible_refined_box's own checks are what keeps a bad refinement from
        # being applied, regardless of source.
        frame = {
            "annotations": [
                {"category": "ball", "source": "manual"},
                {"category": "player", "source": "auto"},
                {"category": "ball", "source": "auto", "opencvRefinement": {"method": "grabcut-contour-v1"}},
                {"category": "spectator", "source": "manual"},  # not a category CATEGORY_ASPECT_BOUNDS covers
            ]
        }
        eligible = ml_worker.refinable_annotations(frame)
        self.assertEqual(len(eligible), 2)
        self.assertTrue(any(item["source"] == "manual" and item["category"] == "ball" for item in eligible))
        self.assertTrue(any(item["source"] == "auto" and item["category"] == "player" for item in eligible))
        self.assertFalse(any(item.get("opencvRefinement") for item in eligible))
        self.assertFalse(any(item["category"] == "spectator" for item in eligible))

    def test_extended_sports_use_specialized_categories(self):
        self.assertEqual(ml_worker.sport_categories("rugby"), ["ball", "player", "referee", "goalpost"])
        self.assertIn("goalkeeper", ml_worker.sport_categories("lacrosse"))
        self.assertIn("goalpost", ml_worker.sport_categories("american_football"))

    def test_benchmark_scores_a_perfect_detection(self):
        truth = [{"id": "frame-1", "annotations": [{"category": "ball", "x": 10, "y": 20, "width": 10, "height": 10}]}]
        predictions = [{"frameID": "frame-1", "category": "ball", "confidence": .9, "box": [10, 20, 20, 30]}]
        metrics = ml_worker.evaluate_predictions(truth, predictions, ["ball"])
        self.assertEqual(metrics["truePositives"], 1)
        self.assertEqual(metrics["falsePositives"], 0)
        self.assertEqual(metrics["falseNegatives"], 0)
        self.assertAlmostEqual(metrics["mAP50"], 1.0)
        self.assertAlmostEqual(metrics["qualityScore"], 100.0)

    def test_box_iou_and_box_iou_xywh_are_independent_helpers(self):
        """Regression test: ml_worker.py used to define box_iou twice — once for
        xyxy corners (used by evaluate_predictions) and once for x/y/width/height
        boxes (used by the OpenCV refinement). The second definition silently
        shadowed the first at module scope, so evaluate_predictions computed IoU
        against the wrong box format. The two helpers must stay distinct and each
        must interpret its own coordinate format correctly.
        """
        self.assertIsNot(ml_worker.box_iou, ml_worker.box_iou_xywh)
        overlap_xyxy = ml_worker.box_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])
        self.assertAlmostEqual(overlap_xyxy, 25.0 / 175.0)
        overlap_xywh = ml_worker.box_iou_xywh((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0))
        self.assertAlmostEqual(overlap_xywh, 25.0 / 175.0)

    def test_benchmark_scores_a_partial_overlap_below_threshold_as_a_miss(self):
        """Unlike the other benchmark tests above, truth and prediction here are
        NOT identical, so a box-format bug (xyxy vs. x/y/width/height) changes
        the outcome instead of canceling out. This is what actually caught the
        box_iou/box_iou_xywh name collision.
        """
        truth = [{"id": "frame-1", "annotations": [{"category": "ball", "x": 0, "y": 0, "width": 20, "height": 20}]}]
        predictions = [{"frameID": "frame-1", "category": "ball", "confidence": .9, "box": [25, 0, 45, 20]}]
        metrics = ml_worker.evaluate_predictions(truth, predictions, ["ball"])
        self.assertEqual(metrics["truePositives"], 0)
        self.assertEqual(metrics["falsePositives"], 1)
        self.assertEqual(metrics["falseNegatives"], 1)

    def test_benchmark_penalizes_false_positives_and_missed_objects(self):
        truth = [{"id": "frame-1", "annotations": [
            {"category": "ball", "x": 10, "y": 20, "width": 10, "height": 10},
            {"category": "ball", "x": 50, "y": 50, "width": 10, "height": 10},
        ]}]
        predictions = [
            {"frameID": "frame-1", "category": "ball", "confidence": .9, "box": [10, 20, 20, 30]},
            {"frameID": "frame-1", "category": "ball", "confidence": .8, "box": [80, 80, 90, 90]},
        ]
        metrics = ml_worker.evaluate_predictions(truth, predictions, ["ball"])
        self.assertEqual(metrics["truePositives"], 1)
        self.assertEqual(metrics["falsePositives"], 1)
        self.assertEqual(metrics["falseNegatives"], 1)
        self.assertAlmostEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], .5)
        self.assertAlmostEqual(metrics["f1"], .5)

    def test_benchmark_maps_model_class_ids_to_sport_schema(self):
        self.assertEqual(ml_worker.benchmark_detection_category("", 0, "basketball", {"ball", "player"}), "ball")
        self.assertEqual(ml_worker.benchmark_detection_category("sports ball", 32, "hockey", {"puck"}), "puck")
        self.assertIsNone(ml_worker.benchmark_detection_category("person", 0, "football", {"ball"}))

    def test_benchmark_command_writes_local_ranking_and_predictions(self):
        class FakeDetections:
            data = {"class_name": ["ball"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0]]
            class_id = [0]
            confidence = [.95]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                self.threshold = threshold
                return FakeDetections() if len(paths) == 1 else [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / "frames" / "one.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
            ml_worker.atomic_json(root / "benchmarks" / "ground-truth.json", {
                "schemaVersion": 1, "createdAt": "2026-08-29T10:00:00Z", "sport": "basketball", "datasetID": "synthetic-dataset",
                "frames": [{"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": [{"category": "ball", "x": 10, "y": 20, "width": 10, "height": 10}]}],
            })
            model_dir = root / "models" / "library" / "synthetic-model"
            (model_dir / "weights").mkdir(parents=True)
            (model_dir / "weights" / "model.pth").write_bytes(b"synthetic")
            ml_worker.atomic_json(model_dir / "manifest.json", {"packageID": "synthetic-model", "sport": "basketball", "modelSize": "nano", "classes": ["ball"], "weights": {"file": "weights/model.pth"}})
            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.benchmark(type("Args", (), {"project": str(root), "threshold": .05, "language": "en"})())
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device
            report = ml_worker.load_json(root / "benchmarks" / "latest.json")
            self.assertEqual(report["results"][0]["rank"], 1)
            self.assertAlmostEqual(report["results"][0]["metrics"]["qualityScore"], 100.0)
            prediction_files = list((root / "benchmarks" / "runs").rglob("*.predictions.json"))
            self.assertEqual(len(prediction_files), 1)
            self.assertNotIn("imageData", prediction_files[0].read_text())

    def test_mps_profile_scales_batch_and_parallel_data_loading(self):
        profile = ml_worker.training_profile(
            "nano", "mps", memory_bytes=16 * 1024**3, cpu_count=8
        )
        self.assertEqual(profile["batch_size"], 2)
        self.assertEqual(profile["grad_accum_steps"], 8)
        self.assertEqual(profile["num_workers"], 4)
        self.assertTrue(profile["persistent_workers"])
        self.assertFalse(profile["pin_memory"])

    def test_large_mps_profile_uses_larger_small_model_batches(self):
        profile = ml_worker.training_profile(
            "small", "mps", memory_bytes=64 * 1024**3, cpu_count=20
        )
        self.assertEqual(profile["batch_size"], 4)
        self.assertEqual(profile["grad_accum_steps"], 4)
        self.assertEqual(profile["num_workers"], 8)
        self.assertFalse(profile["gradient_checkpointing"])

    def test_only_newer_full_checkpoint_is_treated_as_interrupted_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "runs" / "small" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-full-checkpoint")

            interrupted = {"lastTraining": {"completedAt": "2000-01-01T00:00:00Z"}}
            completed = {"lastTraining": {"completedAt": "2999-01-01T00:00:00Z"}}

            self.assertEqual(
                ml_worker.interrupted_run_checkpoint(root, "small", interrupted),
                checkpoint,
            )
            self.assertIsNone(
                ml_worker.interrupted_run_checkpoint(root, "small", completed)
            )

    def _dataset_root_with_categories(self, count: int) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        ml_worker.atomic_json(root / "train" / "_annotations.coco.json", {
            "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": f"class-{i}"} for i in range(count)],
        })
        return root

    # Regression coverage for a real crash: a strict Lightning resume
    # (trainer.fit(ckpt_path=...)) hard-crashes with a state_dict size-mismatch
    # RuntimeError when the checkpoint's detection head was trained for a different
    # number of classes than the project currently has - e.g. after a class was
    # added via auto-labeling while a training run sat interrupted.
    def test_validate_resume_checkpoint_allows_a_matching_checkpoint_through(self):
        dataset_root = self._dataset_root_with_categories(2)
        checkpoint = dataset_root / "last.ckpt"
        checkpoint.write_bytes(b"synthetic")
        original = ml_worker.checkpoint_class_count
        ml_worker.checkpoint_class_count = lambda _: 2
        try:
            result = ml_worker.validate_resume_checkpoint(checkpoint, dataset_root)
        finally:
            ml_worker.checkpoint_class_count = original
        self.assertEqual(result, checkpoint)

    def test_validate_resume_checkpoint_rejects_a_class_count_mismatch(self):
        # The exact scenario from the crash: interrupted at 2 classes, the project
        # now has 4 after auto-labeling added more categories.
        dataset_root = self._dataset_root_with_categories(4)
        checkpoint = dataset_root / "last.ckpt"
        checkpoint.write_bytes(b"synthetic")
        original = ml_worker.checkpoint_class_count
        ml_worker.checkpoint_class_count = lambda _: 2
        try:
            result = ml_worker.validate_resume_checkpoint(checkpoint, dataset_root)
        finally:
            ml_worker.checkpoint_class_count = original
        self.assertIsNone(result)

    def test_validate_resume_checkpoint_allows_through_when_undetermined(self):
        # Conservative default: don't block an otherwise-working resume just because
        # the checkpoint's internal shape couldn't be read.
        dataset_root = self._dataset_root_with_categories(2)
        checkpoint = dataset_root / "last.ckpt"
        checkpoint.write_bytes(b"synthetic")
        original = ml_worker.checkpoint_class_count
        ml_worker.checkpoint_class_count = lambda _: None
        try:
            result = ml_worker.validate_resume_checkpoint(checkpoint, dataset_root)
        finally:
            ml_worker.checkpoint_class_count = original
        self.assertEqual(result, checkpoint)

    # Regression coverage for a second, related crash: RF-DETR's model_class(...)
    # infers num_classes from whatever pretrain_weights checkpoint it's given (or
    # defaults to 90 COCO classes) instead of from the current dataset, so training
    # silently ran with too few classes and crashed deep in loss matching. train()
    # now always passes an explicit num_classes computed from the dataset, filtered
    # through call_with_supported_kwargs so older RF-DETR installs that don't accept
    # the kwarg still work.
    def test_num_classes_is_passed_through_when_supported(self):
        class Model:
            def __init__(self, device, num_classes):
                self.device = device
                self.num_classes = num_classes

        model, ignored = ml_worker.call_with_supported_kwargs(
            Model, {"device": "cpu", "num_classes": 3}
        )
        self.assertEqual(model.num_classes, 3)
        self.assertEqual(ignored, [])

    def test_num_classes_is_dropped_for_older_constructors_without_it(self):
        class Model:
            def __init__(self, device):
                self.device = device

        model, ignored = ml_worker.call_with_supported_kwargs(
            Model, {"device": "cpu", "num_classes": 3}
        )
        self.assertEqual(model.device, "cpu")
        self.assertEqual(ignored, ["num_classes"])

    def test_num_classes_passes_through_a_kwargs_catchall_constructor(self):
        class Model:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        model, ignored = ml_worker.call_with_supported_kwargs(
            Model, {"device": "cpu", "num_classes": 3}
        )
        self.assertEqual(model.kwargs, {"device": "cpu", "num_classes": 3})
        self.assertEqual(ignored, [])

    def test_training_logs_are_preserved_in_unique_history_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "runs" / "small"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.csv").write_text("val/mAP_50_95\n0.42\n")
            (run_dir / "training_config.json").write_text('{"epochs": 20}')

            archived = ml_worker.archive_run_logs(run_dir)

            self.assertIsNotNone(archived)
            self.assertEqual((archived / "metrics.csv").read_text(), "val/mAP_50_95\n0.42\n")
            self.assertEqual((archived / "training_config.json").read_text(), '{"epochs": 20}')

    def test_old_export_signature_does_not_receive_new_optional_arguments(self):
        def old_export(output_dir="output", format="onnx", **kwargs):
            return output_dir, format, kwargs

        supported = ml_worker.explicitly_supported_kwargs(
            old_export,
            {
                "output_name": "custom-name",
                "coreml_precision": "float16",
                "notes": {"privacy": "local"},
            },
        )
        self.assertEqual(supported, {})

    def test_new_export_signature_receives_only_declared_options(self):
        def new_export(
            output_dir="output",
            format="onnx",
            *,
            coreml_precision=None,
            notes=None,
        ):
            return output_dir, format, coreml_precision, notes

        supported = ml_worker.explicitly_supported_kwargs(
            new_export,
            {
                "output_name": "custom-name",
                "coreml_precision": "float16",
                "notes": {"privacy": "local"},
            },
        )
        self.assertEqual(
            supported,
            {"coreml_precision": "float16", "notes": {"privacy": "local"}},
        )

    def test_exchange_package_contains_no_media_or_source_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "runs" / "nano" / "checkpoint_best_total.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-weights")
            project = {
                "schemaVersion": 2,
                "name": "Private Match",
                "sport": "basketball",
                "sourceFolder": "/private/videos/secret",
                "createdAt": "2026-08-28T10:00:00Z",
                "updatedAt": "2026-08-28T10:00:00Z",
                "frames": [self.make_frame("private-video", 1, root)],
            }
            ml_worker.atomic_json(root / "project.json", project)
            args = type("Args", (), {"project": str(root), "model": "nano", "language": "en"})()
            ml_worker.package_model(args)
            package = next((root / "exports" / "share").glob("*.recomodel"))
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "weights/checkpoint_best_total.pth"})
                manifest = archive.read("manifest.json").decode().lower()
                parsed_manifest = json.loads(manifest)
                self.assertTrue(parsed_manifest["description"].startswith("locally fine-tuned rf-detr nano model"))
                self.assertNotIn("secret", manifest)
                self.assertNotIn("private-video", manifest)
                self.assertNotIn("sourcefolder", manifest)

    def test_exchange_package_can_be_verified_installed_and_activated_locally(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            target = Path(temporary) / "target"
            checkpoint = source / "runs" / "nano" / "checkpoint_best_total.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-exchange-weights")
            for root in (source, target):
                root.mkdir(parents=True, exist_ok=True)
                ml_worker.atomic_json(root / "project.json", {
                    "schemaVersion": 2,
                    "name": "Synthetic Basketball",
                    "sport": "basketball",
                    "sourceFolder": str(root),
                    "createdAt": "2026-08-29T10:00:00Z",
                    "updatedAt": "2026-08-29T10:00:00Z",
                    "frames": [self.make_frame("synthetic", 1, root)],
                })
            package_args = type("Args", (), {"project": str(source), "model": "nano", "language": "es"})()
            ml_worker.package_model(package_args)
            package = next((source / "exports" / "share").glob("*.recomodel"))
            manifest = ml_worker.validate_model_package_file(package, "fr")
            install_args = type("Args", (), {"project": str(target), "file": str(package), "language": "es"})()
            ml_worker.install_model_package(install_args)
            active = ml_worker.load_json(target / "models" / "active.json")
            installed = ml_worker.newest_checkpoint(target, "nano")
            self.assertEqual(active["packageID"], manifest["packageID"])
            self.assertIsNotNone(installed)
            self.assertEqual(installed.read_bytes(), b"synthetic-exchange-weights")
            self.assertFalse(any((target / "models").rglob("*.jpg")))

    def test_exchange_package_for_another_sport_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            target = Path(temporary) / "target"
            checkpoint = source / "runs" / "nano" / "checkpoint.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"weights")
            source.mkdir(parents=True, exist_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            ml_worker.atomic_json(source / "project.json", {"sport": "basketball", "frames": []})
            ml_worker.atomic_json(target / "project.json", {"sport": "football", "frames": []})
            ml_worker.package_model(type("Args", (), {"project": str(source), "model": "nano", "language": "en"})())
            package = next((source / "exports" / "share").glob("*.recomodel"))
            with self.assertRaises(SystemExit):
                ml_worker.install_model_package(type("Args", (), {"project": str(target), "file": str(package), "language": "fr"})())


if __name__ == "__main__":
    unittest.main()
