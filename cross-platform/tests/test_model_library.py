import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local_worker = load_module("model_library_local_worker", ROOT / "local_worker.py")
ml_worker = load_module("model_library_ml_worker", ROOT / "ml_worker.py")

try:
    import cv2  # noqa: F401
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class BenchmarkMetricsTests(unittest.TestCase):
    """Regression coverage for the box_iou/box_iou_xywh name collision.

    ml_worker.py used to define two functions named ``box_iou`` (one for xyxy
    boxes used by evaluate_predictions, one for x/y/width/height boxes used by
    the OpenCV box refinement). The second definition silently shadowed the
    first at module scope, so evaluate_predictions computed IoU against the
    wrong box format and every benchmark mAP/precision/recall/F1 number was
    wrong. These tests pin the correct, independent behavior of both helpers.
    """

    def test_box_iou_uses_xyxy_corners(self):
        # Two 10x10 boxes overlapping in a 5x5 corner: intersection 25, union 175.
        overlap = ml_worker.box_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])
        self.assertAlmostEqual(overlap, 25.0 / 175.0)

    def test_box_iou_xywh_uses_x_y_width_height(self):
        # Same geometry expressed as (x, y, width, height).
        overlap = ml_worker.box_iou_xywh((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0))
        self.assertAlmostEqual(overlap, 25.0 / 175.0)

    def test_evaluate_predictions_scores_a_perfect_match_as_one(self):
        ground_truth_frames = [
            {"id": "frame-1", "annotations": [{"category": "ball", "x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}]},
        ]
        predictions = [
            {"frameID": "frame-1", "category": "ball", "confidence": 0.9, "box": [10.0, 10.0, 30.0, 30.0]},
        ]
        metrics = ml_worker.evaluate_predictions(ground_truth_frames, predictions, classes=["ball"])
        self.assertEqual(metrics["truePositives"], 1)
        self.assertEqual(metrics["falsePositives"], 0)
        self.assertEqual(metrics["falseNegatives"], 0)
        self.assertAlmostEqual(metrics["mAP50"], 1.0)
        self.assertAlmostEqual(metrics["meanIoU"], 1.0)

    def test_average_precision_of_perfectly_ranked_detections_is_one(self):
        ap = ml_worker.average_precision([1 / 3, 2 / 3, 1.0], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(ap, 1.0)

    def test_average_precision_penalizes_missed_recall(self):
        # Only one of two ground-truth objects is ever found, at perfect precision:
        # the interpolated AP is the recall reached, since precision is 0 beyond it.
        ap = ml_worker.average_precision([0.5], [1.0])
        self.assertAlmostEqual(ap, 0.5)

    def test_plausible_refined_box_accepts_a_tighter_centered_crop(self):
        original = (100.0, 100.0, 20.0, 20.0)
        candidate = (102.0, 102.0, 16.0, 16.0)
        self.assertTrue(ml_worker.plausible_refined_box(original, candidate, "ball"))

    def test_plausible_refined_box_rejects_a_tiny_candidate(self):
        original = (100.0, 100.0, 20.0, 20.0)
        candidate = (100.0, 100.0, 2.0, 2.0)
        self.assertFalse(ml_worker.plausible_refined_box(original, candidate, "ball"))

    def test_plausible_refined_box_rejects_an_implausible_area_change(self):
        original = (100.0, 100.0, 20.0, 20.0)
        candidate = (90.0, 90.0, 60.0, 60.0)  # 9x the original area
        self.assertFalse(ml_worker.plausible_refined_box(original, candidate, "ball"))

    def test_plausible_refined_box_aspect_ratio_tolerance_is_category_specific(self):
        # A wide, flat box (aspect 4.0) is implausible for a round ball but within
        # the wider tolerance intentionally given to pucks, viewed edge-on more often.
        original = (100.0, 100.0, 20.0, 20.0)
        candidate = (100.0, 108.0, 20.0, 5.0)
        self.assertFalse(ml_worker.plausible_refined_box(original, candidate, "ball"))
        self.assertTrue(ml_worker.plausible_refined_box(original, candidate, "puck"))

    def test_detection_category_map_merges_multiple_selected_categories(self):
        mapping = ml_worker.detection_category_map(["ball", "player"])
        self.assertEqual(mapping["sports ball"], "ball")
        self.assertEqual(mapping["person"], "player")
        mapping_custom = ml_worker.detection_category_map(["hoop"])
        self.assertEqual(mapping_custom["hoop"], "hoop")
        self.assertNotIn("sports ball", mapping_custom)

    def test_plausible_refined_box_covers_every_sport_category(self):
        # CATEGORY_ASPECT_BOUNDS must cover every category any sport_categories()
        # entry can produce, or refinable_annotations() would silently exclude it.
        all_categories = {
            "ball", "player", "referee", "hoop",  # basketball
            "puck", "goalkeeper", "goal",  # hockey (+ football/handball/etc.)
            "goalpost",  # rugby / american football
        }
        self.assertTrue(all_categories.issubset(ml_worker.CATEGORY_ASPECT_BOUNDS.keys()))

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

    def test_plausible_refined_box_rejects_unrealistic_player_aspect(self):
        # An upright player box (tall, narrow) refined into a very wide box is
        # implausible - could be two overlapping players merged into one blob.
        original = (100.0, 100.0, 20.0, 60.0)
        wide_candidate = (95.0, 100.0, 40.0, 30.0)  # aspect 1.33, outside player bounds
        narrow_candidate = (100.0, 100.0, 15.0, 60.0)  # aspect 0.25, within player bounds
        self.assertFalse(ml_worker.plausible_refined_box(original, wide_candidate, "player"))
        self.assertTrue(ml_worker.plausible_refined_box(original, narrow_candidate, "player"))

    def test_evaluate_predictions_rejects_low_overlap_boxes(self):
        # A 20x20 ground-truth box vs. a same-size prediction shifted by 25px:
        # IoU is below the 0.5 threshold, so this must count as FP + FN, not a match.
        ground_truth_frames = [
            {"id": "frame-1", "annotations": [{"category": "ball", "x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0}]},
        ]
        predictions = [
            {"frameID": "frame-1", "category": "ball", "confidence": 0.9, "box": [25.0, 0.0, 45.0, 20.0]},
        ]
        metrics = ml_worker.evaluate_predictions(ground_truth_frames, predictions, classes=["ball"])
        self.assertEqual(metrics["truePositives"], 0)
        self.assertEqual(metrics["falsePositives"], 1)
        self.assertEqual(metrics["falseNegatives"], 1)


class DoctorCudaDetectionTests(unittest.TestCase):
    """Regression coverage for a real report: a Windows user with a working
    NVIDIA GPU had no way to tell from the app whether it was being used -
    doctor() only ever checked torch.backends.mps.is_available() (Mac-only)
    and hardcoded every other case to "cpu", even though detect_device()
    (the function actually used to pick the training/inference device) has
    always checked CUDA correctly. This only tests that doctor() reads
    torch.cuda's real API when present; it fakes the "torch" module entirely
    since doctor() imports it directly rather than through a patchable
    module-level indirection.
    """

    def _install_fake_torch(self, *, cuda_available: bool, device_name: str = "NVIDIA GeForce RTX 4070"):
        import importlib.machinery
        module = types.ModuleType("torch")
        # find_spec("torch") consults sys.modules[name].__spec__ once the name
        # is already loaded - a bare ModuleType has none, which would make
        # doctor() see torch as "not installed" and skip CUDA detection
        # entirely despite the fake module being right there.
        module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        module.cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_name=lambda _index=0: device_name,
        )
        sys.modules["torch"] = module
        self.addCleanup(sys.modules.pop, "torch", None)

    def test_reports_cuda_available_and_device_name(self):
        self._install_fake_torch(cuda_available=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ml_worker.doctor(argparse.Namespace())
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["cudaAvailable"])
        self.assertEqual(payload["gpuName"], "NVIDIA GeForce RTX 4070")
        self.assertEqual(payload["recommendedDevice"], "cuda")

    def test_reports_cpu_when_no_cuda_device_is_available(self):
        self._install_fake_torch(cuda_available=False)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ml_worker.doctor(argparse.Namespace())
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["cudaAvailable"])
        self.assertIsNone(payload["gpuName"])
        self.assertEqual(payload["recommendedDevice"], "cpu")


class ResumeCheckpointValidationTests(unittest.TestCase):
    """Regression coverage for a real crash: a strict Lightning resume
    (trainer.fit(ckpt_path=...)) hard-crashes with a state_dict size-mismatch
    RuntimeError when the checkpoint's detection head was trained for a different
    number of classes than the project currently has - e.g. after a class was added
    via auto-labeling while a training run sat interrupted. validate_resume_checkpoint
    is what train() now uses to detect that and fall back to a normal fine-tuning
    cycle instead of letting the whole run abort.
    """

    def _dataset_root_with_categories(self, count: int) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        ml_worker.atomic_json(root / "train" / "_annotations.coco.json", {
            "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": f"class-{i}"} for i in range(count)],
        })
        return root

    def test_returns_none_when_no_checkpoint_was_offered(self):
        dataset_root = self._dataset_root_with_categories(2)
        self.assertIsNone(ml_worker.validate_resume_checkpoint(None, dataset_root))

    def test_allows_a_matching_checkpoint_through(self):
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

    def test_rejects_a_checkpoint_trained_for_a_different_class_count(self):
        # This is the exact scenario from the crash: interrupted at 2 classes,
        # the project now has 4 after auto-labeling added more categories.
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

    def test_allows_the_checkpoint_through_when_class_count_cant_be_determined(self):
        # Conservative default: don't block an otherwise-working resume just because
        # the checkpoint's internal shape couldn't be read (e.g. no torch installed,
        # or a future RF-DETR version uses a different state_dict key).
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

    def test_dataset_class_count_reads_the_coco_categories_list(self):
        dataset_root = self._dataset_root_with_categories(3)
        self.assertEqual(ml_worker.dataset_class_count(dataset_root, "train"), 3)
        self.assertIsNone(ml_worker.dataset_class_count(dataset_root, "missing-split"))


class AutoLabelActiveModelTests(unittest.TestCase):
    """Regression coverage for a real report: auto-label ran a full inference pass
    for "referee" against a model that was only ever trained on "ball" (activated
    before "referee" was added to the project), silently found nothing for it, and
    folded that into a single misleading "0 boxes" instead of explaining that this
    specific model doesn't know that class yet. The old check only asked "does any
    checkpoint exist at all", not "does this activated model's own manifest cover
    this class" - active_model_classes() is what auto_label() now consults instead.
    """

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


class FieldGeometryTests(unittest.TestCase):
    """Coverage for the field-boundary filter: by request, auto-label should only
    add "player"/"referee"/"goalkeeper" boxes for people whose feet fall inside
    the project's marked field (four image corners + real dimensions), so a
    spectator or bench player standing nearby doesn't silently become "player".

    The test geometry is deliberately axis-aligned (corners form a square, not a
    perspective-distorted quad) so the expected real-world coordinates can be
    worked out by hand: frame pixels (100,100)-(900,900) map to field meters
    (0,0)-(20,20), a uniform scale of 0.025 m/px with a (-100,-100)px offset.
    """

    def _square_field_geometry(self) -> dict:
        return {
            "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "realWidth": 20.0,
            "realLength": 20.0,
        }

    def test_returns_none_without_geometry(self):
        self.assertIsNone(ml_worker.field_membership_checker(None))
        self.assertIsNone(ml_worker.field_membership_checker({}))

    def test_returns_none_for_incomplete_or_invalid_geometry(self):
        self.assertIsNone(ml_worker.field_membership_checker({
            "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],  # only 3 corners
            "realWidth": 20.0, "realLength": 20.0,
        }))
        self.assertIsNone(ml_worker.field_membership_checker({
            "corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "realWidth": 0.0, "realLength": 20.0,  # zero real width
        }))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_accepts_a_point_inside_the_marked_field_and_rejects_outside(self):
        is_on_field = ml_worker.field_membership_checker(self._square_field_geometry())
        self.assertIsNotNone(is_on_field)
        # Frame center (500, 500) -> field (10, 10): well inside the 20x20 field.
        self.assertTrue(is_on_field("player", 480.0, 400.0, 520.0, 500.0, 1000.0, 1000.0))
        # Near the frame's top-left corner (50, 50), well outside the marked
        # square (which starts at pixel 100,100) -> negative field coordinates.
        self.assertFalse(is_on_field("player", 30.0, 20.0, 70.0, 50.0, 1000.0, 1000.0))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is not installed in this environment")
    def test_only_filters_person_shaped_categories(self):
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


class RefereeClothingTests(unittest.TestCase):
    """Coverage for the clothing-based referee heuristic: by request, a generic
    "player" detection whose crop matches the sport's typical referee attire
    (see REFEREE_CLOTHING_PROFILES) gets reclassified to "referee" - still
    source="auto", so it goes through the same manual review as any other
    automatic suggestion rather than being trusted blindly.
    """

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


class BuildDatasetCategoryFilterTests(unittest.TestCase):
    """Coverage for training-on-a-subset: by request, a local training run can
    be restricted to a chosen subset of already-annotated categories (e.g.
    "only ball and referee this time"). build_dataset()'s categories parameter
    must drop both the excluded category from the dataset's category list AND
    every individual box of that category - not just hide it from the list.
    """

    def _frame_with_annotations(self, root: Path, categories: list[str]) -> dict:
        relative = "frames/one.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg")
        return {
            "id": "frame-one", "relativePath": relative, "videoID": "v1", "videoName": "clip.mov",
            "timestamp": 0.0, "width": 1920, "height": 1080,
            "annotations": [
                {"id": f"ann-{index}", "category": category, "x": 1.0, "y": 1.0, "width": 8.0, "height": 8.0, "source": "manual"}
                for index, category in enumerate(categories)
            ],
        }

    def test_restricts_to_the_given_categories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self._frame_with_annotations(root, ["ball", "referee"])
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Test", "sport": "basketball",
                "sourceFolder": str(root), "frames": [frame],
            })

            dataset = ml_worker.build_dataset(root, categories=["ball"])

            coco = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
            self.assertEqual([category["name"] for category in coco["categories"]], ["ball"])
            self.assertEqual(len(coco["annotations"]), 1)

    def test_rejects_a_category_subset_with_no_matching_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = self._frame_with_annotations(root, ["ball"])
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Test", "sport": "basketball",
                "sourceFolder": str(root), "frames": [frame],
            })

            with self.assertRaises(SystemExit):
                ml_worker.build_dataset(root, categories=["referee"])


class CallWithSupportedKwargsModelConstructionTests(unittest.TestCase):
    """Regression coverage for a second, related crash: RF-DETR's model_class(...)
    infers num_classes from whatever pretrain_weights checkpoint it's given (or
    defaults to 90 COCO classes) instead of from the current dataset, so training
    silently ran with too few classes and crashed deep in loss matching. train()
    now always passes an explicit num_classes computed from the dataset, filtered
    through call_with_supported_kwargs so older RF-DETR installs that don't accept
    the kwarg still work.
    """

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


class ModelLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / ".reco-training"
        self.root.mkdir()
        self.document = {
            "schemaVersion": 2,
            "sport": "basketball",
            "frames": [{"reviewStatus": "reviewed", "annotations": [{"category": "ball"}]}],
        }
        ml_worker.atomic_json(self.root / "project.json", self.document)
        self.checkpoint = self.root / "runs" / "small" / "checkpoint_best_total.pth"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"first-model-revision")

    def tearDown(self):
        self.temporary.cleanup()

    def test_archives_activates_and_renames_immutable_revision(self):
        summary = {
            "validationMetrics": {"val/mAP_50_95": 0.435},
            "testMetrics": {"test/mAP_50_95": 0.403},
        }
        manifest_path, manifest, created = ml_worker.archive_local_checkpoint(
            self.root, self.document, "small", self.checkpoint, summary, "local-training"
        )
        self.assertTrue(created)
        active = ml_worker.activate_library_model(self.root, manifest_path, manifest)
        self.assertEqual(active["packageID"], manifest["packageID"])
        self.assertEqual((manifest_path.parent / "weights" / self.checkpoint.name).read_bytes(), b"first-model-revision")

        args = argparse.Namespace(project=str(self.root), package_id=manifest["packageID"], name="Basketball Small Best", language="en")
        with contextlib.redirect_stdout(io.StringIO()):
            ml_worker.rename_model(args)
        renamed = json.loads(manifest_path.read_text())
        self.assertEqual(renamed["displayName"], "Basketball Small Best")

        snapshot = local_worker.model_library_snapshot(self.root)
        self.assertEqual(snapshot["activePackageID"], manifest["packageID"])
        self.assertTrue(snapshot["packages"][0]["isBest"])
        self.assertAlmostEqual(snapshot["packages"][0]["validationScore"], 0.435)
        self.assertAlmostEqual(snapshot["packages"][0]["testScore"], 0.403)

    def test_same_weights_are_not_archived_twice(self):
        first_path, _, _ = ml_worker.archive_local_checkpoint(
            self.root, self.document, "small", self.checkpoint, {}, "local-training"
        )
        second_path, _, created = ml_worker.archive_local_checkpoint(
            self.root, self.document, "small", self.checkpoint, {}, "local-training"
        )
        self.assertFalse(created)
        self.assertEqual(first_path, second_path)

    def test_reads_latest_independent_test_metrics(self):
        (self.checkpoint.parent / "metrics.csv").write_text(
            "epoch,test/mAP_50_95,test/F1,test/precision\n"
            "1,0.336,0.687,0.672\n"
            "2,0.435,0.875,0.875\n",
            encoding="utf-8",
        )
        metrics = ml_worker.latest_test_metrics(self.checkpoint.parent)
        self.assertAlmostEqual(metrics["test/mAP_50_95"], 0.435)
        self.assertAlmostEqual(metrics["test/F1"], 0.875)

    def test_package_name_becomes_portable_file_component(self):
        self.assertEqual(ml_worker.package_file_slug("My Basketball Model #1"), "My-Basketball-Model-1")
        self.assertEqual(ml_worker.package_file_slug("  ...  "), "reco-model")
        self.assertLessEqual(len(ml_worker.package_file_slug("x" * 100)), 60)

    def test_exchange_package_uses_requested_name(self):
        args = argparse.Namespace(project=str(self.root), model="small", name="Hall Model #4", language="en")
        with contextlib.redirect_stdout(io.StringIO()):
            ml_worker.package_model(args)
        packages = list((self.root / "exports" / "share").glob("*.recomodel"))
        self.assertEqual(len(packages), 1)
        self.assertTrue(packages[0].name.startswith("Hall-Model-4-"))
        with zipfile.ZipFile(packages[0]) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["displayName"], "Hall Model #4")


class HeldOutFrameTests(unittest.TestCase):
    """A held-out frame (extracted via "Unabhängiger Modelltest", never part of
    the training folder) must never enter the training dataset - that's the
    actual leakage guard behind the independent model test, not just a label.
    """

    def _frame(self, root: Path, frame_id: str, category: str, held_out: bool = False) -> dict:
        relative = f"frames/{frame_id}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg")
        frame = {
            "id": frame_id, "relativePath": relative, "videoID": "v1", "videoName": "clip.mov",
            "timestamp": 0.0, "width": 1920, "height": 1080,
            "annotations": [{"id": "ann-0", "category": category, "x": 1.0, "y": 1.0, "width": 8.0, "height": 8.0, "source": "manual"}],
        }
        if held_out:
            frame["heldOut"] = True
        return frame

    def test_held_out_frames_are_excluded_from_the_training_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainable = self._frame(root, "trainable", "ball")
            held_out = self._frame(root, "held-out", "ball", held_out=True)
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Test", "sport": "basketball",
                "sourceFolder": str(root), "frames": [trainable, held_out],
            })

            dataset = ml_worker.build_dataset(root)

            coco = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
            image_names = {image["file_name"] for image in coco["images"]}
            self.assertIn("trainable.jpg", image_names)
            self.assertNotIn("held-out.jpg", image_names)

    def test_only_held_out_frames_count_for_training_dataset_is_empty_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            held_out = self._frame(root, "held-out", "ball", held_out=True)
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Test", "sport": "basketball",
                "sourceFolder": str(root), "frames": [held_out],
            })

            with self.assertRaises(SystemExit):
                ml_worker.build_dataset(root)


class CombineModelsTests(unittest.TestCase):
    """Coverage for "ein neues Modell backen": combine two already-installed
    library models into one, each contributing only the categories it was
    assigned - an ensemble at inference time, not real weight merging.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / ".reco-training"
        self.root.mkdir()
        ml_worker.atomic_json(self.root / "project.json", {
            "schemaVersion": 2, "sport": "basketball", "frames": [],
        })
        self.referee_id = self._install("referee-model", "nano", ["referee"], b"referee-weights")
        self.ball_id = self._install("ball-model", "nano", ["ball", "player"], b"ball-weights")

    def tearDown(self):
        self.temporary.cleanup()

    def _install(self, label: str, model_size: str, classes: list[str], weight_bytes: bytes) -> str:
        document = {
            "schemaVersion": 2, "sport": "basketball",
            "frames": [{"reviewStatus": "reviewed", "annotations": [{"category": category} for category in classes]}],
        }
        checkpoint = self.root / "runs" / f"{label}.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(weight_bytes)
        _, manifest, _ = ml_worker.archive_local_checkpoint(self.root, document, model_size, checkpoint, {}, "local-training")
        return str(manifest["packageID"])

    def test_combine_builds_a_manifest_with_per_category_members(self):
        args = argparse.Namespace(
            project=str(self.root),
            member=[f"{self.referee_id}:referee", f"{self.ball_id}:ball,player"],
            name="Combo", language="en",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            ml_worker.combine_models(args)

        active = ml_worker.load_json(self.root / "models" / "active.json")
        manifest = ml_worker.load_json(self.root / "models" / "library" / active["packageID"] / "manifest.json")
        self.assertEqual(manifest["schemaVersion"], 2)
        self.assertEqual(sorted(manifest["classes"]), ["ball", "player", "referee"])
        members = manifest["ensemble"]["members"]
        self.assertEqual(len(members), 2)
        by_id = {member["packageID"]: member for member in members}
        self.assertEqual(by_id[self.referee_id]["categories"], ["referee"])
        self.assertEqual(sorted(by_id[self.ball_id]["categories"]), ["ball", "player"])
        for member in members:
            weight_path = self.root / "models" / "library" / active["packageID"] / member["weightsFile"]
            self.assertTrue(weight_path.is_file())
            self.assertEqual(ml_worker.sha256_file(weight_path), member["sha256"])

    def test_rejects_fewer_than_two_members(self):
        args = argparse.Namespace(project=str(self.root), member=[f"{self.referee_id}:referee"], name="", language="en")
        with self.assertRaises(SystemExit):
            ml_worker.combine_models(args)

    def test_rejects_a_category_assigned_to_two_members(self):
        args = argparse.Namespace(
            project=str(self.root),
            member=[f"{self.referee_id}:referee", f"{self.ball_id}:ball,referee"],
            name="", language="en",
        )
        with self.assertRaises(SystemExit):
            ml_worker.combine_models(args)

    def test_rejects_a_category_the_member_model_does_not_support(self):
        args = argparse.Namespace(
            project=str(self.root),
            member=[f"{self.referee_id}:referee,hoop", f"{self.ball_id}:ball"],
            name="", language="en",
        )
        with self.assertRaises(SystemExit):
            ml_worker.combine_models(args)

    def test_rejects_mismatched_model_sizes(self):
        large_ball_id = self._install("ball-model-large", "small", ["ball"], b"ball-weights-small")
        args = argparse.Namespace(
            project=str(self.root),
            member=[f"{self.referee_id}:referee", f"{large_ball_id}:ball"],
            name="", language="en",
        )
        with self.assertRaises(SystemExit):
            ml_worker.combine_models(args)


class EnsemblePackageValidationTests(unittest.TestCase):
    """A combined/ensemble .recomodel bundles one weights file per member
    instead of the usual single manifest+weights pair - validate_model_package_file()
    must accept that shape (schemaVersion 2) while still requiring every
    member's own checksum to match, and must keep rejecting a malformed one.
    """

    def _write_ensemble_package(self, path: Path, *, corrupt_checksum: bool = False) -> None:
        manifest = {
            "schemaVersion": 2,
            "packageID": "reco-ensemble-basketball-nano-20260101-000000",
            "displayName": "Combo", "sport": "basketball", "modelSize": "nano",
            "classes": ["ball", "referee"],
            "ensemble": {"members": [
                {"packageID": "referee-model", "modelSize": "nano", "categories": ["referee"], "weightsFile": "weights/referee-model.pth", "sha256": ml_worker.hashlib.sha256(b"referee-weights").hexdigest()},
                {"packageID": "ball-model", "modelSize": "nano", "categories": ["ball"], "weightsFile": "weights/ball-model.pth", "sha256": "0" * 64 if corrupt_checksum else ml_worker.hashlib.sha256(b"ball-weights").hexdigest()},
            ]},
        }
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("weights/referee-model.pth", b"referee-weights")
            archive.writestr("weights/ball-model.pth", b"ball-weights")

    def test_accepts_a_valid_ensemble_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "combo.recomodel"
            self._write_ensemble_package(package)
            manifest = ml_worker.validate_model_package_file(package, "en")
            self.assertEqual(manifest["ensemble"]["members"][0]["packageID"], "referee-model")

    def test_rejects_an_ensemble_package_with_a_bad_member_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "combo.recomodel"
            self._write_ensemble_package(package, corrupt_checksum=True)
            with self.assertRaises(SystemExit):
                ml_worker.validate_model_package_file(package, "en")


class WeightsPickleSafetyTests(unittest.TestCase):
    """A matching checksum only proves a weights file matches what its own
    manifest claims - it does not prove the file is safe, since an attacker
    controls both the file and its checksum. verify_weights_pickle_safety()
    is the actual protection: it requires torch.load(..., weights_only=True),
    PyTorch's own allowlist-based safe loader, to accept the file before it is
    trusted. Real torch is not a test dependency here (matches the existing
    DoctorCudaDetectionTests pattern above), so a fake "torch" module is
    injected to test both outcomes of that call, and the ImportError path is
    tested by simply *not* injecting one - torch is not installed in this
    test environment either, so that path runs for real.
    """

    def _install_fake_torch(self, *, accepts: bool):
        import importlib.machinery
        module = types.ModuleType("torch")
        module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        def fake_load(*_args, **_kwargs):
            if not accepts:
                raise RuntimeError("Weights only load failed: disallowed global found")
            return {}
        module.load = fake_load
        sys.modules["torch"] = module
        self.addCleanup(sys.modules.pop, "torch", None)

    def test_accepts_a_file_torch_reports_as_safe(self):
        self._install_fake_torch(accepts=True)
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "model.pth"
            weights.write_bytes(b"looks-like-tensor-data")
            ml_worker.verify_weights_pickle_safety(weights, "en", "model.pth")

    def test_rejects_a_file_torch_reports_as_unsafe(self):
        self._install_fake_torch(accepts=False)
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "model.pth"
            weights.write_bytes(b"pickled-exploit-payload")
            with self.assertRaises(SystemExit):
                ml_worker.verify_weights_pickle_safety(weights, "en", "model.pth")

    def test_skips_gracefully_when_torch_is_not_installed(self):
        sys.modules.pop("torch", None)
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "model.pth"
            weights.write_bytes(b"anything")
            # Must not raise - checksum validation (tested separately) is the
            # only guarantee available without torch, and that is acceptable
            # degraded behavior, not a silent bypass of the checksum check.
            ml_worker.verify_weights_pickle_safety(weights, "en", "model.pth")

    def test_verify_zip_weight_entry_rejects_a_checksum_mismatch_before_scanning_content(self):
        self._install_fake_torch(accepts=True)
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "one.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("weights/one.pth", b"real-bytes")
            with zipfile.ZipFile(package) as archive:
                with self.assertRaises(SystemExit):
                    ml_worker.verify_zip_weight_entry(archive, "weights/one.pth", "0" * 64, "en", "one")

    def test_verify_zip_weight_entry_rejects_unsafe_content_with_a_correct_checksum(self):
        self._install_fake_torch(accepts=False)
        payload = b"real-bytes"
        digest = ml_worker.hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "one.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("weights/one.pth", payload)
            with zipfile.ZipFile(package) as archive:
                with self.assertRaises(SystemExit):
                    ml_worker.verify_zip_weight_entry(archive, "weights/one.pth", digest, "en", "one")

    def test_verify_zip_weight_entry_accepts_matching_checksum_and_safe_content(self):
        self._install_fake_torch(accepts=True)
        payload = b"real-bytes"
        digest = ml_worker.hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "one.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("weights/one.pth", payload)
            with zipfile.ZipFile(package) as archive:
                ml_worker.verify_zip_weight_entry(archive, "weights/one.pth", digest, "en", "one")

    def test_validate_model_package_file_rejects_a_package_whose_weights_are_unsafe(self):
        self._install_fake_torch(accepts=False)
        payload = b"real-bytes"
        manifest = {
            "schemaVersion": 1, "packageID": "reco-basketball-nano-1", "sport": "basketball",
            "modelSize": "nano", "classes": ["ball"],
            "weights": {"file": "weights/checkpoint.pth", "sha256": ml_worker.hashlib.sha256(payload).hexdigest()},
        }
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "model.recomodel"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                archive.writestr("weights/checkpoint.pth", payload)
            with self.assertRaises(SystemExit):
                ml_worker.validate_model_package_file(package, "en")

    def test_rejects_a_symlink_entry_even_with_a_correct_checksum(self):
        payload = b"real-bytes"
        manifest = {
            "schemaVersion": 1, "packageID": "reco-basketball-nano-1", "sport": "basketball",
            "modelSize": "nano", "classes": ["ball"],
            "weights": {"file": "weights/checkpoint.pth", "sha256": ml_worker.hashlib.sha256(payload).hexdigest()},
        }
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "model.recomodel"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                info = zipfile.ZipInfo("weights/checkpoint.pth")
                info.external_attr = (0o120777 << 16)  # S_IFLNK
                archive.writestr(info, payload)
            with self.assertRaises(SystemExit):
                ml_worker.validate_model_package_file(package, "en")


class ModelInstanceSafetyGateTests(unittest.TestCase):
    """model_instance() is the single place almost every real model-loading
    call site funnels through - the mandatory verify_weights_pickle_safety()
    call added there (alongside equivalent ones in ensemble_member_model(),
    benchmark(), flag_ensemble_disagreement(), and export_model()) is the
    actual gate between an on-disk weights file and RF-DETR's own unsafe
    (non-weights_only) torch.load() - regardless of whether that exact file
    was already scanned once at install time (it may not have been, e.g. if
    torch wasn't set up yet back then, or the file reached the library some
    other way).
    """

    def _install_fake_torch(self, *, accepts: bool):
        import importlib.machinery
        module = types.ModuleType("torch")
        module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        module.cuda = types.SimpleNamespace(is_available=lambda: False)
        def fake_load(*_args, **_kwargs):
            if not accepts:
                raise RuntimeError("Weights only load failed: disallowed global found")
            return {}
        module.load = fake_load
        sys.modules["torch"] = module
        self.addCleanup(sys.modules.pop, "torch", None)

    def _project_with_checkpoint(self, root: Path) -> Path:
        ml_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
        checkpoint = root / "runs" / "nano" / "checkpoint_best_total.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"pretend-checkpoint-bytes")
        return checkpoint

    def test_rejects_an_unsafe_checkpoint_before_constructing_the_model(self):
        self._install_fake_torch(accepts=False)
        class FakeModel:
            def __init__(self, **_kwargs):
                raise AssertionError("must not construct the model for a rejected checkpoint")
        original = ml_worker.import_model_class
        ml_worker.import_model_class = lambda *_a, **_k: FakeModel
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._project_with_checkpoint(root)
                with self.assertRaises(SystemExit):
                    ml_worker.model_instance(root, "nano", trained=True, language="en")
        finally:
            ml_worker.import_model_class = original

    def test_constructs_the_model_when_the_checkpoint_is_safe(self):
        self._install_fake_torch(accepts=True)
        class FakeModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        original = ml_worker.import_model_class
        ml_worker.import_model_class = lambda *_a, **_k: FakeModel
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                checkpoint = self._project_with_checkpoint(root)
                model = ml_worker.model_instance(root, "nano", trained=True, language="en")
                self.assertEqual(model.kwargs["pretrain_weights"], str(checkpoint))
        finally:
            ml_worker.import_model_class = original


class AutoLabelEnsembleDispatchTests(unittest.TestCase):
    """Using a combined/ensemble model: auto_label() must run each member only
    for the categories it was assigned when the model was combined, not the
    single-checkpoint path a normal active model uses.
    """

    def test_auto_label_dispatches_each_member_for_its_own_categories(self):
        class RefereeDetections:
            data = {"class_name": ["referee"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0]]
            class_id = [0]
            confidence = [.9]

        class BallDetections:
            data = {"class_name": ["ball"]}
            xyxy = [[40.0, 10.0, 60.0, 70.0]]
            class_id = [0]
            confidence = [.8]

        class RefereeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [RefereeDetections() for _ in paths]

        class BallModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [BallDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {
                "sport": "basketball",
                "frames": [{"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": []}],
            })
            package_id = "reco-ensemble-basketball-nano-test"
            weights_dir = root / "models" / "library" / package_id / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "referee-model.pth").write_bytes(b"referee-weights")
            (weights_dir / "ball-model.pth").write_bytes(b"ball-weights")
            manifest = {
                "schemaVersion": 2, "packageID": package_id, "sport": "basketball", "modelSize": "nano",
                "classes": ["ball", "referee"],
                "ensemble": {"members": [
                    {"packageID": "referee-model", "modelSize": "nano", "categories": ["referee"], "weightsFile": "weights/referee-model.pth", "sha256": "irrelevant-for-this-test"},
                    {"packageID": "ball-model", "modelSize": "nano", "categories": ["ball"], "weightsFile": "weights/ball-model.pth", "sha256": "irrelevant-for-this-test"},
                ]},
            }
            ml_worker.atomic_json(weights_dir.parent / "manifest.json", manifest)
            ml_worker.atomic_json(root / "models" / "active.json", {"packageID": package_id, "modelSize": "nano"})

            # Both members share modelSize "nano" here, so import_model_class()
            # alone can't tell them apart - patch ensemble_member_model()
            # directly instead, dispatching on which member is being loaded.
            original_ensemble_member_model = ml_worker.ensemble_member_model
            original_device = ml_worker.detect_device
            original_emit = ml_worker.emit
            try:
                def fake_ensemble_member_model(package_dir, member, language="de"):
                    return RefereeModel() if member["packageID"] == "referee-model" else BallModel()
                ml_worker.ensemble_member_model = fake_ensemble_member_model
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.emit = lambda message: None
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["ball", "referee"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.ensemble_member_model = original_ensemble_member_model
                ml_worker.detect_device = original_device
                ml_worker.emit = original_emit

            document = ml_worker.load_json(root / "project.json")
            categories = {item["category"] for item in document["frames"][0]["annotations"]}
            self.assertEqual(categories, {"ball", "referee"})


class SimulateBallTrackingTests(unittest.TestCase):
    """Coverage for the ball-tracking simulation player's backend: run the
    active model over an already-extracted frame sequence (extraction itself
    is platform-specific glue done by the caller, not ml_worker.py) and
    return raw per-frame ball detections as a single JSON blob - no emit()
    progress text mixed in, matching the same single-JSON contract
    installModelPackage()/activateModel() already use in MLWorker.swift.
    """

    class _Detections:
        def __init__(self, names, boxes, confidences):
            self.data = {"class_name": names}
            self.xyxy = boxes
            self.class_id = [0] * len(names)
            self.confidence = confidences

    def test_returns_ball_center_per_frame_and_skips_non_ball_detections(self):
        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                results = []
                for path in paths:
                    if path.endswith("one.jpg"):
                        results.append(SimulateBallTrackingTests._Detections(
                            ["sports ball", "person"], [[10.0, 10.0, 20.0, 20.0], [1.0, 1.0, 2.0, 2.0]], [.9, .8],
                        ))
                    else:
                        results.append(SimulateBallTrackingTests._Detections([], [], []))
                return results

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            ml_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
            frames_dir = Path(temporary) / "frames"
            frames_dir.mkdir()
            (frames_dir / "one.jpg").write_bytes(b"fake")
            (frames_dir / "two.jpg").write_bytes(b"fake")

            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                args = type("Args", (), {
                    "project": str(root), "frames_dir": str(frames_dir),
                    "model": "nano", "threshold": 0.25, "language": "en",
                })()
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    ml_worker.simulate_ball_tracking(args)
                payload = json.loads(buffer.getvalue())
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device

            by_file = {item["file"]: item["ball"] for item in payload["frames"]}
            self.assertAlmostEqual(by_file["one.jpg"]["x"], 15.0)
            self.assertAlmostEqual(by_file["one.jpg"]["y"], 15.0)
            self.assertIsNone(by_file["two.jpg"])

    def test_rejects_a_missing_frames_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ml_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
            args = type("Args", (), {
                "project": str(root), "frames_dir": str(root / "missing"),
                "model": "nano", "threshold": 0.25, "language": "en",
            })()
            with self.assertRaises(SystemExit):
                ml_worker.simulate_ball_tracking(args)

    def test_rejects_an_empty_frames_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ml_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
            frames_dir = root / "frames"
            frames_dir.mkdir()
            args = type("Args", (), {
                "project": str(root), "frames_dir": str(frames_dir),
                "model": "nano", "threshold": 0.25, "language": "en",
            })()
            with self.assertRaises(SystemExit):
                ml_worker.simulate_ball_tracking(args)


class ComputeDisagreementTests(unittest.TestCase):
    """compute_disagreement() is the pure IoU-matching core of the
    ensemble-disagreement review-priority signal (flag_ensemble_disagreement)."""

    def test_identical_boxes_have_zero_disagreement(self):
        box = {"category": "ball", "x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}
        self.assertEqual(ml_worker.compute_disagreement([box], [dict(box)]), 0.0)

    def test_both_empty_is_zero_disagreement(self):
        self.assertEqual(ml_worker.compute_disagreement([], []), 0.0)

    def test_one_sided_detection_is_total_disagreement(self):
        box = {"category": "ball", "x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}
        self.assertEqual(ml_worker.compute_disagreement([box], []), 1.0)
        self.assertEqual(ml_worker.compute_disagreement([], [box]), 1.0)

    def test_non_overlapping_boxes_of_the_same_category_both_count_as_unmatched(self):
        primary = [{"category": "ball", "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}]
        secondary = [{"category": "ball", "x": 500.0, "y": 500.0, "width": 10.0, "height": 10.0}]
        self.assertEqual(ml_worker.compute_disagreement(primary, secondary), 1.0)

    def test_different_categories_never_match_even_with_identical_geometry(self):
        primary = [{"category": "ball", "x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}]
        secondary = [{"category": "player", "x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}]
        self.assertEqual(ml_worker.compute_disagreement(primary, secondary), 1.0)

    def test_partial_overlap_below_iou_threshold_counts_as_unmatched(self):
        primary = [{"category": "ball", "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}]
        secondary = [{"category": "ball", "x": 8.0, "y": 8.0, "width": 10.0, "height": 10.0}]
        self.assertEqual(ml_worker.compute_disagreement(primary, secondary, iou_threshold=0.3), 1.0)


class FlagTemporalOutliersTests(unittest.TestCase):
    """flag_temporal_outliers() adapts the neighbor-interpolation idea behind
    BallTrackingResolver.swift to sparse, unevenly-sampled training frames."""

    @staticmethod
    def _frame(frame_id, timestamp, cx, cy, review_status="candidate", category="ball"):
        return {
            "id": frame_id,
            "videoID": "video-1",
            "timestamp": timestamp,
            "width": 1000.0,
            "height": 1000.0,
            "reviewStatus": review_status,
            "annotations": [{"id": f"ann-{frame_id}", "category": category, "x": cx - 5, "y": cy - 5, "width": 10.0, "height": 10.0, "source": "auto"}],
        }

    def test_flags_a_position_far_from_the_interpolated_line(self):
        frames = [
            self._frame("f1", 0.0, 100.0, 100.0),
            self._frame("f2", 1.0, 900.0, 900.0),  # way off the line between f1 and f3
            self._frame("f3", 2.0, 200.0, 200.0),
        ]
        ml_worker.flag_temporal_outliers(frames)
        self.assertEqual(frames[0].get("reviewFlags", []), [])
        self.assertIn("temporal-outlier", frames[1]["reviewFlags"])
        self.assertEqual(frames[2].get("reviewFlags", []), [])

    def test_a_position_on_the_interpolated_line_is_not_flagged(self):
        frames = [
            self._frame("f1", 0.0, 100.0, 100.0),
            self._frame("f2", 1.0, 150.0, 150.0),
            self._frame("f3", 2.0, 200.0, 200.0),
        ]
        ml_worker.flag_temporal_outliers(frames)
        self.assertEqual(frames[1].get("reviewFlags", []), [])

    def test_leading_and_trailing_frames_without_both_neighbors_are_never_flagged(self):
        frames = [
            self._frame("f1", 0.0, 900.0, 900.0),
            self._frame("f2", 1.0, 100.0, 100.0),
            self._frame("f3", 2.0, 110.0, 110.0),
            self._frame("f4", 3.0, 900.0, 900.0),
        ]
        ml_worker.flag_temporal_outliers(frames)
        self.assertEqual(frames[0].get("reviewFlags", []), [])
        self.assertEqual(frames[3].get("reviewFlags", []), [])

    def test_frames_with_zero_or_multiple_boxes_of_the_category_are_skipped(self):
        multi = self._frame("f2", 1.0, 900.0, 900.0)
        multi["annotations"].append({"id": "ann-extra", "category": "ball", "x": 1.0, "y": 1.0, "width": 5.0, "height": 5.0, "source": "auto"})
        frames = [
            self._frame("f1", 0.0, 100.0, 100.0),
            multi,
            self._frame("f3", 2.0, 200.0, 200.0),
        ]
        ml_worker.flag_temporal_outliers(frames)
        self.assertEqual(multi.get("reviewFlags", []), [])

    def test_a_gap_wider_than_the_max_is_never_bridged(self):
        frames = [
            self._frame("f1", 0.0, 100.0, 100.0),
            self._frame("f2", 5.0, 900.0, 900.0),
            self._frame("f3", 10.0, 200.0, 200.0),
        ]
        ml_worker.flag_temporal_outliers(frames, max_gap_seconds=5.0)
        self.assertEqual(frames[1].get("reviewFlags", []), [])

    def test_only_candidate_frames_get_flagged_even_when_a_reviewed_neighbor_is_the_outlier(self):
        frames = [
            self._frame("f1", 0.0, 100.0, 100.0),
            self._frame("f2", 1.0, 900.0, 900.0, review_status="reviewed"),
            self._frame("f3", 2.0, 200.0, 200.0),
        ]
        ml_worker.flag_temporal_outliers(frames)
        self.assertEqual(frames[1].get("reviewFlags", []), [])


if __name__ == "__main__":
    unittest.main()
