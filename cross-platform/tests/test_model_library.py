import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
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


if __name__ == "__main__":
    unittest.main()
