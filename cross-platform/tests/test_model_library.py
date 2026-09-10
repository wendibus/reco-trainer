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
