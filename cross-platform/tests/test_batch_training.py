import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import ml_worker


def box(x: float, y: float, category: str = "player") -> dict:
    return {"category": category, "x": x, "y": y, "width": 10.0, "height": 10.0}


def prediction(frame: str, x: float, y: float, confidence: float, category: str = "player") -> dict:
    return {"frameID": frame, "category": category, "confidence": confidence, "box": [x, y, x + 10.0, y + 10.0]}


class SweepValidationFramesTests(unittest.TestCase):
    def test_studio_split_is_used_instead_of_the_project_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "dataset" / "valid").mkdir(parents=True)
            (root / "dataset" / "valid" / "_annotations.coco.json").write_text(json.dumps({
                "categories": [{"id": 1, "name": "player"}],
                "images": [{"id": 0, "file_name": "real0-a.png"}, {"id": 1, "file_name": "real0-empty.png"}],
                "annotations": [{"image_id": 0, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            }), encoding="utf-8")
            document = {"augmentation": {"mode": "mixed"}, "frames": [{"id": "project", "annotations": [box(0, 0)]}]}
            frames, images = ml_worker.sweep_validation_frames(root, document, "en")
        self.assertEqual(images, root / "dataset" / "valid")
        self.assertEqual(frames, [{"id": "0", "relativePath": "dataset/valid/real0-a.png", "annotations": [{"category": "player", "x": 1, "y": 2, "width": 3, "height": 4}]}])


class ConfidenceSweepTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{"id": "a", "annotations": [box(0, 0), box(50, 0), box(100, 100, "ball")]}]

    def predictions(self):
        return [
            prediction("a", 0, 0, 0.9), prediction("a", 50, 0, 0.8),
            prediction("a", 200, 200, 0.3), prediction("a", 300, 300, 0.2),  # wrong, only at low confidence
            prediction("a", 100, 100, 0.25, "ball"),  # right, but the ball is unsure
        ]

    def test_the_best_threshold_drops_wrong_boxes_and_keeps_right_ones(self):
        result = ml_worker.sweep_thresholds(self.frames, self.predictions(), ["player"])
        self.assertGreaterEqual(result["bestThreshold"], 0.35)
        self.assertLessEqual(result["bestThreshold"], 0.8)
        self.assertEqual(result["f1"], 1.0)

    def test_every_class_gets_its_own_threshold(self):
        result = ml_worker.sweep_thresholds(self.frames, self.predictions(), ["player", "ball"])
        self.assertLessEqual(result["perClass"]["ball"]["threshold"], 0.25)
        self.assertGreaterEqual(result["perClass"]["player"]["threshold"], 0.35)

    def test_without_predictions_nothing_breaks(self):
        result = ml_worker.sweep_thresholds(self.frames, [], ["player"])
        self.assertEqual(result["f1"], 0.0)


class RecommendationTests(unittest.TestCase):
    def test_the_best_model_wins(self):
        results = [{"model": "nano", "status": "done", "mAP50": 0.70}, {"model": "small", "status": "done", "mAP50": 0.80}]
        self.assertEqual(ml_worker.recommended_model(results), "small")

    def test_a_smaller_model_wins_when_the_difference_is_within_one_point(self):
        results = [{"model": "small", "status": "done", "mAP50": 0.795}, {"model": "medium", "status": "done", "mAP50": 0.80}]
        self.assertEqual(ml_worker.recommended_model(results), "small")

    def test_failed_models_do_not_count(self):
        results = [{"model": "nano", "status": "failed"}, {"model": "small", "status": "done", "mAP50": 0.5}, {"model": "medium", "status": "skipped"}]
        self.assertEqual(ml_worker.recommended_model(results), "small")
        self.assertIsNone(ml_worker.recommended_model([{"model": "nano", "status": "failed"}]))


class ProfileTests(unittest.TestCase):
    def test_larger_models_never_get_a_bigger_batch_than_small(self):
        for device in ("mps", "cuda", "cpu"):
            for memory in (8, 16, 32, 64, 128):
                bytes_total = memory * 1024**3
                small = ml_worker.training_profile("small", device, bytes_total, 8)["batch_size"]
                for size in ("medium", "large"):
                    profile = ml_worker.training_profile(size, device, bytes_total, 8)
                    self.assertLessEqual(profile["batch_size"], small)
                    self.assertGreaterEqual(profile["batch_size"], 1)

    def test_all_sizes_can_be_trained(self):
        self.assertEqual(list(ml_worker.MODEL_CLASSES), ["nano", "small", "medium", "large"])


class BatchRunTests(unittest.TestCase):
    """The run itself, with a stand-in for the training step: failures and order, no ML needed."""

    def run_batch(self, models: str, fail: set[str]) -> tuple[dict, str]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "project.json").write_text(json.dumps({"schemaVersion": 1, "sport": "basketball", "frames": []}), encoding="utf-8")
            calls = []

            def fake_step(command, prefix, index, count):
                calls.append((command[2], command[command.index("--model") + 1]))
                if command[2] == "train" and command[command.index("--model") + 1] in fail:
                    return 1, None, ["out of memory"]
                if command[2] == "confidence-sweep":
                    return 0, json.dumps({"mAP50": {"nano": 0.5, "small": 0.7, "medium": 0.9}[command[command.index("--model") + 1]], "bestThreshold": 0.4}), []
                return 0, None, []

            original = ml_worker.run_batch_step
            ml_worker.run_batch_step = fake_step
            try:
                args = type("Args", (), {"project": str(root), "models": models, "epochs": 5, "auto_epochs": True, "patience": 10, "fresh_start": True, "sweep": True, "anonymize_fraction": 0.0, "language": "en"})()
                original_awake = ml_worker.keep_awake
                ml_worker.keep_awake = lambda: None
                try:
                    ml_worker.train_batch(args)
                finally:
                    ml_worker.keep_awake = original_awake
            finally:
                ml_worker.run_batch_step = original
            return json.loads((root / "batch-training" / "latest.json").read_text(encoding="utf-8")), calls

    def test_smallest_first_and_a_failure_does_not_end_the_night(self):
        report, calls = self.run_batch("medium,small,nano", {"small"})
        self.assertEqual([item["model"] for item in report["models"]], ["nano", "small", "medium"])
        self.assertEqual(report["models"][1]["status"], "failed")
        self.assertEqual(report["models"][1]["error"], "out of memory")
        self.assertEqual(report["models"][2]["status"], "done")
        self.assertEqual(report["recommended"], "medium")
        self.assertTrue(report["finishedAt"])
        self.assertNotIn(("confidence-sweep", "small"), calls)


class StaticContractTests(unittest.TestCase):
    def test_a_training_dialog_opens_instead_of_training_right_away(self):
        page = (ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        self.assertIn("setTrainDialogOpen(true)", page)
        self.assertNotIn("onClick={() => void runML('/api/train'", page)


if __name__ == "__main__":
    unittest.main()
