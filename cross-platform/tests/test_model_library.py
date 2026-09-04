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
