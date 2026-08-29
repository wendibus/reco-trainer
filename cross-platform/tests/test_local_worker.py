from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


WORKER_PATH = Path(__file__).parents[1] / "local_worker.py"
SPEC = importlib.util.spec_from_file_location("local_worker", WORKER_PATH)
assert SPEC and SPEC.loader
local_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_worker)


class GroundTruthTests(unittest.TestCase):
    def setUp(self):
        local_worker.STATE.update(selected_folder=None, project_root=None, project=None, busy=False, error=None)

    def test_freeze_creates_explicit_negative_frames_and_dataset_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".reco-training"
            project = {
                "sport": "basketball",
                "frames": [
                    {"id": "positive", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": [{"category": "ball", "x": 1, "y": 2, "width": 3, "height": 4, "source": "manual"}]},
                    {"id": "negative", "relativePath": "frames/two.jpg", "width": 100, "height": 80, "annotations": []},
                ],
            }
            local_worker.STATE.update(project_root=root, project=project)
            status = local_worker.freeze_ground_truth()
            frozen = json.loads((root / "benchmarks" / "ground-truth.json").read_text())
            self.assertEqual(status["groundTruth"]["frameCount"], 2)
            self.assertEqual(status["groundTruth"]["annotationCount"], 1)
            self.assertEqual(frozen["frames"][1]["annotations"], [])
            self.assertEqual(len(frozen["datasetID"]), 64)
            self.assertNotIn("videoName", json.dumps(frozen))

    def test_freeze_rejects_unreviewed_automatic_suggestions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".reco-training"
            project = {"sport": "basketball", "frames": [{"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": [{"category": "ball", "x": 1, "y": 2, "width": 3, "height": 4, "source": "auto"}]}]}
            local_worker.STATE.update(project_root=root, project=project)
            with self.assertRaises(RuntimeError):
                local_worker.freeze_ground_truth()


if __name__ == "__main__":
    unittest.main()
