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

    def test_frame_count_is_per_video_and_safely_bounded(self):
        self.assertEqual(local_worker.frames_per_video(None), 240)
        self.assertEqual(local_worker.frames_per_video("500"), 500)
        self.assertEqual(local_worker.frames_per_video(1), 4)
        self.assertEqual(local_worker.frames_per_video(50_000), 5_000)

    def test_remove_frame_deletes_only_derived_files_and_invalidates_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            root = folder / ".reco-training"
            frame_path = root / "frames" / "synthetic.jpg"
            source_video = folder / "source.mp4"
            frame_path.parent.mkdir(parents=True)
            frame_path.write_bytes(b"synthetic test frame")
            source_video.write_bytes(b"synthetic source video")
            for split in ("train", "valid", "test"):
                split_file = root / "dataset" / split / frame_path.name
                split_file.parent.mkdir(parents=True)
                split_file.write_bytes(b"derived copy")
            benchmark = root / "benchmarks"
            benchmark.mkdir(parents=True)
            (benchmark / "ground-truth.json").write_text("{}")
            (benchmark / "latest.json").write_text("{}")
            project = {
                "sport": "basketball",
                "sourceFolder": str(folder),
                "frames": [{"id": "remove-me", "relativePath": "frames/synthetic.jpg", "annotations": []}],
            }
            local_worker.atomic_json(root / "project.json", project)
            local_worker.STATE.update(selected_folder=folder, project_root=root, project=project)

            local_worker.remove_training_frame({"frameId": "remove-me"})

            self.assertTrue(source_video.exists())
            self.assertFalse(frame_path.exists())
            self.assertEqual(local_worker.STATE.project["frames"], [])
            self.assertFalse((benchmark / "ground-truth.json").exists())
            self.assertFalse((benchmark / "latest.json").exists())
            self.assertTrue(any((root / "backups").glob("project-*.json")))


if __name__ == "__main__":
    unittest.main()
