from __future__ import annotations

import importlib.util
import json
import stat
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

    def test_candidate_requires_human_decision_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".reco-training"
            candidate = {
                "id": "candidate", "relativePath": "frames/candidate.jpg", "reviewStatus": "candidate",
                "annotations": [{"id": "auto", "category": "ball", "x": 1, "y": 2, "width": 3, "height": 4, "source": "auto"}],
            }
            project = {"sport": "basketball", "frames": [candidate]}
            local_worker.atomic_json(root / "project.json", project)
            local_worker.STATE.update(project_root=root, project=project)

            local_worker.review_candidate({"frameId": "candidate", "decision": "ball", "category": "ball"})

            self.assertEqual(candidate["reviewStatus"], "reviewed")
            self.assertEqual(candidate["annotations"][0]["source"], "manual")

    def test_no_ball_candidate_becomes_reviewed_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".reco-training"
            candidate = {
                "id": "candidate", "relativePath": "frames/candidate.jpg", "reviewStatus": "candidate",
                "annotations": [{"id": "auto", "category": "ball", "x": 1, "y": 2, "width": 3, "height": 4, "source": "auto"}],
            }
            project = {"sport": "basketball", "frames": [candidate]}
            local_worker.atomic_json(root / "project.json", project)
            local_worker.STATE.update(project_root=root, project=project)

            local_worker.review_candidate({"frameId": "candidate", "decision": "no-ball", "category": "ball"})

            self.assertEqual(candidate["reviewStatus"], "reviewed")
            self.assertEqual(candidate["annotations"], [])


class PythonVersionWarningTests(unittest.TestCase):
    """python_version_warning() replaced an inline Python check that used to live in
    Start Reco Preview Windows.bat, where it broke the launcher on real Windows
    (cmd.exe aborted the whole script instead of just printing the warning). Doing
    the check here in plain Python, instead of batch, is what makes it testable.
    """

    def setUp(self):
        self.original_version_info = local_worker.sys.version_info

    def tearDown(self):
        local_worker.sys.version_info = self.original_version_info

    def test_no_warning_for_3_11_or_3_12(self):
        local_worker.sys.version_info = (3, 11, 5, "final", 0)
        self.assertIsNone(local_worker.python_version_warning())
        local_worker.sys.version_info = (3, 12, 0, "final", 0)
        self.assertIsNone(local_worker.python_version_warning())

    def test_warns_for_other_versions(self):
        local_worker.sys.version_info = (3, 13, 0, "final", 0)
        warning = local_worker.python_version_warning()
        self.assertIsNotNone(warning)
        self.assertIn("3.13", warning)


class FrameExtractionTests(unittest.TestCase):
    """Covers extract_frames_for_video, the helper factored out of extract_project
    and expand_dataset (which used to build the same ffmpeg/Apple-extractor command
    and the same even-height scaling logic twice, independently)."""

    def _make_executable(self, path: Path, script: str) -> None:
        path.write_text(script)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def test_uses_apple_extractor_and_scales_dimensions_to_even_height(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "frames"
            output_dir.mkdir()
            extractor = root / "fake-extractor.py"
            self._make_executable(extractor, (
                "#!/usr/bin/env python3\n"
                "import sys, pathlib\n"
                "video, output_dir, prefix, count = sys.argv[1:5]\n"
                "for index in range(1, int(count) + 1):\n"
                "    (pathlib.Path(output_dir) / f'{prefix}-{index:06d}.jpg').write_bytes(b'x')\n"
            ))
            video = root / "clip.mov"
            video.write_bytes(b"fake video")

            generated, rate, output_width, output_height = local_worker.extract_frames_for_video(
                video, duration=10.0, width=1920, height=1081, target_count=5,
                output_dir=output_dir, prefix="vid1",
                ffmpeg=None, apple_extractor=extractor,
                failure_message="Extraction failed",
            )

            self.assertEqual(len(generated), 5)
            self.assertEqual(rate, 0.5)
            # 1280/1920 is an exact 2/3 scale factor for this width.
            self.assertEqual(output_width, 1280)
            self.assertEqual(output_height % 2, 0)

    def test_raises_with_video_name_and_stderr_on_extractor_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "frames"
            output_dir.mkdir()
            extractor = root / "failing-extractor.py"
            self._make_executable(extractor, "#!/usr/bin/env python3\nimport sys\nsys.stderr.write('boom')\nsys.exit(1)\n")
            video = root / "clip.mov"
            video.write_bytes(b"fake video")

            with self.assertRaises(RuntimeError) as context:
                local_worker.extract_frames_for_video(
                    video, duration=10.0, width=100, height=100, target_count=5,
                    output_dir=output_dir, prefix="vid1",
                    ffmpeg=None, apple_extractor=extractor,
                    failure_message="Extraction failed",
                )
            self.assertIn("Extraction failed", str(context.exception))
            self.assertIn("clip.mov", str(context.exception))
            self.assertIn("boom", str(context.exception))


if __name__ == "__main__":
    unittest.main()
