from __future__ import annotations

import os as _os
import tempfile as _tempfile
import importlib.util
import json
import re
import stat
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image


WORKER_PATH = Path(__file__).parents[1] / "local_worker.py"
SPEC = importlib.util.spec_from_file_location("local_worker", WORKER_PATH)
assert SPEC and SPEC.loader
local_worker = importlib.util.module_from_spec(SPEC)
# Never touch the real per-computer ML environment from a test.
_os.environ.setdefault("RECO_ML_ENV", str(Path(_tempfile.mkdtemp(prefix="reco-ml-env-")) / "ml-env"))
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
                    {"id": "positive", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": [{"category": "ball", "x": 1, "y": 2, "width": 3, "height": 4, "source": "manual"}], "heldOut": True},
                    {"id": "negative", "relativePath": "frames/two.jpg", "width": 100, "height": 80, "annotations": [], "heldOut": True},
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
        self.assertEqual(local_worker.frames_per_video(None), 60)
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

    def test_no_warning_for_3_10_and_newer(self):
        for minor in (10, 13):
            local_worker.sys.version_info = (3, minor, 0, "final", 0)
            self.assertIsNone(local_worker.python_version_warning())

    def test_warns_for_old_python_when_no_suitable_python_exists(self):
        original = local_worker.system_python

        def none_found():
            raise RuntimeError("none")

        local_worker.system_python = none_found
        try:
            local_worker.sys.version_info = (3, 9, 6, "final", 0)
            warning = local_worker.python_version_warning()
        finally:
            local_worker.system_python = original
        self.assertIsNotNone(warning)
        self.assertIn("3.9", warning)

    def test_no_warning_for_old_python_when_a_newer_one_is_installed(self):
        original = local_worker.system_python
        local_worker.system_python = lambda: Path("/opt/homebrew/bin/python3.12")
        try:
            local_worker.sys.version_info = (3, 9, 6, "final", 0)
            self.assertIsNone(local_worker.python_version_warning())
        finally:
            local_worker.system_python = original


class SystemPythonSelectionTests(unittest.TestCase):
    """rfdetr>=1.9 cannot be installed on Python 3.9 (pip: "No matching distribution
    found"), so ML setup must pick an interpreter >= 3.10 instead of the first python3.
    """

    def setUp(self):
        self.original_candidates = local_worker.python_candidates
        self.original_version_of = local_worker.python_version_of

    def tearDown(self):
        local_worker.python_candidates = self.original_candidates
        local_worker.python_version_of = self.original_version_of

    def _fake(self, root: Path, versions: dict[str, tuple[int, int]]):
        paths = []
        for name in versions:
            path = root / name
            path.write_text("")
            path.chmod(0o755)
            paths.append(path)
        local_worker.python_candidates = lambda: paths
        local_worker.python_version_of = lambda path: versions.get(path.name)

    def test_skips_python_3_9_and_picks_the_first_new_enough_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._fake(Path(temporary), {"python3": (3, 9), "python3.12": (3, 12), "python3.13": (3, 13)})
            self.assertEqual(local_worker.system_python().name, "python3.12")

    def test_raises_a_helpful_error_when_only_old_pythons_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._fake(Path(temporary), {"python3": (3, 9)})
            with self.assertRaises(RuntimeError) as raised:
                local_worker.system_python()
            self.assertIn("3.10", str(raised.exception))


class OutdatedVenvTests(unittest.TestCase):
    """A venv created earlier with Python 3.9 is rebuilt instead of failing on pip again."""

    def setUp(self):
        local_worker.STATE.update(selected_folder=None, project_root=None, project=None, busy=False, error=None, log=[])
        self.original_run_logged = local_worker.run_logged
        self.original_version_of = local_worker.python_version_of
        self.original_system_python = local_worker.system_python
        self.commands: list[list[str]] = []
        local_worker.run_logged = lambda command: self.commands.append(command)
        local_worker.system_python = lambda: Path("/opt/homebrew/bin/python3.12")
        local_worker.detect_nvidia_gpu_name = lambda: None
        self.original_torch_import_error = local_worker.torch_import_error
        local_worker.torch_import_error = lambda path: None

    def tearDown(self):
        local_worker.run_logged = self.original_run_logged
        local_worker.python_version_of = self.original_version_of
        local_worker.system_python = self.original_system_python
        local_worker.torch_import_error = self.original_torch_import_error

    def _project_with_venv(self, root: Path) -> Path:
        local_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
        venv = root / ".runtime" / "venv"
        venv_python = local_worker.venv_python_path(venv)
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("")
        local_worker.STATE.update(project_root=root)
        return venv

    def test_recreates_a_venv_built_with_python_3_9(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv = self._project_with_venv(root)
            local_worker.python_version_of = lambda path: (3, 9)

            local_worker.ml_action("setup", {"useCuda": False})

            self.assertTrue(any(command[1:3] == ["-m", "venv"] for command in self.commands))
            self.assertFalse(local_worker.venv_python_path(venv).exists())

    def test_keeps_a_venv_that_is_new_enough(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project_with_venv(root)
            local_worker.python_version_of = lambda path: (3, 12)

            local_worker.ml_action("setup", {"useCuda": False})

            self.assertFalse(any(command[1:3] == ["-m", "venv"] for command in self.commands))


class TorchFallbackTests(unittest.TestCase):
    """A CUDA PyTorch that installs but cannot load (WinError 127 on c10_cuda.dll, reported on
    Windows) must not leave a broken ML environment: Set up ML switches to the CPU build."""

    def setUp(self):
        local_worker.STATE.update(log=[])
        self.original_run_logged = local_worker.run_logged
        self.original_error = local_worker.torch_import_error
        self.commands: list[list[str]] = []
        local_worker.run_logged = lambda command: self.commands.append(command)

    def tearDown(self):
        local_worker.run_logged = self.original_run_logged
        local_worker.torch_import_error = self.original_error

    def test_working_torch_is_left_alone(self):
        local_worker.torch_import_error = lambda path: None
        local_worker.ensure_torch_importable(Path("/venv/python"))
        self.assertEqual(self.commands, [])

    def test_unloadable_torch_is_replaced_by_the_cpu_build(self):
        results = iter(["OSError: [WinError 127] c10_cuda.dll", None])
        local_worker.torch_import_error = lambda path: next(results)
        with mock.patch.object(local_worker.sys, "platform", "win32"):
            local_worker.ensure_torch_importable(Path("/venv/python"))
        self.assertIn("uninstall", self.commands[0])
        self.assertIn(local_worker.CPU_TORCH_INDEX_URL, self.commands[1])
        self.assertEqual(len(self.commands), 2)

    def test_error_when_even_the_cpu_build_fails(self):
        local_worker.torch_import_error = lambda path: "OSError: still broken"
        with self.assertRaises(RuntimeError) as raised:
            local_worker.ensure_torch_importable(Path("/venv/python"))
        self.assertIn("still broken", str(raised.exception))
        self.assertIn("Visual C++", str(raised.exception))

    def test_import_check_reports_the_last_error_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            broken = Path(temporary) / "python"
            broken.write_text("#!/bin/sh\necho 'Traceback' >&2\necho 'OSError: [WinError 127] boom' >&2\nexit 1\n")
            broken.chmod(0o755)
            self.assertEqual(local_worker.torch_import_error(broken), "OSError: [WinError 127] boom")
            working = Path(temporary) / "ok"
            working.write_text("#!/bin/sh\nexit 0\n")
            working.chmod(0o755)
            self.assertIsNone(local_worker.torch_import_error(working))


class SharedMlEnvTests(unittest.TestCase):
    """The ML environment is set up once per computer and found again in every project."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.shared = self.base / "ml-env"
        self.original_env = _os.environ.get("RECO_ML_ENV")
        _os.environ["RECO_ML_ENV"] = str(self.shared)
        self.original_version_of = local_worker.python_version_of

    def tearDown(self):
        if self.original_env is None:
            _os.environ.pop("RECO_ML_ENV", None)
        else:
            _os.environ["RECO_ML_ENV"] = self.original_env
        local_worker.python_version_of = self.original_version_of
        self.temporary.cleanup()

    def _venv(self, venv: Path) -> Path:
        python = local_worker.venv_python_path(venv)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("")
        return venv

    def test_a_new_project_without_any_environment_points_to_the_shared_one(self):
        self.assertEqual(local_worker.ml_venv(self.base / "a"), self.shared / "venv")

    def test_the_shared_environment_wins_in_every_project(self):
        self._venv(self.shared / "venv")
        self._venv(self.base / "a" / ".runtime" / "venv")
        self.assertEqual(local_worker.ml_venv(self.base / "a"), self.shared / "venv")
        self.assertEqual(local_worker.ml_venv(self.base / "b"), self.shared / "venv")

    def test_an_older_project_environment_is_remembered_for_other_projects(self):
        old = self._venv(self.base / "a" / ".runtime" / "venv")
        self.assertEqual(local_worker.ml_venv(self.base / "a"), old)
        self.assertEqual(local_worker.ml_venv(self.base / "b"), old)

    def test_a_remembered_environment_that_is_gone_falls_back_to_the_shared_one(self):
        old = self._venv(self.base / "a" / ".runtime" / "venv")
        local_worker.ml_venv(self.base / "a")
        local_worker.venv_python_path(old).unlink()
        self.assertEqual(local_worker.ml_venv(self.base / "b"), self.shared / "venv")

    def test_setup_never_replaces_a_too_old_environment_of_another_project(self):
        old = self._venv(self.base / "a" / ".runtime" / "venv")
        local_worker.ml_venv(self.base / "a")
        local_worker.python_version_of = lambda path: (3, 9)
        self.assertEqual(local_worker.ml_setup_target(self.base / "b", old), self.shared / "venv")
        local_worker.python_version_of = lambda path: (3, 12)
        self.assertEqual(local_worker.ml_setup_target(self.base / "b", old), old)


class EmptyProjectTests(unittest.TestCase):
    """A dataset can be imported into a working folder that was never prepared."""

    def test_starts_a_project_without_frames_for_the_chosen_sport(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            root = folder / ".reco-training"
            local_worker.start_empty_project(folder, root, "basketball")
            project = json.loads((root / "project.json").read_text(encoding="utf-8"))
            self.assertEqual((project["sport"], project["frames"], project["name"]), ("basketball", [], folder.name))
            self.assertTrue((root / "frames").is_dir())

    def test_rejects_an_unknown_sport(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                local_worker.start_empty_project(Path(temporary), Path(temporary) / ".reco-training", "chess")


class VenvPythonPathTests(unittest.TestCase):
    """Regression coverage for a real report: a Windows user's "Set Up ML" click
    failed with "[WinError 2] The system cannot find the file specified" -
    ml_action() hardcoded the POSIX venv layout (venv/bin/python3) for every
    platform, which never exists on Windows (venv/Scripts/python.exe there), so
    every subprocess call after venv creation failed outright.
    """

    def setUp(self):
        self.original_name = local_worker.os.name

    def tearDown(self):
        local_worker.os.name = self.original_name

    def test_posix_layout(self):
        # A plain concrete Path here (not PurePosixPath) also exercises that the
        # "posix" branch works with whatever real Path type callers pass in.
        local_worker.os.name = "posix"
        path = local_worker.venv_python_path(PurePosixPath("/project/.runtime/venv"))
        self.assertEqual(path, PurePosixPath("/project/.runtime/venv/bin/python3"))

    def test_windows_layout(self):
        # PureWindowsPath (not the concrete Path/WindowsPath) so this test can
        # build a Windows-flavored path while actually running on macOS/Linux -
        # a concrete Path refuses to instantiate the "wrong" OS's flavor.
        local_worker.os.name = "nt"
        path = local_worker.venv_python_path(PureWindowsPath("/project/.runtime/venv"))
        self.assertEqual(path, PureWindowsPath("/project/.runtime/venv/Scripts/python.exe"))


class FrameExtractionTests(unittest.TestCase):
    """Covers extract_frames_for_video, the helper factored out of extract_project
    and expand_dataset (which used to build the same ffmpeg/Apple-extractor command
    and the same even-height scaling logic twice, independently)."""

    def _make_executable(self, path: Path, script: str) -> None:
        path.write_text(script)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def test_uses_apple_extractor_and_keeps_native_dimensions(self):
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
            # Frames are no longer downscaled to 1280 px: native size is reported.
            self.assertEqual(output_width, 1920)
            self.assertEqual(output_height, 1081)

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


class CancelExtractionTests(unittest.TestCase):
    """The progress dialog's "Cancel": stop ffmpeg/the extractor, keep the project unchanged."""

    SLOW_EXTRACTOR = (
        "#!/usr/bin/env python3\n"
        "import json, sys, time, pathlib\n"
        "if sys.argv[1] == '--probe':\n"
        "    print(json.dumps({'duration': 10.0, 'width': 64, 'height': 32})); sys.exit(0)\n"
        "video, output_dir, prefix, count = sys.argv[1:5]\n"
        "for index in range(1, int(count) + 1):\n"
        "    (pathlib.Path(output_dir) / f'{prefix}-{index:06d}.jpg').write_bytes(b'x')\n"
        "    time.sleep(0.3)\n"
    )

    def setUp(self):
        local_worker.CANCEL_REQUESTED.clear()
        local_worker.STATE.update(selected_folder=None, project_root=None, project=None, busy=False, error=None, log=[], language="en", detail=None, operation="idle")

    def tearDown(self):
        local_worker.CANCEL_REQUESTED.clear()
        local_worker.STATE.update(language="de", busy=False, operation="idle", detail=None)

    def _folder(self, temporary: str) -> tuple[Path, Path, Path]:
        folder = Path(temporary) / "videos"
        folder.mkdir()
        for name in ("a.mp4", "b.mp4"):
            (folder / name).write_bytes(b"fake video")
        root = local_worker.project_root_for(folder)
        extractor = Path(temporary) / "slow-extractor.py"
        extractor.write_text(self.SLOW_EXTRACTOR)
        extractor.chmod(extractor.stat().st_mode | stat.S_IEXEC)
        local_worker.STATE.update(selected_folder=folder, project_root=root)
        return folder, root, extractor

    def test_run_extraction_command_stops_the_process_on_cancel(self):
        threading = __import__("threading")
        timer = threading.Timer(0.5, local_worker.CANCEL_REQUESTED.set)
        timer.start()
        started = __import__("time").time()
        with self.assertRaises(local_worker.OperationCancelled):
            local_worker.run_extraction_command([sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertLess(__import__("time").time() - started, 5)

    def test_cancel_during_preparation_keeps_the_previous_project(self):
        threading = __import__("threading")
        with tempfile.TemporaryDirectory() as temporary:
            _, root, extractor = self._folder(temporary)
            root.mkdir(parents=True, exist_ok=True)
            previous = {"schemaVersion": 2, "sport": "basketball", "frames": [{"id": "old", "relativePath": "frames/old.jpg", "videoID": "v", "annotations": []}]}
            local_worker.atomic_json(root / "project.json", previous)
            with mock.patch.object(local_worker.shutil, "which", lambda name: None), \
                    mock.patch.object(local_worker, "native_frame_extractor", lambda _root: extractor):
                job = local_worker.JOBS.submit("prepare", local_worker.extract_project, "basketball", 20)
                for _ in range(100):
                    detail = local_worker.STATE.detail or {}
                    if detail.get("phase") == "extract" and detail.get("framesDone", 0) >= 1:
                        break
                    __import__("time").sleep(0.1)
                self.assertTrue(local_worker.STATE.snapshot()["cancellable"])
                self.assertEqual(local_worker.STATE.detail["videoCount"], 2)
                self.assertEqual(local_worker.STATE.detail["framesTarget"], 40)
                local_worker.JOBS.cancel(job["id"])
                for _ in range(100):
                    if not local_worker.JOBS.has_pending():
                        break
                    __import__("time").sleep(0.1)
            self.assertFalse(local_worker.JOBS.has_pending())
            self.assertEqual(local_worker.JOBS.snapshot()["recent"][0]["status"], "cancelled")
            self.assertEqual(local_worker.STATE.operation, "cancelled")
            self.assertFalse(local_worker.STATE.busy)
            self.assertIsNone(local_worker.STATE.error)
            self.assertEqual(json.loads((root / "project.json").read_text()), previous)
            self.assertEqual(list(root.glob("frames.next-*")), [])


class WorkingFolderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.root = self.folder / ".reco-training"
        self.root.mkdir()
        saved = (local_worker.STATE.selected_folder, local_worker.STATE.project_root, local_worker.STATE.project, local_worker.STATE.folder_empty)
        self.addCleanup(lambda: local_worker.STATE.update(selected_folder=saved[0], project_root=saved[1], project=saved[2], folder_empty=saved[3]))
        language = local_worker.STATE.language
        self.addCleanup(lambda: setattr(local_worker.STATE, "language", language))
        local_worker.STATE.language = "de"
        local_worker.STATE.update(selected_folder=self.folder, project_root=self.root, project=None, folder_empty=False, operation="selected", error=None)

    def test_an_empty_folder_counts_as_empty_even_with_a_project_inside(self):
        (self.root / "frames").mkdir()
        (self.root / "frames" / "old.jpg").write_bytes(b"x")
        (self.folder / "notes.txt").write_text("x", encoding="utf-8")
        self.assertTrue(local_worker.folder_is_empty(self.folder))
        (self.folder / "a.JPG").write_bytes(b"x")
        self.assertFalse(local_worker.folder_is_empty(self.folder))

    def test_a_video_is_content(self):
        (self.folder / "game.mp4").write_bytes(b"x")
        self.assertFalse(local_worker.folder_is_empty(self.folder))

    def test_preparing_an_empty_folder_says_what_to_add_and_writes_nothing(self):
        local_worker.extract_project("football", 20)
        self.assertEqual(local_worker.STATE.operation, "error")
        self.assertIn("noch leer", local_worker.STATE.error or "")
        self.assertTrue(local_worker.STATE.folder_empty)
        self.assertFalse((self.root / "project.json").exists())

    def test_only_images_keeps_frames_of_videos_prepared_earlier(self):
        earlier = {"id": "v-1", "videoID": "abc", "relativePath": "frames/abc-000001.jpg", "timestamp": 0.0, "width": 10, "height": 10,
                   "annotations": [{"id": "1", "category": "ball", "x": 1, "y": 1, "width": 2, "height": 2}]}
        (self.root / "project.json").write_text(json.dumps({"schemaVersion": 2, "sport": "football", "frames": [earlier]}), encoding="utf-8")
        Image.new("RGB", (40, 20)).save(self.folder / "a.jpg")
        local_worker.extract_project("football", 20)
        self.assertEqual(local_worker.STATE.operation, "ready", local_worker.STATE.error)
        frames = local_worker.STATE.project["frames"]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]["annotations"], earlier["annotations"])
        self.assertTrue(frames[1]["videoID"].startswith("import-"))


class StaticHintsTests(unittest.TestCase):
    def test_every_button_has_a_hint_and_no_native_tooltip_bypasses_the_switch(self):
        for path in (Path(local_worker.__file__).parent / "app").glob("*.tsx"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(" title={", text, path.name)
            for match in re.finditer(r"<button\b(.{0,40})", text):
                self.assertIn("data-hint=", match.group(1), f"{path.name}: {match.group(0)}")


class RunLoggedEncodingTests(unittest.TestCase):
    """Regression coverage for a real Windows crash: local training died with a
    raw UnicodeEncodeError from inside RF-DETR's rich-rendered metrics tables,
    because a subprocess whose stdout is piped (not a real terminal) falls back
    to the OS locale's preferred encoding - a legacy codepage like cp1252 on
    Windows, not UTF-8 - for both the child's writes and this parent's reads.
    """

    def test_forces_utf8_for_both_the_child_and_this_process(self):
        captured: dict = {}

        class FakeProcess:
            stdout = iter(())

            def wait(self):
                return 0

        def fake_popen(command, **kwargs):
            captured.update(kwargs)
            return FakeProcess()

        original_popen = local_worker.subprocess.Popen
        try:
            local_worker.subprocess.Popen = fake_popen
            local_worker.run_logged(["echo", "hi"])
        finally:
            local_worker.subprocess.Popen = original_popen

        self.assertEqual(captured.get("encoding"), "utf-8")
        self.assertEqual(captured.get("env", {}).get("PYTHONIOENCODING"), "utf-8")


class SimulationSlotTests(unittest.TestCase):
    """Two simulation slots (A, B) for the side-by-side model comparison."""

    def setUp(self):
        local_worker.STATE.update(simulation_videos={}, simulation=None, simulation_b=None, language="en")

    def tearDown(self):
        local_worker.STATE.update(simulation_videos={}, language="de")

    def test_each_slot_has_its_own_frames_directory(self):
        root = Path("/project")
        self.assertEqual(local_worker.simulation_frames_dir(root, "a"), root / ".runtime" / "simulation")
        self.assertEqual(local_worker.simulation_frames_dir(root, "b"), root / ".runtime" / "simulation-b")
        self.assertEqual(local_worker.simulation_slot({"slot": "B"}), "b")
        self.assertEqual(local_worker.simulation_slot({"slot": "x"}), "a")

    def test_same_video_reuses_slot_a_without_a_dialog(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip.mp4"
            video.write_bytes(b"video")
            local_worker.STATE.simulation_videos["a"] = video
            original = local_worker.select_video_file
            local_worker.select_video_file = lambda: self.fail("no dialog expected")
            try:
                self.assertEqual(local_worker.simulation_video({"slot": "b", "sameVideo": True}), video)
            finally:
                local_worker.select_video_file = original

    def test_same_video_without_a_simulation_in_a_fails_clearly(self):
        with self.assertRaises(RuntimeError):
            local_worker.simulation_video({"slot": "b", "sameVideo": True})

    def test_slot_b_result_does_not_replace_slot_a(self):
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = {"sport": "basketball", "frames": []}
            local_worker.STATE.update(project_root=root, project=project, simulation={"id": "keep"})
            venv_python = local_worker.venv_python_path(local_worker.ml_venv(root))
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("")
            video = root / "clip.mp4"
            video.write_bytes(b"video")

            def fake_extract(_video, _duration, _width, _height, _count, frames_dir, prefix, *_rest):
                (frames_dir / f"{prefix}-0001.jpg").write_bytes(b"jpg")
                return 1, 12.0, 320, 180

            def fake_run_json(command, on_progress=None):
                commands.append(command)
                return {"frames": [{"file": "sim-0001.jpg", "ball": None}], "model": "Model B"}

            originals = (local_worker.extract_frames_for_video, local_worker.run_json, local_worker.media_info, local_worker.native_frame_extractor)
            local_worker.extract_frames_for_video = fake_extract
            local_worker.run_json = fake_run_json
            local_worker.media_info = lambda *_: (1.0, 320, 180)
            local_worker.native_frame_extractor = lambda *_: None
            try:
                local_worker.simulate_ball_tracking(video, {"slot": "b", "package": "pkg-b", "language": "en"})
            finally:
                (local_worker.extract_frames_for_video, local_worker.run_json, local_worker.media_info, local_worker.native_frame_extractor) = originals
            self.assertEqual(local_worker.STATE.simulation, {"id": "keep"})
            self.assertEqual(local_worker.STATE.simulation_b["model"], "Model B")
            self.assertEqual(local_worker.STATE.simulation_b["video"], "clip.mp4")
            self.assertIn("--package", commands[0])
            self.assertEqual(commands[0][commands[0].index("--package") + 1], "pkg-b")
            self.assertIn(str(root / ".runtime" / "simulation-b"), commands[0])
            self.assertEqual(local_worker.STATE.simulation_videos["b"], video)
            local_worker.STATE.update(project_root=None, project=None, simulation=None, simulation_b=None)

    def test_rejects_a_package_id_with_a_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_worker.STATE.update(project_root=root, project={"sport": "basketball", "frames": []})
            local_worker.simulate_ball_tracking(root / "clip.mp4", {"slot": "b", "package": "../x", "language": "en"})
            self.assertEqual(local_worker.STATE.operation, "error")
            local_worker.STATE.update(project_root=None, project=None, error=None)


class ConsistencyReviewTests(unittest.TestCase):
    """Step-by-step review of frames flagged by the consistency check."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        box = {"id": "a", "category": "player", "x": 1, "y": 2, "width": 30, "height": 60, "source": "manual"}
        self.project = {"sport": "basketball", "frames": [
            {"id": "f1", "annotations": [box, {**box, "id": "b"}], "reviewFlags": ["vlm-issue"], "consistencySuggestion": {"add": []}, "vlmCheck": {"comment": "dup"}},
            {"id": "f2", "annotations": [box], "reviewFlags": ["detector-disagreement"]},
        ]}
        local_worker.atomic_json(self.root / "project.json", self.project)
        local_worker.STATE.update(project_root=self.root, project=self.project, busy=False, language="en")

    def tearDown(self):
        local_worker.STATE.update(project_root=None, project=None, busy=False, language="de")
        self.temporary.cleanup()

    def saved(self) -> dict:
        return {frame["id"]: frame for frame in json.loads((self.root / "project.json").read_text())["frames"]}

    def test_accept_stores_the_suggested_boxes_and_clears_the_flags(self):
        local_worker.review_consistency_frame({"frameId": "f1", "decision": "accept", "annotations": [self.project["frames"][0]["annotations"][0], {"category": "ball", "x": 5, "y": 5, "width": 8, "height": 8, "source": "manual"}, {"category": "unicorn", "x": 0, "y": 0, "width": 1, "height": 1}]})
        frame = self.saved()["f1"]
        self.assertEqual([box["category"] for box in frame["annotations"]], ["player", "ball"])
        self.assertNotIn("reviewFlags", frame)
        self.assertNotIn("consistencySuggestion", frame)
        self.assertEqual(frame["consistencyReview"]["decision"], "accept")
        self.assertEqual(frame["consistencyReview"]["flags"], ["vlm-issue"])
        self.assertEqual(frame["vlmCheck"], {"comment": "dup"})
        self.assertIn("1 images left", local_worker.STATE.message)

    def test_reject_and_edited_keep_the_boxes(self):
        for frame_id, decision in (("f1", "reject"), ("f2", "edited")):
            local_worker.review_consistency_frame({"frameId": frame_id, "decision": decision, "annotations": []})
        saved = self.saved()
        self.assertEqual(len(saved["f1"]["annotations"]), 2)
        self.assertEqual(len(saved["f2"]["annotations"]), 1)
        self.assertEqual(saved["f2"]["consistencyReview"]["decision"], "edited")
        self.assertFalse(any("reviewFlags" in frame for frame in saved.values()))

    def test_rejects_unknown_decisions_and_running_jobs(self):
        with self.assertRaises(RuntimeError):
            local_worker.review_consistency_frame({"frameId": "f1", "decision": "maybe"})
        local_worker.STATE.update(busy=True)
        with self.assertRaises(RuntimeError):
            local_worker.review_consistency_frame({"frameId": "f1", "decision": "reject"})
        self.assertIn("reviewFlags", self.saved()["f1"])


class RunJsonTests(unittest.TestCase):
    """Regression coverage for a real report: the ball-tracking simulation failed
    with "Expecting ',' delimiter: line 1 column 6 (char 5)" because a library
    notice starting with a timestamp ("[2026-...") was printed to stdout ahead of
    the worker's JSON result, and the whole stdout was parsed as one JSON blob.
    """

    def run_python(self, code: str) -> dict:
        return local_worker.run_json([sys.executable, "-c", code])

    def test_parses_clean_single_line_output(self):
        self.assertEqual(self.run_python('print(\'{"frames": [1]}\')'), {"frames": [1]})

    def test_ignores_preceding_log_lines_on_stdout(self):
        code = "print('[2026-09-29 10:00:00] loading model'); print('some warning'); print('{\"frames\": []}')"
        self.assertEqual(self.run_python(code), {"frames": []})

    def test_forwards_stderr_progress_lines_to_the_callback(self):
        seen: list[tuple[int, int]] = []
        code = (
            "import sys; print('RECO_PROGRESS 1 3', file=sys.stderr, flush=True); "
            "print('RECO_PROGRESS 3 3', file=sys.stderr, flush=True); print('{\"ok\": true}')"
        )
        result = local_worker.run_json([sys.executable, "-c", code], on_progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(seen, [(1, 3), (3, 3)])

    def test_progress_lines_are_not_part_of_the_error_message(self):
        code = "import sys; print('RECO_PROGRESS 1 2', file=sys.stderr, flush=True); print('boom', file=sys.stderr); sys.exit(3)"
        with self.assertRaises(RuntimeError) as raised:
            local_worker.run_json([sys.executable, "-c", code])
        self.assertEqual(str(raised.exception), "boom")

    def test_ignores_trailing_blank_lines(self):
        self.assertEqual(self.run_python('print(\'{"ok": true}\'); print(); print()'), {"ok": True})


class ParityActionTests(unittest.TestCase):
    """Actions the web UI gained to match the Mac app: OpenCV box refinement,
    field boundaries and opening the training folder."""

    def setUp(self):
        local_worker.STATE.update(selected_folder=None, project_root=None, project=None, busy=False, error=None, log=[], language="en")

    def tearDown(self):
        local_worker.STATE.update(language="de")

    def _project(self, root: Path) -> dict:
        project = {"sport": "basketball", "frames": []}
        local_worker.atomic_json(root / "project.json", project)
        local_worker.STATE.update(project_root=root, project=project)
        return project

    def test_refine_boxes_runs_the_shared_worker_command_after_a_backup(self):
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            venv_python = local_worker.venv_python_path(root / ".runtime" / "venv")
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("")
            with mock.patch.object(local_worker, "run_logged", lambda command: commands.append(command)):
                local_worker.ml_action("refine-boxes", {"language": "en"})
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][2:], ["refine-boxes", "--project", str(root), "--language", "en", "--tier", "auto"])
            self.assertTrue(list((root / "backups").glob("project-*-boxen.json")))
        self.assertIsNone(local_worker.STATE.error)

    def test_import_labels_uses_the_system_python_and_reports_the_summary(self):
        commands: list[list[str]] = []
        summary = {"imported": 2, "matched": 1, "duplicates": 1, "annotations": 7, "format": "yolo",
                   "skippedClasses": {"ghost": 2}, "imagesWithoutLabels": 0, "invalidLines": 0, "unreadableImages": 0}

        def fake_run_json(command):
            commands.append(command)
            return summary

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            root.mkdir()
            source = Path(temporary) / "labels"
            source.mkdir()
            self._project(root)
            # No venv: importing labels needs only the standard library.
            with mock.patch.object(local_worker, "run_json", fake_run_json), \
                    mock.patch.object(local_worker, "system_python", lambda: Path("/usr/bin/python3")):
                local_worker.ml_action("import-labels", {"language": "en", "source": str(source), "candidates": True})
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][0], "/usr/bin/python3")
            self.assertEqual(commands[0][2:], ["import-labels", "--project", str(root), "--source", str(source.resolve()), "--format", "auto", "--language", "en", "--candidates"])
        self.assertIsNone(local_worker.STATE.error)
        self.assertEqual(local_worker.STATE.operation, "ready")
        self.assertEqual(
            local_worker.STATE.message,
            "Labels imported (YOLO): 3 images, 7 boxes. Added as suggestions to the review queue. "
            "1 already present, skipped. Classes not imported: ghost (2).",
        )

    def test_import_onnx_passes_the_form_values_to_the_worker(self):
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            root.mkdir()
            weights = Path(temporary) / "best.onnx"
            weights.write_bytes(b"onnx")
            self._project(root)
            with mock.patch.object(local_worker, "run_logged", lambda command: commands.append(command)), \
                    mock.patch.object(local_worker, "system_python", lambda: Path("/usr/bin/python3")):
                local_worker.ml_action("import-onnx", {
                    "language": "en", "file": str(weights), "name": "Soccer", "classNames": "person,ball",
                    "inputSize": "1280", "license": "AGPL-3.0",
                })
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][0], "/usr/bin/python3")
            self.assertEqual(commands[0][2:], [
                "install-onnx", "--project", str(root), "--file", str(weights.resolve()), "--name", "Soccer",
                "--class-names", "person,ball", "--input-size", "1280", "--license", "AGPL-3.0", "--language", "en",
            ])
        self.assertIsNone(local_worker.STATE.error)

    def test_import_labels_rejects_a_missing_source_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._project(Path(temporary))
            local_worker.ml_action("import-labels", {"language": "en", "source": str(Path(temporary) / "missing")})
        self.assertIn("not available", local_worker.STATE.error or "")

    def test_refine_boxes_asks_for_ml_setup_when_there_is_no_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._project(Path(temporary))
            local_worker.ml_action("refine-boxes", {"language": "en"})
        self.assertIn("Set up ML", local_worker.STATE.error or "")

    def test_saves_field_geometry_in_the_shape_the_ml_worker_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            corners = [[0.1, 0.2], [0.9, 0.2], [0.95, 0.9], [0.05, 0.9]]
            # Four points need no real size any more; an old client's sizes are ignored.
            local_worker.save_field_geometry({"corners": corners, "realWidth": 15, "realLength": 28})
            saved = json.loads((root / "project.json").read_text())["fieldGeometry"]
            self.assertEqual(saved, {"corners": corners})
            self.assertEqual(local_worker.STATE.snapshot()["fieldGeometry"], saved)

    def test_saves_a_partial_field_outline_with_points_on_the_image_edge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            # The field leaves the frame at the bottom and right: points sit on the edge.
            outline = [[0.1, 0.3], [0.8, 0.25], [1.0, 0.45], [1.0, 1.0], [0.0, 1.0], [0.0, 0.6]]
            local_worker.save_field_geometry({"corners": outline})
            saved = json.loads((root / "project.json").read_text())["fieldGeometry"]
            self.assertEqual(saved, {"corners": outline})

    def test_rejects_an_outline_whose_edges_cross(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            bow_tie = [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]  # corners clicked out of order
            with self.assertRaises(RuntimeError) as raised:
                local_worker.save_field_geometry({"corners": bow_tie})
            self.assertIn("cross", str(raised.exception))
            self.assertNotIn("fieldGeometry", json.loads((root / "project.json").read_text()))

    def test_rejects_invalid_field_geometry(self):
        good = [[0.1, 0.2], [0.9, 0.2], [0.95, 0.9], [0.05, 0.9]]
        cases = [
            {"corners": good[:3]},
            {"corners": [[0.5, 0.5]] * 33},
            {"corners": [[1.5, 0.2]] + good[1:]},
            {"corners": [[float("nan"), 0.2]] + good[1:]},
            {"corners": [[0.1, "x"]] + good[1:]},
            {"corners": []},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._project(root)
            for payload in cases:
                with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                    local_worker.save_field_geometry(payload)
            self.assertNotIn("fieldGeometry", json.loads((root / "project.json").read_text()))

    def test_reveals_the_visible_training_folder_with_the_system_file_manager(self):
        opened: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temporary:
            video_folder = Path(temporary)
            root = video_folder / ".reco-training"
            root.mkdir()
            (video_folder / "Reco Training").mkdir()
            local_worker.STATE.update(project_root=root)
            with mock.patch.object(local_worker.sys, "platform", "linux"), \
                 mock.patch.object(local_worker.shutil, "which", lambda name: "/usr/bin/xdg-open"), \
                 mock.patch.object(local_worker.subprocess, "Popen", lambda command, **kwargs: opened.append(command)):
                local_worker.reveal_training_folder()
        self.assertEqual(opened, [["/usr/bin/xdg-open", str(video_folder / "Reco Training")]])

    def test_reveal_without_a_project_explains_what_to_do(self):
        with self.assertRaises(RuntimeError):
            local_worker.reveal_training_folder()


class CudaSetupTests(unittest.TestCase):
    """Regression coverage for a real report: plain `pip install torch` gives
    a CPU-only wheel on Windows (PyPI only hosts CUDA builds for Linux), so
    "ML einrichten" previously left every Windows user on CPU regardless of
    having a working NVIDIA GPU. ml_action("setup", ...) now installs torch
    from the CUDA index first when a GPU is detected - guarded by
    payload["useCuda"] so it can be switched off if that install itself
    causes trouble.
    """

    def setUp(self):
        local_worker.STATE.update(selected_folder=None, project_root=None, project=None, busy=False, error=None, log=[])
        self.original_os_name = local_worker.os.name
        self.original_run_logged = local_worker.run_logged
        self.original_detect_gpu = local_worker.detect_nvidia_gpu_name
        self.original_torch_import_error = local_worker.torch_import_error
        local_worker.torch_import_error = lambda path: None
        self.commands: list[list[str]] = []
        local_worker.run_logged = lambda command: self.commands.append(command)

    def tearDown(self):
        local_worker.os.name = self.original_os_name
        local_worker.run_logged = self.original_run_logged
        local_worker.detect_nvidia_gpu_name = self.original_detect_gpu
        local_worker.torch_import_error = self.original_torch_import_error

    def _prepare_project_with_existing_venv(self, root: Path, *, os_name: str) -> None:
        # os.name must be set before computing venv_python_path, which
        # branches on it (Scripts/python.exe vs bin/python3) - otherwise the
        # placeholder file ends up at a different path than ml_action() will
        # later check, making it look like the venv doesn't exist yet and
        # triggering an unrelated `python -m venv` creation path instead.
        local_worker.os.name = os_name
        local_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
        venv_python = local_worker.venv_python_path(root / ".runtime" / "venv")
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("")
        local_worker.STATE.update(project_root=root)

    def test_installs_cuda_torch_first_when_a_windows_gpu_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare_project_with_existing_venv(root, os_name="nt")
            local_worker.detect_nvidia_gpu_name = lambda: "NVIDIA GeForce RTX 4070"

            local_worker.ml_action("setup", {"useCuda": True})

            self.assertEqual(len(self.commands), 3)
            self.assertIn("--index-url", self.commands[1])
            self.assertIn("torch", self.commands[1])
            self.assertIn("rfdetr[train,onnx,coreml]>=1.9.0", self.commands[2])
            self.assertIn("--extra-index-url", self.commands[2])

    def test_skips_cuda_install_when_the_toggle_is_off(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare_project_with_existing_venv(root, os_name="nt")
            local_worker.detect_nvidia_gpu_name = lambda: "NVIDIA GeForce RTX 4070"

            local_worker.ml_action("setup", {"useCuda": False})

            self.assertEqual(len(self.commands), 1)
            self.assertIn("rfdetr[train,onnx,coreml]>=1.9.0", self.commands[0])
            self.assertNotIn("--extra-index-url", self.commands[0])

    def test_skips_cuda_install_when_no_gpu_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare_project_with_existing_venv(root, os_name="nt")
            local_worker.detect_nvidia_gpu_name = lambda: None

            local_worker.ml_action("setup", {"useCuda": True})

            self.assertEqual(len(self.commands), 1)
            self.assertNotIn("--extra-index-url", self.commands[0])

    def test_skips_cuda_install_on_non_windows_platforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare_project_with_existing_venv(root, os_name="posix")
            local_worker.detect_nvidia_gpu_name = lambda: "NVIDIA GeForce RTX 4070"

            local_worker.ml_action("setup", {"useCuda": True})

            self.assertEqual(len(self.commands), 1)
            self.assertNotIn("--extra-index-url", self.commands[0])


class SimulationGapCandidateTests(unittest.TestCase):
    """Misses of a tracking simulation become review candidates, and only on request."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"video")
        self.video_id = local_worker.hashlib.sha256(str(self.video).encode("utf-8")).hexdigest()[:12]
        frames_dir = local_worker.simulation_frames_dir(self.root, "a")
        frames_dir.mkdir(parents=True)
        balls = [(100, 100), None, None, None, (300, 200), None, (320, 200)]
        sim_frames = []
        for index, ball in enumerate(balls):
            name = f"sim-{index + 1:04d}.jpg"
            (frames_dir / name).write_bytes(b"jpg-%d" % index)
            sim_frames.append({"file": name, "timestamp": index / 12, "width": 1920, "height": 1080, "ball": {"x": ball[0], "y": ball[1], "confidence": .8} if ball else None})
        venv_python = local_worker.venv_python_path(local_worker.ml_venv(self.root))
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("")
        self.project = {"sport": "basketball", "frames": []}
        local_worker.atomic_json(self.root / "project.json", self.project)
        local_worker.STATE.update(
            project_root=self.root, project=self.project, language="en", error=None,
            simulation={"id": "1", "fps": 12.0, "frames": sim_frames, "package": "pkg-a"}, simulation_videos={"a": self.video},
        )
        self.commands: list[list[str]] = []
        self.original_run_json = local_worker.run_json

        def fake_run_json(command, on_progress=None):
            self.commands.append(command)
            requests = json.loads(Path(command[command.index("--requests") + 1]).read_text())
            return {"frames": [
                {"file": item["file"], "ball": {"x": 190.0, "y": 140.0, "width": 12.0, "height": 12.0, "confidence": .3} if item["file"] == "sim-0003.jpg" else None}
                for item in requests
            ]}

        local_worker.run_json = fake_run_json

    def tearDown(self):
        local_worker.run_json = self.original_run_json
        local_worker.STATE.update(project_root=None, project=None, simulation=None, simulation_b=None, simulation_videos={}, language="de", error=None, busy=False)
        self.temporary.cleanup()

    def test_gap_frames_become_candidates_with_the_recovered_ball(self):
        local_worker.simulation_gaps_to_candidates({"slot": "a", "language": "en"})
        self.assertIsNone(local_worker.STATE.error)
        saved = json.loads((self.root / "project.json").read_text())["frames"]
        self.assertEqual(len(saved), 4)  # three of the first gap, one of the second
        self.assertTrue(all(frame["reviewStatus"] == "candidate" and frame["videoID"] == self.video_id and frame["origin"] == "ball-gap" and not frame.get("heldOut") for frame in saved))
        self.assertTrue(all((self.root / frame["relativePath"]).is_file() for frame in saved))
        with_ball = [frame for frame in saved if frame["annotations"]]
        self.assertEqual(len(with_ball), 1)
        box = with_ball[0]["annotations"][0]
        self.assertEqual((box["category"], box["source"], box["x"], box["width"]), ("ball", "auto", 190.0, 12.0))
        self.assertAlmostEqual(with_ball[0]["timestamp"], 2 / 12)
        self.assertEqual(self.commands[0][self.commands[0].index("--package") + 1], "pkg-a")
        self.assertFalse((self.root / ".runtime" / "ball-gaps-a.json").exists())
        self.assertIn("4 gap images", local_worker.STATE.message)

    def test_a_second_run_adds_nothing_twice(self):
        local_worker.simulation_gaps_to_candidates({"slot": "a", "language": "en"})
        local_worker.simulation_gaps_to_candidates({"slot": "a", "language": "en"})
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(len(json.loads((self.root / "project.json").read_text())["frames"]), 4)

    def test_a_video_with_independent_test_images_is_refused(self):
        self.project["frames"].append({"id": "held", "videoID": self.video_id, "heldOut": True, "annotations": []})
        local_worker.simulation_gaps_to_candidates({"slot": "a", "language": "en"})
        self.assertIn("independent test images", local_worker.STATE.error)
        self.assertEqual(self.commands, [])
        self.assertEqual(len(self.project["frames"]), 1)

    def test_without_a_simulation_it_fails_clearly(self):
        local_worker.STATE.update(simulation=None)
        local_worker.simulation_gaps_to_candidates({"slot": "a", "language": "en"})
        self.assertIn("Simulate a video first", local_worker.STATE.error)


if __name__ == "__main__":
    unittest.main()
