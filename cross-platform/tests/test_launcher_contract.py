"""Contract tests for what the Mac launcher (Swift) relies on in the Python part."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]

# Must stay identical to WorkerController.launchSnippet in the Swift launcher.
LAUNCH_SNIPPET = "import os, sys; os.getpgrp() == os.getpid() or os.setsid(); os.execv(sys.executable, [sys.executable] + sys.argv[1:])"


class LauncherContractTests(unittest.TestCase):
    def test_launch_snippet_is_mirrored_in_swift(self):
        swift = (ROOT.parent / "mac" / "Sources" / "RecoTrainerMac" / "WorkerController.swift").read_text(encoding="utf-8")
        self.assertIn(LAUNCH_SNIPPET, swift)

    def test_launch_snippet_makes_worker_a_process_group_leader_and_passes_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "probe.py"
            script.write_text(
                "import json, os, sys\nprint(json.dumps({'leader': os.getpgrp() == os.getpid(), 'argv': sys.argv[1:]}))\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-c", LAUNCH_SNIPPET, str(script), "--port", "8766"],
                capture_output=True, text=True, check=True, timeout=20,
            )
        data = json.loads(result.stdout)
        self.assertTrue(data["leader"])
        self.assertEqual(data["argv"], ["--port", "8766"])

    def test_launch_snippet_runs_when_already_a_process_group_leader(self):
        # Foundation's Process starts the child as a group leader; os.setsid() would raise EPERM there.
        result = subprocess.run(
            [sys.executable, "-c", LAUNCH_SNIPPET, "-c", "import os; print(os.getpgrp() == os.getpid())"],
            capture_output=True, text=True, timeout=20, process_group=0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "True")

    def test_bundled_frame_extractor_override(self):
        spec_path = ROOT / "local_worker.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_worker_contract", spec_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "reco-frame-extractor"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            previous = os.environ.get("RECO_FRAME_EXTRACTOR")
            os.environ["RECO_FRAME_EXTRACTOR"] = str(binary)
            try:
                self.assertEqual(module.native_frame_extractor(Path(temporary)), binary)
            finally:
                if previous is None:
                    os.environ.pop("RECO_FRAME_EXTRACTOR", None)
                else:
                    os.environ["RECO_FRAME_EXTRACTOR"] = previous

    def test_payload_zip_has_what_the_launcher_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_ui = Path(temporary) / "ui"
            (fake_ui / "assets").mkdir(parents=True)
            (fake_ui / "index.html").write_text("<!doctype html>", encoding="utf-8")
            out = Path(temporary) / "out"
            env = dict(os.environ, RECO_UI_SOURCE=str(fake_ui), RECO_PAYLOAD_MIN_LAUNCHER="0.20.0")
            subprocess.run(["bash", str(ROOT / "scripts" / "build-payload.sh"), str(out)], env=env, check=True, capture_output=True, timeout=60)
            version = json.loads((ROOT / "package.json").read_text())["version"]
            archive = zipfile.ZipFile(out / f"Reco.Trainer.Payload.{version}.zip")
            names = set(archive.namelist())
            # PayloadStore.requiredFiles in the Swift launcher
            for required in ("payload.json", "local_worker.py", "ml_worker.py", "ui/index.html"):
                self.assertIn(required, names)
            # Modules and data the workers load at run time must travel with them.
            for companion in ("anonymize.py", "teachers.py", "augment_studio.py", "vlm.py", "model_registry.json"):
                self.assertIn(companion, names)
            manifest = json.loads(archive.read("payload.json"))
            self.assertEqual(manifest, {"version": version, "minLauncher": "0.20.0"})
            self.assertFalse([n for n in names if n.startswith("/") or ".." in n.split("/")])

    def test_swift_required_files_match_payload(self):
        swift = (ROOT.parent / "mac" / "Sources" / "LauncherCore" / "PayloadStore.swift").read_text(encoding="utf-8")
        self.assertIn('["payload.json", "local_worker.py", "ml_worker.py", "ui/index.html"]', swift)


class NoTrainingImagesInTheRepositoryTests(unittest.TestCase):
    """Pictures of a training set, generated or real, must never reach the repository."""

    ALLOWED = ("mac/Assets/", "mac/Assets.xcassets/", "mac/Sources/RecoTrainerMac/Resources/", "cross-platform/public/", "docs/screenshots/", "docs/training-video/")
    SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".tif", ".tiff"}

    def test_tracked_pictures_are_only_the_apps_own(self):
        repository = Path(__file__).resolve().parents[2]
        try:
            listed = subprocess.run(["git", "ls-files"], cwd=repository, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        pictures = [name for name in listed.splitlines() if Path(name).suffix.lower() in self.SUFFIXES]
        self.assertEqual([name for name in pictures if not name.startswith(self.ALLOWED)], [])


if __name__ == "__main__":
    unittest.main()
