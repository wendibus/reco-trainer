"""The app-managed VLM: platform builds, answer parsing, coordinates, safe downloads."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import vlm  # noqa: E402

spec = importlib.util.spec_from_file_location("vlm_ml_worker", ROOT / "ml_worker.py")
assert spec and spec.loader
ml_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_worker)
REGISTRY = {entry["id"]: entry for entry in json.loads((ROOT / "model_registry.json").read_text())["models"]}


class PlatformTests(unittest.TestCase):
    def test_every_platform_has_a_registered_build(self):
        cases = {
            ("Darwin", "arm64", False): ["llama-server-macos-arm64"],
            ("Windows", "AMD64", True): ["llama-server-windows-cuda", "llama-server-windows-cudart"],
            ("Windows", "AMD64", False): ["llama-server-windows-cpu"],
            ("Linux", "x86_64", True): ["llama-server-linux-cuda", "llama-server-linux-cudart"],
            ("Linux", "x86_64", False): ["llama-server-linux-cpu"],
            ("Linux", "aarch64", False): ["llama-server-linux-arm64"],  # Docker on Apple silicon
        }
        for (system, machine, nvidia), expected in cases.items():
            ids = vlm.runtime_ids(system, machine, nvidia, glibc=(2, 41))
            self.assertEqual(ids, expected)
            for model_id in ids:
                self.assertEqual(REGISTRY[model_id]["license"], "MIT")
        with self.assertRaises(RuntimeError):
            vlm.runtime_ids("Darwin", "x86_64")

    def test_linux_build_follows_the_c_library(self):
        # Ubuntu 22.04 (glibc 2.35) with NVIDIA: the CUDA build needs 2.38, so the CPU build is used.
        self.assertEqual(vlm.runtime_ids("Linux", "x86_64", True, glibc=(2, 35)), ["llama-server-linux-cpu"])
        self.assertEqual(vlm.runtime_ids("Linux", "x86_64", True, glibc=(2, 39)), ["llama-server-linux-cuda", "llama-server-linux-cudart"])
        with self.assertRaises(RuntimeError):
            vlm.runtime_ids("Linux", "aarch64", False, glibc=(2, 36))  # Debian 12 on arm64
        with self.assertRaises(RuntimeError):
            vlm.runtime_ids("Linux", "x86_64", False, glibc=(2, 31))

    def test_cuda_runtime_installs_next_to_its_server(self):
        self.assertEqual(REGISTRY["llama-server-windows-cudart"]["installInto"], "llama-server-windows-cuda")
        self.assertEqual(REGISTRY["llama-server-linux-cudart"]["installInto"], "llama-server-linux-cuda")

    def test_shared_folder_per_platform_and_none_in_docker(self):
        with mock.patch.dict(os.environ, {"RECO_SHARED_MODELS": "/tmp/shared-test"}):
            self.assertEqual(vlm.shared_models_dir(), Path("/tmp/shared-test"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RECO_SHARED_MODELS", None)
            with mock.patch.object(vlm.Path, "exists", lambda self: str(self) == "/.dockerenv"):
                self.assertIsNone(vlm.shared_models_dir())


class AnswerTests(unittest.TestCase):
    def test_json_in_fences_or_prose(self):
        self.assertEqual(vlm.parse_json_answer('```json\n{"missing": 1}\n```'), {"missing": 1})
        self.assertEqual(vlm.parse_json_answer('Sure! [{"label": "ball", "bbox_2d": [1, 2, 3, 4]}] Done.'), [{"label": "ball", "bbox_2d": [1, 2, 3, 4]}])
        with self.assertRaises(ValueError):
            vlm.parse_json_answer("no idea")

    def test_coordinates_are_normalized_per_model_format(self):
        # Qwen3.5 answered [390, 604, 399, 622] for a ball at pixel (499, 435) of a 1280x720 frame.
        x, y, w, h = vlm.to_pixels([390, 604, 399, 622], 1280, 720, "relative-1000")
        self.assertAlmostEqual(x, 499.2)
        self.assertAlmostEqual(y, 434.88)
        self.assertAlmostEqual(w, 11.52)
        self.assertEqual(vlm.to_pixels([10, 20, 30, 50], 1280, 720, "absolute"), (10.0, 20.0, 20.0, 30.0))
        self.assertEqual(vlm.to_pixels([0.5, 0.5, 0.25, 0.75], 100, 100, "relative-1"), (25.0, 50.0, 25.0, 25.0))
        self.assertEqual(vlm.to_pixels([-50, -50, 2000, 2000], 100, 100, "absolute"), (0.0, 0.0, 100.0, 100.0))
        self.assertEqual(REGISTRY["qwen3.5-4b-vlm"]["coordinates"], "relative-1000")

    def test_counts_are_robust(self):
        self.assertEqual(vlm.as_count("2"), 2)
        self.assertEqual(vlm.as_count(None), 0)
        self.assertEqual(vlm.as_count(-3), 0)


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __exit__(self, *exc):
        self.close()
        return False


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_zip_slip_and_foreign_links_are_refused(self):
        bad_zip = self.root / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as archive:
            archive.writestr("../escape.txt", "x")
        with self.assertRaises(SystemExit):
            ml_worker.safe_extract(bad_zip, "zip", self.root / "out1")
        bad_tar = self.root / "bad.tar.gz"
        with tarfile.open(bad_tar, "w:gz") as archive:
            link = tarfile.TarInfo("llama/evil")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(link)
        with self.assertRaises(SystemExit):
            ml_worker.safe_extract(bad_tar, "tar.gz", self.root / "out2")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_runtime_archive_is_checked_unpacked_and_marked_complete(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            data = b"#!/bin/sh\necho server\n"
            info = tarfile.TarInfo("llama-b1/llama-server")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        blob = payload.getvalue()
        entry = {"id": "rt", "name": "runtime", "url": "https://example.com/rt.tar.gz", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob), "archive": "tar.gz"}
        with mock.patch.object(ml_worker.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(blob)):
            folder = ml_worker.download_archive(entry, self.root / "rt", "en")
        server = vlm.find_server_binary(folder)
        self.assertIsNotNone(server)
        if os.name != "nt":
            self.assertTrue(os.access(server, os.X_OK))
        self.assertTrue(ml_worker.completion_marker(folder, "rt").is_file())
        self.assertEqual([path.name for path in folder.iterdir() if path.name.endswith(".tar.gz")], [])

    def test_model_files_are_each_checked(self):
        model, projector = b"model-bytes" * 10, b"projector"
        entry = {"id": "vlm-test", "name": "VLM", "files": [
            {"name": "m.gguf", "url": "https://huggingface.co/x/resolve/abc/m.gguf", "sha256": hashlib.sha256(model).hexdigest(), "bytes": len(model)},
            {"name": "mmproj.gguf", "url": "https://huggingface.co/x/resolve/abc/mmproj.gguf", "sha256": hashlib.sha256(projector).hexdigest(), "bytes": len(projector)},
        ]}
        responses = {"m.gguf": model, "mmproj.gguf": b"tampered"}
        opener = lambda request, **_k: FakeResponse(responses[request.full_url.rsplit("/", 1)[1]])  # noqa: E731
        with mock.patch.object(ml_worker.urllib.request, "urlopen", opener), mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                ml_worker.download_model_files(entry, self.root / "vlm", "en")
            self.assertFalse(ml_worker.completion_marker(self.root / "vlm", "vlm-test").exists())
            self.assertFalse((self.root / "vlm" / "mmproj.gguf").exists())
            responses["mmproj.gguf"] = projector
            ml_worker.download_model_files(entry, self.root / "vlm", "en")
        self.assertTrue(ml_worker.completion_marker(self.root / "vlm", "vlm-test").is_file())
        self.assertEqual((self.root / "vlm" / "m.gguf").read_bytes(), model)


if __name__ == "__main__":
    unittest.main()
