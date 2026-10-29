"""Model registry: allowed licenses only, verified downloads, nothing half-written left behind."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml_worker = load_module("registry_ml_worker", ROOT / "ml_worker.py")
local_worker = load_module("registry_local_worker", ROOT / "local_worker.py")
REGISTRY = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, fail_after: int | None = None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}
        self.fail_after = fail_after

    def read(self, size: int = -1) -> bytes:
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise SystemExit(143)  # what SIGTERM turns into inside ml_worker
        return super().read(size)

    def __exit__(self, *exc):
        self.close()
        return False


class RegistryContentTests(unittest.TestCase):
    def test_only_apache_and_mit_licenses(self):
        self.assertEqual(set(REGISTRY["allowedLicenses"]), {"Apache-2.0", "MIT"})
        for entry in REGISTRY["models"]:
            self.assertIn(entry["license"], REGISTRY["allowedLicenses"], entry["id"])
            self.assertTrue(entry["licenseUrl"].startswith("https://"), entry["id"])

    def test_no_pml_or_ultralytics_or_sam3_models(self):
        text = json.dumps(REGISTRY).lower()
        for forbidden in ("xlarge", "2xl", "pml", "ultralytics", "yolov", "sam3", "sam 3"):
            self.assertNotIn(forbidden, text)

    def test_downloadable_entries_have_https_url_checksum_and_size(self):
        ids = [entry["id"] for entry in REGISTRY["models"]]
        self.assertEqual(len(ids), len(set(ids)))
        for entry in REGISTRY["models"]:
            if entry.get("managedBy"):
                continue
            if entry.get("huggingFace"):
                # Pinned to one commit, so the files cannot change under us.
                self.assertRegex(entry["huggingFace"]["revision"], r"^[0-9a-f]{40}$", entry["id"])
                self.assertRegex(entry["huggingFace"]["repo"], r"^[\w.-]+/[\w.-]+$")
                self.assertGreater(entry["bytes"], 0)
                continue
            self.assertTrue(entry["url"].startswith("https://"), entry["id"])
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)
            self.assertEqual(Path(entry["file"]).name, entry["file"])


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        self.data = b"fake onnx weights" * 1000
        registry = {"schemaVersion": 1, "allowedLicenses": ["MIT"], "models": [{
            "id": "test-model", "name": "Test model", "license": "MIT", "licenseUrl": "https://example.com/LICENSE",
            "url": "https://example.com/test.onnx", "file": "test.onnx", "sha256": hashlib.sha256(self.data).hexdigest(), "bytes": len(self.data),
        }]}
        path = self.project / "registry.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        self.registry_patch = mock.patch.object(ml_worker, "REGISTRY_PATH", path)
        self.registry_patch.start()

    def tearDown(self):
        self.registry_patch.stop()
        self.temporary.cleanup()

    def models_dir(self) -> list[str]:
        folder = self.project / ".runtime" / "models"
        return sorted(item.name for item in folder.iterdir()) if folder.is_dir() else []

    def test_download_is_verified_and_reused(self):
        with mock.patch.object(ml_worker.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(self.data)):
            path = ml_worker.download_registry_model(self.project, "test-model", "en")
        self.assertEqual(path.read_bytes(), self.data)
        self.assertEqual(self.models_dir(), ["test.onnx"])
        with mock.patch.object(ml_worker.urllib.request, "urlopen", side_effect=AssertionError("must not download again")):
            self.assertEqual(ml_worker.download_registry_model(self.project, "test-model", "en"), path)

    def test_wrong_checksum_leaves_nothing_behind(self):
        with mock.patch.object(ml_worker.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(b"tampered")):
            with self.assertRaises(SystemExit) as raised:
                ml_worker.download_registry_model(self.project, "test-model", "en")
        self.assertIn("checksum", str(raised.exception))
        self.assertEqual(self.models_dir(), [])

    def test_cancel_during_download_removes_the_partial_file(self):
        with mock.patch.object(ml_worker.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(self.data, fail_after=4096)):
            with self.assertRaises(SystemExit):
                ml_worker.download_registry_model(self.project, "test-model", "en")
        self.assertEqual(self.models_dir(), [])


class RegistrySnapshotTests(unittest.TestCase):
    def test_status_lists_licenses_and_install_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".runtime" / "models").mkdir(parents=True)
            (root / ".runtime" / "models" / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
            (root / ".runtime" / "models" / "hf" / "florence-2-base").mkdir(parents=True)
            (root / ".runtime" / "models" / "hf" / "florence-2-base" / ".reco-complete").write_text("rev")
            entries = {entry["id"]: entry for entry in local_worker.model_registry_snapshot(root)}
        self.assertTrue(entries["yunet-face"]["installed"])
        self.assertFalse(entries["ppocr-text-detector"]["installed"])
        self.assertIsNone(entries["rf-detr-large"]["installed"])  # managed by the rfdetr package itself
        self.assertIs(entries["sam2.1-hiera-small"]["installed"], False)
        self.assertIs(entries["florence-2-base"]["installed"], True)
        self.assertEqual(entries["yunet-face"]["license"], "MIT")
        for entry in entries.values():
            self.assertTrue(re.fullmatch(r"Apache-2\.0|MIT", entry["license"]))


if __name__ == "__main__":
    unittest.main()
