from __future__ import annotations

import http.client
import importlib.util
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

WORKER_PATH = Path(__file__).parents[1] / "local_worker.py"
SPEC = importlib.util.spec_from_file_location("local_worker_static", WORKER_PATH)
assert SPEC and SPEC.loader
local_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_worker)


class StaticUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        base = Path(cls.temporary.name)
        cls.ui = (base / "ui").resolve()
        (cls.ui / "assets").mkdir(parents=True)
        (cls.ui / "index.html").write_text("<!doctype html><title>Reco</title>", encoding="utf-8")
        (cls.ui / "assets" / "app-abc.js").write_text("console.log(1)", encoding="utf-8")
        (cls.ui / "clip.mp4").write_bytes(bytes(range(200)))
        (base / "secret.txt").write_text("secret", encoding="utf-8")
        cls.original_ui = local_worker.UI_DIR
        local_worker.UI_DIR = cls.ui
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), local_worker.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        local_worker.UI_DIR = cls.original_ui
        cls.temporary.cleanup()

    def get(self, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response, body

    def test_root_and_spa_fallback_serve_index(self):
        for path in ("/", "/index.html", "/some/client/route"):
            response, body = self.get(path)
            self.assertEqual(response.status, 200, path)
            self.assertIn(b"<title>Reco</title>", body)
            self.assertEqual(response.getheader("Cache-Control"), "no-cache")

    def test_hashed_assets_are_cacheable_with_correct_type(self):
        response, body = self.get("/assets/app-abc.js")
        self.assertEqual(response.status, 200)
        self.assertEqual(body, b"console.log(1)")
        self.assertTrue(response.getheader("Content-Type").startswith("text/javascript"))
        self.assertIn("immutable", response.getheader("Cache-Control"))

    def test_missing_file_with_extension_is_404_not_index(self):
        response, _ = self.get("/assets/missing.js")
        self.assertEqual(response.status, 404)

    def test_path_traversal_is_refused(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/../../secret.txt"):
            response, body = self.get(path)
            self.assertNotIn(b"secret", body, path)

    def test_api_paths_are_never_served_as_static(self):
        response, _ = self.get("/api/does-not-exist")
        self.assertEqual(response.status, 404)

    def test_range_requests_for_video(self):
        response, body = self.get("/clip.mp4", {"Range": "bytes=10-19"})
        self.assertEqual(response.status, 206)
        self.assertEqual(body, bytes(range(10, 20)))
        self.assertEqual(response.getheader("Content-Range"), "bytes 10-19/200")
        response, body = self.get("/clip.mp4", {"Range": "bytes=190-"})
        self.assertEqual(body, bytes(range(190, 200)))
        response, body = self.get("/clip.mp4", {"Range": "bytes=-5"})
        self.assertEqual(body, bytes(range(195, 200)))
        response, _ = self.get("/clip.mp4", {"Range": "bytes=500-600"})
        self.assertEqual(response.status, 416)
        response, body = self.get("/clip.mp4")
        self.assertEqual((response.status, len(body)), (200, 200))

    def test_missing_ui_folder_explains_itself(self):
        local_worker.UI_DIR = self.ui / "nope"
        try:
            response, body = self.get("/")
            self.assertEqual(response.status, 404)
            self.assertIn(b"npm run build", body)
        finally:
            local_worker.UI_DIR = self.ui


if __name__ == "__main__":
    unittest.main()
