import contextlib
import io
import json
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import model_hub


def weights_file(path: Path, tokens=1025, head=6, extra=b"", key="x") -> None:
    header = {
        model_hub.POSITION_KEY: {"dtype": "F32", "shape": [1, tokens, 1], "data_offsets": [0, 4]},
        model_hub.CLASS_HEAD_KEY: {"dtype": "F32", "shape": [head], "data_offsets": [4, 8]},
        key: {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 12 + extra)


class FakeGitHub(BaseHTTPRequestHandler):
    """Records every call; fail names a path part that answers with an error."""
    calls: list = []
    fail: str | None = None
    index: bytes = b"{}"
    etag = '"one"'
    tokens: list = []

    def log_message(self, *_):
        pass

    def answer(self, code, body=None, headers=None):
        raw = json.dumps(body).encode() if isinstance(body, (dict, list)) else (body or b"")
        self.send_response(code)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def handle_any(self):
        cls = type(self)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        path = urlparse(self.path).path
        cls.calls.append((self.command, self.path, raw, self.headers.get("Authorization")))
        host = f"http://{self.headers['Host']}"
        if cls.fail and cls.fail in path:
            return self.answer(500, {"message": "boom"})
        if path == "/index.json":
            if self.headers.get("If-None-Match") == cls.etag:
                return self.answer(304)
            return self.answer(200, cls.index, {"ETag": cls.etag})
        if path == "/login/device/code":
            return self.answer(200, {"device_code": "dev", "user_code": "ABCD-1234", "verification_uri": f"{host}/login/device", "interval": 0, "expires_in": 60})
        if path == "/login/oauth/access_token":
            return self.answer(200, cls.tokens.pop(0))
        if self.command == "POST" and path.endswith("/forks"):
            return self.answer(202, {"full_name": "me/reco-models"})
        if self.command == "GET" and path.endswith("/branches/main"):
            return self.answer(200, {"name": "main"})
        if self.command == "GET" and path.endswith("/git/ref/heads/main"):
            return self.answer(200, {"object": {"sha": "base"}})
        if self.command == "GET" and ("/git/ref/heads/" in path or "/contents/" in path):
            return self.answer(404, {"message": "Not Found"})
        if self.command == "POST" and path.endswith("/releases"):
            return self.answer(201, {"id": 7, "upload_url": f"{host}/upload/7{{?name,label}}"})
        if path.startswith("/upload/"):
            return self.answer(201, {"browser_download_url": "https://github.com/me/reco-models/releases/download/t/w.safetensors"})
        if path.endswith("/pulls"):
            return self.answer(201, {"html_url": "https://github.com/o/reco-models/pull/1"})
        return self.answer(200 if self.command != "DELETE" else 204, {} if self.command != "DELETE" else None)

    do_GET = do_POST = do_PUT = do_DELETE = handle_any


class ServerTest(unittest.TestCase):
    def setUp(self):
        FakeGitHub.calls, FakeGitHub.fail, FakeGitHub.tokens = [], None, []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.dir = Path(self.folder.name)


class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "w.safetensors"

    def test_size_and_classes_come_from_the_weights(self):
        weights_file(self.path)
        self.assertEqual(model_hub.inspect_weights(self.path), {"size": "small", "classCount": 5})

    def test_large_and_hidden_data_are_refused(self):
        for options in ({"tokens": 1937}, {"extra": b"hidden"}, {"key": "/Users/me"}):
            weights_file(self.path, **options)
            with self.assertRaises(model_hub.HubError, msg=options):
                model_hub.inspect_weights(self.path)

    def test_a_pickle_is_not_read_as_weights(self):
        self.path.write_bytes(b"\x80\x04cos\nsystem\n(S'ls'\ntR." * 3)
        with self.assertRaises(model_hub.HubError):
            model_hub.inspect_weights(self.path)


class IndexTests(ServerTest):
    def test_filter_by_sport_version_and_classes(self):
        index = {"models": [
            {"id": "a", "sport": "football", "size": "small", "classes": ["ball", "player"], "compat": {"trainer_min": "0.25.0"}},
            {"id": "b", "sport": "basketball", "size": "small", "classes": ["ball"], "compat": {"trainer_min": "0.25.0"}},
            {"id": "c", "sport": "football", "size": "small", "classes": ["ball"], "compat": {"trainer_min": "9.0.0"}},
            {"id": "d", "sport": "football", "size": "small", "classes": ["ball"], "compat": {"trainer_min": "0.1.0"}, "deprecated": True},
            {"id": "e", "sport": "football", "size": "nano", "classes": ["ball"], "compat": {"trainer_min": "0.25.0"}},
        ]}
        found = model_hub.filter_models(index, "football", ["player", "ball"], "0.25.0")
        self.assertEqual([(item["id"], item["classMatch"]) for item in found], [("e", "different"), ("a", "same")])

    def test_cache_etag_and_offline(self):
        FakeGitHub.index = json.dumps({"models": [{"id": "a"}]}).encode()
        cache = self.dir / "index.json"
        url = f"{self.url}/index.json"
        first = model_hub.fetch_index(cache, url)
        self.assertEqual((first["models"], first["offline"]), ([{"id": "a"}], False))
        second = model_hub.fetch_index(cache, url)
        self.assertEqual((second["models"], second["offline"]), ([{"id": "a"}], False))
        self.assertEqual(sum(1 for call in FakeGitHub.calls if call[0] == "GET"), 2)
        gone = model_hub.fetch_index(cache, "http://127.0.0.1:9/index.json")
        self.assertEqual((gone["models"], gone["offline"]), ([{"id": "a"}], True))
        self.assertEqual(model_hub.fetch_index(self.dir / "none.json", "http://127.0.0.1:9/x")["models"], [])


class DownloadTests(ServerTest):
    def test_only_from_the_main_repository_and_with_the_right_hash(self):
        target = self.dir / "file"
        with self.assertRaises(model_hub.HubError):
            model_hub.download("https://evil.example/w.safetensors", target, "0" * 64)
        self.assertEqual(FakeGitHub.calls, [])

    def test_wrong_hash_leaves_nothing_behind(self):
        target = self.dir / "file"
        FakeGitHub.index = b"content"
        original = model_hub.HUB_ASSET_PREFIX
        model_hub.HUB_ASSET_PREFIX = self.url
        self.addCleanup(setattr, model_hub, "HUB_ASSET_PREFIX", original)
        with self.assertRaises(model_hub.HubError):
            model_hub.download(f"{self.url}/index.json", target, "0" * 64)
        self.assertEqual(list(self.dir.iterdir()), [])
        import hashlib
        model_hub.download(f"{self.url}/index.json", target, hashlib.sha256(b"content").hexdigest())
        self.assertEqual(target.read_bytes(), b"content")


def share_model() -> dict:
    return model_hub.build_model_json(
        name="Halle Kunstlicht", description="Vollfeld", version="1.0.0", sport="basketball", size="small", classes=["ball", "player", "referee", "hoop"], parent=None, author="me",
        training={"images": 5, "games": 1, "camera": "fest", "venue": "indoor", "epochs": 3}, self_reported={"mAP50": 0.5}, trainer_version="0.25.0", sha256="a" * 64, size_bytes=12, source_url="",
    )


class ModelJsonTests(unittest.TestCase):
    def test_private_text_is_refused(self):
        for text in ("/Users/me/verein", "/Volumes/Extern/x", "C:\\Users\\me", "ich@example.com"):
            with self.assertRaises(model_hub.HubError, msg=text):
                model_hub.check_public_text(text)

    def test_the_slug_and_the_confirmation(self):
        model = share_model()
        self.assertEqual(model["id"], "halle-kunstlicht")
        self.assertEqual(model["rights"]["confirmed_by"], "me")
        self.assertEqual(model["input_resolution"], 512)
        self.assertNotIn("samples", model)

    def test_the_hub_repository_accepts_what_the_trainer_writes(self):
        tools = Path.home() / "Developer" / "reco-models" / "scripts"
        if not (tools / "hub_tools.py").is_file():
            self.skipTest("reco-models is not next to this checkout")
        sys.path.insert(0, str(tools))
        import hub_tools
        model = share_model()
        model["files"][0]["source_url"] = "https://github.com/me/reco-models/releases/download/basketball-small-halle-kunstlicht-v1.0.0/w.safetensors"
        hub_tools.check_model(model, "sports/basketball/small/halle-kunstlicht/model.json", "me")
        self.assertEqual(hub_tools.POSITION_TOKENS, model_hub.POSITION_TOKENS)


class LoginTests(ServerTest):
    def test_device_flow_waits_for_confirmation(self):
        FakeGitHub.tokens = [{"error": "authorization_pending"}, {"error": "slow_down", "interval": 0}, {"access_token": "tok"}]
        shown = []
        token = model_hub.device_login(lambda code, url: shown.append(code), client_id="id", login=self.url, wait=lambda _: None)
        self.assertEqual((token, shown), ("tok", ["ABCD-1234"]))
        self.assertIn(b"scope=public_repo", FakeGitHub.calls[0][2])

    def test_denied_and_not_configured(self):
        FakeGitHub.tokens = [{"error": "access_denied", "error_description": "denied"}]
        with self.assertRaises(model_hub.HubError):
            model_hub.device_login(lambda *_: None, client_id="id", login=self.url, wait=lambda _: None)
        original = model_hub.HUB_CLIENT_ID
        model_hub.HUB_CLIENT_ID = ""
        self.addCleanup(setattr, model_hub, "HUB_CLIENT_ID", original)
        with self.assertRaises(model_hub.HubError):
            model_hub.device_login(lambda *_: None, login=self.url)


class ShareTests(ServerTest):
    def run_share(self):
        weights = self.dir / "w.safetensors"
        weights_file(weights)
        steps = []
        github = model_hub.GitHub("tok", self.url)
        url = model_hub.share(github, weights, share_model(), steps.append, repo="o/reco-models", wait=lambda _: None)
        return url, steps

    def test_writes_only_into_the_fork_and_opens_a_pull_request(self):
        url, steps = self.run_share()
        self.assertEqual(url, "https://github.com/o/reco-models/pull/1")
        self.assertEqual(steps, ["fork", "upload", "pull-request"])
        writes = [(method, path) for method, path, *_ in FakeGitHub.calls if method != "GET"]
        self.assertEqual({path for _, path in writes if path.startswith("/repos/o/")}, {"/repos/o/reco-models/forks", "/repos/o/reco-models/pulls"})
        put = next(call for call in FakeGitHub.calls if call[0] == "PUT")
        self.assertIn("/repos/me/reco-models/contents/sports/basketball/small/halle-kunstlicht/model.json", put[1])
        self.assertTrue(all(call[3] == "Bearer tok" for call in FakeGitHub.calls))
        pull = json.loads(next(call for call in FakeGitHub.calls if call[1].endswith("/pulls"))[2])
        self.assertEqual(pull["head"], "me:model/basketball-small-halle-kunstlicht-v1.0.0")

    def test_a_failure_removes_the_release_and_the_branch_again(self):
        FakeGitHub.fail = "/pulls"
        with self.assertRaises(model_hub.HubError):
            self.run_share()
        deleted = [path for method, path, *_ in FakeGitHub.calls if method == "DELETE"]
        self.assertEqual(deleted, ["/repos/me/reco-models/git/refs/heads/model/basketball-small-halle-kunstlicht-v1.0.0", "/repos/me/reco-models/releases/7", "/repos/me/reco-models/git/refs/tags/basketball-small-halle-kunstlicht-v1.0.0"])


class CancelTests(ServerTest):
    def test_cancelling_cleans_the_fork_like_a_failure(self):
        weights = self.dir / "w.safetensors"
        weights_file(weights)

        def status(step):
            if step == "pull-request":
                raise SystemExit(143)

        with self.assertRaises(SystemExit):
            model_hub.share(model_hub.GitHub("tok", self.url), weights, share_model(), status, repo="o/reco-models", wait=lambda _: None)
        deleted = [path for method, path, *_ in FakeGitHub.calls if method == "DELETE"]
        self.assertIn("/repos/me/reco-models/releases/7", deleted)


class BeforeAfterTests(unittest.TestCase):
    def run_compare(self, before, after):
        import ml_worker
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "project.json").write_text(json.dumps({"sport": "football", "frames": []}), encoding="utf-8")
            activated = []
            parent = (root / "manifest.json", {"hub": {"lineage": "football/nano/a@1.0.0"}})
            with mock.patch.object(ml_worker, "hub_score_on_own_images", return_value={"mAP50": after, "validationImages": 3, "annotations": 9}), mock.patch.object(ml_worker, "activate_library_model", lambda *args: activated.append(args[1])), mock.patch.object(ml_worker, "require_project", return_value={"sport": "football"}), mock.patch.object(ml_worker, "project_file", return_value=root / "project.json"), contextlib.redirect_stdout(io.StringIO()):
                ml_worker.compare_with_hub_parent(root, "nano", parent, {"mAP50": before}, root / "c.pth", "en")
            return activated, json.loads((root / "project.json").read_text(encoding="utf-8"))["hubComparison"]

    def test_worse_keeps_the_hub_model(self):
        activated, result = self.run_compare(0.8, 0.7)
        self.assertEqual(len(activated), 1)
        self.assertTrue(result["keptParent"])

    def test_better_keeps_the_new_model(self):
        activated, result = self.run_compare(0.7, 0.8)
        self.assertEqual(activated, [])
        self.assertFalse(result["keptParent"])


if __name__ == "__main__":
    unittest.main()
