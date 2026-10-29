"""Job management: queue, cooperative cancel with process-tree kill and cleanup, stages, SSE."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

WORKER_PATH = Path(__file__).parents[1] / "local_worker.py"
SPEC = importlib.util.spec_from_file_location("local_worker_jobs", WORKER_PATH)
assert SPEC and SPEC.loader
local_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_worker)

# Stands in for ml_worker.py: writes a partial export, starts a grandchild, reports a stage and
# progress, then works "forever" - or fails at once when FAIL is in the arguments.
FAKE_ML_WORKER = """
import os, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
if "FAIL" in args:
    print("something broke", flush=True)
    sys.exit(3)
project = Path(args[args.index("--project") + 1])
out = project / "exports" / "onnx"
out.mkdir(parents=True, exist_ok=True)
(out / "reco-partial.onnx").write_bytes(b"half")
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
(project / "grandchild.pid").write_text(str(child.pid))
print("RECO_STAGE 2 3 Schritt zwei", flush=True)
print("RECO_PROGRESS 1 4", flush=True)
time.sleep(120)
"""


def wait_until(condition, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A killed child of a dead parent may linger as a zombie until reaped by init.
    try:
        import subprocess
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(state) and not state.startswith("Z")
    except OSError:
        return True


class JobTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        self.root.mkdir()
        local_worker.atomic_json(self.root / "project.json", {"sport": "basketball", "frames": []})
        local_worker.JOBS.__init__()
        local_worker.CANCEL_REQUESTED.clear()
        local_worker.STATE.update(project_root=self.root, project={"sport": "basketball", "frames": []}, busy=False, error=None,
                                  operation="idle", language="en", log=[], stage=None, detail=None, progress=0.0)
        venv_python = local_worker.venv_python_path(self.root / ".runtime" / "venv")
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.symlink_to(sys.executable)
        self.fake_worker = Path(self.temporary.name) / "fake_ml_worker.py"
        self.fake_worker.write_text(FAKE_ML_WORKER, encoding="utf-8")
        self.patch = mock.patch.object(local_worker, "find_ml_worker", lambda: self.fake_worker)
        self.patch.start()

    def tearDown(self):
        local_worker.CANCEL_REQUESTED.set()
        local_worker.terminate_active_processes()
        wait_until(lambda: not local_worker.JOBS.has_pending(), 10)
        self.patch.stop()
        local_worker.STATE.update(language="de", busy=False, operation="idle", stage=None)
        self.temporary.cleanup()

    def finished(self, job_id: str) -> dict | None:
        return next((job for job in local_worker.JOBS.snapshot()["recent"] if job["id"] == job_id), None)


class QueueTests(JobTestCase):
    def test_jobs_run_one_after_another_and_queued_jobs_can_be_cancelled(self):
        gate = threading.Event()
        order: list[str] = []

        def first():
            order.append("first")
            gate.wait(5)
            local_worker.STATE.update(operation="ready", busy=False)

        def second():
            order.append("second")
            local_worker.STATE.update(operation="ready", busy=False)

        job1 = local_worker.JOBS.submit("first", first)
        job2 = local_worker.JOBS.submit("second", second)
        job3 = local_worker.JOBS.submit("third", second)
        snapshot = local_worker.JOBS.snapshot()
        self.assertEqual(snapshot["current"]["id"], job1["id"])
        self.assertEqual([job["status"] for job in snapshot["queue"]], ["queued", "queued"])

        cancelled = local_worker.JOBS.cancel(job3["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        gate.set()
        self.assertTrue(wait_until(lambda: not local_worker.JOBS.has_pending()))
        self.assertEqual(order, ["first", "second"])
        self.assertEqual(self.finished(job2["id"])["status"], "done")

    def test_a_failing_job_reports_its_error(self):
        job = local_worker.JOBS.submit("export-onnx", local_worker.ml_action, "export-onnx", {"language": "en", "model": "FAIL"})
        self.assertTrue(wait_until(lambda: self.finished(job["id"]) is not None))
        finished = self.finished(job["id"])
        self.assertEqual(finished["status"], "failed")
        self.assertIn("exited with code 3", finished["error"])


class CancelTests(JobTestCase):
    def start_export(self) -> dict:
        exports = self.root / "exports" / "onnx"
        exports.mkdir(parents=True)
        older = exports / "older-export.onnx"
        older.write_bytes(b"complete")
        old_time = time.time() - 3600
        os.utime(older, (old_time, old_time))
        job = local_worker.JOBS.submit("export-onnx", local_worker.ml_action, "export-onnx", {"language": "en"})
        self.assertTrue(wait_until(lambda: (self.root / "grandchild.pid").is_file() and local_worker.STATE.stage is not None))
        return job

    def test_cancel_stops_the_process_tree_and_removes_partial_output(self):
        job = self.start_export()
        grandchild = int((self.root / "grandchild.pid").read_text())
        running = local_worker.JOBS.snapshot()["current"]
        self.assertEqual(running["stage"], {"index": 2, "count": 3, "label": "Schritt zwei"})
        self.assertTrue(running["cancellable"])

        started = time.time()
        self.assertEqual(local_worker.JOBS.cancel(job["id"])["status"], "cancelling")
        self.assertTrue(wait_until(lambda: self.finished(job["id"]) is not None, 10))
        self.assertLess(time.time() - started, 8)
        self.assertEqual(self.finished(job["id"])["status"], "cancelled")
        self.assertFalse((self.root / "exports" / "onnx" / "reco-partial.onnx").exists())
        self.assertTrue((self.root / "exports" / "onnx" / "older-export.onnx").exists())
        self.assertTrue(wait_until(lambda: not pid_alive(grandchild), 5), "grandchild survived the cancel")
        self.assertIn("half-written files were removed", local_worker.STATE.message)

    def test_cancelled_training_keeps_its_checkpoint_and_says_so(self):
        job = local_worker.JOBS.submit("train", local_worker.ml_action, "train", {"language": "en", "epochs": 1})
        self.assertTrue(wait_until(lambda: (self.root / "grandchild.pid").is_file()))
        local_worker.JOBS.cancel(job["id"])
        self.assertTrue(wait_until(lambda: self.finished(job["id"]) is not None, 10))
        self.assertEqual(self.finished(job["id"])["status"], "cancelled")
        self.assertIn("last saved checkpoint is kept", self.finished(job["id"])["message"])

    def test_cancel_endpoint_without_running_job_is_refused(self):
        with self.assertRaises(RuntimeError):
            local_worker.JOBS.cancel("0123456789ab")


class EventStreamTests(JobTestCase):
    def test_events_stream_sends_the_job_list(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), local_worker.Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/events", timeout=5) as response:
                self.assertEqual(response.headers["Content-Type"], "text/event-stream; charset=utf-8")
                self.assertEqual(response.readline().decode().strip(), "event: jobs")
                data = json.loads(response.readline().decode().removeprefix("data: "))
                self.assertEqual(set(data), {"current", "queue", "recent"})
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
