"""Windows with an NVIDIA card: a CPU-only PyTorch gets replaced by the matching CUDA build."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]


def load_worker():
    spec = importlib.util.spec_from_file_location("cuda_repair_local_worker", ROOT / "local_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU = {"version": "2.9.1+cpu", "cuda": None, "available": False, "works": False, "vision": "0.24.1+cpu"}
WORKING = {"version": "2.9.1+cu126", "cuda": "12.6", "available": True, "works": True, "vision": "0.24.1+cu126"}


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker()
        self.commands: list[list[str]] = []
        self.logs: list[str] = []
        self.worker.run_logged = lambda command, extra_env=None: self.commands.append(command)
        self.worker.STATE.append_log = self.logs.append
        self.worker.STATE.update(language="en")

    def tearDown(self):
        self.worker.STATE.update(language="de")

    def run_with(self, reports):
        sequence = iter(reports)
        with mock.patch.object(self.worker, "torch_cuda_report", side_effect=lambda _python: next(sequence)):
            self.worker.ensure_cuda_torch(Path("python"))

    def test_nothing_happens_when_the_card_already_works(self):
        self.run_with([WORKING])
        self.assertEqual(self.commands, [])

    def test_a_cpu_build_is_replaced_by_the_cuda_build_of_the_same_version(self):
        self.run_with([CPU, WORKING])
        self.assertEqual(len(self.commands), 1)
        command = self.commands[0]
        self.assertIn("--force-reinstall", command)
        self.assertIn("--no-deps", command)
        self.assertIn("torch==2.9.1", command)
        self.assertIn("torchvision==0.24.1", command)
        self.assertEqual(command[command.index("--index-url") + 1], "https://download.pytorch.org/whl/cu126")
        self.assertTrue(any("used now" in line for line in self.logs))

    def test_the_next_index_is_tried_when_the_first_does_not_make_the_card_work(self):
        self.run_with([CPU, CPU, WORKING])
        self.assertEqual([command[command.index("--index-url") + 1].rsplit("/", 1)[1] for command in self.commands], ["cu126", "cu128"])

    def test_a_failing_install_is_logged_and_never_fatal(self):
        def failing(command, extra_env=None):
            self.commands.append(command)
            raise RuntimeError("no network")

        self.worker.run_logged = failing
        self.run_with([CPU])
        self.assertEqual(len(self.commands), 2)
        self.assertTrue(any("CPU" in line for line in self.logs))

    def test_a_cuda_build_that_does_not_compute_is_not_reinstalled(self):
        self.run_with([{**WORKING, "works": False, "error": "no kernel image"}])
        self.assertEqual(self.commands, [])
        self.assertTrue(any("no kernel image" in line for line in self.logs))

    def test_an_unreadable_check_changes_nothing(self):
        self.run_with([{}])
        self.assertEqual(self.commands, [])

    def test_the_check_script_reports_what_the_code_reads(self):
        for key in ("version", "cuda", "available", "works", "vision"):
            self.assertIn(f'"{key}"', self.worker.CUDA_CHECK_SCRIPT)
        # on a computer with no torch the script fails; the report is then empty and nothing is touched
        with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="no torch")):
            self.assertEqual(self.worker.torch_cuda_report(Path("python")), {})
        with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="noise\n" + json.dumps(CPU) + "\n", stderr="")):
            self.assertEqual(self.worker.torch_cuda_report(Path("python"))["cuda"], None)

    def test_setup_repairs_before_it_checks_that_torch_loads(self):
        source = (ROOT / "local_worker.py").read_text(encoding="utf-8")
        self.assertLess(source.index("ensure_cuda_torch(venv_python)  # an NVIDIA card"), source.index("            ensure_torch_importable(venv_python)\n            # The accelerator label"))


if __name__ == "__main__":
    unittest.main()
