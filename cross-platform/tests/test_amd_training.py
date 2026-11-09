import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import amd_training
import local_worker
import ml_worker

GOOD_CHECK = {"available": True, "backward": True, "matmulDifference": 0.0005, "device": "AMD Radeon RX 7800 XT"}


def install(folder: Path, check: dict | None = GOOD_CHECK, **state) -> None:
    python = amd_training.venv_python(folder)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    amd_training.write_state(folder, {"verified": check, **state})


class CardTests(unittest.TestCase):
    def test_rdna3_and_rdna4_cards_are_offered_older_ones_not(self):
        for name in ("AMD Radeon RX 7800 XT", "AMD Radeon RX 7900 XTX", "AMD Radeon RX 9070 XT", "AMD Radeon PRO W7900", "AMD Radeon AI PRO R9700", "AMD Radeon RX 7700 XT"):
            self.assertTrue(amd_training.card_supported({"vendor": "amd", "name": name}), name)
        for card in ({"vendor": "amd", "name": "AMD Radeon RX 6800 XT"}, {"vendor": "amd", "name": "AMD Radeon RX 580"}, {"vendor": "amd", "name": "AMD Radeon RX 7600"},
                     {"vendor": "amd", "name": "AMD Radeon(TM) 8060S Graphics"},
                     {"vendor": "intel", "name": "Intel Arc A770"}, None):
            self.assertFalse(amd_training.card_supported(card), card)


class EnvironmentTests(unittest.TestCase):
    def test_install_pins_rfdetr_to_the_normal_environment_and_keeps_amds_torch(self):
        steps = amd_training.install_steps(Path("py.exe"), "1.11.2", Path("c.txt"))
        names = [name for name, _ in steps]
        self.assertEqual(names, ["pip", "ROCm SDK", "PyTorch (ROCm)", "RF-DETR"])
        rfdetr = steps[-1][1]
        self.assertIn("rfdetr[train,onnx]==1.11.2", rfdetr)
        self.assertEqual(rfdetr[rfdetr.index("-c") + 1], "c.txt")
        self.assertIn("torch==2.9.1+rocm7.2.1", amd_training.CONSTRAINTS)
        self.assertTrue(all(url.startswith("https://repo.radeon.com/") for url in amd_training.SDK_WHEELS + amd_training.TORCH_WHEELS))
        # A broken version string falls back to the tested version instead of reaching pip.
        self.assertIn(f"rfdetr[train,onnx]=={amd_training.FALLBACK_RFDETR}", amd_training.install_steps(Path("p"), "1.0; rm -rf", Path("c"))[-1][1])

    def test_only_a_checked_and_switched_on_environment_trains(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertIsNone(amd_training.training_python(folder))
            install(folder, {"available": True, "backward": True, "matmulDifference": 3.0})
            self.assertIsNone(amd_training.training_python(folder))  # wrong results on the card
            install(folder)
            self.assertEqual(amd_training.training_python(folder), amd_training.venv_python(folder))
            install(folder, enabled=False)
            self.assertIsNone(amd_training.training_python(folder))

    def test_status_is_only_offered_on_windows_with_a_supported_card(self):
        card = {"vendor": "amd", "name": "AMD Radeon RX 7800 XT"}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict("os.environ", {"RECO_SIMULATE_AMD": ""}):
            folder = Path(temporary)
            with mock.patch.object(amd_training, "is_windows_x64", return_value=False):
                self.assertIsNone(amd_training.status(folder, card))
            with mock.patch.object(amd_training, "is_windows_x64", return_value=True):
                self.assertEqual(amd_training.status(folder, card)["ready"], False)
                self.assertIsNone(amd_training.status(folder, {"vendor": "amd", "name": "AMD Radeon RX 6600"}))
                install(folder)
                self.assertTrue(amd_training.status(folder, card)["enabled"])


class WorkerTests(unittest.TestCase):
    def run_action(self, action: str, payload: dict, rocm_python: Path | None, yolo_on_card: bool = False):
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
            local_worker.STATE.update(project_root=root, project={"sport": "basketball", "frames": []})
            python = local_worker.venv_python_path(root / ".runtime" / "venv")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")
            with mock.patch.object(local_worker, "run_logged", side_effect=lambda command, env=None: calls.append((command, env))), \
                    mock.patch.object(local_worker, "rocm_status", return_value={"enabled": rocm_python is not None}), \
                    mock.patch.object(local_worker, "remember_rocm_problem"), \
                    mock.patch.object(local_worker, "ensure_yolo"), \
                    mock.patch.object(local_worker, "ensure_yolo_rocm", return_value=yolo_on_card), \
                    mock.patch.object(amd_training, "training_python", return_value=rocm_python):
                local_worker.ml_action(action, payload)
        return calls

    def test_training_uses_the_rocm_python_when_switched_on(self):
        calls = self.run_action("train", {"language": "en"}, Path("C:/rocm/python.exe"))
        self.assertEqual(calls[0][0][0], str(Path("C:/rocm/python.exe")))
        self.assertEqual(calls[0][1], local_worker.ROCM_TRAINING_ENV)
        calls = self.run_action("train", {"language": "en"}, None)
        self.assertNotEqual(calls[0][0][0], str(Path("C:/rocm/python.exe")))
        self.assertIsNone(calls[0][1])

    def test_a_batch_with_yolo_trains_on_the_card_when_yolo_is_set_up_there(self):
        rocm = str(Path("C:/rocm/python.exe"))
        for models in (["nano", "yolo-nano"], ["yolo-medium"]):
            calls = self.run_action("train-batch", {"language": "en", "models": models, "yoloLicenseAccepted": True}, Path("C:/rocm/python.exe"), yolo_on_card=True)
            self.assertEqual(calls[0][0][0], rocm)
            self.assertNotIn("--rfdetr-python", calls[0][0])
            self.assertEqual(calls[0][1], local_worker.ROCM_TRAINING_ENV)

    def test_a_batch_with_yolo_trains_only_rfdetr_on_the_card(self):
        calls = self.run_action("train-batch", {"language": "en", "models": ["nano", "yolo-nano"], "yoloLicenseAccepted": True}, Path("C:/rocm/python.exe"))
        command = calls[0][0]
        self.assertNotEqual(command[0], str(Path("C:/rocm/python.exe")))
        self.assertEqual(command[command.index("--rfdetr-python") + 1], str(Path("C:/rocm/python.exe")))
        self.assertIsNone(calls[0][1])
        calls = self.run_action("train-batch", {"language": "en", "models": ["yolo-nano"], "yoloLicenseAccepted": True}, Path("C:/rocm/python.exe"))
        self.assertNotIn("--rfdetr-python", calls[0][0])
        calls = self.run_action("train-batch", {"language": "en", "models": ["nano", "yolo-nano"], "yoloLicenseAccepted": True}, None)
        self.assertNotIn("--rfdetr-python", calls[0][0])
        calls = self.run_action("train-batch", {"language": "en", "models": ["nano", "small"]}, Path("C:/rocm/python.exe"))
        self.assertEqual(calls[0][0][0], str(Path("C:/rocm/python.exe")))

    def test_a_rocm_training_crash_does_not_start_the_directml_report(self):
        with mock.patch.object(local_worker, "amd_choice", return_value="yes"), \
                mock.patch.object(local_worker, "rocm_status", return_value={"enabled": True}), \
                mock.patch.object(local_worker, "record_gpu_problem") as report:
            local_worker.record_job_failure("train", "exited with code 1")
            report.assert_not_called()
            local_worker.record_job_failure("autolabel", "exited with code 1")
            report.assert_called_once()


class TargetFamilyTests(unittest.TestCase):
    def test_every_rocm_process_skips_the_offload_arch_lookup(self):
        # AMD's rocm_sdk runs offload-arch unless this is set; on paths with spaces that fails noisily.
        self.assertEqual(local_worker.ROCM_TRAINING_ENV["ROCM_SDK_TARGET_FAMILY"], "custom")
        self.assertEqual(ml_worker.ROCM_STEP_ENV, local_worker.ROCM_TRAINING_ENV)


class YoloOnCardTests(unittest.TestCase):
    def test_yolo_install_keeps_amds_torch_and_adds_opencv_only_when_missing(self):
        constraints = Path("C:/rocm/constraints.txt")
        steps = amd_training.yolo_install_steps(Path("C:/rocm/python.exe"), "8.4.180", constraints, has_cv2=True)
        self.assertTrue(all(["-c", str(constraints)] == step[step.index("-c"):step.index("-c") + 2] for step in steps))
        self.assertIn("--no-deps", steps[0])
        self.assertIn("ultralytics==8.4.180", steps[0])
        self.assertFalse(any("opencv" in part for step in steps for part in step))
        steps = amd_training.yolo_install_steps(Path("C:/rocm/python.exe"), "8.4.180", constraints, has_cv2=False)
        self.assertIn("opencv-python-headless>=4.10.0", steps[-1])
        self.assertFalse(any(part.startswith("torch") for step in steps for part in step))

    def test_a_failed_install_leaves_yolo_on_the_cpu(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(local_worker, "rocm_folder", return_value=Path(temporary)), \
                mock.patch.object(local_worker, "run_logged", side_effect=RuntimeError("pip failed")), \
                mock.patch.object(local_worker.subprocess, "run", return_value=mock.Mock(returncode=1)):
            self.assertFalse(local_worker.ensure_yolo_rocm(Path("C:/rocm/python.exe")))
        with mock.patch.object(local_worker.subprocess, "run", return_value=mock.Mock(returncode=0)), \
                mock.patch.object(local_worker, "run_logged") as install:
            self.assertTrue(local_worker.ensure_yolo_rocm(Path("C:/rocm/python.exe")))
            install.assert_not_called()


class SetupTests(unittest.TestCase):
    def setup_with(self, check_output: str):
        commands = []
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "rocm-env"
            main = Path(temporary) / "main" / "python"
            main.parent.mkdir(parents=True)
            main.write_text("")

            def fake_run(command, **kwargs):
                if "-c" in command and "importlib.metadata" in command[-1]:
                    return mock.Mock(stdout="1.11.2\n", stderr="", returncode=0)
                return mock.Mock(stdout=check_output, stderr="", returncode=0)

            def fake_logged(command, env=None):
                commands.append(command)
                if command[1:3] == ["-m", "venv"]:
                    python = amd_training.venv_python(folder)
                    python.parent.mkdir(parents=True, exist_ok=True)
                    python.write_text("")

            with mock.patch.object(local_worker, "rocm_folder", return_value=folder), \
                    mock.patch.object(local_worker, "rocm_status", return_value={"ready": False}), \
                    mock.patch.object(local_worker, "ml_venv", return_value=main.parent), \
                    mock.patch.object(local_worker, "venv_python_path", return_value=main), \
                    mock.patch.object(amd_training, "find_python312", return_value="C:/Python312/python.exe"), \
                    mock.patch.object(local_worker, "run_logged", side_effect=fake_logged), \
                    mock.patch.object(local_worker.subprocess, "run", side_effect=fake_run):
                local_worker.rocm_setup_action({})
            return commands, amd_training.read_state(folder), local_worker.STATE.error

    def test_setup_installs_in_order_and_switches_on_after_the_check(self):
        commands, state, error = self.setup_with(json.dumps(GOOD_CHECK))
        self.assertIsNone(error)
        self.assertEqual(commands[0][1:3], ["-m", "venv"])
        self.assertTrue(any("rfdetr[train,onnx]==1.11.2" in command for command in commands))
        self.assertTrue(state["enabled"])
        self.assertTrue(amd_training.verified(state["verified"]))

    def test_a_card_that_fails_the_check_is_not_used(self):
        commands, state, error = self.setup_with(json.dumps({"available": False}))
        self.assertIn("CPU", error)
        self.assertFalse(amd_training.verified(state["verified"]))


if __name__ == "__main__":
    unittest.main()
