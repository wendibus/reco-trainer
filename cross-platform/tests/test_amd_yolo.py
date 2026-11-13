"""YOLO on an AMD card with MIOpen problems: start over without MIOpen; the other AMD details."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import amd_training  # noqa: E402
import yolo_training  # noqa: E402

MIOPEN_ERROR = "RuntimeError: miopenStatusUnknownError ('type_traits' file not found)"


class FakeUltralytics:
    """Stands in for ultralytics: the first train() call fails the way the tester's card did."""
    def __init__(self, failures: list[str]):
        self.failures = failures
        self.trained = 0
        self.updates: list[dict] = []
        self.settings = types.SimpleNamespace(update=lambda values: self.updates.append(values))
        outer = self

        class YOLO:
            def __init__(self, weights):
                self.weights = weights

            def add_callback(self, *_):
                pass

            def train(self, **options):
                outer.trained += 1
                if outer.failures:
                    raise RuntimeError(outer.failures.pop(0))
                run = Path(options["project"]) / options["name"] / "weights"
                run.mkdir(parents=True, exist_ok=True)
                (run / "best.pt").write_bytes(b"x")

            def val(self, **_):
                return types.SimpleNamespace(results_dict={})

            def export(self, **_):
                path = Path(self.weights).with_suffix(".onnx")
                path.write_bytes(b"onnx")
                return str(path)

        self.YOLO = YOLO


class MiopenRetryTests(unittest.TestCase):
    def run_train(self, failures, device="cuda", env=None):
        fake = FakeUltralytics(list(failures))
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=True)))
        notices: list[str] = []
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(sys.modules, {"ultralytics": fake, "torch": torch}), mock.patch.dict("os.environ", env or {}):
            folder = Path(temp)
            (folder / "data.yaml").write_text("train: images/train\n", encoding="utf-8")
            (folder / "start.pt").write_bytes(b"x")
            try:
                result = yolo_training.train("yolo-nano", folder / "start.pt", folder / "data.yaml", folder / "runs" / "yolo-nano", device, 3, 5, 640, False, lambda *_: None, notices.append)
            except RuntimeError as error:
                return fake, torch, notices, error
            return fake, torch, notices, result

    def test_a_miopen_failure_starts_over_without_miopen_and_says_so(self):
        fake, torch, notices, result = self.run_train([MIOPEN_ERROR])
        self.assertEqual(fake.trained, 2)
        self.assertFalse(torch.backends.cudnn.enabled)
        self.assertEqual(notices, ["miopen"])
        self.assertTrue(Path(result["best"]).name.endswith(".pt"))

    def test_other_errors_and_other_devices_are_not_retried(self):
        fake, _, notices, error = self.run_train(["CUDA out of memory"])
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual((fake.trained, notices), (1, []))
        fake, _, _, error = self.run_train([MIOPEN_ERROR], device="cpu")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(fake.trained, 1)

    def test_it_is_tried_only_once(self):
        fake, _, _, error = self.run_train([MIOPEN_ERROR, MIOPEN_ERROR])
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(fake.trained, 2)

    def test_the_check_after_the_install_can_switch_miopen_off_from_the_start(self):
        fake, torch, notices, _ = self.run_train([], env={"RECO_NO_MIOPEN": "1"})
        self.assertEqual((fake.trained, notices), (1, []))
        self.assertFalse(torch.backends.cudnn.enabled)

    def test_downloaded_weights_go_to_the_app_data_not_the_working_folder(self):
        fake, _, _, _ = self.run_train([])
        self.assertEqual(fake.updates[0]["weights_dir"], str(yolo_training.weights_folder()))
        self.assertEqual(yolo_training.weights_folder().name, "yolo-weights")

    def test_which_errors_count_as_miopen(self):
        for text in (MIOPEN_ERROR, "hiprtc compile failed", "MIOpen error"):
            self.assertTrue(yolo_training.is_miopen_failure(RuntimeError(text)))
        self.assertFalse(yolo_training.is_miopen_failure(RuntimeError("CUDA out of memory")))


class VerifyScriptTests(unittest.TestCase):
    def test_the_check_tries_a_batchnorm_step_but_it_does_not_decide_whether_the_card_works(self):
        self.assertIn("BatchNorm2d", amd_training.VERIFY_SCRIPT)
        self.assertIn('info["batchNorm"]', amd_training.VERIFY_SCRIPT)
        good = {"available": True, "backward": True, "matmulDifference": 0.0, "batchNorm": False}
        self.assertTrue(amd_training.verified(good))

    def test_miopen_broken_reads_the_stored_check(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.assertFalse(amd_training.miopen_broken(folder))
            amd_training.write_state(folder, {"verified": {"available": True, "batchNorm": False}})
            self.assertTrue(amd_training.miopen_broken(folder))
            amd_training.write_state(folder, {"verified": {"available": True, "batchNorm": True}})
            self.assertFalse(amd_training.miopen_broken(folder))
            amd_training.write_state(folder, {"verified": {"available": True}})  # checked before this version: unknown
            self.assertFalse(amd_training.miopen_broken(folder))


class WorkerTests(unittest.TestCase):
    def load(self, name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_rocm_environment_carries_the_switch_only_when_the_check_found_the_problem(self):
        worker = self.load("amd_yolo_local_worker", "local_worker.py")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with mock.patch.object(worker, "rocm_folder", return_value=folder):
                self.assertNotIn("RECO_NO_MIOPEN", worker.rocm_training_env())
                amd_training.write_state(folder, {"verified": {"batchNorm": False}})
                self.assertEqual(worker.rocm_training_env()["RECO_NO_MIOPEN"], "1")
                self.assertEqual(worker.rocm_training_env()["ROCM_SDK_TARGET_FAMILY"], "custom")

    def test_a_step_that_already_printed_its_result_does_not_sound_like_an_error(self):
        ml = self.load("amd_yolo_ml_worker", "ml_worker.py")
        stopped = ml.localized_step_stopped("Medium", "en", finished=True)
        self.assertIn("not an error", stopped)
        self.assertIn("result saved", stopped)
        self.assertIn("did not end by itself and was stopped", ml.localized_step_stopped("Medium", "en"))
        self.assertNotIn("not an error", ml.localized_step_stopped("Medium", "en"))


if __name__ == "__main__":
    unittest.main()
