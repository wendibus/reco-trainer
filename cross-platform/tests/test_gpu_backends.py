"""AMD and Intel graphics: RF-DETR through ONNX Runtime (DirectML) and the ML setup that installs it.

The Vulkan build of the VLM is covered in test_vlm.py.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml_worker = load_module("gpu_test_ml_worker", ROOT / "ml_worker.py")
local_worker = load_module("gpu_test_local_worker", ROOT / "local_worker.py")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def fake_onnxruntime(*providers: str) -> types.ModuleType:
    module = types.ModuleType("onnxruntime")
    module.get_available_providers = lambda: list(providers)  # type: ignore[attr-defined]
    return module


class ProviderChoiceTests(unittest.TestCase):
    def choose(self, setting: str, device: str, *providers: str):
        environment = {"RECO_ONNX_INFERENCE": setting} if setting else {}
        with mock.patch.dict(os.environ, environment, clear=False), \
                mock.patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime(*providers)}), \
                mock.patch.object(ml_worker, "detect_device", return_value=device):
            if not setting:
                os.environ.pop("RECO_ONNX_INFERENCE", None)
            return ml_worker.onnx_inference_providers()

    def test_directml_is_used_only_where_pytorch_has_no_gpu(self):
        self.assertEqual(self.choose("", "cpu", "DmlExecutionProvider", "CPUExecutionProvider"), ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertIsNone(self.choose("", "cuda", "DmlExecutionProvider", "CPUExecutionProvider"))
        self.assertIsNone(self.choose("", "mps", "CPUExecutionProvider"))
        self.assertIsNone(self.choose("", "cpu", "CPUExecutionProvider"))  # plain onnxruntime: keep PyTorch

    def test_setting_switches_it_on_or_off(self):
        self.assertIsNone(self.choose("off", "cpu", "DmlExecutionProvider", "CPUExecutionProvider"))
        self.assertEqual(self.choose("on", "mps", "CPUExecutionProvider"), ["CPUExecutionProvider"])
        self.assertEqual(self.choose("on", "cuda", "DmlExecutionProvider", "CPUExecutionProvider"), ["DmlExecutionProvider", "CPUExecutionProvider"])

    def test_missing_onnxruntime_keeps_pytorch(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}), \
                mock.patch.object(ml_worker, "detect_device", return_value="cpu"):
            self.assertIsNone(ml_worker.onnx_inference_providers())


class AcceleratedModelTests(unittest.TestCase):
    def test_without_providers_the_pytorch_model_is_returned(self):
        model = object()
        with mock.patch.object(ml_worker, "onnx_inference_providers", return_value=None):
            self.assertIs(ml_worker.accelerated_model(model, None, "nano"), model)

    def test_a_failing_export_keeps_pytorch(self):
        model = types.SimpleNamespace(model=object(), export=mock.Mock(side_effect=RuntimeError("opset not supported")))
        messages: list[str] = []
        with mock.patch.object(ml_worker, "onnx_inference_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]), \
                mock.patch.object(ml_worker, "onnx_cache_dir", return_value=Path(self.enterContext(_tempdir()))), \
                mock.patch.dict(sys.modules, {"rfdetr": types.SimpleNamespace(__version__="1.11.2")}), \
                mock.patch.object(ml_worker, "emit", side_effect=messages.append):
            self.assertIs(ml_worker.accelerated_model(model, None, "nano", "en"), model)
        self.assertTrue(any("opset not supported" in message and "PyTorch on the CPU" in message for message in messages))

    def test_onnx_failure_while_detecting_falls_back_to_pytorch(self):
        torch_model = types.SimpleNamespace(predict=mock.Mock(return_value=["torch result"]), class_names=["ball"])
        wrapped = object.__new__(ml_worker.OnnxRfdetrModel)
        wrapped.__dict__.update(torch_model=torch_model, np=None, providers=["DmlExecutionProvider"], fallback_reason=None)
        wrapped._tensor = mock.Mock(side_effect=RuntimeError("DML device removed"))
        with mock.patch.object(ml_worker, "emit") as emitted:
            self.assertEqual(wrapped.predict(["a.jpg"], threshold=0.4), ["torch result"])
            self.assertEqual(wrapped.predict(["b.jpg"], threshold=0.4), ["torch result"])
        self.assertEqual(emitted.call_count, 1)
        self.assertEqual(torch_model.predict.call_count, 2)
        self.assertEqual(wrapped.class_names, ["ball"])  # everything else is the PyTorch model's


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class DecodeTests(unittest.TestCase):
    def test_decode_matches_rfdetr_postprocess(self):
        # 3 queries x 2 classes; logit 0 -> 0.5, 2 -> 0.88, -2 -> 0.12
        logits = np.array([[2.0, -2.0], [0.0, 2.0], [-2.0, -2.0]], dtype=np.float32)
        boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.4, 0.4], [0.9, 0.9, 0.1, 0.1]], dtype=np.float32)
        xyxy, scores, labels = ml_worker.decode_rfdetr_outputs(np, boxes, logits, 200, 100, threshold=0.5, num_select=300)
        # Ranked over all query/class pairs; score 0.5 is not > 0.5. Query 0 class 0 comes before
        # query 1 class 1 on equal scores (stable order), and boxes are clamped to the image.
        self.assertEqual(labels.tolist(), [0, 1])
        np.testing.assert_allclose(scores, [1 / (1 + np.exp(-2.0))] * 2, rtol=1e-6)
        np.testing.assert_allclose(xyxy[0], [80, 30, 120, 70], atol=1e-4)
        np.testing.assert_allclose(xyxy[1], [0, 0, 60, 30], atol=1e-4)

    def test_num_select_caps_the_candidates_before_the_threshold(self):
        logits = np.full((4, 1), 3.0, dtype=np.float32)
        boxes = np.tile(np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32), (4, 1))
        _xyxy, scores, _labels = ml_worker.decode_rfdetr_outputs(np, boxes, logits, 10, 10, threshold=0.1, num_select=2)
        self.assertEqual(len(scores), 2)

    def test_coco_base_models_keep_sparse_category_ids(self):
        coco = types.ModuleType("rfdetr.assets.coco_classes")
        coco.COCO_CLASS_NAMES = ["person", "bicycle"]  # type: ignore[attr-defined]
        coco.COCO_CLASSES = {1: "person", 2: "bicycle"}  # type: ignore[attr-defined]
        base = types.SimpleNamespace(class_names=["person", "bicycle"], model=types.SimpleNamespace(args=types.SimpleNamespace(num_classes=91)))
        tuned = types.SimpleNamespace(class_names=["ball", "player"], model=types.SimpleNamespace(args=types.SimpleNamespace(num_classes=2)))
        with mock.patch.dict(sys.modules, {"rfdetr.assets.coco_classes": coco}):
            self.assertEqual(ml_worker.rfdetr_class_names(base), {1: "person", 2: "bicycle"})
            self.assertEqual(ml_worker.rfdetr_class_names(tuned), {0: "ball", 1: "player"})


import teachers  # noqa: E402


class TierTests(unittest.TestCase):
    def test_directml_reaches_medium_at_most(self):
        self.assertEqual(teachers.choose_tier("directml", vram_gb=16), "medium")
        self.assertEqual(teachers.choose_tier("directml", vram_gb=8), "medium")
        self.assertEqual(teachers.choose_tier("directml", vram_gb=0.5), "weak")  # integrated graphics
        self.assertEqual(teachers.choose_tier("directml"), "weak")

    def test_registry_entries_become_cards(self):
        radeon = teachers.windows_graphics_entry({"DriverDesc": "AMD Radeon RX 7800 XT", "ProviderName": "Advanced Micro Devices, Inc.", "HardwareInformation.qwMemorySize": 16 * 1024**3})
        self.assertEqual(radeon, {"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vramGB": 16.0})
        arc = teachers.windows_graphics_entry({"DriverDesc": "Intel(R) Arc(TM) A770 Graphics", "ProviderName": "Intel Corporation", "HardwareInformation.MemorySize": (16 * 1024**3).to_bytes(8, "little")})
        self.assertEqual((arc["vendor"], arc["vramGB"]), ("intel", 16.0))
        self.assertIsNone(teachers.windows_graphics_entry({"DriverDesc": "Microsoft Basic Display Adapter", "ProviderName": "Microsoft"}))
        self.assertEqual(teachers.windows_graphics_entry({"DriverDesc": "Intel(R) UHD Graphics", "ProviderName": "Intel Corporation"})["vramGB"], 0.0)

    def test_best_amd_or_intel_card_unless_nvidia_is_there(self):
        radeon = {"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vramGB": 16.0}
        uhd = {"name": "Intel(R) UHD Graphics", "vendor": "intel", "vramGB": 0.1}
        with mock.patch.object(teachers, "windows_graphics", return_value=[uhd, radeon]):
            self.assertEqual(teachers.directml_graphics(), radeon)
        with mock.patch.object(teachers, "windows_graphics", return_value=[uhd, {"name": "RTX 4070", "vendor": "nvidia", "vramGB": 12.0}]):
            self.assertIsNone(teachers.directml_graphics())
        self.assertEqual(teachers.windows_graphics() if sys.platform != "win32" else [], [])

    def test_ml_worker_tier_uses_the_directml_card(self):
        with mock.patch.object(ml_worker, "detect_device", return_value="cpu"), \
                mock.patch.object(ml_worker, "onnx_inference_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]), \
                mock.patch.object(teachers, "directml_graphics", return_value={"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vramGB": 16.0}):
            self.assertEqual(ml_worker.hardware_tier(), "medium")
        with mock.patch.object(ml_worker, "detect_device", return_value="cpu"), \
                mock.patch.object(ml_worker, "onnx_inference_providers", return_value=None):
            self.assertEqual(ml_worker.hardware_tier(), "weak")
        self.assertEqual(ml_worker.hardware_tier("strong"), "strong")  # a choice in the app still wins


class SetupTests(unittest.TestCase):
    def test_directml_only_on_windows_x64_without_nvidia(self):
        cases = [("nt", "AMD64", None, True), ("nt", "AMD64", "NVIDIA GeForce RTX 4070", False),
                 ("nt", "ARM64", None, False), ("posix", "x86_64", None, False)]
        for os_name, machine, nvidia, expected in cases:
            with mock.patch.object(local_worker.os, "name", os_name), \
                    mock.patch.object(local_worker.platform, "machine", return_value=machine), \
                    mock.patch.object(local_worker, "amd_choice", return_value="yes"):
                self.assertEqual(local_worker.wants_directml(nvidia), expected, (os_name, machine, nvidia))

    def test_directml_only_after_yes(self):
        for choice, expected in (("yes", True), ("no", False), (None, False)):
            with mock.patch.object(local_worker.os, "name", "nt"), \
                    mock.patch.object(local_worker.platform, "machine", return_value="AMD64"), \
                    mock.patch.object(local_worker, "amd_choice", return_value=choice):
                self.assertEqual(local_worker.wants_directml(None), expected, choice)

    def test_directml_replaces_onnxruntime(self):
        commands: list[list[str]] = []
        with mock.patch.object(local_worker, "run_logged", side_effect=commands.append):
            local_worker.install_directml_runtime(Path("python"))
        self.assertEqual(commands[0][-3:], ["uninstall", "-y", "onnxruntime"])
        self.assertEqual(commands[1][-4:], ["install", "--force-reinstall", "--no-deps", "onnxruntime-directml"])

    def test_failed_directml_install_restores_onnxruntime(self):
        commands: list[list[str]] = []

        def run(command):
            commands.append(command)
            if command[-1] == "onnxruntime-directml":
                raise RuntimeError("no matching distribution")
        with mock.patch.object(local_worker, "run_logged", side_effect=run):
            local_worker.install_directml_runtime(Path("python"))
        self.assertEqual(commands[-1][-1], "onnxruntime")


RADEON = {"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vramGB": 16.0}


class AmdChoiceTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(_tempdir()))
        self.enterContext(mock.patch.object(local_worker.vlm, "app_data_dir", return_value=self.folder))
        self.enterContext(mock.patch.dict(local_worker._GPU_CHOICE, clear=True))
        self.enterContext(mock.patch.dict(os.environ, {}, clear=False))
        for key in ("RECO_ONNX_INFERENCE", "RECO_VLM_RUNTIME", "RECO_SIMULATE_AMD"):
            os.environ.pop(key, None)

    def windows_with(self, card, nvidia=None):
        self.enterContext(mock.patch.object(local_worker.os, "name", "nt"))
        self.enterContext(mock.patch.object(local_worker.platform, "machine", return_value="AMD64"))
        self.enterContext(mock.patch.object(local_worker, "detect_nvidia_gpu_name", return_value=nvidia))
        self.enterContext(mock.patch.object(local_worker.teachers, "directml_graphics", return_value=card))

    def test_offer_only_without_nvidia_and_with_an_amd_or_intel_card(self):
        self.windows_with(RADEON)
        self.assertEqual(local_worker.amd_offer(), {"card": RADEON, "choice": None})
        local_worker._GPU_CHOICE.clear()
        with mock.patch.object(local_worker, "detect_nvidia_gpu_name", return_value="NVIDIA GeForce RTX 4070"):
            self.assertIsNone(local_worker.amd_offer())
        local_worker._GPU_CHOICE.clear()
        with mock.patch.object(local_worker.teachers, "directml_graphics", return_value=None):
            self.assertIsNone(local_worker.amd_offer())

    def test_no_offer_outside_windows(self):
        with mock.patch.object(local_worker.os, "name", "posix"):
            self.assertIsNone(local_worker.amd_offer())

    def test_choice_is_remembered_per_computer(self):
        self.windows_with(RADEON)
        local_worker.save_amd_choice("yes")
        local_worker._GPU_CHOICE.clear()
        self.assertEqual(local_worker.amd_choice(), "yes")
        saved = json.loads((self.folder / "gpu-choice.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["amd"], saved["card"]["name"]), ("yes", RADEON["name"]))
        with self.assertRaises(RuntimeError):
            local_worker.save_amd_choice("maybe")

    def test_without_yes_ml_worker_keeps_the_card_alone(self):
        self.windows_with(RADEON)
        self.assertEqual(local_worker.gpu_environment(), {"RECO_ONNX_INFERENCE": "off", "RECO_VLM_RUNTIME": "cpu"})
        local_worker.save_amd_choice("no")
        self.assertEqual(local_worker.gpu_environment(), {"RECO_ONNX_INFERENCE": "off", "RECO_VLM_RUNTIME": "cpu"})
        os.environ["RECO_VLM_RUNTIME"] = "vulkan"  # set by the user: wins
        self.assertEqual(local_worker.gpu_environment(), {"RECO_ONNX_INFERENCE": "off"})
        local_worker.save_amd_choice("yes")
        self.assertEqual(local_worker.gpu_environment(), {})

    def test_tier_uses_the_card_only_after_yes(self):
        self.windows_with(RADEON)
        self.enterContext(mock.patch.object(local_worker, "nvidia_vram_gb", return_value=None))
        self.enterContext(mock.patch.dict(local_worker._TIER_CACHE, clear=True))
        local_worker.save_amd_choice("no")
        self.assertEqual(local_worker.detected_tier(), "weak")
        local_worker.save_amd_choice("yes")
        self.assertEqual(local_worker.detected_tier(), "medium")


class GpuProblemTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(local_worker, "STATE", local_worker.LocalState()))
        self.submit = self.enterContext(mock.patch.object(local_worker.JOBS, "submit"))

    def test_first_fallback_after_yes_queues_one_diagnosis(self):
        local_worker.STATE.operation = "autolabel"
        local_worker.STATE.log = ["loading model", "ONNX Runtime failed"]
        with mock.patch.object(local_worker, "amd_choice", return_value="yes"):
            local_worker.record_gpu_problem(json.dumps({"reason": "DML device removed", "traceback": "Traceback ..."}))
            local_worker.record_gpu_problem(json.dumps({"reason": "again"}))
        self.assertEqual(self.submit.call_count, 1)
        self.assertEqual(self.submit.call_args.args[:3], ("gpu-diagnose", local_worker.gpu_diagnose_action, {"trigger": "problem"}))
        problem = local_worker.STATE.gpu_problem
        self.assertEqual((problem["reason"], problem["operation"]), ("again", "autolabel"))
        self.assertIn("loading model", problem["log"])
        status = local_worker.gpu_status()["problem"]
        self.assertNotIn("log", status)  # the status poll stays small

    def test_no_diagnosis_without_yes_or_during_one(self):
        with mock.patch.object(local_worker, "amd_choice", return_value="no"):
            local_worker.record_gpu_problem("not json")
        self.assertEqual(local_worker.STATE.gpu_problem["reason"], "not json")
        local_worker.STATE.gpu_problem = None
        local_worker.STATE.gpu_diagnosing = True
        with mock.patch.object(local_worker, "amd_choice", return_value="yes"):
            local_worker.record_gpu_problem(json.dumps({"reason": "inside the diagnosis"}))
        self.assertIsNone(local_worker.STATE.gpu_problem)
        self.submit.assert_not_called()

    def test_run_logged_picks_up_the_fallback_line(self):
        line = "RECO_GPU_FALLBACK " + json.dumps({"reason": "DML failed"})
        with mock.patch.object(local_worker, "record_gpu_problem") as record, \
                mock.patch.object(local_worker, "gpu_environment", return_value={}):
            local_worker.run_logged([sys.executable, "-c", f"print({line!r}); print('done')"])
        record.assert_called_once_with(line.removeprefix("RECO_GPU_FALLBACK "))
        self.assertNotIn(line, local_worker.STATE.log)

    def test_a_crashed_job_after_yes_starts_the_diagnosis(self):
        local_worker.STATE.log = ["OSError: [WinError 126] onnxruntime.dll"]
        with mock.patch.object(local_worker, "amd_choice", return_value="yes"):
            local_worker.record_job_failure("gpu-test", "ignored: the test reports itself")
            local_worker.record_job_failure("autolabel", "The local ML worker exited with code 1.")
        self.assertEqual(self.submit.call_count, 1)
        problem = local_worker.STATE.gpu_problem
        self.assertEqual((problem["reason"], problem["operation"]), ("The local ML worker exited with code 1.", "autolabel"))
        self.assertIn("OSError: [WinError 126] onnxruntime.dll", problem["log"])

    def test_a_crashed_job_without_yes_is_not_a_gpu_problem(self):
        with mock.patch.object(local_worker, "amd_choice", return_value=None):
            local_worker.record_job_failure("train", "exited with code 1")
        self.assertIsNone(local_worker.STATE.gpu_problem)
        self.submit.assert_not_called()

    def test_exit_code_error_names_the_last_exception(self):
        script = "import sys; print('loading'); print('OSError: [WinError 126] DirectML.dll'); sys.exit(1)"
        with mock.patch.object(local_worker, "gpu_environment", return_value={}), \
                self.assertRaisesRegex(RuntimeError, r" 1\D*\(OSError: \[WinError 126\] DirectML.dll\)"):
            local_worker.run_logged([sys.executable, "-c", script])

    def test_a_crashing_diagnosis_still_writes_a_report(self):
        folder = Path(self.enterContext(_tempdir()))
        local_worker.STATE.log = ["GPU diagnosis: DirectML: RF-DETR Nano …"]
        with mock.patch.object(local_worker, "diagnostics_dir", return_value=folder), \
                mock.patch.object(local_worker, "run_logged", side_effect=RuntimeError("exited with code 3221225477")), \
                mock.patch.object(local_worker, "find_ml_worker", return_value=Path("ml_worker.py")), \
                mock.patch.object(local_worker, "amd_offer", return_value=None):
            summary = local_worker.run_gpu_diagnosis("test")
        self.assertFalse(summary["ok"])
        report = Path(summary["reportPath"]).read_text(encoding="utf-8")
        self.assertIn("3221225477", report)
        self.assertIn("DirectML: RF-DETR Nano", report)
        self.assertEqual(list(folder.glob("context-*.json")), [])
        self.assertFalse(local_worker.STATE.gpu_diagnosing)
        with mock.patch.object(local_worker, "diagnostics_dir", return_value=folder):
            self.assertIn("3221225477", local_worker.gpu_report_text()["text"])

    def test_report_endpoint_only_reads_the_diagnostics_folder(self):
        folder = Path(self.enterContext(_tempdir()))
        local_worker.STATE.gpu_diagnosis = {"id": "x", "reportPath": str(Path(__file__))}
        with mock.patch.object(local_worker, "diagnostics_dir", return_value=folder), self.assertRaises(RuntimeError):
            local_worker.gpu_report_text()


class FallbackSignalTests(unittest.TestCase):
    def test_failed_directml_preparation_is_signalled(self):
        model = types.SimpleNamespace(model=object(), export=mock.Mock(side_effect=RuntimeError("opset not supported")))
        with mock.patch.object(ml_worker, "onnx_inference_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]), \
                mock.patch.object(ml_worker, "onnx_cache_dir", return_value=Path(self.enterContext(_tempdir()))), \
                mock.patch.dict(sys.modules, {"rfdetr": types.SimpleNamespace(__version__="1.11.2")}), \
                mock.patch.object(ml_worker, "emit"), \
                mock.patch("builtins.print") as printed:
            ml_worker.accelerated_model(model, None, "nano", "en")
        lines = [call.args[0] for call in printed.call_args_list if str(call.args[0]).startswith("RECO_GPU_FALLBACK ")]
        self.assertEqual(len(lines), 1)
        self.assertIn("opset not supported", json.loads(lines[0].split(" ", 1)[1])["reason"])

    def test_cpu_only_onnx_is_not_a_gpu_failure(self):
        model = types.SimpleNamespace(model=object(), export=mock.Mock(side_effect=RuntimeError("x")))
        with mock.patch.object(ml_worker, "onnx_inference_providers", return_value=["CPUExecutionProvider"]), \
                mock.patch.object(ml_worker, "onnx_cache_dir", return_value=Path(self.enterContext(_tempdir()))), \
                mock.patch.dict(sys.modules, {"rfdetr": types.SimpleNamespace(__version__="1.11.2")}), \
                mock.patch.object(ml_worker, "emit"), mock.patch.object(ml_worker, "gpu_fallback") as fallback:
            ml_worker.accelerated_model(model, None, "nano", "en")
        fallback.assert_not_called()

    def test_vlm_restart_on_the_cpu_is_signalled(self):
        server = types.SimpleNamespace(on_cpu=True, gpu_error=RuntimeError("vk::DeviceLostError"))
        with mock.patch.object(ml_worker, "emit"), mock.patch.object(ml_worker, "gpu_fallback") as fallback:
            ml_worker.report_vlm_fallback(server)
            ml_worker.report_vlm_fallback(types.SimpleNamespace(on_cpu=False, gpu_error=None))
        fallback.assert_called_once()
        self.assertIn("DeviceLostError", fallback.call_args.args[0])


class DiagnosisReportTests(unittest.TestCase):
    def setUp(self):
        import gpu_diagnosis
        self.diagnosis = gpu_diagnosis

    def test_two_onnx_runtimes_at_once_fail_the_package_check(self):
        versions = {"onnxruntime": "1.24.4", "onnxruntime-directml": "1.24.4"}

        def version(name):
            if name in versions:
                return versions[name]
            raise importlib.metadata.PackageNotFoundError(name)
        step = self.diagnosis.Step("packages", "Python packages")
        with mock.patch.object(self.diagnosis.importlib.metadata, "version", side_effect=version):
            self.diagnosis.step_packages(step)
        self.assertEqual(step.status, "failed")
        self.assertIn("onnxruntime, onnxruntime-directml", step.summary)

    def test_a_crashing_step_is_recorded_and_the_rest_still_runs(self):
        folder = Path(self.enterContext(_tempdir()))
        worker = types.SimpleNamespace(emit=lambda _message: None)

        def broken(_step):
            raise OSError("DirectML.dll could not be loaded")
        with mock.patch.object(self.diagnosis, "step_system", side_effect=lambda step, _version: setattr(step, "summary", "Windows 11")), \
                mock.patch.object(self.diagnosis, "step_graphics", side_effect=lambda step: setattr(step, "summary", RADEON["name"])), \
                mock.patch.object(self.diagnosis, "step_dxdiag"), mock.patch.object(self.diagnosis, "step_packages"), \
                mock.patch.object(self.diagnosis, "step_torch"), \
                mock.patch.object(self.diagnosis, "step_onnxruntime", side_effect=broken), \
                mock.patch.object(self.diagnosis, "step_vulkan") as vulkan:
            summary = self.diagnosis.run_diagnosis(worker, folder, None, {"trigger": "test", "log": ["setup line"]}, "0.29.0")
        vulkan.assert_called_once()
        self.assertFalse(summary["ok"])
        failed = [step for step in summary["steps"] if step["status"] == "failed"]
        self.assertEqual([step["key"] for step in failed], ["onnxruntime"])
        report = Path(summary["reportPath"]).read_text(encoding="utf-8")
        self.assertIn("[FAIL] ONNX Runtime providers", report)
        self.assertIn("OSError: DirectML.dll could not be loaded", report)
        self.assertIn("Traceback", report)
        self.assertIn("setup line", report)
        self.assertTrue(Path(summary["reportPath"]).with_suffix(".json").is_file())


class _tempdir:
    def __enter__(self) -> str:
        import tempfile
        self.folder = tempfile.TemporaryDirectory()
        return self.folder.__enter__()

    def __exit__(self, *exc) -> None:
        self.folder.__exit__(*exc)


if __name__ == "__main__":
    unittest.main()
