"""AMD and Intel graphics: RF-DETR through ONNX Runtime (DirectML) and the ML setup that installs it.

The Vulkan build of the VLM is covered in test_vlm.py.
"""
from __future__ import annotations

import importlib.util
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


class SetupTests(unittest.TestCase):
    def test_directml_only_on_windows_x64_without_nvidia(self):
        cases = [("nt", "AMD64", None, True), ("nt", "AMD64", "NVIDIA GeForce RTX 4070", False),
                 ("nt", "ARM64", None, False), ("posix", "x86_64", None, False)]
        for os_name, machine, nvidia, expected in cases:
            with mock.patch.object(local_worker.os, "name", os_name), \
                    mock.patch.object(local_worker.platform, "machine", return_value=machine):
                self.assertEqual(local_worker.wants_directml(nvidia), expected, (os_name, machine, nvidia))

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


class _tempdir:
    def __enter__(self) -> str:
        import tempfile
        self.folder = tempfile.TemporaryDirectory()
        return self.folder.__enter__()

    def __exit__(self, *exc) -> None:
        self.folder.__exit__(*exc)


if __name__ == "__main__":
    unittest.main()
