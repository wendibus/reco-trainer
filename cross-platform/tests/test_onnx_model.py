import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml_worker = load_module("onnx_test_ml_worker", ROOT / "ml_worker.py")

try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401
    from onnx import TensorProto, helper
    from PIL import Image
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def write_constant_detector(path: Path, size: int, rows: list[list[float]]) -> None:
    """A tiny ONNX 'detector' whose NMS-style output (1 x N x 6) is constant."""
    value = helper.make_tensor("rows", TensorProto.FLOAT, [1, len(rows), 6], [v for row in rows for v in row])
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["detections"], value=value)],
        "constant-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, len(rows), 6])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def make_package(onnx_path: Path, output: Path, **overrides) -> None:
    command = [
        sys.executable, str(ROOT / "tools" / "make_onnx_package.py"), str(onnx_path),
        "--name", overrides.get("name", "soccer"), "--sport", overrides.get("sport", "football"),
        "--class-names", overrides.get("class_names", "player,ball,referee"),
        "--input-size", str(overrides.get("input_size", 64)), "-o", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)


@unittest.skipUnless(HAS_ONNX, "onnx / onnxruntime / Pillow are not installed")
class OnnxYoloModelTests(unittest.TestCase):
    def test_predict_maps_letterboxed_boxes_back_to_the_original_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "detector.onnx"
            # 64 px input; a 200x100 image is scaled by 0.32 to 64x32 and padded 16 px top/bottom.
            write_constant_detector(weights, 64, [
                [0, 16, 32, 32, 0.9, 0],    # -> (0, 0, 100, 50), class 0
                [10, 20, 20, 30, 0.1, 1],   # below threshold
                [0, 16, 64, 48, 0.8, 2],    # -> (0, 0, 200, 100), class 2
            ])
            image = root / "frame.png"
            Image.new("RGB", (200, 100), (30, 120, 30)).save(image)

            model = ml_worker.OnnxYoloModel(weights, ["player", "ball", "referee"], 64)
            [detections] = model.predict([str(image)], threshold=0.25)

            self.assertEqual(len(detections.xyxy), 2)
            np.testing.assert_allclose(detections.xyxy[0], [0, 0, 100, 50], atol=1.0)
            np.testing.assert_allclose(detections.xyxy[1], [0, 0, 200, 100], atol=1.0)
            self.assertEqual(list(detections.data["class_name"]), ["player", "referee"])
            self.assertTrue(all(0 <= value <= 200 for box in detections.xyxy for value in box[::2]))

    def test_model_without_built_in_nms_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "raw.onnx"
            value = helper.make_tensor("raw", TensorProto.FLOAT, [1, 8, 10], [0.0] * 80)
            graph = helper.make_graph(
                [helper.make_node("Constant", [], ["out"], value=value)], "raw",
                [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 10])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            model.ir_version = 8
            onnx.save(model, str(weights))
            with self.assertRaises(SystemExit):
                ml_worker.OnnxYoloModel(weights, ["player"], 64, "en")


@unittest.skipUnless(HAS_ONNX, "onnx / onnxruntime / Pillow are not installed")
class OnnxPackageTests(unittest.TestCase):
    def _project(self, root: Path, sport: str = "football") -> None:
        (root / "frames").mkdir(parents=True)
        Image.new("RGB", (200, 100), (30, 120, 30)).save(root / "frames" / "one.png")
        ml_worker.atomic_json(root / "project.json", {
            "sport": sport,
            "frames": [{"id": "one", "relativePath": "frames/one.png", "width": 200, "height": 100, "annotations": []}],
        })

    def _install(self, root: Path, package: Path) -> dict:
        args = argparse.Namespace(project=str(root), file=str(package), language="en")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ml_worker.install_model_package(args)
        return json.loads(output.getvalue().strip().splitlines()[-1])

    def test_install_activates_an_onnx_package_for_any_model_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            weights = Path(temporary) / "detector.onnx"
            write_constant_detector(weights, 64, [[0, 16, 32, 32, 0.9, 0]])
            package = Path(temporary) / "soccer.recomodel"
            make_package(weights, package)
            self._project(root)

            result = self._install(root, package)

            self.assertEqual(result["active"]["packageID"], "soccer")
            self.assertEqual(result["active"]["format"], "onnx-yolo")
            self.assertTrue((root / "models" / "library" / "soccer" / "weights" / "soccer.onnx").is_file())
            for size in ("nano", "small"):
                self.assertEqual(ml_worker.active_model_classes(root, size), {"ball", "player", "referee"})
                self.assertTrue(ml_worker.is_onnx_manifest(ml_worker.active_model_manifest(root, size)))
                # An ONNX file is never mistaken for a trainable RF-DETR checkpoint.
                self.assertIsNone(ml_worker.newest_checkpoint(root, size))

    def test_auto_label_uses_the_active_onnx_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            weights = Path(temporary) / "detector.onnx"
            write_constant_detector(weights, 64, [
                [0, 16, 32, 32, 0.9, 0],   # player
                [32, 16, 48, 24, 0.7, 1],  # ball
                [0, 16, 64, 48, 0.8, 2],   # referee
            ])
            package = Path(temporary) / "soccer.recomodel"
            make_package(weights, package)
            self._project(root)
            self._install(root, package)

            original_device = ml_worker.detect_device
            original_emit = ml_worker.emit
            try:
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.emit = lambda message: None
                args = argparse.Namespace(
                    project=str(root), model="nano", category=["player", "ball", "referee"],
                    threshold=0.25, language="en", candidate_only=False,
                )
                ml_worker.auto_label(args)
            finally:
                ml_worker.detect_device = original_device
                ml_worker.emit = original_emit

            annotations = ml_worker.load_json(root / "project.json")["frames"][0]["annotations"]
            self.assertEqual(sorted(item["category"] for item in annotations), ["ball", "player", "referee"])
            self.assertTrue(all(item["source"] == "auto" for item in annotations))

    def test_package_with_onnx_weights_but_without_the_format_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "detector.onnx"
            write_constant_detector(weights, 64, [[0, 16, 32, 32, 0.9, 0]])
            package = Path(temporary) / "soccer.recomodel"
            make_package(weights, package)
            tampered = Path(temporary) / "tampered.recomodel"
            with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "manifest.json":
                        manifest = json.loads(data)
                        del manifest["format"]
                        data = json.dumps(manifest).encode()
                    target.writestr(name, data)
            with self.assertRaises(SystemExit):
                ml_worker.validate_model_package_file(tampered, "en")

    def test_package_with_invalid_input_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "detector.onnx"
            write_constant_detector(weights, 64, [[0, 16, 32, 32, 0.9, 0]])
            package = Path(temporary) / "soccer.recomodel"
            make_package(weights, package)
            tampered = Path(temporary) / "tampered.recomodel"
            with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "manifest.json":
                        manifest = json.loads(data)
                        manifest["inputSize"] = 100000
                        data = json.dumps(manifest).encode()
                    target.writestr(name, data)
            with self.assertRaises(SystemExit):
                ml_worker.validate_model_package_file(tampered, "en")

    def test_validator_tool_accepts_the_generated_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "detector.onnx"
            write_constant_detector(weights, 64, [[0, 16, 32, 32, 0.9, 0]])
            package = Path(temporary) / "soccer.recomodel"
            make_package(weights, package)
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "validate_model_package.py"), str(package)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("VALID"))
            self.assertEqual(json.loads(result.stdout.split("\n", 1)[1])["format"], "onnx-yolo")

    def test_is_onnx_manifest_only_matches_the_onnx_format(self):
        self.assertTrue(ml_worker.is_onnx_manifest({"format": "onnx-yolo"}))
        self.assertFalse(ml_worker.is_onnx_manifest({"format": None}))
        self.assertFalse(ml_worker.is_onnx_manifest(None))


if __name__ == "__main__":
    unittest.main()


class InstallOnnxTests(unittest.TestCase):
    """install-onnx: wrap a plain .onnx file and install it from the app."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        ml_worker.atomic_json(self.project / "project.json", {"sport": "football", "frames": []})
        self.weights = self.root / "best.onnx"
        self.weights.write_bytes(b"not a real graph - install does not load it")

    def tearDown(self):
        self.temporary.cleanup()

    def install(self, **overrides) -> dict:
        values = dict(project=str(self.project), file=str(self.weights), name="Soccer YOLO",
                      class_names="person, sports ball, referee", input_size=1280, license="", language="en")
        values.update(overrides)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ml_worker.install_onnx_model(argparse.Namespace(**values))
        return json.loads(output.getvalue().strip().splitlines()[-1])

    def test_builds_a_package_in_the_inbox_and_activates_it(self):
        result = self.install()
        self.assertTrue(result["installed"])
        manifest = result["manifest"]
        self.assertEqual(manifest["packageID"], "Soccer-YOLO")
        self.assertEqual(manifest["displayName"], "Soccer YOLO")
        self.assertEqual(manifest["classNames"], ["player", "ball", "referee"])
        self.assertEqual(manifest["format"], ml_worker.ONNX_FORMAT)
        self.assertEqual(manifest["inputSize"], 1280)
        package = self.project / "inbox" / "Soccer-YOLO.recomodel"
        self.assertTrue(package.is_file())
        with zipfile.ZipFile(package) as archive:
            self.assertEqual(sorted(archive.namelist()), ["manifest.json", "weights/Soccer-YOLO.onnx"])
        self.assertTrue((self.project / "models" / "library" / "Soccer-YOLO" / "weights" / "Soccer-YOLO.onnx").is_file())

    def test_rejects_classes_the_sport_does_not_have(self):
        with self.assertRaises(SystemExit) as raised:
            self.install(class_names="player,hoop")
        self.assertIn("not recognised: hoop", str(raised.exception))
        self.assertFalse((self.project / "inbox").exists())

    def test_rejects_a_file_that_is_not_onnx(self):
        other = self.root / "best.pt"
        other.write_bytes(b"x")
        with self.assertRaises(SystemExit):
            self.install(file=str(other))
