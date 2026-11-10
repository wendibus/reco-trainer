"""Export for the Reco camera app: the YOLO end-to-end layout Reco reads (see reco_export.py)."""
import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
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


reco_export = load_module("reco_export_under_test", ROOT / "reco_export.py")

try:
    import numpy as np
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper, numpy_helper
    from PIL import Image
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class NamesAndGeometryTests(unittest.TestCase):
    def test_players_become_person_and_names_use_the_dict_form_reco_parses(self):
        self.assertEqual(reco_export.names_metadata(["ball"]), "{0: 'ball'}")
        self.assertEqual(reco_export.names_metadata(["ball", "player", "referee"]), "{0: 'ball', 1: 'person', 2: 'referee'}")

    def test_letterbox_matches_reco(self):
        # Reco: scale = min(S/w, S/h), new size rounded, padding split evenly.
        self.assertEqual(reco_export.letterbox_geometry(1920, 1080, 960), (0.5, 0.0, 210.0, 960, 540))
        self.assertEqual(reco_export.letterbox_geometry(1080, 1920, 960), (0.5, 210.0, 0.0, 540, 960))

    def test_content_rows_cover_the_picture_only(self):
        self.assertEqual(reco_export.content_rows(16 / 9, 960), (210, 540))
        self.assertEqual(reco_export.content_rows(4 / 3, 960), (120, 720))
        self.assertEqual(reco_export.content_rows(1.0, 960), (0, 960))

    def test_input_size_leaves_room_above_the_model_resolution(self):
        self.assertEqual(reco_export.input_size_for([384]), 960)
        self.assertEqual(reco_export.input_size_for([384, 704]), 1280)

    def test_project_aspect_is_the_median_of_the_images(self):
        sizes = {"a.jpg": (1920, 1080), "b.jpg": (1280, 720), "c.jpg": (640, 480)}
        frames = [{"relativePath": name} for name in sizes] + [{"note": "no path"}]
        aspect = reco_export.project_aspect(Path("/x"), frames, lambda path: sizes[path.name])
        self.assertAlmostEqual(aspect, 16 / 9)
        self.assertAlmostEqual(reco_export.project_aspect(Path("/x"), [], lambda path: None), 16 / 9)


@unittest.skipUnless(HAS_ONNX, "onnx / onnxruntime / Pillow are not installed")
class YoloFilterTests(unittest.TestCase):
    def yolo_file(self, folder: Path, rows) -> Path:
        """A stand-in YOLO end-to-end export: the output is a constant [1, N, 6] plus 0 * input."""
        data = np.array([rows], dtype=np.float32)
        graph = helper.make_graph(
            [helper.make_node("ReduceSum", ["images"], ["total"], keepdims=0),
             helper.make_node("Mul", ["total", "zero"], ["nothing"]),
             helper.make_node("Add", ["table", "nothing"], ["output0"])],
            "yolo",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, len(rows), 6])],
            [numpy_helper.from_array(data, "table"), numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = folder / "best.onnx"
        onnx.save(model, str(path))
        return path

    def test_keeps_only_the_chosen_classes_with_new_ids(self):
        rows = [[1, 2, 3, 4, 0.9, 0], [5, 6, 7, 8, 0.8, 1], [9, 9, 9, 9, 0.7, 2], [1, 1, 2, 2, 0.6, 1]]
        with tempfile.TemporaryDirectory() as temp:
            source = self.yolo_file(Path(temp), rows)
            target = Path(temp) / "reco.onnx"
            size = reco_export.filter_yolo_onnx(source, target, ["player", "ball", "referee"], ["ball", "player"])
            reco_export.write_metadata(target, ["ball", "player"], size, {"author": "test"})
            session = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            output = session.run(None, {"images": np.zeros((1, 3, 64, 64), dtype=np.float32)})[0][0]
            self.assertEqual(size, 64)
            self.assertEqual(len(session.get_outputs()), 1)
            self.assertEqual(output[:, :4].tolist(), [row[:4] for row in rows])
            # player (old 0) -> 1, ball (old 1) -> 0, referee dropped: confidence 0.
            self.assertEqual(output[:, 5].tolist(), [1.0, 0.0, 0.0, 0.0])
            self.assertEqual([round(float(value), 2) for value in output[:, 4]], [0.9, 0.8, 0.0, 0.6])
            names = session.get_modelmeta().custom_metadata_map
            self.assertEqual(names["names"], "{0: 'ball', 1: 'person'}")
            self.assertEqual(names["imgsz"], "[64, 64]")

    def test_refuses_a_file_without_the_reco_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            graph = helper.make_graph([helper.make_node("Identity", ["images"], ["boxes"])], "other",
                                      [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 300, 4])],
                                      [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 300, 4])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            source = Path(temp) / "other.onnx"
            onnx.save(model, str(source))
            with self.assertRaises(ValueError):
                reco_export.filter_yolo_onnx(source, Path(temp) / "out.onnx", ["ball"], ["ball"])

    def test_check_counts_images_where_a_labelled_object_is_found(self):
        # A ball at x 100-120, y 300-320 of a 640 x 360 picture: in a 64 px letterbox that is
        # scale 0.1, pad_y 14 -> box 10-12, 44-46.
        rows = [[10, 44, 12, 46, 0.9, 0]] + [[0, 0, 0, 0, 0.0, 0]] * 3
        with tempfile.TemporaryDirectory() as temp:
            source = self.yolo_file(Path(temp), rows)
            target = Path(temp) / "reco.onnx"
            reco_export.filter_yolo_onnx(source, target, ["ball"], ["ball"])
            reco_export.write_metadata(target, ["ball"], 64, {})
            picture = Image.new("RGB", (640, 360))
            result = reco_export.check_file(target, [(picture, [(100, 300, 20, 20)]), (picture, [(400, 50, 20, 20)])])
            self.assertEqual((result["found"], result["samples"], result["inputSize"]), (1, 2, 64))


class RecoExportCommandTests(unittest.TestCase):
    def setUp(self):
        self.ml_worker = load_module("reco_export_test_ml_worker", ROOT / "ml_worker.py")

    def project(self, temp: str, manifest: dict) -> Path:
        root = Path(temp)
        package = root / "models" / "library" / manifest["packageID"]
        (package / "weights").mkdir(parents=True)
        (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "project.json").write_text(json.dumps({"sport": "football", "frames": []}), encoding="utf-8")
        return root

    def run_command(self, root: Path, package: str, classes: str) -> str:
        args = argparse.Namespace(project=str(root), package=package, classes=classes, language="en")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(self.ml_worker, "require_project", return_value=json.loads((root / "project.json").read_text())):
            self.ml_worker.export_reco(args)
        return output.getvalue()

    def test_unknown_package_and_unknown_classes_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.project(temp, {"packageID": "m1", "displayName": "Ball model", "modelSize": "nano", "classes": ["ball"]})
            with self.assertRaises(SystemExit) as missing:
                self.run_command(root, "../elsewhere", "ball")
            self.assertIn("not found", str(missing.exception))
            with self.assertRaises(SystemExit) as wrong:
                self.run_command(root, "m1", "player")
            self.assertIn("none of the chosen classes", str(wrong.exception))

    def test_ball_is_the_default_class(self):
        parser_help = (ROOT / "ml_worker.py").read_text(encoding="utf-8")
        self.assertIn('reco_parser.add_argument("--classes", default="ball")', parser_help)
        dialog = (ROOT / "app" / "RecoExportDialog.tsx").read_text(encoding="utf-8")
        self.assertIn("classes.includes('ball') ? ['ball']", dialog)

    def test_worker_route_and_output_folder(self):
        source = (ROOT / "local_worker.py").read_text(encoding="utf-8")
        self.assertIn('"/api/export-reco"', source)
        self.assertIn('"export-reco": [root / "exports" / "reco"]', source)


if __name__ == "__main__":
    unittest.main()
