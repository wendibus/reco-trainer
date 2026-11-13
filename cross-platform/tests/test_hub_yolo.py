"""YOLO26 in the model hub: weights only as safetensors, rebuilt locally, AGPL-3.0 visible."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import model_hub  # noqa: E402

try:
    import torch  # noqa: F401
    from safetensors.torch import load_file  # noqa: F401
    from ultralytics.nn.tasks import DetectionModel  # noqa: F401
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def yolo_header(width=16, classes=2, tensors=700) -> dict:
    header = {}
    names = [model_hub.YOLO_FIRST_KEY, *model_hub.YOLO_CLASS_KEYS] + [f"model.{1 + index % 22}.part{index}.weight" for index in range(tensors - 4)]
    for name in names:
        shape = [width, 3, 3, 3] if name == model_hub.YOLO_FIRST_KEY else [classes, 4 * width, 1, 1] if name in model_hub.YOLO_CLASS_KEYS else [1]
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [0, 2]}
    return header


class HeaderTests(unittest.TestCase):
    def test_size_and_classes_come_from_the_header(self):
        self.assertEqual(model_hub.inspect_yolo_header(yolo_header(16, 2)), {"size": "yolo-nano", "classCount": 2, "architecture": "yolo26"})
        self.assertEqual(model_hub.inspect_yolo_header(yolo_header(32, 5))["size"], "yolo-small")
        self.assertEqual(model_hub.inspect_yolo_header(yolo_header(64, 1))["size"], "yolo-medium")

    def test_anything_else_is_refused(self):
        for header in (yolo_header(24), yolo_header(tensors=20), {"x": {"shape": [1]}}, {**yolo_header(), "/Users/me/x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}):
            with self.assertRaises(model_hub.HubError):
                model_hub.inspect_yolo_header(header)
        broken = yolo_header()
        broken[model_hub.YOLO_CLASS_KEYS[1]]["shape"][0] = 3  # the three class heads must agree
        with self.assertRaises(model_hub.HubError):
            model_hub.inspect_yolo_header(broken)

    def test_inspect_weights_tells_the_families_apart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "w.safetensors"
            raw = json.dumps(yolo_header()).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 2)
            self.assertEqual(model_hub.inspect_weights(path)["size"], "yolo-nano")


class ModelJsonTests(unittest.TestCase):
    def build(self, **changes):
        options = dict(name="Hall model", description="Indoor", version="1.0.0", sport="basketball", size="yolo-small", classes=["ball", "player"], parent=None, author="someone",
                       training={"images": 10, "games": 1, "camera": "fixed", "venue": "indoor"}, self_reported={}, trainer_version="0.36.0", sha256="a" * 64, size_bytes=1, source_url="",
                       architecture="yolo26", image_size=960, framework_version="8.4.180")
        return model_hub.build_model_json(**{**options, **changes})

    def test_a_yolo_entry_names_its_license_framework_and_a_trainer_that_knows_it(self):
        entry = self.build()
        self.assertEqual((entry["architecture"], entry["license"], entry["input_resolution"], entry["size"]), ("yolo26", "AGPL-3.0", 960, "yolo-small"))
        self.assertEqual(entry["framework"], {"name": "ultralytics", "version": "8.4.180"})
        self.assertEqual(entry["rights"]["statement_version"], 2)
        self.assertEqual(entry["compat"]["trainer_min"], "0.37.0")  # an older trainer is not shown this model
        self.assertEqual(self.build(trainer_version="0.40.1")["compat"]["trainer_min"], "0.40.1")

    def test_rf_detr_entries_stay_as_they_were(self):
        entry = self.build(size="small", architecture="rf-detr", image_size=None, framework_version=None)
        self.assertEqual((entry["license"], entry["input_resolution"], entry["rights"]["statement_version"], "framework" in entry), ("Apache-2.0", 512, 1, False))

    def test_bad_yolo_details_are_refused(self):
        for changes in ({"size": "yolo-large"}, {"image_size": 512}, {"framework_version": None}):
            with self.assertRaises(model_hub.HubError):
                self.build(**changes)


class IndexTests(unittest.TestCase):
    def test_yolo_models_are_listed_for_a_trainer_that_knows_them_only(self):
        index = {"models": [
            {"sport": "basketball", "size": "yolo-nano", "id": "a", "classes": ["ball"], "compat": {"trainer_min": "0.37.0"}},
            {"sport": "basketball", "size": "small", "id": "b", "classes": ["ball"], "compat": {"trainer_min": "0.25.0"}},
        ]}
        self.assertEqual([m["id"] for m in model_hub.filter_models(index, "basketball", ["ball"], "0.37.0")], ["b", "a"])
        self.assertEqual([m["id"] for m in model_hub.filter_models(index, "basketball", ["ball"], "0.36.0")], ["b"])

    def test_the_library_entry_can_be_the_parent_of_the_next_yolo_training(self):
        entry = model_hub.yolo_library_manifest({"sport": "basketball", "size": "yolo-nano", "id": "a", "version": "1.0.0", "name": "A", "classes": ["player", "ball"], "license": "AGPL-3.0",
                                                 "framework": {"version": "8.4.180"}, "author": "x", "training": {"images": 3}}, "weights/m.onnx", "f" * 64, "weights/best.pt", 640)
        self.assertEqual((entry["yoloVariant"], entry["trainWeights"], entry["format"], entry["source"], entry["classNames"]), ("yolo-nano", "weights/best.pt", "onnx-yolo", "hub", ["player", "ball"]))
        self.assertEqual(entry["license"], "AGPL-3.0")
        self.assertEqual(entry["hub"]["lineage"], "basketball/yolo-nano/a@1.0.0")


@unittest.skipUnless(HAS_YOLO, "ultralytics, torch and safetensors are not installed")
class RoundTripTests(unittest.TestCase):
    def test_share_then_rebuild_gives_the_same_weights_and_hides_local_paths(self):
        import os
        os.environ.setdefault("YOLO_OFFLINE", "1")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            net = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
            net.names = {0: "ball", 1: "player", 2: "referee"}
            checkpoint = folder / "best.pt"
            torch.save({"model": net.half(), "train_args": {"data": "/Users/me/secret/data.yaml"}}, checkpoint)
            weights = folder / "w.safetensors"
            classes = model_hub.sanitize_yolo_checkpoint(checkpoint, weights)
            self.assertEqual(classes, ["ball", "player", "referee"])
            self.assertNotIn(b"/Users/me", weights.read_bytes())
            self.assertEqual(model_hub.inspect_weights(weights), {"size": "yolo-nano", "classCount": 3, "architecture": "yolo26"})
            rebuilt = folder / "rebuilt.pt"
            model_hub.rebuild_yolo_checkpoint(weights, classes, "yolo-nano", rebuilt)
            first = load_file(str(weights))
            again = torch.load(rebuilt, map_location="cpu", weights_only=False)["model"].state_dict()
            self.assertEqual(set(first), set(again))
            self.assertTrue(all(torch.equal(first[name].float(), again[name].float()) for name in first if first[name].is_floating_point()))
            with self.assertRaises(model_hub.HubError):
                model_hub.rebuild_yolo_checkpoint(weights, ["a", "b"], "yolo-nano", folder / "x.pt")  # wrong number of classes
            with self.assertRaises(model_hub.HubError):
                model_hub.rebuild_yolo_checkpoint(weights, classes, "yolo-small", folder / "x.pt")  # wrong size


class ContractTests(unittest.TestCase):
    def test_the_wiring_keeps_the_security_rules(self):
        ml = (ROOT / "ml_worker.py").read_text(encoding="utf-8")
        self.assertIn("model_hub.sanitize_yolo_checkpoint", ml)
        self.assertIn("model_hub.rebuild_yolo_checkpoint", ml)
        self.assertIn('manifest.get("source") == "local-training"', ml)  # only models trained here are shared
        self.assertIn("(is_onnx_manifest(manifest) and not yolo)", ml)  # imported ONNX models still are not
        worker = (ROOT / "local_worker.py").read_text(encoding="utf-8")
        self.assertIn('payload.get("yoloLicenseAccepted") is not True', worker)
        hub = (ROOT / "app" / "ModelHub.tsx").read_text(encoding="utf-8")
        self.assertIn("agplAgree", hub)
        self.assertIn("agplRights", hub)


if __name__ == "__main__":
    unittest.main()
