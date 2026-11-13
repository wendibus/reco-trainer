"""Points promised to the second tester: pre-label only trained classes in the independent test,
offer the field boundary before pre-labelling, make "Extend dataset" easier to find."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_worker():
    spec = importlib.util.spec_from_file_location("tester_points_local_worker", ROOT / "local_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainedClassesTests(unittest.TestCase):
    def library(self, root: Path, classes: list[str], active: bool = True):
        folder = root / "models" / "library" / "m1"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"packageID": "m1", "classes": classes, "sport": "basketball"}), encoding="utf-8")
        if active:
            (root / "models" / "active.json").write_text(json.dumps({"packageID": "m1"}), encoding="utf-8")

    def test_the_classes_of_the_active_model_count(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.library(root, ["ball", "hoop"])
            self.assertEqual(worker.trained_classes(root, {}), {"ball", "hoop"})

    def test_without_an_active_model_the_last_training_decides_and_then_nothing_is_known(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(worker.trained_classes(root, {"lastTraining": {"classes": ["ball"]}}), {"ball"})
            self.assertEqual(worker.trained_classes(root, {}), set())
            self.assertEqual(worker.trained_classes(None, {}), set())

    def test_the_independent_test_pre_labels_only_those_and_falls_back_to_all(self):
        source = (ROOT / "local_worker.py").read_text(encoding="utf-8")
        self.assertIn('[name for name in sport_classes if name in known] or sport_classes', source)
        self.assertNotIn('autolabel_payload["categories"] = list(SPORT_CATEGORIES.get(project.get("sport"), []))', source)


class InterfaceContractTests(unittest.TestCase):
    def test_pre_labelling_asks_about_the_field_first(self):
        page = (ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        for start in ("startActiveLearning()", "startIndependentValidation()", "'/api/autolabel'"):
            self.assertIn(start, page)
        self.assertEqual(page.count("withFieldOffer(() =>"), 3)  # auto-label, extend dataset, independent test
        self.assertIn("noFieldYet()", page)
        dialog = (ROOT / "app" / "FieldOfferDialog.tsx").read_text(encoding="utf-8")
        for key in ("de:", "en:", "es:", "fr:"):
            self.assertIn(key, dialog)

    def test_extend_dataset_stands_out_once_a_model_exists(self):
        page = (ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        self.assertIn("trainedModel && !candidates.length ? 'primary-button'", page)
        self.assertEqual(page.count("nextStep:"), 4)


if __name__ == "__main__":
    unittest.main()
