import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import local_worker


class AugmentReadinessTests(unittest.TestCase):
    def test_missing_ml_environment_is_reported_before_the_folder_dialog(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "project.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(local_worker, "ml_venv", return_value=root / "missing-venv"):
                self.assertEqual(local_worker.augment_state(root).get("needs"), "ml")
                with mock.patch.object(local_worker, "choose_video_folder") as dialog:
                    with self.assertRaises(RuntimeError):
                        local_worker.augment_select_source(root, "en")
                    dialog.assert_not_called()

    def test_ready_project_has_no_hint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "project.json").write_text("{}", encoding="utf-8")
            python = local_worker.venv_python_path(root / "venv")
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            with mock.patch.object(local_worker, "ml_venv", return_value=root / "venv"):
                self.assertNotIn("needs", local_worker.augment_state(root))

    def test_studio_disables_the_button_and_explains_why(self):
        source = (ROOT / "app" / "AugmentationStudio.tsx").read_text(encoding="utf-8")
        self.assertIn("disabled={busy || Boolean(state.needs)}", source)
        worker = (ROOT / "local_worker.py").read_text(encoding="utf-8")
        self.assertIn('{"source": None, "needs": "project"}', worker)


if __name__ == "__main__":
    unittest.main()
