import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import amd_training_probe as probe


class AmdTrainingProbeTests(unittest.TestCase):
    def test_wheels_come_from_amd_and_match_the_constraints(self):
        for url in probe.SDK_WHEELS + probe.TORCH_WHEELS:
            self.assertTrue(url.startswith("https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"))
        for url in probe.TORCH_WHEELS:
            self.assertIn("cp312", url)
            name, version = url.rsplit("/", 1)[1].split("-")[:2]
            self.assertIn(f"{name}=={version.replace('%2B', '+')}", probe.CONSTRAINTS)
        self.assertIn("==", probe.RFDETR)

    def test_it_stays_out_of_the_app_environment(self):
        self.assertEqual(probe.FOLDER.name, "amd-training-probe")

    def test_the_launcher_runs_the_script_once(self):
        bat = (ROOT / "tools" / "Test AMD Training.bat").read_bytes()
        self.assertIn(b"\r\n", bat)
        self.assertNotIn(b"||", bat)


if __name__ == "__main__":
    unittest.main()
