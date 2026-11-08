import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANGUAGES = ("de", "en", "es", "fr")
STEPS = 7


class AugmentGuideTests(unittest.TestCase):
    """The studio guide has seven steps in every language, each with a screenshot in that language."""

    def test_every_language_has_seven_steps(self):
        source = (ROOT / "app" / "AugmentGuide.tsx").read_text(encoding="utf-8")
        for language in LANGUAGES:
            block = source.split(f"\n  {language}: {{", 1)[1].split("\n  },", 1)[0]
            self.assertEqual(len(re.findall(r"^      \['", block, re.M)), STEPS, language)

    def test_screenshots_exist_for_every_step_and_language(self):
        folder = ROOT / "public" / "augment-guide"
        expected = {f"{language}-{step}.jpg" for language in LANGUAGES for step in range(1, STEPS + 1)}
        self.assertEqual({path.name for path in folder.iterdir()}, expected)
        for name in expected:
            data = (folder / name).read_bytes()
            self.assertEqual(data[:3], b"\xff\xd8\xff", name)
            self.assertLess(len(data), 250_000, name)

    def test_screenshots_are_allowed_by_the_image_guard(self):
        ignore = (ROOT.parent / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("!cross-platform/public/**", ignore)


if __name__ == "__main__":
    unittest.main()
