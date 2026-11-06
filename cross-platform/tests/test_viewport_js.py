"""Runs the editor's view-math tests (tests/viewport.test.mjs) when Node 23.6+ is installed."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def node_version() -> tuple[int, int] | None:
    node = shutil.which("node")
    if not node:
        return None
    output = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    try:
        major, minor = (int(part) for part in output.split(".")[:2])
    except ValueError:
        return None
    return major, minor


class ViewportMathTests(unittest.TestCase):
    @unittest.skipUnless((node_version() or (0, 0)) >= (23, 6), "needs Node 23.6+ (runs TypeScript directly)")
    def test_view_math(self):
        result = subprocess.run(["node", "--test", "tests/viewport.test.mjs"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
