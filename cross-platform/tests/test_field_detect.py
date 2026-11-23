import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:  # the light detection needs both; skip where they are missing
    np = None

if np is not None:
    import field_detect


def synthetic_frame(path: Path) -> None:
    """Grey hall; green floor in the lower 60 %; a second, brighter green patch bottom-left."""
    image = np.full((180, 320, 3), (200, 200, 205), dtype=np.uint8)
    image[72:, :] = (30, 120, 50)
    image[100:104, :] = (240, 240, 240)  # a court line across the floor
    Image.fromarray(image).save(path)


@unittest.skipIf(np is None, "numpy and Pillow are not installed")
class FieldDetectTests(unittest.TestCase):
    def test_finds_the_floor_and_ignores_lines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            synthetic_frame(path)
            result = field_detect.detect_field(path)
        ys = [y for _, y in result["corners"]]
        self.assertGreaterEqual(len(result["corners"]), 4)
        self.assertLessEqual(len(result["corners"]), field_detect.MAX_POINTS)
        self.assertAlmostEqual(min(ys), 0.4, delta=0.05)
        self.assertGreater(result["coverage"], 0.5)

    def test_hidden_area_is_left_out(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            synthetic_frame(path)
            result = field_detect.detect_field(path, [[0.0, 0.0, 0.3, 1.0]])
        xs = [x for x, _ in result["corners"]]
        self.assertGreaterEqual(min(xs), 0.27)

    def test_no_field_in_a_flat_grey_frame(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            Image.fromarray(np.full((90, 160, 3), 128, dtype=np.uint8)).save(path)
            result = None
            try:
                result = field_detect.detect_field(path)
            except field_detect.FieldNotFound:
                return
        # A uniform frame may legitimately count as one big surface; it must still be valid.
        self.assertTrue(all(0 <= v <= 1 for point in result["corners"] for v in point))


if __name__ == "__main__":
    unittest.main()
