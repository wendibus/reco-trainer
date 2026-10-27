import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml_worker = load_module("label_import_ml_worker", ROOT / "ml_worker.py")


def run_import(project: Path, source: Path, **overrides) -> dict:
    values = dict(project=str(project), source=str(source), format="auto", class_map="",
                  candidates=False, overwrite=False, language="en")
    values.update(overrides)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        ml_worker.import_labels(argparse.Namespace(**values))
    return json.loads(output.getvalue().strip().splitlines()[-1])


class LabelImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        ml_worker.atomic_json(self.project / "project.json", {"sport": "football", "frames": []})
        self.original_emit = ml_worker.emit
        ml_worker.emit = lambda message: None

    def tearDown(self):
        ml_worker.emit = self.original_emit
        self.temporary.cleanup()

    def frames(self) -> list[dict]:
        return ml_worker.load_json(self.project / "project.json")["frames"]

    def yolo_dataset(self, names_yaml: str = "names:\n  0: person\n  1: ball\n  2: referee\n  3: ghost\n") -> Path:
        source = self.root / "dataset"
        (source / "images" / "train").mkdir(parents=True)
        (source / "labels" / "train").mkdir(parents=True)
        (source / "data.yaml").write_text("path: .\ntrain: images/train\n" + names_yaml, encoding="utf-8")
        Image.new("RGB", (200, 100), (10, 90, 10)).save(source / "images" / "train" / "a.jpg")
        Image.new("RGB", (100, 100), (90, 10, 10)).save(source / "images" / "train" / "b.png")
        (source / "labels" / "train" / "a.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n"            # player: x 50..150, y 25..75
            "1 0.1 0.2 0.1 0.2 0.8\n"       # ball with confidence column
            "3 0.5 0.5 0.1 0.1\n"           # unknown class "ghost"
            "2 0.2 0.2 0.1 0.1 0.9 0.3\n"   # 6 numbers = polygon? (cls + 6 values) -> polygon bbox
            "bad line\n", encoding="utf-8")
        (source / "labels" / "train" / "b.txt").write_text("", encoding="utf-8")  # background image
        Image.new("RGB", (50, 50)).save(source / "images" / "train" / "unlabelled.jpg")  # no label file
        return source

    def test_imports_yolo_dataset_with_pixel_boxes_and_reports_what_was_skipped(self):
        summary = run_import(self.project, self.yolo_dataset())

        self.assertEqual(summary["format"], "yolo")
        self.assertEqual(summary["imported"], 2)
        self.assertEqual(summary["imagesWithoutLabels"], 1)
        self.assertEqual(summary["invalidLines"], 1)
        self.assertEqual(summary["skippedClasses"], {"ghost": 1})
        frames = {Path(frame["relativePath"]).name.split("-", 1)[1]: frame for frame in self.frames()}
        frame = frames["a.jpg"]
        self.assertEqual((frame["width"], frame["height"]), (200, 100))
        self.assertEqual(frame["reviewStatus"], "reviewed")
        self.assertTrue(frame["videoID"].startswith("import-"))
        player = next(item for item in frame["annotations"] if item["category"] == "player")
        self.assertEqual((player["x"], player["y"], player["width"], player["height"]), (50.0, 25.0, 100.0, 50.0))
        self.assertEqual(player["source"], "manual")
        ball = next(item for item in frame["annotations"] if item["category"] == "ball")
        self.assertAlmostEqual(ball["x"], 10.0)
        self.assertAlmostEqual(ball["y"], 10.0)
        self.assertEqual(frames["b.png"]["annotations"], [])
        self.assertTrue((self.project / frame["relativePath"]).is_file())

    def test_polygon_lines_become_their_bounding_box(self):
        source = self.root / "poly"
        (source / "images").mkdir(parents=True)
        (source / "labels").mkdir()
        (source / "data.yaml").write_text("names: [player, ball]\n", encoding="utf-8")
        Image.new("RGB", (100, 100)).save(source / "images" / "p.jpg")
        (source / "labels" / "p.txt").write_text("0 0.1 0.2 0.5 0.2 0.3 0.8\n", encoding="utf-8")

        run_import(self.project, source)

        box = self.frames()[0]["annotations"][0]
        self.assertEqual(box["category"], "player")
        self.assertAlmostEqual(box["x"], 10.0)
        self.assertAlmostEqual(box["y"], 20.0)
        self.assertAlmostEqual(box["width"], 40.0)
        self.assertAlmostEqual(box["height"], 60.0)

    def test_reimporting_the_same_images_is_skipped(self):
        source = self.yolo_dataset()
        run_import(self.project, source)
        again = run_import(self.project, source)
        self.assertEqual(again["imported"], 0)
        self.assertEqual(again["duplicates"], 2)
        self.assertEqual(len(self.frames()), 2)

    def test_class_map_is_needed_without_class_names_and_is_used_when_given(self):
        source = self.root / "bare"
        (source / "images").mkdir(parents=True)
        (source / "labels").mkdir()
        Image.new("RGB", (100, 100)).save(source / "images" / "x.jpg")
        (source / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        with self.assertRaises(SystemExit):
            run_import(self.project, source)
        summary = run_import(self.project, source, class_map="0=person")
        self.assertEqual(summary["annotations"], 1)
        self.assertEqual(self.frames()[0]["annotations"][0]["category"], "player")

    def test_candidates_option_imports_unreviewed_auto_suggestions(self):
        run_import(self.project, self.yolo_dataset(), candidates=True)
        frame = next(item for item in self.frames() if item["relativePath"].endswith("a.jpg"))
        self.assertEqual(frame["reviewStatus"], "candidate")
        self.assertTrue(all(item["source"] == "auto" for item in frame["annotations"]))

    def test_labels_are_attached_to_an_existing_frame_with_the_same_file_name(self):
        (self.project / "frames").mkdir()
        Image.new("RGB", (200, 100)).save(self.project / "frames" / "a.jpg")
        ml_worker.atomic_json(self.project / "project.json", {"sport": "football", "frames": [
            {"id": "f1", "relativePath": "frames/a.jpg", "videoID": "v1", "videoName": "v", "timestamp": 0, "width": 200, "height": 100, "annotations": []},
        ]})

        summary = run_import(self.project, self.yolo_dataset())

        self.assertEqual(summary["matched"], 1)
        self.assertEqual(summary["imported"], 1)  # only b.png is new
        existing = next(frame for frame in self.frames() if frame["id"] == "f1")
        self.assertTrue(existing["annotations"])
        self.assertEqual(existing["reviewStatus"], "reviewed")

    def test_imports_coco_with_class_name_aliases(self):
        source = self.root / "coco"
        source.mkdir()
        Image.new("RGB", (200, 100)).save(source / "c.jpg")
        (source / "_annotations.coco.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "c.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 0, "name": "objects"}, {"id": 1, "name": "Sports Ball"}, {"id": 2, "name": "Goalkeeper"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 10, 50, 80]},
                {"id": 3, "image_id": 1, "category_id": 0, "bbox": [0, 0, 5, 5]},
            ],
        }), encoding="utf-8")

        summary = run_import(self.project, source)

        self.assertEqual(summary["format"], "coco")
        self.assertEqual(summary["annotations"], 2)
        self.assertEqual(summary["skippedClasses"], {"objects": 1})
        by_category = {item["category"]: item for item in self.frames()[0]["annotations"]}
        self.assertEqual((by_category["ball"]["x"], by_category["ball"]["width"]), (10.0, 30.0))
        self.assertIn("goalkeeper", by_category)

    def test_boxes_are_clamped_to_the_image(self):
        source = self.root / "clamp"
        (source / "images").mkdir(parents=True)
        (source / "labels").mkdir()
        (source / "data.yaml").write_text("names: [player]\n", encoding="utf-8")
        Image.new("RGB", (100, 100)).save(source / "images" / "e.jpg")
        (source / "labels" / "e.txt").write_text("0 0.95 0.5 0.3 0.2\n", encoding="utf-8")
        run_import(self.project, source)
        box = self.frames()[0]["annotations"][0]
        self.assertAlmostEqual(box["x"] + box["width"], 100.0)

    def test_image_size_reads_png_and_jpeg_headers(self):
        png, jpg = self.root / "a.png", self.root / "a.jpg"
        Image.new("RGB", (321, 123)).save(png)
        Image.new("RGB", (640, 480)).save(jpg, quality=85)
        self.assertEqual(ml_worker.image_size(png), (321, 123))
        self.assertEqual(ml_worker.image_size(jpg), (640, 480))
        self.assertIsNone(ml_worker.image_size(self.root / "missing.jpg"))


if __name__ == "__main__":
    unittest.main()


class CarryOverImportedFramesTests(unittest.TestCase):
    def test_imported_frames_survive_preparing_videos_again(self):
        local_worker = load_module("label_import_local_worker", ROOT / "local_worker.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            (root / "frames" / "imported").mkdir(parents=True)
            (root / "frames" / "imported" / "abc-a.jpg").write_bytes(b"x")
            (root / "frames" / "video-000001.jpg").write_bytes(b"y")
            project = {"frames": [
                {"id": "i1", "relativePath": "frames/imported/abc-a.jpg", "videoID": "import-1234",
                 "annotations": [{"category": "player"}, {"category": "hoop"}]},
                {"id": "v1", "relativePath": "frames/video-000001.jpg", "videoID": "v", "annotations": []},
                {"id": "evil", "relativePath": "../outside.jpg", "videoID": "import-1234", "annotations": []},
            ]}
            new_frames = Path(temporary) / "frames.next"
            new_frames.mkdir()

            kept = local_worker.carry_over_imported_frames(root, project, new_frames, "football")

            self.assertEqual([frame["id"] for frame in kept], ["i1"])
            self.assertEqual([item["category"] for item in kept[0]["annotations"]], ["player"])
            self.assertTrue((new_frames / "imported" / "abc-a.jpg").is_file())
