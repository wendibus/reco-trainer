"""Consistency check: second detector, timeline, VLM verdicts, hard examples."""
from __future__ import annotations

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
import vlm  # noqa: E402

spec = importlib.util.spec_from_file_location("consistency_ml_worker", ROOT / "ml_worker.py")
assert spec and spec.loader
ml_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_worker)


def person(x: float, category: str = "player") -> dict:
    return {"id": f"{category}-{x}", "category": category, "x": x, "y": 50.0, "width": 40.0, "height": 120.0, "source": "manual"}


class FakeDetections:
    def __init__(self, boxes):
        import numpy as np
        self.xyxy = np.array([[x, 50.0, x + 40.0, 170.0] for x in boxes], dtype=float).reshape(-1, 4)
        self.confidence = np.ones(len(boxes))
        self.class_id = np.zeros(len(boxes), dtype=int)
        self.data = {"class_name": ["person"] * len(boxes)}


class FakeDetector:
    """Finds people at fixed x positions per image file name."""
    def __init__(self, per_image):
        self.per_image = per_image

    def predict(self, paths, threshold=0.0):
        return [FakeDetections(self.per_image[Path(path).name]) for path in paths]


class FakeServer:
    url = "http://127.0.0.1:1"

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


try:
    import numpy  # noqa: F401
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class HelperTests(unittest.TestCase):
    def test_referees_and_goalkeepers_compare_as_people(self):
        self.assertEqual([box["category"] for box in ml_worker.comparable([person(0, "referee"), person(1, "goalkeeper"), {"category": "puck", "x": 0, "y": 0, "width": 1, "height": 1}, {"category": "goal"}])], ["person", "person", "ball"])

    def test_vanished_players_need_a_short_gap_and_a_real_drop(self):
        frames = [
            {"id": "a", "videoID": "v", "timestamp": 0.0, "annotations": [person(0), person(50), person(100), person(150)]},
            {"id": "b", "videoID": "v", "timestamp": 0.5, "annotations": [person(0)]},
            {"id": "c", "videoID": "v", "timestamp": 9.0, "annotations": []},
        ]
        self.assertEqual(ml_worker.flag_vanished_players(frames), 1)
        self.assertEqual(frames[1]["reviewFlags"], ["players-vanished"])
        self.assertNotIn("reviewFlags", frames[2])


@unittest.skipUnless(HAS_NUMPY, "numpy is not installed")
class ConsistencyCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "frames").mkdir()
        frames = []
        for name, labelled in (("agree", [0, 100]), ("missing", [0]), ("vlm-bad", [0, 100]), ("clean", [0, 100])):
            (self.root / "frames" / f"{name}.jpg").write_bytes(b"jpeg")
            frames.append({"id": name, "relativePath": f"frames/{name}.jpg", "videoID": name, "timestamp": 0.0, "width": 300, "height": 200,
                           "reviewStatus": "reviewed", "annotations": [person(x) for x in labelled]})
        ml_worker.atomic_json(self.root / "project.json", {"sport": "basketball", "frames": frames})
        # The second detector agrees on "agree", "vlm-bad" and "clean", but sees a second person on "missing".
        self.detector = FakeDetector({"agree.jpg": [0, 100], "missing.jpg": [0, 100], "vlm-bad.jpg": [0, 100], "clean.jpg": [0, 100]})

    def tearDown(self):
        self.temporary.cleanup()

    def run_check(self, use_vlm: bool, answers: dict[str, dict] | None = None, sample: float = 0.0) -> tuple[dict, dict]:
        asked: list[str] = []

        def fake_ask(_url, image, _prompt, **_kwargs):
            asked.append(Path(image).stem)
            return json.dumps((answers or {}).get(Path(image).stem, {"missing": 0, "duplicates": 0, "comment": "fine"}))

        args = argparse.Namespace(project=str(self.root), iou=0.5, threshold=0.3, vlm=use_vlm, vlm_model="qwen3.5-4b-vlm",
                                  sample=sample, teacher="auto", tier="strong", language="en")
        output = io.StringIO()
        with mock.patch.object(ml_worker, "model_instance", lambda *_a, **_k: self.detector), \
                mock.patch.object(ml_worker, "detect_device", lambda: "cpu"), \
                mock.patch.object(ml_worker, "prepare_vlm", lambda *_a: (Path("server"), Path("m"), Path("p"), "relative-1000")), \
                mock.patch.object(vlm, "LlamaServer", FakeServer), mock.patch.object(vlm, "ask_image", fake_ask), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            ml_worker.consistency_check(args)
        summary = json.loads(output.getvalue().strip().splitlines()[-1])
        frames = {frame["id"]: frame for frame in json.loads((self.root / "project.json").read_text())["frames"]}
        return summary, {"frames": frames, "asked": asked}

    def test_without_vlm_the_disagreement_stays_for_review(self):
        summary, state = self.run_check(use_vlm=False)
        self.assertEqual(summary["detectorDisagreement"], 1)
        self.assertEqual(summary["review"], ["missing"])
        self.assertEqual(state["frames"]["missing"]["reviewFlags"], ["detector-disagreement"])
        self.assertNotIn("reviewFlags", state["frames"]["agree"])

    def test_vlm_clears_resolved_warnings_and_marks_hard_examples(self):
        summary, state = self.run_check(use_vlm=True, answers={"missing": {"missing": 0, "duplicates": 0}}, sample=1.0)
        # The VLM saw nothing wrong on "missing": warning cleared.
        self.assertNotIn("reviewFlags", state["frames"]["missing"])
        self.assertEqual(summary["vlmResolved"], 1)
        self.assertEqual(sorted(state["asked"]), ["agree", "clean", "missing", "vlm-bad"])

        summary, state = self.run_check(use_vlm=True, answers={"vlm-bad": {"missing": 1, "duplicates": 0, "comment": "a player has no box"}}, sample=1.0)
        bad = state["frames"]["vlm-bad"]
        self.assertEqual(bad["reviewFlags"], ["vlm-issue"])
        self.assertTrue(bad["hardExample"])
        self.assertEqual(bad["vlmCheck"]["missing"], 1)
        self.assertEqual(summary["vlmIssues"], 1)

    def test_vlm_only_sees_flagged_frames_without_a_sample(self):
        _, state = self.run_check(use_vlm=True, sample=0.0)
        self.assertEqual(state["asked"], ["missing"])

    def test_hard_examples_go_into_training_twice(self):
        document = json.loads((self.root / "project.json").read_text())
        for frame in document["frames"]:
            frame["hardExample"] = frame["id"] == "agree"
            frame["videoID"] = "one-video"  # all in one split, so "agree" is in train
        ml_worker.atomic_json(self.root / "project.json", document)
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = ml_worker.build_dataset(self.root, "en")
        names = []
        for split in ("train", "valid", "test"):
            coco = json.loads((dataset / split / "_annotations.coco.json").read_text())
            names += [image["file_name"] for image in coco["images"]]
            ids = [image["id"] for image in coco["images"]]
            self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("hard-agree.jpg", names)
        self.assertEqual(names.count("agree.jpg"), 1)


if __name__ == "__main__":
    unittest.main()
