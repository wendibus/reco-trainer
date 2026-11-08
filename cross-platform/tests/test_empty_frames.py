import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import local_worker
import ml_worker


def frame(number: int, video: str, boxes: int = 1, **extra) -> dict:
    annotations = [{"id": f"{number}-{index}", "category": "player", "x": 10, "y": 10, "width": 20, "height": 40} for index in range(boxes)]
    return {"id": f"id{number}", "relativePath": f"frames/f{number}.jpg", "videoID": video, "videoName": video, "timestamp": float(number), "width": 100, "height": 100, "annotations": annotations, **extra}


class TrainsOnTests(unittest.TestCase):
    def test_only_confirmed_empty_frames_train(self):
        self.assertTrue(ml_worker.trains_on(frame(1, "a")))
        self.assertFalse(ml_worker.trains_on(frame(2, "a", boxes=0)))
        self.assertTrue(ml_worker.trains_on(frame(3, "a", boxes=0, emptyConfirmed=True)))
        # "no ball" in the candidate review and imported label files say "reviewed".
        self.assertTrue(ml_worker.trains_on(frame(4, "a", boxes=0, reviewStatus="reviewed")))
        self.assertFalse(ml_worker.trains_on(frame(5, "a", reviewStatus="candidate")))

    def test_dataset_leaves_unconfirmed_empty_frames_out_and_says_so(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "frames").mkdir()
            frames = [frame(index, f"v{index % 3}", boxes=0 if index < 4 else 1, **({"emptyConfirmed": True} if index == 0 else {})) for index in range(18)]
            for item in frames:
                (root / item["relativePath"]).write_bytes(b"x")
            (root / "project.json").write_text(json.dumps({"schemaVersion": 2, "name": "t", "sport": "basketball", "frames": frames}))
            messages = []
            with mock.patch.object(ml_worker, "emit", side_effect=messages.append):
                dataset = ml_worker.build_dataset(root, "en")
            names = set()
            for split in ("train", "valid", "test"):
                names |= {image["file_name"] for image in json.loads((dataset / split / "_annotations.coco.json").read_text())["images"]}
            self.assertIn("f0.jpg", names)
            self.assertFalse({"f1.jpg", "f2.jpg", "f3.jpg"} & names)
            self.assertTrue(any("3 images without annotations" in message for message in messages))


class SaveEmptyTests(unittest.TestCase):
    def test_flag_is_set_and_cleared_by_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = {"schemaVersion": 2, "sport": "basketball", "frames": [frame(1, "a", boxes=0)]}
            with mock.patch.object(local_worker.STATE, "project_root", root), mock.patch.object(local_worker.STATE, "project", project), \
                    mock.patch.object(local_worker, "backup_project"), mock.patch.object(local_worker, "clean_annotations", side_effect=lambda _p, items: items):
                local_worker.save_annotations({"frameId": "id1", "annotations": [], "emptyConfirmed": True})
                self.assertTrue(project["frames"][0]["emptyConfirmed"])
                local_worker.save_annotations({"frameId": "id1", "annotations": [{"category": "player"}]})
                self.assertNotIn("emptyConfirmed", project["frames"][0])


if __name__ == "__main__":
    unittest.main()
