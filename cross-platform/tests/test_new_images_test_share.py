"""How many of the new images become test: dataset_split pins a share of the reviewed candidates
that the last training has not seen, and the worker does it before a training starts."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import dataset_split  # noqa: E402


def load_worker():
    spec = importlib.util.spec_from_file_location("new_images_local_worker", ROOT / "local_worker.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(name: str, video: str = "a", reviewed: str | None = None, **extra) -> dict:
    item = {"id": name, "videoID": video, "timestamp": 1.0, "annotations": [{"category": "ball"}], "reviewStatus": "reviewed" if reviewed else None, **extra}
    if reviewed:
        item["reviewedAt"] = reviewed
    return item


class NewImageTests(unittest.TestCase):
    def test_only_images_reviewed_after_the_last_training_are_new(self):
        trained = "2026-11-10T12:00:00Z"
        self.assertTrue(dataset_split.is_new_image(frame("a", reviewed="2026-11-11T08:00:00Z"), trained))
        self.assertFalse(dataset_split.is_new_image(frame("b", reviewed="2026-11-09T08:00:00Z"), trained))
        self.assertFalse(dataset_split.is_new_image(frame("c"), trained))  # never reviewed as a candidate
        self.assertFalse(dataset_split.is_new_image(frame("d", reviewed="2026-11-11T08:00:00Z", testHold=True), trained))
        self.assertTrue(dataset_split.is_new_image(frame("e", reviewed="2026-11-11T08:00:00+00:00"), None))  # nothing trained yet

    def test_a_share_is_set_aside_as_test_and_stays_test(self):
        frames = [frame(f"n{i}", video=f"v{i % 2}", reviewed="2026-11-11T08:00:00Z") for i in range(10)]
        frames += [frame(f"old{i}", reviewed="2026-11-01T08:00:00Z") for i in range(5)]
        taken = dataset_split.pin_new_images_as_test(frames, 20, "2026-11-10T12:00:00Z")
        self.assertEqual(taken, 2)
        held = sorted(item["id"] for item in frames if item.get("testHold"))
        self.assertTrue(all(name.startswith("n") for name in held))
        # asking again for the rest of the new images does not take more than the share of what is left
        self.assertEqual(dataset_split.pin_new_images_as_test(frames, 0, "2026-11-10T12:00:00Z"), 0)
        result = dataset_split.preview(frames, "en", "2026-11-10T12:00:00Z")
        self.assertEqual(result["newImages"], 8)
        self.assertTrue(all(result["roles"][name] == "test" for name in held))
        self.assertGreaterEqual(result["counts"]["test"], 2)

    def test_at_least_one_image_when_a_share_is_chosen(self):
        frames = [frame(f"n{i}", reviewed="2026-11-11T08:00:00Z") for i in range(3)]
        self.assertEqual(dataset_split.pin_new_images_as_test(frames, 10, None), 1)

    def test_the_choice_is_the_same_each_time(self):
        first = [frame(f"n{i}", reviewed="2026-11-11T08:00:00Z") for i in range(20)]
        second = [frame(f"n{i}", reviewed="2026-11-11T08:00:00Z") for i in reversed(range(20))]
        dataset_split.pin_new_images_as_test(first, 30, None)
        dataset_split.pin_new_images_as_test(second, 30, None)
        self.assertEqual({f["id"] for f in first if f.get("testHold")}, {f["id"] for f in second if f.get("testHold")})


class WorkerTests(unittest.TestCase):
    def test_the_worker_sets_the_share_aside_and_saves_it_before_training(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = {"sport": "basketball", "lastTraining": {"completedAt": "2026-11-10T12:00:00Z"},
                       "frames": [frame(f"n{i}", reviewed="2026-11-11T08:00:00Z") for i in range(10)]}
            (root / "project.json").write_text(json.dumps(project), encoding="utf-8")
            worker.STATE.update(project_root=root, project=project)
            try:
                self.assertEqual(worker.pin_new_images_as_test(root, 30), 3)
                saved = json.loads((root / "project.json").read_text())
                self.assertEqual(sum(1 for item in saved["frames"] if item.get("testHold")), 3)
                self.assertEqual(worker.pin_new_images_as_test(root, 0), 0)
                self.assertEqual(worker.pin_new_images_as_test(root, "x"), 0)
            finally:
                worker.STATE.update(project_root=None, project=None)

    def test_a_decision_in_the_candidate_review_marks_the_time(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = {"id": "c1", "videoID": "a", "timestamp": 1.0, "reviewStatus": "candidate", "annotations": []}
            project = {"sport": "basketball", "frames": [candidate]}
            (root / "project.json").write_text(json.dumps(project), encoding="utf-8")
            worker.STATE.update(project_root=root, project=project, language="en")
            try:
                worker.review_candidate({"frameId": "c1", "decision": "no-ball", "category": "ball"})
                self.assertEqual(candidate["reviewStatus"], "reviewed")
                self.assertTrue(candidate["reviewedAt"].endswith("Z"))
                self.assertTrue(dataset_split.is_new_image(candidate, "2020-01-01T00:00:00Z"))
            finally:
                worker.STATE.update(project_root=None, project=None, language="de")

    def test_analysing_again_reuses_the_video_of_the_slot(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            video.write_bytes(b"video")
            worker.STATE.update(simulation_videos={"a": video})
            try:
                original = worker.select_video_file
                worker.select_video_file = lambda: self.fail("no dialog expected")
                try:
                    self.assertEqual(worker.simulation_video({"slot": "a", "again": True}), video)
                finally:
                    worker.select_video_file = original
                with self.assertRaises(RuntimeError):
                    worker.simulation_video({"slot": "b", "again": True})
                self.assertEqual(worker.requested_section({"start": 30, "length": 540, "every": 3}), {"start": 30.0, "length": 540.0, "fraction": 0.0, "every": 3})
            finally:
                worker.STATE.update(simulation_videos={})


class DialogContractTests(unittest.TestCase):
    def test_the_dialog_asks_and_offers_to_label_the_rest(self):
        dialog = (ROOT / "app" / "TrainDialog.tsx").read_text(encoding="utf-8")
        self.assertIn("newTestPercent", dialog)
        self.assertIn("labelRest", dialog)
        page = (ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        self.assertIn("setSplitFilter('none')", page)
        player = (ROOT / "app" / "BallTrackingPlayer.tsx").read_text(encoding="utf-8")
        self.assertIn("run('a', false, true)", player)


if __name__ == "__main__":
    unittest.main()
