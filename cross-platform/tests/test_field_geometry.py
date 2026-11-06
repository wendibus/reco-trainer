"""Field outline per video (field_geometry.py) and its migration from the single outline."""
import copy
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import field_geometry
import local_worker
import ml_worker

OLD = {"corners": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], "realWidth": 20.0, "realLength": 40.0}
LEFT = [[0.0, 0.2], [0.6, 0.2], [0.6, 1.0], [0.0, 1.0]]


def frame(frame_id: str, video: str) -> dict:
    return {"id": frame_id, "videoName": video, "relativePath": f"frames/{frame_id}.jpg", "width": 1000, "height": 1000, "annotations": []}


def old_project() -> dict:
    """A project as 0.28 stored it: one outline for every frame."""
    return {"schemaVersion": 2, "sport": "football", "fieldGeometry": copy.deepcopy(OLD),
            "frames": [frame("a1", "links.mp4"), frame("a2", "links.mp4"), frame("b1", "rechts.mp4")]}


class MigrationTests(unittest.TestCase):
    def test_old_outline_is_copied_to_every_video_and_kept(self):
        project = old_project()
        self.assertTrue(field_geometry.migrate(project))
        self.assertEqual(project["fieldGeometries"], {"links.mp4": OLD, "rechts.mp4": OLD})
        self.assertEqual(project["fieldGeometryBeforePerVideo"], OLD)
        self.assertNotIn("fieldGeometry", project)
        self.assertFalse(field_geometry.migrate(project), "runs once")

    def test_readers_get_the_same_outline_before_and_after_migration(self):
        before = old_project()
        after = copy.deepcopy(before)
        field_geometry.migrate(after)
        for video in ("links.mp4", "rechts.mp4"):
            self.assertEqual(field_geometry.geometry_for(before, video), field_geometry.geometry_for(after, video))

    def test_a_project_without_videos_keeps_the_single_outline(self):
        project = {"fieldGeometry": copy.deepcopy(OLD), "frames": []}
        self.assertFalse(field_geometry.migrate(project))
        self.assertEqual(field_geometry.geometry_for(project, "new.mp4"), OLD)

    def test_changing_one_video_leaves_the_others(self):
        project = old_project()
        field_geometry.set_geometry(project, ["links.mp4"], {"corners": LEFT})
        self.assertEqual(field_geometry.geometry_for(project, "links.mp4"), {"corners": LEFT})
        self.assertEqual(field_geometry.geometry_for(project, "rechts.mp4"), OLD)
        self.assertIsNone(field_geometry.geometry_for(project, "neu.mp4"), "a video added later has no outline yet")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.enterContext(unittest.mock.patch.object(local_worker, "STATE", local_worker.LocalState()))

    def _open(self, project: dict) -> dict:
        local_worker.atomic_json(self.root / "project.json", project)
        local_worker.STATE.update(project_root=self.root, project=local_worker.load_project(self.root))
        return json.loads((self.root / "project.json").read_text())

    def test_opening_an_existing_project_migrates_it_with_a_backup(self):
        original = old_project()
        stored = self._open(original)
        self.assertEqual(stored["fieldGeometries"], {"links.mp4": OLD, "rechts.mp4": OLD})
        self.assertEqual(stored["frames"], original["frames"], "frames and boxes untouched")
        backups = list((self.root / "backups").glob("project-*feldgrenze-pro-video.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text()), original, "backup holds the project as it was")
        snapshot = local_worker.STATE.snapshot()
        self.assertEqual(snapshot["fieldGeometries"]["rechts.mp4"], OLD)

    def test_saves_the_outline_of_one_video_or_all(self):
        self._open(old_project())
        local_worker.save_field_geometry({"corners": LEFT, "video": "links.mp4"})
        stored = json.loads((self.root / "project.json").read_text())["fieldGeometries"]
        self.assertEqual(stored, {"links.mp4": {"corners": LEFT}, "rechts.mp4": OLD})
        local_worker.save_field_geometry({"corners": LEFT, "video": "links.mp4", "applyToAll": True})
        stored = json.loads((self.root / "project.json").read_text())["fieldGeometries"]
        self.assertEqual(stored, {"links.mp4": {"corners": LEFT}, "rechts.mp4": {"corners": LEFT}})
        with self.assertRaises(RuntimeError):
            local_worker.save_field_geometry({"corners": LEFT, "video": "unknown.mp4"})


class AutoLabelFilterTests(unittest.TestCase):
    @unittest.skipUnless(ml_worker.field_membership_checker({"corners": LEFT}) is not None, "needs OpenCV")
    def test_each_frame_uses_the_outline_of_its_video(self):
        project = old_project()
        field_geometry.set_geometry(project, ["links.mp4"], {"corners": LEFT})
        is_on_field, present = ml_worker.field_membership_by_video(project)
        self.assertTrue(present)
        # feet at (800, 500): outside the left camera's outline, inside the right camera's
        self.assertFalse(is_on_field("player", 780, 400, 820, 500, 1000, 1000, "links.mp4"))
        self.assertTrue(is_on_field("player", 780, 400, 820, 500, 1000, 1000, "rechts.mp4"))
        self.assertTrue(is_on_field("player", 780, 50, 820, 60, 1000, 1000, "neu.mp4"), "no outline, no filter")
        self.assertTrue(is_on_field("ball", 780, 50, 820, 60, 1000, 1000, "links.mp4"), "only people are filtered")

    def test_no_outline_means_no_filter(self):
        self.assertEqual(ml_worker.field_membership_by_video({"frames": [frame("a", "x.mp4")]}), (None, False))


if __name__ == "__main__":
    unittest.main()
