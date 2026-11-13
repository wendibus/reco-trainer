"""Roles chosen per video: dataset_split honours them, the worker stores them."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import dataset_split  # noqa: E402


def frames_of(videos: str, per_video: int = 5) -> list[dict]:
    return [{"id": f"{video}{i}", "videoID": video, "timestamp": float(i), "annotations": [{"category": "ball"}]} for video in videos for i in range(per_video)]


def roles_by_video(splits: dict) -> dict:
    found: dict[str, set] = {}
    for role, items in splits.items():
        for item in items:
            found.setdefault(item["videoID"], set()).add(role)
    return {video: next(iter(roles)) if len(roles) == 1 else roles for video, roles in found.items()}


class SplitWithRolesTests(unittest.TestCase):
    def test_without_choices_nothing_changes(self):
        frames = frames_of("abc")
        self.assertEqual(roles_by_video(dataset_split.split_frames(frames, "en")[0]), roles_by_video(dataset_split.split_frames(frames, "en", {})[0]))

    def test_a_chosen_video_gets_its_role_whole(self):
        splits, warning = dataset_split.split_frames(frames_of("abcd"), "en", {"a": "valid", "b": "test"})
        self.assertEqual(roles_by_video(splits), {"a": "valid", "b": "test", "c": "train", "d": "train"})
        self.assertIsNone(warning)

    def test_a_missing_role_is_filled_from_the_other_videos_but_one_keeps_training(self):
        auto = roles_by_video(dataset_split.split_frames(frames_of("abcd"), "en", {"a": "valid"})[0])
        self.assertEqual(auto["a"], "valid")
        self.assertEqual(sorted(auto.values()), ["test", "train", "train", "valid"])
        # only two videos left over after one choice: no test is made up, a training video stays
        only = roles_by_video(dataset_split.split_frames(frames_of("ab"), "en", {"a": "valid"})[0])
        self.assertEqual(only, {"a": "valid", "b": "train"})

    def test_choosing_training_keeps_a_video_out_of_the_automatic_roles(self):
        roles = roles_by_video(dataset_split.split_frames(frames_of("abc"), "en", {"a": "train", "b": "train"})[0])
        self.assertEqual(roles, {"a": "train", "b": "train", "c": "train"})

    def test_unknown_roles_and_videos_are_ignored(self):
        plain = roles_by_video(dataset_split.split_frames(frames_of("abc"), "en")[0])
        self.assertEqual(roles_by_video(dataset_split.split_frames(frames_of("abc"), "en", {"a": "nonsense", "zzz": "test"})[0]), plain)

    def test_the_preview_uses_the_choices(self):
        result = dataset_split.preview(frames_of("abc"), "en", None, {"a": "test", "b": "valid"})
        self.assertEqual(result["roles"]["a0"], "test")
        self.assertEqual(result["roles"]["b0"], "valid")
        self.assertEqual(result["roles"]["c0"], "train")


class WorkerRoleTests(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location("video_roles_local_worker", ROOT / "local_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_worker_stores_a_role_and_removes_it_on_auto(self):
        worker = self.load()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = {"sport": "basketball", "frames": frames_of("ab") + [{"id": "i1", "videoID": "import-1", "annotations": []}]}
            (root / "project.json").write_text(json.dumps(project), encoding="utf-8")
            worker.STATE.update(project_root=root, project=project, language="en")
            try:
                worker.set_video_role({"videoID": "a", "role": "valid"})
                self.assertEqual(json.loads((root / "project.json").read_text())["videoRoles"], {"a": "valid"})
                self.assertEqual(worker.STATE.snapshot()["videoRoles"], {"a": "valid"})
                self.assertEqual(worker.STATE.snapshot()["splits"]["roles"]["a0"], "valid")
                worker.set_video_role({"videoID": "a", "role": "auto"})
                self.assertNotIn("videoRoles", json.loads((root / "project.json").read_text()))
                for bad in ({"videoID": "a", "role": "boss"}, {"videoID": "nope", "role": "test"}, {"videoID": "import-1", "role": "test"}):
                    with self.assertRaises(RuntimeError):
                        worker.set_video_role(bad)
            finally:
                worker.STATE.update(project_root=None, project=None, language="de")

    def test_every_training_path_reads_the_choices(self):
        source = (ROOT / "ml_worker.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('.get("videoRoles")'), 5)
        self.assertIn('"/api/video-role"', (ROOT / "local_worker.py").read_text(encoding="utf-8"))
        self.assertIn("VideoRoles", (ROOT / "app" / "page.tsx").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
