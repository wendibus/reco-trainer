"""Hardware tiers, teacher helpers, ball interpolation and the SAM 2 refinement path."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import teachers  # noqa: E402

spec = importlib.util.spec_from_file_location("teachers_ml_worker", ROOT / "ml_worker.py")
assert spec and spec.loader
ml_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_worker)

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False


class TierTests(unittest.TestCase):
    def test_tiers_by_device_and_memory(self):
        self.assertEqual(teachers.choose_tier("cuda", vram_gb=16), "strong")
        self.assertEqual(teachers.choose_tier("cuda", vram_gb=12), "medium")
        self.assertEqual(teachers.choose_tier("cuda", vram_gb=4), "weak")
        self.assertEqual(teachers.choose_tier("mps", memory_gb=48), "strong")
        self.assertEqual(teachers.choose_tier("mps", memory_gb=16), "medium")
        self.assertEqual(teachers.choose_tier("mps", memory_gb=8), "weak")
        self.assertEqual(teachers.choose_tier("cpu", memory_gb=128), "weak")

    def test_weak_tier_has_no_sam_and_no_vlm_but_a_teacher(self):
        plan = teachers.tier_plan("weak")
        self.assertIsNone(plan["refiner"])
        self.assertFalse(plan["vlm"])
        self.assertIn(plan["teacher"], teachers.TEACHER_SIZES)
        self.assertEqual(teachers.tier_plan("unknown"), plan)

    def test_every_planned_model_is_in_the_registry_with_an_allowed_license(self):
        registry = json.loads((ROOT / "model_registry.json").read_text())
        entries = {entry["id"]: entry for entry in registry["models"]}
        for plan in teachers.TIER_PLANS.values():
            for key in ("teacher", "refiner", "second"):
                if plan[key]:
                    self.assertIn(plan[key], entries)
                    self.assertIn(entries[plan[key]]["license"], registry["allowedLicenses"])

    def test_explicit_tier_wins_and_teacher_size_follows_the_tier(self):
        self.assertEqual(ml_worker.hardware_tier("medium"), "medium")
        self.assertEqual(ml_worker.teacher_size("auto", "strong"), "large")
        self.assertEqual(ml_worker.teacher_size("auto", "weak"), "small")
        self.assertEqual(ml_worker.teacher_size("nano", "strong"), "nano")


class FlorenceLabelTests(unittest.TestCase):
    def test_free_text_labels_map_to_coco_names(self):
        self.assertEqual(teachers.florence_label("Man"), "person")
        self.assertEqual(teachers.florence_label("basketball"), "sports ball")
        self.assertEqual(teachers.florence_label("tennis ball"), "sports ball")
        self.assertEqual(teachers.florence_label("scoreboard"), "scoreboard")


def frame(frame_id: str, timestamp: float, ball: tuple[float, float] | None, video: str = "v") -> dict:
    annotations = [] if ball is None else [{"id": f"b-{frame_id}", "category": "ball", "x": ball[0], "y": ball[1], "width": 10.0, "height": 10.0}]
    return {"id": frame_id, "videoID": video, "timestamp": timestamp, "width": 1000, "height": 600, "annotations": annotations}


class InterpolationTests(unittest.TestCase):
    def test_fills_a_missing_ball_between_two_close_frames(self):
        frames = [frame("a", 0.0, (100, 100)), frame("b", 0.2, None), frame("c", 0.4, (140, 120))]
        self.assertEqual(teachers.interpolate_ball(frames), 1)
        box = frames[1]["annotations"][0]
        self.assertAlmostEqual(box["x"], 120.0)
        self.assertAlmostEqual(box["y"], 110.0)
        self.assertEqual(box["source"], "auto")
        self.assertTrue(box["interpolated"])

    def test_no_guess_across_long_gaps_big_jumps_or_videos(self):
        far = [frame("a", 0.0, (100, 100)), frame("b", 5.0, None), frame("c", 10.0, (120, 100))]
        jump = [frame("a", 0.0, (0, 100)), frame("b", 0.2, None), frame("c", 0.4, (900, 100))]
        videos = [frame("a", 0.0, (100, 100), "v1"), frame("b", 0.2, None, "v2"), frame("c", 0.4, (120, 100), "v1")]
        for frames in (far, jump):
            self.assertEqual(teachers.interpolate_ball(frames), 0)
        self.assertEqual(teachers.interpolate_ball(videos), 0)

    def test_keeps_an_existing_ball(self):
        frames = [frame("a", 0.0, (100, 100)), frame("b", 0.2, (50, 50)), frame("c", 0.4, (140, 120))]
        self.assertEqual(teachers.interpolate_ball(frames), 0)


@unittest.skipUnless(HAS_CV, "OpenCV / numpy are not installed")
class Sam2RefinePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "frames").mkdir()
        cv2.imwrite(str(self.root / "frames" / "f.jpg"), np.full((200, 300, 3), 120, dtype=np.uint8))
        self.player = {"id": "p", "category": "player", "x": 100.0, "y": 40.0, "width": 60.0, "height": 140.0, "source": "auto"}
        self.ball = {"id": "b", "category": "ball", "x": 20.0, "y": 20.0, "width": 12.0, "height": 12.0, "source": "auto"}
        ml_worker.atomic_json(self.root / "project.json", {"sport": "basketball", "frames": [
            {"id": "f", "relativePath": "frames/f.jpg", "width": 300, "height": 200, "annotations": [self.player, self.ball]},
        ]})

    def tearDown(self):
        self.temporary.cleanup()

    def run_refine(self, refiner) -> list[dict]:
        with mock.patch.object(ml_worker, "sam2_refiner", lambda *_args: (refiner, "sam2.1-hiera-small")), mock.patch.object(ml_worker, "emit", lambda _m: None):
            ml_worker.refine_boxes(argparse.Namespace(project=str(self.root), language="en", method="auto", tier="strong"))
        return json.loads((self.root / "project.json").read_text())["frames"][0]["annotations"]

    def test_plausible_masks_tighten_boxes_and_keep_the_original(self):
        class FakeRefiner:
            def refine(self, image, boxes):
                return [(x + 5, y + 2, w - 10, h - 4) for x, y, w, h in boxes]

        player, ball = self.run_refine(FakeRefiner())
        self.assertEqual((player["x"], player["width"]), (105.0, 50.0))
        self.assertEqual(player["refinement"]["method"], "sam2.1-hiera-small")
        self.assertEqual(player["refinement"]["original"]["width"], 60.0)
        self.assertNotIn("opencvRefinement", player)

    def test_implausible_or_empty_masks_leave_boxes_unchanged(self):
        class WildRefiner:
            def refine(self, image, boxes):
                return [(0.0, 0.0, 300.0, 200.0), None]  # mask spilled over the whole frame / empty

        player, ball = self.run_refine(WildRefiner())
        self.assertEqual((player["x"], player["y"], player["width"], player["height"]), (100.0, 40.0, 60.0, 140.0))
        self.assertNotIn("refinement", player)
        self.assertNotIn("refinement", ball)

    def test_opencv_method_never_loads_sam(self):
        self.assertEqual(ml_worker.sam2_refiner(self.root, "opencv", "strong", "en"), (None, None))

    def test_sam_is_opt_in(self):
        # "auto" keeps OpenCV on every tier; only an explicit "sam2" loads SAM 2.
        for tier in ("strong", "medium", "weak"):
            self.assertEqual(ml_worker.sam2_refiner(self.root, "auto", tier, "en"), (None, None))


class BallGapTests(unittest.TestCase):
    """Frames of a dense clip where the ball was missed between two detections."""

    @staticmethod
    def clip(balls, fps=12.0, width=1920):
        return [
            {"file": f"sim-{index:04d}.jpg", "timestamp": index / fps, "width": width, "height": 1080,
             "ball": {"x": ball[0], "y": ball[1], "confidence": .9} if ball else None}
            for index, ball in enumerate(balls)
        ]

    def test_picks_the_middle_of_a_short_gap_with_the_straight_line_position(self):
        frames = self.clip([(100, 100), None, None, None, (300, 200)])
        picks = teachers.ball_gaps(frames, per_gap=1)
        self.assertEqual([pick["index"] for pick in picks], [2])
        self.assertAlmostEqual(picks[0]["x"], 200.0)
        self.assertAlmostEqual(picks[0]["y"], 150.0)
        self.assertEqual(picks[0]["file"], "sim-0002.jpg")
        self.assertGreaterEqual(picks[0]["radius"], 0.08 * 1920)

    def test_takes_at_most_per_gap_frames_and_never_the_detections_themselves(self):
        frames = self.clip([(100, 100), *[None] * 9, (200, 100)])
        picks = teachers.ball_gaps(frames, per_gap=3)
        self.assertEqual(len(picks), 3)
        self.assertTrue(all(0 < pick["index"] < 10 for pick in picks))
        self.assertEqual(len(teachers.ball_gaps(self.clip([(1, 1), None, (2, 2)]), per_gap=3)), 1)

    def test_skips_long_gaps_far_jumps_and_open_ends(self):
        self.assertEqual(teachers.ball_gaps(self.clip([(100, 100), *[None] * 30, (120, 100)])), [])  # 2.6 s
        self.assertEqual(teachers.ball_gaps(self.clip([(100, 100), None, (1500, 100)])), [])  # cut or wrong detection
        self.assertEqual(teachers.ball_gaps(self.clip([None, None, (100, 100), None, None])), [])  # lost, not bridged
        self.assertEqual(teachers.ball_gaps(self.clip([(100, 100), (110, 100), (120, 100)])), [])

    def test_limit_spreads_the_picks_over_the_whole_clip(self):
        frames = self.clip([(100, 100), None] * 40 + [(100, 100)])
        picks = teachers.ball_gaps(frames, limit=10)
        self.assertEqual(len(picks), 10)
        self.assertGreater(picks[-1]["index"], 60)


if __name__ == "__main__":
    unittest.main()
