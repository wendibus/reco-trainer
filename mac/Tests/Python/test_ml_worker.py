from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path


WORKER_PATH = (
    Path(__file__).parents[2]
    / "Sources"
    / "RecoTrainerMac"
    / "Resources"
    / "ml_worker.py"
)
SPEC = importlib.util.spec_from_file_location("ml_worker", WORKER_PATH)
assert SPEC and SPEC.loader
ml_worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ml_worker)


class DatasetTests(unittest.TestCase):
    def make_frame(self, video_id: str, number: int, root: Path, annotated: bool = True):
        relative = f"frames/{video_id}-{number}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg")
        annotations = []
        if annotated:
            annotations.append(
                {
                    "id": str(uuid.uuid4()),
                    "category": "ball",
                    "x": 10.0,
                    "y": 20.0,
                    "width": 8.0,
                    "height": 8.0,
                    "source": "manual",
                }
            )
        return {
            "id": str(uuid.uuid4()),
            "relativePath": relative,
            "videoID": video_id,
            "videoName": f"{video_id}.mov",
            "timestamp": float(number),
            "width": 1920,
            "height": 1080,
            "annotations": annotations,
        }

    def test_builds_coco_and_keeps_videos_in_separate_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = [
                self.make_frame(video, number, root)
                for video in ("a", "b", "c")
                for number in range(2)
            ]
            project = {
                "schemaVersion": 1,
                "name": "Test",
                "sport": "football",
                "sourceFolder": str(root),
                "createdAt": "2026-08-28T10:00:00Z",
                "updatedAt": "2026-08-28T10:00:00Z",
                "frames": frames,
            }
            ml_worker.atomic_json(root / "project.json", project)

            dataset = ml_worker.build_dataset(root)

            seen_videos = {}
            for split in ("train", "valid", "test"):
                coco = json.loads((dataset / split / "_annotations.coco.json").read_text())
                self.assertEqual(coco["categories"][0]["name"], "ball")
                for image in coco["images"]:
                    video = image["file_name"].split("-")[0]
                    if video in seen_videos:
                        self.assertEqual(seen_videos[video], split)
                    else:
                        seen_videos[video] = split
            self.assertEqual(set(seen_videos.values()), {"train", "valid", "test"})

    def test_single_video_temporal_split(self):
        frames = [
            {
                "videoID": "one",
                "timestamp": float(number),
                "annotations": [],
            }
            for number in range(20)
        ]
        splits, warning = ml_worker.split_frames(frames)
        self.assertIsNotNone(warning)
        self.assertEqual(len(splits["train"]), 14)
        self.assertEqual(len(splits["valid"]), 4)
        self.assertEqual(len(splits["test"]), 2)

    def test_ball_focus_ignores_all_people(self):
        mapping = ml_worker.detection_category_map("ball")
        self.assertEqual(mapping["sports ball"], "ball")
        self.assertNotIn("person", mapping)

    def test_category_specific_skip_preserves_other_classes(self):
        frame = {"annotations": [{"category": "player"}]}
        self.assertFalse(ml_worker.frame_has_category(frame, "ball"))
        self.assertTrue(ml_worker.frame_has_category(frame, "player"))

    def test_extended_sports_use_specialized_categories(self):
        self.assertEqual(ml_worker.sport_categories("rugby"), ["ball", "player", "referee", "goalpost"])
        self.assertIn("goalkeeper", ml_worker.sport_categories("lacrosse"))
        self.assertIn("goalpost", ml_worker.sport_categories("american_football"))

    def test_mps_profile_scales_batch_and_parallel_data_loading(self):
        profile = ml_worker.training_profile(
            "nano", "mps", memory_bytes=16 * 1024**3, cpu_count=8
        )
        self.assertEqual(profile["batch_size"], 2)
        self.assertEqual(profile["grad_accum_steps"], 8)
        self.assertEqual(profile["num_workers"], 4)
        self.assertTrue(profile["persistent_workers"])
        self.assertFalse(profile["pin_memory"])

    def test_large_mps_profile_uses_larger_small_model_batches(self):
        profile = ml_worker.training_profile(
            "small", "mps", memory_bytes=64 * 1024**3, cpu_count=20
        )
        self.assertEqual(profile["batch_size"], 4)
        self.assertEqual(profile["grad_accum_steps"], 4)
        self.assertEqual(profile["num_workers"], 8)
        self.assertFalse(profile["gradient_checkpointing"])

    def test_old_export_signature_does_not_receive_new_optional_arguments(self):
        def old_export(output_dir="output", format="onnx", **kwargs):
            return output_dir, format, kwargs

        supported = ml_worker.explicitly_supported_kwargs(
            old_export,
            {
                "output_name": "custom-name",
                "coreml_precision": "float16",
                "notes": {"privacy": "local"},
            },
        )
        self.assertEqual(supported, {})

    def test_new_export_signature_receives_only_declared_options(self):
        def new_export(
            output_dir="output",
            format="onnx",
            *,
            coreml_precision=None,
            notes=None,
        ):
            return output_dir, format, coreml_precision, notes

        supported = ml_worker.explicitly_supported_kwargs(
            new_export,
            {
                "output_name": "custom-name",
                "coreml_precision": "float16",
                "notes": {"privacy": "local"},
            },
        )
        self.assertEqual(
            supported,
            {"coreml_precision": "float16", "notes": {"privacy": "local"}},
        )

    def test_exchange_package_contains_no_media_or_source_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "runs" / "nano" / "checkpoint_best_total.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-weights")
            project = {
                "schemaVersion": 2,
                "name": "Private Match",
                "sport": "basketball",
                "sourceFolder": "/private/videos/secret",
                "createdAt": "2026-08-28T10:00:00Z",
                "updatedAt": "2026-08-28T10:00:00Z",
                "frames": [self.make_frame("private-video", 1, root)],
            }
            ml_worker.atomic_json(root / "project.json", project)
            args = type("Args", (), {"project": str(root), "model": "nano", "language": "en"})()
            ml_worker.package_model(args)
            package = next((root / "exports" / "share").glob("*.recomodel"))
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "weights/checkpoint_best_total.pth"})
                manifest = archive.read("manifest.json").decode().lower()
                parsed_manifest = json.loads(manifest)
                self.assertTrue(parsed_manifest["description"].startswith("locally fine-tuned rf-detr nano model"))
                self.assertNotIn("secret", manifest)
                self.assertNotIn("private-video", manifest)
                self.assertNotIn("sourcefolder", manifest)

    def test_exchange_package_can_be_verified_installed_and_activated_locally(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            target = Path(temporary) / "target"
            checkpoint = source / "runs" / "nano" / "checkpoint_best_total.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-exchange-weights")
            for root in (source, target):
                root.mkdir(parents=True, exist_ok=True)
                ml_worker.atomic_json(root / "project.json", {
                    "schemaVersion": 2,
                    "name": "Synthetic Basketball",
                    "sport": "basketball",
                    "sourceFolder": str(root),
                    "createdAt": "2026-08-29T10:00:00Z",
                    "updatedAt": "2026-08-29T10:00:00Z",
                    "frames": [self.make_frame("synthetic", 1, root)],
                })
            package_args = type("Args", (), {"project": str(source), "model": "nano", "language": "es"})()
            ml_worker.package_model(package_args)
            package = next((source / "exports" / "share").glob("*.recomodel"))
            manifest = ml_worker.validate_model_package_file(package, "fr")
            install_args = type("Args", (), {"project": str(target), "file": str(package), "language": "es"})()
            ml_worker.install_model_package(install_args)
            active = ml_worker.load_json(target / "models" / "active.json")
            installed = ml_worker.newest_checkpoint(target, "nano")
            self.assertEqual(active["packageID"], manifest["packageID"])
            self.assertIsNotNone(installed)
            self.assertEqual(installed.read_bytes(), b"synthetic-exchange-weights")
            self.assertFalse(any((target / "models").rglob("*.jpg")))

    def test_exchange_package_for_another_sport_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            target = Path(temporary) / "target"
            checkpoint = source / "runs" / "nano" / "checkpoint.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"weights")
            source.mkdir(parents=True, exist_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            ml_worker.atomic_json(source / "project.json", {"sport": "basketball", "frames": []})
            ml_worker.atomic_json(target / "project.json", {"sport": "football", "frames": []})
            ml_worker.package_model(type("Args", (), {"project": str(source), "model": "nano", "language": "en"})())
            package = next((source / "exports" / "share").glob("*.recomodel"))
            with self.assertRaises(SystemExit):
                ml_worker.install_model_package(type("Args", (), {"project": str(target), "file": str(package), "language": "fr"})())


if __name__ == "__main__":
    unittest.main()
