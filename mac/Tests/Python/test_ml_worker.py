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

    def test_unreviewed_candidates_are_excluded_from_training_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewed = self.make_frame("reviewed", 1, root)
            candidate = self.make_frame("candidate", 1, root)
            candidate["reviewStatus"] = "candidate"
            ml_worker.atomic_json(root / "project.json", {
                "schemaVersion": 2, "name": "Active learning", "sport": "basketball",
                "sourceFolder": str(root), "frames": [reviewed, candidate],
            })

            dataset = ml_worker.build_dataset(root)
            names = []
            for split in ("train", "valid", "test"):
                coco = json.loads((dataset / split / "_annotations.coco.json").read_text())
                names.extend(image["file_name"] for image in coco["images"])

            self.assertIn(Path(reviewed["relativePath"]).name, names)
            self.assertNotIn(Path(candidate["relativePath"]).name, names)

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
        mapping = ml_worker.detection_category_map(["ball"])
        self.assertEqual(mapping["sports ball"], "ball")
        self.assertNotIn("person", mapping)

    def test_detection_category_map_merges_multiple_selected_categories(self):
        mapping = ml_worker.detection_category_map(["ball", "player"])
        self.assertEqual(mapping["sports ball"], "ball")
        self.assertEqual(mapping["person"], "player")
        # A category with no base-model alias (e.g. hoop) still maps to itself,
        # so a custom-trained model's own class name passes through unchanged.
        mapping_custom = ml_worker.detection_category_map(["hoop"])
        self.assertEqual(mapping_custom["hoop"], "hoop")
        self.assertNotIn("sports ball", mapping_custom)

    def test_autolabel_adds_multiple_categories_in_one_run(self):
        """A single auto_label() call with multiple --category values should add
        every still-missing selected category from one model pass per frame,
        without duplicating a category a frame already has."""
        class FakeDetections:
            data = {"class_name": ["sports ball", "person"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0], [40.0, 10.0, 60.0, 70.0]]
            class_id = [32, 0]
            confidence = [.9, .8]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                return [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "one.jpg").write_bytes(b"synthetic")
            (root / "frames" / "two.jpg").write_bytes(b"synthetic")
            project = {
                "sport": "basketball",
                "frames": [
                    {"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": []},
                    {"id": "two", "relativePath": "frames/two.jpg", "width": 100, "height": 80, "annotations": [
                        {"id": "existing", "category": "ball", "x": 1, "y": 1, "width": 5, "height": 5, "source": "manual"},
                    ]},
                ],
            }
            ml_worker.atomic_json(root / "project.json", project)
            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                args = type("Args", (), {
                    "project": str(root), "model": "nano", "category": ["ball", "player"],
                    "threshold": 0.25, "language": "en", "candidate_only": False,
                })()
                ml_worker.auto_label(args)
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device

            document = ml_worker.load_json(root / "project.json")
            frame_one = next(frame for frame in document["frames"] if frame["id"] == "one")
            frame_two = next(frame for frame in document["frames"] if frame["id"] == "two")

            self.assertEqual({item["category"] for item in frame_one["annotations"]}, {"ball", "player"})
            self.assertTrue(all(item["source"] == "auto" for item in frame_one["annotations"]))
            # frame two already had a manual "ball" box, so autolabel should only add "player".
            self.assertEqual(sorted(item["category"] for item in frame_two["annotations"]), ["ball", "player"])
            self.assertEqual(sum(item["category"] == "ball" for item in frame_two["annotations"]), 1)

    def test_category_specific_skip_preserves_other_classes(self):
        frame = {"annotations": [{"category": "player"}]}
        self.assertFalse(ml_worker.frame_has_category(frame, "ball"))
        self.assertTrue(ml_worker.frame_has_category(frame, "player"))

    def test_extended_sports_use_specialized_categories(self):
        self.assertEqual(ml_worker.sport_categories("rugby"), ["ball", "player", "referee", "goalpost"])
        self.assertIn("goalkeeper", ml_worker.sport_categories("lacrosse"))
        self.assertIn("goalpost", ml_worker.sport_categories("american_football"))

    def test_benchmark_scores_a_perfect_detection(self):
        truth = [{"id": "frame-1", "annotations": [{"category": "ball", "x": 10, "y": 20, "width": 10, "height": 10}]}]
        predictions = [{"frameID": "frame-1", "category": "ball", "confidence": .9, "box": [10, 20, 20, 30]}]
        metrics = ml_worker.evaluate_predictions(truth, predictions, ["ball"])
        self.assertEqual(metrics["truePositives"], 1)
        self.assertEqual(metrics["falsePositives"], 0)
        self.assertEqual(metrics["falseNegatives"], 0)
        self.assertAlmostEqual(metrics["mAP50"], 1.0)
        self.assertAlmostEqual(metrics["qualityScore"], 100.0)

    def test_box_iou_and_box_iou_xywh_are_independent_helpers(self):
        """Regression test: ml_worker.py used to define box_iou twice — once for
        xyxy corners (used by evaluate_predictions) and once for x/y/width/height
        boxes (used by the OpenCV refinement). The second definition silently
        shadowed the first at module scope, so evaluate_predictions computed IoU
        against the wrong box format. The two helpers must stay distinct and each
        must interpret its own coordinate format correctly.
        """
        self.assertIsNot(ml_worker.box_iou, ml_worker.box_iou_xywh)
        overlap_xyxy = ml_worker.box_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])
        self.assertAlmostEqual(overlap_xyxy, 25.0 / 175.0)
        overlap_xywh = ml_worker.box_iou_xywh((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0))
        self.assertAlmostEqual(overlap_xywh, 25.0 / 175.0)

    def test_benchmark_scores_a_partial_overlap_below_threshold_as_a_miss(self):
        """Unlike the other benchmark tests above, truth and prediction here are
        NOT identical, so a box-format bug (xyxy vs. x/y/width/height) changes
        the outcome instead of canceling out. This is what actually caught the
        box_iou/box_iou_xywh name collision.
        """
        truth = [{"id": "frame-1", "annotations": [{"category": "ball", "x": 0, "y": 0, "width": 20, "height": 20}]}]
        predictions = [{"frameID": "frame-1", "category": "ball", "confidence": .9, "box": [25, 0, 45, 20]}]
        metrics = ml_worker.evaluate_predictions(truth, predictions, ["ball"])
        self.assertEqual(metrics["truePositives"], 0)
        self.assertEqual(metrics["falsePositives"], 1)
        self.assertEqual(metrics["falseNegatives"], 1)

    def test_benchmark_penalizes_false_positives_and_missed_objects(self):
        truth = [{"id": "frame-1", "annotations": [
            {"category": "ball", "x": 10, "y": 20, "width": 10, "height": 10},
            {"category": "ball", "x": 50, "y": 50, "width": 10, "height": 10},
        ]}]
        predictions = [
            {"frameID": "frame-1", "category": "ball", "confidence": .9, "box": [10, 20, 20, 30]},
            {"frameID": "frame-1", "category": "ball", "confidence": .8, "box": [80, 80, 90, 90]},
        ]
        metrics = ml_worker.evaluate_predictions(truth, predictions, ["ball"])
        self.assertEqual(metrics["truePositives"], 1)
        self.assertEqual(metrics["falsePositives"], 1)
        self.assertEqual(metrics["falseNegatives"], 1)
        self.assertAlmostEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], .5)
        self.assertAlmostEqual(metrics["f1"], .5)

    def test_benchmark_maps_model_class_ids_to_sport_schema(self):
        self.assertEqual(ml_worker.benchmark_detection_category("", 0, "basketball", {"ball", "player"}), "ball")
        self.assertEqual(ml_worker.benchmark_detection_category("sports ball", 32, "hockey", {"puck"}), "puck")
        self.assertIsNone(ml_worker.benchmark_detection_category("person", 0, "football", {"ball"}))

    def test_benchmark_command_writes_local_ranking_and_predictions(self):
        class FakeDetections:
            data = {"class_name": ["ball"]}
            xyxy = [[10.0, 20.0, 20.0, 30.0]]
            class_id = [0]
            confidence = [.95]

        class FakeModel:
            def __init__(self, **_): pass
            def predict(self, paths, threshold):
                self.threshold = threshold
                return FakeDetections() if len(paths) == 1 else [FakeDetections() for _ in paths]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / "frames" / "one.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"synthetic")
            ml_worker.atomic_json(root / "project.json", {"sport": "basketball", "frames": []})
            ml_worker.atomic_json(root / "benchmarks" / "ground-truth.json", {
                "schemaVersion": 1, "createdAt": "2026-08-29T10:00:00Z", "sport": "basketball", "datasetID": "synthetic-dataset",
                "frames": [{"id": "one", "relativePath": "frames/one.jpg", "width": 100, "height": 80, "annotations": [{"category": "ball", "x": 10, "y": 20, "width": 10, "height": 10}]}],
            })
            model_dir = root / "models" / "library" / "synthetic-model"
            (model_dir / "weights").mkdir(parents=True)
            (model_dir / "weights" / "model.pth").write_bytes(b"synthetic")
            ml_worker.atomic_json(model_dir / "manifest.json", {"packageID": "synthetic-model", "sport": "basketball", "modelSize": "nano", "classes": ["ball"], "weights": {"file": "weights/model.pth"}})
            original_import = ml_worker.import_model_class
            original_device = ml_worker.detect_device
            try:
                ml_worker.import_model_class = lambda *_: FakeModel
                ml_worker.detect_device = lambda: "cpu"
                ml_worker.benchmark(type("Args", (), {"project": str(root), "threshold": .05, "language": "en"})())
            finally:
                ml_worker.import_model_class = original_import
                ml_worker.detect_device = original_device
            report = ml_worker.load_json(root / "benchmarks" / "latest.json")
            self.assertEqual(report["results"][0]["rank"], 1)
            self.assertAlmostEqual(report["results"][0]["metrics"]["qualityScore"], 100.0)
            prediction_files = list((root / "benchmarks" / "runs").rglob("*.predictions.json"))
            self.assertEqual(len(prediction_files), 1)
            self.assertNotIn("imageData", prediction_files[0].read_text())

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

    def test_only_newer_full_checkpoint_is_treated_as_interrupted_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "runs" / "small" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-full-checkpoint")

            interrupted = {"lastTraining": {"completedAt": "2000-01-01T00:00:00Z"}}
            completed = {"lastTraining": {"completedAt": "2999-01-01T00:00:00Z"}}

            self.assertEqual(
                ml_worker.interrupted_run_checkpoint(root, "small", interrupted),
                checkpoint,
            )
            self.assertIsNone(
                ml_worker.interrupted_run_checkpoint(root, "small", completed)
            )

    def _dataset_root_with_categories(self, count: int) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        ml_worker.atomic_json(root / "train" / "_annotations.coco.json", {
            "images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": f"class-{i}"} for i in range(count)],
        })
        return root

    # Regression coverage for a real crash: a strict Lightning resume
    # (trainer.fit(ckpt_path=...)) hard-crashes with a state_dict size-mismatch
    # RuntimeError when the checkpoint's detection head was trained for a different
    # number of classes than the project currently has - e.g. after a class was
    # added via auto-labeling while a training run sat interrupted.
    def test_validate_resume_checkpoint_allows_a_matching_checkpoint_through(self):
        dataset_root = self._dataset_root_with_categories(2)
        checkpoint = dataset_root / "last.ckpt"
        checkpoint.write_bytes(b"synthetic")
        original = ml_worker.checkpoint_class_count
        ml_worker.checkpoint_class_count = lambda _: 2
        try:
            result = ml_worker.validate_resume_checkpoint(checkpoint, dataset_root)
        finally:
            ml_worker.checkpoint_class_count = original
        self.assertEqual(result, checkpoint)

    def test_validate_resume_checkpoint_rejects_a_class_count_mismatch(self):
        # The exact scenario from the crash: interrupted at 2 classes, the project
        # now has 4 after auto-labeling added more categories.
        dataset_root = self._dataset_root_with_categories(4)
        checkpoint = dataset_root / "last.ckpt"
        checkpoint.write_bytes(b"synthetic")
        original = ml_worker.checkpoint_class_count
        ml_worker.checkpoint_class_count = lambda _: 2
        try:
            result = ml_worker.validate_resume_checkpoint(checkpoint, dataset_root)
        finally:
            ml_worker.checkpoint_class_count = original
        self.assertIsNone(result)

    def test_validate_resume_checkpoint_allows_through_when_undetermined(self):
        # Conservative default: don't block an otherwise-working resume just because
        # the checkpoint's internal shape couldn't be read.
        dataset_root = self._dataset_root_with_categories(2)
        checkpoint = dataset_root / "last.ckpt"
        checkpoint.write_bytes(b"synthetic")
        original = ml_worker.checkpoint_class_count
        ml_worker.checkpoint_class_count = lambda _: None
        try:
            result = ml_worker.validate_resume_checkpoint(checkpoint, dataset_root)
        finally:
            ml_worker.checkpoint_class_count = original
        self.assertEqual(result, checkpoint)

    def test_training_logs_are_preserved_in_unique_history_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "runs" / "small"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.csv").write_text("val/mAP_50_95\n0.42\n")
            (run_dir / "training_config.json").write_text('{"epochs": 20}')

            archived = ml_worker.archive_run_logs(run_dir)

            self.assertIsNotNone(archived)
            self.assertEqual((archived / "metrics.csv").read_text(), "val/mAP_50_95\n0.42\n")
            self.assertEqual((archived / "training_config.json").read_text(), '{"epochs": 20}')

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
