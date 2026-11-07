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
import yolo_training


def write_split(root: Path, split: str, images: list[dict], annotations: list[dict]) -> None:
    folder = root / split
    folder.mkdir(parents=True)
    for image in images:
        (folder / image["file_name"]).write_bytes(b"jpg")
    (folder / "_annotations.coco.json").write_text(json.dumps({
        "categories": [{"id": 2, "name": "player"}, {"id": 1, "name": "ball"}],
        "images": images, "annotations": annotations,
    }), encoding="utf-8")


class CocoToYoloTests(unittest.TestCase):
    def test_labels_are_normalised_centre_boxes_in_category_id_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = {"id": 7, "file_name": "a.jpg", "width": 200, "height": 100}
            write_split(root / "coco", "train", [image], [
                {"image_id": 7, "category_id": 1, "bbox": [10, 20, 20, 10]},
                {"image_id": 7, "category_id": 2, "bbox": [100, 0, 50, 100]},
            ])
            write_split(root / "coco", "valid", [{**image, "file_name": "b.jpg"}], [])
            data, names = yolo_training.coco_to_yolo(root / "coco", root / "yolo")
            self.assertEqual(names, ["ball", "player"])
            lines = (root / "yolo" / "labels" / "train" / "a.txt").read_text().split("\n")
            self.assertEqual(lines[0], "0 0.100000 0.250000 0.100000 0.100000")
            self.assertEqual(lines[1], "1 0.625000 0.500000 0.250000 1.000000")
            self.assertTrue((root / "yolo" / "images" / "train" / "a.jpg").exists())
            # An image without boxes still gets an (empty) label file: a background image.
            self.assertEqual((root / "yolo" / "labels" / "val" / "b.txt").read_text(), "")
            text = data.read_text()
            self.assertIn("val: images/val", text)
            self.assertNotIn("test:", text)

    def test_missing_validation_split_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_split(root / "coco", "train", [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}], [])
            with self.assertRaises(ValueError):
                yolo_training.coco_to_yolo(root / "coco", root / "yolo")


class YoloSettingsTests(unittest.TestCase):
    def test_metric_names_match_what_reco_trainer_compares(self):
        metrics = yolo_training.metrics_from({"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.5, "fitness": 1}, "test")
        self.assertEqual(metrics, {"test/mAP_50": 0.8, "test/mAP_50_95": 0.5})
        self.assertEqual(ml_worker.validation_score(metrics), 0.5)

    def test_variants_are_pinned_by_checksum_and_never_enter_the_registry(self):
        registry_ids = {entry["id"] for entry in ml_worker.load_registry()["models"]}
        for name, entry in yolo_training.VARIANTS.items():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(entry["url"].startswith("https://github.com/ultralytics/assets/releases/download/"))
            self.assertNotIn(name, registry_ids)
        self.assertIn("==", " ".join(yolo_training.INSTALL_ARGS[0]))
        self.assertNotIn("opencv-python", " ".join(" ".join(item) for item in yolo_training.INSTALL_ARGS))

    def test_no_telemetry_and_no_automatic_installs(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict("os.environ", {}, clear=False):
            yolo_training.configure_environment(Path(folder) / "settings")
            import os
            self.assertEqual(os.environ["YOLO_OFFLINE"], "1")
            self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "False")

    def test_batch_size_shrinks_for_large_images_and_the_cpu(self):
        self.assertGreater(yolo_training.batch_size("yolo-nano", "mps", 640), yolo_training.batch_size("yolo-nano", "mps", 1280))
        self.assertLess(yolo_training.batch_size("yolo-small", "cpu", 640), yolo_training.batch_size("yolo-small", "mps", 640))


class EvaluationFolderTests(unittest.TestCase):
    def test_validation_output_stays_inside_the_run(self):
        run_dir = Path("/projects/demo/runs/yolo-nano")
        options = yolo_training.eval_folder(run_dir, "test")
        self.assertEqual(Path(options["project"]), run_dir)
        self.assertEqual(options["name"], "eval-test")
        source = (ROOT / "yolo_training.py").read_text(encoding="utf-8")
        self.assertEqual(source.count(".val("), source.count("**eval_folder(run_dir"))


class AppVersionTests(unittest.TestCase):
    def test_packaged_version_file_is_read_when_package_json_is_gone(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "version.json").write_text('{"version": "9.9.9"}')
            with mock.patch.object(local_worker, "BASE_DIR", Path(folder)):
                self.assertEqual(local_worker.app_version(), "9.9.9")

    def test_packaging_writes_the_version_file_for_windows_and_linux(self):
        script = (ROOT / "scripts" / "package-platforms.sh").read_text(encoding="utf-8")
        self.assertIn('> "$WINDOWS_DIR/version.json"', script)
        self.assertIn('> "$LINUX_DIR/version.json"', script)
        self.assertIn("--exclude '/runs'", script)


class WindowsMemoryTests(unittest.TestCase):
    def test_memory_is_read_without_sysconf_on_windows(self):
        import gpu_diagnosis
        with mock.patch.object(local_worker.os, "sysconf", side_effect=AttributeError, create=True), \
                mock.patch.object(local_worker.os, "name", "nt"), \
                mock.patch.object(gpu_diagnosis, "windows_memory_bytes", return_value=32 * 1024 ** 3):
            self.assertEqual(local_worker.hardware_summary()["memoryGB"], 32)
            self.assertEqual(ml_worker.total_memory_bytes(), 32 * 1024 ** 3)


class WorkerIntegrationTests(unittest.TestCase):
    def test_train_accepts_yolo_variants(self):
        parser = ml_worker.build_parser() if hasattr(ml_worker, "build_parser") else None
        if parser is None:
            self.skipTest("parser is built in main()")
        args = parser.parse_args(["train", "--project", ".", "--model", "yolo-small", "--image-size", "960"])
        self.assertEqual((args.model, args.image_size), ("yolo-small", 960))

    def test_recommendation_prefers_rf_detr_order_then_yolo(self):
        results = [{"model": "yolo-nano", "status": "done", "mAP50": 0.80}, {"model": "small", "status": "done", "mAP50": 0.795}]
        self.assertEqual(ml_worker.recommended_model(results), "small")
        results = [{"model": "yolo-nano", "status": "done", "mAP50": 0.90}, {"model": "small", "status": "done", "mAP50": 0.80}]
        self.assertEqual(ml_worker.recommended_model(results), "yolo-nano")

    def test_yolo_is_only_installed_after_the_license_was_accepted(self):
        with mock.patch.object(local_worker, "yolo_installed", return_value=False), mock.patch.object(local_worker, "run_logged") as run:
            with self.assertRaises(RuntimeError):
                local_worker.ensure_yolo(Path("python"), False)
            run.assert_not_called()
            local_worker.ensure_yolo(Path("python"), True)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn(f"ultralytics=={yolo_training.ULTRALYTICS_VERSION}", commands[0])

    def test_continuation_picks_the_same_yolo_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for package, variant, created in (("a", "yolo-nano", "2026-01-01"), ("b", "yolo-small", "2026-02-01"), ("c", "yolo-nano", "2026-03-01")):
                weights = root / "models" / "library" / package / "weights"
                weights.mkdir(parents=True)
                (weights / "best.pt").write_bytes(b"pt")
                (weights.parent / "manifest.json").write_text(json.dumps({"packageID": package, "yoloVariant": variant, "createdAt": created, "trainWeights": "weights/best.pt"}))
            found = ml_worker.yolo_library_parent(root, "yolo-nano")
            self.assertEqual(found[1]["packageID"], "c")
            self.assertIsNone(ml_worker.yolo_library_parent(root, "yolo-medium"))


if __name__ == "__main__":
    unittest.main()
