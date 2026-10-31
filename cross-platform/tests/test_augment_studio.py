import hashlib
import json
import math
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

try:
    import cv2
    import numpy as np
    import augment_studio as studio
except ImportError:  # the ML environment is not there: nothing to test
    studio = None

WIDTH, HEIGHT = 1200, 500
TEAM_COLOURS = {0: (30, 30, 220), 1: (220, 90, 20)}


def person_height(y: float) -> float:
    """People further away (higher in the picture) are smaller, as in a real camera."""
    return 40.0 + 0.2 * y


def person_tilt(x: float) -> float:
    return 12.0 * (x - WIDTH / 2) / (WIDTH / 2)


def draw_person(image: np.ndarray, mask: np.ndarray, foot_x: float, foot_y: float, team: int) -> list[float]:
    height = person_height(foot_y)
    width = height * 0.32
    lean = math.tan(math.radians(person_tilt(foot_x)))
    top = np.zeros(image.shape[:2], dtype=np.uint8)
    for row in range(int(foot_y - height), int(foot_y)):
        share = (foot_y - row) / height
        cx = int(foot_x + lean * (foot_y - row))
        half = int(width / 2)
        colour = (40, 70, 110) if share > 0.85 else TEAM_COLOURS[team] if share > 0.35 else (30, 30, 30)
        image[row, cx - half:cx + half] = colour
        top[row, cx - half:cx + half] = 1
    mask |= top.astype(bool)
    ys, xs = np.nonzero(top)
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def court_background() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = (60, 120, 170)
    image[:150] = (70, 60, 50)
    cv2.line(image, (0, 300), (WIDTH, 300), (250, 250, 250), 3)
    return image


def write_dataset(folder: Path, frames: int = 16, seed: int = 3, prefix: str = "game") -> None:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    categories = [{"id": 1, "name": "player"}, {"id": 2, "name": "ball"}, {"id": 3, "name": "hoop"}]
    for number in range(frames):
        image = court_background()
        cv2.rectangle(image, (100, 170), (140, 200), (255, 255, 255), -1)
        file_name = f"{prefix}_{number:03d}.jpg"
        image_id = number + 1
        positions = []
        for index in range(10):
            team = index % 2
            for _ in range(40):
                x, y = float(rng.uniform(150, WIDTH - 150)), float(rng.uniform(250, HEIGHT - 20))
                if all(abs(x - other[0]) > 90 for other in positions):
                    break
            positions.append((x, y))
            mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
            box = draw_person(image, mask, x, y, team)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1, "bbox": box, "segmentation": studio.mask_to_rle(mask)})
        cv2.circle(image, (600, 120), 7, (0, 140, 255), -1)
        annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 2, "bbox": [593, 113, 15, 15], "segmentation": studio.mask_to_rle(np.array(cv2.circle(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), (600, 120), 7, 1, -1), dtype=bool))})
        hoop = np.zeros((HEIGHT, WIDTH), dtype=bool)
        hoop[170:200, 100:140] = True
        annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 3, "bbox": [100, 170, 40, 30], "segmentation": studio.mask_to_rle(hoop)})
        cv2.imwrite(str(folder / file_name), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
        images.append({"id": image_id, "file_name": file_name, "width": WIDTH, "height": HEIGHT})
    (folder / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}), encoding="utf-8")


def tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name not in {"run.json"}:
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class RleTests(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        for shape in ((1, 1), (7, 3), (40, 90)):
            mask = rng.random(shape) > 0.5
            self.assertTrue((studio.rle_to_mask(studio.mask_to_rle(mask)) == mask).all())
        self.assertFalse(studio.rle_to_mask(studio.mask_to_rle(np.zeros((5, 5), dtype=bool))).any())
        self.assertTrue(studio.rle_to_mask(studio.mask_to_rle(np.ones((5, 5), dtype=bool))).all())

    def test_matches_pycocotools_when_available(self):
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            self.skipTest("pycocotools is not installed")
        mask = np.random.default_rng(2).random((60, 40)) > 0.7
        encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        self.assertEqual(studio.mask_to_rle(mask)["counts"], encoded["counts"].decode("ascii"))


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class OcclusionLabelTests(unittest.TestCase):
    @staticmethod
    def item(x, y, w, h, keep=False):
        return {"mask": np.ones((h, w), dtype=bool), "x": x, "y": y, "full": float(w * h), "keep": keep}

    def test_the_label_of_a_partly_hidden_person_comes_from_what_is_visible(self):
        far, near = self.item(10, 10, 20, 40), self.item(10, 30, 20, 40)
        result = studio.resolve_visibility([far, near], (100, 100), 0.3)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(far["visibleShare"], 0.5)
        self.assertEqual(studio.tight_box(far["visible"]), (0, 0, 20, 20))
        self.assertEqual(studio.tight_box(near["visible"]), (0, 0, 20, 40))

    def test_a_mostly_hidden_person_is_removed_and_the_ones_behind_are_counted_again(self):
        back, middle, front = self.item(0, 0, 20, 20), self.item(0, 2, 20, 20), self.item(0, 0, 20, 22)
        result = studio.resolve_visibility([back, middle, front], (60, 60), 0.3)
        self.assertEqual(result, [front])

    def test_removing_someone_uncovers_the_person_behind(self):
        back, middle, front = self.item(0, 0, 20, 20), self.item(0, 10, 20, 20), self.item(0, 16, 20, 14)
        # middle sees only 6 rows of 20 (0.3 stays); with a higher limit it goes, and back gets its rows again
        result = studio.resolve_visibility([back, middle, front], (60, 60), 0.35)
        self.assertFalse(any(item is middle for item in result))
        self.assertAlmostEqual(back["visibleShare"], 0.8)  # only the front person covers it now

    def test_people_beside_the_court_stay_whatever_covers_them(self):
        kept, front = self.item(0, 0, 20, 20, keep=True), self.item(0, 0, 20, 20)
        self.assertEqual(len(studio.resolve_visibility([kept, front], (40, 40), 0.3)), 2)


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class CropLabelTests(unittest.TestCase):
    def setUp(self):
        class Fake:
            config = {"cropMinVisible": 0.4}
            class_id = {"player": 1}
        self.context = Fake()
        self.instance = {"class": "player", "session": "s", "role": "player", "team": 0}

    def label(self, x, y, w, h):
        return {"id": "a", "instance": self.instance, "mask": np.ones((h, w), dtype=bool), "x": x, "y": y, "occluded": 0.0, "foot": [x + w / 2, y + h]}

    def run_crop(self, label, window=(100, 100, 200, 200), size=400):
        result, _ = studio.output_labels([label], list(window), size / window[2], (0, 0), size, self.context)
        return result

    def test_inside_is_scaled_and_not_truncated(self):
        result = self.run_crop(self.label(150, 150, 20, 40))
        self.assertEqual(result[0]["bbox"], [100.0, 100.0, 40.0, 80.0])
        self.assertFalse(result[0]["truncated"])

    def test_cut_by_the_edge_is_clipped_and_marked(self):
        result = self.run_crop(self.label(80, 150, 40, 40))  # half of it is outside
        self.assertEqual(result[0]["bbox"], [0.0, 100.0, 40.0, 80.0])
        self.assertTrue(result[0]["truncated"])

    def test_less_than_40_percent_visible_is_dropped(self):
        self.assertEqual(self.run_crop(self.label(70, 150, 40, 40)), [])  # 25 % inside

    def test_outside_is_dropped(self):
        self.assertEqual(self.run_crop(self.label(0, 0, 40, 40)), [])

    def test_the_segmentation_matches_the_box(self):
        mask = studio.rle_to_mask(self.run_crop(self.label(150, 150, 20, 40))[0]["segmentation"])
        self.assertEqual(studio.tight_box(mask), (100, 100, 140, 180))

    def test_padding_keeps_the_aspect_ratio(self):
        image = np.zeros((100, 400, 3), dtype=np.uint8)
        out, window, scale, padding = studio.framed(image, None, 200)
        self.assertEqual(out.shape, (200, 200, 3))
        self.assertEqual((scale, padding), (0.5, (0, 75)))


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class GeometryTests(unittest.TestCase):
    def test_size_map_recovers_the_height_and_ignores_outliers(self):
        rng = np.random.default_rng(4)
        points = np.column_stack((rng.uniform(0, WIDTH, 200), rng.uniform(100, HEIGHT, 200)))
        heights = person_height(points[:, 1]) + rng.normal(0, 1, 200)
        heights[:15] *= 0.4  # hidden or bent over
        model = studio.fit_field(points, heights, WIDTH, HEIGHT)
        for x, y in ((100, 150), (600, 300), (1100, 480)):
            self.assertAlmostEqual(studio.field_at(model, x, y), person_height(y), delta=2.0)

    def test_scale_from_the_size_map_is_the_ratio_of_heights(self):
        model = studio.fit_field(np.array([[x, y] for x in range(0, WIDTH, 100) for y in range(100, HEIGHT, 50)], dtype=float), np.array([person_height(y) for x in range(0, WIDTH, 100) for y in range(100, HEIGHT, 50)]), WIDTH, HEIGHT)
        geometry = studio.Geometry({"sizeMap": model, "tiltMap": {**model, "coefficients": [0.0] * 6}, "polygon": [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]})
        self.assertAlmostEqual(geometry.height_at(600, 400) / geometry.height_at(600, 200), person_height(400) / person_height(200), places=2)

    def test_few_people_still_give_the_perspective(self):
        points = np.array([[300.0, 220.0], [900.0, 300.0], [500.0, 360.0], [700.0, 430.0], [200.0, 480.0]])
        model = studio.fit_size_field(points, person_height(points[:, 1]), WIDTH, HEIGHT)
        self.assertEqual(model["degree"], 1)
        self.assertAlmostEqual(studio.field_at(model, 600, 400), person_height(400), delta=1.0)

    def test_few_people_with_implausible_sizes_give_a_constant_size(self):
        # Larger further away cannot be perspective, and a fit through it must not explode.
        points = np.array([[300.0, 220.0], [900.0, 300.0], [500.0, 360.0], [700.0, 430.0]])
        model = studio.fit_size_field(points, np.array([120.0, 100.0, 90.0, 70.0]), WIDTH, HEIGHT)
        self.assertEqual(model["degree"], 0)
        self.assertEqual(studio.field_at(model, 0, 0), studio.field_at(model, WIDTH, HEIGHT))

    def test_a_field_stays_in_the_range_of_its_samples(self):
        rng = np.random.default_rng(6)
        points = np.column_stack((rng.uniform(500, 700, 14), rng.uniform(300, 320, 14)))
        model = studio.fit_field(points, rng.uniform(95, 105, 14), WIDTH, HEIGHT)
        for x, y in ((0, 0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)):
            self.assertTrue(70 <= studio.field_at(model, x, y) <= 130)

    def session_geometry(self, images: int, degree: int = 1) -> "studio.Geometry":
        size = {"degree": 1, "coefficients": [40.0, 0.0, 100.0], "width": WIDTH, "height": HEIGHT} if degree else {"degree": 0, "coefficients": [100.0], "width": WIDTH, "height": HEIGHT}
        tilt = {"degree": 0, "coefficients": [0.0], "width": WIDTH, "height": HEIGHT}
        return studio.Geometry({"sizeMap": size, "tiltMap": tilt, "polygon": [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], "polygonSource": "estimated", "images": ["x"] * images})

    def feet(self, geometry: "studio.Geometry", old: list[float]) -> list[dict]:
        context = types.SimpleNamespace(geometry={"s": geometry})
        rng = np.random.default_rng(8)
        spots = [studio.find_foot(context, geometry, {"session": "s", "foot": old}, [], rng) for _ in range(60)]
        return [spot for spot in spots if spot]

    def test_a_moving_camera_keeps_people_near_where_they_stood(self):
        old = [600.0, 300.0]
        geometry = self.session_geometry(images=1)
        height = geometry.height_at(*old)
        spots = self.feet(geometry, old)
        self.assertTrue(spots)
        for spot in spots:
            self.assertLessEqual(abs(spot["new"][0] - old[0]), 1.5 * height)
            self.assertLessEqual(abs(spot["new"][1] - old[1]), 0.35 * height)

    def test_without_a_perspective_people_keep_their_depth_and_size(self):
        old = [600.0, 300.0]
        spots = self.feet(self.session_geometry(images=1, degree=0), old)
        self.assertTrue(spots)
        for spot in spots:
            self.assertLessEqual(abs(spot["new"][1] - old[1]), 10.0)
            self.assertEqual(spot["scale"], 1.0)

    def test_a_fixed_camera_uses_the_whole_field_but_never_scales_far(self):
        spots = self.feet(self.session_geometry(images=20), [600.0, 250.0])
        self.assertTrue(spots)
        self.assertGreater(max(abs(spot["new"][0] - 600.0) for spot in spots), 300.0)
        for spot in spots:
            self.assertTrue(studio.MIN_SCALE <= spot["scale"] <= studio.MAX_SCALE)

    def test_homography_gives_more_pixels_per_metre_nearer_the_camera(self):
        # image y 0 is far: the same image distance covers more metres there
        homography = [[0.02, 0, 0], [0, 0.02, 0], [0, 0.002, 1]]
        geometry = studio.Geometry({"sizeMap": {}, "tiltMap": {}, "homography": homography, "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]})
        self.assertGreater(geometry.height_at(100, 400), geometry.height_at(100, 50))

    def test_the_transform_is_uniform_turns_towards_the_lean_and_keeps_the_foot(self):
        placement = {"old": [50.0, 100.0], "new": [300.0, 200.0], "scale": 1.5, "rotate": 10.0, "flip": False}
        matrix = studio.placement_matrix(placement, (40.0, 60.0))
        foot = matrix @ np.array([10.0, 40.0, 1.0])
        self.assertAlmostEqual(foot[0], 300.0)
        self.assertAlmostEqual(foot[1], 200.0)
        singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        self.assertAlmostEqual(singular[0], 1.5)
        self.assertAlmostEqual(singular[1], 1.5)  # never a distortion
        head = matrix @ np.array([10.0, 0.0, 1.0])
        self.assertGreater(head[0], foot[0])  # a positive change leans the head to the right
        self.assertLess(head[1], foot[1])
        self.assertAlmostEqual(studio.mask_tilt(np.eye(3, dtype=bool)), studio.mask_tilt(np.eye(3, dtype=bool)))

    def test_mirroring_keeps_scale_and_foot(self):
        placement = {"old": [50.0, 100.0], "new": [300.0, 200.0], "scale": 1.0, "rotate": 0.0, "flip": True}
        matrix = studio.placement_matrix(placement, (40.0, 60.0))
        self.assertAlmostEqual(float(np.linalg.det(matrix[:2, :2])), -1.0)
        self.assertAlmostEqual((matrix @ np.array([10.0, 40.0, 1.0]))[0], 300.0)
        self.assertAlmostEqual((matrix @ np.array([20.0, 40.0, 1.0]))[0], 290.0)

    def test_lean_is_measured_towards_the_head(self):
        mask = np.zeros((60, 60), dtype=bool)
        for row in range(60):
            mask[row, max(0, 30 - row // 2 + 15):max(0, 30 - row // 2 + 15) + 6] = True  # the head is to the right
        self.assertGreater(studio.mask_tilt(mask), 15.0)


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class SessionAndTeamTests(unittest.TestCase):
    def test_two_teams_and_an_outlier(self):
        red, blue = np.zeros(19), np.zeros(19)
        red[0], blue[10] = 1.0, 1.0
        other = np.zeros(19)
        other[5] = 1.0
        rng = np.random.default_rng(1)
        features = np.array([red + rng.normal(0, 0.01, 19) for _ in range(10)] + [blue + rng.normal(0, 0.01, 19) for _ in range(10)] + [other])
        teams, uncertain, _ = studio.cluster_teams(features)
        self.assertEqual(len(set(teams[:10])), 1)
        self.assertEqual(len(set(teams[10:20])), 1)
        self.assertNotEqual(teams[0], teams[10])
        self.assertEqual(teams[20], -1)
        self.assertTrue(uncertain[20])

    @staticmethod
    def jersey(bin_index: int, count: int) -> "np.ndarray":
        rng = np.random.default_rng(bin_index)
        features = np.zeros((count, 19))
        features[:, bin_index] = 1.0
        return np.clip(features + rng.normal(0, 0.02, features.shape), 0, None)

    def test_a_goalkeeper_does_not_become_a_team_and_merge_the_real_ones(self):
        features = np.vstack((self.jersey(10, 10), self.jersey(18, 10), self.jersey(5, 1), self.jersey(0, 1)))
        teams, _, _ = studio.cluster_teams(features)
        self.assertEqual(len(set(teams[:10])), 1)
        self.assertEqual(len(set(teams[10:20])), 1)
        self.assertNotEqual(teams[0], teams[10])
        self.assertTrue((teams[20:] == -1).all())

    def test_an_existing_split_is_kept(self):
        sessions = {"a": ["train/1.png"], "b": ["valid/2.png"], "c": ["test/3.png"]}
        self.assertEqual(studio.splits_from_folders(sessions), {"a": "train", "b": "valid", "c": "test"})
        self.assertEqual(studio.choose_split(sessions, studio.splits_from_folders(sessions))[0], {"a": "train", "b": "valid", "c": "test"})

    def test_folders_without_a_held_out_part_are_no_split(self):
        self.assertEqual(studio.splits_from_folders({"a": ["train/1.png"], "b": ["train/2.png"]}), {})
        self.assertEqual(studio.splits_from_folders({"a": ["game/1.png"], "b": ["valid/2.png"]}), {})

    def test_session_split_never_cuts_a_session_and_keeps_one_for_training(self):
        sessions = {f"s{number}": [f"s{number}/{index}.jpg" for index in range(10)] for number in range(6)}
        split, warning = studio.choose_split(sessions, {})
        self.assertIsNone(warning)
        self.assertEqual(set(split.values()), {"train", "valid", "test"})
        self.assertEqual(studio.choose_split({"a": ["1"]}, {})[1], "one-session")
        self.assertEqual(studio.choose_split(sessions, {"s0": "train", "s1": "test"})[0]["s1"], "test")

    def test_hints_ignore_split_folders_and_running_numbers(self):
        self.assertEqual(studio.session_hint("train/game_a_0012.jpg"), "game_a")
        self.assertEqual(studio.session_hint("match1/images/valid/cam-7.png"), "match1/cam")

    def test_linked_sessions_form_one_group(self):
        groups = studio.link_groups(["a", "b", "c"], [["a", "b"]])
        self.assertEqual(groups["a"], groups["b"])
        self.assertNotEqual(groups["a"], groups["c"])


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class ValidatorTests(unittest.TestCase):
    @staticmethod
    def annotation(number, role="player", team=0, foot=(10, 10), session="s", bbox=(1, 1, 5, 5), category=1):
        return {"instance_id": str(number), "role": role, "team_id": team, "foot": list(foot), "session_id": session, "bbox": list(bbox), "category_id": category}

    def check(self, annotations, group_of=lambda name: "g"):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        return studio.check_annotations(annotations, 100, {"players": 5, "referees": 3, "balls": 1}, {1}, lambda name: polygon, group_of)

    def test_valid_image(self):
        self.assertEqual(self.check([self.annotation(1), self.annotation(2, team=1)]), [])

    def test_a_third_team_is_rejected(self):
        self.assertIn("teams", self.check([self.annotation(1), self.annotation(2, team=1), self.annotation(3, team=2)]))

    def test_sessions_of_different_games_are_rejected(self):
        problems = self.check([self.annotation(1, session="a"), self.annotation(2, session="b")], group_of=lambda name: name)
        self.assertIn("sessions", problems)

    def test_limits_count_only_people_on_the_field(self):
        on_field = [self.annotation(number) for number in range(5)]
        self.assertEqual(self.check([*on_field, self.annotation(9, foot=(500, 500))]), [])
        self.assertIn("players", self.check([*on_field, self.annotation(9)]))

    def test_other_rules(self):
        self.assertIn("balls", self.check([self.annotation(1, role="ball"), self.annotation(2, role="ball")]))
        self.assertIn("duplicate", self.check([self.annotation(1), self.annotation(1)]))
        self.assertIn("bbox", self.check([self.annotation(1, bbox=(98, 1, 5, 5))]))
        self.assertIn("bbox", self.check([self.annotation(1, bbox=(1, 1, 0, 5))]))
        self.assertIn("class", self.check([self.annotation(1, category=7)]))


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class PlateTests(unittest.TestCase):
    """What stays in the empty hall must not look like an unlabelled person or a missing shoe."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.source = root / "source"
        folder = cls.source / "game"
        write_dataset(folder, frames=3, prefix="a", seed=9)
        document = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        # The first person has no usable mask, the second one a mask without shoes, and a logo is called a ball.
        cls.unmasked = document["annotations"][0]
        del cls.unmasked["segmentation"]
        cls.short = document["annotations"][1]
        short = studio.rle_to_mask(cls.short["segmentation"])
        x, y, w, h = cls.short["bbox"]
        short[int(y + 0.9 * h):] = False
        cls.short["segmentation"] = studio.mask_to_rle(short)
        picture = cv2.imread(str(folder / document["images"][1]["file_name"]))
        cv2.rectangle(picture, (900, 20), (1100, 140), (0, 255, 255), -1)
        cv2.imwrite(str(folder / document["images"][1]["file_name"]), picture, [cv2.IMWRITE_JPEG_QUALITY, 98])
        logo = np.zeros((HEIGHT, WIDTH), dtype=bool)
        logo[20:140, 900:1100] = True
        cls.logo = {"id": 9000, "image_id": 2, "category_id": 2, "bbox": [900, 20, 200, 120], "segmentation": studio.mask_to_rle(logo)}
        document["annotations"].append(cls.logo)
        (folder / "_annotations.coco.json").write_text(json.dumps(document), encoding="utf-8")
        poor = lambda: lambda image, box: (np.ones((8, 8), dtype=bool), (int(box[0]), int(box[1])), 0.1)
        cls.work = root / "work"
        cls.analysis = studio.analyze(cls.source, cls.work, {"variations": 4, "outputSize": 384, "seed": 1}, "en", make_masks=poor)
        cls.session = cls.analysis["sessions"][0]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def plate(self, number: int) -> np.ndarray:
        name = next(value for key, value in self.session["plate"]["files"].items() if key.endswith(f"a_{number:03d}.jpg"))
        return cv2.imread(str(self.work / "cache" / "plates" / name))

    def test_a_person_without_a_mask_leaves_the_plate(self):
        self.assertEqual(self.analysis["dropped"], 1)
        x, y, w, h = self.unmasked["bbox"]
        region = self.plate(0)[int(y + 0.4 * h):int(y + 0.6 * h), int(x + 0.3 * w):int(x + 0.7 * w)].astype(float)
        self.assertGreater(np.abs(region - np.array(TEAM_COLOURS[0], dtype=float)).sum(axis=2).mean(), 60)

    def test_too_little_disk_space_stops_with_a_clear_message(self):
        with self.assertRaisesRegex(studio.StudioError, "disk space"):
            studio.require_space(self.work, 10**18, "en")

    def test_shoes_outside_the_mask_leave_the_plate_and_the_person_stays_in_place(self):
        x, y, w, h = self.short["bbox"]
        plate = self.plate(0)[int(y + 0.92 * h):int(y + h), int(x):int(x + w)].astype(float)
        self.assertGreater(np.abs(plate - np.array((30, 30, 30), dtype=float)).sum(axis=2).mean(), 60)
        instance = next(item for item in self.analysis["instances"] if item["bbox"] == self.short["bbox"])
        self.assertFalse(instance["complete"])

    def test_a_ball_as_big_as_a_person_is_no_ball_and_stays_in_the_picture(self):
        self.assertTrue(any("ignored" in line for line in self.analysis["log"]))
        self.assertFalse(any(item["bbox"] == self.logo["bbox"] for item in self.analysis["instances"]))
        self.assertGreater(self.plate(1)[60:100, 950:1050, 1].mean(), 200)


@unittest.skipIf(studio is None, "needs numpy and OpenCV")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source"
        write_dataset(cls.source / "game-a", prefix="a", seed=3)
        write_dataset(cls.source / "game-b", frames=4, prefix="b", seed=4)
        write_dataset(cls.source / "game-c", frames=4, prefix="c", seed=5)
        cls.config = {"variations": 24, "outputSize": 384, "seed": 11, "splitOverrides": {}}
        cls.work = cls.root / "work"
        cls.analysis = studio.analyze(cls.source, cls.work, cls.config, "en")
        names = {session["id"].split("|")[0]: session["id"] for session in cls.analysis["sessions"]}
        cls.config["splitOverrides"] = {names["game-a/a"]: "train", names["game-b/b"]: "valid", names["game-c/c"]: "test"}
        cls.analysis = studio.analyze(cls.source, cls.work, cls.config, "en")
        cls.report = studio.generate(cls.work, cls.config, "en", processes=1)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_missing_annotations_stop_with_a_clear_message(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(studio.StudioError) as raised:
            studio.load_source(empty, "en")
        self.assertIn("COCO", str(raised.exception))

    def test_requested_images_and_no_violations(self):
        self.assertEqual(self.report["images"], 24)
        self.assertEqual(self.report["violations"], {})
        self.assertEqual(len(list((self.work / "output" / "images").glob("*.jpg"))), 24)
        coco = json.loads((self.work / "output" / "annotations.coco.json").read_text(encoding="utf-8"))
        self.assertEqual(len(coco["images"]), 24)
        for annotation in coco["annotations"]:
            self.assertTrue({"bbox", "segmentation", "category_id", "team_id", "occluded_ratio", "truncated"} <= set(annotation))

    def test_boxes_are_the_tight_box_of_the_mask(self):
        coco = json.loads((self.work / "output" / "annotations.coco.json").read_text(encoding="utf-8"))
        for annotation in coco["annotations"]:
            if annotation["role"] == "static":
                continue
            top_left = studio.tight_box(studio.rle_to_mask(annotation["segmentation"]))
            x, y, w, h = annotation["bbox"]
            self.assertEqual(top_left, (x, y, x + w, y + h))

    def test_held_out_sessions_feed_no_synthetic_image(self):
        allowed = {session["id"] for session in self.analysis["sessions"] if session["split"] == "train"}
        coco = json.loads((self.work / "output" / "annotations.coco.json").read_text(encoding="utf-8"))
        self.assertTrue({annotation["session_id"] for annotation in coco["annotations"]} <= allowed)
        split = json.loads((self.work / "output" / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(split["real"]["valid"]), 4)
        self.assertEqual(len(split["real"]["test"]), 4)
        self.assertFalse(set(split["real"]["train"]) & set(split["real"]["valid"]))

    def test_people_get_the_size_of_their_new_place(self):
        context = studio.Context(self.work, self.config)
        for plan in (studio.plan_image(context, index, 0) for index in range(10)):
            for placement in plan["placements"]:
                if placement.get("fixed") or placement.get("with"):
                    continue
                instance = context.instances[placement["id"]]
                if instance["role"] == "ball":
                    continue
                expected = person_height(placement["new"][1]) / person_height(placement["old"][1])
                self.assertAlmostEqual(placement["scale"], expected, delta=0.12)
                self.assertLessEqual(abs(placement["rotate"]), studio.MAX_TILT_CHANGE_DEG + 1e-6)

    def test_no_person_appears_twice_and_static_objects_stay(self):
        context = studio.Context(self.work, self.config)
        for index in range(10):
            plan = studio.plan_image(context, index, 0)
            ids = [placement["id"] for placement in plan["placements"]]
            self.assertEqual(len(ids), len(set(ids)))
            canvas, labels = studio.render_canvas(context, plan)
            for label in labels:
                if label["instance"]["role"] == "static":
                    self.assertEqual((label["x"], label["y"]), (100, 170))
            self.assertLessEqual(sum(label["instance"]["role"] == "ball" for label in labels), 1)

    def test_reproducible_bit_for_bit_also_in_parallel_and_after_a_cancel(self):
        first = tree_hash(self.work / "output")
        other = self.root / "again"
        shutil.copytree(self.work, other)
        shutil.rmtree(other / "output")
        studio.generate(other, self.config, "en", processes=2)
        self.assertEqual(tree_hash(other / "output"), first)
        # A cancel: only 7 images finished and the next line was cut off in the middle.
        for name, cut in (("manifest.jsonl", True), ("labels.jsonl", False)):
            path = other / "output" / name
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text("".join(lines[:7]) + (lines[7][:20] if cut else ""), encoding="utf-8")
        (other / "output" / "annotations.coco.json").unlink()
        studio.generate(other, self.config, "en", processes=1)
        self.assertEqual(tree_hash(other / "output"), first)

    def test_the_manifest_alone_reproduces_an_image(self):
        context = studio.Context(self.work, self.config)
        entry = json.loads((self.work / "output" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[5])
        data, _, _ = studio.render_plan(context, entry["plan"])
        self.assertEqual(data, (self.work / "output" / "images" / entry["file"]).read_bytes())

    def test_another_seed_gives_other_images(self):
        context = studio.Context(self.work, {**self.config, "seed": 12})
        self.assertNotEqual(studio.plan_image(context, 0, 0)["placements"], studio.Context(self.work, self.config) and studio.plan_image(studio.Context(self.work, self.config), 0, 0)["placements"])

    def test_jersey_variation_is_consistent_per_team_and_keeps_teams_apart(self):
        context = studio.Context(self.work, {**self.config, "jerseyVariation": True})
        plan = studio.plan_image(context, 3, 0)
        self.assertIsNotNone(plan["jersey"])
        session = context.sessions[plan["session"]]
        colours = [studio.bgr_to_lab(studio.shifted_colour(session["teamColours"][team], plan["jersey"]["shifts"][team])) for team in ("0", "1")]
        self.assertGreater(float(np.linalg.norm(colours[0] - colours[1])), studio.MIN_JERSEY_DELTA_E)

    def test_training_dataset_has_real_images_only_in_valid_and_test(self):
        target = self.root / "dataset"
        counts = studio.build_training_dataset(self.work, target, real_weight=3)
        self.assertEqual(counts, {"train": 16 * 3 + 24, "valid": 4, "test": 4})
        for split in ("valid", "test"):
            self.assertFalse(list((target / split).glob("syn-*")))
        self.assertEqual(studio.build_training_dataset(self.work, target, real_weight=1, use_synthetic=False)["train"], 16)
        self.assertFalse(list((target / "train").glob("syn-*")))


if __name__ == "__main__":
    unittest.main()
