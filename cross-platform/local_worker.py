#!/usr/bin/env python3
"""Loopback-only bridge between the Reco web UI and the local Mac ML worker.

The service binds exclusively to 127.0.0.1. It has no upload endpoint and no
outbound networking code. A folder can only be selected through the native
macOS folder dialog (or RECO_TEST_FOLDER during automated tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HOST = os.environ.get("RECO_BIND_HOST", "127.0.0.1")
DEFAULT_PORT = 8766
BASE_DIR = Path(__file__).resolve().parent
ALLOWED_ORIGINS = {"http://localhost:8765", "http://127.0.0.1:8765"}
configured_origin = os.environ.get("RECO_ALLOWED_ORIGIN")
if configured_origin and re.fullmatch(r"http://(?:localhost|127\.0\.0\.1):\d{2,5}", configured_origin):
    ALLOWED_ORIGINS.add(configured_origin)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
DEFAULT_FRAMES_PER_VIDEO = 240
MIN_FRAMES_PER_VIDEO = 4
MAX_FRAMES_PER_VIDEO = 5000
SPORT_CATEGORIES = {
    "football": ["ball", "player", "goalkeeper", "referee", "goal"],
    "futsal": ["ball", "player", "goalkeeper", "referee", "goal"],
    "basketball": ["ball", "player", "referee", "hoop"],
    "handball": ["ball", "player", "goalkeeper", "referee", "goal"],
    "hockey": ["puck", "player", "goalkeeper", "referee", "goal"],
    "rugby": ["ball", "player", "referee", "goalpost"],
    "lacrosse": ["ball", "player", "goalkeeper", "referee", "goal"],
    "american_football": ["ball", "player", "referee", "goalpost"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def backup_project(root: Path, reason: str) -> Path | None:
    """Keep small, local JSON snapshots before a project-changing operation."""
    source = root / "project.json"
    if not source.is_file():
        return None
    backups = root / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    target = backups / f"project-{stamp}-{safe_name(reason)}.json"
    shutil.copy2(source, target)
    for old in sorted(backups.glob("project-*.json"), reverse=True)[30:]:
        old.unlink(missing_ok=True)
    return target


def hardware_summary() -> dict:
    memory = 0
    try:
        memory = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    machine = platform.machine()
    return {
        "machine": machine,
        "cpuCores": os.cpu_count() or 1,
        "memoryGB": round(memory / (1024 ** 3)) if memory else None,
        "accelerator": "Apple GPU · Metal/MPS" if sys.platform == "darwin" and machine == "arm64" else "CPU",
    }


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in " ._-" else "_" for character in value).strip() or "video"


class LocalState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.selected_folder: Path | None = None
        self.project_root: Path | None = None
        self.project: dict | None = None
        self.operation = "idle"
        self.progress = 0.0
        self.message = "Lokaler Worker bereit."
        self.log: list[str] = []
        self.busy = False
        self.error: str | None = None

    def update(self, **values) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def append_log(self, line: str) -> None:
        with self.lock:
            clean = line.rstrip()
            if clean:
                self.log.append(clean)
                self.log = self.log[-250:]

    def snapshot(self) -> dict:
        with self.lock:
            frames = (self.project or {}).get("frames", [])
            training_frames = [frame for frame in frames if frame.get("reviewStatus") != "candidate"]
            candidates = [frame for frame in frames if frame.get("reviewStatus") == "candidate"]
            annotations = [annotation for frame in training_frames for annotation in frame.get("annotations", [])]
            classes = sorted({str(annotation.get("category")) for annotation in annotations})
            return {
                "connected": True,
                "platform": platform.system(),
                "selectedFolder": str(self.selected_folder) if self.selected_folder else None,
                "folderName": self.selected_folder.name if self.selected_folder else None,
                "projectRoot": str(self.project_root) if self.project_root else None,
                "operation": self.operation,
                "progress": self.progress,
                "message": self.message,
                "busy": self.busy,
                "error": self.error,
                "sport": (self.project or {}).get("sport"),
                "framesPerVideo": (self.project or {}).get("framesPerVideo", DEFAULT_FRAMES_PER_VIDEO),
                "stats": {
                    "frameCount": len(training_frames),
                    "candidateCount": len(candidates),
                    "annotatedFrames": sum(bool(frame.get("annotations")) for frame in training_frames),
                    "annotationCount": len(annotations),
                    "automaticCount": sum(annotation.get("source") == "auto" for annotation in annotations),
                    "classes": classes,
                },
                "hardware": hardware_summary(),
                "lastTraining": (self.project or {}).get("lastTraining"),
                "trainingHistory": (self.project or {}).get("trainingHistory", []),
                "modelLibrary": model_library_snapshot(self.project_root),
                "benchmark": benchmark_snapshot(self.project_root),
                "frames": frames,
                "log": "\n".join(self.log[-80:]),
            }


STATE = LocalState()


def project_root_for(folder: Path) -> Path:
    root = folder / ".reco-training"
    # Dot-prefixed folders are hidden on macOS and Linux. Keep the stable
    # internal path, but expose the same project permanently under a visible
    # name. Windows Explorer already shows dot-prefixed directories and often
    # requires elevated privileges for symlinks, so it is intentionally skipped.
    if os.name != "nt":
        visible = folder / "Reco Training"
        if not os.path.lexists(visible):
            try:
                visible.symlink_to(Path(".reco-training"), target_is_directory=True)
            except OSError:
                pass
    return root


def load_project(root: Path) -> dict | None:
    path = root / "project.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_library_snapshot(root: Path | None) -> dict:
    if root is None:
        return {"packages": [], "activePackageID": None}
    packages = []
    library = root / "models" / "library"
    if library.is_dir():
        for manifest_path in sorted(library.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                packages.append({
                    "packageID": manifest.get("packageID"),
                    "displayName": manifest.get("displayName") or manifest.get("packageID"),
                    "createdAt": manifest.get("createdAt"),
                    "sport": manifest.get("sport"),
                    "modelSize": manifest.get("modelSize"),
                    "classes": manifest.get("classes", []),
                    "description": manifest.get("description"),
                    "source": manifest.get("source"),
                    "statistics": manifest.get("statistics", {}),
                    "validationMetrics": manifest.get("validationMetrics") or (manifest.get("trainingSummary") or {}).get("validationMetrics", {}),
                    "testMetrics": manifest.get("testMetrics") or (manifest.get("trainingSummary") or {}).get("testMetrics", {}),
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    active_id = None
    active_path = root / "models" / "active.json"
    if active_path.is_file():
        try:
            active_id = json.loads(active_path.read_text(encoding="utf-8")).get("packageID")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    def score_metrics(metrics: dict) -> float | None:
        for key, value in metrics.items():
            normalized = str(key).lower().replace("val/", "").replace("val_", "")
            normalized = normalized.replace("test/", "").replace("test_", "")
            if normalized in {"map_50_95", "map50_95", "ap/ball", "ap_ball"}:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None
    def score(item: dict) -> float | None:
        test_score = score_metrics(item.get("testMetrics") or {})
        return test_score if test_score is not None else score_metrics(item.get("validationMetrics") or {})
    scored = [(score(item), item.get("packageID")) for item in packages]
    scored = [item for item in scored if item[0] is not None]
    best_id = max(scored, default=(None, None), key=lambda item: item[0])[1]
    for item in packages:
        item["isActive"] = item.get("packageID") == active_id
        item["isBest"] = item.get("packageID") == best_id
        item["validationScore"] = score_metrics(item.get("validationMetrics") or {})
        item["testScore"] = score_metrics(item.get("testMetrics") or {})
    packages.sort(key=lambda item: (not item["isActive"], not item["isBest"], str(item.get("createdAt") or "")))
    return {"packages": packages, "activePackageID": active_id, "bestPackageID": best_id}


def benchmark_snapshot(root: Path | None) -> dict:
    if root is None:
        return {"groundTruth": None, "latest": None}
    benchmark_root = root / "benchmarks"
    ground_truth = None
    latest = None
    try:
        document = json.loads((benchmark_root / "ground-truth.json").read_text(encoding="utf-8"))
        ground_truth = {
            "createdAt": document.get("createdAt"),
            "datasetID": document.get("datasetID"),
            "sport": document.get("sport"),
            "frameCount": len(document.get("frames", [])),
            "annotationCount": sum(len(frame.get("annotations", [])) for frame in document.get("frames", [])),
            "classes": sorted({annotation.get("category") for frame in document.get("frames", []) for annotation in frame.get("annotations", []) if annotation.get("category")}),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    try:
        latest = json.loads((benchmark_root / "latest.json").read_text(encoding="utf-8"))
        if isinstance(latest, dict):
            latest["matchesGroundTruth"] = bool(ground_truth and latest.get("datasetID") == ground_truth.get("datasetID"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        latest = None
    return {"groundTruth": ground_truth, "latest": latest}


def freeze_ground_truth() -> dict:
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError("Kein lokales Projekt geöffnet.")
    frames = [frame for frame in project.get("frames", []) if frame.get("reviewStatus") != "candidate"]
    if not frames:
        raise RuntimeError("Das Projekt enthält keine Testbilder.")
    pending = sum(annotation.get("source") == "auto" for frame in frames for annotation in frame.get("annotations", []))
    if pending:
        raise RuntimeError("Vor dem Modelltest alle automatischen Vorschläge übernehmen, korrigieren oder verwerfen.")
    allowed = set(SPORT_CATEGORIES.get(project.get("sport"), []))
    reference_frames = []
    for frame in frames:
        annotations = []
        for annotation in frame.get("annotations", []):
            category = str(annotation.get("category", ""))
            if category not in allowed:
                continue
            annotations.append({
                "category": category,
                "x": max(0.0, float(annotation.get("x", 0))),
                "y": max(0.0, float(annotation.get("y", 0))),
                "width": max(1.0, float(annotation.get("width", 1))),
                "height": max(1.0, float(annotation.get("height", 1))),
            })
        reference_frames.append({
            "id": str(frame.get("id")),
            "relativePath": str(frame.get("relativePath")),
            "width": int(frame.get("width", 0)),
            "height": int(frame.get("height", 0)),
            "annotations": annotations,
        })
    annotation_count = sum(len(frame["annotations"]) for frame in reference_frames)
    if annotation_count == 0:
        raise RuntimeError("Mindestens eine richtige Box muss vor dem Modelltest festgelegt sein.")
    canonical = json.dumps({"sport": project.get("sport"), "frames": reference_frames}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    reference = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "sport": project.get("sport"),
        "datasetID": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "frames": reference_frames,
        "reviewStatement": "Every frame was explicitly frozen as ground truth; frames without boxes are intentional negatives.",
    }
    atomic_json(root / "benchmarks" / "ground-truth.json", reference)
    STATE.update(message=f"Referenz mit {len(reference_frames)} Bildern und {annotation_count} Boxen festgelegt.")
    return benchmark_snapshot(root)


def choose_video_folder(*, expansion: bool = False) -> Path:
    configured_folder = (
        os.environ.get("RECO_EXPANSION_FOLDER") or os.environ.get("RECO_TEST_EXPANSION_FOLDER")
        if expansion else
        os.environ.get("RECO_VIDEO_FOLDER") or os.environ.get("RECO_TEST_FOLDER")
    )
    prompt = "Ordner mit neuen, noch nicht verwendeten Sportvideos auswählen" if expansion else "Ordner mit Sportvideos auswählen"
    if configured_folder:
        selected = Path(configured_folder).expanduser().resolve()
    elif sys.platform == "darwin":
        script = f'POSIX path of (choose folder with prompt "{prompt}")'
        result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("Ordnerauswahl wurde abgebrochen.")
        selected = Path(result.stdout.strip()).resolve()
    elif sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            raise RuntimeError("PowerShell wurde für die native Ordnerauswahl nicht gefunden.")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = 'Select folder containing sports videos'; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.SelectedPath }"
        )
        result = subprocess.run([powershell, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("Ordnerauswahl wurde abgebrochen.")
        selected = Path(result.stdout.strip()).resolve()
    else:
        dialog = shutil.which("zenity")
        command = [dialog, "--file-selection", "--directory", "--title=Select folder containing sports videos"] if dialog else None
        if command is None and shutil.which("kdialog"):
            command = [shutil.which("kdialog"), "--getexistingdirectory", str(Path.home())]
        if command is None:
            raise RuntimeError("Für die Ordnerauswahl bitte Zenity oder KDialog installieren oder RECO_VIDEO_FOLDER setzen.")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("Ordnerauswahl wurde abgebrochen.")
        selected = Path(result.stdout.strip()).resolve()
    if not selected.is_dir():
        raise RuntimeError("Der ausgewählte Ordner ist nicht verfügbar.")
    return selected


def select_folder() -> Path:
    selected = choose_video_folder()
    root = project_root_for(selected)
    project = load_project(root)
    STATE.update(
        selected_folder=selected,
        project_root=root,
        project=project,
        operation="selected",
        progress=0.0,
        message=(f"Vorhandenes Projekt mit {len(project.get('frames', []))} Frames geöffnet." if project else "Ordner gewählt. Videos können jetzt lokal vorbereitet werden."),
        error=None,
    )
    return selected


def select_model_package() -> Path:
    configured = os.environ.get("RECO_MODEL_PACKAGE") or os.environ.get("RECO_TEST_MODEL_PACKAGE")
    if configured:
        selected = Path(configured).expanduser().resolve()
    elif sys.platform == "darwin":
        script = 'POSIX path of (choose file with prompt "Reco-Modellpaket auswählen")'
        result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("Modellauswahl wurde abgebrochen.")
        selected = Path(result.stdout.strip()).resolve()
    elif sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            raise RuntimeError("PowerShell wurde für die Modellauswahl nicht gefunden.")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
            "$dialog.Filter = 'Reco model package (*.recomodel)|*.recomodel'; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileName }"
        )
        result = subprocess.run([powershell, "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("Modellauswahl wurde abgebrochen.")
        selected = Path(result.stdout.strip()).resolve()
    else:
        dialog = shutil.which("zenity")
        command = [dialog, "--file-selection", "--title=Select Reco model package", "--file-filter=Reco models | *.recomodel"] if dialog else None
        if command is None and shutil.which("kdialog"):
            command = [shutil.which("kdialog"), "--getopenfilename", str(Path.home()), "*.recomodel"]
        if command is None:
            inboxes = []
            if STATE.selected_folder:
                inboxes.extend(STATE.selected_folder.glob("*.recomodel"))
            if STATE.project_root:
                inboxes.extend((STATE.project_root / "inbox").glob("*.recomodel"))
            candidates = sorted({path.resolve() for path in inboxes if path.is_file()})
            if len(candidates) == 1:
                return candidates[0]
            raise RuntimeError("Für die Modellauswahl bitte genau ein .recomodel-Paket in den Videoordner oder .reco-training/inbox legen, Zenity/KDialog installieren oder RECO_MODEL_PACKAGE setzen.")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("Modellauswahl wurde abgebrochen.")
        selected = Path(result.stdout.strip()).resolve()
    if not selected.is_file() or selected.suffix.lower() != ".recomodel":
        raise RuntimeError("Bitte eine vorhandene .recomodel-Datei auswählen.")
    return selected


def discover_videos(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and ".reco-training" not in path.parts),
        key=lambda path: str(path).casefold(),
    )


def media_info(path: Path, apple_extractor: Path | None = None) -> tuple[float, int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        if apple_extractor is None:
            raise RuntimeError("Video-Metadaten können lokal nicht gelesen werden.")
        result = subprocess.run([str(apple_extractor), "--probe", str(path)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Video konnte nicht gelesen werden: {path.name}\n{result.stderr.strip()}")
        payload = json.loads(result.stdout)
        return float(payload["duration"]), int(payload["width"]), int(payload["height"])
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Video konnte nicht gelesen werden: {path.name}\n{result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if not streams or duration <= 0:
        raise RuntimeError(f"Video enthält keine lesbare Bildspur: {path.name}")
    return duration, int(streams[0].get("width") or 0), int(streams[0].get("height") or 0)


def native_frame_extractor(root: Path) -> Path:
    source = BASE_DIR / "frame_extractor.swift"
    if sys.platform != "darwin" or not source.is_file():
        raise RuntimeError("FFmpeg wurde nicht gefunden.")
    compiler = shutil.which("swiftc") or "/usr/bin/swiftc"
    if not Path(compiler).exists():
        raise RuntimeError("Weder FFmpeg noch die Xcode Command Line Tools wurden gefunden.")
    binary = root / ".runtime" / "bin" / "reco-frame-extractor"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
        module_cache = root / ".runtime" / "swift-module-cache"
        module_cache.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([compiler, "-module-cache-path", str(module_cache), str(source), "-o", str(binary)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Lokaler Apple-Frame-Extraktor konnte nicht erstellt werden.\n{result.stderr.strip()}")
    return binary


def frames_per_video(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = DEFAULT_FRAMES_PER_VIDEO
    return min(MAX_FRAMES_PER_VIDEO, max(MIN_FRAMES_PER_VIDEO, count))


def extract_frames_for_video(
    video: Path,
    duration: float,
    width: int,
    height: int,
    target_count: int,
    output_dir: Path,
    prefix: str,
    ffmpeg: str | None,
    apple_extractor: Path | None,
    failure_message: str,
) -> tuple[list[Path], float, int, int]:
    """Extract up to target_count evenly spaced frames from one video.

    Shared by initial project preparation and active-learning dataset expansion,
    which previously built the same ffmpeg/Apple-extractor command and the same
    even-height output scaling twice with only cosmetic differences.

    Returns the generated frame paths (sorted), the sampling rate actually used
    (frames per second of source video, needed by callers to compute each
    frame's timestamp), and the scaled output width/height (max 1280px wide,
    height rounded up to an even number since some encoders reject odd heights).
    """
    rate = max(target_count / duration, 1 / max(duration, 1.0))
    pattern = output_dir / f"{prefix}-%06d.jpg"
    command = (
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"fps={rate:.8f},scale='min(iw,1280)':-2",
         "-frames:v", str(target_count), "-q:v", "2", str(pattern)]
        if ffmpeg else
        [str(apple_extractor), str(video), str(output_dir), prefix, str(target_count)]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{failure_message}: {video.name}\n{result.stderr.strip()}")
    generated = sorted(output_dir.glob(f"{prefix}-*.jpg"))
    scale = min(1.0, 1280 / max(width, 1))
    output_width = max(1, round(width * scale))
    output_height = max(1, round(height * scale))
    if output_height % 2:
        output_height += 1
    return generated, rate, output_width, output_height


def extract_project(sport: str, requested_frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO) -> None:
    try:
        folder = STATE.selected_folder
        root = STATE.project_root
        if folder is None or root is None:
            raise RuntimeError("Zuerst einen Videoordner auswählen.")
        if sport not in SPORT_CATEGORIES:
            raise RuntimeError("Unbekannte Sportart.")
        count_per_video = frames_per_video(requested_frames_per_video)
        existing_project = load_project(root)
        existing_annotations = {
            str(frame.get("id")): [
                annotation for annotation in frame.get("annotations", [])
                if annotation.get("category") in SPORT_CATEGORIES[sport]
            ]
            for frame in (existing_project or {}).get("frames", [])
        }
        backup_project(root, "vorbereitung")
        ffmpeg = shutil.which("ffmpeg")
        apple_extractor = native_frame_extractor(root) if not ffmpeg or not shutil.which("ffprobe") else None
        STATE.update(operation="scanning", busy=True, progress=0.01, message="Suche lokale Videos …", error=None, log=[])
        videos = discover_videos(folder)
        if not videos:
            raise RuntimeError("Im gewählten Ordner wurden keine MP4-, MOV- oder M4V-Videos gefunden.")
        infos = [(video, *media_info(video, apple_extractor)) for video in videos]
        total_duration = max(sum(item[1] for item in infos), 1.0)
        temporary_frames = root / f"frames.next-{uuid.uuid4().hex[:8]}"
        temporary_frames.mkdir(parents=True, exist_ok=False)
        frames: list[dict] = []
        completed = 0.0
        try:
            for video, duration, width, height in infos:
                target = count_per_video
                video_id = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:12]
                STATE.update(operation="extracting", message=f"Extrahiere Trainingsbilder: {video.name}")
                generated, rate, output_width, output_height = extract_frames_for_video(
                    video, duration, width, height, target, temporary_frames, video_id,
                    ffmpeg, apple_extractor, "Frame-Extraktion fehlgeschlagen",
                )
                for index, image_path in enumerate(generated, start=1):
                    frame_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{index}"))
                    frames.append({
                        "id": frame_id,
                        "relativePath": f"frames/{image_path.name}",
                        "videoID": video_id,
                        "videoName": video.name,
                        "timestamp": min((index - 1) / rate, duration),
                        "width": output_width,
                        "height": output_height,
                        "annotations": existing_annotations.get(frame_id, []),
                    })
                completed += duration
                STATE.update(progress=min(completed / total_duration, 0.98))
            root.mkdir(parents=True, exist_ok=True)
            frames_dir = root / "frames"
            previous = root / "frames.previous"
            if previous.exists(): shutil.rmtree(previous)
            if frames_dir.exists(): frames_dir.replace(previous)
            temporary_frames.replace(frames_dir)
            if previous.exists(): shutil.rmtree(previous)
            for directory in ("dataset", "runs", "exports"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            now = utc_now()
            project = {
                "schemaVersion": 2,
                "name": folder.name,
                "sport": sport,
                "sourceFolder": str(folder),
                "createdAt": (existing_project or {}).get("createdAt", now),
                "updatedAt": now,
                "framesPerVideo": count_per_video,
                "frames": frames,
            }
            if existing_project and existing_project.get("lastTraining"):
                project["lastTraining"] = existing_project["lastTraining"]
            if existing_project and existing_project.get("trainingHistory"):
                project["trainingHistory"] = existing_project["trainingHistory"]
            atomic_json(root / "project.json", project)
            STATE.update(project=project, operation="ready", busy=False, progress=1.0, message=f"{len(project['frames'])} echte Frames lokal erstellt.")
        except Exception:
            if temporary_frames.exists(): shutil.rmtree(temporary_frames)
            raise
    except Exception as error:
        STATE.update(operation="error", busy=False, error=str(error), message="Lokale Vorbereitung fehlgeschlagen.")
        STATE.append_log(str(error))


def expand_dataset(folder: Path, payload: dict) -> None:
    """Extract a review inbox from separate videos, then pre-label only that inbox."""
    try:
        root = STATE.project_root
        project = STATE.project
        if root is None or project is None:
            raise RuntimeError("Zuerst ein bestehendes Trainingsprojekt öffnen.")
        if STATE.selected_folder and folder.resolve() == STATE.selected_folder.resolve():
            raise RuntimeError("Für die Erweiterung bitte einen anderen Videoordner wählen.")
        count_per_video = min(500, frames_per_video(payload.get("framesPerVideo", 100)))
        backup_project(root, "active-learning")
        ffmpeg = shutil.which("ffmpeg")
        apple_extractor = native_frame_extractor(root) if not ffmpeg or not shutil.which("ffprobe") else None
        videos = discover_videos(folder)
        if not videos:
            raise RuntimeError("Im gewählten Erweiterungsordner wurden keine Videos gefunden.")
        infos = [(video, *media_info(video, apple_extractor)) for video in videos]
        total_duration = max(sum(item[1] for item in infos), 1.0)
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        new_frames: list[dict] = []
        known_video_ids = {str(frame.get("videoID")) for frame in project.get("frames", [])}
        completed = 0.0
        STATE.update(operation="active-learning-extract", busy=True, progress=0.01, error=None, log=[], message="Extrahiere lokale Prüfkandidaten …")
        for video, duration, width, height in infos:
            target = count_per_video
            video_id = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:12]
            if video_id in known_video_ids:
                completed += duration
                STATE.update(progress=min(completed / total_duration * 0.45, 0.45), message=f"Bereits verwendetes Video übersprungen: {video.name}")
                continue
            prefix = f"al-{video_id}"
            generated, rate, output_width, output_height = extract_frames_for_video(
                video, duration, width, height, target, frames_dir, prefix,
                ffmpeg, apple_extractor, "Kandidaten-Extraktion fehlgeschlagen",
            )
            for index, image_path in enumerate(generated, start=1):
                new_frames.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"active-learning:{video_id}:{index}")),
                    "relativePath": f"frames/{image_path.name}", "videoID": video_id,
                    "videoName": video.name, "timestamp": min((index - 1) / rate, duration),
                    "width": output_width, "height": output_height, "annotations": [],
                    "reviewStatus": "candidate",
                })
            completed += duration
            STATE.update(progress=min(completed / total_duration * 0.45, 0.45), message=f"Neue Videos werden geprüft: {video.name}")

        if not new_frames:
            raise RuntimeError("Alle gewählten Videos wurden bereits verwendet. Bitte neue Videos auswählen.")
        project["frames"] = [*project.get("frames", []), *new_frames]
        project["updatedAt"] = utc_now()
        atomic_json(root / "project.json", project)
        STATE.update(project=project, busy=False, progress=0.5, message=f"{len(new_frames)} Kandidaten extrahiert. Lokale Ballerkennung startet …")
        ml_action("autolabel", {**payload, "candidateOnly": True, "threshold": payload.get("threshold", 0.12)})
    except Exception as error:
        STATE.append_log(str(error))
        STATE.update(operation="error", busy=False, error=str(error), message="Datensatzerweiterung fehlgeschlagen.")


def find_ml_worker() -> Path:
    candidates = [
        BASE_DIR / "ml_worker.py",
        BASE_DIR.parent / "reco-trainer-mac" / "Sources" / "RecoTrainerMac" / "Resources" / "ml_worker.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Der lokale RF-DETR-Worker wurde nicht gefunden.")


def system_python() -> Path:
    candidates = [
        Path("/opt/homebrew/bin/python3.12"), Path("/usr/local/bin/python3.12"),
        Path("/opt/homebrew/bin/python3.11"), Path("/usr/local/bin/python3.11"), Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("Python 3.11 oder 3.12 wurde nicht gefunden.")


def run_logged(command: list[str]) -> None:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    assert process.stdout is not None
    for line in process.stdout:
        STATE.append_log(line)
        clean = line.strip()
        match = re.search(r"(?:^|\s)(\d+)\s*/\s*(\d+)(?:\s|$)", clean)
        values = {"message": clean or STATE.message}
        if match and int(match.group(2)) > 0:
            values["progress"] = min(0.99, int(match.group(1)) / int(match.group(2)))
        STATE.update(**values)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Lokaler ML-Worker wurde mit Code {code} beendet.")


def ml_action(action: str, payload: dict) -> None:
    try:
        root = STATE.project_root
        if root is None or not (root / "project.json").is_file():
            raise RuntimeError("Zuerst Videos lokal vorbereiten.")
        language = payload.get("language", "de")
        if language not in {"de", "en", "es", "fr"}:
            language = "de"
        model = payload.get("model", "nano")
        STATE.update(operation=action, busy=True, progress=0.0, error=None, log=[], message="Lokaler ML-Vorgang startet …")
        venv = root / ".runtime" / "venv"
        venv_python = venv / "bin" / "python3"
        if action == "setup":
            if not venv_python.is_file():
                venv.parent.mkdir(parents=True, exist_ok=True)
                run_logged([str(system_python()), "-m", "venv", str(venv)])
            run_logged([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "rfdetr[train,onnx,coreml]>=1.9.0", "onnxruntime"])
        else:
            worker = find_ml_worker()
            executable = venv_python
            if action == "import-model":
                package_file = Path(str(payload.get("file", ""))).resolve()
                if not package_file.is_file() or package_file.suffix.lower() != ".recomodel":
                    raise RuntimeError("Das ausgewählte Modellpaket ist nicht verfügbar.")
                executable = system_python()
                args = ["install-package", "--project", str(root), "--file", str(package_file), "--language", language]
            elif action == "package-model":
                executable = system_python()
                args = ["package", "--project", str(root), "--model", model, "--name", str(payload.get("name", "")), "--language", language]
            elif action in {"activate-model", "rename-model", "delete-model"}:
                executable = system_python()
                package_id = str(payload.get("packageID", ""))
                args = [action, "--project", str(root), "--package-id", package_id, "--language", language]
                if action == "rename-model":
                    args.extend(["--name", str(payload.get("name", ""))])
            elif not venv_python.is_file():
                raise RuntimeError("ML-Umgebung fehlt. Zuerst „ML einrichten“ anklicken.")
            elif action == "autolabel":
                backup_project(root, "automatisch")
                threshold = min(0.95, max(0.05, float(payload.get("threshold", 0.25))))
                args = ["autolabel", "--project", str(root), "--model", model, "--category", payload.get("category", "ball"), "--threshold", str(threshold), "--language", language]
                if payload.get("candidateOnly"):
                    args.append("--candidate-only")
            elif action == "benchmark":
                threshold = min(0.95, max(0.01, float(payload.get("threshold", 0.05))))
                args = ["benchmark", "--project", str(root), "--threshold", str(threshold), "--language", language]
            elif action == "train":
                backup_project(root, "training")
                args = ["train", "--project", str(root), "--model", model, "--epochs", str(int(payload.get("epochs", 20))), "--language", language]
            elif action == "export-coreml":
                args = ["export", "--project", str(root), "--model", model, "--format", "coreml", "--language", language]
            elif action == "export-onnx":
                args = ["export", "--project", str(root), "--model", model, "--format", "onnx", "--language", language]
            else:
                raise RuntimeError("Unbekannter ML-Vorgang.")
            run_logged([str(executable), str(worker), *args])
        project = load_project(root)
        STATE.update(project=project, operation="ready", busy=False, progress=1.0, message="Lokaler Vorgang abgeschlossen.")
    except Exception as error:
        STATE.append_log(str(error))
        STATE.update(operation="error", busy=False, error=str(error), message="Lokaler ML-Vorgang fehlgeschlagen.")


def save_annotations(payload: dict) -> None:
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError("Kein lokales Projekt geöffnet.")
    frame_id = str(payload.get("frameId", ""))
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise RuntimeError("Ungültige Markierungen.")
    frame = next((item for item in project.get("frames", []) if item.get("id") == frame_id), None)
    if frame is None:
        raise RuntimeError("Frame wurde nicht gefunden.")
    backup_project(root, "markierung")
    allowed = set(SPORT_CATEGORIES.get(project.get("sport"), []))
    clean = []
    for annotation in annotations:
        category = str(annotation.get("category", ""))
        if category not in allowed:
            continue
        clean.append({
            "id": str(annotation.get("id") or uuid.uuid4()), "category": category,
            "x": max(0.0, float(annotation.get("x", 0))), "y": max(0.0, float(annotation.get("y", 0))),
            "width": max(1.0, float(annotation.get("width", 1))), "height": max(1.0, float(annotation.get("height", 1))),
            "confidence": annotation.get("confidence"), "source": str(annotation.get("source", "manual")),
        })
    frame["annotations"] = clean
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    STATE.update(project=project, message=f"{len(clean)} Markierungen lokal gespeichert.")


def remove_training_frame(payload: dict) -> None:
    """Remove one derived training image without ever touching a source video."""
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError("Kein lokales Projekt geöffnet.")
    frame_id = str(payload.get("frameId", ""))
    frames = project.get("frames", [])
    index = next((number for number, item in enumerate(frames) if str(item.get("id")) == frame_id), None)
    if index is None:
        raise RuntimeError("Frame wurde nicht gefunden.")
    frame = frames[index]
    is_candidate = frame.get("reviewStatus") == "candidate"
    target = (root / str(frame.get("relativePath", ""))).resolve()
    frames_root = (root / "frames").resolve()
    if frames_root not in target.parents:
        raise RuntimeError("Ungültiger Frame-Pfad.")

    backup_project(root, "bild-entfernt")
    target.unlink(missing_ok=True)
    for split in ("train", "valid", "test"):
        (root / "dataset" / split / target.name).unlink(missing_ok=True)
    frames.pop(index)
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)

    # A frozen benchmark no longer represents the current image set.
    benchmark_invalidated = False
    if not is_candidate:
        for name in ("ground-truth.json", "latest.json"):
            benchmark_file = root / "benchmarks" / name
            if benchmark_file.is_file():
                benchmark_file.unlink()
                benchmark_invalidated = True
    message = "Trainingsbild lokal entfernt. Das Quellvideo bleibt unverändert."
    if benchmark_invalidated:
        message += " Die Benchmark-Referenz muss neu festgelegt werden."
    STATE.update(project=project, message=message)


def review_candidate(payload: dict) -> None:
    root = STATE.project_root
    project = STATE.project
    if root is None or project is None:
        raise RuntimeError("Kein lokales Projekt geöffnet.")
    frame_id = str(payload.get("frameId", ""))
    decision = str(payload.get("decision", ""))
    frame = next((item for item in project.get("frames", []) if str(item.get("id")) == frame_id), None)
    if frame is None or frame.get("reviewStatus") != "candidate":
        raise RuntimeError("Prüfkandidat wurde nicht gefunden.")
    target = str(payload.get("category") or "ball")
    if decision == "ball":
        matching = [item for item in frame.get("annotations", []) if item.get("category") == target]
        if not matching:
            raise RuntimeError("Zuerst eine passende Ball-Box einzeichnen oder korrigieren.")
        for annotation in matching:
            annotation["source"] = "manual"
    elif decision == "no-ball":
        frame["annotations"] = [item for item in frame.get("annotations", []) if item.get("category") != target]
    else:
        raise RuntimeError("Unbekannte Prüfentscheidung.")
    backup_project(root, "kandidat-geprueft")
    frame["reviewStatus"] = "reviewed"
    project["updatedAt"] = utc_now()
    atomic_json(root / "project.json", project)
    STATE.update(project=project, message="Geprüftes Bild wurde in den Trainingssatz übernommen.")


class Handler(BaseHTTPRequestHandler):
    server_version = "RecoLocalWorker/0.1"

    def log_message(self, format: str, *args) -> None:
        return

    def allowed_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if origin in ALLOWED_ORIGINS else None

    def add_cors(self) -> None:
        origin = self.allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def json_response(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.add_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise RuntimeError("Anfrage ist zu groß.")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_OPTIONS(self) -> None:
        if not self.allowed_origin():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.add_cors()
        self.end_headers()

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/api/status":
            self.json_response(STATE.snapshot())
            return
        if route.path == "/api/frame":
            root = STATE.project_root
            project = STATE.project or {}
            frame_id = (parse_qs(route.query).get("id") or [""])[0]
            frame = next((item for item in project.get("frames", []) if item.get("id") == frame_id), None)
            if root is None or frame is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target = (root / frame["relativePath"]).resolve()
            frames_root = (root / "frames").resolve()
            if frames_root not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.add_cors()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self.allowed_origin():
            self.json_response({"error": "Nur die lokale Reco-Oberfläche darf diesen Worker aufrufen."}, HTTPStatus.FORBIDDEN)
            return
        try:
            payload = self.read_json()
            if self.path == "/api/select-folder":
                folder = select_folder()
                self.json_response({"ok": True, "folder": str(folder), "status": STATE.snapshot()})
            elif self.path == "/api/prepare":
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                count = frames_per_video(payload.get("framesPerVideo"))
                threading.Thread(target=extract_project, args=(payload.get("sport", "basketball"), count), daemon=True).start()
                self.json_response({"ok": True})
            elif self.path == "/api/active-learning":
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                folder = choose_video_folder(expansion=True)
                threading.Thread(target=expand_dataset, args=(folder, payload), daemon=True).start()
                self.json_response({"ok": True, "folder": str(folder)})
            elif self.path == "/api/annotations":
                save_annotations(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/remove-frame":
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                remove_training_frame(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/review-candidate":
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                review_candidate(payload)
                self.json_response({"ok": True, "status": STATE.snapshot()})
            elif self.path == "/api/benchmark-ground-truth":
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                self.json_response({"ok": True, "benchmark": freeze_ground_truth()})
            elif self.path == "/api/import-model":
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                package_file = select_model_package()
                threading.Thread(target=ml_action, args=("import-model", {**payload, "file": str(package_file)}), daemon=True).start()
                self.json_response({"ok": True, "fileName": package_file.name})
            elif self.path in {"/api/setup", "/api/autolabel", "/api/train", "/api/export-coreml", "/api/export-onnx", "/api/package-model", "/api/benchmark", "/api/activate-model", "/api/rename-model", "/api/delete-model"}:
                if STATE.busy:
                    raise RuntimeError("Ein lokaler Vorgang läuft bereits.")
                action = self.path.removeprefix("/api/")
                threading.Thread(target=ml_action, args=(action, payload), daemon=True).start()
                self.json_response({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def python_version_warning() -> str | None:
    """Flag a Python other than 3.11/3.12 without touching platform launcher scripts.

    Video browsing/upload works on any Python 3, but "Set up ML" installs
    rfdetr[train,onnx,coreml], which is only validated against 3.11/3.12.
    """
    if sys.version_info[:2] in {(3, 11), (3, 12)}:
        return None
    return (
        f"Python {sys.version_info[0]}.{sys.version_info[1]} erkannt (empfohlen: 3.11 oder 3.12). "
        "Video-Vorschau und -Upload funktionieren, aber „ML einrichten“ kann fehlschlagen."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reco Trainer loopback worker")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if HOST not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        raise SystemExit("RECO_BIND_HOST muss 127.0.0.1, localhost oder 0.0.0.0 sein.")
    warning = python_version_warning()
    if warning:
        print(f"WARNUNG: {warning}", flush=True)
        STATE.update(message=warning)
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"Reco Local Worker: http://{HOST}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
