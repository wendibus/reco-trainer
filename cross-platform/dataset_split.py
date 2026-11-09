"""Which images train, validate and test. Standard library only, shared by local_worker (the split
shown in the app) and ml_worker (the split the training uses), so both always agree.

Videos are split as a whole: neighboring frames of one video look almost the same, so frames of
one video in training and in validation would make the scores look better than they are.
Imported images carry no scene information and train, unless they come with a role of their own
("split"), which import-labels sets for datasets exported by Reco Trainer: those were split by
video already."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

ROLES = ("train", "valid", "test")


def _text(language: str, german: str, english: str, spanish: str, french: str) -> str:
    return {"de": german, "es": spanish, "fr": french}.get(language, english)


def trains_on(frame: dict[str, Any]) -> bool:
    """Whether a frame belongs in the training data. A frame without boxes only counts when someone
    said it is empty (button "Image is empty", or "no ball" in the candidate review, or an imported
    label file): an untouched frame would otherwise teach that its players and ball are background."""
    if frame.get("reviewStatus") == "candidate":
        return False
    return bool(frame.get("annotations")) or frame.get("reviewStatus") == "reviewed" or bool(frame.get("emptyConfirmed"))


def is_imported_frame(frame: dict[str, Any]) -> bool:
    return str(frame.get("videoID", "")).startswith("import-")


def stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def split_frames(frames: list[dict[str, Any]], language: str = "de") -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    pinned = [frame for frame in frames if is_imported_frame(frame) and frame.get("split") in ROLES]
    pinned_ids = {id(frame) for frame in pinned}
    rest = [frame for frame in frames if id(frame) not in pinned_ids]
    if rest:
        output, warning = _split_by_video(rest, language, has_pinned=bool(pinned))
    else:
        output, warning = {"train": [], "valid": [], "test": []}, None
    for frame in pinned:
        output[frame["split"]].append(frame)
    return output, warning


def _split_by_video(frames: list[dict[str, Any]], language: str, has_pinned: bool) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    imported = [frame for frame in frames if is_imported_frame(frame)]
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if not is_imported_frame(frame):
            by_video[frame["videoID"]].append(frame)
    video_ids = sorted(by_video, key=stable_bucket)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    warning = None

    if imported and not video_ids:
        if has_pinned:
            # The other imported images already brought validation and test images along.
            output["train"] = imported
            return output, None
        # File order would put whole dataset versions into one split: mix by a stable hash.
        ordered = sorted(imported, key=lambda frame: stable_bucket(str(frame["id"])))
        train_end = max(1, int(len(ordered) * 0.7))
        valid_end = max(train_end + 1, int(len(ordered) * 0.9))
        output["train"] = ordered[:train_end]
        output["valid"] = ordered[train_end:valid_end]
        output["test"] = ordered[valid_end:]
        warning = _text(
            language,
            "Nur importierte Bilder: Der Split ist gemischt, gleiche Szenen können in Training und Test liegen; Qualitätswerte sind nicht unabhängig.",
            "Imported images only: the split is mixed, the same scenes can be in training and test; quality metrics are not independent.",
            "Solo imágenes importadas: la división es mixta, las mismas escenas pueden estar en entrenamiento y prueba; las métricas no son independientes.",
            "Images importées uniquement : la répartition est mélangée, les mêmes scènes peuvent se trouver en entraînement et en test ; les métriques ne sont pas indépendantes.",
        )
        return output, warning
    output["train"].extend(imported)

    if len(video_ids) >= 3:
        test_ids = {video_ids[-1]}
        valid_ids = {video_ids[-2]}
        for video_id, items in by_video.items():
            split = "test" if video_id in test_ids else "valid" if video_id in valid_ids else "train"
            output[split].extend(items)
    elif len(video_ids) == 2:
        output["train"].extend(by_video[video_ids[0]])
        output["valid"].extend(by_video[video_ids[1]])
        warning = _text(
            language,
            "Nur zwei Videos: Es wird noch kein unabhängiger Test-Split erzeugt.",
            "Only two videos: no independent test split is created yet.",
            "Solo dos vídeos: todavía no se crea una división de prueba independiente.",
            "Seulement deux vidéos : aucune répartition de test indépendante n’est encore créée.",
        )
    else:
        ordered = sorted((frame for frame in frames if not is_imported_frame(frame)), key=lambda frame: frame["timestamp"])
        train_end = max(1, int(len(ordered) * 0.7))
        valid_end = max(train_end + 1, int(len(ordered) * 0.9))
        output["train"].extend(ordered[:train_end])
        output["valid"] = ordered[train_end:valid_end]
        output["test"] = ordered[valid_end:]
        warning = _text(
            language,
            "Nur ein Video: temporaler Split als Notlösung; Qualitätswerte sind nicht unabhängig.",
            "Only one video: a temporal fallback split is used; quality metrics are not independent.",
            "Solo un vídeo: se usa una división temporal de emergencia; las métricas no son independientes.",
            "Une seule vidéo : une répartition temporelle de secours est utilisée ; les métriques ne sont pas indépendantes.",
        )
    return output, warning


def preview(frames: list[dict[str, Any]], language: str = "de") -> dict[str, Any]:
    """For the app: the role of every image the next training uses, the counts, and what to fix.
    Images that do not train yet (unlabelled, candidates, independent test) get no role."""
    usable = [frame for frame in frames if trains_on(frame) and not frame.get("heldOut")]
    splits, warning = split_frames(usable, language)
    roles = {str(frame.get("id")): role for role, items in splits.items() for frame in items}
    counts = {role: len(items) for role, items in splits.items()}
    total = sum(counts.values())
    problems: list[str] = []
    if total and counts["valid"] < max(10, round(total * 0.05)):
        problems.append(_text(
            language,
            f"Nur {counts['valid']} Validierungsbilder: Die Bewertung und das automatische Aktivieren schwanken stark. Ein weiteres Video hilft.",
            f"Only {counts['valid']} validation images: the score and automatic activation will vary a lot. Another video helps.",
            f"Solo {counts['valid']} imágenes de validación: la puntuación y la activación automática variarán mucho. Otro vídeo ayuda.",
            f"Seulement {counts['valid']} images de validation : le score et l’activation automatique varieront beaucoup. Une autre vidéo aide.",
        ))
    if total and counts["test"] == 0:
        problems.append(_text(
            language,
            "Keine Testbilder: Für einen unabhängigen Test braucht es mindestens drei Videos (eins davon wird Test) oder einen eigenen Test unter „Unabhängiger Modelltest“.",
            "No test images: an independent test needs at least three videos (one of them becomes the test) or a separate one under “Independent model test”.",
            "Sin imágenes de prueba: una prueba independiente necesita al menos tres vídeos (uno de ellos será la prueba) o una propia en «Prueba de modelo independiente».",
            "Aucune image de test : un test indépendant nécessite au moins trois vidéos (l’une devient le test) ou un test séparé sous « Test de modèle indépendant ».",
        ))
    return {"roles": roles, "counts": counts, "warning": warning, "problems": problems}
