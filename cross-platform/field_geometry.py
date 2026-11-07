"""Field outline per video (camera position). Standard library only.

project.json used to hold one outline for every frame ("fieldGeometry"). Since 0.29 each video
has its own outline in "fieldGeometries" ({video name: {"corners": [[x, y], ...]}}), because a
left and a right camera see the field in different places.

Old projects are migrated without losing anything: the old outline is copied to every video the
project has, and kept under "fieldGeometryBeforePerVideo". Readers that meet a project which was
not migrated yet (for example ml_worker on a project the app has not opened since) fall back to
the old outline, so the result is the same either way.
"""
from __future__ import annotations

import copy
from typing import Any

LEGACY_KEY = "fieldGeometry"
KEY = "fieldGeometries"
BACKUP_KEY = "fieldGeometryBeforePerVideo"


def video_key(frame: dict[str, Any]) -> str:
    """The video a frame comes from; imported images without a video share the key ""."""
    return str(frame.get("videoName") or "")


def project_videos(project: dict[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for frame in project.get("frames") or []:
        seen.setdefault(video_key(frame), None)
    return list(seen)


def geometries(project: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    value = (project or {}).get(KEY)
    return {str(video): geometry for video, geometry in value.items() if isinstance(geometry, dict)} if isinstance(value, dict) else {}


def geometry_for(project: dict[str, Any] | None, video: str) -> dict[str, Any] | None:
    """The outline that applies to frames of `video`, or None when it has none."""
    per_video = geometries(project)
    if video in per_video:
        return per_video[video]
    if KEY not in (project or {}):
        legacy = (project or {}).get(LEGACY_KEY)
        return legacy if isinstance(legacy, dict) else None
    return None


def needs_migration(project: dict[str, Any] | None) -> bool:
    """Only once the project has videos: without any, the old outline stays the default."""
    return bool(project) and isinstance(project.get(LEGACY_KEY), dict) and KEY not in project and bool(project_videos(project))


def migrate(project: dict[str, Any]) -> bool:
    """Turn the single outline into one per video. Returns True when `project` was changed."""
    if not needs_migration(project):
        return False
    legacy = project[LEGACY_KEY]
    project[KEY] = {video: copy.deepcopy(legacy) for video in project_videos(project)}
    project[BACKUP_KEY] = copy.deepcopy(legacy)
    del project[LEGACY_KEY]
    return True


def set_geometry(project: dict[str, Any], videos: list[str], geometry: dict[str, Any]) -> None:
    """Store `geometry` for each of `videos` (migrating an old single outline first)."""
    migrate(project)
    per_video = geometries(project)
    for video in videos:
        per_video[video] = copy.deepcopy(geometry)
    project[KEY] = per_video


def clear_geometry(project: dict[str, Any], videos: list[str] | None) -> None:
    """Remove the outline of `videos` (None: of every video and the default for new ones).
    Frames of a video without an outline are not filtered by auto-label."""
    migrate(project)
    if videos is None:
        project.pop(LEGACY_KEY, None)
        if KEY in project:
            project[KEY] = {}
        return
    per_video = geometries(project)
    for video in videos:
        per_video.pop(video, None)
    project[KEY] = per_video
