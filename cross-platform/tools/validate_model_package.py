#!/usr/bin/env python3
"""Offline validator for privacy-safe Reco Trainer model packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"INVALID: {message}")


def validate(package: Path) -> dict:
    if not package.is_file():
        fail("package does not exist")
    if package.stat().st_size > 4 * 1024**3:
        fail("package is too large")
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != 2:
            fail("package must contain exactly manifest.json and one weights file")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                fail(f"unsafe path: {name}")
            if name != "manifest.json" and not (name.startswith("weights/") and path.suffix in {".pth", ".ckpt"}):
                fail(f"disallowed content: {name}")
        if "manifest.json" not in names:
            fail("manifest.json is missing")
        if archive.getinfo("manifest.json").file_size > 1024 * 1024:
            fail("manifest is too large")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schemaVersion") != 1:
            fail("unsupported schema version")
        weight_name = manifest.get("weights", {}).get("file")
        if weight_name not in names:
            fail("weights file is missing")
        if archive.getinfo(weight_name).file_size > 4 * 1024**3:
            fail("weights file is too large")
        package_id = str(manifest.get("packageID", ""))
        if not package_id or len(package_id) > 120 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in package_id):
            fail("invalid package ID")
        if manifest.get("modelSize") not in {"nano", "small"}:
            fail("unsupported model size")
        description = manifest.get("description")
        if description is not None and (not isinstance(description, str) or not description.strip() or len(description) > 500):
            fail("invalid package description")
        checksum = hashlib.sha256()
        with archive.open(weight_name) as weights:
            for chunk in iter(lambda: weights.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != manifest.get("weights", {}).get("sha256"):
            fail("weights checksum mismatch")
        serialized = json.dumps(manifest).lower()
        for forbidden in ("sourcefolder", "videoname", ".mp4", ".mov", ".m4v", "relativepath"):
            if forbidden in serialized:
                fail(f"private media metadata found: {forbidden}")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    manifest = validate(args.package.expanduser().resolve())
    print("VALID")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
