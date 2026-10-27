#!/usr/bin/env python3
"""Wrap an exported YOLO ONNX detector in a Reco Trainer ``.recomodel`` package.

The ONNX file must be exported with NMS built in (output 1 x N x 6: x1, y1, x2, y2,
confidence, class id), e.g. Ultralytics ``end2end``. Imported ONNX packages can pre-label
frames and be benchmarked in Reco Trainer; they cannot be trained or fine-tuned.

Example (class ids 0, 1, 2 = person, ball, referee):

    python3 tools/make_onnx_package.py best.onnx --name soccer --sport football \
        --class-names player,ball,referee --input-size 1920 \
        --license "AGPL-3.0 (Ultralytics YOLO26s fine-tune)" -o soccer.recomodel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SPORT_CLASSES = {
    "football": {"ball", "player", "goalkeeper", "referee", "goal"},
    "futsal": {"ball", "player", "goalkeeper", "referee", "goal"},
    "basketball": {"ball", "player", "referee", "hoop"},
    "handball": {"ball", "player", "goalkeeper", "referee", "goal"},
    "hockey": {"puck", "player", "goalkeeper", "referee", "goal"},
    "rugby": {"ball", "player", "referee", "goalpost"},
    "lacrosse": {"ball", "player", "goalkeeper", "referee", "goal"},
    "american_football": {"ball", "player", "referee", "goalpost"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", type=Path, help="exported ONNX detector with built-in NMS")
    parser.add_argument("--name", required=True, help="package ID and display name, e.g. soccer")
    parser.add_argument("--sport", default="football", choices=sorted(SPORT_CLASSES))
    parser.add_argument("--class-names", required=True, help="comma-separated Reco class names in model class-id order")
    parser.add_argument("--input-size", type=int, required=True, help="square model input size, e.g. 1920")
    parser.add_argument("--license", default="", help="license notice stored in the manifest")
    parser.add_argument("--description", default="")
    parser.add_argument("--model-size", default="small", choices=["nano", "small"],
                        help="placeholder size used by the model library (the ONNX model ignores it)")
    parser.add_argument("-o", "--output", type=Path, help="output path (default: <name>.recomodel)")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", args.name):
        sys.exit("--name may only contain letters, digits, '.', '_' and '-'")
    if not args.onnx.is_file() or args.onnx.suffix.lower() != ".onnx":
        sys.exit("the model file must be an existing .onnx file")
    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    unsupported = set(class_names) - SPORT_CLASSES[args.sport]
    if not class_names or unsupported:
        sys.exit(f"class names must be Reco classes for {args.sport}: {', '.join(sorted(SPORT_CLASSES[args.sport]))}"
                 + (f" (unsupported: {', '.join(sorted(unsupported))})" if unsupported else ""))
    if not 64 <= args.input_size <= 4096:
        sys.exit("--input-size must be between 64 and 4096")

    digest = hashlib.sha256()
    with args.onnx.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    description = args.description.strip() or (
        f"Imported {args.sport} object detector (ONNX, input {args.input_size} px) for "
        f"{', '.join(class_names)}. Pre-labeling and benchmarking only; it cannot be fine-tuned."
    )
    weight_name = f"weights/{args.name}.onnx"
    manifest = {
        "schemaVersion": 1,
        "packageID": args.name,
        "displayName": args.name,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sport": args.sport,
        "modelSize": args.model_size,
        "classes": sorted(set(class_names)),
        "classNames": class_names,
        "format": "onnx-yolo",
        "inputSize": args.input_size,
        "source": "external",
        "description": description[:500],
        "framework": {"name": "ONNX (YOLO, NMS built in)"},
        "weights": {"file": weight_name, "sha256": digest.hexdigest()},
        "privacy": "This package contains model weights and aggregate metadata only. It contains no videos, frames, source paths, or video file names.",
        "license": args.license or "Unknown - the publisher must confirm the license of the imported weights.",
    }
    output = args.output or Path(f"{args.name}.recomodel")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        archive.write(args.onnx, weight_name)
    print(output)


if __name__ == "__main__":
    main()
