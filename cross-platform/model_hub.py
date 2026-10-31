"""Model Hub: exchange of fine-tuned models through a public GitHub repository.

Reads from the main repository only and writes into the user's fork only; a model reaches the
main repository through review. Weights travel as safetensors, which cannot run code when loaded,
and no picture ever goes along. Standard library only, torch is imported where weights are touched.
"""
from __future__ import annotations

import json
import os
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HUB_REPO = os.environ.get("RECO_HUB_REPO", "wendibus/reco-models")
HUB_INDEX_URL = os.environ.get("RECO_HUB_INDEX_URL", f"https://github.com/{HUB_REPO}/releases/download/index/index.json")
HUB_API = os.environ.get("RECO_HUB_API", "https://api.github.com")
HUB_LOGIN = os.environ.get("RECO_HUB_LOGIN", "https://github.com")
# The public id of the OAuth app. Device flow needs no secret.
HUB_CLIENT_ID = os.environ.get("RECO_HUB_CLIENT_ID", "")
HUB_ASSET_PREFIX = os.environ.get("RECO_HUB_ASSET_PREFIX", f"https://github.com/{HUB_REPO}/releases/download/")

HUB_SIZES = {"nano": 384, "small": 512, "medium": 576}
# Parameter counts are too close to tell the sizes apart; the position embedding of the backbone is not.
POSITION_TOKENS = {577: "nano", 1025: "small", 1297: "medium"}
POSITION_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"
CLASS_HEAD_KEY = "transformer.enc_out_class_embed.0.bias"
RIGHTS_STATEMENT_VERSION = 1
MAX_WEIGHTS_BYTES = 500_000_000
TENSOR_KEY = re.compile(r"^[A-Za-z0-9_.]+$")
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
TIMEOUT = 30


class HubError(RuntimeError):
    pass


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(version))[:3]) or (0,)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """The header of a safetensors file, checked without loading anything. Tensors have to fill the
    file exactly and carry plain names, so nothing else can travel along in it."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise HubError("not a safetensors file")
        length = struct.unpack("<Q", prefix)[0]
        if length > min(size - 8, 10_000_000):
            raise HubError("not a safetensors file")
        try:
            header = json.loads(handle.read(length))
        except ValueError as error:
            raise HubError("not a safetensors file") from error
    metadata = header.pop("__metadata__", {})
    if metadata not in ({}, {"format": "pt"}):
        raise HubError("the weights carry unexpected metadata")
    end = 0
    for name, tensor in header.items():
        if not TENSOR_KEY.match(name):
            raise HubError(f"unexpected tensor name: {name[:80]}")
        end = max(end, int(tensor["data_offsets"][1]))
    if 8 + length + end != size:
        raise HubError("the weights file holds more than its tensors")
    return header


def inspect_weights(path: Path) -> dict[str, Any]:
    """Size class and class count, read from the weights instead of taken from the uploader."""
    header = read_safetensors_header(path)
    try:
        tokens = header[POSITION_KEY]["shape"][1]
        head = header[CLASS_HEAD_KEY]["shape"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise HubError("these weights are not an RF-DETR detection model") from error
    size = POSITION_TOKENS.get(tokens)
    if size is None:
        raise HubError("only RF-DETR Nano, Small and Medium can be shared")
    return {"size": size, "classCount": head - 1}


def sanitize_checkpoint(checkpoint: Path, output: Path) -> list[str]:
    """Only the weights go out. A checkpoint also holds the optimizer state and the training
    arguments, and those contain local paths with the user name and folder names."""
    import torch
    from safetensors.torch import save_file

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, args = payload.get("model"), payload.get("args")
    classes = args.get("class_names") if isinstance(args, dict) else None
    if not state or not classes:
        raise HubError("this checkpoint has no weights or class names")
    save_file({name: tensor.clone() for name, tensor in state.items()}, str(output), metadata={"format": "pt"})
    return [str(name) for name in classes]


def write_minimal_checkpoint(weights: dict[str, Any], classes: list[str], output: Path) -> None:
    """A checkpoint RF-DETR can load and continue from, holding weights and class names only."""
    import torch

    torch.save({"model": weights, "args": {"class_names": classes, "num_classes": len(classes)}}, output)


def rebuild_checkpoint(safetensors_path: Path, classes: list[str], output: Path) -> None:
    from safetensors.torch import load_file

    write_minimal_checkpoint(load_file(str(safetensors_path)), classes, output)


def _request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None, method: str | None = None, timeout: int = TIMEOUT):
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers={"User-Agent": "reco-trainer", **(headers or {})}, method=method), timeout=timeout)


def fetch_index(cache: Path, url: str | None = None) -> dict[str, Any]:
    """The index of the hub, with the last copy when offline. offline tells the two apart."""
    url = url or HUB_INDEX_URL
    cached = None
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    headers = {"If-None-Match": cached["etag"]} if cached and cached.get("etag") else {}
    try:
        with _request(url, headers=headers) as response:
            index = json.loads(response.read())
            etag = response.headers.get("ETag")
    except urllib.error.HTTPError as error:
        if error.code == 304 and cached:
            return {**cached["index"], "offline": False}
        if error.code == 404 and not cached:
            return {"models": [], "offline": False, "notYetPublished": True}
        if cached:
            return {**cached["index"], "offline": True}
        return {"models": [], "offline": True}
    except (OSError, ValueError):
        if cached:
            return {**cached["index"], "offline": True}
        return {"models": [], "offline": True}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps({"etag": etag, "index": index}), encoding="utf-8")
    temporary.replace(cache)
    return {**index, "offline": False}


def filter_models(index: dict[str, Any], sport: str, classes: list[str], trainer_version: str) -> list[dict[str, Any]]:
    """Models for this sport and this trainer. classMatch says what happens at fine-tuning: with
    other classes the detection head starts over and only the backbone helps."""
    found = []
    for model in index.get("models", []):
        if model.get("sport") != sport or model.get("deprecated") or model.get("size") not in HUB_SIZES:
            continue
        if version_key(model.get("compat", {}).get("trainer_min", "0")) > version_key(trainer_version):
            continue
        same = not classes or set(model.get("classes", [])) == set(classes)
        found.append({**model, "classMatch": "same" if same else "different"})
    return sorted(found, key=lambda item: (list(HUB_SIZES).index(item["size"]), str(item.get("published_at", ""))), reverse=False)


def find_model(index: dict[str, Any], sport: str, size: str, model_id: str) -> dict[str, Any]:
    for model in index.get("models", []):
        if (model.get("sport"), model.get("size"), model.get("id")) == (sport, size, model_id):
            return model
    raise HubError("model not found in the hub")


def weights_entry(model: dict[str, Any]) -> dict[str, Any]:
    for entry in model.get("files", []):
        if entry.get("format") == "safetensors":
            return entry
    raise HubError("this model has no safetensors weights")


def download(url: str, destination: Path, sha256: str, progress: Callable[[float], None] | None = None) -> None:
    """Only files of the main repository, only with the hash from the index."""
    import hashlib

    if not url.startswith(HUB_ASSET_PREFIX):
        raise HubError("this download does not come from the hub")
    digest = hashlib.sha256()
    written = 0
    temporary = destination.with_name(destination.name + ".part")
    try:
        with _request(url, timeout=120) as response, temporary.open("wb") as sink:
            total = int(response.headers.get("Content-Length") or 0)
            if total > MAX_WEIGHTS_BYTES:
                raise HubError("the file is too large")
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_WEIGHTS_BYTES:
                    raise HubError("the file is too large")
                digest.update(chunk)
                sink.write(chunk)
                if progress and total:
                    progress(written / total)
        if digest.hexdigest() != sha256:
            raise HubError("the checksum of the download does not match")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def library_manifest(model: dict[str, Any], weights_file: str, weights_sha: str) -> dict[str, Any]:
    """Library entry for a downloaded model. The uploader's scores stay out of validationMetrics:
    what counts here is the measurement on the user's own images."""
    now = datetime.now(timezone.utc)
    return {
        "schemaVersion": 1,
        "packageID": f"hub-{model['sport']}-{model['size']}-{model['id']}-{model['version']}".replace("_", "-"),
        "displayName": f"{model['name']} {model['version']}",
        "createdAt": now.isoformat(),
        "sport": model["sport"],
        "modelSize": model["size"],
        "classes": sorted(model["classes"]),
        "description": str(model.get("description", ""))[:400],
        "source": "hub",
        "statistics": {"frameCount": int(model.get("training", {}).get("images") or 0)},
        "weights": {"file": weights_file, "sha256": weights_sha},
        "privacy": "This library entry contains model weights and aggregate metadata only; no videos or frames.",
        "license": model.get("license", "Apache-2.0"),
        "hub": {"lineage": lineage(model), "author": model.get("author"), "selfReported": model.get("metrics", {}).get("self_reported"), "training": model.get("training")},
    }


def lineage(model: dict[str, Any]) -> str:
    return f"{model['sport']}/{model['size']}/{model['id']}@{model['version']}"


def slug(name: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))).strip("-")[:60]


def check_public_text(text: str) -> None:
    """Names and descriptions become public: no paths and no mail addresses."""
    if re.search(r"(/Users/|/home/|[A-Za-z]:\\|/Volumes/|\S+@\S+\.\S+)", text):
        raise HubError("name and description must not contain paths or mail addresses")


def build_model_json(*, name: str, description: str, version: str, sport: str, size: str, classes: list[str], parent: str | None, author: str, training: dict[str, Any],
                     self_reported: dict[str, Any], trainer_version: str, sha256: str, size_bytes: int, source_url: str) -> dict[str, Any]:
    check_public_text(f"{name} {description}")
    model_id = slug(name)
    if not MODEL_ID.match(model_id):
        raise HubError("the name needs at least two letters or digits")
    return {
        "schema": 1, "id": model_id, "name": name, "version": version, "sport": sport, "size": size, "architecture": "rf-detr", "parent": parent, "author": author,
        "license": "Apache-2.0", "description": description, "classes": classes, "input_resolution": HUB_SIZES[size], "training": training, "checkpoint_sanitized": True,
        "rights": {"confirmed_by": author, "confirmed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "statement_version": RIGHTS_STATEMENT_VERSION},
        "metrics": {"self_reported": self_reported}, "compat": {"trainer_min": trainer_version},
        "files": [{"format": "safetensors", "source_url": source_url, "sha256": sha256, "size_bytes": size_bytes}],
    }


class GitHub:
    """The few GitHub calls sharing a model needs."""

    def __init__(self, token: str, api: str | None = None):
        self.token = token
        self.api = (api or HUB_API).rstrip("/")

    def call(self, method: str, path: str, body: Any = None, *, url: str | None = None, data: bytes | None = None, content_type: str = "application/json", ok_missing: bool = False) -> Any:
        payload = data if data is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if payload is not None:
            headers["Content-Type"] = content_type
        try:
            with _request(url or f"{self.api}{path}", payload, headers, method, timeout=600) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if ok_missing and error.code == 404:
                return None
            message = error.read().decode("utf-8", "replace")[:300]
            raise HubError(f"GitHub {method} {path or url}: {error.code} {message}") from error
        except OSError as error:
            raise HubError(f"GitHub {method} {path or url}: {error}") from error
        return json.loads(raw) if raw else None


def device_login(show_code: Callable[[str, str], None], client_id: str | None = None, login: str | None = None, wait: Callable[[float], None] = time.sleep) -> str:
    """Device flow: the user confirms a code in the browser. The token stays in this process and
    is gone when the upload is finished."""
    client_id = client_id or HUB_CLIENT_ID
    login = (login or HUB_LOGIN).rstrip("/")
    if not client_id:
        raise HubError("the hub login is not set up yet")

    def post(path: str, values: dict[str, str]) -> dict[str, Any]:
        with _request(f"{login}{path}", urllib.parse.urlencode(values).encode("utf-8"), {"Accept": "application/json"}, "POST") as response:
            return json.loads(response.read())

    start = post("/login/device/code", {"client_id": client_id, "scope": "public_repo"})
    if "device_code" not in start:
        raise HubError(f"GitHub login: {start.get('error_description') or start.get('error') or 'failed'}")
    show_code(start["user_code"], start["verification_uri"])
    interval = float(start.get("interval", 5))
    deadline = time.monotonic() + float(start.get("expires_in", 900))
    while time.monotonic() < deadline:
        wait(interval)
        result = post("/login/oauth/access_token", {"client_id": client_id, "device_code": start["device_code"], "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
        if result.get("access_token"):
            return str(result["access_token"])
        error = result.get("error")
        if error == "slow_down":
            interval = float(result.get("interval", interval + 5))
        elif error != "authorization_pending":
            raise HubError(f"GitHub login: {result.get('error_description') or error}")
    raise HubError("GitHub login: the code has expired")


def share(github: GitHub, weights: Path, model: dict[str, Any], status: Callable[[str], None], repo: str | None = None, wait: Callable[[float], None] = time.sleep) -> str:
    """Fork, release with the weights, model.json on a branch, pull request. If anything goes wrong
    or the job is cancelled, the release and the branch leave the fork again."""
    repo = repo or HUB_REPO
    owner, name = repo.split("/")
    author = model["author"]
    folder = f"sports/{model['sport']}/{model['size']}/{model['id']}"
    tag = f"{model['sport']}-{model['size']}-{model['id']}-v{model['version']}"
    branch = f"model/{tag}"
    status("fork")
    fork = github.call("POST", f"/repos/{repo}/forks", {})
    fork_repo = fork["full_name"]
    for _ in range(30):
        if github.call("GET", f"/repos/{fork_repo}/branches/main", ok_missing=True):
            break
        wait(2)
    else:
        raise HubError("the fork did not become ready")
    base = github.call("GET", f"/repos/{repo}/git/ref/heads/main")["object"]["sha"]
    if github.call("GET", f"/repos/{fork_repo}/git/ref/heads/{branch}", ok_missing=True):
        raise HubError(f"this version was already uploaded (branch {branch} exists in your fork)")
    release_id = None
    branch_created = False
    try:
        status("upload")
        release = github.call("POST", f"/repos/{fork_repo}/releases", {"tag_name": tag, "name": tag, "target_commitish": "main", "draft": False, "prerelease": True, "body": f"Weights for {model['name']} {model['version']}."})
        release_id = release["id"]
        upload = str(release["upload_url"]).split("{")[0] + f"?name={urllib.parse.quote(weights.name)}"
        asset = github.call("POST", "", url=upload, data=weights.read_bytes(), content_type="application/octet-stream")
        model["files"][0]["source_url"] = asset["browser_download_url"]
        status("pull-request")
        github.call("POST", f"/repos/{fork_repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": base})
        branch_created = True
        import base64
        content = base64.b64encode((json.dumps(model, ensure_ascii=False, indent=2) + "\n").encode("utf-8")).decode("ascii")
        path = f"{folder}/model.json"
        existing = github.call("GET", f"/repos/{fork_repo}/contents/{path}?ref={branch}", ok_missing=True)
        github.call("PUT", f"/repos/{fork_repo}/contents/{path}", {"message": f"Add {model['name']} {model['version']}", "content": content, "branch": branch, **({"sha": existing["sha"]} if existing else {})})
        pull = github.call("POST", f"/repos/{repo}/pulls", {
            "title": f"{model['sport']}/{model['size']}: {model['name']} {model['version']}", "head": f"{author}:{branch}", "base": "main",
            "body": f"{model['description']}\n\nUploaded with Reco Trainer. Classes: {', '.join(model['classes'])}.",
        })
        return str(pull["html_url"])
    except BaseException:
        if branch_created:
            _quietly(github, "DELETE", f"/repos/{fork_repo}/git/refs/heads/{branch}")
        if release_id is not None:
            _quietly(github, "DELETE", f"/repos/{fork_repo}/releases/{release_id}")
            _quietly(github, "DELETE", f"/repos/{fork_repo}/git/refs/tags/{tag}")
        raise


def _quietly(github: GitHub, method: str, path: str) -> None:
    try:
        github.call(method, path)
    except HubError:
        pass
