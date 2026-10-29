"""Local vision-language model (VLM) for the consistency check - run by the app itself.

Reco Trainer downloads a pinned llama.cpp `llama-server` build for the platform (MIT) and a
Qwen3.5 GGUF model (Apache-2.0), starts the server on 127.0.0.1 for the duration of a job and
stops it afterwards, which also frees the GPU memory. Nothing is sent anywhere else.

Standard library only, so the worker can use it without the ML environment.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# --- where large shared models live ------------------------------------------------------------

def shared_models_dir() -> Path | None:
    """One folder per computer for the multi-GB VLM and its runtime (not one copy per project).
    None inside Docker: there the container file system is thrown away, so the project's own
    .runtime folder on the mounted video volume is used instead."""
    configured = os.environ.get("RECO_SHARED_MODELS")
    if configured:
        return Path(configured).expanduser()
    if Path("/.dockerenv").exists():
        return None
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Reco Trainer" / "models"
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "Reco Trainer" / "models"
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")) / "reco-trainer" / "models"


# --- which llama-server build ------------------------------------------------------------------

def runtime_ids(system: str | None = None, machine: str | None = None, nvidia: bool = False) -> list[str]:
    """Registry ids of the llama-server build (plus its CUDA runtime) for this platform."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arm = machine in {"arm64", "aarch64"}
    if system == "darwin":
        if not arm:
            raise RuntimeError("The VLM needs a Mac with Apple silicon.")
        return ["llama-server-macos-arm64"]
    if system == "windows":
        return ["llama-server-windows-cuda", "llama-server-windows-cudart"] if nvidia and not arm else ["llama-server-windows-cpu"]
    if arm:
        return ["llama-server-linux-arm64"]
    return ["llama-server-linux-cuda", "llama-server-linux-cudart"] if nvidia else ["llama-server-linux-cpu"]


def find_server_binary(folder: Path) -> Path | None:
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    for path in sorted(folder.rglob(name)):
        if path.is_file():
            return path
    return None


# --- answers ---------------------------------------------------------------------------------

def parse_json_answer(text: str) -> Any:
    """The model's JSON, also when wrapped in ```json fences or surrounded by prose."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    for opening, closing in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(opening), cleaned.rfind(closing)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start:end + 1])
            except ValueError:
                continue
    raise ValueError(f"no JSON in the model's answer: {text[:200]!r}")


def to_pixels(box: list[float], width: float, height: float, coordinates: str) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) from the model -> (x, y, w, h) in image pixels.

    Qwen3.5 / Qwen3-VL answer in a 0-1000 grid ("relative-1000"); older Qwen2.5-VL answered in
    pixels of the image it saw ("absolute"). The format is stored per model in the registry."""
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    if coordinates == "relative-1000":
        x1, x2 = x1 / 1000 * width, x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height
    elif coordinates == "relative-1":
        x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    return x1, y1, x2 - x1, y2 - y1


def as_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


# --- prompts ---------------------------------------------------------------------------------

def frame_check_prompt(boxes: list[dict[str, Any]], width: int, height: int) -> str:
    listed = [{"label": box["category"], "box": [round(box["x"]), round(box["y"]), round(box["x"] + box["width"]), round(box["y"] + box["height"])]} for box in boxes]
    return (
        f"This is a frame from a sports video ({width}x{height} pixels). These objects were labelled "
        f"(x1, y1, x2, y2 in pixels): {json.dumps(listed)}. Look at the image carefully. "
        "Is any player, referee, goalkeeper or the ball visible on the playing field without a box? "
        "Do two boxes mark the same object? "
        'Answer only with JSON: {"missing": <number>, "duplicates": <number>, "comment": "<one short sentence>"}'
    )


ANONYMIZATION_PROMPT = (
    "This sports frame was anonymized: faces and heads should be blurred. "
    "Can you still clearly recognize any person's face, so that the person could be identified? "
    'Answer only with JSON: {"recognizable_faces": <number>, "comment": "<one short sentence>"}'
)


# --- server and client -----------------------------------------------------------------------

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class LlamaServer:
    """Starts llama-server on 127.0.0.1 for one job and stops it again (frees GPU memory)."""

    def __init__(self, binary: Path, model: Path, mmproj: Path, context: int = 8192, gpu_layers: int = 99) -> None:
        self.binary, self.model, self.mmproj = binary, model, mmproj
        self.context, self.gpu_layers = context, gpu_layers
        self.port = free_port()
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 300.0) -> "LlamaServer":
        command = [
            str(self.binary), "-m", str(self.model), "--mmproj", str(self.mmproj),
            "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.context), "-ngl", str(self.gpu_layers),
        ]
        environment = dict(os.environ)
        if os.name != "nt":
            # The build ships its shared libraries next to the binary.
            library_path = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
            environment[library_path] = str(self.binary.parent) + os.pathsep + environment.get(library_path, "")
        # The server logs a lot; a pipe nobody reads would fill up and block it, so it goes to a file.
        import tempfile
        log = tempfile.NamedTemporaryFile(prefix="reco-llama-server-", suffix=".log", delete=False)
        self.log_path = Path(log.name)
        self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=str(self.binary.parent))
        log.close()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                error = self.log_tail()
                raise RuntimeError(f"llama-server stopped while starting: {error.strip()}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server did not become ready in time")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGTERM) if os.name != "nt" else process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if self.log_path:
            self.log_path.unlink(missing_ok=True)

    def log_tail(self, size: int = 800) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-size:] if self.log_path else ""
        except OSError:
            return ""

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def ask_image(url: str, image_path: Path, prompt: str, max_tokens: int = 400, timeout: float = 300.0) -> str:
    """One question about one image (OpenAI-compatible chat API of llama-server)."""
    data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]
