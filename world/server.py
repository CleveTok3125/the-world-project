"""Start, wait for and stop the model server that backs the narrator.

The simulation and the model server are meant to run together, so the caller
opens :func:`running_server` and the rest of the run happens inside. A server
that is already listening on the port is reused rather than started twice.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

BINARY_ENV = "WORLD_NARRATOR_BINARY"
MODEL_ENV = "WORLD_NARRATOR_MODEL_FILE"
HOST_ENV = "WORLD_NARRATOR_HOST"
PORT_ENV = "WORLD_NARRATOR_PORT"
LOG_ENV = "WORLD_NARRATOR_LOG"
GPU_LAYERS_ENV = "WORLD_NARRATOR_GPU_LAYERS"
GPU_SPLIT_ENV = "WORLD_NARRATOR_GPU_SPLIT"
GPU_MAIN_ENV = "WORLD_NARRATOR_GPU_MAIN"

GPU_LIBRARY_PREFIXES = ("ggml-cuda", "ggml-hip", "ggml-vulkan", "ggml-metal")

DEFAULT_BINARY = Path("vendor/llama.cpp/build/bin/llama-server")
DEFAULT_MODEL_DIRECTORY = Path("models")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_STARTUP_TIMEOUT = 300.0
HEALTH_PATH = "/health"
READY_STATE = "ok"
CONTEXT_SIZE = 4096


class ProcessHandle(Protocol):
    """The little of a subprocess that the server supervisor actually needs."""

    returncode: int | None

    def poll(self) -> int | None:
        """Return the exit code when the process is gone, otherwise ``None``."""
        ...

    def send_signal(self, sig: int) -> None:
        """Ask the process to wind down."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process is gone and return its exit code."""
        ...

    def kill(self) -> None:
        """Stop the process the hard way."""
        ...


@dataclass
class ServerConfig:
    """How to reach and how to launch the model server.

    Attributes:
        binary: Path to the ``llama-server`` executable.
        model: Path to the GGUF model to load.
        host: Address the server binds to.
        port: Port the server listens on.
        startup_timeout: Seconds to wait for the model to finish loading.
        context_size: Context window handed to the server.
        log_path: Where the server writes its own output.
        gpu_layers: How many layers to keep in video memory, or ``None`` to leave
            the server to decide for itself.
        split_mode: How to spread the model over more than one card.
        main_gpu: Which card to treat as the main one.
    """

    binary: Path = DEFAULT_BINARY
    model: Path = field(default_factory=Path)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    context_size: int = CONTEXT_SIZE
    log_path: Path | None = None
    gpu_layers: str | None = None
    split_mode: str = "layer"
    main_gpu: int | None = None

    @property
    def base_url(self) -> str:
        """The root URL a client should talk to."""
        return f"http://{self.host}:{self.port}"

    def command(self) -> list[str]:
        """The exact command line handed to the operating system."""
        line = [
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
        ]
        if self.gpu_layers is not None:
            line += ["--gpu-layers", self.gpu_layers]
            line += ["--split-mode", self.split_mode]
        if self.main_gpu is not None:
            line += ["--main-gpu", str(self.main_gpu)]
        return line

    def has_gpu_backend(self) -> bool:
        """Whether this binary was built with any kind of card backend.

        Asking a card for layers it cannot hold is not an error the server reports
        helpfully, it is a slow run, so it is worth knowing before starting one.
        """
        if not self.binary.is_file():
            return False
        folder = self.binary.parent
        return any(
            next(folder.glob(f"lib{prefix}*"), None) is not None
            for prefix in GPU_LIBRARY_PREFIXES
        )


def find_model(directory: Path = DEFAULT_MODEL_DIRECTORY) -> Path | None:
    """The only, or the largest, GGUF model sitting in ``directory``."""
    if not directory.is_dir():
        return None
    models = sorted(directory.glob("*.gguf"))
    if not models:
        return None
    return max(models, key=lambda path: path.stat().st_size)


def config_from_environment(
    directory: Path = DEFAULT_MODEL_DIRECTORY,
) -> ServerConfig:
    """Read the server settings out of the environment, filling in sane defaults."""
    model = os.environ.get(MODEL_ENV, "").strip()
    if model:
        model_path = Path(model)
    else:
        found = find_model(directory)
        model_path = found if found is not None else directory / "model.gguf"
    log = os.environ.get(LOG_ENV, "")
    return ServerConfig(
        binary=Path(os.environ.get(BINARY_ENV, DEFAULT_BINARY)),
        model=model_path,
        host=os.environ.get(HOST_ENV, DEFAULT_HOST),
        port=int(os.environ.get(PORT_ENV, DEFAULT_PORT)),
        log_path=Path(log) if log else None,
        gpu_layers=os.environ.get(GPU_LAYERS_ENV, "").strip() or None,
        split_mode=os.environ.get(GPU_SPLIT_ENV, "").strip() or "layer",
        main_gpu=_optional_int(os.environ.get(GPU_MAIN_ENV, "")),
    )


def _optional_int(raw: str) -> int | None:
    """Read a number that is allowed to be missing."""
    text = raw.strip()
    return int(text) if text else None


def is_ready(config: ServerConfig, timeout: float = 2.0) -> bool:
    """Whether a server on that address has finished loading and can answer."""
    try:
        with urllib.request.urlopen(
            f"{config.base_url}{HEALTH_PATH}", timeout=timeout
        ) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False
    return body.get("status") == READY_STATE


def wait_until_ready(config: ServerConfig, process: ProcessHandle | None) -> None:
    """Block until the server answers, or explain why it never did."""
    deadline = time.monotonic() + config.startup_timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"the model server exited with code {process.returncode}, "
                f"see {config.log_path or 'the server output'} for the reason"
            )
        if is_ready(config):
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"the model server was still loading after {config.startup_timeout:.0f} seconds"
    )


@contextmanager
def running_server(config: ServerConfig) -> Iterator[ServerConfig]:
    """Run the model server for the duration of the block.

    A server already listening on the address is left alone. Otherwise one is
    started, and it is stopped again when the block ends, however it ends.
    """
    if is_ready(config):
        yield config
        return

    if not config.binary.is_file():
        raise RuntimeError(f"model server not found at {config.binary}")
    if not config.model.is_file():
        raise RuntimeError(f"model not found at {config.model}")
    if config.gpu_layers is not None and not config.has_gpu_backend():
        raise RuntimeError(
            f"{config.binary} was built without a card backend, so "
            f"{GPU_LAYERS_ENV} cannot do anything. Rebuild llama.cpp with CUDA, "
            "HIP, Vulkan or Metal, or unset it to run on the processor."
        )

    process = _spawn(config)
    try:
        wait_until_ready(config, process)
        yield config
    finally:
        _stop(process)


def _spawn(config: ServerConfig) -> ProcessHandle:
    if config.log_path:
        log = config.log_path.open("w")
        try:
            return subprocess.Popen(
                config.command(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
    return subprocess.Popen(
        config.command(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop(process: ProcessHandle) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()