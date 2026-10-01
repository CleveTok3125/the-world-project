from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from world.server import (
    BINARY_ENV,
    GPU_LAYERS_ENV,
    GPU_MAIN_ENV,
    GPU_SPLIT_ENV,
    HOST_ENV,
    LOG_ENV,
    MODEL_ENV,
    PORT_ENV,
    ServerConfig,
    config_from_environment,
    find_model,
    is_ready,
    running_server,
    wait_until_ready,
)

CALLS: list[list[str]] = []


@pytest.fixture
def config(tmp_path: Path):
    """A configuration whose binary and model are real empty files."""
    binary = tmp_path / "llama-server"
    binary.write_text("")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")

    def build(**overrides) -> ServerConfig:
        settings = {"binary": binary, "model": model}
        settings.update(overrides)
        return ServerConfig(**settings)

    return build


def fake_popen(script: str):
    """A stand in for ``subprocess.Popen`` that records commands and does nothing."""

    class FakeProcess:
        def __init__(self, args: list[str], **_: object) -> None:
            CALLS.append(args)
            self.args = args
            self.returncode: int | None = None
            self.signals: list[object] = []
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, signal: object) -> None:
            self.signals.append(signal)
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode or 0

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    def factory(args: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(args, **kwargs)

    return factory


class ScriptedProcess:
    """A process that reports itself as already dead."""

    def __init__(self, *_args: object, code: int = 1, **_: object) -> None:
        self.returncode = code

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, *_: object) -> None:
        return

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def kill(self) -> None:
        return


@pytest.fixture
def clean_environment(monkeypatch):
    for name in (
        BINARY_ENV,
        MODEL_ENV,
        HOST_ENV,
        PORT_ENV,
        LOG_ENV,
        GPU_LAYERS_ENV,
        GPU_SPLIT_ENV,
        GPU_MAIN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    CALLS.clear()
    return monkeypatch


class TestFindModel:
    def test_it_picks_the_only_model(self, tmp_path: Path) -> None:
        (tmp_path / "one.gguf").write_bytes(b"x")

        assert find_model(tmp_path) == tmp_path / "one.gguf"

    def test_it_picks_the_largest_when_several_are_present(self, tmp_path: Path) -> None:
        (tmp_path / "small.gguf").write_bytes(b"x" * 10)
        (tmp_path / "big.gguf").write_bytes(b"x" * 100)

        assert find_model(tmp_path) == tmp_path / "big.gguf"

    def test_it_ignores_files_that_are_not_models(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello")

        assert find_model(tmp_path) is None

    def test_a_missing_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert find_model(tmp_path / "absent") is None

    def test_an_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert find_model(tmp_path) is None


class TestConfiguration:
    def test_the_defaults_point_at_the_vendored_server(self, clean_environment) -> None:
        config = config_from_environment()

        assert config.binary == Path("vendor/llama.cpp/build/bin/llama-server")
        assert config.port == 8080
        assert config.host == "127.0.0.1"

    def test_every_setting_can_come_from_the_environment(
        self, clean_environment, tmp_path: Path
    ) -> None:
        clean_environment.setenv(BINARY_ENV, "/opt/llama-server")
        clean_environment.setenv(MODEL_ENV, str(tmp_path / "m.gguf"))
        clean_environment.setenv(HOST_ENV, "0.0.0.0")
        clean_environment.setenv(PORT_ENV, "9000")
        clean_environment.setenv(LOG_ENV, str(tmp_path / "server.log"))

        config = config_from_environment()

        assert config.binary == Path("/opt/llama-server")
        assert config.model == tmp_path / "m.gguf"
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.log_path == tmp_path / "server.log"

    def test_a_model_in_the_directory_is_found_automatically(
        self, clean_environment, tmp_path: Path
    ) -> None:
        (tmp_path / "picked.gguf").write_bytes(b"x" * 50)

        assert config_from_environment(tmp_path).model == tmp_path / "picked.gguf"

    def test_without_any_model_a_placeholder_path_is_kept(
        self, clean_environment, tmp_path: Path
    ) -> None:
        assert config_from_environment(tmp_path).model == tmp_path / "model.gguf"

    def test_the_command_line_names_the_model_and_the_port(self, tmp_path: Path) -> None:
        config = ServerConfig(binary=Path("/bin/llama-server"), model=tmp_path / "m.gguf")

        command = config.command()

        assert command[0] == "/bin/llama-server"
        assert str(tmp_path / "m.gguf") in command
        assert "8080" in command

    def test_the_base_url_follows_the_host_and_the_port(self) -> None:
        assert ServerConfig(host="1.2.3.4", port=9000).base_url == "http://1.2.3.4:9000"


class TestHealth:
    def test_a_server_that_answers_ok_is_ready(self) -> None:
        with _fake_health("ok") as url:
            assert is_ready(_config_for(url))

    def test_a_server_still_loading_is_not_ready(self) -> None:
        with _fake_health("loading model") as url:
            assert not is_ready(_config_for(url))

    def test_nothing_listening_is_not_ready(self) -> None:
        assert not is_ready(ServerConfig(host="127.0.0.1", port=9))

    def test_a_non_json_reply_is_not_ready(self) -> None:
        with _fake_health("not json at all") as url:
            assert not is_ready(_config_for(url))


class TestRunningServer:
    def test_a_server_already_listening_is_reused(
        self, clean_environment, monkeypatch
    ) -> None:
        def fail(*_: object, **__: object) -> None:
            raise AssertionError("must not start a second server")

        monkeypatch.setattr(subprocess, "Popen", fail)
        with _fake_health("ok") as url, running_server(_config_for(url)) as config:
            assert config.base_url == url

    def test_a_missing_server_binary_is_reported(self, clean_environment) -> None:
        config = ServerConfig(
            binary=Path("/nowhere/llama-server"), model=Path("/nowhere/m.gguf")
        )

        with pytest.raises(RuntimeError, match="model server not found"), running_server(
            config
        ):
            pass

    def test_a_missing_model_is_reported(self, clean_environment, tmp_path: Path) -> None:
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")

        config = ServerConfig(binary=binary, model=tmp_path / "absent.gguf")

        with pytest.raises(RuntimeError, match="model not found"), running_server(config):
            pass

    def test_the_server_is_started_when_it_is_not_listening(
        self, clean_environment, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(subprocess, "Popen", fake_popen(""))
        monkeypatch.setattr("world.server.wait_until_ready", lambda *a, **k: None)
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")

        with running_server(ServerConfig(binary=binary, model=model)):
            pass

        assert CALLS
        assert str(model) in CALLS[0]

    def test_the_server_is_stopped_when_the_block_ends(
        self, clean_environment, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(subprocess, "Popen", fake_popen(""))
        monkeypatch.setattr("world.server.wait_until_ready", lambda *a, **k: None)
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")

        with running_server(ServerConfig(binary=binary, model=model)) as config:
            pass

        assert config.port == 8080

    def test_a_failure_inside_the_block_still_stops_the_server(
        self, clean_environment, monkeypatch, tmp_path: Path
    ) -> None:
        stopped: list[bool] = []
        monkeypatch.setattr(subprocess, "Popen", fake_popen(""))
        monkeypatch.setattr("world.server.wait_until_ready", lambda *a, **k: None)
        monkeypatch.setattr(
            "world.server._stop", lambda process: stopped.append(True)
        )
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")

        with (
            pytest.raises(RuntimeError, match="boom"),
            running_server(ServerConfig(binary=binary, model=model)),
        ):
            raise RuntimeError("boom")

        assert stopped == [True]

    def test_the_server_log_is_written_when_a_path_is_given(
        self, clean_environment, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(subprocess, "Popen", fake_popen(""))
        monkeypatch.setattr("world.server.wait_until_ready", lambda *a, **k: None)
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        log = tmp_path / "server.log"

        with running_server(ServerConfig(binary=binary, model=model, log_path=log)):
            pass

        assert log.is_file()

    def test_no_log_file_is_created_without_a_path(
        self, clean_environment, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(subprocess, "Popen", fake_popen(""))
        monkeypatch.setattr("world.server.wait_until_ready", lambda *a, **k: None)
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")

        with running_server(ServerConfig(binary=binary, model=model)):
            pass

        assert not list(tmp_path.glob("*.log"))


class TestRunningOnACard:
    def test_the_processor_is_used_unless_a_card_is_asked_for(self, config) -> None:
        line = config().command()

        assert "--gpu-layers" not in line
        assert "--main-gpu" not in line

    def test_asking_for_a_card_puts_the_layers_in_the_command(self, config) -> None:
        line = config(gpu_layers="all").command()

        assert line[line.index("--gpu-layers") + 1] == "all"
        assert line[line.index("--split-mode") + 1] == "layer"

    def test_a_particular_card_can_be_named(self, config) -> None:
        line = config(gpu_layers="20", main_gpu=1).command()

        assert line[line.index("--main-gpu") + 1] == "1"
        assert line[line.index("--gpu-layers") + 1] == "20"

    def test_how_to_split_across_cards_can_be_chosen(self, config) -> None:
        line = config(gpu_layers="all", split_mode="row").command()

        assert line[line.index("--split-mode") + 1] == "row"

    def test_a_binary_with_a_card_backend_is_recognised(self, tmp_path) -> None:
        binary = tmp_path / "llama-server"
        binary.write_text("")
        (tmp_path / "libggml-cuda.so").write_text("")

        assert ServerConfig(binary=binary).has_gpu_backend() is True

    def test_a_processor_only_binary_says_so(self, config) -> None:
        assert config().has_gpu_backend() is False

    def test_a_missing_binary_says_it_has_no_backend(self, tmp_path) -> None:
        assert ServerConfig(binary=tmp_path / "nothing").has_gpu_backend() is False

    def test_asking_for_a_card_on_a_processor_only_build_is_refused(self, config) -> None:
        with (
            pytest.raises(RuntimeError, match="without a card backend"),
            running_server(config(gpu_layers="all")),
        ):
            pass

    def test_the_settings_come_from_the_environment(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(GPU_LAYERS_ENV, "all")
        monkeypatch.setenv(GPU_SPLIT_ENV, "row")
        monkeypatch.setenv(GPU_MAIN_ENV, "1")

        config = config_from_environment(tmp_path)

        assert config.gpu_layers == "all"
        assert config.split_mode == "row"
        assert config.main_gpu == 1

    def test_no_layers_set_means_no_card_asking(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv(GPU_LAYERS_ENV, raising=False)
        monkeypatch.setenv(GPU_MAIN_ENV, "  ")

        config = config_from_environment(tmp_path)

        assert config.gpu_layers is None
        assert config.main_gpu is None


class TestWaitUntilReady:
    def test_it_returns_as_soon_as_the_server_answers(
        self, clean_environment, monkeypatch
    ) -> None:
        answers = iter([False, False, True])
        monkeypatch.setattr("world.server.is_ready", lambda *a, **k: next(answers))
        monkeypatch.setattr("time.sleep", lambda _s: None)

        wait_until_ready(ServerConfig(startup_timeout=30), None)

    def test_a_server_that_dies_early_is_reported(
        self, clean_environment, monkeypatch
    ) -> None:
        monkeypatch.setattr("world.server.is_ready", lambda *a, **k: False)

        with pytest.raises(RuntimeError, match="exited with code 3"):
            wait_until_ready(ServerConfig(), ScriptedProcess(code=3))

    def test_a_server_that_never_loads_is_reported(
        self, clean_environment, monkeypatch
    ) -> None:
        monkeypatch.setattr("world.server.is_ready", lambda *a, **k: False)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        monkeypatch.setattr("time.monotonic", _counting_clock())

        with pytest.raises(RuntimeError, match="still loading"):
            wait_until_ready(ServerConfig(startup_timeout=0.2), None)


def _counting_clock():
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def clock() -> float:
        return next(ticks, 99.0)

    return clock


def _config_for(url: str) -> ServerConfig:
    """A configuration aimed at whatever port the fake health server took."""
    host, port = url.rsplit("://", 1)[1].split(":")
    return ServerConfig(host=host, port=int(port))


class _fake_health:
    """Serve a fixed health payload on a free port for the duration of the block."""

    def __init__(self, status: str) -> None:
        self.status = status
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if outer.status == "ok":
                    body = json.dumps({"status": "ok"}).encode()
                else:
                    body = outer.status.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_: object) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()