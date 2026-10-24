"""Exercise owned backend processes and the independent YAM config without hardware."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
from pathlib import Path

import pytest

from core.app.console.rsi import RsiBackend, RsiProxy
from core.config import load_config

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def backend_config(tmp_path):
    script = tmp_path / "backend.py"
    script.write_text('''
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ready"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

def stop(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, signal.SIG_IGN if len(sys.argv) > 3 else stop)
server = HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
try:
    server.serve_forever()
finally:
    server.server_close()
    Path(sys.argv[2]).write_text("graceful shutdown")
''')
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    return {"url": f"http://127.0.0.1:{port}",
            "command": [sys.executable, str(script), str(port), str(tmp_path / "stopped")],
            "cwd": str(tmp_path), "start_timeout_s": 3}


def test_owned_backend_readiness_graceful_exit_and_signal_restore(backend_config, tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    with RsiBackend(backend_config) as backend:
        assert backend.process is not None and backend.process.poll() is None
        assert RsiProxy(backend_config["url"]).request("GET", "/status")[0] == 200
        assert signal.getsignal(signal.SIGTERM) != previous
    assert backend.process.poll() == 0
    assert (tmp_path / "stopped").read_text() == "graceful shutdown"
    assert signal.getsignal(signal.SIGTERM) == previous


def test_existing_service_is_rejected_and_external_context_does_not_stop_it(backend_config):
    with RsiBackend(backend_config) as owner:
        with pytest.raises(RuntimeError, match="already has a service"):
            with RsiBackend(backend_config):
                pytest.fail("Duplicate backend must not start")
        with RsiBackend({**backend_config, "command": None}) as external:
            assert external.process is None
        assert owner.process.poll() is None
        assert RsiProxy(backend_config["url"]).request("GET", "/status")[0] == 200


def test_start_failure_and_timeout_clean_up_owned_processes(backend_config):
    failed = RsiBackend({**backend_config,
                         "command": [sys.executable, "-c", "raise SystemExit(7)"]})
    with pytest.raises(RuntimeError, match="exit code 7"):
        with failed:
            pytest.fail("Failed backend must not become ready")
    assert failed.process.poll() == 7
    waiting = RsiBackend({**backend_config, "start_timeout_s": 0.15,
                          "command": [sys.executable, "-c", "import time; time.sleep(30)"]})
    with pytest.raises(TimeoutError, match="readiness timed out"):
        with waiting:
            pytest.fail("Unresponsive backend must not become ready")
    assert waiting.process.poll() is not None


def test_unresponsive_shutdown_kills_owned_process_group(backend_config):
    backend = RsiBackend({**backend_config, "command": [*backend_config["command"], "ignore-term"]})
    backend.STOP_TIMEOUT_S = 0.1
    with backend:
        process = backend.process
    assert process.poll() == -signal.SIGKILL


def test_yam_config_reads_runtime_paths_without_loading_credentials(tmp_path, monkeypatch):
    agent = tmp_path / "agent"
    agent.mkdir()
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"dataset_root": str(tmp_path / "dataset"),
                                  "episode": 40, "service_port": 18810}))
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    monkeypatch.setenv("EVA_RSI_ROOT", str(agent))
    monkeypatch.setenv("EVA_RSI_CONFIG", str(runtime))
    monkeypatch.setenv("EVA_RSI_CREDENTIALS", str(credentials))
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    config = load_config(ROOT / "configs/rsi/yam.py")
    assert config.console.initial_tab == "rsi" and config.robot.type == "dual_yam"
    assert config.transport.type == "dataset" and config.transport.episode_id == 40
    assert config.transport.dataset_dir == str(tmp_path / "dataset")
    assert len(config.robot.initial_qpos) == 14
    assert set(config.transport.dataset_keys.video_keys) == {
        "cam_high", "cam_left_wrist", "cam_right_wrist",
    }
    assert config.inference_cfg.setup_warmup_chunks == 0 and not config.rollout.storage.enabled
    assert config.rsi.url == "http://127.0.0.1:18810" and config.rsi.cwd == str(agent)
    assert config.rsi.command[:4] == [str(agent / ".venv/bin/python"),
                                    str(agent / "tools/serve_rsi.py"), "--config", str(runtime)]
    assert config.rsi.command[-2:] == ["--credentials", str(credentials)]
    assert Path(config.rsi.command[5]).parent == agent / "work_dirs/rsi"
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    monkeypatch.delenv("EVA_RSI_ROOT")
    monkeypatch.delenv("EVA_RSI_CREDENTIALS")
    default = load_config(ROOT / "configs/rsi/yam.py")
    assert default.rsi.cwd == str(ROOT.parents[1])
    assert "--credentials" not in default.rsi.command


def test_main_closes_backend_before_restart(backend_config, monkeypatch):
    import main as entry

    config = load_config(ROOT / "configs/00_base/defaults.py")
    config.rsi = backend_config
    monkeypatch.setattr(entry, "parse_args", lambda: (config, 8080, "yam.py", False))
    monkeypatch.setattr(entry, "run", lambda *args, **kwargs: True)
    receipts = []

    def record_restart(executable, argv):
        assert RsiProxy(backend_config["url"]).request("GET", "/status")[0] == 502
        receipts.append(executable)

    monkeypatch.setattr(entry.os, "execv", record_restart)
    entry.main()
    assert receipts == [sys.executable]


def test_main_interrupt_cleans_backend_and_returns_without_traceback(backend_config, monkeypatch):
    import main as entry

    config = load_config(ROOT / "configs/00_base/defaults.py")
    config.rsi = backend_config
    monkeypatch.setattr(entry, "parse_args", lambda: (config, 8080, "yam.py", False))

    def interrupt(*args, **kwargs):
        assert RsiProxy(backend_config["url"]).request("GET", "/status")[0] == 200
        raise KeyboardInterrupt

    monkeypatch.setattr(entry, "run", interrupt)
    assert entry.main() is None
    assert RsiProxy(backend_config["url"]).request("GET", "/status")[0] == 502
