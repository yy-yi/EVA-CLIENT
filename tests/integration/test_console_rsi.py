"""Exercise RSI routes over TCP without a robot, model service, or Isaac runtime."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.app.console.server import _resolve_initial_tab
from tests.integration._harness import console_config

pytestmark = pytest.mark.integration


@pytest.fixture
def rsi_service(monkeypatch: pytest.MonkeyPatch) -> Generator[list[tuple[str, str, dict]]]:
    calls: list[tuple[str, str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            calls.append(("GET", self.path, {}))
            if self.path == "/artifacts/preview.png":
                payload = b"preview-image"
                status, content_type = 200, "image/png"
            elif self.path == "/frame":
                payload = json.dumps(
                    {"qpos": [0.0] * 14, "cameras": {}, "simulation": {"available": False}}
                ).encode()
                status, content_type = 200, "application/json"
            elif self.path == "/artifacts":
                payload = json.dumps(
                    {"items": [{"name": "CPU preview", "url": "/artifacts/preview.png"}]}
                ).encode()
                status, content_type = 200, "application/json"
            else:
                payload = json.dumps(
                    {"status": "ready", "robot": {"connected": False, "source": "dataset"}}
                ).encode()
                status, content_type = 200, "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, body))
            payload = json.dumps({"ok": self.path != "/seek", "error": "out of range"}).encode()
            self.send_response(400 if self.path == "/seek" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            return

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("EVA_RSI_URL", f"http://127.0.0.1:{service.server_port}")
    try:
        yield calls
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


def test_rsi_missing_configuration_is_explicit(console, monkeypatch):
    monkeypatch.delenv("EVA_RSI_URL", raising=False)
    response = console.get("/api/rsi/status")
    assert response.status == 503
    assert response.json["configured"] is False
    assert "EVA_RSI_URL" in response.json["error"]
    assert console.runtime.command_queue.empty()


def test_rsi_proxy_reads_frames_and_artifacts_without_robot_commands(console, rsi_service):
    response = console.get("/api/rsi/status")
    assert response.status == 200
    assert response.json["robot"] == {"connected": False, "source": "dataset"}
    assert len(console.get("/api/rsi/frame").json["qpos"]) == 14
    assert console.get("/api/rsi/artifacts").json["items"][0]["name"] == "CPU preview"
    image = console.get("/api/rsi/artifacts/preview.png")
    assert image.raw == b"preview-image"
    assert image.headers["content-type"] == "image/png"
    assert console.runtime.command_queue.empty()
    assert len(rsi_service) == 4


def test_rsi_commands_forward_body_and_preserve_rejection(console, rsi_service):
    assert console.post("/api/rsi/run").json["ok"]
    rejected = console.post("/api/rsi/seek", {"index": 999})
    assert rejected.status == 400
    assert rejected.json == {"ok": False, "error": "out of range"}
    assert console.post("/api/rsi/stop").json["ok"]
    assert rsi_service == [
        ("POST", "/run", {}),
        ("POST", "/seek", {"index": 999}),
        ("POST", "/stop", {}),
    ]
    assert console.runtime.command_queue.empty()


def test_rsi_rejects_robot_routes(console, rsi_service):
    assert console.post("/api/rsi/reset").status == 404
    assert console.post("/api/rsi/action", {"qpos": [0] * 14}).status == 404
    assert rsi_service == []
    assert console.runtime.command_queue.empty()


def test_rsi_unreachable_service_reports_gateway_error(console, monkeypatch):
    with ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler) as unused:
        port = unused.server_port
    monkeypatch.setenv("EVA_RSI_URL", f"http://127.0.0.1:{port}")
    response = console.get("/api/rsi/status")
    assert response.status == 502
    assert "连接失败" in response.json["error"]


def test_rsi_initial_tab_and_assets(console):
    config = console_config()
    config.console.initial_tab = "rsi"
    assert _resolve_initial_tab(config) == "rsi"
    console.config.console.initial_tab = "rsi"
    assert console.get("/api/config").json["initial_tab"] == "rsi"
    assert b'data-tab="rsi"' in console.get("/").raw
    assert b"/api/rsi/status" in console.get("/js/rsi.js").raw
