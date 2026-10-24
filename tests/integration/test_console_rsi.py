"""Exercise RSI routes over TCP without a robot, model service, or Isaac runtime."""

from __future__ import annotations

import json
import os
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
            if self.path == "/runs/recorded/download":
                payload = b"recorded-run-archive" * 65536
                status, content_type = 200, "application/zip"
            elif self.path == "/artifacts/preview.png":
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


def test_rsi_streams_run_archive_through_existing_console(console, rsi_service):
    response = console.get("/api/rsi/runs/recorded/download")
    assert response.status == 200
    assert response.raw == b"recorded-run-archive" * 65536
    assert response.headers["content-type"] == "application/zip"
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


def test_rsi_proxy_prefers_configured_url(console, rsi_service, monkeypatch):
    configured_url = os.environ["EVA_RSI_URL"]
    console.config.rsi = {"url": configured_url, "command": None, "cwd": None,
                          "start_timeout_s": 1}
    monkeypatch.setenv("EVA_RSI_URL", "http://127.0.0.1:1")
    response = console.get("/api/rsi/status")
    assert response.status == 200
    assert rsi_service == [("GET", "/status", {})]


def test_rsi_tasks_reuse_collection_store_and_persist(console, tmp_path):
    console.runtime.console_ctx.output_dir = str(tmp_path)
    created = console.post("/api/rsi/tasks", {"task_id": "pull_drawer", "prompt_en": "Pull drawer"})
    assert created.status == 200
    task = next(item for item in created.json["tasks"] if item["task_id"] == "pull_drawer")
    assert task["scene_ids"] == ["rsi_observation"]
    assert console.post("/api/rsi/task", {"task_id": "pull_drawer"}).status == 200
    assert console.get("/api/rsi/tasks").json["selected_task_id"] == "pull_drawer"
    console.runtime.console_ctx.rsi_controls = None
    reloaded = console.get("/api/rsi/tasks").json
    assert any(item["task_id"] == "pull_drawer" for item in reloaded["tasks"])
    assert (tmp_path / "rsi/tasks/tasks.csv").is_file()
    assert console.runtime.command_queue.empty()


def test_rsi_synthetic_connection_cannot_home_or_send(console, tmp_path):
    console.runtime.console_ctx.output_dir = str(tmp_path)
    status = console.get("/api/rsi/client-status").json
    assert status["robot_connected"] is False
    assert status["hardware_enabled"] is False
    assert status["mode"] == "manual"
    assert console.post("/api/rsi/home").status == 409
    assert console.post("/api/rsi/send", {"candidate_id": "candidate-1"}).status == 409
    assert console.runtime.command_queue.empty()


def test_rsi_stop_revokes_auto_and_uses_existing_halt(console, tmp_path, rsi_service):
    console.runtime.console_ctx.output_dir = str(tmp_path)
    assert console.post("/api/rsi/mode", {"mode": "auto"}).json["mode"] == "auto"
    assert console.post("/api/rsi/stop").status == 200
    assert console.get("/api/rsi/client-status").json["mode"] == "manual"
    assert console.runtime.command_queue.get_nowait() == "web:halt"
    assert console.post("/api/rsi/mode", {"mode": "bad"}).status == 409
    assert console.post("/api/rsi/halt").status == 200
    assert console.runtime.command_queue.get_nowait() == "web:halt"


def test_rsi_send_requires_replay_and_deduplicates(console, tmp_path, monkeypatch):
    from core.app.console.rsi_control import RsiControls

    console.runtime.console_ctx.output_dir = str(tmp_path)
    console.post("/api/rsi/tasks", {"task_id": "pull", "prompt_en": "Pull drawer"})
    console.post("/api/rsi/task", {"task_id": "pull"})
    monkeypatch.setattr(RsiControls, "require_hardware", staticmethod(lambda ctx: None))
    candidate = {"candidate_id": "first", "task_id": "pull", "task_prompt": "Pull drawer",
                 "review_passed": True,
                 "replay_completed": False, "fps": console.config.inference_cfg.publish_rate,
                 "initial_qpos": console.runtime.transport.get_latest_qpos().tolist(),
                 "qpos": [console.runtime.transport.get_latest_qpos().tolist()]}
    monkeypatch.setattr(
        "core.app.console.rsi.RsiProxy.request",
        lambda *args: (200, json.dumps(candidate).encode(), "application/json"),
    )
    assert console.post("/api/rsi/send", {"candidate_id": "first"}).status == 409
    candidate["replay_completed"] = True
    assert console.post("/api/rsi/send", {"candidate_id": "first"}).status == 200
    assert console.runtime.command_queue.get_nowait() == "web:rsi_send"
    assert console.post("/api/rsi/send", {"candidate_id": "first"}).status == 409
    console.post("/api/rsi/halt")
    assert console.post("/api/rsi/send", {"candidate_id": "first"}).status == 409
    assert console.get("/api/rsi/client-status").json["pending_steps"] == 0


@pytest.mark.parametrize("disconnect", [False, True])
def test_rsi_send_records_existing_episode_format(console, tmp_path, monkeypatch, disconnect):
    import pyarrow.parquet as pq

    from core.app.console.rsi_control import RsiControls

    console.runtime.console_ctx.output_dir = str(tmp_path)
    console.post("/api/rsi/tasks", {"task_id": "pull", "prompt_en": "Pull drawer"})
    console.post("/api/rsi/task", {"task_id": "pull"})
    checks = []

    def feedback_guard(ctx):
        checks.append(True)
        if disconnect and len(checks) == 4:
            raise ValueError("Robot feedback lost")

    monkeypatch.setattr(RsiControls, "require_hardware", staticmethod(feedback_guard))
    candidate = {"candidate_id": "recorded", "task_id": "pull", "task_prompt": "Pull drawer",
                 "review_passed": True,
                 "replay_completed": True, "fps": console.config.inference_cfg.publish_rate,
                 "initial_qpos": console.runtime.transport.get_latest_qpos().tolist(),
                 "qpos": [console.runtime.transport.get_latest_qpos().tolist()] * 2}
    monkeypatch.setattr(
        "core.app.console.rsi.RsiProxy.request",
        lambda *args: (200, json.dumps(candidate).encode(), "application/json"),
    )
    assert console.post("/api/rsi/send", {"candidate_id": "recorded"}).status == 200
    console.pump()
    error = console.get("/api/rsi/client-status").json["last_error"]
    assert error == ("Robot feedback lost" if disconnect else "")
    assert len(checks) == 4
    dataset = tmp_path / "rsi/executions"
    paths = list(dataset.glob("data/chunk-*/*.parquet"))
    assert len(paths) == 1
    table = pq.read_table(paths[0])
    # The shared recorder resamples the synthetic transport's raw timestamps.
    assert table.num_rows >= (1 if disconnect else 2)
    assert "action.qpos" in table.column_names
    assert "observations.state.qpos" in table.column_names
    assert len(list(dataset.glob("videos/chunk-*/*/*.mp4"))) == 3
    assert console.runtime.episode_logger is None


def test_rsi_fake_zmq_with_fresh_feedback_is_not_real_hardware(console, tmp_path, monkeypatch):
    ctx = console.runtime.console_ctx
    ctx.output_dir = str(tmp_path)
    selection = ctx.device_settings.workspace.initial_selection(console.config)
    ctx.device_settings.workspace.saved = {
        "selected": selection, "overrides": {"robot": {selection["robot"]: {"mode": "fake"}}},
    }
    console.config.transport.type = "zmq"
    monkeypatch.setattr(ctx.obs_reader, "seconds_since_last_recv", lambda: 0.0)
    status = console.get("/api/rsi/client-status").json
    assert status["robot_connected"] is False
    assert status["hardware_enabled"] is False
    assert console.post("/api/rsi/home").status == 409
    assert console.post("/api/rsi/send", {"candidate_id": "fake"}).status == 409
    assert console.runtime.command_queue.empty()


def test_rsi_send_rejects_different_initial_arm_pose(console, tmp_path, monkeypatch):
    from core.app.console.rsi_control import RsiControls

    console.runtime.console_ctx.output_dir = str(tmp_path)
    console.post("/api/rsi/tasks", {"task_id": "pull", "prompt_en": "Pull drawer"})
    console.post("/api/rsi/task", {"task_id": "pull"})
    monkeypatch.setattr(RsiControls, "require_hardware", staticmethod(lambda ctx: None))
    initial = console.runtime.transport.get_latest_qpos().tolist()
    initial[0] += 0.1
    candidate = {"candidate_id": "moved", "task_id": "pull", "task_prompt": "Pull drawer",
                 "review_passed": True,
                 "replay_completed": True, "fps": console.config.inference_cfg.publish_rate,
                 "initial_qpos": initial, "qpos": [initial]}
    monkeypatch.setattr(
        "core.app.console.rsi.RsiProxy.request",
        lambda *args: (200, json.dumps(candidate).encode(), "application/json"),
    )
    response = console.post("/api/rsi/send", {"candidate_id": "moved"})
    assert response.status == 409
    assert "differs from candidate start" in response.json["error"]
    assert console.runtime.command_queue.empty()


def test_rsi_stop_cancels_home_before_command_pump(console, tmp_path, monkeypatch):
    from core.app.console.rsi_control import RsiControls

    console.runtime.console_ctx.output_dir = str(tmp_path)
    monkeypatch.setattr(RsiControls, "require_hardware", staticmethod(lambda ctx: None))
    resets = []
    monkeypatch.setattr("core.app.handlers.control.run_reset", lambda *args: resets.append(args))
    assert console.post("/api/rsi/home").status == 200
    assert console.post("/api/rsi/halt").status == 200
    console.pump()
    assert resets == []


def test_rsi_stop_during_candidate_fetch_cancels_send(console, tmp_path, monkeypatch):
    from core.app.console.rsi_control import RsiControls

    ctx = console.runtime.console_ctx
    ctx.output_dir = str(tmp_path)
    console.post("/api/rsi/tasks", {"task_id": "pull", "prompt_en": "Pull drawer"})
    console.post("/api/rsi/task", {"task_id": "pull"})
    monkeypatch.setattr(RsiControls, "require_hardware", staticmethod(lambda ctx: None))
    initial = console.runtime.transport.get_latest_qpos().tolist()
    candidate = {"candidate_id": "cancelled", "task_id": "pull", "task_prompt": "Pull drawer",
                 "review_passed": True,
                 "replay_completed": True, "fps": console.config.inference_cfg.publish_rate,
                 "initial_qpos": initial, "qpos": [initial]}

    def fetch(*args):
        ctx.rsi_controls.halt(ctx)
        return 200, json.dumps(candidate).encode(), "application/json"

    monkeypatch.setattr("core.app.console.rsi.RsiProxy.request", fetch)
    response = console.post("/api/rsi/send", {"candidate_id": "cancelled"})
    assert response.status == 409
    assert "cancelled" in response.json["error"]
    assert console.runtime.command_queue.get_nowait() == "web:halt"
    assert console.runtime.command_queue.empty()


def test_rsi_active_send_blocks_run_changes_but_allows_stop(console, tmp_path, rsi_service):
    ctx = console.runtime.console_ctx
    ctx.output_dir = str(tmp_path)
    assert console.get("/api/rsi/tasks").status == 200
    ctx.rsi_controls.executing = True
    for route in ("run", "model", "seek", "replay"):
        assert console.post(f"/api/rsi/{route}").status == 409
    assert rsi_service == []
    assert console.post("/api/rsi/stop").status == 200
    assert rsi_service == [("POST", "/stop", {})]
    assert ctx.rsi_controls.mode == "manual"


def test_editing_same_task_invalidates_old_reviewed_candidate(console, tmp_path, monkeypatch):
    from core.app.console.rsi_control import RsiControls

    console.runtime.console_ctx.output_dir = str(tmp_path)
    console.post("/api/rsi/tasks", {"task_id": "pull", "prompt_en": "Pull drawer"})
    console.post("/api/rsi/task", {"task_id": "pull"})
    monkeypatch.setattr(RsiControls, "require_hardware", staticmethod(lambda ctx: None))
    initial = console.runtime.transport.get_latest_qpos().tolist()
    candidate = {"candidate_id": "old", "task_id": "pull", "task_prompt": "Pull drawer",
                 "review_passed": True, "replay_completed": True,
                 "fps": console.config.inference_cfg.publish_rate,
                 "initial_qpos": initial, "qpos": [initial]}
    monkeypatch.setattr("core.app.console.rsi.RsiProxy.request",
                        lambda *args: (200, json.dumps(candidate).encode(), "application/json"))
    edited = console.post("/api/rsi/tasks", {"task_id": "pull", "prompt_en": "Push drawer"})
    assert edited.status == 200
    response = console.post("/api/rsi/send", {"candidate_id": "old"})
    assert response.status == 409
    assert "Task changed" in response.json["error"]
    assert console.runtime.command_queue.empty()
