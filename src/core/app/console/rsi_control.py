"""RSI operator state using the console task store and existing motion handlers."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from core.app.console.rsi import RsiProxy
from core.app.state import SessionMode, SessionStatus
from core.recorder.episode import EpisodeLogger
from tools.datasets.store import TaskSetStore


class RsiControls:
    def __init__(self, root: Path, initial_tasks: list[dict]) -> None:
        self.root = root
        self.store = TaskSetStore(root / "tasks")
        self.lock = threading.RLock()
        self.mode = "manual"
        self.selected_task_id = None
        self.candidate_id = None
        self.sent_candidate_ids: set[str] = set()
        self.pending = None
        self.executing = False
        self.generation = 0
        self.last_error = ""
        state = self.store.state()
        if not state["scenes"]:
            self.store.upsert("scenes", None, {"scene_id": "rsi_observation", "placements": []})
        if not state["tasks"]:
            for task in initial_tasks:
                self.save_task(task)

    def tasks(self) -> dict:
        return {"ok": True, "tasks": self.store.state()["tasks"],
                "selected_task_id": self.selected_task_id}

    def save_task(self, body: dict) -> dict:
        if not str(body.get("prompt_en", "")).strip():
            raise ValueError("Task prompt is required")
        with self.lock:
            if self.pending is not None or self.executing:
                raise ValueError("Stop the current send before editing tasks")
            existing = {task["task_id"]: task for task in self.store.state()["tasks"]}
            payload = {"scene_ids": ["rsi_observation"], "scene_epsiodes_count": [1],
                       "total_epsiodes_count": 1, **existing.get(body["task_id"], {}), **body}
            current_id = body["task_id"] if body["task_id"] in existing else None
            self.store.upsert("tasks", current_id, payload)
            return self.tasks()

    def select_task(self, task_id: str) -> dict:
        with self.lock:
            if self.pending is not None or self.executing:
                raise ValueError("Stop the current send before changing tasks")
            if task_id not in {task["task_id"] for task in self.store.state()["tasks"]}:
                raise ValueError("Unknown RSI task")
            self.selected_task_id = task_id
            self.mode = "manual"
            self.generation += 1
            return self.tasks()

    def status(self, ctx, snapshot: dict) -> dict:
        live = snapshot["transport_type"] in {"zmq", "ros1", "ros2"}
        connected = bool(live and snapshot["transport_connected"]
                         and self.robot_mode(ctx) == "real")
        return {"ok": True, "robot_type": snapshot["robot_type"],
                "transport_type": snapshot["transport_type"], "robot_connected": connected,
                "hardware_enabled": connected, "devices": ctx.device_settings.status(),
                "session_status": snapshot["session_status"], "mode": self.mode,
                "candidate_id": self.candidate_id,
                "sent_candidate_ids": sorted(self.sent_candidate_ids),
                "pending_steps": 0 if self.pending is None else len(self.pending["qpos"]),
                "last_error": self.last_error or snapshot["last_error"]}

    @staticmethod
    def robot_mode(ctx) -> str:
        config = ctx.runtime.active_config or ctx.config
        workspace = ctx.device_settings.workspace
        return workspace.resolve(workspace.initial_selection(config))["robot"]["mode"]

    @staticmethod
    def require_hardware(ctx) -> None:
        config = ctx.runtime.active_config or ctx.config
        reader = ctx.obs_reader or ctx.runtime.transport
        qpos = reader.get_latest_qpos()
        freshness = reader.seconds_since_last_recv()
        if (config.transport.type not in {"zmq", "ros1", "ros2"}
                or RsiControls.robot_mode(ctx) != "real"
                or freshness is None or freshness > 2 or qpos is None):
            raise ValueError("Live feedback is required; recorded/synthetic sources cannot send")
        runtime, session = ctx.runtime, ctx.session
        if (runtime.collection_teleop_armed or runtime.collection_teleop_active
                or session.manual_publish_active or session.status is SessionStatus.RUNNING):
            raise ValueError("Stop the active robot operation first")

    @staticmethod
    def check_start(ctx, candidate: dict) -> None:
        from core.app.handlers.control import MANUAL_MAX_QPOS_STEP

        config = ctx.runtime.active_config or ctx.config
        measured = np.asarray(ctx.runtime.transport.get_latest_qpos(), dtype=np.float32)
        initial = np.asarray(candidate["initial_qpos"], dtype=np.float32)
        if (measured.shape != (ctx.runtime.robot.total_action_dim,)
                or measured.shape != initial.shape or not np.isfinite(measured).all()
                or not np.isfinite(initial).all()):
            raise ValueError("Live robot and candidate initial joint states must match dimensions")
        mask = np.asarray(ctx.runtime.robot.gripper_mask, dtype=bool)
        tolerance = config.inference_cfg.get("manual_max_qpos_step", MANUAL_MAX_QPOS_STEP)
        if np.any(np.abs(measured[~mask] - initial[~mask]) > tolerance):
            raise ValueError("Live arm pose differs from candidate start; infer and Replay again")
        if config.robot.type == "dual_yam":
            if config.robot.gripper_open != 1.0 or config.robot.gripper_close != 0.0:
                raise ValueError("RSI YAM requires normalized gripper units: 0 closed, 1 open")
            if np.any((measured[mask] < 0) | (measured[mask] > 1)
                      | (initial[mask] < 0) | (initial[mask] > 1)):
                raise ValueError("YAM gripper state is outside normalized [0, 1]")

    def set_mode(self, mode: str) -> dict:
        if mode not in {"manual", "auto"}:
            raise ValueError("RSI mode must be manual or auto")
        with self.lock:
            if self.pending is not None or self.executing:
                raise ValueError("Stop the current send before changing dispatch mode")
            if self.mode != mode:
                self.generation += 1
            self.mode = mode
            return {"ok": True, "mode": self.mode}

    def halt(self, ctx) -> None:
        with self.lock:
            self.mode = "manual"
            self.generation += 1
            self.pending = None
            ctx.session.interrupt_requested = True
            ctx.runtime.command_queue.put("web:halt")

    @staticmethod
    def candidate(proxy: RsiProxy) -> dict:
        status, raw, _ = proxy.request("GET", "/candidate")
        value = json.loads(raw)
        if status != 200:
            raise ValueError(value["error"])
        if (not value.get("candidate_id") or not value["review_passed"]
                or not value["replay_completed"]):
            raise ValueError("Review and complete simulation Replay before sending")
        return value

    def stage_send(self, ctx, proxy: RsiProxy, candidate_id: str) -> dict:
        with self.lock:
            generation = self.generation
        self.require_hardware(ctx)
        candidate = self.candidate(proxy)
        self.check_start(ctx, candidate)
        qpos = np.asarray(candidate["qpos"], dtype=np.float32)
        if (candidate["candidate_id"] != candidate_id or qpos.ndim != 2 or not len(qpos)
                or qpos.shape[1] != ctx.runtime.robot.total_action_dim
                or not np.isfinite(qpos).all()):
            raise ValueError("Candidate identity or joint trajectory is invalid")
        config = ctx.runtime.active_config or ctx.config
        if candidate["fps"] != config.inference_cfg.publish_rate:
            raise ValueError("Candidate fps must match the configured robot publish rate")
        with self.lock:
            if generation != self.generation:
                raise ValueError("Send cancelled by STOP, task, or dispatch mode change")
            if (self.pending is not None or self.executing
                    or candidate_id in self.sent_candidate_ids):
                raise ValueError("Candidate already queued or sent; infer a new candidate")
            selected = next((task for task in self.store.state()["tasks"]
                             if task["task_id"] == self.selected_task_id), None)
            if (selected is None or candidate.get("task_id") != self.selected_task_id
                    or candidate.get("task_prompt") != selected["prompt_en"]):
                raise ValueError("Task changed; infer and Replay a new candidate")
            self.candidate_id = candidate_id
            self.pending = {"candidate_id": candidate_id, "qpos": qpos,
                            "generation": self.generation, "proxy": proxy}
            self.sent_candidate_ids.add(candidate_id)
            self.last_error = ""
            ctx.runtime.command_queue.put("web:rsi_send")
        return {"ok": True, "queued": True, "candidate_id": candidate_id}

    def execute(self, ctx) -> None:
        from core.app.handlers.control import run_pending_chunk_on_real
        from core.app.handlers.recording import end_episode, start_episode

        with self.lock:
            pending = self.pending
            if pending is None:
                return
            self.executing = True
        try:
            self.require_hardware(ctx)
            current = self.candidate(pending["proxy"])
            self.check_start(ctx, current)
            if current["candidate_id"] != pending["candidate_id"]:
                raise ValueError("Candidate changed after queuing; infer and Replay again")
            runtime, session = ctx.runtime, ctx.session
            config = runtime.active_config or ctx.config
            with self.lock:
                if pending["generation"] != self.generation:
                    return
                if runtime.episode_logger is not None:
                    raise ValueError("Finish the existing episode before sending RSI actions")
                runtime.episode_logger = EpisodeLogger(
                    log_dir=self.root / "executions", robot=runtime.robot,
                    fps=int(config.inference_cfg.publish_rate),
                    dataset_keys=config.transport.dataset_keys,
                    convert_bgr_to_rgb=config.transport.convert_bgr_to_rgb,
                    gripper_open=config.robot.get("gripper_open"),
                    gripper_close=config.robot.get("gripper_close"),
                    gripper_threshold=config.robot.get("gripper_threshold"),
                )
                task = next(task for task in self.store.state()["tasks"]
                            if task["task_id"] == self.selected_task_id)
                session.selected_task = task["prompt_en"]
                session.mode = SessionMode.STEP
                session.status = SessionStatus.READY
                session.interrupt_requested = False
                session.gripper_locks.clear()
                session.follow_human_gripper = False
                session.pending_real_chunk = pending["qpos"]
            try:
                start_episode(runtime, session)
                completed = run_pending_chunk_on_real(
                    config, runtime, session, feedback_guard=lambda: self.require_hardware(ctx),
                )
                if not completed:
                    self.last_error = "Robot trajectory was stopped before completion"
            finally:
                try:
                    end_episode(runtime)
                    runtime.episode_logger.finalize()
                finally:
                    runtime.episode_logger = None
                    session.pending_real_chunk = None
                    session.status = SessionStatus.READY
        except (ValueError, OSError, RuntimeError) as error:
            self.last_error = str(error)
            ctx.session.last_error = str(error)
        finally:
            with self.lock:
                self.pending = None
                self.executing = False

    def home(self, ctx, generation: int) -> None:
        from core.app.handlers.control import run_reset

        try:
            with self.lock:
                if generation != self.generation:
                    return
            self.require_hardware(ctx)
            ctx.session.mode = SessionMode.STEP
            run_reset(ctx.runtime.active_config or ctx.config, ctx.runtime, ctx.session)
        except (ValueError, OSError, RuntimeError) as error:
            self.last_error = str(error)
            ctx.session.last_error = str(error)
