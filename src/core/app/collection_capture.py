"""Background raw snapshot capture for collection and simulator episodes."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)
SIM_TICK_DRAIN_TIMEOUT_S = 5.0


class CollectionCaptureRunner:
    """Move raw snapshot capture off the control/state-machine loop."""

    def __init__(
        self,
        runtime: Any,
        fps: float,
        max_raw_snapshots_per_tick: int,
        simulator_ticks: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"collection capture fps must be positive, got {fps}")
        if max_raw_snapshots_per_tick <= 0:
            raise ValueError(
                "collection capture max_raw_snapshots_per_tick must be positive"
            )
        self._runtime = runtime
        self._interval_s = 1.0 / float(fps)
        self._max_raw_snapshots_per_tick = max_raw_snapshots_per_tick
        self._simulator_ticks = simulator_ticks
        self._last_source_frame_id: int | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="eva-collection-capture",
            daemon=True,
        )
        self._error: BaseException | None = None

    def start(self) -> None:
        """Start the background capture thread."""
        self._thread.start()

    def stop(self, final_source_frame_id: int | None = None) -> None:
        """Stop the capture thread and surface any background failure."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        if self._error is not None:
            raise RuntimeError("collection capture runner failed") from self._error
        if final_source_frame_id is None:
            self._drain_available_raw(self._max_raw_snapshots_per_tick)
        else:
            self._drain_through_source_frame(final_source_frame_id)

    @property
    def is_running(self) -> bool:
        """True while the capture thread is alive."""
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._capture_tick()
                self._stop_event.wait(self._interval_s)
        except BaseException as exc:
            self._error = exc
            logger.exception("Collection capture runner failed")

    def _capture_tick(self) -> bool:
        if not self._active_loggers():
            return False
        recorded = False
        for _ in range(self._max_raw_snapshots_per_tick):
            if self._stop_event.is_set():
                break
            snapshot = self._runtime.transport.acquire_collection_raw()
            if snapshot is None:
                break
            self._ingest(snapshot)
            recorded = True
        return recorded

    def _drain_available_raw(self, max_snapshots: int) -> bool:
        if not self._active_loggers():
            return False
        recorded = False
        for _ in range(max_snapshots):
            snapshot = self._runtime.transport.acquire_collection_raw()
            if snapshot is None:
                return recorded
            self._ingest(snapshot)
            recorded = True
        return recorded

    def _drain_through_source_frame(self, final_source_frame_id: int) -> None:
        if not self._simulator_ticks:
            raise RuntimeError("source-frame drain requires simulator tick capture")
        if final_source_frame_id < 0:
            raise ValueError(f"simulator stop frame must be non-negative: {final_source_frame_id}")
        if final_source_frame_id == 0 and self._last_source_frame_id is None:
            return
        expected_last = final_source_frame_id - 1
        deadline = time.monotonic() + SIM_TICK_DRAIN_TIMEOUT_S
        while self._last_source_frame_id != expected_last:
            if (
                self._last_source_frame_id is not None
                and self._last_source_frame_id >= final_source_frame_id
            ):
                raise ValueError(
                    "simulator capture passed stop boundary: "
                    f"last={self._last_source_frame_id}, stop={final_source_frame_id}"
                )
            snapshot = self._runtime.transport.acquire_collection_raw()
            if snapshot is not None:
                self._ingest(snapshot)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out draining simulator frames through "
                    f"{expected_last}; last={self._last_source_frame_id}"
                )
            time.sleep(0.001)

    def _active_loggers(self) -> list[Any]:
        loggers = [
            logger_obj
            for logger_obj in (
                getattr(self._runtime, "episode_logger", None),
                getattr(self._runtime, "rollout_episode_logger", None),
            )
            if logger_obj is not None and getattr(logger_obj, "has_active_episode", True)
        ]
        if self._simulator_ticks:
            return loggers
        return [logger_obj for logger_obj in loggers if logger_obj.is_collection_enabled]

    def _ingest(self, snapshot: Any) -> None:
        loggers = self._active_loggers()
        if self._simulator_ticks:
            frame_id = snapshot.source_frame_id
            expected = 0 if self._last_source_frame_id is None else self._last_source_frame_id + 1
            if frame_id != expected:
                raise ValueError(
                    f"non-contiguous simulator frames: expected {expected}, got {frame_id}"
                )
            self._last_source_frame_id = frame_id
            for logger_obj in loggers:
                logger_obj.ingest_raw_episode_snapshot(snapshot)
            return
        for logger_obj in loggers:
            logger_obj.ingest_collection_snapshot(snapshot)


def start_collection_capture(
    runtime: Any,
    fps: float,
    max_raw_snapshots_per_tick: int,
) -> None:
    """Attach and start one collection capture runner on the runtime."""
    if getattr(runtime, "collection_capture_runner", None) is not None:
        raise RuntimeError("collection capture runner is already active")
    runner = CollectionCaptureRunner(runtime, fps, max_raw_snapshots_per_tick)
    runtime.collection_capture_runner = runner
    runner.start()


def stop_collection_capture(runtime: Any) -> None:
    """Stop and detach the active collection capture runner if present."""
    runner = getattr(runtime, "collection_capture_runner", None)
    if runner is None:
        return
    try:
        runner.stop()
    finally:
        runtime.collection_capture_runner = None


def start_sim_tick_capture(
    runtime: Any,
    fps: float,
    max_raw_snapshots_per_tick: int,
) -> None:
    """Capture simulator-owned state/action ticks for an active rollout."""
    if getattr(runtime, "collection_capture_runner", None) is not None:
        raise RuntimeError("raw capture runner is already active")
    runner = CollectionCaptureRunner(
        runtime,
        fps,
        max_raw_snapshots_per_tick,
        simulator_ticks=True,
    )
    runtime.collection_capture_runner = runner
    runner.start()


def stop_sim_tick_capture(runtime: Any, final_source_frame_id: int) -> None:
    """Drain through the simulator stop boundary and detach the capture runner."""
    runner = getattr(runtime, "collection_capture_runner", None)
    if runner is None:
        return
    runner.stop(final_source_frame_id)
    runtime.collection_capture_runner = None
