#!/usr/bin/env python3
"""Export one RSI candidate alongside aligned recorded state/RGB, without hardware IO.

Uses the Collection/Eval EpisodeLogger format. The action column contains model
targets; the observations remain independent recorded evidence, never execution
feedback. --task-file freezes a Collection task JSON object with prompt_en.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Generator
from contextlib import ExitStack, closing
from itertools import islice
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")]

from core.config import ConfigDict  # noqa: E402
from core.recorder.episode import EpisodeLogger  # noqa: E402
from core.registry import ROBOT_REGISTRY  # noqa: E402
from core.types import Observation  # noqa: E402
from core.utils.lerobot import LeRobotDatasetIO  # noqa: E402
from robots.zoo import dual_yam  # noqa: E402, F401
from tools.conversion.source import video_frames  # noqa: E402


def export_episode(
    dataset_root: Path,
    episode: int,
    predictions: Path,
    output: Path,
    task_definition: dict[str, Any],
) -> dict[str, Any]:
    """Write a new dataset using only candidate actions and indexed source observations."""
    prediction = json.loads(predictions.read_text(encoding="utf-8"))
    task = task_definition["prompt_en"]
    robot = ROBOT_REGISTRY.build("dual_yam")
    actions = np.asarray(prediction["actions"], dtype=np.float32)
    indices = np.asarray(prediction["action_indices"], dtype=np.int64)
    fps = float(prediction["control_hz"])
    if (
        actions.ndim != 2 or actions.shape[0] == 0
        or actions.shape[1] != robot.initial_qpos.size or not np.isfinite(actions).all()
        or indices.shape != (actions.shape[0],)
        or not np.array_equal(indices, prediction["action_indices"])
        or not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
        or indices[0] < 0 or not np.isfinite(fps) or fps <= 0 or not task.strip()
    ):
        raise ValueError("candidate requires nonempty finite 14-D actions and contiguous indices")
    if (
        prediction["source"] != "agent_reviewed_sandbox_candidate"
        or not prediction["candidate_id"] or prediction["hardware_executed_steps"] != 0
    ):
        raise ValueError("export requires a reviewed, unexecuted recorded-open-loop candidate")

    source = LeRobotDatasetIO(dataset_root)
    info = json.loads((dataset_root / "meta/info.json").read_text())
    if not np.isclose(fps, float(info["fps"])) or not fps.is_integer():
        raise ValueError("candidate and recorded video rates must match at an integer FPS")
    keys = ConfigDict(
        state_key="observations.state.qpos", eef_key="observations.state.eef",
        action_key="action.qpos",
        video_keys={
            cam.observation_key: f"observation.images.{cam.observation_key}"
            for cam in robot.observation_schema.cameras
        },
    )
    table = pq.read_table(source.episode_parquet(episode), columns=[keys.state_key, "timestamp"])
    states = np.asarray(table[keys.state_key].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if indices[-1] >= len(states) or states.shape[1:] != actions.shape[1:]:
        raise ValueError("candidate indices or dimensions do not match the recorded episode")
    videos = source.episode_video_paths(episode, keys.video_keys)
    if set(videos) != set(keys.video_keys):
        raise FileNotFoundError("recorded episode requires all three YAM camera videos")

    metadata = {
        "scope": "recorded_open_loop", "hardware_executed_steps": 0,
        "action_semantics": "model_candidate_targets_not_executed",
        "feedback": "recorded; independent of predictions",
        "source_dataset_root": str(dataset_root.resolve()), "source_episode_index": episode,
        "source_action_indices": indices.tolist(),
        "source_timestamps": timestamps[indices].tolist(),
        "candidate_id": prediction["candidate_id"],
        "observation": prediction["observation"], "control_hz": fps, "task": task.strip(),
        "task_definition": task_definition,
    }
    output.mkdir(parents=True)
    recorder = EpisodeLogger(output, robot, int(fps), keys, convert_bgr_to_rgb=False)
    recorder.start_episode(task.strip())
    recorder.set_episode_meta(**metadata)
    with ExitStack() as stack:
        readers = {}
        for camera, path in videos.items():
            decoder = cast(Generator[np.ndarray, None, None], video_frames(path))
            frames = stack.enter_context(closing(decoder))
            readers[camera] = islice(frames, int(indices[0]), int(indices[-1]) + 1)
        for offset, (index, action) in enumerate(zip(indices, actions, strict=True)):
            images = {camera: next(reader, None) for camera, reader in readers.items()}
            missing = [camera for camera, image in images.items() if image is None]
            if missing:
                raise ValueError(f"source frame {index} missing from camera videos: {missing}")
            recorder.record_step(
                Observation(images=cast(dict[str, np.ndarray], images), state_qpos=states[index]),
                action, timestamp=offset / fps,
            )
    if not recorder.end_episode():
        raise RuntimeError("EpisodeLogger did not save the candidate")
    recorder.finalize()
    metadata_path = output / "meta/rsi_episode.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (output / "meta/rsi_prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2) + "\n",
    )
    saved = LeRobotDatasetIO(output)
    return {
        "status": "saved", "dataset_root": str(output.resolve()), "episode_index": 0,
        "frames": len(actions), "parquet": str(saved.episode_parquet(0).resolve()),
        "videos": {key: str(path.resolve()) for key, path in
                   saved.episode_video_paths(0, keys.video_keys).items()},
        "metadata": str(metadata_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file", type=Path)
    args = parser.parse_args()
    definition = (json.loads(args.task_file.read_text(encoding="utf-8")) if args.task_file
                  else {"prompt_en": args.task})
    result = export_episode(
        args.dataset_root, args.episode, args.predictions, args.output, definition,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
