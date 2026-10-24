"""RSI export reuses the actual episode writer and dataset/video readers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from core.config import ConfigDict
from core.recorder.episode import EpisodeLogger
from core.registry import ROBOT_REGISTRY
from core.types import Observation
from core.utils.lerobot import LeRobotDatasetIO
from tools.conversion.source import video_frames
from tools.rsi_export_episode import export_episode

pytestmark = pytest.mark.integration


@pytest.fixture
def recorded_candidate(tmp_path):
    source = tmp_path / "source"
    robot = ROBOT_REGISTRY.build("dual_yam")
    video_keys = {
        camera.observation_key: f"observation.images.{camera.observation_key}"
        for camera in robot.observation_schema.cameras
    }
    keys = ConfigDict(
        state_key="observations.state.qpos", action_key="action.qpos",
        eef_key="observations.state.eef", video_keys=video_keys,
    )
    recorder = EpisodeLogger(source, robot, 30, keys, convert_bgr_to_rgb=False)
    states = np.arange(4 * 14, dtype=np.float32).reshape(4, 14) / 100
    recorder.start_episode("recorded source task")
    for index, state in enumerate(states):
        images = {
            camera: np.full((32, 32, 3), (40 + index * 20, 80 + offset * 20, 150), np.uint8)
            for offset, camera in enumerate(video_keys)
        }
        recorder.record_step(Observation(images=images, state_qpos=state), state + 3,
                             timestamp=index / 30)
    assert recorder.end_episode()
    recorder.finalize()
    actions = states[1:] - 0.05
    candidate = {
        "source": "agent_reviewed_sandbox_candidate", "candidate_id": "candidate_test",
        "actions": actions.tolist(), "action_indices": [1, 2, 3], "control_hz": 30,
        "hardware_executed_steps": 0, "feedback": "recorded; independent of predictions",
        "observation": {"frame_id": 1, "world_instance_id": "recorded_test"},
    }
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(candidate))
    return source, predictions, states, actions, video_keys


def test_export_real_episode_writer_round_trip(recorded_candidate, tmp_path):
    source, predictions, states, actions, video_keys = recorded_candidate
    output = tmp_path / "export"
    task = {"task_id": "drawer_1", "prompt_en": "Approach the drawer", "prompt_zh": "靠近抽屉"}
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(task))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[2] / "tools/rsi_export_episode.py"),
         "--dataset-root", str(source), "--episode", "0", "--predictions", str(predictions),
         "--output", str(output), "--task-file", str(task_file)],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "saved" and receipt["frames"] == 3
    assert Path(receipt["dataset_root"]).is_absolute()
    store = LeRobotDatasetIO(output)
    table = pq.read_table(store.episode_parquet(0))
    np.testing.assert_array_equal(table["observations.state.qpos"].to_pylist(), states[1:])
    np.testing.assert_array_equal(table["action.qpos"].to_pylist(), actions)
    np.testing.assert_allclose(table["timestamp"].to_pylist(), np.arange(3) / 30, atol=1e-8)
    loaded_actions, loaded_task = store.load_trajectory(0, "action.qpos")
    np.testing.assert_array_equal(loaded_actions, actions)
    assert loaded_task == task["prompt_en"]
    info = json.loads((output / "meta/info.json").read_text())
    assert (info["total_frames"], info["total_videos"], info["robot_type"]) == (3, 3, "dual_yam")
    row = json.loads((output / "meta/episodes.jsonl").read_text())
    assert row["scope"] == "recorded_open_loop" and row["hardware_executed_steps"] == 0
    assert row["source_action_indices"] == [1, 2, 3]
    assert row["task_definition"] == task
    assert row["action_semantics"] == "model_candidate_targets_not_executed"
    assert json.loads(Path(receipt["metadata"]).read_text())["candidate_id"] == "candidate_test"
    source_videos = LeRobotDatasetIO(source).episode_video_paths(0, video_keys)
    for camera, output_video in store.episode_video_paths(0, video_keys).items():
        frames = list(video_frames(output_video))
        original = list(video_frames(source_videos[camera]))
        assert len(frames) == 3 and all(frame.shape == (32, 32, 3) for frame in frames)
        # H.264 recompression is lossy; tolerate small pixel changes, preserving RGB order.
        for saved, observed in zip(frames, original[1:], strict=True):
            assert np.abs(saved.astype(float).mean((0, 1)) - observed.mean((0, 1))).max() < 8
    with pytest.raises(FileExistsError):
        export_episode(source, 0, predictions, output, task)


@pytest.mark.parametrize("review", [{"passed": False, "subgoal": "handle grasp failed"}, None])
def test_unexported_trial_collection_data(recorded_candidate, tmp_path, review):
    source, predictions, states, actions, video_keys = recorded_candidate
    payload = json.loads(predictions.read_text())
    payload.update(source="unexported_sandbox_trial", candidate_id=None,
                   trial_id="trial_failed_grasp", review=review)
    predictions.write_text(json.dumps(payload))
    output = tmp_path / "trial_export"
    receipt = export_episode(source, 0, predictions, output, {"prompt_en": "Approach the handle"})
    table = pq.read_table(receipt["parquet"])
    np.testing.assert_array_equal(table["observations.state.qpos"].to_pylist(), states[1:])
    np.testing.assert_array_equal(table["action.qpos"].to_pylist(), actions)
    metadata = json.loads(Path(receipt["metadata"]).read_text())
    assert metadata["prediction_source"] == "unexported_sandbox_trial"
    assert metadata["candidate_id"] is None and metadata["trial_id"] == "trial_failed_grasp"
    assert metadata["review"] == review and metadata["hardware_executed_steps"] == 0
    assert metadata["action_semantics"] == "model_sandbox_trial_targets_not_hardware_executed"
    row = json.loads((output / "meta/episodes.jsonl").read_text())
    assert row["review"] == review and row["candidate_id"] is None
    assert json.loads((output / "meta/rsi_prediction.json").read_text()) == payload
    assert set(receipt["videos"]) == set(video_keys)
    for path in receipt["videos"].values():
        frames = list(video_frames(Path(path)))
        assert len(frames) == 3 and all(frame.shape == (32, 32, 3) for frame in frames)


@pytest.mark.parametrize("changes", [
    {"actions": [], "action_indices": []},
    {"action_indices": [0, 2, 3]},
    {"action_indices": [2, 3, 4]},
    {"control_hz": 15},
    {"hardware_executed_steps": 1},
    {"candidate_id": None},
    {"source": "unexported_sandbox_trial", "candidate_id": None, "trial_id": ""},
    {"source": "unexported_sandbox_trial", "trial_id": "trial"},
    {"source": "unexported_sandbox_trial", "candidate_id": None,
     "trial_id": "trial", "review": False},
])
def test_invalid_candidate_never_creates_episode(recorded_candidate, tmp_path, changes):
    source, predictions, _, _, _ = recorded_candidate
    payload = json.loads(predictions.read_text())
    payload.update(changes)
    predictions.write_text(json.dumps(payload))
    output = tmp_path / "invalid_export"
    with pytest.raises(ValueError):
        export_episode(source, 0, predictions, output, {"prompt_en": "test task"})
    assert not output.exists()
