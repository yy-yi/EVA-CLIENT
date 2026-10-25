"""Dual YAM RSI workspace with recorded RGB/state and an owned local backend."""

import datetime
import json
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

_base_ = ["../00_base/defaults.py"]

__agent_root = pathlib.Path(os.environ.get(
    "EVA_RSI_ROOT", pathlib.Path("{{ fileDirname }}/yam.py").resolve().parents[4],
)).expanduser().resolve()
__runtime_path = pathlib.Path(os.environ.get(
    "EVA_RSI_CONFIG", __agent_root / "configs/rsi/yam.json",
)).expanduser().resolve()
__runtime = json.loads(__runtime_path.read_text())
__output = (__agent_root / "work_dirs/rsi"
            / ("yam_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")))
__command = [str(__agent_root / ".venv/bin/python"), str(__agent_root / "tools/serve_rsi.py"),
             "--config", str(__runtime_path), "--output", str(__output)]
if os.environ.get("EVA_RSI_CREDENTIALS"):
    __command += ["--credentials", str(pathlib.Path(
        os.environ["EVA_RSI_CREDENTIALS"],
    ).expanduser().resolve())]

rsi = dict(url=f"http://127.0.0.1:{__runtime['service_port']}", command=__command,
           cwd=str(__agent_root), start_timeout_s=60.0,
           tasks_dir=str(__agent_root / "work_dirs/rsi/client"),
           initial_tasks=__runtime.get("tasks", []))
console = dict(initial_tab="rsi")
robot = dict(type="dual_yam")
rollout = dict(storage=dict(enabled=False))
inference_cfg = dict(obs_space=dict(type="JointState"), action_space=dict(type="JointState"),
                     setup_warmup_chunks=0)
