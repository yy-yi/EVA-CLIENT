[← Back to README](../README.md)

# 📦 Install

Requirements:

- Python 3.11 (the default project environment uses Python 3.11.15)
- [uv](https://github.com/astral-sh/uv) for dependency management
- ROS Noetic — **only** for the `ros1` transport; `ros2` / `zmq` / `dataset` need no ROS

```bash
git clone https://github.com/Noietch/EVA-CLIENT.git
cd EVA-CLIENT
curl -LsSf https://astral.sh/uv/install.sh | sh

# Standard install (non-ROS backends)
uv sync --extra dev
source .venv/bin/activate
```

<details>
<summary><b>ROS1 install</b> — expose the system ROS install to the venv</summary>

```bash
uv venv --python 3.11.15 --system-site-packages .venv
uv sync --extra dev
```

</details>

Optional hardware extras:

| Extra    | Use it for                                                                    |
|----------|-------------------------------------------------------------------------------|
| `dev`    | test / lint / type-check tooling — always recommended                         |
| `franka` | Franka arms (`dual_franka`)                                                   |
| `arx`    | ARX R5 arm (`arx_r5`); vendored driver in-tree                                |
| `ur5e`   | UR5e arm with DH gripper; vendored driver in-tree                             |
| `camera` | Orbbec depth cameras (Linux x86_64/aarch64 + macOS arm64 wheels; Python 3.11) |
| `viz`    | Faster 3D robot-model loading in the web view                                 |

Extras stack; for ARX development, run `uv sync --extra dev --extra arx`.

Verify the install with no ROS and no policy server:

```bash
eva --config configs/00_openloop/dual_agilex_piper_openloop.py
```
