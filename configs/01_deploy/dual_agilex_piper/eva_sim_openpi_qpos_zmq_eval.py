"""Dual Piper: EVA Sim evaluation strategy overrides."""

_base_ = ["eva_sim_openpi_qpos_zmq.py"]

inference_strategies = dict(
    sync=dict(
        args=dict(
            execute_horizon=5,
            sync_wait_max_ticks=0,
        ),
    ),
)
