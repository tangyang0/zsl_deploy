#!/usr/bin/env python3
"""Read one ZSL-1 state and preview the same policy used by deployment.

This script never calls ``sendMotorCmd``. It still opens an SDK receive
connection, so do not run it alongside another program using the same SDK.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from policy_deploy import ACTION_SCALE, DEFAULT_MODEL, DEFAULT_Q, build_obs, projected_gravity, validate_session

DEFAULT_SDK_LIB = "/home/tangyang/workspace/genisom_l1_sdk_old/lib/zsl-1/x86_64"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sdk-lib", default=DEFAULT_SDK_LIB)
    p.add_argument("--local-ip", default="192.168.234.16")
    p.add_argument("--dog-ip", default="192.168.234.1")
    p.add_argument("--port", type=int, default=43988)
    return p.parse_args()


def main(cli):
    model = Path(cli.model).expanduser().resolve()
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name, output_name = validate_session(session)
    sys.path.insert(0, cli.sdk_lib)
    import mc_sdk_zsl_1_py as sdk

    robot = sdk.LowLevel()
    robot.initRobot(cli.local_ip, cli.port, cli.dog_ip)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if robot.checkConnect() and robot.haveMotorData():
            state = robot.getMotorState()
            q = np.stack(
                [np.asarray(state.q_abad, np.float32), np.asarray(state.q_hip, np.float32), np.asarray(state.q_knee, np.float32)],
                axis=1,
            ).reshape(-1)
            qd = np.stack(
                [np.asarray(state.qd_abad, np.float32), np.asarray(state.qd_hip, np.float32), np.asarray(state.qd_knee, np.float32)],
                axis=1,
            ).reshape(-1)
            obs = build_obs(q, qd, np.asarray(robot.getBodyGyro(), np.float32), projected_gravity(robot.getQuaternion()), np.zeros(3, np.float32), np.zeros(12, np.float32))
            action = np.asarray(session.run([output_name], {input_name: obs[None, :]})[0][0], np.float32)
            if action.shape != (12,) or not np.isfinite(action).all():
                raise ValueError(f"策略输出异常：shape={action.shape}")
            print("model:", model)
            print("obs shape:", obs.shape)
            print("action:", np.round(action, 4))
            print("q_des:", np.round(DEFAULT_Q + ACTION_SCALE * action, 4))
            print("策略预览完成，没有发送任何电机命令。")
            return
        time.sleep(0.1)
    raise RuntimeError("10 秒内没有收到有效电机状态")


if __name__ == "__main__":
    main(parse_args())
