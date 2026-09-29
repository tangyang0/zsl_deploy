#!/usr/bin/env python3
"""Receive-only connectivity and state test for a ZSL-1."""

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk-lib", default="/home/tangyang/workspace/genisom_l1_sdk_old/lib/zsl-1/x86_64")
    p.add_argument("--local-ip", default="192.168.234.16")
    p.add_argument("--dog-ip", default="192.168.234.1")
    p.add_argument("--port", type=int, default=43988)
    p.add_argument("--timeout", type=float, default=10.0)
    return p.parse_args()


def main(args):
    sys.path.insert(0, args.sdk_lib)
    import mc_sdk_zsl_1_py

    robot = mc_sdk_zsl_1_py.LowLevel()
    robot.initRobot(args.local_ip, args.port, args.dog_ip)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        connected = robot.checkConnect()
        has_data = robot.haveMotorData()
        if connected and has_data:
            state = robot.getMotorState()
            print("checkConnect:", connected)
            print("haveMotorData:", has_data)
            print("q_abad:", list(state.q_abad))
            print("q_hip :", list(state.q_hip))
            print("q_knee:", list(state.q_knee))
            print("quaternion:", list(robot.getQuaternion()))
            print("body gyro:", list(robot.getBodyGyro()))
            print("body acc :", list(robot.getBodyAcc()))
            return
        time.sleep(0.1)
    print("checkConnect:", connected)
    print("haveMotorData:", has_data)
    raise RuntimeError(f"{args.timeout:.1f} 秒内没有收到电机状态数据")


if __name__ == "__main__":
    main(parse_args())
