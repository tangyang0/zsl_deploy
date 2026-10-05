#!/usr/bin/env python3
"""Run a RobotLab ZSL-1 velocity policy through the legacy ZSL-1 SDK.

The live controller is a small state machine: keep the startup pose in
low-level damping, press ``s`` for the stand-up transition, then press ``t``
to start ONNX policy control. ``d`` cancels/ends safely by switching the
current pose to damping. In policy control the arrow keys request ±0.2 m/s
and an inactive/released key means zero velocity.

The policy is evaluated at 50 Hz and the latest position target is sent at
500 Hz by a separate sender thread.

Experimental variant: raw policy actions are NOT amplitude-clipped.
Joint targets are clamped to the SDK's hard per-joint windows so
sendMotorCmd never rejects them. Action scales, finite-output checks, and
the damping state machine are retained.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import onnxruntime as ort


DEFAULT_SDK_LIB = "/home/tangyang/workspace/genisom_l1_sdk_old/lib/zsl-1/x86_64"
DEFAULT_MODEL = ("/home/tangyang/workspace/robot_lab/logs/rsl_rl/zsibot_zsl1_flat/ty_2026-09-23_18-56-21_smooth_ft/exported/policy.onnx")
LOCAL_IP = "192.168.234.16"
DOG_IP = "192.168.234.1"
PORT = 43988
SEND_DT = 0.002
POLICY_DT = 0.020
TRANSITION_DT = 2.0
LOW_SPEED = 0.5
DAMPING_KD = 3.0

# RobotLab joint order: [FAR, FBL, RAR, RBL] = [FR, FL, RR, RL].
DEFAULT_Q = np.tile(np.array([0.0, 0.8, -1.5], dtype=np.float32), 4)
# Intermediate target used only by the SDK's LowLevel stand-up transition.
# It is deliberately not used as a damping target: on this robot it can lift
# the legs briefly when the robot is already lying on the ground.
STANDUP_INTERMEDIATE_Q = np.tile(np.array([0.0, 1.4, -2.4], dtype=np.float32), 4)
ACTION_SCALE = np.tile(np.array([0.125, 0.25, 0.25], dtype=np.float32), 4)
# Hard joint windows enforced by the SDK inside sendMotorCmd ("invalid
# {abad,hip,knee} cmd, expect ... rad"); targets outside are rejected with
# return -1 and the sender thread aborts. Policy targets are clamped here so
# unclipped actions can never trip that check.
Q_LIMIT_LO = np.tile(np.array([-0.48, -1.15, -2.9], dtype=np.float32), 4)
Q_LIMIT_HI = np.tile(np.array([0.48, 2.97, -0.65], dtype=np.float32), 4)

# Initial gains; this experimental variant does not clip policy actions.
POLICY_KP = 20.0
POLICY_KD = 0.7
TRANSITION_KP = 80.0
TRANSITION_KD = 1.0
STATE_DAMPING = "damping"
STATE_STANDING = "standing"
STATE_TEST = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ONNX policy path")
    parser.add_argument("--sdk-lib", default=DEFAULT_SDK_LIB, help="Directory containing mc_sdk_zsl_1_py")
    parser.add_argument("--local-ip", default=LOCAL_IP)
    parser.add_argument("--dog-ip", default=DOG_IP)
    parser.add_argument("--port", type=int, default=PORT)
    # Hidden compatibility options for old launch files. Live mode always
    # uses the state-machine keyboard controller below.
    parser.add_argument("--keyboard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"), default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--key-timeout", type=float, default=0.15,
        help="方向键最后一次事件后的保持时间（秒，默认 0.15）",
    )
    parser.add_argument("--kp", type=float, default=POLICY_KP, help="Policy position gain")
    parser.add_argument("--kd", type=float, default=POLICY_KD, help="Policy velocity gain")
    parser.add_argument("--transition-kp", type=float, default=TRANSITION_KP)
    parser.add_argument("--transition-kd", type=float, default=TRANSITION_KD)
    parser.add_argument(
        "--stop-hold-seconds", type=float, default=3.0,
        help="退出时继续发送当前姿态阻尼命令的时间（秒）",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and infer once without SDK/robot")
    return parser.parse_args()


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dtype=np.float32
    )


def projected_gravity(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        raise ValueError("四元数必须是 4 个有限数")
    norm = float(np.linalg.norm(q))
    if norm < 1e-3:
        raise ValueError("收到无效的零四元数")
    q = q / norm
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    gravity_world = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    return quat_mul(quat_mul(q_conj, gravity_world), q)[1:]


def _array4(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != 4 or not np.isfinite(result).all():
        raise ValueError(f"SDK 的 {name} 必须是 4 个有限数")
    return result


def read_q(robot) -> np.ndarray:
    state = robot.getMotorState()
    return np.stack(
        [_array4(state.q_abad, "q_abad"), _array4(state.q_hip, "q_hip"), _array4(state.q_knee, "q_knee")],
        axis=1,
    ).reshape(-1)


def read_q_qd_imu(robot):
    state = robot.getMotorState()
    q = np.stack(
        [_array4(state.q_abad, "q_abad"), _array4(state.q_hip, "q_hip"), _array4(state.q_knee, "q_knee")],
        axis=1,
    ).reshape(-1)
    qd = np.stack(
        [_array4(state.qd_abad, "qd_abad"), _array4(state.qd_hip, "qd_hip"), _array4(state.qd_knee, "qd_knee")],
        axis=1,
    ).reshape(-1)
    gyro = np.asarray(robot.getBodyGyro(), dtype=np.float32).reshape(-1)
    if gyro.size != 3 or not np.isfinite(gyro).all():
        raise ValueError("SDK 的 body gyro 必须是 3 个有限数")
    return q, qd, gyro, projected_gravity(robot.getQuaternion())


def make_command(sdk, q_des: np.ndarray, kp: float, kd: float):
    q_des = np.asarray(q_des, dtype=np.float32).reshape(-1)
    if q_des.size != 12 or not np.isfinite(q_des).all():
        raise ValueError("目标关节角必须是 12 个有限数")
    if not np.isfinite([kp, kd]).all() or kp < 0.0 or kd < 0.0:
        raise ValueError("kp/kd 必须是非负有限数")
    cmd = sdk.MotorCommand()
    for i in range(4):
        j = 3 * i
        cmd.q_des_abad[i] = float(q_des[j])
        cmd.q_des_hip[i] = float(q_des[j + 1])
        cmd.q_des_knee[i] = float(q_des[j + 2])
        cmd.kp_abad[i] = cmd.kp_hip[i] = cmd.kp_knee[i] = kp
        cmd.kd_abad[i] = cmd.kd_hip[i] = cmd.kd_knee[i] = kd
    return cmd


def send_q(robot, sdk, q_des, kp, kd):
    ret = robot.sendMotorCmd(make_command(sdk, q_des, kp, kd))
    if ret is not None and ret < 0:
        raise RuntimeError(f"sendMotorCmd failed: {ret}")


def wait_for_state(robot, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if robot.checkConnect() and robot.haveMotorData():
            read_q_qd_imu(robot)
            return
        time.sleep(0.1)
    raise RuntimeError("10 秒内没有收到机器人状态数据")


def check_live(robot):
    if not robot.checkConnect() or not robot.haveMotorData():
        raise RuntimeError("机器人连接或状态数据已失效")


def transition(robot, sdk, start_q, target_q, duration, kp, kd, keyboard=None):
    if duration <= 0.0:
        send_q(robot, sdk, target_q, kp, kd)
        return True
    print(f"姿态过渡开始，耗时 {duration:.1f} 秒")
    t0 = time.monotonic()
    next_tick = t0
    last_check = t0
    while True:
        now = time.monotonic()
        if keyboard is not None:
            stop, events = keyboard.poll()
            if stop or "DAMPING" in events:
                print("过渡过程中收到退出/阻尼请求，停止当前过渡")
                return False
        if now - last_check >= 0.1:
            check_live(robot)
            last_check = now
        ratio = min((now - t0) / duration, 1.0)
        send_q(robot, sdk, (1.0 - ratio) * start_q + ratio * target_q, kp, kd)
        if ratio >= 1.0:
            return True
        next_tick += SEND_DT
        time.sleep(max(0.0, next_tick - time.monotonic()))


def build_obs(q, qd, gyro, gravity, command, last_action) -> np.ndarray:
    parts = [np.asarray(x, dtype=np.float32).reshape(-1) for x in (q, qd, gyro, gravity, command, last_action)]
    if [x.size for x in parts] != [12, 12, 3, 3, 3, 12]:
        raise ValueError("observation 各段尺寸不符合 ZSL-1 policy")
    obs = np.concatenate([parts[2] * 0.25, parts[3], parts[4], parts[0] - DEFAULT_Q, parts[1] * 0.05, parts[5]])
    if not np.isfinite(obs).all():
        raise ValueError("observation 包含 NaN 或 Inf")
    return obs.astype(np.float32)


def validate_session(session):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"策略必须是单输入单输出，实际为 {len(inputs)} 输入/{len(outputs)} 输出")
    input_info, output_info = inputs[0], outputs[0]
    if input_info.shape[-1] != 45 or output_info.shape[-1] != 12:
        raise ValueError(f"策略接口应为 [1,45] -> [1,12]，实际为 {input_info.shape} -> {output_info.shape}")
    action = np.asarray(session.run([output_info.name], {input_info.name: np.zeros((1, 45), np.float32)})[0])
    if action.shape != (1, 12) or not np.isfinite(action).all():
        raise ValueError(f"策略预检输出异常：shape={action.shape}")
    return input_info.name, output_info.name


class KeyboardCommand:
    """POSIX keyboard controller for the deployment state machine.

    A POSIX terminal does not report key-release events. Terminal key-repeat
    events therefore refresh a short activity timeout; once it expires the
    command is zero. This gives the requested release-to-zero behavior while
    still allowing a held arrow key to keep moving.
    """

    _steps = {
        "UP": np.array([LOW_SPEED, 0.0, 0.0], np.float32),
        "DOWN": np.array([-LOW_SPEED, 0.0, 0.0], np.float32),
        # The SDK examples use +vy for left and -vy for right.
        "LEFT": np.array([0.0, LOW_SPEED, 0.0], np.float32),
        "RIGHT": np.array([0.0, -LOW_SPEED, 0.0], np.float32),
    }
    _escape_keys = {b"\x1b[A": "UP", b"\x1b[B": "DOWN", b"\x1b[C": "RIGHT", b"\x1b[D": "LEFT"}
    _byte_keys = {b"8": "UP", b"2": "DOWN", b"4": "LEFT", b"6": "RIGHT"}
    _event_keys = {b"s": "STANDUP", b"t": "TEST", b"d": "DAMPING", b"x": "EXIT"}

    def __init__(self, hold_timeout=0.15):
        if not np.isfinite(hold_timeout) or hold_timeout <= 0.0:
            raise ValueError("--key-timeout 必须是正数")
        self.hold_timeout = float(hold_timeout)
        self.command = np.zeros(3, dtype=np.float32)
        self._active = {}
        self._buffer = bytearray()
        self._old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("实时部署需要在可交互终端中运行")
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print("状态：阻尼；按 s 进入 standup 过渡，完成后按 t 进入测试")
        print("d：放弃/停止测试，切换当前姿态阻尼并自然下趴；X/Ctrl+C 也是安全退出")
        print("测试控制：↑/↓ 前进/后退，←/→ 左移/右移，速度 ±0.2 m/s；松开即归零")
        return self

    def __exit__(self, *_):
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _register(self, key, now, events):
        mapped = self._event_keys.get(key)
        if mapped is not None:
            events.add(mapped)
            return
        mapped = self._byte_keys.get(key)
        if mapped is not None:
            self._active[mapped] = now

    def _refresh_command(self, now):
        expired = [key for key, stamp in self._active.items() if now - stamp > self.hold_timeout]
        for key in expired:
            del self._active[key]
        self.command[:] = 0.0
        for key in self._active:
            self.command += self._steps[key]

    def clear_motion(self, clear_buffer=False):
        """Forget active directions, optionally discarding pending input."""
        self._active.clear()
        if clear_buffer:
            self._buffer.clear()
        self.command[:] = 0.0

    def poll(self):
        events = set()
        while select.select([sys.stdin], [], [], 0.0)[0]:
            chunk = os.read(sys.stdin.fileno(), 64)
            if not chunk:
                break
            self._buffer.extend(chunk)
        now = time.monotonic()
        while self._buffer:
            if self._buffer[0:1] == b"\x1b":
                # Keep an incomplete escape sequence for the next poll.
                if len(self._buffer) < 3:
                    break
                encoded = bytes(self._buffer[:3])
                key = self._escape_keys.get(encoded)
                del self._buffer[:3]
                if key is not None:
                    self._active[key] = now
                continue
            key = bytes(self._buffer[:1]).lower()
            del self._buffer[:1]
            if key == b" ":
                self.clear_motion()
                continue
            self._register(key, now, events)
        self._refresh_command(now)
        return "EXIT" in events, events


class CommandSender:
    """Send the latest target independently of ONNX inference timing."""

    def __init__(self, robot, sdk, initial_q, kp, kd):
        self.robot, self.sdk = robot, sdk
        self.q_des = np.asarray(initial_q, dtype=np.float32).copy()
        self.kp, self.kd = kp, kd
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="zsl1-command-sender", daemon=True)

    def start(self):
        self._thread.start()

    def set_target(self, q_des):
        q_des = np.asarray(q_des, dtype=np.float32).reshape(-1)
        if q_des.size != 12 or not np.isfinite(q_des).all():
            raise ValueError("发送目标必须是 12 个有限数")
        with self._lock:
            self.q_des = q_des.copy()

    def _run(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                with self._lock:
                    q_des = self.q_des.copy()
                send_q(self.robot, self.sdk, q_des, self.kp, self.kd)
            except BaseException as exc:
                self.error = exc
                self._stop.set()
                return
            next_tick += SEND_DT
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def hold_damping(robot, sdk, q_des, seconds):
    if seconds <= 0.0:
        return
    print(f"已进入阻尼，继续发送阻尼命令 {seconds:.1f} 秒后退出")
    deadline, next_tick = time.monotonic() + seconds, time.monotonic()
    while time.monotonic() < deadline:
        try:
            send_q(robot, sdk, q_des, 0.0, DAMPING_KD)
        except Exception as exc:
            print(f"阻尼命令发送失败：{exc}", file=sys.stderr)
            return
        next_tick += SEND_DT
        time.sleep(max(0.0, next_tick - time.monotonic()))


def safe_damping_shutdown(robot, sdk, seconds):
    """Switch to damping at the current pose without commanding a lift."""
    try:
        current_q = read_q(robot)
    except Exception as exc:
        print(f"无法读取退出姿态，未发送阻尼命令：{exc}", file=sys.stderr)
        return
    print("停止策略，保持当前关节目标并进入阻尼；不再主动抬升")
    hold_damping(robot, sdk, current_q, seconds)


def main(args: argparse.Namespace):
    print("实验版本：策略动作无限幅；关节目标 = 默认姿态 + ACTION_SCALE × 原始动作，并按 SDK 关节限位裁剪")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 ONNX 模型：{model_path}")
    if args.key_timeout <= 0.0 or not np.isfinite(args.key_timeout):
        raise ValueError("--key-timeout 必须是正数")
    if args.stop_hold_seconds < 0.0 or not np.isfinite(args.stop_hold_seconds):
        raise ValueError("--stop-hold-seconds 不能为负数")
    if args.command is not None:
        legacy_command = np.asarray(args.command, dtype=np.float32)
        if not np.isfinite(legacy_command).all() or np.any(np.abs(legacy_command) > 1.0):
            raise ValueError("--command 的三个值必须在 [-1, 1] 内")
        print("警告：实时模式已改为键盘状态机，忽略旧参数 --command", file=sys.stderr)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name, output_name = validate_session(session)
    print(f"ONNX 预检通过：[1,45] -> [1,12]，模型：{model_path}")
    if args.dry_run:
        print("dry-run 完成，未加载 SDK，未连接机器人，未发送电机命令")
        return

    sys.path.insert(0, args.sdk_lib)
    import mc_sdk_zsl_1_py as sdk

    robot = sdk.LowLevel()
    robot.initRobot(args.local_ip, args.port, args.dog_ip)
    sender = None
    state = STATE_DAMPING
    try:
        wait_for_state(robot)
        print("SDK 已连接，收到有效电机/IMU 状态")
        start_q = read_q(robot)
        # kp=0, kd=3 is the damping command used by the legacy LowLevel demo.
        # Keep the pose that was already present: using an active low-pose
        # target here could lift a robot that is already lying on the ground.
        # A single LowLevel connection is used because the SDK forbids using
        # HighLevel and LowLevel concurrently.
        sender = CommandSender(robot, sdk, start_q, 0.0, DAMPING_KD)
        sender.start()
        print("当前状态：阻尼；按 s 开始 standup 过渡")

        keyboard = KeyboardCommand(args.key_timeout)
        with keyboard:
            last_action = np.zeros(12, dtype=np.float32)
            command = np.zeros(3, dtype=np.float32)
            q_des = DEFAULT_Q.copy()
            clip_steps = 0
            next_infer = time.monotonic()
            last_check = next_infer
            last_print = 0.0
            while True:
                now = time.monotonic()
                if sender is not None and sender.error is not None:
                    raise RuntimeError(f"发送线程停止：{sender.error}")
                stop, events = keyboard.poll()
                if stop:
                    print("收到 X，退出控制")
                    break

                if state == STATE_DAMPING and "STANDUP" in events:
                    keyboard.clear_motion(clear_buffer=True)
                    sender.stop()
                    sender = None
                    start_q = read_q(robot)
                    print("收到 s：开始 standup 过渡")
                    completed = transition(
                        robot, sdk, start_q, STANDUP_INTERMEDIATE_Q, TRANSITION_DT,
                        args.transition_kp, args.transition_kd, keyboard,
                    )
                    if completed:
                        completed = transition(
                            robot, sdk, STANDUP_INTERMEDIATE_Q, DEFAULT_Q, TRANSITION_DT,
                            args.transition_kp, args.transition_kd, keyboard,
                        )
                    if not completed:
                        break
                    sender = CommandSender(robot, sdk, DEFAULT_Q, args.transition_kp, args.transition_kd)
                    sender.start()
                    state = STATE_STANDING
                    keyboard.clear_motion(clear_buffer=True)
                    print("standup 完成；当前保持站立，按 t 进入测试")
                elif state == STATE_DAMPING and "TEST" in events:
                    print("当前仍是阻尼状态，请先按 s 完成 standup")
                elif state in (STATE_STANDING, STATE_TEST) and "DAMPING" in events:
                    keyboard.clear_motion(clear_buffer=True)
                    print("收到 d：安全停止，切换当前姿态阻尼并自然下趴")
                    break
                elif state == STATE_STANDING and "TEST" in events:
                    keyboard.clear_motion(clear_buffer=True)
                    sender.stop()
                    sender = CommandSender(robot, sdk, DEFAULT_Q, args.kp, args.kd)
                    sender.start()
                    state = STATE_TEST
                    last_action[:] = 0.0
                    command[:] = 0.0
                    clip_steps = 0
                    next_infer = now
                    print("收到 t：进入策略测试；无方向键时 command=0")

                command = keyboard.command.copy()
                if now - last_check >= 0.1:
                    check_live(robot)
                    last_check = now
                if state == STATE_TEST and now >= next_infer:
                    q, qd, gyro, gravity = read_q_qd_imu(robot)
                    obs = build_obs(q, qd, gyro, gravity, command, last_action)
                    raw_action = np.asarray(session.run([output_name], {input_name: obs[None, :]})[0][0], np.float32)
                    if raw_action.shape != (12,) or not np.isfinite(raw_action).all():
                        raise ValueError(f"策略输出异常：shape={raw_action.shape}")
                    last_action = raw_action.copy()
                    q_raw = DEFAULT_Q + ACTION_SCALE * last_action
                    q_des = np.clip(q_raw, Q_LIMIT_LO, Q_LIMIT_HI)
                    if not np.array_equal(q_des, q_raw):
                        clip_steps += 1
                    sender.set_target(q_des)
                    next_infer += POLICY_DT
                    if now - next_infer > POLICY_DT:
                        next_infer = now + POLICY_DT
                    if now - last_print > 1.0:
                        print(f"state  : {state}")
                        print(f"command: {np.round(command, 3)}")
                        print(f"action : {np.round(last_action, 3)}")
                        print(f"q_des  : {np.round(q_des, 3)}")
                        if clip_steps:
                            print(f"clip   : {clip_steps} 个控制周期触及 SDK 关节限位")
                        last_print = now
                time.sleep(0.0005)
    finally:
        if sender is not None:
            sender.stop()
        # Every exit path, including Ctrl+C/X and an exit before `t`, switches
        # to damping at the measured current pose. It never commands the
        # stand-up intermediate target during shutdown.
        safe_damping_shutdown(robot, sdk, args.stop_hold_seconds)


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        main(cli_args)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在停止")
    except Exception as exc:
        print(f"部署失败：{exc}", file=sys.stderr)
        raise
