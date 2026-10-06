#!/usr/bin/env python3
"""Run a RobotLab ZSL-1 velocity policy through the legacy ZSL-1 SDK.

The live controller is a small state machine: keep the startup pose in
low-level damping, press ``s`` for the stand-up transition, then press ``t``
to start ONNX policy control. ``d`` cancels/ends safely by switching the
current pose to damping. In policy control the arrow keys request linear
velocity and Z/C request left/right yaw rate; an inactive/released key
means zero command.

The policy is evaluated at 50 Hz and the latest position target is sent at
500 Hz by a separate sender thread.

Experimental variant: raw policy actions are NOT amplitude-clipped.
Joint targets are clamped to the SDK's hard per-joint windows so
sendMotorCmd never rejects them. Action scales, finite-output checks, and
the damping state machine are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort


# SDK and models are bundled in this repo; the SDK directory is selected by
# machine architecture so the same tree runs on the x86_64 host and the dog's
# aarch64 main computer without --sdk-lib.
_REPO_DIR = Path(__file__).resolve().parent
_ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(
    os.uname().machine, os.uname().machine
)
DEFAULT_SDK_LIB = str(_REPO_DIR / "sdk" / _ARCH)
DEFAULT_MODEL = str(_REPO_DIR / "models" / "smooth_ft.onnx")
LOCAL_IP = "192.168.234.16"
DOG_IP = "192.168.234.1"
PORT = 43988
SEND_DT = 0.002
POLICY_DT = 0.020
TRANSITION_DT = 2.0
# Speed caps shared by the keyboard steps and the web-stick scaling; defaults
# equal the training command limits (lin ±1.0 m/s, yaw ±1.0 rad/s). Tighten
# per run with --max-lin-speed / --max-yaw-speed.
MAX_LIN_SPEED = 1.0
MAX_YAW_SPEED = 1.0
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
# unclipped actions can never trip that check. The bounds keep a small inward
# margin: float32 cannot represent -2.9/-0.65/2.97 exactly and rounds them to
# the outside of the window, so a target pinned exactly at an edge would still
# fail the SDK's double-precision check.
Q_LIMIT_MARGIN = 1e-3
Q_LIMIT_LO = np.tile(np.array([-0.48, -1.15, -2.9], dtype=np.float32), 4) + Q_LIMIT_MARGIN
Q_LIMIT_HI = np.tile(np.array([0.48, 2.97, -0.65], dtype=np.float32), 4) - Q_LIMIT_MARGIN

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
    parser.add_argument(
        "--max-lin-speed", type=float, default=MAX_LIN_SPEED, metavar="MPS",
        help=f"键盘挡位/网页摇杆的线速度上限 m/s（默认 {MAX_LIN_SPEED}，训练指令上限 1.0）",
    )
    parser.add_argument(
        "--max-yaw-speed", type=float, default=MAX_YAW_SPEED, metavar="RADPS",
        help=f"键盘挡位/网页摇杆的角速度上限 rad/s（默认 {MAX_YAW_SPEED}，训练指令上限 1.0）",
    )
    parser.add_argument("--kp", type=float, default=POLICY_KP, help="Policy position gain")
    parser.add_argument("--kd", type=float, default=POLICY_KD, help="Policy velocity gain")
    parser.add_argument("--transition-kp", type=float, default=TRANSITION_KP)
    parser.add_argument("--transition-kd", type=float, default=TRANSITION_KD)
    parser.add_argument(
        "--stop-hold-seconds", type=float, default=3.0,
        help="退出时继续发送当前姿态阻尼命令的时间（秒）",
    )
    parser.add_argument(
        "--web-control", type=int, nargs="?", const=8080, default=None, metavar="PORT",
        help="开启内置 Web 控制页（默认端口 8080）：Retroid/手机浏览器打开，"
             "触摸双摇杆或浏览器 Gamepad API 实体摇杆，含 站起/测试/急停 按钮",
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

    _escape_keys = {b"\x1b[A": "UP", b"\x1b[B": "DOWN", b"\x1b[C": "RIGHT", b"\x1b[D": "LEFT"}
    _byte_keys = {
        b"8": "UP", b"2": "DOWN", b"4": "LEFT", b"6": "RIGHT",
        b"z": "TURN_L", b"7": "TURN_L", b"c": "TURN_R", b"9": "TURN_R",
    }
    _event_keys = {b"s": "STANDUP", b"t": "TEST", b"d": "DAMPING", b"x": "EXIT"}

    def __init__(self, hold_timeout=0.15, lin_speed=MAX_LIN_SPEED, yaw_speed=MAX_YAW_SPEED):
        if not np.isfinite(hold_timeout) or hold_timeout <= 0.0:
            raise ValueError("--key-timeout 必须是正数")
        self.hold_timeout = float(hold_timeout)
        self.command = np.zeros(3, dtype=np.float32)
        # The SDK examples use +vy for left and -vy for right.
        self._steps = {
            "UP": np.array([lin_speed, 0.0, 0.0], np.float32),
            "DOWN": np.array([-lin_speed, 0.0, 0.0], np.float32),
            "LEFT": np.array([0.0, lin_speed, 0.0], np.float32),
            "RIGHT": np.array([0.0, -lin_speed, 0.0], np.float32),
            # Yaw follows play.py's Se2Keyboard binding (Z/NUMPAD_7 = +omega_z
            # left turn, X/NUMPAD_9 = -omega_z right turn); X is already the
            # emergency exit here, so right turn moves to C.
            "TURN_L": np.array([0.0, 0.0, yaw_speed], np.float32),
            "TURN_R": np.array([0.0, 0.0, -yaw_speed], np.float32),
        }
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
        lin, yaw = self._steps["UP"][0], self._steps["TURN_L"][2]
        print(
            f"测试控制：↑/↓ 前进/后退，←/→ 左移/右移（±{lin:.2f} m/s）；"
            f"z/7 左转，c/9 右转（±{yaw:.2f} rad/s）；松开即归零"
        )
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


class WebControl:
    """Built-in web gamepad: serves a single HTML page with two touch
    joysticks plus the browser Gamepad API (so the Retroid Pocket 4's
    physical sticks work too) and receives commands over HTTP POST.
    Command values are normalized [-1, 1] and scaled in the main loop.
    """

    def __init__(self, port: int, lin_speed=MAX_LIN_SPEED, yaw_speed=MAX_YAW_SPEED):
        self.command = np.zeros(3, dtype=np.float32)
        self._scale = np.array([lin_speed, lin_speed, yaw_speed], np.float32)
        self.events: set = set()
        self.last_rx = 0.0
        self.info = {"state": "-", "command": [0.0, 0.0, 0.0], "clip": 0}
        self.lock = threading.Lock()
        ctrl = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    self._send(200, WEB_PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    with ctrl.lock:
                        self._send(200, json.dumps(ctrl.info).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    data = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self._send(400, b"bad json", "text/plain")
                    return
                if self.path == "/cmd":
                    with ctrl.lock:
                        ctrl.command[:] = (
                            float(data.get("vx", 0.0)),
                            float(data.get("vy", 0.0)),
                            float(data.get("wz", 0.0)),
                        )
                        ctrl.last_rx = time.monotonic()
                    self._send(204, b"", "text/plain")
                elif self.path == "/event":
                    event = str(data.get("event", ""))
                    if event in ("STANDUP", "TEST", "DAMPING"):
                        with ctrl.lock:
                            ctrl.events.add(event)
                    self._send(204, b"", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")

            def log_message(self, *_):
                pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="web-control").start()

    def active(self) -> bool:
        return time.monotonic() - self.last_rx < 0.3

    def scaled_command(self) -> np.ndarray:
        with self.lock:
            return np.clip(self.command, -1.0, 1.0) * self._scale

    def drain_events(self) -> set:
        with self.lock:
            events, self.events = self.events, set()
            return events

    def set_info(self, **kwargs):
        with self.lock:
            self.info.update(kwargs)


WEB_PAGE = """<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>ZSL-1 Web 控制</title>
<style>
  html,body{height:100%}
  body{margin:0;background:#1b1f24;color:#d8dee6;font-family:sans-serif;
       display:flex;flex-direction:column;height:100vh;height:100dvh;
       touch-action:none;user-select:none;overflow:hidden}
  #top{padding:6px 12px;font-size:14px;background:#11151a;flex:none}
  #top b{color:#7ee787}
  #btns{display:flex;gap:10px;padding:8px;justify-content:center;flex:none}
  button{flex:1;padding:14px 0;font-size:17px;border:none;border-radius:10px;color:#fff}
  #b_s{background:#2d7d46}#b_t{background:#2a5d9f}
  #b_d{background:#b3352c;font-weight:bold;font-size:19px}
  #pads{flex:1;min-height:0;display:flex;justify-content:space-around;align-items:center}
  .pad{border-radius:50%;background:#262c33;position:relative;border:2px solid #3a424c;
       width:min(38vw,26vh,230px);height:min(38vw,26vh,230px);
       width:min(38vw,26dvh,230px);height:min(38vw,26dvh,230px)}
  .stick{position:absolute;left:50%;top:50%;width:34%;height:34%;border-radius:50%;
         background:#5a86ff;transform:translate(-50%,-50%);opacity:.85}
</style></head><body>
<div id="top">状态: <b id="st">-</b> &nbsp;|&nbsp; 指令: <span id="cm">0.00 0.00 0.00</span>
 &nbsp;|&nbsp; <span id="src">触摸</span></div>
<div id="btns">
  <button id="b_s" onclick="sendEvent('STANDUP')">站起 (s)</button>
  <button id="b_t" onclick="sendEvent('TEST')">测试 (t)</button>
  <button id="b_d" onclick="sendEvent('DAMPING')">急停 (d)</button>
</div>
<div id="pads">
  <div class="pad" id="padL"><div class="stick" id="stickL"></div></div>
  <div class="pad" id="padR"><div class="stick" id="stickR"></div></div>
</div>
<script>
const dead = 0.08;
let L = {x:0, y:0}, R = {x:0, y:0}, padPointer = {};
function clamp(v){return Math.max(-1, Math.min(1, v));}
function dz(v){return Math.abs(v) < dead ? 0 : (v - Math.sign(v)*dead)/(1-dead);}
function bindPad(el, store){
  el.addEventListener('pointerdown', e=>{padPointer[e.pointerId]=el;el.setPointerCapture(e.pointerId);move(e);});
  el.addEventListener('pointermove', e=>{if(padPointer[e.pointerId]===el)move(e);});
  const end = e=>{if(padPointer[e.pointerId]===el){delete padPointer[e.pointerId];store.x=0;store.y=0;}};
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
  function move(e){
    const r = el.getBoundingClientRect();
    store.x = clamp((e.clientX-(r.left+r.width/2))/(r.width/2));
    store.y = clamp((e.clientY-(r.top+r.height/2))/(r.height/2));
  }
}
bindPad(document.getElementById('padL'), L);
bindPad(document.getElementById('padR'), R);
let gp = null;
window.addEventListener('gamepadconnected', e=>{gp=e.gamepad;document.getElementById('src').textContent='手柄:'+gp.id.slice(0,18);});
window.addEventListener('gamepaddisconnected', ()=>{gp=null;document.getElementById('src').textContent='触摸';});
function axes(){
  if(gp){
    const g = navigator.getGamepads()[gp.index];
    if(g) return {vx:dz(-g.axes[1]), vy:dz(g.axes[0]), wz:dz(-g.axes[2])};
  }
  return {vx:dz(-L.y), vy:dz(L.x), wz:dz(R.x)};
}
function sendEvent(ev){fetch('/event',{method:'POST',body:JSON.stringify({event:ev})});}
setInterval(()=>{
  const a = axes();
  fetch('/cmd',{method:'POST',body:JSON.stringify(a)});
  document.getElementById('cm').textContent =
    a.vx.toFixed(2)+' '+a.vy.toFixed(2)+' '+a.wz.toFixed(2);
},50);
setInterval(async()=>{
  try{ const s = await (await fetch('/state')).json();
       document.getElementById('st').textContent = s.state + (s.clip ? ' (限位'+s.clip+')' : '');
  }catch(e){}
},500);
</script></body></html>
"""


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
    for name, value in (("--max-lin-speed", args.max_lin_speed), ("--max-yaw-speed", args.max_yaw_speed)):
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError(f"{name} 必须在 (0, 1.0] 内（训练指令范围为 ±1.0）")
    if args.dry_run:
        print("dry-run 完成，未加载 SDK，未连接机器人，未发送电机命令")
        return

    sys.path.insert(0, args.sdk_lib)
    import mc_sdk_zsl_1_py as sdk

    web = None
    if args.web_control:
        web = WebControl(args.web_control, args.max_lin_speed, args.max_yaw_speed)
        print(f"Web 控制页已开启: http://<本机IP>:{args.web_control} （Retroid/手机浏览器打开）")

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

        keyboard = KeyboardCommand(args.key_timeout, args.max_lin_speed, args.max_yaw_speed)
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
                if web is not None:
                    events |= web.drain_events()
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
                if web is not None:
                    if web.active():
                        command = web.scaled_command()
                    web.set_info(state=state, command=np.round(command, 3).tolist(), clip=clip_steps)
                if now - last_check >= 0.1:
                    check_live(robot)
                    last_check = now
                if state == STATE_TEST and now >= next_infer:
                    q, qd, gyro, gravity = read_q_qd_imu(robot)
                    obs = build_obs(q, qd, gyro, gravity, command, last_action)
                    t_infer = time.perf_counter()
                    raw_action = np.asarray(session.run([output_name], {input_name: obs[None, :]})[0][0], np.float32)
                    print(f"infer  : {(time.perf_counter() - t_infer) * 1000.0:.3f} ms")
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
