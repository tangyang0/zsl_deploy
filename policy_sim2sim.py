#!/usr/bin/env python3
"""Sim2sim playback of an exported RobotLab policy in MuJoCo.

Loads the training URDF (with an injected floating base), runs the same
ONNX policy and the same observation/action pipeline as policy_deploy.py:
obs = [ang_vel*0.25, projected_gravity, command, joint_pos-default,
joint_vel*0.05, last_action], target = default + ACTION_SCALE*action
clamped to the SDK joint windows, torque = kp*(q_des-q) - kd*qd with the
training effort limit. Physics runs at 200 Hz (training sim dt), the
policy at 50 Hz.

The interactive mode shows mujoco's own 3D viewer plus a small pygame
control panel with play.py's exact Se2Keyboard semantics: the command is the
sum over keys currently held in the panel — hold to move, release to stop,
any combination works, zero command until a key is held. Binding: arrows or
numpad/main-row 8/2/4/6 for linear velocity, z/7 for left yaw, x/9 for right
yaw, l or Space to zero, r to reset the robot. The native viewer alone
(--native-viewer) has a press-only key callback (no release or repeat
events, verified by synthetic key injection), so there keys stay active
until cleared. Falls (base height below 0.2 m) auto-reset.
"""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

from policy_deploy import (
    ACTION_SCALE,
    DEFAULT_MODEL,
    DEFAULT_Q,
    MAX_LIN_SPEED,
    Q_LIMIT_HI,
    Q_LIMIT_LO,
    MAX_YAW_SPEED,
    validate_session,
)

SOURCE_URDF = Path(
    "/home/tangyang/workspace/robot_lab/source/robot_lab/data/Robots/"
    "zsibot/zsl1_description/urdf/zsl1.urdf"
)
SOURCE_MESH_DIR = SOURCE_URDF.parent.parent / "meshes"

# RobotLab policy order: [FAR, FBL, RAR, RBL] = [FR, FL, RR, RL] x [ABAD, HIP, KNEE].
POLICY_JOINTS = [
    f"{leg}_{joint}"
    for leg in ("FAR", "FBL", "RAR", "RBL")
    for joint in ("ABAD_JOINT", "HIP_JOINT", "KNEE_JOINT")
]

SIM_DT = 0.005          # training physics rate (200 Hz)
DECIMATION = 4          # policy rate = SIM_DT * DECIMATION = 50 Hz
KP = 20.0               # matches training actuator stiffness / deploy gains
KD = 0.7
EFFORT_LIMIT = 28.0     # training DCMotor effort limit
RESET_HEIGHT = 0.4      # training init_state base height
FALL_HEIGHT = 0.2

# GLFW key codes used by the mujoco viewer: arrows 262-265, KP_0..KP_9 = 320..329.
_KEY_ARROW = {265: "UP", 264: "DOWN", 263: "LEFT", 262: "RIGHT"}
_KEY_NUMPAD = {320 + i: str(i) for i in range(10)}
# Same binding as play.py's Se2Keyboard (Z/NUMPAD_7 = +omega_z, X/NUMPAD_9
# = -omega_z, L resets); "c" stays as an extra alias for right yaw.
_KEY_STEPS = {
    "8": np.array([MAX_LIN_SPEED, 0.0, 0.0], np.float32),
    "2": np.array([-MAX_LIN_SPEED, 0.0, 0.0], np.float32),
    "4": np.array([0.0, MAX_LIN_SPEED, 0.0], np.float32),
    "6": np.array([0.0, -MAX_LIN_SPEED, 0.0], np.float32),
    "z": np.array([0.0, 0.0, MAX_YAW_SPEED], np.float32),
    "7": np.array([0.0, 0.0, MAX_YAW_SPEED], np.float32),
    "x": np.array([0.0, 0.0, -MAX_YAW_SPEED], np.float32),
    "c": np.array([0.0, 0.0, -MAX_YAW_SPEED], np.float32),
    "9": np.array([0.0, 0.0, -MAX_YAW_SPEED], np.float32),
}
_KEY_STEPS.update({name: vec for name, vec in (
    ("UP", _KEY_STEPS["8"]), ("DOWN", _KEY_STEPS["2"]),
    ("LEFT", _KEY_STEPS["4"]), ("RIGHT", _KEY_STEPS["6"]),
)})


def _decode_key(keycode):
    if keycode in _KEY_ARROW:
        return _KEY_ARROW[keycode]
    if keycode in _KEY_NUMPAD:
        return _KEY_NUMPAD[keycode]
    if 0 < keycode < 256:
        return chr(keycode).lower()
    return None


def build_mujoco_urdf() -> str:
    """Patch the training URDF for MuJoCo: floating base + absolute meshes."""
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    root.append(ET.fromstring(
        '<mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco>'
    ))
    root.append(ET.fromstring('<link name="world"/>'))
    root.append(ET.fromstring(
        '<joint name="anchor" type="floating">'
        '<parent link="world"/><child link="BASE_LINK"/></joint>'
    ))
    for mesh in root.iter("mesh"):
        filename = Path(mesh.get("filename"))
        mesh.set("filename", str((SOURCE_URDF.parent / filename).resolve()))
    return ET.tostring(root, encoding="unicode")


def load_mujoco():
    spec = mujoco.MjSpec.from_string(build_mujoco_urdf())
    # The URDF world has no ground; add the floor the policy was trained on.
    # Motion reference comes from visual-only dark grid strips on a light
    # floor: plain geometry renders everywhere, while procedural textures
    # silently failed to upload in the GLFW viewer on this machine (they
    # did render offscreen, so this is a context-dependent GL quirk).
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20.0, 20.0, 0.1], pos=[0.0, 0.0, 0.0],
        friction=[1.0, 0.005, 0.0001], rgba=[0.88, 0.88, 0.88, 1.0],
    )
    for i in range(-20, 21):
        if i == 0:
            continue
        spec.worldbody.add_geom(
            name=f"grid_x_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015, 20.0, 0.002], pos=[i, 0.0, 0.002],
            rgba=[0.25, 0.25, 0.28, 1.0],
            contype=0, conaffinity=0, group=1,
        )
        spec.worldbody.add_geom(
            name=f"grid_y_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[20.0, 0.015, 0.002], pos=[0.0, i, 0.002],
            rgba=[0.25, 0.25, 0.28, 1.0],
            contype=0, conaffinity=0, group=1,
        )
    spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
    model = spec.compile()
    model.opt.timestep = SIM_DT
    joint_ids = {model.joint(i).name: i for i in range(model.njnt)}
    missing = [name for name in POLICY_JOINTS if name not in joint_ids]
    if missing:
        raise RuntimeError(f"MuJoCo 模型缺少关节: {missing}")
    qadr = np.array([model.jnt_qposadr[joint_ids[name]] for name in POLICY_JOINTS])
    vadr = np.array([model.jnt_dofadr[joint_ids[name]] for name in POLICY_JOINTS])
    base_body = model.body("BASE_LINK").id
    return model, qadr, vadr, base_body


def reset_robot(model, data, qadr):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, RESET_HEIGHT)
    data.qpos[3] = 1.0  # quaternion (w, x, y, z)
    data.qpos[qadr] = DEFAULT_Q
    mujoco.mj_forward(model, data)


def step_policy(model, data, session, output_name, qadr, vadr, base_body,
                command, last_action):
    """One 50 Hz cycle: build obs, infer, clamp target, run DECIMATION substeps."""
    # Free-joint qvel[3:6] is the angular velocity in the base frame
    # (verified against quaternion finite differences).
    ang_vel = data.qvel[3:6]
    # Projected gravity: gravity direction expressed in the base frame.
    quat = data.qpos[3:7].copy()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])

    q = data.qpos[qadr]
    qd = data.qvel[vadr]
    obs = np.concatenate([
        ang_vel * 0.25, projected_gravity, command, q - DEFAULT_Q, qd * 0.05, last_action,
    ]).astype(np.float32)
    action = np.asarray(session.run([output_name], {session.get_inputs()[0].name: obs[None, :]})[0][0], np.float32)
    if action.shape != (12,) or not np.isfinite(action).all():
        raise RuntimeError(f"策略输出异常: shape={action.shape}")

    q_target = DEFAULT_Q + ACTION_SCALE * action
    q_des = np.clip(q_target, Q_LIMIT_LO, Q_LIMIT_HI)
    clipped = int(not np.array_equal(q_des, q_target))
    for _ in range(DECIMATION):
        tau = KP * (q_des - data.qpos[qadr]) - KD * data.qvel[vadr]
        data.qfrc_applied[vadr] = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        mujoco.mj_step(model, data)
    return action, clipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ONNX policy path")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                        default=[0.0, 0.0, 0.0], help="Initial velocity command (default zero: no walking until a key is held)")
    parser.add_argument("--headless", type=int, default=0, metavar="STEPS",
                        help="Run N policy steps without a viewer and exit (smoke test)")
    parser.add_argument("--native-viewer", action="store_true",
                        help="Use mujoco's own viewer (keys are press-only: no release events)")
    return parser.parse_args()


def main(args):
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    validate_session(session)
    model, qadr, vadr, base_body = load_mujoco()
    data = mujoco.MjData(model)
    reset_robot(model, data, qadr)
    command = np.asarray(args.command, dtype=np.float32)
    print(f"模型: {args.model}")
    print(f"初始指令: {command}")

    if args.headless > 0:
        last_action = np.zeros(12, dtype=np.float32)
        falls, clip_steps = 0, 0
        for step in range(args.headless):
            last_action, clipped = step_policy(
                model, data, session, session.get_outputs()[0].name,
                qadr, vadr, base_body, command, last_action,
            )
            clip_steps += clipped
            if data.qpos[2] < FALL_HEIGHT:
                falls += 1
                reset_robot(model, data, qadr)
                last_action[:] = 0.0
            if step % 100 == 0:
                print(f"step {step:5d}  height={data.qpos[2]:.3f}  "
                      f"|action|max={np.abs(last_action).max():.2f}  falls={falls}")
        print(f"headless 完成: {args.headless} 步, 摔倒 {falls} 次, 触限位 {clip_steps} 步, "
              f"最终高度 {data.qpos[2]:.3f} m")
        return

    if args.native_viewer:
        run_native_viewer(model, data, session, qadr, vadr, base_body, command)
    else:
        run_interactive(model, data, session, qadr, vadr, base_body, command)


PANEL_W, PANEL_H = 620, 130


def _pygame_keymap():
    """pygame key -> command step, same binding as play.py's Se2Keyboard."""
    import pygame
    fwd = np.array([MAX_LIN_SPEED, 0.0, 0.0], np.float32)
    left = np.array([0.0, MAX_LIN_SPEED, 0.0], np.float32)
    turn_l = np.array([0.0, 0.0, MAX_YAW_SPEED], np.float32)
    return {
        pygame.K_UP: fwd, pygame.K_KP8: fwd, pygame.K_8: fwd,
        pygame.K_DOWN: -fwd, pygame.K_KP2: -fwd, pygame.K_2: -fwd,
        pygame.K_LEFT: left, pygame.K_KP4: left, pygame.K_4: left,
        pygame.K_RIGHT: -left, pygame.K_KP6: -left, pygame.K_6: -left,
        pygame.K_z: turn_l, pygame.K_KP7: turn_l,
        pygame.K_x: -turn_l, pygame.K_KP9: -turn_l, pygame.K_c: -turn_l,
    }


def run_interactive(model, data, session, qadr, vadr, base_body, command):
    """play.py-style keyboard: a small pygame panel reads the full key state
    (hold to move, release to stop, combinations sum), while mujoco's own
    GLFW viewer shows the 3D scene. No offscreen GL context is created —
    EGL/GLX offscreen rendering failed on some setups of this machine."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((PANEL_W, PANEL_H))
    pygame.display.set_caption("policy_sim2sim 控制 — 点击此窗口按键")
    # The default pygame font has no CJK glyphs (the panel showed boxes);
    # pick any installed Noto CJK face, falling back to the default font.
    # CJK glyphs are wide: keep the size small enough for the longest line.
    font = None
    for name in ("notosanscjksc", "notosanscjkhk", "notoserifcjksc", "notoserifcjkhk"):
        path = pygame.font.match_font(name)
        if path:
            font = pygame.font.Font(path, 18)
            break
    if font is None:
        font = pygame.font.SysFont(None, 18)
    keymap = _pygame_keymap()
    output_name = session.get_outputs()[0].name
    print("在控制面板按住按键: ↑↓←→/8246 平移, z/7 左转, x/9 右转, l/空格 清零, r 复位")
    last_action = np.zeros(12, dtype=np.float32)
    clip_steps, falls, running = 0, 0, True
    next_policy = time.monotonic()
    clock = pygame.time.Clock()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while running and viewer.is_running():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_robot(model, data, qadr)
                    last_action[:] = 0.0
                    print("已复位")
            keys = pygame.key.get_pressed()
            command[:] = 0.0
            if not (keys[pygame.K_l] or keys[pygame.K_SPACE]):
                for key, step in keymap.items():
                    if keys[key]:
                        command += step
            now = time.monotonic()
            while now >= next_policy and running:
                last_action, clipped = step_policy(
                    model, data, session, output_name,
                    qadr, vadr, base_body, command, last_action,
                )
                clip_steps += clipped
                if data.qpos[2] < FALL_HEIGHT:
                    falls += 1
                    reset_robot(model, data, qadr)
                    last_action[:] = 0.0
                next_policy += SIM_DT * DECIMATION
                if now - next_policy > 0.1:
                    next_policy = now
            with viewer.lock():
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.3]
            viewer.sync()
            screen.fill((24, 24, 24))
            lines = (
                f"command [{command[0]:+.2f} {command[1]:+.2f} {command[2]:+.2f}]"
                f"   height {data.qpos[2]:.2f} m   pos ({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f})",
                f"按住: ↑↓←→/8246 平移   z/7 左转   x/9 右转   l/空格 清零   r 复位",
                f"falls {falls}   clip {clip_steps}   关闭任一窗口退出",
            )
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, (0, 255, 120) if i == 0 else (200, 200, 200)), (12, 12 + 34 * i))
            pygame.display.flip()
            clock.tick(60)
        if not running:
            viewer.close()
    pygame.quit()
    if clip_steps:
        print(f"退出；本次运行共 {clip_steps} 个周期触及限位裁剪")


def run_native_viewer(model, data, session, qadr, vadr, base_body, command):
    """MuJoCo's own viewer. Its key callback fires once per press with no
    repeat/release events (verified with synthetic key injection and on the
    real keyboard), so a pressed direction stays active until cleared —
    press once to keep moving, `l`/`Space` to stop. Prefer the default pygame
    window for hold-to-move semantics."""
    state = {"key": None}

    def key_callback(keycode):
        key = _decode_key(keycode)
        if key in (" ", "l"):
            state["key"] = None
        elif key == "r":
            state["reset"] = True
        elif key in _KEY_STEPS:
            state["key"] = key  # last key wins: replaces the previous command

    print("注意: 原生 viewer 无松键/重复事件，按一次持续生效且新键覆盖旧指令，l/空格 停止"
          "（建议用默认 pygame 窗口获得按住即走、松开即停）")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_action = np.zeros(12, dtype=np.float32)
        clip_steps, next_loop = 0, time.monotonic()
        last_printed = None
        while viewer.is_running():
            if state.pop("reset", False):
                reset_robot(model, data, qadr)
                last_action[:] = 0.0
                print("已复位")
            now = time.monotonic()
            command[:] = _KEY_STEPS[state["key"]] if state["key"] else 0.0
            rounded = tuple(np.round(command, 3))
            if rounded != last_printed:
                print(f"command: {rounded}")
                last_printed = rounded
            last_action, clipped = step_policy(
                model, data, session, session.get_outputs()[0].name,
                qadr, vadr, base_body, command, last_action,
            )
            clip_steps += clipped
            if data.qpos[2] < FALL_HEIGHT:
                print(f"摔倒 (height={data.qpos[2]:.3f})，1 秒后自动复位")
                time.sleep(1.0)
                reset_robot(model, data, qadr)
                last_action[:] = 0.0
            with viewer.lock():
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.3]
            viewer.sync()
            next_loop += SIM_DT * DECIMATION
            sleep = next_loop - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_loop = time.monotonic()
        if clip_steps:
            print(f"退出；本次运行共 {clip_steps} 个周期触及限位裁剪")


if __name__ == "__main__":
    main(parse_args())
