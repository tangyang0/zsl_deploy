# ZSL-1 policy deployment

Deployment and sim2sim playback of exported RobotLab velocity policies (flat or rough terrain, any `[1,45] -> [1,12]` ONNX) through the legacy ZSL-1 LowLevel Python SDK: `policy_deploy.py` runs the policy on the robot, `policy_sim2sim.py` replays it in MuJoCo first.

## Environment setup

The SDK Python binding is built for CPython 3.10 (`mc_sdk_zsl_1_py.cpython-310-*`), so the environment must be Python 3.10. `numpy` and `onnxruntime` are the only pip dependencies for deployment; the MuJoCo playback below additionally needs `pip install mujoco pygame-ce`. The SDK itself needs no installation: `policy_deploy.py` loads the binding at runtime from `--sdk-lib` (default: the x86_64 build inside `genisom_l1_sdk_old`).

```bash
conda create -n zsl_sdk_py310 python=3.10 -y
conda activate zsl_sdk_py310
pip install numpy onnxruntime      # deployment
pip install mujoco pygame-ce       # optional: sim2sim playback
```

## Quick start

```bash
cd /home/tangyang/workspace/zsl_deploy
python policy_deploy.py --dry-run   # validate model only: no SDK, no robot
```

The default model is the `smooth_ft` policy. Pass `--model` to load another exported policy (e.g. `config2` or the rough-terrain run); the interface is checked at startup, so a policy with a different layout fails loudly.

## Sim2sim playback in MuJoCo

`policy_sim2sim.py` replays the same ONNX policy in MuJoCo before touching the robot. It patches the training URDF on the fly (floating base, absolute mesh paths, and an injected floor because the URDF world has no ground), then runs the identical pipeline as deployment: the observation/action constants are imported from `policy_deploy.py`, physics runs at the training rate of 200 Hz with the policy at 50 Hz, PD gains `kp=20`/`kd=0.7` and the 28 Nm effort limit match training, and joint targets use the same SDK-window clamp with clip counting. The base-frame angular velocity comes from free-joint `qvel[3:6]` (verified against quaternion finite differences).

The floor is a light plane with dark grid strips every meter (visual-only geoms, no physics) so walking is visible. It deliberately avoids textures: on this machine procedural textures rendered offscreen but were silently dropped by the GLFW viewer's context (verified by capturing the actual window), while plain geometry always renders.

```bash
python policy_sim2sim.py --headless 500 --command 0.5 0 0   # no-viewer smoke test
python policy_sim2sim.py                                    # interactive: 3D viewer + control panel
```

Interactive mode (default) opens mujoco's own 3D viewer plus a small pygame control panel with play.py's exact keyboard semantics: the command is the sum over keys held in the panel — hold to move, release to stop, combinations work, and the command starts at zero (the robot stands until you press). Binding matches Se2Keyboard: arrows or numpad/main-row `8/2/4/6` for linear velocity, `z`/`7` left yaw, `x`/`9` right yaw, `l`/`Space` to zero, `r` to reset the robot. The camera lookat follows the robot each frame (mouse orbit/zoom still work). Falls (base below `0.2 m`) auto-reset. The pygame panel exists because the viewer's key callback has no release events; it is a plain software window, avoiding the offscreen EGL/GLX contexts that failed on some sessions of this machine. Requires `pip install pygame-ce`.

`--native-viewer` runs the viewer alone (its key callback fires once per press with no release/repeat events — verified by synthetic key injection — so keys there stay active until cleared). Use it to sanity-check transfer before hardware: e.g. the `smooth_ft` policy tracks `0.5 m/s` at `0.45 m/s` here, while `config2` only accepts low speeds in MuJoCo (`0.25 m/s` tracks, `0.5` falls) despite walking in Isaac.

## Where to run and network settings

**Host mode (default)**: run on the inference machine over WiFi. Local IP `192.168.234.16`, dog `192.168.234.1`. Override with `--local-ip`/`--dog-ip` if the wireless address differs. Requires the dog-side `/opt/export/config/sdk_config.yaml` to contain `target_ip: "192.168.234.16"` (the motion controller pushes state to this address; changes need a robot reboot to take effect).

**Onboard mode**: run on the dog's main computer (`ssh l1`, files staged in `~/zsl_deploy_onboard`, SDK in `sdk/`). Set the dog-side `target_ip` to `192.168.234.1` (the dog's own ap0 address, matching the factory backup) and reboot, then:

```bash
python3 policy_deploy.py --sdk-lib sdk \
    --local-ip 192.168.234.1 --dog-ip 192.168.234.1 --model models/<policy>.onnx
```

`--dog-ip` must stay `192.168.234.1` even onboard: `mc_ctrl` binds its command socket to the ap0 address, never to loopback. Switching the dog-side `target_ip` between host and onboard mode always requires a reboot, and the other side loses its SDK connection while it points elsewhere. Only one SDK client may be active at a time.

## Keyboard state machine

```bash
python policy_deploy.py
```

The program keeps the startup pose on the ground and starts **low-level damping** without commanding a lift. Press `s` to run the stand-up transition (ground pose -> intermediate pose -> default standing pose), then press `t` to start ONNX policy testing. Before `t`, the robot only holds the standing pose; no policy inference is performed. Press `d` after `s` to cancel/stop before testing and safely switch the current pose to damping. `X` and `Ctrl-C` are also safe exits.

While testing:

- Arrow keys (or numpad `8/2/4/6`): forward/backward and left/right at `±LOW_SPEED` (currently `0.8 m/s`, within the training command range).
- `z` (or numpad `7`): turn left at `+1.0 rad/s`; `c` (or numpad `9`): turn right. Play.py's X binding is not used here because `X` is the safe exit.
- Active directions sum, but a POSIX terminal only auto-repeats the most recently pressed key: holding two direction keys keeps only the last one once `--key-timeout` (default `0.15 s`) expires. Combine directions by tapping alternately, or press one at a time.
- `Space` clears the command immediately; releasing a key zeroes it after the same timeout.
- The hidden `--keyboard` and `--command` options remain accepted for old launch files but are ignored.

## Web control (Retroid / phone)

`--web-control [PORT]` starts a built-in HTTP server (default 8080) serving a single touch gamepad page: two virtual joysticks (left = vx/vy, right X = yaw), and — on Android handhelds like the bundled Retroid Pocket 4 — the browser Gamepad API reads the physical sticks directly. Buttons drive the same state machine: `站起 (s)`, `测试 (t)`, `急停 (d)`. Commands are normalized sticks scaled by `LOW_SPEED`/`TURN_SPEED` with a dead zone; if the page stops sending for 0.3 s the command falls back to the keyboard source. The page cannot send `EXIT` on purpose: closing the browser or losing WiFi only zeroes the command and never kills the deployment. The vendor app cannot be reused for this: it talks to the SDK channel exclusively, so while your policy runs the app has no link at all.

Use `--model`, `--sdk-lib`, `--local-ip`, `--dog-ip`, `--port`, `--kp`, `--kd`, `--key-timeout`, and `--web-control` to override defaults.

## Policy action handling

Experimental variant: raw policy actions are **not** amplitude-clamped. Joint targets are `default pose + ACTION_SCALE * raw action`, then clamped to the hard windows enforced by the SDK's `sendMotorCmd` (abad `±0.48`, hip `-1.15~2.97`, knee `-2.9~-0.65` rad) with a small inward margin (`1e-3 rad`): float32 cannot represent the knee/hip bounds exactly and rounds to the outside, so a target pinned at an edge would still be rejected. The 1 Hz status print shows `clip : N` whenever the clamp engaged during the last second, which quantifies how far the policy wants to exceed the hardware windows.

The initial policy gains `kp=20`, `kd=0.7` match the training actuator stiffness/damping; change them only after hardware validation.

## Shutdown and damping

Every exit path (`d`, `X`, `Ctrl-C`, or a sender error) switches to the legacy LowLevel damping command (`kp=0`, `kd=3`) at the measured current pose and holds it for `--stop-hold-seconds` so the robot can naturally lower itself. The stand-up intermediate pose is never commanded during shutdown. The SDK documents that HighLevel and LowLevel cannot be used concurrently, so this deployment keeps one LowLevel connection rather than trying to call `HighLevel.passive()` in parallel. This software does not replace the robot's physical emergency stop.
