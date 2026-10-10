# ZSL-1 policy deployment

Deployment and sim2sim playback of exported RobotLab velocity policies (flat or rough terrain, any `[1,45] -> [1,12]` ONNX) through the legacy ZSL-1 LowLevel Python SDK. Two tools, two stages: `policy_sim2sim.py` replays the policy in MuJoCo first (sim2sim), `policy_deploy.py` then runs it on the robot (sim2real). Both share the bundled `sdk/` (both architectures) and `models/` (smooth_ft by default, plus config2 and the rough-terrain policy).

## Environment setup

The SDK Python binding is built for CPython 3.10 (`mc_sdk_zsl_1_py.cpython-310-*`), so the environment must be Python 3.10. `numpy` and `onnxruntime` are the only pip dependencies for deployment; the MuJoCo playback below additionally needs `pip install mujoco pygame-ce`. Everything else is bundled in this repo: both SDK builds live under `sdk/` (selected automatically by machine architecture — x86_64 host or the dog's aarch64 main computer), and the ONNX policies live under `models/`. No external files are needed.

```bash
conda create -n zsl_sdk_py310 python=3.10 -y
conda activate zsl_sdk_py310
pip install numpy onnxruntime      # deployment
pip install mujoco pygame-ce       # optional: sim2sim playback
```

## Sim2sim: MuJoCo playback

`policy_sim2sim.py` replays the same ONNX policy in MuJoCo before touching the robot. It patches the training URDF on the fly (floating base, absolute mesh paths, and an injected floor because the URDF world has no ground), then runs the identical pipeline as deployment: the observation/action constants are imported from `policy_deploy.py`, physics runs at the training rate of 200 Hz with the policy at 50 Hz, PD gains `kp=20`/`kd=0.7` and the 28 Nm effort limit match training, and joint targets use the same SDK-window clamp with clip counting. The base-frame angular velocity comes from free-joint `qvel[3:6]` (verified against quaternion finite differences).

The floor is a light plane with dark grid strips every meter (visual-only geoms, no physics) so walking is visible. It deliberately avoids textures: on this machine procedural textures rendered offscreen but were silently dropped by the GLFW viewer's context (verified by capturing the actual window), while plain geometry always renders.

```bash
python policy_sim2sim.py --headless 500 --command 0.5 0 0   # no-viewer smoke test
python policy_sim2sim.py                                    # interactive: 3D viewer + control panel
```

Interactive mode (default) opens mujoco's own 3D viewer plus a small pygame control panel with play.py's exact keyboard semantics: the command is the sum over keys held in the panel — hold to move, release to stop, combinations work, and the command starts at zero (the robot stands until you press). Binding matches Se2Keyboard: arrows or numpad/main-row `8/2/4/6` for linear velocity, `z`/`7` left yaw, `x`/`9` right yaw, `l`/`Space` to zero, `r` to reset the robot. The camera lookat follows the robot each frame (mouse orbit/zoom still work). Falls (base below `0.2 m`) auto-reset. The pygame panel exists because the viewer's key callback has no release events; it is a plain software window, avoiding the offscreen EGL/GLX contexts that failed on some sessions of this machine.

```bash
python policy_sim2sim.py --native-viewer
```

Runs the viewer alone: its key callback has no release/repeat events (verified by synthetic key injection and on the real keyboard), so a pressed direction stays active until cleared, and the latest key replaces the previous command — press once to keep moving, press another key to switch direction, `l`/`Space` to stop. The camera follows the robot in this mode too. Use sim2sim to sanity-check transfer before hardware: e.g. the `smooth_ft` policy tracks `0.5 m/s` at `0.45 m/s` here, while `config2` only accepts low speeds in MuJoCo (`0.25 m/s` tracks, `0.5` falls) despite walking in Isaac.

## Sim2real: robot deployment

### Preparation

#### SDK source and prerequisites

The bundled `sdk/` binaries come from the vendor's official SDK repository [zsibot/genisom_l1_sdk_old](https://github.com/zsibot/genisom_l1_sdk_old.git) (online docs: [zsibot.github.io/genisom_l1_sdk_old](https://zsibot.github.io/genisom_l1_sdk_old/)), distributed under the **BSD 3-Clause License** (Copyright (c) 2025, ZsiBot). Only the two `.so` files per architecture were copied into this repo; the full license text is preserved at `sdk/LICENSE` as its binary-redistribution clause requires.

The SDK repo's own requirements must be satisfied before deployment:

- **Version match**: the SDK protocol differs across firmware versions. Check the dog's version with `grep -oP 'motion-control_\K[^_]+' /etc/release/*[^rootfs]*.yaml` and use the matching SDK release — this repo bundles the one verified on our unit; for anything else go to the upstream repo.
- The vendor recommends running the SDK program on a compute board **wired to the robot** (ethernet) rather than WiFi.
- The dog-side `/opt/export/config/sdk_config.yaml` `target_ip` must point at the controlling machine, and the dog must be **rebooted** after changing it.
- Only one SDK client may connect at a time — the vendor remote/app is locked out while this script runs, and vice versa.
- Keep system resources free: the SDK docs warn that motion control can fail under resource starvation.

#### Where to run and what to change

The two modes differ in only two things: which machine runs the script, and where the dog pushes its state (`target_ip` in the dog-side `/opt/export/config/sdk_config.yaml` — every change requires a robot reboot to take effect). The `LOCAL_IP`/`DOG_IP` constants both default to `192.168.234.1` — **onboard mode (running on the dog) is the default**, no network flags needed.

**Deploy on the dog (onboard mode, default)**

1. On the dog, set `/opt/export/config/sdk_config.yaml` to `target_ip: "192.168.234.1"` (the dog's own ap0 address, matching the factory backup), then reboot the dog.
2. Over SSH, just run:

   ```bash
   python3 policy_deploy.py --web-control
   ```

   Full example with the rough policy and speed caps:

   ```bash
   python3 policy_deploy.py --web-control --model models/rough.onnx --max-x-speed 0.4 --max-y-speed 0.2 --max-yaw-speed 0.8
   ```

**Deploy on the inference machine**

1. Connect the machine to the dog's WiFi AP; it should get `192.168.234.16` (the shipped default). If DHCP gave a different address, edit `LOCAL_IP` or pass `--local-ip <actual address>`.
2. On the dog, set `target_ip: "192.168.234.16"` (the dog pushes state to this address), then reboot the dog.
3. Run:

   ```bash
   python policy_deploy.py --local-ip 192.168.234.16 --web-control
   ```

`--dog-ip` always stays `192.168.234.1`: `mc_ctrl` binds its command socket to the ap0 address, never to loopback. Switching `target_ip` between modes always requires a reboot, and the other side has no SDK connection while it points elsewhere; only one SDK client may be active at a time.

### Operation

#### Launch and options

```bash
python policy_deploy.py --dry-run   # validate model only: no SDK, no robot
```

The default model is `models/smooth_ft.onnx`. Pass `--model` to load another bundled policy (`models/config2.onnx`, `models/rough.onnx`) or any exported `[1,45] -> [1,12]` policy; the interface is checked at startup, so a policy with a different layout fails loudly. All defaults can be overridden with `--model`, `--sdk-lib`, `--local-ip`, `--dog-ip`, `--port`, `--kp`, `--kd`, `--key-timeout`, `--max-x-speed`, `--max-y-speed`, `--max-yaw-speed`, and `--web-control`.

#### Keyboard control

```bash
python policy_deploy.py
```

The program keeps the startup pose on the ground and starts **low-level damping** without commanding a lift. Press `s` to run the stand-up transition (ground pose -> intermediate pose -> default standing pose), then press `t` to start ONNX policy testing. Before `t`, the robot only holds the standing pose; no policy inference is performed. Press `d` after `s` to cancel/stop before testing and safely switch the current pose to damping. `X` and `Ctrl-C` are also safe exits.

While testing:

- Arrow keys (or numpad `8/2/4/6`): forward/backward at `±MAX_X_SPEED` and left/right at `±MAX_Y_SPEED` (defaults `1.0 m/s`, the training command limit; tighten with `--max-x-speed` / `--max-y-speed` — the trained vy range is only ±0.3 m/s).
- `z` (or numpad `7`): turn left at `+MAX_YAW_SPEED` (default `1.0 rad/s`; tighten with `--max-yaw-speed`); `c` (or numpad `9`): turn right. Play.py's X binding is not used here because `X` is the safe exit.
- Active directions sum, but a POSIX terminal only auto-repeats the most recently pressed key: holding two direction keys keeps only the last one once `--key-timeout` (default `0.15 s`) expires. Combine directions by tapping alternately, or press one at a time.
- `Space` clears the command immediately; releasing a key zeroes it after the same timeout.
- The hidden `--keyboard` and `--command` options remain accepted for old launch files but are ignored.

#### Web control (Retroid / phone)

```bash
python policy_deploy.py --web-control        # or: --web-control 9090 for another port
```

This starts a built-in HTTP server (default port 8080) serving a single touch gamepad page: two virtual joysticks (left = vx/vy, right X = yaw), and — on Android handhelds like the bundled Retroid Pocket 4 — the browser Gamepad API reads the physical sticks directly. Buttons drive the same state machine: `站起 (s)`, `测试 (t)`, `急停 (d)`. Commands are normalized sticks scaled by `MAX_X_SPEED`/`MAX_Y_SPEED`/`MAX_YAW_SPEED` with a dead zone (`--max-x-speed`/`--max-y-speed`/`--max-yaw-speed` apply to the keyboard and the web sticks alike); if the page stops sending for 0.3 s the command falls back to the keyboard source. The page cannot send `EXIT` on purpose: closing the browser or losing WiFi only zeroes the command and never kills the deployment. The vendor app cannot be reused for this: it talks to the SDK channel exclusively, so while your policy runs the app has no link at all.

### Mechanism and safety

#### Action handling and joint limits

Experimental variant: raw policy actions are **not** amplitude-clamped. Joint targets are `default pose + ACTION_SCALE * raw action`, then clamped to the hard windows enforced by the SDK's `sendMotorCmd` (abad `±0.48`, hip `-1.15~2.97`, knee `-2.9~-0.65` rad) with a small inward margin (`1e-3 rad`): float32 cannot represent the knee/hip bounds exactly and rounds to the outside, so a target pinned at an edge would still be rejected. The 1 Hz status print shows `clip : N` whenever the clamp engaged during the last second, which quantifies how far the policy wants to exceed the hardware windows.

The initial policy gains `kp=20`, `kd=0.7` match the training actuator stiffness/damping; change them only after hardware validation.

#### Shutdown and damping

Every exit path (`d`, `X`, `Ctrl-C`, or a sender error) switches to the legacy LowLevel damping command (`kp=0`, `kd=3`) at the measured current pose and holds it for `--stop-hold-seconds` so the robot can naturally lower itself. The stand-up intermediate pose is never commanded during shutdown. The SDK documents that HighLevel and LowLevel cannot be used concurrently, so this deployment keeps one LowLevel connection rather than trying to call `HighLevel.passive()` in parallel. This software does not replace the robot's physical emergency stop.
