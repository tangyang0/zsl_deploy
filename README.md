# ZSL-1 policy deployment

Single-script deployment of exported RobotLab velocity policies (flat or
rough terrain, any `[1,45] -> [1,12]` ONNX) through the legacy ZSL-1
LowLevel Python SDK. Run with the Python 3.10 environment that matches the
SDK extension:

```bash
conda activate zsl_sdk_py310
cd /home/tangyang/workspace/zsl_deploy
python policy_deploy.py --dry-run   # validate model only: no SDK, no robot
```

The default model is the `smooth_ft` policy. Pass `--model` to load another
exported policy (e.g. `config2` or the rough-terrain run); the interface is
checked at startup, so a policy with a different layout fails loudly.

## Where to run and network settings

**Host mode (default)**: run on the inference machine over WiFi. Local IP
`192.168.234.16`, dog `192.168.234.1`. Override with `--local-ip`/`--dog-ip`
if the wireless address differs. Requires the dog-side
`/opt/export/config/sdk_config.yaml` to contain `target_ip: "192.168.234.16"`
(the motion controller pushes state to this address; changes need a robot
reboot to take effect).

**Onboard mode**: run on the dog's main computer (`ssh l1`, files staged in
`~/zsl_deploy_onboard`, SDK in `sdk/`). Set the dog-side `target_ip` to
`127.0.0.1` and reboot, then:

```bash
python3 policy_deploy.py --sdk-lib sdk \
    --local-ip 127.0.0.1 --dog-ip 192.168.234.1 --model models/<policy>.onnx
```

`--dog-ip` must stay `192.168.234.1` even onboard: `mc_ctrl` binds its
command socket to the ap0 address, never to loopback. Switching the dog-side
`target_ip` between host and onboard mode always requires a reboot, and the
other side loses its SDK connection while it points elsewhere. Only one SDK
client may be active at a time.

## Keyboard state machine

```bash
python policy_deploy.py
```

The program keeps the startup pose on the ground and starts **low-level
damping** without commanding a lift. Press `s` to run the stand-up transition
(ground pose -> intermediate pose -> default standing pose), then press `t`
to start ONNX policy testing. Before `t`, the robot only holds the standing
pose; no policy inference is performed. Press `d` after `s` to cancel/stop
before testing and safely switch the current pose to damping. `X` and
`Ctrl-C` are also safe exits.

While testing:

- Arrow keys (or numpad `8/2/4/6`): forward/backward and left/right at
  `±LOW_SPEED` (currently `0.8 m/s`, within the training command range).
- `z` (or numpad `7`): turn left at `+1.0 rad/s`; `c` (or numpad `9`): turn
  right. Play.py's X binding is not used here because `X` is the safe exit.
- Active directions sum, but a POSIX terminal only auto-repeats the most
  recently pressed key: holding two direction keys keeps only the last one
  once `--key-timeout` (default `0.15 s`) expires. Combine directions by
  tapping alternately, or press one at a time.
- `Space` clears the command immediately; releasing a key zeroes it after
  the same timeout.
- The hidden `--keyboard` and `--command` options remain accepted for old
  launch files but are ignored.

Use `--model`, `--sdk-lib`, `--local-ip`, `--dog-ip`, `--port`, `--kp`,
`--kd`, and `--key-timeout` to override defaults.

## Policy action handling

Experimental variant: raw policy actions are **not** amplitude-clamped.
Joint targets are `default pose + ACTION_SCALE * raw action`, then clamped
to the hard windows enforced by the SDK's `sendMotorCmd` (abad `±0.48`,
hip `-1.15~2.97`, knee `-2.9~-0.65` rad) with a small inward margin
(`1e-3 rad`): float32 cannot represent the knee/hip bounds exactly and
rounds to the outside, so a target pinned at an edge would still be
rejected. The 1 Hz status print shows `clip : N` whenever the clamp engaged
during the last second, which quantifies how far the policy wants to exceed
the hardware windows.

The initial policy gains `kp=20`, `kd=0.7` match the training actuator
stiffness/damping; change them only after hardware validation.

## Shutdown and damping

Every exit path (`d`, `X`, `Ctrl-C`, or a sender error) switches to the
legacy LowLevel damping command (`kp=0`, `kd=3`) at the measured current
pose and holds it for `--stop-hold-seconds` so the robot can naturally
lower itself. The stand-up intermediate pose is never commanded during
shutdown. The SDK documents that HighLevel and LowLevel cannot be used
concurrently, so this deployment keeps one LowLevel connection rather than
trying to call `HighLevel.passive()` in parallel. This software does not
replace the robot's physical emergency stop.
