# ZSL-1 policy deployment

These scripts use the legacy ZSL-1 LowLevel Python SDK and the exported
RobotLab ONNX policy. Run them with the Python 3.10 environment that matches
the SDK extension:

```bash
conda activate zsl_sdk_py310
cd /home/tangyang/workspace/zsl_deploy
```

当前无线网络默认配置为本机 `192.168.234.16`、机器狗 `192.168.234.1`。
如果无线网卡地址不同，可通过 `--local-ip` 和 `--dog-ip` 覆盖。

First validate the model without loading the SDK or connecting to the robot:

```bash
python policy_deploy.py --dry-run
```

Then check that the network and SDK are returning state. This is receive-only
and does not send motor commands:

```bash
python sdk_state_test.py
```

The deployment program uses the following explicit keyboard state machine:

```bash
python policy_deploy.py
```

The program keeps the startup pose on the ground and starts **low-level
damping** without commanding a lift. Press `s` to run the stand-up transition
(ground pose -> intermediate pose -> default standing pose), then press `t`
to start ONNX policy testing. Before `t`, the robot only holds the standing
pose; no policy inference is performed. Press `d` after `s` to cancel/stop
before testing and safely switch the current pose to damping.

During testing, the arrow keys (or numpad `8/2/4/6`) continuously refresh a
low-speed command: up/down = forward/backward and left/right = left/right,
with magnitude `0.25 m/s`. This is intentionally above the training command
threshold of `0.2 m/s`. The terminal does not expose physical key-release
events, so the controller treats a direction as active only while repeated
key events arrive; after `--key-timeout` seconds without one (default `0.15`),
the command becomes `[0, 0, 0]`. `Space` is an explicit immediate clear.
There is no keyboard yaw command in this mode.

The initial policy gains are `kp=20`, `kd=0.7`; change them only after
hardware validation. `X` and `Ctrl-C` are also safe exits: the policy sender
is stopped, the measured current pose is retained as the target, and then
`kp=0, kd=3` damping is sent for several seconds so the robot can naturally
lower itself. They do not command the stand-up intermediate pose or
immediately release the current policy target.
The hidden `--keyboard` and `--command` options remain accepted for old
launch files, but live control always follows this state machine and ignores
`--command`.

To adjust the release timeout, for example:

```bash
python policy_deploy.py --key-timeout 0.20
```

Use `--model`, `--sdk-lib`, `--local-ip`, `--dog-ip`, and `--port` to override
the paths or network settings. `policy_preview.py` uses the same default rough
model and only performs one inference after receiving a state.

The damping state is implemented with the legacy LowLevel damping command
(`kp=0`, `kd=3`) at the measured current pose. The SDK documents that
HighLevel and LowLevel cannot be used concurrently, so this deployment keeps
one LowLevel connection rather than trying to call `HighLevel.passive()` in
parallel. This software does not replace the robot's physical emergency
stop. Do not run the preview, state test, and deployment programs at the same
time because they each create an SDK connection.
