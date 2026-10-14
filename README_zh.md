# ZSL-1 策略部署

把 RobotLab 导出的速度策略 ONNX（平地或粗糙地形，任意 `[1,45] -> [1,12]` 接口）经 ZSL-1 旧版 LowLevel Python SDK 落到真机，分两个阶段、两个工具：`policy_sim2sim.py` 先在 MuJoCo 里回放验证（sim2sim），`policy_deploy.py` 再上真机运行（sim2real）。两者共用子模块引用的 `sdk/`（genisom_l1_sdk_old，双架构）与内置的 `models/`（默认 `smooth_ft`，含粗糙地形策略）。

## 环境搭建

```bash
git clone --recurse-submodules https://github.com/tangyang0/zsl_deploy.git
# 已经克隆过的旧副本补齐子模块：
git submodule update --init
```

SDK 的 Python 绑定按 CPython 3.10 编译（`mc_sdk_zsl_1_py.cpython-310-*`），因此环境必须是 Python 3.10。部署只需 `numpy` 和 `onnxruntime` 两个 pip 依赖；下面的 MuJoCo 回放额外需要 `pip install mujoco pygame-ce`。ONNX 策略内置在 `models/` 下。SDK 以 git 子模块引用（`sdk/` → [zsibot/genisom_l1_sdk_old](https://github.com/zsibot/genisom_l1_sdk_old.git)）：克隆时加 `--recurse-submodules`，或克隆后执行 `git submodule update --init`；脚本按机器架构自动选用 `sdk/lib/zsl-1/<架构>`。

```bash
conda create -n zsl_sdk_py310 python=3.10 -y
conda activate zsl_sdk_py310
pip install numpy onnxruntime      # 部署
pip install mujoco pygame-ce       # 可选：sim2sim 回放
```

## sim2sim：MuJoCo 回放

`policy_sim2sim.py` 在上真机之前，用同一份 ONNX 策略在 MuJoCo 里回放。它实时给训练 URDF 打补丁（浮动基座、mesh 绝对路径、注入地面），并跑与部署完全一致的管线：观测/动作常量直接从 `policy_deploy.py` 导入，物理 200 Hz、策略 50 Hz，PD 增益 `kp=20`/`kd=0.7` 与 28 Nm 力矩限幅同训练，关节目标同样按 SDK 硬窗口裁剪并统计 clip 次数。机体系角速度取自由关节的 `qvel[3:6]`。

地面是浅色平面加深色网格线（纯视觉几何，不参与碰撞），用来观察位移。不用纹理，纹理在 viewer 的 GL 上下文里会丢失。

```bash
python policy_sim2sim.py --headless 500 --command 0.5 0 0   # 无窗口冒烟测试
python policy_sim2sim.py                                    # 交互：3D viewer + 控制面板
```

交互模式（默认）打开 mujoco 原生 3D viewer 和一个 pygame 控制面板，键盘语义与 play.py 一致：指令 = 当前按住的所有键之和，按住即走、松开即停、可组合，初始为零。键位同 Se2Keyboard：方向键或小键盘/主键区 `8/2/4/6` 平移，`z`/`7` 左转，`x`/`9` 右转，`l`/`空格` 清零，`r` 复位。相机注视点跟随机器狗（鼠标旋转/缩放可用），摔倒自动复位。键盘输入放在 pygame 面板上，因为 viewer 的按键回调没有松开事件，拿不到按住状态。

```bash
python policy_sim2sim.py --native-viewer
```

只开原生 viewer：按键回调没有松开/重复事件，按一次持续生效，新按键覆盖旧指令，`l`/`空格` 停止；相机同样跟随。策略在 MuJoCo 和 Isaac 里的表现可能差很多（`config2` 在 Isaac 正常，在 MuJoCo `0.5 m/s` 就摔），上真机前先在这里过一遍。

## sim2real：真机部署

### 准备

#### SDK 来源与前置要求

内置 `sdk/` 的二进制来自厂商官方 SDK 仓库 [zsibot/genisom_l1_sdk_old](https://github.com/zsibot/genisom_l1_sdk_old.git)（在线文档：[zsibot.github.io/genisom_l1_sdk_old](https://zsibot.github.io/genisom_l1_sdk_old/)），采用 **BSD 3-Clause 协议**（Copyright (c) 2025, ZsiBot）。上游仓库以 `sdk/` 子模块整体引用（钉在本机验证过的提交），协议全文随子模块自带。

部署前须先满足 SDK 仓库自身的要求：

- **版本匹配**：SDK 与机器狗本体内程序的通讯协议随版本不同。在狗上执行 `grep -oP 'motion-control_\K[^_]+' /etc/release/*[^rootfs]*.yaml` 查询本机版本，并选用对应版本的 SDK——本仓库内置的是本机验证过的版本，其他版本到上游仓库获取。
- 厂商建议 SDK 程序运行在与机器人**网线直连**的算力板上，避免 WiFi 干扰。
- 狗端 `/opt/export/config/sdk_config.yaml` 的 `target_ip` 必须指向控制端机器，且修改后需**重启机器狗**生效。
- 同一时刻只允许一个 SDK 客户端连接——本脚本运行期间厂商遥控器/App 被屏蔽，反之亦然。
- 保证系统资源充足：SDK 文档警告资源不足时可能出现运动控制模块失效。

#### 部署位置与网络配置

两种模式的区别只有两件事：脚本在哪台机器上跑，以及狗把状态推给谁（狗端 `/opt/export/config/sdk_config.yaml` 的 `target_ip`，每次修改都必须重启狗才生效）。默认常量 `LOCAL_IP`/`DOG_IP` 均为 `192.168.234.1`——**默认在狗上运行（板载模式）**，无需任何网络参数。

**在狗上部署（默认模式）**

1. 在狗上把 `/opt/export/config/sdk_config.yaml` 的 `target_ip` 改为 `192.168.234.1`（狗自己的 ap0 地址），重启狗。
2. SSH 上去直接运行：

   ```bash
   python3 policy_deploy.py --web-control
   ```

   使用 rough 策略并限速的完整示例：

   ```bash
   python3 policy_deploy.py --web-control --model models/rough.onnx --max-x-speed 0.4 --max-y-speed 0.2 --max-yaw-speed 0.8
   ```

   狗上默认绑定核心 5,6（核 7 留给运控，0-4 常被感知/编码占用）；其他机器默认不绑定，`--pin-cores` 可指定，`none` 关闭。

**在推理机上部署**

1. 推理机连狗的 WiFi AP，正常会拿到 `192.168.234.16`（出厂默认）。若 DHCP 分的不是这个地址，改 `LOCAL_IP` 常量或运行时加 `--local-ip <实际地址>`。
2. 在狗上把 `target_ip` 改为 `"192.168.234.16"`（狗按此地址推送状态），然后重启狗。
3. 运行：

   ```bash
   python policy_deploy.py --local-ip 192.168.234.16 --web-control
   ```

`--dog-ip` 恒为 `192.168.234.1`：`mc_ctrl` 的指令套接字绑在 ap0 地址上、从不监听 loopback。两种模式间切换 `target_ip` 都要重启狗，且指向前一侧时另一侧没有 SDK 连接；同一时刻只允许一个 SDK 客户端。

### 操作

#### 启动与参数

```bash
python policy_deploy.py --dry-run   # 仅验证模型：不加载 SDK、不连接机器人
```

默认模型是 `models/smooth_ft.onnx`；用 `--model` 换成内置的 `models/rough.onnx` 或任意导出的 `[1,45] -> [1,12]` 策略，接口在启动时校验，布局不符会直接报错。所有默认值都可用 `--model`、`--sdk-lib`、`--local-ip`、`--dog-ip`、`--port`、`--kp`、`--kd`、`--key-timeout`、`--max-x-speed`、`--max-y-speed`、`--max-yaw-speed`、`--web-control`、`--pin-cores` 覆盖。

#### 键盘控制

```bash
python policy_deploy.py
```

程序先保持开机趴姿并进入**低层阻尼**，不主动抬升。按 `s` 执行起立过渡（趴姿 → 中间姿态 → 默认站姿），再按 `t` 进入 ONNX 策略测试；按 `t` 之前只保持站立、不做推理。测试前想放弃按 `d`，安全切回当前姿态阻尼。`X` 和 `Ctrl-C` 也是安全退出。

测试中的键位：

- 方向键（或小键盘 `8/2/4/6`）：前进/后退 `±MAX_X_SPEED`、左移/右移 `±MAX_Y_SPEED`（默认 `1.0 m/s`，即训练指令上限；可用 `--max-x-speed` / `--max-y-speed` 收紧——训练时 vy 范围仅 ±0.3 m/s）。
- `z`（或小键盘 `7`）：`+MAX_YAW_SPEED` 左转（默认 `1.0 rad/s`，可用 `--max-yaw-speed` 收紧）；`c`（或小键盘 `9`）：右转；play.py 的 X 在这里留给安全退出。
- 生效中的方向会叠加，但 POSIX 终端只自动重复最后按下的那个键：同时按住两个方向键时，`--key-timeout`（默认 `0.15 s`）过后只剩最后一个。需要组合时交替点按，或一次只按一个。
- `空格` 立即清零；松开按键同样在超时后归零。
- 隐藏参数 `--keyboard`、`--command` 为兼容旧启动脚本保留，实际被忽略。

#### Web 控制（Retroid / 手机）

```bash
python policy_deploy.py --web-control        # 换端口: --web-control 9090
```

开启内置 HTTP 服务（默认端口 8080），浏览器打开即得单页触摸手柄：左侧虚拟摇杆控制 vx/vy，右摇杆 X 轴控制转向；在 Retroid Pocket 4 这类安卓掌机上，浏览器 Gamepad API 会直接读取**实体摇杆**。页面上 `站起 (s)`、`测试 (t)`、`急停 (d)` 按钮驱动同一套状态机。指令为归一化摇杆值乘 `MAX_X_SPEED`/`MAX_Y_SPEED`/`MAX_YAW_SPEED`（带死区；`--max-x-speed`/`--max-y-speed`/`--max-yaw-speed` 对键盘挡位和网页摇杆同时生效）；页面停止发送超过 0.3 秒即回落到键盘指令源。网页没有退出功能：关闭页面或断网只会让指令归零，不影响部署进程。原厂 App 独占 SDK 通道，不能用来控制本部署。

### 机制与安全

#### 动作处理与关节限位

实验变体：策略原始动作不做幅度钳位。关节目标 = `默认姿态 + ACTION_SCALE × 原始动作`，再裁剪进 `sendMotorCmd` 的硬窗口（abad `±0.48`、hip `-1.15~2.97`、knee `-2.9~-0.65` rad），留 `1e-3 rad` 余量（float32 表示不了边界值，取到边界仍会被拒）。1 Hz 打印里的 `clip : N` 是自进入测试以来发生裁剪的控制周期数。

初始增益 `kp=20`、`kd=0.7` 与训练执行器刚度/阻尼一致；硬件验证前不要改动。

#### 退出与阻尼

所有退出路径（`d`、`X`、`Ctrl-C`、发送线程出错）都会在实测当前姿态上切换为旧版 LowLevel 阻尼指令（`kp=0`、`kd=3`）并保持 `--stop-hold-seconds` 秒，让机器人自然趴下。退出过程不会命令起立中间姿态。SDK 文档说明 HighLevel 与 LowLevel 不能并用，因此本部署始终保持单条 LowLevel 连接，而不是并行调用 `HighLevel.passive()`。本软件不能替代实体急停。
