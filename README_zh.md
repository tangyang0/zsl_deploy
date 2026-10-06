# ZSL-1 策略部署

部署与 sim2sim 回放：把 RobotLab 导出的速度策略 ONNX（平地或粗糙地形，任意 `[1,45] -> [1,12]` 接口）通过 ZSL-1 旧版 LowLevel Python SDK 跑在真机上——`policy_deploy.py` 负责真机部署，`policy_sim2sim.py` 负责先在 MuJoCo 里回放验证。

## 环境搭建

SDK 的 Python 绑定按 CPython 3.10 编译（`mc_sdk_zsl_1_py.cpython-310-*`），因此环境必须是 Python 3.10。部署只需 `numpy` 和 `onnxruntime` 两个 pip 依赖；下面的 MuJoCo 回放额外需要 `pip install mujoco pygame-ce`。SDK 本体免安装：`policy_deploy.py` 运行时通过 `--sdk-lib` 加载绑定（默认指向 `genisom_l1_sdk_old` 内的 x86_64 版本）。

```bash
conda create -n zsl_sdk_py310 python=3.10 -y
conda activate zsl_sdk_py310
pip install numpy onnxruntime      # 部署
pip install mujoco pygame-ce       # 可选：sim2sim 回放
```

## 快速开始

```bash
cd /home/tangyang/workspace/zsl_deploy
python policy_deploy.py --dry-run   # 仅验证模型：不加载 SDK、不连接机器人
```

默认模型是 `smooth_ft` 策略；用 `--model` 换其他导出的策略（如 `config2` 或粗糙地形 run），接口在启动时校验，布局不符会直接报错。

## MuJoCo sim2sim 回放

`policy_sim2sim.py` 在上真机之前，用同一份 ONNX 策略在 MuJoCo 里回放。它实时给训练 URDF 打补丁（浮动基座、mesh 绝对路径、注入地面），并跑与部署完全一致的管线：观测/动作常量直接从 `policy_deploy.py` 导入，物理 200 Hz、策略 50 Hz，PD 增益 `kp=20`/`kd=0.7` 与 28 Nm 力矩限幅同训练，关节目标同样按 SDK 硬窗口裁剪并统计 clip 次数。机体系角速度直接取自由关节的 `qvel[3:6]`（已用四元数有限差分验证）。

地面为浅色平面加每米一条深色网格线（纯 visual 几何，不参与碰撞），机器狗走动时有明确位移参照。刻意不用纹理：本机上程序化纹理在离屏渲染正常、却在 GLFW viewer 的上下文里被静默丢弃（已截窗验证），几何线条则在任何上下文都能渲染。

```bash
python policy_sim2sim.py --headless 500 --command 0.5 0 0   # 无窗口冒烟测试
python policy_sim2sim.py                                    # 交互：3D viewer + 控制面板
```

交互模式（默认）打开 mujoco 原生 3D viewer 和一个 pygame 控制小面板，键盘语义与 play.py 完全一致：指令 = 面板中当前按住的所有键之和——按住即走、松开即停、可任意组合，初始指令为零（不按键机器人就站着）。键位与 Se2Keyboard 相同：方向键或小键盘/主键区 `8/2/4/6` 平移，`z`/`7` 左转，`x`/`9` 右转，`l`/`空格` 清零，`r` 复位。相机注视点每帧跟随机器狗（鼠标旋转/缩放仍可用）。摔倒（基座低于 `0.2 m`）自动复位。之所以加 pygame 面板，是因为 viewer 的按键回调没有松开事件；面板是纯软件窗口，避开了本机某些会话上失败的离屏 EGL/GLX 上下文。需要 `pip install pygame-ce`。

`--native-viewer` 只开原生 viewer（其按键回调每次按下只触发一次、无重复/松开事件——已用合成按键注入验证——因此该模式下按键持续生效直到清零）。上真机前先用它筛策略：例如 `smooth_ft` 在此以 `0.45 m/s` 跟踪 `0.5 m/s` 指令，而 `config2` 在 MuJoCo 里只吃低速（`0.25` 可跟踪、`0.5` 摔倒），尽管它在 Isaac 里能走。

## 运行位置与网络设置

**推理机模式（默认）**：通过 WiFi 在推理机上运行。本机 IP `192.168.234.16`，机器狗 `192.168.234.1`，无线地址不同时用 `--local-ip`/`--dog-ip` 覆盖。前提是狗端 `/opt/export/config/sdk_config.yaml` 里为 `target_ip: "192.168.234.16"`（运控按此地址推送状态；改动需重启机器狗生效）。

**板载模式**：在狗的主控上运行（`ssh l1`，文件在 `~/zsl_deploy_onboard`，SDK 在 `sdk/`）。将狗端 `target_ip` 改为 `192.168.234.1`（狗自己的 ap0 地址，与出厂备份一致）并重启后：

```bash
python3 policy_deploy.py --sdk-lib sdk \
    --local-ip 192.168.234.1 --dog-ip 192.168.234.1 --model models/<policy>.onnx
```

即使在狗上运行，`--dog-ip` 也必须保持 `192.168.234.1`：`mc_ctrl` 的指令套接字绑定在 ap0 地址上，从不监听 loopback。在推理机/板载两种模式之间切换 `target_ip` 都需要重启，且另一侧在此期间收不到 SDK 数据；同一时刻只允许一个 SDK 客户端。

## 键盘状态机

```bash
python policy_deploy.py
```

程序先保持开机趴姿并进入**低层阻尼**，不主动抬升。按 `s` 执行起立过渡（趴姿 → 中间姿态 → 默认站姿），再按 `t` 进入 ONNX 策略测试；按 `t` 之前只保持站立、不做推理。测试前想放弃按 `d`，安全切回当前姿态阻尼。`X` 和 `Ctrl-C` 也是安全退出。

测试中的键位：

- 方向键（或小键盘 `8/2/4/6`）：前进/后退、左移/右移，速度 `±LOW_SPEED`（当前 `0.8 m/s`，在训练指令范围内）。
- `z`（或小键盘 `7`）：+1.0 rad/s 左转；`c`（或小键盘 `9`）：右转。play.py 的 X 键位此处未用，因为 `X` 是安全退出。
- 生效中的方向会叠加，但 POSIX 终端只自动重复最后按下的那个键：同时按住两个方向键时，`--key-timeout`（默认 `0.15 s`）过后只剩最后一个。需要组合时交替点按，或一次只按一个。
- `空格` 立即清零；松开按键同样在超时后归零。
- 隐藏参数 `--keyboard`、`--command` 为兼容旧启动脚本保留，实际被忽略。

## Web 控制（Retroid / 手机）

`--web-control [端口]` 开启内置 HTTP 服务（默认 8080），浏览器打开即得单页触摸手柄：左侧虚拟摇杆控制 vx/vy，右摇杆 X 轴控制转向；在 Retroid Pocket 4 这类安卓掌机上，浏览器 Gamepad API 会直接读取**实体摇杆**。页面上 `站起 (s)`、`测试 (t)`、`急停 (d)` 按钮驱动同一套状态机。指令为归一化摇杆值乘 `LOW_SPEED`/`TURN_SPEED`（带死区）；页面停止发送超过 0.3 秒即回落到键盘指令源。网页**故意**不提供退出功能：关闭浏览器或断 WiFi 只会令指令归零，绝不会杀掉部署进程。原厂 App 无法复用：它独占 SDK 通道，你的策略运行时 App 根本没有链路。

可用 `--model`、`--sdk-lib`、`--local-ip`、`--dog-ip`、`--port`、`--kp`、`--kd`、`--key-timeout`、`--web-control` 覆盖默认值。

## 策略动作处理

实验变体：策略原始动作**不做幅度钳位**。关节目标 = `默认姿态 + ACTION_SCALE × 原始动作`，再裁剪进 `sendMotorCmd` 的硬窗口（abad `±0.48`、hip `-1.15~2.97`、knee `-2.9~-0.65` rad），并留 `1e-3 rad` 内缩余量：float32 无法精确表示 knee/hip 边界、会向窗口外侧取整，钉在边界上的目标仍会被拒绝。1 Hz 状态打印里的 `clip : N` 表示最近一秒限位裁剪生效的控制周期数，用来量化策略想越界的程度。

初始增益 `kp=20`、`kd=0.7` 与训练执行器刚度/阻尼一致；硬件验证前不要改动。

## 退出与阻尼

所有退出路径（`d`、`X`、`Ctrl-C`、发送线程出错）都会在实测当前姿态上切换为旧版 LowLevel 阻尼指令（`kp=0`、`kd=3`）并保持 `--stop-hold-seconds` 秒，让机器人自然趴下。退出过程绝不会命令起立中间姿态。SDK 文档说明 HighLevel 与 LowLevel 不能并用，因此本部署始终保持单条 LowLevel 连接，而不是并行调用 `HighLevel.passive()`。本软件不能替代实体急停。
