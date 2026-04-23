# Learning

这个目录现在只保留一条训练和部署链：

```text
PyTorch joint MLP -> ONNX -> ESP-PPQ -> .espdl -> ESP-DL
```

目标不是完整智能导航，而是先把低层控制做稳。  
当前模型不是三个独立网络，而是：

- 一个共享主干
- 三个输出头
- 同时预测 `depth / forward / yaw` 三个残差

也就是：

- `u_total = u_base + u_residual`
- `forward_cmd_total = forward_cmd_base + forward_cmd_residual`
- `yaw_cmd_total = yaw_cmd_base + yaw_cmd_residual`

## 当前结构

- [data.py](data.py)
  共享数据层。负责 CSV 读取、样本过滤、滑动窗口、target 回退和标准化。
- [model.py](model.py)
  共享主干 + 三输出头的 PyTorch MLP。
- [train.py](train.py)
  训练入口。输出一个 joint bundle。
- [export.py](export.py)
  导出入口。把一个 joint bundle 导出成一个 ONNX，必要时继续量化成一个 `.espdl`。
- [tests](tests)
  数据层、联合训练、联合导出的测试。

## 阅读路线

1. 先看 [data.py](data.py)
2. 再看 [train.py](train.py)
3. 再看 [model.py](model.py)
4. 最后看 [export.py](export.py)

## 方法概述

当前方法是：

```text
手写基线控制器 + 神经网络残差补偿
```

网络输入是最近几帧的状态、基线命令和任务编码。  
网络输出是三个轴的残差修正量：

- 深度头输出 `u_residual`
- 前进头输出 `forward_cmd_residual`
- 转向头输出 `yaw_cmd_residual`

这样做的目的不是替代基线控制器，而是在基线之上补偿：

- 水动力扰动
- 耦合效应
- 传感器噪声下的非线性偏差
- 推进周期和姿态之间的关系

## 前进和转向编码

当前默认支持你现在这套离散任务编码：

| 字段 | 推荐编码 | 含义 |
| --- | --- | --- |
| `forward_cmd_base` | `0 / 1` | `0` 表示当前没有前进任务，`1` 表示当前正在前进 |
| `forward_cmd_residual` | 初期先记 `0` | 神经网络未接入前先置零 |
| `forward_cmd_total` | 初期先等于 `forward_cmd_base` | 当前最终前进命令 |
| `forward_phase_interval_ms` | 正数 | 当前推进周期，网络会学习推进周期和姿态/转向的关系 |
| `yaw_cmd_base` | `-1 / 0 / 1` | `-1` 左转，`0` 不转，`1` 右转 |
| `yaw_cmd_residual` | 初期先记 `0` | 神经网络未接入前先置零 |
| `yaw_cmd_total` | 初期先等于 `yaw_cmd_base` | 当前最终转向命令 |

说明：

- 这里的 `forward_cmd_base` 和 `yaw_cmd_base` 可以是离散任务编码，不要求你现在就有连续推进量。
- `forward_phase_interval_ms` 很重要，因为你明确说过你更想学习“推进周期和转向/姿态的关系”。

## 训练数据 CSV

推荐至少输出这些列：

```text
session_id
timestamp_ms
dt_ms
robot_mode
control_mode
depth_valid
imu_valid
battery_v
target_depth_cm
filtered_depth_cm
depth_speed_cm_s
depth_accel_cm_s2
roll_deg
pitch_deg
gyro_x_deg_s
gyro_y_deg_s
gyro_z_deg_s
front_distance_cm
left_distance_cm
right_distance_cm
depth_err_cm
u_base
u_residual
u_total
forward_cmd_base
forward_cmd_residual
forward_cmd_total
forward_phase_interval_ms
yaw_cmd_base
yaw_cmd_residual
yaw_cmd_total
buoyancy_dir_applied
buoyancy_pwm_applied
actuator_mask
balancing
emergency_stop
```

要求：

1. 文件必须是 `UTF-8 CSV`，第一行必须是表头。
2. 一行代表一个控制周期的完整状态。
3. 数值列只写纯数字。
4. 同一个 `session_id` 内的 `timestamp_ms` 必须单调递增。
5. `L50` 这种目标深度命令执行期间的数据要保留。
6. 前进和转向任务执行期间的数据也要保留。
7. 只过滤真正的 `j/k` 浮力直控帧。
8. 如果暂时没有显式 residual 列，至少要保留每一轴的 `base` 和 `total`。

## 自动 residual 回退

如果 CSV 里没有显式 residual 列，训练代码会自动回退为：

- `u_residual = u_total - u_base`
- `forward_cmd_residual = forward_cmd_total - forward_cmd_base`
- `yaw_cmd_residual = yaw_cmd_total - yaw_cmd_base`

所以最少你也要把 `base` 和 `total` 打出来。

## 默认输入特征

默认 joint 模型使用这一组共享特征：

- `depth_err_cm`
- `depth_speed_cm_s`
- `depth_accel_cm_s2`
- `roll_deg`
- `pitch_deg`
- `gyro_x_deg_s`
- `gyro_y_deg_s`
- `gyro_z_deg_s`
- `battery_v`
- `buoyancy_pwm_applied`
- `front_distance_cm`
- `left_distance_cm`
- `right_distance_cm`
- `u_base`
- `forward_cmd_base`
- `forward_phase_interval_ms`
- `yaw_cmd_base`

这些特征会按时间窗口展开，变成：

```text
t-4_*
t-3_*
t-2_*
t-1_*
t-0_*
```

默认窗口大小是 `5`。

## 默认输出目标

joint 模型同时学习三列：

- `depth -> u_residual`
- `forward -> forward_cmd_residual`
- `yaw -> yaw_cmd_residual`

## 训练

```powershell
python -m learning.train `
  --csv D:\path\to\sensors.csv `
  --output-dir artifacts\joint_model `
  --hidden-dims 24 12 `
  --epochs 300 `
  --batch-size 64
```

训练输出：

- `joint_model.pt`
- `joint_manifest.json`

`joint_model.pt` 里包含：

- 模型结构
- 权重
- 输入标准化参数
- 输出标准化参数
- 轴顺序
- target 列映射

## 导出 ONNX

```powershell
python -m learning.export `
  --model artifacts\joint_model\joint_model.pt `
  --output artifacts\joint_model\joint_model.onnx `
  --opset 13
```

输出：

- `joint_model.onnx`
- `joint_model.metadata.json`

ONNX 的输出是一个张量：

```text
residuals[0] = depth residual
residuals[1] = forward residual
residuals[2] = yaw residual
```

顺序由 `axis_names` 决定，默认是：

```text
["depth", "forward", "yaw"]
```

## 导出 `.espdl`

```powershell
python -m learning.export `
  --model artifacts\joint_model\joint_model.pt `
  --output artifacts\joint_model\joint_model.onnx `
  --espdl-output artifacts\joint_model\joint_model.espdl `
  --calibration-csv D:\path\to\sensors.csv `
  --target esp32s3 `
  --num-bits 8 `
  --calib-steps 32 `
  --export-test-values
```

输出：

- `joint_model.espdl`
- `joint_model.info`
- `joint_model.json`
- `joint_model.espdl.metadata.json`

## ESP32-S3 / ESP-DL 对接说明

你现在是 `ESP32-S3`，所以这条链是可用的：

```text
PC 上训练 joint_model.pt
-> 导出 joint_model.onnx
-> ESP-PPQ 量化
-> joint_model.espdl
-> ESP-DL 在 ESP32-S3 上推理
```

板端拿到输出后，按默认轴顺序解释即可：

```text
index 0 -> depth residual
index 1 -> forward residual
index 2 -> yaw residual
```

## 依赖

训练和 ONNX 导出：

- `torch`
- `onnx`
- `onnxscript`

`.espdl` 导出：

- `esp-ppq`
  导入路径是 `ppq`

## 测试

在 `learning` 目录下运行：

```powershell
python -m unittest discover tests -v
```

说明：

- 不依赖 `torch` 的测试会一直跑。
- 依赖 `torch` / `onnx` / `ppq` 的测试会在缺依赖时自动跳过。
