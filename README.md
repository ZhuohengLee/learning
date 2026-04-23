# Learning

这个目录现在只保留一条训练和部署链：

```text
PyTorch MLP -> ONNX -> ESP-PPQ -> .espdl -> ESP-DL
```

目标不是完整导航，而是先把三条低层控制轴做稳：

- 深度：`u_total = u_base + u_residual`
- 前进：`forward_cmd_total = forward_cmd_base + forward_cmd_residual`
- 转向：`yaw_cmd_total = yaw_cmd_base + yaw_cmd_residual`

## 当前结构

- [data.py](data.py)
  共享数据层。负责 CSV 读取、训练样本过滤、滑动窗口、target 回退和标准化。
- [model.py](model.py)
  PyTorch MLP 模型定义。
- [train.py](train.py)
  顶层训练入口。
- [export.py](export.py)
  顶层导出入口，负责 ONNX 和 `.espdl`。
- [tests](tests)
  共享数据层测试和 PyTorch 导出链测试。

旧的纯 Python MLP 链路已经删除。

## 阅读顺序

1. 先看 [data.py](data.py)
2. 再看 [train.py](train.py)
3. 最后看 [export.py](export.py)

## 前进和转向任务编码

当前默认支持离散任务编码：

| 字段 | 推荐编码 | 含义 |
| --- | --- | --- |
| `forward_cmd_base` | `0 / 1` | `0` = 当前没有前进任务，`1` = 当前正在前进 |
| `forward_cmd_residual` | `0` | 神经网络未接入前先记 0 |
| `forward_cmd_total` | 先等于 `forward_cmd_base` | 当前最终前进命令 |
| `yaw_cmd_base` | `-1 / 0 / 1` | `-1` = 左转，`0` = 不转，`1` = 右转 |
| `yaw_cmd_residual` | `0` | 神经网络未接入前先记 0 |
| `yaw_cmd_total` | 先等于 `yaw_cmd_base` | 当前最终转向命令 |

## 训练数据 CSV

推荐固定输出这些列：

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
5. `L50` 这类目标深度命令执行期间的数据要保留。
6. 前进和转向任务执行期间的数据也要保留。
7. 只过滤真正的 `j/k` 浮力直控帧。
8. 如果没有显式 residual 列，至少保留每一轴的 `base` 和 `total`。

## 默认特征和 target

默认共享输入特征：

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

默认 target：

- `depth` -> `u_residual`
- `forward` -> `forward_cmd_residual`
- `yaw` -> `yaw_cmd_residual`

如果显式 residual 列不存在，会自动回退到：

- `u_residual` -> `u_total - u_base`
- `forward_cmd_residual` -> `forward_cmd_total - forward_cmd_base`
- `yaw_cmd_residual` -> `yaw_cmd_total - yaw_cmd_base`

## 训练

```powershell
python -m learning.train `
  --csv D:\path\to\sensors.csv `
  --output-dir artifacts\pytorch_axis_models `
  --hidden-dims 24 12 `
  --epochs 300 `
  --batch-size 64
```

训练输出：

- `depth_model.pt`
- `forward_model.pt`
- `yaw_model.pt`
- `axis_manifest.json`

## 导出 ONNX

```powershell
python -m learning.export `
  --model-dir artifacts\pytorch_axis_models `
  --output-dir artifacts\onnx_axis_models `
  --opset 18
```

输出：

- `depth_model.onnx`
- `forward_model.onnx`
- `yaw_model.onnx`
- 每个 ONNX 对应一个 `.metadata.json`
- `axis_manifest.json`

## 导出 .espdl

```powershell
python -m learning.export `
  --model-dir artifacts\pytorch_axis_models `
  --output-dir artifacts\onnx_axis_models `
  --espdl-output-dir artifacts\espdl_axis_models `
  --calibration-csv D:\path\to\sensors.csv `
  --target esp32s3 `
  --num-bits 8 `
  --calib-steps 32 `
  --export-test-values
```

输出：

- `depth_model.espdl`
- `forward_model.espdl`
- `yaw_model.espdl`
- 每个 `.espdl` 对应一个 `.info`
- 每个 `.espdl` 对应一个 `.json`
- 每个 `.espdl` 对应一个 `.espdl.metadata.json`
- `espdl_axis_manifest.json`

## 依赖

训练和 ONNX 导出：

- `torch`
- `onnx`

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
