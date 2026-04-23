# Learning

这个目录用于训练“基线控制器 + 神经网络残差补偿”模型

核心形式是：

```text
final_command = base_controller_command + neural_residual
```

当前覆盖三条控制轴：

- 深度轴：`u_total = u_base + u_residual`
- 前进轴：`forward_cmd_total = forward_cmd_base + forward_cmd_residual`
- 转向轴：`yaw_cmd_total = yaw_cmd_base + yaw_cmd_residual`

## 当前结构

- [data.py](data.py)
  共享数据层。负责 CSV 读取、训练样本过滤、滑动窗口、target 回退、标准化。
- [python_mlp](python_mlp)
  旧链路。
  训练产物是 `JSON bundle`，导出产物是给现有固件直接使用的 `ESP32 C++ 头文件`。
- [pytorch_mlp](pytorch_mlp)
  新链路。
  训练产物是 `.pt bundle`，导出分两段：
  1. `PyTorch -> ONNX`
  2. `ONNX -> ESP-PPQ -> .espdl`
- [tests](tests)
  测试共享数据层、旧链路和新链路。

顶层不再保留多余的 `train.py` / `export.py` / `model.py` 包装文件。

## 什么时候用哪套链路

- 如果你现在要继续沿用你现有的自定义板端推理代码：
  用 `python_mlp`
- 如果你现在要为 `ESP32-S3 + ESP-DL` 准备标准部署链：
  用 `pytorch_mlp`

## 推荐阅读顺序

1. 先看 [data.py](data.py)
   两套后端共用同一套采数合同和样本构建逻辑。
2. 再看 [python_mlp/train.py](python_mlp/train.py)
   这是旧链路的训练入口。
3. 然后看 [pytorch_mlp/train.py](pytorch_mlp/train.py)
   这是新的 PyTorch 三轴训练入口。
4. 最后看两个后端各自的 `export.py`
   一个导出 C++ 头文件，一个导出 ONNX / `.espdl`。

## 前进和转向的任务编码

当前默认支持离散任务编码：

| 字段 | 推荐编码 | 含义 |
| --- | --- | --- |
| `forward_cmd_base` | `0 / 1` | `0` = 当前没有前进任务，`1` = 当前正在执行前进任务 |
| `forward_cmd_residual` | 先固定 `0` | 神经网络未接入前先不补偿 |
| `forward_cmd_total` | 先等于 `forward_cmd_base` | 最终前进任务命令 |
| `yaw_cmd_base` | `-1 / 0 / 1` | `-1` = 左转，`0` = 不转，`1` = 右转 |
| `yaw_cmd_residual` | 先固定 `0` | 神经网络未接入前先不补偿 |
| `yaw_cmd_total` | 先等于 `yaw_cmd_base` | 最终转向任务命令 |

这意味着第一版模型不要求前进和转向一定是连续力度，只要日志编码稳定一致，就可以先训练。

## 训练数据 CSV 格式

推荐固定输出下面这些列：

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

采数要求：

1. 文件必须是 `UTF-8 CSV`，第一行必须是表头。
2. 一行代表一个控制周期的完整状态。
3. 数值列只写纯数字，不要把单位写进单元格。
4. 同一个 `session_id` 内的 `timestamp_ms` 必须单调递增。
5. `L50` 这种高层闭环命令执行期间的数据要保留。
6. 前进和转向任务执行期间的数据也要保留。
7. 只过滤真正的 `j/k` 浮力直控帧，不过滤正常的目标深度 / 前进 / 转向任务帧。
8. 如果没有显式 residual 列，至少要保留每一轴的 `base` 和 `total`，训练代码会自动用 `total - base` 回退。

## 神经网络还没接上时怎么填

如果板端还没有神经网络推理，先这样记：

- `u_residual = 0`
- `u_total = u_base`
- `forward_cmd_residual = 0`
- `forward_cmd_total = forward_cmd_base`
- `yaw_cmd_residual = 0`
- `yaw_cmd_total = yaw_cmd_base`

如果前进和转向还没有连续量，直接这样记：

- `forward_cmd_base = 0` 表示当前没有前进任务
- `forward_cmd_base = 1` 表示当前正在前进
- `yaw_cmd_base = -1` 表示左转
- `yaw_cmd_base = 0` 表示不转
- `yaw_cmd_base = 1` 表示右转

## 当前默认特征

共享输入特征默认是：

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

## 旧链路：自定义 Python MLP

训练：

```powershell
python -m learning.python_mlp.train `
  --csv D:\path\to\sensors.csv `
  --output-dir artifacts\axis_models
```

导出到当前自定义固件头文件：

```powershell
python -m learning.python_mlp.export `
  --model-dir artifacts\axis_models `
  --output-dir D:\path\to\Squid-Robot\ESP32
```

输出：

- `depth_model.json`
- `forward_model.json`
- `yaw_model.json`
- `axis_manifest.json`
- `DepthResidualModelData.h`
- `ForwardResidualModelData.h`
- `YawResidualModelData.h`

## 新链路：PyTorch MLP

训练：

```powershell
python -m learning.pytorch_mlp.train `
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

### 第一步：导出 ONNX

```powershell
python -m learning.pytorch_mlp.export `
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

### 第二步：继续导出 `.espdl`

```powershell
python -m learning.pytorch_mlp.export `
  --model-dir artifacts\pytorch_axis_models `
  --output-dir artifacts\onnx_axis_models `
  --espdl-output-dir artifacts\espdl_axis_models `
  --calibration-csv D:\path\to\sensors.csv `
  --target esp32s3 `
  --num-bits 8 `
  --calib-steps 32 `
  --export-test-values
```

这条命令会先导出 ONNX，再继续调用 `ESP-PPQ` 做 PTQ 量化，并输出：

- `depth_model.espdl`
- `forward_model.espdl`
- `yaw_model.espdl`
- 每个 `.espdl` 对应一个 `.info`
- 每个 `.espdl` 对应一个 `.json`
- 每个 `.espdl` 额外对应一个 `.espdl.metadata.json`
- `espdl_axis_manifest.json`

说明：

- `--calibration-csv` 必须是和训练同格式的代表性真实数据，最好来自水下真实控制过程。
- `--calib-steps` 控制每个轴最多抽取多少个代表性窗口用于 PTQ。
- `--export-test-values` 会把一组真实输入/输出嵌进 `.espdl`，方便后面在板端调用 `model->test()` 做一致性检查。
- 当 `--target esp32` 时，代码会按乐鑫官方要求把 ESP-PPQ 目标映射成 `c`。

## 依赖

旧链路：

- Python 标准库

新链路训练和 ONNX 导出：

- `torch`
- `onnx`

新链路 `.espdl` 导出：

- `esp-ppq`
  安装后导入路径是 `ppq`

## ESP-DL 相关说明

你之前拿到的那套 `optimizer/calibrator/export_coefficient_to_cpp` 流程可以作为历史参考，但当前更推荐的主线是：

```text
PyTorch -> ONNX -> ESP-PPQ -> .espdl -> ESP-DL
```

也就是说，`pytorch_mlp` 现在已经负责：

- 训练 PyTorch MLP
- 导出 ONNX
- 基于代表性 CSV 校准数据继续导出 `.espdl`

还没包含的部分是：

- ESP-IDF 工程里如何加载这些 `.espdl`
- 板端前处理 / 后处理封装
- 与你现有控制循环的最终接线

## 测试

在 `learning` 目录下运行：

```powershell
python -m unittest discover tests -v
```

说明：

- 旧链路测试会一直跑。
- 新链路里不依赖 `torch` 的测试也会一直跑。
- 依赖 `torch` / `onnx` / `ppq` 的测试会在缺依赖时自动跳过。
