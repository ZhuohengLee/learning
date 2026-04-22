# Learning

这个仓库放的是残差控制训练代码，目标不是完整智能导航，而是先把低层控制做稳。

核心思路：

```text
final_command = base_controller_command + neural_residual
```

当前已经覆盖三条控制轴：

- 深度轴：`u_total = u_base + u_residual`
- 前进轴：`forward_cmd_total = forward_cmd_base + forward_cmd_residual`
- 转向轴：`yaw_cmd_total = yaw_cmd_base + yaw_cmd_residual`

也就是说，基线控制器先保证系统可控，神经网络只学习“还要补多少”。

## 现在只有两个主脚本

- [train.py](train.py)
  唯一训练入口。一次读一个 CSV，同时训练 `depth / forward / yaw` 三个模型。
- [export.py](export.py)
  唯一导出入口。既可以一次导出三轴，也可以单独导出一个模型 bundle。

剩下的 Python 文件只负责：

- [data.py](data.py)
  训练数据读取、过滤、滑动窗口、target 回退规则。
- [model.py](model.py)
  纯 Python 小型 MLP。

## 阅读路线

建议按这个顺序看：

1. [data.py](data.py)
   先看训练数据是怎么过滤、切片、构造 target 的。
2. [train.py](train.py)
   看公开单入口如何一次训练 `depth / forward / yaw` 三个模型。
3. [model.py](model.py)
   看小型 MLP 的结构和训练过程。
4. [export.py](export.py)
   看训练后怎么导出成 ESP32 头文件。

## 发给采数同事的直接要求

如果别人要帮你采训练数据，直接把下面这几条发给他：

1. 日志格式必须是 `UTF-8 CSV`，第一行必须是表头。
2. 一行代表一个控制周期的完整状态，不要把单位写进单元格里。
3. 数值列只能放纯数字，比如 `12.5`、`0`、`-3.2`，不能写成 `12.5cm`。
4. 状态位统一用 `0/1`。
5. 同一个 `session_id` 内，`timestamp_ms` 必须单调递增。
6. `L50` 这种高层闭环命令执行期间的数据要保留，前进和转向任务的数据也要保留。
7. 真正的 `j/k` 浮力直控帧不需要专门采，代码会自动过滤这类直控样本。
8. 如果某个轴没有打印显式 residual 列，也必须至少打印这个轴的 `base` 和 `total`，这样训练代码才能用 `total - base` 自动恢复 residual target。

## 正式 CSV 表头

当前推荐固定使用下面这一组列，并按这个顺序输出：

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

最少要点：

- `forward_phase_interval_ms` 现在必须打印，因为前进和转向模型要学习推进周期和姿态/转向之间的关系。
- `imu_valid=0` 的帧会被过滤，所以如果 IMU 还没接好，不要拿那批数据训练。
- `depth_valid=0`、`balancing=1`、`emergency_stop=1` 的帧也会被过滤。

## 可以直接参考的 CSV 示例

下面是一段最小示例，别人照这个格式输出就行：

```csv
session_id,timestamp_ms,dt_ms,robot_mode,control_mode,depth_valid,imu_valid,battery_v,target_depth_cm,filtered_depth_cm,depth_speed_cm_s,depth_accel_cm_s2,roll_deg,pitch_deg,gyro_x_deg_s,gyro_y_deg_s,gyro_z_deg_s,front_distance_cm,left_distance_cm,right_distance_cm,depth_err_cm,u_base,u_residual,u_total,forward_cmd_base,forward_cmd_residual,forward_cmd_total,forward_phase_interval_ms,yaw_cmd_base,yaw_cmd_residual,yaw_cmd_total,buoyancy_dir_applied,buoyancy_pwm_applied,actuator_mask,balancing,emergency_stop
session_0007,15320,40,auto,auto,1,1,11.9,50.0,46.8,-2.4,0.3,1.8,-0.6,0.7,-1.1,0.2,85.0,110.0,96.0,3.2,24.0,-3.0,21.0,35.0,4.0,39.0,180.0,12.0,-2.0,10.0,1,132,13,0,0
```

## 训练时真正会用到哪些列

### 样本过滤字段

当下面这些列存在时，`data.py` 会用它们过滤无效样本：

- `session_id`
- `timestamp_ms`
- `control_mode`
- `depth_valid`
- `imu_valid`
- `balancing`
- `emergency_stop`

过滤规则：

- `depth_valid=0` 的帧会被丢掉
- `imu_valid=0` 的帧会被丢掉
- `balancing=1` 的帧会被丢掉
- `emergency_stop=1` 的帧会被丢掉
- `L50` 这类高层目标命令产生的闭环执行帧会被保留
- 前进 / 转向任务帧同样会被保留
- 只有真正的 `j/k` 浮力直控帧会被丢掉
- 时间戳倒退或相邻帧时间差过大时，会被切成不同序列

### 默认共享输入特征

`train.py` 默认使用下面这些输入列训练三个轴：

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

### 三个轴的 target 规则

默认 target：

- `depth` -> `u_residual`
- `forward` -> `forward_cmd_residual`
- `yaw` -> `yaw_cmd_residual`

如果显式 residual 列不存在，代码会自动回退到：

- `u_residual` -> `u_total - u_base`
- `forward_cmd_residual` -> `forward_cmd_total - forward_cmd_base`
- `yaw_cmd_residual` -> `yaw_cmd_total - yaw_cmd_base`

这就是为什么日志里至少要保留每个轴的 `base` 和 `total`。

## 训练命令

在这个仓库根目录下运行：

```powershell
python train.py `
  --csv D:\path\to\sensors.csv `
  --output-dir artifacts\axis_models
```

常用参数：

- `--window-size`
- `--feature-columns`
- `--hidden-dims`
- `--epochs`
- `--learning-rate`
- `--l2`
- `--val-fraction`
- `--max-dt-ms`
- `--seed`
- `--print-every`
- `--depth-target-column`
- `--forward-target-column`
- `--yaw-target-column`

示例：

```powershell
python train.py `
  --csv D:\path\to\sensors.csv `
  --output-dir artifacts\axis_models `
  --window-size 5 `
  --hidden-dims 24 12 `
  --epochs 400 `
  --learning-rate 0.008
```

训练完成后会输出：

- `artifacts\axis_models\depth_model.json`
- `artifacts\axis_models\forward_model.json`
- `artifacts\axis_models\yaw_model.json`
- `artifacts\axis_models\axis_manifest.json`

## 导出到 ESP32

### 一次导出三轴

```powershell
python export.py `
  --model-dir artifacts\axis_models `
  --output-dir D:\path\to\Squid-Robot\ESP32
```

它会生成：

- `DepthResidualModelData.h`
- `ForwardResidualModelData.h`
- `YawResidualModelData.h`

### 单独导出一个 bundle

```powershell
python export.py `
  --model artifacts\axis_models\forward_model.json `
  --output D:\path\to\Squid-Robot\ESP32\ForwardResidualModelData.h `
  --namespace forward_residual_model `
  --include-guard ESP32_FORWARD_RESIDUAL_MODEL_DATA_H
```

当前导出器支持两种板端特征契约：

- `depth_legacy`
- `shared_three_axis`

## 测试

在仓库根目录运行：

```powershell
python -m unittest discover tests -v
```

## 如果以后改了数据格式

最少要同步检查这些文件：

- [data.py](data.py)
- [train.py](train.py)
- [export.py](export.py)
- 这个 README
