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

## 当前推荐的命令编码

这个仓库当前默认兼容两种命令表达：

- 连续控制量
  例如 `u_base` 直接是 `-100 ~ 100` 这样的连续指令。
- 离散任务编码
  适合你目前这种“还没有前进多少、转向多少连续量”的阶段。

当前对前进轴和转向轴，推荐先用离散任务编码：

| 字段 | 推荐编码 | 含义 |
| --- | --- | --- |
| `forward_cmd_base` | `0 / 1` | `0` = 当前没有前进任务，`1` = 当前正在执行前进任务 |
| `forward_cmd_residual` | 先固定 `0` | 神经网络未接入前先不补偿 |
| `forward_cmd_total` | 先等于 `forward_cmd_base` | 最终前进任务命令 |
| `yaw_cmd_base` | `-1 / 0 / 1` | `-1` = 左转，`0` = 不转，`1` = 右转 |
| `yaw_cmd_residual` | 先固定 `0` | 神经网络未接入前先不补偿 |
| `yaw_cmd_total` | 先等于 `yaw_cmd_base` | 最终转向任务命令 |

这意味着当前训练代码不会假设前进和转向一定是连续力度。只要日志里编码稳定一致，第一版模型就能先学“任务状态 + 推进周期 + 姿态/扰动”的关系。

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
9. 如果前进和转向还没有连续量，就按本 README 里的离散任务编码来记：
   `forward_cmd_base = 0/1`
   `yaw_cmd_base = -1/0/1`

## 采数同事的实现顺序

推荐把一行日志定义为“同一个控制周期结束时的完整快照”，按下面顺序实现：

1. 周期开始先记录 `timestamp_ms`，并用它和上一周期时间差得到 `dt_ms`。
2. 读取本周期传感器原始值，并更新 `depth_valid`、`imu_valid`、`battery_v`、距离传感器状态。
3. 用滤波器或状态估计器得到 `filtered_depth_cm`、`depth_speed_cm_s`、`depth_accel_cm_s2`、`roll_deg`、`pitch_deg`、`gyro_*_deg_s`。
4. 读取当前任务和模式，得到 `robot_mode`、`control_mode`、`target_depth_cm`、`forward_phase_interval_ms`。
5. 在控制器里显式计算内部控制量：
   `depth_err_cm = target_depth_cm - filtered_depth_cm`
6. 计算三个轴的基线输出、残差输出、最终输出：
   `u_base / u_residual / u_total`
   `forward_cmd_base / forward_cmd_residual / forward_cmd_total`
   `yaw_cmd_base / yaw_cmd_residual / yaw_cmd_total`
   如果前进/转向还没有连续控制量，就直接使用离散任务编码：
   `forward_cmd_base = 0/1`
   `yaw_cmd_base = -1/0/1`
7. 把最终输出映射成真正下发到执行器的方向、PWM 和开关状态，得到
   `buoyancy_dir_applied`、`buoyancy_pwm_applied`、`actuator_mask`
8. 最后再把这一整行一次性写入 CSV。训练会把空字符串当成缺失值，所以训练相关列不要留空。

## 每个字段怎么得到

下面这张表是给实现采数的人看的，重点是说明“这个字段应该从哪里来”。

| 字段 | 怎么得到 | 备注 |
| --- | --- | --- |
| `session_id` | 当前一次开机或一次采样任务的固定会话 ID | 推荐直接用 SD 目录名或启动时间戳 |
| `timestamp_ms` | `millis()` 或单调递增系统时钟 | 必须单调递增 |
| `dt_ms` | `timestamp_ms - previous_timestamp_ms` | 第一帧可以写 `0` |
| `robot_mode` | 主状态机当前模式 | 例如 idle / auto / manual 的编码值 |
| `control_mode` | 控制输入来源 | 用来区分任务控制和直控 |
| `depth_valid` | 深度传感器健康标志 | 无效帧训练时会过滤 |
| `imu_valid` | IMU 健康标志 | 无效帧训练时会过滤 |
| `battery_v` | 电池 ADC 读数换算成电压 | 不要一直写 `0.00` |
| `target_depth_cm` | 当前深度控制器内部实际使用的目标深度 | 不要留空；没有目标就写当前保持深度 |
| `filtered_depth_cm` | 深度传感器原始值经过滤波后的结果 | 推荐记录控制器真正用到的值 |
| `depth_speed_cm_s` | 深度的一阶差分或状态估计器输出 | 单位固定为 cm/s |
| `depth_accel_cm_s2` | 深度速度的一阶差分或状态估计器输出 | 单位固定为 cm/s^2 |
| `roll_deg` / `pitch_deg` | IMU 姿态解算结果 | 用控制器实际使用的姿态值 |
| `gyro_x_deg_s` / `gyro_y_deg_s` / `gyro_z_deg_s` | IMU 陀螺仪角速度读数 | 单位固定为 deg/s |
| `front_distance_cm` / `left_distance_cm` / `right_distance_cm` | 距离传感器测距值，统一换算到 cm | 如果当前没有这些传感器，不要留空；临时可写 `0`，但训练时必须同步把这些列从 `--feature-columns` 里删掉 |
| `depth_err_cm` | `target_depth_cm - filtered_depth_cm` | 这是训练默认深度模型的重要输入 |
| `u_base` | 手写深度基线控制器输出 | 不是传感器量，是控制器内部变量 |
| `u_residual` | 神经网络残差输出 | 神经网络未接入前固定写 `0` |
| `u_total` | `u_base + u_residual` 再经过限幅后的结果 | 这是最终深度控制指令 |
| `forward_cmd_base` | 手写前进任务编码或前进控制器输出 | 如果还没有连续前进量，推荐 `0` = 不前进，`1` = 前进 |
| `forward_cmd_residual` | 神经网络前进残差输出 | 神经网络未接入前固定写 `0` |
| `forward_cmd_total` | `forward_cmd_base + forward_cmd_residual` | 在离散编码阶段通常就等于 `0` 或 `1` |
| `forward_phase_interval_ms` | 当前推进周期设置值 | 前进和转向模型都会用到 |
| `yaw_cmd_base` | 手写转向任务编码或转向控制器输出 | 如果还没有连续转向量，推荐 `-1` = 左转，`0` = 不转，`1` = 右转 |
| `yaw_cmd_residual` | 神经网络转向残差输出 | 神经网络未接入前固定写 `0` |
| `yaw_cmd_total` | `yaw_cmd_base + yaw_cmd_residual` | 在离散编码阶段通常就等于 `-1 / 0 / 1` |
| `buoyancy_dir_applied` | 根据 `u_total` 实际下发的浮力方向 | 记录最终真正执行的方向 |
| `buoyancy_pwm_applied` | 根据 `u_total` 实际下发的 PWM | 记录最终真正执行的 PWM |
| `actuator_mask` | 本周期实际打开了哪些执行器的 bitmask | 用最终执行状态，不是目标状态 |
| `balancing` | 是否处于 balance / trim 特殊模式 | 该模式下样本会被过滤 |
| `emergency_stop` | 是否处于急停 | 急停帧会被过滤 |

## 神经网络还没接上时怎么写

如果目前板端还没有神经网络推理，先按下面方式记录，这也是合法训练数据：

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

换句话说，第一阶段最重要的不是“已经有残差”，而是先把基线控制器内部量打印出来。

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

下面是一段最小示例，别人照这个格式输出就行。这里前进和转向都使用离散任务编码：

```csv
session_id,timestamp_ms,dt_ms,robot_mode,control_mode,depth_valid,imu_valid,battery_v,target_depth_cm,filtered_depth_cm,depth_speed_cm_s,depth_accel_cm_s2,roll_deg,pitch_deg,gyro_x_deg_s,gyro_y_deg_s,gyro_z_deg_s,front_distance_cm,left_distance_cm,right_distance_cm,depth_err_cm,u_base,u_residual,u_total,forward_cmd_base,forward_cmd_residual,forward_cmd_total,forward_phase_interval_ms,yaw_cmd_base,yaw_cmd_residual,yaw_cmd_total,buoyancy_dir_applied,buoyancy_pwm_applied,actuator_mask,balancing,emergency_stop
session_0007,15320,40,auto,auto,1,1,11.9,50.0,46.8,-2.4,0.3,1.8,-0.6,0.7,-1.1,0.2,85.0,110.0,96.0,3.2,24.0,0.0,24.0,1,0,1,180.0,1,0,1,1,132,13,0,0
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

说明：

- 当前代码允许 `forward_cmd_base` 用 `0/1` 离散编码
- 当前代码允许 `yaw_cmd_base` 用 `-1/0/1` 离散编码
- 训练时它们会被当作普通数值特征，不要求必须是连续力度

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
