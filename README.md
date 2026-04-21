# Residual Control Training

This directory contains a small, dependency-free training scaffold for the
robot's residual controllers.

It is intentionally narrow:

- Input: control telemetry exported from the robot as CSV
- Model: small MLP regressors
- Output: bounded residual corrections on top of the baseline controllers
- Goal: keep the baseline controllers in charge, and let the networks learn only
  bounded corrections

This code uses only the Python standard library so you can inspect and run it
without installing PyTorch first. It is suitable for validating the data format,
feature pipeline, and export path. Once the data schema is stable, you can port
the same feature/target contract to PyTorch if you want faster training.

## Control architecture

The current direction is:

- `depth` stays a predictive + residual buoyancy controller
- `forward` is now a continuous forward-effort controller on the ESP32
- `yaw` is now a continuous bidirectional turn controller on the ESP32

That means training should follow three small models:

- `depth_model.json`
- `forward_model.json`
- `yaw_model.json`

The depth exporter is already wired end-to-end today.
The forward and yaw training bundles are now produced separately so their
onboard inference wrappers can be added without changing the data contract.

## Expected CSV columns

The original depth-only feature set expects these columns:

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

The shared multi-axis training flow expects these columns:

- `depth_err_cm`
- `depth_speed_cm_s`
- `depth_accel_cm_s2`
- `roll_deg`
- `pitch_deg`
- `gyro_x_deg_s`
- `gyro_y_deg_s`
- `gyro_z_deg_s`
- `front_distance_cm`
- `left_distance_cm`
- `right_distance_cm`
- `battery_v`
- `u_base`
- `forward_cmd_base`
- `yaw_cmd_base`

The loader also uses these gating columns when present:

- `session_id`
- `timestamp_ms`
- `control_mode`
- `depth_valid`
- `imu_valid`
- `balancing`
- `emergency_stop`

Target resolution order:

1. explicit `--target-column`
2. `residual_target_pwm`
3. `u_residual`
4. `u_total - u_base`

## Train a model

```powershell
python learning\train_residual.py `
  --csv D:\path\to\control_telemetry.csv `
  --output D:\working\squid robot\code\learning\artifacts\residual_model.json
```

Useful overrides:

```powershell
python learning\train_residual.py `
  --csv D:\path\to\control_telemetry.csv `
  --output learning\artifacts\residual_model.json `
  --window-size 5 `
  --hidden-dims 24 12 `
  --epochs 400 `
  --learning-rate 0.008 `
  --target-column residual_target_pwm
```

## Output bundle

The training script writes a JSON file containing:

- model architecture and weights
- input normalization statistics
- target normalization statistics
- feature order
- training metadata

This makes the feature order explicit so you can later reproduce the same
inference logic on the ESP32 side.

## Train depth only

```powershell
python learning\train_residual.py `
  --csv D:\path\to\control_telemetry.csv `
  --output D:\working\squid robot\code\learning\artifacts\residual_model.json
```

## Train all three axes

```powershell
python learning\train_axis_models.py `
  --csv D:\path\to\control_telemetry.csv `
  --output-dir D:\working\squid robot\code\learning\artifacts\axis_models
```

Default targets used by the multi-axis trainer:

- `depth` -> `u_residual`
- `forward` -> `forward_cmd_residual`
- `yaw` -> `yaw_cmd_residual`

You can override each target column from the CLI.
If a `*_residual` column is missing but matching `base` and `total` columns are
present, the trainer derives the residual target as `total - base`.

## Run inference on ESP32

The intended deployment split is:

- training on your computer
- inference only on the ESP32

After you train a real model bundle, export it into the firmware header:

```powershell
python learning\export_to_esp32.py `
  --model learning\artifacts\residual_model.json `
  --output ESP32\ResidualModelData.h
```

Then rebuild and flash the `ESP32` firmware. At runtime:

- `rn stat` shows whether the residual model is present and warmed up
- `rn on` enables onboard residual inference
- `rn off` disables it and falls back to the baseline controller only

Safety notes:

- the residual output is still bounded on the ESP32 side
- the baseline controller remains the primary controller
- the exporter refuses to generate a header if the feature order does not match
  the current ESP32 inference contract

Current integration status:

- battery voltage is already wired into the ESP32 inference path
- IMU fields are already reserved in the model contract
- forward and yaw are now continuous command controllers on the ESP32 side
- SD logging now records `forward_cmd_*` and `yaw_cmd_*` fields for training
- you still need to connect your actual IMU driver to
  `DepthController::setImuState(...)` in `ESP32/ESP32.ino` once that sensor
  code exists in this repo

## Tests

Run the local tests with:

```powershell
python -m unittest discover learning\tests
```

The tests cover:

- CSV loading and filtering
- sliding-window example construction
- normalization round trips
- basic MLP training behavior
- export structure
- multi-axis model export flow
