"""Training helpers for residual robot-control experiments.

The `learning` package covers three steps:
1. load telemetry exported by the robot,
2. train small residual neural networks in pure Python,
3. export trained weights into formats the ESP32 firmware can use.

Reading route:
1. Start with `train_residual.py` if you only care about depth residual training.
2. Start with `train_axis_models.py` if you care about depth + forward + yaw together.
3. Then read `data.py` to understand how telemetry becomes training examples.
4. Then read `model.py` to see the exact MLP and training loop.
5. Finally read `export_to_esp32.py` to see how trained weights are turned into firmware headers.
"""
