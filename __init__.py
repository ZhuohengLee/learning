"""Training helpers for residual robot-control experiments.

The `learning` package covers three steps:
1. load telemetry exported by the robot,
2. train small residual neural networks in pure Python,
3. export trained weights into formats the ESP32 firmware can use.

Reading route:
1. Start with `train.py` because it is the public single-entry trainer for depth + forward + yaw.
2. Then read `export.py` because it is the public single-entry exporter for both one-bundle and three-axis deployment.
3. Then read `data.py` to understand how telemetry becomes training examples.
4. Then read `model.py` to see the exact MLP and training loop.
"""
