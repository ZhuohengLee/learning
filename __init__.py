"""Training helpers for residual robot-control experiments.

The `learning` package centers on one shared data layer plus one top-level deployment path:
1. `data.py` holds the telemetry schema, filtering rules, and normalization helpers.
2. `model.py` defines the PyTorch MLP.
3. `train.py` trains depth / forward / yaw residual models.
4. `export.py` exports ONNX and optional `.espdl` artifacts.

Reading route:
1. Start with `data.py` because it defines the telemetry contract shared by training and export.
2. Then read `train.py` for model training.
3. Then read `export.py` for ONNX / `.espdl` export.
"""
