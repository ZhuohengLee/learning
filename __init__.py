"""Training helpers for residual robot-control experiments.

The `learning` package centers on one shared data layer plus one top-level deployment path:
1. `data.py` holds the telemetry schema, filtering rules, and normalization helpers.
2. `model.py` defines the shared-trunk PyTorch MLP with three output heads.
3. `train.py` trains one joint depth / forward / yaw residual model.
4. `export.py` exports one joint ONNX and optional `.espdl` artifact.

Reading route:
1. Start with `data.py` because it defines the telemetry contract shared by training and export.
2. Then read `train.py` for model training.
3. Then read `export.py` for ONNX / `.espdl` export.
"""
