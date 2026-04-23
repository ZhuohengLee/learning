"""Training helpers for residual robot-control experiments.

The `learning` package has one shared data layer and two backends:
1. `data.py` holds the telemetry schema, filtering rules, and normalization helpers.
2. `python_mlp/` keeps the original hand-written Python MLP -> JSON -> C++ header flow.
3. `pytorch_mlp/` is the new PyTorch backend meant for ONNX / ESP-DL style deployment.

Reading route:
1. Start with `data.py` because both backends share the same telemetry contract.
2. Then read `python_mlp/train.py` if you want the current lightweight firmware path.
3. Then read `pytorch_mlp/train.py` if you want the new PyTorch training path.
"""
