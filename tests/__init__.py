"""Test package for the learning module.

The files in this package verify:
1. data loading and window construction,
2. shared telemetry filtering and target fallback behavior,
3. top-level PyTorch training and export behavior when torch is installed.

Reading route:
1. Start with `test_data.py` to understand the telemetry-to-dataset contract.
2. Then read `test_pytorch_backend.py` for the top-level training / ONNX export path.
"""
