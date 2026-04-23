"""PyTorch backend for residual-control experiments.

Reading route:
1. Start with `train.py` because it is the public training entry for this backend.
2. Then read `export.py` because it exports `.pt` bundles into ONNX and optional `.espdl` artifacts.
3. Finally read `model.py` for the exact network definition.
"""
