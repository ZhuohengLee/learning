"""Test package for the learning module.

The files in this package verify:
1. data loading and window construction,
2. the pure-Python MLP backend,
3. JSON-to-header export logic for the legacy firmware path,
4. multi-axis training behavior,
5. optional PyTorch backend behavior when torch is installed.

Reading route:
1. Start with `test_data.py` to understand the expected telemetry-to-dataset behavior.
2. Then read `test_model.py` to see the minimal expectations for the pure-Python MLP.
3. Then read `test_export.py` for the legacy firmware export contract.
4. Then read `test_train.py` for the three-axis training flow.
5. Finally read `test_pytorch_backend.py` if you want the optional PyTorch path.
"""
