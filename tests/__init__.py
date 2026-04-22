"""Test package for the learning module.

The files in this package verify:
1. data loading and window construction,
2. the pure-Python MLP implementation,
3. JSON-to-header export logic,
4. multi-axis training and export behavior.

Reading route:
1. Start with `test_data.py` to understand the expected telemetry-to-dataset behavior.
2. Then read `test_model.py` to see the minimal expectations for the MLP itself.
3. Then read `test_export_to_esp32.py` for the firmware export contract.
4. Finally read `test_train_axis_models.py` for the end-to-end multi-axis flow.
"""
