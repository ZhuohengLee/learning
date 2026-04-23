"""Custom pure-Python MLP backend for residual-control experiments.

Reading route:
1. Start with `train.py` because it is the implementation behind the legacy public trainer.
2. Then read `export.py` because it turns JSON bundles into firmware headers.
3. Then read `model.py` for the hand-written MLP and optimizer logic.
"""

