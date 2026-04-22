"""Tests for the small pure-Python MLP regressor.

Reading route:
1. Start with `test_training_reduces_loss_on_simple_mapping()` to see the core learning expectation.
2. Then read `test_export_contains_architecture_and_parameters()` to see serialization expectations.
"""

import sys  # Adjust the import path when the standalone repo is executed directly.
import unittest  # Use the standard-library test framework.
from pathlib import Path  # Resolve the standalone repo path safely across platforms.

if __package__ in {None, ""}:  # Detect direct test-module execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Add the parent of the repo root so `learning` can be imported.

from learning.model import MLPRegressor  # Import the model under test.


class ModelTests(unittest.TestCase):  # Group model-behavior tests together.
    """Check that the toy MLP both trains and exports correctly."""

    def test_training_reduces_loss_on_simple_mapping(self) -> None:
        """Verify that gradient descent can fit an easy additive mapping."""

        model = MLPRegressor(input_dim=2, hidden_dims=(6,), seed=7)  # Build a tiny network with one hidden layer.
        samples = [  # Define a simple supervised mapping the network should learn quickly.
            ([0.0, 0.0], 0.0),  # Zero plus zero should map to zero.
            ([1.0, 0.0], 1.0),  # One plus zero should map to one.
            ([0.0, 1.0], 1.0),  # Zero plus one should map to one.
            ([1.0, 1.0], 2.0),  # One plus one should map to two.
        ]

        baseline_loss = model.mean_squared_error(samples)  # Measure loss before any training.
        for _ in range(200):  # Run enough epochs for the tiny mapping to converge.
            model.train_epoch(samples, learning_rate=0.03)  # Update model weights with gradient descent.
        trained_loss = model.mean_squared_error(samples)  # Measure loss after training.

        self.assertLess(trained_loss, baseline_loss)  # Training should reduce the loss.
        self.assertLess(trained_loss, 0.02)  # The final loss should be very small on this easy task.

    def test_export_contains_architecture_and_parameters(self) -> None:
        """Verify that `to_dict()` preserves structure and parameter tensors."""

        model = MLPRegressor(input_dim=3, hidden_dims=(4, 2), seed=5)  # Build a network with two hidden layers.
        exported = model.to_dict()  # Serialize the model into plain Python data.

        self.assertEqual(exported["input_dim"], 3)  # Export should preserve input width.
        self.assertEqual(exported["hidden_dims"], [4, 2])  # Export should preserve hidden-layer widths.
        self.assertEqual(len(exported["layers"]), 3)  # Export should contain two hidden layers plus one output layer.
        self.assertIn("weights", exported["layers"][0])  # Each layer export should include weights.
        self.assertIn("biases", exported["layers"][0])  # Each layer export should include biases.


if __name__ == "__main__":  # Allow running this file directly for local debugging.
    unittest.main()  # Execute the tests.
