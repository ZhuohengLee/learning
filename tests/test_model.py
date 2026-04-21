import unittest

from learning.model import MLPRegressor


class ModelTests(unittest.TestCase):
    def test_training_reduces_loss_on_simple_mapping(self) -> None:
        model = MLPRegressor(input_dim=2, hidden_dims=(6,), seed=7)
        samples = [
            ([0.0, 0.0], 0.0),
            ([1.0, 0.0], 1.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 1.0], 2.0),
        ]

        baseline_loss = model.mean_squared_error(samples)
        for _ in range(200):
            model.train_epoch(samples, learning_rate=0.03)
        trained_loss = model.mean_squared_error(samples)

        self.assertLess(trained_loss, baseline_loss)
        self.assertLess(trained_loss, 0.02)

    def test_export_contains_architecture_and_parameters(self) -> None:
        model = MLPRegressor(input_dim=3, hidden_dims=(4, 2), seed=5)
        exported = model.to_dict()

        self.assertEqual(exported["input_dim"], 3)
        self.assertEqual(exported["hidden_dims"], [4, 2])
        self.assertEqual(len(exported["layers"]), 3)
        self.assertIn("weights", exported["layers"][0])
        self.assertIn("biases", exported["layers"][0])


if __name__ == "__main__":
    unittest.main()
