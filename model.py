"""A small pure-Python MLP regressor for residual control experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass
class DenseLayer:
    """One dense layer with either tanh or linear activation."""

    weights: list[list[float]]
    biases: list[float]
    activation: str


class MLPRegressor:
    """Small multilayer perceptron with batch gradient descent."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int] = (24, 12),
        seed: int = 7,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain only positive widths")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self._rng = random.Random(seed)
        layer_dims = [input_dim, *self.hidden_dims, 1]
        self.layers: list[DenseLayer] = []
        for index in range(len(layer_dims) - 1):
            in_dim = layer_dims[index]
            out_dim = layer_dims[index + 1]
            scale = math.sqrt(2.0 / (in_dim + out_dim))
            weights = [
                [self._rng.uniform(-scale, scale) for _ in range(in_dim)]
                for _ in range(out_dim)
            ]
            biases = [0.0 for _ in range(out_dim)]
            activation = "linear" if index == len(layer_dims) - 2 else "tanh"
            self.layers.append(DenseLayer(weights=weights, biases=biases, activation=activation))

    def predict_one(self, features: Sequence[float]) -> float:
        activations = [float(value) for value in features]
        if len(activations) != self.input_dim:
            raise ValueError("feature dimension does not match model input_dim")

        for layer in self.layers:
            next_activations = []
            for neuron_weights, bias in zip(layer.weights, layer.biases):
                z = sum(weight * value for weight, value in zip(neuron_weights, activations)) + bias
                next_activations.append(_activate(z, layer.activation))
            activations = next_activations

        return activations[0]

    def predict_batch(self, features: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_one(feature_vector) for feature_vector in features]

    def mean_squared_error(self, samples: Sequence[tuple[Sequence[float], float]]) -> float:
        if not samples:
            return 0.0
        error_sum = 0.0
        for features, target in samples:
            diff = self.predict_one(features) - float(target)
            error_sum += diff * diff
        return error_sum / len(samples)

    def train_epoch(
        self,
        samples: Sequence[tuple[Sequence[float], float]],
        *,
        learning_rate: float = 0.01,
        l2: float = 0.0,
        shuffle: bool = True,
    ) -> float:
        if not samples:
            raise ValueError("cannot train on an empty sample list")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0.0:
            raise ValueError("l2 must be non-negative")

        ordered_samples = list(samples)
        if shuffle:
            self._rng.shuffle(ordered_samples)

        weight_grads = [
            [[0.0 for _ in neuron_weights] for neuron_weights in layer.weights]
            for layer in self.layers
        ]
        bias_grads = [[0.0 for _ in layer.biases] for layer in self.layers]
        total_loss = 0.0

        for features, target in ordered_samples:
            layer_inputs: list[list[float]] = []
            layer_outputs: list[list[float]] = []
            activations = [float(value) for value in features]
            if len(activations) != self.input_dim:
                raise ValueError("feature dimension does not match model input_dim")

            for layer in self.layers:
                layer_inputs.append(activations)
                outputs: list[float] = []
                for neuron_weights, bias in zip(layer.weights, layer.biases):
                    z = sum(weight * value for weight, value in zip(neuron_weights, activations)) + bias
                    outputs.append(_activate(z, layer.activation))
                layer_outputs.append(outputs)
                activations = outputs

            prediction = layer_outputs[-1][0]
            diff = prediction - float(target)
            total_loss += diff * diff

            deltas: list[list[float]] = [[] for _ in self.layers]
            deltas[-1] = [2.0 * diff * _activate_derivative(prediction, self.layers[-1].activation)]

            for layer_index in range(len(self.layers) - 2, -1, -1):
                next_layer = self.layers[layer_index + 1]
                current_outputs = layer_outputs[layer_index]
                current_deltas: list[float] = []
                for neuron_index, current_output in enumerate(current_outputs):
                    propagated = 0.0
                    for next_neuron_index, next_delta in enumerate(deltas[layer_index + 1]):
                        propagated += next_layer.weights[next_neuron_index][neuron_index] * next_delta
                    current_deltas.append(
                        propagated
                        * _activate_derivative(current_output, self.layers[layer_index].activation)
                    )
                deltas[layer_index] = current_deltas

            for layer_index, layer in enumerate(self.layers):
                inputs = layer_inputs[layer_index]
                for neuron_index, delta in enumerate(deltas[layer_index]):
                    bias_grads[layer_index][neuron_index] += delta
                    for input_index, input_value in enumerate(inputs):
                        weight_grads[layer_index][neuron_index][input_index] += delta * input_value

        sample_count = len(ordered_samples)
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron_weights in enumerate(layer.weights):
                for input_index, weight in enumerate(neuron_weights):
                    grad = weight_grads[layer_index][neuron_index][input_index] / sample_count
                    if l2:
                        grad += l2 * weight
                    neuron_weights[input_index] -= learning_rate * grad
                layer.biases[neuron_index] -= (
                    learning_rate * bias_grads[layer_index][neuron_index] / sample_count
                )

        return total_loss / sample_count

    def to_dict(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "layers": [
                {
                    "activation": layer.activation,
                    "weights": layer.weights,
                    "biases": layer.biases,
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MLPRegressor":
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=tuple(int(value) for value in payload["hidden_dims"]),
            seed=0,
        )
        layers_payload = payload["layers"]
        if not isinstance(layers_payload, list) or len(layers_payload) != len(model.layers):
            raise ValueError("invalid layer payload")

        restored_layers: list[DenseLayer] = []
        for existing_layer, layer_payload in zip(model.layers, layers_payload):
            if not isinstance(layer_payload, dict):
                raise ValueError("layer payload must be a mapping")
            restored_layers.append(
                DenseLayer(
                    weights=[
                        [float(weight) for weight in row]
                        for row in layer_payload["weights"]  # type: ignore[index]
                    ],
                    biases=[float(value) for value in layer_payload["biases"]],  # type: ignore[index]
                    activation=str(layer_payload.get("activation", existing_layer.activation)),
                )
            )
        model.layers = restored_layers
        return model


def _activate(value: float, activation: str) -> float:
    if activation == "linear":
        return value
    if activation == "tanh":
        return math.tanh(value)
    raise ValueError(f"unsupported activation: {activation}")


def _activate_derivative(output_value: float, activation: str) -> float:
    if activation == "linear":
        return 1.0
    if activation == "tanh":
        return 1.0 - output_value * output_value
    raise ValueError(f"unsupported activation: {activation}")
