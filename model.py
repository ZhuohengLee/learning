"""A small pure-Python MLP regressor for residual control experiments.

Reading route:
1. Start with `MLPRegressor.__init__()` to see the network shape and initialization.
2. Then read `predict_one()` to understand the forward pass.
3. Then read `train_epoch()` to understand the full batch-gradient training logic.
4. Finally read `to_dict()` / `from_dict()` to see how models are serialized.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import math  # Provide tanh and square-root utilities.
import random  # Shuffle samples and initialize weights reproducibly.
from dataclasses import dataclass  # Define lightweight layer records.
from typing import Sequence  # Accept generic ordered collections as inputs.


@dataclass  # Store one dense layer's parameters in a simple mutable record.
class DenseLayer:
    """One dense layer with either tanh or linear activation."""

    weights: list[list[float]]  # Matrix shaped as `[out_dim][in_dim]`.
    biases: list[float]  # One bias term per output neuron.
    activation: str  # Activation name, currently `"tanh"` or `"linear"`.


class MLPRegressor:
    """Small multilayer perceptron with batch gradient descent."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int] = (24, 12),
        seed: int = 7,
    ) -> None:
        if input_dim < 1:  # Refuse invalid model input dimensions.
            raise ValueError("input_dim must be positive")  # Explain the constructor constraint.
        if any(width < 1 for width in hidden_dims):  # Refuse hidden layers with zero or negative width.
            raise ValueError("hidden_dims must contain only positive widths")  # Explain the constructor constraint.

        self.input_dim = input_dim  # Remember the required input vector width.
        self.hidden_dims = tuple(hidden_dims)  # Store the hidden-layer layout as an immutable tuple.
        self._rng = random.Random(seed)  # Use a local deterministic RNG for reproducibility.
        layer_dims = [input_dim, *self.hidden_dims, 1]  # Append a single scalar output neuron.
        self.layers: list[DenseLayer] = []  # Accumulate initialized dense layers here.
        for index in range(len(layer_dims) - 1):  # Build every layer transition in order.
            in_dim = layer_dims[index]  # Width of the previous activation vector.
            out_dim = layer_dims[index + 1]  # Width of the next activation vector.
            scale = math.sqrt(2.0 / (in_dim + out_dim))  # Keep tanh activations inside a reasonable range.
            weights = [
                [self._rng.uniform(-scale, scale) for _ in range(in_dim)]  # Sample each input weight uniformly.
                for _ in range(out_dim)  # Repeat for every output neuron.
            ]  # Initialize one weight row per output neuron.
            biases = [0.0 for _ in range(out_dim)]  # Start every bias at zero.
            activation = "linear" if index == len(layer_dims) - 2 else "tanh"  # Use linear output and tanh hidden layers.
            self.layers.append(DenseLayer(weights=weights, biases=biases, activation=activation))  # Store the layer.

    def predict_one(self, features: Sequence[float]) -> float:
        """Run one forward pass and return a scalar prediction."""

        activations = [float(value) for value in features]  # Copy the input into a mutable float list.
        if len(activations) != self.input_dim:  # Verify the caller supplied the expected width.
            raise ValueError("feature dimension does not match model input_dim")  # Explain the mismatch.

        for layer in self.layers:  # Propagate through each dense layer in order.
            next_activations = []  # Collect this layer's output activations.
            for neuron_weights, bias in zip(layer.weights, layer.biases):  # Walk each neuron in the current layer.
                z = sum(weight * value for weight, value in zip(neuron_weights, activations)) + bias  # Compute the affine term.
                next_activations.append(_activate(z, layer.activation))  # Apply the configured activation function.
            activations = next_activations  # Feed this layer's outputs into the next layer.

        return activations[0]  # The network always ends with one scalar output.

    def predict_batch(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Run forward passes for a batch of feature vectors."""

        return [self.predict_one(feature_vector) for feature_vector in features]  # Reuse the single-sample path.

    def mean_squared_error(self, samples: Sequence[tuple[Sequence[float], float]]) -> float:
        """Compute the average squared error on a list of samples."""

        if not samples:  # Define the loss on an empty set as zero for convenience.
            return 0.0  # Avoid dividing by zero in that degenerate case.
        error_sum = 0.0  # Accumulate squared residuals here.
        for features, target in samples:  # Visit every `(features, target)` pair.
            diff = self.predict_one(features) - float(target)  # Compute the scalar prediction error.
            error_sum += diff * diff  # Accumulate squared error.
        return error_sum / len(samples)  # Return the mean squared error.

    def train_epoch(
        self,
        samples: Sequence[tuple[Sequence[float], float]],
        *,
        learning_rate: float = 0.01,
        l2: float = 0.0,
        shuffle: bool = True,
    ) -> float:
        """Train once over the full sample list using batch gradient descent."""

        if not samples:  # Refuse to train on an empty dataset.
            raise ValueError("cannot train on an empty sample list")  # Explain why training failed.
        if learning_rate <= 0.0:  # Enforce a positive optimizer step size.
            raise ValueError("learning_rate must be positive")  # Explain the constraint.
        if l2 < 0.0:  # Enforce a non-negative weight-decay coefficient.
            raise ValueError("l2 must be non-negative")  # Explain the constraint.

        ordered_samples = list(samples)  # Copy the sample list so shuffling does not mutate the caller input.
        if shuffle:  # Shuffle only when requested by the caller.
            self._rng.shuffle(ordered_samples)  # Use the model-local RNG for reproducible order.

        weight_grads = [
            [[0.0 for _ in neuron_weights] for neuron_weights in layer.weights]  # Match the layer weight shape exactly.
            for layer in self.layers  # Repeat for every layer.
        ]  # Allocate one accumulated weight-gradient matrix per layer.
        bias_grads = [[0.0 for _ in layer.biases] for layer in self.layers]  # Allocate one accumulated bias vector per layer.
        total_loss = 0.0  # Accumulate training loss over the epoch.

        for features, target in ordered_samples:  # Visit every training sample once.
            layer_inputs: list[list[float]] = []  # Cache each layer's input activations for backprop.
            layer_outputs: list[list[float]] = []  # Cache each layer's output activations for backprop.
            activations = [float(value) for value in features]  # Copy the input into a mutable float list.
            if len(activations) != self.input_dim:  # Recheck width even during training.
                raise ValueError("feature dimension does not match model input_dim")  # Explain the mismatch.

            for layer in self.layers:  # Run the forward pass layer by layer.
                layer_inputs.append(activations)  # Save the current input activations for later gradients.
                outputs: list[float] = []  # Collect neuron outputs for this layer.
                for neuron_weights, bias in zip(layer.weights, layer.biases):  # Walk each neuron in the current layer.
                    z = sum(weight * value for weight, value in zip(neuron_weights, activations)) + bias  # Compute affine term.
                    outputs.append(_activate(z, layer.activation))  # Apply the activation function.
                layer_outputs.append(outputs)  # Save the full output vector for backprop.
                activations = outputs  # Feed the output vector into the next layer.

            prediction = layer_outputs[-1][0]  # Read the scalar network output.
            diff = prediction - float(target)  # Compute prediction minus target.
            total_loss += diff * diff  # Accumulate squared loss for reporting.

            deltas: list[list[float]] = [[] for _ in self.layers]  # Allocate one delta vector per layer.
            deltas[-1] = [2.0 * diff * _activate_derivative(prediction, self.layers[-1].activation)]  # Seed output-layer delta.

            for layer_index in range(len(self.layers) - 2, -1, -1):  # Walk hidden layers backward.
                next_layer = self.layers[layer_index + 1]  # Inspect the layer ahead for delta propagation.
                current_outputs = layer_outputs[layer_index]  # Read this layer's forward outputs.
                current_deltas: list[float] = []  # Collect deltas for this layer.
                for neuron_index, current_output in enumerate(current_outputs):  # Compute one hidden-neuron delta at a time.
                    propagated = 0.0  # Accumulate the gradient signal from the next layer.
                    for next_neuron_index, next_delta in enumerate(deltas[layer_index + 1]):  # Walk every neuron in the next layer.
                        propagated += next_layer.weights[next_neuron_index][neuron_index] * next_delta  # Project delta backward.
                    current_deltas.append(
                        propagated
                        * _activate_derivative(current_output, self.layers[layer_index].activation)
                    )  # Multiply by the local derivative and store the hidden delta.
                deltas[layer_index] = current_deltas  # Save the full delta vector for this layer.

            for layer_index, layer in enumerate(self.layers):  # Accumulate parameter gradients layer by layer.
                inputs = layer_inputs[layer_index]  # Read the forward inputs that fed this layer.
                for neuron_index, delta in enumerate(deltas[layer_index]):  # Walk every neuron delta in the current layer.
                    bias_grads[layer_index][neuron_index] += delta  # Bias gradient is the delta itself.
                    for input_index, input_value in enumerate(inputs):  # Walk every input connection to this neuron.
                        weight_grads[layer_index][neuron_index][input_index] += delta * input_value  # Accumulate weight gradient.

        sample_count = len(ordered_samples)  # Use the epoch sample count to average gradients.
        for layer_index, layer in enumerate(self.layers):  # Apply one update to every layer.
            for neuron_index, neuron_weights in enumerate(layer.weights):  # Walk every neuron weight row.
                for input_index, weight in enumerate(neuron_weights):  # Walk every individual weight.
                    grad = weight_grads[layer_index][neuron_index][input_index] / sample_count  # Convert sum to mean gradient.
                    if l2:  # Add weight decay only when requested.
                        grad += l2 * weight  # Penalize large weights with L2 regularization.
                    neuron_weights[input_index] -= learning_rate * grad  # Apply the gradient-descent step.
                layer.biases[neuron_index] -= (
                    learning_rate * bias_grads[layer_index][neuron_index] / sample_count
                )  # Apply the averaged bias update.

        return total_loss / sample_count  # Report mean training loss for the epoch.

    def to_dict(self) -> dict[str, object]:
        """Export the model architecture and parameters into plain Python data."""

        return {
            "input_dim": self.input_dim,  # Store the required input width.
            "hidden_dims": list(self.hidden_dims),  # Store hidden-layer widths as plain integers.
            "layers": [
                {
                    "activation": layer.activation,  # Keep the activation type.
                    "weights": layer.weights,  # Keep the full weight matrix.
                    "biases": layer.biases,  # Keep the full bias vector.
                }
                for layer in self.layers  # Repeat for every dense layer.
            ],  # Store each layer's parameters in order.
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MLPRegressor":
        """Restore a model instance from exported parameters."""

        model = cls(
            input_dim=int(payload["input_dim"]),  # Restore the input width.
            hidden_dims=tuple(int(value) for value in payload["hidden_dims"]),  # Restore hidden-layer widths.
            seed=0,  # Seed is irrelevant because weights will be overwritten.
        )  # Construct a shape-compatible empty model first.
        layers_payload = payload["layers"]  # Read the serialized layer list.
        if not isinstance(layers_payload, list) or len(layers_payload) != len(model.layers):  # Validate layer count.
            raise ValueError("invalid layer payload")  # Explain malformed serialized models.

        restored_layers: list[DenseLayer] = []  # Rebuild the layer list here.
        for existing_layer, layer_payload in zip(model.layers, layers_payload):  # Walk expected and serialized layers together.
            if not isinstance(layer_payload, dict):  # Ensure each serialized layer is a mapping.
                raise ValueError("layer payload must be a mapping")  # Explain malformed serialized models.
            restored_layers.append(
                DenseLayer(
                    weights=[
                        [float(weight) for weight in row]  # Convert every serialized weight to float.
                        for row in layer_payload["weights"]  # type: ignore[index]
                    ],  # Restore the full weight matrix.
                    biases=[float(value) for value in layer_payload["biases"]],  # type: ignore[index]
                    activation=str(layer_payload.get("activation", existing_layer.activation)),  # Preserve activation name.
                )
            )  # Append a fully restored DenseLayer.
        model.layers = restored_layers  # Replace the placeholder layers with restored ones.
        return model  # Return the reconstructed model instance.


def _activate(value: float, activation: str) -> float:
    """Apply the configured activation function."""

    if activation == "linear":  # Output layer uses identity activation.
        return value  # Return the affine term unchanged.
    if activation == "tanh":  # Hidden layers use hyperbolic tangent.
        return math.tanh(value)  # Apply the nonlinearity.
    raise ValueError(f"unsupported activation: {activation}")  # Explain unknown activation names.


def _activate_derivative(output_value: float, activation: str) -> float:
    """Return the derivative using the already-computed neuron output."""

    if activation == "linear":  # Derivative of identity is constant.
        return 1.0  # Return the linear derivative.
    if activation == "tanh":  # Derivative can be expressed from tanh output directly.
        return 1.0 - output_value * output_value  # Use `1 - tanh(x)^2`.
    raise ValueError(f"unsupported activation: {activation}")  # Explain unknown activation names.
