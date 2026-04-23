"""PyTorch MLP definitions for residual-control experiments."""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

from typing import Sequence  # Accept generic ordered collections as inputs.

_TORCH_IMPORT_ERROR: Exception | None = None  # Preserve the original import failure for debugging.

try:  # Import PyTorch lazily so the rest of the package can work without it.
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
except ImportError as error:  # Keep the backend importable only when used explicitly.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = error  # Remember the underlying failure.


def require_torch() -> None:
    """Raise a clear error when the PyTorch backend is used without torch installed."""

    if torch is None or nn is None:  # Reject use of the backend when PyTorch is unavailable.
        raise ImportError(
            "PyTorch backend requires `torch`. Install PyTorch before using learning.pytorch_mlp."
        ) from _TORCH_IMPORT_ERROR


if nn is not None:  # Define the real model only when PyTorch is installed.

    class TorchResidualMLP(nn.Module):
        """Small two-hidden-layer MLP that mirrors the custom Python backend."""

        def __init__(self, *, input_dim: int, hidden_dims: Sequence[int] = (24, 12)) -> None:
            super().__init__()  # Initialize the parent `nn.Module`.
            if input_dim < 1:  # Refuse invalid model input dimensions.
                raise ValueError("input_dim must be positive")  # Explain the constructor constraint.
            if len(hidden_dims) != 2:  # Keep the architecture aligned with the existing firmware path.
                raise ValueError("TorchResidualMLP currently expects exactly two hidden layers")  # Explain the limit.
            if any(width < 1 for width in hidden_dims):  # Refuse hidden layers with zero or negative width.
                raise ValueError("hidden_dims must contain only positive widths")  # Explain the constructor constraint.

            hidden1, hidden2 = (int(width) for width in hidden_dims)  # Unpack the two hidden widths explicitly.
            self.input_dim = int(input_dim)  # Remember the expected flat input width.
            self.hidden_dims = (hidden1, hidden2)  # Preserve the two hidden-layer widths.
            self.network = nn.Sequential(  # Build the exact forward graph in one readable block.
                nn.Linear(self.input_dim, hidden1),  # First dense layer.
                nn.Tanh(),  # Match the custom backend's hidden activation.
                nn.Linear(hidden1, hidden2),  # Second dense layer.
                nn.Tanh(),  # Match the custom backend's hidden activation.
                nn.Linear(hidden2, 1),  # Scalar residual output.
            )

        def forward(self, inputs):  # type: ignore[override]
            """Run one forward pass on a `[batch, input_dim]` float tensor."""

            return self.network(inputs)  # Delegate to the sequential graph.

else:

    class TorchResidualMLP:  # type: ignore[no-redef]
        """Fallback placeholder that raises a clear error when instantiated without torch."""

        def __init__(self, *args, **kwargs) -> None:
            require_torch()  # Always raise the dependency error.

