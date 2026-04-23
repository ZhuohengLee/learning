"""PyTorch MLP definitions for joint residual-control experiments."""

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


def _axis_head_key(axis_name: str) -> str:
    """Convert a public axis name into a safe internal submodule key."""

    return f"axis_{axis_name}"  # Prefix head names so axes like `forward` do not collide with `nn.Module` attributes.


def require_torch() -> None:
    """Raise a clear error when the PyTorch backend is used without torch installed."""

    if torch is None or nn is None:  # Reject use of the backend when PyTorch is unavailable.
        raise ImportError(
            "PyTorch backend requires `torch`. Install PyTorch before using learning."
        ) from _TORCH_IMPORT_ERROR


if nn is not None:  # Define the real model only when PyTorch is installed.

    class TorchJointResidualMLP(nn.Module):
        """Small joint MLP with one shared trunk and one scalar head per control axis."""

        def __init__(
            self,
            *,
            input_dim: int,
            axis_names: Sequence[str] = ("depth", "forward", "yaw"),
            hidden_dims: Sequence[int] = (24, 12),
        ) -> None:
            super().__init__()  # Initialize the parent `nn.Module`.
            if input_dim < 1:  # Refuse invalid model input dimensions.
                raise ValueError("input_dim must be positive")  # Explain the constructor constraint.
            if len(hidden_dims) != 2:  # Keep the architecture aligned with the current firmware plan.
                raise ValueError("TorchJointResidualMLP currently expects exactly two hidden layers")  # Explain the limit.
            if any(width < 1 for width in hidden_dims):  # Refuse hidden layers with zero or negative width.
                raise ValueError("hidden_dims must contain only positive widths")  # Explain the constructor constraint.
            normalized_axis_names = [str(name).strip() for name in axis_names]  # Normalize axis names once for all metadata and head keys.
            if not normalized_axis_names or any(not name for name in normalized_axis_names):  # Refuse empty head-name lists.
                raise ValueError("axis_names must contain at least one non-empty axis name")  # Explain the constructor constraint.
            if len(set(normalized_axis_names)) != len(normalized_axis_names):  # Require unique head names for a stable state_dict layout.
                raise ValueError("axis_names must be unique")  # Explain the constructor constraint.

            hidden1, hidden2 = (int(width) for width in hidden_dims)  # Unpack the two hidden widths explicitly.
            self.input_dim = int(input_dim)  # Remember the expected flat input width.
            self.axis_names = tuple(normalized_axis_names)  # Preserve the public joint output order.
            self.head_keys = tuple(_axis_head_key(axis_name) for axis_name in self.axis_names)  # Preserve safe internal head keys in the same output order.
            self.hidden_dims = (hidden1, hidden2)  # Preserve the two hidden-layer widths.
            self.trunk = nn.Sequential(  # Build the shared nonlinear encoder first.
                nn.Linear(self.input_dim, hidden1),  # First dense layer.
                nn.Tanh(),  # Hidden activation.
                nn.Linear(hidden1, hidden2),  # Second dense layer.
                nn.Tanh(),  # Hidden activation.
            )
            self.heads = nn.ModuleDict(  # Attach one scalar head per logical control axis.
                {
                    head_key: nn.Linear(hidden2, 1)  # Emit one residual scalar for this control axis.
                    for head_key in self.head_keys
                }
            )

        def forward(self, inputs):  # type: ignore[override]
            """Run one forward pass on a `[batch, input_dim]` float tensor."""

            shared_features = self.trunk(inputs)  # Encode the shared state once for all axes.
            return torch.cat(  # type: ignore[union-attr]
                [self.heads[head_key](shared_features) for head_key in self.head_keys],  # Concatenate scalar head outputs into `[batch, axis_count]`.
                dim=1,
            )

else:

    class TorchJointResidualMLP:  # type: ignore[no-redef]
        """Fallback placeholder that raises a clear error when instantiated without torch."""

        def __init__(self, *args, **kwargs) -> None:
            require_torch()  # Always raise the dependency error.
