"""Train depth, forward, and yaw residual models from one telemetry CSV.

Reading route:
1. Start with `main()` to see the single-entry training workflow.
2. Then read `parse_args()` to understand the unified CLI surface.
3. Then read `train_residual_models()` because it wires the public entry into the shared axis trainer.
4. Finally read `learning.train_axis_models.train_single_axis()` for one model's train/eval/export path.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the training CLI.
import json  # Export the manifest in JSON format.
from pathlib import Path  # Build output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.
from typing import Sequence  # Accept generic ordered collections as inputs.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add the repo root to `sys.path`.

from learning.data import (  # Reuse telemetry loading and dataset-prep helpers.
    DEFAULT_UNIFIED_FEATURE_COLUMNS,  # Shared default feature set used for all three public-axis models.
    DEFAULT_WINDOW_SIZE,  # Default number of stacked frames per sample.
    load_control_rows,  # Load raw telemetry rows from CSV.
)
from learning.train_axis_models import (  # Reuse the already-tested axis training helpers.
    DEFAULT_AXIS_TARGETS,  # Default residual target column per logical control axis.
    train_axis_models,  # Train and export one residual bundle per control axis.
)


def main() -> None:
    """CLI entry point for unified three-axis residual training."""

    args = parse_args()  # Parse all user-supplied training options.
    rows = load_control_rows(args.csv)  # Load raw telemetry rows from the requested CSV.
    output_dir = Path(args.output_dir)  # Normalize the requested output directory.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory before training starts.

    manifest = train_residual_models(  # Train depth, forward, and yaw from one shared command.
        rows=rows,  # Pass raw telemetry rows.
        output_dir=output_dir,  # Save exported model bundles into this directory.
        feature_columns=tuple(args.feature_columns),  # Use the configured shared feature set for all three axes.
        window_size=args.window_size,  # Stack this many recent frames per example.
        hidden_dims=tuple(args.hidden_dims),  # Use the configured hidden-layer widths.
        epochs=args.epochs,  # Train for this many epochs per axis.
        learning_rate=args.learning_rate,  # Use this optimizer step size.
        l2=args.l2,  # Use this L2 regularization coefficient.
        val_fraction=args.val_fraction,  # Reserve this fraction for validation.
        max_dt_ms=args.max_dt_ms,  # Break sequences when frame gaps become too large.
        seed=args.seed,  # Make training reproducible.
        print_every=args.print_every,  # Print progress at this interval.
        axis_targets={
            "depth": args.depth_target_column,  # Depth residual target column.
            "forward": args.forward_target_column,  # Forward residual target column.
            "yaw": args.yaw_target_column,  # Yaw residual target column.
        },
    )
    manifest_path = output_dir / "axis_manifest.json"  # Name the summary manifest file.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Save the manifest as pretty JSON.
    print(f"saved axis manifest to {manifest_path}")  # Tell the user where the manifest was written.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for unified three-axis training."""

    parser = argparse.ArgumentParser(  # Build the CLI parser.
        description="Train depth, forward, and yaw residual MLPs from one control telemetry CSV.",  # Describe the tool.
    )
    parser.add_argument("--csv", required=True, help="Path to control telemetry CSV")  # Input telemetry file.
    parser.add_argument("--output-dir", required=True, help="Directory for exported model bundles")  # Output directory.
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)  # Stacked history length.
    parser.add_argument(
        "--feature-columns",  # Flag name on the CLI.
        nargs="+",  # Accept one or more feature-column names.
        default=list(DEFAULT_UNIFIED_FEATURE_COLUMNS),  # Start from the shared public three-axis feature list.
        help="Shared base feature columns used by the depth, forward, and yaw models",  # Explain the option.
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[24, 12])  # Hidden-layer widths.
    parser.add_argument("--epochs", type=int, default=300)  # Number of optimization epochs.
    parser.add_argument("--learning-rate", type=float, default=0.01)  # Gradient-descent step size.
    parser.add_argument("--l2", type=float, default=1e-4)  # L2 regularization coefficient.
    parser.add_argument("--val-fraction", type=float, default=0.25)  # Fraction of data reserved for validation.
    parser.add_argument("--max-dt-ms", type=float, default=80.0)  # Maximum allowed time gap inside one sequence.
    parser.add_argument("--seed", type=int, default=7)  # Random seed for reproducibility.
    parser.add_argument("--print-every", type=int, default=25)  # How often to print progress during training.
    parser.add_argument("--depth-target-column", default=DEFAULT_AXIS_TARGETS["depth"])  # Explicit depth target name.
    parser.add_argument("--forward-target-column", default=DEFAULT_AXIS_TARGETS["forward"])  # Explicit forward target name.
    parser.add_argument("--yaw-target-column", default=DEFAULT_AXIS_TARGETS["yaw"])  # Explicit yaw target name.
    return parser.parse_args()  # Return the parsed argument namespace.


def train_residual_models(
    *,
    rows: Sequence[dict[str, str]],
    output_dir: Path,
    feature_columns: Sequence[str],
    window_size: int,
    hidden_dims: Sequence[int],
    epochs: int,
    learning_rate: float,
    l2: float,
    val_fraction: float,
    max_dt_ms: float,
    seed: int,
    print_every: int,
    axis_targets: dict[str, str],
) -> dict[str, object]:
    """Train depth, forward, and yaw residual models and return a manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists.
    return train_axis_models(
        rows=rows,  # Pass raw telemetry rows straight into the shared axis trainer.
        output_dir=output_dir,  # Save bundles into this directory.
        feature_columns=feature_columns,  # Reuse one shared feature set for all three axes.
        window_size=window_size,  # Stack this many recent frames per example.
        hidden_dims=hidden_dims,  # Use the configured hidden-layer widths.
        epochs=epochs,  # Train for this many epochs per axis.
        learning_rate=learning_rate,  # Use this optimizer step size.
        l2=l2,  # Use this regularization coefficient.
        val_fraction=val_fraction,  # Reserve this fraction for validation.
        max_dt_ms=max_dt_ms,  # Break sequences when frame gaps become too large.
        seed=seed,  # Make training reproducible.
        print_every=print_every,  # Print progress at this interval.
        axis_targets=axis_targets,  # Train the requested logical control axes.
    )  # Delegate the actual per-axis training/export workflow.


if __name__ == "__main__":  # Run the CLI when the file is executed directly.
    main()  # Start unified three-axis training.
