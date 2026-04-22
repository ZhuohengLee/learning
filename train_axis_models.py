"""Train separate residual models for depth, forward, and yaw control.

Reading route:
1. Start with `main()` to see the command-line workflow.
2. Then read `train_axis_models()` because it coordinates all axis-specific training.
3. Then read `train_single_axis()` to understand one model's train/eval/export path.
4. Finally read `_augment_missing_axis_targets()` to see how older logs are upgraded.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the training CLI.
import copy  # Snapshot the best model parameters during training.
import json  # Export model bundles and the manifest in JSON format.
import math  # Compute RMSE for validation metrics.
from pathlib import Path  # Build output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.
from typing import Sequence  # Accept generic ordered collections as inputs.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add the repo root to `sys.path`.

from learning.data import (  # Reuse telemetry loading and dataset-prep helpers.
    DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,  # Default shared feature set for multi-axis training.
    DEFAULT_WINDOW_SIZE,  # Default number of stacked frames per sample.
    Example,  # Typed training example record.
    build_examples,  # Convert raw rows into windowed examples.
    fit_standardizer,  # Fit z-score normalization statistics.
    load_control_rows,  # Load raw telemetry rows from CSV.
    split_examples_by_session,  # Split examples without session leakage.
)
from learning.model import MLPRegressor  # Tiny pure-Python regressor used for training.


DEFAULT_AXIS_TARGETS = {
    "depth": "u_residual",  # Residual depth-control output.
    "forward": "forward_cmd_residual",  # Residual forward-control output.
    "yaw": "yaw_cmd_residual",  # Residual yaw-control output.
}


def main() -> None:
    """CLI entry point for multi-axis residual training."""

    args = parse_args()  # Parse all user-supplied training options.
    rows = load_control_rows(args.csv)  # Load raw telemetry rows from the requested CSV.
    output_dir = Path(args.output_dir)  # Normalize the requested output directory.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory before training starts.

    manifest = train_axis_models(  # Train one model per requested control axis.
        rows=rows,  # Pass raw telemetry rows.
        output_dir=output_dir,  # Save bundles into this directory.
        feature_columns=args.feature_columns,  # Use the configured shared feature set.
        window_size=args.window_size,  # Stack this many recent frames per example.
        hidden_dims=tuple(args.hidden_dims),  # Use the configured hidden-layer widths.
        epochs=args.epochs,  # Train for this many epochs per axis.
        learning_rate=args.learning_rate,  # Use this optimizer step size.
        l2=args.l2,  # Use this L2 regularization coefficient.
        val_fraction=args.val_fraction,  # Reserve this fraction for validation.
        max_dt_ms=args.max_dt_ms,  # Break sequences when frame gaps become too large.
        seed=args.seed,  # Make training reproducible.
        print_every=args.print_every,  # Print progress at this interval.
        axis_targets={  # Select target columns for each logical control axis.
            "depth": args.depth_target_column,  # Depth residual target column.
            "forward": args.forward_target_column,  # Forward residual target column.
            "yaw": args.yaw_target_column,  # Yaw residual target column.
        },
    )
    manifest_path = output_dir / "axis_manifest.json"  # Name the summary manifest file.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Save the manifest as pretty JSON.
    print(f"saved axis manifest to {manifest_path}")  # Tell the user where the manifest was written.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for axis-wise training."""

    parser = argparse.ArgumentParser(  # Build the CLI parser.
        description="Train separate residual models for depth, forward, and yaw control.",  # Describe the tool.
    )
    parser.add_argument("--csv", required=True, help="Path to control telemetry CSV")  # Input telemetry file.
    parser.add_argument("--output-dir", required=True, help="Directory for exported model bundles")  # Output directory.
    parser.add_argument(
        "--feature-columns",  # Flag name on the CLI.
        nargs="+",  # Accept one or more feature-column names.
        default=list(DEFAULT_MULTI_AXIS_FEATURE_COLUMNS),  # Start from the default shared feature list.
        help="Shared feature columns used for every axis model",  # Explain the option.
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)  # Stacked history length.
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


def train_axis_models(
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
    """Train one residual model per control axis and return a manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists.
    prepared_rows = _augment_missing_axis_targets(rows, axis_targets)  # Backfill residual targets when older logs lack them.
    manifest: dict[str, object] = {
        "feature_columns": list(feature_columns),  # Record the shared base feature set.
        "window_size": window_size,  # Record the shared stacked history length.
        "axes": {},  # Fill this mapping with one entry per trained axis.
    }

    for axis_name, target_column in axis_targets.items():  # Visit each requested control axis.
        if not target_column:  # Skip axes whose target column was intentionally disabled.
            continue  # Do not train or export anything for this axis.

        dataset = build_examples(
            prepared_rows,  # Use rows after residual-target backfilling.
            feature_columns,  # Reuse the shared feature set.
            window_size=window_size,  # Stack this many frames per example.
            target_column=target_column,  # Read the axis-specific residual target.
            max_dt_ms=max_dt_ms,  # Break sequences when frame gaps become too large.
        )  # Build a supervised dataset for this specific target column.
        if not dataset.examples:  # Refuse to proceed when this axis has no usable examples.
            raise ValueError(f"no trainable examples were built for axis {axis_name}")  # Explain the failing axis.

        train_set, val_set = split_examples_by_session(
            dataset,  # Use the axis-specific dataset.
            val_fraction=val_fraction,  # Reserve this fraction for validation.
            seed=seed,  # Make the split reproducible.
        )  # Split examples without session leakage.
        bundle, metrics = train_single_axis(
            train_examples=train_set.examples,  # Use the training split.
            val_examples=val_set.examples,  # Use the validation split.
            feature_names=dataset.feature_names,  # Preserve the flattened feature naming.
            feature_columns=dataset.feature_columns,  # Preserve the base feature naming.
            window_size=dataset.window_size,  # Preserve the stacked history length.
            hidden_dims=hidden_dims,  # Use the configured hidden-layer widths.
            epochs=epochs,  # Train for this many epochs.
            learning_rate=learning_rate,  # Use this optimizer step size.
            l2=l2,  # Use this regularization coefficient.
            seed=seed,  # Make initialization reproducible.
            print_every=print_every,  # Print progress at this interval.
            source_axis=axis_name,  # Label logs and metadata with the current axis.
            target_column=target_column,  # Remember which column produced the target.
        )  # Train, evaluate, and export one axis model.

        output_path = output_dir / f"{axis_name}_model.json"  # Name the bundle file for this axis.
        output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")  # Save the bundle as pretty JSON.
        print(f"[{axis_name}] saved model bundle to {output_path}")  # Tell the user where the axis bundle was written.
        manifest["axes"][axis_name] = {  # type: ignore[index]
            "target_column": target_column,  # Record the axis target column.
            "output_path": str(output_path),  # Record the axis bundle path.
            **metrics,  # Include dataset sizes and best validation loss.
        }

    return manifest  # Return the summary manifest for all trained axes.


def _augment_missing_axis_targets(
    rows: Sequence[dict[str, str]],
    axis_targets: dict[str, str],
) -> list[dict[str, str]]:
    """Backfill residual columns from total-base pairs when needed."""

    derived_pairs = {
        axis_targets.get("depth", ""): ("u_total", "u_base"),  # Recover depth residual from total minus base.
        axis_targets.get("forward", ""): ("forward_cmd_total", "forward_cmd_base"),  # Recover forward residual.
        axis_targets.get("yaw", ""): ("yaw_cmd_total", "yaw_cmd_base"),  # Recover yaw residual.
    }  # Map residual target columns to `(total, base)` fallback pairs.

    prepared_rows: list[dict[str, str]] = []  # Accumulate copied and backfilled rows here.
    for row in rows:  # Visit every raw telemetry row once.
        copied = dict(row)  # Copy the row so the caller's data is not mutated.
        for target_column, pair in derived_pairs.items():  # Visit every possible residual fallback mapping.
            if not target_column:  # Ignore disabled axes with empty target names.
                continue  # Nothing to backfill for this entry.
            raw_target = copied.get(target_column, "").strip()  # Read the explicit residual field when present.
            if raw_target:  # Keep rows that already carry the explicit residual value.
                continue  # No fallback computation is required.

            total_key, base_key = pair  # Unpack the fallback total/base column names.
            raw_total = copied.get(total_key, "").strip()  # Read the total command text.
            raw_base = copied.get(base_key, "").strip()  # Read the base command text.
            if raw_total and raw_base:  # Only derive a residual when both ingredients exist.
                copied[target_column] = str(float(raw_total) - float(raw_base))  # Recover residual as total minus base.
        prepared_rows.append(copied)  # Store the copied row after all backfills.

    return prepared_rows  # Return the copied and backfilled row list.


def train_single_axis(
    *,
    train_examples: Sequence[Example],
    val_examples: Sequence[Example],
    feature_names: Sequence[str],
    feature_columns: Sequence[str],
    window_size: int,
    hidden_dims: Sequence[int],
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    print_every: int,
    source_axis: str,
    target_column: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train, evaluate, and export one axis-specific residual model."""

    train_features = [example.features for example in train_examples]  # Extract training feature vectors.
    train_targets = [[example.target] for example in train_examples]  # Wrap targets as 1D vectors for fitting.
    feature_standardizer = fit_standardizer(train_features)  # Fit input normalization on training features only.
    target_standardizer = fit_standardizer(train_targets)  # Fit target normalization on training targets only.

    normalized_train = _normalize_samples(
        train_examples,  # Use the training split.
        feature_standardizer,  # Apply feature normalization.
        target_standardizer,  # Apply target normalization.
    )  # Convert train examples into normalized tuples.
    normalized_val = _normalize_samples(
        val_examples,  # Use the validation split.
        feature_standardizer,  # Reuse training feature normalization.
        target_standardizer,  # Reuse training target normalization.
    )  # Convert validation examples into normalized tuples.

    model = MLPRegressor(
        input_dim=len(feature_names),  # Match the flattened feature-vector width.
        hidden_dims=tuple(hidden_dims),  # Use the configured hidden-layer widths.
        seed=seed,  # Make initialization reproducible.
    )  # Build the regression model with the requested architecture.
    best_snapshot = copy.deepcopy(model.to_dict())  # Snapshot the initial weights as a fallback best model.
    best_val_loss = math.inf  # Track the best validation loss seen so far.

    for epoch in range(1, epochs + 1):  # Run the requested number of epochs.
        train_loss = model.train_epoch(
            normalized_train,  # Use normalized train samples.
            learning_rate=learning_rate,  # Apply the configured learning rate.
            l2=l2,  # Apply the configured L2 penalty.
            shuffle=True,  # Shuffle samples each epoch for a healthier optimization path.
        )  # Train once over the normalized training set.
        val_loss = model.mean_squared_error(normalized_val)  # Evaluate normalized validation loss after the update.
        if val_loss < best_val_loss:  # Keep the best-performing validation checkpoint.
            best_val_loss = val_loss  # Record the new best validation loss.
            best_snapshot = copy.deepcopy(model.to_dict())  # Deep-copy the current parameters.

        if epoch == 1 or epoch == epochs or epoch % print_every == 0:  # Print first, periodic, and final stats.
            metrics = _denormalized_metrics(
                model,  # Evaluate the current model.
                normalized_val,  # Use normalized validation samples.
                target_standardizer,  # Undo target normalization for reporting.
            )  # Convert metrics back to the original command scale for readability.
            print(
                f"[{source_axis}] epoch={epoch:4d} "  # Show axis name and epoch number.
                f"train_loss={train_loss:.6f} "  # Show normalized train loss.
                f"val_loss={val_loss:.6f} "  # Show normalized validation loss.
                f"val_mae={metrics['mae']:.3f} "  # Show validation MAE in original command units.
                f"val_rmse={metrics['rmse']:.3f}"  # Show validation RMSE in original command units.
            )  # Emit one compact training-progress line.

    best_model = MLPRegressor.from_dict(best_snapshot)  # Restore the best validation checkpoint for export.
    bundle = {
        "metadata": {
            "axis": source_axis,  # Record the logical control axis.
            "window_size": window_size,  # Record the stacked-frame count.
            "feature_columns": list(feature_columns),  # Record the base input-feature order.
            "feature_names": list(feature_names),  # Record the flattened feature order.
            "target_column": target_column,  # Record the exact target column used.
            "train_examples": len(train_examples),  # Record the training-set size.
            "val_examples": len(val_examples),  # Record the validation-set size.
            "epochs": epochs,  # Record the training epoch count.
            "learning_rate": learning_rate,  # Record the optimizer step size.
            "l2": l2,  # Record the regularization coefficient.
            "best_val_loss": best_val_loss,  # Record the best normalized validation loss.
        },  # Store human-readable training metadata.
        "input_standardizer": feature_standardizer.to_dict(),  # Export feature normalization statistics.
        "target_standardizer": target_standardizer.to_dict(),  # Export target normalization statistics.
        "model": best_model.to_dict(),  # Export the best model weights and architecture.
    }  # Assemble the axis-specific export bundle.
    metrics = {
        "train_examples": len(train_examples),  # Record the training-set size.
        "val_examples": len(val_examples),  # Record the validation-set size.
        "best_val_loss": best_val_loss,  # Record the best normalized validation loss.
    }  # Build the small summary inserted into the manifest.
    return bundle, metrics  # Return both the full bundle and the compact manifest entry.


def _normalize_samples(
    examples: Sequence[Example],
    feature_standardizer,
    target_standardizer,
) -> list[tuple[list[float], float]]:
    """Convert raw examples into normalized `(features, target)` tuples."""

    samples: list[tuple[list[float], float]] = []  # Accumulate normalized samples here.
    for example in examples:  # Visit every raw example once.
        normalized_features = feature_standardizer.normalize(example.features)  # Normalize the feature vector.
        normalized_target = target_standardizer.normalize([example.target])[0]  # Normalize the scalar target.
        samples.append((normalized_features, normalized_target))  # Store the normalized pair.
    return samples  # Return the full normalized sample list.


def _denormalized_metrics(model, samples, target_standardizer) -> dict[str, float]:
    """Report validation metrics back in the original command scale."""

    if not samples:  # Handle empty validation sets defensively.
        return {"mae": 0.0, "rmse": 0.0}  # Report zeros instead of dividing by zero.

    absolute_error_sum = 0.0  # Accumulate absolute error in original units.
    squared_error_sum = 0.0  # Accumulate squared error in original units.
    for features, normalized_target in samples:  # Visit every normalized validation sample.
        normalized_prediction = model.predict_one(features)  # Predict in normalized target space.
        prediction = target_standardizer.denormalize([normalized_prediction])[0]  # Convert prediction back to original units.
        target = target_standardizer.denormalize([normalized_target])[0]  # Convert target back to original units.
        diff = prediction - target  # Measure the prediction error in original units.
        absolute_error_sum += abs(diff)  # Accumulate absolute error.
        squared_error_sum += diff * diff  # Accumulate squared error.

    count = len(samples)  # Use the sample count to compute mean metrics.
    return {
        "mae": absolute_error_sum / count,  # Mean absolute error.
        "rmse": math.sqrt(squared_error_sum / count),  # Root mean squared error.
    }  # Return both MAE and RMSE in original units.


if __name__ == "__main__":  # Run the CLI when the file is executed directly.
    main()  # Start axis-wise training.
