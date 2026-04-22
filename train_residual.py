"""Train a residual buoyancy controller from CSV telemetry logs.

Reading route:
1. Start with `main()` to see the full training pipeline end to end.
2. Then read `parse_args()` to understand the configurable inputs.
3. Then read `_normalize_samples()` to see how examples are prepared for the MLP.
4. Finally read `_denormalized_metrics()` to understand the printed MAE/RMSE values.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the training CLI.
import copy  # Snapshot the best model parameters during training.
import json  # Export the trained bundle in JSON format.
import math  # Compute RMSE for validation metrics.
from pathlib import Path  # Build output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add the repo root to `sys.path`.

from learning.data import (  # Reuse telemetry loading and dataset-prep helpers.
    DEFAULT_FEATURE_COLUMNS,  # Default depth-model feature set.
    DEFAULT_WINDOW_SIZE,  # Default number of stacked frames per sample.
    Example,  # Typed training example record.
    fit_standardizer,  # Fit z-score normalization statistics.
    load_control_rows,  # Load raw telemetry rows from CSV.
    build_examples,  # Convert raw rows into windowed examples.
    split_examples_by_session,  # Split examples without session leakage.
)
from learning.model import MLPRegressor  # Tiny pure-Python regressor used for training.


def main() -> None:
    """CLI entry point for depth residual training."""

    args = parse_args()  # Parse all user-supplied training options.
    rows = load_control_rows(args.csv)  # Load raw telemetry rows from the requested CSV.
    dataset = build_examples(  # Convert raw rows into fixed-width supervised examples.
        rows,  # Pass the raw telemetry rows.
        args.feature_columns,  # Use the configured input feature columns.
        window_size=args.window_size,  # Stack this many recent frames per example.
        target_column=args.target_column,  # Read this explicit residual target when provided.
        max_dt_ms=args.max_dt_ms,  # Break sequences when frame gaps become too large.
    )
    if not dataset.examples:  # Refuse to train when no usable data was produced.
        raise SystemExit("no trainable examples were built from the supplied CSV")  # Exit with a clear message.

    train_set, val_set = split_examples_by_session(  # Split examples without leaking sessions across sets.
        dataset,  # Use the prepared example set.
        val_fraction=args.val_fraction,  # Reserve this fraction for validation.
        seed=args.seed,  # Make the split reproducible.
    )

    train_features = [example.features for example in train_set.examples]  # Extract training feature vectors.
    train_targets = [[example.target] for example in train_set.examples]  # Wrap targets as 1D vectors for fitting.
    feature_standardizer = fit_standardizer(train_features)  # Fit input normalization on training features only.
    target_standardizer = fit_standardizer(train_targets)  # Fit target normalization on training targets only.

    normalized_train = _normalize_samples(  # Convert train examples into normalized tuples.
        train_set.examples,  # Use the training split.
        feature_standardizer,  # Apply feature normalization.
        target_standardizer,  # Apply target normalization.
    )
    normalized_val = _normalize_samples(  # Convert validation examples into normalized tuples.
        val_set.examples,  # Use the validation split.
        feature_standardizer,  # Reuse training feature normalization.
        target_standardizer,  # Reuse training target normalization.
    )

    model = MLPRegressor(  # Build the regression model with the requested architecture.
        input_dim=len(dataset.feature_names),  # Match the flattened feature-vector width.
        hidden_dims=tuple(args.hidden_dims),  # Use the configured hidden-layer widths.
        seed=args.seed,  # Make initialization reproducible.
    )
    best_snapshot = copy.deepcopy(model.to_dict())  # Snapshot the initial weights as a fallback best model.
    best_val_loss = math.inf  # Track the best validation loss seen so far.

    for epoch in range(1, args.epochs + 1):  # Run the requested number of epochs.
        train_loss = model.train_epoch(  # Train once over the normalized training set.
            normalized_train,  # Use normalized train samples.
            learning_rate=args.learning_rate,  # Apply the configured learning rate.
            l2=args.l2,  # Apply the configured L2 penalty.
            shuffle=True,  # Shuffle samples each epoch for a healthier optimization path.
        )
        val_loss = model.mean_squared_error(normalized_val)  # Evaluate normalized validation loss after the update.

        if val_loss < best_val_loss:  # Keep the best-performing validation checkpoint.
            best_val_loss = val_loss  # Record the new best validation loss.
            best_snapshot = copy.deepcopy(model.to_dict())  # Deep-copy the current parameters.

        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:  # Print first, periodic, and final stats.
            metrics = _denormalized_metrics(  # Convert metrics back to the original PWM scale for readability.
                model,  # Evaluate the current model.
                normalized_val,  # Use normalized validation samples.
                target_standardizer,  # Undo target normalization for reporting.
            )
            print(  # Emit one compact training-progress line.
                f"epoch={epoch:4d} "  # Show the current epoch number.
                f"train_loss={train_loss:.6f} "  # Show normalized train loss.
                f"val_loss={val_loss:.6f} "  # Show normalized validation loss.
                f"val_mae_pwm={metrics['mae']:.3f} "  # Show validation MAE in original PWM units.
                f"val_rmse_pwm={metrics['rmse']:.3f}"  # Show validation RMSE in original PWM units.
            )

    best_model = MLPRegressor.from_dict(best_snapshot)  # Restore the best validation checkpoint for export.
    output_path = Path(args.output)  # Normalize the destination path.
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories when needed.

    export_payload = {  # Assemble the full training bundle expected by exporters and firmware tools.
        "metadata": {  # Store human-readable training metadata.
            "source_csv": str(Path(args.csv).resolve()),  # Remember the exact CSV used for training.
            "window_size": args.window_size,  # Record the stacked-frame count.
            "feature_columns": list(args.feature_columns),  # Record the base input-feature order.
            "feature_names": list(dataset.feature_names),  # Record the flattened feature order.
            "target_column": args.target_column or "auto",  # Record target resolution mode.
            "train_examples": len(train_set.examples),  # Record the training-set size.
            "val_examples": len(val_set.examples),  # Record the validation-set size.
            "epochs": args.epochs,  # Record the training epoch count.
            "learning_rate": args.learning_rate,  # Record the optimizer step size.
            "l2": args.l2,  # Record the regularization coefficient.
            "best_val_loss": best_val_loss,  # Record the best normalized validation loss.
        },
        "input_standardizer": feature_standardizer.to_dict(),  # Export feature normalization statistics.
        "target_standardizer": target_standardizer.to_dict(),  # Export target normalization statistics.
        "model": best_model.to_dict(),  # Export the best model weights and architecture.
    }
    output_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")  # Save the bundle as pretty JSON.
    print(f"saved model bundle to {output_path}")  # Tell the user where the bundle was written.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for training."""

    parser = argparse.ArgumentParser(  # Build the CLI parser.
        description="Train a residual buoyancy MLP from control telemetry CSV logs.",  # Describe the tool.
    )
    parser.add_argument("--csv", required=True, help="Path to control_telemetry.csv")  # Input telemetry file.
    parser.add_argument(
        "--output",  # Flag name on the CLI.
        required=True,  # Force the user to choose an output path.
        help="Where to write the exported model JSON bundle",  # Explain what gets written there.
    )
    parser.add_argument(
        "--window-size",  # Flag name on the CLI.
        type=int,  # Parse the value as an integer.
        default=DEFAULT_WINDOW_SIZE,  # Use the package default unless overridden.
        help="How many historical frames to stack into one training example",  # Explain the option.
    )
    parser.add_argument(
        "--feature-columns",  # Flag name on the CLI.
        nargs="+",  # Accept one or more feature-column names.
        default=list(DEFAULT_FEATURE_COLUMNS),  # Start from the default depth feature list.
        help="Base feature columns expected in the telemetry CSV",  # Explain the option.
    )
    parser.add_argument(
        "--target-column",  # Flag name on the CLI.
        default="",  # Default to automatic target resolution.
        help="Optional explicit residual target column. If omitted, the loader tries residual_target_pwm, then u_residual, then u_total-u_base.",  # Explain fallback behavior.
    )
    parser.add_argument(
        "--hidden-dims",  # Flag name on the CLI.
        nargs="+",  # Accept one or more integer widths.
        type=int,  # Parse widths as integers.
        default=[24, 12],  # Use the default two-hidden-layer layout.
        help="Hidden layer widths for the MLP",  # Explain the option.
    )
    parser.add_argument("--epochs", type=int, default=300)  # Number of optimization epochs.
    parser.add_argument("--learning-rate", type=float, default=0.01)  # Gradient-descent step size.
    parser.add_argument("--l2", type=float, default=1e-4)  # L2 regularization coefficient.
    parser.add_argument("--val-fraction", type=float, default=0.25)  # Fraction of data reserved for validation.
    parser.add_argument("--max-dt-ms", type=float, default=80.0)  # Maximum allowed time gap inside one sequence.
    parser.add_argument("--seed", type=int, default=7)  # Random seed for reproducibility.
    parser.add_argument("--print-every", type=int, default=25)  # How often to print progress during training.
    return parser.parse_args()  # Return the parsed argument namespace.


def _normalize_samples(
    examples: list[Example],
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
    """Report MAE/RMSE back in the original PWM scale."""

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
    main()  # Start training.
