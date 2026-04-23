"""Train one joint residual model with a shared trunk and three output heads.

Reading route:
1. Start with `main()` for CLI wiring.
2. Then read `train_model()` because it is the public joint-training entry.
3. Then read `_train_joint_bundle()` for the actual optimization loop.
4. Finally read the normalization and device helpers near the bottom.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the training CLI.
import json  # Export the manifest in JSON format.
import math  # Compute RMSE for validation metrics.
from pathlib import Path  # Build output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.
from typing import Sequence  # Accept generic ordered collections as inputs.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Add the repo root so `learning` can be imported.

from learning.data import (  # Reuse telemetry loading and dataset-prep helpers.
    DEFAULT_JOINT_TARGET_COLUMNS,  # Default joint output targets for the public trainer.
    DEFAULT_UNIFIED_FEATURE_COLUMNS,  # Default shared feature set for the joint model.
    DEFAULT_WINDOW_SIZE,  # Default number of stacked frames per sample.
    MultiAxisExample,  # Typed joint training example record.
    MultiAxisExampleSet,  # Typed joint dataset wrapper.
    build_multi_axis_examples,  # Convert raw rows into windowed joint examples.
    fit_standardizer,  # Fit z-score normalization statistics.
    load_control_rows,  # Load raw telemetry rows from CSV.
    split_examples_by_session,  # Split examples without session leakage.
)
from learning.model import TorchJointResidualMLP, require_torch  # Import the PyTorch backend and dependency guard.


JOINT_MODEL_FILENAME = "joint_model.pt"  # Keep the exported joint bundle name explicit.


def main() -> None:
    """CLI entry point for joint three-axis PyTorch training."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    args = parse_args()  # Parse all user-supplied training options.
    rows = load_control_rows(args.csv)  # Load raw telemetry rows from the requested CSV.
    output_dir = Path(args.output_dir)  # Normalize the requested output directory.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory before training starts.

    manifest = train_model(
        rows=rows,  # Pass raw telemetry rows.
        output_dir=output_dir,  # Save the exported joint model bundle into this directory.
        feature_columns=tuple(args.feature_columns),  # Use the configured shared feature set.
        window_size=args.window_size,  # Stack this many recent frames per example.
        hidden_dims=tuple(args.hidden_dims),  # Use the configured hidden-layer widths.
        epochs=args.epochs,  # Train for this many epochs.
        learning_rate=args.learning_rate,  # Use this optimizer step size.
        l2=args.l2,  # Use this weight-decay coefficient.
        val_fraction=args.val_fraction,  # Reserve this fraction for validation.
        max_dt_ms=args.max_dt_ms,  # Break sequences when frame gaps become too large.
        seed=args.seed,  # Make training reproducible.
        print_every=args.print_every,  # Print progress at this interval.
        batch_size=args.batch_size,  # Use the configured minibatch size.
        device_name=args.device,  # Choose the requested torch device.
        axis_targets={
            "depth": args.depth_target_column,  # Depth residual target column.
            "forward": args.forward_target_column,  # Forward residual target column.
            "yaw": args.yaw_target_column,  # Yaw residual target column.
        },
    )  # Train one joint residual bundle for all requested axes.
    manifest_path = output_dir / "joint_manifest.json"  # Name the summary manifest file.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Save the manifest as pretty JSON.
    print(f"saved joint manifest to {manifest_path}")  # Tell the user where the manifest was written.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for joint three-axis PyTorch training."""

    parser = argparse.ArgumentParser(
        description="Train one shared-trunk PyTorch MLP that predicts depth, forward, and yaw residuals from one telemetry CSV.",
    )
    parser.add_argument("--csv", required=True, help="Path to control telemetry CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for the exported joint `.pt` bundle")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(DEFAULT_UNIFIED_FEATURE_COLUMNS),
        help="Shared base feature columns used by the joint depth/forward/yaw PyTorch model.",
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[24, 12])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-dt-ms", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="Torch device name, for example `cpu` or `cuda`")
    parser.add_argument("--depth-target-column", default=DEFAULT_JOINT_TARGET_COLUMNS["depth"])
    parser.add_argument("--forward-target-column", default=DEFAULT_JOINT_TARGET_COLUMNS["forward"])
    parser.add_argument("--yaw-target-column", default=DEFAULT_JOINT_TARGET_COLUMNS["yaw"])
    return parser.parse_args()


def train_model(
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
    batch_size: int,
    device_name: str,
    axis_targets: dict[str, str],
) -> dict[str, object]:
    """Train one joint PyTorch residual model and return a manifest."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    import torch  # type: ignore[import-not-found]

    if batch_size < 1:  # Refuse invalid minibatch sizes.
        raise ValueError("batch_size must be positive")  # Explain the contract clearly.

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists before training starts.
    prepared_rows = _augment_missing_axis_targets(rows, axis_targets)  # Backfill missing residual columns from total-base pairs.
    dataset = build_multi_axis_examples(
        prepared_rows,  # Use the prepared telemetry rows with residual fallbacks applied.
        feature_columns,  # Use the requested shared feature set.
        axis_targets=axis_targets,  # Learn all requested axes together.
        window_size=window_size,  # Stack the requested number of frames.
        max_dt_ms=max_dt_ms,  # Respect the requested sequence-gap threshold.
    )
    if not dataset.examples:  # Reject empty joint datasets immediately.
        raise ValueError("no trainable examples were built for the joint model")  # Explain why training cannot continue.

    split_train, split_val = split_examples_by_session(dataset, val_fraction=val_fraction, seed=seed)  # Keep session leakage out of validation.
    if not isinstance(split_train, MultiAxisExampleSet) or not isinstance(split_val, MultiAxisExampleSet):  # Narrow the union returned by the shared splitter.
        raise TypeError("joint training expected MultiAxisExampleSet splits")  # Fail loudly if the splitter contract changes unexpectedly.

    device = _resolve_device(torch=torch, device_name=device_name)  # Resolve the requested torch device once.
    bundle, metrics = _train_joint_bundle(
        train_examples=split_train.examples,  # Feed the training windows into the optimizer loop.
        val_examples=split_val.examples,  # Hold out validation windows for model selection.
        feature_names=dataset.feature_names,  # Preserve the expanded feature-name contract in the exported bundle.
        feature_columns=dataset.feature_columns,  # Preserve the base feature-column contract in the exported bundle.
        window_size=dataset.window_size,  # Preserve the stacked-frame width in the exported bundle.
        axis_names=dataset.axis_names,  # Preserve the joint output order in the exported bundle.
        target_columns=dataset.target_columns,  # Preserve the logical-axis to CSV-target mapping in the exported bundle.
        hidden_dims=hidden_dims,  # Use the requested hidden-layer widths.
        epochs=epochs,  # Train for the requested number of epochs.
        learning_rate=learning_rate,  # Use the requested optimizer step size.
        l2=l2,  # Use the requested weight decay.
        seed=seed,  # Make weight initialization reproducible.
        print_every=print_every,  # Emit progress at the requested interval.
        batch_size=batch_size,  # Use the requested minibatch size.
        device=device,  # Train on the resolved device.
    )

    output_path = output_dir / JOINT_MODEL_FILENAME  # Name the exported joint model bundle predictably.
    torch.save(bundle, output_path)  # Persist the trained joint bundle to disk.
    print(f"saved joint PyTorch model bundle to {output_path}")  # Tell the user where the bundle was written.

    manifest_axes = {  # Flatten the exported per-axis metadata for the summary manifest.
        axis_name: {
            "target_column": dataset.target_columns[axis_name],  # Record which CSV column trained this logical axis.
            **metrics["axes"][axis_name],  # Record validation metrics for this axis.
        }
        for axis_name in dataset.axis_names
    }
    return {
        "backend": "pytorch",  # Record the backend used to create this bundle.
        "model_type": "shared_trunk_multi_head_mlp",  # Record the network topology in one short label.
        "model_path": str(output_path),  # Record the exported bundle path.
        "feature_columns": list(dataset.feature_columns),  # Record the shared base feature contract.
        "window_size": dataset.window_size,  # Record the stacked-frame width.
        "device": str(device),  # Record the device used during training.
        "axis_names": list(dataset.axis_names),  # Record the joint output order.
        "target_columns": dict(dataset.target_columns),  # Record the logical-axis to CSV-target mapping.
        "train_examples": metrics["train_examples"],  # Record how many training windows were used.
        "val_examples": metrics["val_examples"],  # Record how many validation windows were used.
        "best_val_loss": metrics["best_val_loss"],  # Record the normalized model-selection loss.
        "axes": manifest_axes,  # Record per-axis validation metrics in original command units.
    }


def _augment_missing_axis_targets(
    rows: Sequence[dict[str, str]],
    axis_targets: dict[str, str],
) -> list[dict[str, str]]:
    """Backfill residual columns from total-base pairs when needed."""

    derived_pairs = {
        axis_targets.get("depth", ""): ("u_total", "u_base"),
        axis_targets.get("forward", ""): ("forward_cmd_total", "forward_cmd_base"),
        axis_targets.get("yaw", ""): ("yaw_cmd_total", "yaw_cmd_base"),
    }

    prepared_rows: list[dict[str, str]] = []  # Accumulate copied rows with derived residuals filled in.
    for row in rows:  # Walk every raw telemetry row once.
        copied = dict(row)  # Copy the CSV row so the caller's data is never mutated.
        for target_column, pair in derived_pairs.items():  # Check every residual target that may need a fallback.
            if not target_column:  # Ignore axes whose target columns were intentionally disabled.
                continue  # Move to the next target column.
            raw_target = copied.get(target_column, "").strip()  # Read the explicit residual value when present.
            if raw_target:  # Prefer an explicit residual value over derived math.
                continue  # Leave the explicit value unchanged.
            total_key, base_key = pair  # Unpack the total/base pair for this axis.
            raw_total = copied.get(total_key, "").strip()  # Read the raw total command.
            raw_base = copied.get(base_key, "").strip()  # Read the raw base command.
            if raw_total and raw_base:  # Only derive a residual when both ingredients exist.
                copied[target_column] = str(float(raw_total) - float(raw_base))  # Recover residual as total minus base.
        prepared_rows.append(copied)  # Keep the copied row even when no fallback was needed.
    return prepared_rows  # Return the prepared row list without mutating the caller's input.


def _train_joint_bundle(
    *,
    train_examples: Sequence[MultiAxisExample],
    val_examples: Sequence[MultiAxisExample],
    feature_names: Sequence[str],
    feature_columns: Sequence[str],
    window_size: int,
    axis_names: Sequence[str],
    target_columns: dict[str, str],
    hidden_dims: Sequence[int],
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    print_every: int,
    batch_size: int,
    device,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train, evaluate, and export one joint PyTorch model bundle."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found]

    torch.manual_seed(seed)  # Make weight initialization reproducible.

    train_features = [example.features for example in train_examples]  # Collect training features into one matrix-like list.
    train_targets = [example.targets for example in train_examples]  # Collect joint targets into one matrix-like list.
    feature_standardizer = fit_standardizer(train_features)  # Fit input normalization from training features only.
    target_standardizer = fit_standardizer(train_targets)  # Fit joint-output normalization from training targets only.

    train_feature_tensor, train_target_tensor = _build_normalized_tensors(
        torch=torch,  # Build tensors with the torch module selected above.
        examples=train_examples,  # Normalize the training windows.
        feature_standardizer=feature_standardizer,  # Normalize features with train-only statistics.
        target_standardizer=target_standardizer,  # Normalize targets with train-only statistics.
    )
    val_feature_tensor, val_target_tensor = _build_normalized_tensors(
        torch=torch,  # Build tensors with the torch module selected above.
        examples=val_examples,  # Normalize the validation windows.
        feature_standardizer=feature_standardizer,  # Reuse the training feature standardizer.
        target_standardizer=target_standardizer,  # Reuse the training target standardizer.
    )

    train_loader = DataLoader(
        TensorDataset(train_feature_tensor, train_target_tensor),  # Pair normalized features with normalized joint targets.
        batch_size=min(batch_size, len(train_examples)),  # Avoid empty trailing batches on tiny datasets.
        shuffle=True,  # Shuffle training windows each epoch for better SGD mixing.
    )

    model = TorchJointResidualMLP(
        input_dim=len(feature_names),  # Match the flattened shared feature width.
        axis_names=tuple(axis_names),  # Preserve the joint output order inside the model heads.
        hidden_dims=tuple(hidden_dims),  # Use the requested hidden-layer widths.
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)  # Use Adam for stable small-MLP training.
    loss_fn = nn.MSELoss()  # Optimize average normalized MSE across all three heads.
    best_snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}  # Preserve the best model parameters seen so far.
    best_val_loss = math.inf  # Track the best normalized validation loss seen so far.

    for epoch in range(1, epochs + 1):  # Run the requested number of SGD epochs.
        model.train()  # Enable gradient updates for this epoch.
        train_loss_sum = 0.0  # Accumulate loss weighted by batch size.
        train_sample_count = 0  # Count how many training windows contributed to the epoch loss.
        for batch_features, batch_targets in train_loader:  # Walk every minibatch once.
            batch_features = batch_features.to(device)  # Move input features onto the requested device.
            batch_targets = batch_targets.to(device)  # Move target vectors onto the requested device.
            optimizer.zero_grad()  # Clear stale gradients before the new forward pass.
            predictions = model(batch_features)  # Predict normalized residual vectors for this minibatch.
            loss = loss_fn(predictions, batch_targets)  # Measure average normalized MSE across all heads.
            loss.backward()  # Backpropagate gradients through the joint network.
            optimizer.step()  # Apply one optimizer update step.
            batch_size_now = int(batch_features.shape[0])  # Measure how many windows this minibatch contained.
            train_loss_sum += float(loss.detach().cpu()) * batch_size_now  # Accumulate weighted batch loss.
            train_sample_count += batch_size_now  # Accumulate batch sample count.

        train_loss = train_loss_sum / max(1, train_sample_count)  # Convert the weighted sum back into average epoch loss.
        val_loss = _evaluate_normalized_mse(
            model=model,  # Evaluate the current model snapshot.
            features=val_feature_tensor,  # Use held-out validation features.
            targets=val_target_tensor,  # Use held-out validation targets.
            device=device,  # Run evaluation on the same device.
            torch=torch,  # Reuse the current torch module.
        )
        if val_loss < best_val_loss:  # Track the best model snapshot by normalized validation loss.
            best_val_loss = val_loss  # Update the best observed validation loss.
            best_snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}  # Freeze the best parameters on CPU.

        if epoch == 1 or epoch == epochs or epoch % print_every == 0:  # Emit progress on the first epoch, periodic checkpoints, and the last epoch.
            axis_metrics = _denormalized_axis_metrics(
                model=model,  # Evaluate the current model snapshot.
                features=val_feature_tensor,  # Use held-out validation features.
                targets=val_target_tensor,  # Use held-out validation targets.
                device=device,  # Run evaluation on the same device.
                torch=torch,  # Reuse the current torch module.
                axis_names=axis_names,  # Format metrics in the stable joint output order.
                target_standardizer=target_standardizer,  # Map normalized outputs back to original command units.
            )
            axis_rmse_summary = " ".join(  # Build one compact per-axis RMSE summary that follows the saved axis order.
                f"{axis_name}_rmse={axis_metrics[axis_name]['rmse']:.3f}"  # Format one denormalized RMSE token for this axis.
                for axis_name in axis_names
            )
            print(
                f"[joint] epoch={epoch:4d} "  # Prefix the progress line with the public model name.
                f"train_loss={train_loss:.6f} "  # Show average normalized training loss.
                f"val_loss={val_loss:.6f} "  # Show average normalized validation loss.
                f"{axis_rmse_summary}"  # Show denormalized RMSE per axis.
            )

    model.load_state_dict(best_snapshot)  # Restore the best validation snapshot before exporting.
    best_axis_metrics = _denormalized_axis_metrics(
        model=model,  # Evaluate the selected best model snapshot.
        features=val_feature_tensor,  # Use held-out validation features.
        targets=val_target_tensor,  # Use held-out validation targets.
        device=device,  # Run evaluation on the same device.
        torch=torch,  # Reuse the current torch module.
        axis_names=axis_names,  # Format metrics in the stable joint output order.
        target_standardizer=target_standardizer,  # Map normalized outputs back to original command units.
    )

    bundle = {
        "metadata": {
            "backend": "pytorch",  # Record which backend produced this bundle.
            "model_type": "shared_trunk_multi_head_mlp",  # Record the network topology in one short label.
            "axis_names": list(axis_names),  # Preserve the joint output order for export and firmware integration.
            "window_size": window_size,  # Preserve the stacked-frame width.
            "feature_columns": list(feature_columns),  # Preserve the base feature-column contract.
            "feature_names": list(feature_names),  # Preserve the expanded flattened feature names.
            "target_columns": dict(target_columns),  # Preserve the logical-axis to CSV-target mapping.
            "train_examples": len(train_examples),  # Record how many training windows were used.
            "val_examples": len(val_examples),  # Record how many validation windows were used.
            "epochs": epochs,  # Record the requested training epoch count.
            "learning_rate": learning_rate,  # Record the optimizer step size.
            "l2": l2,  # Record the weight-decay coefficient.
            "best_val_loss": best_val_loss,  # Record the selected normalized validation loss.
            "best_axis_metrics": best_axis_metrics,  # Record per-axis validation metrics in original command units.
        },
        "input_standardizer": feature_standardizer.to_dict(),  # Export input normalization for deployment parity.
        "target_standardizer": target_standardizer.to_dict(),  # Export joint-output normalization for deployment parity.
        "model_spec": {
            "input_dim": len(feature_names),  # Record the flattened shared feature width.
            "hidden_dims": [int(value) for value in hidden_dims],  # Record the two hidden-layer widths.
            "output_dim": len(axis_names),  # Record the number of joint output heads.
            "axis_names": list(axis_names),  # Record the output ordering expected by the heads.
            "hidden_activation": "tanh",  # Record the shared hidden activation.
            "output_activation": "linear",  # Record the output activation.
        },
        "state_dict": best_snapshot,  # Export the selected best model parameters.
    }
    metrics = {
        "train_examples": len(train_examples),  # Mirror the training-window count into the summary manifest.
        "val_examples": len(val_examples),  # Mirror the validation-window count into the summary manifest.
        "best_val_loss": best_val_loss,  # Mirror the selected normalized validation loss into the summary manifest.
        "axes": best_axis_metrics,  # Mirror the denormalized per-axis metrics into the summary manifest.
    }
    return bundle, metrics  # Return both the serialized bundle and the summary metrics.


def _build_normalized_tensors(
    *,
    torch,
    examples: Sequence[MultiAxisExample],
    feature_standardizer,
    target_standardizer,
):
    """Convert joint examples into normalized float32 tensors."""

    normalized_features = [
        feature_standardizer.normalize(example.features)  # Normalize each flattened input window.
        for example in examples
    ]
    normalized_targets = [
        target_standardizer.normalize(example.targets)  # Normalize each joint target vector in axis order.
        for example in examples
    ]
    return (
        torch.tensor(normalized_features, dtype=torch.float32),  # Convert normalized inputs into one dense tensor.
        torch.tensor(normalized_targets, dtype=torch.float32),  # Convert normalized joint targets into one dense tensor.
    )


def _evaluate_normalized_mse(*, model, features, targets, device, torch) -> float:
    """Evaluate average MSE in normalized joint-target space."""

    model.eval()  # Switch to inference mode before evaluation.
    with torch.no_grad():  # Disable gradients for validation.
        predictions = model(features.to(device))  # Predict normalized residual vectors.
        errors = predictions - targets.to(device)  # Measure normalized prediction error.
        return float((errors * errors).mean().detach().cpu())  # Return average normalized MSE across all heads.


def _denormalized_axis_metrics(
    *,
    model,
    features,
    targets,
    device,
    torch,
    axis_names: Sequence[str],
    target_standardizer,
) -> dict[str, dict[str, float]]:
    """Report per-axis validation metrics back in the original command scale."""

    if int(features.shape[0]) == 0:  # Handle degenerate validation splits defensively.
        return {
            axis_name: {"mae": 0.0, "rmse": 0.0}  # Emit zero metrics when no validation samples exist.
            for axis_name in axis_names
        }

    model.eval()  # Switch to inference mode before evaluation.
    with torch.no_grad():  # Disable gradients for validation.
        predictions = model(features.to(device)).detach().cpu().tolist()  # Collect normalized predictions as plain Python lists.
    target_values = targets.detach().cpu().tolist()  # Collect normalized targets as plain Python lists.

    absolute_error_sums = {axis_name: 0.0 for axis_name in axis_names}  # Accumulate per-axis absolute error in original units.
    squared_error_sums = {axis_name: 0.0 for axis_name in axis_names}  # Accumulate per-axis squared error in original units.
    for normalized_prediction_row, normalized_target_row in zip(predictions, target_values):  # Walk prediction/target pairs together.
        prediction_row = target_standardizer.denormalize(normalized_prediction_row)  # Map the predicted joint vector back to command units.
        target_row = target_standardizer.denormalize(normalized_target_row)  # Map the target joint vector back to command units.
        for axis_index, axis_name in enumerate(axis_names):  # Walk the joint vector in the stable axis order.
            diff = prediction_row[axis_index] - target_row[axis_index]  # Measure denormalized error for this axis.
            absolute_error_sums[axis_name] += abs(diff)  # Accumulate denormalized absolute error.
            squared_error_sums[axis_name] += diff * diff  # Accumulate denormalized squared error.

    count = len(predictions)  # Count how many validation windows were evaluated.
    return {
        axis_name: {
            "mae": absolute_error_sums[axis_name] / count,  # Convert accumulated absolute error into average MAE.
            "rmse": math.sqrt(squared_error_sums[axis_name] / count),  # Convert accumulated squared error into RMSE.
        }
        for axis_name in axis_names
    }


def _resolve_device(*, torch, device_name: str):
    """Resolve the requested torch device with a CPU fallback."""

    requested = device_name.strip().lower()  # Normalize the requested device name once.
    if requested == "cuda" and not torch.cuda.is_available():  # Guard against requesting CUDA on a CPU-only machine.
        print("requested cuda but CUDA is unavailable; falling back to cpu")  # Tell the user about the fallback explicitly.
        return torch.device("cpu")  # Fall back to CPU instead of failing.
    return torch.device(device_name)  # Accept the requested device when it is available.


if __name__ == "__main__":  # Allow running this file directly as a script.
    main()  # Execute the CLI entry point.
