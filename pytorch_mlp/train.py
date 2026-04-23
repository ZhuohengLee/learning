"""Train depth, forward, and yaw residual models with the PyTorch backend.

The telemetry contract matches `learning.data`. The output of this backend is a `.pt`
bundle per axis plus a JSON manifest. These bundles are intended for later ONNX export
and then ESP-PPQ / ESP-DL deployment.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the training CLI.
import json  # Export the manifest in JSON format.
import math  # Compute RMSE for validation metrics.
from pathlib import Path  # Build output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.
from typing import Sequence  # Accept generic ordered collections as inputs.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Add the repo root so `learning` can be imported.

from learning.data import (  # Reuse telemetry loading and dataset-prep helpers.
    DEFAULT_UNIFIED_FEATURE_COLUMNS,  # Default shared feature set for all three public axis models.
    DEFAULT_WINDOW_SIZE,  # Default number of stacked frames per sample.
    Example,  # Typed training example record.
    build_examples,  # Convert raw rows into windowed examples.
    fit_standardizer,  # Fit z-score normalization statistics.
    load_control_rows,  # Load raw telemetry rows from CSV.
    split_examples_by_session,  # Split examples without session leakage.
)
from learning.pytorch_mlp.model import TorchResidualMLP, require_torch  # Import the PyTorch backend and dependency guard.


DEFAULT_AXIS_TARGETS = {
    "depth": "u_residual",  # Residual depth-control output.
    "forward": "forward_cmd_residual",  # Residual forward-control output.
    "yaw": "yaw_cmd_residual",  # Residual yaw-control output.
}  # Keep the public three-axis defaults in one place.


def main() -> None:
    """CLI entry point for unified three-axis PyTorch training."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    args = parse_args()  # Parse all user-supplied training options.
    rows = load_control_rows(args.csv)  # Load raw telemetry rows from the requested CSV.
    output_dir = Path(args.output_dir)  # Normalize the requested output directory.
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory before training starts.

    manifest = train_models(
        rows=rows,  # Pass raw telemetry rows.
        output_dir=output_dir,  # Save exported model bundles into this directory.
        feature_columns=tuple(args.feature_columns),  # Use the configured shared feature set.
        window_size=args.window_size,  # Stack this many recent frames per example.
        hidden_dims=tuple(args.hidden_dims),  # Use the configured hidden-layer widths.
        epochs=args.epochs,  # Train for this many epochs per axis.
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
    )  # Train one residual bundle per requested axis.
    manifest_path = output_dir / "axis_manifest.json"  # Name the summary manifest file.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Save the manifest as pretty JSON.
    print(f"saved axis manifest to {manifest_path}")  # Tell the user where the manifest was written.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for unified three-axis PyTorch training."""

    parser = argparse.ArgumentParser(
        description="Train depth, forward, and yaw residual PyTorch MLPs from one control telemetry CSV.",
    )
    parser.add_argument("--csv", required=True, help="Path to control telemetry CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for exported .pt bundles")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(DEFAULT_UNIFIED_FEATURE_COLUMNS),
        help="Shared base feature columns used by the depth, forward, and yaw PyTorch models.",
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
    parser.add_argument("--depth-target-column", default=DEFAULT_AXIS_TARGETS["depth"])
    parser.add_argument("--forward-target-column", default=DEFAULT_AXIS_TARGETS["forward"])
    parser.add_argument("--yaw-target-column", default=DEFAULT_AXIS_TARGETS["yaw"])
    return parser.parse_args()


def train_models(
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
    """Train one PyTorch residual model per control axis and return a manifest."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    import torch  # type: ignore[import-not-found]

    if batch_size < 1:  # Refuse invalid minibatch sizes.
        raise ValueError("batch_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = _augment_missing_axis_targets(rows, axis_targets)
    device = _resolve_device(torch=torch, device_name=device_name)
    manifest: dict[str, object] = {
        "backend": "pytorch_mlp",
        "feature_columns": list(feature_columns),
        "window_size": window_size,
        "device": str(device),
        "axes": {},
    }

    for axis_name, target_column in axis_targets.items():
        if not target_column:
            continue

        dataset = build_examples(
            prepared_rows,
            feature_columns,
            window_size=window_size,
            target_column=target_column,
            max_dt_ms=max_dt_ms,
        )
        if not dataset.examples:
            raise ValueError(f"no trainable examples were built for axis {axis_name}")

        train_set, val_set = split_examples_by_session(dataset, val_fraction=val_fraction, seed=seed)
        bundle, metrics = _train_single_axis(
            train_examples=train_set.examples,
            val_examples=val_set.examples,
            feature_names=dataset.feature_names,
            feature_columns=dataset.feature_columns,
            window_size=dataset.window_size,
            hidden_dims=hidden_dims,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
            print_every=print_every,
            batch_size=batch_size,
            device=device,
            source_axis=axis_name,
            target_column=target_column,
        )

        output_path = output_dir / f"{axis_name}_model.pt"
        torch.save(bundle, output_path)
        print(f"[{axis_name}] saved PyTorch model bundle to {output_path}")
        manifest["axes"][axis_name] = {  # type: ignore[index]
            "target_column": target_column,
            "output_path": str(output_path),
            **metrics,
        }

    return manifest


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

    prepared_rows: list[dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        for target_column, pair in derived_pairs.items():
            if not target_column:
                continue
            raw_target = copied.get(target_column, "").strip()
            if raw_target:
                continue
            total_key, base_key = pair
            raw_total = copied.get(total_key, "").strip()
            raw_base = copied.get(base_key, "").strip()
            if raw_total and raw_base:
                copied[target_column] = str(float(raw_total) - float(raw_base))
        prepared_rows.append(copied)
    return prepared_rows


def _train_single_axis(
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
    batch_size: int,
    device,
    source_axis: str,
    target_column: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train, evaluate, and export one axis-specific PyTorch model bundle."""

    require_torch()
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found]

    torch.manual_seed(seed)

    train_features = [example.features for example in train_examples]
    train_targets = [[example.target] for example in train_examples]
    feature_standardizer = fit_standardizer(train_features)
    target_standardizer = fit_standardizer(train_targets)

    train_feature_tensor, train_target_tensor = _build_normalized_tensors(
        torch=torch,
        examples=train_examples,
        feature_standardizer=feature_standardizer,
        target_standardizer=target_standardizer,
    )
    val_feature_tensor, val_target_tensor = _build_normalized_tensors(
        torch=torch,
        examples=val_examples,
        feature_standardizer=feature_standardizer,
        target_standardizer=target_standardizer,
    )

    train_loader = DataLoader(
        TensorDataset(train_feature_tensor, train_target_tensor),
        batch_size=min(batch_size, len(train_examples)),
        shuffle=True,
    )

    model = TorchResidualMLP(input_dim=len(feature_names), hidden_dims=tuple(hidden_dims)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    loss_fn = nn.MSELoss()
    best_snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_loss = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            batch_size_now = batch_features.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * batch_size_now
            train_sample_count += int(batch_size_now)

        train_loss = train_loss_sum / max(1, train_sample_count)
        val_loss = _evaluate_normalized_mse(
            model=model,
            features=val_feature_tensor,
            targets=val_target_tensor,
            device=device,
            torch=torch,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        if epoch == 1 or epoch == epochs or epoch % print_every == 0:
            metrics = _denormalized_metrics(
                model=model,
                features=val_feature_tensor,
                targets=val_target_tensor,
                device=device,
                torch=torch,
                target_standardizer=target_standardizer,
            )
            print(
                f"[{source_axis}] epoch={epoch:4d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae={metrics['mae']:.3f} "
                f"val_rmse={metrics['rmse']:.3f}"
            )

    bundle = {
        "metadata": {
            "backend": "pytorch_mlp",
            "axis": source_axis,
            "window_size": window_size,
            "feature_columns": list(feature_columns),
            "feature_names": list(feature_names),
            "target_column": target_column,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "best_val_loss": best_val_loss,
        },
        "input_standardizer": feature_standardizer.to_dict(),
        "target_standardizer": target_standardizer.to_dict(),
        "model_spec": {
            "input_dim": len(feature_names),
            "hidden_dims": [int(value) for value in hidden_dims],
            "output_dim": 1,
            "hidden_activation": "tanh",
            "output_activation": "linear",
        },
        "state_dict": best_snapshot,
    }
    metrics = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_val_loss": best_val_loss,
    }
    return bundle, metrics


def _build_normalized_tensors(
    *,
    torch,
    examples: Sequence[Example],
    feature_standardizer,
    target_standardizer,
):
    """Convert examples into normalized float32 tensors."""

    normalized_features = [
        feature_standardizer.normalize(example.features)
        for example in examples
    ]
    normalized_targets = [
        [target_standardizer.normalize([example.target])[0]]
        for example in examples
    ]
    return (
        torch.tensor(normalized_features, dtype=torch.float32),
        torch.tensor(normalized_targets, dtype=torch.float32),
    )


def _evaluate_normalized_mse(*, model, features, targets, device, torch) -> float:
    """Evaluate MSE in normalized target space."""

    model.eval()
    with torch.no_grad():
        predictions = model(features.to(device))
        errors = predictions - targets.to(device)
        return float((errors * errors).mean().detach().cpu())


def _denormalized_metrics(*, model, features, targets, device, torch, target_standardizer) -> dict[str, float]:
    """Report validation metrics back in the original command scale."""

    if int(features.shape[0]) == 0:
        return {"mae": 0.0, "rmse": 0.0}

    model.eval()
    with torch.no_grad():
        predictions = model(features.to(device)).detach().cpu().flatten().tolist()
    target_values = targets.detach().cpu().flatten().tolist()

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    for normalized_prediction, normalized_target in zip(predictions, target_values):
        prediction = target_standardizer.denormalize([float(normalized_prediction)])[0]
        target = target_standardizer.denormalize([float(normalized_target)])[0]
        diff = prediction - target
        absolute_error_sum += abs(diff)
        squared_error_sum += diff * diff

    count = len(predictions)
    return {
        "mae": absolute_error_sum / count,
        "rmse": math.sqrt(squared_error_sum / count),
    }


def _resolve_device(*, torch, device_name: str):
    """Resolve the requested torch device with a CPU fallback."""

    requested = device_name.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("requested cuda but CUDA is unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()

