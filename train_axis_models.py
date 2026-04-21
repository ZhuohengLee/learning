"""Train separate residual models for depth, forward, and yaw control."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.data import (
    DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,
    DEFAULT_WINDOW_SIZE,
    Example,
    build_examples,
    fit_standardizer,
    load_control_rows,
    split_examples_by_session,
)
from learning.model import MLPRegressor


DEFAULT_AXIS_TARGETS = {
    "depth": "u_residual",
    "forward": "forward_cmd_residual",
    "yaw": "yaw_cmd_residual",
}


def main() -> None:
    args = parse_args()
    rows = load_control_rows(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = train_axis_models(
        rows=rows,
        output_dir=output_dir,
        feature_columns=args.feature_columns,
        window_size=args.window_size,
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        val_fraction=args.val_fraction,
        max_dt_ms=args.max_dt_ms,
        seed=args.seed,
        print_every=args.print_every,
        axis_targets={
            "depth": args.depth_target_column,
            "forward": args.forward_target_column,
            "yaw": args.yaw_target_column,
        },
    )
    manifest_path = output_dir / "axis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved axis manifest to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate residual models for depth, forward, and yaw control.",
    )
    parser.add_argument("--csv", required=True, help="Path to control telemetry CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for exported model bundles")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(DEFAULT_MULTI_AXIS_FEATURE_COLUMNS),
        help="Shared feature columns used for every axis model",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[24, 12])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-dt-ms", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--depth-target-column", default=DEFAULT_AXIS_TARGETS["depth"])
    parser.add_argument("--forward-target-column", default=DEFAULT_AXIS_TARGETS["forward"])
    parser.add_argument("--yaw-target-column", default=DEFAULT_AXIS_TARGETS["yaw"])
    return parser.parse_args()


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
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = _augment_missing_axis_targets(rows, axis_targets)
    manifest: dict[str, object] = {
        "feature_columns": list(feature_columns),
        "window_size": window_size,
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

        train_set, val_set = split_examples_by_session(
            dataset,
            val_fraction=val_fraction,
            seed=seed,
        )
        bundle, metrics = train_single_axis(
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
            source_axis=axis_name,
            target_column=target_column,
        )

        output_path = output_dir / f"{axis_name}_model.json"
        output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"[{axis_name}] saved model bundle to {output_path}")
        manifest["axes"][axis_name] = {
            "target_column": target_column,
            "output_path": str(output_path),
            **metrics,
        }

    return manifest


def _augment_missing_axis_targets(
    rows: Sequence[dict[str, str]],
    axis_targets: dict[str, str],
) -> list[dict[str, str]]:
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
    train_features = [example.features for example in train_examples]
    train_targets = [[example.target] for example in train_examples]
    feature_standardizer = fit_standardizer(train_features)
    target_standardizer = fit_standardizer(train_targets)

    normalized_train = _normalize_samples(train_examples, feature_standardizer, target_standardizer)
    normalized_val = _normalize_samples(val_examples, feature_standardizer, target_standardizer)

    model = MLPRegressor(
        input_dim=len(feature_names),
        hidden_dims=tuple(hidden_dims),
        seed=seed,
    )
    best_snapshot = copy.deepcopy(model.to_dict())
    best_val_loss = math.inf

    for epoch in range(1, epochs + 1):
        train_loss = model.train_epoch(
            normalized_train,
            learning_rate=learning_rate,
            l2=l2,
            shuffle=True,
        )
        val_loss = model.mean_squared_error(normalized_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_snapshot = copy.deepcopy(model.to_dict())

        if epoch == 1 or epoch == epochs or epoch % print_every == 0:
            metrics = _denormalized_metrics(model, normalized_val, target_standardizer)
            print(
                f"[{source_axis}] epoch={epoch:4d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae={metrics['mae']:.3f} "
                f"val_rmse={metrics['rmse']:.3f}"
            )

    best_model = MLPRegressor.from_dict(best_snapshot)
    bundle = {
        "metadata": {
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
        "model": best_model.to_dict(),
    }
    metrics = {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_val_loss": best_val_loss,
    }
    return bundle, metrics


def _normalize_samples(
    examples: Sequence[Example],
    feature_standardizer,
    target_standardizer,
) -> list[tuple[list[float], float]]:
    samples: list[tuple[list[float], float]] = []
    for example in examples:
        normalized_features = feature_standardizer.normalize(example.features)
        normalized_target = target_standardizer.normalize([example.target])[0]
        samples.append((normalized_features, normalized_target))
    return samples


def _denormalized_metrics(model, samples, target_standardizer) -> dict[str, float]:
    if not samples:
        return {"mae": 0.0, "rmse": 0.0}

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    for features, normalized_target in samples:
        normalized_prediction = model.predict_one(features)
        prediction = target_standardizer.denormalize([normalized_prediction])[0]
        target = target_standardizer.denormalize([normalized_target])[0]
        diff = prediction - target
        absolute_error_sum += abs(diff)
        squared_error_sum += diff * diff

    count = len(samples)
    return {
        "mae": absolute_error_sum / count,
        "rmse": math.sqrt(squared_error_sum / count),
    }


if __name__ == "__main__":
    main()
