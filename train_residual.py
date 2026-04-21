"""Train a residual buoyancy controller from CSV telemetry logs."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.data import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_WINDOW_SIZE,
    Example,
    fit_standardizer,
    load_control_rows,
    build_examples,
    split_examples_by_session,
)
from learning.model import MLPRegressor


def main() -> None:
    args = parse_args()
    rows = load_control_rows(args.csv)
    dataset = build_examples(
        rows,
        args.feature_columns,
        window_size=args.window_size,
        target_column=args.target_column,
        max_dt_ms=args.max_dt_ms,
    )
    if not dataset.examples:
        raise SystemExit("no trainable examples were built from the supplied CSV")

    train_set, val_set = split_examples_by_session(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_features = [example.features for example in train_set.examples]
    train_targets = [[example.target] for example in train_set.examples]
    feature_standardizer = fit_standardizer(train_features)
    target_standardizer = fit_standardizer(train_targets)

    normalized_train = _normalize_samples(
        train_set.examples,
        feature_standardizer,
        target_standardizer,
    )
    normalized_val = _normalize_samples(
        val_set.examples,
        feature_standardizer,
        target_standardizer,
    )

    model = MLPRegressor(
        input_dim=len(dataset.feature_names),
        hidden_dims=tuple(args.hidden_dims),
        seed=args.seed,
    )
    best_snapshot = copy.deepcopy(model.to_dict())
    best_val_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        train_loss = model.train_epoch(
            normalized_train,
            learning_rate=args.learning_rate,
            l2=args.l2,
            shuffle=True,
        )
        val_loss = model.mean_squared_error(normalized_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_snapshot = copy.deepcopy(model.to_dict())

        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            metrics = _denormalized_metrics(
                model,
                normalized_val,
                target_standardizer,
            )
            print(
                f"epoch={epoch:4d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_mae_pwm={metrics['mae']:.3f} "
                f"val_rmse_pwm={metrics['rmse']:.3f}"
            )

    best_model = MLPRegressor.from_dict(best_snapshot)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_payload = {
        "metadata": {
            "source_csv": str(Path(args.csv).resolve()),
            "window_size": args.window_size,
            "feature_columns": list(args.feature_columns),
            "feature_names": list(dataset.feature_names),
            "target_column": args.target_column or "auto",
            "train_examples": len(train_set.examples),
            "val_examples": len(val_set.examples),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "best_val_loss": best_val_loss,
        },
        "input_standardizer": feature_standardizer.to_dict(),
        "target_standardizer": target_standardizer.to_dict(),
        "model": best_model.to_dict(),
    }
    output_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    print(f"saved model bundle to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a residual buoyancy MLP from control telemetry CSV logs.",
    )
    parser.add_argument("--csv", required=True, help="Path to control_telemetry.csv")
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the exported model JSON bundle",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="How many historical frames to stack into one training example",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Base feature columns expected in the telemetry CSV",
    )
    parser.add_argument(
        "--target-column",
        default="",
        help="Optional explicit residual target column. If omitted, the loader tries residual_target_pwm, then u_residual, then u_total-u_base.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[24, 12],
        help="Hidden layer widths for the MLP",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-dt-ms", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def _normalize_samples(
    examples: list[Example],
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
