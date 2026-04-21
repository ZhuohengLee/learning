"""CSV loading and feature preparation for residual-control training."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_WINDOW_SIZE = 5
DEFAULT_FEATURE_COLUMNS = [
    "depth_err_cm",
    "depth_speed_cm_s",
    "depth_accel_cm_s2",
    "roll_deg",
    "pitch_deg",
    "gyro_x_deg_s",
    "gyro_y_deg_s",
    "gyro_z_deg_s",
    "battery_v",
    "buoyancy_pwm_applied",
]
DEFAULT_MULTI_AXIS_FEATURE_COLUMNS = [
    "depth_err_cm",
    "depth_speed_cm_s",
    "depth_accel_cm_s2",
    "roll_deg",
    "pitch_deg",
    "gyro_x_deg_s",
    "gyro_y_deg_s",
    "gyro_z_deg_s",
    "front_distance_cm",
    "left_distance_cm",
    "right_distance_cm",
    "battery_v",
    "u_base",
    "forward_cmd_base",
    "yaw_cmd_base",
]
DEFAULT_TARGET_PRIORITY = ["residual_target_pwm", "u_residual"]


@dataclass(frozen=True)
class Example:
    """One training example for residual control."""

    session_id: str
    timestamp_ms: int
    features: list[float]
    target: float


@dataclass(frozen=True)
class ExampleSet:
    """Prepared dataset plus feature naming metadata."""

    examples: list[Example]
    feature_columns: list[str]
    feature_names: list[str]
    window_size: int


@dataclass(frozen=True)
class Standardizer:
    """Column-wise z-score normalization."""

    means: list[float]
    stds: list[float]

    def normalize(self, values: Sequence[float]) -> list[float]:
        if len(values) != len(self.means):
            raise ValueError("value dimension does not match standardizer")
        return [
            (float(value) - mean) / std
            for value, mean, std in zip(values, self.means, self.stds)
        ]

    def denormalize(self, values: Sequence[float]) -> list[float]:
        if len(values) != len(self.means):
            raise ValueError("value dimension does not match standardizer")
        return [
            float(value) * std + mean
            for value, mean, std in zip(values, self.means, self.stds)
        ]

    def to_dict(self) -> dict[str, list[float]]:
        return {"means": self.means, "stds": self.stds}


def load_control_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load telemetry rows from a CSV file."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"telemetry CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def compute_feature_names(feature_columns: Sequence[str], window_size: int) -> list[str]:
    """Expand base feature names into flattened window feature names."""

    feature_names: list[str] = []
    for offset in range(window_size):
        lag = window_size - 1 - offset
        for column in feature_columns:
            feature_names.append(f"t-{lag}_{column}")
    return feature_names


def fit_standardizer(samples: Sequence[Sequence[float]]) -> Standardizer:
    """Fit mean/std statistics for a matrix-like sample list."""

    if not samples:
        raise ValueError("cannot fit a standardizer on an empty dataset")

    width = len(samples[0])
    if width == 0:
        raise ValueError("cannot fit a standardizer with zero-width samples")

    means = [0.0] * width
    for row in samples:
        if len(row) != width:
            raise ValueError("all samples must have the same width")
        for index, value in enumerate(row):
            means[index] += float(value)
    means = [value / len(samples) for value in means]

    variances = [0.0] * width
    for row in samples:
        for index, value in enumerate(row):
            diff = float(value) - means[index]
            variances[index] += diff * diff

    stds = []
    for variance in variances:
        std = math.sqrt(variance / len(samples))
        stds.append(std if std > 1e-8 else 1.0)

    return Standardizer(means=means, stds=stds)


def split_examples_by_session(
    dataset: ExampleSet,
    val_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[ExampleSet, ExampleSet]:
    """Split examples by session id to reduce train/validation leakage."""

    if not dataset.examples:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    grouped: dict[str, list[Example]] = {}
    for example in dataset.examples:
        grouped.setdefault(example.session_id, []).append(example)

    sessions = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(sessions)

    if len(sessions) == 1:
        cutoff = max(1, int(round(len(dataset.examples) * (1.0 - val_fraction))))
        cutoff = min(cutoff, len(dataset.examples))
        train_examples = dataset.examples[:cutoff]
        val_examples = dataset.examples[cutoff:] or dataset.examples[-1:]
    else:
        target_val_sessions = max(1, int(round(len(sessions) * val_fraction)))
        target_val_sessions = min(target_val_sessions, len(sessions) - 1)
        val_session_ids = set(sessions[:target_val_sessions])
        train_examples = [
            example for example in dataset.examples if example.session_id not in val_session_ids
        ]
        val_examples = [
            example for example in dataset.examples if example.session_id in val_session_ids
        ]

    train_set = ExampleSet(
        examples=train_examples,
        feature_columns=list(dataset.feature_columns),
        feature_names=list(dataset.feature_names),
        window_size=dataset.window_size,
    )
    val_set = ExampleSet(
        examples=val_examples,
        feature_columns=list(dataset.feature_columns),
        feature_names=list(dataset.feature_names),
        window_size=dataset.window_size,
    )
    return train_set, val_set


def build_examples(
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_column: str | None = None,
    max_dt_ms: float = 80.0,
) -> ExampleSet:
    """Convert raw telemetry rows into fixed-width sliding window examples."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")

    sequences = _split_into_sequences(
        rows=rows,
        feature_columns=feature_columns,
        max_dt_ms=max_dt_ms,
    )
    feature_names = compute_feature_names(feature_columns, window_size)
    examples: list[Example] = []

    for sequence in sequences:
        if len(sequence) < window_size:
            continue

        for end_index in range(window_size - 1, len(sequence)):
            window = sequence[end_index - window_size + 1 : end_index + 1]
            flattened: list[float] = []
            for row in window:
                for column in feature_columns:
                    flattened.append(_read_float(row, column))

            final_row = window[-1]
            examples.append(
                Example(
                    session_id=_session_id(final_row),
                    timestamp_ms=int(_read_float(final_row, "timestamp_ms", default=0.0)),
                    features=flattened,
                    target=_resolve_target(final_row, target_column),
                )
            )

    return ExampleSet(
        examples=examples,
        feature_columns=list(feature_columns),
        feature_names=feature_names,
        window_size=window_size,
    )


def _split_into_sequences(
    *,
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    max_dt_ms: float,
) -> list[list[dict[str, str]]]:
    sequences: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    previous_session = ""

    for row in rows:
        if not _row_is_trainable(row, feature_columns):
            if current:
                sequences.append(current)
                current = []
            previous_session = ""
            continue

        session_id = _session_id(row)
        if current:
            previous = current[-1]
            previous_ts = _read_float(previous, "timestamp_ms", default=0.0)
            current_ts = _read_float(row, "timestamp_ms", default=0.0)
            if (
                session_id != previous_session
                or current_ts < previous_ts
                or current_ts - previous_ts > max_dt_ms
            ):
                sequences.append(current)
                current = []

        current.append(row)
        previous_session = session_id

    if current:
        sequences.append(current)

    return sequences


def _row_is_trainable(row: dict[str, str], feature_columns: Sequence[str]) -> bool:
    if row.get("depth_valid") and not _read_flag(row, "depth_valid"):
        return False
    if row.get("imu_valid") and not _read_flag(row, "imu_valid"):
        return False
    if _read_flag(row, "balancing", default=False):
        return False
    if _read_flag(row, "emergency_stop", default=False):
        return False

    control_mode = row.get("control_mode", "").strip().lower()
    if control_mode and control_mode not in {"1", "auto"}:
        return False

    try:
        for column in feature_columns:
            _read_float(row, column)
        _resolve_target(row, target_column=None)
    except ValueError:
        return False

    return True


def _resolve_target(row: dict[str, str], target_column: str | None) -> float:
    if target_column:
        return _read_float(row, target_column)

    for field in DEFAULT_TARGET_PRIORITY:
        raw_value = row.get(field, "").strip()
        if raw_value:
            return float(raw_value)

    if row.get("u_total", "").strip() and row.get("u_base", "").strip():
        return _read_float(row, "u_total") - _read_float(row, "u_base")

    raise ValueError("row does not contain a residual training target")


def _read_float(row: dict[str, str], field: str, *, default: float | None = None) -> float:
    raw_value = row.get(field)
    if raw_value is None or raw_value == "":
        if default is not None:
            return default
        raise ValueError(f"missing numeric field: {field}")
    return float(raw_value)


def _read_flag(row: dict[str, str], field: str, *, default: bool = True) -> bool:
    raw_value = row.get(field)
    if raw_value is None or raw_value == "":
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean-like field {field}={raw_value!r}")


def _session_id(row: dict[str, str]) -> str:
    value = row.get("session_id", "").strip()
    return value or "default"
