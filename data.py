"""CSV loading and feature preparation for residual-control training.

Reading route:
1. Start with `build_examples()` because it is the main entry point from training scripts.
2. Then read `split_examples_by_session()` to understand train/validation splitting.
3. Then read `fit_standardizer()` to understand normalization.
4. Finally read `_split_into_sequences()`, `_row_is_trainable()`, and `_resolve_target()`
   to see how raw telemetry rows are filtered and converted into supervised targets.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import csv  # Read telemetry exported in CSV format.
import math  # Compute square roots for standard deviation.
import random  # Shuffle sessions before train/validation splitting.
from dataclasses import dataclass  # Define lightweight immutable records.
from pathlib import Path  # Accept filesystem paths in a platform-safe way.
from typing import Sequence  # Type sequences without forcing a specific container.


DEFAULT_WINDOW_SIZE = 5  # Stack this many consecutive frames into one feature vector.
DEFAULT_FEATURE_COLUMNS = [
    "depth_err_cm",  # Signed depth tracking error in centimeters.
    "depth_speed_cm_s",  # Vertical speed estimate in centimeters per second.
    "depth_accel_cm_s2",  # Vertical acceleration estimate in centimeters per second squared.
    "roll_deg",  # Roll attitude in degrees.
    "pitch_deg",  # Pitch attitude in degrees.
    "gyro_x_deg_s",  # Gyroscope x-axis angular speed.
    "gyro_y_deg_s",  # Gyroscope y-axis angular speed.
    "gyro_z_deg_s",  # Gyroscope z-axis angular speed.
    "battery_v",  # Battery voltage measured by the robot.
    "buoyancy_pwm_applied",  # Previous buoyancy actuator PWM actually applied.
]
DEFAULT_MULTI_AXIS_FEATURE_COLUMNS = [
    "depth_err_cm",  # Shared depth error feature for all control axes.
    "depth_speed_cm_s",  # Shared vertical speed feature for all control axes.
    "depth_accel_cm_s2",  # Shared vertical acceleration feature for all control axes.
    "roll_deg",  # Shared roll feature for all control axes.
    "pitch_deg",  # Shared pitch feature for all control axes.
    "gyro_x_deg_s",  # Shared x-axis gyro feature for all control axes.
    "gyro_y_deg_s",  # Shared y-axis gyro feature for all control axes.
    "gyro_z_deg_s",  # Shared z-axis gyro feature for all control axes.
    "front_distance_cm",  # Front sonar distance in centimeters.
    "left_distance_cm",  # Left sonar distance in centimeters.
    "right_distance_cm",  # Right sonar distance in centimeters.
    "battery_v",  # Battery voltage shared across axes.
    "u_base",  # Hand-written depth controller output before residual trim.
    "forward_cmd_base",  # Hand-written forward command before residual trim.
    "forward_phase_interval_ms",  # Current forward propulsion phase interval in milliseconds.
    "yaw_cmd_base",  # Hand-written yaw command before residual trim.
]
DEFAULT_TARGET_PRIORITY = ["residual_target_pwm", "u_residual"]  # Try these residual fields in order.
TARGET_FALLBACK_PAIRS = {
    "residual_target_pwm": ("u_total", "u_base"),  # Recover depth residual when the legacy explicit field is absent.
    "u_residual": ("u_total", "u_base"),  # Recover depth residual from total minus base.
    "forward_cmd_residual": ("forward_cmd_total", "forward_cmd_base"),  # Recover forward residual from total minus base.
    "yaw_cmd_residual": ("yaw_cmd_total", "yaw_cmd_base"),  # Recover yaw residual from total minus base.
}  # Map residual field names to their `(total, base)` fallback columns.


@dataclass(frozen=True)  # Freeze the record so examples are not mutated during training.
class Example:
    """One training example for residual control."""

    session_id: str  # Session label used to keep train/validation splits clean.
    timestamp_ms: int  # Timestamp of the last frame inside the stacked window.
    features: list[float]  # Flattened feature vector built from the window.
    target: float  # Residual command the model should predict.


@dataclass(frozen=True)  # Freeze the record so dataset metadata stays consistent.
class ExampleSet:
    """Prepared dataset plus feature naming metadata."""

    examples: list[Example]  # All prepared examples in chronological order.
    feature_columns: list[str]  # Base feature names before window flattening.
    feature_names: list[str]  # Expanded feature names after window flattening.
    window_size: int  # Number of frames stacked into each example.


@dataclass(frozen=True)  # Freeze the record so normalization statistics stay stable.
class Standardizer:
    """Column-wise z-score normalization."""

    means: list[float]  # Per-column mean used for normalization.
    stds: list[float]  # Per-column standard deviation used for normalization.

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Apply column-wise z-score normalization."""

        if len(values) != len(self.means):  # Guard against mismatched feature dimensions.
            raise ValueError("value dimension does not match standardizer")  # Fail fast on invalid input.
        return [  # Return a new normalized list without mutating the input.
            (float(value) - mean) / std  # Apply `(x - mean) / std` independently per column.
            for value, mean, std in zip(values, self.means, self.stds)  # Walk values and stats together.
        ]

    def denormalize(self, values: Sequence[float]) -> list[float]:
        """Map normalized values back into the original feature space."""

        if len(values) != len(self.means):  # Guard against mismatched feature dimensions.
            raise ValueError("value dimension does not match standardizer")  # Fail fast on invalid input.
        return [  # Return a new denormalized list without mutating the input.
            float(value) * std + mean  # Undo z-score normalization independently per column.
            for value, mean, std in zip(values, self.means, self.stds)  # Walk values and stats together.
        ]

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize the normalization statistics for JSON export."""

        return {"means": self.means, "stds": self.stds}  # Emit a simple JSON-friendly payload.


def load_control_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load telemetry rows from a CSV file."""

    path = Path(csv_path)  # Normalize string or Path inputs into a Path object.
    if not path.is_file():  # Verify the CSV exists before opening it.
        raise FileNotFoundError(f"telemetry CSV not found: {path}")  # Raise a clear path-specific error.

    with path.open("r", newline="", encoding="utf-8") as handle:  # Open in text mode with stable CSV handling.
        reader = csv.DictReader(handle)  # Parse the header and rows as dictionaries.
        if not reader.fieldnames:  # Reject files that have no readable header row.
            raise ValueError(f"CSV has no header: {path}")  # Fail with a message tied to the file path.
        return [dict(row) for row in reader]  # Convert each row into a plain dict for downstream code.


def compute_feature_names(feature_columns: Sequence[str], window_size: int) -> list[str]:
    """Expand base feature names into flattened window feature names."""

    feature_names: list[str] = []  # Accumulate names in the same order as flattening.
    for offset in range(window_size):  # Walk every position inside the time window.
        lag = window_size - 1 - offset  # Convert the loop position into a readable time lag.
        for column in feature_columns:  # Repeat every base feature for that lag.
            feature_names.append(f"t-{lag}_{column}")  # Match the exact flattened feature ordering.
    return feature_names  # Return the full expanded name list.


def fit_standardizer(samples: Sequence[Sequence[float]]) -> Standardizer:
    """Fit mean/std statistics for a matrix-like sample list."""

    if not samples:  # Refuse to compute statistics from an empty dataset.
        raise ValueError("cannot fit a standardizer on an empty dataset")  # Explain why fitting failed.

    width = len(samples[0])  # Use the first sample to define the expected feature width.
    if width == 0:  # Reject degenerate zero-width feature vectors.
        raise ValueError("cannot fit a standardizer with zero-width samples")  # Explain why fitting failed.

    means = [0.0] * width  # Start running column sums at zero.
    for row in samples:  # Visit every sample row once.
        if len(row) != width:  # Enforce a consistent feature width across all rows.
            raise ValueError("all samples must have the same width")  # Fail fast on malformed data.
        for index, value in enumerate(row):  # Walk every column of the current row.
            means[index] += float(value)  # Accumulate the column sum as a float.
    means = [value / len(samples) for value in means]  # Convert sums into per-column means.

    variances = [0.0] * width  # Start variance accumulators at zero.
    for row in samples:  # Visit every sample row again for variance.
        for index, value in enumerate(row):  # Walk every column of the current row.
            diff = float(value) - means[index]  # Measure deviation from the column mean.
            variances[index] += diff * diff  # Accumulate squared deviation.

    stds = []  # Accumulate per-column standard deviations here.
    for variance in variances:  # Convert each variance term independently.
        std = math.sqrt(variance / len(samples))  # Use population standard deviation for consistency.
        stds.append(std if std > 1e-8 else 1.0)  # Clamp tiny std values to avoid division by zero.

    return Standardizer(means=means, stds=stds)  # Package the fitted statistics into an immutable record.


def split_examples_by_session(
    dataset: ExampleSet,
    val_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[ExampleSet, ExampleSet]:
    """Split examples by session id to reduce train/validation leakage."""

    if not dataset.examples:  # Refuse to split an empty dataset.
        raise ValueError("cannot split an empty dataset")  # Explain why splitting failed.
    if not 0.0 <= val_fraction < 1.0:  # Enforce a sane validation fraction range.
        raise ValueError("val_fraction must be in [0, 1)")  # Explain the accepted interval.

    grouped: dict[str, list[Example]] = {}  # Bucket examples by session id.
    for example in dataset.examples:  # Visit every prepared example once.
        grouped.setdefault(example.session_id, []).append(example)  # Append the example to its session bucket.

    sessions = list(grouped.keys())  # Extract the distinct session identifiers.
    rng = random.Random(seed)  # Build a deterministic RNG for reproducible splits.
    rng.shuffle(sessions)  # Shuffle session order before partitioning.

    if len(sessions) == 1:  # Special-case a single-session dataset.
        cutoff = max(1, int(round(len(dataset.examples) * (1.0 - val_fraction))))  # Keep at least one train sample.
        cutoff = min(cutoff, len(dataset.examples))  # Clamp the cutoff inside dataset bounds.
        train_examples = dataset.examples[:cutoff]  # Use the front portion for training.
        val_examples = dataset.examples[cutoff:] or dataset.examples[-1:]  # Ensure validation is never empty.
    else:  # Use session-level splitting when multiple sessions exist.
        target_val_sessions = max(1, int(round(len(sessions) * val_fraction)))  # Request at least one validation session.
        target_val_sessions = min(target_val_sessions, len(sessions) - 1)  # Keep at least one training session.
        val_session_ids = set(sessions[:target_val_sessions])  # Mark the shuffled validation session ids.
        train_examples = [
            example for example in dataset.examples if example.session_id not in val_session_ids
        ]  # Build the training subset from non-validation sessions.
        val_examples = [
            example for example in dataset.examples if example.session_id in val_session_ids
        ]  # Build the validation subset from validation sessions.

    train_set = ExampleSet(
        examples=train_examples,  # Attach the selected training examples.
        feature_columns=list(dataset.feature_columns),  # Copy base feature-column metadata.
        feature_names=list(dataset.feature_names),  # Copy expanded feature-name metadata.
        window_size=dataset.window_size,  # Preserve the original window size.
    )
    val_set = ExampleSet(
        examples=val_examples,  # Attach the selected validation examples.
        feature_columns=list(dataset.feature_columns),  # Copy base feature-column metadata.
        feature_names=list(dataset.feature_names),  # Copy expanded feature-name metadata.
        window_size=dataset.window_size,  # Preserve the original window size.
    )
    return train_set, val_set  # Return both splits together.


def build_examples(
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_column: str | None = None,
    max_dt_ms: float = 80.0,
) -> ExampleSet:
    """Convert raw telemetry rows into fixed-width sliding window examples."""

    if window_size < 1:  # Refuse zero-length or negative windows.
        raise ValueError("window_size must be at least 1")  # Explain the lower bound.
    if not feature_columns:  # Refuse a dataset with no input features.
        raise ValueError("feature_columns must not be empty")  # Explain why preparation failed.

    sequences = _split_into_sequences(
        rows=rows,  # Pass all raw CSV rows.
        feature_columns=feature_columns,  # Validate these feature columns inside each row.
        target_column=target_column,  # Validate rows against the target actually requested by the caller.
        max_dt_ms=max_dt_ms,  # Cut sequences when the time gap gets too large.
    )
    feature_names = compute_feature_names(feature_columns, window_size)  # Build the flattened feature-name list.
    examples: list[Example] = []  # Accumulate prepared training examples here.

    for sequence in sequences:  # Process each contiguous trainable run separately.
        if len(sequence) < window_size:  # Skip runs that are too short for one full window.
            continue  # Nothing can be extracted from this sequence.

        for end_index in range(window_size - 1, len(sequence)):  # Slide the trailing window over the sequence.
            window = sequence[end_index - window_size + 1 : end_index + 1]  # Extract the current time window.
            flattened: list[float] = []  # Flatten all window frames into one vector.
            for row in window:  # Walk frames from oldest to newest inside the window.
                for column in feature_columns:  # Walk features in the configured base order.
                    flattened.append(_read_float(row, column))  # Append the numeric feature value.

            final_row = window[-1]  # Use the newest frame for timestamp and target resolution.
            examples.append(
                Example(
                    session_id=_session_id(final_row),  # Attach the current session id.
                    timestamp_ms=int(_read_float(final_row, "timestamp_ms", default=0.0)),  # Keep the newest frame time.
                    features=flattened,  # Store the flattened window vector.
                    target=_resolve_target(final_row, target_column),  # Resolve the regression target for this window.
                )
            )

    return ExampleSet(
        examples=examples,  # Attach every prepared example.
        feature_columns=list(feature_columns),  # Copy the configured base feature columns.
        feature_names=feature_names,  # Attach the computed flattened feature names.
        window_size=window_size,  # Preserve the configured window size.
    )


def _split_into_sequences(
    *,
    rows: Sequence[dict[str, str]],
    feature_columns: Sequence[str],
    target_column: str | None,
    max_dt_ms: float,
) -> list[list[dict[str, str]]]:
    """Break the log into contiguous trainable stretches."""

    sequences: list[list[dict[str, str]]] = []  # Accumulate contiguous trainable runs.
    current: list[dict[str, str]] = []  # Hold the run currently being built.
    previous_session = ""  # Track the previous session id to detect boundaries.

    for row in rows:  # Walk through the raw CSV in chronological file order.
        if not _row_is_trainable(row, feature_columns, target_column):  # Reject rows from invalid regimes or missing features.
            if current:  # Close the current run if one is open.
                sequences.append(current)  # Save the finished run.
                current = []  # Start fresh after the invalid boundary.
            previous_session = ""  # Clear the previous-session marker after a hard boundary.
            continue  # Move to the next raw row.

        session_id = _session_id(row)  # Resolve the current row's session id.
        if current:  # Only compare against a previous frame when the run is non-empty.
            previous = current[-1]  # Inspect the last accepted row in the current run.
            previous_ts = _read_float(previous, "timestamp_ms", default=0.0)  # Read the last accepted timestamp.
            current_ts = _read_float(row, "timestamp_ms", default=0.0)  # Read the current row timestamp.
            if (
                session_id != previous_session  # Session changed.
                or current_ts < previous_ts  # Time moved backward.
                or current_ts - previous_ts > max_dt_ms  # Time gap is too large.
            ):
                sequences.append(current)  # Close the previous contiguous run.
                current = []  # Start a new run from the current row.

        current.append(row)  # Add the current trainable row to the active run.
        previous_session = session_id  # Remember this session for the next iteration.

    if current:  # Flush the final run if the file ended mid-sequence.
        sequences.append(current)  # Save the last contiguous run.

    return sequences  # Return all trainable contiguous runs.


def _row_is_trainable(
    row: dict[str, str],
    feature_columns: Sequence[str],
    target_column: str | None,
) -> bool:
    """Return True only for sensor-valid control frames that match the target regime."""

    if row.get("depth_valid") and not _read_flag(row, "depth_valid"):  # Drop rows with invalid depth sensing.
        return False  # Depth-dependent control cannot learn from these rows.
    if row.get("imu_valid") and not _read_flag(row, "imu_valid"):  # Drop rows with invalid IMU sensing when present.
        return False  # Motion-state features would be untrustworthy.
    if _read_flag(row, "balancing", default=False):  # Exclude balance windows.
        return False  # Balance mode is a different control regime.
    if _read_flag(row, "emergency_stop", default=False):  # Exclude emergency-stop windows.
        return False  # Emergency behavior should not be learned as nominal control.

    if _row_is_direct_buoyancy_override(row):  # Reject only the explicit `j/k` buoyancy override regime.
        return False  # Direct ascend/descend commands bypass the controller and should not be imitated.

    control_mode = _normalized_control_mode(row)  # Normalize the control mode field when present.
    if _target_requires_autonomous_depth_control(target_column) and control_mode and control_mode not in {"1", "auto"}:
        return False  # Depth residual training still requires autonomous closed-loop control frames.

    try:  # Validate numeric feature and target availability using the same helper path as training.
        for column in feature_columns:  # Check every requested feature column.
            _read_float(row, column)  # Fail if any feature is missing or malformed.
        _resolve_target(row, target_column=target_column)  # Fail if the requested residual target cannot be resolved.
    except ValueError:  # Catch any missing or malformed field error.
        return False  # Reject rows that cannot become valid training examples.

    return True  # The row belongs to the requested trainable control regime.


def _target_requires_autonomous_depth_control(target_column: str | None) -> bool:
    """Return True when the requested target belongs to the depth-control axis."""

    return target_column in {None, "", "residual_target_pwm", "u_residual"}  # Treat auto-target mode as depth training.


def _normalized_control_mode(row: dict[str, str]) -> str:
    """Return the normalized control-mode token when the CSV provides one."""

    return row.get("control_mode", "").strip().lower()  # Normalize capitalization and whitespace consistently.


def _row_is_direct_buoyancy_override(row: dict[str, str]) -> bool:
    """Detect the direct `j/k` buoyancy override signature exported by the firmware."""

    control_mode = _normalized_control_mode(row)  # Read the optional manual/auto mode flag.
    if control_mode and control_mode not in {"0", "manual"}:  # Restrict this signature check to manual-style frames.
        return False  # AUTO depth control may legitimately drive buoyancy in the same direction.

    buoyancy_direction = int(round(_read_float(row, "buoyancy_dir_applied", default=0.0)))  # Read the buoyancy direction.
    if buoyancy_direction not in {1, 2}:  # Only ascend/descend can correspond to `j/k`.
        return False  # Stop and balance frames are not manual vertical overrides.

    buoyancy_pwm = _read_float(row, "buoyancy_pwm_applied", default=0.0)  # Read the applied buoyancy PWM.
    base_output = _read_float(row, "u_base", default=0.0)  # Read the base depth-controller output.
    residual_output = _read_float(row, "u_residual", default=0.0)  # Read the residual depth trim.
    total_output = _read_float(row, "u_total", default=0.0)  # Read the exported total buoyancy command.
    return (
        buoyancy_pwm >= 250.0  # Manual overrides drive the buoyancy pump at the hard manual PWM.
        and abs(base_output) <= 1e-6  # Manual overrides bypass the base controller.
        and abs(residual_output) <= 1e-6  # Manual overrides bypass residual compensation too.
        and abs(abs(total_output) - 100.0) <= 1e-6  # Manual overrides pin the exported command to +/-100.
    )  # Match the exact signature produced by `manualAscend()` / `manualDescend()`.


def _resolve_target(row: dict[str, str], target_column: str | None) -> float:
    """Resolve the residual target value for one row."""

    if target_column:  # Respect an explicitly requested target column first.
        return _resolve_named_target(row, target_column)  # Read that exact residual field or its documented fallback.

    for field in DEFAULT_TARGET_PRIORITY:  # Try the known residual fields in preferred order.
        try:  # Probe each supported residual field in priority order.
            return _resolve_named_target(row, field)  # Return the first explicit or derived residual value that works.
        except ValueError:  # Ignore missing fields and keep trying lower-priority options.
            continue  # Move on to the next known residual field.

    raise ValueError("row does not contain a residual training target")  # Explain why target resolution failed.


def _resolve_named_target(row: dict[str, str], field: str) -> float:
    """Resolve one named residual field, including its documented total-base fallback."""

    raw_value = row.get(field, "").strip()  # Read and normalize the raw explicit residual field.
    if raw_value:  # Prefer an explicit residual value when the CSV already provides one.
        return float(raw_value)  # Convert the chosen residual field into a float.

    pair = TARGET_FALLBACK_PAIRS.get(field)  # Look up the documented total/base fallback pair for this target name.
    if pair is not None:  # Only attempt fallback math for target names that define one.
        total_field, base_field = pair  # Unpack the total and base column names.
        raw_total = row.get(total_field, "").strip()  # Read the raw total-command field.
        raw_base = row.get(base_field, "").strip()  # Read the raw base-command field.
        if raw_total and raw_base:  # Only derive a residual when both ingredients are present.
            return _read_float(row, total_field) - _read_float(row, base_field)  # Recover residual as total minus base.

    raise ValueError(f"row does not contain target field: {field}")  # Explain which named residual target could not be resolved.


def _read_float(row: dict[str, str], field: str, *, default: float | None = None) -> float:
    """Read one numeric field from a CSV row."""

    raw_value = row.get(field)  # Fetch the raw field text.
    if raw_value is None or raw_value == "":  # Treat missing or empty text as absent.
        if default is not None:  # Allow callers to provide a fallback for optional fields.
            return default  # Use the fallback value instead of failing.
        raise ValueError(f"missing numeric field: {field}")  # Explain which field is missing.
    return float(raw_value)  # Convert the raw text into a float.


def _read_flag(row: dict[str, str], field: str, *, default: bool = True) -> bool:
    """Read one boolean-like field from a CSV row."""

    raw_value = row.get(field)  # Fetch the raw field text.
    if raw_value is None or raw_value == "":  # Treat missing or empty text as absent.
        return default  # Fall back to the caller-provided default.

    normalized = raw_value.strip().lower()  # Normalize capitalization and whitespace.
    if normalized in {"1", "true", "yes", "on"}:  # Accept common truthy spellings.
        return True  # Return the boolean value.
    if normalized in {"0", "false", "no", "off"}:  # Accept common falsy spellings.
        return False  # Return the boolean value.
    raise ValueError(f"invalid boolean-like field {field}={raw_value!r}")  # Explain malformed flag values.


def _session_id(row: dict[str, str]) -> str:
    """Return the session identifier used for train/validation splitting."""

    value = row.get("session_id", "").strip()  # Read and normalize the optional session field.
    return value or "default"  # Fall back to a stable default when the field is absent.
