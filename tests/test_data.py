"""Tests for telemetry loading, window building, and normalization.

Reading route:
1. Start with `test_build_examples_filters_invalid_rows_and_respects_windows()`.
2. Then read `test_build_examples_falls_back_to_u_total_minus_u_base_target()`.
3. Then read `test_fit_standardizer_produces_round_trip_statistics()`.
4. Finish with `test_compute_feature_names_matches_window_size()` as the simplest helper check.
"""

import csv  # Build temporary CSV fixtures for edge-case tests.
import sys  # Adjust the import path when the standalone repo is executed directly.
import tempfile  # Create temporary files safely during testing.
import unittest  # Use the standard-library test framework.
from pathlib import Path  # Build fixture paths safely across platforms.

if __package__ in {None, ""}:  # Detect direct test-module execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Add the parent of the repo root so `learning` can be imported.

from learning.data import (  # Import the data-pipeline helpers under test.
    DEFAULT_FEATURE_COLUMNS,  # Default depth-training feature set.
    DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,  # Default shared multi-axis feature set.
    DEFAULT_WINDOW_SIZE,  # Default stacked-window length.
    ExampleSet,  # Dataset wrapper type returned by `build_examples`.
    build_examples,  # Convert raw rows into supervised examples.
    compute_feature_names,  # Expand feature names across a time window.
    fit_standardizer,  # Fit normalization statistics.
    load_control_rows,  # Load telemetry rows from CSV.
)


FIXTURE_PATH = (  # Point to the checked-in sample telemetry CSV.
    Path(__file__).resolve().parent / "fixtures" / "sample_control_telemetry.csv"
)


class DataPipelineTests(unittest.TestCase):  # Group data-pipeline tests together.
    """Verify the training dataset pipeline on both fixture and synthetic data."""

    def test_build_examples_filters_invalid_rows_and_respects_windows(self) -> None:
        """Verify that invalid rows are skipped and windows are built correctly."""

        rows = load_control_rows(FIXTURE_PATH)  # Load the checked-in telemetry fixture.
        examples = build_examples(rows, DEFAULT_FEATURE_COLUMNS, window_size=3)  # Build 3-frame training examples.

        self.assertIsInstance(examples, ExampleSet)  # Result should be wrapped in `ExampleSet`.
        self.assertEqual(len(examples.examples), 4)  # Fixture should yield four valid windowed examples.
        self.assertEqual(len(examples.examples[0].features), len(DEFAULT_FEATURE_COLUMNS) * 3)  # Feature length should equal feature-count times window-size.
        self.assertEqual(examples.examples[0].session_id, "A")  # First valid example should come from session A.
        self.assertEqual(examples.examples[-1].session_id, "B")  # Last valid example should come from session B.
        self.assertEqual(examples.examples[0].target, 2.0)  # First example should keep the expected residual target.
        self.assertEqual(examples.examples[-1].target, 0.0)  # Last example should keep the expected residual target.

    def test_compute_feature_names_matches_window_size(self) -> None:
        """Verify that flattened feature names preserve lag ordering."""

        names = compute_feature_names(["depth_err_cm", "roll_deg"], window_size=3)  # Expand two base features over three lags.
        self.assertEqual(
            names,  # Compare against the exact expected flattened ordering.
            [
                "t-2_depth_err_cm",  # Oldest depth-error feature.
                "t-2_roll_deg",  # Oldest roll feature.
                "t-1_depth_err_cm",  # Middle depth-error feature.
                "t-1_roll_deg",  # Middle roll feature.
                "t-0_depth_err_cm",  # Newest depth-error feature.
                "t-0_roll_deg",  # Newest roll feature.
            ],
        )

    def test_fit_standardizer_produces_round_trip_statistics(self) -> None:
        """Verify that normalization and denormalization are self-consistent."""

        rows = load_control_rows(FIXTURE_PATH)  # Load the checked-in telemetry fixture.
        examples = build_examples(rows, DEFAULT_FEATURE_COLUMNS, window_size=3)  # Build 3-frame training examples.
        standardizer = fit_standardizer([example.features for example in examples.examples])  # Fit feature normalization.
        normalized = [standardizer.normalize(example.features) for example in examples.examples]  # Normalize every example.

        first_column_mean = sum(row[0] for row in normalized) / len(normalized)  # Measure the normalized first-column mean.
        self.assertAlmostEqual(first_column_mean, 0.0, places=6)  # Normalized features should have near-zero mean.
        restored = standardizer.denormalize(normalized[0])  # Map one normalized sample back to original units.
        self.assertEqual(len(restored), len(examples.examples[0].features))  # Round-trip should preserve feature length.
        for got, want in zip(restored, examples.examples[0].features):  # Compare each restored value against the original.
            self.assertAlmostEqual(got, want, places=6)  # Round-trip error should be numerically tiny.

    def test_build_examples_falls_back_to_u_total_minus_u_base_target(self) -> None:
        """Verify that residual target fallback uses `u_total - u_base`."""

        tests_dir = Path(__file__).resolve().parent  # Place the temporary file near the test folder.
        with tempfile.NamedTemporaryFile(
            mode="w",  # Open the file in text-write mode.
            newline="",  # Let the CSV writer manage newline handling.
            encoding="utf-8",  # Use UTF-8 for deterministic text output.
            suffix=".csv",  # Give the temporary file a CSV suffix.
            dir=tests_dir,  # Keep the file inside the test directory.
            delete=False,  # Keep the file after close so the loader can reopen it.
        ) as handle:
            csv_path = Path(handle.name)  # Capture the generated temporary file path.
            writer = csv.DictWriter(
                handle,  # Write rows into the open temporary file.
                fieldnames=[
                    "session_id",  # Session id used by the splitter.
                    "timestamp_ms",  # Frame timestamp in milliseconds.
                    "dt_ms",  # Delta time between frames.
                    "control_mode",  # Autonomous/manual control flag.
                    "depth_valid",  # Depth sensor validity flag.
                    "imu_valid",  # IMU validity flag.
                    "balancing",  # Balance-mode flag.
                    "emergency_stop",  # Emergency-stop flag.
                    "depth_err_cm",  # Depth-error feature.
                    "depth_speed_cm_s",  # Depth-speed feature.
                    "depth_accel_cm_s2",  # Depth-acceleration feature.
                    "roll_deg",  # Roll feature.
                    "pitch_deg",  # Pitch feature.
                    "gyro_x_deg_s",  # Gyro-x feature.
                    "gyro_y_deg_s",  # Gyro-y feature.
                    "gyro_z_deg_s",  # Gyro-z feature.
                    "battery_v",  # Battery-voltage feature.
                    "buoyancy_pwm_applied",  # Previous buoyancy PWM feature.
                    "u_base",  # Base control output.
                    "u_total",  # Total control output after residual trim.
                ],
            )  # Build a CSV writer with the minimal required fields.
            writer.writeheader()  # Emit the CSV header row.
            for index in range(DEFAULT_WINDOW_SIZE):  # Write exactly one full window of valid rows.
                writer.writerow(
                    {
                        "session_id": "X",  # Keep all rows in one synthetic session.
                        "timestamp_ms": index * 50,  # Space frames 50 ms apart.
                        "dt_ms": 50,  # Record the frame delta in milliseconds.
                        "control_mode": 1,  # Mark the row as autonomous.
                        "depth_valid": 1,  # Mark depth as valid.
                        "imu_valid": 1,  # Mark IMU as valid.
                        "balancing": 0,  # Keep the row outside balance mode.
                        "emergency_stop": 0,  # Keep the row outside emergency-stop mode.
                        "depth_err_cm": 1.0 - index * 0.1,  # Vary depth error slightly across the window.
                        "depth_speed_cm_s": -0.2,  # Keep depth speed constant for simplicity.
                        "depth_accel_cm_s2": 0.01,  # Keep depth acceleration constant for simplicity.
                        "roll_deg": 0.2,  # Keep roll constant for simplicity.
                        "pitch_deg": -0.1,  # Keep pitch constant for simplicity.
                        "gyro_x_deg_s": 0.01,  # Keep gyro-x constant for simplicity.
                        "gyro_y_deg_s": -0.02,  # Keep gyro-y constant for simplicity.
                        "gyro_z_deg_s": 0.03,  # Keep gyro-z constant for simplicity.
                        "battery_v": 11.7,  # Keep battery voltage constant for simplicity.
                        "buoyancy_pwm_applied": 120,  # Keep previous buoyancy PWM constant for simplicity.
                        "u_base": 100,  # Use a fixed base control output.
                        "u_total": 107,  # Use a fixed total control output.
                    }
                )  # Write one synthetic telemetry row.

        try:  # Ensure the temporary file is cleaned up even if assertions fail.
            rows = load_control_rows(csv_path)  # Reload the synthetic CSV through the normal loader.
            examples = build_examples(
                rows,  # Pass the synthetic telemetry rows.
                DEFAULT_FEATURE_COLUMNS,  # Use the default depth feature set.
                window_size=DEFAULT_WINDOW_SIZE,  # Build exactly one full-size window.
            )  # Let the pipeline infer residual targets automatically.
            self.assertEqual(len(examples.examples), 1)  # Exactly one full window should produce one example.
            self.assertEqual(examples.examples[0].target, 7.0)  # Residual target should equal `u_total - u_base`.
        finally:
            csv_path.unlink(missing_ok=True)  # Delete the temporary CSV regardless of test outcome.

    def test_build_examples_supports_axis_specific_total_minus_base_fallback(self) -> None:
        """Verify that axis-specific targets can be derived from matching total/base pairs."""

        tests_dir = Path(__file__).resolve().parent  # Place the temporary file near the test folder.
        with tempfile.NamedTemporaryFile(
            mode="w",  # Open the file in text-write mode.
            newline="",  # Let the CSV writer manage newline handling.
            encoding="utf-8",  # Use UTF-8 for deterministic text output.
            suffix=".csv",  # Give the temporary file a CSV suffix.
            dir=tests_dir,  # Keep the file inside the test directory.
            delete=False,  # Keep the file after close so the loader can reopen it.
        ) as handle:
            csv_path = Path(handle.name)  # Capture the generated temporary file path.
            writer = csv.DictWriter(
                handle,  # Write rows into the open temporary file.
                fieldnames=[
                    "session_id",  # Session id used by the splitter.
                    "timestamp_ms",  # Frame timestamp in milliseconds.
                    "control_mode",  # Autonomous/manual control flag.
                    "depth_valid",  # Depth sensor validity flag.
                    "imu_valid",  # IMU validity flag.
                    "balancing",  # Balance-mode flag.
                    "emergency_stop",  # Emergency-stop flag.
                    *DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,  # Shared multi-axis input features.
                    "forward_cmd_total",  # Forward total control output after residual trim.
                ],
            )  # Build a CSV writer with the minimal required fields.
            writer.writeheader()  # Emit the CSV header row.
            for index in range(DEFAULT_WINDOW_SIZE):  # Write exactly one full window of valid rows.
                writer.writerow(
                    {
                        "session_id": "Y",  # Keep all rows in one synthetic session.
                        "timestamp_ms": index * 50,  # Space frames 50 ms apart.
                        "control_mode": 1,  # Mark the row as autonomous.
                        "depth_valid": 1,  # Mark depth as valid.
                        "imu_valid": 1,  # Mark IMU as valid.
                        "balancing": 0,  # Keep the row outside balance mode.
                        "emergency_stop": 0,  # Keep the row outside emergency-stop mode.
                        "depth_err_cm": 0.5 + index * 0.1,  # Vary depth error slightly across the window.
                        "depth_speed_cm_s": -0.1,  # Keep depth speed constant for simplicity.
                        "depth_accel_cm_s2": 0.02,  # Keep depth acceleration constant for simplicity.
                        "roll_deg": 0.0,  # Keep roll constant for simplicity.
                        "pitch_deg": 0.0,  # Keep pitch constant for simplicity.
                        "gyro_x_deg_s": 0.01,  # Keep gyro-x constant for simplicity.
                        "gyro_y_deg_s": 0.02,  # Keep gyro-y constant for simplicity.
                        "gyro_z_deg_s": 0.03,  # Keep gyro-z constant for simplicity.
                        "front_distance_cm": 60.0,  # Keep front sonar constant for simplicity.
                        "left_distance_cm": 40.0,  # Keep left sonar constant for simplicity.
                        "right_distance_cm": 38.0,  # Keep right sonar constant for simplicity.
                        "battery_v": 11.8,  # Keep battery voltage constant for simplicity.
                        "u_base": 80.0,  # Depth base command required by the shared feature contract.
                        "forward_cmd_base": 25.0,  # Forward base command used for fallback residual recovery.
                        "forward_phase_interval_ms": 1000.0,  # Forward phase interval included in the shared feature contract.
                        "yaw_cmd_base": -5.0,  # Yaw base command required by the shared feature contract.
                        "forward_cmd_total": 29.0,  # Forward total command so fallback should recover a residual of 4.
                    }
                )  # Write one synthetic telemetry row.

        try:  # Ensure the temporary file is cleaned up even if assertions fail.
            rows = load_control_rows(csv_path)  # Reload the synthetic CSV through the normal loader.
            examples = build_examples(
                rows,  # Pass the synthetic telemetry rows.
                DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,  # Use the default multi-axis feature set.
                window_size=DEFAULT_WINDOW_SIZE,  # Build exactly one full-size window.
                target_column="forward_cmd_residual",  # Request the forward residual target explicitly.
            )  # Let the pipeline derive the forward residual from total minus base.
            self.assertEqual(len(examples.examples), 1)  # Exactly one full window should produce one example.
            self.assertEqual(examples.examples[0].target, 4.0)  # Forward residual should equal `forward_cmd_total - forward_cmd_base`.
        finally:
            csv_path.unlink(missing_ok=True)  # Delete the temporary CSV regardless of test outcome.

    def test_build_examples_keeps_manual_forward_frames_for_forward_training(self) -> None:
        """Verify that manual forward/yaw task frames are still trainable for axis models."""

        tests_dir = Path(__file__).resolve().parent  # Place the temporary file near the test folder.
        with tempfile.NamedTemporaryFile(
            mode="w",  # Open the file in text-write mode.
            newline="",  # Let the CSV writer manage newline handling.
            encoding="utf-8",  # Use UTF-8 for deterministic text output.
            suffix=".csv",  # Give the temporary file a CSV suffix.
            dir=tests_dir,  # Keep the file inside the test directory.
            delete=False,  # Keep the file after close so the loader can reopen it.
        ) as handle:
            csv_path = Path(handle.name)  # Capture the generated temporary file path.
            writer = csv.DictWriter(
                handle,  # Write rows into the open temporary file.
                fieldnames=[
                    "session_id",  # Session id used by the splitter.
                    "timestamp_ms",  # Frame timestamp in milliseconds.
                    "control_mode",  # Autonomous/manual control flag.
                    "depth_valid",  # Depth sensor validity flag.
                    "imu_valid",  # IMU validity flag.
                    "balancing",  # Balance-mode flag.
                    "emergency_stop",  # Emergency-stop flag.
                    "buoyancy_dir_applied",  # Applied buoyancy direction for direct-manual filtering.
                    "buoyancy_pwm_applied",  # Applied buoyancy PWM for direct-manual filtering.
                    "u_total",  # Exported depth command used by the direct-manual filter.
                    "u_residual",  # Exported depth residual used by the direct-manual filter.
                    *DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,  # Shared multi-axis input features.
                    "forward_cmd_total",  # Forward total control output after residual trim.
                ],
            )  # Build a CSV writer with the minimal required fields.
            writer.writeheader()  # Emit the CSV header row.
            for index in range(DEFAULT_WINDOW_SIZE):  # Write exactly one full window of valid rows.
                writer.writerow(
                    {
                        "session_id": "M",  # Keep all rows in one synthetic session.
                        "timestamp_ms": index * 50,  # Space frames 50 ms apart.
                        "control_mode": 0,  # Mark the row as manual so this test exercises the new rule.
                        "depth_valid": 1,  # Mark depth as valid.
                        "imu_valid": 1,  # Mark IMU as valid.
                        "balancing": 0,  # Keep the row outside balance mode.
                        "emergency_stop": 0,  # Keep the row outside emergency-stop mode.
                        "buoyancy_dir_applied": 0,  # No direct vertical override is active.
                        "buoyancy_pwm_applied": 0,  # No buoyancy pump output is active.
                        "u_total": 0.0,  # Depth output stays idle during manual forward testing.
                        "u_residual": 0.0,  # Depth residual stays idle during manual forward testing.
                        "depth_err_cm": 0.2 + index * 0.1,  # Vary depth error slightly across the window.
                        "depth_speed_cm_s": -0.05,  # Keep depth speed constant for simplicity.
                        "depth_accel_cm_s2": 0.01,  # Keep depth acceleration constant for simplicity.
                        "roll_deg": 0.0,  # Keep roll constant for simplicity.
                        "pitch_deg": 0.0,  # Keep pitch constant for simplicity.
                        "gyro_x_deg_s": 0.01,  # Keep gyro-x constant for simplicity.
                        "gyro_y_deg_s": 0.02,  # Keep gyro-y constant for simplicity.
                        "gyro_z_deg_s": 0.03,  # Keep gyro-z constant for simplicity.
                        "front_distance_cm": 70.0,  # Keep front sonar constant for simplicity.
                        "left_distance_cm": 45.0,  # Keep left sonar constant for simplicity.
                        "right_distance_cm": 40.0,  # Keep right sonar constant for simplicity.
                        "battery_v": 11.9,  # Keep battery voltage constant for simplicity.
                        "u_base": 0.0,  # Depth base controller is idle in manual forward mode.
                        "forward_cmd_base": 60.0,  # Forward base command remains active in manual mode.
                        "forward_phase_interval_ms": 900.0 + index * 20.0,  # Vary the forward phase interval across the window.
                        "yaw_cmd_base": 15.0,  # Keep yaw base command fixed as described by the user.
                        "forward_cmd_total": 64.0,  # Forward total command so fallback should recover a residual of 4.
                    }
                )  # Write one synthetic telemetry row.

        try:  # Ensure the temporary file is cleaned up even if assertions fail.
            rows = load_control_rows(csv_path)  # Reload the synthetic CSV through the normal loader.
            examples = build_examples(
                rows,  # Pass the synthetic telemetry rows.
                DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,  # Use the default multi-axis feature set.
                window_size=DEFAULT_WINDOW_SIZE,  # Build exactly one full-size window.
                target_column="forward_cmd_residual",  # Request the forward residual target explicitly.
            )  # Let the pipeline derive the forward residual from total minus base.
            self.assertEqual(len(examples.examples), 1)  # Manual forward frames should remain trainable.
            self.assertEqual(examples.examples[0].target, 4.0)  # Forward residual should equal `forward_cmd_total - forward_cmd_base`.
        finally:
            csv_path.unlink(missing_ok=True)  # Delete the temporary CSV regardless of test outcome.

    def test_build_examples_keeps_manual_depth_target_frames_for_depth_training(self) -> None:
        """Verify that manual `L50`-style depth-target frames remain trainable."""

        tests_dir = Path(__file__).resolve().parent  # Place the temporary file near the test folder.
        with tempfile.NamedTemporaryFile(
            mode="w",  # Open the file in text-write mode.
            newline="",  # Let the CSV writer manage newline handling.
            encoding="utf-8",  # Use UTF-8 for deterministic text output.
            suffix=".csv",  # Give the temporary file a CSV suffix.
            dir=tests_dir,  # Keep the file inside the test directory.
            delete=False,  # Keep the file after close so the loader can reopen it.
        ) as handle:
            csv_path = Path(handle.name)  # Capture the generated temporary file path.
            writer = csv.DictWriter(
                handle,  # Write rows into the open temporary file.
                fieldnames=[
                    "session_id",  # Session id used by the splitter.
                    "timestamp_ms",  # Frame timestamp in milliseconds.
                    "dt_ms",  # Delta time between frames.
                    "control_mode",  # Autonomous/manual control flag.
                    "depth_valid",  # Depth sensor validity flag.
                    "imu_valid",  # IMU validity flag.
                    "balancing",  # Balance-mode flag.
                    "emergency_stop",  # Emergency-stop flag.
                    "buoyancy_dir_applied",  # Applied buoyancy direction for direct-manual filtering.
                    "u_residual",  # Explicit depth residual target.
                    *DEFAULT_FEATURE_COLUMNS,  # Default depth-training input features.
                    "u_base",  # Base control output.
                    "u_total",  # Total control output after residual trim.
                ],
            )  # Build a CSV writer with the minimal required fields.
            writer.writeheader()  # Emit the CSV header row.
            for index in range(DEFAULT_WINDOW_SIZE):  # Write exactly one full window of valid rows.
                writer.writerow(
                    {
                        "session_id": "L",  # Keep all rows in one synthetic session.
                        "timestamp_ms": index * 50,  # Space frames 50 ms apart.
                        "dt_ms": 50,  # Record the frame delta in milliseconds.
                        "control_mode": 0,  # Mark the row as manual to match `L50` issued in manual mode.
                        "depth_valid": 1,  # Mark depth as valid.
                        "imu_valid": 1,  # Mark IMU as valid.
                        "balancing": 0,  # Keep the row outside balance mode.
                        "emergency_stop": 0,  # Keep the row outside emergency-stop mode.
                        "buoyancy_dir_applied": 2,  # Depth controller is actively descending toward the target.
                        "u_residual": 5.0,  # Residual target stays explicit for this synthetic case.
                        "depth_err_cm": 10.0 - index,  # Vary the depth error across the approach to the target.
                        "depth_speed_cm_s": -0.2,  # Keep depth speed constant for simplicity.
                        "depth_accel_cm_s2": 0.03,  # Keep depth acceleration constant for simplicity.
                        "roll_deg": 0.1,  # Keep roll constant for simplicity.
                        "pitch_deg": -0.1,  # Keep pitch constant for simplicity.
                        "gyro_x_deg_s": 0.01,  # Keep gyro-x constant for simplicity.
                        "gyro_y_deg_s": 0.02,  # Keep gyro-y constant for simplicity.
                        "gyro_z_deg_s": 0.03,  # Keep gyro-z constant for simplicity.
                        "battery_v": 11.8,  # Keep battery voltage constant for simplicity.
                        "buoyancy_pwm_applied": 140,  # Closed-loop depth control is active but not in `j/k` full-manual mode.
                        "u_base": 90.0,  # Base controller contributes the dominant effort.
                        "u_total": 95.0,  # Total command equals base plus residual trim.
                    }
                )  # Write one synthetic telemetry row.

        try:  # Ensure the temporary file is cleaned up even if assertions fail.
            rows = load_control_rows(csv_path)  # Reload the synthetic CSV through the normal loader.
            examples = build_examples(
                rows,  # Pass the synthetic telemetry rows.
                DEFAULT_FEATURE_COLUMNS,  # Use the default depth feature set.
                window_size=DEFAULT_WINDOW_SIZE,  # Build exactly one full-size window.
                target_column="u_residual",  # Request the depth residual target explicitly.
            )  # Let the pipeline keep manual target-hold depth frames.
            self.assertEqual(len(examples.examples), 1)  # Manual `L50`-style frames should remain trainable.
            self.assertEqual(examples.examples[0].target, 5.0)  # The explicit residual target should survive unchanged.
        finally:
            csv_path.unlink(missing_ok=True)  # Delete the temporary CSV regardless of test outcome.

    def test_build_examples_filters_direct_manual_buoyancy_override_frames(self) -> None:
        """Verify that only the explicit `j/k` buoyancy override signature is rejected."""

        tests_dir = Path(__file__).resolve().parent  # Place the temporary file near the test folder.
        with tempfile.NamedTemporaryFile(
            mode="w",  # Open the file in text-write mode.
            newline="",  # Let the CSV writer manage newline handling.
            encoding="utf-8",  # Use UTF-8 for deterministic text output.
            suffix=".csv",  # Give the temporary file a CSV suffix.
            dir=tests_dir,  # Keep the file inside the test directory.
            delete=False,  # Keep the file after close so the loader can reopen it.
        ) as handle:
            csv_path = Path(handle.name)  # Capture the generated temporary file path.
            writer = csv.DictWriter(
                handle,  # Write rows into the open temporary file.
                fieldnames=[
                    "session_id",  # Session id used by the splitter.
                    "timestamp_ms",  # Frame timestamp in milliseconds.
                    "dt_ms",  # Delta time between frames.
                    "control_mode",  # Autonomous/manual control flag.
                    "depth_valid",  # Depth sensor validity flag.
                    "imu_valid",  # IMU validity flag.
                    "balancing",  # Balance-mode flag.
                    "emergency_stop",  # Emergency-stop flag.
                    "buoyancy_dir_applied",  # Applied buoyancy direction for direct-manual filtering.
                    "buoyancy_pwm_applied",  # Applied buoyancy PWM for direct-manual filtering.
                    "u_residual",  # Explicit depth residual used by the direct-manual filter.
                    *DEFAULT_FEATURE_COLUMNS,  # Default depth-training input features.
                    "u_base",  # Base control output.
                    "u_total",  # Total control output after residual trim.
                ],
            )  # Build a CSV writer with the minimal required fields.
            writer.writeheader()  # Emit the CSV header row.
            for index in range(DEFAULT_WINDOW_SIZE):  # Write exactly one full window of synthetic rows.
                writer.writerow(
                    {
                        "session_id": "J",  # Keep all rows in one synthetic session.
                        "timestamp_ms": index * 50,  # Space frames 50 ms apart.
                        "dt_ms": 50,  # Record the frame delta in milliseconds.
                        "control_mode": 0,  # Mark the row as manual so the direct override signature is meaningful.
                        "depth_valid": 1,  # Mark depth as valid.
                        "imu_valid": 1,  # Mark IMU as valid.
                        "balancing": 0,  # Keep the row outside balance mode.
                        "emergency_stop": 0,  # Keep the row outside emergency-stop mode.
                        "buoyancy_dir_applied": 1,  # Match the manual ascend signature.
                        "buoyancy_pwm_applied": 255,  # Match the hard manual buoyancy PWM.
                        "u_residual": 0.0,  # Manual overrides bypass residual compensation.
                        "depth_err_cm": 1.0,  # Keep depth error constant for simplicity.
                        "depth_speed_cm_s": 0.0,  # Keep depth speed constant for simplicity.
                        "depth_accel_cm_s2": 0.0,  # Keep depth acceleration constant for simplicity.
                        "roll_deg": 0.0,  # Keep roll constant for simplicity.
                        "pitch_deg": 0.0,  # Keep pitch constant for simplicity.
                        "gyro_x_deg_s": 0.0,  # Keep gyro-x constant for simplicity.
                        "gyro_y_deg_s": 0.0,  # Keep gyro-y constant for simplicity.
                        "gyro_z_deg_s": 0.0,  # Keep gyro-z constant for simplicity.
                        "battery_v": 11.7,  # Keep battery voltage constant for simplicity.
                        "u_base": 0.0,  # Manual overrides bypass the base controller.
                        "u_total": -100.0,  # Match the exported manual ascend command.
                    }
                )  # Write one synthetic telemetry row.

        try:  # Ensure the temporary file is cleaned up even if assertions fail.
            rows = load_control_rows(csv_path)  # Reload the synthetic CSV through the normal loader.
            examples = build_examples(
                rows,  # Pass the synthetic telemetry rows.
                DEFAULT_FEATURE_COLUMNS,  # Use the default depth feature set.
                window_size=DEFAULT_WINDOW_SIZE,  # Attempt to build one full-size window.
            )  # Let the pipeline auto-resolve the depth residual target.
            self.assertEqual(len(examples.examples), 0)  # Direct `j/k` override frames should be rejected entirely.
        finally:
            csv_path.unlink(missing_ok=True)  # Delete the temporary CSV regardless of test outcome.


if __name__ == "__main__":  # Allow running this file directly for local debugging.
    unittest.main()  # Execute the tests.
