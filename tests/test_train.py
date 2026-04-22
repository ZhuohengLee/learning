"""Tests for the public single-entry three-axis residual trainer.

Reading route:
1. Start with `test_train_models_exports_three_bundles_with_shared_features()`.
2. Then inspect the synthetic telemetry rows to see the minimum shared schema.
3. Then focus on the `train_models(...)` call.
4. Finish with the assertions that verify all three bundles use one shared feature set.
"""

import csv  # Build a synthetic telemetry CSV for the public trainer test.
import json  # Read exported model bundles back from disk.
import shutil  # Remove temporary test directories after the test.
import sys  # Adjust the import path when the standalone repo is executed directly.
import unittest  # Use the standard-library test framework.
from pathlib import Path  # Build temporary paths safely across platforms.

if __package__ in {None, ""}:  # Detect direct test-module execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Add the parent of the repo root so `learning` can be imported.

from learning.data import DEFAULT_UNIFIED_FEATURE_COLUMNS  # Reuse the public trainer's default shared feature set.
from learning.train import train_models  # Import the public single-entry trainer under test.


class TrainModelsTests(unittest.TestCase):  # Group public-trainer tests together.
    """Check that the public single-entry trainer exports three shared-feature bundles."""

    def test_train_models_exports_three_bundles_with_shared_features(self) -> None:
        """Verify that the public trainer exports depth, forward, and yaw with one shared feature list."""

        tests_dir = Path(__file__).resolve().parent  # Place temporary artifacts beside the test file.
        temp_path = tests_dir / "_train_case"  # Use a dedicated temporary directory for this test.
        if temp_path.exists():  # Remove leftovers from previous interrupted runs.
            shutil.rmtree(temp_path, ignore_errors=True)  # Best-effort cleanup before the test starts.
        temp_path.mkdir(parents=True, exist_ok=True)  # Create the temporary directory tree.
        try:  # Ensure temporary artifacts are removed even if assertions fail.
            csv_path = temp_path / "telemetry.csv"  # Name the synthetic telemetry file.
            output_dir = temp_path / "models"  # Name the directory that will receive exported bundles.

            fieldnames = [
                "session_id",  # Session id used by the splitter.
                "timestamp_ms",  # Frame timestamp in milliseconds.
                "control_mode",  # Autonomous/manual control flag.
                "depth_valid",  # Depth sensor validity flag.
                "imu_valid",  # IMU validity flag.
                "balancing",  # Balance-mode flag.
                "emergency_stop",  # Emergency-stop flag.
                *DEFAULT_UNIFIED_FEATURE_COLUMNS,  # Shared public input features used for all three axes.
                "u_total",  # Depth total control output used to recover residuals.
                "forward_cmd_total",  # Forward total control output used to recover residuals.
                "yaw_cmd_total",  # Yaw total control output used to recover residuals.
            ]  # Define the minimal synthetic telemetry schema required by the public trainer.

            with csv_path.open("w", newline="", encoding="utf-8") as handle:  # Open the synthetic CSV for writing.
                writer = csv.DictWriter(handle, fieldnames=fieldnames)  # Build a CSV writer with the chosen schema.
                writer.writeheader()  # Emit the CSV header row.
                for session_index, session_id in enumerate(["A", "B"]):  # Create two sessions so one can land in validation.
                    for step in range(6):  # Create enough rows per session for multiple windows.
                        writer.writerow(
                            {
                                "session_id": session_id,  # Keep rows grouped by synthetic session.
                                "timestamp_ms": session_index * 1000 + step * 50,  # Space frames 50 ms apart per session.
                                "control_mode": 1,  # Mark all rows as trainable non-direct-control frames.
                                "depth_valid": 1,  # Mark depth as valid.
                                "imu_valid": 1,  # Mark IMU as valid.
                                "balancing": 0,  # Keep the row outside balance mode.
                                "emergency_stop": 0,  # Keep the row outside emergency-stop mode.
                                "depth_err_cm": 2.0 - step * 0.2,  # Vary depth error smoothly across the session.
                                "depth_speed_cm_s": -0.3 + step * 0.02,  # Vary depth speed slightly.
                                "depth_accel_cm_s2": 0.05,  # Keep depth acceleration constant for simplicity.
                                "roll_deg": 0.1 * step,  # Vary roll slightly across the session.
                                "pitch_deg": -0.05 * step,  # Vary pitch slightly across the session.
                                "gyro_x_deg_s": 0.01 * step,  # Vary gyro-x slightly across the session.
                                "gyro_y_deg_s": 0.02 * step,  # Vary gyro-y slightly across the session.
                                "gyro_z_deg_s": 0.03 * step,  # Vary gyro-z slightly across the session.
                                "battery_v": 11.8 - session_index * 0.1,  # Use a slightly different battery voltage per session.
                                "buoyancy_pwm_applied": 120.0 + step,  # Include buoyancy history in the shared input state.
                                "front_distance_cm": 60.0 - step,  # Vary front sonar distance slightly.
                                "left_distance_cm": 35.0 + step,  # Vary left sonar distance slightly.
                                "right_distance_cm": 32.0 - step * 0.5,  # Vary right sonar distance slightly.
                                "u_base": 78.0 + step,  # Depth base command used by both features and residual fallback.
                                "forward_cmd_base": 68.0 - step,  # Forward base command used by both features and fallback.
                                "forward_phase_interval_ms": 1000.0 + session_index * 50.0,  # Forward propulsion timing feature.
                                "yaw_cmd_base": -18.0 + step,  # Yaw base command used by both features and fallback.
                                "u_total": 80.0 + step,  # Depth total command so `u_residual` can be recovered.
                                "forward_cmd_total": 70.0 - step,  # Forward total command so residual can be recovered.
                                "yaw_cmd_total": -20.0 + step,  # Yaw total command so residual can be recovered.
                            }
                        )  # Write one synthetic telemetry row.

            with csv_path.open("r", newline="", encoding="utf-8") as handle:  # Reopen the synthetic CSV for reading.
                reader = csv.DictReader(handle)  # Parse rows as dictionaries.
                rows = [dict(row) for row in reader]  # Materialize all rows for the trainer API.

            manifest = train_models(
                rows=rows,  # Pass the synthetic telemetry rows.
                output_dir=output_dir,  # Write exported bundles into the temporary model directory.
                feature_columns=DEFAULT_UNIFIED_FEATURE_COLUMNS,  # Use the public trainer's shared feature list.
                window_size=3,  # Stack three frames per example.
                hidden_dims=(8, 4),  # Use a small test-friendly architecture.
                epochs=10,  # Keep training short so the test runs quickly.
                learning_rate=0.02,  # Use a moderate learning rate.
                l2=1e-4,  # Apply a small amount of L2 regularization.
                val_fraction=0.5,  # Reserve one synthetic session for validation.
                max_dt_ms=80.0,  # Treat 50 ms row spacing as contiguous.
                seed=7,  # Make the split and initialization reproducible.
                print_every=5,  # Print progress midway and at the end.
                axis_targets={
                    "depth": "u_residual",  # Train the depth residual model.
                    "forward": "forward_cmd_residual",  # Train the forward residual model.
                    "yaw": "yaw_cmd_residual",  # Train the yaw residual model.
                },
            )  # Train and export all requested axis models from one entry point.

            self.assertEqual(manifest["feature_columns"], list(DEFAULT_UNIFIED_FEATURE_COLUMNS))  # Manifest should record the shared feature list once.
            self.assertIn("depth", manifest["axes"])  # Manifest should contain a depth entry.
            self.assertIn("forward", manifest["axes"])  # Manifest should contain a forward entry.
            self.assertIn("yaw", manifest["axes"])  # Manifest should contain a yaw entry.

            for axis_name in ("depth", "forward", "yaw"):  # Inspect every expected exported axis bundle.
                bundle_path = output_dir / f"{axis_name}_model.json"  # Build the expected bundle path.
                self.assertTrue(bundle_path.is_file())  # Exported bundle file should exist on disk.
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))  # Load the bundle back from disk.
                self.assertEqual(bundle["metadata"]["axis"], axis_name)  # Bundle metadata should record the correct axis.
                self.assertEqual(bundle["metadata"]["window_size"], 3)  # Bundle metadata should preserve the chosen window size.
                self.assertEqual(bundle["metadata"]["feature_columns"], list(DEFAULT_UNIFIED_FEATURE_COLUMNS))  # All three axes should share one feature order.
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)  # Best-effort cleanup of all temporary artifacts.


if __name__ == "__main__":  # Allow running this file directly for local debugging.
    unittest.main()  # Execute the tests.
