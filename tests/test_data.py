import csv
import tempfile
import unittest
from pathlib import Path

from learning.data import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_WINDOW_SIZE,
    ExampleSet,
    build_examples,
    compute_feature_names,
    fit_standardizer,
    load_control_rows,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "sample_control_telemetry.csv"
)


class DataPipelineTests(unittest.TestCase):
    def test_build_examples_filters_invalid_rows_and_respects_windows(self) -> None:
        rows = load_control_rows(FIXTURE_PATH)
        examples = build_examples(rows, DEFAULT_FEATURE_COLUMNS, window_size=3)

        self.assertIsInstance(examples, ExampleSet)
        self.assertEqual(len(examples.examples), 4)
        self.assertEqual(len(examples.examples[0].features), len(DEFAULT_FEATURE_COLUMNS) * 3)
        self.assertEqual(examples.examples[0].session_id, "A")
        self.assertEqual(examples.examples[-1].session_id, "B")
        self.assertEqual(examples.examples[0].target, 2.0)
        self.assertEqual(examples.examples[-1].target, 0.0)

    def test_compute_feature_names_matches_window_size(self) -> None:
        names = compute_feature_names(["depth_err_cm", "roll_deg"], window_size=3)
        self.assertEqual(
            names,
            [
                "t-2_depth_err_cm",
                "t-2_roll_deg",
                "t-1_depth_err_cm",
                "t-1_roll_deg",
                "t-0_depth_err_cm",
                "t-0_roll_deg",
            ],
        )

    def test_fit_standardizer_produces_round_trip_statistics(self) -> None:
        rows = load_control_rows(FIXTURE_PATH)
        examples = build_examples(rows, DEFAULT_FEATURE_COLUMNS, window_size=3)
        standardizer = fit_standardizer([example.features for example in examples.examples])
        normalized = [standardizer.normalize(example.features) for example in examples.examples]

        first_column_mean = sum(row[0] for row in normalized) / len(normalized)
        self.assertAlmostEqual(first_column_mean, 0.0, places=6)
        restored = standardizer.denormalize(normalized[0])
        self.assertEqual(len(restored), len(examples.examples[0].features))
        for got, want in zip(restored, examples.examples[0].features):
            self.assertAlmostEqual(got, want, places=6)

    def test_build_examples_falls_back_to_u_total_minus_u_base_target(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            suffix=".csv",
            dir=tests_dir,
            delete=False,
        ) as handle:
            csv_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "session_id",
                    "timestamp_ms",
                    "dt_ms",
                    "control_mode",
                    "depth_valid",
                    "imu_valid",
                    "balancing",
                    "emergency_stop",
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
                    "u_base",
                    "u_total",
                ],
            )
            writer.writeheader()
            for index in range(DEFAULT_WINDOW_SIZE):
                writer.writerow(
                    {
                        "session_id": "X",
                        "timestamp_ms": index * 50,
                        "dt_ms": 50,
                        "control_mode": 1,
                        "depth_valid": 1,
                        "imu_valid": 1,
                        "balancing": 0,
                        "emergency_stop": 0,
                        "depth_err_cm": 1.0 - index * 0.1,
                        "depth_speed_cm_s": -0.2,
                        "depth_accel_cm_s2": 0.01,
                        "roll_deg": 0.2,
                        "pitch_deg": -0.1,
                        "gyro_x_deg_s": 0.01,
                        "gyro_y_deg_s": -0.02,
                        "gyro_z_deg_s": 0.03,
                        "battery_v": 11.7,
                        "buoyancy_pwm_applied": 120,
                        "u_base": 100,
                        "u_total": 107,
                    }
                )

        try:
            rows = load_control_rows(csv_path)
            examples = build_examples(
                rows,
                DEFAULT_FEATURE_COLUMNS,
                window_size=DEFAULT_WINDOW_SIZE,
            )
            self.assertEqual(len(examples.examples), 1)
            self.assertEqual(examples.examples[0].target, 7.0)
        finally:
            csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
