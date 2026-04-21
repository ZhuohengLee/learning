import csv
import json
import shutil
import unittest
from pathlib import Path

from learning.data import DEFAULT_MULTI_AXIS_FEATURE_COLUMNS
from learning.train_axis_models import train_axis_models


class TrainAxisModelsTests(unittest.TestCase):
    def test_train_axis_models_exports_three_bundles(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        temp_path = tests_dir / "_axis_model_case"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            csv_path = temp_path / "telemetry.csv"
            output_dir = temp_path / "models"

            fieldnames = [
                "session_id",
                "timestamp_ms",
                "control_mode",
                "depth_valid",
                "imu_valid",
                "balancing",
                "emergency_stop",
                *DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,
                "u_base",
                "u_total",
                "forward_cmd_base",
                "forward_cmd_total",
                "yaw_cmd_base",
                "yaw_cmd_total",
            ]

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for session_index, session_id in enumerate(["A", "B"]):
                    for step in range(6):
                        writer.writerow(
                            {
                                "session_id": session_id,
                                "timestamp_ms": session_index * 1000 + step * 50,
                                "control_mode": 1,
                                "depth_valid": 1,
                                "imu_valid": 1,
                                "balancing": 0,
                                "emergency_stop": 0,
                                "depth_err_cm": 2.0 - step * 0.2,
                                "depth_speed_cm_s": -0.3 + step * 0.02,
                                "depth_accel_cm_s2": 0.05,
                                "roll_deg": 0.1 * step,
                                "pitch_deg": -0.05 * step,
                                "gyro_x_deg_s": 0.01 * step,
                                "gyro_y_deg_s": 0.02 * step,
                                "gyro_z_deg_s": 0.03 * step,
                                "front_distance_cm": 60.0 - step,
                                "left_distance_cm": 35.0 + step,
                                "right_distance_cm": 32.0 - step * 0.5,
                                "battery_v": 11.8 - session_index * 0.1,
                                "u_base": 78.0 + step,
                                "u_total": 80.0 + step,
                                "forward_cmd_base": 68.0 - step,
                                "forward_cmd_total": 70.0 - step,
                                "yaw_cmd_base": -18.0 + step,
                                "yaw_cmd_total": -20.0 + step,
                            }
                        )

            rows = []
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = [dict(row) for row in reader]

            manifest = train_axis_models(
                rows=rows,
                output_dir=output_dir,
                feature_columns=DEFAULT_MULTI_AXIS_FEATURE_COLUMNS,
                window_size=3,
                hidden_dims=(8, 4),
                epochs=10,
                learning_rate=0.02,
                l2=1e-4,
                val_fraction=0.5,
                max_dt_ms=80.0,
                seed=7,
                print_every=5,
                axis_targets={
                    "depth": "u_residual",
                    "forward": "forward_cmd_residual",
                    "yaw": "yaw_cmd_residual",
                },
            )

            self.assertIn("depth", manifest["axes"])
            self.assertIn("forward", manifest["axes"])
            self.assertIn("yaw", manifest["axes"])

            for axis_name in ("depth", "forward", "yaw"):
                bundle_path = output_dir / f"{axis_name}_model.json"
                self.assertTrue(bundle_path.is_file())
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                self.assertEqual(bundle["metadata"]["axis"], axis_name)
                self.assertEqual(bundle["metadata"]["window_size"], 3)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
