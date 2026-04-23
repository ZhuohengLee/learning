"""Optional tests for the PyTorch joint backend.

These tests are skipped automatically when `torch` is not installed, so the shared
data pipeline can still be tested in lightweight environments.
"""

import csv  # Build a synthetic telemetry CSV for the backend tests.
import importlib.util  # Detect optional dependencies without importing them eagerly.
import json  # Read exported metadata sidecars back from disk.
import shutil  # Remove temporary directories after each test.
import sys  # Adjust the import path when the standalone repo is executed directly.
import unittest  # Use the standard-library test framework.
from pathlib import Path  # Build temporary paths safely across platforms.

if __package__ in {None, ""}:  # Detect direct test-module execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Add the parent of the repo root so `learning` can be imported.

from learning.data import DEFAULT_UNIFIED_FEATURE_COLUMNS  # Reuse the shared public joint feature contract.


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None  # Detect whether torch is installed.
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None  # Detect whether ONNX export dependencies are installed.


def _write_minimal_csv(csv_path: Path) -> None:
    """Write one small shared-schema telemetry CSV for joint-backend tests."""

    fieldnames = [
        "session_id",
        "timestamp_ms",
        "control_mode",
        "depth_valid",
        "imu_valid",
        "balancing",
        "emergency_stop",
        *DEFAULT_UNIFIED_FEATURE_COLUMNS,
        "u_total",
        "forward_cmd_total",
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
                        "battery_v": 11.8 - session_index * 0.1,
                        "buoyancy_pwm_applied": 120.0 + step,
                        "front_distance_cm": 60.0 - step,
                        "left_distance_cm": 35.0 + step,
                        "right_distance_cm": 32.0 - step * 0.5,
                        "u_base": 78.0 + step,
                        "forward_cmd_base": float(step % 2),
                        "forward_phase_interval_ms": 1000.0 + session_index * 50.0,
                        "yaw_cmd_base": float((step % 3) - 1),
                        "u_total": 80.0 + step,
                        "forward_cmd_total": float(step % 2),
                        "yaw_cmd_total": float((step % 3) - 1),
                    }
                )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch backend requires torch")
class PytorchBackendTests(unittest.TestCase):
    """Verify that the optional joint PyTorch backend can train compatible bundles."""

    def test_train_model_exports_one_joint_pt_bundle(self) -> None:
        """Verify that the PyTorch backend writes one joint `.pt` bundle with three axes."""

        from learning.train import train_model  # Import lazily so the module is only loaded when torch exists.

        tests_dir = Path(__file__).resolve().parent
        temp_path = tests_dir / "_pytorch_train_case"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            csv_path = temp_path / "telemetry.csv"
            output_dir = temp_path / "models"
            _write_minimal_csv(csv_path)

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]

            manifest = train_model(
                rows=rows,
                output_dir=output_dir,
                feature_columns=DEFAULT_UNIFIED_FEATURE_COLUMNS,
                window_size=3,
                hidden_dims=(8, 4),
                epochs=2,
                learning_rate=1e-3,
                l2=1e-4,
                val_fraction=0.5,
                max_dt_ms=80.0,
                seed=7,
                print_every=1,
                batch_size=4,
                device_name="cpu",
                axis_targets={
                    "depth": "u_residual",
                    "forward": "forward_cmd_residual",
                    "yaw": "yaw_cmd_residual",
                },
            )

            self.assertEqual(manifest["backend"], "pytorch")
            self.assertEqual(manifest["model_type"], "shared_trunk_multi_head_mlp")
            self.assertEqual(manifest["axis_names"], ["depth", "forward", "yaw"])
            self.assertTrue((output_dir / "joint_model.pt").is_file())
            self.assertIn("depth", manifest["axes"])
            self.assertIn("forward", manifest["axes"])
            self.assertIn("yaw", manifest["axes"])
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


class PytorchCalibrationTests(unittest.TestCase):
    """Verify that calibration windows can be rebuilt from the shared telemetry contract."""

    def test_prepare_calibration_inputs_reuses_bundle_feature_contract(self) -> None:
        """Verify that ESP-PPQ calibration inputs are rebuilt with the saved joint feature contract."""

        from learning.export import prepare_calibration_inputs

        tests_dir = Path(__file__).resolve().parent
        temp_path = tests_dir / "_pytorch_calibration_case"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            csv_path = temp_path / "telemetry.csv"
            _write_minimal_csv(csv_path)

            input_dim = len(DEFAULT_UNIFIED_FEATURE_COLUMNS) * 2
            calibration = prepare_calibration_inputs(
                bundle={
                    "metadata": {
                        "backend": "pytorch",
                        "model_type": "shared_trunk_multi_head_mlp",
                        "axis_names": ["depth", "forward", "yaw"],
                        "window_size": 2,
                        "feature_columns": list(DEFAULT_UNIFIED_FEATURE_COLUMNS),
                        "target_columns": {
                            "depth": "u_residual",
                            "forward": "forward_cmd_residual",
                            "yaw": "yaw_cmd_residual",
                        },
                    },
                    "input_standardizer": {
                        "means": [0.0] * input_dim,
                        "stds": [1.0] * input_dim,
                    },
                },
                calibration_csv=csv_path,
                max_dt_ms=80.0,
                limit=4,
            )

            self.assertEqual(calibration["sample_count"], 4)
            self.assertEqual(calibration["input_dim"], input_dim)
            self.assertEqual(calibration["window_size"], 2)
            self.assertEqual(calibration["axis_names"], ["depth", "forward", "yaw"])
            self.assertEqual(calibration["target_columns"]["depth"], "u_residual")
            self.assertEqual(len(calibration["samples"]), 4)
            self.assertEqual(len(calibration["samples"][0]), input_dim)
            self.assertEqual(len(calibration["timestamps_ms"]), 4)
            self.assertEqual(len(calibration["session_ids"]), 4)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


@unittest.skipUnless(TORCH_AVAILABLE and ONNX_AVAILABLE, "PyTorch ONNX export requires torch and onnx")
class PytorchExportTests(unittest.TestCase):
    """Verify that the optional PyTorch exporter can write joint ONNX plus metadata sidecars."""

    def test_export_model_writes_onnx_and_metadata(self) -> None:
        """Verify that a saved joint `.pt` bundle exports to ONNX with a JSON sidecar."""

        import torch  # type: ignore[import-not-found]

        from learning.export import export_model

        tests_dir = Path(__file__).resolve().parent
        temp_path = tests_dir / "_pytorch_export_case"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            input_dim = len(DEFAULT_UNIFIED_FEATURE_COLUMNS) * 2
            bundle_path = temp_path / "joint_model.pt"
            output_path = temp_path / "joint_model.onnx"
            torch.save(
                {
                    "metadata": {
                        "backend": "pytorch",
                        "model_type": "shared_trunk_multi_head_mlp",
                        "axis_names": ["depth", "forward", "yaw"],
                        "window_size": 2,
                        "feature_columns": list(DEFAULT_UNIFIED_FEATURE_COLUMNS),
                        "feature_names": ["f"] * input_dim,
                        "target_columns": {
                            "depth": "u_residual",
                            "forward": "forward_cmd_residual",
                            "yaw": "yaw_cmd_residual",
                        },
                    },
                    "input_standardizer": {
                        "means": [0.0] * input_dim,
                        "stds": [1.0] * input_dim,
                    },
                    "target_standardizer": {"means": [0.0, 0.0, 0.0], "stds": [1.0, 1.0, 1.0]},
                    "model_spec": {
                        "input_dim": input_dim,
                        "hidden_dims": [4, 2],
                        "output_dim": 3,
                        "axis_names": ["depth", "forward", "yaw"],
                        "hidden_activation": "tanh",
                        "output_activation": "linear",
                    },
                    "state_dict": {
                        "trunk.0.weight": torch.zeros(4, input_dim),
                        "trunk.0.bias": torch.zeros(4),
                        "trunk.2.weight": torch.zeros(2, 4),
                        "trunk.2.bias": torch.zeros(2),
                        "heads.axis_depth.weight": torch.zeros(1, 2),
                        "heads.axis_depth.bias": torch.zeros(1),
                        "heads.axis_forward.weight": torch.zeros(1, 2),
                        "heads.axis_forward.bias": torch.zeros(1),
                        "heads.axis_yaw.weight": torch.zeros(1, 2),
                        "heads.axis_yaw.bias": torch.zeros(1),
                    },
                },
                bundle_path,
            )

            export_model(model_path=bundle_path, output_path=output_path, opset=13)

            self.assertTrue(output_path.is_file())
            metadata = json.loads(output_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["backend"], "pytorch")
            self.assertEqual(metadata["metadata"]["model_type"], "shared_trunk_multi_head_mlp")
            self.assertEqual(metadata["metadata"]["axis_names"], ["depth", "forward", "yaw"])
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
