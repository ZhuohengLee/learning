"""Tests for converting a trained bundle into an ESP32 header.

Reading route:
1. Start with `test_render_header_contains_expected_dimensions()` to see the happy-path export contract.
2. Then read `test_render_header_accepts_shared_three_axis_contract()` to see the unified three-axis export path.
3. Finally read `test_render_header_rejects_feature_order_mismatch()` to see the strict safety check.
"""

import json  # Load the sample model bundle from disk.
import shutil  # Remove temporary export directories after the test.
import sys  # Adjust the import path when the standalone repo is executed directly.
import unittest  # Use the standard-library test framework.
from pathlib import Path  # Build fixture paths safely across platforms.

if __package__ in {None, ""}:  # Detect direct test-module execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Add the parent of the repo root so `learning` can be imported.

from learning.data import DEFAULT_UNIFIED_FEATURE_COLUMNS  # Reuse the shared public feature contract.
from learning.export_axis_models_to_esp32 import export_axis_models  # Import the three-axis batch exporter under test.
from learning.export_to_esp32 import render_header  # Import the exporter under test.


SAMPLE_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "sample_model.json"  # Reuse the checked-in sample bundle.


def _build_shared_payload(*, axis_name: str) -> dict[str, object]:
    """Build one minimal shared-feature bundle for exporter tests."""

    feature_columns = list(DEFAULT_UNIFIED_FEATURE_COLUMNS)  # Reuse the exact shared feature order used by the public trainer.
    input_dim = len(feature_columns) * 2  # Use a small two-frame window for the synthetic payload.
    return {
        "metadata": {
            "axis": axis_name,  # Tag the payload with its logical control axis.
            "feature_columns": feature_columns,  # Export the shared three-axis feature order.
            "window_size": 2,  # Use a short test-friendly window size.
        },
        "input_standardizer": {
            "means": [0.0] * input_dim,  # Use identity normalization for the synthetic payload.
            "stds": [1.0] * input_dim,  # Use identity normalization for the synthetic payload.
        },
        "target_standardizer": {
            "means": [0.0],  # Keep target denormalization trivial.
            "stds": [1.0],  # Keep target denormalization trivial.
        },
        "model": {
            "input_dim": input_dim,  # Match the flattened shared feature width.
            "hidden_dims": [4, 2],  # Use a tiny two-hidden-layer network.
            "layers": [
                {"weights": [[0.0] * input_dim for _ in range(4)], "biases": [0.0] * 4},  # First hidden layer.
                {"weights": [[0.0] * 4 for _ in range(2)], "biases": [0.0] * 2},  # Second hidden layer.
                {"weights": [[0.0] * 2], "biases": [0.0]},  # Output layer.
            ],
        },
    }  # Return a minimal but valid shared-feature payload.


class ExportToEsp32Tests(unittest.TestCase):  # Group header-export tests together.
    """Verify that header generation keeps the firmware contract intact."""

    def test_render_header_contains_expected_dimensions(self) -> None:
        """Verify that a valid bundle renders the expected C++ constants."""

        payload = json.loads(SAMPLE_MODEL_PATH.read_text(encoding="utf-8"))  # Load the sample model bundle.
        header = render_header(payload=payload, namespace="test_model", include_guard="TEST_MODEL_H")  # Render the bundle into header text.

        self.assertIn("#ifndef TEST_MODEL_H", header)  # Header should start with the requested include guard.
        self.assertIn("namespace test_model {", header)  # Header should open the requested namespace.
        self.assertIn("constexpr bool kModelAvailable = true;", header)  # Header should mark the model as present.
        self.assertIn("constexpr uint8_t kWindowSize = 3;", header)  # Header should export the window size.
        self.assertIn("constexpr uint8_t kBaseFeatureCount = 10;", header)  # Header should export the base feature count.
        self.assertIn("constexpr uint16_t kInputDim = 30;", header)  # Header should export the flattened input width.
        self.assertIn("constexpr uint8_t kHidden1Size = 8;", header)  # Header should export the first hidden-layer width.
        self.assertIn("constexpr uint8_t kHidden2Size = 4;", header)  # Header should export the second hidden-layer width.
        self.assertIn('#include "ResidualModelSpec.h"', header)  # Header should depend on the generic runtime model-view struct.
        self.assertIn("// Feature contract: depth_legacy", header)  # Header should record which firmware contract it targets.
        self.assertIn("// Feature order: depth_err_cm, depth_speed_cm_s", header)  # Header should document feature order.
        self.assertIn("inline ResidualModelView getModelView()", header)  # Header should expose the generic runtime view helper.
        self.assertIn("constexpr float kLayer2Biases[1]", header)  # Header should contain output-layer biases.

    def test_render_header_accepts_shared_three_axis_contract(self) -> None:
        """Verify that shared three-axis feature bundles export cleanly for firmware."""

        payload = _build_shared_payload(axis_name="forward")  # Build a minimal but valid shared-feature payload.

        header = render_header(payload=payload, namespace="shared_axis_model", include_guard="SHARED_AXIS_MODEL_H")  # Render the synthetic bundle into a header.

        self.assertIn("// Feature contract: shared_three_axis", header)  # Header should identify the shared three-axis contract.
        self.assertIn("constexpr uint8_t kBaseFeatureCount = 17;", header)  # Header should export the shared base feature count.
        self.assertIn("buoyancy_pwm_applied, front_distance_cm", header)  # Header should preserve the unified feature order.

    def test_export_axis_models_writes_three_headers(self) -> None:
        """Verify that the batch exporter writes depth, forward, and yaw headers together."""

        tests_dir = Path(__file__).resolve().parent  # Place temporary artifacts beside the test file.
        temp_path = tests_dir / "_axis_export_case"  # Use a dedicated temporary directory for this test.
        if temp_path.exists():  # Remove leftovers from previous interrupted runs.
            shutil.rmtree(temp_path, ignore_errors=True)  # Best-effort cleanup before the test starts.
        temp_path.mkdir(parents=True, exist_ok=True)  # Create the temporary directory tree.
        try:  # Ensure temporary artifacts are removed even if assertions fail.
            model_dir = temp_path / "models"  # Store synthetic axis bundles here.
            output_dir = temp_path / "headers"  # Store generated headers here.
            model_dir.mkdir(parents=True, exist_ok=True)  # Create the synthetic model directory.

            for axis_name in ("depth", "forward", "yaw"):  # Create one synthetic bundle per control axis.
                bundle_path = model_dir / f"{axis_name}_model.json"  # Build the bundle path for this axis.
                bundle_path.write_text(json.dumps(_build_shared_payload(axis_name=axis_name)), encoding="utf-8")  # Save the synthetic bundle.

            export_axis_models(model_dir=model_dir, output_dir=output_dir)  # Export all three synthetic bundles into headers.

            self.assertTrue((output_dir / "DepthResidualModelData.h").is_file())  # Depth header should be created.
            self.assertTrue((output_dir / "ForwardResidualModelData.h").is_file())  # Forward header should be created.
            self.assertTrue((output_dir / "YawResidualModelData.h").is_file())  # Yaw header should be created.
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)  # Best-effort cleanup of all temporary artifacts.

    def test_render_header_rejects_feature_order_mismatch(self) -> None:
        """Verify that mismatched feature order is rejected before export."""

        payload = json.loads(SAMPLE_MODEL_PATH.read_text(encoding="utf-8"))  # Load the sample model bundle.
        payload["metadata"]["feature_columns"] = [  # Deliberately scramble the first two features.
            "depth_speed_cm_s",  # Move speed ahead of error to break the contract.
            "depth_err_cm",  # Move error behind speed to break the contract.
            "depth_accel_cm_s2",  # Keep the rest unchanged.
            "roll_deg",  # Keep the rest unchanged.
            "pitch_deg",  # Keep the rest unchanged.
            "gyro_x_deg_s",  # Keep the rest unchanged.
            "gyro_y_deg_s",  # Keep the rest unchanged.
            "gyro_z_deg_s",  # Keep the rest unchanged.
            "battery_v",  # Keep the rest unchanged.
            "buoyancy_pwm_applied",  # Keep the rest unchanged.
        ]

        with self.assertRaisesRegex(ValueError, "feature_columns must match one of the supported ESP32 inference contracts"):  # Expect a strict contract error.
            render_header(payload=payload)  # Try exporting the malformed bundle.


if __name__ == "__main__":  # Allow running this file directly for local debugging.
    unittest.main()  # Execute the tests.
