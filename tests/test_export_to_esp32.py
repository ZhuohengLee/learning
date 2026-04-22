"""Tests for converting a trained bundle into an ESP32 header.

Reading route:
1. Start with `test_render_header_contains_expected_dimensions()` to see the happy-path export contract.
2. Then read `test_render_header_rejects_feature_order_mismatch()` to see the strict safety check.
"""

import json  # Load the sample model bundle from disk.
import unittest  # Use the standard-library test framework.
from pathlib import Path  # Build fixture paths safely across platforms.

from learning.export_to_esp32 import render_header  # Import the exporter under test.


SAMPLE_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "sample_model.json"  # Reuse the checked-in sample bundle.


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
        self.assertIn("// Feature order: depth_err_cm, depth_speed_cm_s", header)  # Header should document feature order.
        self.assertIn("constexpr float kLayer2Biases[1]", header)  # Header should contain output-layer biases.

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

        with self.assertRaisesRegex(ValueError, "feature_columns must match the ESP32 inference contract"):  # Expect a strict contract error.
            render_header(payload=payload)  # Try exporting the malformed bundle.


if __name__ == "__main__":  # Allow running this file directly for local debugging.
    unittest.main()  # Execute the tests.
