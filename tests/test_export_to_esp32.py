import json
import unittest
from pathlib import Path

from learning.export_to_esp32 import render_header


SAMPLE_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "sample_model.json"


class ExportToEsp32Tests(unittest.TestCase):
    def test_render_header_contains_expected_dimensions(self) -> None:
        payload = json.loads(SAMPLE_MODEL_PATH.read_text(encoding="utf-8"))
        header = render_header(payload=payload, namespace="test_model", include_guard="TEST_MODEL_H")

        self.assertIn("#ifndef TEST_MODEL_H", header)
        self.assertIn("namespace test_model {", header)
        self.assertIn("constexpr bool kModelAvailable = true;", header)
        self.assertIn("constexpr uint8_t kWindowSize = 3;", header)
        self.assertIn("constexpr uint8_t kBaseFeatureCount = 10;", header)
        self.assertIn("constexpr uint16_t kInputDim = 30;", header)
        self.assertIn("constexpr uint8_t kHidden1Size = 8;", header)
        self.assertIn("constexpr uint8_t kHidden2Size = 4;", header)
        self.assertIn("// Feature order: depth_err_cm, depth_speed_cm_s", header)
        self.assertIn("constexpr float kLayer2Biases[1]", header)

    def test_render_header_rejects_feature_order_mismatch(self) -> None:
        payload = json.loads(SAMPLE_MODEL_PATH.read_text(encoding="utf-8"))
        payload["metadata"]["feature_columns"] = [
            "depth_speed_cm_s",
            "depth_err_cm",
            "depth_accel_cm_s2",
            "roll_deg",
            "pitch_deg",
            "gyro_x_deg_s",
            "gyro_y_deg_s",
            "gyro_z_deg_s",
            "battery_v",
            "buoyancy_pwm_applied",
        ]

        with self.assertRaisesRegex(ValueError, "feature_columns must match the ESP32 inference contract"):
            render_header(payload=payload)


if __name__ == "__main__":
    unittest.main()
