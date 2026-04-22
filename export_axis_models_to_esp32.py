"""Export depth, forward, and yaw model bundles into ESP32 header files.

Reading route:
1. Start with `main()` to see the directory-in / three-headers-out workflow.
2. Then read `parse_args()` to understand the deployment CLI surface.
3. Then read `export_axis_models()` because it maps each axis bundle to one firmware header.
4. Finally read `AXIS_EXPORTS` to see the default namespace and include-guard mapping.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the export CLI.
import json  # Load the serialized model bundles from disk.
from pathlib import Path  # Build input and output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add the repo root to `sys.path`.

from learning.export_to_esp32 import render_header  # Reuse the single-bundle header renderer.


AXIS_EXPORTS = {
    "depth": {
        "filename": "DepthResidualModelData.h",  # Default output header for the depth model.
        "namespace": "depth_residual_model",  # Firmware namespace used by the depth controller.
        "include_guard": "ESP32_DEPTH_RESIDUAL_MODEL_DATA_H",  # Include guard for the depth header.
    },
    "forward": {
        "filename": "ForwardResidualModelData.h",  # Default output header for the forward model.
        "namespace": "forward_residual_model",  # Firmware namespace used by the forward controller.
        "include_guard": "ESP32_FORWARD_RESIDUAL_MODEL_DATA_H",  # Include guard for the forward header.
    },
    "yaw": {
        "filename": "YawResidualModelData.h",  # Default output header for the yaw model.
        "namespace": "yaw_residual_model",  # Firmware namespace used by the yaw controller.
        "include_guard": "ESP32_YAW_RESIDUAL_MODEL_DATA_H",  # Include guard for the yaw header.
    },
}  # Map each logical control axis to its default firmware export target.


def main() -> None:
    """CLI entry point for exporting all three axis bundles at once."""

    args = parse_args()  # Parse all user-supplied export options.
    export_axis_models(
        model_dir=Path(args.model_dir),  # Read axis bundles from this directory.
        output_dir=Path(args.output_dir),  # Write generated headers into this directory.
    )


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for three-axis export."""

    parser = argparse.ArgumentParser(
        description="Export depth, forward, and yaw model bundles into ESP32 header files.",  # Describe the tool.
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing depth/forward/yaw model JSON bundles")  # Bundle directory.
    parser.add_argument("--output-dir", required=True, help="Directory that will receive the generated ESP32 headers")  # Header directory.
    return parser.parse_args()  # Return the parsed argument namespace.


def export_axis_models(*, model_dir: Path, output_dir: Path) -> None:
    """Export every expected axis bundle into its matching ESP32 header."""

    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists before writing headers.
    for axis_name, export_config in AXIS_EXPORTS.items():  # Visit every supported control axis.
        bundle_path = model_dir / f"{axis_name}_model.json"  # Build the expected JSON bundle path for this axis.
        if not bundle_path.is_file():  # Fail fast when a required axis bundle is missing.
            raise FileNotFoundError(f"missing axis model bundle: {bundle_path}")  # Explain which bundle is absent.

        payload = json.loads(bundle_path.read_text(encoding="utf-8"))  # Load the trained model bundle from disk.
        header_text = render_header(
            payload=payload,  # Convert this bundle into an ESP32 header.
            namespace=str(export_config["namespace"]),  # Use the axis-specific firmware namespace.
            include_guard=str(export_config["include_guard"]),  # Use the axis-specific include guard.
        )
        output_path = output_dir / str(export_config["filename"])  # Build the destination header path for this axis.
        output_path.write_text(header_text, encoding="utf-8")  # Save the generated header text.
        print(f"[{axis_name}] wrote ESP32 model header to {output_path}")  # Tell the user where this axis header was written.


if __name__ == "__main__":  # Run the CLI when the file is executed directly.
    main()  # Start three-axis ESP32 header export.
