"""Export a trained residual model bundle into an ESP32 C++ header.

Reading route:
1. Start with `main()` to see the file-in / file-out workflow.
2. Then read `render_header()` because it defines the full firmware export format.
3. Then read `_read_layer()` to see how JSON weights are parsed.
4. Finally read `_format_float()`, `_format_1d_array()`, and `_format_2d_array()`
   to see how Python values become C++ constants.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the export CLI.
import json  # Load the serialized model bundle from disk.
from pathlib import Path  # Build input and output paths safely across platforms.


EXPECTED_FEATURE_COLUMNS = [
    "depth_err_cm",  # Depth tracking error expected by firmware inference.
    "depth_speed_cm_s",  # Vertical speed expected by firmware inference.
    "depth_accel_cm_s2",  # Vertical acceleration expected by firmware inference.
    "roll_deg",  # Roll feature expected by firmware inference.
    "pitch_deg",  # Pitch feature expected by firmware inference.
    "gyro_x_deg_s",  # Gyroscope x-axis feature expected by firmware inference.
    "gyro_y_deg_s",  # Gyroscope y-axis feature expected by firmware inference.
    "gyro_z_deg_s",  # Gyroscope z-axis feature expected by firmware inference.
    "battery_v",  # Battery-voltage feature expected by firmware inference.
    "buoyancy_pwm_applied",  # Previous buoyancy PWM feature expected by firmware inference.
]


def main() -> None:
    """CLI entry point for turning a JSON model bundle into a C++ header."""

    args = parse_args()  # Parse all user-supplied export options.
    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))  # Load the trained model bundle from JSON.
    header_text = render_header(
        payload=payload,  # Pass the full JSON bundle.
        namespace=args.namespace,  # Use the requested C++ namespace.
        include_guard=args.include_guard,  # Use the requested include guard.
    )  # Convert the bundle into C++ header text.
    output_path = Path(args.output)  # Normalize the destination path.
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories when needed.
    output_path.write_text(header_text, encoding="utf-8")  # Save the generated header to disk.
    print(f"wrote ESP32 model header to {output_path}")  # Tell the user where the header was written.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for header generation."""

    parser = argparse.ArgumentParser(
        description="Render a trained residual model JSON bundle as an ESP32 C++ header.",  # Describe the tool.
    )  # Build the CLI parser.
    parser.add_argument("--model", required=True, help="Path to residual_model.json")  # Input bundle path.
    parser.add_argument("--output", required=True, help="Path to the generated .h file")  # Output header path.
    parser.add_argument(
        "--namespace",  # Flag name on the CLI.
        default="residual_model",  # Use the default firmware namespace unless overridden.
        help="C++ namespace for the generated constants",  # Explain the option.
    )
    parser.add_argument(
        "--include-guard",  # Flag name on the CLI.
        default="ESP32_RESIDUAL_MODEL_DATA_H",  # Use the default include guard unless overridden.
        help="Header include guard",  # Explain the option.
    )
    return parser.parse_args()  # Return the parsed argument namespace.


def render_header(
    *,
    payload: dict[str, object],
    namespace: str = "residual_model",
    include_guard: str = "ESP32_RESIDUAL_MODEL_DATA_H",
) -> str:
    """Render the model bundle as a self-contained ESP32 header file."""

    metadata = payload["metadata"]  # Read the bundle metadata section.
    input_standardizer = payload["input_standardizer"]  # Read input normalization statistics.
    target_standardizer = payload["target_standardizer"]  # Read target normalization statistics.
    model = payload["model"]  # Read model architecture and weights.

    if not isinstance(metadata, dict):  # Validate the metadata payload type.
        raise ValueError("metadata must be a mapping")  # Explain malformed model bundles.
    if not isinstance(input_standardizer, dict):  # Validate the input standardizer payload type.
        raise ValueError("input_standardizer must be a mapping")  # Explain malformed model bundles.
    if not isinstance(target_standardizer, dict):  # Validate the target standardizer payload type.
        raise ValueError("target_standardizer must be a mapping")  # Explain malformed model bundles.
    if not isinstance(model, dict):  # Validate the model payload type.
        raise ValueError("model must be a mapping")  # Explain malformed model bundles.

    window_size = int(metadata["window_size"])  # Restore the stacked-frame count.
    feature_columns = [str(value) for value in metadata["feature_columns"]]  # Restore base feature order.
    input_dim = int(model["input_dim"])  # Restore flattened input width.
    hidden_dims = [int(value) for value in model["hidden_dims"]]  # Restore hidden-layer widths.
    layers = model["layers"]  # Read serialized layer payloads.
    if not isinstance(layers, list) or len(hidden_dims) != 2 or len(layers) != 3:  # Enforce the tiny firmware network shape.
        raise ValueError("export currently supports exactly two hidden layers")  # Explain unsupported architectures.
    if window_size < 1:  # Reject degenerate window sizes.
        raise ValueError("window_size must be positive")  # Explain malformed bundles.

    base_feature_count = len(feature_columns)  # Count features before window flattening.
    if base_feature_count * window_size != input_dim:  # Ensure metadata matches the serialized model width.
        raise ValueError("window_size * base_feature_count must match input_dim")  # Explain inconsistent bundles.
    if feature_columns != EXPECTED_FEATURE_COLUMNS:  # Ensure training and firmware agree on exact feature order.
        raise ValueError(
            "feature_columns must match the ESP32 inference contract exactly: "
            + ", ".join(EXPECTED_FEATURE_COLUMNS)
        )  # Explain the strict firmware contract.

    means = [float(value) for value in input_standardizer["means"]]  # Restore input means as floats.
    stds = [float(value) for value in input_standardizer["stds"]]  # Restore input standard deviations as floats.
    target_mean = float(target_standardizer["means"][0])  # Restore scalar target mean.
    target_std = float(target_standardizer["stds"][0])  # Restore scalar target standard deviation.

    if len(means) != input_dim or len(stds) != input_dim:  # Ensure normalization vectors match the model width.
        raise ValueError("input standardizer dimension does not match input_dim")  # Explain inconsistent bundles.

    layer0 = _read_layer(layers[0])  # Parse the first hidden layer.
    layer1 = _read_layer(layers[1])  # Parse the second hidden layer.
    layer2 = _read_layer(layers[2])  # Parse the output layer.

    lines = [
        f"#ifndef {include_guard}",  # Start the include guard.
        f"#define {include_guard}",  # Define the include guard symbol.
        "",  # Separate the guard from the comment block.
        "// Auto-generated by learning/export_to_esp32.py.",  # Document the generator.
        "// Regenerate this file after retraining the residual controller.",  # Tell users not to edit manually.
        "// Feature order: " + ", ".join(feature_columns),  # Freeze the exact firmware input order in the header.
        "",  # Separate comments from includes.
        "#include <Arduino.h>",  # Pull in Arduino integer types.
        "",  # Separate includes from namespace content.
        f"namespace {namespace} {{",  # Open the requested namespace.
        "",  # Separate the namespace header from constants.
        "constexpr bool kModelAvailable = true;",  # Flag that the generated model exists.
        f"constexpr uint8_t kWindowSize = {window_size};",  # Emit the stacked-frame count.
        f"constexpr uint8_t kBaseFeatureCount = {base_feature_count};",  # Emit feature count before flattening.
        f"constexpr uint16_t kInputDim = {input_dim};",  # Emit flattened input width.
        f"constexpr uint8_t kHidden1Size = {hidden_dims[0]};",  # Emit first hidden-layer width.
        f"constexpr uint8_t kHidden2Size = {hidden_dims[1]};",  # Emit second hidden-layer width.
        f"constexpr float kTargetMean = {_format_float(target_mean)};",  # Emit target mean as a C++ float literal.
        f"constexpr float kTargetStd = {_format_float(target_std)};",  # Emit target std as a C++ float literal.
        "",  # Separate scalar constants from arrays.
        "// Per-input normalization used before the forward pass.",  # Explain the normalization arrays.
        _format_1d_array("kInputMeans", means),  # Emit the per-input mean vector.
        "",  # Separate mean and std arrays visually.
        _format_1d_array("kInputStds", stds),  # Emit the per-input std vector.
        "",  # Separate normalization arrays from network weights.
        "// Dense layer weights are emitted as compile-time constants so the",  # Explain why arrays are generated.
        "// firmware can run inference without filesystem access.",  # Complete the explanation line.
        _format_2d_array("kLayer0Weights", layer0["weights"]),  # Emit first-layer weight matrix.
        "",  # Separate matrices from bias vectors.
        _format_1d_array("kLayer0Biases", layer0["biases"]),  # Emit first-layer bias vector.
        "",  # Separate first and second layer parameters.
        _format_2d_array("kLayer1Weights", layer1["weights"]),  # Emit second-layer weight matrix.
        "",  # Separate matrices from bias vectors.
        _format_1d_array("kLayer1Biases", layer1["biases"]),  # Emit second-layer bias vector.
        "",  # Separate second and output layer parameters.
        _format_2d_array("kLayer2Weights", layer2["weights"]),  # Emit output-layer weight matrix.
        "",  # Separate matrices from bias vectors.
        _format_1d_array("kLayer2Biases", layer2["biases"]),  # Emit output-layer bias vector.
        "",  # Separate parameter arrays from namespace close.
        "}  // namespace " + namespace,  # Close the namespace.
        "",  # Separate namespace close from include-guard close.
        f"#endif  // {include_guard}",  # Close the include guard.
        "",  # End the file with a trailing newline when joined.
    ]  # Assemble the generated header line by line.
    return "\n".join(lines)  # Return the final header text.


def _read_layer(layer_payload: object) -> dict[str, list[list[float]] | list[float]]:
    """Parse one exported layer into typed float lists."""

    if not isinstance(layer_payload, dict):  # Ensure each layer payload is a mapping.
        raise ValueError("layer payload must be a mapping")  # Explain malformed serialized models.

    weights = [[float(value) for value in row] for row in layer_payload["weights"]]  # Restore the weight matrix.
    biases = [float(value) for value in layer_payload["biases"]]  # Restore the bias vector.
    return {"weights": weights, "biases": biases}  # Return the parsed layer payload.


def _format_float(value: float) -> str:
    """Format one float as a stable C++ float literal."""

    text = f"{value:.9f}".rstrip("0").rstrip(".")  # Trim redundant trailing zeros for compact output.
    if "." not in text:  # Ensure integer-looking values still become float literals.
        text += ".0"  # Add the explicit decimal part.
    return text + "f"  # Append the C++ float suffix.


def _format_1d_array(name: str, values: list[float]) -> str:
    """Render a flat float array declaration for the generated header."""

    body = ", ".join(_format_float(value) for value in values)  # Format every element as a float literal.
    return f"constexpr float {name}[{len(values)}] = {{ {body} }};"  # Return the full array declaration.


def _format_2d_array(name: str, rows: list[list[float]]) -> str:
    """Render a 2D float array declaration for the generated header."""

    if not rows or not rows[0]:  # Reject empty matrices because firmware expects fixed dimensions.
        raise ValueError(f"{name} must not be empty")  # Explain malformed serialized models.

    rendered_rows = []  # Accumulate one rendered row string at a time.
    for row in rows:  # Visit every matrix row in order.
        rendered_rows.append("    { " + ", ".join(_format_float(value) for value in row) + " }")  # Format one row.
    return (
        f"constexpr float {name}[{len(rows)}][{len(rows[0])}] = {{\n"  # Emit the declaration header.
        + ",\n".join(rendered_rows)  # Emit all matrix rows separated by commas.
        + "\n};"  # Close the declaration.
    )  # Return the full multi-line 2D array declaration.


if __name__ == "__main__":  # Run the CLI when the file is executed directly.
    main()  # Start header export.
