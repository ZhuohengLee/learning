"""Export residual model bundles into ESP32 C++ header files.

Reading route:
1. Start with `main()` to see the public export entry point.
2. Then read `parse_args()` to understand single-bundle vs three-axis modes.
3. Then read `export_axis_models()` for the batch depth/forward/yaw path.
4. Finally read `render_header()` for the actual firmware header format.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the export CLI.
import json  # Load the serialized model bundles from disk.
from pathlib import Path  # Build input and output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add the parent of the repo root so `learning` can be imported.

from learning.data import DEFAULT_FEATURE_COLUMNS, DEFAULT_UNIFIED_FEATURE_COLUMNS  # Reuse the training-side feature contracts.


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
SUPPORTED_FEATURE_CONTRACTS = {
    "depth_legacy": list(DEFAULT_FEATURE_COLUMNS),  # Keep supporting the original depth-only firmware contract.
    "shared_three_axis": list(DEFAULT_UNIFIED_FEATURE_COLUMNS),  # Support the shared feature contract used by depth/forward/yaw together.
}  # Map stable contract names to exact per-frame feature order.
MAX_ESP32_INPUT_DIM = 256  # Keep runtime working memory bounded on the microcontroller.
MAX_ESP32_HIDDEN1_SIZE = 64  # Match the first hidden-layer ceiling enforced by firmware inference.
MAX_ESP32_HIDDEN2_SIZE = 64  # Match the second hidden-layer ceiling enforced by firmware inference.


def main() -> None:
    """CLI entry point for exporting one or three residual bundles."""

    args = parse_args()  # Parse all user-supplied export options.
    if args.model_dir:  # Use batch mode when a model directory is supplied.
        export_axis_models(
            model_dir=Path(args.model_dir),  # Read axis bundles from this directory.
            output_dir=Path(args.output_dir),  # Write generated headers into this directory.
        )
        return  # Stop after finishing the batch export.

    export_model(
        model_path=Path(args.model),  # Read one JSON model bundle from this path.
        output_path=Path(args.output),  # Write one generated header to this path.
        namespace=args.namespace,  # Use the requested C++ namespace.
        include_guard=args.include_guard,  # Use the requested include guard.
    )  # Run the single-bundle export path.


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for model export."""

    parser = argparse.ArgumentParser(
        description="Export one residual model bundle or a depth/forward/yaw model directory into ESP32 headers.",  # Describe the tool.
    )  # Build the CLI parser.
    parser.add_argument("--model", help="Path to a single residual model bundle JSON file")  # Optional single-bundle input path.
    parser.add_argument("--output", help="Path to the generated .h file for single-bundle export")  # Optional single-bundle output path.
    parser.add_argument("--model-dir", help="Directory containing depth/forward/yaw model JSON bundles")  # Optional three-axis input directory.
    parser.add_argument("--output-dir", help="Directory that will receive the generated ESP32 headers")  # Optional three-axis output directory.
    parser.add_argument(
        "--namespace",  # Flag name on the CLI.
        default="residual_model",  # Use the default firmware namespace unless overridden.
        help="C++ namespace for single-bundle export",  # Explain the option.
    )
    parser.add_argument(
        "--include-guard",  # Flag name on the CLI.
        default="ESP32_RESIDUAL_MODEL_DATA_H",  # Use the default include guard unless overridden.
        help="Header include guard for single-bundle export",  # Explain the option.
    )
    args = parser.parse_args()  # Parse the raw CLI first so cross-argument validation can run.

    single_mode = bool(args.model)  # Detect the one-bundle export path.
    batch_mode = bool(args.model_dir)  # Detect the depth/forward/yaw batch export path.
    if single_mode == batch_mode:  # Reject ambiguous or empty mode selection.
        parser.error("choose either --model for single-bundle export or --model-dir for batch export")  # Explain the allowed modes.
    if single_mode and not args.output:  # Require a destination file in single-bundle mode.
        parser.error("--output is required when using --model")  # Explain the missing argument.
    if batch_mode and not args.output_dir:  # Require a destination directory in batch mode.
        parser.error("--output-dir is required when using --model-dir")  # Explain the missing argument.
    if args.output and not single_mode:  # Reject a single-bundle output path in batch mode.
        parser.error("--output can only be used with --model")  # Explain the invalid combination.
    if args.output_dir and not batch_mode:  # Reject a batch output directory in single-bundle mode.
        parser.error("--output-dir can only be used with --model-dir")  # Explain the invalid combination.
    return args  # Return the validated argument namespace.


def export_model(
    *,
    model_path: Path,
    output_path: Path,
    namespace: str = "residual_model",
    include_guard: str = "ESP32_RESIDUAL_MODEL_DATA_H",
) -> None:
    """Export one JSON bundle into one ESP32 header file."""

    payload = json.loads(model_path.read_text(encoding="utf-8"))  # Load the trained model bundle from JSON.
    header_text = render_header(
        payload=payload,  # Convert the bundle into C++ header text.
        namespace=namespace,  # Use the requested C++ namespace.
        include_guard=include_guard,  # Use the requested include guard.
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories when needed.
    output_path.write_text(header_text, encoding="utf-8")  # Save the generated header to disk.
    print(f"wrote ESP32 model header to {output_path}")  # Tell the user where the header was written.


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
    contract_name = _resolve_feature_contract(feature_columns)  # Validate and name the exact firmware feature contract.
    if input_dim > MAX_ESP32_INPUT_DIM:  # Reject models that exceed the runtime scratch-buffer limit.
        raise ValueError(f"input_dim exceeds ESP32 inference limit of {MAX_ESP32_INPUT_DIM}")  # Explain the failing model size.
    if hidden_dims[0] > MAX_ESP32_HIDDEN1_SIZE:  # Reject oversized first hidden layers before deployment.
        raise ValueError(f"hidden1 exceeds ESP32 inference limit of {MAX_ESP32_HIDDEN1_SIZE}")  # Explain the failing model size.
    if hidden_dims[1] > MAX_ESP32_HIDDEN2_SIZE:  # Reject oversized second hidden layers before deployment.
        raise ValueError(f"hidden2 exceeds ESP32 inference limit of {MAX_ESP32_HIDDEN2_SIZE}")  # Explain the failing model size.

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
        "// Auto-generated by learning/export.py.",  # Document the generator.
        "// Regenerate this file after retraining the residual controller.",  # Tell users not to edit manually.
        "// Feature contract: " + contract_name,  # Freeze which firmware-side feature contract this bundle targets.
        "// Feature order: " + ", ".join(feature_columns),  # Freeze the exact firmware input order in the header.
        "",  # Separate comments from includes.
        '#include "ResidualModelSpec.h"',  # Pull in the generic model-view struct used by runtime inference.
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
        "",  # Separate parameter arrays from the runtime view helper.
        "inline ResidualModelView getModelView() {",  # Expose this namespace as one generic runtime model view.
        "    return ResidualModelView{",  # Start the returned struct literal.
        "        kModelAvailable,",  # Pass the model availability flag.
        "        kWindowSize,",  # Pass the stacked-frame count.
        "        kBaseFeatureCount,",  # Pass the per-frame feature count.
        "        kInputDim,",  # Pass the flattened input width.
        "        kHidden1Size,",  # Pass the first hidden-layer width.
        "        kHidden2Size,",  # Pass the second hidden-layer width.
        "        kTargetMean,",  # Pass the target mean.
        "        kTargetStd,",  # Pass the target std.
        "        &kInputMeans[0],",  # Pass a pointer to the input means.
        "        &kInputStds[0],",  # Pass a pointer to the input stds.
        "        &kLayer0Weights[0][0],",  # Pass a flat pointer to the first-layer weights.
        "        &kLayer0Biases[0],",  # Pass a pointer to the first-layer biases.
        "        &kLayer1Weights[0][0],",  # Pass a flat pointer to the second-layer weights.
        "        &kLayer1Biases[0],",  # Pass a pointer to the second-layer biases.
        "        &kLayer2Weights[0][0],",  # Pass a flat pointer to the output-layer weights.
        "        &kLayer2Biases[0],",  # Pass a pointer to the output-layer biases.
        "    };",  # Close the returned struct literal.
        "}",  # Close the helper function.
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


def _resolve_feature_contract(feature_columns: list[str]) -> str:
    """Validate the feature order against the known firmware-side contracts."""

    for contract_name, expected_columns in SUPPORTED_FEATURE_CONTRACTS.items():  # Visit every supported firmware contract.
        if feature_columns == expected_columns:  # Match the feature order exactly.
            return contract_name  # Return the stable contract name for comments and diagnostics.

    supported_contracts = [
        f"{name}: " + ", ".join(columns)  # Render each supported contract on one readable line.
        for name, columns in SUPPORTED_FEATURE_CONTRACTS.items()  # Visit every supported firmware contract.
    ]
    raise ValueError(  # Explain all supported contracts instead of one hard-coded order.
        "feature_columns must match one of the supported ESP32 inference contracts:\n"
        + "\n".join(supported_contracts)
    )


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
    main()  # Start model export.
