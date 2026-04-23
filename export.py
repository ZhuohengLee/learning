"""Export one joint residual bundle into ONNX and optionally into `.espdl`.

Reading route:
1. Start with `main()` for CLI wiring.
2. Then read `export_model()` because it is the public joint-export entry.
3. Then read `prepare_calibration_inputs()` to see how calibration data is rebuilt from CSV.
4. Finally read `export_espdl_from_onnx()` to see how ESP-PPQ is invoked.
"""

from __future__ import annotations  # Defer type-hint evaluation for cleaner forward references.

import argparse  # Parse command-line arguments for the export CLI.
import contextlib  # Temporarily redirect noisy exporter stdout on Windows GBK consoles.
import importlib.util  # Detect optional dependencies such as `onnx` and `ppq`.
import io  # Hold redirected exporter logs in memory.
import json  # Write metadata sidecars and manifests.
from pathlib import Path  # Build input and output paths safely across platforms.
import sys  # Adjust import path when running this file as a script.
from typing import Sequence  # Type ordered collections without requiring a concrete container.

if __package__ in {None, ""}:  # Detect direct script execution outside package mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Add the repo root so `learning` can be imported.

from learning.data import (  # Reuse the shared telemetry contract for calibration-data preparation.
    Standardizer,  # Rebuild input normalization from the saved training bundle.
    build_multi_axis_examples,  # Convert calibration CSV rows into windowed joint feature vectors.
    load_control_rows,  # Load calibration telemetry from the shared CSV format.
)
from learning.model import TorchJointResidualMLP, require_torch  # Import the joint PyTorch backend and dependency guard.


SUPPORTED_ESPDL_TARGETS = ("esp32", "esp32s3", "esp32p4")  # Keep the supported ESP-DL deployment targets explicit.


def main() -> None:
    """CLI entry point for exporting one joint PyTorch bundle into ONNX / `.espdl` artifacts."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    _require_onnx()  # ONNX export is always the first stage of the pipeline.
    args = parse_args()  # Parse all user-supplied export options.
    export_model(
        model_path=Path(args.model),  # Read the joint `.pt` bundle from this path.
        output_path=Path(args.output),  # Write the ONNX export to this path.
        opset=args.opset,  # Use the configured ONNX opset.
        espdl_path=Path(args.espdl_output) if args.espdl_output else None,  # Optionally continue into `.espdl`.
        calibration_csv=Path(args.calibration_csv) if args.calibration_csv else None,  # Use this telemetry CSV for post-training quantization.
        calib_steps=args.calib_steps,  # Use at most this many calibration samples.
        max_dt_ms=args.max_dt_ms,  # Reuse the same sequence-gap threshold used during training.
        target=args.target,  # Quantize for this ESP-DL target platform.
        num_of_bits=args.num_of_bits,  # Use this quantization precision.
        device_name=args.device,  # Run ESP-PPQ on this torch device.
        export_test_values=args.export_test_values,  # Optionally embed a deterministic board-side test vector.
        verbose=args.verbose,  # Forward the requested ESP-PPQ verbosity level.
    )


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for joint PyTorch ONNX / ESP-DL export."""

    parser = argparse.ArgumentParser(
        description=(
            "Export one shared-trunk PyTorch residual model bundle into one ONNX file, "
            "and optionally continue into one ESP-PPQ `.espdl` deployment artifact."
        ),
    )
    parser.add_argument("--model", required=True, help="Path to the joint residual model bundle `.pt` file")
    parser.add_argument("--output", required=True, help="Path to the generated joint `.onnx` file")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version used for export")
    parser.add_argument("--espdl-output", help="Optional `.espdl` output path")
    parser.add_argument("--calibration-csv", help="Telemetry CSV used to build ESP-PPQ calibration inputs")
    parser.add_argument("--calib-steps", type=int, default=32, help="Maximum number of calibration samples")
    parser.add_argument("--max-dt-ms", type=float, default=80.0, help="Maximum allowed timestamp gap when rebuilding windows")
    parser.add_argument(
        "--target",
        choices=list(SUPPORTED_ESPDL_TARGETS),
        default="esp32s3",
        help="ESP-DL deployment target; `esp32` is mapped to ESP-PPQ target `c` as required by Espressif docs.",
    )
    parser.add_argument("--num-bits", dest="num_of_bits", type=int, choices=[8, 16], default=8, help="Quantization bit width for ESP-PPQ")
    parser.add_argument("--device", default="cpu", help="Torch device used by ESP-PPQ, for example `cpu` or `cuda`")
    parser.add_argument(
        "--export-test-values",
        action="store_true",
        help="Embed one deterministic test input/output pair into the exported `.espdl` model for board-side verification.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level forwarded to ESP-PPQ")
    args = parser.parse_args()  # Parse the full CLI once.

    if args.espdl_output and not args.calibration_csv:  # Enforce representative calibration data for PTQ.
        parser.error("--calibration-csv is required when exporting `.espdl` artifacts")  # Explain the PTQ dependency clearly.
    if args.calib_steps < 1:  # Reject invalid calibration sample limits.
        parser.error("--calib-steps must be positive")  # Explain the lower bound clearly.
    return args  # Return the validated CLI arguments.


def export_model(
    *,
    model_path: Path,
    output_path: Path,
    opset: int = 13,
    espdl_path: Path | None = None,
    calibration_csv: Path | None = None,
    calib_steps: int = 32,
    max_dt_ms: float = 80.0,
    target: str = "esp32s3",
    num_of_bits: int = 8,
    device_name: str = "cpu",
    export_test_values: bool = False,
    verbose: int = 1,
) -> dict[str, object]:
    """Export one joint `.pt` residual bundle into ONNX and optionally into `.espdl` artifacts."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    _require_onnx()  # ONNX export dependencies are required for the first stage.
    import torch  # type: ignore[import-not-found]

    bundle = torch.load(model_path, map_location="cpu")  # Load the saved joint PyTorch bundle from disk.
    model = _restore_model_from_bundle(bundle)  # Recreate the `nn.Module` with trained weights.
    input_dim = _require_positive_int(bundle["model_spec"]["input_dim"], field_name="model_spec.input_dim")  # Recover the fixed flat input width from the bundle.
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create the ONNX output directory before export.
    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)  # Match the fixed batch-1 ESP-DL input contract.
    with contextlib.redirect_stdout(io.StringIO()):  # Suppress torch.onnx Unicode-rich progress logs that break on GBK consoles.
        torch.onnx.export(
            model,  # Export the restored joint PyTorch module.
            dummy_input,  # Trace the network with one fixed-size batch-1 input.
            output_path,  # Write the ONNX graph here.
            input_names=["input"],  # Keep the single input name stable for firmware integration.
            output_names=["residuals"],  # Emit one `[batch, 3]` residual tensor ordered by `axis_names`.
            opset_version=opset,  # Use the requested ONNX opset version.
            do_constant_folding=True,  # Fold constants during export for a simpler graph.
        )

    metadata = _bundle_metadata(bundle, model_path=model_path, onnx_path=output_path)  # Build the JSON-safe ONNX metadata payload.
    metadata_path = output_path.with_suffix(".metadata.json")  # Place the ONNX sidecar next to the ONNX file.
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")  # Save the ONNX metadata sidecar.
    print(f"exported ONNX model to {output_path}")  # Tell the user where the ONNX file was written.
    print(f"wrote ONNX metadata to {metadata_path}")  # Tell the user where the ONNX sidecar was written.

    result: dict[str, object] = {
        "backend": "pytorch",  # Record the backend used to create these artifacts.
        "source_bundle": str(model_path),  # Record the original `.pt` bundle path.
        "onnx_path": str(output_path),  # Record the ONNX output path.
        "onnx_metadata_path": str(metadata_path),  # Record the ONNX metadata sidecar path.
        "opset": opset,  # Record the ONNX opset used for export.
    }
    if espdl_path is None:  # Stop after ONNX export when the caller did not request ESP-PPQ.
        return result  # Return only the ONNX artifact information.

    if calibration_csv is None:  # Guard against callers that bypassed CLI validation.
        raise ValueError("calibration_csv is required when exporting `.espdl` artifacts")  # Explain the PTQ dependency clearly.

    calibration = prepare_calibration_inputs(
        bundle=bundle,  # Use the saved feature contract and standardizer from the bundle.
        calibration_csv=calibration_csv,  # Read representative calibration windows from this telemetry CSV.
        max_dt_ms=max_dt_ms,  # Reuse the requested sequence-gap threshold.
        limit=calib_steps,  # Limit the calibration set to the requested number of samples.
    )
    espdl_result = export_espdl_from_onnx(
        onnx_path=output_path,  # Quantize the ONNX file just exported above.
        espdl_path=espdl_path,  # Write the quantized ESP-DL binary here.
        calibration_samples=calibration["samples"],  # Feed representative normalized calibration vectors into ESP-PPQ.
        target=target,  # Use the requested deployment target.
        num_of_bits=num_of_bits,  # Use the requested quantization precision.
        device_name=device_name,  # Run quantization on this torch device.
        export_test_values=export_test_values,  # Optionally embed a deterministic board-side test sample.
        verbose=verbose,  # Forward the requested verbosity level.
    )

    espdl_metadata_path = Path(f"{espdl_path}.metadata.json")  # Keep an additional project-specific sidecar next to the `.espdl` file.
    espdl_metadata_path.write_text(
        json.dumps(
            {
                "backend": "pytorch",  # Record the backend used to create the deployment artifact.
                "source_bundle": str(model_path),  # Preserve the source `.pt` path for traceability.
                "onnx_path": str(output_path),  # Record the ONNX intermediate used for quantization.
                "calibration_csv": str(calibration_csv),  # Record which telemetry CSV powered PTQ.
                "calibration_samples": calibration["sample_count"],  # Record how many representative samples were used.
                "window_size": calibration["window_size"],  # Record the stacked-frame window size used for calibration.
                "feature_columns": calibration["feature_columns"],  # Record the base feature contract used during calibration.
                "axis_names": calibration["axis_names"],  # Record the joint output order used for calibration and firmware parsing.
                "target_columns": calibration["target_columns"],  # Record the logical-axis to CSV-target mapping.
                "requested_target": target,  # Preserve the user-facing chip target.
                "esp_ppq_target": espdl_result["esp_ppq_target"],  # Preserve the quantizer target actually passed to ESP-PPQ.
                "num_of_bits": num_of_bits,  # Preserve the quantization precision.
                "export_test_values": export_test_values,  # Record whether board-side test values were embedded.
                "artifacts": espdl_result,  # Record the `.espdl/.info/.json` artifact paths.
                "metadata": metadata["metadata"],  # Keep the original model metadata for firmware integration.
                "input_standardizer": metadata["input_standardizer"],  # Preserve input normalization for firmware-side parity checks.
                "target_standardizer": metadata["target_standardizer"],  # Preserve output normalization for diagnostics.
                "model_spec": metadata["model_spec"],  # Preserve the trained model architecture summary.
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote ESP-DL metadata to {espdl_metadata_path}")  # Tell the user where the `.espdl` sidecar was written.

    result.update(
        {
            "espdl_path": espdl_result["espdl_path"],  # Record the exported `.espdl` file.
            "espdl_info_path": espdl_result["info_path"],  # Record the `.info` debug file.
            "espdl_quantization_json_path": espdl_result["quantization_json_path"],  # Record the ESP-PPQ JSON sidecar.
            "espdl_metadata_path": str(espdl_metadata_path),  # Record the project-specific `.espdl` metadata sidecar.
            "espdl_target": target,  # Record the user-facing deployment target.
            "espdl_num_of_bits": num_of_bits,  # Record the quantization precision.
            "espdl_calibration_csv": str(calibration_csv),  # Record the calibration CSV used for PTQ.
            "espdl_calibration_samples": calibration["sample_count"],  # Record the number of calibration windows used.
        }
    )
    return result  # Return the full export summary.


def prepare_calibration_inputs(
    *,
    bundle: dict[str, object],
    calibration_csv: Path,
    max_dt_ms: float = 80.0,
    limit: int = 32,
) -> dict[str, object]:
    """Build normalized representative calibration inputs from the shared telemetry CSV contract."""

    if limit < 1:  # Reject invalid calibration sample limits.
        raise ValueError("limit must be positive")  # Explain the lower bound clearly.

    metadata = _require_mapping(bundle.get("metadata"), field_name="metadata")  # Recover the saved feature contract from the bundle.
    feature_columns = _require_string_list(metadata.get("feature_columns"), field_name="metadata.feature_columns")  # Recover the base feature list.
    window_size = _require_positive_int(metadata.get("window_size"), field_name="metadata.window_size")  # Recover the stacked-frame window size.
    axis_names = _require_string_list(metadata.get("axis_names"), field_name="metadata.axis_names")  # Recover the joint output order.
    target_columns = _require_string_mapping(metadata.get("target_columns"), field_name="metadata.target_columns")  # Recover the logical-axis to CSV-target mapping.
    input_standardizer = _standardizer_from_payload(bundle.get("input_standardizer"))  # Recover the saved input normalization statistics.

    rows = load_control_rows(calibration_csv)  # Load the representative telemetry CSV from disk.
    dataset = build_multi_axis_examples(
        rows,  # Rebuild windows from the shared telemetry contract.
        feature_columns,  # Use the same base feature columns used during training.
        axis_targets=target_columns,  # Require the same joint target contract used during training.
        window_size=window_size,  # Use the same stacked-frame width used during training.
        max_dt_ms=max_dt_ms,  # Split sequences with the requested timestamp-gap threshold.
    )
    if not dataset.examples:  # Reject calibration files that do not yield any trainable windows.
        raise ValueError(f"no trainable calibration examples were built from {calibration_csv}")  # Explain the failure clearly.
    if dataset.axis_names != axis_names:  # Detect bundle/CSV mismatches early.
        raise ValueError("calibration axis order does not match the saved model metadata")  # Explain the mismatch clearly.

    selected_examples = _select_evenly_spaced_examples(dataset.examples, limit)  # Spread calibration picks across the whole log instead of taking only the earliest rows.
    normalized_samples = [
        input_standardizer.normalize(example.features)  # Apply the same input normalization used during training.
        for example in selected_examples
    ]
    return {
        "source_csv": str(calibration_csv),  # Record which telemetry CSV produced these calibration samples.
        "sample_count": len(normalized_samples),  # Record how many representative windows were selected.
        "input_dim": len(normalized_samples[0]),  # Record the flat feature width expected by the model.
        "window_size": window_size,  # Record the stacked-frame width used to rebuild windows.
        "feature_columns": feature_columns,  # Record the base feature contract used to rebuild windows.
        "axis_names": axis_names,  # Record the stable joint output order.
        "target_columns": target_columns,  # Record the logical-axis to CSV-target mapping.
        "samples": normalized_samples,  # Return the normalized calibration vectors ready for ESP-PPQ.
        "timestamps_ms": [example.timestamp_ms for example in selected_examples],  # Record which frames were picked for traceability.
        "session_ids": [example.session_id for example in selected_examples],  # Record which sessions supplied the calibration windows.
    }


def export_espdl_from_onnx(
    *,
    onnx_path: Path,
    espdl_path: Path,
    calibration_samples: Sequence[Sequence[float]],
    target: str = "esp32s3",
    num_of_bits: int = 8,
    device_name: str = "cpu",
    export_test_values: bool = False,
    verbose: int = 1,
) -> dict[str, object]:
    """Run ESP-PPQ PTQ on one ONNX file and export the `.espdl/.info/.json` artifacts."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    _require_esp_ppq()  # Reject direct `.espdl` export when ESP-PPQ is not installed.
    import torch  # type: ignore[import-not-found]
    from ppq import QuantizationSettingFactory  # type: ignore[import-not-found]
    from ppq.api import espdl_quantize_onnx  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]

    if not calibration_samples:  # Reject quantization requests without representative calibration inputs.
        raise ValueError("calibration_samples must not be empty")  # Explain the lower bound clearly.

    input_dim = len(calibration_samples[0])  # Infer the flat feature width from the first representative sample.
    if input_dim < 1:  # Reject degenerate empty feature vectors.
        raise ValueError("calibration samples must have a positive feature dimension")  # Explain the lower bound clearly.
    if any(len(sample) != input_dim for sample in calibration_samples):  # Enforce a consistent flat feature width across all samples.
        raise ValueError("all calibration samples must have the same feature width")  # Explain the shape contract clearly.

    espdl_path.parent.mkdir(parents=True, exist_ok=True)  # Create the `.espdl` output directory before quantization starts.
    quant_setting = QuantizationSettingFactory.espdl_setting()  # Use ESP-DL-compatible default PTQ settings.
    quant_target = _normalize_esp_ppq_target(target)  # Map `esp32` to ESP-PPQ target `c` as required by Espressif docs.

    calibration_tensors = [
        torch.tensor(sample, dtype=torch.float32)  # Convert each normalized sample into a float32 tensor.
        for sample in calibration_samples
    ]
    dataloader = DataLoader(
        dataset=calibration_tensors,  # Feed the representative normalized vectors into ESP-PPQ.
        batch_size=1,  # Match the fixed batch-1 contract documented by ESP-DL.
        shuffle=False,  # Keep calibration order deterministic for reproducibility.
    )

    def _collate_fn(batch):  # type: ignore[no-untyped-def]
        """Move one calibration batch onto the requested device for ESP-PPQ."""

        return batch.to(device_name)  # Match Espressif's documented `collate_fn(batch) -> batch.to(device)` pattern.

    quantize_kwargs: dict[str, object] = {
        "onnx_import_file": str(onnx_path),  # Read the just-exported ONNX file from disk.
        "espdl_export_file": str(espdl_path),  # Write the quantized ESP-DL binary here.
        "calib_dataloader": dataloader,  # Feed representative calibration windows into PTQ.
        "calib_steps": len(calibration_tensors),  # Run one PTQ calibration step per representative sample.
        "input_shape": [1, input_dim],  # Declare the fixed `[batch=1, features=input_dim]` model input shape.
        "target": quant_target,  # Use the chip-specific ESP-PPQ target expected by Espressif's quantizer.
        "num_of_bits": num_of_bits,  # Use the requested quantization precision.
        "collate_fn": _collate_fn,  # Move calibration batches onto the requested device.
        "setting": quant_setting,  # Use the default ESP-DL-compatible quantization settings.
        "device": device_name,  # Run quantization on this torch device.
        "error_report": True,  # Ask ESP-PPQ to emit quantization error information for debugging.
        "skip_export": False,  # Export the `.espdl` artifact instead of only building an in-memory graph.
        "export_test_values": export_test_values,  # Optionally embed a deterministic test input/output pair for board-side verification.
        "verbose": verbose,  # Forward the requested ESP-PPQ verbosity level.
    }
    if export_test_values:  # Provide one deterministic test vector so `model->test()` on the board checks a real sample.
        quantize_kwargs["inputs"] = [torch.tensor([calibration_samples[0]], dtype=torch.float32)]  # Match the single-input model contract.

    espdl_quantize_onnx(**quantize_kwargs)  # Run post-training quantization and export the final deployment artifact.

    info_path = espdl_path.with_suffix(".info")  # ESP-PPQ exports the text debug file using the same stem.
    quantization_json_path = espdl_path.with_suffix(".json")  # ESP-PPQ exports quantization metadata using the same stem.
    print(f"exported ESP-DL model to {espdl_path}")  # Tell the user where the `.espdl` file was written.
    print(f"expected ESP-DL debug info at {info_path}")  # Tell the user where the `.info` file should appear.
    print(f"expected ESP-DL quantization JSON at {quantization_json_path}")  # Tell the user where the quantization JSON should appear.
    return {
        "espdl_path": str(espdl_path),  # Record the final deployment artifact path.
        "info_path": str(info_path),  # Record the companion `.info` debug file path.
        "quantization_json_path": str(quantization_json_path),  # Record the companion quantization JSON path.
        "esp_ppq_target": quant_target,  # Record the chip-specific target actually passed into ESP-PPQ.
        "requested_target": target,  # Preserve the original user-facing target for traceability.
    }


def _restore_model_from_bundle(bundle: dict[str, object]):
    """Restore a `TorchJointResidualMLP` from one saved `.pt` bundle."""

    require_torch()  # Fail early with a clear message when PyTorch is unavailable.
    model_spec = _require_mapping(bundle.get("model_spec"), field_name="model_spec")  # Read the saved architecture specification.
    axis_names = _require_string_list(model_spec.get("axis_names"), field_name="model_spec.axis_names")  # Recover the saved output order.
    model = TorchJointResidualMLP(
        input_dim=_require_positive_int(model_spec.get("input_dim"), field_name="model_spec.input_dim"),  # Restore the trained flat input width.
        axis_names=axis_names,  # Restore the trained output-head ordering.
        hidden_dims=[int(value) for value in _require_list(model_spec.get("hidden_dims"), field_name="model_spec.hidden_dims")],  # Restore the two hidden-layer widths.
    )
    state_dict = _require_mapping(bundle.get("state_dict"), field_name="state_dict")  # Read the saved parameter dictionary.
    model.load_state_dict(state_dict)  # Restore the trained weights into the recreated module.
    model.eval()  # Freeze the model into inference mode before ONNX export.
    return model  # Return the restored inference-ready model.


def _bundle_metadata(bundle: dict[str, object], *, model_path: Path, onnx_path: Path) -> dict[str, object]:
    """Extract the JSON-safe metadata needed after ONNX export."""

    metadata = _require_mapping(bundle.get("metadata"), field_name="metadata")  # Read the saved training metadata payload.
    return {
        "backend": "pytorch",  # Record which backend produced the exported artifact.
        "source_bundle": str(model_path),  # Record which `.pt` bundle produced this ONNX file.
        "onnx_path": str(onnx_path),  # Record the ONNX output path.
        "metadata": metadata,  # Preserve the saved training metadata payload.
        "input_standardizer": bundle.get("input_standardizer", {}),  # Preserve input normalization statistics.
        "target_standardizer": bundle.get("target_standardizer", {}),  # Preserve output normalization statistics.
        "model_spec": bundle.get("model_spec", {}),  # Preserve the trained model architecture summary.
    }


def _standardizer_from_payload(payload: object) -> Standardizer:
    """Recreate a `Standardizer` from the serialized bundle payload."""

    mapping = _require_mapping(payload, field_name="input_standardizer")  # Validate that the serialized payload is a mapping.
    means = _require_float_list(mapping.get("means"), field_name="input_standardizer.means")  # Recover the saved per-column means.
    stds = _require_float_list(mapping.get("stds"), field_name="input_standardizer.stds")  # Recover the saved per-column std values.
    if len(means) != len(stds):  # Reject malformed normalization payloads.
        raise ValueError("input_standardizer means/stds length mismatch")  # Explain the mismatch clearly.
    return Standardizer(means=means, stds=stds)  # Rebuild the immutable standardizer record used by the shared data layer.


def _select_evenly_spaced_examples(examples: Sequence[object], limit: int) -> list[object]:
    """Select up to `limit` examples spread roughly evenly across the full log."""

    if limit < 1:  # Reject invalid selection limits.
        raise ValueError("limit must be positive")  # Explain the lower bound clearly.
    if len(examples) <= limit:  # Keep all examples when the log is already short enough.
        return list(examples)  # Return every example in chronological order.
    if limit == 1:  # Special-case the degenerate one-sample selection.
        return [examples[len(examples) // 2]]  # Pick the median example for balance.

    last_index = len(examples) - 1  # Compute the final valid example index once.
    indices = sorted(  # Build a deterministic set of representative indices across the full sequence.
        {
            round(position * last_index / (limit - 1))  # Spread picks from the start of the log to the end of the log.
            for position in range(limit)
        }
    )
    return [examples[index] for index in indices]  # Return the selected representative examples in chronological order.


def _normalize_esp_ppq_target(target: str) -> str:
    """Map the user-facing chip target onto the value expected by ESP-PPQ."""

    normalized = target.strip().lower()  # Normalize casing and whitespace consistently.
    if normalized == "esp32":  # Follow Espressif's documented special case for classic ESP32.
        return "c"  # ESP-PPQ expects `c` because ESP32 operators are implemented in C.
    return normalized  # ESP32-S3 and ESP32-P4 use their regular chip-name targets directly.


def _require_mapping(payload: object, *, field_name: str) -> dict[str, object]:
    """Validate that `payload` is a JSON-like mapping and return it."""

    if not isinstance(payload, dict):  # Reject malformed JSON-like payloads early.
        raise ValueError(f"{field_name} must be a mapping")  # Explain the type contract clearly.
    return payload  # Narrow the type after validation.


def _require_list(payload: object, *, field_name: str) -> list[object]:
    """Validate that `payload` is a JSON-like list and return it."""

    if not isinstance(payload, list):  # Reject malformed JSON-like payloads early.
        raise ValueError(f"{field_name} must be a list")  # Explain the type contract clearly.
    return payload  # Narrow the type after validation.


def _require_float_list(payload: object, *, field_name: str) -> list[float]:
    """Validate that `payload` is a list-like collection of floats."""

    return [float(value) for value in _require_list(payload, field_name=field_name)]  # Convert every numeric-looking entry into a real float.


def _require_string_list(payload: object, *, field_name: str) -> list[str]:
    """Validate that `payload` is a list-like collection of strings."""

    return [str(value) for value in _require_list(payload, field_name=field_name)]  # Convert every entry into a plain string.


def _require_string_mapping(payload: object, *, field_name: str) -> dict[str, str]:
    """Validate that `payload` is a mapping of strings to strings."""

    mapping = _require_mapping(payload, field_name=field_name)  # Validate that the payload is a mapping first.
    return {str(key): str(value) for key, value in mapping.items()}  # Normalize every key/value into a plain string.


def _require_positive_int(payload: object, *, field_name: str) -> int:
    """Validate that `payload` is a positive integer-like value."""

    value = int(payload)  # Convert integer-like JSON values into a real Python int.
    if value < 1:  # Reject zero and negative values.
        raise ValueError(f"{field_name} must be positive")  # Explain the lower bound clearly.
    return value  # Return the validated positive integer.


def _require_onnx() -> None:
    """Raise a clear error when ONNX export dependencies are missing."""

    if importlib.util.find_spec("onnx") is None:  # Detect whether the ONNX package is importable.
        raise ImportError("PyTorch ONNX export requires the `onnx` package. Install it before using learning.export.")  # Explain the missing dependency clearly.
    if importlib.util.find_spec("onnxscript") is None:  # Detect whether the Torch ONNX exporter support package is importable.
        raise ImportError(
            "PyTorch ONNX export also requires `onnxscript` with current torch releases. Install it before using learning.export."
        )  # Explain the missing dependency clearly.


def _require_esp_ppq() -> None:
    """Raise a clear error when ESP-PPQ dependencies are missing."""

    if importlib.util.find_spec("ppq") is None:  # Detect whether the ESP-PPQ package is importable.
        raise ImportError(
            "ESP-DL export requires `esp-ppq` (import path `ppq`). Install it before requesting `.espdl` export."
        )  # Explain the missing dependency clearly.


if __name__ == "__main__":  # Allow running this file directly as a script.
    main()  # Execute the CLI entry point.
