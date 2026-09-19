from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnx
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the robust classifier to portable edge formats.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


class TemperatureWrapper(nn.Module):
    def __init__(self, model: nn.Module, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images) / self.temperature


def benchmark(model, example: torch.Tensor, iterations: int = 80) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(example)
        started = time.perf_counter()
        for _ in range(iterations):
            model(example)
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    wrapper = TemperatureWrapper(model, checkpoint.get("temperature", 1.0)).eval()
    example = torch.zeros(1, 3, checkpoint.get("image_size", 224), checkpoint.get("image_size", 224))

    onnx_path = args.output_dir / "waste_bin_robust_v2.onnx"
    torch.onnx.export(
        wrapper,
        example,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized_wrapper = TemperatureWrapper(quantized, checkpoint.get("temperature", 1.0)).eval()
    quantized_path = args.output_dir / "waste_bin_robust_v2_dynamic_int8.torchscript.pt"
    torch.jit.trace(quantized_wrapper, example).save(str(quantized_path))

    float_script = args.output_dir / "waste_bin_robust_v2.torchscript.pt"
    report = {
        "onnx": {"path": str(onnx_path.resolve()), "bytes": onnx_path.stat().st_size, "checker_passed": True, "opset": 17},
        "dynamic_int8_torchscript": {"path": str(quantized_path.resolve()), "bytes": quantized_path.stat().st_size},
        "float_checkpoint_bytes": args.checkpoint.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cpu_single_view_inference_ms": {
            "float": benchmark(wrapper, example),
            "dynamic_int8": benchmark(quantized_wrapper, example),
        },
        "deployment_note": "For the i.MX93 Ethos-U65 NPU, use the ONNX model as an interchange artifact, then perform full-integer post-training quantization with representative camera frames, convert to fully quantized TFLite with NXP eIQ, and compile it with the Arm Vela flow. Dynamic TorchScript INT8 is a CPU reference, not an Ethos-U65 artifact.",
    }
    (args.output_dir / "edge_export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
