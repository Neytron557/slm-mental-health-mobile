#!/usr/bin/env python3
"""Quantize a merged model with PyTorch AO dynamic INT8."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to the merged float model")
    parser.add_argument("--output-path", required=True, help="Directory for the quantized checkpoint")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=["Linear"],
        help="Module class names to quantize dynamically (default: Linear)",
    )
    return parser.parse_args()


def resolve_modules(names: list[str]) -> set[type[torch.nn.Module]]:
    mapping = {}
    for name in names:
        if not hasattr(torch.nn, name):
            raise ValueError(f"torch.nn has no module named {name}")
        module_cls = getattr(torch.nn, name)
        mapping[module_cls] = None
    return set(mapping.keys())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = resolve_modules(args.modules)

    print(f"Loading base model from {args.model_path} in float32 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    print("Applying torch.ao.quantization.quantize_dynamic...")
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        modules,
        dtype=torch.qint8,
    )

    print("Saving quantized model to", output_dir)
    quantized.save_pretrained(output_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("PyTorch AO dynamic quantization complete.")


if __name__ == "__main__":
    main()
