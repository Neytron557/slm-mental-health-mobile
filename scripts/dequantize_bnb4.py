#!/usr/bin/env python3
"""Convert a bitsandbytes 4-bit model directory back to full-precision weights."""
import argparse
from pathlib import Path

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import QuantState, dequantize_nf4
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to the existing 4-bit model")
    parser.add_argument("--output-path", required=True, help="Directory to save the recovered fp16 model")
    return parser.parse_args()


def convert_linear4bit(module: bnb.nn.Linear4bit) -> torch.nn.Linear:
    state = module.state_dict()
    quant_inputs = {
        key.split("weight.", 1)[1]: value
        for key, value in state.items()
        if key != "weight"
    }
    quant_state = QuantState.from_dict(quant_inputs, device=torch.device("cpu"))
    weight = dequantize_nf4(state["weight"], quant_state=quant_state).to(torch.float16)
    out_features, in_features = weight.shape
    converted = torch.nn.Linear(
        in_features,
        out_features,
        bias=module.bias is not None,
        dtype=torch.float16,
    )
    converted.weight.data.copy_(weight)
    if module.bias is not None:
        converted.bias.data.copy_(module.bias.data.to(torch.float16))
    return converted


def recursively_convert(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, bnb.nn.Linear4bit):
            setattr(module, name, convert_linear4bit(child))
        else:
            recursively_convert(child)


def main() -> None:
    args = parse_args()
    source = Path(args.model_path)
    target = Path(args.output_path)
    target.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(str(source), device_map="cpu")

    recursively_convert(model)

    model.save_pretrained(target)

    print("Done")


if __name__ == "__main__":
    main()
