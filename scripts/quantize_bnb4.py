#!/usr/bin/env python3
"""Quantize a merged LoRA model to 4-bit bitsandbytes format (NF4)."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to the merged full-precision model")
    parser.add_argument("--output-path", required=True, help="Directory to save the 4-bit model")
    parser.add_argument("--quant-type", default="nf4", choices=["nf4", "fp4"], help="Quantization type for 4-bit")
    parser.add_argument("--double-quant", action="store_true", help="Enable double quantization")
    parser.add_argument("--compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Computation dtype for dequantized matmuls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading model from {args.model_path} with bitsandbytes 4-bit ({args.quant_type})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print("Saving 4-bit quantized weights...")
    model.save_pretrained(output_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("bitsandbytes 4-bit quantization complete.")


if __name__ == "__main__":
    main()
