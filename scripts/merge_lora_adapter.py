#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save standalone weights."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Path to the original base model")
    parser.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--output", required=True, help="Directory to write the merged model")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Precision to load the base model before merging (float32 is safest, float16 saves memory)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to from_pretrained (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    print(f"🔄 Loading base model from {args.base_model} ({args.dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    print(f"🔧 Loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("🧬 Merging LoRA weights into the base model (this may take a moment)...")
    merged_model = model.merge_and_unload()

    print("💾 Saving merged model to", args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merged_model.to("cpu")
    merged_model.save_pretrained(args.output)

    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print("✅ Done. The directory can now be used as a standalone model.")


if __name__ == "__main__":
    main()
