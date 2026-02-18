#!/usr/bin/env python3
"""Quantize a merged LoRA model with AutoAWQ (e.g., 4-bit GEMM)."""
import argparse
import json
import random
from pathlib import Path
from typing import List

from awq import AutoAWQForCausalLM
import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to merged fp16/fp32 model")
    parser.add_argument("--output-path", required=True, help="Directory to store quantized model")
    parser.add_argument("--w-bit", type=int, default=4, help="Weight bit-width")
    parser.add_argument("--q-group-size", type=int, default=128, help="Group size for AWQ")
    parser.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV"], help="AWQ kernel version")
    parser.add_argument("--zero-point", action="store_true", help="Enable per-group zero points")
    parser.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--calibration-file", default="../finetuning_scripts/merged_counsel_dataset.json", help="JSON file with conversation texts")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokens per calibration example")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_path = Path(args.calibration_file).resolve()
    if calib_path.exists():
        with calib_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts: List[str] = []
        for item in data:
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)
        if not texts:
            raise ValueError("Calibration file did not contain any non-empty 'text' fields")
        random.seed(0)
        random.shuffle(texts)
        calib_texts = texts[: args.calib_samples]
        print(f"Loaded {len(calib_texts)} calibration texts from {calib_path}")
    else:
        calib_texts = None
        print(f"Calibration file {calib_path} not found. Falling back to AWQ built-in calibration dataset.")

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    print(f"Loading model from {args.model_path} for AWQ quantization...")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    print("Running AWQ quantization...")
    calib_data = calib_texts if calib_texts is not None else "pile"
    # Older AutoAWQ interfaces accept calib_data without explicit sample cap.
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
    )

    print("Saving AWQ model to", output_dir)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print("AWQ quantization complete.")


if __name__ == "__main__":
    main()
