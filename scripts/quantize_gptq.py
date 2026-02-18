#!/usr/bin/env python3
"""Quantize a fine-tuned causal LM with AutoGPTQ (e.g., 4-bit W4G128)."""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


def load_calibration_texts(path: Path, max_samples: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Calibration JSON must be a list of objects with a 'text' field")
    texts = [str(item.get("text", "")).strip() for item in data]
    texts = [t for t in texts if t]
    random.seed(0)
    random.shuffle(texts)
    return texts[:max_samples]


def prepare_examples(texts: List[str], tokenizer: AutoTokenizer, max_length: int) -> List[Dict[str, List[int]]]:
    examples: List[Dict[str, List[int]]] = []
    for text in texts:
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = tokenized.get("input_ids")
        attention_mask = tokenized.get("attention_mask")
        if not input_ids or not attention_mask:
            continue
        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
    if not examples:
        raise ValueError("No valid calibration examples after tokenization")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to merged fine-tuned model")
    parser.add_argument("--output-path", required=True, help="Directory to save the quantized model")
    parser.add_argument(
        "--calibration-file",
        default="finetuning_scripts/merged_counsel_dataset.json",
        help="JSON file with calibration conversations",
    )
    parser.add_argument("--calibration-samples", type=int, default=128, help="Number of calibration texts")
    parser.add_argument("--max-length", type=int, default=512, help="Token length for calibration")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bit-width")
    parser.add_argument("--group-size", type=int, default=128, help="GPTQ group size")
    parser.add_argument(
        "--desc-act",
        action="store_true",
        help="Enable desc_act for GPTQ (defaults to False)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size")
    parser.add_argument("--use-triton", action="store_true", help="Enable Triton kernels if available")
    return parser.parse_args()


def deduplicate_tied_weights(model_dir: Path) -> None:
    safetensor_files = sorted(model_dir.glob("gptq_model-*.safetensors"))
    if not safetensor_files:
        return

    lm_head_key = "lm_head.weight"
    embed_key = "model.embed_tokens.weight"

    for tensor_path in safetensor_files:
        weights = load_file(str(tensor_path))
        if lm_head_key not in weights or embed_key not in weights:
            continue

        lm_head = weights[lm_head_key]
        embed = weights[embed_key]
        if lm_head.shape != embed.shape:
            continue

        if torch.equal(lm_head, embed):
            weights.pop(lm_head_key)
            tmp_path = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
            save_file(weights, str(tmp_path))
            tmp_path.replace(tensor_path)
            print(f"Removed duplicated lm_head weights from {tensor_path.name}")
        break


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_path).resolve()
    calib_path = Path(args.calibration_file).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    texts = load_calibration_texts(calib_path, args.calibration_samples)
    examples = prepare_examples(texts, tokenizer, args.max_length)

    quant_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=True,
        true_sequential=True,
        damp_percent=0.01,
    )

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    model.quantize(
        examples,
        batch_size=args.batch_size,
        use_triton=args.use_triton,
        use_cuda_fp16=torch.cuda.is_available(),
        cache_examples_on_gpu=torch.cuda.is_available(),
    )

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    deduplicate_tied_weights(output_path)


if __name__ == "__main__":
    main()
