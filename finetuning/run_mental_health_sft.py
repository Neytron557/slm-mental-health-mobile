#!/usr/bin/env python3
"""
LoRA SFT driver for the unified mental-health corpus.

Expected dataset layout (produced by prepare_mental_health_corpus.py):
    intern_summer/data/mental_health_corpus/train.jsonl
    intern_summer/data/mental_health_corpus/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
torch.optim.AdamW.train = lambda self: None  # accelerate wrapper expects optimizers to expose train()
torch.optim.AdamW.eval = lambda self: None  # accelerate wrapper expects optimizers to expose eval()
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "intern_summer" / "data" / "mental_health_corpus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a base model on the mental-health SFT corpus with LoRA.")
    parser.add_argument("--model", required=True, help="Path or HF ID of the base model.")
    parser.add_argument("--train-file", type=Path, default=CORPUS_ROOT / "train.jsonl", help="Path to train JSONL.")
    parser.add_argument("--val-file", type=Path, default=CORPUS_ROOT / "val.jsonl", help="Path to validation JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length for training.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluation frequency (steps).")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint save frequency (steps).")
    parser.add_argument("--logging-steps", type=int, default=20, help="Logging frequency (steps).")
    parser.add_argument("--save-total-limit", type=int, default=2, help="How many checkpoints to keep.")
    parser.add_argument("--lr-scheduler", default="cosine", help="LR scheduler type for Transformers TrainingArguments.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max training steps (overrides epochs).")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for training.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--target-modules", nargs="*", default=None, help="Optional custom LoRA target modules.")
    return parser.parse_args()


def formatting_func(example: Dict[str, str]) -> str:
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    return f"{prompt}\n\nAssistant: {response}"


def build_bnb_config(disable_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if disable_4bit:
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_datasets(train_file: Path, val_file: Path):
    data_files = {"train": str(train_file)}
    if val_file and val_file.exists():
        data_files["validation"] = str(val_file)
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"]
    eval_ds = ds.get("validation")
    return train_ds, eval_ds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = load_datasets(args.train_file, args.val_file)
    train_ds = train_ds.map(lambda ex: {"text": formatting_func(ex)})
    keep_cols = {"text"}
    drop_cols = [col for col in train_ds.column_names if col not in keep_cols]
    if drop_cols:
        train_ds = train_ds.remove_columns(drop_cols)
    if eval_ds is not None:
        eval_ds = eval_ds.map(lambda ex: {"text": formatting_func(ex)})
        drop_cols = [col for col in eval_ds.column_names if col not in keep_cols]
        if drop_cols:
            eval_ds = eval_ds.remove_columns(drop_cols)

    bnb_config = build_bnb_config(args.no_4bit)
    compute_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=None if bnb_config else compute_dtype,
    )
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = args.target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    eval_strategy = "steps" if eval_ds is not None else "no"
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_ds is not None,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        max_seq_length=args.max_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "model": args.model,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "output_dir": str(args.output_dir),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "use_4bit": not args.no_4bit,
    }
    with (args.output_dir / "finetune_args.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
