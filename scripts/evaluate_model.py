#!/usr/bin/env python
"""Evaluate causal language models on a fixed mental-health prompt.

The script reports model storage size, load time, latency to first token,
full generation latency, and throughput. Results are appended to a JSONL file
for easy comparison across variants.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from bert_score import score as bertscore
try:
    from awq import AutoAWQForCausalLM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoAWQForCausalLM = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None

try:
    from optimum.bettertransformer import BetterTransformer
except Exception:  # pragma: no cover - optional dependency or version mismatch
    BetterTransformer = None


def get_directory_size_mb(path: Path) -> float:
    total_bytes = 0
    for file_path in path.rglob("*"):
        if file_path.is_symlink():  # avoid double counting quantized weights via symlinks
            continue
        if file_path.is_file():
            total_bytes += file_path.stat().st_size
    return total_bytes / (1024 ** 2)


def resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_quant_config(args: argparse.Namespace):
    if args.load_in_4bit or args.load_in_8bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes is required for 4/8-bit loading but is not installed.")
        return BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return None


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def detect_quantization(model_path: Path):
    config_path = model_path / "config.json"
    config = load_json(config_path)
    if not config:
        return {}
    quant_config = config.get("quantization_config") or {}
    quant_method = quant_config.get("quant_method") or quant_config.get("quant_method_name")
    quant_config["quant_method"] = quant_method
    return quant_config


def load_base_model(args, tokenizer):
    model_path = Path(args.model_path)
    quant_config = detect_quantization(model_path)
    quant_method = (quant_config.get("quant_method") or "none").lower()

    base_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.attn_impl:
        base_kwargs["attn_implementation"] = args.attn_impl

    def resolve_dtype():
        if args.dtype == "float32":
            return torch.float32
        if args.dtype == "bfloat16":
            return torch.bfloat16
        if args.dtype == "float16":
            return torch.float16

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            cfg_dtype = getattr(config, "torch_dtype", None)
        except Exception:
            cfg_dtype = None

        if isinstance(cfg_dtype, str):
            cfg_dtype = getattr(torch, cfg_dtype, None)

        if cfg_dtype is torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            cfg_dtype = torch.float16

        if cfg_dtype is not None:
            return cfg_dtype

        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    preferred_dtype = resolve_dtype()

    if quant_method in {"awq", "autoawq", "awq-4bit"}:
        if AutoAWQForCausalLM is None:
            raise RuntimeError("autoawq is required to load AWQ quantized models")
        wrapper = AutoAWQForCausalLM.from_quantized(
            str(model_path),
            trust_remote_code=True,
            device_map=base_kwargs.get("device_map"),
            fuse_layers=False,
        )
        model = wrapper.model
    elif quant_method in {"bitsandbytes", "bnb"} or quant_config.get("load_in_4bit"):
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes is required to load 4-bit models")
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            **base_kwargs,
        )
    elif quant_method == "gptq":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            **base_kwargs,
        )
    else:
        model_kwargs = dict(base_kwargs)
        if preferred_dtype is not None:
            model_kwargs["torch_dtype"] = preferred_dtype
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **model_kwargs,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    if args.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required when providing --adapter-path")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        if args.merge_adapter:
            model = model.merge_and_unload()
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM latency and throughput on a fixed prompt")
    parser.add_argument("--model-path", required=True, help="Path to the model directory")
    parser.add_argument("--variant-name", required=True, help="Human-readable tag for the evaluated variant")
    parser.add_argument("--output", default="scripts/eval_results.jsonl", help="JSONL file to append metrics")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument("--adapter-path", help="Path to a LoRA adapter to load on top of the base model")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge LoRA weights into the base model before evaluation")
    parser.add_argument("--load-in-4bit", action="store_true", help="Deprecated flag (kept for compatibility)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Deprecated flag (kept for compatibility)")
    parser.add_argument(
        "--attn-impl",
        choices=["sdpa", "flash_attention_2", "eager", "auto"],
        default=None,
        help="Attention backend to request when loading the model",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
        help="Preferred floating point dtype for non-quantized models",
    )
    parser.add_argument("--use-compile", action="store_true", help="Apply torch.compile for inference")
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode to use when --use-compile is set",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Enable dynamic shape support when compiling",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 kernels for matmul/convolution on Ampere GPUs",
    )
    parser.add_argument(
        "--use-awq-loader",
        action="store_true",
        help="Load the model with AutoAWQForCausalLM.from_quantized",
    )
    parser.add_argument("--use-bettertransformer", action="store_true", help="Convert to BetterTransformer after loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--reference-text",
        help="Optional reference response used to compute BERTScore precision / recall / F1.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        help="Path to a text file containing the reference response for BERTScore.",
    )
    args = parser.parse_args()

    reference_text: str | None = None
    if args.reference_path is not None:
        if not args.reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {args.reference_path}")
        reference_text = args.reference_path.read_text(encoding="utf-8").strip()
    elif args.reference_text:
        reference_text = args.reference_text.strip()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    size_mb = get_directory_size_mb(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    load_start = time.perf_counter()
    model = load_base_model(args, tokenizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_time = time.perf_counter() - load_start

    if args.use_bettertransformer:
        if BetterTransformer is None:
            raise RuntimeError("optimum is required for BetterTransformer but is not installed.")
        model = BetterTransformer.transform(model)

    if args.use_compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=True, dynamic=args.compile_dynamic)

    model.eval()

    device = resolve_device(model)

    messages = [
        {
            "role": "system",
            "content": (
            (
                "You are a supportive, non-judgmental mental-health assistant. "
                "Do NOT diagnose, prescribe, or claim to replace professional care. Avoid clinical labels. "
                "If you detect possible risk of self-harm, harm to others, or abuse: acknowledge feelings briefly, "
                "encourage contacting trusted people or emergency services, offer crisis resources if a location is shared. "
                "Otherwise reflect the user's emotion in one short sentence, ask one gentle open question, and offer 2-3 "
                "low-risk coping ideas like grounding, breathing, or journaling. Style warm, plain language, no absolutes, max ~120 tokens. "
                "When explicitly asked to classify from a fixed set of labels, reply with only the chosen label."
            )
        ),
    },
    {
        "role": "user",
        "content": (
            "I've been feeling completely overwhelmed lately. No matter how much I try to keep up with everything, I always feel like I'm falling behind. It's starting to affect my sleep and my relationships. What should I do?"
        ),
    },
    ]

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        prompt = (
            f"{system_prompt}\n\n"
            f"User: {user_prompt}\n\n"
            "Assistant:"
        )
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Warm-up pass to reduce compilation overhead for fairer measurements.
    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        first_start = time.perf_counter()
        first_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        first_token_time = time.perf_counter() - first_start

    first_token_id = first_output[0, input_ids.shape[1]]
    first_token_text = tokenizer.decode(first_token_id)

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_start = time.perf_counter()
        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - gen_start

    sequences = gen_output.sequences
    generated_tokens = sequences[0, input_ids.shape[1]:]
    num_new_tokens = generated_tokens.numel()
    throughput = num_new_tokens / gen_time if gen_time > 0 else float("inf")
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    bertscore_metrics = None
    if reference_text:
        candidate = generated_text.strip()
        reference = reference_text.strip()
        if candidate and reference:
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
            P, R, F = bertscore([
                candidate,
            ], [
                reference,
            ], lang="en", model_type=BERTSCORE_MODEL_TYPE, device=device_arg)
            bertscore_metrics = {
                "precision": round(float(P.mean().item()), 6),
                "recall": round(float(R.mean().item()), 6),
                "f1": round(float(F.mean().item()), 6),
            }
        else:
            bertscore_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    result = {
        "variant": args.variant_name,
        "model_path": str(model_path),
        "size_mb": round(size_mb, 2),
        "load_time_s": round(load_time, 4),
        "first_token_time_s": round(first_token_time, 4),
        "inference_time_s": round(gen_time, 4),
        "throughput_toks_per_s": round(throughput, 2),
        "first_token": first_token_text,
        "generated_text": generated_text.strip(),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
            "use_compile": args.use_compile,
            "use_bettertransformer": args.use_bettertransformer,
        },
    }

    if bertscore_metrics is not None:
        result["bertscore"] = bertscore_metrics

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    if bertscore_metrics is not None:
        print(
            "BERTScore -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}".format(
                **bertscore_metrics
            )
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
BERTSCORE_MODEL_TYPE = "bert-base-uncased"
