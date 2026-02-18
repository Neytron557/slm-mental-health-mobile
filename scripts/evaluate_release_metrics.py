#!/usr/bin/env python3
"""Compute BERTScore, classification F1, empathy, and toxicity metrics for release models."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:  # optional dependencies used for specific quantization schemes
    from awq import AutoAWQForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    AutoAWQForCausalLM = None

try:  # optional when adapters are needed (not used here but kept for compatibility)
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover
    PeftModel = None

from bert_score import score as bertscore
from transformers import AutoModelForSequenceClassification

REPO_ROOT = Path(__file__).resolve().parent
INTERN_ROOT = REPO_ROOT / "intern_summer"
DEFAULT_MODELS_DIR = INTERN_ROOT / "models" / "release_models"
BERT_DATA_PATH = INTERN_ROOT / "generation_results.csv"
DREDDIT_PATH = INTERN_ROOT / "Dreddit_preds.csv"
SDCNL_PATH = INTERN_ROOT / "SDCNL_preds.csv"
BERT_PROMPT_LIMIT = 30
BERTSCORE_MODEL_TYPE = "bert-base-uncased"
BERTSCORE_DEVICE = "cuda"

DEFAULT_TEXT_BATCH_SIZE = 8 if torch.cuda.is_available() else 2
DEFAULT_CLASS_BATCH_SIZE = 32 if torch.cuda.is_available() else 4
ENCODING_MAX_LENGTH = 768
DEFAULT_OUTPUT_DIR = INTERN_ROOT / "metrics_outputs"
FORCE_FP32_MODELS = {"gemma-3-270m-it", "gemma-3-270m-it-finetuned"}
FORCE_4BIT_MODELS = {"ibm-granite_granite-4.0-h-micro", "ibm-granite_granite-4.0-h-micro-finetuned-bnb4"}
LABEL_SYNONYMS = {
    "suicidal": ("suicide", "suicidality"),
    "depression": ("depressed", "depressive"),
    "yes": ("yeah", "y", "affirmative"),
    "no": ("nah", "negative"),
}
MODEL_OVERRIDES = {
    "ibm-granite_granite-4.0-h-micro": {
        "text_batch_size": 1,
        "class_batch_size": 2,
        "max_input_length": 128,
        "bert_max_new_tokens": 12,
        "class_max_new_tokens": 2,
        "use_cache_text": False,
        "use_cache_class": False,
    },
    "ibm-granite_granite-4.0-h-micro-finetuned-bnb4": {
        "text_batch_size": 1,
        "class_batch_size": 2,
        "max_input_length": 128,
        "bert_max_new_tokens": 12,
        "class_max_new_tokens": 2,
        "use_cache_text": False,
        "use_cache_class": False,
    },
}
BERT_MAX_NEW_TOKENS = 128
CLASS_MAX_NEW_TOKENS = 6
POS_LABEL_DREDDIT = "yes"
POS_LABEL_SDCNL = "suicidal"

EMPATHY_MODEL = "bdotloh/roberta-base-empathy"
TOXICITY_MODEL = "unitary/toxic-bert"

@dataclass
class GenerationResult:
    prompts: List[str]
    references: List[str]

@dataclass
class ClassificationDataset:
    prompts: List[str]
    labels: List[str]


def load_csv_text_pairs(path: Path) -> GenerationResult:
    prompts, references = [], []
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompts.append(row["input"].strip())
            references.append(row["output"].strip())
    return GenerationResult(prompts, references)


def load_classification_dataset(path: Path) -> ClassificationDataset:
    prompts, labels = [], []
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompts.append(row["prompt"].strip())
            labels.append(row["gt"].strip().lower())
    return ClassificationDataset(prompts, labels)


def detect_quantization(model_path: Path) -> Dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    import json

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    quant_cfg = data.get("quantization_config") or {}
    quant_cfg["quant_method"] = (quant_cfg.get("quant_method") or quant_cfg.get("quant_method_name") or "none").lower()
    return quant_cfg


def resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


T = TypeVar("T")


def load_model_and_tokenizer(model_path: Path, force_cpu: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    use_gpu = torch.cuda.is_available() and not force_cpu
    force_fp32 = model_path.name in FORCE_FP32_MODELS

    quant_cfg = detect_quantization(model_path)
    quant_method = quant_cfg.get("quant_method", "none")
    effective_quant_method = quant_method if use_gpu else "none"

    base_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if use_gpu else None,
    }

    if use_gpu and model_path.name in FORCE_4BIT_MODELS:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            **base_kwargs,
        )
    elif effective_quant_method in {"awq", "autoawq", "awq-4bit"}:
        if AutoAWQForCausalLM is None:
            raise RuntimeError("autoawq is required to load AWQ quantized models")
        wrapper = AutoAWQForCausalLM.from_quantized(
            str(model_path),
            trust_remote_code=True,
            device_map=base_kwargs.get("device_map"),
            fuse_layers=False,
        )
        model = wrapper.model
    elif effective_quant_method in {"bitsandbytes", "bnb"} or (use_gpu and quant_cfg.get("load_in_4bit")):
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            **base_kwargs,
        )
    elif effective_quant_method == "gptq":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **base_kwargs,
        )
    else:
        dtype = torch.float16 if use_gpu and not force_fp32 else torch.float32
        if quant_cfg.get("torch_dtype") == "bfloat16" and torch.cuda.is_available() and not force_cpu:
            dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            **base_kwargs,
        )

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def chunk_list(items: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def maybe_format_prompt(tokenizer, prompt: str) -> str:
    """Apply chat template when available so chat models respond with assistant turns."""

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    template = getattr(tokenizer, "chat_template", None)
    if not apply_template or not template:
        return prompt
    try:
        return apply_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def prepare_batch(tokenizer, batch: List[str]) -> List[str]:
    return [maybe_format_prompt(tokenizer, item) for item in batch]


def generate_texts(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
    use_cache: bool = True,
    max_input_length: int = ENCODING_MAX_LENGTH,
) -> List[str]:
    device = resolve_device(model)
    results: List[str] = []
    bad_words = None
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        bad_words = [[tokenizer.pad_token_id]]
    for batch in chunk_list(prompts, batch_size):
        formatted = prepare_batch(tokenizer, batch)
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=model.config.eos_token_id,
                use_cache=use_cache,
                bad_words_ids=bad_words,
            )
        generated = outputs[:, encoded["input_ids"].shape[1]:]
        for seq in generated:
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            results.append(text)
    return results


def generate_labels(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
    use_cache: bool = True,
    max_input_length: int = ENCODING_MAX_LENGTH,
) -> List[str]:
    device = resolve_device(model)
    predictions: List[str] = []
    bad_words = None
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        bad_words = [[tokenizer.pad_token_id]]
    for batch in chunk_list(prompts, batch_size):
        formatted = prepare_batch(tokenizer, batch)
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=model.config.eos_token_id,
                use_cache=use_cache,
                bad_words_ids=bad_words,
            )
        generated = outputs[:, encoded["input_ids"].shape[1]:]
        for seq in generated:
            txt = tokenizer.decode(seq, skip_special_tokens=True).strip().lower()
            predictions.append(txt)
    return predictions


def _sum_label_logprobs_for_sequence(model, tokenizer, input_ids: torch.Tensor, label_ids: torch.Tensor) -> float:
    """Compute sum of log-probs for label_ids when appended to input_ids.

    input_ids: shape [1, L]
    label_ids: shape [1, T]
    """
    device = next(model.parameters()).device
    combo = torch.cat([input_ids, label_ids], dim=1).to(device)
    attn = torch.ones_like(combo)
    with torch.no_grad():
        outputs = model(input_ids=combo, attention_mask=attn)
        logits = outputs.logits  # [1, L+T, V]
        start = input_ids.shape[1]
        total = 0.0
        for pos in range(label_ids.shape[1]):
            idx = start + pos - 1 if pos > 0 else start - 1
            step_logits = logits[0, idx]
            step_lp = torch.log_softmax(step_logits, dim=-1)[label_ids[0, pos]]
            total += float(step_lp.item())
    return total


def classify_via_logprobs(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    candidates: Tuple[str, str] = ("suicidal", "depression"),
    max_input_length: int = ENCODING_MAX_LENGTH,
) -> List[str]:
    """Classify prompts by scoring log-prob of candidate labels.

    Returns a list of predicted labels (one of candidates) with same length as prompts.
    """
    device = resolve_device(model)
    # Pre-tokenize candidate variants (with and without leading space)
    cand_variants = {}
    for cand in candidates:
        ids_plain = tokenizer(cand, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        ids_spc = tokenizer(" " + cand, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        cand_variants[cand] = (ids_plain, ids_spc)

    preds: List[str] = []
    for batch in chunk_list(prompts, batch_size):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        input_ids_all = encoded["input_ids"].to(device)
        attn_all = encoded.get("attention_mask")
        if attn_all is not None:
            attn_all = attn_all.to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids_all, attention_mask=attn_all)

        # Score each sample individually (keeps memory predictable across quantization backends)
        for i in range(input_ids_all.shape[0]):
            input_ids = input_ids_all[i : i + 1]
            best_label = None
            best_score = float('-inf')
            for cand, (ids_plain, ids_spc) in cand_variants.items():
                s1 = _sum_label_logprobs_for_sequence(model, tokenizer, input_ids, ids_plain)
                s2 = _sum_label_logprobs_for_sequence(model, tokenizer, input_ids, ids_spc)
                score = max(s1, s2)
                if score > best_score:
                    best_score = score
                    best_label = cand
            preds.append(best_label or candidates[0])
    return preds


def run_with_batch_backoff(
    label: str,
    initial_batch_size: int,
    operation: Callable[[int], T],
) -> Tuple[T, int]:
    """Execute generation with retry-on-OOM logic that halves the batch size when needed."""

    current = initial_batch_size
    if current <= 0:
        raise ValueError("Batch size must be positive")

    while True:
        try:
            result = operation(current)
            return result, current
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message:
                raise
            if current == 1:
                raise
            next_batch = max(1, current // 2)
            print(
                f"CUDA OOM during {label}; retrying with batch size {next_batch}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current = next_batch


def extract_label(text: str, valid_labels: Tuple[str, ...]) -> str:
    text_lower = text.lower()
    # Strip common chat template prefixes before token analysis.
    text_lower = re.sub(r"\b(?:assistant|human|user|system)\s*[:>-]?", " ", text_lower)
    tokens = re.findall(r"[a-z]+", text_lower)
    if tokens:
        for token in tokens:
            for label in valid_labels:
                lcase = label.lower()
                if token == lcase or token.startswith(lcase):
                    return lcase
                for alias in LABEL_SYNONYMS.get(lcase, ()):  # type: ignore[arg-type]
                    if token == alias or token.startswith(alias):
                        return lcase
    for label in valid_labels:
        if label.lower() in text_lower:
            return label.lower()
        for alias in LABEL_SYNONYMS.get(label.lower(), ()):  # type: ignore[arg-type]
            if alias in text_lower:
                return label.lower()
    return "unknown"


def binary_f1(preds: List[str], refs: List[str], positive: str) -> float:
    tp = fp = fn = 0
    for pred, ref in zip(preds, refs):
        pred_pos = pred == positive
        ref_pos = ref == positive
        if pred_pos and ref_pos:
            tp += 1
        elif pred_pos and not ref_pos:
            fp += 1
        elif not pred_pos and ref_pos:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_empathy_model():
    tokenizer = AutoTokenizer.from_pretrained(EMPATHY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMPATHY_MODEL)
    model.eval()
    return model, tokenizer


def load_toxicity_model():
    tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL)
    model.eval()
    return model, tokenizer


def score_empathy(texts: List[str], model, tokenizer) -> float:
    device = resolve_device(model)
    scores: List[float] = []
    for batch in chunk_list(texts, 16):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, 0].tolist())  # label 0 == Empathy
    return float(sum(scores) / len(scores)) if scores else 0.0


def score_toxicity(texts: List[str], model, tokenizer) -> float:
    device = resolve_device(model)
    scores: List[float] = []
    for batch in chunk_list(texts, 16):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits)
            combined = 1.0 - torch.prod(1.0 - probs, dim=1)
        scores.extend(combined.tolist())
    return float(sum(scores) / len(scores)) if scores else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for one or more language models.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Base directory containing model subdirectories to evaluate.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Optional list of subdirectory names to skip inside --models-dir.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=None,
        help="Override batch size for generative text evaluation (defaults adapt to device).",
    )
    parser.add_argument(
        "--class-batch-size",
        type=int,
        default=None,
        help="Override batch size for classification-style generation (defaults adapt to device).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing aggregated metrics and start from scratch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where metrics and plots will be stored (defaults to metrics_outputs).",
    )
    return parser.parse_args()


def iter_model_paths(base_dir: Path, exclude: Iterable[str]) -> List[Path]:
    exclude_names = {Path(item).name for item in exclude}
    if not base_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {base_dir}")
    model_paths = []
    for candidate in sorted(base_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name in exclude_names:
            continue
        model_paths.append(candidate)
    return model_paths


def generate_metric_plots(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    metric_sections = ("bertscore", "f1", "qualitative")
    if not all(results.get(section) for section in metric_sections):
        print("Metrics incomplete; skipping plot generation.")
        return

    model_names = sorted(
        set(results["bertscore"].keys())
        & set(results["f1"].keys())
        & set(results["qualitative"].keys())
    )
    if not model_names:
        print("No evaluated models available for plotting.")
        return

    indices = range(len(model_names))

    bert_precision = [results["bertscore"][name]["precision"] for name in model_names]
    bert_recall = [results["bertscore"][name]["recall"] for name in model_names]
    bert_f1 = [results["bertscore"][name]["f1"] for name in model_names]
    dred_f1 = [results["f1"][name]["dreddit"] for name in model_names]
    sdcnl_f1 = [results["f1"][name]["sdcnl"] for name in model_names]
    empathy_scores = [results["qualitative"][name]["empathy"] for name in model_names]
    toxicity_scores = [results["qualitative"][name]["toxicity"] for name in model_names]

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_plot = output_dir / "metrics_summary.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    saved_paths = []

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    ax.bar([i - bar_width for i in indices], bert_precision, width=bar_width, label="Precision", color="#4C72B0")
    ax.bar(indices, bert_recall, width=bar_width, label="Recall", color="#55A868")
    ax.bar([i + bar_width for i in indices], bert_f1, width=bar_width, label="F1", color="#C44E52")
    ax.set_title("BERTScore Precision / Recall / F1")
    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "metrics_bertscore.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    saved_paths.append(plot_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    ax.bar([i - bar_width / 2 for i in indices], dred_f1, width=bar_width, label="Dreddit", color="#55A868")
    ax.bar([i + bar_width / 2 for i in indices], sdcnl_f1, width=bar_width, label="SDCNL", color="#C44E52")
    ax.set_title("Classification F1")
    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "metrics_f1.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    saved_paths.append(plot_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    ax.bar([i - bar_width / 2 for i in indices], empathy_scores, width=bar_width, label="Empathy", color="#8172B2")
    ax.bar([i + bar_width / 2 for i in indices], toxicity_scores, width=bar_width, label="Toxicity", color="#CCB974")
    ax.set_title("Empathy vs Toxicity")
    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "metrics_qualitative.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    saved_paths.append(plot_path)

    print("Saved metric plots:")
    for path in saved_paths:
        print(f" - {path}")


def main():
    args = parse_args()
    output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).resolve()
    results_path = output_dir / "multi_metric_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_batch_size(requested: int | None, default: int, label: str) -> int:
        value = requested if requested and requested > 0 else default
        if value <= 0:
            raise ValueError(f"{label} batch size must be positive")
        return value

    text_batch_size = resolve_batch_size(
        args.text_batch_size,
        DEFAULT_TEXT_BATCH_SIZE,
        "Text generation",
    )
    class_batch_size = resolve_batch_size(
        args.class_batch_size,
        DEFAULT_CLASS_BATCH_SIZE,
        "Classification",
    )
    print(
        "Using batch sizes -> text: %s, classification: %s"
        % (text_batch_size, class_batch_size)
    )

    text_data = load_csv_text_pairs(BERT_DATA_PATH)
    if BERT_PROMPT_LIMIT and len(text_data.prompts) > BERT_PROMPT_LIMIT:
        text_data = GenerationResult(
            prompts=text_data.prompts[:BERT_PROMPT_LIMIT],
            references=text_data.references[:BERT_PROMPT_LIMIT],
        )
    dred_dataset = load_classification_dataset(DREDDIT_PATH)
    sdcnl_dataset = load_classification_dataset(SDCNL_PATH)

    empathy_model, empathy_tok = load_empathy_model()
    toxicity_model, toxicity_tok = load_toxicity_model()

    results = {
        "bertscore": {},
        "f1": {},
        "qualitative": {},
    }
    if not args.fresh and results_path.exists():
        try:
            with results_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            for key in ("bertscore", "f1", "qualitative"):
                if isinstance(loaded.get(key), dict):
                    results[key].update(loaded[key])
        except Exception:
            pass

    model_paths = iter_model_paths(args.models_dir, args.exclude)
    models_in_run = [path.name for path in model_paths]

    # Drop stale metrics for models we are about to refresh.
    for key in ("bertscore", "f1", "qualitative"):
        for model_name in models_in_run:
            results[key].pop(model_name, None)

    failed_models_existing = []
    if not args.fresh and results_path.exists():
        try:
            with results_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            failed_models_existing = [
                name
                for name in loaded.get("failed_models", [])
                if name not in models_in_run
            ]
        except Exception:
            pass

    failed_models: List[str] = []
    for model_path in model_paths:
        model_name = model_path.name
        print(f"Evaluating {model_name}...")
        responses = dred_raw = sdcnl_raw = None
        dred_preds: List[str] = []
        sdcnl_preds: List[str] = []
        success = False

        overrides = MODEL_OVERRIDES.get(model_name, {})
        model_text_batch_size = overrides.get("text_batch_size", text_batch_size)
        model_class_batch_size = overrides.get("class_batch_size", class_batch_size)
        model_max_input_length = overrides.get("max_input_length", ENCODING_MAX_LENGTH)
        model_bert_tokens = overrides.get("bert_max_new_tokens", BERT_MAX_NEW_TOKENS)
        model_class_tokens = overrides.get("class_max_new_tokens", CLASS_MAX_NEW_TOKENS)

        for force_cpu in (False, True):
            try:
                model = tokenizer = None
                model, tokenizer = load_model_and_tokenizer(model_path, force_cpu=force_cpu)
                try:
                    try:
                        responses, _ = run_with_batch_backoff(
                            f"text generation for {model_name}",
                            model_text_batch_size,
                            lambda batch: generate_texts(
                                model,
                                tokenizer,
                                text_data.prompts,
                                model_bert_tokens,
                                batch,
                                max_input_length=model_max_input_length,
                                use_cache=overrides.get("use_cache_text", True),
                            ),
                        )
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        print(
                            f"Re-running text generation for {model_name} with cache disabled due to OOM..."
                        )
                        responses, _ = run_with_batch_backoff(
                            f"text generation for {model_name} (no cache)",
                            model_text_batch_size,
                            lambda batch: generate_texts(
                                model,
                                tokenizer,
                                text_data.prompts,
                                model_bert_tokens,
                                batch,
                                use_cache=False,
                                max_input_length=model_max_input_length,
                            ),
                        )

                    try:
                        dred_raw, class_batch_used = run_with_batch_backoff(
                            f"Dreddit classification for {model_name}",
                            model_class_batch_size,
                            lambda batch: generate_labels(
                                model,
                                tokenizer,
                                dred_dataset.prompts,
                                model_class_tokens,
                                batch,
                                max_input_length=model_max_input_length,
                                use_cache=overrides.get("use_cache_class", True),
                            ),
                        )
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        print(
                            f"Re-running Dreddit classification for {model_name} with cache disabled due to OOM..."
                        )
                        dred_raw, class_batch_used = run_with_batch_backoff(
                            f"Dreddit classification for {model_name} (no cache)",
                            model_class_batch_size,
                            lambda batch: generate_labels(
                                model,
                                tokenizer,
                                dred_dataset.prompts,
                                model_class_tokens,
                                batch,
                                use_cache=False,
                                max_input_length=model_max_input_length,
                            ),
                        )
                    dred_preds = [extract_label(txt, ("yes", "no")) for txt in dred_raw]

                    # Use log-prob based classification for SDCNL to avoid refusals/verbosity
                    sdcnl_preds, _ = run_with_batch_backoff(
                        f"SDCNL log-prob classification for {model_name}",
                        class_batch_used,
                        lambda batch: classify_via_logprobs(
                            model,
                            tokenizer,
                            sdcnl_dataset.prompts,
                            batch,
                            candidates=("suicidal", "depression"),
                            max_input_length=model_max_input_length,
                        ),
                    )
                    sdcnl_raw = sdcnl_preds[:]  # store predictions as raw outputs for record
                    success = True
                    break
                finally:
                    # Free memory between attempts.
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as exc:  # pylint: disable=broad-except
                is_oom = isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
                if not force_cpu and is_oom:
                    print(f"GPU OOM while evaluating {model_name}; retrying on CPU...")
                    continue
                print(f"Failed to evaluate {model_name}: {exc}")
                break

        if not success or responses is None or dred_raw is None or sdcnl_raw is None:
            failed_models.append(model_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        device_arg = BERTSCORE_DEVICE if torch.cuda.is_available() else "cpu"
        P, R, F = bertscore(
            responses,
            text_data.references,
            lang="en",
            model_type=BERTSCORE_MODEL_TYPE,
            device=device_arg,
        )
        results["bertscore"][model_name] = {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F.mean().item()),
        }

        results["f1"][model_name] = {
            "dreddit": binary_f1(dred_preds, dred_dataset.labels, POS_LABEL_DREDDIT),
            "sdcnl": binary_f1(sdcnl_preds, sdcnl_dataset.labels, POS_LABEL_SDCNL),
        }

        empathy_score = score_empathy(responses, empathy_model, empathy_tok)
        toxicity_score = score_toxicity(responses, toxicity_model, toxicity_tok)
        results["qualitative"][model_name] = {
            "empathy": empathy_score,
            "toxicity": toxicity_score,
        }

        output_dump = output_dir / f"{model_name}_outputs.json"
        with output_dump.open("w", encoding="utf-8") as handle:
            json.dump({
                "responses": responses,
                "dreddit_raw": dred_raw,
                "sdcnl_raw": sdcnl_raw,
            }, handle, ensure_ascii=False, indent=2)

        # Clean up per-model intermediate collections to limit memory footprint.
        del responses, dred_raw, sdcnl_raw

    combined_failed = failed_models_existing + failed_models
    if combined_failed:
        results["failed_models"] = combined_failed
    elif "failed_models" in results:
        results.pop("failed_models", None)

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f"Saved aggregated metrics to {results_path}")

    generate_metric_plots(results, output_dir)


if __name__ == "__main__":
    main()
