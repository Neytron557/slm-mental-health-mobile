#!/usr/bin/env python3
"""
Build a unified mental-health SFT corpus that mixes counseling dialogs and
classification-style prompts aligned with our evaluation suites.

Sources:
- nbertagnolli/counsel-chat, andreshere/counsel_chat
- Amod/mental_health_counseling_conversations
- Sulav/mental_health_counseling_conversations_sharegpt
- alexandreteles/mental-health-conversational-data
- heliosbrahma/mental_health_chatbot_dataset
- asmaab/dreaddit (train+validation only)
- jingjietan/sdcnl-suicide (train+validation only)
- thisiskeithkwan/Reddit_C-SSRS (train split)
- SahandSab/EmoBench (emotional_understanding)
- google-research-datasets/go_emotions (train split, simplified config)
- local iclab_dataset/train.csv (depression yes/no)
- local optimized_train.csv (7-way emotion classification prompts)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "intern_summer" / "data" / "mental_health_corpus"


@dataclass
class Example:
    prompt: str
    response: str
    source: str
    task: str


def dataset_slice(ds: Dataset, cap: int | None, rng: random.Random) -> Iterable[dict]:
    if cap is None or cap <= 0 or cap >= len(ds):
        for row in ds:
            yield row
        return
    indices = rng.sample(range(len(ds)), cap)
    for idx in indices:
        yield ds[int(idx)]


def trim_pair(prompt: str, response: str, max_chars: int) -> tuple[str, str] | None:
    if max_chars <= 0:
        return prompt, response
    text = f"{prompt}\n{response}"
    if len(text) > max_chars:
        return None
    return prompt, response


def load_counsel_chat(rng: random.Random, max_chars: int, cap: int | None) -> List[Example]:
    examples: List[Example] = []
    template = (
        "You are an empathetic mental health counselor. Provide a concise, supportive reply to the client.\n\n"
        "Client:\n{question}\n\nAssistant:"
    )
    datasets = [
        ("nbertagnolli/counsel-chat", "train", "counsel_chat_nbertagnolli"),
        ("andreshere/counsel_chat", "train", "counsel_chat_andreshere"),
    ]
    for name, split, source in datasets:
        ds = load_dataset(name, split=split)
        for row in dataset_slice(ds, cap, rng):
            question = row.get("questionText") or row.get("Context") or ""
            answer = row.get("answerText") or row.get("Response") or ""
            prompt = template.format(question=question.strip())
            pair = trim_pair(prompt, answer.strip(), max_chars)
            if pair:
                examples.append(Example(prompt=pair[0], response=pair[1], source=source, task="counseling"))
    return examples


def load_mental_health_counseling_sets(
    rng: random.Random, max_chars: int, cap: int | None
) -> List[Example]:
    examples: List[Example] = []
    template = (
        "You are a compassionate therapist. Read the client's note and respond with empathy and actionable support.\n\n"
        "Client:\n{context}\n\nAssistant:"
    )
    datasets = [
        ("Amod/mental_health_counseling_conversations", "train", "amod_counseling"),
        ("Sulav/mental_health_counseling_conversations_sharegpt", "train", "sharegpt_counseling"),
        ("alexandreteles/mental-health-conversational-data", "train", "alexandre_counseling"),
    ]
    for name, split, source in datasets:
        ds = load_dataset(name, split=split)
        for row in dataset_slice(ds, cap, rng):
            context = row.get("Context") or row.get("Context:") or row.get("Context ")
            if not context:
                context = row.get("text") or ""
            response = row.get("Response") or row.get("text") or ""
            prompt = template.format(context=str(context).strip())
            pair = trim_pair(prompt, str(response).strip(), max_chars)
            if pair:
                examples.append(Example(prompt=pair[0], response=pair[1], source=source, task="counseling"))

    tiny_ds = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")
    for row in dataset_slice(tiny_ds, cap, rng):
        text = str(row.get("text", "")).strip()
        if "<HUMAN>:" in text and "<ASSISTANT>:" in text:
            human, assistant = text.split("<ASSISTANT>:", maxsplit=1)
            prompt = human.replace("<HUMAN>:", "Client:").strip()
            pair = trim_pair(prompt, assistant.strip(), max_chars)
            if pair:
                examples.append(Example(prompt=pair[0], response=pair[1], source="heliosbrahma_qa", task="counseling"))
    return examples


def load_dreaddit(rng: random.Random, max_chars: int, cap: int | None) -> List[Example]:
    ds = load_dataset("asmaab/dreaddit")
    merged = concatenate_datasets([ds["train"], ds["validation"]])
    examples: List[Example] = []
    for row in dataset_slice(merged, cap, rng):
        text = str(row["text"]).strip()
        label = "stressed" if int(row["label"]) == 1 else "not stressed"
        prompt = (
            "Determine whether the Reddit post indicates stress. "
            "Reply with 'stressed' or 'not stressed'.\n\n"
            f"Post:\n{text}\n\nAnswer:"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="dreaddit", task="stress_classification"))
    return examples


def load_sdcnl(rng: random.Random, max_chars: int, cap: int | None) -> List[Example]:
    ds = load_dataset("jingjietan/sdcnl-suicide")
    merged = concatenate_datasets([ds["train"], ds["validation"]])
    examples: List[Example] = []
    for row in dataset_slice(merged, cap, rng):
        text = str(row["selftext"]).strip()
        label = "suicidal" if int(row["is_suicide"]) == 1 else "depression"
        prompt = (
            "Classify the post. Use 'suicidal' when the writer expresses intent or desire to end their life. "
            "Use 'depression' when the post reflects depression without explicit suicidal intent. "
            "Reply with a single word.\n\n"
            f"Post:\n{text}\n\nAnswer (suicidal/depression):"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="sdcnl", task="suicide_vs_depression"))
    return examples


def load_cssrs(rng: random.Random, max_chars: int, cap: int | None) -> List[Example]:
    ds = load_dataset("thisiskeithkwan/Reddit_C-SSRS", split="train")
    examples: List[Example] = []
    options = ("Supportive", "Ideation", "Indicator", "Behavior", "Attempt")
    option_text = ", ".join(options)
    for row in dataset_slice(ds, cap, rng):
        text = str(row["Post"]).strip()
        label = str(row["Label"]).strip()
        prompt = (
            "Label the suicide risk level using the Columbia-Suicide Severity Rating Scale. "
            "Choose one option and answer with the option only.\n"
            f"Options: {option_text}\n\nPost:\n{text}\n\nAnswer:"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="cssrs", task="suicide_risk"))
    return examples


def load_emobench(rng: random.Random, max_chars: int, cap: int | None) -> List[Example]:
    ds = load_dataset("SahandSab/EmoBench", "emotional_understanding", split="train")
    examples: List[Example] = []
    for row in dataset_slice(ds, cap, rng):
        scenario = str(row["scenario"]).strip()
        label = str(row["emotion_label"]).strip()
        options = ", ".join(row.get("emotion_choices", []))
        prompt = (
            "Identify the most appropriate emotion for the scenario and answer with the emotion word only.\n"
            f"Options: {options}\n\nScenario:\n{scenario}\n\nAnswer:"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="emobench", task="emotion_classification"))
    return examples


def load_goemotions(rng: random.Random, max_chars: int, cap: int | None) -> List[Example]:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    options = tuple(ds.features["labels"].feature.names)
    option_text = ", ".join(options)
    examples: List[Example] = []
    for row in dataset_slice(ds, cap, rng):
        text = str(row["text"]).strip()
        if not text:
            continue
        labels = row["labels"]
        if not labels:
            continue
        label_idx = labels[0]
        if not isinstance(label_idx, int) or label_idx >= len(options):
            continue
        label = options[label_idx].lower()
        prompt = (
            "Identify the best matching emotion for the Reddit comment. "
            "Choose one label from the GoEmotions inventory and reply with the label only.\n"
            f"Options: {option_text}\n\nComment:\n{text}\n\nAnswer:"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="goemotions", task="emotion_classification"))
    return examples


def load_iclab_local(max_chars: int, rng: random.Random, cap: int | None) -> List[Example]:
    dataset_path = REPO_ROOT / "iclab_dataset" / "train.csv"
    examples: List[Example] = []
    if not dataset_path.exists():
        return examples
    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    indices = range(len(rows))
    if cap is not None and cap > 0 and cap < len(rows):
        indices = rng.sample(indices, cap)
    for idx in indices:
        row = rows[idx]
        text = row.get("text-prompted", "").strip()
        label = row.get("label", "").strip().lower()
        if not text or not label:
            continue
        prompt = (
            "As a clinician, decide whether the following post shows depression. "
            "Reply with 'yes' or 'no'.\n\nPost:\n"
            f"{text}\n\nAnswer:"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="iclab_depression", task="depression_detection"))
    return examples


def load_optimized_emotions(max_chars: int, rng: random.Random, cap: int | None) -> List[Example]:
    dataset_path = REPO_ROOT / "optimized_train.csv"
    examples: List[Example] = []
    if not dataset_path.exists():
        return examples
    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    indices = range(len(rows))
    if cap is not None and cap > 0 and cap < len(rows):
        indices = rng.sample(indices, cap)
    for idx in indices:
        row = rows[idx]
        prompt_text = row.get("text-prompted", "").strip()
        label = row.get("label", "").strip().lower()
        if not prompt_text or not label:
            continue
        prompt = (
            "Read the short narrative and select the best emotion. "
            "Options: joy, guilt, fear, anger, sadness, disgust, shame.\n\n"
            f"Text:\n{prompt_text}\n\nAnswer:"
        )
        pair = trim_pair(prompt, label, max_chars)
        if pair:
            examples.append(Example(prompt=pair[0], response=pair[1], source="optimized_emotions", task="emotion_classification"))
    return examples


def deduplicate(examples: Sequence[Example]) -> List[Example]:
    seen = set()
    unique: List[Example] = []
    for ex in examples:
        key = hashlib.sha256(f"{ex.prompt}|||{ex.response}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)
    return unique


def write_jsonl(path: Path, rows: Iterable[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ex in rows:
            handle.write(
                json.dumps(
                    {
                        "prompt": ex.prompt,
                        "response": ex.response,
                        "source": ex.source,
                        "task": ex.task,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_corpus(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    max_chars = args.max_chars
    cap = args.per_source_cap
    examples: List[Example] = []
    examples += load_counsel_chat(rng, max_chars, cap)
    examples += load_mental_health_counseling_sets(rng, max_chars, cap)
    examples += load_dreaddit(rng, max_chars, cap)
    examples += load_sdcnl(rng, max_chars, cap)
    examples += load_cssrs(rng, max_chars, cap)
    examples += load_emobench(rng, max_chars, cap)
    examples += load_goemotions(rng, max_chars, cap)
    examples += load_iclab_local(max_chars, rng, cap)
    examples += load_optimized_emotions(max_chars, rng, cap)

    examples = deduplicate(examples)
    rng.shuffle(examples)

    split_idx = int(len(examples) * args.train_ratio)
    train_set = examples[:split_idx]
    val_set = examples[split_idx:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_set)
    write_jsonl(args.output_dir / "val.jsonl", val_set)

    stats: Dict[str, Dict[str, int]] = {}
    for ex in examples:
        stats.setdefault(ex.task, {})
        stats[ex.task].setdefault(ex.source, 0)
        stats[ex.task][ex.source] += 1
    summary = {
        "total_examples": len(examples),
        "train_examples": len(train_set),
        "val_examples": len(val_set),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "max_chars": args.max_chars,
        "per_source_cap": args.per_source_cap,
        "by_task": stats,
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(train_set)} train and {len(val_set)} val examples to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a multi-source mental-health SFT corpus.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSONL output.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Portion of data used for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2800,
        help="Skip examples whose prompt+response exceed this many characters. Set <=0 to disable.",
    )
    parser.add_argument(
        "--per-source-cap",
        type=int,
        default=None,
        help="Optional cap per source to keep dataset size manageable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_corpus(parse_args())
