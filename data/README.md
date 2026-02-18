# Training Data

This directory contains the curated training datasets used to fine-tune the mental health SLMs.

## Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `deepseek_10k_train.csv` | 15,524 | `text-prompted`, `label` | Base instruction-response dataset used with DeepSeek-style prompting |
| `subset_train.csv` | 15,524 | `text-prompted`, `label` | Subset split for controlled training/ablation experiments |
| `optimized_train.csv` | 29,596 | `text-prompted`, `label`, `prompt_variation` | Augmented dataset with multiple prompt phrasings per sample |

## Format

Each row contains:
- **`text-prompted`**: The full instruction prompt formatted for the model (e.g., *"As a psychologist, read this post and identify the emotion..."*)
- **`label`**: The ground-truth classification target
- **`prompt_variation`** *(optimized_train.csv only)*: Index of the prompt template variant used

## Labels

The datasets cover three complementary mental health classification tasks:

| Label | Task |
|-------|------|
| `joy`, `anger`, `fear`, `sadness`, `disgust`, `guilt`, `shame` | **Emotion classification** (ISEAR-style) |
| `yes`, `no` | **Stress / binary detection** (Dreaddit-style) |
| `suicidal`, `depression` | **Suicidal ideation detection** (SDCNL-style) |
| `minimum`, `mild`, `moderate`, `severe` | **Depression severity** (PHQ-9 scale) |

## Source Datasets

The training data was curated and merged from 14+ sources:

- **Counseling**: `nbertagnolli/counsel-chat`, `andreshere/counsel_chat`, `Amod/mental_health_counseling_conversations`
- **Emotion**: GoEmotions (Google), EmoBench
- **Stress detection**: Dreaddit (Reddit stress corpus)
- **Suicidal ideation**: SDCNL, Reddit C-SSRS
- **Depression severity**: DepSeverity corpus
- **ICLab internal**: depression detection dataset

The merging and prompt construction pipeline is in [`finetuning/prepare_mental_health_corpus.py`](../finetuning/prepare_mental_health_corpus.py).

## Notes

- Large intermediate files (merged JSONL, parquet caches) are excluded from the repo via `.gitignore`. Regenerate them by running `prepare_mental_health_corpus.py`.
- Model prediction outputs (`*_preds.csv`, `generation_results*.csv`) are also excluded — they are produced by the evaluation scripts.
