# slm-mental-health-mobile

Fine-tuning and on-device deployment of Small Language Models (SLMs) for mental health applications. This project explores supervised fine-tuning of 270M–3B parameter models on curated mental health datasets, followed by quantization and deployment to Android (Samsung Galaxy S22) via ONNX Runtime GenAI.

---

## Overview

Mental health AI on consumer mobile devices has the potential to provide private, always-available support without data leaving the device. This project addresses two core challenges:

1. **Fine-tuning efficacy** — Can small (sub-3B) models learn therapeutic response patterns and mental health classification from multi-source datasets?
2. **Mobile viability** — Can these models be compressed to fit and run efficiently on a mid-range Android device (6–8 GB RAM)?

### Research pipeline

```
Data Curation → Fine-tuning (LoRA/SFT) → Quantization → Evaluation → Android Deployment
     ↓                    ↓                    ↓               ↓              ↓
14+ datasets         7 base models        AWQ/GPTQ/BnB4    5 datasets    ORT GenAI
76k examples       270M–3B params         0.85–3.4 GB       metrics      Galaxy S22
```

---

## Repository Structure

```
slm-mental-health-mobile/
├── finetuning/                  # Training scripts
│   ├── prepare_mental_health_corpus.py  # Merges 14+ datasets → unified JSONL (76k examples)
│   ├── run_mental_health_sft.py         # Primary SFT training driver (LoRA + TRL)
│   ├── advanced_counsel_finetune.py     # Advanced training with quality monitoring
│   └── test_model_response.py          # Quick inference sanity check
├── scripts/                     # Post-training utilities
│   ├── quantize_awq.py          # 4-bit AWQ quantization (mobile-primary)
│   ├── quantize_gptq.py         # 4-bit GPTQ quantization
│   ├── quantize_bnb4.py         # BitsAndBytes 4-bit training-time quantization
│   ├── dequantize_bnb4.py       # Convert BnB4 → FP16 for ONNX conversion
│   ├── quantize_torch_ao.py     # TorchAO quantization (experimental)
│   ├── merge_lora_adapter.py    # Merge LoRA adapters into base weights
│   ├── evaluate_model.py        # Evaluation on 5 mental health datasets
│   ├── evaluate_release_metrics.py  # Release candidate metrics (BertScore, F1, empathy)
│   └── generate_charts.py       # Comparison charts across model variants
├── android_deployment/          # Android ONNX Runtime deployment pipeline
│   ├── README.md                # Full 19-step deployment playbook
│   ├── DEPLOYMENT_STATUS.md     # Current progress tracking
│   ├── config/models.yml        # Registry of 7 release model configurations
│   ├── prompts/                 # Benchmark prompt templates
│   ├── templates/               # Chat formatting templates
│   └── scripts/                 # Conversion, staging, benchmarking scripts
│       ├── convert_models.py    # ORT GenAI + ONNX fallback conversion
│       ├── stage_models.py      # ADB push to Android device
│       ├── run_benchmarks.py    # On-device benchmark runner
│       └── aggregate_results.py # Results consolidation
├── data/                        # Training datasets
│   ├── README.md
│   ├── deepseek_10k_train.csv   # 15.5k instruction-response pairs
│   ├── optimized_train.csv      # 29.6k pairs (multi-prompt augmented)
│   └── subset_train.csv         # 15.5k pairs (ablation subset)
├── results/metrics/             # Evaluation outputs (JSON + charts)
├── docs/                        # Research documentation
│   ├── comprehensive_solution_summary.md  # Strategy: quantization phases, mobile arch
│   ├── mental_health_finetune_plan.md     # Hyperparameters per model
│   └── slm_cot_tool_use_report.md         # Chain-of-Thought & tool-use research notes
├── requirements.txt
└── .gitignore
```

---

## Models

Seven base architectures were fine-tuned and released:

| Model | Params | Quantization | Format | Notes |
|-------|--------|-------------|--------|-------|
| Gemma-3-270M-IT | 270M | FP16 | HF | Smallest, fastest inference |
| TinyLlama-1.1B-Chat | 1.1B | BnB4 | HF | Quick sanity baseline |
| Qwen2.5-1.5B-Instruct | 1.5B | AWQ | HF | Best compression ratio |
| IBM Granite-4.0-h-micro | ~1B | BnB4 | HF | IBM micro architecture |
| IBM Granite-3.1-2B-Instruct | 2B | BnB4 | HF | IBM 2B variant |
| Llama-3.2-3B-Instruct | 3B | GPTQ (W4G128) | HF | Meta flagship for size |
| SmolLM3-3B-Base | 3B | BnB4 | HF | HuggingFace efficient 3B |

Target mobile footprint: **≤ 1 GB** (4-bit AWQ quantized Qwen/Llama) for on-device inference.

> Model weights are hosted on HuggingFace Hub (not included in this repo due to size). See `android_deployment/config/models.yml` for source paths.

---

## Datasets

The training corpus is built from 14+ sources, unified into ~76,000 instruction-following examples:

| Category | Sources | Examples |
|----------|---------|---------|
| Emotion classification | GoEmotions, EmoBench, ISEAR | ~51,000 |
| Depression counseling | ICLab dataset, DepSeverity | ~15,200 |
| Stress detection | Dreaddit | ~2,800 |
| Suicidal ideation | SDCNL, Reddit C-SSRS | ~1,600 |
| Therapeutic conversation | counsel-chat (3 variants), Amod, Alexandre, HeliosBrahma | ~5,800 |

See [`data/README.md`](data/README.md) for dataset format and label descriptions.

---

## Evaluation Results

Evaluated on 5 held-out mental health benchmarks (Llama-3.2-3B variants):

| Dataset | Task | Base Acc | FT Merged Acc | FT AWQ Acc |
|---------|------|----------|---------------|-----------|
| Dreaddit | Stress detection (binary) | 57.4% | **54.3%** | 53.5% |
| SDCNL | Suicidal ideation (binary) | 50.0% | **53.1%** | 52.3% |
| GoEmotions | Emotion (27-class) | 27.0% | **41.2%** | 40.5% |
| EmoBench | Emotion understanding | 22.0% | **22.0%** | 18.0% |
| C-SSRS | Suicide risk (multi-class) | 12.0% | **19.0%** | 12.0% |

The merged (LoRA-integrated FP16) model generally outperforms quantized variants, with AWQ offering the best accuracy-to-size trade-off among compressed formats.

Charts and raw JSON results are in [`results/metrics/`](results/metrics/).

---

## Setup

### Requirements

- Python 3.10+
- CUDA 11.8 or 12.x (GPU strongly recommended for training)
- ~20 GB disk for model checkpoints (not included)

```bash
git clone https://github.com/Neytron557/slm-mental-health-mobile.git
cd slm-mental-health-mobile

python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA (adjust cu118 to your CUDA version)
pip install torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

---

## Usage

### 1. Prepare training corpus

Merges all 14+ source datasets into a unified JSONL file:

```bash
python finetuning/prepare_mental_health_corpus.py
# Output: data/merged_counsel_dataset.json (~76k examples)
```

### 2. Fine-tune a model

```bash
# SFT with LoRA (4-bit BnB quantization by default)
python finetuning/run_mental_health_sft.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --output_dir models/llama-3.2-3b-finetuned \
  --dataset_path data/merged_counsel_dataset.json \
  --num_epochs 3

# Advanced training with therapeutic quality monitoring
python finetuning/advanced_counsel_finetune.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --output_dir models/llama-3.2-3b-advanced
```

### 3. Merge LoRA adapters

```bash
python scripts/merge_lora_adapter.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --adapter_path models/llama-3.2-3b-finetuned \
  --output_path models/llama-3.2-3b-merged
```

### 4. Quantize for mobile

```bash
# AWQ 4-bit (recommended for mobile)
python scripts/quantize_awq.py \
  --model_path models/llama-3.2-3b-merged \
  --output_path models/llama-3.2-3b-awq \
  --calibration_data data/merged_counsel_dataset.json

# GPTQ 4-bit (alternative)
python scripts/quantize_gptq.py \
  --model_path models/llama-3.2-3b-merged \
  --output_path models/llama-3.2-3b-gptq
```

### 5. Evaluate

```bash
# Evaluate on 5 mental health benchmarks
python scripts/evaluate_model.py \
  --model_path models/llama-3.2-3b-awq \
  --output_file results/metrics/my_eval.json

# Release metrics (BertScore + F1 + empathy scoring)
python scripts/evaluate_release_metrics.py
```

### 6. Android deployment

See [`android_deployment/README.md`](android_deployment/README.md) for the full ONNX Runtime GenAI conversion and on-device benchmarking pipeline targeting Samsung Galaxy S22.

---

## Acknowledgements

This work was conducted at ICLab as part of a summer research internship. The project builds on:

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [counsel-chat](https://github.com/nbertagnolli/counsel-chat) by Nicolas Bertagnolli
- ONNX Runtime GenAI for mobile inference
