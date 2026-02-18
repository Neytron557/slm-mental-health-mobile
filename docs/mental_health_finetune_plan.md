# Mental-Health Fine-Tuning Plan

## 0) Build the unified SFT corpus

```
cd /home/iclab/slm_iclab-1/intern_summer/finetuning_scripts
python3 prepare_mental_health_corpus.py --output-dir ../data/mental_health_corpus --per-source-cap 0
```
- Sources: counsel-chat (2x), Amod + ShareGPT counseling sets, alexandreteles, heliosbrahma, Dreaddit (train+val), SDCNL (train+val), Reddit C-SSRS (train), EmoBench emotional_understanding, GoEmotions train (simplified), iclab_dataset/train.csv, optimized_train.csv.
- Files produced: `intern_summer/data/mental_health_corpus/train.jsonl`, `val.jsonl`, `stats.json`.

## 1) Training template

```
python3 run_mental_health_sft.py \
  --model <BASE_MODEL_PATH> \
  --output-dir <OUTPUT_DIR> \
  --max-length 1024 \
  --batch-size 1 \
  --grad-accum <ACCUM> \
  --epochs <EPOCHS> \
  --lr <LR> \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 \
  --eval-steps 250 --save-steps 250 --logging-steps 20 \
  --seed 42
```
- Uses 4-bit loading by default (set `--no-4bit` if you want full/bf16).
- Validation runs if `val.jsonl` is present; checkpoints saved under `--output-dir`.

## 2) Per-model knobs (all paths under `intern_summer/models`)

| Model | Output dir | Accum | Epochs | LR | Notes |
|-------|------------|-------|--------|----|-------|
| HuggingFaceTB_SmolLM3-3B-Base | release_candidates/SmolLM3-mental-health-lora | 16 | 2 | 2e-5 | Keep 4-bit; fits in 24 GB. |
| TinyLlama_TinyLlama-1.1B-Chat-v1.0 | release_candidates/TinyLlama-mental-health-lora | 12 | 3 | 2.5e-5 | Can bump batch to 2 if VRAM allows. |
| ibm-granite_granite-3.1-2b-instruct | release_candidates/granite-3.1-mental-health-lora | 16 | 2.5 | 1.5e-5 | Shorter context (stay at 1024). |
| ibm-granite_granite-4.0-h-micro | release_candidates/granite-4.0h-mental-health-lora | 12 | 3 | 2e-5 | 4-bit recommended to save memory. |
| qwen2_5_1_5b_instruct | release_candidates/qwen2.5-1.5b-mental-health-lora | 12 | 3 | 2e-5 | Defaults target standard q/k/v/o proj modules. |
| Llama-3.2-3B-Instruct | release_candidates/llama3.2-3b-mental-health-lora | 16 | 2 | 1.8e-5 | Matches eval prompts; keep 4-bit. |
| gemma-3-270m-it | release_candidates/gemma3-270m-mental-health-lora | 4 | 3 | 3e-5 | Run with `--no-4bit --bf16` to avoid unnecessary quant during training. |

## 3) Ready-to-run commands

```
# 3B SmolLM3
python3 run_mental_health_sft.py --model ../models/HuggingFaceTB_SmolLM3-3B-Base \
  --output-dir ../models/release_candidates/SmolLM3-mental-health-lora \
  --grad-accum 16 --epochs 2 --lr 2e-5

# TinyLlama 1.1B
python3 run_mental_health_sft.py --model ../models/TinyLlama_TinyLlama-1.1B-Chat-v1.0 \
  --output-dir ../models/release_candidates/TinyLlama-mental-health-lora \
  --grad-accum 12 --epochs 3 --lr 2.5e-5

# Granite 3.1 2B
python3 run_mental_health_sft.py --model ../models/ibm-granite_granite-3.1-2b-instruct \
  --output-dir ../models/release_candidates/granite-3.1-mental-health-lora \
  --grad-accum 16 --epochs 2.5 --lr 1.5e-5

# Granite 4.0-h micro
python3 run_mental_health_sft.py --model ../models/ibm-granite_granite-4.0-h-micro \
  --output-dir ../models/release_candidates/granite-4.0h-mental-health-lora \
  --grad-accum 12 --epochs 3 --lr 2e-5

# Qwen2.5 1.5B
python3 run_mental_health_sft.py --model ../models/qwen2_5_1_5b_instruct \
  --output-dir ../models/release_candidates/qwen2.5-1.5b-mental-health-lora \
  --grad-accum 12 --epochs 3 --lr 2e-5

# Llama-3.2 3B Instruct
python3 run_mental_health_sft.py --model ../models/Llama-3.2-3B-Instruct \
  --output-dir ../models/release_candidates/llama3.2-3b-mental-health-lora \
  --grad-accum 16 --epochs 2 --lr 1.8e-5

# Gemma3 270M (no 4-bit during training)
python3 run_mental_health_sft.py --model ../models/gemma-3-270m-it \
  --output-dir ../models/release_candidates/gemma3-270m-mental-health-lora \
  --grad-accum 4 --epochs 3 --lr 3e-5 --no-4bit --bf16
```

After fine-tuning, quantize per your earlier picks (BnB4/AWQ/GPTQ) before pushing release artifacts under `intern_summer/models/release_models`.
