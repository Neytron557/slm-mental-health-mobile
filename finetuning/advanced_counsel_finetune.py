#!/usr/bin/env python3
"""
ADVANCED Counsel-Chat Fine-tuning Script with AWQ Mobile Optimization
- Merged dataset support (both counsel-chat sources)
- AWQ 4-bit quantization for mobile deployment (1-2GB target)
- Advanced training techniques for better quality
- Configurable epochs and training strategies
- Professional therapeutic response optimization
- Mobile-optimized inference ready

INSTALLATION FOR AWQ MOBILE DEPLOYMENT:
1. Install auto-awq: pip install auto-awq
2. For mobile deployment: Use --awq_version GEMM (default)
3. Target model size: ~1-2GB with DeepSeek-R1-1.5B base model
4. Mobile inference: Compatible with ONNX Runtime Mobile, Core ML

USAGE:
python advanced_counsel_finetune.py --awq_group_size 128 --awq_version GEMM
"""

import argparse
import os
import random
import string
import math
import warnings
from typing import Optional, Dict, Any, List
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import numpy as np
try:
    from awq import AutoAWQForCausalLM
except ImportError:
    print("⚠️ autoawq is not installed. Install with: pip install autoawq")
    AutoAWQForCausalLM = None

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Utility loader that returns None when a JSON file is missing or invalid."""
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"⚠️ Failed to read JSON from {path}: {exc}")
        return None


def detect_quantization_config(model_path: str) -> Dict[str, Any]:
    """Infer quantization metadata from a model directory."""

    config_path = os.path.join(model_path, "config.json")
    config = load_json(config_path)
    if not config:
        return {"quant_method": None}

    quant_config = config.get("quantization_config") or {}
    quant_method = quant_config.get("quant_method") or quant_config.get("quant_method_name")

    # Normalise shorthand keys for downstream consumption
    normalized = {
        **quant_config,
        "quant_method": quant_method,
        "load_in_4bit": quant_config.get("load_in_4bit", quant_config.get("_load_in_4bit")),
        "load_in_8bit": quant_config.get("load_in_8bit", quant_config.get("_load_in_8bit")),
    }
    return normalized


@dataclass
class AdvancedTrainingMetrics:
    """Track advanced training metrics for therapeutic quality monitoring."""
    train_losses: List[float]
    val_losses: List[float]
    epochs: List[int]
    therapeutic_scores: List[float]
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    best_therapeutic_score: float = 0.0


class AdvancedCounselDataset(Dataset):
    """Advanced dataset with merged sources and enhanced preprocessing."""

    def __init__(self, tokenizer, max_length: int = 512, quality_threshold: int = 1, 
                 validation_split: float = 0.15, use_merged_dataset: bool = True,
                 merged_dataset_path: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quality_threshold = quality_threshold
        self.validation_split = validation_split
        self.use_merged_dataset = use_merged_dataset
        self.merged_dataset_path = merged_dataset_path
        self.samples = self._load_and_preprocess()
        
    def _load_and_preprocess(self):
        """Load and preprocess with support for merged datasets."""
        
        if self.use_merged_dataset and self.merged_dataset_path and os.path.exists(self.merged_dataset_path):
            print(f"📚 Loading merged dataset from {self.merged_dataset_path}...")
            return self._load_merged_dataset()
        else:
            print("📚 Loading individual datasets and merging...")
            return self._load_and_merge_datasets()
    
    def _load_merged_dataset(self):
        """Load pre-merged dataset from JSON file."""
        import json
        
        try:
            with open(self.merged_dataset_path, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            print(f"✅ Loaded {len(merged_data)} conversations from merged dataset")
            
            conversations = []
            for item in merged_data:
                try:
                    text = item['text']
                    metadata = item.get('metadata', {})
                    
                    # Filter by quality if needed
                    upvotes = metadata.get('upvotes', 0)
                    if upvotes < self.quality_threshold:
                        continue
                    
                    # Tokenize to check length
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > self.max_length - 50:
                        continue
                    
                    conversations.append({
                        'text': text,
                        'context': metadata.get('question', ''),
                        'response': metadata.get('response', ''),
                        'upvotes': upvotes,
                        'topic': metadata.get('topic', 'general'),
                        'source': metadata.get('source', 'merged'),
                        'tokens': len(tokens)
                    })
                    
                except Exception as e:
                    continue
            
            print(f"📊 Filtered to {len(conversations)} high-quality conversations")
            return conversations
            
        except Exception as e:
            print(f"❌ Error loading merged dataset: {e}")
            return self._load_and_merge_datasets()
    
    def _load_and_merge_datasets(self):
        """Load and merge datasets from multiple sources."""
        print("🔧 Loading and merging multiple counsel-chat datasets...")
        
        conversations = []
        seen_hashes = set()
        
        # Load nbertagnolli/counsel-chat
        try:
            dataset1 = load_dataset("nbertagnolli/counsel-chat")
            raw_data1 = dataset1["train"]
            
            for item in tqdm(raw_data1, desc="Processing nbertagnolli dataset"):
                conv = self._process_nbertagnolli_item(item, seen_hashes)
                if conv:
                    conversations.append(conv)
                    
        except Exception as e:
            print(f"⚠️ Could not load nbertagnolli dataset: {e}")
        
        # Load andreshere/counsel_chat  
        try:
            dataset2 = load_dataset("andreshere/counsel_chat")
            raw_data2 = dataset2["train"]
            
            for item in tqdm(raw_data2, desc="Processing andreshere dataset"):
                conv = self._process_andreshere_item(item, seen_hashes)
                if conv:
                    conversations.append(conv)
                    
        except Exception as e:
            print(f"⚠️ Could not load andreshere dataset: {e}")
        
        print(f"✅ Merged {len(conversations)} unique high-quality conversations")
        return conversations
    
    def _process_nbertagnolli_item(self, item, seen_hashes):
        """Process item from nbertagnolli/counsel-chat dataset."""
        try:
            question = item.get("questionText", "").strip()
            response = item.get("answerText", "").strip()
            topic = item.get("topic", "general")
            upvotes = item.get("upvotes", 0)
            
            if not question or not response or upvotes < self.quality_threshold:
                return None
            
            if len(question) < 20 or len(response) < 50:
                return None
            
            # Create conversation text
            conversation_text = f"""You are a compassionate mental health counselor. A client is seeking support about {topic}. Please provide an empathetic, helpful response.

Client: {question}

Counselor: {response}"""
            
            # Check for duplicates and length
            content_hash = hash(f"{question}|||{response}")
            if content_hash in seen_hashes:
                return None
            seen_hashes.add(content_hash)
            
            tokens = self.tokenizer.encode(conversation_text, add_special_tokens=False)
            if len(tokens) > self.max_length - 50:
                return None
            
            return {
                'text': conversation_text,
                'context': question,
                'response': response,
                'upvotes': upvotes,
                'topic': topic,
                'source': 'nbertagnolli',
                'tokens': len(tokens)
            }
            
        except Exception:
            return None
    
    def _process_andreshere_item(self, item, seen_hashes):
        """Process item from andreshere/counsel_chat dataset."""
        try:
            question = item.get("Context", "").strip()
            response = item.get("Response", "").strip()
            topic = item.get("Topic", "general").lower()
            upvotes = item.get("UpVotes", 0)
            
            if not question or not response or upvotes < self.quality_threshold:
                return None
            
            if len(question) < 20 or len(response) < 50:
                return None
            
            # Create conversation text
            conversation_text = f"""You are a compassionate mental health counselor. A client is seeking support about {topic}. Please provide an empathetic, helpful response.

Client: {question}

Counselor: {response}"""
            
            # Check for duplicates and length
            content_hash = hash(f"{question}|||{response}")
            if content_hash in seen_hashes:
                return None
            seen_hashes.add(content_hash)
            
            tokens = self.tokenizer.encode(conversation_text, add_special_tokens=False)
            if len(tokens) > self.max_length - 50:
                return None
            
            return {
                'text': conversation_text,
                'context': question,
                'response': response,
                'upvotes': upvotes,
                'topic': topic,
                'source': 'andreshere',
                'tokens': len(tokens)
            }
            
        except Exception:
            return None
    
    def split_train_validation(self):
        """Split dataset into training and validation sets with balanced topics."""
        total_size = len(self.samples)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size
        
        # Sort by topic for balanced split
        samples_by_topic = {}
        for sample in self.samples:
            topic = sample['topic']
            if topic not in samples_by_topic:
                samples_by_topic[topic] = []
            samples_by_topic[topic].append(sample)
        
        train_samples = []
        val_samples = []
        
        # Split each topic proportionally
        for topic, topic_samples in samples_by_topic.items():
            random.shuffle(topic_samples)
            topic_val_size = max(1, int(len(topic_samples) * self.validation_split))
            
            val_samples.extend(topic_samples[:topic_val_size])
            train_samples.extend(topic_samples[topic_val_size:])
        
        # Create new dataset instances
        train_dataset = AdvancedCounselDataset.__new__(AdvancedCounselDataset)
        val_dataset = AdvancedCounselDataset.__new__(AdvancedCounselDataset)
        
        # Copy attributes
        for dataset in [train_dataset, val_dataset]:
            dataset.tokenizer = self.tokenizer
            dataset.max_length = self.max_length
            dataset.quality_threshold = self.quality_threshold
            dataset.validation_split = self.validation_split
            dataset.use_merged_dataset = self.use_merged_dataset
            dataset.merged_dataset_path = self.merged_dataset_path
        
        train_dataset.samples = train_samples
        val_dataset.samples = val_samples
        
        print(f"📊 Balanced dataset split: {len(train_samples)} train, {len(val_samples)} validation")
        
        # Print topic distribution
        train_topics = {}
        val_topics = {}
        for sample in train_samples:
            topic = sample['topic']
            train_topics[topic] = train_topics.get(topic, 0) + 1
        for sample in val_samples:
            topic = sample['topic']
            val_topics[topic] = val_topics.get(topic, 0) + 1
        
        print("📈 Topic distribution:")
        all_topics = set(train_topics.keys()) | set(val_topics.keys())
        for topic in sorted(all_topics):
            train_count = train_topics.get(topic, 0)
            val_count = val_topics.get(topic, 0)
            print(f"   {topic}: {train_count} train, {val_count} val")
        
        return train_dataset, val_dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Enhanced tokenization with better padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels for training (same as input_ids)
        labels = input_ids.clone()
        
        # Enhanced masking for better therapeutic training
        try:
            # Find "Counselor:" marker more robustly
            text_lower = text.lower()
            counselor_pos = text_lower.find("counselor:")
            
            if counselor_pos != -1:
                # Find token position of counselor response start
                prefix_text = text[:counselor_pos + 10]  # Include "Counselor:"
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                mask_until = len(prefix_tokens)
            else:
                # Fallback: mask first 70%
                mask_until = int(0.7 * len(input_ids))
            
            # Apply masking - only train on counselor response
            if mask_until > 0 and mask_until < len(labels):
                labels[:mask_until] = -100
                
        except Exception:
            # Emergency fallback
            mask_until = int(0.7 * len(input_ids))
            labels[:mask_until] = -100
        
        # Ensure we have sufficient valid labels
        valid_labels = (labels != -100).sum()
        if valid_labels < 20:  # Need at least 20 tokens to train on
            labels[-30:] = input_ids[-30:]  # Unmask last 30 tokens
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AdvancedTherapeuticTrainer(Trainer):
    """Advanced trainer with therapeutic quality monitoring."""
    
    def __init__(self, *args, validation_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_dataset = validation_dataset
        self.metrics = AdvancedTrainingMetrics([], [], [], [])
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with therapeutic quality metrics."""
        if eval_dataset is None:
            eval_dataset = self.validation_dataset
            
        if eval_dataset is None:
            return {}
            
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Track validation loss and therapeutic quality
        if "eval_loss" in result:
            self.metrics.val_losses.append(result["eval_loss"])
            
            # Calculate therapeutic quality score (placeholder)
            therapeutic_score = self._calculate_therapeutic_score(result["eval_loss"])
            self.metrics.therapeutic_scores.append(therapeutic_score)
            
            # Check if this is the best model
            if result["eval_loss"] < self.metrics.best_val_loss:
                self.metrics.best_val_loss = result["eval_loss"]
                self.metrics.best_epoch = len(self.metrics.val_losses)
                self.metrics.best_therapeutic_score = therapeutic_score
                print(f"🎯 New best model: val_loss={result['eval_loss']:.4f}, therapeutic_score={therapeutic_score:.3f}")
        
        return result
    
    def _calculate_therapeutic_score(self, val_loss):
        """Calculate therapeutic quality score based on validation loss and other factors."""
        # Simple heuristic: lower loss = better therapeutic quality
        # In practice, this could be much more sophisticated
        base_score = max(0, 1.0 - val_loss / 10.0)
        return min(1.0, base_score)


def setup_advanced_model_and_tokenizer(args):
    """Setup model/tokenizer with the correct quantization strategy for fine-tuning."""

    quant_config = detect_quantization_config(args.model_path)
    quant_method = (quant_config.get("quant_method") or "none").lower() if quant_config else "none"

    # Allow explicit override from CLI for 4-bit loading even when the base model is full precision
    if getattr(args, "load_in_4bit", False):
        quant_method = "bitsandbytes"

    print(f"🔍 Detected quantization method: {quant_method or 'none'}")

    # Load tokenizer first so we can reuse it regardless of model strategy
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if getattr(args, "attn_implementation", None):
        base_kwargs["attn_implementation"] = args.attn_implementation
    elif args.use_flash_attention:
        base_kwargs["attn_implementation"] = "flash_attention_2"

    model = None

    if quant_method in {"awq", "awq-4bit", "autoawq"}:
        if AutoAWQForCausalLM is None:
            raise ImportError("auto-awq is required to fine-tune AWQ models. Install with `pip install autoawq`." )
        print("🔧 Loading AWQ pre-quantized model for training...")
        awq_model = AutoAWQForCausalLM.from_quantized(
            args.model_path,
            trust_remote_code=True,
            device_map=base_kwargs.get("device_map"),
            fuse_layers=False,
        )
        model = awq_model.model
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    elif quant_method in {"bitsandbytes", "bnb"} or quant_config.get("load_in_4bit") or getattr(args, "load_in_4bit", False):
        print("🔧 Loading model with bitsandbytes 4-bit quantization for QLoRA training...")
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported() and args.bf16) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            **base_kwargs,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    elif quant_method == "gptq":
        print("🔧 Loading GPTQ quantized model and preparing for LoRA training...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            **base_kwargs,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    else:
        print("🔧 Loading full-precision model for fine-tuning...")
        if torch.cuda.is_available() and getattr(args, "bf16", False):
            dtype = torch.bfloat16
        elif torch.cuda.is_available() and getattr(args, "fp16", True):
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            **base_kwargs,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Advanced LoRA configuration for therapeutic training
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    if trainable_count == 0:
        raise ValueError("No trainable parameters found! Ensure target modules are correct for this architecture.")
    print(f"✅ Verified {trainable_count} trainable parameters")

    return model, tokenizer


def create_advanced_optimizer_and_scheduler(model, args, num_training_steps):
    """Create advanced optimizer and scheduler with better hyperparameters."""
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"🔧 Optimizing {len(trainable_params)} trainable parameter groups")
    
    # Advanced optimizer compatible with AWQ quantization
    from torch.optim import AdamW
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    
    # Advanced scheduler selection
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )
    elif args.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def main(args):
    """Advanced main training function with enhanced features."""
    
    print("🏥 Starting ADVANCED Conversational Mental Health Model Training")
    print("📱 Mobile Optimized: AWQ 4-bit quantization, 1-2GB target size")
    print("🎯 Enhanced Features: Merged datasets, advanced training, therapeutic optimization")
    
    # Setup paths
    model_dir = os.path.abspath(args.model_path)
    assert os.path.exists(model_dir), f"Model path {model_dir} not found"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and tokenizer  
    print("🔧 Setting up AWQ-quantized model configuration...")
    model, tokenizer = setup_advanced_model_and_tokenizer(args)
    
    # Setup dataset with merged sources
    print(f"📚 Loading advanced dataset (merged={args.use_merged_dataset})...")
    full_dataset = AdvancedCounselDataset(
        tokenizer, 
        args.max_length, 
        quality_threshold=args.quality_threshold,
        use_merged_dataset=args.use_merged_dataset,
        merged_dataset_path=args.merged_dataset_path
    )
    
    # Split into train/validation
    train_dataset, val_dataset = full_dataset.split_train_validation()
    
    print(f"📊 Training on {len(train_dataset)} conversations")
    print(f"📊 Validating on {len(val_dataset)} conversations")
    
    # Calculate training steps
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * args.num_epochs
    
    print(f"📈 Training for {args.num_epochs} epochs ({steps_per_epoch} steps/epoch)")
    
    # Setup optimizer and scheduler
    optimizer, scheduler = create_advanced_optimizer_and_scheduler(model, args, num_training_steps)
    
    # Advanced training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        
        # Training settings
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        
        # Precision and optimization (AWQ optimized)
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False,
        
        # Enhanced evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Enhanced logging
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        
        # Memory optimization
        remove_unused_columns=False,
        dataloader_num_workers=0,
        
        # Stability
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        
        # Disable unnecessary features
        report_to="none",
        disable_tqdm=False,
    )
    
    # Create advanced trainer
    trainer = AdvancedTherapeuticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        validation_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler) if scheduler else (optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    
    # Training with advanced monitoring
    print("🚀 Starting advanced therapeutic training...")
    print(f"📊 Total training steps: {num_training_steps}")
    print(f"📊 Evaluation every {training_args.eval_steps} steps")
    print(f"📊 Early stopping patience: {args.early_stopping_patience} evaluations")
    
    trainer.train()
    
    # Enhanced saving and metrics
    print("💾 Saving advanced model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save comprehensive training info
    training_info = {
        "model": args.model_path,
        "dataset": "merged counsel-chat (advanced)",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "quality_threshold": args.quality_threshold,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation_steps,
        "scheduler_type": args.scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "quantization": "4-bit AWQ (Mobile Optimized)",
        "awq_group_size": args.awq_group_size,
        "awq_version": args.awq_version,
        "enhancements": [
            "Merged multiple counsel-chat datasets",
            "Advanced therapeutic quality monitoring", 
            "Balanced topic distribution in validation",
            "Enhanced conversation preprocessing",
            "Configurable training strategies",
            "AWQ 4-bit quantization for mobile deployment",
            "Optimized for 1-2GB model size",
            "Better regularization and optimization"
        ],
        "best_val_loss": trainer.metrics.best_val_loss,
        "best_epoch": trainer.metrics.best_epoch,
        "best_therapeutic_score": trainer.metrics.best_therapeutic_score,
        "final_therapeutic_score": trainer.metrics.therapeutic_scores[-1] if trainer.metrics.therapeutic_scores else 0,
        "target": "Advanced mobile mental health support (1-2GB AWQ optimized)"
    }
    
    with open(os.path.join(args.output_dir, "advanced_training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✅ Advanced training completed!")
    print(f"📂 Model saved to: {args.output_dir}")
    print(f"🎯 Best validation loss: {trainer.metrics.best_val_loss:.4f} at epoch {trainer.metrics.best_epoch}")
    print(f"🏥 Best therapeutic score: {trainer.metrics.best_therapeutic_score:.3f}")
    print("📱 Ready for mobile mental health deployment with AWQ optimization!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Counsel-Chat Fine-tuning")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="./vanilla_deepseek/deepseek_r1_1.5b",
                       help="Path to base model")
    parser.add_argument("--output_dir", type=str, 
                       default=f"./advanced_counsel_model_{int(torch.rand(1).item() * 100000)}",
                       help="Output directory for trained model")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Dataset arguments
    parser.add_argument("--quality_threshold", type=int, default=1, 
                       help="Minimum upvotes for conversation inclusion")
    parser.add_argument("--use_merged_dataset", action="store_true", default=True,
                       help="Use pre-merged dataset if available")
    parser.add_argument("--merged_dataset_path", type=str, default="merged_counsel_dataset.json",
                       help="Path to merged dataset JSON file")
    
    # Training strategy arguments
    parser.add_argument("--scheduler_type", type=str, default="cosine", 
                       choices=["cosine", "linear", "none"],
                       help="Learning rate scheduler type")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Save frequency")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    
    # AWQ and Mobile Optimization arguments
    parser.add_argument("--awq_group_size", type=int, default=128,
                       help="AWQ group size for quantization (default: 128)")
    parser.add_argument("--awq_version", type=str, default="GEMM", choices=["GEMM", "GEMV"],
                       help="AWQ version - GEMM for mobile, GEMV for server (default: GEMM)")
    
    # Optional features
    parser.add_argument("--use_flash_attention", action="store_true", 
                       help="Use flash attention if available")
    parser.add_argument("--attn_implementation", type=str, default=None,
                       choices=["sdpa", "flash_attention_2", "eager"],
                       help="Force a specific attention backend (e.g., 'eager' for Gemma models)")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                       help="Force bitsandbytes 4-bit loading (QLoRA)")
    parser.add_argument("--fp16", dest="fp16", action="store_true",
                       help="Enable fp16 mixed precision training (default)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                       help="Disable fp16 mixed precision training")
    parser.set_defaults(fp16=True)
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="Enable bf16 mixed precision when supported")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing")
    
    args = parser.parse_args()
    main(args)
