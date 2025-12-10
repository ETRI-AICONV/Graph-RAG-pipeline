"""
HotpotQA Graph-RAG Pipeline - Modular Components
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
import re
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.graph_utils import flatten_context, get_gold_indices, supporting_fact_em, supporting_fact_f1, batch_embed, build_hierarchical_graph
from utils.gpu_utils import set_device_auto, get_available_gpus
from .config import DEVICE, AVAILABLE_GPUS, MODEL_NAME, HIDDEN_DIM, GNN_LAYERS, THRESHOLD

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
import torch.distributed as dist

# ==================== T5 Generation ====================
def normalize_answer(s: str) -> str:
    """HotpotQA 표준 정규화"""
    import string
    
    # 소문자 변환
    s = s.lower()
    
    # 구두점 제거 (마침표, 쉼표 등)
    s = s.translate(str.maketrans("", "", string.punctuation))
    
    # 공백 정규화
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    
    return s

def compute_answer_em_f1(pred: str, gold: str):
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    em = 1.0 if p == g else 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return em, 0.0
    common = set(p_tokens) & set(g_tokens)
    if len(common) == 0:
        return em, 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return em, f1


def prepare_t5_training_data(cache, mode="gold"):
    """Prepare T5 training data
    mode: 'gold' = use gold supporting facts, 'retrieved' = use retrieved facts
    """
    data = []
    for sid, entry in cache.items():
        try:
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            
            if not question or not answer:
                continue
            
            if mode == "gold":
                # Use gold supporting facts
                gold_idx = entry.get("gold", [])
                evidences = [entry["sents"][i] for i in gold_idx if i < len(entry["sents"])]
            else:
                # Use retrieved facts (requires model)
                raise NotImplementedError("Retrieved mode requires passing model")
            
            if evidences:
                data.append({
                    "question": question,
                    "evidences": evidences,  # Keep as list for training format
                    "answer": answer
                })
        except Exception as e:
            print(f"Error preparing data for sample {sid}: {e}")
            continue
    
    return data


def train_t5_generator(train_data, val_data, model_name="t5-base", output_dir=None, num_epochs=3, skip_training=False, batch_size=32):
    """Train T5 generator with multi-GPU support"""
    # Default output_dir if not provided (for backward compatibility)
    if output_dir is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(SCRIPT_DIR, "models", "t5_generator")
    print(f"\n{'='*80}")
    print(f"Training Generator ({model_name}) for {num_epochs} epochs...")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Structure: Question and contexts concatenated into single input")
    
    # Set environment variables to suppress warnings and optimize memory
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizers fork warning
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce memory fragmentation
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.parallel')  # Suppress torch gather warning
    
    # Multi-GPU support: Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        device = torch.device('cuda')
        # Don't restrict CUDA_VISIBLE_DEVICES - use all available GPUs
    else:
        device = torch.device('cpu')
    
    # Clean up any existing distributed process group (if any)
    if dist.is_available():
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
    
    # Format data for T5: Concatenate question and evidences into single string
    def format_t5_input(question: str, evidences: list) -> str:
        """Format input for T5: concatenate all passages"""
        parts = [f"question: {question}"]
        for evidence in evidences[:3]:  # Max 3 passages
            if evidence:  # Skip empty
                parts.append(f"context: {evidence}")
        return " ".join(parts)
    
    formatted_train = []
    for item in train_data:
        # Split evidences back to list if it's a string
        if isinstance(item['evidences'], str):
            evidences = item['evidences'].split(' ')
        else:
            evidences = item['evidences']
        evidences = evidences[:3]  # Max 3 passages
        
        # For T5: concatenate into single string
        input_text = format_t5_input(item['question'], evidences)
        formatted_train.append({
            "input": input_text,
            "output": item['answer']
        })
    
    formatted_val = []
    for item in val_data:
        # Split evidences back to list if it's a string
        if isinstance(item['evidences'], str):
            evidences = item['evidences'].split(' ')
        else:
            evidences = item['evidences']
        evidences = evidences[:3]  # Max 3 passages
        
        # For T5: concatenate into single string
        input_text = format_t5_input(item['question'], evidences)
        formatted_val.append({
            "input": input_text,
            "output": item['answer']
        })
    
    train_dataset = Dataset.from_list(formatted_train)
    val_dataset = Dataset.from_list(formatted_val)
    
    print(f"Formatted datasets created: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 원본 output 저장 (디버깅용)
    train_original_outputs = [item["output"] for item in formatted_train]
    val_original_outputs = [item["output"] for item in formatted_val]
    
    # Load tokenizer and model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model for T5
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Tokenize for T5: Single input string (not separate passages)
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            examples["output"],
            max_length=64,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["input", "output"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["input", "output"])
    print("Tokenization complete (single input string)")
    
    # Training args - Multi-GPU optimized
    print("Setting up training arguments...")
    
    # Optimize batch size for multi-GPU: Use maximum batch size per GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # For T5: Use larger batch sizes
    if num_gpus >= 4:
        per_device_batch_size = max(24, batch_size)
    elif num_gpus >= 2:
        per_device_batch_size = max(16, batch_size)
    else:
        per_device_batch_size = batch_size
    
    # Minimize gradient accumulation for speed
    effective_batch_size = per_device_batch_size * num_gpus
    gradient_accumulation_steps = 1
    
    # Calculate warmup steps (10% of total steps or minimum 100)
    train_samples = len(train_dataset)
    effective_batch = per_device_batch_size * num_gpus * gradient_accumulation_steps
    steps_per_epoch = max(1, train_samples // effective_batch) if effective_batch > 0 else train_samples
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(100, int(total_steps * 0.1))
    
    # Note: Do NOT set max_steps when using num_train_epochs
    # If both are set, max_steps takes precedence and will override num_train_epochs
    # For small datasets, we rely on num_train_epochs only
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,  # Optimized for multi-GPU
        per_device_eval_batch_size=per_device_batch_size,  # Use same batch size for eval
        gradient_accumulation_steps=gradient_accumulation_steps,  # Maintain effective batch size
        dataloader_num_workers=8,  # Increase workers for speed
        dataloader_prefetch_factor=4,  # Increase prefetch for speed
        fp16=True,  # Enable mixed precision for faster training and less memory
        gradient_checkpointing=False,  # Disable for speed
        learning_rate=3e-5,  # Reduced learning rate to prevent overfitting
        warmup_steps=warmup_steps,  # Dynamic warmup based on dataset size
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        eval_strategy="epoch",  # Evaluate on validation set at end of each epoch
        save_strategy="epoch",  # Save checkpoint each epoch
        logging_steps=999999,  # Disable step-by-step logging, only log at epoch end
        disable_tqdm=False,  # Enable progress bar for each epoch
        logging_strategy="epoch",  # Only log at end of each epoch
        predict_with_generate=True,  # Generation 확인을 위해 활성화
        generation_max_length=64,  # Generation 최대 길이
        save_total_limit=2,  # Keep last 2 checkpoints
        load_best_model_at_end=True,  # Load best model based on eval_loss
        metric_for_best_model="eval_loss",  # Use eval_loss to determine best model
        greater_is_better=False,  # Lower eval_loss is better
        save_steps=999999,  # Disable step-based saving (use epoch-based)
        no_cuda=(device.type == 'cpu'),  # Use CUDA if available
        ddp_find_unused_parameters=False,  # Optimize for multi-GPU (faster)
        dataloader_pin_memory=True,  # Pin memory for faster data transfer to GPU
        report_to='none',  # Disable wandb/tensorboard
        dataloader_drop_last=False,  # Don't drop last batch for small datasets
        remove_unused_columns=False,  # Avoid column processing issues
        skip_memory_metrics=True,  # Skip memory metrics to avoid issues
        max_grad_norm=1.0,  # Gradient clipping for stability
        optim="adamw_torch",  # Use standard AdamW optimizer
        max_steps=-1,  # Use num_train_epochs only, don't override
    )
    
    # CRITICAL: Override any distributed settings
    training_args.local_rank = -1
    # Note: world_size and process_index are read-only properties, cannot be set directly
    # They will be automatically set to 1 and 0 respectively when local_rank=-1
    
    print(f"[Generator Training] Multi-GPU training enabled")
    print(f"[Generator Training] Batch size: {training_args.per_device_train_batch_size}")
    
    # Use standard DataCollatorForSeq2Seq for T5
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # Custom callback to print epoch loss and show epoch-based progress (like retriever training)
    from transformers import TrainerCallback, TrainerState
    try:
        from transformers import EarlyStoppingCallback
    except ImportError:
        # Fallback for older transformers versions
        from transformers.trainer_callback import EarlyStoppingCallback
    
    class EpochLossCallback(TrainerCallback):
        def __init__(self, num_epochs):
            self.num_epochs = num_epochs
            self.current_epoch = 0
        
        def on_epoch_begin(self, args, state, control, **kwargs):
            """Print epoch start message"""
            # state.epoch is 0-indexed float (0.0, 1.0, 2.0, ...)
            # Convert to 1-indexed integer for display
            self.current_epoch = int(state.epoch) + 1
            print(f"\n{'='*80}")
            print(f"Epoch {self.current_epoch}/{self.num_epochs}")
            print(f"{'='*80}")
            # Progress bar will be shown by tqdm during training
        
        def on_epoch_end(self, args, state, control, **kwargs):
            """Print epoch end summary with loss values"""
            # state.epoch is 0-indexed float, convert to 1-indexed
            self.current_epoch = int(state.epoch) + 1
            
            # Memory cleanup after each epoch
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Get train loss and eval loss from log history
            train_loss = None
            eval_loss = None
            
            if state.log_history:
                # Find the last log entry for this epoch
                epoch_logs = [log for log in state.log_history 
                             if 'epoch' in log and log.get('epoch', -1) == state.epoch]
                
                if epoch_logs:
                    last_log = epoch_logs[-1]
                    train_loss = last_log.get('train_loss') or last_log.get('loss')
                    eval_loss = last_log.get('eval_loss')
            
            # Print epoch results (loss values calculated at epoch end)
            print(f"\n{'='*80}")
            if train_loss is not None and eval_loss is not None:
                print(f"Epoch {self.current_epoch}/{self.num_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {eval_loss:.6f}")
            elif train_loss is not None:
                print(f"Epoch {self.current_epoch}/{self.num_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
            elif eval_loss is not None:
                print(f"Epoch {self.current_epoch}/{self.num_epochs} Summary:")
                print(f"  Val Loss: {eval_loss:.6f}")
            print(f"{'='*80}\n")
        
        def on_step_end(self, args, state, control, **kwargs):
            # Periodic memory cleanup every 200 steps to prevent OOM (reduced frequency for speed)
            if state.global_step % 200 == 0 and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
    # Multi-GPU support: Let transformers Trainer handle distributed training automatically
    # Trainer - Use standard Seq2SeqTrainer for T5
    from transformers import Seq2SeqTrainer
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EpochLossCallback(num_epochs)],  # Add callbacks (early stopping disabled)
    )
    
    # Multi-GPU: Let Trainer automatically handle distributed training
    # Trainer will detect and use all available GPUs automatically
    
    print("Starting training...")
    
    training_successful = False
    
    try:
        if skip_training:
            print("\n[Skipping training - using pre-trained model]")
            training_successful = True
        else:
            # Final check: ensure distributed is not initialized
            if dist.is_available() and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except:
                    pass
            
            # Final memory cleanup before training
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            trainer.train()
            print("Training completed successfully")
            
            # Manually save model after training
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            training_successful = True
    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*80}")
        print(f"Training error: {error_msg}")
        print(f"{'='*80}")
        
        # Check if it's a distributed error
        if "process group" in error_msg.lower() or "distributed" in error_msg.lower():
            print("\nWARNING:  DISTRIBUTED TRAINING ERROR DETECTED")
            print("This is a known issue with transformers Trainer trying to initialize distributed training.")
            print("\nAttempting to fix by patching Trainer methods...")
            
            # Try to patch and retry
            try:
                # More aggressive patching
                import transformers.trainer
                original_init_distributed = getattr(transformers.trainer, '_setup_distributed', None)
                if original_init_distributed:
                    def noop_setup_distributed(*args, **kwargs):
                        pass
                    transformers.trainer._setup_distributed = noop_setup_distributed
                
                # Retry training
                trainer.train()
                print("Training completed successfully after retry")
                training_successful = True
            except Exception as e2:
                print(f"Retry also failed: {e2}")
                training_successful = False
        else:
            training_successful = False
        
        # 오류가 발생해도 최소한의 학습은 진행되었을 수 있으므로 확인
        if not training_successful:
            try:
                # Checkpoint가 있는지 확인
                import glob
                checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
                if checkpoints:
                    print(f"Found {len(checkpoints)} checkpoints - loading latest checkpoint")
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    print(f"Loading latest checkpoint: {latest_checkpoint}")
                    # Load T5 model
                    model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint).to(device)
                    trainer.model = model
                    print(f"OK: Loaded model from checkpoint")
                    training_successful = True
            except Exception as e2:
                print(f"WARNING:  Failed to load from checkpoint: {e2}")
                pass  # Silent fail - will use untrained model
    
    # 학습이 성공했는지 확인
    if not training_successful:
        print("\n" + "="*80)
        print("WARNING:  CRITICAL WARNING: Generator training failed or incomplete!")
        print("WARNING:  The model may not be trained, which will result in EM=0, F1=0")
        print("="*80)
        print("\nTo fix this:")
        print("1. Check the training logs above for errors")
        print("2. Try reducing batch size or number of epochs")
        print("3. Ensure sufficient GPU memory")
        print("="*80 + "\n")
    
    # Model saving is already handled in training block
    # Only save tokenizer if not already saved
    if not os.path.exists(os.path.join(output_dir, "tokenizer_config.json")):
        print(f"Saving tokenizer to {output_dir}")
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved")
    
    return model, tokenizer

