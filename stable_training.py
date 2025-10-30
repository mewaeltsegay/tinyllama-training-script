#!/usr/bin/env python3
"""
Stable TinyLlama Training Script

Production-ready version of the simple, stable training approach.
Based on the successful test_basic_training.py results.
"""

import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stable_training.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class TigrinyaDataset(Dataset):
    """Simple dataset class for Tigrinya JSONL data."""
    
    def __init__(self, tokenized_texts: List[Dict[str, Any]]):
        self.data = tokenized_texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long)
        }


class StableTrainingCallback(TrainerCallback):
    """Simple callback for monitoring training stability."""
    
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer
        self.step_count = 0
        self.nan_detected = False
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log metrics to tensorboard."""
        if logs and self.tb_writer:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if not (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))):
                        self.tb_writer.add_scalar(key, value, state.global_step)
                    else:
                        logger.error(f"Invalid metric {key}: {value}")
                        self.nan_detected = True
            
            self.tb_writer.flush()
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Monitor for gradient issues."""
        self.step_count += 1
        
        # Check gradients every 10 steps
        if self.step_count % 10 == 0:
            nan_count = 0
            inf_count = 0
            total_norm = 0.0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_count += 1
                    if torch.isinf(param.grad).any():
                        inf_count += 1
                    total_norm += param.grad.data.norm(2).item() ** 2
            
            total_norm = total_norm ** 0.5
            
            if nan_count > 0 or inf_count > 0:
                logger.error(f"Gradient issues at step {state.global_step}: NaN={nan_count}, Inf={inf_count}")
                logger.error("Stopping training due to gradient instability")
                control.should_training_stop = True
                self.nan_detected = True
            else:
                logger.debug(f"Step {state.global_step}: gradient norm = {total_norm:.4f}")


def load_dataset(
    dataset_path: str, 
    tokenizer, 
    max_length: int = 512, 
    max_samples: int = None
) -> TigrinyaDataset:
    """
    Load and tokenize dataset from JSONL file.
    
    Args:
        dataset_path: Path to JSONL file
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to load
        
    Returns:
        TigrinyaDataset instance
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    texts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {i+1}")
                continue
    
    logger.info(f"Loaded {len(texts)} texts from dataset")
    
    # Tokenize texts
    tokenized = []
    for i, text in enumerate(texts):
        try:
            tokens = tokenizer(
                text, 
                truncation=True, 
                padding=False, 
                max_length=max_length,
                return_tensors=None
            )
            
            # Skip very short sequences
            if len(tokens['input_ids']) > 5:
                tokenized.append(tokens)
            
        except Exception as e:
            logger.warning(f"Failed to tokenize text {i}: {e}")
            continue
    
    logger.info(f"Successfully tokenized {len(tokenized)} sequences")
    
    if len(tokenized) == 0:
        raise ValueError("No valid sequences found in dataset")
    
    return TigrinyaDataset(tokenized)


def create_training_arguments(
    output_dir: str,
    hardware_config: Dict[str, Any],
    num_epochs: int = 1,
    warmup_steps: int = None,
    save_steps: int = None,
    logging_steps: int = None,
    eval_dataset_available: bool = False
) -> TrainingArguments:
    """Create hardware-optimized training arguments."""
    
    # Use hardware config defaults, override with specific values if provided
    learning_rate = hardware_config.get("learning_rate", 5e-6)
    batch_size = hardware_config.get("batch_size", 1)
    gradient_accumulation_steps = hardware_config.get("gradient_accumulation_steps", 8)
    max_grad_norm = hardware_config.get("max_grad_norm", 1.0)
    use_bf16 = hardware_config.get("use_bf16", False)
    dataloader_num_workers = hardware_config.get("dataloader_num_workers", 0)
    torch_compile = hardware_config.get("torch_compile", False)
    pin_memory = hardware_config.get("pin_memory", True)
    
    # Set defaults if not provided
    if warmup_steps is None:
        warmup_steps = hardware_config.get("warmup_steps", 100)
    if save_steps is None:
        save_steps = hardware_config.get("save_steps", 500)
    if logging_steps is None:
        logging_steps = hardware_config.get("logging_steps", 10)
    
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Learning rate and optimization
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=hardware_config.get("lr_scheduler_type", "cosine"),
        weight_decay=hardware_config.get("weight_decay", 0.01),
        
        # Optimizer settings
        optim=hardware_config.get("optim", "adamw_torch"),
        adam_beta1=hardware_config.get("adam_beta1", 0.9),
        adam_beta2=hardware_config.get("adam_beta2", 0.999),
        adam_epsilon=hardware_config.get("adam_epsilon", 1e-8),
        
        # Mixed precision
        bf16=use_bf16,
        fp16=not use_bf16 and hardware_config.get("fp16", False),  # Use FP16 if not BF16
        tf32=hardware_config.get("tf32", False),  # Enable TF32 on supported hardware
        
        # Gradient clipping
        max_grad_norm=max_grad_norm,
        
        # Logging and saving
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps" if eval_dataset_available else "no",
        eval_steps=hardware_config.get("eval_steps", save_steps) if eval_dataset_available else None,
        
        # Performance settings
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=hardware_config.get("dataloader_drop_last", True),
        group_by_length=hardware_config.get("group_by_length", False),
        remove_unused_columns=hardware_config.get("remove_unused_columns", False),
        
        # Advanced features
        torch_compile=torch_compile,
        
        # Reporting
        report_to=[],  # No external reporting
        disable_tqdm=False,
        
        # Reproducibility
        seed=42,
        data_seed=42,
    )
    
    return args


def get_hardware_config() -> Dict[str, Any]:
    """Get hardware-optimized configuration based on detected GPU."""
    config = {
        "use_bf16": False,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "learning_rate": 5e-6,
        "dataloader_num_workers": 0,
        "torch_compile": False,
        "pin_memory": True
    }
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU configuration")
        return config
    
    try:
        device_name = torch.cuda.get_device_name(0).lower()
        memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
        
        logger.info(f"Detected GPU: {device_name} ({memory_gb}GB)")
        
        if "h100" in device_name:
            # H100 80GB - High-performance configuration
            logger.info("Configuring for H100 80GB - High Performance Mode")
            config.update({
                "use_bf16": True,           # BF16 is optimal on H100
                "batch_size": 8,           # Much larger batch size
                "gradient_accumulation_steps": 4,  # Reduce accumulation since batch is larger
                "max_grad_norm": 1.0,      # Standard clipping
                "learning_rate": 1e-4,     # Higher learning rate for faster training
                "dataloader_num_workers": 8,  # Utilize CPU cores
                "torch_compile": True,     # Enable compilation for speed
                "pin_memory": True,
                "warmup_ratio": 0.1,       # 10% warmup
                "lr_scheduler_type": "cosine_with_restarts",
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,        # Optimized for large models
                "adam_epsilon": 1e-8,
                "max_length": 2048,        # Longer sequences
                "save_steps": 100,         # More frequent saves
                "eval_steps": 100,
                "logging_steps": 10,
                "dataloader_drop_last": True,
                "group_by_length": True,   # Optimize batching
                "length_column_name": "length",
                "remove_unused_columns": False,
                "optim": "adamw_torch_fused",  # Fused optimizer for H100
                "tf32": True,              # Enable TF32 for even better performance
            })
            
        elif "a100" in device_name:
            # A100 - High performance but slightly more conservative
            logger.info("Configuring for A100 - High Performance Mode")
            config.update({
                "use_bf16": True,
                "batch_size": 6,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "learning_rate": 8e-5,
                "dataloader_num_workers": 6,
                "torch_compile": True,
                "pin_memory": True,
                "max_length": 1536,
                "optim": "adamw_torch_fused",
                "tf32": True,
            })
            
        elif any(gpu in device_name for gpu in ['rtx', '4090', '3090']):
            # Consumer GPUs - More conservative
            logger.info(f"Configuring for consumer GPU: {device_name}")
            if memory_gb >= 20:  # RTX 4090, 3090
                config.update({
                    "use_bf16": False,  # Use FP16 for consumer cards
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 5e-5,
                    "dataloader_num_workers": 4,
                    "max_length": 1024,
                })
            else:  # Smaller consumer GPUs
                config.update({
                    "use_bf16": False,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "learning_rate": 3e-5,
                    "dataloader_num_workers": 2,
                    "max_length": 512,
                })
        else:
            logger.info(f"Using default configuration for {device_name}")
            
    except Exception as e:
        logger.warning(f"Could not detect GPU details: {e}, using default configuration")
    
    return config


def setup_h100_optimizations():
    """Setup additional optimizations for H100."""
    try:
        # Enable TF32 for even better performance on H100
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            if "h100" in device_name:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 optimizations for H100")
                
                # Set optimal memory allocation strategy
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
                logger.info("Configured memory allocation for H100")
                
                # Enable CUDA graphs if available (experimental)
                try:
                    torch._C._cuda_set_sync_debug_mode(0)  # Disable sync debug for performance
                    logger.info("Disabled CUDA sync debug for performance")
                except:
                    pass
                    
    except Exception as e:
        logger.warning(f"Could not setup H100 optimizations: {e}")


def log_system_info():
    """Log detailed system information for optimization."""
    logger.info("=== System Information ===")
    
    # CPU info
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        logger.info(f"CPU: {cpu_count} cores")
        if cpu_freq:
            logger.info(f"CPU Frequency: {cpu_freq.current:.0f} MHz")
        logger.info(f"RAM: {memory.total // 1024**3} GB ({memory.available // 1024**3} GB available)")
        
    except ImportError:
        logger.info("psutil not available, skipping detailed CPU info")
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory // 1024**3
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Memory: {memory_gb} GB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            logger.info(f"  Multiprocessors: {props.multi_processor_count}")
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(i) // 1024**3
            reserved = torch.cuda.memory_reserved(i) // 1024**3
            logger.info(f"  Current Usage: {allocated} GB allocated, {reserved} GB reserved")
    
    # PyTorch info
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")


def validate_setup(model, tokenizer, train_dataset) -> bool:
    """
    Validate training setup before starting.
    
    Returns:
        True if setup is valid, False otherwise
    """
    logger.info("Validating training setup...")
    
    # Check model
    if model is None:
        logger.error("Model is None")
        return False
    
    # Check dataset
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Training dataset is empty or None")
        return False
    
    # Check tokenizer
    if tokenizer is None:
        logger.error("Tokenizer is None")
        return False
    
    # Test a small batch
    try:
        logger.info("Testing data collator and forward pass...")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=None
        )
        
        # Get a few samples
        samples = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
        batch = data_collator(samples)
        
        logger.info(f"Test batch shape: {batch['input_ids'].shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model = model.cuda()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            if loss is None:
                logger.error("Model returned None loss")
                return False
            
            loss_value = loss.item()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Model returned invalid loss: {loss_value}")
                return False
            
            logger.info(f"Test forward pass successful, loss: {loss_value:.4f}")
        
    except Exception as e:
        logger.error(f"Setup validation failed: {e}")
        return False
    
    logger.info("Training setup validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stable TinyLlama Training")
    
    # Model and data
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--dataset-path", default="dataset/train.jsonl")
    parser.add_argument("--output-dir", default="output_stable")
    
    # Training parameters (will be overridden by hardware detection)
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate (auto-optimized for hardware)")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (auto-optimized for hardware)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation (auto-optimized)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length (auto-optimized for hardware)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    
    # Dataset limits
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    
    # Hardware options
    parser.add_argument("--force-bf16", action="store_true", help="Force BF16 usage")
    parser.add_argument("--force-fp32", action="store_true", help="Force FP32 usage")
    
    args = parser.parse_args()
    
    logger.info("=== Stable TinyLlama Training ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Initial Learning rate: {args.learning_rate} (may be auto-optimized)")
    logger.info(f"Initial Batch size: {args.batch_size} (may be auto-optimized)")
    logger.info(f"Initial Gradient accumulation: {args.gradient_accumulation_steps} (may be auto-optimized)")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Max samples: {args.max_samples}")
    
    # Log detailed system information
    log_system_info()
    
    # Setup hardware-specific optimizations
    setup_h100_optimizations()
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,  # Start with FP32 for stability
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        logger.info(f"Model loaded with {model.num_parameters():,} parameters")
        
        # Load dataset
        train_dataset = load_dataset(
            args.dataset_path,
            tokenizer,
            max_length=args.max_length,
            max_samples=args.max_samples
        )
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate setup
        if not validate_setup(model, tokenizer, train_dataset):
            logger.error("Setup validation failed!")
            return 1
        
        # Get hardware-optimized configuration
        hardware_config = get_hardware_config()
        
        # Override with command line arguments if provided
        if args.force_fp32:
            hardware_config["use_bf16"] = False
            hardware_config["fp16"] = False
        elif args.force_bf16:
            hardware_config["use_bf16"] = True
            hardware_config["fp16"] = False
        
        # Override hardware config with explicit command line args
        if hasattr(args, 'learning_rate') and args.learning_rate != 5e-6:
            hardware_config["learning_rate"] = args.learning_rate
        if hasattr(args, 'batch_size') and args.batch_size != 1:
            hardware_config["batch_size"] = args.batch_size
        if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps != 8:
            hardware_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if hasattr(args, 'max_grad_norm') and args.max_grad_norm != 1.0:
            hardware_config["max_grad_norm"] = args.max_grad_norm
        
        # Update max_length from hardware config
        if "max_length" in hardware_config and args.max_length == 512:
            args.max_length = hardware_config["max_length"]
            logger.info(f"Using hardware-optimized max_length: {args.max_length}")
        
        logger.info("Hardware Configuration:")
        for key, value in hardware_config.items():
            logger.info(f"  {key}: {value}")
        
        # Create training arguments
        training_args = create_training_arguments(
            output_dir=args.output_dir,
            hardware_config=hardware_config,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            eval_dataset_available=False  # No eval dataset in this version
        )
        
        # Create data collator
        pad_to_multiple_of = 8 if hardware_config.get("use_bf16", False) else None
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        
        # Create tensorboard writer
        tb_writer = SummaryWriter(log_dir=Path(args.output_dir) / "tensorboard")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Add monitoring callback
        callback = StableTrainingCallback(tb_writer)
        trainer.add_callback(callback)
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Check if training was successful
        if callback.nan_detected:
            logger.error("Training failed due to numerical instability")
            return 1
        
        # Save the final model
        trainer.save_model()
        trainer.save_state()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        logger.info(f"Training runtime: {train_result.training_runtime:.1f}s")
        logger.info(f"Samples per second: {train_result.train_samples_per_second:.2f}")
        
        # Close tensorboard
        tb_writer.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())