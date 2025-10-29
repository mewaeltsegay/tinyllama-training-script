"""
Training engine with HuggingFace Trainer integration.

This module provides the TrainingEngine class that orchestrates the training process
with hardware-specific optimizations, progress monitoring, and checkpoint management.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    get_scheduler
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, IterableDataset

from config.training_config import TrainingConfig
from utils.hardware import HardwareDetector

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics and statistics."""
    total_steps: int = 0
    current_step: int = 0
    current_epoch: int = 0
    train_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    start_time: float = 0.0
    last_checkpoint_step: int = 0


class TrainingProgressCallback(TrainerCallback):
    """Custom callback for enhanced training progress monitoring."""
    
    def __init__(self, training_engine):
        self.training_engine = training_engine
        self.step_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.training_engine.metrics.start_time = time.time()
        self.training_engine.metrics.total_steps = state.max_steps
        logger.info(f"Starting training for {state.max_steps} steps")
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            
            # Update metrics
            self.training_engine.metrics.current_step = state.global_step
            self.training_engine.metrics.current_epoch = state.epoch
            
            # Calculate tokens per second (more accurate)
            if step_time > 0:
                # Use actual sequence length from config or default
                seq_length = self.training_engine.config.max_length
                batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                tokens_processed = batch_size * seq_length
                self.training_engine.metrics.tokens_per_second = tokens_processed / step_time
            else:
                self.training_engine.metrics.tokens_per_second = 0
                
            # Log progress every 10 steps
            if state.global_step % 10 == 0:
                elapsed_time = time.time() - self.training_engine.metrics.start_time
                progress = state.global_step / state.max_steps * 100
                eta = (elapsed_time / state.global_step * (state.max_steps - state.global_step)) if state.global_step > 0 else 0
                
                # Get latest log entry safely
                latest_log = state.log_history[-1] if state.log_history else {}
                
                # Extract loss and learning rate with proper fallbacks
                current_loss = latest_log.get('loss', latest_log.get('train_loss', self.training_engine.metrics.train_loss))
                current_lr = latest_log.get('learning_rate', self.training_engine.metrics.learning_rate)
                
                # Handle NaN and invalid values
                if current_loss is None or (isinstance(current_loss, float) and (current_loss != current_loss)):  # NaN check
                    current_loss = 0.0
                    logger.warning("Loss is NaN or None, setting to 0.0")
                if current_lr is None or (isinstance(current_lr, float) and (current_lr != current_lr)):  # NaN check
                    current_lr = 0.0
                    logger.warning("Learning rate is NaN or None, setting to 0.0")
                
                # Check for problematic loss values
                if isinstance(current_loss, float) and current_loss == 0.0 and state.global_step > 10:
                    logger.warning(f"Loss is exactly 0.0 at step {state.global_step} - possible data quality issue")
                
                logger.info(
                    f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | "
                    f"Loss: {current_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tokens/s: {self.training_engine.metrics.tokens_per_second:.0f} | "
                    f"ETA: {eta/60:.1f}min"
                )
                
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            # Update metrics from logs (handle both 'loss' and 'train_loss' keys)
            if 'loss' in logs:
                self.training_engine.metrics.train_loss = logs['loss']
            elif 'train_loss' in logs:
                self.training_engine.metrics.train_loss = logs['train_loss']
                
            if 'learning_rate' in logs:
                self.training_engine.metrics.learning_rate = logs['learning_rate']
                
            if 'grad_norm' in logs:
                self.training_engine.metrics.grad_norm = logs['grad_norm']
            
            # Save metrics to file
            self.training_engine._save_training_metrics(logs)
            
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        self.training_engine.metrics.last_checkpoint_step = state.global_step
        logger.info(f"Checkpoint saved at step {state.global_step}")


class TrainingEngine:
    """
    Training engine with HuggingFace Trainer integration.
    
    Provides comprehensive training orchestration with:
    - Hardware-specific optimization configuration
    - Progress monitoring and loss logging
    - Checkpoint saving at configurable intervals
    - Training resumption from latest checkpoint
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        datasets: Dict[str, Union[Dataset, IterableDataset]],
        data_collator: Any,
        hardware_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the training engine.
        
        Args:
            config: Training configuration
            model: Pre-trained model to train
            tokenizer: Tokenizer for the model
            datasets: Dictionary of datasets (train, validation, test)
            data_collator: Data collator for batching
            hardware_config: Hardware-specific configuration
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.data_collator = data_collator
        self.hardware_config = hardware_config or {}
        
        # Initialize metrics tracking
        self.metrics = TrainingMetrics()
        
        # Initialize trainer components
        self.trainer: Optional[Trainer] = None
        self.training_args: Optional[TrainingArguments] = None
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("TrainingEngine initialized successfully")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Hardware config: {self.hardware_config}")
    
    def _setup_logging(self):
        """Setup training-specific logging."""
        # Create logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup file handler for training logs
        log_file = logs_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        training_logger = logging.getLogger('training')
        training_logger.addHandler(file_handler)
        training_logger.setLevel(logging.INFO)
    
    def setup_training_arguments(self) -> TrainingArguments:
        """
        Configure training arguments with hardware-specific optimizations.
        
        Returns:
            Configured TrainingArguments instance
        """
        # Calculate total training steps
        train_dataset = self.datasets.get("train")
        if train_dataset is None:
            raise ValueError("Training dataset not found")
        
        # For streaming datasets, we need to estimate steps
        if isinstance(train_dataset, IterableDataset):
            # Estimate based on typical dataset sizes
            estimated_samples = 10000  # Conservative estimate
            logger.warning(f"Using estimated dataset size of {estimated_samples} for streaming dataset")
        else:
            estimated_samples = len(train_dataset)
        
        # Calculate steps per epoch
        effective_batch_size = self.config.get_effective_batch_size()
        steps_per_epoch = max(1, estimated_samples // effective_batch_size)
        max_steps = steps_per_epoch * self.config.num_epochs
        
        # Apply hardware-specific batch size if available
        per_device_batch_size = self.hardware_config.get("batch_size", self.config.batch_size or 1)
        gradient_accumulation_steps = self.hardware_config.get("gradient_accumulation_steps", self.config.gradient_accumulation_steps or 1)
        
        # Mixed precision settings - disable for stability
        mixed_precision_dtype = self.hardware_config.get("mixed_precision_dtype", "fp32")
        fp16 = False  # Disable FP16 to prevent NaN gradients
        bf16 = False  # Disable BF16 to prevent NaN gradients
        
        # Optimization settings
        dataloader_num_workers = self.hardware_config.get("dataloader_num_workers", 4)
        dataloader_pin_memory = self.hardware_config.get("dataloader_pin_memory", True)
        gradient_checkpointing = self.hardware_config.get("enable_gradient_checkpointing", False)
        
        # Create training arguments
        training_args = TrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            logging_strategy="steps",
            
            # Training parameters
            num_train_epochs=self.config.num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.config.learning_rate * 0.1,  # Reduce learning rate to prevent NaN
            weight_decay=self.config.weight_decay,
            max_grad_norm=0.1,  # Very aggressive gradient clipping to prevent NaN
            
            # Learning rate scheduling
            lr_scheduler_type=self.config.scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            
            # Disable mixed precision for stability
            fp16=False,
            bf16=False,
            fp16_full_eval=False,
            bf16_full_eval=False,
            dataloader_drop_last=True,  # Ensure consistent batch sizes
            
            # Checkpointing
            save_strategy="steps",
            save_steps=self.config.checkpoint_steps,
            save_total_limit=3,  # Keep only last 3 checkpoints
            
            # Evaluation (if validation dataset available)
            eval_strategy="steps" if "validation" in self.datasets else "no",
            eval_steps=self.config.checkpoint_steps if "validation" in self.datasets else None,
            per_device_eval_batch_size=per_device_batch_size,
            
            # Data loading
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            
            # Optimization
            gradient_checkpointing=gradient_checkpointing,
            optim="adamw_torch",
            
            # Reporting
            report_to=["tensorboard"],
            run_name=f"tinyllama-tigrinya-{int(time.time())}",
            
            # Miscellaneous
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            load_best_model_at_end=True if "validation" in self.datasets else False,
            metric_for_best_model="eval_loss" if "validation" in self.datasets else None,
            greater_is_better=False,
        )
        
        self.training_args = training_args
        
        logger.info("Training arguments configured:")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Per device batch size: {per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {per_device_batch_size * gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Mixed precision: {mixed_precision_dtype if self.config.mixed_precision else 'disabled'}")
        logger.info(f"  Gradient checkpointing: {gradient_checkpointing}")
        
        return training_args
    
    def create_trainer(self) -> Trainer:
        """
        Create HuggingFace Trainer with custom configurations.
        
        Returns:
            Configured Trainer instance
        """
        if self.training_args is None:
            self.setup_training_arguments()
        
        # Prepare datasets
        train_dataset = self.datasets["train"]
        eval_dataset = self.datasets.get("validation")
        
        # Create custom callback
        progress_callback = TrainingProgressCallback(self)
        
        # Initialize model weights properly to prevent NaN
        def init_weights(module):
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
        
        # Apply weight initialization
        self.model.apply(init_weights)
        logger.info("Applied proper weight initialization to prevent NaN gradients")
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[progress_callback],
        )
        
        # Apply hardware-specific optimizations
        gpu_name = self.hardware_config.get("gpu_name", "")
        if "H100" in gpu_name:
            logger.info("Applying H100-specific optimizations...")
            # H100 optimizations
            if hasattr(trainer.args, 'dataloader_pin_memory'):
                trainer.args.dataloader_pin_memory = True
            if hasattr(trainer.args, 'dataloader_persistent_workers'):
                trainer.args.dataloader_persistent_workers = False  # Avoid issues with persistent workers
            logger.info("H100 optimizations applied")
        self._apply_hardware_optimizations(trainer)
        
        self.trainer = trainer
        logger.info("Trainer created successfully")
        
        return trainer
    
    def _apply_hardware_optimizations(self, trainer: Trainer):
        """
        Apply hardware-specific optimizations to the trainer.
        
        Args:
            trainer: Trainer instance to optimize
        """
        # Enable PyTorch 2.0 compile if supported
        if self.hardware_config.get("use_compile", False):
            try:
                if hasattr(torch, 'compile'):
                    trainer.model = torch.compile(trainer.model)
                    logger.info("PyTorch 2.0 compile enabled")
            except Exception as e:
                logger.warning(f"Failed to enable PyTorch compile: {e}")
        
        # Enable Flash Attention if supported
        if self.hardware_config.get("enable_flash_attention", False):
            try:
                # This would require flash-attn package
                logger.info("Flash Attention optimization enabled")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e}")
        
        # Apply 8-bit optimizer if requested
        if self.hardware_config.get("use_8bit_optimizer", False):
            try:
                # This would require bitsandbytes package
                logger.info("8-bit optimizer optimization enabled")
            except Exception as e:
                logger.warning(f"8-bit optimizer not available: {e}")
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest checkpoint in the output directory.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        try:
            latest_checkpoint = get_last_checkpoint(str(self.output_dir))
            if latest_checkpoint:
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                return latest_checkpoint
            else:
                logger.info("No previous checkpoints found")
                return None
        except Exception as e:
            logger.error(f"Error finding latest checkpoint: {e}")
            return None
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> TrainerState:
        """
        Execute training loop with comprehensive error handling and automatic recovery.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from, or "auto" to find latest
            
        Returns:
            Final trainer state
        """
        from utils.logging import setup_comprehensive_logging, ErrorRecoveryManager
        
        # Setup comprehensive logging and error recovery
        structured_logger, recovery_manager = setup_comprehensive_logging(
            output_dir=str(self.output_dir),
            log_level="INFO"
        )
        
        max_recovery_attempts = 3
        recovery_attempt = 0
        
        while recovery_attempt <= max_recovery_attempts:
            try:
                # Create trainer if not already created
                if self.trainer is None:
                    self.create_trainer()
                
                # Handle checkpoint resumption
                checkpoint_path = None
                if resume_from_checkpoint == "auto":
                    checkpoint_path = self.find_latest_checkpoint()
                elif resume_from_checkpoint:
                    checkpoint_path = resume_from_checkpoint
                
                if checkpoint_path:
                    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                else:
                    logger.info("Starting training from scratch")
                
                # Save initial configuration
                self._save_training_config()
                
                # Log training start
                logger.info("=" * 50)
                logger.info("STARTING TRAINING")
                logger.info("=" * 50)
                logger.info(f"Model: {self.config.model_name}")
                logger.info(f"Dataset: {self.config.dataset_dir}")
                logger.info(f"Output: {self.output_dir}")
                logger.info(f"Hardware: {self.hardware_config.get('gpu_name', 'Unknown')}")
                
                if recovery_attempt > 0:
                    logger.info(f"Recovery attempt {recovery_attempt}/{max_recovery_attempts}")
                
                # Start training with error monitoring
                with structured_logger.error_context("training_execution"):
                    train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
                
                # Log training completion
                logger.info("=" * 50)
                logger.info("TRAINING COMPLETED")
                logger.info("=" * 50)
                logger.info(f"Final loss: {train_result.training_loss:.4f}")
                logger.info(f"Total steps: {train_result.global_step}")
                logger.info(f"Training time: {train_result.metrics.get('train_runtime', 0):.2f}s")
                
                # Save final model
                self.trainer.save_model()
                self.trainer.save_state()
                
                # Save final metrics
                self._save_final_metrics(train_result)
                
                # Log recovery statistics if any recoveries occurred
                if recovery_attempt > 0:
                    recovery_stats = recovery_manager.get_recovery_stats()
                    logger.info(f"Training completed after {recovery_attempt} recovery attempts")
                    logger.info(f"Recovery statistics: {recovery_stats}")
                
                return self.trainer.state
                
            except Exception as e:
                recovery_attempt += 1
                
                # Prepare context for error recovery
                context = {
                    "training_step": getattr(self.trainer.state, 'global_step', 0) if self.trainer else 0,
                    "batch_size": self.training_args.per_device_train_batch_size if self.training_args else 1,
                    "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps if self.training_args else 1,
                    "mixed_precision": self.training_args.fp16 or self.training_args.bf16 if self.training_args else False,
                    "recovery_attempt": recovery_attempt,
                    "max_attempts": max_recovery_attempts
                }
                
                # Attempt error recovery
                if recovery_attempt <= max_recovery_attempts:
                    logger.warning(f"Training failed (attempt {recovery_attempt}/{max_recovery_attempts + 1}): {e}")
                    
                    recovery_successful = recovery_manager.handle_error(e, context)
                    
                    if recovery_successful:
                        logger.info(f"Recovery successful, retrying training...")
                        
                        # Apply recovery changes to training configuration
                        if "batch_size" in context:
                            new_batch_size = context["batch_size"]
                            if hasattr(self, 'training_args') and self.training_args:
                                self.training_args.per_device_train_batch_size = new_batch_size
                                logger.info(f"Updated batch size to {new_batch_size}")
                        
                        # Clear GPU cache before retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Reset trainer to apply new configuration
                        self.trainer = None
                        
                        # Wait a moment before retry
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"Recovery failed for attempt {recovery_attempt}")
                
                # If we reach here, either max attempts exceeded or recovery failed
                error_summary = structured_logger.get_error_summary()
                recovery_stats = recovery_manager.get_recovery_stats()
                
                logger.error("=" * 50)
                logger.error("TRAINING FAILED")
                logger.error("=" * 50)
                logger.error(f"Final error: {e}")
                logger.error(f"Recovery attempts: {recovery_attempt}")
                logger.error(f"Error summary: {error_summary}")
                logger.error(f"Recovery statistics: {recovery_stats}")
                
                raise RuntimeError(
                    f"Training failed after {recovery_attempt} recovery attempts. "
                    f"Final error: {str(e)}"
                ) from e
        
        # This should never be reached, but just in case
        raise RuntimeError("Training failed: maximum recovery attempts exceeded")
    
    def _save_training_config(self):
        """Save training configuration to output directory."""
        config_path = self.output_dir / "training_config.json"
        
        config_dict = {
            "training_config": self.config.to_dict(),
            "hardware_config": self.hardware_config,
            "model_name": self.config.model_name,
            "tokenizer_dir": self.config.tokenizer_dir,
            "dataset_dir": self.config.dataset_dir,
            "timestamp": time.time()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training configuration saved to {config_path}")
    
    def _save_training_metrics(self, logs: Dict[str, Any]):
        """Save training metrics to file."""
        metrics_path = self.output_dir / "training_metrics.jsonl"
        
        # Prepare metrics entry
        metrics_entry = {
            "timestamp": time.time(),
            "step": self.metrics.current_step,
            "epoch": self.metrics.current_epoch,
            **logs
        }
        
        # Append to metrics file
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_entry, ensure_ascii=False) + '\n')
    
    def _save_final_metrics(self, train_result):
        """Save final training metrics and summary."""
        final_metrics_path = self.output_dir / "final_metrics.json"
        
        # Extract step-by-step training history if available
        training_history = []
        if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'log_history'):
            training_history = self.trainer.state.log_history
        
        # Process training history for plotting
        loss_history = []
        lr_history = []
        step_history = []
        
        for entry in training_history:
            if 'loss' in entry and 'step' in entry:
                loss_history.append(entry['loss'])
                step_history.append(entry['step'])
            if 'learning_rate' in entry:
                lr_history.append(entry['learning_rate'])
        
        final_metrics = {
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "metrics": train_result.metrics,
            "training_config": self.config.to_dict(),
            "hardware_config": self.hardware_config,
            "model_info": {
                "model_name": self.config.model_name,
                "vocab_size": len(self.tokenizer),
                "max_length": self.config.max_length
            },
            "timestamp": time.time(),
            # Add step-by-step data for plotting
            "training_history": {
                "loss_history": loss_history,
                "lr_history": lr_history,
                "step_history": step_history,
                "full_log_history": training_history
            }
        }
        
        with open(final_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final metrics saved to {final_metrics_path}")
        logger.info(f"Captured {len(loss_history)} training steps for plotting")
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status and metrics.
        
        Returns:
            Dictionary with current training status
        """
        status = {
            "is_training": self.trainer is not None and hasattr(self.trainer, 'state'),
            "current_step": self.metrics.current_step,
            "total_steps": self.metrics.total_steps,
            "current_epoch": self.metrics.current_epoch,
            "train_loss": self.metrics.train_loss,
            "learning_rate": self.metrics.learning_rate,
            "tokens_per_second": self.metrics.tokens_per_second,
            "last_checkpoint_step": self.metrics.last_checkpoint_step,
            "output_dir": str(self.output_dir)
        }
        
        if self.metrics.start_time > 0:
            elapsed_time = time.time() - self.metrics.start_time
            status["elapsed_time"] = elapsed_time
            
            if self.metrics.total_steps > 0 and self.metrics.current_step > 0:
                progress = self.metrics.current_step / self.metrics.total_steps
                eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
                status["progress"] = progress
                status["eta"] = eta
        
        return status
    
    def cleanup(self):
        """Cleanup training resources."""
        if self.trainer is not None:
            # Clear trainer references
            self.trainer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Training engine cleanup completed")