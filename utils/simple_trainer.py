#!/usr/bin/env python3
"""
Simple, Stable Training Loop

Back-to-basics approach that focuses on fundamental stability without overcomplication.
Key principles:
1. Conservative learning rates
2. Simple gradient clipping
3. Basic mixed precision (BF16 only on H100)
4. Minimal logging
5. No complex recovery mechanisms
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from transformers import TrainingArguments, Trainer
from torch.utils.tensorboard import SummaryWriter
import os

logger = logging.getLogger(__name__)


class SimpleStableTrainer:
    """
    Simple, stable trainer that avoids overcomplication.
    
    Focus on proven stability techniques:
    - Conservative learning rates
    - Simple gradient clipping
    - Basic mixed precision
    - Fail-fast on NaN (no complex recovery)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./output",
        learning_rate: float = 1e-5,  # Very conservative
        max_grad_norm: float = 0.5,   # Conservative clipping
        use_bf16: bool = None,        # Auto-detect based on hardware
        batch_size: int = 1,          # Start small
        gradient_accumulation_steps: int = 8,
        num_epochs: int = 1,
        save_steps: int = 1000,
        logging_steps: int = 10,
        warmup_steps: int = 100
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.warmup_steps = warmup_steps
        
        # Auto-detect BF16 support
        if use_bf16 is None:
            self.use_bf16 = self._should_use_bf16()
        else:
            self.use_bf16 = use_bf16
        
        # Setup tensorboard
        self.tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
        
        logger.info(f"SimpleStableTrainer initialized:")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Max grad norm: {self.max_grad_norm}")
        logger.info(f"  BF16: {self.use_bf16}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
    
    def _should_use_bf16(self) -> bool:
        """Auto-detect if BF16 should be used based on hardware."""
        if not torch.cuda.is_available():
            return False
        
        try:
            device_name = torch.cuda.get_device_name(0).lower()
            # Only use BF16 on modern hardware that supports it well
            if any(gpu in device_name for gpu in ['h100', 'a100']):
                logger.info(f"Detected {device_name}, enabling BF16")
                return True
            else:
                logger.info(f"Detected {device_name}, using FP32 for stability")
                return False
        except Exception:
            logger.warning("Could not detect GPU, using FP32")
            return False
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create simple, stable training arguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            
            # Learning rate and optimization
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            
            # Mixed precision - only BF16 on supported hardware
            bf16=self.use_bf16,
            fp16=False,  # Avoid FP16 for stability
            
            # Gradient clipping
            max_grad_norm=self.max_grad_norm,
            
            # Logging and saving
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=3,
            
            # Evaluation
            evaluation_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.save_steps if self.eval_dataset else None,
            
            # Stability settings
            dataloader_pin_memory=True,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            
            # Disable advanced features that can cause instability
            torch_compile=False,
            optim="adamw_torch",  # Use standard optimizer
            
            # Reporting
            report_to=[],  # No external reporting
            disable_tqdm=False,
            
            # Memory management
            dataloader_drop_last=True,
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
    
    def create_data_collator(self):
        """Create simple data collator for causal LM."""
        from transformers import DataCollatorForLanguageModeling
        
        # Use the standard transformers data collator
        # It's well-tested and stable
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if self.use_bf16 else None,
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Run simple, stable training.
        
        Returns:
            Training metrics dictionary
        """
        logger.info("Starting simple stable training...")
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create data collator
        data_collator = self.create_data_collator()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Add custom callback for monitoring
        trainer.add_callback(SimpleTrainingCallback(self.tb_writer))
        
        try:
            # Train the model
            train_result = trainer.train()
            
            # Save the final model
            trainer.save_model()
            trainer.save_state()
            
            # Log final metrics
            logger.info("Training completed successfully!")
            logger.info(f"Final train loss: {train_result.training_loss:.4f}")
            
            return {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.training_runtime,
                "train_samples_per_second": train_result.train_samples_per_second,
                "train_steps_per_second": train_result.train_steps_per_second,
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.tb_writer:
                self.tb_writer.close()
    
    def validate_setup(self) -> bool:
        """
        Validate training setup before starting.
        
        Returns:
            True if setup is valid, False otherwise
        """
        logger.info("Validating training setup...")
        
        # Check model
        if self.model is None:
            logger.error("Model is None")
            return False
        
        # Check dataset
        if self.train_dataset is None or len(self.train_dataset) == 0:
            logger.error("Training dataset is empty or None")
            return False
        
        # Check tokenizer
        if self.tokenizer is None:
            logger.error("Tokenizer is None")
            return False
        
        # Check if tokenizer has pad token
        if self.tokenizer.pad_token is None:
            logger.warning("Tokenizer has no pad token, setting to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Test a small batch
        try:
            logger.info("Testing data collator with small batch...")
            data_collator = self.create_data_collator()
            
            # Get a few samples
            samples = [self.train_dataset[i] for i in range(min(3, len(self.train_dataset)))]
            batch = data_collator(samples)
            
            logger.info(f"Test batch shape: {batch['input_ids'].shape}")
            logger.info(f"Test batch has labels: {'labels' in batch}")
            
            # Test forward pass
            logger.info("Testing model forward pass...")
            self.model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    self.model = self.model.cuda()
                
                outputs = self.model(**batch)
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


class SimpleTrainingCallback:
    """Simple callback for basic monitoring."""
    
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer
        self.step_count = 0
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log metrics to tensorboard."""
        if logs and self.tb_writer:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))):
                    self.tb_writer.add_scalar(key, value, state.global_step)
            
            self.tb_writer.flush()
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check for NaN gradients and fail fast."""
        self.step_count += 1
        
        # Check gradients every few steps
        if self.step_count % 10 == 0:
            nan_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_count += 1
            
            if nan_count > 0:
                logger.error(f"NaN gradients detected in {nan_count} parameters at step {state.global_step}")
                logger.error("Stopping training - check learning rate, data, or model initialization")
                control.should_training_stop = True


def create_simple_trainer(
    model: nn.Module,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir: str = "./output",
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 1
) -> SimpleStableTrainer:
    """
    Factory function to create a simple stable trainer.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        output_dir: Output directory
        learning_rate: Learning rate (conservative default)
        batch_size: Batch size (start small)
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of epochs
        
    Returns:
        SimpleStableTrainer instance
    """
    return SimpleStableTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs
    )


if __name__ == "__main__":
    print("Simple Stable Trainer")
    print("Back-to-basics approach for stable training without overcomplication")