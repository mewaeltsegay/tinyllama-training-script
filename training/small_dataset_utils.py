"""
Training utilities for small dataset optimization.

This module provides utilities to integrate small dataset optimization techniques
with the HuggingFace Trainer and other training components.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import Dataset

from utils.small_dataset_optimizer import SmallDatasetOptimizer, SmallDatasetTrainingMixin

logger = logging.getLogger(__name__)


class SmallDatasetTrainingArguments(TrainingArguments):
    """
    Extended TrainingArguments with small dataset optimization parameters.
    """
    
    def __init__(
        self,
        *args,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "warmup_cosine",
        num_cycles: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.scheduler_type = scheduler_type
        self.num_cycles = num_cycles


class SmallDatasetTrainer(Trainer, SmallDatasetTrainingMixin):
    """
    Extended Trainer with small dataset optimization capabilities.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: SmallDatasetTrainingArguments,
        train_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        small_dataset_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the SmallDatasetTrainer.
        
        Args:
            model: The model to train
            args: Training arguments with small dataset parameters
            train_dataset: Training dataset
            tokenizer: Tokenizer instance
            small_dataset_config: Configuration for small dataset optimizations
            **kwargs: Additional arguments for Trainer
        """
        self.small_dataset_config = small_dataset_config or {}
        self.tokenizer = tokenizer
        self.original_train_dataset = train_dataset
        
        # Initialize small dataset optimizer
        self.small_dataset_optimizer = SmallDatasetOptimizer(
            tokenizer=tokenizer,
            max_length=self.small_dataset_config.get("max_length", 512),
            token_dropout_prob=self.small_dataset_config.get("token_dropout_prob", 0.1),
            length_variation_ratio=self.small_dataset_config.get("length_variation_ratio", 0.3),
            augmentation_factor=self.small_dataset_config.get("augmentation_factor", 2),
            seed=self.small_dataset_config.get("seed", None)
        )
        
        # Apply data augmentation to training dataset
        logger.info("Applying small dataset optimizations to training data...")
        augmented_train_dataset = self.small_dataset_optimizer.augment_dataset(train_dataset)
        
        # Initialize parent Trainer with augmented dataset
        super().__init__(
            model=model,
            args=args,
            train_dataset=augmented_train_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # Apply model regularization
        self.small_dataset_optimizer.apply_regularization_to_model(model, args.weight_decay)
        
        # Create gradient clipper
        self.gradient_clipper = self.small_dataset_optimizer.create_gradient_clipper(args.max_grad_norm)
        
        logger.info(f"Initialized SmallDatasetTrainer with augmented dataset: "
                   f"{len(train_dataset)} -> {len(augmented_train_dataset)} examples")
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create optimizer and scheduler with small dataset optimizations.
        
        Args:
            num_training_steps: Total number of training steps
        """
        # Create optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Create learning rate scheduler
        if self.args.scheduler_type == "warmup_cosine":
            self.lr_scheduler = self.small_dataset_optimizer.create_warmup_cosine_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                warmup_ratio=self.args.warmup_ratio,
                num_cycles=self.args.num_cycles
            )
        else:
            # Fallback to default scheduler
            super().create_optimizer_and_scheduler(num_training_steps)
        
        logger.info(f"Created optimizer and scheduler for {num_training_steps} training steps")
    
    def training_step(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a training step with gradient clipping.
        
        Args:
            model: The model being trained
            inputs: The inputs and targets for the model
            
        Returns:
            The loss tensor
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Forward pass
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # Backward pass
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        # Apply gradient clipping
        if self.gradient_clipper:
            grad_norm = self.gradient_clipper(model)
            if hasattr(self, 'state') and self.state.log_history:
                # Log gradient norm for monitoring
                self.state.log_history[-1]["grad_norm"] = grad_norm.item()
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get information about the applied optimizations.
        
        Returns:
            Dictionary with optimization information
        """
        return {
            "original_dataset_size": len(self.original_train_dataset),
            "augmented_dataset_size": len(self.train_dataset),
            "augmentation_ratio": len(self.train_dataset) / len(self.original_train_dataset),
            "weight_decay": self.args.weight_decay,
            "max_grad_norm": self.args.max_grad_norm,
            "warmup_ratio": self.args.warmup_ratio,
            "scheduler_type": self.args.scheduler_type,
            "optimization_config": self.small_dataset_optimizer.get_optimization_config(
                num_training_steps=self.args.max_steps or 1000,
                warmup_ratio=self.args.warmup_ratio,
                weight_decay=self.args.weight_decay,
                max_grad_norm=self.args.max_grad_norm
            )
        }


def create_small_dataset_training_args(
    config: Dict[str, Any],
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    **kwargs
) -> SmallDatasetTrainingArguments:
    """
    Create training arguments optimized for small datasets.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        logging_steps: Steps between logging
        save_steps: Steps between saving checkpoints
        eval_steps: Steps between evaluation
        **kwargs: Additional arguments
        
    Returns:
        SmallDatasetTrainingArguments instance
    """
    args = SmallDatasetTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=config.get("warmup_ratio", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        scheduler_type=config.get("scheduler_type", "warmup_cosine"),
        num_cycles=config.get("num_cycles", 0.5),
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps" if eval_steps > 0 else "no",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard by default
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=config.get("mixed_precision", True) and torch.cuda.is_available(),
        **kwargs
    )
    
    logger.info(f"Created small dataset training arguments: {args}")
    return args


def setup_small_dataset_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: Dict[str, Any],
    data_collator: Any
) -> Tuple[SmallDatasetTrainer, SmallDatasetTrainingArguments]:
    """
    Set up training with small dataset optimizations.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: Configuration dictionary
        data_collator: Data collator for batching
        
    Returns:
        Tuple of (trainer, training_arguments)
    """
    # Create training arguments
    training_args = create_small_dataset_training_args(
        config=config,
        output_dir=config.get("output_dir", "output"),
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 5e-5),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("checkpoint_steps", 500),
        eval_steps=config.get("eval_steps", 500) if eval_dataset else 0
    )
    
    # Create trainer
    trainer = SmallDatasetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        small_dataset_config=config
    )
    
    logger.info("Set up small dataset training with optimizations")
    return trainer, training_args