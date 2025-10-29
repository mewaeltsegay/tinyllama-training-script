"""
Small dataset optimization techniques for TinyLlama Tigrinya training.

This module implements various optimization techniques specifically designed for
training on small datasets, including data augmentation, regularization, and
learning rate scheduling.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Union
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SmallDatasetOptimizer:
    """
    Optimizer for small dataset training with data augmentation and regularization.
    
    Implements techniques including:
    - Token dropout augmentation
    - Sequence length variation
    - Repeated sampling with different random seeds
    - Learning rate scheduling with warmup and cosine annealing
    - Regularization techniques
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        token_dropout_prob: float = 0.1,
        length_variation_ratio: float = 0.3,
        augmentation_factor: int = 2,
        seed: Optional[int] = None
    ):
        """
        Initialize the small dataset optimizer.
        
        Args:
            tokenizer: Tokenizer for handling special tokens
            max_length: Maximum sequence length
            token_dropout_prob: Probability of dropping tokens for augmentation
            length_variation_ratio: Ratio of length variation (0.3 means 30% variation)
            augmentation_factor: Factor by which to augment the dataset
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_dropout_prob = token_dropout_prob
        self.length_variation_ratio = length_variation_ratio
        self.augmentation_factor = augmentation_factor
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        logger.info(f"Initialized SmallDatasetOptimizer with augmentation_factor={augmentation_factor}")
    
    def apply_token_dropout_augmentation(self, dataset: Dataset) -> Dataset:
        """
        Apply token dropout augmentation to the dataset.
        
        Args:
            dataset: Input dataset with tokenized examples
            
        Returns:
            Augmented dataset with token dropout applied
        """
        def apply_token_dropout(examples):
            """Apply token dropout for regularization."""
            augmented_examples = {"input_ids": [], "labels": []}
            
            # Get special token IDs
            unk_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
            special_token_ids = set()
            
            for token in [self.tokenizer.bos_token, self.tokenizer.eos_token, 
                         self.tokenizer.pad_token, self.tokenizer.unk_token]:
                if token:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None:
                        special_token_ids.add(token_id)
            
            for input_ids, labels in zip(examples["input_ids"], examples["labels"]):
                # Create a copy for dropout
                dropped_input_ids = input_ids.copy()
                
                # Apply dropout (replace with unk token)
                for i in range(len(dropped_input_ids)):
                    if random.random() < self.token_dropout_prob:
                        # Don't drop special tokens
                        if dropped_input_ids[i] not in special_token_ids:
                            dropped_input_ids[i] = unk_token_id
                
                augmented_examples["input_ids"].append(dropped_input_ids)
                augmented_examples["labels"].append(labels)  # Keep original labels
            
            return augmented_examples
        
        # Apply token dropout to create augmented data
        dropout_dataset = dataset.map(
            apply_token_dropout,
            batched=True,
            batch_size=1000,
            desc="Applying token dropout augmentation"
        )
        
        logger.info(f"Applied token dropout augmentation: {len(dataset)} -> {len(dropout_dataset)} examples")
        return dropout_dataset
    
    def apply_sequence_length_variation(self, dataset: Dataset) -> Dataset:
        """
        Apply sequence length variation through random truncation.
        
        Args:
            dataset: Input dataset with tokenized examples
            
        Returns:
            Dataset with varied sequence lengths
        """
        def apply_length_variation(examples):
            """Apply random sequence length variation."""
            varied_examples = {"input_ids": [], "labels": []}
            
            for input_ids, labels in zip(examples["input_ids"], examples["labels"]):
                # Calculate variation range
                original_length = len(input_ids)
                min_length = max(64, int(original_length * (1 - self.length_variation_ratio)))
                max_length = min(self.max_length, original_length)
                
                if max_length > min_length:
                    # Randomly choose a truncation point
                    random_length = random.randint(min_length, max_length)
                    truncated_input_ids = input_ids[:random_length]
                    truncated_labels = labels[:random_length]
                else:
                    truncated_input_ids = input_ids
                    truncated_labels = labels
                
                varied_examples["input_ids"].append(truncated_input_ids)
                varied_examples["labels"].append(truncated_labels)
            
            return varied_examples
        
        # Apply length variation
        length_varied_dataset = dataset.map(
            apply_length_variation,
            batched=True,
            batch_size=1000,
            desc="Applying sequence length variation"
        )
        
        logger.info(f"Applied sequence length variation: {len(dataset)} -> {len(length_varied_dataset)} examples")
        return length_varied_dataset
    
    def create_repeated_samples(self, dataset: Dataset, num_repeats: int = 2) -> Dataset:
        """
        Create repeated samples with different random seeds for small datasets.
        
        Args:
            dataset: Input dataset
            num_repeats: Number of times to repeat the dataset with different seeds
            
        Returns:
            Dataset with repeated samples
        """
        repeated_datasets = [dataset]
        
        for repeat_idx in range(1, num_repeats):
            # Set different seed for each repetition
            repeat_seed = (self.seed or 42) + repeat_idx * 1000
            random.seed(repeat_seed)
            torch.manual_seed(repeat_seed)
            
            # Shuffle the dataset with the new seed
            repeated_dataset = dataset.shuffle(seed=repeat_seed)
            repeated_datasets.append(repeated_dataset)
        
        # Concatenate all repeated datasets
        final_dataset = concatenate_datasets(repeated_datasets)
        
        logger.info(f"Created repeated samples: {len(dataset)} -> {len(final_dataset)} examples ({num_repeats} repeats)")
        return final_dataset
    
    def augment_dataset(self, dataset: Dataset) -> Dataset:
        """
        Apply all data augmentation techniques to the dataset.
        
        Args:
            dataset: Input training dataset
            
        Returns:
            Fully augmented dataset
        """
        original_size = len(dataset)
        augmented_datasets = [dataset]  # Start with original
        
        # Apply token dropout augmentation
        if self.augmentation_factor >= 2:
            dropout_dataset = self.apply_token_dropout_augmentation(dataset)
            augmented_datasets.append(dropout_dataset)
        
        # Apply sequence length variation
        if self.augmentation_factor >= 3:
            length_varied_dataset = self.apply_sequence_length_variation(dataset)
            augmented_datasets.append(length_varied_dataset)
        
        # Create repeated samples if needed
        if self.augmentation_factor >= 4:
            repeated_dataset = self.create_repeated_samples(dataset, num_repeats=2)
            augmented_datasets.append(repeated_dataset)
        
        # Combine all augmented datasets
        final_dataset = concatenate_datasets(augmented_datasets)
        
        # Shuffle the final combined dataset
        final_dataset = final_dataset.shuffle(seed=self.seed)
        
        logger.info(f"Applied all augmentation techniques: {original_size} -> {len(final_dataset)} examples")
        return final_dataset
    
    def create_warmup_cosine_scheduler(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_ratio: float = 0.1,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ) -> LambdaLR:
        """
        Create a learning rate scheduler with warmup and cosine annealing.
        
        Args:
            optimizer: The optimizer to schedule
            num_training_steps: Total number of training steps
            warmup_ratio: Ratio of warmup steps to total steps
            num_cycles: Number of cosine cycles (0.5 means half cycle)
            last_epoch: Last epoch for resuming training
            
        Returns:
            Learning rate scheduler
        """
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine annealing
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
        
        logger.info(f"Created warmup cosine scheduler: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
        return scheduler
    
    def apply_regularization_to_model(self, model: nn.Module, weight_decay: float = 0.01) -> None:
        """
        Apply regularization techniques to the model.
        
        Args:
            model: The model to apply regularization to
            weight_decay: Weight decay coefficient
        """
        # Apply weight decay to all parameters except biases and layer norms
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norm parameters
                if 'bias' not in name and 'LayerNorm' not in name and 'layer_norm' not in name:
                    if not hasattr(param, 'weight_decay'):
                        param.weight_decay = weight_decay
        
        logger.info(f"Applied weight decay regularization: {weight_decay}")
    
    def create_gradient_clipper(self, max_grad_norm: float = 1.0):
        """
        Create a gradient clipping function.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Function to clip gradients
        """
        def clip_gradients(model: nn.Module) -> float:
            """Clip gradients and return the gradient norm."""
            return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        logger.info(f"Created gradient clipper with max_norm={max_grad_norm}")
        return clip_gradients
    
    def get_optimization_config(
        self,
        num_training_steps: int,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get a complete optimization configuration for small datasets.
        
        Args:
            num_training_steps: Total number of training steps
            warmup_ratio: Ratio of warmup steps
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Dictionary with optimization configuration
        """
        config = {
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "num_training_steps": num_training_steps,
            "num_warmup_steps": int(num_training_steps * warmup_ratio),
            "scheduler_type": "warmup_cosine",
            "augmentation_factor": self.augmentation_factor,
            "token_dropout_prob": self.token_dropout_prob,
            "length_variation_ratio": self.length_variation_ratio
        }
        
        logger.info(f"Created optimization config for small dataset training: {config}")
        return config


class SmallDatasetTrainingMixin:
    """
    Mixin class to add small dataset optimization capabilities to training engines.
    """
    
    def setup_small_dataset_optimizations(
        self,
        optimizer: SmallDatasetOptimizer,
        model: nn.Module,
        train_dataset: Dataset,
        num_training_steps: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set up all small dataset optimizations.
        
        Args:
            optimizer: Small dataset optimizer instance
            model: The model to optimize
            train_dataset: Training dataset
            num_training_steps: Total training steps
            config: Training configuration
            
        Returns:
            Dictionary with optimization results and configurations
        """
        results = {}
        
        # Apply data augmentation
        augmented_dataset = optimizer.augment_dataset(train_dataset)
        results["augmented_dataset"] = augmented_dataset
        results["augmentation_ratio"] = len(augmented_dataset) / len(train_dataset)
        
        # Apply model regularization
        weight_decay = config.get("weight_decay", 0.01)
        optimizer.apply_regularization_to_model(model, weight_decay)
        results["weight_decay"] = weight_decay
        
        # Create gradient clipper
        max_grad_norm = config.get("max_grad_norm", 1.0)
        gradient_clipper = optimizer.create_gradient_clipper(max_grad_norm)
        results["gradient_clipper"] = gradient_clipper
        results["max_grad_norm"] = max_grad_norm
        
        # Get optimization configuration
        warmup_ratio = config.get("warmup_ratio", 0.1)
        opt_config = optimizer.get_optimization_config(
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm
        )
        results["optimization_config"] = opt_config
        
        logger.info(f"Set up small dataset optimizations: {len(train_dataset)} -> {len(augmented_dataset)} examples")
        return results


def create_small_dataset_optimizer(
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any]
) -> SmallDatasetOptimizer:
    """
    Factory function to create a SmallDatasetOptimizer from configuration.
    
    Args:
        tokenizer: Tokenizer instance
        config: Configuration dictionary
        
    Returns:
        Configured SmallDatasetOptimizer instance
    """
    return SmallDatasetOptimizer(
        tokenizer=tokenizer,
        max_length=config.get("max_length", 512),
        token_dropout_prob=config.get("token_dropout_prob", 0.1),
        length_variation_ratio=config.get("length_variation_ratio", 0.3),
        augmentation_factor=config.get("augmentation_factor", 2),
        seed=config.get("seed", None)
    )