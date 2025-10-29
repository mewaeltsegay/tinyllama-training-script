"""
Training configuration dataclass for TinyLlama Tigrinya continuous pretraining.

This module defines the TrainingConfig dataclass that holds all training parameters
and their default values, supporting command-line configuration and validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration class for TinyLlama Tigrinya training parameters."""
    
    # Model parameters
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    
    # Training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 512
    
    # Hardware optimization parameters (auto-detected if None)
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    mixed_precision: bool = True
    
    # I/O parameters
    dataset_dir: str = "dataset"
    tokenizer_dir: str = "tokenizer"
    output_dir: str = "output"
    checkpoint_steps: int = 500
    
    # Small dataset optimization parameters
    max_samples: Optional[int] = None  # Limit dataset size for testing
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data augmentation parameters
    token_dropout_prob: float = 0.1
    length_variation_ratio: float = 0.3
    augmentation_factor: int = 2
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"
    num_cycles: float = 0.5
    
    # Inference parameters
    inference_prompts: Optional[List[str]] = field(default_factory=lambda: [
        "ሰላም! ከመይ ኣሎኻ?",  # Hello! How are you?
        "ሎሚ ጽቡቕ መዓልቲ እዩ።",  # Today is a good day.
        "ትግርኛ ቋንቋ ጽቡቕ እዩ።"  # Tigrinya language is good.
    ])
    max_new_tokens: int = 100
    temperature: float = 0.8
    
    # Logging parameters
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
        self._ensure_directories()
    
    def _validate_parameters(self) -> None:
        """Validate all configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.max_length <= 0:
            raise ValueError("Max length must be positive")
        
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        
        if self.checkpoint_steps <= 0:
            raise ValueError("Checkpoint steps must be positive")
        
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("Warmup ratio must be between 0.0 and 1.0")
        
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        
        if self.max_new_tokens <= 0:
            raise ValueError("Max new tokens must be positive")
        
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("Log level must be one of: DEBUG, INFO, WARNING, ERROR")
        
        if not (0.0 <= self.token_dropout_prob <= 1.0):
            raise ValueError("Token dropout probability must be between 0.0 and 1.0")
        
        if not (0.0 <= self.length_variation_ratio <= 1.0):
            raise ValueError("Length variation ratio must be between 0.0 and 1.0")
        
        if self.augmentation_factor < 1:
            raise ValueError("Augmentation factor must be at least 1")
        
        # Valid scheduler types based on transformers library
        valid_schedulers = [
            "linear", "cosine", "cosine_with_restarts", "polynomial", 
            "constant", "constant_with_warmup", "inverse_sqrt", 
            "reduce_lr_on_plateau", "cosine_with_min_lr", "cosine_warmup_with_min_lr", 
            "warmup_stable_decay"
        ]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"Scheduler type must be one of: {valid_schedulers}")
        
        if self.num_cycles <= 0:
            raise ValueError("Number of cycles must be positive")
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        dataset_path = Path(self.dataset_dir)
        tokenizer_path = Path(self.tokenizer_dir)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
        
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {self.tokenizer_dir}")
        
        # Create output directory if it doesn't exist
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size considering gradient accumulation."""
        batch_size = self.batch_size or 1
        grad_accum = self.gradient_accumulation_steps or 1
        return batch_size * grad_accum
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "dataset_dir": self.dataset_dir,
            "tokenizer_dir": self.tokenizer_dir,
            "output_dir": self.output_dir,
            "checkpoint_steps": self.checkpoint_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "token_dropout_prob": self.token_dropout_prob,
            "length_variation_ratio": self.length_variation_ratio,
            "augmentation_factor": self.augmentation_factor,
            "scheduler_type": self.scheduler_type,
            "num_cycles": self.num_cycles,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "log_level": self.log_level,
            "effective_batch_size": self.get_effective_batch_size()
        }
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        config_dict = self.to_dict()
        return "\n".join([f"{key}: {value}" for key, value in config_dict.items()])
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            self._validate_parameters()
            return True
        except ValueError:
            return False
    
    def validate_directories(self) -> bool:
        """
        Validate that required directories exist.
        
        Returns:
            True if directories exist, False otherwise
        """
        try:
            self._ensure_directories()
            return True
        except (FileNotFoundError, PermissionError):
            return False
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            TrainingConfig instance
        """
        # Filter out keys that are not valid parameters
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


def parse_arguments(args=None):
    """
    Parse command-line arguments for training configuration.
    
    Args:
        args: List of arguments to parse (defaults to sys.argv)
        
    Returns:
        Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyLlama Tigrinya Training")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, 
                       default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                       help="Model name or path")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (auto-detected if not specified)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="Gradient accumulation steps (auto-detected if not specified)")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Enable mixed precision training")
    
    # I/O parameters
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                       help="Dataset directory")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer",
                       help="Tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--checkpoint_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    
    # Small dataset optimization parameters
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--token_dropout_prob", type=float, default=0.1,
                       help="Token dropout probability")
    parser.add_argument("--length_variation_ratio", type=float, default=0.3,
                       help="Length variation ratio")
    parser.add_argument("--augmentation_factor", type=int, default=2,
                       help="Data augmentation factor")
    
    # Learning rate scheduling
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", 
                               "constant", "constant_with_warmup", "inverse_sqrt", 
                               "reduce_lr_on_plateau", "cosine_with_min_lr", 
                               "cosine_warmup_with_min_lr", "warmup_stable_decay"],
                       help="Learning rate scheduler type")
    parser.add_argument("--num_cycles", type=float, default=0.5,
                       help="Number of cycles for cosine scheduler")
    
    # Inference parameters
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum new tokens for inference")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for text generation")
    parser.add_argument("--inference_prompts", nargs="*", 
                       default=["ሰላም! ከመይ ኣሎኻ?", "ሎሚ ጽቡቕ መዓልቲ እዩ።", "ትግርኛ ቋንቋ ጽቡቕ እዩ።"],
                       help="Inference prompts")
    
    # Logging parameters
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args(args)