#!/usr/bin/env python3
"""
TinyLlama Continuous Pretraining for Tigrinya Language

Main script for continuous pretraining of TinyLlama on Tigrinya language datasets.
Optimized for resource-constrained environments and small datasets with automatic
hardware detection and optimization.
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Any

# Fix console encoding for Windows
import os
if os.name == 'nt':  # Windows
    try:
        import codecs
        # Try to reconfigure stdout/stderr for UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        else:
            # Fallback for older Python versions
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach(), errors='replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach(), errors='replace')
    except Exception:
        # If encoding fix fails, continue without it
        pass

import torch
from transformers import TrainerCallback, Trainer, TrainingArguments

from config.training_config import TrainingConfig
from utils.training_stability import apply_stability_to_training_args
from utils.mixed_precision_manager import MixedPrecisionManager as StandaloneMixedPrecisionManager


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration with Unicode support."""
    import sys
    import io
    
    # Configure stdout to handle Unicode properly
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    # Create handlers with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    file_handler = logging.FileHandler('training.log', encoding='utf-8', errors='replace')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def parse_arguments() -> TrainingConfig:
    """Parse command-line arguments and return TrainingConfig."""
    parser = argparse.ArgumentParser(
        description="TinyLlama Continuous Pretraining for Tigrinya Language\n\n"
                   "This script performs continuous pretraining of TinyLlama on Tigrinya datasets "
                   "with automatic hardware optimization and small dataset techniques.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="For usage examples, run: python tinyllama_tigrinya_training.py --examples"
    )
    
    # Add examples argument
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples and exit"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="HuggingFace model name or path"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    
    # Hardware optimization
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    # I/O parameters
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory containing JSONL dataset files"
    )
    
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="tokenizer",
        help="Directory containing SentencePiece tokenizer files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for model checkpoints and logs"
    )
    
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    
    # Small dataset optimization
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from dataset (for testing/debugging)"
    )
    
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduling"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    # Data augmentation parameters
    parser.add_argument(
        "--token-dropout-prob",
        type=float,
        default=0.1,
        help="Probability of token dropout for data augmentation"
    )
    
    parser.add_argument(
        "--length-variation-ratio",
        type=float,
        default=0.3,
        help="Ratio of sequence length variation for augmentation"
    )
    
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=2,
        help="Factor by which to augment the dataset"
    )
    
    # Learning rate scheduling
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", 
                "constant", "constant_with_warmup", "inverse_sqrt", 
                "reduce_lr_on_plateau", "cosine_with_min_lr", 
                "cosine_warmup_with_min_lr", "warmup_stable_decay"],
        help="Type of learning rate scheduler"
    )
    
    parser.add_argument(
        "--num-cycles",
        type=float,
        default=0.5,
        help="Number of cosine cycles for scheduler"
    )
    
    # Inference parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate during inference"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for text generation"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Handle examples request
    if args.examples:
        print_usage_examples()
        sys.exit(0)
    
    # Validate arguments
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    
    if args.num_epochs <= 0:
        parser.error("Number of epochs must be positive")
    
    if args.max_length <= 0:
        parser.error("Max length must be positive")
    
    if args.checkpoint_steps <= 0:
        parser.error("Checkpoint steps must be positive")
    
    if not (0.0 <= args.warmup_ratio <= 1.0):
        parser.error("Warmup ratio must be between 0.0 and 1.0")
    
    if args.weight_decay < 0:
        parser.error("Weight decay must be non-negative")
    
    if args.max_grad_norm <= 0:
        parser.error("Max gradient norm must be positive")
    
    if args.temperature <= 0:
        parser.error("Temperature must be positive")
    
    if not (0.0 <= args.token_dropout_prob <= 1.0):
        parser.error("Token dropout probability must be between 0.0 and 1.0")
    
    if not (0.0 <= args.length_variation_ratio <= 1.0):
        parser.error("Length variation ratio must be between 0.0 and 1.0")
    
    if args.augmentation_factor < 1:
        parser.error("Augmentation factor must be at least 1")
    
    if args.num_cycles <= 0:
        parser.error("Number of cycles must be positive")
    
    # Validate directories exist
    dataset_path = Path(args.dataset_dir)
    tokenizer_path = Path(args.tokenizer_dir)
    
    if not dataset_path.exists():
        parser.error(f"Dataset directory does not exist: {args.dataset_dir}")
    
    if not tokenizer_path.exists():
        parser.error(f"Tokenizer directory does not exist: {args.tokenizer_dir}")
    
    # Create TrainingConfig from arguments
    config = TrainingConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=not args.no_mixed_precision,
        dataset_dir=args.dataset_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        checkpoint_steps=args.checkpoint_steps,
        max_samples=args.max_samples,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        token_dropout_prob=args.token_dropout_prob,
        length_variation_ratio=args.length_variation_ratio,
        augmentation_factor=args.augmentation_factor,
        scheduler_type=args.scheduler_type,
        num_cycles=args.num_cycles,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_level=args.log_level
    )
    
    return config


def main() -> None:
    """
    Main entry point for TinyLlama Tigrinya continuous pretraining.
    
    Orchestrates the complete training pipeline with integrated stability fixes:
    1. Configuration parsing and validation
    2. Hardware detection and stability configuration
    3. Dataset loading with fixed data collator
    4. Model management with gradient stabilization
    5. Training execution with tensorboard logging
    6. Inference validation and result reporting
    7. Final cleanup and summary
    """
    from utils.logging import setup_comprehensive_logging
    from utils.hardware import HardwareDetector
    from data.dataset_loader import TigrinyaDatasetLoader
    from model.model_manager import TinyLlamaManager
    from training.trainer import TrainingEngine
    from inference.generator import InferenceEngine
    
    # Import all stability fixes
    from utils.tensorboard_logger import create_tensorboard_logger
    from data.fixed_data_collator import create_fixed_data_collator, LossValidator
    from utils.gradient_stabilizer import create_gradient_stabilizer

    from utils.minimal_stability import create_minimal_stability_configurator
    
    structured_logger = None
    recovery_manager = None
    start_time = time.time()
    
    # Initialize component references for cleanup
    hardware_detector = None
    dataset_loader = None
    model_manager = None
    training_engine = None
    inference_engine = None
    tensorboard_logger = None
    gradient_stabilizer = None
    mixed_precision_manager = None
    stability_configurator = None
    
    # Set up basic logging first (before any potential errors)
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # ============================================================================
        # PHASE 1: CONFIGURATION AND VALIDATION
        # ============================================================================
        
        # Parse command-line arguments
        config = parse_arguments()
        
        # Update logging level based on config
        setup_logging(config.log_level)
        logger = logging.getLogger(__name__)
        
        # Create output directory for comprehensive logging
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup comprehensive logging and error recovery
        structured_logger, recovery_manager = setup_comprehensive_logging(
            output_dir=str(output_dir),
            log_level=config.log_level
        )
        
        logger.info("=" * 80)
        logger.info("TINYLLAMA TIGRINYA CONTINUOUS PRETRAINING WITH STABILITY FIXES")
        logger.info("=" * 80)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Dataset: {config.dataset_dir}")
        logger.info(f"Tokenizer: {config.tokenizer_dir}")
        logger.info(f"Output: {config.output_dir}")
        logger.info(f"Configuration: {config}")
        
        # Validate critical paths and files with error handling
        with structured_logger.error_context("initial_validation"):
            # Validate dataset directory
            dataset_path = Path(config.dataset_dir)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {config.dataset_dir}")
            
            # Validate tokenizer directory
            tokenizer_path = Path(config.tokenizer_dir)
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer directory does not exist: {config.tokenizer_dir}")
            
            # Check for required tokenizer files
            required_tokenizer_files = ["sentencepiece.model", "tokenizer_config.json"]
            missing_files = []
            for file_name in required_tokenizer_files:
                file_path = tokenizer_path / file_name
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                raise FileNotFoundError(f"Required tokenizer files not found: {missing_files}")
            
            # Check for dataset files
            dataset_files = list(dataset_path.glob("*.jsonl"))
            if not dataset_files:
                raise FileNotFoundError(f"No JSONL dataset files found in {config.dataset_dir}")
            
            logger.info(f"[OK] Validation completed: {len(dataset_files)} dataset files found")
        
        # ============================================================================
        # PHASE 2: HARDWARE DETECTION AND STABILITY CONFIGURATION
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 2: Hardware Detection and Stability Configuration")
        logger.info("-" * 50)
        
        with structured_logger.error_context("hardware_detection"):
            # Initialize hardware detector
            hardware_detector = HardwareDetector()
            hardware_detector.log_hardware_info()
            
            # Create minimal stability configurator (no complex recovery needed)
            stability_configurator = create_minimal_stability_configurator()
            stability_config = stability_configurator.detect_and_configure()
            
            logger.info(f"[OK] Stability configuration: {stability_config}")
            
            # Override config with stability-optimized settings if not explicitly set
            if config.batch_size is None:
                config.batch_size = stability_config["batch_size"]
                logger.info(f"[OK] Auto-configured batch size: {config.batch_size}")
            
            if config.gradient_accumulation_steps is None:
                config.gradient_accumulation_steps = stability_config["gradient_accumulation_steps"]
                logger.info(f"[OK] Auto-configured gradient accumulation steps: {config.gradient_accumulation_steps}")
            
            # Update learning rate and other stability parameters
            config.learning_rate = stability_config["learning_rate"]
            config.max_grad_norm = stability_config["max_grad_norm"]
            config.warmup_ratio = stability_config["warmup_ratio"]
            config.weight_decay = stability_config["weight_decay"]
            
            logger.info(f"[OK] Applied stability parameters - LR: {config.learning_rate}, Max Grad Norm: {config.max_grad_norm}")
        
        # ============================================================================
        # PHASE 3: INITIALIZE TENSORBOARD LOGGING (REPLACES WANDB)
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 3: Initialize TensorBoard Logging System")
        logger.info("-" * 50)
        
        with structured_logger.error_context("tensorboard_setup"):
            # Create tensorboard logger (replaces wandb)
            tensorboard_logger = create_tensorboard_logger(
                output_dir=str(output_dir),
                experiment_name="tinyllama_tigrinya_training"
            )
            
            logger.info(f"[OK] TensorBoard logger initialized")
            logger.info(f"[OK] TensorBoard logs: {tensorboard_logger.log_dir}")
            logger.info(f"[OK] View with: tensorboard --logdir {tensorboard_logger.log_dir}")
        
        # ============================================================================
        # PHASE 4: DATASET LOADING WITH FIXED DATA COLLATOR
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 4: Dataset Loading with Fixed Data Collator")
        logger.info("-" * 50)
        
        with structured_logger.error_context("dataset_loading"):
            # Initialize dataset loader with small dataset optimizations
            small_dataset_config = {
                "token_dropout_prob": config.token_dropout_prob,
                "length_variation_ratio": config.length_variation_ratio,
                "augmentation_factor": config.augmentation_factor
            }
            
            dataset_loader = TigrinyaDatasetLoader(
                dataset_dir=config.dataset_dir,
                tokenizer_dir=config.tokenizer_dir,
                max_length=config.max_length,
                streaming=False,  # Use non-streaming for small datasets
                seed=42,
                small_dataset_config=small_dataset_config,
                max_samples=config.max_samples
            )
            
            # Validate dataset format
            if not dataset_loader.validate_dataset_format():
                raise ValueError("Dataset format validation failed")
            
            # Load and preprocess datasets
            datasets = dataset_loader.load_datasets()
            logger.info(f"[OK] Loaded datasets: {list(datasets.keys())}")
            
            for split, dataset in datasets.items():
                logger.info(f"  {split}: {len(dataset)} examples")
        
        # ============================================================================
        # PHASE 5: MODEL LOADING WITH GRADIENT STABILIZATION
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 5: Model Loading with Gradient Stabilization")
        logger.info("-" * 50)
        
        with structured_logger.error_context("model_loading"):
            # Initialize model manager
            model_manager = TinyLlamaManager(
                model_name=config.model_name,
                tokenizer_dir=config.tokenizer_dir
            )
            
            # Load model and tokenizer
            model, tokenizer = model_manager.load_model_and_tokenizer()
            logger.info("[OK] Model and tokenizer loaded successfully")
            
            # Create fixed data collator (replaces problematic default collator)
            data_collator = create_fixed_data_collator(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
                max_length=config.max_length,
                conservative_labeling=True  # Use conservative labeling for stability
            )
            logger.info("[OK] Fixed data collator created (resolves zero loss issue)")
            
            # Initialize loss validator
            loss_validator = LossValidator(tokenizer)
            
            # Create gradient stabilizer system
            gradient_stabilizer, mixed_precision_manager = create_gradient_stabilizer(
                max_grad_norm=config.max_grad_norm,
                gpu_type="auto",
                enable_mixed_precision=config.mixed_precision
            )
            
            # Initialize model weights for stability
            gradient_stabilizer.initialize_model_weights(model)
            logger.info("[OK] Gradient stabilizer initialized and model weights initialized")
            
            # Setup mixed precision scaler
            scaler = mixed_precision_manager.setup_scaler(enabled=config.mixed_precision)
            if scaler:
                logger.info(f"[OK] Mixed precision scaler initialized with scale: {scaler.get_scale()}")
            else:
                logger.info("[OK] Mixed precision disabled or not available")
            
            # Log model information
            model_info = model_manager.get_model_info()
            logger.info(f"[OK] Model info: {model_info}")
        
        # ============================================================================
        # PHASE 6: TRAINING EXECUTION WITH INTEGRATED STABILITY FIXES
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 6: Training Execution with Integrated Stability Fixes")
        logger.info("-" * 50)
        
        with structured_logger.error_context("training_execution"):
            # Create custom training engine with all fixes integrated
            class StabilizedTrainingEngine(TrainingEngine):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.tensorboard_logger = tensorboard_logger
                    self.gradient_stabilizer = gradient_stabilizer
                    self.mixed_precision_manager = mixed_precision_manager
                    self.loss_validator = loss_validator
                    self.stability_configurator = stability_configurator
                
                def setup_training_arguments(self):
                    """Override to apply stability configuration."""
                    # Calculate total training steps
                    train_dataset = self.datasets.get("train")
                    if train_dataset is None:
                        raise ValueError("Training dataset not found")
                    
                    # For streaming datasets, we need to estimate steps
                    if hasattr(train_dataset, '__len__'):
                        estimated_samples = len(train_dataset)
                    else:
                        # Estimate based on typical dataset sizes
                        estimated_samples = 10000  # Conservative estimate
                        logger.warning(f"Using estimated dataset size of {estimated_samples} for streaming dataset")
                    
                    # Calculate steps per epoch
                    effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
                    steps_per_epoch = max(1, estimated_samples // effective_batch_size)
                    max_steps = steps_per_epoch * self.config.num_epochs
                    
                    # Create base training arguments with stability settings
                    
                    training_args = TrainingArguments(
                        # Output and logging
                        output_dir=str(self.output_dir),
                        logging_dir=str(self.output_dir / "tensorboard"),
                        logging_steps=10,
                        logging_strategy="steps",
                        
                        # Training parameters
                        max_steps=max_steps,
                        per_device_train_batch_size=self.config.batch_size,
                        gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                        learning_rate=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                        max_grad_norm=self.config.max_grad_norm,
                        
                        # Learning rate scheduling
                        lr_scheduler_type=self.config.scheduler_type,
                        warmup_ratio=self.config.warmup_ratio,
                        
                        # Mixed precision (will be overridden by mixed precision manager)
                        fp16=False,  # Will be set by mixed precision manager
                        bf16=False,  # Will be set by mixed precision manager
                        
                        # Checkpointing
                        save_strategy="steps",
                        save_steps=self.config.checkpoint_steps,
                        save_total_limit=3,
                        
                        # Evaluation
                        eval_strategy="no",  # Disable evaluation for stability
                        
                        # Data loading
                        dataloader_num_workers=0,  # Conservative for Windows
                        dataloader_pin_memory=True,
                        dataloader_drop_last=True,
                        dataloader_prefetch_factor=None,  # Must be None when num_workers=0
                        
                        # Reporting - TensorBoard only (no wandb)
                        report_to=["tensorboard"],
                        run_name=f"tinyllama_tigrinya_{int(time.time())}",
                        
                        # Stability settings
                        seed=42,
                        data_seed=42,
                        remove_unused_columns=False,
                        skip_memory_metrics=True,
                        dataloader_persistent_workers=False,
                    )
                    
                    # Apply stability configuration
                    training_args = self.stability_configurator.apply_to_training_arguments(training_args)
                    
                    # Apply mixed precision settings
                    standalone_mp_manager = StandaloneMixedPrecisionManager(gpu_type="auto")
                    training_args = standalone_mp_manager.update_training_arguments(training_args)
                    
                    # Ensure wandb is disabled
                    training_args.report_to = ["tensorboard"]  # Only tensorboard
                    
                    self.training_args = training_args
                    return training_args
                
                def create_trainer(self):
                    """Override to integrate all stability fixes."""
                    if self.training_args is None:
                        self.setup_training_arguments()
                    
                    # Prepare datasets
                    train_dataset = self.datasets["train"]
                    eval_dataset = self.datasets.get("validation")
                    
                    # Create custom callback with integrated fixes
                    class StabilizedTrainingCallback(TrainerCallback):
                        def __init__(self, training_engine):
                            self.training_engine = training_engine
                            self.step_start_time = None
                            self.tensorboard_logger = training_engine.tensorboard_logger
                            self.gradient_stabilizer = training_engine.gradient_stabilizer
                            self.loss_validator = training_engine.loss_validator
                        
                        def on_train_begin(self, args, state, control, **kwargs):
                            """Called at the beginning of training."""
                            self.training_engine.metrics.start_time = time.time()
                            self.training_engine.metrics.total_steps = state.max_steps
                            logger.info(f"Starting stabilized training for {state.max_steps} steps")
                        
                        def on_step_begin(self, args, state, control, **kwargs):
                            """Called at the beginning of each training step."""
                            self.step_start_time = time.time()
                        
                        def on_step_end(self, args, state, control, **kwargs):
                            """Enhanced step end with stability monitoring."""
                            if self.step_start_time is not None:
                                step_time = time.time() - self.step_start_time
                                
                                # Update metrics
                                self.training_engine.metrics.current_step = state.global_step
                                self.training_engine.metrics.current_epoch = state.epoch
                                
                                # Get model for gradient analysis
                                model = kwargs.get('model', self.training_engine.model)
                                
                                # Perform gradient stabilization
                                grad_norm, is_healthy = self.gradient_stabilizer.clip_gradients(model)
                                
                                if not is_healthy:
                                    logger.error(f"Unhealthy gradients detected at step {state.global_step}")
                                    # Let the trainer handle the error
                                    return
                                
                                # Log gradients to tensorboard
                                if state.global_step % 10 == 0:
                                    grad_stats = self.tensorboard_logger.log_gradients(model, state.global_step)
                                    
                                    # Log hardware metrics
                                    self.tensorboard_logger.log_hardware_metrics(state.global_step)
                                
                                # Calculate tokens per second
                                if step_time > 0:
                                    seq_length = self.training_engine.config.max_length
                                    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                                    tokens_processed = batch_size * seq_length
                                    self.training_engine.metrics.tokens_per_second = tokens_processed / step_time
                                
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
                                    
                                    # Validate loss value
                                    if current_loss == 0.0 and state.global_step > 10:
                                        logger.error(f"CRITICAL: Zero loss detected at step {state.global_step}")
                                        # This indicates data collator issues
                                    
                                    logger.info(
                                        f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | "
                                        f"Loss: {current_loss:.4f} | "
                                        f"LR: {current_lr:.2e} | "
                                        f"Grad Norm: {grad_norm:.4f} | "
                                        f"Tokens/s: {self.training_engine.metrics.tokens_per_second:.0f} | "
                                        f"ETA: {eta/60:.1f}min"
                                    )
                        
                        def on_log(self, args, state, control, logs=None, **kwargs):
                            """Enhanced logging with tensorboard integration."""
                            if logs:
                                # Update metrics from logs
                                if 'loss' in logs:
                                    self.training_engine.metrics.train_loss = logs['loss']
                                elif 'train_loss' in logs:
                                    self.training_engine.metrics.train_loss = logs['train_loss']
                                    
                                if 'learning_rate' in logs:
                                    self.training_engine.metrics.learning_rate = logs['learning_rate']
                                    
                                if 'grad_norm' in logs:
                                    self.training_engine.metrics.grad_norm = logs['grad_norm']
                                
                                # Log to tensorboard (replaces wandb)
                                tb_metrics = {}
                                for key, value in logs.items():
                                    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf'))):
                                        tb_metrics[key] = value
                                
                                if tb_metrics:
                                    self.tensorboard_logger.log_metrics(tb_metrics, state.global_step)
                                
                                # Save metrics to file
                                self.training_engine._save_training_metrics(logs)
                        
                        def on_save(self, args, state, control, **kwargs):
                            """Called when a checkpoint is saved."""
                            self.training_engine.metrics.last_checkpoint_step = state.global_step
                            logger.info(f"Checkpoint saved at step {state.global_step}")
                        
                        def on_train_end(self, args, state, control, **kwargs):
                            """Called at the end of training."""
                            # Log final hyperparameters and metrics
                            hparams = {
                                "learning_rate": args.learning_rate,
                                "batch_size": args.per_device_train_batch_size,
                                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                "max_grad_norm": args.max_grad_norm,
                                "warmup_steps": args.warmup_steps,
                                "fp16": args.fp16,
                                "bf16": args.bf16
                            }
                            
                            final_metrics = {
                                "final_loss": self.training_engine.metrics.train_loss,
                                "total_steps": state.global_step
                            }
                            
                            self.tensorboard_logger.log_hyperparameters(hparams, final_metrics)
                            logger.info("Training completed - TensorBoard logging finalized")
                    
                    # Create custom callback
                    progress_callback = StabilizedTrainingCallback(self)
                    
                    # Create trainer with all fixes
                    trainer = Trainer(
                        model=model,
                        args=self.training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer=tokenizer,
                        data_collator=data_collator,  # Use fixed data collator
                        callbacks=[progress_callback],
                    )
                    
                    self.trainer = trainer
                    logger.info("[OK] Stabilized trainer created with all fixes integrated")
                    
                    return trainer
            
            # Initialize stabilized training engine
            training_engine = StabilizedTrainingEngine(
                config=config,
                model=model,
                tokenizer=tokenizer,
                datasets=datasets,
                data_collator=data_collator,
                hardware_config=stability_config
            )
            
            # Setup training arguments and create trainer
            training_args = training_engine.setup_training_arguments()
            trainer = training_engine.create_trainer()
            
            logger.info("[OK] Stabilized training engine initialized")
            logger.info(f"[OK] Training configuration: {training_engine.get_training_status()}")
            
            # Log initial hyperparameters to tensorboard
            hparams = {
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "max_grad_norm": training_args.max_grad_norm,
                "warmup_ratio": training_args.warmup_ratio,
                "weight_decay": training_args.weight_decay,
                "fp16": training_args.fp16,
                "bf16": training_args.bf16,
                "model_name": config.model_name,
                "max_length": config.max_length
            }
            tensorboard_logger.log_hyperparameters(hparams)
            
            # Execute training with automatic checkpoint resumption
            logger.info("Starting stabilized training...")
            final_state = training_engine.train(resume_from_checkpoint="auto")
            
            logger.info("[OK] Training completed successfully")
            logger.info(f"[OK] Final training state: {training_engine.get_training_status()}")
        
        # ============================================================================
        # PHASE 7: INFERENCE VALIDATION AND RESULT REPORTING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 7: Inference Validation and Result Reporting")
        logger.info("-" * 50)
        
        with structured_logger.error_context("inference_validation"):
            # Initialize inference engine with trained model
            inference_engine = InferenceEngine(model=model, tokenizer=tokenizer)
            
            # Run validation with predefined Tigrinya prompts
            validation_output_file = output_dir / "validation_results.json"
            
            validation_results = inference_engine.validate_training(
                prompts=None,  # Use default Tigrinya prompts
                output_file=str(validation_output_file),
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                samples_per_prompt=3
            )
            
            logger.info("[OK] Inference validation completed")
            logger.info(f"[OK] Validation results saved to: {validation_output_file}")
            
            # Evaluate generation quality
            quality_metrics = inference_engine.evaluate_generation_quality(validation_results)
            logger.info(f"[OK] Generation quality metrics: {quality_metrics}")
        
        # ============================================================================
        # PHASE 8: VISUALIZATION AND PLOTTING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 8: Creating Training Visualizations")
        logger.info("-" * 50)
        
        try:
            from utils.plotting import TrainingPlotter, create_plots_from_training_output
            
            # Create comprehensive plots and dashboard
            logger.info("Creating training plots and dashboard...")
            created_plots = create_plots_from_training_output(str(output_dir))
            
            logger.info(f"[OK] Created {len(created_plots)} visualization files:")
            for plot_path in created_plots:
                logger.info(f"  - {plot_path}")
                
        except ImportError as e:
            logger.warning(f"Plotting dependencies not available: {e}")
            logger.warning("Install matplotlib and seaborn for visualization: pip install matplotlib seaborn")
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")
        
        # ============================================================================
        # PHASE 9: FINAL CLEANUP AND SUMMARY REPORTING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 9: Final Cleanup and Summary Reporting")
        logger.info("-" * 50)
        
        # Get stability report
        stability_report = stability_configurator.get_stability_report()
        gradient_report = gradient_stabilizer.get_stability_report()
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Prepare final summary with stability information
        final_summary = {
            "execution_time": total_time,
            "training_completed": True,
            "validation_completed": True,
            "model_info": model_info,
            "stability_config": stability_config,
            "training_config": config.to_dict(),
            "final_training_metrics": training_engine.get_training_status(),
            "validation_metrics": quality_metrics,
            "stability_report": stability_report,
            "gradient_stability_report": gradient_report,
            "fixes_applied": {
                "tensorboard_logging": True,
                "fixed_data_collator": True,
                "gradient_stabilization": True,
                "mixed_precision_optimization": True,
                "hardware_specific_configuration": True,
                "wandb_removed": True
            },
            "output_files": {
                "model_checkpoint": str(output_dir),
                "validation_results": str(validation_output_file),
                "training_logs": str(output_dir / "logs"),
                "tensorboard_logs": str(tensorboard_logger.log_dir),
                "training_config": str(output_dir / "training_config.json"),
                "final_metrics": str(output_dir / "final_metrics.json")
            }
        }
        
        # Save final summary
        summary_file = output_dir / "execution_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        # Cleanup resources
        if training_engine:
            training_engine.cleanup()
        
        if tensorboard_logger:
            tensorboard_logger.close()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log final success message
        logger.info("=" * 80)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY WITH ALL STABILITY FIXES")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_time/60:.2f} minutes")
        logger.info(f"Model trained and saved to: {output_dir}")
        logger.info(f"Validation results: {validation_output_file}")
        logger.info(f"TensorBoard logs: {tensorboard_logger.log_dir}")
        logger.info(f"Execution summary: {summary_file}")
        
        # Log stability summary
        logger.info("STABILITY FIXES APPLIED:")
        logger.info("  ✓ Removed wandb dependencies, replaced with TensorBoard")
        logger.info("  ✓ Fixed data collator to resolve zero loss issue")
        logger.info("  ✓ Implemented gradient stabilization and NaN recovery")
        logger.info("  ✓ Applied hardware-specific mixed precision settings")
        logger.info("  ✓ Configured conservative training parameters for stability")
        
        # Log error summary if any errors were handled
        if structured_logger:
            error_summary = structured_logger.get_error_summary()
            if error_summary["total_errors"] > 0:
                logger.info(f"Execution completed with {error_summary['total_errors']} handled errors")
                logger.info(f"Error summary: {error_summary}")
            else:
                logger.info("Execution completed with no errors")
        
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        _cleanup_resources(training_engine)
        sys.exit(0)
        
    except FileNotFoundError as e:
        error_msg = f"Required file or directory not found: {e}"
        logger.error(error_msg)
        if structured_logger:
            structured_logger.log_error(e, {"error_type": "file_validation"})
        _cleanup_resources(training_engine)
        sys.exit(1)
        
    except PermissionError as e:
        error_msg = f"Permission denied: {e}"
        logger.error(error_msg)
        if structured_logger:
            structured_logger.log_error(e, {"error_type": "permission_error"})
        _cleanup_resources(training_engine)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Training failed with unexpected error: {e}"
        logger.error(error_msg)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        if structured_logger:
            structured_logger.log_error(e, {
                "error_type": "unexpected_error",
                "phase": "main_execution"
            })
            
            # Log final error summary
            error_summary = structured_logger.get_error_summary()
            logger.error(f"Final error summary: {error_summary}")
        
        _cleanup_resources(training_engine)
        sys.exit(1)


def _cleanup_resources(training_engine: Optional[Any]) -> None:
    """
    Cleanup training resources in case of errors or interruption.
    
    Args:
        training_engine: Training engine instance to cleanup (optional)
    """
    try:
        if training_engine:
            training_engine.cleanup()
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch not available
        
        logger = logging.getLogger(__name__)
        logger.info("Resources cleaned up successfully")
        
    except Exception as e:
        # Don't raise exceptions during cleanup
        print(f"Warning: Error during cleanup: {e}")


def print_usage_examples() -> None:
    """Print usage examples and help information."""
    examples = """
USAGE EXAMPLES:

1. Basic training with default settings:
   python tinyllama_tigrinya_training.py

2. Custom learning rate and epochs:
   python tinyllama_tigrinya_training.py --learning-rate 1e-4 --num-epochs 5

3. Specify custom directories:
   python tinyllama_tigrinya_training.py \\
       --dataset-dir "my_dataset" \\
       --tokenizer-dir "my_tokenizer" \\
       --output-dir "my_output"

4. Hardware-specific optimization (manual override):
   python tinyllama_tigrinya_training.py \\
       --batch-size 2 \\
       --gradient-accumulation-steps 4 \\
       --no-mixed-precision

5. Small dataset optimization:
   python tinyllama_tigrinya_training.py \\
       --token-dropout-prob 0.15 \\
       --augmentation-factor 3 \\
       --warmup-ratio 0.2

6. Custom inference parameters:
   python tinyllama_tigrinya_training.py \\
       --max-new-tokens 150 \\
       --temperature 0.9

7. Debug mode with verbose logging:
   python tinyllama_tigrinya_training.py --log-level DEBUG

DIRECTORY STRUCTURE:
Your workspace should be organized as follows:

workspace/
├── dataset/
│   ├── train.jsonl          # Training data (required)
│   ├── validation.jsonl     # Validation data (optional)
│   └── test.jsonl          # Test data (optional)
├── tokenizer/
│   ├── sentencepiece.model  # SentencePiece model (required)
│   ├── sentencepiece.vocab  # SentencePiece vocabulary (required)
│   └── tokenizer_config.json # Tokenizer configuration (required)
└── output/                  # Output directory (created automatically)
    ├── checkpoint-*/        # Model checkpoints
    ├── logs/               # Training logs
    ├── validation_results.json
    └── execution_summary.json

DATASET FORMAT:
Each line in JSONL files should contain:
{"text": "ሰላም! ከመይ ኣሎኻ? ሎምስ እንታይ ገይርካ?"}

HARDWARE SUPPORT:
- Automatic GPU detection and optimization
- Supported GPUs: RTX 4050, L4, A100, H100
- Automatic fallback to CPU if no GPU available
- Memory-optimized configurations for each GPU type

For more information, see the README.md file.
"""
    print(examples)


def create_example_usage_documentation() -> None:
    """Create example usage documentation file."""
    try:
        readme_content = """# TinyLlama Tigrinya Continuous Pretraining

This script performs continuous pretraining of TinyLlama on Tigrinya language datasets with automatic hardware optimization and small dataset techniques.

## Quick Start

```bash
# Basic training with default settings
python tinyllama_tigrinya_training.py

# Show usage examples
python tinyllama_tigrinya_training.py --examples

# Custom configuration
python tinyllama_tigrinya_training.py --learning-rate 1e-4 --num-epochs 5 --batch-size 2
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-compatible GPU (optional, CPU fallback available)

## Directory Structure

```
workspace/
├── dataset/
│   ├── train.jsonl          # Training data (required)
│   ├── validation.jsonl     # Validation data (optional)
│   └── test.jsonl          # Test data (optional)
├── tokenizer/
│   ├── sentencepiece.model  # SentencePiece model (required)
│   ├── sentencepiece.vocab  # SentencePiece vocabulary (required)
│   └── tokenizer_config.json # Tokenizer configuration (required)
└── output/                  # Output directory (created automatically)
    ├── checkpoint-*/        # Model checkpoints
    ├── logs/               # Training logs
    ├── validation_results.json
    └── execution_summary.json
```

## Dataset Format

Each line in JSONL files should contain:
```json
{"text": "ሰላም! ከመይ ኣሎኻ? ሎምስ እንታይ ገይርካ?"}
```

## Hardware Support

- Automatic GPU detection and optimization
- Supported GPUs: RTX 4050, L4, A100, H100
- Automatic fallback to CPU if no GPU available
- Memory-optimized configurations for each GPU type

## Key Features

- **Automatic Hardware Optimization**: Detects GPU type and optimizes batch sizes, memory usage
- **Small Dataset Techniques**: Data augmentation, regularization, learning rate scheduling
- **Comprehensive Logging**: Structured logging with error recovery
- **Checkpoint Management**: Automatic saving and resumption
- **Inference Validation**: Automatic text generation validation after training
- **Error Recovery**: Automatic recovery from common training errors (OOM, etc.)

## Command Line Options

Run `python tinyllama_tigrinya_training.py --help` for full list of options.

## Output Files

After successful training, you'll find:
- `output/checkpoint-*/`: Model checkpoints
- `output/validation_results.json`: Generated text samples
- `output/execution_summary.json`: Complete execution summary
- `output/logs/`: Detailed training logs
"""
        
        readme_path = Path("USAGE.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"Usage documentation created: {readme_path}")
        
    except Exception as e:
        print(f"Warning: Could not create usage documentation: {e}")


if __name__ == "__main__":
    main()