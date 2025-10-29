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

import torch

from config.training_config import TrainingConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


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
    
    Orchestrates the complete training pipeline:
    1. Configuration parsing and validation
    2. Hardware detection and GPU optimization
    3. Dataset loading and preprocessing
    4. Model management and tokenizer integration
    5. Training execution with monitoring
    6. Inference validation and result reporting
    7. Final cleanup and summary
    """
    from utils.logging import setup_comprehensive_logging
    from utils.hardware import HardwareDetector
    from data.dataset_loader import TigrinyaDatasetLoader
    from model.model_manager import TinyLlamaManager
    from training.trainer import TrainingEngine
    from inference.generator import InferenceEngine
    
    structured_logger = None
    recovery_manager = None
    start_time = time.time()
    
    # Initialize component references for cleanup
    hardware_detector = None
    dataset_loader = None
    model_manager = None
    training_engine = None
    inference_engine = None
    
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
        logger.info("TINYLLAMA TIGRINYA CONTINUOUS PRETRAINING")
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
            
            logger.info(f"✓ Validation completed: {len(dataset_files)} dataset files found")
        
        # ============================================================================
        # PHASE 2: HARDWARE DETECTION AND GPU OPTIMIZATION
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 2: Hardware Detection and GPU Optimization")
        logger.info("-" * 50)
        
        with structured_logger.error_context("hardware_detection"):
            hardware_detector = HardwareDetector()
            hardware_detector.log_hardware_info()
            
            # Detect GPU configuration
            hardware_config = hardware_detector.detect_gpu_config()
            logger.info(f"✓ Hardware configuration: {hardware_config}")
            
            # Override config with hardware-optimized settings if not explicitly set
            if config.batch_size is None:
                config.batch_size = hardware_config["batch_size"]
                logger.info(f"✓ Auto-configured batch size: {config.batch_size}")
            
            if config.gradient_accumulation_steps is None:
                config.gradient_accumulation_steps = hardware_config["gradient_accumulation_steps"]
                logger.info(f"✓ Auto-configured gradient accumulation steps: {config.gradient_accumulation_steps}")
        
        # ============================================================================
        # PHASE 3: DATASET LOADING AND PREPROCESSING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 3: Dataset Loading and Preprocessing")
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
                small_dataset_config=small_dataset_config
            )
            
            # Validate dataset format
            if not dataset_loader.validate_dataset_format():
                raise ValueError("Dataset format validation failed")
            
            # Load and preprocess datasets
            datasets = dataset_loader.load_datasets()
            logger.info(f"✓ Loaded datasets: {list(datasets.keys())}")
            
            for split, dataset in datasets.items():
                logger.info(f"  {split}: {len(dataset)} examples")
            
            # Create data collator
            data_collator = dataset_loader.create_data_collator(mlm=False)
            logger.info("✓ Data collator created for causal language modeling")
        
        # ============================================================================
        # PHASE 4: MODEL MANAGEMENT AND TOKENIZER INTEGRATION
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 4: Model Management and Tokenizer Integration")
        logger.info("-" * 50)
        
        with structured_logger.error_context("model_loading"):
            # Initialize model manager
            model_manager = TinyLlamaManager(
                model_name=config.model_name,
                tokenizer_dir=config.tokenizer_dir
            )
            
            # Load model and tokenizer
            model, tokenizer = model_manager.load_model_and_tokenizer()
            logger.info("✓ Model and tokenizer loaded successfully")
            
            # Log model information
            model_info = model_manager.get_model_info()
            logger.info(f"✓ Model info: {model_info}")
        
        # ============================================================================
        # PHASE 5: TRAINING EXECUTION WITH MONITORING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 5: Training Execution with Monitoring")
        logger.info("-" * 50)
        
        with structured_logger.error_context("training_execution"):
            # Initialize training engine
            training_engine = TrainingEngine(
                config=config,
                model=model,
                tokenizer=tokenizer,
                datasets=datasets,
                data_collator=data_collator,
                hardware_config=hardware_config
            )
            
            # Setup training arguments and create trainer
            training_args = training_engine.setup_training_arguments()
            trainer = training_engine.create_trainer()
            
            logger.info("✓ Training engine initialized")
            logger.info(f"✓ Training configuration: {training_engine.get_training_status()}")
            
            # Execute training with automatic checkpoint resumption
            logger.info("Starting training...")
            final_state = training_engine.train(resume_from_checkpoint="auto")
            
            logger.info("✓ Training completed successfully")
            logger.info(f"✓ Final training state: {training_engine.get_training_status()}")
        
        # ============================================================================
        # PHASE 6: INFERENCE VALIDATION AND RESULT REPORTING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 6: Inference Validation and Result Reporting")
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
            
            logger.info("✓ Inference validation completed")
            logger.info(f"✓ Validation results saved to: {validation_output_file}")
            
            # Evaluate generation quality
            quality_metrics = inference_engine.evaluate_generation_quality(validation_results)
            logger.info(f"✓ Generation quality metrics: {quality_metrics}")
        
        # ============================================================================
        # PHASE 7: FINAL CLEANUP AND SUMMARY REPORTING
        # ============================================================================
        
        logger.info("-" * 50)
        logger.info("PHASE 7: Final Cleanup and Summary Reporting")
        logger.info("-" * 50)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Prepare final summary
        final_summary = {
            "execution_time": total_time,
            "training_completed": True,
            "validation_completed": True,
            "model_info": model_info,
            "hardware_config": hardware_config,
            "training_config": config.to_dict(),
            "final_training_metrics": training_engine.get_training_status(),
            "validation_metrics": quality_metrics,
            "output_files": {
                "model_checkpoint": str(output_dir),
                "validation_results": str(validation_output_file),
                "training_logs": str(output_dir / "logs"),
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
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log final success message
        logger.info("=" * 80)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {total_time/60:.2f} minutes")
        logger.info(f"Model trained and saved to: {output_dir}")
        logger.info(f"Validation results: {validation_output_file}")
        logger.info(f"Execution summary: {summary_file}")
        
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