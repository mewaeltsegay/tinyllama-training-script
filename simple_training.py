#!/usr/bin/env python3
"""
Simple TinyLlama Training Script

Back-to-basics approach that focuses on stability over complexity.
No complex recovery mechanisms, just proven stable training techniques.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simple_training.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, AutoModelForCausalLM
from data.dataset_loader import TigrinyaDatasetLoader
from utils.simple_trainer import create_simple_trainer


def main():
    parser = argparse.ArgumentParser(description="Simple TinyLlama Training")
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="output_simple")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000, help="Limit dataset size for testing")
    
    args = parser.parse_args()
    
    logger.info("=== Simple TinyLlama Training ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Max samples: {args.max_samples}")
    
    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {device_name}")
    else:
        logger.info("CUDA not available, using CPU")
    
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
        logger.info("Loading dataset...")
        dataset_loader = TigrinyaDatasetLoader(
            dataset_dir=args.dataset_dir,
            tokenizer=tokenizer,
            max_length=512,  # Conservative sequence length
            cache_dir="./cache"
        )
        
        train_dataset = dataset_loader.load_train_dataset()
        
        # Limit dataset size for testing
        if args.max_samples and len(train_dataset) > args.max_samples:
            logger.info(f"Limiting dataset from {len(train_dataset)} to {args.max_samples} samples")
            train_dataset = train_dataset.select(range(args.max_samples))
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create simple trainer
        logger.info("Creating trainer...")
        trainer = create_simple_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs
        )
        
        # Validate setup
        logger.info("Validating setup...")
        if not trainer.validate_setup():
            logger.error("Setup validation failed!")
            return 1
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())