#!/usr/bin/env python3
"""
Test script for small dataset optimization techniques.

This script tests the SmallDatasetOptimizer functionality to ensure
all optimization techniques are working correctly.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.training_config import TrainingConfig
from data.dataset_loader import TigrinyaDatasetLoader
from utils.small_dataset_optimizer import SmallDatasetOptimizer, create_small_dataset_optimizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_small_dataset_optimizer():
    """Test the SmallDatasetOptimizer functionality."""
    try:
        logger.info("Testing SmallDatasetOptimizer...")
        
        # Create a test configuration
        config = TrainingConfig(
            dataset_dir="dataset",
            tokenizer_dir="tokenizer",
            max_length=512,
            token_dropout_prob=0.1,
            length_variation_ratio=0.3,
            augmentation_factor=2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            scheduler_type="warmup_cosine",
            num_cycles=0.5
        )
        
        logger.info(f"Configuration: {config}")
        
        # Test dataset loader with small dataset optimizations
        logger.info("Testing dataset loader with small dataset optimizations...")
        
        dataset_loader = TigrinyaDatasetLoader(
            dataset_dir=config.dataset_dir,
            tokenizer_dir=config.tokenizer_dir,
            max_length=config.max_length,
            streaming=False,
            seed=42,
            small_dataset_config={
                "token_dropout_prob": config.token_dropout_prob,
                "length_variation_ratio": config.length_variation_ratio,
                "augmentation_factor": config.augmentation_factor
            }
        )
        
        # Validate dataset format
        if not dataset_loader.validate_dataset_format():
            logger.error("Dataset format validation failed")
            return False
        
        # Load datasets
        datasets = dataset_loader.load_datasets()
        logger.info(f"Loaded datasets: {list(datasets.keys())}")
        
        for split, dataset in datasets.items():
            logger.info(f"{split} dataset size: {len(dataset)}")
        
        # Test small dataset optimizer directly
        logger.info("Testing SmallDatasetOptimizer directly...")
        
        optimizer = create_small_dataset_optimizer(
            tokenizer=dataset_loader.tokenizer,
            config=config.to_dict()
        )
        
        # Test optimization configuration
        opt_config = optimizer.get_optimization_config(
            num_training_steps=1000,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm
        )
        
        logger.info(f"Optimization configuration: {opt_config}")
        
        # Test data collator
        logger.info("Testing data collator...")
        data_collator = dataset_loader.create_data_collator()
        
        # Test with a small batch
        if "train" in datasets:
            train_dataset = datasets["train"]
            test_batch = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
            collated_batch = data_collator(test_batch)
            
            logger.info(f"Test batch shapes:")
            for key, value in collated_batch.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key}: {value.shape}")
        
        logger.info("✓ All small dataset optimization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("Starting small dataset optimization tests...")
    
    # Check if required directories exist
    if not Path("dataset").exists():
        logger.error("Dataset directory not found. Please ensure dataset/ directory exists.")
        return False
    
    if not Path("tokenizer").exists():
        logger.error("Tokenizer directory not found. Please ensure tokenizer/ directory exists.")
        return False
    
    # Run tests
    success = test_small_dataset_optimizer()
    
    if success:
        logger.info("All tests passed successfully!")
        return True
    else:
        logger.error("Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)