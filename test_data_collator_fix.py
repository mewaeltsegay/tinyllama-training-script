#!/usr/bin/env python3
"""
Test script to validate the data collator fix for H100 sequence length issues.
"""

import logging
from data.dataset_loader import TigrinyaDatasetLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_collator():
    """Test the data collator with variable-length sequences."""
    try:
        logger.info("Testing data collator fix...")
        
        # Initialize dataset loader
        dataset_loader = TigrinyaDatasetLoader(
            dataset_dir="dataset",
            tokenizer_dir="tokenizer",
            max_length=512,
            streaming=False,
            max_samples=10  # Small sample for testing
        )
        
        # Load datasets
        datasets = dataset_loader.load_datasets()
        logger.info(f"Loaded datasets: {list(datasets.keys())}")
        
        # Create data collator
        data_collator = dataset_loader.create_data_collator(mlm=False)
        logger.info("Data collator created successfully")
        
        # Test with a small batch
        train_dataset = datasets['train']
        test_batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        
        logger.info("Testing data collator with sample batch...")
        logger.info(f"Sample batch sizes: {[len(item['input_ids']) for item in test_batch]}")
        
        # Test the collator
        collated_batch = data_collator(test_batch)
        
        logger.info("Data collator test successful!")
        logger.info(f"Collated batch shape: {collated_batch['input_ids'].shape}")
        logger.info(f"Labels shape: {collated_batch['labels'].shape}")
        logger.info(f"Attention mask shape: {collated_batch['attention_mask'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data collator test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_data_collator()
    if success:
        print("✅ Data collator fix validated successfully!")
    else:
        print("❌ Data collator test failed!")