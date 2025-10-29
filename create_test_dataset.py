#!/usr/bin/env python3
"""
Create a small test dataset with proper Tigrinya content to verify training works.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_dataset():
    """Create a small test dataset with real Tigrinya sentences."""
    
    # Sample Tigrinya sentences (basic greetings and phrases)
    tigrinya_samples = [
        "ሰላም ኣለኹም። ከመይ ኣለኹም?",  # Hello. How are you?
        "ጽቡቕ እዩ። ኣመስግነካ።",        # Good. Thank you.
        "ሎሚ ጽቡቕ መዓልቲ እዩ።",       # Today is a good day.
        "ኣነ ትግርኛ ይመሃር ኣለኹ።",      # I am learning Tigrinya.
        "እዚ መጽሓፍ ጽቡቕ እዩ።",        # This book is good.
        "ካብ ኣበይ ኢኹም?",             # Where are you from?
        "ኣነ ካብ ኤርትራ እየ።",         # I am from Eritrea.
        "ብዙሕ ኣመስግነካ።",            # Thank you very much.
        "ሰላም ኩን።",                  # Goodbye.
        "ጽቡቕ ለይቲ።",               # Good night.
        "ናብ ቤት ትምህርቲ ይኸይድ ኣለኹ።", # I am going to school.
        "እዚ ቤት ዓቢ እዩ።",           # This house is big.
        "ማይ ክሰቲ እፈቱ።",            # I want to drink water.
        "ሎሚ ዝሓለፈ መዓልቲ ጽቡቕ ነበረ።", # Yesterday was a good day.
        "ጌጋ ኣይገበርኩን።",            # I didn't make a mistake.
    ]
    
    # Create dataset directory
    dataset_dir = Path("dataset_test")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create train, validation, and test splits
    train_size = len(tigrinya_samples) * 20  # Repeat each 20 times for training
    val_size = len(tigrinya_samples) * 3     # Repeat each 3 times for validation
    test_size = len(tigrinya_samples) * 2    # Repeat each 2 times for test
    
    # Create training data (with some variation)
    logger.info("Creating training dataset...")
    with open(dataset_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for repeat in range(20):
            for i, text in enumerate(tigrinya_samples):
                # Add some variation by occasionally combining sentences
                if repeat % 3 == 0 and i < len(tigrinya_samples) - 1:
                    combined_text = f"{text} {tigrinya_samples[i + 1]}"
                    f.write(json.dumps({"text": combined_text}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    
    # Create validation data
    logger.info("Creating validation dataset...")
    with open(dataset_dir / "validation.jsonl", 'w', encoding='utf-8') as f:
        for repeat in range(3):
            for text in tigrinya_samples:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    
    # Create test data
    logger.info("Creating test dataset...")
    with open(dataset_dir / "test.jsonl", 'w', encoding='utf-8') as f:
        for repeat in range(2):
            for text in tigrinya_samples:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    
    logger.info(f"Created test dataset in {dataset_dir}/")
    logger.info(f"  - train.jsonl: ~{train_size} samples")
    logger.info(f"  - validation.jsonl: ~{val_size} samples") 
    logger.info(f"  - test.jsonl: ~{test_size} samples")
    
    return dataset_dir

if __name__ == "__main__":
    create_test_dataset()
    print("\n✅ Test dataset created successfully!")
    print("Now you can test training with: python tinyllama_tigrinya_training.py --dataset_dir dataset_test")