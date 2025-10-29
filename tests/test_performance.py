#!/usr/bin/env python3
"""
Performance benchmarks for different GPU configurations.

Tests performance characteristics and optimization effectiveness across different hardware setups.
"""

import unittest
import time
import tempfile
import shutil
import json
import torch
import psutil
from pathlib import Path
import sys
import logging
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from data.dataset_loader import TigrinyaDatasetLoader
from utils.hardware import HardwareDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for different GPU configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        self.dataset_dir = self.temp_path / "dataset"
        self.output_dir = self.temp_path / "output"
        self.dataset_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create minimal benchmark dataset
        self.create_benchmark_dataset()
        
        # Skip tests if required resources are not available
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_benchmark_dataset(self):
        """Create a standardized benchmark dataset."""
        # Create consistent benchmark data for performance testing
        benchmark_texts = [
            "ሰላም! ከመይ ኣሎኻ? ሎሚ ጽቡቕ መዓልቲ እዩ።",
            "ትግርኛ ሰሚታዊ ቋንቋ እዩ። ኣብ ኤርትራን ሰሜን ኢትዮጵያን ይዝረብ።",
            "ገዓዝ ፊደል ይጥቀም። ብዙሓት ሰባት ትግርኛ ይዛረቡ።",
            "ኣብ ዓለም ብዙሕ ቋንቋታት ኣሎ። ኩሉ ቋንቋ ኣገዳሲ እዩ።",
            "ትምህርቲ ኣገዳሲ እዩ። ኩሉ ሰብ ክመሃር ኣለዎ።"
        ] * 20  # Repeat for more data
        
        # Create train.jsonl
        train_file = self.dataset_dir / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in benchmark_texts:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
        
        # Create validation.jsonl
        val_file = self.dataset_dir / "validation.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for text in benchmark_texts[:5]:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
    
    def test_hardware_detection_performance(self):
        """Test performance of hardware detection."""
        start_time = time.time()
        
        hardware_detector = HardwareDetector()
        gpu_config = hardware_detector.detect_gpu_config()
        
        detection_time = time.time() - start_time
        
        # Hardware detection should be fast (< 5 seconds)
        self.assertLess(detection_time, 5.0, "Hardware detection took too long")
        
        # Verify configuration is complete
        required_keys = [
            "gpu_name", "gpu_memory_gb", "batch_size", "gradient_accumulation_steps",
            "mixed_precision_dtype", "max_memory_usage"
        ]
        
        for key in required_keys:
            self.assertIn(key, gpu_config, f"Missing key in GPU config: {key}")
        
        logger.info(f"Hardware detection completed in {detection_time:.2f}s")
        logger.info(f"Detected configuration: {gpu_config['gpu_name']} with {gpu_config['gpu_memory_gb']}GB")


if __name__ == '__main__':
    # Set up comprehensive logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run performance benchmarks
    unittest.main(verbosity=2)