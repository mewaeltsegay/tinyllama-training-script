#!/usr/bin/env python3
"""
Simple performance benchmarks for the TinyLlama training system.
"""

import unittest
import time
import tempfile
import shutil
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from utils.hardware import HardwareDetector


class TestSimplePerformance(unittest.TestCase):
    """Simple performance tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        self.dataset_dir = self.temp_path / "dataset"
        self.tokenizer_dir = self.temp_path / "tokenizer"
        self.output_dir = self.temp_path / "output"
        
        for dir_path in [self.dataset_dir, self.tokenizer_dir, self.output_dir]:
            dir_path.mkdir()
        
        # Create minimal test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create minimal test files."""
        # Create a minimal dataset
        train_file = self.dataset_dir / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump({"text": "ሰላም! ከመይ ኣሎኻ?"}, f, ensure_ascii=False)
            f.write('\n')
        
        # Create minimal tokenizer config
        config_file = self.tokenizer_dir / "tokenizer_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({"vocab_size": 32000}, f)
        
        # Create minimal sentencepiece model (empty file for testing)
        (self.tokenizer_dir / "sentencepiece.model").touch()
    
    def test_hardware_detection_performance(self):
        """Test hardware detection performance."""
        start_time = time.time()
        
        hardware_detector = HardwareDetector()
        gpu_config = hardware_detector.detect_gpu_config()
        
        detection_time = time.time() - start_time
        
        # Hardware detection should be fast (< 10 seconds)
        self.assertLess(detection_time, 10.0, "Hardware detection took too long")
        
        # Verify configuration is complete
        required_keys = [
            "gpu_name", "gpu_memory_gb", "batch_size", "gradient_accumulation_steps"
        ]
        
        for key in required_keys:
            self.assertIn(key, gpu_config, f"Missing key in GPU config: {key}")
        
        print(f"Hardware detection completed in {detection_time:.2f}s")
    
    def test_config_creation_performance(self):
        """Test configuration creation performance."""
        start_time = time.time()
        
        # Create multiple configurations
        for i in range(100):
            config = TrainingConfig(
                dataset_dir=str(self.dataset_dir),
                tokenizer_dir=str(self.tokenizer_dir),
                output_dir=str(self.output_dir),
                learning_rate=1e-4 + i * 1e-6,
                batch_size=2
            )
        
        creation_time = time.time() - start_time
        
        # Should create 100 configs quickly
        self.assertLess(creation_time, 2.0, "Config creation too slow")
        
        print(f"Created 100 configurations in {creation_time:.3f}s")
    
    def test_batch_size_calculation_performance(self):
        """Test batch size calculation performance."""
        hardware_detector = HardwareDetector()
        
        start_time = time.time()
        
        # Test different memory scenarios
        memory_scenarios = [4096, 8192, 16384, 24576]  # MB
        model_size_mb = 2200
        
        for memory_mb in memory_scenarios:
            batch_size = hardware_detector.get_optimal_batch_size(memory_mb, model_size_mb)
            self.assertGreaterEqual(batch_size, 1)
        
        calculation_time = time.time() - start_time
        
        # Should calculate quickly
        self.assertLess(calculation_time, 1.0, "Batch size calculation too slow")
        
        print(f"Calculated batch sizes for {len(memory_scenarios)} scenarios in {calculation_time:.3f}s")


if __name__ == '__main__':
    unittest.main(verbosity=2)