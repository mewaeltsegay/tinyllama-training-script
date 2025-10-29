#!/usr/bin/env python3
"""
Unit tests for configuration parsing and validation.

Tests the TrainingConfig class and command-line argument parsing.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig, parse_arguments


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / "dataset").mkdir()
        (self.temp_path / "tokenizer").mkdir()
        (self.temp_path / "output").mkdir()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        # Test default values
        self.assertEqual(config.model_name, "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.dataset_dir, "dataset")
        self.assertEqual(config.tokenizer_dir, "tokenizer")
        self.assertEqual(config.output_dir, "output")
        self.assertEqual(config.checkpoint_steps, 500)
        
        # Test small dataset optimization defaults
        self.assertEqual(config.warmup_ratio, 0.1)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.max_grad_norm, 1.0)
        self.assertEqual(config.token_dropout_prob, 0.1)
        self.assertEqual(config.length_variation_ratio, 0.3)
        self.assertEqual(config.augmentation_factor, 2)
        
        # Test scheduler defaults
        self.assertEqual(config.scheduler_type, "cosine")
        self.assertEqual(config.num_cycles, 0.5)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            max_length=256,
            dataset_dir=str(self.temp_path / "dataset"),
            tokenizer_dir=str(self.temp_path / "tokenizer"),
            output_dir=str(self.temp_path / "output")
        )
        self.assertTrue(config.validate())
        
        # Test invalid learning rate - should raise exception during creation
        with self.assertRaises(ValueError):
            TrainingConfig(
                learning_rate=-1.0,
                dataset_dir=str(self.temp_path / "dataset"),
                tokenizer_dir=str(self.temp_path / "tokenizer"),
                output_dir=str(self.temp_path / "output")
            )
        
        # Test invalid num_epochs - should raise exception during creation
        with self.assertRaises(ValueError):
            TrainingConfig(
                num_epochs=0,
                dataset_dir=str(self.temp_path / "dataset"),
                tokenizer_dir=str(self.temp_path / "tokenizer"),
                output_dir=str(self.temp_path / "output")
            )
        
        # Test invalid max_length - should raise exception during creation
        with self.assertRaises(ValueError):
            TrainingConfig(
                max_length=0,
                dataset_dir=str(self.temp_path / "dataset"),
                tokenizer_dir=str(self.temp_path / "tokenizer"),
                output_dir=str(self.temp_path / "output")
            )
        
        # Test invalid scheduler type - should raise exception during creation
        with self.assertRaises(ValueError):
            TrainingConfig(
                scheduler_type="invalid_scheduler",
                dataset_dir=str(self.temp_path / "dataset"),
                tokenizer_dir=str(self.temp_path / "tokenizer"),
                output_dir=str(self.temp_path / "output")
            )
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = TrainingConfig(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=4
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['learning_rate'], 1e-4)
        self.assertEqual(config_dict['num_epochs'], 5)
        self.assertEqual(config_dict['batch_size'], 4)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'learning_rate': 1e-4,
            'num_epochs': 5,
            'batch_size': 4,
            'max_length': 256
        }
        
        config = TrainingConfig.from_dict(config_dict)
        
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.max_length, 256)
    
    def test_directory_validation(self):
        """Test directory path validation."""
        # Test with existing directories
        config = TrainingConfig(
            dataset_dir=str(self.temp_path / "dataset"),
            tokenizer_dir=str(self.temp_path / "tokenizer"),
            output_dir=str(self.temp_path / "output")
        )
        self.assertTrue(config.validate_directories())
        
        # Test with non-existing directories - should raise exception during creation
        with self.assertRaises(FileNotFoundError):
            TrainingConfig(
                dataset_dir="nonexistent_dataset",
                tokenizer_dir="nonexistent_tokenizer"
            )
    
    def test_scheduler_type_validation(self):
        """Test scheduler type validation."""
        # Test valid scheduler types
        valid_schedulers = [
            "linear", "cosine", "cosine_with_restarts", "polynomial", 
            "constant", "constant_with_warmup", "inverse_sqrt", 
            "reduce_lr_on_plateau", "cosine_with_min_lr", 
            "cosine_warmup_with_min_lr", "warmup_stable_decay"
        ]
        
        for scheduler in valid_schedulers:
            config = TrainingConfig(
                scheduler_type=scheduler,
                dataset_dir=str(self.temp_path / "dataset"),
                tokenizer_dir=str(self.temp_path / "tokenizer"),
                output_dir=str(self.temp_path / "output")
            )
            self.assertEqual(config.scheduler_type, scheduler)
        
        # Test invalid scheduler type
        with self.assertRaises(ValueError):
            TrainingConfig(
                scheduler_type="invalid_scheduler_type",
                dataset_dir=str(self.temp_path / "dataset"),
                tokenizer_dir=str(self.temp_path / "tokenizer"),
                output_dir=str(self.temp_path / "output")
            )


class TestArgumentParsing(unittest.TestCase):
    """Test cases for command-line argument parsing."""
    
    def test_parse_basic_arguments(self):
        """Test parsing of basic command-line arguments."""
        # Skip this test if parse_arguments is not implemented
        try:
            from config.training_config import parse_arguments
        except ImportError:
            self.skipTest("parse_arguments function not implemented")
        
        args = [
            "--learning_rate", "1e-4",
            "--num_epochs", "5",
            "--batch_size", "4",
            "--max_length", "256"
        ]
        
        parsed_args = parse_arguments(args)
        
        self.assertEqual(parsed_args.learning_rate, 1e-4)
        self.assertEqual(parsed_args.num_epochs, 5)
        self.assertEqual(parsed_args.batch_size, 4)
        self.assertEqual(parsed_args.max_length, 256)
    
    def test_parse_path_arguments(self):
        """Test parsing of path arguments."""
        try:
            from config.training_config import parse_arguments
        except ImportError:
            self.skipTest("parse_arguments function not implemented")
        
        args = [
            "--dataset_dir", "custom_dataset",
            "--tokenizer_dir", "custom_tokenizer",
            "--output_dir", "custom_output"
        ]
        
        parsed_args = parse_arguments(args)
        
        self.assertEqual(parsed_args.dataset_dir, "custom_dataset")
        self.assertEqual(parsed_args.tokenizer_dir, "custom_tokenizer")
        self.assertEqual(parsed_args.output_dir, "custom_output")
    
    def test_parse_optimization_arguments(self):
        """Test parsing of small dataset optimization arguments."""
        try:
            from config.training_config import parse_arguments
        except ImportError:
            self.skipTest("parse_arguments function not implemented")
        
        args = [
            "--warmup_ratio", "0.2",
            "--weight_decay", "0.02",
            "--max_grad_norm", "2.0",
            "--token_dropout_prob", "0.15",
            "--augmentation_factor", "3"
        ]
        
        parsed_args = parse_arguments(args)
        
        self.assertEqual(parsed_args.warmup_ratio, 0.2)
        self.assertEqual(parsed_args.weight_decay, 0.02)
        self.assertEqual(parsed_args.max_grad_norm, 2.0)
        self.assertEqual(parsed_args.token_dropout_prob, 0.15)
        self.assertEqual(parsed_args.augmentation_factor, 3)
    
    def test_parse_inference_arguments(self):
        """Test parsing of inference-related arguments."""
        try:
            from config.training_config import parse_arguments
        except ImportError:
            self.skipTest("parse_arguments function not implemented")
        
        args = [
            "--max_new_tokens", "150",
            "--temperature", "0.9",
            "--inference_prompts", "ሰላም", "ከመይ ኣሎኻ"
        ]
        
        parsed_args = parse_arguments(args)
        
        self.assertEqual(parsed_args.max_new_tokens, 150)
        self.assertEqual(parsed_args.temperature, 0.9)
        self.assertEqual(parsed_args.inference_prompts, ["ሰላም", "ከመይ ኣሎኻ"])
    
    def test_default_arguments(self):
        """Test that default arguments are properly set."""
        try:
            from config.training_config import parse_arguments
        except ImportError:
            self.skipTest("parse_arguments function not implemented")
        
        parsed_args = parse_arguments([])
        
        # Check that defaults match TrainingConfig defaults
        config = TrainingConfig()
        self.assertEqual(parsed_args.learning_rate, config.learning_rate)
        self.assertEqual(parsed_args.num_epochs, config.num_epochs)
        self.assertEqual(parsed_args.max_length, config.max_length)


if __name__ == '__main__':
    unittest.main()