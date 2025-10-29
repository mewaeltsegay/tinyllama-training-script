#!/usr/bin/env python3
"""
Unit tests for dataset loading and tokenization correctness.

Tests the TigrinyaDatasetLoader class and tokenization functionality.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import TigrinyaDatasetLoader
from config.training_config import TrainingConfig


class TestDatasetLoader(unittest.TestCase):
    """Test cases for TigrinyaDatasetLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test dataset directory
        self.dataset_dir = self.temp_path / "dataset"
        self.dataset_dir.mkdir()
        
        # Create test tokenizer directory (copy from actual tokenizer if exists)
        self.tokenizer_dir = self.temp_path / "tokenizer"
        self.tokenizer_dir.mkdir()
        
        # Create sample JSONL files
        self.create_sample_datasets()
        self.create_sample_tokenizer()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_datasets(self):
        """Create sample JSONL dataset files."""
        # Sample Tigrinya texts
        sample_texts = [
            "ሰላም! ከመይ ኣሎኻ?",
            "ሎሚ ጽቡቕ መዓልቲ እዩ።",
            "ኣብ ትግራይ ብዙሕ ሰብ ይነብር።",
            "ትግርኛ ሰሚታዊ ቋንቋ እዩ።",
            "ኣብ ኤርትራን ሰሜን ኢትዮጵያን ይዝረብ።",
            "ገዓዝ ፊደል ይጥቀም።",
            "ብዙሓት ሰባት ትግርኛ ይዛረቡ።",
            "ኣብ ዓለም ብዙሕ ቋንቋታት ኣሎ።"
        ]
        
        # Create train.jsonl
        train_file = self.dataset_dir / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in sample_texts[:6]:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
        
        # Create validation.jsonl
        val_file = self.dataset_dir / "validation.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for text in sample_texts[6:7]:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
        
        # Create test.jsonl
        test_file = self.dataset_dir / "test.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for text in sample_texts[7:8]:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
    
    def create_sample_tokenizer(self):
        """Create a minimal tokenizer configuration for testing."""
        # Create a basic tokenizer config
        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizer",
            "vocab_size": 32000,
            "model_max_length": 2048,
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>"
        }
        
        config_file = self.tokenizer_dir / "tokenizer_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Create a minimal vocab file (this is a simplified version)
        vocab_file = self.tokenizer_dir / "tokenizer.json"
        minimal_vocab = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 2, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            ],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "BPE",
                "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
                "merges": []
            }
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(minimal_vocab, f, indent=2)
    
    def test_dataset_loader_initialization(self):
        """Test dataset loader initialization."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",  # Use real tokenizer
            max_length=512
        )
        
        self.assertEqual(str(loader.dataset_dir), str(self.dataset_dir))
        self.assertEqual(str(loader.tokenizer_dir), "tokenizer")
        self.assertEqual(loader.max_length, 512)
    
    def test_dataset_format_validation(self):
        """Test dataset format validation."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",
            max_length=512
        )
        
        # Should validate successfully with our sample data
        self.assertTrue(loader.validate_dataset_format())
    
    def test_load_datasets(self):
        """Test loading of datasets."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",
            max_length=512
        )
        
        datasets = loader.load_datasets()
        
        # Check that all expected splits are loaded
        expected_splits = ["train", "validation", "test"]
        for split in expected_splits:
            self.assertIn(split, datasets)
            self.assertGreater(len(datasets[split]), 0)
        
        # Check dataset sizes (may be augmented for training data)
        self.assertGreaterEqual(len(datasets["train"]), 6)  # May be augmented
        self.assertEqual(len(datasets["validation"]), 1)
        self.assertEqual(len(datasets["test"]), 1)
    
    def test_data_collator_creation(self):
        """Test data collator creation."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",
            max_length=512
        )
        
        data_collator = loader.create_data_collator()
        self.assertIsNotNone(data_collator)
    
    def test_small_dataset_optimizations(self):
        """Test small dataset optimization techniques."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",
            max_length=512,
            small_dataset_config={
                "token_dropout_prob": 0.1,
                "length_variation_ratio": 0.3,
                "augmentation_factor": 2
            }
        )
        
        datasets = loader.load_datasets()
        
        # Check that datasets are loaded
        self.assertIn("train", datasets)
        self.assertGreater(len(datasets["train"]), 0)
    
    def test_streaming_dataset_support(self):
        """Test streaming dataset functionality."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",
            max_length=512,
            streaming=True
        )
        
        datasets = loader.load_datasets()
        
        # Check that streaming datasets are created
        self.assertIn("train", datasets)
        # Note: Streaming datasets don't have len(), so we can't check size directly
    
    def test_invalid_dataset_directory(self):
        """Test handling of invalid dataset directory."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        with self.assertRaises((FileNotFoundError, ValueError)):
            loader = TigrinyaDatasetLoader(
                dataset_dir="nonexistent_directory",
                tokenizer_dir="tokenizer",
                max_length=512
            )
            loader.load_datasets()
    
    def test_corrupted_jsonl_handling(self):
        """Test handling of corrupted JSONL files."""
        # Skip this test if we don't have a real tokenizer
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
        
        # Create a corrupted JSONL file
        corrupted_file = self.dataset_dir / "corrupted.jsonl"
        with open(corrupted_file, 'w', encoding='utf-8') as f:
            f.write('{"text": "valid line"}\n')
            f.write('invalid json line\n')
            f.write('{"text": "another valid line"}\n')
        
        loader = TigrinyaDatasetLoader(
            dataset_dir=str(self.dataset_dir),
            tokenizer_dir="tokenizer",
            max_length=512
        )
        
        # Should handle corrupted data gracefully
        # The exact behavior depends on implementation
        try:
            datasets = loader.load_datasets()
            # If it succeeds, that's fine - it handled the corruption
            self.assertIsNotNone(datasets)
        except Exception as e:
            # If it fails, it should be a specific, expected error
            self.assertIsInstance(e, (ValueError, json.JSONDecodeError))


class TestTokenizationCorrectness(unittest.TestCase):
    """Test cases for tokenization correctness."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip all tests if real tokenizer is not available
        if not Path("tokenizer").exists():
            self.skipTest("Real tokenizer directory not available")
    
    def test_tigrinya_text_tokenization(self):
        """Test tokenization of Tigrinya text."""
        loader = TigrinyaDatasetLoader(
            dataset_dir="dataset",
            tokenizer_dir="tokenizer",
            max_length=512
        )
        
        # Test tokenization of sample Tigrinya text
        sample_text = "ሰላም! ከመይ ኣሎኻ?"
        
        # Tokenize the text
        tokens = loader.tokenizer.encode(sample_text)
        
        # Check that tokenization produces reasonable results
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Check that we can decode back to text
        decoded_text = loader.tokenizer.decode(tokens)
        self.assertIsInstance(decoded_text, str)
        
        # The decoded text should contain the original Ge'ez characters
        # (exact match might not be possible due to tokenization)
        self.assertTrue(any(char in decoded_text for char in "ሰላም"))
    
    def test_special_tokens_handling(self):
        """Test handling of special tokens."""
        loader = TigrinyaDatasetLoader(
            dataset_dir="dataset",
            tokenizer_dir="tokenizer",
            max_length=512
        )
        
        # Test that special tokens are properly configured
        tokenizer = loader.tokenizer
        
        # Check for essential special tokens
        self.assertIsNotNone(tokenizer.pad_token)
        self.assertIsNotNone(tokenizer.eos_token)
        
        # Test tokenization with special tokens
        text_with_special = f"{tokenizer.bos_token}ሰላም{tokenizer.eos_token}"
        tokens = tokenizer.encode(text_with_special)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_sequence_length_handling(self):
        """Test handling of different sequence lengths."""
        loader = TigrinyaDatasetLoader(
            dataset_dir="dataset",
            tokenizer_dir="tokenizer",
            max_length=128  # Shorter max length for testing
        )
        
        # Test short text
        short_text = "ሰላም"
        short_tokens = loader.tokenizer.encode(short_text, max_length=128, truncation=True)
        self.assertLessEqual(len(short_tokens), 128)
        
        # Test long text (create by repeating)
        long_text = "ሰላም! ከመይ ኣሎኻ? " * 50  # Should exceed max_length
        long_tokens = loader.tokenizer.encode(long_text, max_length=128, truncation=True)
        self.assertLessEqual(len(long_tokens), 128)


if __name__ == '__main__':
    unittest.main()