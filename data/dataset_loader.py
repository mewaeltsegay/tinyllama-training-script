"""
Dataset loading utilities for Tigrinya JSONL files.

This module provides the TigrinyaDatasetLoader class for loading and preprocessing
JSONL datasets with SentencePiece tokenization, streaming support, and optimizations
for small datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Any
import random

import torch
from datasets import Dataset, load_dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer
)
import sentencepiece as spm

from utils.small_dataset_optimizer import SmallDatasetOptimizer

logger = logging.getLogger(__name__)


class TigrinyaDatasetLoader:
    """
    Dataset loader for Tigrinya JSONL files with SentencePiece tokenization.
    
    Supports streaming datasets, data shuffling, sequence length handling,
    and small dataset optimizations including data augmentation.
    """
    
    def __init__(
        self,
        dataset_dir: str,
        tokenizer_dir: str,
        max_length: int = 512,
        streaming: bool = False,
        seed: Optional[int] = None,
        small_dataset_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_dir: Directory containing JSONL dataset files
            tokenizer_dir: Directory containing SentencePiece tokenizer files
            max_length: Maximum sequence length for tokenization
            streaming: Whether to use streaming datasets for large files
            seed: Random seed for reproducibility
            small_dataset_config: Configuration for small dataset optimizations
            max_samples: Maximum number of samples to use from dataset (for testing/debugging)
        """
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.max_length = max_length
        self.streaming = streaming
        self.seed = seed
        self.small_dataset_config = small_dataset_config or {}
        self.max_samples = max_samples
        
        # Initialize random state
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Validate directories
        self._validate_directories()
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Initialize small dataset optimizer
        self.small_dataset_optimizer = None
        if not streaming:  # Only for non-streaming datasets
            self.small_dataset_optimizer = SmallDatasetOptimizer(
                tokenizer=self.tokenizer,
                max_length=max_length,
                token_dropout_prob=self.small_dataset_config.get("token_dropout_prob", 0.1),
                length_variation_ratio=self.small_dataset_config.get("length_variation_ratio", 0.3),
                augmentation_factor=self.small_dataset_config.get("augmentation_factor", 2),
                seed=seed
            )
        
        logger.info(f"Initialized TigrinyaDatasetLoader with max_length={max_length}, streaming={streaming}")
    
    def _validate_directories(self) -> None:
        """Validate that required directories and files exist with comprehensive error handling."""
        from utils.logging import setup_comprehensive_logging
        
        # Setup basic logging for validation
        if not hasattr(self, '_validation_logger'):
            self._validation_logger = logging.getLogger("dataset_validation")
        
        # Validate dataset directory
        if not self.dataset_dir.exists():
            error_msg = f"Dataset directory does not exist: {self.dataset_dir}"
            self._validation_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check if dataset directory is readable
        try:
            list(self.dataset_dir.iterdir())
        except PermissionError as e:
            error_msg = f"Cannot read dataset directory: {self.dataset_dir} - {e}"
            self._validation_logger.error(error_msg)
            raise PermissionError(error_msg) from e
        
        # Validate tokenizer directory
        if not self.tokenizer_dir.exists():
            error_msg = f"Tokenizer directory does not exist: {self.tokenizer_dir}"
            self._validation_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check for required tokenizer files with detailed validation
        required_files = ["sentencepiece.model", "tokenizer_config.json"]
        missing_files = []
        
        for file_name in required_files:
            file_path = self.tokenizer_dir / file_name
            if not file_path.exists():
                missing_files.append(str(file_path))
            elif not file_path.is_file():
                error_msg = f"Required tokenizer path is not a file: {file_path}"
                self._validation_logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            elif file_path.stat().st_size == 0:
                error_msg = f"Required tokenizer file is empty: {file_path}"
                self._validation_logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        if missing_files:
            error_msg = f"Required tokenizer files not found: {missing_files}"
            self._validation_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate dataset files exist and are readable
        dataset_files = list(self.dataset_dir.glob("*.jsonl"))
        if not dataset_files:
            error_msg = f"No JSONL dataset files found in {self.dataset_dir}"
            self._validation_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check dataset files are readable and non-empty
        for dataset_file in dataset_files:
            try:
                if dataset_file.stat().st_size == 0:
                    self._validation_logger.warning(f"Dataset file is empty: {dataset_file}")
                    continue
                
                # Test read first few lines
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3:  # Test first 3 lines
                            break
                        try:
                            json.loads(line.strip())
                        except json.JSONDecodeError as e:
                            self._validation_logger.warning(
                                f"Potential JSON error in {dataset_file} line {i+1}: {e}"
                            )
                            
            except (PermissionError, UnicodeDecodeError) as e:
                error_msg = f"Cannot read dataset file {dataset_file}: {e}"
                self._validation_logger.error(error_msg)
                raise PermissionError(error_msg) from e
        
        self._validation_logger.info(f"Validation completed: {len(dataset_files)} dataset files found")
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the SentencePiece tokenizer from the tokenizer directory.
        
        Returns:
            Loaded tokenizer instance
        """
        try:
            # Load tokenizer configuration
            config_path = self.tokenizer_dir / "tokenizer_config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
            
            # Load SentencePiece model
            sp_model_path = self.tokenizer_dir / "sentencepiece.model"
            
            # Create a basic tokenizer that can work with SentencePiece
            # We'll use AutoTokenizer but configure it for our SentencePiece model
            tokenizer = AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                use_fast=False
            )
            
            # Load our custom SentencePiece model
            sp_processor = spm.SentencePieceProcessor()
            sp_processor.load(str(sp_model_path))
            
            # Store the SentencePiece processor and config for later use
            tokenizer.sp_model = sp_processor
            tokenizer._tigrinya_vocab_size = tokenizer_config["vocab_size"]
            
            # Set special tokens
            special_tokens = tokenizer_config.get("special_tokens", ["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
            tokenizer.unk_token = special_tokens[0]
            tokenizer.bos_token = special_tokens[1]
            tokenizer.eos_token = special_tokens[2]
            tokenizer.pad_token = special_tokens[3]
            
            # Override tokenization methods to use our SentencePiece model
            def custom_tokenize(text):
                return sp_processor.encode(text, out_type=str)
            
            def custom_convert_tokens_to_ids(tokens):
                if isinstance(tokens, str):
                    return sp_processor.piece_to_id(tokens)
                return [sp_processor.piece_to_id(token) for token in tokens]
            
            def custom_convert_ids_to_tokens(ids, skip_special_tokens=False):
                if isinstance(ids, int):
                    return sp_processor.id_to_piece(ids)
                return [sp_processor.id_to_piece(id_) for id_ in ids]
            
            def custom_vocab_size():
                return tokenizer_config["vocab_size"]
            
            # Monkey patch the methods
            tokenizer._tokenize = custom_tokenize
            tokenizer.convert_tokens_to_ids = custom_convert_tokens_to_ids
            tokenizer.convert_ids_to_tokens = custom_convert_ids_to_tokens
            
            # Store vocab_size in a way that doesn't conflict with the property
            # We'll access it through our custom method
            
            logger.info(f"Loaded SentencePiece tokenizer with vocab_size={tokenizer._tigrinya_vocab_size}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _get_dataset_files(self) -> Dict[str, Path]:
        """
        Get available dataset files from the dataset directory.
        
        Returns:
            Dictionary mapping split names to file paths
        """
        dataset_files = {}
        
        # Look for standard split files
        for split_name in ["train", "validation", "test"]:
            file_path = self.dataset_dir / f"{split_name}.jsonl"
            if file_path.exists():
                dataset_files[split_name] = file_path
        
        if not dataset_files:
            raise FileNotFoundError(f"No JSONL dataset files found in {self.dataset_dir}")
        
        logger.info(f"Found dataset files: {list(dataset_files.keys())}")
        return dataset_files
    
    def _load_jsonl_dataset(self, file_path: Path, split: str) -> Union[Dataset, IterableDataset]:
        """
        Load a JSONL dataset file with robust error handling for corrupted data.
        
        Args:
            file_path: Path to the JSONL file
            split: Dataset split name
            
        Returns:
            Loaded dataset
        """
        try:
            # First, validate and clean the JSONL file
            cleaned_file_path = self._validate_and_clean_jsonl(file_path)
            
            if self.streaming:
                # Use streaming dataset for large files
                dataset = load_dataset(
                    "json",
                    data_files=str(cleaned_file_path),
                    streaming=True,
                    split="train"  # load_dataset requires a split name
                )
                logger.info(f"Loaded streaming dataset from {file_path}")
            else:
                # Load entire dataset into memory
                dataset = load_dataset(
                    "json",
                    data_files=str(cleaned_file_path),
                    split="train"
                )
                
                # Limit samples if specified
                if self.max_samples is not None:
                    original_size = len(dataset)
                    dataset = dataset.select(range(min(self.max_samples, original_size)))
                    logger.info(f"Limited {split} dataset from {original_size} to {len(dataset)} samples")
                
                logger.info(f"Loaded dataset from {file_path} with {len(dataset)} examples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            # Try to provide more specific error information
            if "JSONDecodeError" in str(type(e)):
                logger.error("Dataset contains invalid JSON. Run validation to identify corrupted lines.")
            elif "FileNotFoundError" in str(type(e)):
                logger.error("Dataset file not found. Check file path and permissions.")
            elif "PermissionError" in str(type(e)):
                logger.error("Permission denied accessing dataset file.")
            raise
    
    def _validate_and_clean_jsonl(self, file_path: Path) -> Path:
        """
        Validate and clean JSONL file, removing corrupted lines.
        
        Args:
            file_path: Path to the original JSONL file
            
        Returns:
            Path to cleaned JSONL file (may be the same as input if no cleaning needed)
        """
        corrupted_lines = []
        valid_lines = []
        total_lines = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        # Validate JSON structure
                        data = json.loads(line)
                        
                        # Validate required fields
                        if not isinstance(data, dict):
                            corrupted_lines.append((line_num, "Not a JSON object"))
                            continue
                        
                        if "text" not in data:
                            corrupted_lines.append((line_num, "Missing 'text' field"))
                            continue
                        
                        if not isinstance(data["text"], str):
                            corrupted_lines.append((line_num, "'text' field is not a string"))
                            continue
                        
                        if len(data["text"].strip()) == 0:
                            corrupted_lines.append((line_num, "Empty text content"))
                            continue
                        
                        # Line is valid
                        valid_lines.append(line)
                        
                    except json.JSONDecodeError as e:
                        corrupted_lines.append((line_num, f"JSON decode error: {e}"))
                        continue
                    except UnicodeDecodeError as e:
                        corrupted_lines.append((line_num, f"Unicode decode error: {e}"))
                        continue
                    except Exception as e:
                        corrupted_lines.append((line_num, f"Unexpected error: {e}"))
                        continue
        
        except Exception as e:
            logger.error(f"Failed to read JSONL file {file_path}: {e}")
            raise
        
        # Log validation results
        if corrupted_lines:
            logger.warning(f"Found {len(corrupted_lines)} corrupted lines in {file_path}")
            logger.warning(f"Valid lines: {len(valid_lines)}/{total_lines}")
            
            # Log first few corrupted lines for debugging
            for i, (line_num, error) in enumerate(corrupted_lines[:5]):
                logger.warning(f"  Line {line_num}: {error}")
            
            if len(corrupted_lines) > 5:
                logger.warning(f"  ... and {len(corrupted_lines) - 5} more corrupted lines")
            
            # Create cleaned file if we have corrupted lines
            if len(valid_lines) == 0:
                raise ValueError(f"No valid lines found in {file_path}")
            
            # Create cleaned file
            cleaned_file_path = file_path.parent / f"{file_path.stem}_cleaned.jsonl"
            with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                for line in valid_lines:
                    f.write(line + '\n')
            
            logger.info(f"Created cleaned dataset file: {cleaned_file_path}")
            return cleaned_file_path
        
        else:
            logger.info(f"Dataset file {file_path} is clean ({len(valid_lines)} valid lines)")
            return file_path
    
    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """
        Tokenize text examples.
        
        Args:
            examples: Batch of examples with 'text' field
            
        Returns:
            Tokenized examples
        """
        # Tokenize the texts with proper padding and truncation
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=self.max_length,
            return_attention_mask=False,  # Not needed for causal LM
            return_token_type_ids=False,
            return_tensors=None  # Return lists, not tensors
        )
        
        # For causal language modeling, labels are the same as input_ids
        # Ensure labels are properly formatted as lists
        if isinstance(tokenized["input_ids"][0], list):
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        else:
            tokenized["labels"] = tokenized["input_ids"].copy()
        
        # Validate tokenization results
        if len(tokenized["input_ids"]) != len(tokenized["labels"]):
            raise ValueError(f"Mismatch between input_ids and labels length: {len(tokenized['input_ids'])} vs {len(tokenized['labels'])}")
        
        # Check for any sequences that are too long
        for i, ids in enumerate(tokenized["input_ids"]):
            if len(ids) > self.max_length:
                logger.warning(f"Sequence {i} length {len(ids)} exceeds max_length {self.max_length}")
        
        return tokenized
    
    def _apply_small_dataset_optimizations(self, dataset: Dataset) -> Dataset:
        """
        Apply optimizations for small datasets using the SmallDatasetOptimizer.
        
        Args:
            dataset: Input dataset (already tokenized)
            
        Returns:
            Augmented dataset
        """
        if self.streaming or self.small_dataset_optimizer is None:
            # Skip augmentation for streaming datasets or if optimizer not available
            return dataset
        
        # Use the dedicated small dataset optimizer
        return self.small_dataset_optimizer.augment_dataset(dataset)
    
    def load_datasets(self) -> Dict[str, Union[Dataset, IterableDataset]]:
        """
        Load all available datasets with tokenization and preprocessing.
        
        Returns:
            Dictionary mapping split names to processed datasets
        """
        dataset_files = self._get_dataset_files()
        datasets = {}
        
        for split, file_path in dataset_files.items():
            # Load raw dataset
            raw_dataset = self._load_jsonl_dataset(file_path, split)
            
            # Tokenize the dataset
            if self.streaming:
                # For streaming datasets, apply tokenization on-the-fly
                tokenized_dataset = raw_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=raw_dataset.column_names
                )
            else:
                # For regular datasets, tokenize and cache
                tokenized_dataset = raw_dataset.map(
                    self._tokenize_function,
                    batched=True,
                    remove_columns=raw_dataset.column_names,
                    desc=f"Tokenizing {split} dataset"
                )
                
                # Apply small dataset optimizations for training set
                if split == "train":
                    tokenized_dataset = self._apply_small_dataset_optimizations(tokenized_dataset)
                
                # Shuffle the dataset
                tokenized_dataset = tokenized_dataset.shuffle(seed=self.seed)
            
            datasets[split] = tokenized_dataset
        
        return datasets
    
    def create_data_collator(self, mlm: bool = False):
        """
        Create a fixed data collator for causal language modeling that resolves zero loss issues.
        
        Args:
            mlm: Whether to use masked language modeling (False for causal LM)
            
        Returns:
            FixedDataCollator instance that properly handles padding and labels
        """
        # Import the fixed data collator
        from data.fixed_data_collator import FixedDataCollator
        
        # Use the fixed data collator with conservative labeling for stability
        data_collator = FixedDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            max_length=self.max_length,
            ignore_pad_token_for_loss=True,
            conservative_labeling=True  # Enable conservative labeling to prevent gradient explosion
        )
        
        logger.info(f"Created FixedDataCollator for causal language modeling")
        logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token}")
        logger.info(f"Tokenizer pad_token_id: {self.tokenizer.pad_token_id}")
        logger.info(f"Max sequence length: {self.max_length}")
        
        return data_collator
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded datasets.
        
        Returns:
            Dictionary with dataset information
        """
        dataset_files = self._get_dataset_files()
        info = {
            "dataset_dir": str(self.dataset_dir),
            "tokenizer_dir": str(self.tokenizer_dir),
            "max_length": self.max_length,
            "streaming": self.streaming,
            "vocab_size": self.tokenizer._tigrinya_vocab_size,
            "special_tokens": {
                "unk_token": self.tokenizer.unk_token,
                "bos_token": self.tokenizer.bos_token,
                "eos_token": self.tokenizer.eos_token,
                "pad_token": self.tokenizer.pad_token,
            },
            "available_splits": list(dataset_files.keys()),
            "dataset_files": {split: str(path) for split, path in dataset_files.items()}
        }
        
        return info
    
    def validate_dataset_format(self, sample_size: int = 10) -> bool:
        """
        Validate that dataset files have the correct format.
        
        Args:
            sample_size: Number of samples to validate from each file
            
        Returns:
            True if all files are valid, False otherwise
        """
        dataset_files = self._get_dataset_files()
        
        for split, file_path in dataset_files.items():
            try:
                logger.info(f"Validating {split} dataset format...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= sample_size:
                            break
                        
                        try:
                            data = json.loads(line.strip())
                            if not isinstance(data, dict):
                                logger.error(f"Invalid format in {file_path} line {i+1}: not a JSON object")
                                return False
                            
                            if "text" not in data:
                                logger.error(f"Invalid format in {file_path} line {i+1}: missing 'text' field")
                                return False
                            
                            if not isinstance(data["text"], str):
                                logger.error(f"Invalid format in {file_path} line {i+1}: 'text' field is not a string")
                                return False
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in {file_path} line {i+1}: {e}")
                            return False
                
                logger.info(f"✓ {split} dataset format is valid")
                
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
                return False
        
        logger.info("All dataset files have valid format")
        return True