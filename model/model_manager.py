"""
Model management for TinyLlama and tokenizer handling.

This module provides the TinyLlamaManager class for loading TinyLlama models,
integrating Tigrinya SentencePiece tokenizers, and managing model checkpoints.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from sentencepiece import SentencePieceProcessor

logger = logging.getLogger(__name__)


class TinyLlamaManager:
    """
    Manages TinyLlama model loading, tokenizer integration, and checkpointing.
    
    This class handles:
    - Loading TinyLlama model and Tigrinya SentencePiece tokenizer
    - Resizing token embeddings to match Tigrinya vocabulary (32K tokens)
    - Model checkpoint saving and loading functionality
    - Model initialization and embedding layer adjustment
    """
    
    def __init__(self, model_name: str, tokenizer_dir: str):
        """
        Initialize TinyLlamaManager.
        
        Args:
            model_name: HuggingFace model identifier for TinyLlama
            tokenizer_dir: Directory containing Tigrinya SentencePiece tokenizer files
        """
        self.model_name = model_name
        self.tokenizer_dir = Path(tokenizer_dir)
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
        # Validate tokenizer directory
        self._validate_tokenizer_directory()
        
        logger.info(f"Initialized TinyLlamaManager with model: {model_name}")
        logger.info(f"Tokenizer directory: {self.tokenizer_dir}")
    
    def _validate_tokenizer_directory(self) -> None:
        """Validate that tokenizer directory contains required files."""
        if not self.tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_dir}")
        
        required_files = [
            "sentencepiece.model",
            "sentencepiece.vocab", 
            "tokenizer_config.json"
        ]
        
        for file_name in required_files:
            file_path = self.tokenizer_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required tokenizer file not found: {file_path}")
        
        logger.info("Tokenizer directory validation passed")
    
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load TinyLlama model and Tigrinya SentencePiece tokenizer with comprehensive error handling.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If model or tokenizer loading fails after all recovery attempts
        """
        from utils.logging import setup_comprehensive_logging
        
        # Setup error logging
        structured_logger, recovery_manager = setup_comprehensive_logging(
            output_dir="logs",
            log_level="INFO"
        )
        
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                logger.info(f"Loading TinyLlama model (attempt {attempt}/{max_attempts}): {self.model_name}")
                
                # Load TinyLlama model with error handling
                with structured_logger.error_context("model_loading", model_name=self.model_name):
                    self.model = self._load_model_with_fallbacks()
                
                logger.info("TinyLlama model loaded successfully")
                
                # Load Tigrinya tokenizer with error handling
                with structured_logger.error_context("tokenizer_loading", tokenizer_dir=str(self.tokenizer_dir)):
                    self.tokenizer = self._load_tigrinya_tokenizer()
                
                # Resize token embeddings to match Tigrinya vocabulary
                with structured_logger.error_context("embedding_resize"):
                    self.resize_token_embeddings(self.model, self.tokenizer)
                
                logger.info("Model and tokenizer loaded and configured successfully")
                return self.model, self.tokenizer
                
            except Exception as e:
                context = {
                    "model_name": self.model_name,
                    "tokenizer_dir": str(self.tokenizer_dir),
                    "attempt": attempt,
                    "max_attempts": max_attempts
                }
                
                if attempt < max_attempts:
                    logger.warning(f"Model loading attempt {attempt} failed: {e}")
                    
                    # Attempt recovery
                    recovery_successful = recovery_manager.handle_error(e, context)
                    
                    if recovery_successful:
                        logger.info("Recovery successful, retrying model loading...")
                        
                        # Clear GPU cache before retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        continue
                    else:
                        logger.warning(f"Recovery failed for attempt {attempt}")
                
                # If we reach here, either max attempts exceeded or recovery failed
                error_summary = structured_logger.get_error_summary()
                logger.error(f"Model loading failed after {attempt} attempts")
                logger.error(f"Error summary: {error_summary}")
                
                raise RuntimeError(
                    f"Model loading failed after {attempt} attempts. Final error: {str(e)}"
                ) from e
        
        # This should never be reached
        raise RuntimeError("Model loading failed: maximum attempts exceeded")
    
    def _load_model_with_fallbacks(self) -> PreTrainedModel:
        """
        Load model with multiple fallback strategies for error recovery.
        
        Returns:
            Loaded model
        """
        loading_strategies = [
            # Strategy 1: Standard loading with auto device mapping
            {
                "name": "standard_auto",
                "kwargs": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "trust_remote_code": True
                }
            },
            # Strategy 2: CPU loading first, then move to GPU
            {
                "name": "cpu_first",
                "kwargs": {
                    "torch_dtype": torch.float16,
                    "device_map": "cpu",
                    "trust_remote_code": True
                }
            },
            # Strategy 3: Float32 fallback
            {
                "name": "fp32_fallback",
                "kwargs": {
                    "torch_dtype": torch.float32,
                    "device_map": "auto",
                    "trust_remote_code": True
                }
            },
            # Strategy 4: Minimal loading
            {
                "name": "minimal",
                "kwargs": {
                    "trust_remote_code": True
                }
            }
        ]
        
        for strategy in loading_strategies:
            try:
                logger.info(f"Trying model loading strategy: {strategy['name']}")
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **strategy["kwargs"]
                )
                
                logger.info(f"Model loaded successfully with strategy: {strategy['name']}")
                return model
                
            except Exception as e:
                logger.warning(f"Strategy {strategy['name']} failed: {e}")
                
                # Clear GPU cache between attempts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue
        
        raise RuntimeError("All model loading strategies failed")
    
    def _load_tigrinya_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load Tigrinya SentencePiece tokenizer with comprehensive error handling and fallbacks.
        
        Returns:
            HuggingFace tokenizer configured for Tigrinya
        """
        tokenizer_strategies = [
            # Strategy 1: LlamaTokenizer with full configuration
            self._load_tokenizer_llama_full,
            # Strategy 2: LlamaTokenizer with minimal configuration
            self._load_tokenizer_llama_minimal,
            # Strategy 3: AutoTokenizer fallback
            self._load_tokenizer_auto_fallback,
            # Strategy 4: SentencePiece direct fallback
            self._load_tokenizer_sentencepiece_direct
        ]
        
        for i, strategy in enumerate(tokenizer_strategies, 1):
            try:
                logger.info(f"Trying tokenizer loading strategy {i}/{len(tokenizer_strategies)}")
                tokenizer = strategy()
                logger.info(f"Tokenizer loaded successfully with strategy {i}")
                return tokenizer
                
            except Exception as e:
                logger.warning(f"Tokenizer strategy {i} failed: {e}")
                if i == len(tokenizer_strategies):
                    # Last strategy failed
                    raise RuntimeError(f"All tokenizer loading strategies failed. Last error: {str(e)}") from e
                continue
        
        raise RuntimeError("Tokenizer loading failed: no strategies succeeded")
    
    def _load_tokenizer_llama_full(self) -> PreTrainedTokenizer:
        """Load tokenizer using LlamaTokenizer with full configuration."""
        # Load tokenizer configuration
        config_path = self.tokenizer_dir / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        
        vocab_size = tokenizer_config.get("vocab_size", 32000)
        logger.info(f"Loading Tigrinya tokenizer with vocab size: {vocab_size}")
        
        # Create LlamaTokenizer from SentencePiece model
        sp_model_path = self.tokenizer_dir / "sentencepiece.model"
        if not sp_model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
        
        tokenizer = LlamaTokenizer(
            vocab_file=str(sp_model_path),
            add_bos_token=True,
            add_eos_token=True,
            sp_model_kwargs={"add_bos": True, "add_eos": True},
            clean_up_tokenization_spaces=False
        )
        
        # Set special tokens
        special_tokens = tokenizer_config.get("special_tokens", [])
        if special_tokens:
            tokenizer.add_special_tokens({
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "mask_token": "<mask>"
            })
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Tigrinya tokenizer loaded with {len(tokenizer)} tokens")
        return tokenizer
    
    def _load_tokenizer_llama_minimal(self) -> PreTrainedTokenizer:
        """Load tokenizer using LlamaTokenizer with minimal configuration."""
        sp_model_path = self.tokenizer_dir / "sentencepiece.model"
        if not sp_model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
        
        tokenizer = LlamaTokenizer(vocab_file=str(sp_model_path))
        
        # Set basic special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Minimal Tigrinya tokenizer loaded with {len(tokenizer)} tokens")
        return tokenizer
    
    def _load_tokenizer_auto_fallback(self) -> PreTrainedTokenizer:
        """Load tokenizer using AutoTokenizer as fallback."""
        try:
            # Try to load from the tokenizer directory
            tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            logger.info("Loaded tokenizer using AutoTokenizer from directory")
            return tokenizer
        except:
            # Fallback to base TinyLlama tokenizer
            logger.warning("Loading base TinyLlama tokenizer as fallback")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Fallback tokenizer loaded with {len(tokenizer)} tokens")
            return tokenizer
    
    def _load_tokenizer_sentencepiece_direct(self) -> PreTrainedTokenizer:
        """Load tokenizer using direct SentencePiece integration."""
        sp_model_path = self.tokenizer_dir / "sentencepiece.model"
        if not sp_model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
        
        # Load base tokenizer and replace with SentencePiece
        base_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load SentencePiece processor
        sp_processor = SentencePieceProcessor()
        sp_processor.load(str(sp_model_path))
        
        # Create custom tokenizer class
        class CustomSentencePieceTokenizer:
            def __init__(self, sp_processor, base_tokenizer):
                self.sp_processor = sp_processor
                self.base_tokenizer = base_tokenizer
                self.vocab_size = sp_processor.get_piece_size()
                
                # Set special tokens
                self.bos_token = "<s>"
                self.eos_token = "</s>"
                self.pad_token = "<pad>"
                self.unk_token = "<unk>"
                
                self.bos_token_id = sp_processor.piece_to_id(self.bos_token)
                self.eos_token_id = sp_processor.piece_to_id(self.eos_token)
                self.pad_token_id = sp_processor.piece_to_id(self.pad_token)
                self.unk_token_id = sp_processor.piece_to_id(self.unk_token)
            
            def encode(self, text, add_special_tokens=True):
                tokens = self.sp_processor.encode(text, out_type=int)
                if add_special_tokens:
                    tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
                return tokens
            
            def decode(self, token_ids, skip_special_tokens=True):
                if skip_special_tokens:
                    # Filter out special tokens
                    special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                    token_ids = [tid for tid in token_ids if tid not in special_ids]
                return self.sp_processor.decode(token_ids)
            
            def __len__(self):
                return self.vocab_size
            
            def save_pretrained(self, path):
                # Delegate to base tokenizer
                self.base_tokenizer.save_pretrained(path)
        
        custom_tokenizer = CustomSentencePieceTokenizer(sp_processor, base_tokenizer)
        logger.info(f"Direct SentencePiece tokenizer loaded with {len(custom_tokenizer)} tokens")
        
        return custom_tokenizer
    
    def resize_token_embeddings(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        """
        Resize model token embeddings to match Tigrinya tokenizer vocabulary.
        
        Args:
            model: TinyLlama model to resize
            tokenizer: Tigrinya tokenizer with target vocabulary size
        """
        try:
            original_vocab_size = model.config.vocab_size
            new_vocab_size = len(tokenizer)
            
            logger.info(f"Resizing embeddings from {original_vocab_size} to {new_vocab_size}")
            
            if original_vocab_size != new_vocab_size:
                # Resize token embeddings
                model.resize_token_embeddings(new_vocab_size)
                
                # Update model config
                model.config.vocab_size = new_vocab_size
                
                logger.info("Token embeddings resized successfully")
                
                # Log embedding layer info
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_shape = model.model.embed_tokens.weight.shape
                    logger.info(f"Embedding layer shape: {embed_shape}")
                elif hasattr(model, 'embed_tokens'):
                    embed_shape = model.embed_tokens.weight.shape
                    logger.info(f"Embedding layer shape: {embed_shape}")
            else:
                logger.info("Vocabulary sizes match, no resizing needed")
                
        except Exception as e:
            logger.error(f"Failed to resize token embeddings: {str(e)}")
            raise RuntimeError(f"Embedding resize failed: {str(e)}") from e
    
    def save_checkpoint(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                       output_dir: str, step: int, training_args: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint with tokenizer and training metadata.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            output_dir: Base output directory
            step: Training step number
            training_args: Optional training arguments to save
            
        Returns:
            Path to saved checkpoint directory
        """
        try:
            checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving checkpoint to: {checkpoint_dir}")
            
            # Save model
            model.save_pretrained(checkpoint_dir)
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Save training metadata
            metadata = {
                "step": step,
                "model_name": self.model_name,
                "vocab_size": len(tokenizer),
                "tokenizer_dir": str(self.tokenizer_dir)
            }
            
            if training_args:
                metadata["training_args"] = training_args
            
            metadata_path = checkpoint_dir / "training_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Checkpoint saved successfully at step {step}")
            return str(checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint saving failed: {str(e)}") from e
    
    def load_checkpoint(self, checkpoint_dir: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
        """
        Load model checkpoint with tokenizer and training metadata.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        try:
            checkpoint_path = Path(checkpoint_dir)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
            # Load metadata
            metadata_path = checkpoint_path / "training_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            logger.info(f"Checkpoint loaded successfully from step {metadata.get('step', 'unknown')}")
            
            self.model = model
            self.tokenizer = tokenizer
            
            return model, tokenizer, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}") from e
    
    def get_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint in the output directory.
        
        Args:
            output_dir: Directory to search for checkpoints
            
        Returns:
            Path to latest checkpoint directory, or None if no checkpoints found
        """
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return None
            
            # Find all checkpoint directories
            checkpoint_dirs = [
                d for d in output_path.iterdir() 
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
            
            if not checkpoint_dirs:
                return None
            
            # Sort by step number
            def get_step_number(checkpoint_dir):
                try:
                    return int(checkpoint_dir.name.split("-")[1])
                except (IndexError, ValueError):
                    return 0
            
            latest_checkpoint = max(checkpoint_dirs, key=get_step_number)
            logger.info(f"Latest checkpoint found: {latest_checkpoint}")
            
            return str(latest_checkpoint)
            
        except Exception as e:
            logger.error(f"Failed to find latest checkpoint: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and tokenizer.
        
        Returns:
            Dictionary containing model and tokenizer information
        """
        info = {
            "model_name": self.model_name,
            "tokenizer_dir": str(self.tokenizer_dir),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }
        
        if self.model is not None:
            info.update({
                "model_config": self.model.config.to_dict(),
                "model_dtype": str(self.model.dtype),
                "model_device": str(self.model.device) if hasattr(self.model, 'device') else "unknown"
            })
        
        if self.tokenizer is not None:
            info.update({
                "vocab_size": len(self.tokenizer),
                "special_tokens": {
                    "bos_token": self.tokenizer.bos_token,
                    "eos_token": self.tokenizer.eos_token,
                    "pad_token": self.tokenizer.pad_token,
                    "unk_token": self.tokenizer.unk_token
                }
            })
        
        return info