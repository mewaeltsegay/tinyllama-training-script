#!/usr/bin/env python3
"""
Fixed Data Collator for Causal Language Modeling

This module provides a robust data collator that fixes the zero loss issue
by properly handling input/label formatting, padding, and sequence processing
for causal language modeling tasks.
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedTokenizer
import numpy as np

logger = logging.getLogger(__name__)


class FixedDataCollator:
    """
    Fixed data collator that resolves zero loss issues in causal language modeling.
    
    Key fixes:
    1. Proper label creation with correct shifting for causal LM
    2. Correct padding strategy that preserves loss computation
    3. Validation to ensure non-padding tokens exist in batches
    4. Proper handling of special tokens and attention masks
    5. Robust error handling and logging for debugging
    6. Gradient stabilization through conservative label handling
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pad_to_multiple_of: Optional[int] = 8,
        return_tensors: str = "pt",
        max_length: Optional[int] = None,
        ignore_pad_token_for_loss: bool = True,
        conservative_labeling: bool = True
    ):
        """
        Initialize the fixed data collator.
        
        Args:
            tokenizer: The tokenizer to use for padding
            pad_to_multiple_of: Pad sequences to multiple of this value
            return_tensors: Type of tensors to return ("pt" for PyTorch)
            max_length: Maximum sequence length (optional)
            ignore_pad_token_for_loss: Whether to ignore pad tokens in loss calculation
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.max_length = max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.conservative_labeling = conservative_labeling
        
        # Ensure pad token is properly configured
        self._setup_pad_token()
        
        # Set up loss ignore index
        self.label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
        
        logger.info(f"FixedDataCollator initialized:")
        logger.info(f"  pad_token: '{self.tokenizer.pad_token}' (id: {self.tokenizer.pad_token_id})")
        logger.info(f"  eos_token: '{self.tokenizer.eos_token}' (id: {self.tokenizer.eos_token_id})")
        logger.info(f"  bos_token: '{self.tokenizer.bos_token}' (id: {self.tokenizer.bos_token_id})")
        logger.info(f"  pad_to_multiple_of: {self.pad_to_multiple_of}")
        logger.info(f"  label_pad_token_id: {self.label_pad_token_id}")
        logger.info(f"  max_length: {self.max_length}")
        logger.info(f"  conservative_labeling: {self.conservative_labeling}")
    
    def _setup_pad_token(self):
        """Ensure pad token is properly configured."""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                logger.info("Set pad_token to unk_token")
            else:
                # Add a new pad token
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                logger.info("Added new pad_token: '<pad>'")
        
        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch with proper padding and label handling.
        
        Args:
            features: List of feature dictionaries containing 'input_ids'
            
        Returns:
            Batch dictionary with input_ids, attention_mask, and labels
        """
        if not features:
            raise ValueError("Cannot create batch from empty features list")
        
        # Validate and extract input sequences
        input_ids_list = self._extract_input_ids(features)
        
        if not input_ids_list:
            raise ValueError("No valid input sequences found in features")
        
        # Calculate target sequence length
        max_length = self._calculate_target_length(input_ids_list)
        
        # Create padded batch
        batch = self._create_padded_batch(input_ids_list, max_length)
        
        # Validate the batch
        self._validate_batch(batch)
        
        # Log batch statistics for debugging
        self._log_batch_stats(batch)
        
        return batch
    
    def _extract_input_ids(self, features: List[Dict[str, Any]]) -> List[List[int]]:
        """Extract and validate input_ids from features."""
        input_ids_list = []
        
        for i, feature in enumerate(features):
            if "input_ids" not in feature:
                logger.warning(f"Feature {i} missing 'input_ids', skipping")
                continue
            
            input_ids = feature["input_ids"]
            
            # Convert to list if tensor
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            elif not isinstance(input_ids, list):
                logger.warning(f"Feature {i} 'input_ids' has invalid type {type(input_ids)}, skipping")
                continue
            
            # Validate sequence
            if len(input_ids) == 0:
                logger.warning(f"Feature {i} has empty input_ids, skipping")
                continue
            
            # Truncate if necessary
            if self.max_length and len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                logger.debug(f"Truncated feature {i} from {len(feature['input_ids'])} to {self.max_length}")
            
            # Ensure sequence has at least one non-pad token
            non_pad_tokens = sum(1 for token_id in input_ids if token_id != self.tokenizer.pad_token_id)
            if non_pad_tokens == 0:
                logger.warning(f"Feature {i} contains only padding tokens, skipping")
                continue
            
            input_ids_list.append(input_ids)
        
        return input_ids_list
    
    def _calculate_target_length(self, input_ids_list: List[List[int]]) -> int:
        """Calculate the target length for padding."""
        max_length = max(len(ids) for ids in input_ids_list)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Respect max_length constraint
        if self.max_length:
            max_length = min(max_length, self.max_length)
        
        return max_length
    
    def _create_padded_batch(self, input_ids_list: List[List[int]], target_length: int) -> Dict[str, torch.Tensor]:
        """Create padded batch with proper labels and attention masks."""
        batch_size = len(input_ids_list)
        
        # Initialize tensors
        input_ids = torch.full((batch_size, target_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, target_length), dtype=torch.long)
        labels = torch.full((batch_size, target_length), self.label_pad_token_id, dtype=torch.long)
        
        for i, sequence in enumerate(input_ids_list):
            seq_len = len(sequence)
            
            # Fill input_ids
            input_ids[i, :seq_len] = torch.tensor(sequence, dtype=torch.long)
            
            # Fill attention_mask (1 for real tokens, 0 for padding)
            attention_mask[i, :seq_len] = 1
            
            # Fill labels for causal language modeling
            # CRITICAL FIX: For causal LM, we need to ensure proper label handling
            # Labels should be input_ids shifted by 1 position, but since the model
            # handles shifting internally, we copy input_ids to labels but ensure
            # we have meaningful targets for loss computation
            
            if self.conservative_labeling:
                # Conservative approach: mask more aggressively to prevent gradient explosion
                # This reduces the training signal but improves stability
                
                # Start with all positions masked
                labels[i, :] = self.label_pad_token_id
                
                # Only unmask positions that are safe for training
                if seq_len > 2:
                    # For longer sequences, only use middle portion for training
                    # Skip first token (often BOS) and last few tokens
                    start_idx = 1 if (sequence[0] == self.tokenizer.bos_token_id) else 0
                    end_idx = max(start_idx + 1, seq_len - 1)  # Leave at least one token
                    
                    # Unmask the safe range
                    labels[i, start_idx:end_idx] = torch.tensor(sequence[start_idx:end_idx], dtype=torch.long)
                    
                elif seq_len == 2:
                    # For 2-token sequences, only predict the second token
                    labels[i, 1] = sequence[1]
                    
                elif seq_len == 1:
                    # For single tokens, skip training to avoid instability
                    # Keep all positions masked
                    pass
                    
            else:
                # Standard approach: copy sequence to labels
                labels[i, :seq_len] = torch.tensor(sequence, dtype=torch.long)
                
                # Handle special tokens
                if (seq_len > 0 and 
                    self.tokenizer.bos_token_id is not None and 
                    sequence[0] == self.tokenizer.bos_token_id and
                    seq_len > 1):
                    # Mask BOS token
                    labels[i, 0] = self.label_pad_token_id
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate the created batch to ensure it will produce valid loss."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        # Check tensor shapes
        if not (input_ids.shape == labels.shape == attention_mask.shape):
            raise ValueError(
                f"Shape mismatch: input_ids {input_ids.shape}, "
                f"labels {labels.shape}, attention_mask {attention_mask.shape}"
            )
        
        # Check for valid tokens in labels
        valid_label_positions = (labels != self.label_pad_token_id)
        total_valid_tokens = valid_label_positions.sum().item()
        
        if total_valid_tokens == 0:
            raise ValueError("Batch contains no valid tokens for loss calculation")
        
        # Check attention mask consistency
        attention_tokens = attention_mask.sum().item()
        if attention_tokens == 0:
            raise ValueError("Batch has no attention tokens")
        
        # Validate token ranges
        vocab_size = len(self.tokenizer)
        if input_ids.max().item() >= vocab_size:
            raise ValueError(f"input_ids contains token {input_ids.max().item()} >= vocab_size {vocab_size}")
        
        if input_ids.min().item() < 0:
            raise ValueError(f"input_ids contains negative token {input_ids.min().item()}")
        
        # Check for reasonable token distribution
        batch_size, seq_len = input_ids.shape
        total_positions = batch_size * seq_len
        valid_ratio = total_valid_tokens / total_positions
        
        if valid_ratio < 0.05:  # Less than 5% valid tokens
            logger.warning(f"Batch has very low valid token ratio: {valid_ratio:.2%}")
        
        logger.debug(f"Batch validation passed: {total_valid_tokens}/{total_positions} valid tokens ({valid_ratio:.2%})")
    
    def _log_batch_stats(self, batch: Dict[str, torch.Tensor]) -> None:
        """Log batch statistics for debugging."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        batch_size, seq_len = input_ids.shape
        
        # Calculate statistics
        total_tokens = batch_size * seq_len
        attention_tokens = attention_mask.sum().item()
        valid_label_tokens = (labels != self.label_pad_token_id).sum().item()
        pad_tokens = (input_ids == self.tokenizer.pad_token_id).sum().item()
        
        # Log statistics
        logger.debug(f"Batch created: shape={input_ids.shape}")
        logger.debug(f"  Total positions: {total_tokens}")
        logger.debug(f"  Attention tokens: {attention_tokens} ({attention_tokens/total_tokens:.2%})")
        logger.debug(f"  Valid label tokens: {valid_label_tokens} ({valid_label_tokens/total_tokens:.2%})")
        logger.debug(f"  Padding tokens: {pad_tokens} ({pad_tokens/total_tokens:.2%})")
        
        # Log sequence length distribution
        seq_lengths = attention_mask.sum(dim=1).tolist()
        logger.debug(f"  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")


class LossValidator:
    """
    Utility class to validate loss computation and diagnose zero loss issues.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.zero_loss_count = 0
        self.validation_history = []
    
    def validate_batch_for_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate a batch to ensure it will produce meaningful loss.
        
        Args:
            batch: Batch dictionary with input_ids, labels, attention_mask
            
        Returns:
            Validation report dictionary
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }
        
        # Basic shape validation
        if not (input_ids.shape == labels.shape == attention_mask.shape):
            report["valid"] = False
            report["errors"].append(f"Shape mismatch: {input_ids.shape} vs {labels.shape} vs {attention_mask.shape}")
            return report
        
        batch_size, seq_len = input_ids.shape
        total_positions = batch_size * seq_len
        
        # Count different token types
        valid_labels = (labels != -100).sum().item()
        attention_tokens = attention_mask.sum().item()
        pad_tokens = (input_ids == self.tokenizer.pad_token_id).sum().item()
        
        report["stats"] = {
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "total_positions": total_positions,
            "valid_labels": valid_labels,
            "attention_tokens": attention_tokens,
            "pad_tokens": pad_tokens,
            "valid_label_ratio": valid_labels / total_positions,
            "attention_ratio": attention_tokens / total_positions
        }
        
        # Validation checks
        if valid_labels == 0:
            report["valid"] = False
            report["errors"].append("No valid labels for loss calculation (all labels are -100)")
        
        if attention_tokens == 0:
            report["valid"] = False
            report["errors"].append("No attention tokens (all attention_mask values are 0)")
        
        if valid_labels / total_positions < 0.05:
            report["warnings"].append(f"Very low valid label ratio: {valid_labels/total_positions:.2%}")
        
        # Check for token range issues
        if input_ids.max().item() >= len(self.tokenizer):
            report["valid"] = False
            report["errors"].append(f"Token ID {input_ids.max().item()} exceeds vocab size {len(self.tokenizer)}")
        
        if input_ids.min().item() < 0:
            report["valid"] = False
            report["errors"].append(f"Negative token ID found: {input_ids.min().item()}")
        
        # Store validation history
        self.validation_history.append(report)
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
        
        return report
    
    def validate_model_forward_pass(self, model, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate that the model forward pass produces meaningful loss.
        
        Args:
            model: The model to test
            batch: Batch dictionary
            
        Returns:
            Validation report dictionary
        """
        report = {
            "valid": True,
            "loss_value": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            with torch.no_grad():
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                if loss is None:
                    report["valid"] = False
                    report["errors"].append("Model returned None loss")
                    return report
                
                loss_value = loss.item()
                report["loss_value"] = loss_value
                
                # Validate loss value
                if np.isnan(loss_value):
                    report["valid"] = False
                    report["errors"].append("Model returned NaN loss")
                elif np.isinf(loss_value):
                    report["valid"] = False
                    report["errors"].append("Model returned infinite loss")
                elif loss_value == 0.0:
                    self.zero_loss_count += 1
                    report["valid"] = False
                    report["errors"].append(f"Model returned zero loss (count: {self.zero_loss_count})")
                elif loss_value < 0:
                    report["warnings"].append(f"Negative loss value: {loss_value}")
                elif loss_value > 100:
                    report["warnings"].append(f"Very high loss value: {loss_value}")
                
        except Exception as e:
            report["valid"] = False
            report["errors"].append(f"Model forward pass failed: {str(e)}")
        
        return report
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation history."""
        if not self.validation_history:
            return {"status": "no_validations"}
        
        recent_reports = self.validation_history[-10:]
        
        return {
            "total_validations": len(self.validation_history),
            "recent_validations": len(recent_reports),
            "recent_valid_count": sum(1 for r in recent_reports if r["valid"]),
            "zero_loss_count": self.zero_loss_count,
            "common_errors": self._get_common_errors(recent_reports),
            "avg_valid_label_ratio": np.mean([r["stats"]["valid_label_ratio"] for r in recent_reports if "stats" in r])
        }
    
    def _get_common_errors(self, reports: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get common error patterns from reports."""
        error_counts = {}
        for report in reports:
            for error in report.get("errors", []):
                error_type = error.split(":")[0]  # Get error type before colon
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts


def create_fixed_data_collator(
    tokenizer: PreTrainedTokenizer,
    pad_to_multiple_of: Optional[int] = 8,
    max_length: Optional[int] = None,
    conservative_labeling: bool = True
) -> FixedDataCollator:
    """
    Factory function to create a fixed data collator.
    
    Args:
        tokenizer: The tokenizer to use
        pad_to_multiple_of: Pad sequences to multiple of this value
        max_length: Maximum sequence length
        conservative_labeling: Whether to use conservative labeling for stability
        
    Returns:
        FixedDataCollator instance
    """
    return FixedDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_length,
        ignore_pad_token_for_loss=True,
        conservative_labeling=conservative_labeling
    )


if __name__ == "__main__":
    print("Fixed Data Collator for Causal Language Modeling")
    print("This module provides a robust data collator that fixes zero loss issues.")