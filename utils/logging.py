"""
Logging utilities for training metrics and progress.

This module provides structured logging for training metrics, progress monitoring,
and comprehensive error handling with automatic recovery mechanisms.
"""

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from contextlib import contextmanager

import torch


@dataclass
class TrainingMetrics:
    """Structured container for training metrics."""
    timestamp: float
    step: int
    epoch: float
    train_loss: float
    learning_rate: float
    grad_norm: float
    tokens_per_second: float
    samples_per_second: float
    gpu_memory_used: float
    gpu_memory_total: float
    batch_size: int
    sequence_length: int


@dataclass
class ErrorEvent:
    """Structured container for error events."""
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[str] = None
    recovery_successful: bool = False


class StructuredLogger:
    """
    Structured logger for training metrics and error handling.
    
    Provides JSON-formatted logging for metrics, errors, and recovery actions
    with automatic file rotation and real-time monitoring capabilities.
    """
    
    def __init__(
        self,
        output_dir: str,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        max_file_size_mb: int = 100
    ):
        """
        Initialize the structured logger.
        
        Args:
            output_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_console: Whether to log to console
            enable_file: Whether to log to files
            max_file_size_mb: Maximum log file size before rotation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.max_file_size = max_file_size_mb * 1024 * 1024
        
        # Initialize loggers
        self._setup_loggers()
        
        # Metrics tracking
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        self.errors_file = self.output_dir / "errors.jsonl"
        self.recovery_file = self.output_dir / "recovery_actions.jsonl"
        
        # Thread-safe logging
        self._lock = threading.Lock()
        
        # Error tracking
        self.error_counts = {}
        self.recovery_attempts = {}
        
        self.logger.info("StructuredLogger initialized", extra={
            "output_dir": str(self.output_dir),
            "log_level": log_level,
            "console_enabled": enable_console,
            "file_enabled": enable_file
        })
    
    def _setup_loggers(self):
        """Setup main logger with appropriate handlers."""
        self.logger = logging.getLogger("tinyllama_training")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file:
            log_file = self.output_dir / "training.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """
        Log structured training metrics.
        
        Args:
            metrics: TrainingMetrics object containing current metrics
        """
        with self._lock:
            try:
                # Log to structured metrics file
                metrics_dict = asdict(metrics)
                with open(self.metrics_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics_dict, ensure_ascii=False) + '\n')
                
                # Log summary to main logger
                self.logger.info(
                    f"Step {metrics.step} | Loss: {metrics.train_loss:.4f} | "
                    f"LR: {metrics.learning_rate:.2e} | "
                    f"Tokens/s: {metrics.tokens_per_second:.0f} | "
                    f"GPU: {metrics.gpu_memory_used:.1f}/{metrics.gpu_memory_total:.1f}GB"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to log metrics: {e}")
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_action: Optional[str] = None
    ) -> str:
        """
        Log structured error information.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            recovery_action: Description of recovery action taken
            
        Returns:
            Error ID for tracking
        """
        error_id = f"error_{int(time.time())}_{id(error)}"
        
        with self._lock:
            try:
                error_event = ErrorEvent(
                    timestamp=time.time(),
                    error_type=type(error).__name__,
                    error_message=str(error),
                    stack_trace=traceback.format_exc(),
                    context=context,
                    recovery_action=recovery_action
                )
                
                # Track error counts
                error_type = error_event.error_type
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                # Log to structured errors file
                error_dict = asdict(error_event)
                error_dict['error_id'] = error_id
                with open(self.errors_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_dict, ensure_ascii=False) + '\n')
                
                # Log to main logger
                self.logger.error(
                    f"Error [{error_id}]: {error_event.error_type} - {error_event.error_message}",
                    extra={"error_id": error_id, "context": context}
                )
                
                if recovery_action:
                    self.logger.info(f"Recovery action for [{error_id}]: {recovery_action}")
                
            except Exception as e:
                # Fallback logging if structured logging fails
                self.logger.error(f"Failed to log error: {e}")
                self.logger.error(f"Original error: {error}")
        
        return error_id
    
    def log_recovery_success(self, error_id: str, details: Dict[str, Any]):
        """
        Log successful recovery from an error.
        
        Args:
            error_id: ID of the error that was recovered from
            details: Details about the recovery
        """
        with self._lock:
            try:
                recovery_event = {
                    "timestamp": time.time(),
                    "error_id": error_id,
                    "recovery_successful": True,
                    "details": details
                }
                
                with open(self.recovery_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(recovery_event, ensure_ascii=False) + '\n')
                
                self.logger.info(f"Successfully recovered from error [{error_id}]", extra=details)
                
            except Exception as e:
                self.logger.error(f"Failed to log recovery success: {e}")
    
    def log_progress(
        self,
        current_step: int,
        total_steps: int,
        elapsed_time: float,
        eta: float,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log training progress information.
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
            elapsed_time: Elapsed training time in seconds
            eta: Estimated time to completion in seconds
            additional_info: Additional progress information
        """
        progress_percent = (current_step / total_steps) * 100 if total_steps > 0 else 0
        
        progress_info = {
            "timestamp": time.time(),
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percent": progress_percent,
            "elapsed_time": elapsed_time,
            "eta": eta,
            "steps_per_second": current_step / elapsed_time if elapsed_time > 0 else 0
        }
        
        if additional_info:
            progress_info.update(additional_info)
        
        self.logger.info(
            f"Progress: {current_step}/{total_steps} ({progress_percent:.1f}%) | "
            f"Elapsed: {elapsed_time/60:.1f}min | ETA: {eta/60:.1f}min",
            extra=progress_info
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of errors encountered during training.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "recovery_attempts": dict(self.recovery_attempts)
        }
    
    @contextmanager
    def error_context(self, operation: str, **context):
        """
        Context manager for automatic error logging.
        
        Args:
            operation: Description of the operation being performed
            **context: Additional context information
        """
        context["operation"] = operation
        context["start_time"] = time.time()
        
        try:
            yield
            context["duration"] = time.time() - context["start_time"]
            self.logger.debug(f"Completed: {operation}", extra=context)
        except Exception as e:
            context["duration"] = time.time() - context["start_time"]
            error_id = self.log_error(e, context)
            raise RuntimeError(f"Operation failed [{error_id}]: {operation}") from e


class ErrorRecoveryManager:
    """
    Manager for automatic error recovery and fallback mechanisms.
    
    Handles common training errors with automatic recovery strategies
    including OOM handling, data corruption, and hardware failures.
    """
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize the error recovery manager.
        
        Args:
            logger: StructuredLogger instance for logging recovery actions
        """
        self.logger = logger
        self.recovery_strategies = {
            "OutOfMemoryError": self._handle_oom_error,
            "RuntimeError": self._handle_runtime_error,
            "FileNotFoundError": self._handle_file_not_found,
            "JSONDecodeError": self._handle_json_decode_error,
            "ConnectionError": self._handle_connection_error
        }
        
        # Recovery state tracking
        self.batch_size_reductions = 0
        self.max_batch_size_reductions = 3
        self.original_batch_size = None
        self.corrupted_files = set()
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to recover from an error using appropriate strategy.
        
        Args:
            error: Exception that occurred
            context: Context information about the error
            recovery_config: Configuration for recovery attempts
            
        Returns:
            True if recovery was successful, False otherwise
        """
        error_type = type(error).__name__
        error_id = self.logger.log_error(error, context)
        
        # Check if we have a recovery strategy
        if error_type not in self.recovery_strategies:
            self.logger.logger.warning(f"No recovery strategy for {error_type}")
            return False
        
        try:
            recovery_strategy = self.recovery_strategies[error_type]
            recovery_successful = recovery_strategy(error, context, recovery_config or {})
            
            if recovery_successful:
                self.logger.log_recovery_success(error_id, {
                    "strategy": recovery_strategy.__name__,
                    "context": context
                })
            
            return recovery_successful
            
        except Exception as recovery_error:
            self.logger.log_error(recovery_error, {
                "original_error": str(error),
                "recovery_strategy": recovery_strategy.__name__,
                "context": context
            })
            return False
    
    def _handle_oom_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_config: Dict[str, Any]
    ) -> bool:
        """Handle Out of Memory errors with automatic batch size reduction."""
        if self.batch_size_reductions >= self.max_batch_size_reductions:
            self.logger.logger.error("Maximum batch size reductions reached, cannot recover from OOM")
            return False
        
        current_batch_size = context.get("batch_size", 1)
        if self.original_batch_size is None:
            self.original_batch_size = current_batch_size
        
        # Reduce batch size by half
        new_batch_size = max(1, current_batch_size // 2)
        self.batch_size_reductions += 1
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        recovery_action = f"Reduced batch size from {current_batch_size} to {new_batch_size}"
        self.logger.logger.info(recovery_action)
        
        # Update context with new batch size
        context["batch_size"] = new_batch_size
        context["recovery_action"] = recovery_action
        
        return True
    
    def _handle_runtime_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_config: Dict[str, Any]
    ) -> bool:
        """Handle runtime errors with context-specific recovery."""
        error_message = str(error).lower()
        
        # Handle CUDA out of memory specifically
        if "out of memory" in error_message or "cuda" in error_message:
            return self._handle_oom_error(error, context, recovery_config)
        
        # Handle model loading errors
        if "model" in error_message and "load" in error_message:
            recovery_action = "Attempting to reload model with fallback configuration"
            self.logger.logger.info(recovery_action)
            context["use_fallback_model"] = True
            return True
        
        return False
    
    def _handle_file_not_found(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_config: Dict[str, Any]
    ) -> bool:
        """Handle file not found errors with fallback paths."""
        missing_file = str(error)
        
        # Check if it's a dataset file
        if "dataset" in missing_file.lower() or ".jsonl" in missing_file:
            recovery_action = "Skipping corrupted or missing dataset file"
            self.logger.logger.warning(recovery_action)
            self.corrupted_files.add(missing_file)
            context["skip_file"] = True
            return True
        
        # Check if it's a tokenizer file
        if "tokenizer" in missing_file.lower():
            recovery_action = "Using fallback tokenizer configuration"
            self.logger.logger.warning(recovery_action)
            context["use_fallback_tokenizer"] = True
            return True
        
        return False
    
    def _handle_json_decode_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_config: Dict[str, Any]
    ) -> bool:
        """Handle JSON decode errors in dataset files."""
        file_path = context.get("file_path", "unknown")
        line_number = context.get("line_number", "unknown")
        
        recovery_action = f"Skipping corrupted JSON line {line_number} in {file_path}"
        self.logger.logger.warning(recovery_action)
        
        context["skip_line"] = True
        context["recovery_action"] = recovery_action
        
        return True
    
    def _handle_connection_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_config: Dict[str, Any]
    ) -> bool:
        """Handle connection errors with retry logic."""
        max_retries = recovery_config.get("max_retries", 3)
        current_retry = context.get("retry_count", 0)
        
        if current_retry >= max_retries:
            self.logger.logger.error(f"Maximum retries ({max_retries}) reached for connection error")
            return False
        
        retry_delay = min(2 ** current_retry, 30)  # Exponential backoff, max 30s
        recovery_action = f"Retrying connection after {retry_delay}s (attempt {current_retry + 1}/{max_retries})"
        self.logger.logger.info(recovery_action)
        
        time.sleep(retry_delay)
        context["retry_count"] = current_retry + 1
        
        return True
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts."""
        return {
            "batch_size_reductions": self.batch_size_reductions,
            "original_batch_size": self.original_batch_size,
            "corrupted_files_count": len(self.corrupted_files),
            "corrupted_files": list(self.corrupted_files)
        }


def setup_comprehensive_logging(
    output_dir: str,
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True
) -> tuple[StructuredLogger, ErrorRecoveryManager]:
    """
    Setup comprehensive logging and error recovery system.
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        
    Returns:
        Tuple of (StructuredLogger, ErrorRecoveryManager)
    """
    # Create structured logger
    structured_logger = StructuredLogger(
        output_dir=output_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file
    )
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(structured_logger)
    
    structured_logger.logger.info("Comprehensive logging and error recovery system initialized")
    
    return structured_logger, recovery_manager