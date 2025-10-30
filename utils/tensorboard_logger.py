#!/usr/bin/env python3
"""
TensorBoard Logger for TinyLlama Tigrinya Training

This module provides a comprehensive TensorBoard logging system to replace wandb
dependencies with local tensorboard logging for training metrics and monitoring.
"""

import logging
import math
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorboardLogger:
    """
    TensorBoard logger for training metrics and monitoring.
    
    Provides comprehensive logging capabilities including:
    - Training metrics (loss, learning rate, etc.)
    - Gradient statistics and monitoring
    - Model parameter tracking
    - Hardware utilization metrics
    """
    
    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        flush_secs: int = 30,
        max_queue: int = 10
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for tensorboard logs
            flush_secs: How often to flush logs to disk
            max_queue: Maximum number of events to queue
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SummaryWriter
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
            max_queue=max_queue
        )
        
        # Track logging state
        self.step_count = 0
        self.is_closed = False
        
        logger.info(f"TensorboardLogger initialized with log_dir: {self.log_dir}")
        logger.info(f"View logs with: tensorboard --logdir {self.log_dir}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log training metrics to tensorboard.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (uses internal counter if None)
        """
        if self.is_closed:
            logger.warning("Attempted to log to closed TensorboardLogger")
            return
        
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        for key, value in metrics.items():
            if self._is_valid_metric(key, value):
                self.writer.add_scalar(key, value, step)
            else:
                logger.warning(f"Skipping invalid metric {key}: {value}")
    
    def log_gradients(self, model: nn.Module, step: Optional[int] = None) -> Dict[str, float]:
        """
        Log gradient statistics for model parameters.
        
        Args:
            model: PyTorch model to analyze
            step: Training step (uses internal counter if None)
            
        Returns:
            Dictionary with gradient statistics
        """
        if self.is_closed:
            logger.warning("Attempted to log to closed TensorboardLogger")
            return {}
        
        if step is None:
            step = self.step_count
        
        total_norm = 0.0
        param_count = 0
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Calculate parameter gradient norm
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Log individual layer gradients (sample a few key layers)
                if any(key in name for key in ['embed', 'lm_head', 'layer.0', 'layer.11']):
                    self.writer.add_scalar(f"gradients/layers/{name}", param_norm, step)
                
                # Collect statistics
                gradient_stats[name] = param_norm
        
        # Calculate total gradient norm
        total_norm = total_norm ** 0.5 if param_count > 0 else 0.0
        
        # Log aggregate gradient statistics
        if param_count > 0:
            self.writer.add_scalar("gradients/total_norm", total_norm, step)
            self.writer.add_scalar("gradients/param_count", param_count, step)
            
            # Log gradient distribution statistics
            grad_values = list(gradient_stats.values())
            if grad_values:
                self.writer.add_scalar("gradients/mean", sum(grad_values) / len(grad_values), step)
                self.writer.add_scalar("gradients/max", max(grad_values), step)
                self.writer.add_scalar("gradients/min", min(grad_values), step)
        
        return {
            "total_norm": total_norm,
            "param_count": param_count,
            "mean_grad": sum(gradient_stats.values()) / len(gradient_stats) if gradient_stats else 0.0
        }
    
    def log_model_weights(self, model: nn.Module, step: Optional[int] = None) -> None:
        """
        Log model weight statistics.
        
        Args:
            model: PyTorch model to analyze
            step: Training step (uses internal counter if None)
        """
        if self.is_closed:
            logger.warning("Attempted to log to closed TensorboardLogger")
            return
        
        if step is None:
            step = self.step_count
        
        for name, param in model.named_parameters():
            if param.data is not None:
                # Log weight statistics for key layers
                if any(key in name for key in ['embed', 'lm_head', 'layer.0', 'layer.11']):
                    weight_norm = param.data.norm(2).item()
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    
                    self.writer.add_scalar(f"weights/{name}/norm", weight_norm, step)
                    self.writer.add_scalar(f"weights/{name}/mean", weight_mean, step)
                    self.writer.add_scalar(f"weights/{name}/std", weight_std, step)
    
    def log_learning_rate(self, lr: float, step: Optional[int] = None) -> None:
        """
        Log learning rate.
        
        Args:
            lr: Current learning rate
            step: Training step (uses internal counter if None)
        """
        if step is None:
            step = self.step_count
        
        self.log_metrics({"learning_rate": lr}, step)
    
    def log_loss(self, loss: float, loss_type: str = "train_loss", step: Optional[int] = None) -> None:
        """
        Log training or validation loss.
        
        Args:
            loss: Loss value
            loss_type: Type of loss (train_loss, eval_loss, etc.)
            step: Training step (uses internal counter if None)
        """
        if step is None:
            step = self.step_count
        
        self.log_metrics({loss_type: loss}, step)
    
    def log_hardware_metrics(self, step: Optional[int] = None) -> None:
        """
        Log hardware utilization metrics if available.
        
        Args:
            step: Training step (uses internal counter if None)
        """
        if self.is_closed:
            return
        
        if step is None:
            step = self.step_count
        
        try:
            # Log GPU memory usage if CUDA is available
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                
                self.writer.add_scalar("hardware/gpu_memory_allocated_gb", gpu_memory_allocated, step)
                self.writer.add_scalar("hardware/gpu_memory_reserved_gb", gpu_memory_reserved, step)
                
                # Log GPU utilization if nvidia-ml-py is available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.writer.add_scalar("hardware/gpu_utilization_percent", gpu_util.gpu, step)
                except ImportError:
                    pass  # nvidia-ml-py not available
                except Exception as e:
                    logger.debug(f"Could not get GPU utilization: {e}")
        
        except Exception as e:
            logger.debug(f"Could not log hardware metrics: {e}")
    
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        step: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log a complete training step with all standard metrics.
        
        Args:
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            step: Training step
            additional_metrics: Additional metrics to log
        """
        # Log core metrics
        metrics = {
            "train_loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log all metrics
        self.log_metrics(metrics, step)
        
        # Log hardware metrics periodically
        if step % 50 == 0:  # Every 50 steps
            self.log_hardware_metrics(step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text data (e.g., generated samples).
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Training step (uses internal counter if None)
        """
        if self.is_closed:
            return
        
        if step is None:
            step = self.step_count
        
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log hyperparameters and optionally final metrics.
        
        Args:
            hparams: Hyperparameter dictionary
            metrics: Final metrics dictionary
        """
        if self.is_closed:
            return
        
        # Convert all values to appropriate types for tensorboard
        clean_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)
        
        self.writer.add_hparams(clean_hparams, metrics or {})
    
    def flush(self) -> None:
        """Flush pending logs to disk."""
        if not self.is_closed:
            self.writer.flush()
    
    def close(self) -> None:
        """Close the tensorboard logger and cleanup resources."""
        if not self.is_closed:
            self.writer.flush()
            self.writer.close()
            self.is_closed = True
            logger.info("TensorboardLogger closed")
    
    def _is_valid_metric(self, key: str, value: Any) -> bool:
        """
        Check if a metric value is valid for logging.
        
        Args:
            key: Metric name
            value: Metric value
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, (int, float)):
            return False
        
        if math.isnan(value) or math.isinf(value):
            return False
        
        return True
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_tensorboard_logger(
    output_dir: str,
    experiment_name: Optional[str] = None
) -> TensorboardLogger:
    """
    Create a tensorboard logger with standard configuration.
    
    Args:
        output_dir: Base output directory
        experiment_name: Optional experiment name for subdirectory
        
    Returns:
        Configured TensorboardLogger instance
    """
    if experiment_name:
        log_dir = os.path.join(output_dir, "tensorboard", experiment_name)
    else:
        log_dir = os.path.join(output_dir, "tensorboard")
    
    return TensorboardLogger(log_dir=log_dir)


if __name__ == "__main__":
    # Example usage
    logger = create_tensorboard_logger("test_output")
    
    # Log some example metrics
    logger.log_metrics({
        "train_loss": 2.5,
        "learning_rate": 1e-4,
        "grad_norm": 0.8
    }, step=1)
    
    logger.close()
    print("TensorBoard logger test completed")