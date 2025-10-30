"""
Gradient Stability System for Training Stability

This module provides comprehensive gradient stability management including:
- NaN gradient detection and recovery
- Gradient clipping and norm computation
- Proper weight initialization
- Mixed precision management
- Hardware-specific stability configurations
"""

import torch
import torch.nn as nn
import logging
import math
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GradientStats:
    """Statistics for gradient monitoring"""
    total_norm: float
    max_norm: float
    min_norm: float
    nan_count: int
    inf_count: int
    zero_count: int
    param_count: int
    
    def is_healthy(self) -> bool:
        """Check if gradients are in a healthy state"""
        return (
            self.nan_count == 0 and
            self.inf_count == 0 and
            math.isfinite(self.total_norm) and
            self.total_norm > 0.0
        )


class GradientStabilizer:
    """
    Comprehensive gradient stability management system
    
    Handles NaN detection, gradient clipping, and recovery mechanisms
    to ensure stable training across different hardware configurations.
    """
    
    def __init__(self, max_grad_norm: float = 1.0, nan_recovery_enabled: bool = True):
        self.max_grad_norm = max_grad_norm
        self.nan_recovery_enabled = nan_recovery_enabled
        self.nan_recovery_count = 0
        self.max_nan_recoveries = 5
        
        # Statistics tracking
        self.gradient_history = []
        self.recovery_history = []
        
        logging.info(f"GradientStabilizer initialized with max_grad_norm={max_grad_norm}")
    
    def compute_gradient_stats(self, model: nn.Module) -> GradientStats:
        """
        Compute comprehensive gradient statistics
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            GradientStats object with detailed gradient information
        """
        total_norm = 0.0
        max_norm = 0.0
        min_norm = float('inf')
        nan_count = 0
        inf_count = 0
        zero_count = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_count += 1
                grad_data = param.grad.data
                
                # Check for NaN values
                if torch.isnan(grad_data).any():
                    nan_count += 1
                    logging.warning(f"NaN gradient detected in parameter: {name}")
                    continue
                
                # Check for infinite values
                if torch.isinf(grad_data).any():
                    inf_count += 1
                    logging.warning(f"Infinite gradient detected in parameter: {name}")
                    continue
                
                # Compute parameter norm
                param_norm = grad_data.norm(2).item()
                
                if param_norm == 0.0:
                    zero_count += 1
                else:
                    total_norm += param_norm ** 2
                    max_norm = max(max_norm, param_norm)
                    min_norm = min(min_norm, param_norm)
        
        # Compute total norm
        if total_norm > 0:
            total_norm = total_norm ** 0.5
        
        if min_norm == float('inf'):
            min_norm = 0.0
        
        return GradientStats(
            total_norm=total_norm,
            max_norm=max_norm,
            min_norm=min_norm,
            nan_count=nan_count,
            inf_count=inf_count,
            zero_count=zero_count,
            param_count=param_count
        )
    
    def clip_gradients(self, model: nn.Module) -> Tuple[float, bool]:
        """
        Clip gradients and handle NaN/inf values
        
        Args:
            model: PyTorch model to clip gradients for
            
        Returns:
            Tuple of (gradient_norm, is_healthy)
        """
        # First, compute gradient statistics
        grad_stats = self.compute_gradient_stats(model)
        
        # Handle NaN or infinite gradients
        if not grad_stats.is_healthy():
            if self.nan_recovery_enabled:
                recovery_success = self._recover_from_nan_gradients(model)
                if not recovery_success:
                    return float('nan'), False
            else:
                logging.error("Unhealthy gradients detected but recovery is disabled")
                return float('nan'), False
        
        # Perform gradient clipping
        if grad_stats.total_norm > 0:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.max_grad_norm
            ).item()
        else:
            clipped_norm = 0.0
        
        # Log gradient statistics
        self._log_gradient_stats(grad_stats, clipped_norm)
        
        # Store statistics for monitoring
        self.gradient_history.append(grad_stats)
        if len(self.gradient_history) > 1000:  # Keep last 1000 entries
            self.gradient_history.pop(0)
        
        return clipped_norm, True
    
    def _recover_from_nan_gradients(self, model: nn.Module) -> bool:
        """
        Attempt to recover from NaN gradients
        
        Args:
            model: PyTorch model with NaN gradients
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if self.nan_recovery_count >= self.max_nan_recoveries:
            logging.error(f"Maximum NaN recoveries ({self.max_nan_recoveries}) exceeded")
            return False
        
        self.nan_recovery_count += 1
        logging.warning(f"Attempting NaN gradient recovery #{self.nan_recovery_count}")
        
        # Strategy 1: Zero out NaN gradients
        nan_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_params.append(name)
                    param.grad.zero_()
        
        if nan_params:
            logging.info(f"Zeroed gradients for parameters: {nan_params}")
        
        # Strategy 2: Reinitialize problematic parameters if too many NaN recoveries
        if self.nan_recovery_count > 2:
            self._reinitialize_unstable_parameters(model, nan_params)
        
        # Record recovery attempt
        self.recovery_history.append({
            'step': self.nan_recovery_count,
            'nan_params': nan_params,
            'strategy': 'zero_gradients'
        })
        
        return True
    
    def _reinitialize_unstable_parameters(self, model: nn.Module, nan_params: list):
        """
        Reinitialize parameters that consistently produce NaN gradients
        
        Args:
            model: PyTorch model
            nan_params: List of parameter names with NaN gradients
        """
        logging.warning("Reinitializing unstable parameters due to repeated NaN gradients")
        
        for name, param in model.named_parameters():
            if name in nan_params:
                logging.info(f"Reinitializing parameter: {name}")
                
                if len(param.shape) >= 2:
                    # Linear/Conv layers
                    nn.init.xavier_uniform_(param.data)
                elif len(param.shape) == 1:
                    # Bias terms or layer norm
                    nn.init.zeros_(param.data)
                else:
                    # Scalar parameters
                    nn.init.constant_(param.data, 0.01)
    
    def _log_gradient_stats(self, grad_stats: GradientStats, clipped_norm: float):
        """Log gradient statistics for monitoring"""
        if grad_stats.param_count > 0:
            logging.info(
                f"Gradient Stats - Total Norm: {grad_stats.total_norm:.6f}, "
                f"Clipped Norm: {clipped_norm:.6f}, "
                f"Max: {grad_stats.max_norm:.6f}, Min: {grad_stats.min_norm:.6f}, "
                f"NaN: {grad_stats.nan_count}, Inf: {grad_stats.inf_count}, "
                f"Zero: {grad_stats.zero_count}/{grad_stats.param_count}"
            )
    
    def initialize_model_weights(self, model: nn.Module):
        """
        Initialize model weights for numerical stability
        
        Args:
            model: PyTorch model to initialize
        """
        logging.info("Initializing model weights for numerical stability")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                logging.debug(f"Initialized Linear layer: {name}")
                
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                logging.debug(f"Initialized Embedding layer: {name}")
                
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for layer norm
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                logging.debug(f"Initialized LayerNorm layer: {name}")
                
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # Kaiming initialization for conv layers
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                logging.debug(f"Initialized Conv layer: {name}")
    
    def reset_recovery_count(self):
        """Reset the NaN recovery counter (call after successful training steps)"""
        if self.nan_recovery_count > 0:
            logging.info(f"Resetting NaN recovery count from {self.nan_recovery_count} to 0")
            self.nan_recovery_count = 0
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive stability report
        
        Returns:
            Dictionary with stability metrics and recommendations
        """
        if not self.gradient_history:
            return {"status": "no_data", "message": "No gradient history available"}
        
        recent_stats = self.gradient_history[-10:]  # Last 10 gradient computations
        
        # Compute averages
        avg_norm = sum(s.total_norm for s in recent_stats if math.isfinite(s.total_norm)) / len(recent_stats)
        total_nan_count = sum(s.nan_count for s in recent_stats)
        total_inf_count = sum(s.inf_count for s in recent_stats)
        
        # Determine stability status
        if total_nan_count > 0 or total_inf_count > 0:
            status = "unstable"
        elif avg_norm > self.max_grad_norm * 2:
            status = "high_gradients"
        elif avg_norm < 1e-8:
            status = "vanishing_gradients"
        else:
            status = "stable"
        
        return {
            "status": status,
            "avg_gradient_norm": avg_norm,
            "nan_recovery_count": self.nan_recovery_count,
            "total_nan_gradients": total_nan_count,
            "total_inf_gradients": total_inf_count,
            "max_grad_norm": self.max_grad_norm,
            "recovery_history": self.recovery_history[-5:],  # Last 5 recoveries
            "recommendations": self._get_stability_recommendations(status, avg_norm)
        }
    
    def _get_stability_recommendations(self, status: str, avg_norm: float) -> list:
        """Generate stability recommendations based on current status"""
        recommendations = []
        
        if status == "unstable":
            recommendations.extend([
                "Reduce learning rate by 50%",
                "Enable gradient clipping with lower threshold",
                "Check data preprocessing for invalid values",
                "Consider using mixed precision training"
            ])
        elif status == "high_gradients":
            recommendations.extend([
                f"Reduce max_grad_norm from {self.max_grad_norm} to {self.max_grad_norm * 0.5}",
                "Reduce learning rate",
                "Increase gradient accumulation steps"
            ])
        elif status == "vanishing_gradients":
            recommendations.extend([
                "Increase learning rate carefully",
                "Check model architecture for bottlenecks",
                "Verify weight initialization",
                "Consider residual connections"
            ])
        else:
            recommendations.append("Training appears stable - continue monitoring")
        
        return recommendations


class MixedPrecisionManager:
    """
    Manages mixed precision training configurations for different hardware
    """
    
    def __init__(self, gpu_type: str = "auto"):
        self.gpu_type = self._detect_gpu_type() if gpu_type == "auto" else gpu_type.lower()
        self.scaler = None
        
        logging.info(f"MixedPrecisionManager initialized for GPU type: {self.gpu_type}")
    
    def _detect_gpu_type(self) -> str:
        """Detect GPU type automatically"""
        if not torch.cuda.is_available():
            return "cpu"
        
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        if "h100" in gpu_name:
            return "h100"
        elif "a100" in gpu_name:
            return "a100"
        elif "l4" in gpu_name:
            return "l4"
        elif "rtx" in gpu_name and "4050" in gpu_name:
            return "rtx_4050"
        elif "rtx" in gpu_name:
            return "rtx_general"
        else:
            return "unknown"
    
    def get_precision_config(self) -> Dict[str, Any]:
        """
        Get stable mixed precision configuration by GPU type
        
        Returns:
            Dictionary with precision configuration parameters
        """
        configs = {
            "rtx_4050": {
                "fp16": True,
                "bf16": False,
                "fp16_opt_level": "O1",  # Conservative
                "loss_scale": 128.0,     # Lower scale for stability
                "gradient_clipping": True,
                "max_grad_norm": 0.5,
            },
            "rtx_general": {
                "fp16": True,
                "bf16": False,
                "fp16_opt_level": "O1",
                "loss_scale": 256.0,
                "gradient_clipping": True,
                "max_grad_norm": 0.8,
            },
            "h100": {
                "fp16": False,
                "bf16": True,            # BF16 more stable on H100
                "fp16_opt_level": "O2",
                "loss_scale": None,      # Auto scaling
                "gradient_clipping": True,
                "max_grad_norm": 1.0,
            },
            "a100": {
                "fp16": False,
                "bf16": True,
                "fp16_opt_level": "O2",
                "loss_scale": None,
                "gradient_clipping": True,
                "max_grad_norm": 1.0,
            },
            "l4": {
                "fp16": True,
                "bf16": False,
                "fp16_opt_level": "O1",
                "loss_scale": 256.0,
                "gradient_clipping": True,
                "max_grad_norm": 0.8,
            },
            "cpu": {
                "fp16": False,
                "bf16": False,
                "fp16_opt_level": "O0",
                "loss_scale": None,
                "gradient_clipping": True,
                "max_grad_norm": 1.0,
            },
            "unknown": {
                "fp16": True,
                "bf16": False,
                "fp16_opt_level": "O1",  # Conservative default
                "loss_scale": 128.0,
                "gradient_clipping": True,
                "max_grad_norm": 0.5,
            }
        }
        
        config = configs.get(self.gpu_type, configs["unknown"])
        logging.info(f"Using precision config for {self.gpu_type}: {config}")
        
        return config
    
    def setup_scaler(self, enabled: bool = True, init_scale: float = None) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Setup gradient scaler for mixed precision training
        
        Args:
            enabled: Whether to enable gradient scaling
            init_scale: Initial scale value (uses config default if None)
            
        Returns:
            GradScaler instance or None if not enabled
        """
        if not enabled or not torch.cuda.is_available():
            return None
        
        config = self.get_precision_config()
        
        if config.get("loss_scale") is None:
            # Auto scaling
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            # Fixed or configured scaling
            scale = init_scale if init_scale is not None else config["loss_scale"]
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=scale,
                growth_factor=1.1,       # Slower growth for stability
                backoff_factor=0.8,      # Faster backoff on overflow
                growth_interval=1000     # Less frequent scaling updates
            )
        
        logging.info(f"Gradient scaler initialized with init_scale={self.scaler.get_scale()}")
        return self.scaler
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss if scaler is available"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Step optimizer with gradient scaling
        
        Returns:
            True if optimizer step was taken, False if skipped due to overflow
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Check if step was skipped due to overflow
            return self.scaler.get_scale() > 0
        else:
            optimizer.step()
            return True


# Convenience function for easy integration
def create_gradient_stabilizer(max_grad_norm: float = 1.0, 
                             gpu_type: str = "auto",
                             enable_mixed_precision: bool = True) -> Tuple[GradientStabilizer, MixedPrecisionManager]:
    """
    Create a complete gradient stability system
    
    Args:
        max_grad_norm: Maximum gradient norm for clipping
        gpu_type: GPU type for hardware-specific optimizations
        enable_mixed_precision: Whether to enable mixed precision training
        
    Returns:
        Tuple of (GradientStabilizer, MixedPrecisionManager)
    """
    # Create gradient stabilizer
    stabilizer = GradientStabilizer(max_grad_norm=max_grad_norm)
    
    # Create mixed precision manager
    precision_manager = MixedPrecisionManager(gpu_type=gpu_type)
    
    # Configure gradient stabilizer based on precision config
    if enable_mixed_precision:
        precision_config = precision_manager.get_precision_config()
        stabilizer.max_grad_norm = precision_config.get("max_grad_norm", max_grad_norm)
    
    return stabilizer, precision_manager