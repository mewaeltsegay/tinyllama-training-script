"""
Mixed Precision Manager for hardware-specific training configurations.

This module provides GPU-specific mixed precision settings with conservative scaling
parameters to ensure training stability across different hardware configurations.
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training on specific hardware."""
    fp16: bool
    bf16: bool
    fp16_opt_level: str
    loss_scale: Optional[float]
    gradient_scaler_enabled: bool
    scaler_init_scale: float
    scaler_growth_factor: float
    scaler_backoff_factor: float
    scaler_growth_interval: int
    dataloader_pin_memory: bool
    torch_compile: bool


class MixedPrecisionManager:
    """
    Manages hardware-specific mixed precision settings for stable training.
    
    Provides GPU-specific configurations with conservative scaling parameters
    to prevent gradient overflow and ensure numerical stability.
    """
    
    # Hardware-specific mixed precision configurations
    PRECISION_CONFIGS = {
        "rtx_4050": MixedPrecisionConfig(
            fp16=True,
            bf16=False,
            fp16_opt_level="O1",        # Conservative mixed precision
            loss_scale=128.0,           # Lower scale for stability
            gradient_scaler_enabled=True,
            scaler_init_scale=128.0,    # Conservative initial scale
            scaler_growth_factor=1.1,   # Slower growth
            scaler_backoff_factor=0.8,  # Faster backoff on overflow
            scaler_growth_interval=1000, # Less frequent scaling updates
            dataloader_pin_memory=True,
            torch_compile=False         # Disable for older hardware
        ),
        "h100": MixedPrecisionConfig(
            fp16=False,
            bf16=True,                  # BF16 more stable on H100
            fp16_opt_level="O2",        # Aggressive mixed precision
            loss_scale=None,            # Auto scaling with BF16
            gradient_scaler_enabled=False, # Not needed with BF16
            scaler_init_scale=65536.0,  # Higher scale for BF16
            scaler_growth_factor=2.0,   # Standard growth
            scaler_backoff_factor=0.5,  # Standard backoff
            scaler_growth_interval=2000, # Standard interval
            dataloader_pin_memory=True,
            torch_compile=True          # Enable for modern hardware
        ),
        "a100": MixedPrecisionConfig(
            fp16=False,
            bf16=True,                  # BF16 preferred on A100
            fp16_opt_level="O2",
            loss_scale=None,            # Auto scaling with BF16
            gradient_scaler_enabled=False, # Not needed with BF16
            scaler_init_scale=65536.0,
            scaler_growth_factor=2.0,
            scaler_backoff_factor=0.5,
            scaler_growth_interval=2000,
            dataloader_pin_memory=True,
            torch_compile=True
        ),
        "l4": MixedPrecisionConfig(
            fp16=True,
            bf16=False,                 # FP16 for L4
            fp16_opt_level="O1",        # Conservative for stability
            loss_scale=256.0,           # Moderate scale
            gradient_scaler_enabled=True,
            scaler_init_scale=256.0,    # Moderate initial scale
            scaler_growth_factor=1.2,   # Moderate growth
            scaler_backoff_factor=0.75, # Moderate backoff
            scaler_growth_interval=1500, # Moderate interval
            dataloader_pin_memory=True,
            torch_compile=False         # Conservative for L4
        ),
        "default": MixedPrecisionConfig(
            fp16=True,
            bf16=False,
            fp16_opt_level="O1",        # Most conservative
            loss_scale=64.0,            # Very conservative scale
            gradient_scaler_enabled=True,
            scaler_init_scale=64.0,     # Very conservative
            scaler_growth_factor=1.05,  # Very slow growth
            scaler_backoff_factor=0.9,  # Gentle backoff
            scaler_growth_interval=2000, # Infrequent updates
            dataloader_pin_memory=True,
            torch_compile=False         # Disable by default
        )
    }
    
    def __init__(self, gpu_type: Optional[str] = None):
        """
        Initialize MixedPrecisionManager for specific GPU type.
        
        Args:
            gpu_type: GPU type identifier (rtx_4050, h100, a100, l4)
        """
        self.gpu_type = self._normalize_gpu_type(gpu_type) if gpu_type else "default"
        self.config = self.PRECISION_CONFIGS.get(self.gpu_type, self.PRECISION_CONFIGS["default"])
        self.scaler = None
        
        logger.info(f"Initialized MixedPrecisionManager for GPU type: {self.gpu_type}")
        self._log_config()
    
    def _normalize_gpu_type(self, gpu_type: str) -> str:
        """
        Normalize GPU type string to match configuration keys.
        
        Args:
            gpu_type: Raw GPU type string
            
        Returns:
            Normalized GPU type key
        """
        gpu_type_lower = gpu_type.lower().replace(" ", "_").replace("-", "_")
        
        # Map common GPU names to configuration keys
        gpu_mappings = {
            "rtx_4050": "rtx_4050",
            "rtx4050": "rtx_4050",
            "geforce_rtx_4050": "rtx_4050",
            "h100": "h100",
            "tesla_h100": "h100",
            "h100_sxm5": "h100",
            "a100": "a100",
            "tesla_a100": "a100",
            "a100_sxm4": "a100",
            "l4": "l4",
            "tesla_l4": "l4",
            "l4_24gb": "l4"
        }
        
        # Check for exact matches first
        if gpu_type_lower in gpu_mappings:
            return gpu_mappings[gpu_type_lower]
        
        # Check for partial matches
        for pattern, config_key in gpu_mappings.items():
            if pattern in gpu_type_lower or gpu_type_lower in pattern:
                return config_key
        
        logger.warning(f"Unknown GPU type '{gpu_type}', using default configuration")
        return "default"
    
    def get_precision_config(self) -> Dict[str, Any]:
        """
        Get mixed precision configuration for the current GPU type.
        
        Returns:
            Dictionary containing mixed precision settings
        """
        return {
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            "fp16_opt_level": self.config.fp16_opt_level,
            "loss_scale": self.config.loss_scale,
            "dataloader_pin_memory": self.config.dataloader_pin_memory,
            "torch_compile": self.config.torch_compile,
            "gradient_scaler_config": {
                "enabled": self.config.gradient_scaler_enabled,
                "init_scale": self.config.scaler_init_scale,
                "growth_factor": self.config.scaler_growth_factor,
                "backoff_factor": self.config.scaler_backoff_factor,
                "growth_interval": self.config.scaler_growth_interval
            }
        }
    
    def setup_scaler(self, enabled: Optional[bool] = None) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Setup gradient scaler for mixed precision training.
        
        Args:
            enabled: Override scaler enabled setting
            
        Returns:
            GradScaler instance or None if disabled
        """
        scaler_enabled = enabled if enabled is not None else self.config.gradient_scaler_enabled
        
        if not scaler_enabled or not torch.cuda.is_available():
            logger.info("Gradient scaler disabled or CUDA not available")
            self.scaler = None
            return None
        
        # Don't use scaler with BF16 as it's not needed
        if self.config.bf16:
            logger.info("BF16 enabled, gradient scaler not needed")
            self.scaler = None
            return None
        
        try:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.scaler_init_scale,
                growth_factor=self.config.scaler_growth_factor,
                backoff_factor=self.config.scaler_backoff_factor,
                growth_interval=self.config.scaler_growth_interval,
                enabled=True
            )
            
            logger.info(f"Gradient scaler initialized with scale: {self.config.scaler_init_scale}")
            return self.scaler
            
        except Exception as e:
            logger.error(f"Failed to initialize gradient scaler: {e}")
            self.scaler = None
            return None
    
    def update_training_arguments(self, training_args: TrainingArguments) -> TrainingArguments:
        """
        Update TrainingArguments with hardware-appropriate precision settings.
        
        Args:
            training_args: Original training arguments
            
        Returns:
            Updated training arguments with mixed precision settings
        """
        try:
            # Get precision config
            precision_config = self.get_precision_config()
            
            # Update mixed precision settings
            training_args.fp16 = precision_config["fp16"]
            training_args.bf16 = precision_config["bf16"]
            training_args.dataloader_pin_memory = precision_config["dataloader_pin_memory"]
            
            # Ensure prefetch_factor is None when num_workers=0 to avoid DataLoader error
            if hasattr(training_args, 'dataloader_num_workers') and training_args.dataloader_num_workers == 0:
                training_args.dataloader_prefetch_factor = None
            
            # Set loss scale if using FP16
            if precision_config["fp16"] and precision_config["loss_scale"] is not None:
                training_args.fp16_opt_level = precision_config["fp16_opt_level"]
                # Note: loss_scale is handled by the scaler, not training_args
            
            # Enable torch compile if supported
            if hasattr(training_args, 'torch_compile') and precision_config["torch_compile"]:
                training_args.torch_compile = True
            
            logger.info("Updated training arguments with mixed precision settings")
            self._log_training_args_update(training_args)
            
            return training_args
            
        except Exception as e:
            logger.error(f"Failed to update training arguments: {e}")
            return training_args
    
    def get_autocast_context(self) -> torch.cuda.amp.autocast:
        """
        Get autocast context manager for mixed precision training.
        
        Returns:
            Autocast context manager with appropriate dtype
        """
        if not torch.cuda.is_available():
            # Return CPU autocast for CPU training
            return torch.cpu.amp.autocast(enabled=False)
        
        if self.config.bf16:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True)
        elif self.config.fp16:
            return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
        else:
            return torch.cuda.amp.autocast(enabled=False)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.
        
        Args:
            loss: Unscaled loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer, 
                      model: torch.nn.Module) -> bool:
        """
        Step optimizer with gradient scaling if enabled.
        
        Args:
            optimizer: Optimizer to step
            model: Model for gradient clipping
            
        Returns:
            True if optimizer step was successful, False if skipped due to overflow
        """
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Check if step was skipped due to overflow
            return not self.scaler._found_inf_per_device
        else:
            # Standard optimizer step without scaling
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return True
    
    def get_scaler_state(self) -> Dict[str, Any]:
        """
        Get current state of the gradient scaler.
        
        Returns:
            Dictionary with scaler state information
        """
        if self.scaler is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "scale": self.scaler.get_scale(),
            "growth_tracker": self.scaler._growth_tracker,
            "found_inf": bool(self.scaler._found_inf_per_device)
        }
    
    def validate_precision_support(self) -> Dict[str, bool]:
        """
        Validate mixed precision support on current hardware.
        
        Returns:
            Dictionary indicating support for different precision types
        """
        support = {
            "fp16": False,
            "bf16": False,
            "autocast": False,
            "grad_scaler": False
        }
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, mixed precision not supported")
            return support
        
        try:
            # Check FP16 support
            if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                support["fp16"] = True
                support["autocast"] = True
                support["grad_scaler"] = True
            
            # Check BF16 support (requires newer GPUs)
            if hasattr(torch, 'bfloat16'):
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                # BF16 supported on Ampere (8.x) and newer
                if props.major >= 8:
                    support["bf16"] = True
                    logger.info(f"BF16 supported on {props.name} (compute {props.major}.{props.minor})")
                else:
                    logger.info(f"BF16 not supported on {props.name} (compute {props.major}.{props.minor})")
            
        except Exception as e:
            logger.error(f"Error checking precision support: {e}")
        
        return support
    
    def _log_config(self):
        """Log current mixed precision configuration."""
        logger.info(f"Mixed Precision Configuration for {self.gpu_type}:")
        logger.info(f"  FP16: {self.config.fp16}")
        logger.info(f"  BF16: {self.config.bf16}")
        logger.info(f"  Optimization Level: {self.config.fp16_opt_level}")
        logger.info(f"  Loss Scale: {self.config.loss_scale}")
        logger.info(f"  Gradient Scaler: {self.config.gradient_scaler_enabled}")
        if self.config.gradient_scaler_enabled:
            logger.info(f"    Init Scale: {self.config.scaler_init_scale}")
            logger.info(f"    Growth Factor: {self.config.scaler_growth_factor}")
            logger.info(f"    Backoff Factor: {self.config.scaler_backoff_factor}")
            logger.info(f"    Growth Interval: {self.config.scaler_growth_interval}")
    
    def _log_training_args_update(self, training_args: TrainingArguments):
        """Log training arguments updates."""
        logger.info("Training Arguments Updated:")
        logger.info(f"  FP16: {training_args.fp16}")
        logger.info(f"  BF16: {training_args.bf16}")
        logger.info(f"  Pin Memory: {training_args.dataloader_pin_memory}")
        if hasattr(training_args, 'torch_compile'):
            logger.info(f"  Torch Compile: {training_args.torch_compile}")
    
    @classmethod
    def create_for_gpu(cls, gpu_name: str) -> 'MixedPrecisionManager':
        """
        Create MixedPrecisionManager instance for specific GPU.
        
        Args:
            gpu_name: Name of the GPU
            
        Returns:
            MixedPrecisionManager instance configured for the GPU
        """
        return cls(gpu_type=gpu_name)
    
    @classmethod
    def get_supported_gpu_types(cls) -> list[str]:
        """
        Get list of supported GPU types.
        
        Returns:
            List of supported GPU type identifiers
        """
        return list(cls.PRECISION_CONFIGS.keys())


def create_mixed_precision_manager(gpu_name: Optional[str] = None) -> MixedPrecisionManager:
    """
    Factory function to create MixedPrecisionManager.
    
    Args:
        gpu_name: Name of the GPU (optional, will auto-detect if None)
        
    Returns:
        MixedPrecisionManager instance
    """
    if gpu_name is None:
        # Try to auto-detect GPU
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Auto-detected GPU: {gpu_name}")
            else:
                logger.warning("CUDA not available, using default configuration")
        except Exception as e:
            logger.warning(f"Failed to auto-detect GPU: {e}, using default configuration")
    
    return MixedPrecisionManager(gpu_type=gpu_name)