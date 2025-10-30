"""
Mixed Precision Integration for Training.

This module provides utilities to integrate the MixedPrecisionManager with
the training process, including TrainingArguments updates and trainer setup.
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import TrainingArguments, Trainer

from utils.mixed_precision_manager import MixedPrecisionManager, create_mixed_precision_manager
from utils.hardware import HardwareDetector

logger = logging.getLogger(__name__)


def setup_mixed_precision_training(
    training_args: TrainingArguments,
    hardware_config: Optional[Dict[str, Any]] = None,
    gpu_name: Optional[str] = None
) -> tuple[TrainingArguments, MixedPrecisionManager]:
    """
    Setup mixed precision training with hardware-specific configurations.
    
    Args:
        training_args: Original training arguments
        hardware_config: Hardware configuration dictionary
        gpu_name: GPU name for mixed precision configuration
        
    Returns:
        Tuple of (updated_training_args, mixed_precision_manager)
    """
    try:
        # Determine GPU type for mixed precision configuration
        if gpu_name is None and hardware_config:
            gpu_name = hardware_config.get("gpu_name")
        
        if gpu_name is None:
            # Try to auto-detect GPU
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"Auto-detected GPU for mixed precision: {gpu_name}")
            except Exception as e:
                logger.warning(f"Failed to auto-detect GPU: {e}")
        
        # Create mixed precision manager
        mp_manager = create_mixed_precision_manager(gpu_name)
        
        # Update training arguments with mixed precision settings
        updated_args = mp_manager.update_training_arguments(training_args)
        
        # Validate mixed precision support
        precision_support = mp_manager.validate_precision_support()
        logger.info(f"Mixed precision support: {precision_support}")
        
        # Warn if requested precision is not supported
        precision_config = mp_manager.get_precision_config()
        if precision_config["fp16"] and not precision_support["fp16"]:
            logger.warning("FP16 requested but not supported, falling back to FP32")
            updated_args.fp16 = False
        
        if precision_config["bf16"] and not precision_support["bf16"]:
            logger.warning("BF16 requested but not supported, falling back to FP16 or FP32")
            updated_args.bf16 = False
            if precision_support["fp16"]:
                updated_args.fp16 = True
        
        logger.info("Mixed precision training setup completed successfully")
        return updated_args, mp_manager
        
    except Exception as e:
        logger.error(f"Failed to setup mixed precision training: {e}")
        logger.warning("Falling back to default precision settings")
        
        # Create default mixed precision manager
        mp_manager = MixedPrecisionManager(gpu_type="default")
        return training_args, mp_manager


def configure_trainer_with_mixed_precision(
    trainer: Trainer,
    mp_manager: MixedPrecisionManager,
    hardware_config: Optional[Dict[str, Any]] = None
) -> Trainer:
    """
    Configure trainer with mixed precision settings and gradient scaler.
    
    Args:
        trainer: HuggingFace Trainer instance
        mp_manager: MixedPrecisionManager instance
        hardware_config: Hardware configuration dictionary
        
    Returns:
        Configured trainer with mixed precision support
    """
    try:
        # Setup gradient scaler
        scaler = mp_manager.setup_scaler()
        
        # Store mixed precision manager and scaler in trainer for access during training
        trainer.mp_manager = mp_manager
        trainer.scaler = scaler
        
        # Log mixed precision configuration
        precision_config = mp_manager.get_precision_config()
        logger.info("Trainer configured with mixed precision:")
        logger.info(f"  FP16: {precision_config['fp16']}")
        logger.info(f"  BF16: {precision_config['bf16']}")
        logger.info(f"  Gradient Scaler: {scaler is not None}")
        
        if scaler:
            scaler_state = mp_manager.get_scaler_state()
            logger.info(f"  Scaler Scale: {scaler_state.get('scale', 'N/A')}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Failed to configure trainer with mixed precision: {e}")
        return trainer


def create_mixed_precision_callback(mp_manager: MixedPrecisionManager):
    """
    Create a training callback for mixed precision monitoring.
    
    Args:
        mp_manager: MixedPrecisionManager instance
        
    Returns:
        TrainerCallback for mixed precision monitoring
    """
    from transformers import TrainerCallback
    
    class MixedPrecisionCallback(TrainerCallback):
        """Callback for monitoring mixed precision training."""
        
        def __init__(self, mp_manager: MixedPrecisionManager):
            self.mp_manager = mp_manager
            self.overflow_count = 0
            self.total_steps = 0
        
        def on_step_end(self, args, state, control, **kwargs):
            """Monitor gradient scaler state after each step."""
            self.total_steps += 1
            
            # Check scaler state if available
            scaler_state = self.mp_manager.get_scaler_state()
            if scaler_state.get("enabled", False):
                if scaler_state.get("found_inf", False):
                    self.overflow_count += 1
                    logger.warning(f"Gradient overflow detected at step {state.global_step}")
                
                # Log scaler statistics every 100 steps
                if state.global_step % 100 == 0:
                    overflow_rate = self.overflow_count / self.total_steps * 100
                    logger.info(f"Mixed precision stats - Scale: {scaler_state.get('scale', 'N/A')}, "
                              f"Overflow rate: {overflow_rate:.2f}%")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Add mixed precision metrics to logs."""
            if logs and hasattr(self.mp_manager, 'scaler') and self.mp_manager.scaler:
                scaler_state = self.mp_manager.get_scaler_state()
                if scaler_state.get("enabled", False):
                    logs["grad_scale"] = scaler_state.get("scale", 0.0)
                    logs["grad_overflow_rate"] = (self.overflow_count / max(1, self.total_steps)) * 100
    
    return MixedPrecisionCallback(mp_manager)


def apply_hardware_specific_precision_optimizations(
    trainer: Trainer,
    hardware_config: Dict[str, Any],
    mp_manager: MixedPrecisionManager
):
    """
    Apply hardware-specific precision optimizations to the trainer.
    
    Args:
        trainer: HuggingFace Trainer instance
        hardware_config: Hardware configuration dictionary
        mp_manager: MixedPrecisionManager instance
    """
    try:
        gpu_name = hardware_config.get("gpu_name", "").lower()
        
        # RTX 4050 specific optimizations
        if "rtx" in gpu_name and "4050" in gpu_name:
            logger.info("Applying RTX 4050 mixed precision optimizations...")
            
            # Use more conservative gradient accumulation
            if hasattr(trainer.args, 'gradient_accumulation_steps'):
                trainer.args.gradient_accumulation_steps = max(
                    trainer.args.gradient_accumulation_steps, 8
                )
            
            # Ensure FP16 with conservative scaling
            trainer.args.fp16 = True
            trainer.args.bf16 = False
            
            logger.info("RTX 4050 optimizations applied")
        
        # H100 specific optimizations
        elif "h100" in gpu_name:
            logger.info("Applying H100 mixed precision optimizations...")
            
            # Use BF16 for better stability on H100
            trainer.args.bf16 = True
            trainer.args.fp16 = False
            
            # Enable torch compile if available
            if hasattr(trainer.args, 'torch_compile'):
                trainer.args.torch_compile = True
            
            # H100-specific advanced optimizations
            if hasattr(trainer.args, 'tf32'):
                trainer.args.tf32 = True  # Enable TensorFloat-32
            
            # Optimize for H100's memory bandwidth
            if hasattr(trainer.args, 'dataloader_pin_memory'):
                trainer.args.dataloader_pin_memory = True
            
            # Use fused optimizer if available
            try:
                # This will be used when creating the optimizer
                trainer._fused_optimizer = True
                logger.info("Fused optimizer enabled for H100")
            except:
                pass
            
            logger.info("H100 optimizations applied")
        
        # A100 specific optimizations
        elif "a100" in gpu_name:
            logger.info("Applying A100 mixed precision optimizations...")
            
            # Use BF16 for A100
            trainer.args.bf16 = True
            trainer.args.fp16 = False
            
            # Enable torch compile if available
            if hasattr(trainer.args, 'torch_compile'):
                trainer.args.torch_compile = True
            
            logger.info("A100 optimizations applied")
        
        # L4 specific optimizations
        elif "l4" in gpu_name:
            logger.info("Applying L4 mixed precision optimizations...")
            
            # Use FP16 with moderate settings for L4
            trainer.args.fp16 = True
            trainer.args.bf16 = False
            
            # Moderate gradient accumulation
            if hasattr(trainer.args, 'gradient_accumulation_steps'):
                trainer.args.gradient_accumulation_steps = max(
                    trainer.args.gradient_accumulation_steps, 4
                )
            
            logger.info("L4 optimizations applied")
        
        else:
            logger.info(f"Using default mixed precision settings for {gpu_name}")
        
        # Validate final configuration
        precision_config = mp_manager.get_precision_config()
        logger.info(f"Final mixed precision configuration: FP16={trainer.args.fp16}, BF16={trainer.args.bf16}")
        
    except Exception as e:
        logger.error(f"Failed to apply hardware-specific precision optimizations: {e}")


def get_mixed_precision_training_summary(
    mp_manager: MixedPrecisionManager,
    hardware_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get a summary of mixed precision training configuration.
    
    Args:
        mp_manager: MixedPrecisionManager instance
        hardware_config: Hardware configuration dictionary
        
    Returns:
        Dictionary with mixed precision training summary
    """
    try:
        precision_config = mp_manager.get_precision_config()
        scaler_state = mp_manager.get_scaler_state()
        precision_support = mp_manager.validate_precision_support()
        
        summary = {
            "gpu_type": mp_manager.gpu_type,
            "precision_config": precision_config,
            "scaler_enabled": scaler_state.get("enabled", False),
            "scaler_scale": scaler_state.get("scale", "N/A"),
            "precision_support": precision_support,
            "hardware_config": hardware_config or {}
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate mixed precision summary: {e}")
        return {"error": str(e)}


# Convenience function for easy integration
def setup_complete_mixed_precision_training(
    training_args: TrainingArguments,
    trainer: Optional[Trainer] = None,
    hardware_config: Optional[Dict[str, Any]] = None,
    gpu_name: Optional[str] = None
) -> tuple[TrainingArguments, MixedPrecisionManager, Optional[Trainer]]:
    """
    Complete setup for mixed precision training with all optimizations.
    
    Args:
        training_args: Original training arguments
        trainer: Optional trainer to configure
        hardware_config: Hardware configuration dictionary
        gpu_name: GPU name for mixed precision configuration
        
    Returns:
        Tuple of (updated_training_args, mixed_precision_manager, configured_trainer)
    """
    logger.info("Setting up complete mixed precision training...")
    
    # Setup mixed precision
    updated_args, mp_manager = setup_mixed_precision_training(
        training_args, hardware_config, gpu_name
    )
    
    # Configure trainer if provided
    configured_trainer = None
    if trainer is not None:
        configured_trainer = configure_trainer_with_mixed_precision(
            trainer, mp_manager, hardware_config
        )
        
        # Apply hardware-specific optimizations
        if hardware_config:
            apply_hardware_specific_precision_optimizations(
                configured_trainer, hardware_config, mp_manager
            )
    
    # Log summary
    summary = get_mixed_precision_training_summary(mp_manager, hardware_config)
    logger.info(f"Mixed precision training setup complete: {summary}")
    
    return updated_args, mp_manager, configured_trainer