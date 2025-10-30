#!/usr/bin/env python3
"""
Minimal Stability Configuration - No NaN Recovery Needed
Focus on preventing issues rather than complex recovery.
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_minimal_stable_config() -> Dict[str, Any]:
    """
    Get minimal, rock-solid configuration that prevents NaN gradients.
    No complex recovery needed - just stable settings.
    """
    
    # Detect GPU
    if not torch.cuda.is_available():
        return {
            "learning_rate": 1e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_grad_norm": 0.5,
            "mixed_precision": False,
            "fp16": False,
            "bf16": False,
        }
    
    device_name = torch.cuda.get_device_name(0).lower()
    device_props = torch.cuda.get_device_properties(0)
    compute_cap = device_props.major + device_props.minor / 10.0
    
    # Base stable config
    config = {
        "learning_rate": 1e-5,      # Conservative
        "batch_size": 1,            # Start small
        "gradient_accumulation_steps": 16,
        "max_grad_norm": 0.5,       # Aggressive clipping
        "weight_decay": 0.001,      # Low weight decay
        "warmup_ratio": 0.3,        # Lots of warmup
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 0,
        "mixed_precision": True,
    }
    
    # GPU-specific precision settings
    if "h100" in device_name:
        # H100: Use BF16 but with VERY conservative settings
        config.update({
            "fp16": False,
            "bf16": True,
            "learning_rate": 5e-6,  # Even more conservative for H100
            "batch_size": 4,        # Smaller batch
            "gradient_accumulation_steps": 8,
        })
        logger.info("H100 detected: Using conservative BF16 settings")
        
    elif compute_cap >= 8.0:
        # Modern GPUs: Use BF16 if supported
        config.update({
            "fp16": False,
            "bf16": True,
            "learning_rate": 1e-5,
        })
        logger.info(f"Modern GPU detected (CC {compute_cap}): Using BF16")
        
    else:
        # Older GPUs: Conservative FP16
        config.update({
            "fp16": True,
            "bf16": False,
            "learning_rate": 5e-6,  # Very conservative
        })
        logger.info(f"Older GPU detected (CC {compute_cap}): Using conservative FP16")
    
    return config


def apply_minimal_config_to_training_args(training_args, config: Dict[str, Any]):
    """Apply minimal stable configuration to training arguments."""
    
    # Mixed precision
    training_args.fp16 = config["fp16"]
    training_args.bf16 = config["bf16"]
    
    # Training parameters
    training_args.per_device_train_batch_size = config["batch_size"]
    training_args.gradient_accumulation_steps = config["gradient_accumulation_steps"]
    training_args.max_grad_norm = config["max_grad_norm"]
    
    # Dataloader settings
    training_args.dataloader_pin_memory = config["dataloader_pin_memory"]
    training_args.dataloader_num_workers = config["dataloader_num_workers"]
    training_args.dataloader_prefetch_factor = None  # Avoid Windows issues
    
    # Disable potentially problematic features
    training_args.gradient_checkpointing = False  # Can cause NaN issues
    training_args.torch_compile = False           # Can be unstable
    
    logger.info("Applied minimal stable configuration:")
    logger.info(f"  Mixed precision: FP16={config['fp16']}, BF16={config['bf16']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Max grad norm: {config['max_grad_norm']}")
    
    return training_args


class MinimalStabilityConfigurator:
    """Minimal stability configurator - no complex recovery needed."""
    
    def __init__(self):
        self.config = get_minimal_stable_config()
        logger.info("MinimalStabilityConfigurator initialized")
    
    def detect_and_configure(self) -> Dict[str, Any]:
        """Return minimal stable configuration."""
        return self.config
    
    def apply_to_training_arguments(self, training_args):
        """Apply configuration to training arguments."""
        return apply_minimal_config_to_training_args(training_args, self.config)
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get simple stability report."""
        return {
            "configuration_status": "minimal_stable",
            "mixed_precision": f"FP16={self.config['fp16']}, BF16={self.config['bf16']}",
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
            "max_grad_norm": self.config["max_grad_norm"]
        }


def create_minimal_stability_configurator() -> MinimalStabilityConfigurator:
    """Create minimal stability configurator."""
    return MinimalStabilityConfigurator()


if __name__ == "__main__":
    # Test minimal configurator
    configurator = create_minimal_stability_configurator()
    config = configurator.detect_and_configure()
    print("Minimal Stable Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")