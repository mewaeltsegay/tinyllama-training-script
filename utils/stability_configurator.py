#!/usr/bin/env python3
"""
Stability Configurator
Integrates hardware detection, mixed precision, and training stability configuration.
"""

import torch
import logging
from typing import Dict, Any, Optional
from .hardware import HardwareDetector
from .mixed_precision_manager import MixedPrecisionManager
from .training_stability import create_stability_manager, apply_stability_to_training_args

logger = logging.getLogger(__name__)


class StabilityConfigurator:
    """
    Comprehensive stability configurator that integrates hardware detection,
    mixed precision management, and training stability configuration.
    """
    
    def __init__(self):
        """Initialize the stability configurator."""
        self.hardware_detector = HardwareDetector()
        self.mixed_precision_manager = None
        self.stability_manager = None
        self.gpu_config = None
        self.stability_config = None
        
        logger.info("StabilityConfigurator initialized")
    
    def detect_and_configure(self) -> Dict[str, Any]:
        """
        Detect hardware and configure optimal stability settings.
        
        Returns:
            Dictionary with stability configuration
        """
        try:
            # Detect GPU configuration
            self.gpu_config = self.hardware_detector.detect_gpu_config()
            gpu_name = self.gpu_config.get("gpu_name", "Unknown")
            
            logger.info(f"Detected GPU: {gpu_name}")
            logger.info(f"GPU Memory: {self.gpu_config.get('gpu_memory_gb', 0)} GB")
            logger.info(f"Mixed Precision: {self.gpu_config.get('mixed_precision_dtype', 'fp32')}")
            
            # Create mixed precision manager based on detected GPU
            gpu_type = self._map_gpu_name_to_type(gpu_name)
            self.mixed_precision_manager = MixedPrecisionManager(gpu_type=gpu_type)
            
            # Create stability manager with higher recovery limit for small datasets
            from .training_stability import StabilityConfig
            stability_config = StabilityConfig(
                max_grad_norm=1.0,
                gpu_type=gpu_type,
                mixed_precision_enabled=self.gpu_config.get("mixed_precision", True),
                nan_recovery_enabled=True,
                max_nan_recoveries=20,  # Higher limit for small datasets
                lr_reduction_factor=0.8,  # More conservative LR reduction
                gradient_stats_interval=5,  # More frequent monitoring
                stability_report_interval=10
            )
            
            from .training_stability import TrainingStabilityManager
            self.stability_manager = TrainingStabilityManager(config=stability_config)
            
            # Build comprehensive stability configuration
            self.stability_config = {
                # Hardware configuration
                "gpu_name": gpu_name,
                "gpu_memory_gb": self.gpu_config.get("gpu_memory_gb", 0),
                "gpu_type": gpu_type,
                
                # Training configuration
                "batch_size": self.gpu_config.get("batch_size", 1),
                "gradient_accumulation_steps": self.gpu_config.get("gradient_accumulation_steps", 8),
                "max_grad_norm": 1.0,
                "learning_rate": 5e-5,  # Default learning rate
                "warmup_ratio": 0.1,    # Default warmup ratio
                "weight_decay": 0.01,   # Default weight decay
                
                # Mixed precision configuration
                "mixed_precision": self.gpu_config.get("mixed_precision", True),
                "mixed_precision_dtype": self.gpu_config.get("mixed_precision_dtype", "fp16"),
                "enable_flash_attention": self.gpu_config.get("enable_flash_attention", False),
                "enable_gradient_checkpointing": self.gpu_config.get("enable_gradient_checkpointing", True),
                
                # Optimization settings
                "use_compile": self.gpu_config.get("use_compile", False),
                "dataloader_pin_memory": self.gpu_config.get("dataloader_pin_memory", True),
                "dataloader_num_workers": self.gpu_config.get("dataloader_num_workers", 0),
                
                # Stability features
                "nan_recovery_enabled": True,
                "gradient_monitoring_enabled": True,
                "stability_reporting_enabled": True,
                
                # Detection status
                "detection_errors": self.gpu_config.get("detection_errors", []),
                "fallback_used": self.gpu_config.get("fallback_used", False)
            }
            
            logger.info("Stability configuration completed successfully")
            return self.stability_config
            
        except Exception as e:
            logger.error(f"Error in stability configuration: {e}")
            # Return safe fallback configuration
            return self._get_fallback_stability_config()
    
    def apply_to_training_arguments(self, training_args):
        """
        Apply stability configuration to Hugging Face TrainingArguments.
        
        Args:
            training_args: TrainingArguments instance to modify
            
        Returns:
            Modified TrainingArguments instance
        """
        if not self.stability_config:
            logger.warning("No stability configuration available, using defaults")
            return training_args
        
        try:
            # Apply mixed precision settings
            if self.mixed_precision_manager:
                training_args = self.mixed_precision_manager.update_training_arguments(training_args)
            
            # Apply stability-focused modifications
            gpu_type = self.stability_config.get("gpu_type", "auto")
            apply_stability_to_training_args(training_args, gpu_type)
            
            # Apply batch size if not already set
            if not hasattr(training_args, 'per_device_train_batch_size') or training_args.per_device_train_batch_size is None:
                training_args.per_device_train_batch_size = self.stability_config.get("batch_size", 1)
            
            # Apply gradient accumulation if not already set
            if not hasattr(training_args, 'gradient_accumulation_steps') or training_args.gradient_accumulation_steps is None:
                training_args.gradient_accumulation_steps = self.stability_config.get("gradient_accumulation_steps", 8)
            
            # Apply gradient clipping
            if not hasattr(training_args, 'max_grad_norm') or training_args.max_grad_norm is None:
                training_args.max_grad_norm = self.stability_config.get("max_grad_norm", 1.0)
            
            logger.info("Applied stability configuration to training arguments")
            logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
            logger.info(f"  Mixed precision: FP16={getattr(training_args, 'fp16', False)}, BF16={getattr(training_args, 'bf16', False)}")
            
            return training_args
            
        except Exception as e:
            logger.error(f"Error applying stability configuration to training arguments: {e}")
            return training_args
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Get comprehensive stability report.
        
        Returns:
            Dictionary with stability metrics and status
        """
        report = {
            "timestamp": torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else None,
            "configuration_status": "configured" if self.stability_config else "not_configured",
            "hardware_detection_status": "success" if self.gpu_config else "failed",
            "mixed_precision_status": "enabled" if self.mixed_precision_manager else "disabled",
            "stability_manager_status": "active" if self.stability_manager else "inactive"
        }
        
        # Add hardware information
        if self.gpu_config:
            report["hardware"] = {
                "gpu_name": self.gpu_config.get("gpu_name", "Unknown"),
                "gpu_memory_gb": self.gpu_config.get("gpu_memory_gb", 0),
                "mixed_precision_dtype": self.gpu_config.get("mixed_precision_dtype", "fp32"),
                "detection_errors": len(self.gpu_config.get("detection_errors", [])),
                "fallback_used": self.gpu_config.get("fallback_used", False)
            }
        
        # Add stability configuration
        if self.stability_config:
            report["stability"] = {
                "batch_size": self.stability_config.get("batch_size", 1),
                "gradient_accumulation_steps": self.stability_config.get("gradient_accumulation_steps", 8),
                "max_grad_norm": self.stability_config.get("max_grad_norm", 1.0),
                "nan_recovery_enabled": self.stability_config.get("nan_recovery_enabled", False),
                "gradient_monitoring_enabled": self.stability_config.get("gradient_monitoring_enabled", False)
            }
        
        # Add mixed precision information
        if self.mixed_precision_manager:
            mp_config = self.mixed_precision_manager.get_precision_config()
            report["mixed_precision"] = {
                "fp16": mp_config.get("fp16", False),
                "bf16": mp_config.get("bf16", False),
                "torch_compile": mp_config.get("torch_compile", False),
                "flash_attention": mp_config.get("attention_implementation", "eager") == "flash_attention_2"
            }
        
        # Add stability manager metrics if available
        if self.stability_manager:
            try:
                stability_metrics = self.stability_manager.get_stability_metrics()
                report["metrics"] = stability_metrics
            except Exception as e:
                logger.warning(f"Could not get stability metrics: {e}")
                report["metrics"] = {"error": str(e)}
        
        return report
    
    def _map_gpu_name_to_type(self, gpu_name: str) -> str:
        """
        Map GPU name to type identifier for mixed precision manager.
        
        Args:
            gpu_name: GPU name from hardware detection
            
        Returns:
            GPU type identifier
        """
        gpu_name_lower = gpu_name.lower()
        
        if "h100" in gpu_name_lower:
            return "h100"
        elif "a100" in gpu_name_lower:
            return "a100"
        elif "rtx 4090" in gpu_name_lower or "4090" in gpu_name_lower:
            return "rtx_4090"
        elif "rtx 4080" in gpu_name_lower or "4080" in gpu_name_lower:
            return "rtx_4080"
        elif "rtx 4070" in gpu_name_lower or "4070" in gpu_name_lower:
            return "rtx_4070"
        elif "rtx 4060" in gpu_name_lower or "4060" in gpu_name_lower:
            return "rtx_4060"
        elif "rtx 4050" in gpu_name_lower or "4050" in gpu_name_lower:
            return "rtx_4050"
        elif "l4" in gpu_name_lower:
            return "l4"
        elif "t4" in gpu_name_lower:
            return "t4"
        elif "v100" in gpu_name_lower:
            return "v100"
        else:
            return "unknown"
    
    def _get_fallback_stability_config(self) -> Dict[str, Any]:
        """
        Get fallback stability configuration when detection fails.
        
        Returns:
            Safe fallback configuration
        """
        logger.warning("Using fallback stability configuration")
        
        return {
            "gpu_name": "Unknown",
            "gpu_memory_gb": 8,
            "gpu_type": "unknown",
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_grad_norm": 1.0,
            "learning_rate": 5e-5,  # Default learning rate
            "warmup_ratio": 0.1,    # Default warmup ratio
            "weight_decay": 0.01,   # Default weight decay
            "mixed_precision": True,
            "mixed_precision_dtype": "fp16",
            "enable_flash_attention": False,
            "enable_gradient_checkpointing": True,
            "use_compile": False,
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 0,
            "nan_recovery_enabled": True,
            "gradient_monitoring_enabled": True,
            "stability_reporting_enabled": True,
            "detection_errors": ["Fallback configuration used"],
            "fallback_used": True
        }


def create_stability_configurator() -> StabilityConfigurator:
    """
    Factory function to create a StabilityConfigurator instance.
    
    Returns:
        StabilityConfigurator instance
    """
    return StabilityConfigurator()


if __name__ == "__main__":
    # Test stability configurator
    print("=" * 50)
    print("STABILITY CONFIGURATOR TEST")
    print("=" * 50)
    
    configurator = create_stability_configurator()
    
    # Test configuration detection
    config = configurator.detect_and_configure()
    print(f"Configuration: {config}")
    
    # Test stability report
    report = configurator.get_stability_report()
    print(f"Stability Report: {report}")
    
    print("✅ Stability configurator test completed")