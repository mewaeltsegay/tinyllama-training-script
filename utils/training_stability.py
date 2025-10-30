"""
Training Stability Integration Module

Provides a unified interface for all training stability components including:
- Gradient stabilization
- NaN recovery
- Mixed precision management
- Hardware-specific configurations
- Comprehensive monitoring and reporting
"""

import torch
import torch.nn as nn
import logging
import math
from typing import Dict, Any, Optional, Tuple, Callable, List
from dataclasses import dataclass

from .gradient_stabilizer import GradientStabilizer, MixedPrecisionManager, GradientStats
from .nan_recovery import NaNRecoveryManager, RecoveryStrategy


@dataclass
class StabilityConfig:
    """Configuration for training stability system"""
    # Gradient stabilization
    max_grad_norm: float = 1.0
    gradient_clipping_enabled: bool = True
    
    # NaN recovery
    nan_recovery_enabled: bool = True
    max_nan_recoveries: int = 10
    lr_reduction_factor: float = 0.5
    min_learning_rate: float = 1e-8
    
    # Mixed precision
    mixed_precision_enabled: bool = True
    gpu_type: str = "auto"
    
    # Monitoring
    log_gradient_stats: bool = True
    gradient_stats_interval: int = 10
    stability_report_interval: int = 100
    
    # Checkpointing for recovery
    enable_checkpoint_rollback: bool = False
    checkpoint_interval: int = 100


class TrainingStabilityManager:
    """
    Unified training stability management system
    
    Integrates gradient stabilization, NaN recovery, and mixed precision
    to provide comprehensive training stability across different hardware.
    """
    
    def __init__(self, config: StabilityConfig = None):
        self.config = config or StabilityConfig()
        
        # Initialize components
        self.gradient_stabilizer = GradientStabilizer(
            max_grad_norm=self.config.max_grad_norm,
            nan_recovery_enabled=self.config.nan_recovery_enabled
        )
        
        self.nan_recovery_manager = NaNRecoveryManager(
            max_recoveries=self.config.max_nan_recoveries,
            lr_reduction_factor=self.config.lr_reduction_factor,
            min_learning_rate=self.config.min_learning_rate,
            enable_checkpoint_rollback=self.config.enable_checkpoint_rollback
        )
        
        self.mixed_precision_manager = MixedPrecisionManager(
            gpu_type=self.config.gpu_type
        )
        
        # Setup mixed precision scaler
        if self.config.mixed_precision_enabled:
            self.scaler = self.mixed_precision_manager.setup_scaler(enabled=True)
        else:
            self.scaler = None
        
        # Monitoring state
        self.step_count = 0
        self.last_gradient_stats = None
        self.stability_alerts = []
        
        logging.info("TrainingStabilityManager initialized successfully")
        self._log_configuration()
    
    def _log_configuration(self):
        """Log the current stability configuration"""
        precision_config = self.mixed_precision_manager.get_precision_config()
        
        logging.info("Training Stability Configuration:")
        logging.info(f"  GPU Type: {self.mixed_precision_manager.gpu_type}")
        logging.info(f"  Max Grad Norm: {self.config.max_grad_norm}")
        logging.info(f"  Mixed Precision: {self.config.mixed_precision_enabled}")
        logging.info(f"  Precision Config: {precision_config}")
        logging.info(f"  NaN Recovery: {self.config.nan_recovery_enabled}")
        logging.info(f"  Checkpoint Rollback: {self.config.enable_checkpoint_rollback}")
    
    def initialize_model(self, model: nn.Module):
        """
        Initialize model weights for numerical stability
        
        Args:
            model: PyTorch model to initialize
        """
        logging.info("Initializing model for training stability")
        self.gradient_stabilizer.initialize_model_weights(model)
    
    def process_training_step(self, 
                            model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            loss: torch.Tensor,
                            step: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a complete training step with stability management
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            loss: Computed loss tensor
            step: Current training step
            
        Returns:
            Tuple of (success, metrics_dict)
        """
        self.step_count = step
        metrics = {}
        
        try:
            # Save checkpoint if enabled
            if self.config.enable_checkpoint_rollback:
                self.nan_recovery_manager.save_checkpoint(model, optimizer, step)
            
            # Scale loss if using mixed precision
            if self.scaler is not None:
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
            else:
                loss.backward()
            
            # Process gradients and check for stability
            gradient_norm, is_healthy = self.gradient_stabilizer.clip_gradients(model)
            
            # Store gradient statistics
            self.last_gradient_stats = self.gradient_stabilizer.compute_gradient_stats(model)
            
            # Add gradient metrics
            metrics.update({
                'gradient_norm': gradient_norm,
                'gradient_healthy': is_healthy,
                'gradient_nan_count': self.last_gradient_stats.nan_count,
                'gradient_inf_count': self.last_gradient_stats.inf_count,
                'gradient_max_norm': self.last_gradient_stats.max_norm,
                'gradient_min_norm': self.last_gradient_stats.min_norm
            })
            
            # Handle unhealthy gradients
            if not is_healthy:
                logging.warning(f"Unhealthy gradients detected at step {step}")
                
                if self.config.nan_recovery_enabled:
                    recovery_success = self.nan_recovery_manager.handle_nan_gradients(
                        model, optimizer, step
                    )
                    
                    metrics['recovery_attempted'] = True
                    metrics['recovery_success'] = recovery_success
                    
                    if not recovery_success:
                        logging.error("Recovery failed - training should be stopped")
                        return False, metrics
                else:
                    logging.error("Unhealthy gradients detected but recovery is disabled")
                    return False, metrics
            
            # Perform optimizer step
            if self.scaler is not None:
                # Mixed precision step
                step_taken = self.mixed_precision_manager.step_optimizer(optimizer)
                metrics['optimizer_step_taken'] = step_taken
            else:
                # Regular step
                optimizer.step()
                metrics['optimizer_step_taken'] = True
            
            # Zero gradients for next iteration
            optimizer.zero_grad()
            
            # Reset recovery count on successful steps
            if is_healthy and step % 50 == 0:  # Reset every 50 successful steps
                self.gradient_stabilizer.reset_recovery_count()
                self.nan_recovery_manager.reset_recovery_count()
            
            # Log gradient statistics periodically
            if (self.config.log_gradient_stats and 
                step % self.config.gradient_stats_interval == 0):
                self._log_gradient_statistics(step)
            
            # Generate stability report periodically
            if step % self.config.stability_report_interval == 0:
                self._generate_stability_report(step)
            
            return True, metrics
            
        except Exception as e:
            logging.error(f"Error in training step {step}: {e}")
            metrics['error'] = str(e)
            return False, metrics
    
    def _log_gradient_statistics(self, step: int):
        """Log detailed gradient statistics"""
        if self.last_gradient_stats is None:
            return
        
        stats = self.last_gradient_stats
        logging.info(
            f"Step {step} Gradient Stats - "
            f"Norm: {stats.total_norm:.6f}, "
            f"Max: {stats.max_norm:.6f}, "
            f"Min: {stats.min_norm:.6f}, "
            f"NaN: {stats.nan_count}, "
            f"Inf: {stats.inf_count}, "
            f"Zero: {stats.zero_count}/{stats.param_count}"
        )
    
    def _generate_stability_report(self, step: int):
        """Generate and log comprehensive stability report"""
        gradient_report = self.gradient_stabilizer.get_stability_report()
        recovery_report = self.nan_recovery_manager.get_recovery_report()
        
        logging.info(f"=== Stability Report (Step {step}) ===")
        logging.info(f"Gradient Status: {gradient_report.get('status', 'unknown')}")
        logging.info(f"Average Gradient Norm: {gradient_report.get('avg_gradient_norm', 0):.6f}")
        logging.info(f"NaN Recovery Count: {recovery_report.get('total_recoveries', 0)}")
        
        # Check for stability alerts
        self._check_stability_alerts(gradient_report, recovery_report, step)
    
    def _check_stability_alerts(self, gradient_report: Dict, recovery_report: Dict, step: int):
        """Check for stability issues and generate alerts"""
        alerts = []
        
        # Check gradient health
        if gradient_report.get('status') == 'unstable':
            alerts.append(f"Unstable gradients detected at step {step}")
        
        # Check recovery frequency
        recovery_count = recovery_report.get('total_recoveries', 0)
        if recovery_count > self.config.max_nan_recoveries * 0.8:
            alerts.append(f"High recovery count: {recovery_count}/{self.config.max_nan_recoveries}")
        
        # Check recent success rate
        recent_success = recovery_report.get('recent_success_rate', 1.0)
        if recent_success < 0.5:
            alerts.append(f"Low recovery success rate: {recent_success:.2f}")
        
        # Log alerts
        for alert in alerts:
            logging.warning(f"STABILITY ALERT: {alert}")
            self.stability_alerts.append({'step': step, 'alert': alert})
        
        # Keep only recent alerts
        self.stability_alerts = self.stability_alerts[-20:]
    
    def get_precision_config(self) -> Dict[str, Any]:
        """Get current mixed precision configuration"""
        return self.mixed_precision_manager.get_precision_config()
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get current stability metrics for monitoring"""
        gradient_report = self.gradient_stabilizer.get_stability_report()
        recovery_report = self.nan_recovery_manager.get_recovery_report()
        
        return {
            'gradient_stability': gradient_report,
            'recovery_status': recovery_report,
            'recent_alerts': self.stability_alerts[-5:],  # Last 5 alerts
            'mixed_precision_config': self.get_precision_config(),
            'scaler_scale': self.scaler.get_scale() if self.scaler else None
        }
    
    def should_stop_training(self) -> Tuple[bool, str]:
        """
        Determine if training should be stopped due to stability issues
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check if maximum recoveries exceeded
        recovery_report = self.nan_recovery_manager.get_recovery_report()
        recovery_count = recovery_report.get('total_recoveries', 0)
        
        if recovery_count >= self.config.max_nan_recoveries:
            return True, f"Maximum NaN recoveries exceeded: {recovery_count}/{self.config.max_nan_recoveries}"
        
        # Check for persistent instability
        gradient_report = self.gradient_stabilizer.get_stability_report()
        if gradient_report.get('status') == 'unstable':
            # Count recent unstable alerts
            recent_unstable_alerts = sum(
                1 for alert in self.stability_alerts[-10:] 
                if 'unstable' in alert.get('alert', '').lower()
            )
            
            if recent_unstable_alerts >= 5:
                return True, "Persistent gradient instability detected"
        
        # Check learning rate
        if hasattr(self, '_last_lr') and self._last_lr < self.config.min_learning_rate:
            return True, f"Learning rate too low: {self._last_lr} < {self.config.min_learning_rate}"
        
        return False, ""
    
    def update_learning_rate(self, optimizer: torch.optim.Optimizer):
        """Update internal learning rate tracking"""
        if optimizer.param_groups:
            self._last_lr = optimizer.param_groups[0]['lr']
    
    def get_recommendations(self) -> List[str]:
        """Get stability recommendations based on current state"""
        gradient_report = self.gradient_stabilizer.get_stability_report()
        recovery_report = self.nan_recovery_manager.get_recovery_report()
        
        recommendations = []
        
        # Add gradient-based recommendations
        recommendations.extend(gradient_report.get('recommendations', []))
        
        # Add recovery-based recommendations
        recommendations.extend(recovery_report.get('recommendations', []))
        
        # Add mixed precision recommendations
        precision_config = self.get_precision_config()
        if precision_config.get('fp16') and self.mixed_precision_manager.gpu_type in ['rtx_4050']:
            recommendations.append("Consider using bf16 instead of fp16 for better stability")
        
        return list(set(recommendations))  # Remove duplicates


# Convenience functions for easy integration

def create_stability_manager(max_grad_norm: float = 1.0,
                           gpu_type: str = "auto",
                           enable_mixed_precision: bool = True,
                           enable_nan_recovery: bool = True) -> TrainingStabilityManager:
    """
    Create a training stability manager with sensible defaults
    
    Args:
        max_grad_norm: Maximum gradient norm for clipping
        gpu_type: GPU type for hardware-specific optimizations
        enable_mixed_precision: Whether to enable mixed precision training
        enable_nan_recovery: Whether to enable NaN recovery mechanisms
        
    Returns:
        Configured TrainingStabilityManager instance
    """
    config = StabilityConfig(
        max_grad_norm=max_grad_norm,
        mixed_precision_enabled=enable_mixed_precision,
        gpu_type=gpu_type,
        nan_recovery_enabled=enable_nan_recovery
    )
    
    return TrainingStabilityManager(config)


def apply_stability_to_training_args(training_args, gpu_type: str = "auto") -> None:
    """
    Apply stability-focused modifications to Hugging Face TrainingArguments
    
    Args:
        training_args: Hugging Face TrainingArguments instance
        gpu_type: GPU type for hardware-specific optimizations
    """
    # Create temporary manager to get precision config
    temp_manager = MixedPrecisionManager(gpu_type)
    precision_config = temp_manager.get_precision_config()
    
    # Apply precision settings
    if precision_config.get('fp16'):
        training_args.fp16 = True
        training_args.bf16 = False
    elif precision_config.get('bf16'):
        training_args.fp16 = False
        training_args.bf16 = True
    
    # Apply gradient clipping
    if precision_config.get('gradient_clipping'):
        training_args.max_grad_norm = precision_config.get('max_grad_norm', 1.0)
    
    # Apply conservative settings for stability
    if not hasattr(training_args, 'dataloader_pin_memory'):
        training_args.dataloader_pin_memory = False  # Reduce memory pressure
    
    if not hasattr(training_args, 'gradient_checkpointing'):
        training_args.gradient_checkpointing = True  # Save memory, improve stability
    
    logging.info(f"Applied stability settings to training arguments for {gpu_type}")
    logging.info(f"  FP16: {training_args.fp16}, BF16: {training_args.bf16}")
    logging.info(f"  Max Grad Norm: {training_args.max_grad_norm}")