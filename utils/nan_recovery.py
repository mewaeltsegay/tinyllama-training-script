"""
NaN Recovery Manager for Training Stability

Advanced recovery mechanisms for handling NaN gradients and training instabilities.
Provides multiple recovery strategies and automatic fallback mechanisms.
"""

import torch
import torch.nn as nn
import logging
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class RecoveryStrategy(Enum):
    """Available recovery strategies for NaN gradients"""
    ZERO_GRADIENTS = "zero_gradients"
    REDUCE_LEARNING_RATE = "reduce_lr"
    REINITIALIZE_LAYERS = "reinit_layers"
    RESET_OPTIMIZER = "reset_optimizer"
    CHECKPOINT_ROLLBACK = "checkpoint_rollback"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    step: int
    strategy: RecoveryStrategy
    success: bool
    nan_params: List[str]
    lr_before: float
    lr_after: float
    additional_info: Dict[str, Any]


class NaNRecoveryManager:
    """
    Advanced NaN recovery system with multiple strategies
    
    Implements a hierarchical recovery approach:
    1. Zero gradients (least disruptive)
    2. Reduce learning rate
    3. Reinitialize problematic layers
    4. Reset optimizer state
    5. Rollback to previous checkpoint (if available)
    """
    
    def __init__(self, 
                 max_recoveries: int = 10,
                 lr_reduction_factor: float = 0.5,
                 min_learning_rate: float = 1e-8,
                 enable_checkpoint_rollback: bool = False):
        
        self.max_recoveries = max_recoveries
        self.lr_reduction_factor = lr_reduction_factor
        self.min_learning_rate = min_learning_rate
        self.enable_checkpoint_rollback = enable_checkpoint_rollback
        
        # Recovery tracking
        self.recovery_count = 0
        self.recovery_history: List[RecoveryAttempt] = []
        self.strategy_success_rates: Dict[RecoveryStrategy, List[bool]] = {
            strategy: [] for strategy in RecoveryStrategy
        }
        
        # Checkpoint management
        self.checkpoints: Dict[int, Dict[str, Any]] = {}
        self.checkpoint_interval = 100  # Save checkpoint every N steps
        
        logging.info(f"NaNRecoveryManager initialized with max_recoveries={max_recoveries}")
    
    def handle_nan_gradients(self, 
                           model: nn.Module, 
                           optimizer: torch.optim.Optimizer,
                           step: int,
                           current_lr: float = None) -> bool:
        """
        Handle NaN gradients using hierarchical recovery strategies
        
        Args:
            model: PyTorch model with NaN gradients
            optimizer: Optimizer instance
            step: Current training step
            current_lr: Current learning rate (extracted from optimizer if None)
            
        Returns:
            True if recovery was successful, False if training should stop
        """
        if self.recovery_count >= self.max_recoveries:
            logging.error(f"Maximum recoveries ({self.max_recoveries}) exceeded - stopping training")
            return False
        
        # Extract current learning rate if not provided
        if current_lr is None:
            current_lr = self._get_current_lr(optimizer)
        
        # Identify NaN parameters
        nan_params = self._identify_nan_parameters(model)
        
        logging.warning(f"NaN gradients detected at step {step}. Attempting recovery #{self.recovery_count + 1}")
        logging.info(f"NaN parameters: {nan_params}")
        
        # Try recovery strategies in order of increasing disruption
        recovery_strategies = [
            RecoveryStrategy.ZERO_GRADIENTS,
            RecoveryStrategy.REDUCE_LEARNING_RATE,
            RecoveryStrategy.REINITIALIZE_LAYERS,
            RecoveryStrategy.RESET_OPTIMIZER
        ]
        
        if self.enable_checkpoint_rollback:
            recovery_strategies.append(RecoveryStrategy.CHECKPOINT_ROLLBACK)
        
        # Select strategy based on recovery history and success rates
        strategy = self._select_recovery_strategy(recovery_strategies, nan_params)
        
        # Execute recovery
        success = self._execute_recovery_strategy(
            strategy, model, optimizer, step, nan_params, current_lr
        )
        
        # Record recovery attempt
        new_lr = self._get_current_lr(optimizer)
        recovery_attempt = RecoveryAttempt(
            step=step,
            strategy=strategy,
            success=success,
            nan_params=nan_params,
            lr_before=current_lr,
            lr_after=new_lr,
            additional_info={}
        )
        
        self.recovery_history.append(recovery_attempt)
        self.strategy_success_rates[strategy].append(success)
        
        if success:
            self.recovery_count += 1
            logging.info(f"Recovery successful using strategy: {strategy.value}")
        else:
            logging.error(f"Recovery failed using strategy: {strategy.value}")
        
        return success
    
    def _identify_nan_parameters(self, model: nn.Module) -> List[str]:
        """Identify parameters with NaN gradients"""
        nan_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_params.append(name)
        
        return nan_params
    
    def _select_recovery_strategy(self, 
                                strategies: List[RecoveryStrategy], 
                                nan_params: List[str]) -> RecoveryStrategy:
        """
        Select the most appropriate recovery strategy based on history and context
        
        Args:
            strategies: Available recovery strategies
            nan_params: Parameters with NaN gradients
            
        Returns:
            Selected recovery strategy
        """
        # If this is the first few recoveries, start with least disruptive
        if self.recovery_count < 3:
            return strategies[0]
        
        # If many parameters have NaN, use more aggressive strategy
        if len(nan_params) > 10:
            return RecoveryStrategy.REINITIALIZE_LAYERS
        
        # Check success rates of previous strategies
        for strategy in strategies:
            success_rate = self._get_strategy_success_rate(strategy)
            if success_rate > 0.5:  # If strategy has >50% success rate
                return strategy
        
        # Default to the next strategy in the hierarchy
        recovery_level = min(self.recovery_count // 2, len(strategies) - 1)
        return strategies[recovery_level]
    
    def _get_strategy_success_rate(self, strategy: RecoveryStrategy) -> float:
        """Calculate success rate for a recovery strategy"""
        attempts = self.strategy_success_rates[strategy]
        if not attempts:
            return 0.0
        return sum(attempts) / len(attempts)
    
    def _execute_recovery_strategy(self, 
                                 strategy: RecoveryStrategy,
                                 model: nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 step: int,
                                 nan_params: List[str],
                                 current_lr: float) -> bool:
        """Execute the selected recovery strategy"""
        
        try:
            if strategy == RecoveryStrategy.ZERO_GRADIENTS:
                return self._zero_gradients(model, nan_params)
            
            elif strategy == RecoveryStrategy.REDUCE_LEARNING_RATE:
                return self._reduce_learning_rate(optimizer, current_lr)
            
            elif strategy == RecoveryStrategy.REINITIALIZE_LAYERS:
                return self._reinitialize_layers(model, nan_params)
            
            elif strategy == RecoveryStrategy.RESET_OPTIMIZER:
                return self._reset_optimizer_state(optimizer)
            
            elif strategy == RecoveryStrategy.CHECKPOINT_ROLLBACK:
                return self._rollback_to_checkpoint(model, optimizer, step)
            
            else:
                logging.error(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logging.error(f"Recovery strategy {strategy.value} failed with exception: {e}")
            return False
    
    def _zero_gradients(self, model: nn.Module, nan_params: List[str]) -> bool:
        """Zero out gradients for NaN parameters"""
        logging.info("Executing strategy: Zero gradients")
        
        zeroed_count = 0
        for name, param in model.named_parameters():
            if name in nan_params and param.grad is not None:
                param.grad.zero_()
                zeroed_count += 1
        
        logging.info(f"Zeroed gradients for {zeroed_count} parameters")
        return zeroed_count > 0
    
    def _reduce_learning_rate(self, optimizer: torch.optim.Optimizer, current_lr: float) -> bool:
        """Reduce learning rate for all parameter groups"""
        logging.info("Executing strategy: Reduce learning rate")
        
        new_lr = current_lr * self.lr_reduction_factor
        
        if new_lr < self.min_learning_rate:
            logging.warning(f"Learning rate would be too low ({new_lr} < {self.min_learning_rate})")
            return False
        
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = new_lr
            logging.info(f"Reduced learning rate from {old_lr} to {new_lr}")
        
        return True
    
    def _reinitialize_layers(self, model: nn.Module, nan_params: List[str]) -> bool:
        """Reinitialize layers with NaN gradients"""
        logging.info("Executing strategy: Reinitialize layers")
        
        reinitialized_count = 0
        
        for name, param in model.named_parameters():
            if name in nan_params:
                logging.info(f"Reinitializing parameter: {name}")
                
                # Determine initialization strategy based on parameter shape and name
                if 'weight' in name.lower():
                    if len(param.shape) >= 2:
                        # Linear/Conv weights
                        nn.init.xavier_uniform_(param.data)
                    else:
                        # 1D weights (e.g., layer norm)
                        nn.init.ones_(param.data)
                elif 'bias' in name.lower():
                    # Bias terms
                    nn.init.zeros_(param.data)
                elif 'embedding' in name.lower():
                    # Embedding weights
                    nn.init.normal_(param.data, mean=0.0, std=0.02)
                else:
                    # Default initialization
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.zeros_(param.data)
                
                reinitialized_count += 1
        
        logging.info(f"Reinitialized {reinitialized_count} parameters")
        return reinitialized_count > 0
    
    def _reset_optimizer_state(self, optimizer: torch.optim.Optimizer) -> bool:
        """Reset optimizer state (momentum, etc.)"""
        logging.info("Executing strategy: Reset optimizer state")
        
        try:
            # Clear optimizer state
            optimizer.state.clear()
            logging.info("Optimizer state cleared successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to reset optimizer state: {e}")
            return False
    
    def _rollback_to_checkpoint(self, 
                              model: nn.Module, 
                              optimizer: torch.optim.Optimizer, 
                              current_step: int) -> bool:
        """Rollback to the most recent checkpoint"""
        logging.info("Executing strategy: Checkpoint rollback")
        
        if not self.checkpoints:
            logging.warning("No checkpoints available for rollback")
            return False
        
        # Find the most recent checkpoint before current step
        available_steps = [step for step in self.checkpoints.keys() if step < current_step]
        
        if not available_steps:
            logging.warning("No suitable checkpoint found for rollback")
            return False
        
        rollback_step = max(available_steps)
        checkpoint = self.checkpoints[rollback_step]
        
        try:
            # Restore model state
            model.load_state_dict(checkpoint['model_state'])
            
            # Restore optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            logging.info(f"Successfully rolled back to checkpoint at step {rollback_step}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to rollback to checkpoint: {e}")
            return False
    
    def save_checkpoint(self, 
                       model: nn.Module, 
                       optimizer: torch.optim.Optimizer, 
                       step: int):
        """Save a checkpoint for potential rollback"""
        if not self.enable_checkpoint_rollback:
            return
        
        if step % self.checkpoint_interval == 0:
            self.checkpoints[step] = {
                'model_state': model.state_dict().copy(),
                'optimizer_state': optimizer.state_dict().copy(),
                'step': step
            }
            
            # Keep only the last 5 checkpoints to save memory
            if len(self.checkpoints) > 5:
                oldest_step = min(self.checkpoints.keys())
                del self.checkpoints[oldest_step]
            
            logging.debug(f"Checkpoint saved at step {step}")
    
    def _get_current_lr(self, optimizer: torch.optim.Optimizer) -> float:
        """Extract current learning rate from optimizer"""
        if optimizer.param_groups:
            return optimizer.param_groups[0]['lr']
        return 0.0
    
    def reset_recovery_count(self):
        """Reset recovery count after successful training period"""
        if self.recovery_count > 0:
            logging.info(f"Resetting recovery count from {self.recovery_count} to 0")
            self.recovery_count = 0
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive recovery report"""
        if not self.recovery_history:
            return {"status": "no_recoveries", "message": "No recovery attempts recorded"}
        
        # Calculate strategy success rates
        strategy_stats = {}
        for strategy, attempts in self.strategy_success_rates.items():
            if attempts:
                success_rate = sum(attempts) / len(attempts)
                strategy_stats[strategy.value] = {
                    "attempts": len(attempts),
                    "success_rate": success_rate,
                    "last_used": max([r.step for r in self.recovery_history if r.strategy == strategy], default=0)
                }
        
        # Recent recovery trends
        recent_recoveries = self.recovery_history[-10:]
        recent_success_rate = sum(r.success for r in recent_recoveries) / len(recent_recoveries) if recent_recoveries else 0
        
        return {
            "total_recoveries": self.recovery_count,
            "max_recoveries": self.max_recoveries,
            "recent_success_rate": recent_success_rate,
            "strategy_statistics": strategy_stats,
            "most_recent_recovery": self.recovery_history[-1].__dict__ if self.recovery_history else None,
            "checkpoints_available": len(self.checkpoints),
            "recommendations": self._generate_recovery_recommendations()
        }
    
    def _generate_recovery_recommendations(self) -> List[str]:
        """Generate recommendations based on recovery history"""
        recommendations = []
        
        if self.recovery_count > self.max_recoveries * 0.8:
            recommendations.append("Approaching maximum recovery limit - consider stopping training")
        
        if self.recovery_count > 5:
            recommendations.extend([
                "High number of recoveries detected",
                "Consider reducing learning rate globally",
                "Check data preprocessing for invalid values",
                "Verify model architecture for numerical stability"
            ])
        
        # Strategy-specific recommendations
        zero_grad_success = self._get_strategy_success_rate(RecoveryStrategy.ZERO_GRADIENTS)
        if zero_grad_success < 0.3:
            recommendations.append("Zero gradient strategy has low success rate - consider more aggressive recovery")
        
        lr_reduction_success = self._get_strategy_success_rate(RecoveryStrategy.REDUCE_LEARNING_RATE)
        if lr_reduction_success > 0.8:
            recommendations.append("Learning rate reduction is highly effective - consider starting with lower LR")
        
        return recommendations