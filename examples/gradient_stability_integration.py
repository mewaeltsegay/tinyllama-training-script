"""
Example: Integrating Gradient Stability System into Training

This example demonstrates how to integrate the gradient stability system
into your training loop for robust and stable training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TrainingArguments
import logging
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.training_stability import (
    TrainingStabilityManager, 
    StabilityConfig, 
    create_stability_manager,
    apply_stability_to_training_args
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def example_basic_integration():
    """Basic integration example with a simple model"""
    print("=== Basic Integration Example ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.Linear(10, 1)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create stability manager with default settings
    stability_manager = create_stability_manager(
        max_grad_norm=1.0,
        enable_mixed_precision=torch.cuda.is_available(),
        enable_nan_recovery=True
    )
    
    # Initialize model for stability
    stability_manager.initialize_model(model)
    
    # Training loop
    criterion = nn.MSELoss()
    
    for step in range(10):
        # Generate dummy data
        x = torch.randn(32, 100)
        y = torch.randn(32, 1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Process training step with stability management
        success, metrics = stability_manager.process_training_step(
            model, optimizer, loss, step
        )
        
        if not success:
            print(f"Training failed at step {step}")
            break
        
        # Check if training should stop due to stability issues
        should_stop, reason = stability_manager.should_stop_training()
        if should_stop:
            print(f"Training stopped due to stability issues: {reason}")
            break
        
        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}, "
                  f"Grad Norm = {metrics.get('gradient_norm', 0):.6f}")
    
    # Get final stability report
    stability_metrics = stability_manager.get_stability_metrics()
    print(f"Final gradient status: {stability_metrics['gradient_stability']['status']}")
    
    return True


def example_advanced_configuration():
    """Advanced configuration example with custom settings"""
    print("\n=== Advanced Configuration Example ===")
    
    # Create custom stability configuration
    config = StabilityConfig(
        max_grad_norm=0.5,                    # More aggressive clipping
        gradient_clipping_enabled=True,
        nan_recovery_enabled=True,
        max_nan_recoveries=15,                # Allow more recovery attempts
        lr_reduction_factor=0.3,              # More aggressive LR reduction
        min_learning_rate=1e-7,
        mixed_precision_enabled=True,
        gpu_type="rtx_4050",                  # Specific GPU configuration
        log_gradient_stats=True,
        gradient_stats_interval=5,            # Log every 5 steps
        stability_report_interval=20,         # Report every 20 steps
        enable_checkpoint_rollback=True,      # Enable checkpoint rollback
        checkpoint_interval=50
    )
    
    # Create stability manager with custom config
    stability_manager = TrainingStabilityManager(config)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(50, 100),
        nn.LayerNorm(100),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Create optimizer with higher learning rate (potentially unstable)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # Initialize model
    stability_manager.initialize_model(model)
    
    # Get precision configuration
    precision_config = stability_manager.get_precision_config()
    print(f"Using precision config: {precision_config}")
    
    # Training loop with potential instability
    criterion = nn.MSELoss()
    
    for step in range(30):
        # Generate data with potential for causing instability
        x = torch.randn(16, 50)
        if step > 10:  # Introduce extreme values after step 10
            x = x * (1 + step * 0.1)  # Gradually increase magnitude
        
        y = torch.randn(16, 1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Update learning rate tracking
        stability_manager.update_learning_rate(optimizer)
        
        # Process training step
        success, metrics = stability_manager.process_training_step(
            model, optimizer, loss, step
        )
        
        if not success:
            print(f"Training failed at step {step}")
            
            # Get recommendations
            recommendations = stability_manager.get_recommendations()
            print("Stability recommendations:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                print(f"  - {rec}")
            break
        
        # Check stopping criteria
        should_stop, reason = stability_manager.should_stop_training()
        if should_stop:
            print(f"Training stopped: {reason}")
            break
        
        # Log progress
        if step % 10 == 0:
            recovery_attempted = metrics.get('recovery_attempted', False)
            recovery_success = metrics.get('recovery_success', True)
            
            print(f"Step {step}: Loss = {loss.item():.6f}, "
                  f"Grad Norm = {metrics.get('gradient_norm', 0):.6f}, "
                  f"Recovery = {recovery_attempted}/{recovery_success}")
    
    return True


def example_huggingface_integration():
    """Example of integrating with Hugging Face TrainingArguments"""
    print("\n=== Hugging Face Integration Example ===")
    
    # Create base training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
    )
    
    # Apply stability configurations to training arguments
    apply_stability_to_training_args(training_args, gpu_type="rtx_4050")
    
    print(f"Updated training args:")
    print(f"  FP16: {training_args.fp16}")
    print(f"  BF16: {training_args.bf16}")
    print(f"  Max Grad Norm: {training_args.max_grad_norm}")
    print(f"  Gradient Checkpointing: {getattr(training_args, 'gradient_checkpointing', 'Not set')}")
    
    # Create stability manager for use alongside Hugging Face trainer
    stability_manager = create_stability_manager(
        max_grad_norm=training_args.max_grad_norm,
        gpu_type="rtx_4050"
    )
    
    # In a real scenario, you would integrate this with a custom Trainer class
    # or use callbacks to monitor gradient stability
    
    print("✓ Stability system ready for Hugging Face integration")
    
    return True


def example_monitoring_and_alerts():
    """Example of monitoring and alerting functionality"""
    print("\n=== Monitoring and Alerts Example ===")
    
    # Create stability manager with monitoring enabled
    config = StabilityConfig(
        log_gradient_stats=True,
        gradient_stats_interval=1,  # Log every step for demo
        stability_report_interval=5
    )
    
    stability_manager = TrainingStabilityManager(config)
    
    # Create model
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=1.0)  # High LR for instability
    
    stability_manager.initialize_model(model)
    
    # Training with monitoring
    criterion = nn.MSELoss()
    
    for step in range(15):
        # Create increasingly unstable conditions
        scale = 1.0 + step * 0.5
        x = torch.randn(8, 10) * scale
        y = torch.randn(8, 1)
        
        output = model(x)
        loss = criterion(output, y)
        
        success, metrics = stability_manager.process_training_step(
            model, optimizer, loss, step
        )
        
        # Monitor specific metrics
        grad_norm = metrics.get('gradient_norm', 0)
        nan_count = metrics.get('gradient_nan_count', 0)
        
        if nan_count > 0:
            print(f"⚠️  Step {step}: NaN gradients detected!")
        
        if grad_norm > 10.0:
            print(f"⚠️  Step {step}: High gradient norm: {grad_norm:.2f}")
        
        if not success:
            print(f"❌ Step {step}: Training step failed")
            break
    
    # Get comprehensive stability metrics
    stability_metrics = stability_manager.get_stability_metrics()
    
    print("\nFinal Stability Report:")
    print(f"  Gradient Status: {stability_metrics['gradient_stability']['status']}")
    print(f"  Recovery Count: {stability_metrics['recovery_status']['total_recoveries']}")
    print(f"  Recent Alerts: {len(stability_metrics['recent_alerts'])}")
    
    return True


def main():
    """Run all integration examples"""
    print("Gradient Stability System Integration Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Integration", example_basic_integration),
        ("Advanced Configuration", example_advanced_configuration),
        ("Hugging Face Integration", example_huggingface_integration),
        ("Monitoring and Alerts", example_monitoring_and_alerts)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\nRunning: {name}")
            success = example_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nTo integrate into your training:")
    print("1. Import: from utils.training_stability import create_stability_manager")
    print("2. Create: stability_manager = create_stability_manager()")
    print("3. Initialize: stability_manager.initialize_model(model)")
    print("4. Use: success, metrics = stability_manager.process_training_step(model, optimizer, loss, step)")


if __name__ == "__main__":
    main()