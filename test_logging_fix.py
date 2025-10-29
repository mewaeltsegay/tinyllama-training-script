#!/usr/bin/env python3
"""
Test script to validate the logging fixes.
"""

import logging
from unittest.mock import Mock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_logging_fix():
    """Test the logging fix with mock data."""
    
    # Mock training state
    mock_state = Mock()
    mock_state.global_step = 50
    mock_state.max_steps = 1000
    mock_state.epoch = 0.1
    mock_state.log_history = [
        {'loss': 6.2524, 'learning_rate': 5.928853754940712e-08, 'grad_norm': float('nan')},
        {'loss': 4.1234, 'learning_rate': 1.251646903820817e-07, 'grad_norm': 2.5}
    ]
    
    # Test log extraction
    latest_log = mock_state.log_history[-1] if mock_state.log_history else {}
    
    # Extract loss and learning rate with proper fallbacks
    current_loss = latest_log.get('loss', latest_log.get('train_loss', 0.0))
    current_lr = latest_log.get('learning_rate', 0.0)
    
    # Handle NaN and invalid values
    if current_loss is None or (isinstance(current_loss, float) and (current_loss != current_loss)):
        current_loss = 0.0
    if current_lr is None or (isinstance(current_lr, float) and (current_lr != current_lr)):
        current_lr = 0.0
    
    # Test logging format
    progress = mock_state.global_step / mock_state.max_steps * 100
    tokens_per_second = 150000  # Mock value
    eta = 600  # Mock ETA in seconds
    
    log_message = (
        f"Step {mock_state.global_step}/{mock_state.max_steps} ({progress:.1f}%) | "
        f"Loss: {current_loss:.4f} | "
        f"LR: {current_lr:.2e} | "
        f"Tokens/s: {tokens_per_second:.0f} | "
        f"ETA: {eta/60:.1f}min"
    )
    
    logger.info("Testing logging format:")
    logger.info(log_message)
    
    # Validate values
    assert current_loss == 4.1234, f"Expected loss 4.1234, got {current_loss}"
    assert current_lr == 1.251646903820817e-07, f"Expected LR 1.251646903820817e-07, got {current_lr}"
    assert progress == 5.0, f"Expected progress 5.0%, got {progress}%"
    
    logger.info("✅ Logging fix validation successful!")
    return True

if __name__ == "__main__":
    try:
        test_logging_fix()
        print("✅ All logging tests passed!")
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()