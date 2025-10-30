#!/usr/bin/env python3
"""
Conservative training settings for small datasets to minimize NaN recovery.
"""

import subprocess
import sys

def run_conservative_training():
    """Run training with conservative settings to minimize gradient instability."""
    
    # Conservative training parameters
    cmd = [
        sys.executable, "tinyllama_tigrinya_training.py",
        "--dataset-dir", "dataset",
        "--tokenizer", "tokenizer",
        "--max-samples", "10",
        "--checkpoint-steps", "5",
        "--num-epochs", "1",
        "--learning-rate", "1e-5",      # Lower learning rate
        "--batch-size", "1",            # Explicit small batch
        "--gradient-accumulation-steps", "4",  # Smaller accumulation
        "--max-grad-norm", "0.5",       # More aggressive clipping
        "--weight-decay", "0.001",      # Lower weight decay
        "--warmup-ratio", "0.2"         # More warmup
    ]
    
    print("Running conservative training to minimize NaN recovery...")
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    exit_code = run_conservative_training()
    if exit_code == 0:
        print("✅ Conservative training completed successfully!")
    else:
        print("❌ Training failed with exit code:", exit_code)
    sys.exit(exit_code)