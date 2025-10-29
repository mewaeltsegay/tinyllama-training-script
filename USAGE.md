# TinyLlama Tigrinya Continuous Pretraining

This script performs continuous pretraining of TinyLlama on Tigrinya language datasets with automatic hardware optimization and small dataset techniques.

## Quick Start

```bash
# Basic training with default settings
python tinyllama_tigrinya_training.py

# Show usage examples
python tinyllama_tigrinya_training.py --examples

# Custom configuration
python tinyllama_tigrinya_training.py --learning-rate 1e-4 --num-epochs 5 --batch-size 2
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-compatible GPU (optional, CPU fallback available)

## Directory Structure

```
workspace/
├── dataset/
│   ├── train.jsonl          # Training data (required)
│   ├── validation.jsonl     # Validation data (optional)
│   └── test.jsonl          # Test data (optional)
├── tokenizer/
│   ├── sentencepiece.model  # SentencePiece model (required)
│   ├── sentencepiece.vocab  # SentencePiece vocabulary (required)
│   └── tokenizer_config.json # Tokenizer configuration (required)
└── output/                  # Output directory (created automatically)
    ├── checkpoint-*/        # Model checkpoints
    ├── logs/               # Training logs
    ├── validation_results.json
    └── execution_summary.json
```

## Dataset Format

Each line in JSONL files should contain:
```json
{"text": "ሰላም! ከመይ ኣሎኻ? ሎምስ እንታይ ገይርካ?"}
```

## Hardware Support

- Automatic GPU detection and optimization
- Supported GPUs: RTX 4050, L4, A100, H100
- Automatic fallback to CPU if no GPU available
- Memory-optimized configurations for each GPU type

## Key Features

- **Automatic Hardware Optimization**: Detects GPU type and optimizes batch sizes, memory usage
- **Small Dataset Techniques**: Data augmentation, regularization, learning rate scheduling
- **Comprehensive Logging**: Structured logging with error recovery
- **Checkpoint Management**: Automatic saving and resumption
- **Inference Validation**: Automatic text generation validation after training
- **Error Recovery**: Automatic recovery from common training errors (OOM, etc.)

## Command Line Options

Run `python tinyllama_tigrinya_training.py --help` for full list of options.

## Output Files

After successful training, you'll find:
- `output/checkpoint-*/`: Model checkpoints
- `output/validation_results.json`: Generated text samples
- `output/execution_summary.json`: Complete execution summary
- `output/logs/`: Detailed training logs
