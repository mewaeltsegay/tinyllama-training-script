# TinyLlama Continuous Pretraining for Tigrinya

This project implements continuous pretraining of the TinyLlama language model on Tigrinya language datasets, optimized for resource-constrained environments and small datasets.

## Project Structure

```
├── tinyllama_tigrinya_training.py  # Main training script
├── config/
│   ├── __init__.py
│   └── training_config.py          # Configuration dataclass
├── data/
│   ├── __init__.py
│   └── dataset_loader.py           # Dataset loading utilities
├── model/
│   ├── __init__.py
│   └── model_manager.py            # Model management
├── training/
│   ├── __init__.py
│   └── trainer.py                  # Training engine
├── inference/
│   ├── __init__.py
│   └── generator.py                # Inference engine
├── utils/
│   ├── __init__.py
│   ├── hardware.py                 # Hardware detection
│   └── logging.py                  # Logging utilities
├── dataset/                        # Tigrinya JSONL files
├── tokenizer/                      # SentencePiece tokenizer files
├── output/                         # Model checkpoints and logs
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Usage

```bash
# Basic usage with default parameters
python tinyllama_tigrinya_training.py

# Custom configuration
python tinyllama_tigrinya_training.py \
    --learning-rate 1e-4 \
    --num-epochs 5 \
    --batch-size 2 \
    --output-dir ./my_output

# See all available options
python tinyllama_tigrinya_training.py --help
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Tigrinya dataset in JSONL format
- SentencePiece tokenizer files

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The training script supports extensive command-line configuration for all parameters including:

- Model and training parameters
- Hardware optimization settings
- I/O paths and checkpoint frequency
- Small dataset optimization techniques
- Inference validation parameters

All parameters have sensible defaults and include validation to ensure correct usage.