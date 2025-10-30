#!/usr/bin/env python3
"""
Ultra-basic training test to isolate issues.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Ultra-Basic Training Test ===")
    
    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {device_name}")
    else:
        logger.info("CUDA not available")
        return
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        # Load real dataset directly
        logger.info("Loading real dataset...")
        import json
        from torch.utils.data import Dataset
        
        # Read JSONL file
        texts = []
        with open("dataset/train.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Limit to 100 samples
                    break
                data = json.loads(line)
                texts.append(data.get("text", ""))
        
        logger.info(f"Loaded {len(texts)} texts from dataset")
        
        # Tokenize texts
        tokenized = []
        for text in texts:
            if text.strip():  # Skip empty texts
                tokens = tokenizer(text, truncation=True, padding=False, max_length=256)
                if len(tokens['input_ids']) > 5:  # Skip very short sequences
                    tokenized.append(tokens)
        
        logger.info(f"Tokenized {len(tokenized)} valid sequences")
        
        # Create dataset
        class SimpleDataset(Dataset):
            def __init__(self, tokenized_texts):
                self.data = tokenized_texts
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long)
                }
        
        train_dataset = SimpleDataset(tokenized)
        
        logger.info(f"Dataset size: {len(train_dataset)}")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=None
        )
        
        # Test data collator
        logger.info("Testing data collator...")
        batch = data_collator([train_dataset[0], train_dataset[1]])
        logger.info(f"Batch shape: {batch['input_ids'].shape}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            # Move batch to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logger.info(f"Test loss: {loss.item():.4f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("Invalid loss detected!")
                return
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-6,  # Conservative but not too low
            warmup_steps=2,
            max_grad_norm=1.0,   # Allow higher gradients since they're working
            logging_steps=1,
            save_steps=10,
            bf16=False,          # Use FP32 for maximum stability
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
            disable_tqdm=False,
            seed=42,
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train for just a few steps
        logger.info("Starting training...")
        train_result = trainer.train()
        
        logger.info("Training completed!")
        logger.info(f"Final loss: {train_result.training_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()