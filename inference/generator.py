"""
Inference engine for text generation and validation.

This module provides the InferenceEngine class for generating Tigrinya text
using trained TinyLlama models and validating training results.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Inference engine for text generation using trained TinyLlama models.
    
    This class handles:
    - Loading trained models from checkpoints
    - Configurable text generation with temperature and length controls
    - Automatic inference validation after training completion
    - Sample generation with predefined Tigrinya prompts
    - Saving generated samples to validation output files
    """
    
    def __init__(self, model: Optional[PreTrainedModel] = None, 
                 tokenizer: Optional[PreTrainedTokenizer] = None):
        """
        Initialize InferenceEngine.
        
        Args:
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model is not None and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        
        logger.info(f"InferenceEngine initialized with device: {self.device}")
    
    def load_model_from_checkpoint(self, checkpoint_dir: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing model checkpoint
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            checkpoint_path = Path(checkpoint_dir)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
            # Ensure model is in eval mode
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully from checkpoint")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model from checkpoint: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100, 
                     temperature: float = 0.8, do_sample: bool = True,
                     top_p: float = 0.9, top_k: int = 50,
                     repetition_penalty: float = 1.1) -> str:
        """
        Generate Tigrinya text from a given prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to use sampling or greedy decoding
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If model or tokenizer not loaded
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation")
        
        try:
            logger.debug(f"Generating text for prompt: {prompt[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Create generation config
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
            )
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Remove the original prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            logger.debug(f"Generated text: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}") from e
    
    def generate_multiple_samples(self, prompts: List[str], 
                                max_new_tokens: int = 100,
                                temperature: float = 0.8,
                                samples_per_prompt: int = 1) -> Dict[str, List[str]]:
        """
        Generate multiple text samples for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens per generation
            temperature: Sampling temperature
            samples_per_prompt: Number of samples to generate per prompt
            
        Returns:
            Dictionary mapping prompts to lists of generated samples
        """
        results = {}
        
        for prompt in prompts:
            logger.info(f"Generating {samples_per_prompt} samples for prompt: {prompt}")
            samples = []
            
            for i in range(samples_per_prompt):
                try:
                    generated = self.generate_text(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    samples.append(generated)
                    logger.debug(f"Sample {i+1}/{samples_per_prompt} generated")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate sample {i+1} for prompt '{prompt}': {str(e)}")
                    samples.append(f"[Generation failed: {str(e)}]")
            
            results[prompt] = samples
        
        return results
    
    def validate_training(self, prompts: Optional[List[str]] = None, 
                         output_file: Optional[str] = None,
                         max_new_tokens: int = 100,
                         temperature: float = 0.8,
                         samples_per_prompt: int = 3) -> Dict[str, Any]:
        """
        Run inference validation and save results to file.
        
        Args:
            prompts: List of prompts to use (uses defaults if None)
            output_file: Path to save validation results (auto-generated if None)
            max_new_tokens: Maximum tokens to generate per sample
            temperature: Generation temperature
            samples_per_prompt: Number of samples per prompt
            
        Returns:
            Dictionary containing validation results and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before validation")
        
        # Use default Tigrinya prompts if none provided
        if prompts is None:
            prompts = [
                "ሰላም! ከመይ ኣሎኻ?",  # Hello! How are you?
                "ሎሚ ጽቡቕ መዓልቲ እዩ።",  # Today is a good day.
                "ትግርኛ ቋንቋ ጽቡቕ እዩ።",  # Tigrinya language is good.
                "ኣብ ትግራይ ብዙሕ ሰብ ይነብር።",  # Many people live in Tigray.
                "ትምህርቲ ኣዝዩ ኣገዳሲ እዩ።"  # Education is very important.
            ]
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"validation_results_{timestamp}.json"
        
        logger.info(f"Starting validation with {len(prompts)} prompts")
        logger.info(f"Generation parameters: max_tokens={max_new_tokens}, temp={temperature}")
        
        try:
            # Generate samples
            generation_results = self.generate_multiple_samples(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                samples_per_prompt=samples_per_prompt
            )
            
            # Prepare validation results
            validation_results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_info": self._get_model_info(),
                    "generation_config": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "samples_per_prompt": samples_per_prompt
                    },
                    "total_prompts": len(prompts),
                    "total_samples": len(prompts) * samples_per_prompt
                },
                "results": []
            }
            
            # Format results
            for prompt, samples in generation_results.items():
                result_entry = {
                    "prompt": prompt,
                    "samples": samples,
                    "sample_count": len(samples),
                    "successful_generations": len([s for s in samples if not s.startswith("[Generation failed")])
                }
                validation_results["results"].append(result_entry)
            
            # Save results to file
            self._save_validation_results(validation_results, output_file)
            
            logger.info(f"Validation completed successfully. Results saved to: {output_file}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise RuntimeError(f"Validation failed: {str(e)}") from e
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None or self.tokenizer is None:
            return {"error": "Model or tokenizer not loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "vocab_size": len(self.tokenizer),
            "device": str(self.device)
        }
        
        if hasattr(self.model, 'config'):
            config = self.model.config
            info.update({
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None)
            })
        
        return info
    
    def _save_validation_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save validation results to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Validation results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {str(e)}")
            raise RuntimeError(f"Failed to save results: {str(e)}") from e
    
    def evaluate_generation_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of generated text samples.
        
        Args:
            results: Validation results dictionary
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            metrics = {
                "total_prompts": len(results["results"]),
                "total_samples": 0,
                "successful_generations": 0,
                "failed_generations": 0,
                "average_length": 0,
                "length_distribution": [],
                "success_rate": 0.0
            }
            
            total_length = 0
            lengths = []
            
            for result in results["results"]:
                samples = result["samples"]
                metrics["total_samples"] += len(samples)
                
                for sample in samples:
                    if sample.startswith("[Generation failed"):
                        metrics["failed_generations"] += 1
                    else:
                        metrics["successful_generations"] += 1
                        length = len(sample.split())
                        lengths.append(length)
                        total_length += length
            
            if metrics["successful_generations"] > 0:
                metrics["average_length"] = total_length / metrics["successful_generations"]
                metrics["length_distribution"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "median": sorted(lengths)[len(lengths)//2] if lengths else 0
                }
            
            if metrics["total_samples"] > 0:
                metrics["success_rate"] = metrics["successful_generations"] / metrics["total_samples"]
            
            logger.info(f"Generation quality metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate generation quality: {str(e)}")
            return {"error": str(e)}
    
    def run_interactive_generation(self, max_new_tokens: int = 100, 
                                 temperature: float = 0.8) -> None:
        """
        Run interactive text generation session.
        
        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before interactive generation")
        
        print("=== Interactive Tigrinya Text Generation ===")
        print("Enter prompts to generate text. Type 'quit' to exit.")
        print(f"Generation settings: max_tokens={max_new_tokens}, temperature={temperature}")
        print()
        
        while True:
            try:
                prompt = input("Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print("Generating...")
                generated = self.generate_text(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                print(f"Generated: {generated}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Generation error: {str(e)}")
                continue