"""
Hardware detection and GPU optimization utilities.

This module provides automatic GPU detection and optimization for different hardware configurations.
"""

import torch
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Configuration for specific GPU types."""
    name: str
    memory_gb: int
    batch_size: int
    gradient_accumulation_steps: int
    mixed_precision_dtype: str
    enable_flash_attention: bool
    enable_gradient_checkpointing: bool
    max_memory_usage: float  # Fraction of total memory to use


class HardwareDetector:
    """Detects GPU hardware and provides optimal training configurations."""
    
    # GPU-specific configurations based on design document
    GPU_CONFIGS = {
        "RTX 4050": GPUConfig(
            name="RTX 4050",
            memory_gb=6,
            batch_size=1,
            gradient_accumulation_steps=8,
            mixed_precision_dtype="fp32",  # Use FP32 to avoid gradient issues
            enable_flash_attention=False,
            enable_gradient_checkpointing=True,
            max_memory_usage=0.85
        ),
        "L4": GPUConfig(
            name="L4",
            memory_gb=24,
            batch_size=4,
            gradient_accumulation_steps=2,
            mixed_precision_dtype="fp16",
            enable_flash_attention=True,
            enable_gradient_checkpointing=False,
            max_memory_usage=0.90
        ),
        "A100": GPUConfig(
            name="A100",
            memory_gb=40,
            batch_size=8,
            gradient_accumulation_steps=1,
            mixed_precision_dtype="bf16",
            enable_flash_attention=True,
            enable_gradient_checkpointing=False,
            max_memory_usage=0.90
        ),
        "H100": GPUConfig(
            name="H100",
            memory_gb=80,
            batch_size=16,
            gradient_accumulation_steps=1,
            mixed_precision_dtype="bf16",
            enable_flash_attention=True,
            enable_gradient_checkpointing=False,
            max_memory_usage=0.90
        )
    }
    
    def __init__(self):
        self.gpu_info = None
        self.detected_config = None
    
    def detect_gpu_config(self) -> Dict[str, Any]:
        """
        Detect GPU and return optimal training configuration with comprehensive error handling.
        
        Returns:
            Dict containing GPU configuration parameters
        """
        detection_errors = []
        
        try:
            # Check CUDA availability with detailed error reporting
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, checking reasons...")
                
                # Try to determine why CUDA is not available
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        logger.warning("nvidia-smi not available or failed")
                    else:
                        logger.info("NVIDIA drivers detected but CUDA not available in PyTorch")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("NVIDIA drivers not detected")
                except Exception as e:
                    logger.warning(f"Error checking NVIDIA drivers: {e}")
                
                logger.warning("Falling back to CPU configuration")
                return self._get_cpu_config()
            
            # Attempt GPU detection with multiple fallback strategies
            gpu_name, gpu_memory_gb = None, None
            
            try:
                gpu_name, gpu_memory_gb = self._get_gpu_info()
                logger.info(f"Detected GPU: {gpu_name} with {gpu_memory_gb}GB memory")
            except Exception as e:
                detection_errors.append(f"GPU info detection failed: {e}")
                logger.warning(f"Primary GPU detection failed: {e}")
                
                # Try alternative detection methods
                try:
                    gpu_name, gpu_memory_gb = self._get_gpu_info_fallback()
                    logger.info(f"Fallback detection successful: {gpu_name} with {gpu_memory_gb}GB memory")
                except Exception as fallback_error:
                    detection_errors.append(f"Fallback GPU detection failed: {fallback_error}")
                    logger.error(f"All GPU detection methods failed")
                    return self._get_fallback_config(detection_errors)
            
            # Match GPU to known configurations with error handling
            try:
                gpu_config = self._match_gpu_config(gpu_name, gpu_memory_gb)
            except Exception as e:
                detection_errors.append(f"GPU config matching failed: {e}")
                logger.warning(f"GPU config matching failed: {e}, using memory-based fallback")
                gpu_config = self._get_memory_based_config(gpu_memory_gb)
            
            # Calculate optimal batch size with error handling
            try:
                optimal_batch_size = self.get_optimal_batch_size(
                    gpu_memory_gb * 1024,  # Convert to MB
                    model_size_mb=2200  # TinyLlama approximate size in MB
                )
                
                # Override batch size if calculated value is different
                if optimal_batch_size != gpu_config.batch_size:
                    logger.info(f"Adjusting batch size from {gpu_config.batch_size} to {optimal_batch_size} based on memory calculation")
                    gpu_config.batch_size = optimal_batch_size
                    
            except Exception as e:
                detection_errors.append(f"Batch size calculation failed: {e}")
                logger.warning(f"Batch size calculation failed: {e}, using conservative default")
                gpu_config.batch_size = 1  # Conservative fallback
            
            # Get optimization settings with error handling
            try:
                optimizations = self.enable_optimizations(gpu_config.name)
            except Exception as e:
                detection_errors.append(f"Optimization settings failed: {e}")
                logger.warning(f"Optimization settings failed: {e}, using safe defaults")
                optimizations = self._get_safe_optimizations()
            
            # Build final configuration
            config = {
                "gpu_name": gpu_config.name,
                "gpu_memory_gb": gpu_memory_gb,
                "batch_size": gpu_config.batch_size,
                "gradient_accumulation_steps": gpu_config.gradient_accumulation_steps,
                "mixed_precision": True,
                "mixed_precision_dtype": gpu_config.mixed_precision_dtype,
                "enable_flash_attention": gpu_config.enable_flash_attention,
                "enable_gradient_checkpointing": gpu_config.enable_gradient_checkpointing,
                "max_memory_usage": gpu_config.max_memory_usage,
                "detection_errors": detection_errors,
                **optimizations
            }
            
            # Validate final configuration
            try:
                self._validate_gpu_config(config)
            except Exception as e:
                detection_errors.append(f"Config validation failed: {e}")
                logger.warning(f"GPU config validation failed: {e}, applying safety corrections")
                config = self._apply_safety_corrections(config)
            
            self.detected_config = config
            
            if detection_errors:
                logger.warning(f"GPU detection completed with {len(detection_errors)} warnings")
            else:
                logger.info("GPU detection completed successfully")
            
            return config
            
        except Exception as e:
            detection_errors.append(f"Critical detection error: {e}")
            logger.error(f"Critical error in GPU detection: {e}")
            return self._get_fallback_config(detection_errors)
    
    def _get_gpu_info(self) -> Tuple[str, int]:
        """
        Get GPU name and memory information.
        
        Returns:
            Tuple of (gpu_name, memory_in_gb)
        """
        try:
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(0)
            
            # Get GPU memory in GB
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = int(gpu_memory_bytes / (1024**3))
            
            self.gpu_info = (gpu_name, gpu_memory_gb)
            return gpu_name, gpu_memory_gb
            
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            raise
    
    def _match_gpu_config(self, gpu_name: str, gpu_memory_gb: int) -> GPUConfig:
        """
        Match detected GPU to known configurations.
        
        Args:
            gpu_name: Name of the detected GPU
            gpu_memory_gb: GPU memory in GB
            
        Returns:
            GPUConfig object for the matched GPU
        """
        # Direct name matching
        for config_name, config in self.GPU_CONFIGS.items():
            if config_name.lower() in gpu_name.lower():
                return config
        
        # Memory-based matching for unknown GPUs
        if gpu_memory_gb <= 8:
            logger.info(f"Unknown GPU with {gpu_memory_gb}GB, using RTX 4050 config")
            return self.GPU_CONFIGS["RTX 4050"]
        elif gpu_memory_gb <= 24:
            logger.info(f"Unknown GPU with {gpu_memory_gb}GB, using L4 config")
            return self.GPU_CONFIGS["L4"]
        elif gpu_memory_gb <= 40:
            logger.info(f"Unknown GPU with {gpu_memory_gb}GB, using A100 config")
            return self.GPU_CONFIGS["A100"]
        else:
            logger.info(f"Unknown GPU with {gpu_memory_gb}GB, using H100 config")
            return self.GPU_CONFIGS["H100"]
    
    def get_optimal_batch_size(self, gpu_memory_mb: int, model_size_mb: int) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            gpu_memory_mb: Available GPU memory in MB
            model_size_mb: Model size in MB
            
        Returns:
            Optimal batch size
        """
        # Reserve memory for model, optimizer states, gradients, and activations
        # Model: 1x, Optimizer states: 2x, Gradients: 1x, Activations: variable
        base_memory_usage = model_size_mb * 4  # Conservative estimate
        
        # Available memory for batch processing (use 85% of total to be safe)
        available_memory = gpu_memory_mb * 0.85 - base_memory_usage
        
        if available_memory <= 0:
            logger.warning("Insufficient GPU memory, using batch size 1")
            return 1
        
        # Estimate memory per sample (rough approximation for sequence length 512)
        memory_per_sample = 50  # MB per sample for TinyLlama with seq_len=512
        
        optimal_batch_size = max(1, int(available_memory / memory_per_sample))
        
        # Cap at reasonable maximum
        optimal_batch_size = min(optimal_batch_size, 32)
        
        logger.info(f"Calculated optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def enable_optimizations(self, gpu_type: str) -> Dict[str, bool]:
        """
        Enable GPU-specific optimizations.
        
        Args:
            gpu_type: Type of GPU detected
            
        Returns:
            Dictionary of optimization flags
        """
        optimizations = {
            "use_cpu_offload": False,
            "use_8bit_optimizer": False,
            "use_compile": False,
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 0  # Use 0 for Windows compatibility
        }
        
        # Enable optimizations based on GPU type
        if gpu_type in ["A100", "H100"]:
            optimizations["use_compile"] = True  # PyTorch 2.0 compile for newer GPUs
            optimizations["dataloader_num_workers"] = 0  # Keep 0 for Windows
        elif gpu_type == "RTX 4050":
            optimizations["use_8bit_optimizer"] = False  # Disable to avoid FP16 gradient issues
            optimizations["dataloader_num_workers"] = 0  # Keep 0 for Windows
        
        return optimizations
    
    def _get_cpu_config(self) -> Dict[str, Any]:
        """
        Get configuration for CPU-only training.
        
        Returns:
            CPU training configuration
        """
        logger.warning("Using CPU configuration - training will be significantly slower")
        return {
            "gpu_name": "CPU",
            "gpu_memory_gb": 0,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "mixed_precision": False,
            "mixed_precision_dtype": "fp32",
            "enable_flash_attention": False,
            "enable_gradient_checkpointing": True,
            "max_memory_usage": 0.8,
            "use_cpu_offload": False,
            "use_8bit_optimizer": False,
            "use_compile": False,
            "dataloader_pin_memory": False,
            "dataloader_num_workers": 0
        }
    
    def _get_gpu_info_fallback(self) -> Tuple[str, int]:
        """
        Fallback method for GPU info detection using alternative approaches.
        
        Returns:
            Tuple of (gpu_name, memory_in_gb)
        """
        try:
            # Try using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_gb = int(memory_info.total / (1024**3))
                pynvml.nvmlShutdown()
                return gpu_name, gpu_memory_gb
            except ImportError:
                pass
            
            # Try using subprocess to call nvidia-smi
            try:
                import subprocess
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=name,memory.total', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        parts = lines[0].split(', ')
                        if len(parts) >= 2:
                            gpu_name = parts[0].strip()
                            gpu_memory_gb = int(float(parts[1]) / 1024)  # Convert MB to GB
                            return gpu_name, gpu_memory_gb
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass
            
            # Last resort: use torch with minimal info
            if torch.cuda.is_available():
                gpu_name = "Unknown GPU"
                # Try to get memory from torch
                try:
                    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = int(gpu_memory_bytes / (1024**3))
                except:
                    gpu_memory_gb = 8  # Conservative estimate
                
                return gpu_name, gpu_memory_gb
            
            raise RuntimeError("All fallback GPU detection methods failed")
            
        except Exception as e:
            logger.error(f"Fallback GPU detection failed: {e}")
            raise
    
    def _get_memory_based_config(self, gpu_memory_gb: int) -> GPUConfig:
        """
        Get GPU configuration based only on memory size when name matching fails.
        
        Args:
            gpu_memory_gb: GPU memory in GB
            
        Returns:
            GPUConfig object based on memory size
        """
        if gpu_memory_gb <= 8:
            logger.info(f"Using RTX 4050-like config for {gpu_memory_gb}GB GPU")
            return GPUConfig(
                name=f"Unknown_{gpu_memory_gb}GB",
                memory_gb=gpu_memory_gb,
                batch_size=1,
                gradient_accumulation_steps=8,
                mixed_precision_dtype="fp32",  # Use FP32 to avoid gradient issues
                enable_flash_attention=False,
                enable_gradient_checkpointing=True,
                max_memory_usage=0.85
            )
        elif gpu_memory_gb <= 16:
            logger.info(f"Using mid-range config for {gpu_memory_gb}GB GPU")
            return GPUConfig(
                name=f"Unknown_{gpu_memory_gb}GB",
                memory_gb=gpu_memory_gb,
                batch_size=2,
                gradient_accumulation_steps=4,
                mixed_precision_dtype="fp16",
                enable_flash_attention=False,
                enable_gradient_checkpointing=True,
                max_memory_usage=0.85
            )
        elif gpu_memory_gb <= 24:
            logger.info(f"Using L4-like config for {gpu_memory_gb}GB GPU")
            return GPUConfig(
                name=f"Unknown_{gpu_memory_gb}GB",
                memory_gb=gpu_memory_gb,
                batch_size=4,
                gradient_accumulation_steps=2,
                mixed_precision_dtype="fp16",
                enable_flash_attention=True,
                enable_gradient_checkpointing=False,
                max_memory_usage=0.90
            )
        elif gpu_memory_gb <= 40:
            logger.info(f"Using A100-like config for {gpu_memory_gb}GB GPU")
            return GPUConfig(
                name=f"Unknown_{gpu_memory_gb}GB",
                memory_gb=gpu_memory_gb,
                batch_size=8,
                gradient_accumulation_steps=1,
                mixed_precision_dtype="bf16",
                enable_flash_attention=True,
                enable_gradient_checkpointing=False,
                max_memory_usage=0.90
            )
        else:
            logger.info(f"Using H100-like config for {gpu_memory_gb}GB GPU")
            return GPUConfig(
                name=f"Unknown_{gpu_memory_gb}GB",
                memory_gb=gpu_memory_gb,
                batch_size=16,
                gradient_accumulation_steps=1,
                mixed_precision_dtype="bf16",
                enable_flash_attention=True,
                enable_gradient_checkpointing=False,
                max_memory_usage=0.90
            )
    
    def _get_safe_optimizations(self) -> Dict[str, bool]:
        """
        Get safe optimization settings when detection fails.
        
        Returns:
            Dictionary of conservative optimization flags
        """
        return {
            "use_cpu_offload": False,
            "use_8bit_optimizer": False,
            "use_compile": False,
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 0
        }
    
    def _validate_gpu_config(self, config: Dict[str, Any]) -> None:
        """
        Validate GPU configuration for consistency and safety.
        
        Args:
            config: GPU configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        required_fields = [
            "gpu_name", "gpu_memory_gb", "batch_size", "gradient_accumulation_steps",
            "mixed_precision_dtype", "max_memory_usage"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate ranges
        if config["batch_size"] < 1:
            raise ValueError(f"Invalid batch size: {config['batch_size']}")
        
        if config["gradient_accumulation_steps"] < 1:
            raise ValueError(f"Invalid gradient accumulation steps: {config['gradient_accumulation_steps']}")
        
        if not 0.1 <= config["max_memory_usage"] <= 1.0:
            raise ValueError(f"Invalid max memory usage: {config['max_memory_usage']}")
        
        if config["mixed_precision_dtype"] not in ["fp16", "bf16", "fp32"]:
            raise ValueError(f"Invalid mixed precision dtype: {config['mixed_precision_dtype']}")
    
    def _apply_safety_corrections(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply safety corrections to potentially invalid configuration.
        
        Args:
            config: Configuration to correct
            
        Returns:
            Corrected configuration
        """
        corrected_config = config.copy()
        
        # Ensure minimum safe values
        corrected_config["batch_size"] = max(1, corrected_config.get("batch_size", 1))
        corrected_config["gradient_accumulation_steps"] = max(1, corrected_config.get("gradient_accumulation_steps", 1))
        corrected_config["max_memory_usage"] = max(0.5, min(0.95, corrected_config.get("max_memory_usage", 0.8)))
        
        # Ensure valid mixed precision dtype
        if corrected_config.get("mixed_precision_dtype") not in ["fp16", "bf16", "fp32"]:
            corrected_config["mixed_precision_dtype"] = "fp16"
        
        # Disable risky optimizations if validation failed
        corrected_config["enable_flash_attention"] = False
        corrected_config["use_compile"] = False
        corrected_config["use_8bit_optimizer"] = False
        
        logger.info("Applied safety corrections to GPU configuration")
        return corrected_config
    
    def _get_fallback_config(self, detection_errors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get fallback configuration when detection fails.
        
        Args:
            detection_errors: List of errors encountered during detection
        
        Returns:
            Conservative fallback configuration
        """
        if detection_errors:
            logger.warning(f"GPU detection failed with {len(detection_errors)} errors:")
            for error in detection_errors:
                logger.warning(f"  - {error}")
        
        logger.warning("Using conservative fallback configuration")
        
        return {
            "gpu_name": "Unknown",
            "gpu_memory_gb": 8,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "mixed_precision": True,
            "mixed_precision_dtype": "fp16",
            "enable_flash_attention": False,
            "enable_gradient_checkpointing": True,
            "max_memory_usage": 0.8,
            "use_cpu_offload": False,
            "use_8bit_optimizer": True,
            "use_compile": False,
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 0,
            "detection_errors": detection_errors or [],
            "fallback_used": True
        }
    
    def get_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Get current GPU memory usage information.
        
        Returns:
            Dictionary with memory usage info or None if not available
        """
        if not torch.cuda.is_available():
            return None
        
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(0) / (1024**3)   # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved,
                "utilization": reserved / total
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return None
    
    def log_hardware_info(self):
        """Log detailed hardware information."""
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        else:
            logger.info("CUDA not available, using CPU")
            
        if self.detected_config:
            logger.info("Detected Configuration:")
            for key, value in self.detected_config.items():
                logger.info(f"  {key}: {value}")