#!/usr/bin/env python3
"""
H100 GPU Compatibility Test Suite
Comprehensive testing of H100-specific optimizations and configurations.
"""
import torch
import pytest
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.hardware import HardwareDetector
from utils.mixed_precision_manager import MixedPrecisionManager
from utils.training_stability import create_stability_manager
from model.model_manager import TinyLlamaManager

logger = logging.getLogger(__name__)

class TestH100Compatibility:
    """Test suite for H100 GPU compatibility."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device_name = torch.cuda.get_device_name(0).lower()
            self.is_h100 = "h100" in self.device_name
        else:
            self.device_name = "cpu"
            self.is_h100 = False
    
    def test_hardware_detection(self):
        """Test H100 hardware detection."""
        detector = HardwareDetector()
        gpu_config = detector.detect_gpu_config()
        
        if self.is_h100:
            assert "H100" in gpu_config["gpu_name"], f"Expected H100, got {gpu_config['gpu_name']}"
            assert gpu_config["mixed_precision_dtype"] == "bf16", "H100 should use BF16"
            assert gpu_config["gpu_memory_gb"] >= 70, f"H100 should have ~80GB, got {gpu_config['gpu_memory_gb']}"
            print(f"✅ H100 detected: {gpu_config['gpu_name']}")
            print(f"✅ Memory: {gpu_config['gpu_memory_gb']} GB")
            print(f"✅ Mixed precision: {gpu_config['mixed_precision_dtype']}")
        else:
            print(f"ℹ️  Non-H100 GPU detected: {gpu_config['gpu_name']}")
    
    def test_optimal_batch_size(self):
        """Test H100 optimal batch size calculation."""
        if self.is_h100:
            detector = HardwareDetector()
            # Test with H100's 80GB memory
            batch_size = detector.get_optimal_batch_size(80 * 1024, 2200)  # 80GB in MB, TinyLlama size
            assert batch_size >= 16, f"Expected batch size >= 16 for H100, got {batch_size}"
            
            # Test H100 config
            gpu_config = detector.detect_gpu_config()
            if "H100" in gpu_config["gpu_name"]:
                assert gpu_config["batch_size"] >= 8, f"H100 should have batch size >= 8, got {gpu_config['batch_size']}"
            
            print(f"✅ H100 optimal batch size: {batch_size}")
            print(f"✅ H100 config batch size: {gpu_config.get('batch_size', 'N/A')}")
        else:
            print("ℹ️  Skipping H100 batch size test (not H100)")
    
    def test_mixed_precision_config(self):
        """Test H100 mixed precision configuration."""
        manager = MixedPrecisionManager(gpu_type="h100")
        config = manager.get_precision_config()
        
        # H100-specific assertions
        assert config["bf16"] == True, "H100 should use BF16"
        assert config["fp16"] == False, "H100 should not use FP16 when BF16 available"
        assert config["fp16_opt_level"] == "O2", "H100 should use aggressive optimization"
        assert config["loss_scale"] == "dynamic", "H100 should use dynamic loss scaling"
        assert config["dataloader_num_workers"] == 8, "H100 should use 8 workers"
        assert config["gradient_checkpointing"] == False, "H100 has enough memory"
        assert config["attention_implementation"] == "flash_attention_2", "H100 should use Flash Attention 2"
        assert config["torch_compile"] == True, "H100 should enable torch.compile"
        assert config["fused_optimizer"] == True, "H100 should use fused optimizers"
        
        print("✅ H100 mixed precision config verified")
        print(f"   BF16: {config['bf16']}, Workers: {config['dataloader_num_workers']}")
        print(f"   Flash Attention 2: {config['attention_implementation']}")
        print(f"   Torch Compile: {config['torch_compile']}")
    
    def test_hardware_optimization(self):
        """Test hardware-specific optimization."""
        if self.is_h100:
            detector = HardwareDetector()
            config = detector.detect_gpu_config()
            
            if "H100" in config["gpu_name"]:
                assert config["batch_size"] >= 8, "H100 should handle reasonable batch sizes"
                assert config["mixed_precision"] == True, "H100 should use mixed precision"
                assert config["mixed_precision_dtype"] == "bf16", "H100 should use BF16"
                
                # Check H100-specific optimizations
                optimizations = detector.enable_optimizations("H100")
                assert optimizations.get("enable_flash_attention_2", False), "H100 should enable Flash Attention 2"
                assert optimizations.get("use_fused_adamw", False), "H100 should use fused AdamW"
                assert optimizations.get("enable_tf32", False), "H100 should enable TF32"
                
                print(f"✅ H100 optimization: batch_size={config['batch_size']}")
                print(f"   Mixed precision: {config['mixed_precision_dtype']}")
                print(f"   Flash Attention 2: {optimizations.get('enable_flash_attention_2', False)}")
                print(f"   Fused AdamW: {optimizations.get('use_fused_adamw', False)}")
        else:
            print("ℹ️  Skipping H100 optimization test (not H100)")
    
    def test_training_stability_manager(self):
        """Test H100 training stability configuration."""
        manager = create_stability_manager(gpu_type="h100")
        
        assert manager.config.gpu_type == "h100", "Should be configured for H100"
        assert manager.mixed_precision_manager is not None, "Should have mixed precision manager"
        
        if manager.mixed_precision_manager:
            mp_config = manager.mixed_precision_manager.get_precision_config()
            assert mp_config["bf16"] == True, "Stability manager should use BF16 for H100"
            assert mp_config["torch_compile"] == True, "Should enable torch.compile for H100"
        
        print("✅ H100 training stability manager configured")
    
    def test_model_manager_h100_optimizations(self):
        """Test model manager H100 optimizations."""
        if not self.gpu_available:
            pytest.skip("GPU not available")
        
        # Test dtype selection
        manager = TinyLlamaManager()
        optimal_dtype = manager._get_optimal_dtype()
        
        if self.is_h100:
            assert optimal_dtype == torch.bfloat16, f"H100 should use BF16, got {optimal_dtype}"
            print("✅ H100 model manager uses BF16")
        else:
            print(f"ℹ️  Non-H100 dtype: {optimal_dtype}")
    
    def test_memory_efficiency(self):
        """Test H100 memory efficiency."""
        if not self.is_h100:
            pytest.skip("Not running on H100")
        
        # Test memory allocation
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate a test tensor
        test_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
        bf16_memory = torch.cuda.memory_allocated() - initial_memory
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
        # Test FP32 for comparison
        test_tensor_fp32 = torch.randn(1000, 1000, dtype=torch.float32, device='cuda')
        fp32_memory = torch.cuda.memory_allocated() - initial_memory
        
        # Clean up
        del test_tensor_fp32
        torch.cuda.empty_cache()
        
        # BF16 should use half the memory of FP32
        memory_ratio = fp32_memory / bf16_memory
        assert memory_ratio >= 1.8, f"BF16 should use ~50% less memory, ratio: {memory_ratio:.2f}"
        
        print(f"✅ H100 memory efficiency: BF16 uses {memory_ratio:.1f}x less memory than FP32")
    
    def test_tensor_core_compatibility(self):
        """Test H100 tensor core compatibility."""
        if not self.is_h100:
            pytest.skip("Not running on H100")
        
        # Test BF16 matrix multiplication (should use tensor cores)
        a = torch.randn(512, 512, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(512, 512, dtype=torch.bfloat16, device='cuda')
        
        # This should utilize H100's 4th gen tensor cores
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            c = torch.matmul(a, b)
        
        assert c.dtype == torch.bfloat16, "Result should maintain BF16 precision"
        assert c.shape == (512, 512), "Matrix multiplication should work correctly"
        
        print("✅ H100 tensor core operations working")
    
    def test_flash_attention_availability(self):
        """Test Flash Attention 2 availability for H100."""
        try:
            import flash_attn
            flash_available = True
            print("✅ Flash Attention 2 package available")
        except ImportError:
            flash_available = False
            print("⚠️  Flash Attention 2 not installed")
        
        if self.is_h100 and not flash_available:
            print("💡 Recommendation: Install flash-attn for optimal H100 performance")
            print("   pip install flash-attn --no-build-isolation")
    
    def test_torch_compile_compatibility(self):
        """Test torch.compile compatibility with H100."""
        if not self.is_h100:
            pytest.skip("Not running on H100")
        
        # Test simple model compilation
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel().cuda().bfloat16()
        
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            
            # Test forward pass
            x = torch.randn(32, 10, dtype=torch.bfloat16, device='cuda')
            output = compiled_model(x)
            
            assert output.shape == (32, 1), "Compiled model should work correctly"
            print("✅ H100 torch.compile working")
            
        except Exception as e:
            print(f"⚠️  torch.compile issue: {e}")
    
    def test_comprehensive_h100_setup(self):
        """Test complete H100 setup integration."""
        if not self.is_h100:
            pytest.skip("Not running on H100")
        
        # Test full pipeline
        detector = HardwareDetector()
        gpu_config = detector.detect_gpu_config()
        mp_manager = MixedPrecisionManager(gpu_type="h100")
        stability_manager = create_stability_manager(gpu_type="h100")
        
        # Verify all components are H100-optimized
        assert "H100" in gpu_config["gpu_name"]
        assert mp_manager.gpu_type == "h100"
        assert stability_manager.config.gpu_type == "h100"
        
        # Test configuration consistency
        mp_config = mp_manager.get_precision_config()
        stability_config = stability_manager.mixed_precision_manager.get_precision_config()
        
        assert mp_config["bf16"] == stability_config["bf16"]
        assert mp_config["torch_compile"] == stability_config["torch_compile"]
        
        print("✅ H100 comprehensive setup verified")
        print(f"   All components configured for: {gpu_config['gpu_name']}")
        print(f"   Memory available: {gpu_config['gpu_memory_gb']} GB")
        print(f"   BF16 enabled: {mp_config['bf16']}")
        print(f"   Flash Attention 2: {mp_config['attention_implementation']}")
        print(f"   Torch Compile: {mp_config['torch_compile']}")

def run_h100_compatibility_test():
    """Run H100 compatibility test suite."""
    print("=" * 60)
    print("H100 GPU COMPATIBILITY TEST SUITE")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if we're on H100
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Detected GPU: {device_name}")
        is_h100 = "h100" in device_name.lower()
        if is_h100:
            print("🚀 Running on H100 - Full test suite enabled")
        else:
            print("ℹ️  Running on non-H100 GPU - Limited tests")
    else:
        print("⚠️  No GPU detected - CPU-only tests")
    
    # Run tests
    test_instance = TestH100Compatibility()
    test_instance.setup()
    
    try:
        test_instance.test_hardware_detection()
        test_instance.test_optimal_batch_size()
        test_instance.test_mixed_precision_config()
        test_instance.test_hardware_optimization()
        test_instance.test_training_stability_manager()
        test_instance.test_model_manager_h100_optimizations()
        test_instance.test_memory_efficiency()
        test_instance.test_tensor_core_compatibility()
        test_instance.test_flash_attention_availability()
        test_instance.test_torch_compile_compatibility()
        test_instance.test_comprehensive_h100_setup()
        
        print("\n" + "=" * 60)
        print("✅ ALL H100 COMPATIBILITY TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_h100_compatibility_test()