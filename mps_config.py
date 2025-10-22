# MPS Configuration for M3 Ultra Mac Studio
# This file contains optimal settings for Apple Silicon with Metal Performance Shaders

import os
import torch

def configure_mps_for_m3_ultra():
    """Configure MPS for optimal performance on M3 Ultra Mac Studio"""
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this system")
        return False
    
    print("🍎 Configuring MPS for M3 Ultra Mac Studio...")
    
    # Core MPS environment variables
    mps_env_vars = {
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Use all available memory
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',         # Enable fallback to CPU if needed
        'PYTORCH_TUNABLE_OPs': '1',                 # Enable tunable operations
        'PYTORCH_MPS_ALLOCATOR_POLICY': '1',         # Enable allocator policy
        'PYTORCH_MPS_ALLOCATOR_CACHE_SIZE': '0',     # Disable cache size limit
        'PYTORCH_MPS_LOG_LEVEL': 'INFO',             # Enable logging for debugging
        'PYTORCH_MPS_BACKEND': '1',                  # Force MPS backend
    }
    
    # Set environment variables
    for key, value in mps_env_vars.items():
        os.environ.setdefault(key, value)
        print(f"✅ {key} = {value}")
    
    # Additional PyTorch optimizations
    try:
        # Enable memory efficient attention if available
        if hasattr(torch.backends.mps, 'enable_memory_efficient_attention'):
            torch.backends.mps.enable_memory_efficient_attention(True)
            print("✅ Memory efficient attention enabled")
        
        # Set default device to MPS
        torch.set_default_device('mps')
        print("✅ Default device set to MPS")
        
    except Exception as e:
        print(f"⚠️  Could not set some MPS optimizations: {e}")
    
    print("🚀 MPS configuration complete!")
    return True

def get_optimal_device():
    """Get the optimal device for this system"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        configure_mps_for_m3_ultra()
        return "mps"
    else:
        return "cpu"

def print_system_info():
    """Print system information for debugging"""
    print("\n📊 System Information:")
    print(f"   - PyTorch version: {torch.__version__}")
    print(f"   - CUDA available: {torch.cuda.is_available()}")
    print(f"   - MPS available: {torch.backends.mps.is_available()}")
    print(f"   - MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        try:
            # Test MPS functionality
            test_tensor = torch.randn(10, 10, device='mps')
            print(f"   - MPS test tensor: {test_tensor.device}")
            print("✅ MPS functionality verified")
        except Exception as e:
            print(f"❌ MPS test failed: {e}")
    
    print()

if __name__ == "__main__":
    print_system_info()
    device = get_optimal_device()
    print(f"🎯 Selected device: {device}")
