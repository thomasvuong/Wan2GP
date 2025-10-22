#!/bin/bash

# MPS Optimization Script for M3 Ultra Mac Studio
# This script sets optimal environment variables for Metal Performance Shaders

echo "🚀 Optimizing for M3 Ultra Mac Studio with MPS acceleration..."

# Enable MPS debugging (helps identify issues)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Allow PyTorch to use more memory (critical for 512GB RAM system)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# For better performance with large models
export PYTORCH_TUNABLE_OPs=1

# Additional MPS optimizations
export PYTORCH_MPS_ALLOCATOR_POLICY=1
export PYTORCH_MPS_ALLOCATOR_CACHE_SIZE=0

# Memory management optimizations for large RAM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable verbose MPS logging for debugging
export PYTORCH_MPS_LOG_LEVEL=INFO

# Force MPS backend
export PYTORCH_MPS_BACKEND=1

echo "✅ MPS environment variables set successfully!"
echo "📊 System Info:"
echo "   - Device: M3 Ultra Mac Studio"
echo "   - RAM: 512GB"
echo "   - GPU: 80-core GPU"
echo "   - MPS Available: $(python -c 'import torch; print(torch.backends.mps.is_available())')"
echo "   - MPS Built: $(python -c 'import torch; print(torch.backends.mps.is_built())')"

# Start the application
echo "🎬 Starting Wan2GP with MPS acceleration..."
python wgp.py --server-port 7861
