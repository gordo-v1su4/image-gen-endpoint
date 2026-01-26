#!/bin/bash
set -e

echo "🚀 ImageGen Endpoint Startup Script"
echo "===================================="

# Configuration
MODELS_PATH="${MODELS_PATH:-/app/models}"
HF_HOME="${HF_HOME:-$MODELS_PATH/huggingface}"
TORCH_HOME="${TORCH_HOME:-$MODELS_PATH/torch}"

export HF_HOME
export TORCH_HOME

echo "📁 Models path: $MODELS_PATH"
echo "📁 HF cache: $HF_HOME"

# Create directories
mkdir -p "$MODELS_PATH"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"

# Function to check if a model exists
check_model_exists() {
    local model_path=$1
    local model_name=$2

    if [ -d "$model_path" ] && [ "$(ls -A $model_path 2>/dev/null)" ]; then
        echo "  ✅ $model_name already cached at $model_path"
        return 0
    else
        echo "  ⬇️  $model_name not found, will download"
        return 1
    fi
}

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available - running on CPU')
"

# Download models if HF_TOKEN is set and models aren't present
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "🔑 HuggingFace token found"
    
    # Login to HuggingFace
    python3 -c "
from huggingface_hub import login
import os
login(token=os.environ.get('HF_TOKEN'), add_to_git_credential=False)
print('✅ Logged into HuggingFace')
"
fi

# Check existing models
echo ""
echo "🔍 Checking for cached models..."
check_model_exists "$MODELS_PATH/qwen-2512" "Qwen-Image-2512"
QWEN_EXISTS=$?
check_model_exists "$MODELS_PATH/lightning-loras" "Lightning LoRAs"
LIGHTNING_EXISTS=$?
check_model_exists "$MODELS_PATH/flux-klein-4b" "FLUX.2-klein-4B"
FLUX_EXISTS=$?

# Check if we should download models
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-false}"

if [ "$DOWNLOAD_MODELS" = "true" ]; then
    echo ""
    echo "📥 Downloading missing models..."

    # Download Qwen-Image-2512 FP8 (only if not cached)
    if [ $QWEN_EXISTS -ne 0 ]; then
        echo "  → Downloading Qwen-Image-2512..."
        python3 -c "
from huggingface_hub import snapshot_download
import os

models_path = os.environ.get('MODELS_PATH', '/app/models')

try:
    snapshot_download(
        'Qwen/Qwen-Image-2512',
        local_dir=f'{models_path}/qwen-2512',
        local_dir_use_symlinks=False,
        ignore_patterns=['*.md', '*.txt'],
    )
    print('    ✅ Qwen-Image-2512 downloaded')
except Exception as e:
    print(f'    ⚠️  Qwen-Image-2512: {e}')
"
    fi

    # Download Lightning LoRAs (only if not cached)
    if [ $LIGHTNING_EXISTS -ne 0 ]; then
        echo "  → Downloading Lightning LoRAs..."
        python3 -c "
from huggingface_hub import snapshot_download
import os

models_path = os.environ.get('MODELS_PATH', '/app/models')

try:
    snapshot_download(
        'lightx2v/Qwen-Image-2512-Lightning',
        local_dir=f'{models_path}/lightning-loras',
        local_dir_use_symlinks=False,
    )
    print('    ✅ Lightning LoRAs downloaded')
except Exception as e:
    print(f'    ⚠️  Lightning LoRAs: {e}')
"
    fi

    # Download FLUX Klein 4B (only if not cached)
    if [ $FLUX_EXISTS -ne 0 ]; then
        echo "  → Downloading FLUX.2-klein-4B..."
        python3 -c "
from huggingface_hub import snapshot_download
import os

models_path = os.environ.get('MODELS_PATH', '/app/models')

try:
    snapshot_download(
        'black-forest-labs/FLUX.2-klein-4b-fp8',
        local_dir=f'{models_path}/flux-klein-4b',
        local_dir_use_symlinks=False,
        ignore_patterns=['*.md', '*.txt'],
    )
    print('    ✅ FLUX.2-klein-4B downloaded')
except Exception as e:
    print(f'    ⚠️  FLUX Klein 4B: {e}')
"
    fi

    if [ $QWEN_EXISTS -eq 0 ] && [ $LIGHTNING_EXISTS -eq 0 ] && [ $FLUX_EXISTS -eq 0 ]; then
        echo "  ✅ All models already cached, skipping downloads"
    fi
else
    echo ""
    echo "ℹ️  Model download disabled (set DOWNLOAD_MODELS=true to enable)"
fi

echo ""
echo "✅ Startup complete!"
echo ""
