#!/bin/bash
set -e

MODELS_DIR="/opt/models/qwen-gguf"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "🔍 Checking for required models..."

# Check if all models exist
if [ -f "qwen-image-2512-Q4_K_M.gguf" ] && \
   [ -f "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf" ] && \
   [ -f "qwen_image_vae.safetensors" ]; then
    echo "✅ All models already present. Skipping download."
    ls -lh
    exit 0
fi

echo "📥 Starting model downloads (this may take a while on first run)..."

# Download main model if missing
if [ ! -f "qwen-image-2512-Q4_K_M.gguf" ]; then
    echo "⬇️  Downloading Qwen-Image-2512 Q4_K_M (13.2 GB)..."
    curl -L -C - -o qwen-image-2512-Q4_K_M.gguf \
        https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf
    echo "✅ Main model downloaded"
fi

# Download text encoder if missing
if [ ! -f "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf" ]; then
    echo "⬇️  Downloading Qwen2.5-VL text encoder (4.5 GB)..."
    curl -L -C - -o Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
        https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
    echo "✅ Text encoder downloaded"
fi

# Download VAE if missing
if [ ! -f "qwen_image_vae.safetensors" ]; then
    echo "⬇️  Downloading VAE (243 MB)..."
    curl -L -C - -o qwen_image_vae.safetensors \
        https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
    echo "✅ VAE downloaded"
fi

# Verify all models
echo "🔍 Verifying downloads..."
test -f qwen-image-2512-Q4_K_M.gguf || { echo "❌ Main model missing"; exit 1; }
test -f Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf || { echo "❌ Text encoder missing"; exit 1; }
test -f qwen_image_vae.safetensors || { echo "❌ VAE missing"; exit 1; }

echo "✅ All models verified successfully"
ls -lh
