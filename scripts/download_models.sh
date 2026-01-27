#!/bin/bash
set -e

MODELS_DIR="/opt/models/qwen-gguf"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "🔍 Checking for required models..."

# Required models for generation and editing
GENERATION_MODEL="qwen-image-2512-Q4_K_M.gguf"
EDIT_MODEL="qwen-image-edit-2511-Q4_K_M.gguf"
TEXT_ENCODER="Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
VAE="qwen_image_vae.safetensors"

# Check if all models exist
if [ -f "$GENERATION_MODEL" ] && \
   [ -f "$EDIT_MODEL" ] && \
   [ -f "$TEXT_ENCODER" ] && \
   [ -f "$VAE" ]; then
    echo "✅ All models already present. Skipping download."
    ls -lh
    exit 0
fi

echo "📥 Starting model downloads (this may take a while on first run)..."

# Download generation model if missing
if [ ! -f "$GENERATION_MODEL" ]; then
    echo "⬇️  Downloading Qwen-Image-2512 Q4_K_M (13.2 GB)..."
    curl -L -C - -o "$GENERATION_MODEL" \
        "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/$GENERATION_MODEL"
    echo "✅ Generation model downloaded"
fi

# Download edit model if missing
if [ ! -f "$EDIT_MODEL" ]; then
    echo "⬇️  Downloading Qwen-Image-Edit-2511 Q4_K_M (13.2 GB)..."
    curl -L -C - -o "$EDIT_MODEL" \
        "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/$EDIT_MODEL"
    echo "✅ Edit model downloaded"
fi

# Download text encoder if missing
if [ ! -f "$TEXT_ENCODER" ]; then
    echo "⬇️  Downloading Qwen2.5-VL text encoder (4.5 GB)..."
    curl -L -C - -o "$TEXT_ENCODER" \
        "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/$TEXT_ENCODER"
    echo "✅ Text encoder downloaded"
fi

# Download VAE if missing
if [ ! -f "$VAE" ]; then
    echo "⬇️  Downloading VAE (243 MB)..."
    curl -L -C - -o "$VAE" \
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/$VAE"
    echo "✅ VAE downloaded"
fi

# Verify all models
echo "🔍 Verifying downloads..."
test -f "$GENERATION_MODEL" || { echo "❌ Generation model missing"; exit 1; }
test -f "$EDIT_MODEL" || { echo "❌ Edit model missing"; exit 1; }
test -f "$TEXT_ENCODER" || { echo "❌ Text encoder missing"; exit 1; }
test -f "$VAE" || { echo "❌ VAE missing"; exit 1; }

echo "✅ All models verified successfully"
ls -lh
