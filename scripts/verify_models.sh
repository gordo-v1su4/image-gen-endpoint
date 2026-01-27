#!/bin/bash
# Verify all GGUF models downloaded correctly

set -e

MODELS_DIR="/opt/models/qwen-gguf"

echo "Verifying GGUF models..."

# Check files exist
test -f "$MODELS_DIR/qwen-image-2512-Q4_K_M.gguf" || { echo "❌ Main model missing"; exit 1; }
test -f "$MODELS_DIR/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf" || { echo "❌ Text encoder missing"; exit 1; }
test -f "$MODELS_DIR/qwen_image_vae.safetensors" || { echo "❌ VAE missing"; exit 1; }

# Check file sizes (approximate)
main_size=$(stat -c%s "$MODELS_DIR/qwen-image-2512-Q4_K_M.gguf")
encoder_size=$(stat -c%s "$MODELS_DIR/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf")
vae_size=$(stat -c%s "$MODELS_DIR/qwen_image_vae.safetensors")

# Minimum sizes (slightly less than expected to account for rounding)
MIN_MAIN=13000000000     # 13GB
MIN_ENCODER=4000000000   # 4GB
MIN_VAE=240000000        # 240MB

if [ $main_size -lt $MIN_MAIN ]; then
    echo "❌ Main model too small: $main_size bytes"
    exit 1
fi

if [ $encoder_size -lt $MIN_ENCODER ]; then
    echo "❌ Text encoder too small: $encoder_size bytes"
    exit 1
fi

if [ $vae_size -lt $MIN_VAE ]; then
    echo "❌ VAE too small: $vae_size bytes"
    exit 1
fi

echo "✅ All models verified successfully"
echo "  Main model: $(numfmt --to=iec-i --suffix=B $main_size)"
echo "  Text encoder: $(numfmt --to=iec-i --suffix=B $encoder_size)"
echo "  VAE: $(numfmt --to=iec-i --suffix=B $vae_size)"
