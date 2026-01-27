FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential pkg-config \
    python3.11 python3-pip curl wget \
    && rm -rf /var/lib/apt/lists/*

# Build stable-diffusion.cpp with RTX 4090 support (compute_89)
WORKDIR /opt
RUN echo "Building stable-diffusion.cpp for RTX 4090 (compute_89)..." && \
    git clone --recursive https://github.com/leejet/stable-diffusion.cpp && \
    cd stable-diffusion.cpp && \
    mkdir build && cd build && \
    cmake .. -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89" && \
    cmake --build . --config Release -j$(nproc) && \
    echo "Build complete!"

# Download GGUF models (all public repos, no HF token needed)
# Using -L to follow redirects, -C - to resume interrupted downloads
WORKDIR /opt/models/qwen-gguf
RUN echo "Downloading Qwen-Image-2512 Q4_K_M (13.2 GB)..." && \
    curl -L -C - -o qwen-image-2512-Q4_K_M.gguf \
    https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf && \
    echo "Downloading Qwen2.5-VL text encoder (4.5 GB)..." && \
    curl -L -C - -o Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf && \
    echo "Downloading VAE (243 MB)..." && \
    curl -L -C - -o qwen_image_vae.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors && \
    echo "Verifying downloads..." && \
    ls -lh

# Verify all models downloaded successfully
RUN test -f qwen-image-2512-Q4_K_M.gguf || { echo "❌ Main model missing"; exit 1; } && \
    test -f Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf || { echo "❌ Text encoder missing"; exit 1; } && \
    test -f qwen_image_vae.safetensors || { echo "❌ VAE missing"; exit 1; } && \
    echo "✅ All models verified successfully"

# Copy application
WORKDIR /app
COPY . /app

# Copy and run verification script
COPY scripts/verify_models.sh /tmp/
RUN chmod +x /tmp/verify_models.sh && /tmp/verify_models.sh

# Install Python dependencies
RUN pip3 install --no-cache-dir -e .

# Environment variables
ENV SD_CLI_PATH=/opt/stable-diffusion.cpp/build/bin/sd-cli
ENV GGUF_MODELS_PATH=/opt/models/qwen-gguf
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
