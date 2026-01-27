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

# Create models directory (models will be downloaded at runtime)
RUN mkdir -p /opt/models/qwen-gguf

# Copy application
WORKDIR /app
COPY . /app

# Make scripts executable
RUN chmod +x /app/scripts/download_models.sh /app/scripts/entrypoint.sh

# Install Python dependencies
RUN pip3 install --no-cache-dir -e .

# Environment variables
ENV SD_CLI_PATH=/opt/stable-diffusion.cpp/build/bin/sd-cli
ENV GGUF_MODELS_PATH=/opt/models/qwen-gguf
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use entrypoint script that downloads models on first run
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
