# Build stage for stable-diffusion.cpp
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS sd-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (per Unsloth docs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Clone and build stable-diffusion.cpp with CUDA support
# Use -j4 instead of -j$(nproc) to avoid OOM during CUDA compilation
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp /opt/stable-diffusion.cpp \
    && cd /opt/stable-diffusion.cpp \
    && mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON \
    && cmake --build . -j4

# Verify sd-cli was built
RUN ls -la /opt/stable-diffusion.cpp/build/bin/sd-cli


# Runtime stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python 3.10 and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    curl wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Copy built sd-cli from builder stage
COPY --from=sd-builder /opt/stable-diffusion.cpp/build/bin/sd-cli /opt/stable-diffusion.cpp/build/bin/sd-cli

# Set CUDA environment for runtime
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install uv for faster dependency installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create models directory
RUN mkdir -p /opt/models/qwen-gguf

# Copy application
WORKDIR /app
COPY . /app

# Make scripts executable
RUN chmod +x /app/scripts/download_models.sh /app/scripts/entrypoint.sh

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    uv pip install --system --python python3.10 --no-cache -e .

# Environment variables for GGUF models
ENV GGUF_MODELS_PATH=/opt/models/qwen-gguf \
    SD_CLI_PATH=/opt/stable-diffusion.cpp/build/bin/sd-cli \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Use entrypoint script that downloads models on first run
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
