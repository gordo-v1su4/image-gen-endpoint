# Diffusers-only image generation API
# Supports: Qwen FP8 Lightning (4-step), FLUX Klein 4B (4-step)
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

# Set CUDA environment for runtime
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install uv for faster dependency installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create models directory
RUN mkdir -p /opt/models/huggingface

# Copy application
WORKDIR /app
COPY . /app

# Make scripts executable
RUN chmod +x /app/scripts/entrypoint.sh

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    uv pip install --system --python python3.10 --no-cache -e .

# Environment variables
ENV HF_HOME=/opt/models/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
