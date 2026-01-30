# Diffusers-only image generation API
# Supports: Qwen-Image-2512 Lightning (4-step) with FP8 quantization
# Using devel image for nvcc (required by optimum-quanto FP8)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Build args - models download at container start, not build
ARG DOWNLOAD_MODELS=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (curl, git, g++ for CUDA extension compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates g++ \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Target RTX 4090 (Ada Lovelace, compute 8.9) for Quanto CUDA compilation
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Install uv for Python and dependency management (direct binary download)
# Download uv binary directly to avoid installer script segfault
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        UV_ARCH="x86_64-unknown-linux-gnu"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        UV_ARCH="aarch64-unknown-linux-gnu"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    mkdir -p /root/.local/bin && \
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}.tar.gz | tar -xz --strip-components=1 -C /root/.local/bin && \
    chmod +x /root/.local/bin/uv /root/.local/bin/uvx
ENV PATH="/root/.local/bin:${PATH}"

# Install Python 3.10 using uv
RUN uv python install 3.10
# Create symlinks to uv's Python (uv installs to ~/.local/share/uv/python/cpython-3.10.x/bin/)
RUN for py in /root/.local/share/uv/python/cpython-3.10.*/bin/python3.10; do \
        if [ -f "$py" ]; then \
            ln -sf "$py" /usr/local/bin/python3 && \
            ln -sf "$py" /usr/local/bin/python && \
            break; \
        fi; \
    done

# Set HF cache directory (will be mounted as volume)
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p /root/.cache/huggingface

# Copy application
WORKDIR /app
COPY . /app

# Fix line endings and make scripts executable
RUN sed -i 's/\r$//' /app/scripts/entrypoint.sh /app/scripts/download_models.py && \
    chmod +x /app/scripts/entrypoint.sh /app/scripts/download_models.py

# Install Python dependencies using uv (creates venv at .venv)
RUN uv sync --no-cache
ENV PATH="/app/.venv/bin:${PATH}"

# Verify peft is installed (required for LoRA)
RUN python -c "import peft; print('peft version:', peft.__version__)"

# Pre-download models during build (optional, ~40GB total)
# Set DOWNLOAD_MODELS=false to skip and download at runtime instead
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
        echo "Pre-downloading models..." && \
        python3 /app/scripts/download_models.py; \
    else \
        echo "Skipping model download (will download on first use)"; \
    fi

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
