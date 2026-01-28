FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

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

# Install Python dependencies only
RUN python3 -m pip install --upgrade pip && \
    uv pip install --system --python python3.10 --no-cache -e .

ENV GGUF_MODELS_PATH=/opt/models/qwen-gguf \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Use entrypoint script that downloads models on first run
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
