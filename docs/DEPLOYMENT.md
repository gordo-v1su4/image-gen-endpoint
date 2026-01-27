# Deployment Guide

## Coolify Volume Configuration

To ensure models are persisted across deployments and aren't re-downloaded every time, you **MUST** configure a persistent volume in Coolify.

### Step 1: Add Persistent Volume in Coolify

1. Go to your application in Coolify
2. Navigate to **Storage** tab
3. Click **Add Volume**
4. Configure the volume:
   - **Source**: `/data/models` (or any persistent path on your host)
   - **Destination**: `/app/models`
   - **Name**: `imagegen-models`

### Step 2: Environment Variables

Set these environment variables in Coolify:

```env
# Models path (matches volume destination)
MODELS_PATH=/app/models

# HuggingFace cache directories
HF_HOME=/app/models/huggingface
TORCH_HOME=/app/models/torch

# Enable model downloads on first run
DOWNLOAD_MODELS=true

# Optional: HuggingFace token for private models
HF_TOKEN=your_hf_token_here

# CUDA settings
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
```

### Step 3: First Deployment

On the first deployment with `DOWNLOAD_MODELS=true`:
- Models will be downloaded to `/app/models` (your persistent volume)
- This will take 10-30 minutes depending on your connection
- Models will be cached for future deployments

### Step 4: Subsequent Deployments

After the first deployment, the startup script will:
1. Check if models exist in the persistent volume
2. Skip downloads if models are already cached
3. Only download missing models

**Result**: Deployments will be much faster (seconds instead of minutes)!

## Volume Structure

Your persistent volume will look like this:

```
/app/models/
├── qwen-2512/              # Qwen-Image-2512 model (~10GB)
├── lightning-loras/        # Lightning LoRAs (~2GB)
├── flux-klein-4b/          # FLUX Klein 4B (~8GB)
├── huggingface/            # HF cache
└── torch/                  # PyTorch cache
```

## Verifying Persistent Storage

Check the logs on deployment. You should see:

```
🔍 Checking for cached models...
  ✅ Qwen-Image-2512 already cached at /app/models/qwen-2512
  ✅ Lightning LoRAs already cached at /app/models/lightning-loras
  ✅ FLUX.2-klein-4B already cached at /app/models/flux-klein-4b
```

If you see `⬇️ not found, will download` then the volume isn't configured correctly.

## Troubleshooting

### Models Keep Re-downloading

**Problem**: Models download on every deployment

**Solution**:
1. Check that the volume is correctly mounted in Coolify
2. Verify the source path exists and has correct permissions
3. Ensure `MODELS_PATH` environment variable matches volume destination

### Out of Disk Space

**Problem**: Not enough space for models

**Solution**:
- Qwen-Image-2512: ~10GB
- Lightning LoRAs: ~2GB
- FLUX Klein 4B: ~8GB
- Total: ~20GB minimum recommended

Ensure your persistent volume has at least 25GB free space.

### Permission Errors

**Problem**: Container can't write to volume

**Solution**:
```bash
# On the host machine
sudo chown -R 1000:1000 /data/models
sudo chmod -R 755 /data/models
```

The container runs as user `appuser` (UID 1000).

## Docker Compose Example

If deploying with docker-compose:

```yaml
version: '3.8'

services:
  imagegen:
    image: gordov1su4/imagegen-endpoint:latest
    volumes:
      - models-cache:/app/models
    environment:
      - MODELS_PATH=/app/models
      - HF_HOME=/app/models/huggingface
      - TORCH_HOME=/app/models/torch
      - DOWNLOAD_MODELS=true
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  models-cache:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/models
```

## Manual Volume Management

### Backup Models

```bash
# On host machine
tar -czf models-backup.tar.gz /data/models/
```

### Restore Models

```bash
# On host machine
tar -xzf models-backup.tar.gz -C /
```

### Pre-download Models

To speed up first deployment, pre-download models manually:

```bash
# On host machine with GPU
docker run -v /data/models:/app/models \
  -e MODELS_PATH=/app/models \
  -e DOWNLOAD_MODELS=true \
  -e HF_TOKEN=your_token \
  --gpus all \
  gordov1su4/imagegen-endpoint:latest \
  /bin/bash -c "./scripts/startup.sh && sleep 10"
```

This downloads models before the actual deployment.
