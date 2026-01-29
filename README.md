# Image Generation API

Production-ready FastAPI image generation service powered by **Qwen-Image-2512 GGUF** and **FLUX.2 Klein 4B** models.

## Features

- **Real AI image generation** using quantized GGUF models via stable-diffusion.cpp
- **FLUX.2 Klein 4B** support via diffusers (4-step generation)
- **AI-powered image editing** with Qwen-Image-Edit-2511
- **Optimized for RTX 4090** (24GB VRAM)
- **OpenAI-compatible API** format
- **Multiple aspect ratios**: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
- **VRAM management**: Auto-clearing between generations, queue lock for sequential processing

## Available Models

| Model | Type | VRAM | Steps | Description |
|-------|------|------|-------|-------------|
| `qwen-2512-gguf` | GGUF | ~19GB | 40 | High-quality text-to-image (default) |
| `qwen-edit-gguf` | GGUF | ~19GB | 40 | AI-powered image editing |
| `flux-klein-4b` | Diffusers | ~13GB | 4 | Fast 4-step generation |

## Supported Resolutions

| Aspect Ratio | Resolution |
|--------------|------------|
| 1:1 | 1328x1328 |
| 16:9 | 1664x928 |
| 9:16 | 928x1664 |
| 4:3 | 1472x1104 |
| 3:4 | 1104x1472 |

## Quick Start

### Docker Deployment (Recommended)

```bash
# Clone repository
git clone https://github.com/gordo-v1su4/image-gen-endpoint.git
cd image-gen-endpoint

# Build and run with GPU support
docker compose up --build -d

# View logs
docker compose logs -f
```

Models (~30GB) download automatically on first run.

### API Usage

**Generate Image:**
```bash
curl -X POST https://your-server.com/v1/images/create \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic lion in the African savanna at golden hour",
    "model": "qwen-2512-gguf",
    "width": 1328,
    "height": 1328,
    "steps": 25
  }'
```

**With FLUX (faster, 4 steps):**
```bash
curl -X POST https://your-server.com/v1/images/create \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cyberpunk cityscape at night",
    "model": "flux-klein-4b",
    "width": 1024,
    "height": 1024,
    "steps": 4
  }'
```

**Response Format (OpenAI-compatible):**
```json
{
  "data": [{"b64_json": "...base64 encoded image..."}],
  "created": 1706500000,
  "model": "qwen-2512-gguf"
}
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with GPU info |
| `/v1/images/create` | POST | Generate image from prompt |
| `/v1/images/edit` | POST | Edit image with AI |
| `/docs` | GET | Swagger UI documentation |
| `/test-ui` | GET | Web-based test interface |

## Requirements

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **CUDA**: 12.0+
- **Docker**: With NVIDIA Container Toolkit configured
- **Storage**: ~35GB for models

### Server Setup

```bash
# Install NVIDIA Container Toolkit
sudo apt-get install nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GGUF_MODELS_PATH` | `/opt/models/qwen-gguf` | Path to GGUF model files |
| `SD_CLI_PATH` | `/opt/stable-diffusion.cpp/build/bin/sd-cli` | Path to sd-cli binary |
| `OFFLOAD_TO_CPU` | `false` | Offload to CPU for lower VRAM |
| `SD_VERBOSE` | `false` | Enable verbose sd-cli output |
| `DOWNLOAD_MODELS` | `true` | Auto-download models on startup |

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [GGUF Quick Start](docs/DEPLOY_GGUF_QUICKSTART.md)
- [GGUF Implementation](docs/README_GGUF.md)
- [VRAM Analysis](docs/VRAM_ANALYSIS.md)

## License

Apache 2.0 - See LICENSE for details.
