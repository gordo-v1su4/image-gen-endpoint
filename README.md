# Image Generation API

Production-ready FastAPI image generation service powered by **diffusers** with 4-step distilled models.

## Features

- **4-step fast generation** with distilled/Lightning models
- **Qwen-Image-2512 FP8 Lightning** - High quality, 4 steps
- **FLUX.2 Klein 4B** - Fast distilled generation, 4 steps
- **OpenAI-compatible API** format
- **Automatic model caching** via HuggingFace Hub
- **VRAM management** with auto-clearing between generations

## Available Models

| Model ID | Steps | VRAM | Description |
|----------|-------|------|-------------|
| `qwen-2512-fp8-4step` | 4 | ~20GB | Qwen FP8 with 4-step distillation (default) |
| `flux-klein-4b` | 4 | ~13GB | FLUX.2 Klein 4B distilled |

### Model Details

**qwen-2512-fp8-4step:**
- **Source:** [lightx2v/Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)
- **File:** `qwen_image_2512_fp8_e4m3fn_scaled_4steps_v1.0.safetensors` (20.5 GB)
- **Format:** FP8 (e4m3fn) quantized with 4-step Lightning distillation baked in
- **Max Resolution:** 1664x1664

**flux-klein-4b:**
- **Source:** [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- **Format:** Native 4-step distilled model
- **Max Resolution:** 1280x1280

## Supported Resolutions

**Qwen Lightning:**
| Aspect Ratio | Resolution |
|--------------|------------|
| 1:1 | 1328x1328 |
| 16:9 | 1664x928 |
| 9:16 | 928x1664 |

**FLUX Klein:**
| Aspect Ratio | Resolution |
|--------------|------------|
| 1:1 | 1024x1024 |
| 16:9 | 1280x720 |
| 9:16 | 720x1280 |

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

Models download automatically from HuggingFace on first use (~30GB total).

### API Usage

**Generate with Qwen Lightning (default):**
```bash
curl -X POST https://your-server.com/v1/images/create \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic lion in the African savanna at golden hour",
    "width": 1328,
    "height": 1328,
    "steps": 4,
    "guidance_scale": 1.0
  }'
```

**Generate with FLUX Klein:**
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
  "model": "qwen-2512-fp8-4step"
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

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent with 20GB+ VRAM
- **CUDA**: 12.0+
- **Docker**: With NVIDIA Container Toolkit configured
- **Storage**: ~40GB for cached models

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
| `HF_HOME` | `/opt/models/huggingface` | HuggingFace cache directory |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |

## Architecture

```
┌─────────────────────────────────────┐
│         FastAPI Application         │
├─────────────────────────────────────┤
│         Model Manager               │
│  ┌─────────────┐ ┌───────────────┐  │
│  │Qwen Lightning│ │ FLUX Klein   │  │
│  │  (diffusers) │ │ (diffusers)  │  │
│  └─────────────┘ └───────────────┘  │
├─────────────────────────────────────┤
│         HuggingFace Hub             │
│      (automatic model caching)      │
└─────────────────────────────────────┘
```

## Model Sources

- **Qwen FP8 Lightning**: [lightx2v/Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)
- **FLUX Klein 4B**: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

## License

Apache 2.0 - See LICENSE for details.
