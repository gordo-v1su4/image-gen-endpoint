# Image Generation API

FastAPI-based image generation service using Qwen-Image-2512 (GGUF) and FLUX models.

## Features

- Real AI-powered text-to-image generation with Qwen-Image-2512 (Q4_K_M GGUF)
- Optimized for NVIDIA RTX 4090 (24GB VRAM)
- Multiple aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
- OpenAI-compatible API format
- FastAPI REST API

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -e .

# Run server
uvicorn app.main:app --reload
```

### Docker Deployment
```bash
docker build -t image-gen-api .
docker run --gpus all -p 8000:8000 image-gen-api
```

### API Usage
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2512-gguf",
    "prompt": "a beautiful sunset",
    "width": 1328,
    "height": 1328
  }' \
  --output image.png
```

## Documentation

Full documentation available in [docs/](docs/):
- [Deployment Guide](docs/DEPLOYMENT.md)
- [GGUF Implementation](docs/GGUF_IMPLEMENTATION_PLAN.md)
- [VRAM Analysis](docs/VRAM_ANALYSIS.md)

## Requirements

- NVIDIA GPU with 24GB+ VRAM (RTX 4090 recommended)
- CUDA 12.0+
- Docker with NVIDIA Container Runtime

## License

See LICENSE file for details.
