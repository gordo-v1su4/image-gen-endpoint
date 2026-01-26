# 🎨 ImageGen Endpoint

Production-ready image generation and editing API powered by state-of-the-art AI models.

## Features

- **Image Generation**: Text-to-image with Qwen-Image-2512 and FLUX.2 Klein
- **Image Editing**: AI-powered editing with Qwen-Image-2512
- **Basic Operations**: Resize, crop, rotate, filters (Pillow-based)
- **Fast Inference**: Lightning LoRAs for 4-8 step generation
- **GPU Optimized**: FP8 quantization for RTX 4090 (24GB VRAM)
- **Auto-deploy**: GitHub Actions → Coolify integration

## Supported Models

| Model | Type | VRAM | License |
|-------|------|------|---------|
| Qwen-Image-2512 | Generation/Editing | ~16GB (FP8) | Apache 2.0 |
| FLUX.2-klein-4B | Generation | ~8GB | Apache 2.0 |
| FLUX.2-klein-9B | Generation | ~12GB | Non-commercial |

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/gordo-v1su4/image-gen-endpoint.git
cd image-gen-endpoint

# Create virtual environment with UV
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Test UI

Access the test UI at `/test-ui/` while the server is running:
- **Local**: http://localhost:8000/test-ui/
- **Production**: https://image.v1su4.com/test-ui/
- **Modern UI**: https://image.v1su4.com/test-ui/webapp.html

### Production Deployment

For production deployment with Coolify and persistent model storage, see [DEPLOYMENT.md](DEPLOYMENT.md).

## API Endpoints

### Interactive Documentation

Access the interactive API documentation:
- **Swagger UI**: `/docs` - Interactive API explorer
- **ReDoc**: `/redoc` - Alternative documentation format
- **OpenAPI Schema**: `/openapi.json` - Raw specification

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_total": 24.0,
  "gpu_memory_used": 0.5
}
```

### Generate Image
```bash
POST /v1/images/create
Content-Type: application/json

{
  "prompt": "A beautiful sunset over mountains",
  "model": "qwen-2512",
  "width": 1328,
  "height": 1328,
  "steps": 8,
  "use_lightning": true
}
```

### Edit Image
```bash
POST /v1/images/edit
Content-Type: multipart/form-data

image: <file>
operations: [{"type": "resize", "params": {"width": 512, "height": 512}}]
prompt: "Make it look like anime"
model: "qwen-2512"
```

### Available Edit Operations

- `resize`: `{"width": int, "height": int}`
- `crop`: `{"x": int, "y": int, "width": int, "height": int}`
- `rotate`: `{"degrees": float, "expand": bool}`
- `flip`: `{"direction": "horizontal"|"vertical"}`
- `filter`: `{"type": "blur"|"sharpen"|"grayscale", "strength": float}`
- `adjust`: `{"brightness": float, "contrast": float, "saturation": float}`

## Docker Deployment

### Build and Run
```bash
docker build -t imagegen-endpoint .
docker run --gpus all -p 8000:8000 -v /path/to/models:/app/models imagegen-endpoint
```

### Docker Compose
```bash
docker-compose up -d
```

## Coolify Deployment

1. **Create Service**: Add new service from Git repository
2. **Configure Build**: Dockerfile is auto-detected
3. **Set Environment Variables**:
   - `HF_TOKEN`: Your HuggingFace token (for gated models)
   - `DOWNLOAD_MODELS`: Set to `true` on first deploy
4. **Configure Storage**: Mount persistent volume to `/app/models`
5. **Enable GPU**: Ensure NVIDIA Container Toolkit is installed on host
6. **Set Health Check**: Path `/health`, interval 30s

### GitHub Actions Auto-Deploy

Add secrets to your repository:
- `COOLIFY_WEBHOOK_URL`: Your Coolify webhook URL
- `COOLIFY_TOKEN`: Your Coolify API token

Pushes to `main` will automatically trigger deployment.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_PATH` | `/app/models` | Path for model storage |
| `HF_HOME` | `/app/models/huggingface` | HuggingFace cache |
| `HF_TOKEN` | - | HuggingFace API token |
| `DOWNLOAD_MODELS` | `false` | Download models on startup |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

## Performance Tips

1. **Use Lightning LoRAs**: 4-8 steps vs 40 steps = 5-10x speedup
2. **FP8 Models**: ~50% VRAM reduction with minimal quality loss
3. **FLUX Klein 4B**: Sub-second generation for previews
4. **Model Switching**: Only one model loaded at a time to maximize VRAM

## License

MIT License - see LICENSE file.

## Contributing

See CONTRIBUTING.md for guidelines.
