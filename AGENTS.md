# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Build & Run Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Lint & format
ruff check app/
black app/

# Tests
pytest tests/
pytest tests/test_specific.py::test_function  # single test

# Docker
docker build -t imagegen-endpoint .
docker run --gpus all -p 8000:8000 -v $(pwd)/models:/app/models imagegen-endpoint
docker-compose up -d
```

## Architecture

### Request Flow
```
FastAPI → Routes → Services → Model Manager → AI Models
                      ↓
              Image Processor (Pillow operations)
```

### Key Components

**Routes** (`app/routes/`):
- `create.py` - POST `/v1/images/create` - text-to-image generation
- `edit.py` - POST `/v1/images/edit` - image editing with operations + optional AI prompt
- `health.py` - GET `/health` - CUDA/GPU status check

**Services** (`app/services/`):
- `model_manager.py` - Singleton `model_manager` handles loading/unloading models, VRAM management. Only one model loaded at a time. Contains `MODEL_CONFIGS` dict with all model definitions.
- `image_processor.py` - Pillow-based operations (resize, crop, rotate, filters, adjustments). `apply_edit_operations()` chains multiple operations.

**Models** (`app/models.py`):
- All Pydantic schemas. `ModelType` enum defines available models. `EditOperation` for chainable edits.

### Model Configuration

Models are defined in `MODEL_CONFIGS` in `model_manager.py`. Each entry specifies:
- HuggingFace repo IDs (base and FP8 variants)
- Lightning LoRA repos for fast inference
- VRAM requirements
- License info

Current models: `qwen-2512`, `flux-klein-4b`, `flux-klein-9b`

### Image I/O Pattern

All image transfers use base64 encoding via `app/utils/image_utils.py`. Upload accepts both `multipart/form-data` (file) and base64 in form field.

## Deployment

- Docker image: `gordov1su4/imagegen-endpoint`
- GitHub Actions workflow in `.github/workflows/deploy.yaml` builds, pushes to Docker Hub, then triggers Coolify webhook
- Required secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `COOLIFY_WEBHOOK_URL`, `COOLIFY_TOKEN`
- Environment: `MODELS_PATH`, `HF_HOME`, `HF_TOKEN`, `DOWNLOAD_MODELS`
- Models persist in `/app/models` volume (100GB+ recommended)

## Important Notes

- AI model inference is stubbed with placeholders (`generate_placeholder_image`). Search for `# TODO:` to find integration points.
- Lightning LoRAs only work with FP8 models, not GGUF quantized versions.
- The `test-ui/` directory is excluded from git for local testing only.
- Always use `.yaml` extension (not `.yml`) for Coolify compatibility.
