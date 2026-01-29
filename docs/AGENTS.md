# AGENTS.md

This file provides guidance to AI assistants when working with code in this repository.

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
docker run --gpus all -p 8000:8000 -v hf_cache:/opt/models/huggingface imagegen-endpoint
docker-compose up -d
```

## Architecture

### Request Flow
```
FastAPI → Routes → Services → Model Manager → diffusers Pipeline
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
- HuggingFace repo IDs
- VRAM requirements
- Default steps and guidance scale
- Supported resolutions

Current models:
- `qwen-2512-lightning` - Qwen FP8 Lightning (4-step, default)
- `flux-klein-4b` - FLUX.2 Klein 4B distilled (4-step)

### Image I/O Pattern

All image transfers use base64 encoding via `app/utils/image_utils.py`. Upload accepts both `multipart/form-data` (file) and base64 in form field.

## Deployment

- Docker image built from `Dockerfile`
- Uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base image
- Models auto-download from HuggingFace on first use
- Environment: `HF_HOME`, `CUDA_VISIBLE_DEVICES`
- Models cached in `/opt/models/huggingface` volume (~40GB)

## Important Notes

- All models use diffusers library with 4-step distilled/Lightning inference
- Models download from HuggingFace Hub automatically on first use
- VRAM cleared between generations to prevent memory buildup
- Use `guidance_scale=1.0` for Lightning/distilled models
- Always use `.yaml` extension (not `.yml`) for Coolify compatibility
