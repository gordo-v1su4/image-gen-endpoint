# Model Implementation Notes

## What Was Implemented

### ✅ Real Model Loading (model_manager.py)
- Implemented actual model loading using `diffusers.DiffusionPipeline`
- Support for FP8 quantized models for VRAM efficiency
- Automatic GPU memory management with fallback to CPU
- Lightning LoRA support for 4-8 step fast generation
- Memory optimizations: CPU offload, VAE slicing, VAE tiling

### ✅ Real Image Generation (model_manager.py)
- Replaced placeholder with actual diffusion model inference
- Full parameter support: prompt, negative_prompt, width, height, steps, guidance_scale, seed
- Reproducible generation with seed support
- Progress logging for debugging

### ✅ Updated Endpoint (create.py)
- Replaced `generate_placeholder_image()` with `model_manager.generate_image()`
- Passes all user parameters to the model
- Maintains backward compatibility with response format

## Quick Test

### 1. Set Environment Variables
```bash
export HF_TOKEN="your_huggingface_token_here"
export CUDA_VISIBLE_DEVICES=0  # If you have GPU
```

### 2. Install Dependencies
```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 3. Run the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test Image Generation
```bash
curl -X POST http://localhost:8000/v1/images/create \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "model": "flux-klein-4b",
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "use_lightning": true
  }'
```

## Model Download Sizes

First-time model loading will download models from HuggingFace:

- **qwen-2512**: ~16GB (FP8), ~32GB (full precision)
- **flux-klein-4b**: ~8GB (FP8), ~16GB (full precision)
- **flux-klein-9b**: ~12GB (FP8), ~24GB (full precision)

Models are cached in `HF_HOME` directory (default: `~/.cache/huggingface`).

## VRAM Requirements

- **qwen-2512**: 16GB VRAM (FP8) or 32GB (full)
- **flux-klein-4b**: 8GB VRAM (FP8) or 16GB (full)
- **flux-klein-9b**: 12GB VRAM (FP8) or 24GB (full)

CPU offloading is enabled automatically to reduce VRAM usage.

## Performance Tips

1. **Use FP8 Models**: Automatically selected when available (50% VRAM reduction)
2. **Enable Lightning LoRA**: Set `use_lightning: true` for 4-8 step generation
3. **GPU Memory**: Models auto-unload when VRAM is insufficient
4. **First Generation**: Takes longer due to model loading (~30-60s)
5. **Subsequent Generations**: Fast once model is loaded (~2-10s depending on steps)

## Troubleshooting

### Out of Memory (OOM)
- Reduce image dimensions (try 512x512 or 768x768)
- Use fewer steps (4-8 with Lightning)
- Ensure no other models are loaded (check `/health` endpoint)

### Model Download Fails
- Check HuggingFace token is set: `export HF_TOKEN=...`
- Verify internet connection
- Check HuggingFace model access (some models require approval)

### Slow Generation
- First generation loads the model (expected)
- Check if GPU is being used: `nvidia-smi`
- Verify CUDA is available: Check server logs for "Device: cuda"

## Next Steps

1. **Deploy to Production**: Use Docker or Coolify (see DEPLOYMENT.md)
2. **Add Model Warming**: Pre-load popular models on startup
3. **Add Caching**: Cache common generations
4. **Monitor VRAM**: Add /metrics endpoint for Prometheus

## Code Changes Summary

### app/services/model_manager.py
- `load_model()`: Implemented real diffusers pipeline loading
- `generate_image()`: Implemented real inference with torch

### app/routes/create.py
- Replaced `generate_placeholder_image()` with `model_manager.generate_image()`
- Now calls actual AI models instead of returning placeholders

### Result
- ✅ Real AI-generated images (no more placeholders!)
- ✅ Full parameter support
- ✅ GPU optimization
- ✅ Lightning LoRA support
