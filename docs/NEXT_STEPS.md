# Next Steps: Deploy GGUF to 4090

## What Just Happened

We discovered that:
1. **Qwen-2512 in bfloat16** uses **20-22 GB VRAM** (not 12 GB as configured)
2. This crashed your **5090 with 32GB** and won't work on **4090 with 24GB**
3. **Solution:** Use **GGUF Q4_K_M format** which only uses **14-16 GB**

## Documents Created

### 📋 Planning & Analysis
- **[VRAM_ANALYSIS.md](VRAM_ANALYSIS.md)** - Full breakdown of VRAM usage, what went wrong, and why GGUF fixes it
- **[GGUF_IMPLEMENTATION_PLAN.md](GGUF_IMPLEMENTATION_PLAN.md)** - Detailed technical implementation guide

### 🚀 Deployment
- **[DEPLOY_GGUF_QUICKSTART.md](DEPLOY_GGUF_QUICKSTART.md)** - Step-by-step deployment instructions for Coolify
- **[.env.gguf.example](.env.gguf.example)** - Environment variables needed for GGUF

## Quick Decision Matrix

### Should I use GGUF? ✅ YES

| Format | VRAM | Fits 4090? | Quality | Speed |
|--------|------|------------|---------|-------|
| BF16 | 20-22 GB | ❌ NO | Perfect | Fast |
| **GGUF Q4_K_M** | **14-16 GB** | ✅ **YES** | **Excellent** | **Good** |

### What about other options?

- ❌ **Lightning LoRA** - Still uses 18-20 GB base VRAM (reduces steps, not memory)
- ❌ **Nunchaku** - Removed per your request
- ✅ **FLUX Klein 4B** - Keep as backup (8-10 GB)

## Immediate Next Steps

### 1. Test GGUF on Your 4090 Server (30 minutes)

```bash
# SSH to server
ssh your-4090-server

# Build stable-diffusion.cpp
git clone --recursive https://github.com/leejet/stable-diffusion.cpp /opt/sd-cpp
cd /opt/sd-cpp
mkdir build && cd build
cmake .. -DSD_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Download models
mkdir -p /opt/models/qwen-gguf
cd /opt/models/qwen-gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen_image_vae.safetensors

# Test inference
/opt/sd-cpp/build/bin/sd-cli \
  --model qwen-image-2512-Q4_K_M.gguf \
  --text-encoder-model Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
  --vae qwen_image_vae.safetensors \
  --prompt "a beautiful sunset over mountains" \
  --output test.png \
  --steps 40 \
  --width 1328 \
  --height 1328

# Check VRAM usage
nvidia-smi
```

**Expected:**
- VRAM: ~15-16 GB during generation
- Time: 10-20 seconds
- Output: `test.png` with real AI art

### 2. Implement GGUF Wrapper (1-2 hours)

Create `app/services/gguf_manager.py`:

```python
import asyncio
import os
import uuid
from pathlib import Path
from PIL import Image
from typing import Optional

class GGUFModelManager:
    """Manages GGUF models via stable-diffusion.cpp."""

    def __init__(self):
        self.sd_cli = Path(os.getenv("SD_CLI_PATH", "/opt/sd-cpp/build/bin/sd-cli"))
        self.models_path = Path(os.getenv("GGUF_MODELS_PATH", "/opt/models/qwen-gguf"))
        self.model_file = os.getenv("GGUF_MODEL_FILE", "qwen-image-2512-Q4_K_M.gguf")
        self.text_encoder = os.getenv("GGUF_TEXT_ENCODER", "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf")
        self.vae = os.getenv("GGUF_VAE", "qwen_image_vae.safetensors")

    async def generate_image(
        self,
        prompt: str,
        width: int = 1328,
        height: int = 1328,
        steps: int = 40,
        cfg_scale: float = 2.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate image using GGUF model."""
        output_path = f"/tmp/gen_{uuid.uuid4()}.png"

        cmd = [
            str(self.sd_cli),
            "--model", str(self.models_path / self.model_file),
            "--text-encoder-model", str(self.models_path / self.text_encoder),
            "--vae", str(self.models_path / self.vae),
            "--prompt", prompt,
            "--output", output_path,
            "--steps", str(steps),
            "--sampling-method", "euler",
            "--cfg-scale", str(cfg_scale),
            "--width", str(width),
            "--height", str(height),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"GGUF generation failed: {stderr.decode()}")

        image = Image.open(output_path)
        return image
```

### 3. Update model_manager.py

Add GGUF support:

```python
from .gguf_manager import GGUFModelManager

# In ModelManager.__init__
self.gguf_manager = GGUFModelManager()

# In generate_image
if model_name == "qwen-2512-gguf":
    return await self.gguf_manager.generate_image(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=guidance_scale,
        seed=seed,
    )
```

### 4. Update MODEL_CONFIGS

Add GGUF config:

```python
MODEL_CONFIGS = {
    "qwen-2512-gguf": {
        "model_type": "gguf",
        "vram": 16,  # GB (Q4_K_M)
        "steps": 40,
        "guidance_scale": 2.5,
        "description": "Qwen-2512 GGUF Q4_K_M - Optimized for 24GB VRAM",
        "supported_sizes": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
    },
    # Keep FLUX models as-is
}
```

### 5. Deploy to Coolify

```bash
git add .
git commit -m "Add GGUF support for 4090 deployment"
git push origin main
```

Coolify will auto-deploy.

### 6. Verify Deployment

```bash
# Check health
curl https://image.v1su4.com/health

# Should show:
# {
#   "gpu_memory_used": 15.x,  # NOT 0.0!
#   "models_loaded": ["qwen-2512-gguf"]
# }

# Test generation
curl -X POST https://image.v1su4.com/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2512-gguf",
    "prompt": "a beautiful sunset",
    "width": 1328,
    "height": 1328
  }' \
  --output test_output.png
```

## Timeline Estimate

| Task | Time |
|------|------|
| Build stable-diffusion.cpp | 5-10 min |
| Download GGUF models | 10-20 min |
| Test inference | 5 min |
| Implement wrapper | 1-2 hours |
| Update configs | 30 min |
| Deploy & verify | 15 min |
| **Total** | **2-3 hours** |

## Rollback Plan

If GGUF doesn't work:
1. Keep FLUX Klein 4B (8-10 GB, already working)
2. Try Q3_K_M GGUF (smaller, 11-13 GB)
3. Consider using CPU offloading with bfloat16 (slow but works)

## Success Criteria

✅ Deployment is successful when:
- [ ] `nvidia-smi` shows 14-16 GB VRAM used
- [ ] Generation takes 10-20 seconds
- [ ] Images are real AI art (not placeholders)
- [ ] No crashes or OOM errors
- [ ] Can generate 1328x1328 images consistently

## Questions?

- **"Is Q4_K_M good enough?"** - Yes! It's the recommended quantization level with minimal quality loss
- **"Can I use Lightning LoRA with GGUF?"** - No, they're incompatible approaches
- **"What if I want higher quality?"** - Try Q5_K_M or Q6_K, but they use more VRAM
- **"Will this work on 4090?"** - Yes, 15 GB fits comfortably in 24 GB

## Resources

- [VRAM_ANALYSIS.md](VRAM_ANALYSIS.md) - Understanding the problem
- [GGUF_IMPLEMENTATION_PLAN.md](GGUF_IMPLEMENTATION_PLAN.md) - Full technical details
- [DEPLOY_GGUF_QUICKSTART.md](DEPLOY_GGUF_QUICKSTART.md) - Deployment steps
- [Unsloth GGUF Guide](https://unsloth.ai/docs/models/qwen-image-2512/stable-diffusion.cpp)
- [HuggingFace GGUF Models](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF)

## Summary

**Problem:** Qwen-2512 uses 20GB+ VRAM → won't fit on 4090 (24GB)
**Solution:** Use GGUF Q4_K_M format → only 15GB VRAM
**Action:** Follow steps 1-6 above to deploy

You're ready to start! Begin with Step 1 (test GGUF on your 4090 server).
