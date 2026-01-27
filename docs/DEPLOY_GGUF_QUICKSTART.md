# Quick Start: Deploy GGUF on Coolify (4090 - 24GB)

## TL;DR - What Changed

**Problem:** Qwen-2512 bfloat16 uses 20GB+ VRAM → crashes on 4090 (24GB)
**Solution:** Use GGUF Q4_K_M format → uses only 15GB VRAM ✅

## Deploy to Coolify - Step by Step

### Step 1: SSH to Your 4090 Server

```bash
ssh your-coolify-server
cd /path/to/deployment
```

### Step 2: Build stable-diffusion.cpp

```bash
# Install dependencies
sudo apt update
sudo apt install -y git cmake build-essential pkg-config

# Verify CUDA (should show 12.8)
nvcc --version

# Clone and build
git clone --recursive https://github.com/leejet/stable-diffusion.cpp /opt/sd-cpp
cd /opt/sd-cpp
mkdir build && cd build

# Build with CUDA support
cmake .. -DSD_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Verify binary exists
ls -la bin/sd-cli
```

**Expected time:** 5-10 minutes

### Step 3: Download GGUF Models

```bash
# Create directory
mkdir -p /opt/models/qwen-gguf
cd /opt/models/qwen-gguf

# Download models (total: ~15GB)
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen_image_vae.safetensors

# Verify downloads
ls -lh
```

**Expected time:** 10-20 minutes (depending on connection)

### Step 4: Test GGUF Inference

```bash
# Test generation
/opt/sd-cpp/build/bin/sd-cli \
  --model /opt/models/qwen-gguf/qwen-image-2512-Q4_K_M.gguf \
  --text-encoder-model /opt/models/qwen-gguf/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
  --vae /opt/models/qwen-gguf/qwen_image_vae.safetensors \
  --prompt "a beautiful mountain landscape at sunset, photorealistic" \
  --output test_output.png \
  --steps 40 \
  --sampling-method euler \
  --cfg-scale 2.5 \
  --width 1328 \
  --height 1328

# Check VRAM usage during generation
watch -n 1 nvidia-smi
```

**Expected results:**
- Generation time: 10-20s
- VRAM usage: ~15-16 GB
- Output: `test_output.png` with real AI-generated image

### Step 5: Update Environment Variables

In Coolify or your `.env` file:

```bash
# GGUF Configuration
SD_CLI_PATH=/opt/sd-cpp/build/bin/sd-cli
GGUF_MODELS_PATH=/opt/models/qwen-gguf
GGUF_MODEL_FILE=qwen-image-2512-Q4_K_M.gguf
GGUF_TEXT_ENCODER=Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
GGUF_VAE=qwen_image_vae.safetensors

# Generation defaults
DEFAULT_STEPS=40
DEFAULT_CFG_SCALE=2.5
```

### Step 6: Update Your Code

You'll need to implement a `GGUFModelManager` wrapper. See [GGUF_IMPLEMENTATION_PLAN.md](GGUF_IMPLEMENTATION_PLAN.md) for full code.

Basic wrapper:

```python
import asyncio
from pathlib import Path
from PIL import Image

class GGUFModelManager:
    def __init__(self):
        self.sd_cli = Path(os.getenv("SD_CLI_PATH", "/opt/sd-cpp/build/bin/sd-cli"))
        self.models_path = Path(os.getenv("GGUF_MODELS_PATH", "/opt/models/qwen-gguf"))

    async def generate_image(self, prompt: str, width=1328, height=1328):
        output_path = f"/tmp/gen_{uuid.uuid4()}.png"

        cmd = [
            str(self.sd_cli),
            "--model", str(self.models_path / "qwen-image-2512-Q4_K_M.gguf"),
            "--text-encoder-model", str(self.models_path / "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"),
            "--vae", str(self.models_path / "qwen_image_vae.safetensors"),
            "--prompt", prompt,
            "--output", output_path,
            "--steps", "40",
            "--cfg-scale", "2.5",
            "--width", str(width),
            "--height", str(height),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

        return Image.open(output_path)
```

### Step 7: Deploy via Coolify

Push changes to GitHub, Coolify will auto-deploy.

```bash
git add .
git commit -m "Switch to GGUF format for 4090 compatibility"
git push origin main
```

## Verification Checklist

After deployment, verify:

- [ ] `nvidia-smi` shows ~15-16 GB VRAM used (not 0 GB!)
- [ ] `/health` endpoint shows models loaded
- [ ] First generation takes 30-60s (model loading)
- [ ] Subsequent generations take 10-20s
- [ ] Images are real AI art (not placeholders!)

## VRAM Comparison

| Model | Format | VRAM | Fits 4090? |
|-------|--------|------|------------|
| Qwen-2512 | bfloat16 | 20-22 GB | ❌ NO |
| Qwen-2512 | Q4_K_M GGUF | 15-16 GB | ✅ YES |
| FLUX Klein 4B | bfloat16 | 8-10 GB | ✅ YES |

## Troubleshooting

### "sd-cli: command not found"
```bash
ls -la /opt/sd-cpp/build/bin/sd-cli  # Should exist
chmod +x /opt/sd-cpp/build/bin/sd-cli
```

### "Cannot open model file"
```bash
ls -la /opt/models/qwen-gguf/  # Check files exist
# Re-download if needed
```

### "CUDA out of memory"
```bash
# Check what's using VRAM
nvidia-smi
# Should only use ~15-16 GB for Qwen GGUF
# If higher, check for other processes
```

### Still getting placeholders
```bash
# Check if GGUF integration is working
/opt/sd-cpp/build/bin/sd-cli --help  # Should show help
# Test generation manually (see Step 4)
```

## Performance Expectations

### On RTX 4090 (24GB)

| Metric | Value |
|--------|-------|
| First request | 30-60s (loads model into VRAM) |
| Subsequent requests | 10-20s per image |
| VRAM usage | 15-16 GB |
| Quality | Excellent (Q4_K_M is high quality) |
| Max resolution | 1328x1328 (1:1), 1664x928 (16:9), 1472x1104 (4:3) |

## Cost Savings

By using GGUF instead of full bfloat16:
- **VRAM saved:** 5-7 GB (24% less)
- **Storage saved:** 27 GB (40.9 GB → 13.2 GB)
- **Quality loss:** Minimal (Q4_K_M uses smart quantization)
- **Speed impact:** Slightly slower but acceptable

## Next Steps

1. Complete Step 1-4 above to test GGUF works
2. Implement `GGUFModelManager` in your codebase
3. Test locally before deploying to Coolify
4. Deploy and verify with checklist above

## Questions?

See full implementation details in [GGUF_IMPLEMENTATION_PLAN.md](GGUF_IMPLEMENTATION_PLAN.md)
