# GGUF Implementation Plan for RTX 4090 (24GB VRAM)

## Problem Analysis

### Current Issue
- **Qwen-Image-2512** in bfloat16 format uses **20GB+ VRAM**
- Crashed on RTX 5090 (32GB)
- Model config incorrectly estimated 12GB
- **Will NOT work on RTX 4090 with 24GB VRAM**

### Root Cause
The `diffusers` library loads full-precision models (bfloat16) which are too large for consumer GPUs.

## Solution: GGUF Format

### What is GGUF?
- Quantized model format from llama.cpp ecosystem
- Dramatically reduces VRAM requirements
- Uses stable-diffusion.cpp instead of diffusers
- Quality remains excellent with Q4_K_M quantization

### VRAM Requirements (RTX 4090 - 24GB)

| Model | Format | File Size | VRAM Usage | Fits? |
|-------|--------|-----------|------------|-------|
| Qwen-Image-2512 | bfloat16 | 40.9 GB | ~20-22 GB | ❌ NO |
| Qwen-Image-2512 | Q4_K_M GGUF | 13.2 GB | ~15 GB | ✅ YES |
| FLUX.2 Klein 4B | bfloat16 | ~8 GB | ~8-10 GB | ✅ YES |
| FLUX.2 Klein 9B | bfloat16 | ~18 GB | ~12-15 GB | ✅ YES |

## Recommended Configuration

### Primary Model: Qwen-Image-2512-Q4_K_M (GGUF)
- **Repo**: `unsloth/Qwen-Image-2512-GGUF`
- **File**: `qwen-image-2512-Q4_K_M.gguf` (13.2 GB)
- **Text Encoder**: `Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`
- **VAE**: `qwen_image_vae.safetensors`
- **Total VRAM**: ~15-16 GB
- **Quality**: Excellent (4-bit quantization with smart upcasting)
- **Speed**: ~40 steps recommended

### Backup Model: FLUX.2 Klein 4B (diffusers)
- Keep as-is for fast generation
- Works well with existing `diffusers` code
- Good fallback option

## Implementation Options

### Option 1: stable-diffusion.cpp Binary (Recommended for MVP)
**Pros:**
- Easy to implement (subprocess calls)
- No Python dependencies
- Proven stable
- Direct from official docs

**Cons:**
- Need to build from source
- CLI-based (not library)

**Implementation:**
```python
import subprocess
from pathlib import Path

def generate_with_gguf(prompt: str, output_path: str):
    cmd = [
        "./bin/sd-cli",
        "--model", "./models/qwen-image-2512-Q4_K_M.gguf",
        "--text-encoder-model", "./models/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "--vae", "./models/qwen_image_vae.safetensors",
        "--prompt", prompt,
        "--output", output_path,
        "--steps", "40",
        "--sampling-method", "euler",
        "--cfg-scale", "2.5",
        "--width", "1328",
        "--height", "1328",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result
```

### Option 2: Python Bindings (Future Enhancement)
Look for Python bindings to stable-diffusion.cpp for better integration.

### Option 3: ComfyUI (Alternative)
ComfyUI supports GGUF natively, but requires running ComfyUI server.

## Implementation Steps

### Phase 1: Build stable-diffusion.cpp (On 4090 Server)

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y git cmake build-essential pkg-config

# 2. Setup CUDA (if not already)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version  # Verify CUDA 12.8

# 3. Clone and build
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
mkdir build && cd build
cmake .. -DSD_CUDA=ON
cmake --build . --config Release -j$(nproc)

# 4. Verify binary
ls -la bin/sd-cli  # Should exist and be executable
```

### Phase 2: Download Models

```bash
# Create models directory
mkdir -p models/qwen-gguf

# Download from HuggingFace
cd models/qwen-gguf

# 1. Main diffusion model (13.2 GB)
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf

# 2. Text encoder
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf

# 3. VAE
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen_image_vae.safetensors
```

### Phase 3: Update model_manager.py

Create a new `GGUFModelManager` class that wraps stable-diffusion.cpp:

```python
class GGUFModelManager:
    """Manages GGUF models via stable-diffusion.cpp."""

    def __init__(self, sd_cli_path: str = "./stable-diffusion.cpp/build/bin/sd-cli"):
        self.sd_cli_path = Path(sd_cli_path)
        self.models_path = Path("./models/qwen-gguf")

    async def generate_image(
        self,
        prompt: str,
        width: int = 1328,
        height: int = 1328,
        steps: int = 40,
        cfg_scale: float = 2.5,
        seed: Optional[int] = None,
    ):
        """Generate image using GGUF model."""
        output_path = Path(f"./temp/{uuid.uuid4()}.png")

        cmd = [
            str(self.sd_cli_path),
            "--model", str(self.models_path / "qwen-image-2512-Q4_K_M.gguf"),
            "--text-encoder-model", str(self.models_path / "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"),
            "--vae", str(self.models_path / "qwen_image_vae.safetensors"),
            "--prompt", prompt,
            "--output", str(output_path),
            "--steps", str(steps),
            "--sampling-method", "euler",
            "--cfg-scale", str(cfg_scale),
            "--width", str(width),
            "--height", str(height),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        # Run subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"sd-cli failed: {stderr.decode()}")

        # Load and return image
        from PIL import Image
        image = Image.open(output_path)
        return image
```

### Phase 4: Update MODEL_CONFIGS

```python
MODEL_CONFIGS = {
    "qwen-2512-gguf": {
        "model_type": "gguf",
        "model_file": "qwen-image-2512-Q4_K_M.gguf",
        "text_encoder": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "vae": "qwen_image_vae.safetensors",
        "vram": 16,  # GB (Q4_K_M quantized)
        "steps": 40,  # Recommended for GGUF
        "guidance_scale": 2.5,
        "description": "Qwen Image-2512 GGUF Q4_K_M - Optimized for 24GB VRAM",
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
    "flux-klein-4b": {
        # Keep existing diffusers-based config
        "model_type": "diffusers",
        # ... existing config
    },
}
```

## Testing Plan

### Local Testing (Before Deployment)

```bash
# Test GGUF inference locally
./stable-diffusion.cpp/build/bin/sd-cli \
  --model models/qwen-gguf/qwen-image-2512-Q4_K_M.gguf \
  --text-encoder-model models/qwen-gguf/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
  --vae models/qwen-gguf/qwen_image_vae.safetensors \
  --prompt "a beautiful mountain landscape at sunset" \
  --output test_output.png \
  --steps 40 \
  --sampling-method euler \
  --cfg-scale 2.5 \
  --width 1328 \
  --height 1328
```

### Expected Results
- First run: ~30-60s (model loading)
- Subsequent: ~10-20s per image (40 steps)
- VRAM usage: ~15-16 GB
- Output: High-quality AI-generated image

## Deployment to Coolify

### Dockerfile Updates

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential pkg-config \
    python3.11 python3-pip

# Clone and build stable-diffusion.cpp
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp /opt/sd-cpp
WORKDIR /opt/sd-cpp
RUN mkdir build && cd build && \
    cmake .. -DSD_CUDA=ON && \
    cmake --build . --config Release -j$(nproc)

# Copy app
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -e .

# Download models (or mount volume)
RUN mkdir -p /app/models/qwen-gguf

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
MODELS_PATH=/app/models/qwen-gguf
SD_CLI_PATH=/opt/sd-cpp/build/bin/sd-cli
CUDA_VISIBLE_DEVICES=0
```

## Performance Expectations

### RTX 4090 (24GB VRAM) - GGUF Q4_K_M

| Metric | Value |
|--------|-------|
| Model loading time | 10-20s |
| VRAM usage | 15-16 GB |
| Generation time (1328x1328, 40 steps) | 10-20s |
| Quality | Excellent |
| Max concurrent requests | 1 (single GPU) |

### Comparison to bfloat16

| Format | VRAM | Speed | Quality | Fits 4090? |
|--------|------|-------|---------|------------|
| bfloat16 | 20-22 GB | Fast | Perfect | ❌ NO |
| Q4_K_M GGUF | 15-16 GB | Medium | Excellent | ✅ YES |

## Cleanup Tasks

### Remove Nunchaku References
- ❌ Delete `test_qwen_nunchaku.py`
- ❌ Remove Nunchaku from documentation
- ❌ Don't add Nunchaku configs to model_manager.py

### Remove Lightning LoRA (For GGUF)
Lightning LoRA is for reducing steps in diffusers models. GGUF already handles optimization, so:
- Keep Lightning configs for FLUX models (they still use diffusers)
- Don't try to combine GGUF + Lightning (incompatible)

## Next Steps

1. ✅ Research complete - GGUF is the solution
2. ⏳ Build stable-diffusion.cpp on 4090 server
3. ⏳ Download GGUF models (13.2 GB main model)
4. ⏳ Implement GGUFModelManager wrapper
5. ⏳ Test locally before deploying
6. ⏳ Update Coolify deployment with new Dockerfile

## Unsloth Official Reference

**Source:** https://unsloth.ai/docs/models/qwen-image-2512/stable-diffusion.cpp

### Environment Requirements

- **Minimum RAM:** 13.2+ GB combined memory for 4-bit quantized models
- **GPU:** Optional but recommended (CUDA support)
- **OS:** Ubuntu 22.04+ recommended

### Build Dependencies

```bash
sudo apt update
sudo apt install -y git cmake build-essential pkg-config
```

### CUDA Configuration

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Verify installation
nvcc --version
ldconfig -p | grep -E 'libcudart\.so|libcublas\.so'
```

### Build from Source

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
cmake --build . -j"$(nproc)"
```

### Required Model Files

#### Generation Model (Qwen-Image-2512)
| Component | File | Size | Source |
|-----------|------|------|--------|
| Diffusion Model | `qwen-image-2512-Q4_K_M.gguf` | ~13 GB | [unsloth/Qwen-Image-2512-GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF) |
| Text Encoder | `Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf` | ~4.5 GB | [unsloth/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| VAE | `qwen_image_vae.safetensors` | ~243 MB | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) |

#### Edit Model (Qwen-Image-Edit-2511)
| Component | File | Size | Source |
|-----------|------|------|--------|
| Diffusion Model | `qwen-image-edit-2511-Q4_K_M.gguf` | ~13 GB | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) |
| Text Encoder | `Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf` | ~4.5 GB | (shared with generation) |
| VAE | `qwen_image_vae.safetensors` | ~243 MB | (shared with generation) |

**Total storage required:** ~31 GB (both models share text encoder and VAE)

### CLI Parameters Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--diffusion-model` | Path to GGUF diffusion model | `qwen-image-2512-Q4_K_M.gguf` |
| `--llm` | Path to text encoder GGUF | `Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf` |
| `--vae` | Path to VAE safetensors | `qwen_image_vae.safetensors` |
| `--cfg-scale` | Guidance scale | `2.5` |
| `--sampling-method` | Sampling method | `euler` |
| `--steps` | Number of diffusion steps | `40` |
| `-H` | Image height | `1328` |
| `-W` | Image width | `1328` |
| `-p` | Prompt text | `'a cat'` |
| `-o` | Output file path | `output.png` |
| `--diffusion-fa` | Enable flash attention | (flag) |
| `--flow-shift` | Flow shift value | `3` |
| `--offload-to-cpu` | Offload to CPU when VRAM limited | (flag) |
| `-v` | Verbose output | (flag) |
| `--seed` | Random seed for reproducibility | `42` |

### Full CLI Example - Generation

```bash
./build/bin/sd-cli \
    --diffusion-model ./models/qwen-image-2512-Q4_K_M.gguf \
    --vae ./models/qwen_image_vae.safetensors \
    --llm ./models/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --cfg-scale 2.5 \
    --sampling-method euler \
    --steps 40 \
    -H 1328 -W 1328 \
    --diffusion-fa \
    --flow-shift 3 \
    -p 'a beautiful mountain landscape at sunset, photorealistic' \
    -o output.png \
    -v
```

### Full CLI Example - Image Editing

```bash
./build/bin/sd-cli \
    --diffusion-model ./models/qwen-image-edit-2511-Q4_K_M.gguf \
    --vae ./models/qwen_image_vae.safetensors \
    --llm ./models/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --cfg-scale 4.0 \
    --sampling-method euler \
    --steps 40 \
    -H 1328 -W 1328 \
    --diffusion-fa \
    --flow-shift 3 \
    -i input_image.png \
    -p 'make the sky more dramatic with storm clouds' \
    -o edited_output.png \
    -v
```

### Memory Optimization Flags

For systems with limited VRAM:
```bash
--offload-to-cpu  # Offload model weights to CPU when not in use
```

### Quantization Options

Available quantization levels (quality/size tradeoff):

| Quantization | Quality | Size | Use Case |
|--------------|---------|------|----------|
| Q4_K_M | Excellent | ~13 GB | **Recommended** - Best balance |
| Q4_K_S | Very Good | ~12 GB | Smaller but slightly lower quality |
| Q8_0 | Near Perfect | ~20 GB | When VRAM allows |

---

## Questions?

- GGUF format is battle-tested for Qwen-Image-2512
- Q4_K_M provides best quality/VRAM balance
- Will definitely fit on 4090 with 24GB VRAM
- Lightning LoRA not needed with GGUF (already optimized)
