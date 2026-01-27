# GGUF Migration Summary

## 🚨 Critical Issue Identified

Your Qwen-2512 configuration **will not work** on the RTX 4090 (24GB VRAM) deployment target.

### The Problem
- **Configured VRAM:** 12 GB (in model_manager.py)
- **Actual VRAM:** 20-22 GB for bfloat16 format
- **Target GPU:** RTX 4090 with 24 GB
- **Result:** Won't fit / will crash

### The Solution: GGUF Format
- **GGUF Q4_K_M:** Only 14-16 GB VRAM
- **Quality:** Excellent (minimal loss)
- **Fits 4090:** ✅ YES, comfortably

## 📚 Documentation Created

Start here based on what you need:

### 🎯 Just Want to Deploy?
→ **[DEPLOY_GGUF_QUICKSTART.md](DEPLOY_GGUF_QUICKSTART.md)**
- Step-by-step deployment guide
- Copy-paste commands
- 30-minute setup

### 🤔 Want to Understand the Problem?
→ **[VRAM_ANALYSIS.md](VRAM_ANALYSIS.md)**
- Why your 5090 crashed
- VRAM comparison tables
- Before/after analysis

### 🔧 Need Technical Details?
→ **[GGUF_IMPLEMENTATION_PLAN.md](GGUF_IMPLEMENTATION_PLAN.md)**
- Full implementation guide
- Code examples
- Architecture decisions

### 📋 What Do I Do Next?
→ **[NEXT_STEPS.md](NEXT_STEPS.md)**
- Action items checklist
- Timeline estimates
- Success criteria

## Quick Start (5 minutes)

### 1. SSH to your 4090 server
```bash
ssh your-coolify-server
```

### 2. Build stable-diffusion.cpp
```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp /opt/sd-cpp
cd /opt/sd-cpp/build
cmake .. -DSD_CUDA=ON && cmake --build . --config Release -j$(nproc)
```

### 3. Download GGUF models
```bash
mkdir -p /opt/models/qwen-gguf && cd /opt/models/qwen-gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
wget https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen_image_vae.safetensors
```

### 4. Test it works
```bash
/opt/sd-cpp/build/bin/sd-cli \
  --model qwen-image-2512-Q4_K_M.gguf \
  --text-encoder-model Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
  --vae qwen_image_vae.safetensors \
  --prompt "beautiful sunset" \
  --output test.png
```

See [NEXT_STEPS.md](NEXT_STEPS.md) for full implementation guide.

## VRAM Comparison

| Format | File Size | VRAM Usage | Fits 4090? | Quality |
|--------|-----------|------------|------------|---------|
| BF16 (current) | 40.9 GB | 20-22 GB | ❌ NO | Perfect |
| **GGUF Q4_K_M** | **13.2 GB** | **14-16 GB** | ✅ **YES** | Excellent |
| FLUX Klein 4B | ~8 GB | 8-10 GB | ✅ YES | Good |

## Important: Image Dimensions

**Qwen-Image-2512 uses different native resolutions than FLUX:**

| Aspect Ratio | Qwen Dimensions | FLUX Dimensions |
|--------------|-----------------|-----------------|
| 1:1 | **1328 x 1328** | 1024 x 1024 |
| 16:9 | **1664 x 928** | 1024 x 576 |
| 9:16 | **928 x 1664** | 576 x 1024 |
| 4:3 | **1472 x 1104** | N/A |
| 3:4 | **1104 x 1472** | N/A |
| 3:2 | **1584 x 1056** | N/A |
| 2:3 | **1056 x 1584** | N/A |

**Use Qwen's native resolutions for best quality!** The model is trained on these specific dimensions.

## What Changed

### ✅ Completed
- [x] Identified VRAM issue (20GB vs 12GB config)
- [x] Researched GGUF solution
- [x] Created comprehensive documentation
- [x] Removed Nunchaku references
- [x] Created deployment guides

### ⏳ To Do (Next)
- [ ] Build stable-diffusion.cpp on 4090
- [ ] Download GGUF models
- [ ] Implement GGUFModelManager wrapper
- [ ] Test locally
- [ ] Deploy to Coolify
- [ ] Verify real inference works

## Key Takeaways

1. **bfloat16 is too big** - Uses 20-22 GB VRAM (not 12 GB)
2. **Lightning LoRA doesn't help** - Reduces steps, not VRAM
3. **GGUF is the solution** - Only 14-16 GB with excellent quality
4. **Q4_K_M is optimal** - Best balance of quality/size
5. **stable-diffusion.cpp required** - diffusers doesn't support GGUF

## Files Created

```
📄 README_GGUF.md (this file)           - Overview
📄 VRAM_ANALYSIS.md                     - Problem analysis
📄 GGUF_IMPLEMENTATION_PLAN.md          - Technical guide
📄 DEPLOY_GGUF_QUICKSTART.md            - Deployment steps
📄 NEXT_STEPS.md                        - Action items
📄 .env.gguf.example                    - Environment config
```

## Timeline

**Estimated time to deploy:** 2-3 hours
- Build: 10 min
- Download: 20 min
- Implementation: 1-2 hours
- Testing: 30 min

## Support

- **Unsloth GGUF Guide:** https://unsloth.ai/docs/models/qwen-image-2512/stable-diffusion.cpp
- **HuggingFace Models:** https://huggingface.co/unsloth/Qwen-Image-2512-GGUF
- **stable-diffusion.cpp:** https://github.com/leejet/stable-diffusion.cpp

---

**Bottom Line:** Your current setup won't work on the 4090. GGUF format is the solution. Start with [NEXT_STEPS.md](NEXT_STEPS.md).
