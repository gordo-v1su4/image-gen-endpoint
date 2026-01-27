# Setting Up Real AI Image Generation

## Current Situation

The production server at `https://image.v1su4.com` is currently returning **placeholder images**, not real AI-generated images. This is because:

1. ✅ The code is implemented for real inference
2. ❌ The AI models were never downloaded
3. ❌ GPU memory shows 0.0 GB used (no models loaded)
4. ⚠️  Processing times of 3-11ms are too fast (real AI takes 2-10 seconds)

## What You're Getting Now

The images you're seeing have:
- Dark gray background with grid pattern
- Text overlay showing dimensions
- Prompt text displayed on the image
- These are **placeholders generated with PIL**, not AI

## Setting Up Real AI Inference

### Option 1: Download Models Locally (Recommended for Testing)

#### Requirements
- **GPU**: NVIDIA GPU with 8-24GB VRAM (RTX 3080/3090/4090)
- **Storage**: 10-50GB for models
- **RAM**: 16GB+ recommended
- **Python**: 3.10+

#### Steps

1. **Install Dependencies**
```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install all dependencies including HuggingFace Hub
uv pip install -e ".[dev]" huggingface_hub
```

2. **Set Up HuggingFace Token** (if needed)
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"
```

3. **Download Models**
```bash
# Run the download script
python3 download_models.py
```

This will:
- Check GPU availability
- Download models from HuggingFace
- Cache models in `./models/huggingface/`
- Guide you through model selection

4. **Start the Server Locally**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Test Real Image Generation**
```bash
# Update test script to use localhost
sed -i 's|https://image.v1su4.com|http://localhost:8000|' test_image_gen.py

# Run tests
python3 test_image_gen.py
```

### Option 2: Deploy to Production with Models

#### On Coolify/Production Server

1. **SSH to production server**
```bash
ssh your-server
cd /path/to/image-gen-endpoint
```

2. **Set environment variables**
```bash
export HF_TOKEN="your_token"
export MODELS_PATH="/persistent/storage/models"
export DOWNLOAD_MODELS="true"
```

3. **Download models on first startup**
```bash
# In Dockerfile or startup script
python3 download_models.py
```

4. **Verify models loaded**
```bash
curl https://image.v1su4.com/health
# Should show: "gpu_memory_used": > 8.0
```

### Option 3: Use HuggingFace MCP & Smol Agents

For a more automated approach using AI agents to handle the setup:

1. **Install Smol Agents**
```bash
pip install smolagents huggingface_hub
```

2. **Create Agent Script**
```python
from smolagents import CodeAgent
from smolagents.tools import HfApiTool

# Create agent with HuggingFace tools
agent = CodeAgent(tools=[HfApiTool()])

# Let agent download and set up models
agent.run(\"\"\"
Download the following models for image generation:
1. Qwen/Qwen-Image-2512 (FP8 variant if available)
2. black-forest-labs/FLUX.2-klein-4B

Store them in ./models/huggingface/ directory.
Verify download integrity and report model sizes.
\"\"\")
```

3. **Run setup**
```bash
python3 setup_with_agents.py
```

## Model Information

### Qwen-Image-2512
- **Size**: ~16GB (FP8), ~32GB (FP16)
- **VRAM**: 16GB minimum
- **Speed**: 2-10s per image (8 steps with Lightning)
- **Quality**: High quality, good at complex prompts

### FLUX Klein 4B
- **Size**: ~8GB (FP8), ~16GB (FP16)
- **VRAM**: 8GB minimum
- **Speed**: 1-5s per image
- **Quality**: Fast, good for previews

## Verifying Real Inference

Once models are downloaded, you'll see:

1. **Server logs show**:
   - "Loading model: qwen-2512..."
   - "Model loaded successfully!"
   - Component verification messages

2. **Health endpoint shows**:
   ```json
   {
     "gpu_memory_used": 8.5,  // NOT 0.0!
     "models_loaded": ["qwen-2512"]
   }
   ```

3. **Generation times**:
   - First: 30-60s (model loading)
   - Subsequent: 2-10s (actual inference)
   - NOT 3-11ms (that's placeholder speed)

4. **Image quality**:
   - Actual AI-generated art
   - Follows prompts accurately
   - No grid pattern or text overlay

## Troubleshooting

### "No GPU Available"
- Check: `nvidia-smi`
- Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA drivers if needed

### "Out of Memory"
- Use smaller model (flux-klein-4b instead of qwen-2512)
- Reduce image dimensions (664x664 instead of 1328x1328 for Qwen)
- Close other GPU applications

### "Model Download Fails"
- Check HuggingFace token is valid
- Verify internet connection
- Some models need manual approval on HuggingFace
- Check disk space (need 50GB+ free)

### "Still Getting Placeholders"
- Check if models directory is empty: `ls -la models/`
- Verify GPU memory used > 0: `curl localhost:8000/health`
- Check server logs for errors
- Restart server after downloading models

## Performance Expectations

### With Placeholder (Current)
- Response time: 3-11ms
- Images: Dark grid with text
- GPU usage: 0%

### With Real AI (After Setup)
- First request: 30-60s (loads model)
- Subsequent: 2-10s per image
- GPU usage: 50-90%
- Images: Actual AI art

## Next Steps

1. **Choose your setup option** (local testing recommended first)
2. **Download models** using the script
3. **Verify models loaded** (check health endpoint)
4. **Generate test images**
5. **Compare before/after** - you'll see the difference immediately!

## Questions?

- Check [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for code details
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Review [README.md](README.md) for API documentation
