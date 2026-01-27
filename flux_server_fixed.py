import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import Flux2KleinPipeline  # Correct class from official docs
import base64

MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")

app = FastAPI(title="FLUX.2 klein 4B inference")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Txt2ImgRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 4  # Official example uses 4 steps
    guidance: float = 1.0  # Official example uses 1.0
    seed: int = 0

pipe: Flux2KleinPipeline | None = None

@app.on_event("startup")
def load_model():
    global pipe
    device = "cuda"
    dtype = torch.bfloat16  # Official example uses bfloat16

    print(f"🚀 Loading FLUX.2 [klein] 4B from {MODEL_ID}...")

    # Official approach from HuggingFace model card
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()  # Official recommendation to save VRAM

    print("✅ Model loaded successfully!")

@app.get("/")
def root():
    return {
        "name": "FLUX.2 klein 4B API",
        "model": MODEL_ID,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    cuda_available = torch.cuda.is_available()
    gpu_info = {}
    if cuda_available:
        gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_info["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info["vram_used_gb"] = torch.cuda.memory_allocated(0) / (1024**3)

    return {
        "ok": True,
        "model": MODEL_ID,
        "cuda_available": cuda_available,
        **gpu_info
    }

@app.post("/txt2img")
def txt2img(req: Txt2ImgRequest):
    """Generate image and return PNG directly."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(req.seed)

    print(f"🎨 Generating: {req.prompt[:50]}...")

    # Official approach from HuggingFace model card
    image = pipe(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance,
        num_inference_steps=req.steps,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/v1/images/create")
def create_image_openai_compatible(req: Txt2ImgRequest):
    """OpenAI-compatible endpoint that returns base64 JSON."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(req.seed)

    print(f"🎨 Generating: {req.prompt[:50]}...")

    image = pipe(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance,
        num_inference_steps=req.steps,
        generator=generator,
    ).images[0]

    # Convert to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        "success": True,
        "data": {
            "images": [{
                "id": "flux-gen",
                "data": b64_data,
                "format": "png",
                "metadata": {
                    "width": req.width,
                    "height": req.height,
                    "model_used": MODEL_ID,
                    "steps": req.steps,
                    "seed": req.seed
                }
            }]
        },
        "metadata": {
            "model": MODEL_ID,
            "prompt": req.prompt
        }
    }
