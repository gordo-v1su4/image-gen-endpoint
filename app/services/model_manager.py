"""Model management service for loading and caching AI models."""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path

# Model configurations
MODEL_CONFIGS = {
    "qwen-2512": {
        "repo_id": "Qwen/Qwen-Image-2512",
        "fp8_repo": "Qwen/Qwen-Image-2512",  # FP8 variant
        "lightning_repo": "lightx2v/Qwen-Image-2512-Lightning",
        "lightning_4step": "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
        "lightning_8step": "Qwen-Image-2512-Lightning-8steps-V1.0.safetensors",
        "vram_fp8": 16,  # GB
        "pipeline_class": "QwenImagePipeline",
    },
    "flux-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "fp8_repo": "black-forest-labs/FLUX.2-klein-4b-fp8",
        "vram_fp8": 8,
        "pipeline_class": "FluxKleinPipeline",
        "license": "apache-2.0",
    },
    "flux-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "fp8_repo": "black-forest-labs/FLUX.2-klein-9b-fp8",
        "vram_fp8": 12,
        "pipeline_class": "FluxKleinPipeline",
        "license": "non-commercial",
    },
}


class ModelManager:
    """Manages loading, caching, and switching between AI models."""
    
    def __init__(self, models_path: Optional[str] = None):
        self.models_path = Path(models_path or os.environ.get("MODELS_PATH", "./models"))
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used = torch.cuda.memory_allocated(0) / (1024**3)
        return {
            "total": round(total, 2),
            "used": round(used, 2),
            "free": round(total - used, 2),
        }
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if model files exist locally."""
        if model_name not in MODEL_CONFIGS:
            return False
        
        model_dir = self.models_path / model_name
        return model_dir.exists() and any(model_dir.iterdir())
    
    def get_model_status(self, model_name: str) -> str:
        """Get model status: ready, loading, not_loaded, unknown."""
        if model_name in self.loaded_models:
            return "ready"
        elif self.is_model_downloaded(model_name):
            return "not_loaded"
        else:
            return "not_downloaded"
    
    def list_models(self) -> list:
        """List all available models with their status."""
        models = []
        for name, config in MODEL_CONFIGS.items():
            models.append({
                "name": name,
                "status": self.get_model_status(name),
                "vram_required": config.get("vram_fp8", 0),
                "license": config.get("license", "check-repo"),
            })
        return models
    
    async def load_model(self, model_name: str, use_lightning: bool = True, steps: int = 8):
        """Load a model into memory."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = MODEL_CONFIGS[model_name]
        
        # Check VRAM
        gpu_mem = self.get_gpu_memory()
        if gpu_mem["free"] < config.get("vram_fp8", 0):
            # Unload other models if needed
            await self.unload_all_models()
        
        # TODO: Implement actual model loading with diffusers
        # This is a placeholder for the actual implementation
        print(f"Loading model: {model_name}")
        print(f"  Repo: {config['repo_id']}")
        print(f"  Use Lightning: {use_lightning}")
        print(f"  Device: {self.device}")
        
        # Placeholder - actual implementation would use diffusers
        self.loaded_models[model_name] = {
            "config": config,
            "pipeline": None,  # Would be actual pipeline
            "use_lightning": use_lightning,
        }
        
        return self.loaded_models[model_name]
    
    async def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def unload_all_models(self):
        """Unload all models from memory."""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1328,
        height: int = 1328,
        steps: int = 8,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        use_lightning: bool = True,
    ):
        """Generate an image using the specified model."""
        # Load model if not loaded
        model_data = await self.load_model(model_name, use_lightning, steps)
        
        # TODO: Implement actual generation
        # This is a placeholder
        from app.services.image_processor import generate_placeholder_image
        return generate_placeholder_image(width, height, f"Model: {model_name}")
    
    async def edit_image(
        self,
        model_name: str,
        image,
        prompt: str,
        steps: int = 8,
        use_lightning: bool = True,
    ):
        """Edit an image using the specified model."""
        model_data = await self.load_model(model_name, use_lightning, steps)
        
        # TODO: Implement actual editing
        # This is a placeholder
        return image

# Global model manager instance
model_manager = ModelManager()