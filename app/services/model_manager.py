"""Model management service for loading and caching AI models.

Supports:
- Qwen-Image-2512 FP8 Lightning (4-step)
- FLUX.2 Klein 4B (4-step)
"""

import gc
import math
import os
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Import diffusers
try:
    from diffusers import FluxPipeline, DiffusionPipeline
    from huggingface_hub import hf_hub_download
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    FluxPipeline = None
    DiffusionPipeline = None
    logger.warning(f"diffusers not available: {e}")

# Model configurations
MODEL_CONFIGS = {
    "qwen-2512-lightning": {
        "model_type": "qwen_lightning",
        "repo_id": "lightx2v/Qwen-Image-2512-Lightning",
        "diffusion_file": "qwen_image_2512_fp8_e4m3fn_scaled_4steps_v1.0.safetensors",
        "text_encoder_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "vae_repo": "Qwen/Qwen-Image-2512",
        "vram": 20,  # GB
        "steps": 4,
        "guidance_scale": 1.0,
        "description": "Qwen-Image-2512 FP8 Lightning - 4-step fast generation",
        "license": "apache-2.0",
        "supported_sizes": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
        },
        "default_size": (1328, 1328),
    },
    "flux-klein-4b": {
        "model_type": "flux",
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "vram": 13,  # GB
        "steps": 4,
        "guidance_scale": 1.0,
        "description": "FLUX.2 Klein 4B - Fast 4-step distilled generation",
        "license": "apache-2.0",
        "supported_sizes": {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1152, 896),
            "3:4": (896, 1152),
        },
        "default_size": (1024, 1024),
    },
}


def _clear_vram():
    """Clear VRAM after generation."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("VRAM cleared")
    gc.collect()


class ModelManager:
    """Manages loading, caching, and switching between AI models."""
    
    def __init__(self, models_path: Optional[str] = None):
        self.models_path = Path(models_path or os.environ.get("HF_HOME", "/opt/models/huggingface"))
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.loaded_models: Dict[str, Any] = {}
        self.current_model: Optional[str] = None
        
        # Set device/dtype
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            self.device = "cpu"
            self.dtype = None
        
        logger.info(f"ModelManager initialized: device={self.device}")
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used = torch.cuda.memory_allocated(0) / (1024**3)
        return {
            "total": round(total, 2),
            "used": round(used, 2),
            "free": round(total - used, 2),
        }
    
    def get_model_status(self, model_name: str) -> str:
        """Get model status."""
        if model_name in self.loaded_models:
            return "ready"
        return "not_loaded"
    
    def list_models(self) -> list:
        """List all available models."""
        models = []
        for name, config in MODEL_CONFIGS.items():
            models.append({
                "name": name,
                "status": self.get_model_status(name),
                "vram_required": config.get("vram", 0),
                "steps": config.get("steps", 4),
                "license": config.get("license", "check-repo"),
                "description": config.get("description", ""),
            })
        return models

    async def load_model(self, model_name: str):
        """Load a model into memory."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        config = MODEL_CONFIGS[model_name]
        
        # Unload current model if different
        if self.current_model and self.current_model != model_name:
            await self.unload_model(self.current_model)

        logger.info(f"Loading model: {model_name}")
        
        try:
            if config["model_type"] == "flux":
                pipeline = await self._load_flux(config)
            elif config["model_type"] == "qwen_lightning":
                pipeline = await self._load_qwen_lightning(config)
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")

            self.loaded_models[model_name] = {
                "config": config,
                "pipeline": pipeline,
            }
            self.current_model = model_name
            
            logger.info(f"Model {model_name} loaded successfully")
            return self.loaded_models[model_name]

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise ValueError(f"Failed to load model {model_name}: {str(e)}")

    async def _load_flux(self, config: dict):
        """Load FLUX Klein model."""
        logger.info(f"Loading FLUX from {config['repo_id']}")
        
        pipeline = FluxPipeline.from_pretrained(
            config["repo_id"],
            torch_dtype=self.dtype,
        )
        pipeline = pipeline.to(self.device)
        
        # Enable memory optimizations
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
            
        return pipeline

    async def _load_qwen_lightning(self, config: dict):
        """Load Qwen FP8 Lightning model."""
        logger.info(f"Loading Qwen Lightning from {config['repo_id']}")
        
        # Download the pre-baked FP8 Lightning model
        model_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["diffusion_file"],
        )
        
        # Load using DiffusionPipeline with single_file
        pipeline = DiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=self.dtype,
        )
        pipeline = pipeline.to(self.device)
        
        # Enable memory optimizations
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
            
        return pipeline
    
    async def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if self.current_model == model_name:
                self.current_model = None
            _clear_vram()
            logger.info(f"Model {model_name} unloaded")
    
    async def unload_all_models(self):
        """Unload all models from memory."""
        self.loaded_models.clear()
        self.current_model = None
        _clear_vram()
        logger.info("All models unloaded")
    
    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Generate an image using the specified model."""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not available")
            
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")

        # Load model if not loaded
        model_data = await self.load_model(model_name)
        pipeline = model_data["pipeline"]

        if pipeline is None:
            raise ValueError(f"Pipeline not loaded for {model_name}")

        # Set random seed
        generator = None
        if seed is not None and TORCH_AVAILABLE:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Generating {width}x{height} image with {model_name} ({steps} steps)")

        try:
            # Generate image
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            image = result.images[0]
            logger.info("Generation complete")
            return image
            
        finally:
            # Clear VRAM after generation
            _clear_vram()

    async def edit_image(
        self,
        model_name: str,
        image,
        prompt: str,
        steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Edit an image (placeholder - requires img2img pipeline)."""
        logger.warning(f"Image editing not yet implemented for {model_name}")
        return image


# Global model manager instance
model_manager = ModelManager()
