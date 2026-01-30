"""Model management service for Qwen-Image Lightning models.

Uses FP8 quantization via Quanto for efficient GPU inference.
Target: ~14-16GB VRAM on 24GB GPUs (RTX 4090).
"""

import gc
import math
import os
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import torch
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch available: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Import diffusers with Quanto support
DIFFUSERS_AVAILABLE = False
QUANTO_AVAILABLE = False
try:
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import QwenImageTransformer2DModel
    from huggingface_hub import hf_hub_download
    DIFFUSERS_AVAILABLE = True
    logger.info("diffusers available")
    
    # Check for Quanto quantization support
    try:
        from diffusers import QuantoConfig
        QUANTO_AVAILABLE = True
        logger.info("Quanto FP8 quantization available")
    except ImportError:
        logger.warning("Quanto not available - will use bf16")
except ImportError as e:
    DiffusionPipeline = None
    logger.warning(f"diffusers not available: {e}")

# Model configuration - Qwen-Image-2512 Lightning (4-step)
MODEL_CONFIG = {
    "name": "qwen-image-2512-lightning",
    "base_repo": "Qwen/Qwen-Image-2512",
    "lora_repo": "lightx2v/Qwen-Image-2512-Lightning",
    "lora_file": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    "steps": 4,
    "guidance_scale": 1.0,
    "supported_sizes": {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
    },
    "default_size": (1024, 1024),
}


def _clear_vram():
    """Clear VRAM."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class ModelManager:
    """Manages Qwen-Image Lightning model with FP8 quantization."""
    
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU: {gpu_name}, VRAM: {gpu_mem:.1f}GB")
        else:
            self.device = "cpu"
            self.dtype = None
        
        logger.info(f"ModelManager: device={self.device}, dtype={self.dtype}")
        logger.info(f"Quanto FP8: {'enabled' if QUANTO_AVAILABLE else 'disabled (using bf16)'}")
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory stats."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used = torch.cuda.memory_allocated(0) / (1024**3)
        return {"total": round(total, 2), "used": round(used, 2), "free": round(total - used, 2)}
    
    def list_models(self) -> list:
        """List available models."""
        return [{
            "name": MODEL_CONFIG["name"],
            "status": "ready" if self.model_loaded else "not_loaded",
            "steps": MODEL_CONFIG["steps"],
            "description": "Qwen-Image-2512 Lightning 4-step (FP8)" if QUANTO_AVAILABLE else "Qwen-Image-2512 Lightning 4-step",
        }]
    
    async def load_model(self, model_name: str = None):
        """Load the Qwen-Image Lightning model with FP8 quantization."""
        if self.model_loaded and self.pipeline is not None:
            logger.info("Model already loaded")
            return {"config": MODEL_CONFIG, "pipeline": self.pipeline}
        
        logger.info("=" * 50)
        logger.info("LOADING QWEN-IMAGE-2512 LIGHTNING MODEL")
        if QUANTO_AVAILABLE:
            logger.info("Using FP8 quantization for efficient VRAM usage")
        logger.info("=" * 50)
        
        config = MODEL_CONFIG
        
        # Step 1: Load transformer (with FP8 quantization if available)
        logger.info(f"[1/4] Loading transformer from {config['base_repo']}...")
        
        if QUANTO_AVAILABLE:
            # FP8 quantization - reduces VRAM by ~50%
            quantization_config = QuantoConfig(weights_dtype="float8")
            transformer = QwenImageTransformer2DModel.from_pretrained(
                config["base_repo"],
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=self.dtype,
            )
            logger.info("Transformer loaded with FP8 quantization!")
        else:
            # Fallback to bf16
            transformer = QwenImageTransformer2DModel.from_pretrained(
                config["base_repo"],
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            logger.info("Transformer loaded (bf16)!")
        
        mem = self.get_gpu_memory()
        logger.info(f"VRAM after transformer: {mem['used']:.1f}GB / {mem['total']:.1f}GB")
        
        # Step 2: Create scheduler with Lightning config (shift=3)
        logger.info("[2/4] Creating Lightning scheduler (shift=3)...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "use_dynamic_shifting": True,
            "time_shift_type": "exponential",
        })
        logger.info("Scheduler created!")
        
        # Step 3: Load full pipeline
        logger.info(f"[3/4] Loading pipeline from {config['base_repo']}...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            config["base_repo"],
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=self.dtype,
        )
        logger.info("Pipeline loaded!")
        
        mem = self.get_gpu_memory()
        logger.info(f"VRAM after pipeline: {mem['used']:.1f}GB / {mem['total']:.1f}GB")
        
        # Step 4: Download and apply Lightning LoRA
        logger.info(f"[4/4] Downloading Lightning LoRA from {config['lora_repo']}...")
        lora_path = hf_hub_download(
            repo_id=config["lora_repo"],
            filename=config["lora_file"],
        )
        logger.info(f"LoRA downloaded: {lora_path}")
        
        logger.info("Applying LoRA weights...")
        self.pipeline.load_lora_weights(lora_path)
        logger.info("LoRA applied!")
        
        # Move to GPU - full GPU inference, no CPU offload
        logger.info(f"Moving pipeline to {self.device}...")
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory optimizations for VAE
        if hasattr(self.pipeline, "vae"):
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()
            logger.info("VAE slicing/tiling enabled")
        
        self.model_loaded = True
        
        # Log final GPU memory
        mem = self.get_gpu_memory()
        logger.info("=" * 50)
        logger.info(f"MODEL LOADED! VRAM: {mem['used']:.1f}GB / {mem['total']:.1f}GB")
        logger.info("=" * 50)
        
        return {"config": config, "pipeline": self.pipeline}
    
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
        """Generate an image using full GPU inference."""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not available")
        
        # Load model if needed
        if not self.model_loaded:
            await self.load_model()
        
        logger.info(f"Generating {width}x{height} image, {steps} steps, seed={seed}")
        
        # Generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate on GPU
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=guidance_scale,
                generator=generator,
            )
        
        image = result.images[0]
        logger.info("Generation complete!")
        
        _clear_vram()
        return image
    
    async def unload_model(self, model_name: str = None):
        """Unload model."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.model_loaded = False
        _clear_vram()
        logger.info("Model unloaded")
    
    async def unload_all_models(self):
        """Unload all models."""
        await self.unload_model()


# Global instance
model_manager = ModelManager()
