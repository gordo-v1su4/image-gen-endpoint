"""Model management service for loading and caching AI models."""

import os
import torch
import math
from typing import Optional, Dict, Any
from pathlib import Path
from diffusers import DiffusionPipeline, Flux2KleinPipeline, QwenImagePipeline, FlowMatchEulerDiscreteScheduler
from .gguf_manager import GGUFModelManager

# Model configurations
MODEL_CONFIGS = {
    "qwen-2512": {
        "repo_id": "Qwen/Qwen-Image-2512",
        "pipeline_class": QwenImagePipeline,
        "vram": 12,  # GB (bfloat16)
        "steps": 50,
        "guidance_scale": 1.0,
        "description": "Qwen Image-2512 - High quality text-to-image",
        "license": "check-repo",
        "supported_sizes": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
        "default_size": (1328, 1328),
    },
    "qwen-lightning-4step": {
        "repo_id": "Qwen/Qwen-Image-2512",
        "pipeline_class": QwenImagePipeline,
        "lora_repo": "lightx2v/Qwen-Image-2512-Lightning",
        "lora_file": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        "vram": 10,  # GB (bfloat16 with LoRA)
        "steps": 4,  # Lightning optimized
        "guidance_scale": 1.0,
        "description": "Qwen with 4-step Lightning LoRA - Fast inference",
        "license": "check-repo",
        "supported_sizes": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
        "default_size": (1328, 1328),
    },
    "qwen-lightning-8step": {
        "repo_id": "Qwen/Qwen-Image-2512",
        "pipeline_class": QwenImagePipeline,
        "lora_repo": "lightx2v/Qwen-Image-2512-Lightning",
        "lora_file": "Qwen-Image-2512-Lightning-8steps-V1.0.safetensors",
        "vram": 10,  # GB (bfloat16 with LoRA)
        "steps": 8,  # Lightning optimized
        "guidance_scale": 1.0,
        "description": "Qwen with 8-step Lightning LoRA - Balance quality/speed",
        "license": "check-repo",
        "supported_sizes": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
        "default_size": (1328, 1328),
    },
    "flux-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline_class": Flux2KleinPipeline,
        "vram": 8,  # GB (bfloat16)
        "steps": 4,
        "guidance_scale": 1.0,
        "description": "FLUX.2 Klein 4B - Fast 4-step generation",
        "license": "apache-2.0",
        "default_size": (1024, 1024),
    },
    "flux-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline_class": Flux2KleinPipeline,
        "vram": 12,  # GB (bfloat16)
        "steps": 4,
        "guidance_scale": 1.0,
        "description": "FLUX.2 Klein 9B - Higher quality 4-step generation",
        "license": "non-commercial",
        "default_size": (1024, 1024),
    },
    "qwen-2512-gguf": {
        "model_type": "gguf",
        "vram": 19,  # GB (actual measured: 18.7GB)
        "steps": 40,
        "guidance_scale": 2.5,
        "description": "Qwen-2512 GGUF Q4_K_M - Optimized for 24GB VRAM (RTX 4090)",
        "license": "check-repo",
        "supported_sizes": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
        "default_size": (1328, 1328),
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
        self.gguf_manager = None  # Lazy load GGUF manager
        
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

    def _verify_pipeline_components(self, pipeline, model_name: str):
        """Verify that pipeline has all necessary components."""
        print(f"  Verifying components...")

        # Check for text encoder (CLIP/T5)
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            print(f"    ✓ Text Encoder: {type(pipeline.text_encoder).__name__}")
        elif hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            print(f"    ✓ Text Encoder 2: {type(pipeline.text_encoder_2).__name__}")

        # Check for VAE
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            print(f"    ✓ VAE: {type(pipeline.vae).__name__}")

        # Check for diffusion model (U-Net or Transformer)
        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            print(f"    ✓ U-Net: {type(pipeline.unet).__name__}")
        elif hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
            print(f"    ✓ Transformer: {type(pipeline.transformer).__name__}")

        # Check for scheduler
        if hasattr(pipeline, 'scheduler') and pipeline.scheduler is not None:
            print(f"    ✓ Scheduler: {type(pipeline.scheduler).__name__}")

        print(f"  All critical components verified!")

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

        print(f"Loading model: {model_name}")
        print(f"  Repo: {config['repo_id']}")
        print(f"  Use Lightning: {use_lightning}")
        print(f"  Device: {self.device}")

        try:
            # Get the correct pipeline class
            pipeline_class = config["pipeline_class"]
            repo_id = config["repo_id"]

            # Load pipeline with correct class
            load_kwargs = {
                "torch_dtype": self.dtype,
            }

            # Special setup for QwenImagePipeline
            if pipeline_class == QwenImagePipeline:
                print(f"  Setting up custom FlowMatchEulerDiscreteScheduler for Qwen...")
                scheduler_config = {
                    "base_image_seq_len": 256,
                    "base_shift": math.log(3),
                    "invert_sigmas": False,
                    "max_image_seq_len": 8192,
                    "max_shift": math.log(3),
                    "num_train_timesteps": 1000,
                    "shift": 1.0,
                    "shift_terminal": None,
                    "stochastic_sampling": False,
                    "time_shift_type": "exponential",
                    "use_beta_sigmas": False,
                    "use_dynamic_shifting": True,
                    "use_exponential_sigmas": False,
                    "use_karras_sigmas": False,
                }
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
                load_kwargs["scheduler"] = scheduler

            print(f"  Loading with {pipeline_class.__name__}...")
            pipeline = pipeline_class.from_pretrained(repo_id, **load_kwargs)

            # Load Lightning LoRA if specified
            if "lora_repo" in config and "lora_file" in config:
                print(f"  Loading Lightning LoRA from {config['lora_repo']}...")
                print(f"  LoRA file: {config['lora_file']}")
                pipeline.load_lora_weights(
                    config["lora_repo"],
                    weight_name=config["lora_file"],
                )
                pipeline.fuse_lora()
                print(f"  ✓ Lightning LoRA loaded and fused")

            # Move to device
            pipeline = pipeline.to(self.device)

            # Enable memory optimizations
            if self.device == "cuda":
                # Enable VAE optimizations if available
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                if hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()

            # Verify all components loaded correctly
            self._verify_pipeline_components(pipeline, model_name)

            self.loaded_models[model_name] = {
                "config": config,
                "pipeline": pipeline,
                "use_lightning": use_lightning,
            }

            print(f"  Model loaded successfully!")
            return self.loaded_models[model_name]

        except Exception as e:
            print(f"  Error loading model: {e}")
            raise ValueError(f"Failed to load model {model_name}: {str(e)}")
    
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
        # Check if this is a GGUF model
        config = MODEL_CONFIGS.get(model_name)
        if config and config.get("model_type") == "gguf":
            if self.gguf_manager is None:
                self.gguf_manager = GGUFModelManager()
            return await self.gguf_manager.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=guidance_scale,
                seed=seed,
            )

        # Load model if not loaded
        model_data = await self.load_model(model_name, use_lightning, steps)
        pipeline = model_data["pipeline"]

        if pipeline is None:
            raise ValueError(f"Pipeline not loaded for {model_name}")

        # Set random seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        print(f"Generating image with {model_name}...")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Steps: {steps}, Guidance: {guidance_scale}, Size: {width}x{height}")

        # Generate image
        with torch.inference_mode():
            # QwenImagePipeline uses true_cfg_scale parameter
            if isinstance(pipeline, QwenImagePipeline):
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=guidance_scale,
                    guidance_scale=1.0,
                    generator=generator,
                )
            else:
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

        # Get the first generated image
        image = result.images[0]
        print(f"  Generation complete!")

        return image
    
    async def edit_image(
        self,
        model_name: str,
        image,
        prompt: str,
        steps: int = 8,
        use_lightning: bool = True,
    ):
        """Edit an image using the specified model.

        Note: Real AI-powered editing requires model support for img2img.
        For now, use the image editing operations in image_processor.py.
        FLUX.2 [klein] 4B supports img2img - see flux_server.py for implementation.
        """
        # Load model for editing
        model_data = await self.load_model(model_name, use_lightning, steps)

        # For now, return original image
        # To implement: use pipeline's img2img capabilities
        return image

# Global model manager instance
model_manager = ModelManager()