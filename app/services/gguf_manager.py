"""GGUF model manager using stable-diffusion.cpp."""
import asyncio
import os
import uuid
from pathlib import Path
from PIL import Image
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Model configurations for GGUF models
GGUF_MODEL_CONFIGS = {
    "qwen-2512-gguf": {
        "diffusion_model": "qwen-image-2512-Q4_K_M.gguf",
        "text_encoder": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "vae": "qwen_image_vae.safetensors",
        "default_steps": 40,
        "default_cfg": 2.5,
        "flow_shift": 3,
        "default_size": (1328, 1328),
        "description": "Qwen-Image-2512 - High quality text-to-image generation",
    },
    "qwen-edit-gguf": {
        "diffusion_model": "qwen-image-edit-2511-Q4_K_M.gguf",
        "text_encoder": "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "vae": "qwen_image_vae.safetensors",
        "default_steps": 40,
        "default_cfg": 4.0,
        "flow_shift": 3,
        "default_size": (1328, 1328),
        "description": "Qwen-Image-Edit-2511 - AI-powered image editing",
    },
}


class GGUFModelManager:
    """Manages GGUF models via stable-diffusion.cpp CLI."""

    def __init__(self, model_name: str = "qwen-2512-gguf"):
        self.sd_cli = Path(os.getenv("SD_CLI_PATH", "/opt/stable-diffusion.cpp/build/bin/sd-cli"))
        self.models_path = Path(os.getenv("GGUF_MODELS_PATH", "/opt/models/qwen-gguf"))
        self.current_model = model_name

        if not self.sd_cli.exists():
            raise FileNotFoundError(f"sd-cli not found at {self.sd_cli}")

        # Verify model files exist for the selected model
        self._verify_model_files(model_name)

        logger.info(f"✅ GGUF Model Manager initialized with {model_name} at {self.models_path}")

    def _verify_model_files(self, model_name: str):
        """Verify that required model files exist."""
        if model_name not in GGUF_MODEL_CONFIGS:
            raise ValueError(f"Unknown GGUF model: {model_name}")

        config = GGUF_MODEL_CONFIGS[model_name]
        required_files = [
            config["diffusion_model"],
            config["text_encoder"],
            config["vae"],
        ]

        missing = []
        for file in required_files:
            if not (self.models_path / file).exists():
                missing.append(file)

        if missing:
            logger.warning(f"⚠️ Missing model files for {model_name}: {missing}")
            # Don't raise error - allow partial initialization

    def get_model_config(self, model_name: str = None):
        """Get configuration for a model."""
        model_name = model_name or self.current_model
        return GGUF_MODEL_CONFIGS.get(model_name)

    async def generate_image(
        self,
        prompt: str,
        width: int = 1328,
        height: int = 1328,
        steps: int = None,
        cfg_scale: float = None,
        seed: Optional[int] = None,
        model_name: str = None,
    ) -> Image.Image:
        """Generate image using GGUF model.

        Args:
            prompt: Text prompt for image generation
            width: Image width (default: 1328 for Qwen 1:1 aspect ratio)
            height: Image height (default: 1328 for Qwen 1:1 aspect ratio)
            steps: Number of inference steps (default from model config)
            cfg_scale: Classifier-free guidance scale (default from model config)
            seed: Random seed for reproducibility (optional)
            model_name: GGUF model to use (default: qwen-2512-gguf)

        Returns:
            PIL Image object
        """
        model_name = model_name or "qwen-2512-gguf"
        config = GGUF_MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unknown GGUF model: {model_name}")

        # Use defaults from config if not specified
        steps = steps or config["default_steps"]
        cfg_scale = cfg_scale or config["default_cfg"]

        output_path = f"/tmp/gen_{uuid.uuid4()}.png"

        cmd = [
            str(self.sd_cli),
            "--diffusion-model", str(self.models_path / config["diffusion_model"]),
            "--llm", str(self.models_path / config["text_encoder"]),
            "--vae", str(self.models_path / config["vae"]),
            "-p", prompt,
            "-o", output_path,
            "--steps", str(steps),
            "--sampling-method", "euler",
            "--cfg-scale", str(cfg_scale),
            "-H", str(height),
            "-W", str(width),
            "--diffusion-fa",
            "--flow-shift", str(config.get("flow_shift", 3)),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.info(f"🎨 Generating {width}x{height} image with {model_name} ({steps} steps)")
        logger.debug(f"Command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"❌ GGUF generation failed: {error_msg}")
            raise RuntimeError(f"GGUF generation failed: {error_msg}")

        logger.info(f"✅ Image generated successfully at {output_path}")

        image = Image.open(output_path)
        return image

    async def edit_image(
        self,
        prompt: str,
        input_image: Image.Image,
        width: int = None,
        height: int = None,
        steps: int = None,
        cfg_scale: float = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Edit an image using the Qwen-Image-Edit model.

        Args:
            prompt: Text prompt describing the edit
            input_image: PIL Image to edit
            width: Output width (default: input image width)
            height: Output height (default: input image height)
            steps: Number of inference steps (default: 40)
            cfg_scale: Classifier-free guidance scale (default: 4.0)
            seed: Random seed for reproducibility (optional)

        Returns:
            PIL Image object
        """
        config = GGUF_MODEL_CONFIGS.get("qwen-edit-gguf")
        if not config:
            raise ValueError("Edit model not configured")

        # Use input image dimensions if not specified
        width = width or input_image.width
        height = height or input_image.height
        steps = steps or config["default_steps"]
        cfg_scale = cfg_scale or config["default_cfg"]

        # Save input image to temp file
        input_path = f"/tmp/edit_input_{uuid.uuid4()}.png"
        output_path = f"/tmp/edit_output_{uuid.uuid4()}.png"
        input_image.save(input_path)

        cmd = [
            str(self.sd_cli),
            "--diffusion-model", str(self.models_path / config["diffusion_model"]),
            "--llm", str(self.models_path / config["text_encoder"]),
            "--vae", str(self.models_path / config["vae"]),
            "-p", prompt,
            "-i", input_path,  # Input image for editing
            "-o", output_path,
            "--steps", str(steps),
            "--sampling-method", "euler",
            "--cfg-scale", str(cfg_scale),
            "-H", str(height),
            "-W", str(width),
            "--diffusion-fa",
            "--flow-shift", str(config.get("flow_shift", 3)),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.info(f"✏️ Editing image with qwen-edit-gguf ({steps} steps)")
        logger.debug(f"Command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        # Clean up input file
        try:
            os.remove(input_path)
        except OSError:
            pass

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"❌ GGUF edit failed: {error_msg}")
            raise RuntimeError(f"GGUF edit failed: {error_msg}")

        logger.info(f"✅ Image edited successfully at {output_path}")

        image = Image.open(output_path)
        return image
