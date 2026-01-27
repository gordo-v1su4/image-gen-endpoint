"""GGUF model manager using stable-diffusion.cpp."""
import asyncio
import os
import uuid
from pathlib import Path
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GGUFModelManager:
    """Manages GGUF models via stable-diffusion.cpp CLI."""

    def __init__(self):
        self.sd_cli = Path(os.getenv("SD_CLI_PATH", "/opt/stable-diffusion.cpp/build/bin/sd-cli"))
        self.models_path = Path(os.getenv("GGUF_MODELS_PATH", "/opt/models/qwen-gguf"))

        if not self.sd_cli.exists():
            raise FileNotFoundError(f"sd-cli not found at {self.sd_cli}")

        required_files = [
            "qwen-image-2512-Q4_K_M.gguf",
            "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
            "qwen_image_vae.safetensors"
        ]

        for file in required_files:
            if not (self.models_path / file).exists():
                raise FileNotFoundError(f"Model file not found: {file}")

        logger.info(f"✅ GGUF Model Manager initialized with models at {self.models_path}")

    async def generate_image(
        self,
        prompt: str,
        width: int = 1328,
        height: int = 1328,
        steps: int = 40,
        cfg_scale: float = 2.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate image using GGUF model.

        Args:
            prompt: Text prompt for image generation
            width: Image width (default: 1328 for Qwen 1:1 aspect ratio)
            height: Image height (default: 1328 for Qwen 1:1 aspect ratio)
            steps: Number of inference steps (default: 40)
            cfg_scale: Classifier-free guidance scale (default: 2.5)
            seed: Random seed for reproducibility (optional)

        Returns:
            PIL Image object
        """
        output_path = f"/tmp/gen_{uuid.uuid4()}.png"

        cmd = [
            str(self.sd_cli),
            "--diffusion-model", str(self.models_path / "qwen-image-2512-Q4_K_M.gguf"),
            "--llm", str(self.models_path / "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"),
            "--vae", str(self.models_path / "qwen_image_vae.safetensors"),
            "-p", prompt,
            "-o", output_path,
            "--steps", str(steps),
            "--sampling-method", "euler",
            "--cfg-scale", str(cfg_scale),
            "-H", str(height),
            "-W", str(width),
            "--diffusion-fa",
            "--flow-shift", "3",
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        logger.info(f"🎨 Generating {width}x{height} image with {steps} steps")
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
