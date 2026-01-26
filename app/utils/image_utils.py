"""Image encoding/decoding utilities."""

import base64
import io
from PIL import Image
from fastapi import UploadFile


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(data: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Remove data URL prefix if present
    if "," in data:
        data = data.split(",", 1)[1]
    
    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes))


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load PIL Image from FastAPI UploadFile."""
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def load_image_from_base64(data: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    return base64_to_image(data).convert("RGB")
