"""Pydantic models for API request/response schemas."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model types."""
    QWEN_2512 = "qwen-2512"
    QWEN_EDIT_2511 = "qwen-edit-2511"
    FLUX_KLEIN_4B = "flux-klein-4b"
    FLUX_KLEIN_9B = "flux-klein-9b"


class ImageFormat(str, Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ImageCreateRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    model: ModelType = Field(ModelType.QWEN_2512, description="Model to use")
    width: int = Field(1328, ge=256, le=2048, description="Image width")
    height: int = Field(1328, ge=256, le=2048, description="Image height")
    steps: int = Field(8, ge=1, le=50, description="Number of inference steps")
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    use_lightning: bool = Field(True, description="Use Lightning LoRA for faster inference")


class EditOperation(BaseModel):
    """Single editing operation."""
    type: str = Field(..., description="Operation type: resize, crop, rotate, filter, etc.")
    params: Dict[str, Any] = Field(..., description="Operation-specific parameters")


class ImageEditRequest(BaseModel):
    """Request model for image editing."""
    prompt: Optional[str] = Field(None, description="Text prompt for AI editing")
    operations: List[EditOperation] = Field(..., description="List of edit operations")
    model: ModelType = Field(ModelType.QWEN_EDIT_2511, description="Model to use")
    steps: int = Field(8, ge=1, le=50, description="Number of inference steps")
    use_lightning: bool = Field(True, description="Use Lightning LoRA")


class ImageMetadata(BaseModel):
    """Metadata about a generated/edited image."""
    width: int
    height: int
    format: ImageFormat
    model_used: str
    steps: int
    seed: Optional[int] = None


class ImageResponse(BaseModel):
    """Single image in response."""
    id: str = Field(..., description="Unique image ID")
    data: str = Field(..., description="Base64-encoded image data")
    format: ImageFormat
    metadata: ImageMetadata


class ImageCreateResponse(BaseModel):
    """Response model for image generation."""
    success: bool = True
    data: Dict[str, List[ImageResponse]] = Field(..., description="Generated images")
    metadata: Dict[str, Any] = Field(..., description="Request metadata")


class ImageEditResponse(BaseModel):
    """Response model for image editing."""
    success: bool = True
    data: Dict[str, List[ImageResponse]] = Field(..., description="Edited images")
    metadata: Dict[str, Any] = Field(..., description="Request metadata")


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    type: ModelType
    status: str  # "ready", "loading", "not_loaded"
    vram_usage: Optional[float] = None  # GB
    quantization: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for /models endpoint."""
    success: bool = True
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = "healthy"
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_total: Optional[float] = None  # GB
    gpu_memory_used: Optional[float] = None  # GB
