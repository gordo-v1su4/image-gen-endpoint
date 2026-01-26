"""Image generation endpoint."""

import time
import uuid
from fastapi import APIRouter, HTTPException

from app.models import (
    ImageCreateRequest,
    ImageCreateResponse,
    ImageResponse,
    ImageMetadata,
    ImageFormat,
)
from app.services.image_processor import generate_placeholder_image
from app.utils.image_utils import image_to_base64

router = APIRouter()


@router.post("/create", response_model=ImageCreateResponse)
async def create_image(request: ImageCreateRequest):
    """Generate an image from a text prompt."""
    start_time = time.time()
    
    try:
        # For now, generate a placeholder image
        # TODO: Integrate actual model inference
        image = generate_placeholder_image(
            width=request.width,
            height=request.height,
            text=f"Prompt: {request.prompt[:50]}..."
        )
        
        image_data = image_to_base64(image)
        image_id = str(uuid.uuid4())
        processing_time = int((time.time() - start_time) * 1000)
        
        response_image = ImageResponse(
            id=image_id,
            data=image_data,
            format=ImageFormat.PNG,
            metadata=ImageMetadata(
                width=request.width,
                height=request.height,
                format=ImageFormat.PNG,
                model_used=request.model.value,
                steps=request.steps,
                seed=request.seed,
            ),
        )
        
        return ImageCreateResponse(
            success=True,
            data={"images": [response_image]},
            metadata={
                "processing_time_ms": processing_time,
                "model": request.model.value,
                "prompt": request.prompt,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
