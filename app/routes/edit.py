"""Image editing endpoint."""

import json
import time
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.models import (
    ImageEditResponse,
    ImageResponse,
    ImageMetadata,
    ImageFormat,
    EditOperation,
    OpenAIImageData,
)
from app.services.image_processor import apply_edit_operations
from app.services.model_manager import model_manager
from app.utils.image_utils import image_to_base64, load_image_from_upload, load_image_from_base64

router = APIRouter()


@router.post("/edit", response_model=ImageEditResponse)
async def edit_image(
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    operations: str = Form(...),
    prompt: Optional[str] = Form(None),
    model: str = Form("qwen-2512"),
    steps: int = Form(8),
):
    """Edit an image with specified operations."""
    start_time = time.time()
    
    try:
        ops_list = json.loads(operations)
        edit_ops = [EditOperation(**op) for op in ops_list]
        
        if image:
            input_image = await load_image_from_upload(image)
        elif image_base64:
            input_image = load_image_from_base64(image_base64)
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        result_image = apply_edit_operations(input_image, edit_ops)
        image_data = image_to_base64(result_image)
        
        image_id = str(uuid.uuid4())
        processing_time = int((time.time() - start_time) * 1000)
        
        response_image = ImageResponse(
            id=image_id,
            data=image_data,
            format=ImageFormat.PNG,
            metadata=ImageMetadata(
                width=result_image.width,
                height=result_image.height,
                format=ImageFormat.PNG,
                model_used=model,
                steps=steps,
            ),
        )

        # Create OpenAI-compatible format
        data_uri = f"data:image/png;base64,{image_data}"
        openai_image = OpenAIImageData(
            b64_json=image_data,
            url=data_uri
        )

        # Return both formats for maximum compatibility
        return ImageEditResponse(
            success=True,
            data=[openai_image],  # OpenAI-compatible format for Gemini
            created=int(time.time()),
            model=model,
            images=[response_image],  # Original format for backward compatibility
            metadata={
                "processing_time_ms": processing_time,
                "operations": [op.type for op in edit_ops],
                "prompt": prompt,
            },
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid operations JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit/ai", response_model=ImageEditResponse)
async def edit_image_ai(
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    prompt: str = Form(..., description="Text prompt describing the edit"),
    model: str = Form("qwen-2512-fp8-4step", description="Model to use for editing"),
    steps: int = Form(4, ge=1, le=50, description="Number of inference steps"),
    guidance_scale: float = Form(1.0, ge=0.0, le=20.0, description="Guidance scale"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
):
    """Edit an image using AI (Qwen-Image-Edit-2511).

    This endpoint uses the Qwen-Image-Edit model to perform AI-powered edits
    based on natural language prompts.

    Example prompts:
    - "make the sky more dramatic with storm clouds"
    - "change the color of the car to red"
    - "add a cat sitting on the table"
    - "remove the person in the background"
    """
    start_time = time.time()

    try:
        # Load input image
        if image:
            input_image = await load_image_from_upload(image)
        elif image_base64:
            input_image = load_image_from_base64(image_base64)
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        # Perform AI editing
        result_image = await model_manager.edit_image(
            model_name=model,
            image=input_image,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        image_data = image_to_base64(result_image)
        image_id = str(uuid.uuid4())
        processing_time = int((time.time() - start_time) * 1000)

        response_image = ImageResponse(
            id=image_id,
            data=image_data,
            format=ImageFormat.PNG,
            metadata=ImageMetadata(
                width=result_image.width,
                height=result_image.height,
                format=ImageFormat.PNG,
                model_used=model,
                steps=steps,
                seed=seed,
            ),
        )

        # Create OpenAI-compatible format
        data_uri = f"data:image/png;base64,{image_data}"
        openai_image = OpenAIImageData(
            b64_json=image_data,
            url=data_uri
        )

        return ImageEditResponse(
            success=True,
            data=[openai_image],
            created=int(time.time()),
            model=model,
            images=[response_image],
            metadata={
                "processing_time_ms": processing_time,
                "prompt": prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
