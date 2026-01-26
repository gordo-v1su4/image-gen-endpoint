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
)
from app.services.image_processor import apply_edit_operations
from app.utils.image_utils import image_to_base64, load_image_from_upload, load_image_from_base64

router = APIRouter()


@router.post("/edit", response_model=ImageEditResponse)
async def edit_image(
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    operations: str = Form(...),
    prompt: Optional[str] = Form(None),
    model: str = Form("qwen-edit-2511"),
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
        
        return ImageEditResponse(
            success=True,
            data={"images": [response_image]},
            metadata={
                "processing_time_ms": processing_time,
                "model": model,
                "operations": [op.type for op in edit_ops],
                "prompt": prompt,
            },
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid operations JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
