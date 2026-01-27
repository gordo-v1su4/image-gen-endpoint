"""Image processing service using Pillow."""

from typing import List
from PIL import Image, ImageFilter, ImageEnhance

from app.models import EditOperation


# Placeholder function removed - only real AI inference is supported
# To set up real AI models, run: ./setup_flux_klein.py


def apply_edit_operations(image: Image.Image, operations: List[EditOperation]) -> Image.Image:
    """Apply a list of edit operations to an image."""
    result = image.copy()
    
    for op in operations:
        result = apply_single_operation(result, op)
    
    return result


def apply_single_operation(image: Image.Image, operation: EditOperation) -> Image.Image:
    """Apply a single edit operation."""
    op_type = operation.type.lower()
    params = operation.params
    
    if op_type == "resize":
        width = params.get("width", image.width)
        height = params.get("height", image.height)
        return image.resize((width, height), Image.Resampling.LANCZOS)
    
    elif op_type == "crop":
        x = params.get("x", 0)
        y = params.get("y", 0)
        width = params.get("width", image.width)
        height = params.get("height", image.height)
        return image.crop((x, y, x + width, y + height))
    
    elif op_type == "rotate":
        degrees = params.get("degrees", 0)
        expand = params.get("expand", True)
        return image.rotate(degrees, expand=expand, resample=Image.Resampling.BICUBIC)
    
    elif op_type == "flip":
        direction = params.get("direction", "horizontal")
        if direction == "horizontal":
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    elif op_type == "filter":
        filter_type = params.get("type", "blur")
        strength = params.get("strength", 1.0)
        
        if filter_type == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=strength * 2))
        elif filter_type == "sharpen":
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(1 + strength)
        elif filter_type == "grayscale":
            return image.convert("L").convert("RGB")
        elif filter_type == "contour":
            return image.filter(ImageFilter.CONTOUR)
        elif filter_type == "emboss":
            return image.filter(ImageFilter.EMBOSS)
    
    elif op_type == "adjust":
        result = image
        if "brightness" in params:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(params["brightness"])
        if "contrast" in params:
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(params["contrast"])
        if "saturation" in params:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(params["saturation"])
        return result
    
    return image
