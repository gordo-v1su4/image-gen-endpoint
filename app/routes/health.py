"""Health check endpoint."""

from fastapi import APIRouter

from app.models import HealthResponse

router = APIRouter()


def get_gpu_info() -> dict:
    """Get GPU information if available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            return {
                "cuda_available": True,
                "gpu_name": gpu_name,
                "gpu_memory_total": round(gpu_memory_total, 2),
                "gpu_memory_used": round(gpu_memory_used, 2),
            }
        else:
            return {"cuda_available": False, "reason": "CUDA not available"}
    except ImportError:
        return {"cuda_available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"cuda_available": False, "reason": str(e)}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and GPU status."""
    gpu_info = get_gpu_info()
    return HealthResponse(status="healthy", **gpu_info)
