# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
FastAPI API routes for AutoLabeler.
Aggregates all domain-specific sub-routers into a single router.
"""

import torch
import psutil

from fastapi import APIRouter, HTTPException

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import SystemStatus
from ..services.model_manager import get_model_manager
from ..services.labeler import get_labeling_service

# Import sub-routers
from .datasets import router as datasets_router
from .jobs import router as jobs_router
from .models import router as models_router
from .export_routes import router as export_router
from .hitl import router as hitl_router
from .platform import router as platform_router

logger = get_logger("api")
router = APIRouter()

model_manager = get_model_manager()
labeling_service = get_labeling_service()

# ── Include all sub-routers ────────────────────────────────────
router.include_router(datasets_router)
router.include_router(jobs_router)
router.include_router(models_router)
router.include_router(export_router)
router.include_router(hitl_router)
router.include_router(platform_router)


# ── System-level endpoints ─────────────────────────────────────

@router.get("/health", response_model=SystemStatus)
async def health_check():
    """Check system health and status."""
    cpu_usage = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()

    gpu_info = model_manager.get_gpu_info()

    return SystemStatus(
        status="healthy",
        version=settings.APP_VERSION,
        gpu_available=torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
        gpu_info=gpu_info,
        cpu_count=psutil.cpu_count(),
        cpu_usage=cpu_usage,
        memory_gb=round(mem.total / (1024**3), 2),
        memory_usage_percent=mem.percent,
        disk_space_gb=round(psutil.disk_usage('/').free / (1024**3), 2),
        active_jobs=len([j for j in labeling_service.active_jobs.values() if j.status == "running"]),
        models_loaded=list(model_manager.models.keys()),
        device_preference=str(model_manager.device_preference)
    )


@router.post("/system/device")
async def set_system_device(device: str):
    """Set the global system device preference."""
    try:
        model_manager.set_device(device)
        return {"status": "success", "device": model_manager.device_preference}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/device")
async def get_device_info():
    """Get current device configuration and available options."""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_info = None
        if gpu_available:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            }

        return {
            "current_device": model_manager.device,
            "gpu_available": gpu_available,
            "gpu_info": gpu_info,
            "options": ["gpu", "cpu", "auto"]
        }
    except Exception:
        return {
            "current_device": "cpu",
            "gpu_available": False,
            "gpu_info": None,
            "options": ["cpu"]
        }


@router.post("/device")
async def set_device(device: str):
    """Set the device for model inference."""
    try:
        device = device.lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            if not torch.cuda.is_available():
                raise HTTPException(status_code=400, detail="GPU not available")
            device = "cuda"
        elif device != "cpu":
            raise HTTPException(status_code=400, detail="Invalid device. Use 'gpu', 'cpu', or 'auto'")

        model_manager.set_device(device)

        return {
            "message": f"Device set to {device}",
            "device": device
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
