# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Model management API routes."""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..core.config import settings
from ..core.logging import get_logger
from ..services.model_manager import get_model_manager

logger = get_logger("api.models")
router = APIRouter(tags=["models"])

model_manager = get_model_manager()


@router.get("/models/status")
async def get_models_status():
    """Get the download status of models."""
    return model_manager.get_download_status()


@router.post("/models/download")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Trigger a model download in the background."""
    valid_models = ["yolo", "yolov26", "sam", "sam2"]
    if model_name.lower() not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Valid options: {valid_models}")

    status = model_manager.get_download_status().get(model_name.lower())
    if status and status["status"] == "downloading":
        return {"message": f"{model_name} is already downloading", "status": status}

    background_tasks.add_task(model_manager.download_model_task, model_name.lower())
    return {"message": f"Started download for {model_name}"}


@router.get("/models")
async def list_models():
    """List available AI models."""
    return {
        "models": [
            {
                "name": "YOLOv8",
                "variants": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                "tasks": ["object_detection", "instance_segmentation"],
                "loaded": "yolo" in model_manager.models
            },
            {
                "name": "YOLOv26",
                "variants": ["yolov26x", "yolov26l"],
                "tasks": ["object_detection", "instance_segmentation"],
                "loaded": "yolov26" in model_manager.models
            },
            {
                "name": "SAM",
                "variants": ["vit_h", "vit_l", "vit_b"],
                "tasks": ["semantic_segmentation", "instance_segmentation"],
                "loaded": "sam" in model_manager.models
            },
            {
                "name": "SAM2",
                "variants": ["sam2_l", "sam2_b", "sam2_t"],
                "tasks": ["semantic_segmentation", "instance_segmentation"],
                "loaded": "sam2" in model_manager.models
            }
        ]
    }


@router.post("/models/load")
async def load_model(model_name: str, variant: Optional[str] = None):
    """Load a specific model."""
    try:
        if model_name.lower() == "yolo":
            variant = variant or settings.YOLOV8_MODEL
            model_manager.load_yolo(variant)
            return {"message": f"YOLO model {variant} loaded"}

        elif model_name.lower() in ["yolov26", "yolo26"]:
            variant = variant or settings.YOLOV26_MODEL
            model_manager.load_yolo(variant)
            return {"message": f"YOLOv26 model {variant} loaded"}

        elif model_name.lower() == "sam":
            variant = variant or "vit_h"
            model_manager.load_sam(variant)
            return {"message": f"SAM model {variant} loaded"}

        elif model_name.lower() == "sam2":
            variant = variant or settings.SAM2_MODEL
            model_manager.load_sam2(variant)
            return {"message": f"SAM2 model {variant} loaded"}

        else:
            valid_models = ["yolo", "yolov26", "sam", "sam2"]
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}. Valid options: {valid_models}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model(model_name: str):
    """Unload a model to free memory."""
    model_manager.unload_model(model_name.lower())
    return {"message": f"Model {model_name} unloaded"}
