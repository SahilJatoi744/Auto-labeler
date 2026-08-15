# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Labeling job management API routes."""

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import (
    DatasetInfo, LabelingJob, LabelingProgress, ImageLabels,
    LabelAnnotation, JobCreate, RefineRequest
)
from ..services.labeler import get_labeling_service
from ..services.model_manager import get_model_manager
from ..services.platform import get_platform_service
from ..services.intelligence import get_intelligence_service

logger = get_logger("api.jobs")
router = APIRouter(tags=["jobs"])

labeling_service = get_labeling_service()
model_manager = get_model_manager()
platform = get_platform_service()
intelligence = get_intelligence_service()


@router.post("/jobs", response_model=LabelingJob)
async def create_labeling_job(job_config: JobCreate, background_tasks: BackgroundTasks):
    """Create a new labeling job."""
    try:
        dataset_id = job_config.dataset_id
        metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        with open(metadata_path) as f:
            data = json.load(f)

        dataset_info = DatasetInfo(
            id=data["dataset_id"],
            name=data.get("name", dataset_id),
            path=data.get("path", ""),
            total_images=data.get("total_images", 0),
            valid_images=data.get("valid_images", 0),
            corrupted_images=data.get("corrupted_images", 0),
            total_size_mb=data.get("total_size_mb", 0.0),
            formats=data.get("formats", {}),
            status=data.get("status", "pending")
        )

        job = labeling_service.create_job(
            dataset_info=dataset_info,
            task_type=job_config.task_type,
            class_hierarchy=job_config.class_hierarchy,
            strategy=job_config.strategy,
            confidence_threshold=job_config.confidence_threshold,
            models_config=job_config.models_config or {}
        )

        # Run labeling in the background
        background_tasks.add_task(labeling_service.run_job, job.id)
        platform.record_audit_event("job.create", "job", job.id, {"task_type": job_config.task_type.value})

        return job
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[LabelingJob])
async def get_all_jobs():
    """Get all labeling jobs."""
    try:
        return labeling_service.get_all_jobs()
    except Exception as e:
        logger.error(f"Failed to get jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=LabelingJob)
async def get_job(job_id: str):
    """Get job information."""
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/start")
async def start_labeling_job(job_id: str, background_tasks: BackgroundTasks):
    """Start a labeling job."""
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Run labeling in the background
    background_tasks.add_task(labeling_service.run_job, job_id)
    return {"status": "success", "message": "Job started in the background", "job_id": job_id}


@router.post("/jobs/{job_id}/stop")
async def stop_labeling_job(job_id: str):
    """Stop a running job."""
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    labeling_service.stop_job(job_id)
    return {"status": "success", "message": "Job stop signal sent"}


@router.delete("/jobs/{job_id}")
async def delete_labeling_job(job_id: str):
    """Delete a labeling job and its associated files."""
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        labeling_service.stop_job(job_id)
    labeling_service.delete_job(job_id)
    platform.record_audit_event("job.delete", "job", job_id, {})
    return {"status": "success", "message": "Job deleted"}


@router.get("/jobs/{job_id}/progress", response_model=LabelingProgress)
async def get_job_progress(job_id: str):
    """Get job progress (HTTP endpoint)."""
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return LabelingProgress(
        job_id=job.id,
        total_images=job.total_images,
        processed_images=job.processed_images,
        failed_images=job.failed_images,
        current_image=job.progress.get("current_image") if job.progress else None,
        status=job.status
    )


@router.websocket("/jobs/{job_id}/progress")
async def stream_progress_ws(websocket: WebSocket, job_id: str):
    """WebSocket endpoint to stream job progress."""
    await websocket.accept()
    try:
        async for progress in labeling_service.stream_progress(job_id):
            # Send serialized progress data
            await websocket.send_json({
                "job_id": progress.job_id,
                "total_images": progress.total_images,
                "processed_images": progress.processed_images,
                "failed_images": progress.failed_images,
                "current_image": progress.current_image,
                "current_model": progress.current_model,
                "estimated_time_remaining": progress.estimated_time_remaining,
                "status": progress.status,
                "errors": progress.errors
            })
    except Exception as e:
        logger.error(f"WebSocket progress stream error: {e}")
    finally:
        await websocket.close()


def resolve_image_url(path_str: str, dataset_id: str) -> Optional[str]:
    """Convert an absolute filesystem path to a /uploads/ URL.

    Strategy:
    1. If the path contains 'uploads/', split on it and use the remainder.
    2. If the dataset_id appears in the path, use everything after it.
    3. Fallback: use the filename and search for it under the dataset dir.
    4. Final fallback: construct URL from dataset_id + filename.

    After constructing the candidate URL, verify the file exists under
    UPLOAD_DIR.  If not, walk the dataset directory to find the file by name.
    """
    if not path_str:
        return None

    p_str = str(path_str).replace("\\", "/")
    filename = p_str.rsplit("/", 1)[-1] if "/" in p_str else p_str

    # --- Strategy 1: split on 'uploads/' ---
    candidate = None
    if "uploads/" in p_str:
        rel_str = p_str.split("uploads/")[-1]
        candidate = f"/uploads/{rel_str}"

    # --- Strategy 2: split on dataset_id ---
    if candidate is None and dataset_id and dataset_id in p_str:
        rel_str = p_str.split(dataset_id)[-1].lstrip("/")
        candidate = f"/uploads/{dataset_id}/{rel_str}"

    # --- Verify candidate exists on disk ---
    if candidate:
        check_path = settings.UPLOAD_DIR / candidate.lstrip("/").replace("uploads/", "", 1)
        if check_path.exists():
            return candidate
        # The split-based path didn't resolve; fall through to search

    # --- Strategy 3: search dataset directory for the file ---
    if dataset_id and filename:
        dataset_dir = settings.UPLOAD_DIR / dataset_id
        if dataset_dir.exists():
            for root, _dirs, files in __import__("os").walk(str(dataset_dir)):
                if filename in files:
                    found = Path(root) / filename
                    rel = found.relative_to(settings.UPLOAD_DIR)
                    return f"/uploads/{str(rel).replace(chr(92), '/')}"

    # --- Strategy 4: best-effort from filename ---
    if dataset_id:
        return f"/uploads/{dataset_id}/{filename}"

    parts = p_str.split("/")
    if len(parts) >= 2:
        return f"/uploads/{'/'.join(parts[-2:])}"
    return f"/uploads/{filename}"


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str, image_id: Optional[str] = None):
    """Get labeling results for a job."""
    results = labeling_service.get_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Results not found")

    job = labeling_service.get_job(job_id)
    dataset_id = job.dataset_id if job else None

    if image_id:
        for r in results:
            if r.image_id == image_id:
                r_dict = r.model_dump()
                if not r_dict.get("image_url") and dataset_id:
                    metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"
                    try:
                        with open(metadata_path) as f:
                            meta = json.load(f)
                            img_data = next((img for img in meta.get("images", []) if img["id"] == r.image_id), None)
                            if img_data:
                                r_dict["image_url"] = resolve_image_url(img_data["path"], dataset_id)
                    except Exception:
                        pass
                return r_dict
        raise HTTPException(status_code=404, detail="Image not found in results")

    formatted_results = []
    meta_images = []
    if dataset_id:
        metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
                meta_images = meta.get("images", [])
        except Exception:
            pass

    img_map = {img["id"]: img for img in meta_images}

    for r in results:
        r_dict = r.model_dump()
        # Only resolve from metadata if image_url is not already set
        if not r_dict.get("image_url"):
            img_data = img_map.get(r.image_id)
            if img_data:
                r_dict["image_url"] = resolve_image_url(img_data["path"], dataset_id)
        formatted_results.append(r_dict)

    return {
        "job_id": job_id,
        "total_images": len(results),
        "results": formatted_results
    }


@router.put("/jobs/{job_id}/results")
async def update_job_results_put(job_id: str, results: List[ImageLabels]):
    """Update labeling results for a job (PUT method)."""
    try:
        labeling_service.update_results(job_id, results)
        platform.record_audit_event("job.results.update", "job", job_id, {"images": len(results)})
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        logger.error(f"Failed to update results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/refine", response_model=Optional[LabelAnnotation])
async def refine_job_result(job_id: str, request: RefineRequest):
    """Refine a specific image in a job using a text prompt."""
    try:
        annotation = await labeling_service.refine_image(
            job_id, request.image_id, request.prompt
        )
        if annotation is None:
            raise HTTPException(status_code=404, detail="Could not find object matching prompt")
        return annotation
    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
