# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Export API routes."""

import shutil

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import ExportConfig, ExportResult
from ..services.labeler import get_labeling_service
from ..services.exporter import get_export_service
from ..services.platform import get_platform_service

logger = get_logger("api.export")
router = APIRouter(tags=["export"])

labeling_service = get_labeling_service()
export_service = get_export_service()
platform = get_platform_service()


@router.post("/export", response_model=ExportResult)
async def export_labels(config: ExportConfig):
    """Export labeled dataset in specified format."""
    try:
        results = labeling_service.get_results(config.job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Job results not found")

        job = labeling_service.get_job(config.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        result = export_service.export_dataset(
            job_id=config.job_id,
            results=results,
            class_hierarchy=job.class_hierarchy,
            config=config
        )
        platform.validate_export(config.job_id, results)
        platform.record_audit_event("export.create", "job", config.job_id, {"format": config.format.value})
        platform.record_metric("exports.created", 1, {"job_id": config.job_id, "format": config.format.value})
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{export_id}/download")
async def download_export(export_id: str, format: str = "zip"):
    """Download exported dataset as ZIP file."""
    export_dir = None
    for d in settings.OUTPUT_DIR.iterdir():
        if d.name.startswith(export_id):
            export_dir = d
            break

    if not export_dir or not export_dir.exists():
        raise HTTPException(status_code=404, detail="Export not found")

    zip_path = settings.OUTPUT_DIR / f"{export_id}.zip"
    if not zip_path.exists():
        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            root_dir=export_dir
        )

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{export_id}.zip"
    )


@router.post("/jobs/{job_id}/export", response_model=ExportResult)
async def export_job_dataset(job_id: str, config: ExportConfig):
    """Export labeled dataset to COCO, Pascal VOC, or YOLO format."""
    try:
        job = labeling_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        results = labeling_service.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="No results found for this job")

        config.job_id = job_id

        export_result = export_service.export_dataset(
            job_id,
            results,
            job.class_hierarchy,
            config
        )
        platform.validate_export(job_id, results)
        platform.record_audit_event("export.create", "job", job_id, {"format": config.format.value})
        platform.record_metric("exports.created", 1, {"job_id": job_id, "format": config.format.value})
        return export_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
