# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Human-in-the-Loop and Active Learning API routes."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import (
    RefinementRequest, AnnotationUpdate, AnnotationStatus,
    FlaggedAnnotation, UncertaintyDetails, ActiveLearningConfig,
    ActiveLearningResult, ImageLabels, ActiveLearningRequest,
    FlaggedSample, ExportConfig, ExportResult
)
from ..services.labeler import get_labeling_service
from ..services.model_manager import get_model_manager
from ..services.active_learning import get_active_learning_service, SamplingStrategy

logger = get_logger("api.hitl")
router = APIRouter(tags=["hitl"])

labeling_service = get_labeling_service()
model_manager = get_model_manager()
al_service = get_active_learning_service()


def resolve_image_url(path_str: str, dataset_id: str = None) -> Optional[str]:
    """Convert an absolute filesystem path to a /uploads/ URL.

    Strategy:
    1. If the path contains 'uploads/', split on it and use the remainder.
    2. If the dataset_id appears in the path, use everything after it.
    3. Fallback: use the filename and search for it under the dataset dir.
    4. Final fallback: construct URL from dataset_id + filename.

    After constructing the candidate URL, verify the file exists on disk under
    UPLOAD_DIR. If not, walk the dataset directory to find the file by name.
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


@router.post("/jobs/{job_id}/annotations/{annotation_id}/refine")
async def refine_annotation(
    job_id: str,
    annotation_id: int,
    refinement: RefinementRequest
):
    """Refine an annotation mask using SAM2's interactive prompt interface."""
    try:
        job = labeling_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        results = labeling_service.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")

        annotation = None
        image_labels = None
        for img_labels in results:
            for ann in img_labels.annotations:
                if ann.id == annotation_id:
                    annotation = ann
                    image_labels = img_labels
                    break
            if annotation:
                break

        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")

        from PIL import Image
        import numpy as np

        metadata_path = settings.UPLOAD_DIR / f"{job.dataset_id}_metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)

        img_info = next((img for img in meta["images"] if img["id"] == image_labels.image_id), None)
        if not img_info:
            raise HTTPException(status_code=404, detail="Image not found in dataset")

        image = np.array(Image.open(img_info["path"]).convert("RGB"))

        if refinement.points:
            points = [(p.x, p.y) for p in refinement.points]
            labels = [p.label for p in refinement.points]

            sam_model = model_manager.models.get("sam2")
            if sam_model is None:
                model_manager.load_sam2()
                sam_model = model_manager.models["sam2"]

            results = sam_model(
                image,
                points=[points],
                labels=[labels],
                verbose=False
            )

            if results and results[0].masks is not None:
                mask_data = results[0].masks.data[0].cpu().numpy()
                new_polygon = model_manager._mask_to_polygon(mask_data)

                annotation.segmentation.polygon = new_polygon

                if annotation.attributes is None:
                    annotation.attributes = {}
                annotation.attributes["status"] = AnnotationStatus.CORRECTED.value

                return {
                    "message": "Annotation refined successfully",
                    "annotation_id": annotation_id,
                    "new_polygon": new_polygon
                }

        elif refinement.bbox_adjustment:
            bbox = refinement.bbox_adjustment
            input_box = [[bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height]]

            sam_model = model_manager.models.get("sam2")
            if sam_model is None:
                model_manager.load_sam2()
                sam_model = model_manager.models["sam2"]

            results = sam_model(
                image,
                bboxes=input_box,
                verbose=False
            )

            if results and results[0].masks is not None:
                mask_data = results[0].masks.data[0].cpu().numpy()
                new_polygon = model_manager._mask_to_polygon(mask_data)

                annotation.bbox = bbox
                annotation.segmentation.polygon = new_polygon

                if annotation.attributes is None:
                    annotation.attributes = {}
                annotation.attributes["status"] = AnnotationStatus.CORRECTED.value

                return {
                    "message": "Annotation refined with new bbox",
                    "annotation_id": annotation_id,
                    "new_polygon": new_polygon
                }

        return {"message": "No refinement applied", "annotation_id": annotation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/annotations/{annotation_id}/approve")
async def approve_annotation(job_id: str, annotation_id: int):
    """Mark an annotation as human-verified."""
    try:
        results = labeling_service.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")

        for img_labels in results:
            for ann in img_labels.annotations:
                if ann.id == annotation_id:
                    if ann.attributes is None:
                        ann.attributes = {}
                    ann.attributes["status"] = AnnotationStatus.APPROVED.value
                    ann.attributes["approved_at"] = str(datetime.now())

                    al_service.record_correction(
                        img_labels.image_id,
                        str(annotation_id),
                        "approve"
                    )

                    return {
                        "message": "Annotation approved",
                        "annotation_id": annotation_id,
                        "status": AnnotationStatus.APPROVED.value
                    }

        raise HTTPException(status_code=404, detail="Annotation not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/annotations/{annotation_id}/update")
async def update_annotation(job_id: str, annotation_id: int, update: AnnotationUpdate):
    """Update an annotation (class, bbox, polygon, or status)."""
    try:
        results = labeling_service.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")

        for img_labels in results:
            for ann in img_labels.annotations:
                if ann.id == annotation_id:
                    if update.status:
                        if ann.attributes is None:
                            ann.attributes = {}
                        ann.attributes["status"] = update.status.value

                    if update.class_id is not None:
                        ann.class_id = update.class_id
                    if update.class_name is not None:
                        ann.class_name = update.class_name
                    if update.bbox is not None:
                        ann.bbox = update.bbox
                    if update.polygon is not None:
                        ann.segmentation.polygon = update.polygon

                    al_service.record_correction(
                        img_labels.image_id,
                        str(annotation_id),
                        "modify"
                    )

                    return {
                        "message": "Annotation updated",
                        "annotation_id": annotation_id
                    }

        raise HTTPException(status_code=404, detail="Annotation not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}/annotations/{annotation_id}")
async def delete_annotation(job_id: str, annotation_id: int):
    """Delete an annotation."""
    try:
        results = labeling_service.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")

        for img_labels in results:
            for i, ann in enumerate(img_labels.annotations):
                if ann.id == annotation_id:
                    img_labels.annotations.pop(i)
                    al_service.record_correction(
                        img_labels.image_id,
                        str(annotation_id),
                        "delete"
                    )
                    return {"message": "Annotation deleted", "annotation_id": annotation_id}

        raise HTTPException(status_code=404, detail="Annotation not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/flagged", response_model=List[FlaggedAnnotation])
async def get_flagged_annotations(job_id: str):
    """Get all annotations flagged for human review."""
    try:
        job = labeling_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        results = labeling_service.get_results(job_id)
        if not results:
            return []

        metadata_path = settings.UPLOAD_DIR / f"{job.dataset_id}_metadata.json"
        img_map = {}
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
                img_map = {img["id"]: img for img in meta.get("images", [])}
        except Exception:
            pass

        flagged = []
        for img_labels in results:
            all_anns = img_labels.annotations
            for ann in all_anns:
                score = al_service.score_uncertainty(
                    ann, all_anns, image_size=(1920, 1080)
                )

                if score.total_score > 0.5 or ann.confidence < 0.3:
                    img_data = img_map.get(img_labels.image_id)
                    image_url = None
                    if img_data:
                        image_url = resolve_image_url(img_data["path"], job.dataset_id)

                    reason = "Low confidence" if ann.confidence < 0.3 else "High uncertainty"
                    if score.detection_mask_disagreement > 0.4:
                        reason = "Detection-mask disagreement"

                    flagged.append(FlaggedAnnotation(
                        image_id=img_labels.image_id,
                        annotation_id=ann.id,
                        image_url=image_url,
                        class_name=ann.class_name or "unknown",
                        confidence=ann.confidence,
                        uncertainty=UncertaintyDetails(
                            total=score.total_score,
                            confidence=score.confidence_uncertainty,
                            detection_mask_disagreement=score.detection_mask_disagreement,
                            semantic_instance_disagreement=score.semantic_instance_disagreement,
                            size=score.size_uncertainty,
                            overlap=score.overlap_uncertainty
                        ),
                        reason=reason
                    ))

        flagged.sort(key=lambda x: -x.uncertainty.total)
        return flagged

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get flagged annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Active Learning ──────────────────────────────────────────────

@router.post("/jobs/{job_id}/active-learning/select", response_model=ActiveLearningResult)
async def select_samples_for_labeling(job_id: str, config: ActiveLearningConfig):
    """Select most informative samples for human labeling using active learning."""
    try:
        job = labeling_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        results = labeling_service.get_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")

        from ..services.active_learning import SampleScore

        scored_samples = []
        for img_labels in results:
            sample_score = al_service.score_image(
                img_labels.image_id, "",
                img_labels.annotations,
                image_size=(1920, 1080)
            )
            scored_samples.append(sample_score)

        filtered = [s for s in scored_samples if s.total_uncertainty >= config.min_uncertainty]

        strategy = SamplingStrategy(config.strategy) if config.strategy in ["uncertainty", "diversity", "hybrid", "random"] else SamplingStrategy.UNCERTAINTY

        selected_ids = al_service.select_samples_for_labeling(
            filtered, strategy=strategy, n_samples=config.n_samples
        )

        avg_unc = sum(s.total_uncertainty for s in filtered if s.image_id in selected_ids) / max(len(selected_ids), 1)

        return ActiveLearningResult(
            selected_count=len(selected_ids),
            selected_image_ids=selected_ids,
            strategy_used=strategy.value,
            avg_uncertainty=avg_unc
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Active learning selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/active-learning/summary")
async def get_active_learning_summary(job_id: str):
    """Get summary of active learning metrics for a job."""
    try:
        job = labeling_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        results = labeling_service.get_results(job_id)
        if not results:
            return {"total_images": 0, "avg_uncertainty": 0}

        uncertainties = []
        flagged_count = 0

        for img_labels in results:
            sample_score = al_service.score_image(
                img_labels.image_id, "",
                img_labels.annotations,
                image_size=(1920, 1080)
            )
            uncertainties.append(sample_score.total_uncertainty)
            if sample_score.should_flag:
                flagged_count += 1

        return {
            "job_id": job_id,
            "total_images": len(results),
            "avg_uncertainty": sum(uncertainties) / max(len(uncertainties), 1),
            "max_uncertainty": max(uncertainties) if uncertainties else 0,
            "min_uncertainty": min(uncertainties) if uncertainties else 0,
            "flagged_images": flagged_count,
            "corrections_recorded": len(al_service.label_history)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/active-learning/export-retraining")
async def export_for_retraining(job_id: str):
    """Export corrected annotations for model retraining."""
    try:
        job = labeling_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        output_path = settings.OUTPUT_DIR / f"{job_id}_retraining"
        export_data = al_service.export_for_retraining(output_path)

        return {
            "message": "Export complete",
            "output_path": str(output_path),
            **export_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/active-learning/score/{job_id}")
async def score_job_uncertainty(job_id: str):
    """Calculate uncertainty scores for a job's results."""
    results = labeling_service.get_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Job results not found or job failed")

    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    metadata_path = settings.UPLOAD_DIR / f"{job.dataset_id}_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Dataset metadata not found")

    with open(metadata_path) as f:
        data = json.load(f)

    scored_count = 0
    for res in results:
        img_meta = next((img for img in data.get("images", []) if img["id"] == res.image_id), None)
        path = img_meta["path"] if img_meta else ""

        al_service.score_image(
            image_id=res.image_id,
            image_path=path,
            annotations=res.annotations,
            image_size=(img_meta["width"], img_meta["height"]) if img_meta else (1920, 1080)
        )
        scored_count += 1

    return {"message": f"Scored {scored_count} images", "job_id": job_id}


@router.post("/active-learning/select", response_model=List[str])
async def select_active_learning_samples(request: ActiveLearningRequest):
    """Select the most informative samples for human labeling."""
    try:
        strategy_enum = SamplingStrategy(request.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    selected_ids = al_service.select_samples_for_labeling(
        strategy=strategy_enum,
        n_samples=request.n_samples
    )
    return selected_ids


@router.get("/active-learning/flagged", response_model=List[FlaggedSample])
async def get_flagged_samples():
    """Get samples flagged for human review due to high uncertainty."""
    flagged = al_service.get_flagged_annotations()
    return [FlaggedSample(**f) for f in flagged]
