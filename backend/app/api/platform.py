# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Platform metadata API: projects, lineage, review, RLHF, gateway, metrics."""

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ..core.config import settings
from ..models.schemas import ActiveLearningConfig
from ..services.labeler import get_labeling_service
from ..services.platform import get_platform_service
from ..services.active_learning import get_active_learning_service, SamplingStrategy
from ..services.intelligence import get_intelligence_service
from ..services.model_manager import get_model_manager

router = APIRouter(tags=["platform"])
platform = get_platform_service()
labeling_service = get_labeling_service()
al_service = get_active_learning_service()
intelligence = get_intelligence_service()
model_manager = get_model_manager()


@router.post("/workspaces")
async def create_workspace(payload: Dict[str, Any]):
    return platform.create_workspace(payload["name"], payload.get("description"))


@router.get("/workspaces")
async def list_workspaces():
    return platform.list_workspaces()


@router.post("/projects")
async def create_project(payload: Dict[str, Any]):
    return platform.create_project(payload["workspace_id"], payload["name"], payload.get("description"))


@router.get("/projects")
async def list_projects(workspace_id: Optional[str] = None):
    return platform.list_projects(workspace_id)


@router.post("/datasets/{dataset_id}/versions")
async def create_dataset_version(dataset_id: str, payload: Dict[str, Any]):
    return platform.create_dataset_version(
        dataset_id=dataset_id,
        project_id=payload.get("project_id"),
        version_name=payload.get("version_name", "v1"),
        source=payload.get("source", "manual"),
        manifest=payload.get("manifest", {}),
    )


@router.get("/datasets/{dataset_id}/versions")
async def list_dataset_versions(dataset_id: str):
    return platform.list_dataset_versions(dataset_id)


@router.get("/datasets/{dataset_id}/lineage")
async def list_lineage(dataset_id: str):
    return platform.list_lineage(dataset_id)


@router.get("/audit/events")
async def list_audit_events(limit: int = 100):
    return platform.list_audit_events(limit)


@router.get("/workers/queue")
async def list_worker_queue(limit: int = 100):
    return platform.list_queue(limit)


@router.post("/workers/queue/claim")
async def claim_worker_job(worker_id: str = "local-worker"):
    item = platform.claim_next_job(worker_id)
    if not item:
        return {"status": "empty"}
    return item


@router.post("/workers/run-next")
async def run_next_worker_job(worker_id: str = "local-worker", execute_labeling: bool = False):
    item = platform.claim_next_job(worker_id)
    if not item:
        return {"status": "empty"}

    task_type = item["task_type"]
    payload = item.get("payload", {})
    try:
        if task_type == "quality_evaluation":
            job_id = payload["job_id"]
            report = await evaluate_job_quality(job_id)
            completed = platform.complete_queue_job(item["id"], {"report_id": report.get("report_id"), "job_id": job_id})
            return {"status": "completed", "task_type": task_type, "queue_item": completed, "output": report}

        if task_type == "export_validation":
            job_id = payload["job_id"]
            results = labeling_service.get_results(job_id)
            if results is None:
                raise HTTPException(status_code=404, detail="Job results not found")
            validation = platform.validate_export(job_id, results)
            completed = platform.complete_queue_job(item["id"], {"validation_id": validation["id"], "job_id": job_id})
            platform.record_audit_event("worker.export_validation.complete", "job", job_id, {"queue_id": item["id"]})
            return {"status": "completed", "task_type": task_type, "queue_item": completed, "output": validation}

        if task_type == "labeling":
            job_id = payload["job_id"]
            if execute_labeling:
                results = await labeling_service.run_job(job_id)
                completed = platform.complete_queue_job(item["id"], {"job_id": job_id, "images": len(results)})
                return {"status": "completed", "task_type": task_type, "queue_item": completed, "images": len(results)}
            completed = platform.complete_queue_job(item["id"], {"job_id": job_id, "status": "manual_start_required"})
            return {
                "status": "completed",
                "task_type": task_type,
                "queue_item": completed,
                "output": {"message": "Labeling jobs are created durably. Use the Start button or pass execute_labeling=true to run this worker endpoint."},
            }

        failed = platform.fail_queue_job(item["id"], f"Unsupported task type: {task_type}")
        return {"status": "failed", "task_type": task_type, "queue_item": failed}
    except HTTPException:
        platform.fail_queue_job(item["id"], "HTTP error during worker execution")
        raise
    except Exception as exc:
        failed = platform.fail_queue_job(item["id"], str(exc))
        return {"status": "failed", "task_type": task_type, "queue_item": failed, "error": str(exc)}


@router.get("/model-gateway/runs")
async def list_model_runs(limit: int = 100):
    return platform.list_model_runs(limit)


@router.get("/model-gateway/catalog")
async def get_model_catalog():
    return {"models": intelligence.get_model_catalog(), "integrations": model_manager.get_advanced_model_status()}


@router.get("/model-gateway/integrations/status")
async def get_model_integration_status():
    return model_manager.get_advanced_model_status()


@router.post("/model-gateway/integrations/{model_key}/prepare")
async def prepare_model_integration(model_key: str, payload: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    result = model_manager.prepare_advanced_model(
        model_key,
        allow_download=bool(payload.get("allow_download", False)),
        token=payload.get("token"),
    )
    platform.record_audit_event("model.prepare", "model", model_key, {"status": result["status"], "message": result["message"]})
    return result


@router.post("/model-gateway/recommend")
async def recommend_models(payload: Dict[str, Any]):
    task_type = payload.get("task_type", "instance_segmentation")
    class_names = payload.get("class_names") or []
    device = payload.get("device", "auto")
    limit = int(payload.get("limit", 5))
    return {
        "request": {"task_type": task_type, "class_names": class_names, "device": device, "limit": limit},
        "recommendations": intelligence.recommend_models(task_type, class_names, device, limit),
    }


@router.get("/observability/metrics")
async def list_metrics(limit: int = 100):
    return platform.list_metrics(limit)


@router.post("/exports/{job_id}/validate")
async def validate_export(job_id: str):
    results = labeling_service.get_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Job results not found")
    return platform.validate_export(job_id, results)


@router.get("/exports/validations")
async def list_export_validations(job_id: Optional[str] = None):
    return platform.list_export_validations(job_id)


@router.post("/jobs/{job_id}/quality/enqueue")
async def enqueue_quality_evaluation(job_id: str):
    if not labeling_service.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    item = platform.enqueue_job("quality_evaluation", {"job_id": job_id})
    platform.record_audit_event("quality.enqueue", "job", job_id, {"queue_id": item["id"]})
    return item


@router.post("/jobs/{job_id}/quality/evaluate")
async def evaluate_job_quality(job_id: str):
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    results = labeling_service.get_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Job results not found")

    class_names = [c.name for c in job.class_hierarchy.classes]
    task_type = job.task_type.value if hasattr(job.task_type, "value") else str(job.task_type)
    report = intelligence.evaluate_job_quality(job_id, task_type, results, class_names)
    saved = platform.save_evaluation_report(job_id, report)
    for image_score in report.get("image_scores", []):
        platform.save_quality_score(
            job_id,
            image_score.get("image_id") or "",
            image_score.get("score", 0),
            image_score.get("issues", []),
        )
    platform.record_metric("quality.avg_score", report["summary"]["avg_quality_score"], {"job_id": job_id, "task_type": task_type})
    platform.record_model_run(
        "annotation-quality-agent",
        "quality_evaluation",
        {"job_id": job_id, "images": report["summary"]["image_count"]},
        {"avg_quality_score": report["summary"]["avg_quality_score"], "high_priority_images": report["summary"]["high_priority_images"]},
    )
    platform.record_audit_event("quality.evaluate", "job", job_id, {"report_id": saved["id"]})
    return {**report, "report_id": saved["id"], "created_at": saved["created_at"]}


@router.get("/jobs/{job_id}/quality/reports")
async def list_quality_reports(job_id: Optional[str] = None):
    return platform.list_evaluation_reports(job_id)


@router.get("/jobs/{job_id}/quality/scores")
async def list_quality_scores(job_id: Optional[str] = None):
    return platform.list_quality_scores(job_id)


@router.get("/datasets/{dataset_id}/health")
async def get_dataset_health(dataset_id: str):
    metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Dataset metadata not found")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    versions = platform.list_dataset_versions(dataset_id)
    lineage = platform.list_lineage(dataset_id)
    health = intelligence.summarize_dataset_health(dataset_id, metadata, versions, lineage)
    platform.record_metric("dataset.health_score", health["health_score"], {"dataset_id": dataset_id})
    return health


@router.get("/review/{job_id}/queue")
async def get_review_queue(job_id: str):
    from .hitl import get_flagged_annotations

    flagged = await get_flagged_annotations(job_id)
    return {"job_id": job_id, "items": flagged, "count": len(flagged)}


@router.post("/review/{job_id}/active-learning/select")
async def active_learning_select(job_id: str, config: ActiveLearningConfig):
    job = labeling_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    results = labeling_service.get_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")

    scores = [
        al_service.score_image(r.image_id, "", r.annotations)
        for r in results
    ]
    filtered = [s for s in scores if s.total_uncertainty >= config.min_uncertainty]
    strategy = SamplingStrategy(config.strategy) if config.strategy in SamplingStrategy._value2member_map_ else SamplingStrategy.UNCERTAINTY
    selected = al_service.select_samples_for_labeling(filtered, strategy, config.n_samples)
    avg = sum(s.total_uncertainty for s in filtered if s.image_id in selected) / max(len(selected), 1)
    return {"job_id": job_id, "selected_image_ids": selected, "selected_count": len(selected), "avg_uncertainty": avg}


@router.post("/preferences/items")
async def create_preference_item(payload: Dict[str, Any]):
    return platform.create_preference_item(
        project_id=payload.get("project_id"),
        image_id=payload["image_id"],
        prompt=payload["prompt"],
        candidates=payload["candidates"],
    )


@router.get("/preferences/items")
async def list_preference_items(project_id: Optional[str] = None):
    return platform.list_preference_items(project_id)


@router.post("/preferences/items/{item_id}/votes")
async def record_preference_vote(item_id: str, payload: Dict[str, Any]):
    return platform.record_preference_vote(
        item_id=item_id,
        selected_candidate_id=payload["selected_candidate_id"],
        rationale=payload.get("rationale"),
    )


@router.get("/preferences/votes")
async def list_preference_votes(item_id: Optional[str] = None):
    return platform.list_preference_votes(item_id)
