# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Research-aware image annotation intelligence services.

This module keeps model recommendation and QA heuristics lightweight. It does
not download or run foundation models. Runtime integration remains owned by the
model manager and labeling service; this service describes what should be used,
why, and how risky each output looks.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from ..core.config import settings


def _task_value(task_type: Any) -> str:
    return getattr(task_type, "value", str(task_type))


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value.__dict__) if hasattr(value, "__dict__") else {}


def _bbox_dict(annotation: Dict[str, Any]) -> Optional[Dict[str, float]]:
    bbox = annotation.get("bbox")
    if not bbox:
        return None
    data = _as_dict(bbox)
    try:
        return {
            "x": float(data.get("x", 0)),
            "y": float(data.get("y", 0)),
            "width": float(data.get("width", 0)),
            "height": float(data.get("height", 0)),
        }
    except (TypeError, ValueError):
        return None


def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _has_segmentation(annotation: Dict[str, Any]) -> bool:
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return False
    data = _as_dict(segmentation)
    polygon = data.get("polygon")
    if polygon:
        return True
    return bool(data.get("rle") or data.get("mask_path"))


class ImageIntelligenceService:
    """Model advice, annotation QA, and dataset-health heuristics."""

    def __init__(self):
        self.catalog = [
            {
                "id": "sam3_concept",
                "name": "SAM 3 Concept Segmentation",
                "provider": "Meta",
                "year": 2025,
                "tasks": ["object_detection", "instance_segmentation", "semantic_segmentation"],
                "runtime_status": "optional_external",
                "availability": self._availability(settings.SAM3_MODEL),
                "recommended_for": [
                    "open-vocabulary instance masks",
                    "text-prompted object concepts",
                    "high-recall pre-labeling before human review",
                ],
                "strengths": [
                    "Text and visual concept prompts",
                    "Finds all matching concept instances",
                    "Best fit for mask-first annotation when available",
                ],
                "constraints": [
                    "Requires SAM 3 runtime and weights",
                    "Falls back to YOLO-World plus SAM2 in this local app unless installed",
                ],
                "research_basis": "Meta introduced SAM 3 in 2025 for promptable concept segmentation with text and visual prompts.",
                "local_config": {
                    "preferred_runtime": "sam3",
                    "sam3_model": settings.SAM3_MODEL,
                    "fallback_profile": "yolo_world_sam2",
                    "use_yolo_world": True,
                },
                "base_score": 96,
            },
            {
                "id": "yolo26_sam2_hybrid",
                "name": "YOLO plus SAM2 Hybrid",
                "provider": "Local Ultralytics pipeline",
                "year": 2026,
                "tasks": ["object_detection", "instance_segmentation", "semantic_segmentation"],
                "runtime_status": "local",
                "availability": self._availability(settings.YOLOV26_MODEL, settings.SAM2_MODEL),
                "recommended_for": [
                    "production local image annotation",
                    "fast bounding boxes with high-quality masks",
                    "offline batch labeling",
                ],
                "strengths": [
                    "Already integrated in this app",
                    "Good speed/quality balance",
                    "Works without external model gateway",
                ],
                "constraints": ["Closed-class performance depends on user class aliases"],
                "research_basis": "Combines detector proposals with SAM2-style promptable mask refinement.",
                "local_config": {"use_yolo_world": False},
                "base_score": 92,
            },
            {
                "id": "yolo_world_sam2",
                "name": "YOLO-World plus SAM2",
                "provider": "Local Ultralytics pipeline",
                "year": 2024,
                "tasks": ["object_detection", "instance_segmentation", "semantic_segmentation"],
                "runtime_status": "local",
                "availability": self._availability("yolov8s-world.pt", settings.SAM2_MODEL),
                "recommended_for": [
                    "custom classes with descriptive names",
                    "prompt-based detection",
                    "domain classes not covered by standard YOLO labels",
                ],
                "strengths": [
                    "Open-vocabulary prompt path",
                    "Pairs naturally with SAM2 for masks",
                    "Useful fallback for SAM 3-like concept prompts",
                ],
                "constraints": ["Prompt wording can affect recall"],
                "research_basis": "Open-vocabulary detectors improve custom-label bootstrapping before mask refinement.",
                "local_config": {"preferred_runtime": "yolo_world", "use_yolo_world": True},
                "base_score": 88,
            },
            {
                "id": "grounding_dino_15",
                "name": "Grounding DINO 1.5",
                "provider": "IDEA Research",
                "year": 2024,
                "tasks": ["object_detection", "instance_segmentation"],
                "runtime_status": "optional_external",
                "availability": "not_installed",
                "recommended_for": [
                    "open-set detection",
                    "large custom vocabularies",
                    "external model-gateway detection before SAM masks",
                ],
                "strengths": [
                    "Strong open-set detection",
                    "Pro and Edge profiles cover quality/speed tradeoffs",
                ],
                "constraints": ["Not wired into the local runtime yet"],
                "research_basis": "Grounding DINO 1.5 reports strong COCO and LVIS zero-shot detection results.",
                "local_config": {
                    "preferred_runtime": "grounding_dino",
                    "grounding_dino_model_id": settings.GROUNDING_DINO_MODEL_ID,
                    "fallback_profile": "yolo_world_sam2",
                    "use_yolo_world": True,
                },
                "base_score": 86,
            },
            {
                "id": "dinov3_quality",
                "name": "DINOv3 Dataset Intelligence",
                "provider": "Meta",
                "year": 2025,
                "tasks": ["dataset_quality", "active_learning", "semantic_segmentation"],
                "runtime_status": "optional_external",
                "availability": "not_installed",
                "recommended_for": [
                    "dataset curation",
                    "near-duplicate detection",
                    "diversity sampling",
                    "visual feature quality checks",
                ],
                "strengths": [
                    "Dense image features",
                    "Strong backbone for quality and diversity analytics",
                ],
                "constraints": ["Not a direct auto-labeling model"],
                "research_basis": "Meta describes DINOv3 as a versatile vision foundation model with strong dense features.",
                "local_config": {
                    "preferred_runtime": "dinov3",
                    "dinov3_model_id": settings.DINOV3_MODEL_ID,
                    "quality_only": True,
                },
                "base_score": 82,
            },
            {
                "id": "sam2_refiner",
                "name": "SAM2 Interactive Refiner",
                "provider": "Meta / Ultralytics runtime",
                "year": 2024,
                "tasks": ["instance_segmentation", "semantic_segmentation"],
                "runtime_status": "local",
                "availability": self._availability(settings.SAM2_MODEL),
                "recommended_for": ["mask refinement", "review corrections", "box-to-mask generation"],
                "strengths": ["Local mask refinement", "Strong promptable segmentation baseline"],
                "constraints": ["Needs detector prompts for full automation"],
                "research_basis": "SAM2 improves image segmentation speed and accuracy versus SAM while supporting promptable segmentation.",
                "local_config": {"preferred_runtime": "sam2", "sam_model": settings.SAM2_MODEL},
                "base_score": 78,
            },
        ]

    def _availability(self, *model_files: str) -> str:
        missing = [name for name in model_files if not (settings.MODELS_DIR / name).exists()]
        return "ready" if not missing else "missing_files"

    def get_model_catalog(self) -> List[Dict[str, Any]]:
        return deepcopy(self.catalog)

    def get_model_profile(self, model_id: str) -> Optional[Dict[str, Any]]:
        for item in self.catalog:
            if item["id"] == model_id:
                return deepcopy(item)
        return None

    def recommend_models(
        self,
        task_type: str,
        class_names: Optional[List[str]] = None,
        device: str = "auto",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        task = _task_value(task_type)
        classes = [c.strip() for c in (class_names or []) if c and c.strip()]
        lower_classes = [c.lower() for c in classes]
        descriptive_prompting = any(len(c.split()) > 1 for c in lower_classes)
        custom_domain = bool(classes) and not set(lower_classes).issubset(
            {"person", "car", "truck", "bus", "bicycle", "motorcycle", "chair", "table", "dog", "cat"}
        )
        scored: List[Dict[str, Any]] = []

        for profile in self.catalog:
            if task not in profile["tasks"]:
                continue
            score = float(profile["base_score"])
            reasons = []

            if profile["availability"] == "ready":
                score += 6
                reasons.append("runtime artifacts are available locally")
            elif profile["runtime_status"] == "optional_external":
                score -= 4
                reasons.append("requires an external or newly installed runtime")

            if task == "instance_segmentation":
                if "SAM" in profile["name"] or "sam" in profile["id"]:
                    score += 8
                    reasons.append("mask-first task benefits from SAM-family segmentation")
                if "yolo" in profile["id"]:
                    score += 4
                    reasons.append("detector proposals reduce manual mask prompting")

            if task == "object_detection":
                if "grounding" in profile["id"] or "yolo" in profile["id"]:
                    score += 6
                    reasons.append("detection task benefits from detector-first models")

            if task == "semantic_segmentation" and profile["id"] in {"sam3_concept", "yolo_world_sam2", "sam2_refiner"}:
                score += 5
                reasons.append("segmentation profile can produce or refine masks")

            if custom_domain or descriptive_prompting:
                if profile["id"] in {"sam3_concept", "grounding_dino_15", "yolo_world_sam2"}:
                    score += 10
                    reasons.append("custom or descriptive classes benefit from open-vocabulary prompting")
                elif profile["id"] == "yolo26_sam2_hybrid":
                    score -= 2
                    reasons.append("closed-vocabulary detector may need aliases for custom classes")

            if device == "cpu" and profile["id"] in {"sam3_concept", "grounding_dino_15", "dinov3_quality"}:
                score -= 8
                reasons.append("large foundation model profile is better on GPU or external gateway")
            elif device in {"gpu", "cuda"} and profile["id"] in {"sam3_concept", "yolo26_sam2_hybrid", "yolo_world_sam2"}:
                score += 4
                reasons.append("GPU improves throughput for this profile")

            item = deepcopy(profile)
            item["score"] = round(max(0.0, min(100.0, score)), 1)
            item["why"] = reasons or ["general fit for this task"]
            scored.append(item)

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, min(limit, 20))]

    def runtime_config_for_profile(self, model_id: Optional[str]) -> Dict[str, Any]:
        if not model_id:
            return {}
        profile = self.get_model_profile(model_id)
        if not profile:
            return {"selected_model_id": model_id, "model_profile_status": "unknown"}
        config = deepcopy(profile.get("local_config", {}))
        config.update(
            {
                "selected_model_id": model_id,
                "selected_model_name": profile["name"],
                "model_profile_status": profile["runtime_status"],
                "model_profile_availability": profile["availability"],
            }
        )
        return config

    def score_image_quality(self, image_labels: Any, task_type: str) -> Dict[str, Any]:
        image = _as_dict(image_labels)
        annotations = [_as_dict(item) for item in image.get("annotations", [])]
        task = _task_value(task_type)
        score = 100.0
        issues: List[Dict[str, Any]] = []
        boxes: List[Dict[str, Any]] = []
        low_confidence_count = 0
        invalid_geometry_count = 0
        missing_segmentation_count = 0

        if not annotations:
            score -= 28
            issues.append(
                {
                    "severity": "warning",
                    "code": "empty_image",
                    "message": "No annotations were produced for this image.",
                }
            )

        for index, annotation in enumerate(annotations):
            confidence = float(annotation.get("confidence", 0.0))
            bbox = _bbox_dict(annotation)
            if confidence < 0.35:
                low_confidence_count += 1
                penalty = 14 if confidence >= 0.2 else 22
                score -= penalty
                issues.append(
                    {
                        "severity": "warning",
                        "code": "low_confidence",
                        "annotation_id": annotation.get("id", index),
                        "message": f"Annotation confidence {confidence:.2f} is below review threshold.",
                    }
                )

            if bbox is None or bbox["width"] <= 0 or bbox["height"] <= 0:
                invalid_geometry_count += 1
                score -= 25
                issues.append(
                    {
                        "severity": "error",
                        "code": "invalid_bbox",
                        "annotation_id": annotation.get("id", index),
                        "message": "Annotation has missing or invalid bounding-box geometry.",
                    }
                )
            else:
                boxes.append({"annotation": annotation, "bbox": bbox})

            if task in {"instance_segmentation", "semantic_segmentation"} and not _has_segmentation(annotation):
                missing_segmentation_count += 1
                score -= 10
                issues.append(
                    {
                        "severity": "warning",
                        "code": "missing_segmentation",
                        "annotation_id": annotation.get("id", index),
                        "message": "Segmentation task annotation has no polygon, RLE, or mask path.",
                    }
                )

            uncertainty = _as_dict(annotation.get("attributes")).get("uncertainty", {})
            uncertainty_total = _as_dict(uncertainty).get("total_uncertainty")
            if uncertainty_total is not None and float(uncertainty_total) > 0.5:
                score -= 8
                issues.append(
                    {
                        "severity": "warning",
                        "code": "high_uncertainty",
                        "annotation_id": annotation.get("id", index),
                        "message": "Model uncertainty is high for this annotation.",
                    }
                )

        duplicate_pairs = 0
        for left_index in range(len(boxes)):
            for right_index in range(left_index + 1, len(boxes)):
                left = boxes[left_index]
                right = boxes[right_index]
                same_class = left["annotation"].get("class_name") == right["annotation"].get("class_name")
                iou = _bbox_iou(left["bbox"], right["bbox"])
                if same_class and iou >= 0.75:
                    duplicate_pairs += 1
                    score -= 12
                    issues.append(
                        {
                            "severity": "warning",
                            "code": "duplicate_overlap",
                            "message": f"Two same-class boxes overlap heavily with IoU {iou:.2f}.",
                        }
                    )

        score = round(max(0.0, min(100.0, score)), 1)
        if score < 75 or any(issue["severity"] == "error" for issue in issues):
            priority = "high"
        elif score < 90 or issues:
            priority = "medium"
        else:
            priority = "low"

        return {
            "image_id": image.get("image_id"),
            "score": score,
            "grade": self._grade(score),
            "review_priority": priority,
            "issues": issues,
            "statistics": {
                "annotations": len(annotations),
                "low_confidence": low_confidence_count,
                "invalid_geometry": invalid_geometry_count,
                "missing_segmentation": missing_segmentation_count,
                "duplicate_pairs": duplicate_pairs,
            },
        }

    def evaluate_job_quality(
        self,
        job_id: str,
        task_type: str,
        results: Iterable[Any],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        image_scores = [self.score_image_quality(item, task_type) for item in results]
        image_count = len(image_scores)
        annotation_count = 0
        empty_images = 0
        class_distribution: Dict[str, int] = {}
        low_confidence_annotations = 0

        for image_score, image in zip(image_scores, results):
            image_dict = _as_dict(image)
            annotations = [_as_dict(item) for item in image_dict.get("annotations", [])]
            annotation_count += len(annotations)
            if not annotations:
                empty_images += 1
            low_confidence_annotations += image_score["statistics"]["low_confidence"]
            for annotation in annotations:
                class_name = annotation.get("class_name") or str(annotation.get("class_id", "unknown"))
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

        avg_quality = round(
            sum(item["score"] for item in image_scores) / image_count,
            1,
        ) if image_count else 0.0
        high_priority = [item["image_id"] for item in image_scores if item["review_priority"] == "high"]
        missing_classes = [
            name for name in (class_names or [])
            if name and class_distribution.get(name, 0) == 0
        ]

        actions: List[str] = []
        if high_priority:
            actions.append("Review high-priority images before export.")
        if low_confidence_annotations:
            actions.append("Raise human review coverage for low-confidence annotations.")
        if empty_images:
            actions.append("Inspect empty images to distinguish true negatives from model misses.")
        if missing_classes:
            actions.append("Check aliases or prompt wording for classes with zero detections.")
        if not actions:
            actions.append("Run a spot-check review sample before export.")

        return {
            "job_id": job_id,
            "task_type": _task_value(task_type),
            "summary": {
                "image_count": image_count,
                "annotation_count": annotation_count,
                "avg_quality_score": avg_quality,
                "empty_images": empty_images,
                "high_priority_images": len(high_priority),
                "low_confidence_annotations": low_confidence_annotations,
                "class_distribution": class_distribution,
                "missing_classes": missing_classes,
            },
            "image_scores": image_scores,
            "recommended_actions": actions,
        }

    def summarize_dataset_health(
        self,
        dataset_id: str,
        metadata: Dict[str, Any],
        versions: Optional[List[Dict[str, Any]]] = None,
        lineage: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        total = int(metadata.get("total_images") or 0)
        valid = int(metadata.get("valid_images") or 0)
        corrupted = int(metadata.get("corrupted_images") or 0)
        formats = metadata.get("formats") or {}
        versions = versions or []
        lineage = lineage or []
        score = 100.0
        warnings: List[str] = []
        recommendations: List[str] = []

        if total == 0:
            score = 0.0
            warnings.append("Dataset has no images.")
            recommendations.append("Upload a ZIP or folder containing valid image files.")
        else:
            valid_ratio = valid / total
            corrupted_ratio = corrupted / total
            score -= (1.0 - valid_ratio) * 45.0
            score -= corrupted_ratio * 20.0
            if corrupted:
                warnings.append(f"{corrupted} corrupted or unreadable images were detected.")
                recommendations.append("Remove, repair, or replace corrupted images before production labeling.")

        if len(formats) > 3:
            score -= 4
            recommendations.append("Normalize image formats if downstream training expects a narrow format set.")
        if not versions:
            score -= 8
            warnings.append("No dataset version is recorded.")
            recommendations.append("Create or confirm an immutable dataset version before labeling.")
        if not lineage:
            score -= 6
            warnings.append("No lineage events are recorded.")
            recommendations.append("Record upload and preprocessing lineage for reproducibility.")

        if total and valid < max(5, math.ceil(total * 0.2)):
            recommendations.append("Dataset is small or sparse; use active learning and manual review aggressively.")

        return {
            "dataset_id": dataset_id,
            "health_score": round(max(0.0, min(100.0, score)), 1),
            "status": "healthy" if score >= 90 else "needs_review" if score >= 70 else "at_risk",
            "summary": {
                "total_images": total,
                "valid_images": valid,
                "corrupted_images": corrupted,
                "formats": formats,
                "versions": len(versions),
                "lineage_events": len(lineage),
            },
            "warnings": warnings,
            "recommendations": recommendations or ["Dataset metadata looks ready for labeling."],
        }

    def _grade(self, score: float) -> str:
        if score >= 95:
            return "A"
        if score >= 85:
            return "B"
        if score >= 75:
            return "C"
        if score >= 60:
            return "D"
        return "F"


intelligence_service = ImageIntelligenceService()


def get_intelligence_service() -> ImageIntelligenceService:
    return intelligence_service
