# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Active Learning Service.
Implements uncertainty scoring and sample selection for efficient annotation.
Uses detection-mask disagreement and model confidence for sample prioritization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
import cv2

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import LabelAnnotation, SegmentationMask, BoundingBox

logger = get_logger("active_learning")


class SamplingStrategy(str, Enum):
    """Active learning sampling strategies."""
    UNCERTAINTY = "uncertainty"      # Highest uncertainty scores
    DIVERSITY = "diversity"          # Most different from labeled set
    HYBRID = "hybrid"                # Combination of both
    RANDOM = "random"                # Random baseline


@dataclass
class UncertaintyScore:
    """Uncertainty metrics for an annotation or image."""
    total_score: float                    # Overall uncertainty (0-1, higher = more uncertain)
    confidence_uncertainty: float         # 1 - model confidence
    detection_mask_disagreement: float    # Mismatch between bbox and polygon
    semantic_instance_disagreement: float # Mismatch between semantic and instance class
    size_uncertainty: float               # Very small or very large objects
    overlap_uncertainty: float            # Overlapping detections
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total_score,
            "confidence": self.confidence_uncertainty,
            "detection_mask_disagreement": self.detection_mask_disagreement,
            "semantic_instance_disagreement": self.semantic_instance_disagreement,
            "size": self.size_uncertainty,
            "overlap": self.overlap_uncertainty
        }


@dataclass
class SampleScore:
    """Active learning score for an image."""
    image_id: str
    image_path: str
    total_uncertainty: float
    annotation_count: int
    flagged_count: int
    annotation_uncertainties: List[UncertaintyScore]
    
    @property
    def should_flag(self) -> bool:
        """Whether this sample should be flagged for human review."""
        return self.total_uncertainty > 0.5 or self.flagged_count > 0


class ActiveLearningService:
    """
    Service for active learning sample selection and uncertainty estimation.
    
    Key Features:
    1. Uncertainty scoring based on multiple signals
    2. Detection-mask disagreement detection
    3. Sample prioritization for efficient labeling
    4. Flagging system for human review
    """
    
    # Weights for combining uncertainty components
    UNCERTAINTY_WEIGHTS = {
        "confidence": 0.3,
        "detection_mask_disagreement": 0.25,
        "semantic_instance_disagreement": 0.2,
        "size": 0.15,
        "overlap": 0.1
    }
    
    # Thresholds for flagging
    FLAG_THRESHOLDS = {
        "min_confidence": 0.3,          # Flag if below this
        "max_disagreement": 0.4,        # Flag if above this
        "min_area_ratio": 0.001,        # Flag if object is tiny
        "max_overlap_iou": 0.7          # Flag if highly overlapping
    }
    
    def __init__(self):
        self.scored_samples: Dict[str, SampleScore] = {}
        self.label_history: Dict[str, List[str]] = {}  # image_id -> correction history
    
    def score_uncertainty(
        self,
        annotation: LabelAnnotation,
        all_annotations: List[LabelAnnotation] = None,
        semantic_class: Optional[str] = None,
        image_size: Tuple[int, int] = (1920, 1080)
    ) -> UncertaintyScore:
        """
        Compute uncertainty score for a single annotation.
        
        Args:
            annotation: The annotation to score
            all_annotations: All annotations in the image (for overlap calculation)
            semantic_class: Class from semantic segmentation (for disagreement)
            image_size: (width, height) of the image
            
        Returns:
            UncertaintyScore with detailed metrics
        """
        all_annotations = all_annotations or []
        
        # 1. Confidence uncertainty (simple)
        confidence_unc = 1.0 - annotation.confidence
        
        # 2. Detection-mask disagreement
        det_mask_disagree = self._compute_detection_mask_disagreement(annotation)
        
        # 3. Semantic-instance disagreement
        sem_inst_disagree = self._compute_semantic_instance_disagreement(
            annotation.class_name, semantic_class
        )
        
        # 4. Size uncertainty
        size_unc = self._compute_size_uncertainty(annotation, image_size)
        
        # 5. Overlap uncertainty
        overlap_unc = self._compute_overlap_uncertainty(annotation, all_annotations)
        
        # Weighted combination
        total = (
            self.UNCERTAINTY_WEIGHTS["confidence"] * confidence_unc +
            self.UNCERTAINTY_WEIGHTS["detection_mask_disagreement"] * det_mask_disagree +
            self.UNCERTAINTY_WEIGHTS["semantic_instance_disagreement"] * sem_inst_disagree +
            self.UNCERTAINTY_WEIGHTS["size"] * size_unc +
            self.UNCERTAINTY_WEIGHTS["overlap"] * overlap_unc
        )
        
        return UncertaintyScore(
            total_score=min(1.0, total),
            confidence_uncertainty=confidence_unc,
            detection_mask_disagreement=det_mask_disagree,
            semantic_instance_disagreement=sem_inst_disagree,
            size_uncertainty=size_unc,
            overlap_uncertainty=overlap_unc
        )
    
    def _compute_detection_mask_disagreement(
        self,
        annotation: LabelAnnotation
    ) -> float:
        """
        Compute disagreement between bounding box and segmentation mask.
        High disagreement indicates poor mask quality or detection error.
        """
        if annotation.bbox is None or annotation.segmentation is None:
            return 0.0
        
        if annotation.segmentation.polygon is None:
            return 0.5  # No polygon = moderate uncertainty
        
        # Get bbox area
        bbox_area = annotation.bbox.width * annotation.bbox.height
        
        if bbox_area < 1:
            return 1.0  # Invalid bbox
        
        # Estimate polygon area (sum of contour areas)
        polygon_area = 0.0
        for poly in annotation.segmentation.polygon:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                polygon_area += cv2.contourArea(pts)
        
        if polygon_area < 1:
            return 0.8  # Empty polygon = high uncertainty
        
        # Compute disagreement as difference from expected ratio
        # Ideal: polygon covers 50-90% of bbox (objects rarely fill bbox completely)
        fill_ratio = polygon_area / bbox_area
        
        if fill_ratio < 0.2:
            # Very sparse - likely wrong detection
            return 0.7
        elif fill_ratio > 1.2:
            # Polygon larger than bbox - likely wrong
            return 0.6
        elif fill_ratio < 0.4 or fill_ratio > 1.0:
            # Moderate disagreement
            return 0.3
        else:
            # Good agreement
            return 0.0
    
    def _compute_semantic_instance_disagreement(
        self,
        instance_class: str,
        semantic_class: Optional[str]
    ) -> float:
        """
        Compute disagreement between instance and semantic class predictions.
        """
        if semantic_class is None:
            return 0.0  # No semantic prediction
        
        instance_lower = instance_class.lower()
        semantic_lower = semantic_class.lower()
        
        if instance_lower == semantic_lower:
            return 0.0  # Perfect agreement
        
        # Check for related classes (e.g., "car" and "vehicle")
        related_mappings = {
            "car": ["vehicle", "automobile", "sedan"],
            "truck": ["vehicle", "lorry"],
            "person": ["pedestrian", "human", "rider"],
            "bicycle": ["bike", "cycle"],
            "motorcycle": ["motorbike", "bike"],
            "bus": ["vehicle"],
            "train": ["vehicle", "rail"],
        }
        
        for key, related in related_mappings.items():
            if instance_lower == key and semantic_lower in related:
                return 0.1  # Minor disagreement
            if semantic_lower == key and instance_lower in related:
                return 0.1
        
        # Complete disagreement
        return 0.8
    
    def _compute_size_uncertainty(
        self,
        annotation: LabelAnnotation,
        image_size: Tuple[int, int]
    ) -> float:
        """
        Compute uncertainty based on object size.
        Very small or very large objects are harder to annotate correctly.
        """
        if annotation.bbox is None:
            return 0.5
        
        img_w, img_h = image_size
        img_area = img_w * img_h
        
        if img_area < 1:
            return 0.5
        
        obj_area = annotation.bbox.width * annotation.bbox.height
        area_ratio = obj_area / img_area
        
        if area_ratio < 0.001:
            # Very tiny object
            return 0.8
        elif area_ratio < 0.005:
            # Small object
            return 0.4
        elif area_ratio > 0.5:
            # Very large (might be background)
            return 0.6
        elif area_ratio > 0.3:
            # Large
            return 0.2
        else:
            # Normal size
            return 0.0
    
    def _compute_overlap_uncertainty(
        self,
        annotation: LabelAnnotation,
        all_annotations: List[LabelAnnotation]
    ) -> float:
        """
        Compute uncertainty based on overlap with other annotations.
        High overlap suggests potential duplicate or occlusion issues.
        """
        if annotation.bbox is None or len(all_annotations) < 2:
            return 0.0
        
        max_iou = 0.0
        
        for other in all_annotations:
            if other.id == annotation.id or other.bbox is None:
                continue
            
            iou = self._calculate_iou(annotation.bbox, other.bbox)
            max_iou = max(max_iou, iou)
        
        if max_iou > 0.7:
            return 0.8  # High overlap
        elif max_iou > 0.5:
            return 0.4  # Moderate overlap
        elif max_iou > 0.3:
            return 0.2  # Some overlap
        else:
            return 0.0
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate IoU of two bounding boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def score_image(
        self,
        image_id: str,
        image_path: str,
        annotations: List[LabelAnnotation],
        semantic_predictions: Optional[Dict[str, str]] = None,
        image_size: Tuple[int, int] = (1920, 1080)
    ) -> SampleScore:
        """
        Score an entire image for active learning.
        
        Args:
            image_id: Unique identifier for the image
            image_path: Path to the image file
            annotations: All annotations for this image
            semantic_predictions: Optional map of annotation_id -> semantic class
            image_size: (width, height) of the image
            
        Returns:
            SampleScore for the image
        """
        semantic_predictions = semantic_predictions or {}
        
        annotation_scores = []
        flagged_count = 0
        
        for ann in annotations:
            semantic_class = semantic_predictions.get(str(ann.id))
            
            score = self.score_uncertainty(
                ann, annotations, semantic_class, image_size
            )
            annotation_scores.append(score)
            
            # Check if should flag
            if self._should_flag_annotation(ann, score):
                flagged_count += 1
        
        # Image-level uncertainty is max of annotation uncertainties
        # (one bad annotation makes the whole image uncertain)
        total_uncertainty = max(
            [s.total_score for s in annotation_scores], default=0.0
        )
        
        sample_score = SampleScore(
            image_id=image_id,
            image_path=image_path,
            total_uncertainty=total_uncertainty,
            annotation_count=len(annotations),
            flagged_count=flagged_count,
            annotation_uncertainties=annotation_scores
        )
        
        self.scored_samples[image_id] = sample_score
        
        return sample_score
    
    def _should_flag_annotation(
        self,
        annotation: LabelAnnotation,
        score: UncertaintyScore
    ) -> bool:
        """Check if an annotation should be flagged for human review."""
        # Low confidence
        if annotation.confidence < self.FLAG_THRESHOLDS["min_confidence"]:
            return True
        
        # High detection-mask disagreement
        if score.detection_mask_disagreement > self.FLAG_THRESHOLDS["max_disagreement"]:
            return True
        
        # High total uncertainty
        if score.total_score > 0.6:
            return True
        
        return False
    
    def select_samples_for_labeling(
        self,
        scored_samples: Optional[List[SampleScore]] = None,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
        n_samples: int = 100
    ) -> List[str]:
        """
        Select most informative samples for human labeling.
        
        Args:
            scored_samples: Pre-scored samples (uses internal cache if None)
            strategy: Sampling strategy to use
            n_samples: Number of samples to select
            
        Returns:
            List of image_ids for human labeling
        """
        if scored_samples is None:
            scored_samples = list(self.scored_samples.values())
        
        if not scored_samples:
            logger.warning("No samples to select from")
            return []
        
        if strategy == SamplingStrategy.UNCERTAINTY:
            # Sort by uncertainty (highest first)
            sorted_samples = sorted(
                scored_samples,
                key=lambda x: -x.total_uncertainty
            )
        
        elif strategy == SamplingStrategy.DIVERSITY:
            # For diversity, use annotation counts and class distribution
            # (simplified - full implementation would use embeddings)
            sorted_samples = sorted(
                scored_samples,
                key=lambda x: (x.annotation_count, -x.total_uncertainty)
            )
        
        elif strategy == SamplingStrategy.HYBRID:
            # Combine uncertainty with diversity (alternating selection)
            by_uncertainty = sorted(
                scored_samples, key=lambda x: -x.total_uncertainty
            )
            by_diversity = sorted(
                scored_samples, key=lambda x: x.annotation_count
            )
            
            selected = []
            used = set()
            
            for i in range(min(n_samples, len(scored_samples))):
                if i % 2 == 0:
                    source = by_uncertainty
                else:
                    source = by_diversity
                
                for sample in source:
                    if sample.image_id not in used:
                        selected.append(sample)
                        used.add(sample.image_id)
                        break
            
            sorted_samples = selected
        
        else:  # RANDOM
            import random
            sorted_samples = scored_samples.copy()
            random.shuffle(sorted_samples)
        
        selected_ids = [s.image_id for s in sorted_samples[:n_samples]]
        
        logger.info(
            f"Selected {len(selected_ids)} samples using {strategy.value} strategy"
        )
        
        return selected_ids
    
    def get_flagged_annotations(
        self,
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all annotations flagged for human review.
        
        Returns:
            List of flagged annotation details
        """
        flagged = []
        
        for sample in self.scored_samples.values():
            if sample.flagged_count == 0:
                continue
            
            for i, score in enumerate(sample.annotation_uncertainties):
                if score.total_score > 0.5:
                    flagged.append({
                        "image_id": sample.image_id,
                        "image_path": sample.image_path,
                        "annotation_index": i,
                        "uncertainty_score": score.total_score,
                        "uncertainty_details": score.to_dict()
                    })
        
        return sorted(flagged, key=lambda x: -x["uncertainty_score"])
    
    def record_correction(
        self,
        image_id: str,
        annotation_id: str,
        correction_type: str
    ):
        """
        Record a human correction for future analysis.
        
        Args:
            image_id: Image that was corrected
            annotation_id: Annotation that was corrected
            correction_type: Type of correction (delete, modify, add)
        """
        if image_id not in self.label_history:
            self.label_history[image_id] = []
        
        self.label_history[image_id].append({
            "annotation_id": annotation_id,
            "correction_type": correction_type,
            "timestamp": str(Path("").absolute())  # Placeholder
        })
        
        logger.info(f"Recorded {correction_type} correction for {annotation_id}")
    
    def get_retraining_candidates(
        self,
        min_corrections: int = 5
    ) -> List[str]:
        """
        Get images with enough corrections to be useful for retraining.
        
        Returns:
            List of image_ids suitable for retraining
        """
        candidates = []
        
        for image_id, history in self.label_history.items():
            if len(history) >= min_corrections:
                candidates.append(image_id)
        
        return candidates
    
    def export_for_retraining(
        self,
        output_path: Path,
        image_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export corrected annotations for model retraining.
        
        Args:
            output_path: Directory to save exports
            image_ids: Specific images to export (all corrected if None)
            
        Returns:
            Summary of exported data
        """
        if image_ids is None:
            image_ids = list(self.label_history.keys())
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "images": [],
            "correction_count": 0,
            "image_count": len(image_ids)
        }
        
        for image_id in image_ids:
            if image_id in self.scored_samples:
                sample = self.scored_samples[image_id]
                export_data["images"].append({
                    "id": image_id,
                    "path": sample.image_path,
                    "annotations": sample.annotation_count,
                    "corrections": len(self.label_history.get(image_id, []))
                })
                export_data["correction_count"] += len(
                    self.label_history.get(image_id, [])
                )
        
        # Save summary
        summary_path = output_path / "retraining_summary.json"
        with open(summary_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(image_ids)} images for retraining to {output_path}")
        
        return export_data


# Global instance
_al_service: Optional[ActiveLearningService] = None


def get_active_learning_service() -> ActiveLearningService:
    """Get the global active learning service instance."""
    global _al_service
    if _al_service is None:
        _al_service = ActiveLearningService()
    return _al_service
