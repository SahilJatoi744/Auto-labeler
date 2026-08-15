# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Pydantic models for API request/response validation.
Defines all data structures for the application.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Supported computer vision tasks."""
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"


class ExportFormat(str, Enum):
    """Supported export formats."""
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    YOLO = "yolo"


class LabelingStrategy(str, Enum):
    """Labeling strategies."""
    RULE_BASED = "rule_based"
    AI_ASSISTED = "ai_assisted"
    HYBRID = "hybrid"


class ClassDefinition(BaseModel):
    """Definition of a class/label."""
    id: int = Field(..., description="Unique class ID")
    name: str = Field(..., description="Class name")
    description: Optional[str] = Field(None, description="Class description")
    parent_id: Optional[int] = Field(None, description="Parent class ID for hierarchy")
    color: Optional[str] = Field(None, description="Hex color for visualization")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes")
    
    @field_validator('color')
    @classmethod
    def validate_color(cls, v):
        if v and not v.startswith('#'):
            return f'#{v}'
        return v


class ClassHierarchy(BaseModel):
    """Hierarchical class structure."""
    classes: List[ClassDefinition] = Field(..., description="List of class definitions")
    
    def get_children(self, parent_id: int) -> List[ClassDefinition]:
        """Get all child classes of a parent."""
        return [c for c in self.classes if c.parent_id == parent_id]
    
    def get_root_classes(self) -> List[ClassDefinition]:
        """Get all root classes (no parent)."""
        return [c for c in self.classes if c.parent_id is None]
    
    def validate_hierarchy(self) -> bool:
        """Validate the class hierarchy has no cycles."""
        visited = set()
        
        def has_cycle(class_id: int, path: set) -> bool:
            if class_id in path:
                return True
            if class_id in visited:
                return False
            path.add(class_id)
            visited.add(class_id)
            for child in self.get_children(class_id):
                if has_cycle(child.id, path):
                    return True
            path.remove(class_id)
            return False
        
        for cls in self.get_root_classes():
            if has_cycle(cls.id, set()):
                return False
        return True


class DatasetInfo(BaseModel):
    """Information about an uploaded dataset."""
    id: str = Field(..., description="Dataset unique ID")
    name: str = Field(..., description="Dataset name")
    path: str = Field(..., description="Dataset storage path")
    total_images: int = Field(0, description="Total number of images")
    valid_images: int = Field(0, description="Number of valid images")
    corrupted_images: int = Field(0, description="Number of corrupted images")
    total_size_mb: float = Field(0.0, description="Total size in MB")
    formats: Dict[str, int] = Field(default_factory=dict, description="Image format counts")
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field("pending", description="Dataset status")


class ImageInfo(BaseModel):
    """Information about a single image."""
    id: str = Field(..., description="Image unique ID")
    filename: str = Field(..., description="Image filename")
    path: str = Field(..., description="Image path")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    format: str = Field(..., description="Image format")
    size_bytes: int = Field(..., description="File size in bytes")
    status: str = Field("pending", description="Processing status")


class BoundingBox(BaseModel):
    """Bounding box annotation."""
    x: float = Field(..., description="X coordinate (top-left)")
    y: float = Field(..., description="Y coordinate (top-left)")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height


class SegmentationMask(BaseModel):
    """Segmentation mask annotation."""
    polygon: Optional[List[List[List[float]]]] = Field(None, description="Polygon points [[[x,y], [x,y]], ...]")
    rle: Optional[Dict[str, Any]] = Field(None, description="Run-length encoding")
    mask_path: Optional[str] = Field(None, description="Path to mask file")

    @field_validator("polygon", mode="before")
    @classmethod
    def normalize_polygon(cls, value):
        if value is None:
            return value
        normalized = []
        for contour in value:
            if not contour:
                continue
            first = contour[0]
            if isinstance(first, (int, float)):
                points = []
                coords = list(contour)
                for i in range(0, len(coords) - 1, 2):
                    points.append([float(coords[i]), float(coords[i + 1])])
                if points:
                    normalized.append(points)
            else:
                normalized.append([[float(point[0]), float(point[1])] for point in contour if len(point) >= 2])
        return normalized


class LabelAnnotation(BaseModel):
    """Single label annotation for an object."""
    id: int = Field(..., description="Annotation ID")
    image_id: str = Field(..., description="Parent image ID")
    class_id: int = Field(..., description="Class ID")
    class_name: Optional[str] = Field(None, description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box")
    segmentation: Optional[SegmentationMask] = Field(None, description="Segmentation mask")
    area: Optional[float] = Field(None, description="Object area")
    iscrowd: bool = Field(False, description="Is crowd annotation")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes")


class ImageLabels(BaseModel):
    """All labels for a single image."""
    image_id: str = Field(..., description="Image ID")
    image_url: Optional[str] = Field(None, description="Static URL to the image")
    annotations: List[LabelAnnotation] = Field(default_factory=list)
    status: str = Field("labeled", description="Labeling status")
    processed_at: Optional[datetime] = Field(None)
    processing_time_ms: Optional[float] = Field(None)


class LabelingJob(BaseModel):
    """Labeling job configuration and status."""
    id: str = Field(..., description="Job unique ID")
    dataset_id: str = Field(..., description="Dataset ID")
    task_type: TaskType = Field(..., description="Type of labeling task")
    strategy: LabelingStrategy = Field(default=LabelingStrategy.AI_ASSISTED)
    class_hierarchy: ClassHierarchy = Field(..., description="Class definitions")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    models_config: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field("pending", description="Job status")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    progress: Dict[str, Any] = Field(default_factory=dict)
    total_images: int = Field(0)
    processed_images: int = Field(0)
    failed_images: int = Field(0)


class JobCreate(BaseModel):
    """Request model for creating a labeling job."""
    dataset_id: str
    task_type: TaskType
    class_hierarchy: ClassHierarchy
    strategy: LabelingStrategy = LabelingStrategy.AI_ASSISTED
    confidence_threshold: float = 0.5
    models_config: Optional[Dict[str, Any]] = None
    device: str = Field(default="auto", description="Device to use: 'gpu', 'cpu', or 'auto' (detect automatically)")
    custom_classes: Optional[List[str]] = Field(None, description="Custom class names for detection/segmentation")



class LabelingProgress(BaseModel):
    """Real-time labeling progress."""
    job_id: str = Field(..., description="Job ID")
    total_images: int = Field(0)
    processed_images: int = Field(0)
    failed_images: int = Field(0)
    current_image: Optional[str] = Field(None)
    current_model: Optional[str] = Field(None)
    estimated_time_remaining: Optional[int] = Field(None, description="Seconds remaining")
    status: str = Field("running")
    errors: List[str] = Field(default_factory=list)


class ExportConfig(BaseModel):
    """Export configuration."""
    job_id: str = Field(..., description="Job ID to export")
    format: ExportFormat = Field(default=ExportFormat.COCO)
    split_ratios: Dict[str, float] = Field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    include_unlabeled: bool = Field(default=False)
    min_confidence: Optional[float] = Field(None)
    output_path: Optional[str] = Field(None)


class ExportResult(BaseModel):
    """Export result information."""
    export_id: str = Field(..., description="Export unique ID")
    job_id: str = Field(..., description="Source job ID")
    format: ExportFormat = Field(..., description="Export format")
    output_path: str = Field(..., description="Output directory path")
    file_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to exported files")
    statistics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Information about an AI model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    task_types: List[TaskType] = Field(..., description="Supported task types")
    description: Optional[str] = Field(None)
    download_url: Optional[str] = Field(None)
    is_downloaded: bool = Field(False)
    file_size_mb: Optional[float] = Field(None)


class SystemStatus(BaseModel):
    """System health and status."""
    status: str = Field("healthy")
    version: str = Field(...)
    gpu_available: bool = Field(False)
    gpu_info: Optional[Dict[str, Any]] = Field(None)
    cpu_count: int = Field(0)
    cpu_usage: float = Field(0.0)
    memory_gb: float = Field(0.0)
    memory_usage_percent: float = Field(0.0)
    disk_space_gb: float = Field(0.0)
    active_jobs: int = Field(0)
    models_loaded: List[str] = Field(default_factory=list)
    device_preference: str = Field("auto")

# ==================== Active Learning Schemas ====================

class UncertaintyScore(BaseModel):
    """Uncertainty metrics."""
    total_score: float
    confidence_uncertainty: float
    detection_mask_disagreement: float
    semantic_instance_disagreement: float
    size_uncertainty: float
    overlap_uncertainty: float


class SampleScore(BaseModel):
    """Score for an image sample."""
    image_id: str
    image_path: str
    total_uncertainty: float
    annotation_count: int
    flagged_count: int
    annotation_uncertainties: List[UncertaintyScore]


class ActiveLearningRequest(BaseModel):
    """Request to select samples for active learning."""
    job_id: str = Field(..., description="Job ID to select from")
    strategy: str = Field("uncertainty", description="Sampling strategy: uncertainty, diversity, hybrid, random")
    n_samples: int = Field(100, description="Number of samples to select")


class ActiveLearningConfig(BaseModel):
    """Configuration for active learning selection."""
    strategy: str = "uncertainty"
    n_samples: int = 100
    min_uncertainty: float = 0.5


class ActiveLearningResult(BaseModel):
    """Result of active learning selection."""
    selected_count: int
    selected_image_ids: List[str]
    strategy_used: str
    avg_uncertainty: float


class FlaggedSample(BaseModel):
    """Flagged sample for human review."""
    image_id: str
    image_path: str
    annotation_index: int
    uncertainty_score: float
    uncertainty_details: Dict[str, float]


# ==================== Human-in-the-Loop Schemas ====================

class AnnotationStatus(str, Enum):
    """Status of an annotation in the review pipeline."""
    AUTO = "auto"           # Auto-generated, not reviewed
    FLAGGED = "flagged"     # Flagged for human review
    APPROVED = "approved"   # Human approved
    CORRECTED = "corrected" # Human corrected


class RefinementType(str, Enum):
    """Type of refinement for SAM2 interactive correction."""
    POINT_ADD = "point_add"           # Add positive point
    POINT_REMOVE = "point_remove"     # Add negative point
    BBOX_ADJUST = "bbox_adjust"       # Adjust bounding box
    STROKE_ADD = "stroke_add"         # Add inclusion stroke
    STROKE_REMOVE = "stroke_remove"   # Add exclusion stroke


class PointPrompt(BaseModel):
    """Point prompt for SAM2 refinement."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    label: int = Field(1, description="1 for positive (include), 0 for negative (exclude)")


class RefinementRequest(BaseModel):
    """Request for refining an annotation mask."""
    refinement_type: RefinementType = Field(..., description="Type of refinement")
    points: Optional[List[PointPrompt]] = Field(None, description="Point prompts for SAM2")
    bbox_adjustment: Optional[BoundingBox] = Field(None, description="New bounding box")
    stroke_mask: Optional[str] = Field(None, description="Base64 encoded binary stroke mask")


class UncertaintyDetails(BaseModel):
    """Detailed uncertainty metrics for an annotation."""
    total: float = Field(..., ge=0.0, le=1.0, description="Overall uncertainty score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="1 - model confidence")
    detection_mask_disagreement: float = Field(..., ge=0.0, le=1.0)
    semantic_instance_disagreement: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0, le=1.0, description="Size-based uncertainty")
    overlap: float = Field(..., ge=0.0, le=1.0, description="Overlap-based uncertainty")


class FlaggedAnnotation(BaseModel):
    """Annotation flagged for human review."""
    image_id: str
    annotation_id: int
    image_url: Optional[str] = None
    class_name: str
    confidence: float
    uncertainty: UncertaintyDetails = Field(..., description="Detailed uncertainty metrics")
    reason: str = Field(default="Low confidence", description="Reason for flagging")


class AnnotationUpdate(BaseModel):
    """Request for updating an annotation status."""
    status: AnnotationStatus
    class_id: Optional[int] = Field(None, description="New class ID if changed")
    class_name: Optional[str] = Field(None, description="New class name if changed")
    bbox: Optional[BoundingBox] = Field(None, description="Updated bounding box")
    polygon: Optional[List[List[float]]] = Field(None, description="Updated polygon")


class ActiveLearningConfig(BaseModel):
    """Configuration for active learning sample selection."""
    strategy: str = Field("uncertainty", description="Selection strategy: uncertainty, diversity, hybrid, random")
    n_samples: int = Field(100, ge=1, le=10000, description="Number of samples to select")
    min_uncertainty: float = Field(0.3, ge=0.0, le=1.0, description="Min uncertainty to consider")


class ActiveLearningResult(BaseModel):
    """Result of active learning sample selection."""
    selected_count: int
    selected_image_ids: List[str]
    strategy_used: str
    avg_uncertainty: float


class RefineRequest(BaseModel):
    """Request for prompt-based label refinement."""
    image_id: str
    prompt: str
