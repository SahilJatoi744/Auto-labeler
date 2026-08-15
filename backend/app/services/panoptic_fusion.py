# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Panoptic Fusion Engine.
Merges instance segmentation (YOLO+SAM2) with semantic segmentation (DeepLabV3+)
to create complete scene understanding with both 'things' and 'stuff'.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..core.logging import get_logger
from ..models.schemas import LabelAnnotation, SegmentationMask, BoundingBox

logger = get_logger("panoptic_fusion")


@dataclass
class PanopticSegment:
    """A segment in panoptic output."""
    id: int
    class_name: str
    class_id: int
    is_thing: bool  # True for instances (car, person), False for stuff (road, sky)
    confidence: float
    bbox: Optional[BoundingBox] = None
    polygon: Optional[List[List[float]]] = None
    mask: Optional[np.ndarray] = None
    area: float = 0.0


@dataclass
class PanopticResult:
    """Complete panoptic segmentation result."""
    segments: List[PanopticSegment] = field(default_factory=list)
    panoptic_map: Optional[np.ndarray] = None  # Per-pixel segment IDs
    class_map: Optional[np.ndarray] = None      # Per-pixel class IDs
    image_size: Tuple[int, int] = (0, 0)        # (height, width)
    
    def get_things(self) -> List[PanopticSegment]:
        """Get all 'thing' segments (instances)."""
        return [s for s in self.segments if s.is_thing]
    
    def get_stuff(self) -> List[PanopticSegment]:
        """Get all 'stuff' segments (semantic regions)."""
        return [s for s in self.segments if not s.is_thing]
    
    def to_coco_panoptic(self) -> Dict[str, Any]:
        """Convert to COCO panoptic format."""
        segments_info = []
        for seg in self.segments:
            segments_info.append({
                "id": seg.id,
                "category_id": seg.class_id,
                "iscrowd": 0,
                "area": seg.area,
                "bbox": [seg.bbox.x, seg.bbox.y, seg.bbox.width, seg.bbox.height] if seg.bbox else None
            })
        
        return {
            "segments_info": segments_info,
            "image_size": {"height": self.image_size[0], "width": self.image_size[1]}
        }


class PanopticFusionEngine:
    """
    Fuses instance and semantic segmentation outputs.
    
    Pipeline:
    1. Start with instance masks (things: cars, people, animals)
    2. Fill remaining pixels with semantic masks (stuff: road, sky, building)
    3. Resolve overlaps using priority strategy
    """
    
    # Default 'thing' classes (have instances)
    THING_CLASSES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush", "rider"
    }
    
    # Default 'stuff' classes (amorphous regions)
    STUFF_CLASSES = {
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "vegetation", "terrain", "sky", "water", "ground", "grass",
        "mountain", "sand", "snow", "ceiling", "floor", "window", "door"
    }
    
    def __init__(self):
        self.next_segment_id = 1
    
    def fuse(
        self,
        instance_annotations: List[LabelAnnotation],
        semantic_masks: Dict[str, np.ndarray],
        image_size: Tuple[int, int],
        priority: str = "instance",
        overlap_threshold: float = 0.5
    ) -> PanopticResult:
        """
        Merge instance and semantic segmentation outputs.
        
        Args:
            instance_annotations: Instance annotations from YOLO+SAM2
            semantic_masks: Dict of class_name -> binary mask from semantic seg
            image_size: (height, width) of the image
            priority: "instance" (default) or "semantic" for overlap handling
            overlap_threshold: IoU threshold for considering overlap
            
        Returns:
            PanopticResult with fused segmentation
        """
        h, w = image_size
        self.next_segment_id = 1
        
        # Initialize output maps
        panoptic_map = np.zeros((h, w), dtype=np.int32)
        class_map = np.zeros((h, w), dtype=np.int32)
        
        segments = []
        
        # Step 1: Add instance masks (things)
        if priority == "instance":
            segments = self._add_instances_first(
                instance_annotations, panoptic_map, class_map, segments, image_size
            )
            segments = self._add_semantic_to_gaps(
                semantic_masks, panoptic_map, class_map, segments, image_size
            )
        else:
            # Add semantic first, then overlay instances
            segments = self._add_semantic_first(
                semantic_masks, panoptic_map, class_map, segments, image_size
            )
            segments = self._overlay_instances(
                instance_annotations, panoptic_map, class_map, segments, image_size
            )
        
        logger.info(
            f"Panoptic fusion complete: {len([s for s in segments if s.is_thing])} things, "
            f"{len([s for s in segments if not s.is_thing])} stuff regions"
        )
        
        return PanopticResult(
            segments=segments,
            panoptic_map=panoptic_map,
            class_map=class_map,
            image_size=image_size
        )
    
    def _add_instances_first(
        self,
        annotations: List[LabelAnnotation],
        panoptic_map: np.ndarray,
        class_map: np.ndarray,
        segments: List[PanopticSegment],
        image_size: Tuple[int, int]
    ) -> List[PanopticSegment]:
        """Add instance masks to panoptic map with priority."""
        h, w = image_size
        
        # Sort by confidence (higher confidence = painted last = visible)
        sorted_anns = sorted(annotations, key=lambda x: x.confidence)
        
        for ann in sorted_anns:
            if ann.segmentation is None or ann.segmentation.polygon is None:
                continue
            
            # Convert polygon to mask
            mask = self._polygon_to_mask(ann.segmentation.polygon, (h, w))
            
            if mask is None or mask.sum() < 10:
                continue
            
            segment_id = self.next_segment_id
            self.next_segment_id += 1
            
            # Paint on panoptic map
            panoptic_map[mask > 0] = segment_id
            class_map[mask > 0] = ann.class_id
            
            # Create segment
            segment = PanopticSegment(
                id=segment_id,
                class_name=ann.class_name,
                class_id=ann.class_id,
                is_thing=True,
                confidence=ann.confidence,
                bbox=ann.bbox,
                polygon=ann.segmentation.polygon,
                area=float(mask.sum())
            )
            segments.append(segment)
        
        return segments
    
    def _add_semantic_to_gaps(
        self,
        semantic_masks: Dict[str, np.ndarray],
        panoptic_map: np.ndarray,
        class_map: np.ndarray,
        segments: List[PanopticSegment],
        image_size: Tuple[int, int]
    ) -> List[PanopticSegment]:
        """Fill gaps (zero pixels) with semantic masks."""
        h, w = image_size
        
        # Sort by typical importance (road first, then others)
        priority_order = ["road", "sidewalk", "building", "vegetation", "sky"]
        
        sorted_classes = []
        for cls in priority_order:
            if cls in semantic_masks:
                sorted_classes.append(cls)
        for cls in semantic_masks:
            if cls not in sorted_classes:
                sorted_classes.append(cls)
        
        for class_name in sorted_classes:
            mask = semantic_masks[class_name]
            
            # Only fill where panoptic_map is 0 (no instance)
            gap_mask = (panoptic_map == 0) & (mask > 0)
            
            if gap_mask.sum() < 100:  # Skip tiny regions
                continue
            
            segment_id = self.next_segment_id
            self.next_segment_id += 1
            
            # Get a class ID for this stuff class (use negative or high numbers)
            class_id = 1000 + len(segments)  # Stuff classes get high IDs
            
            # Paint on panoptic map
            panoptic_map[gap_mask] = segment_id
            class_map[gap_mask] = class_id
            
            # Convert mask to polygon
            polygon = self._mask_to_polygon(gap_mask.astype(np.uint8))
            
            # Calculate bounding box
            coords = np.where(gap_mask)
            if len(coords[0]) > 0:
                bbox = BoundingBox(
                    x=float(coords[1].min()),
                    y=float(coords[0].min()),
                    width=float(coords[1].max() - coords[1].min()),
                    height=float(coords[0].max() - coords[0].min())
                )
            else:
                bbox = None
            
            segment = PanopticSegment(
                id=segment_id,
                class_name=class_name,
                class_id=class_id,
                is_thing=False,
                confidence=0.8,  # Default confidence for semantic
                bbox=bbox,
                polygon=polygon,
                area=float(gap_mask.sum())
            )
            segments.append(segment)
        
        return segments
    
    def _add_semantic_first(
        self,
        semantic_masks: Dict[str, np.ndarray],
        panoptic_map: np.ndarray,
        class_map: np.ndarray,
        segments: List[PanopticSegment],
        image_size: Tuple[int, int]
    ) -> List[PanopticSegment]:
        """Add semantic masks first (for semantic priority mode)."""
        h, w = image_size
        
        for class_name, mask in semantic_masks.items():
            if mask.sum() < 100:
                continue
            
            segment_id = self.next_segment_id
            self.next_segment_id += 1
            
            class_id = 1000 + len(segments)
            
            panoptic_map[mask > 0] = segment_id
            class_map[mask > 0] = class_id
            
            polygon = self._mask_to_polygon(mask)
            
            coords = np.where(mask > 0)
            bbox = BoundingBox(
                x=float(coords[1].min()),
                y=float(coords[0].min()),
                width=float(coords[1].max() - coords[1].min()),
                height=float(coords[0].max() - coords[0].min())
            ) if len(coords[0]) > 0 else None
            
            segment = PanopticSegment(
                id=segment_id,
                class_name=class_name,
                class_id=class_id,
                is_thing=False,
                confidence=0.8,
                bbox=bbox,
                polygon=polygon,
                area=float(mask.sum())
            )
            segments.append(segment)
        
        return segments
    
    def _overlay_instances(
        self,
        annotations: List[LabelAnnotation],
        panoptic_map: np.ndarray,
        class_map: np.ndarray,
        segments: List[PanopticSegment],
        image_size: Tuple[int, int]
    ) -> List[PanopticSegment]:
        """Overlay instances on top of semantic (for semantic priority mode)."""
        # Same as _add_instances_first but overwrites existing pixels
        return self._add_instances_first(
            annotations, panoptic_map, class_map, segments, image_size
        )
    
    def _polygon_to_mask(
        self, 
        polygon: List[List[float]], 
        image_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Convert polygon to binary mask."""
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for poly in polygon:
            if len(poly) < 6:
                continue
            
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        
        return mask
    
    def _mask_to_polygon(
        self, 
        mask: np.ndarray,
        simplify_factor: float = 0.002
    ) -> Optional[List[List[float]]]:
        """Convert binary mask to polygon."""
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask
        
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )
        
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 100 or len(contour) < 3:
                continue
            
            epsilon = simplify_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                points = approx.reshape(-1, 2).tolist()
                polygons.append([float(c) for pt in points for c in pt])
        
        return polygons if polygons else None
    
    def merge_overlapping_instances(
        self,
        annotations: List[LabelAnnotation],
        iou_threshold: float = 0.7
    ) -> List[LabelAnnotation]:
        """
        Merge highly overlapping instances of the same class.
        
        Useful for removing duplicate detections from different scales.
        """
        if len(annotations) <= 1:
            return annotations
        
        # Sort by confidence (descending)
        sorted_anns = sorted(annotations, key=lambda x: -x.confidence)
        
        keep = [True] * len(sorted_anns)
        
        for i in range(len(sorted_anns)):
            if not keep[i]:
                continue
            
            for j in range(i + 1, len(sorted_anns)):
                if not keep[j]:
                    continue
                
                # Only merge same class
                if sorted_anns[i].class_name != sorted_anns[j].class_name:
                    continue
                
                # Calculate IoU
                iou = self._calculate_bbox_iou(
                    sorted_anns[i].bbox, sorted_anns[j].bbox
                )
                
                if iou > iou_threshold:
                    keep[j] = False
        
        return [ann for ann, k in zip(sorted_anns, keep) if k]
    
    def _calculate_bbox_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate IoU of two bounding boxes."""
        if box1 is None or box2 is None:
            return 0.0
        
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


# Global instance
_fusion_engine: Optional[PanopticFusionEngine] = None


def get_fusion_engine() -> PanopticFusionEngine:
    """Get the global panoptic fusion engine instance."""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = PanopticFusionEngine()
    return _fusion_engine
