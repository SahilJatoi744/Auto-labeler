# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Main labeling service that orchestrates the labeling pipeline.
Combines preprocessing, AI inference, and post-processing.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import cv2
import numpy as np

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import (
    BoundingBox, ClassHierarchy, DatasetInfo, ImageInfo, ImageLabels,
    LabelAnnotation, LabelingJob, LabelingProgress, SegmentationMask,
    TaskType, LabelingStrategy
)
from .preprocessor import get_preprocessor
from .model_manager import get_model_manager
from .platform import get_platform_service

logger = get_logger("labeler")


class LabelingService:
    """
    Main service for automatic image labeling.
    Orchestrates the entire labeling pipeline.
    """
    
    # Class alias map: loaded from settings.DEFAULT_ALIAS_MAP (configurable via .env)
    # Override per-job via class_hierarchy.classes[].attributes.aliases
    @property
    def ALIAS_MAP(self):
        return settings.DEFAULT_ALIAS_MAP

    def __init__(self):
        self.preprocessor = get_preprocessor()
        self.model_manager = get_model_manager()
        self.platform = get_platform_service()
        self.logger = get_logger("labeler")
        
        # Track active jobs
        self.active_jobs: Dict[str, LabelingJob] = {}
        self.job_results: Dict[str, List[ImageLabels]] = {}
        self._stop_flags: Dict[str, bool] = {}
        
        # Load existing jobs from disk
        self._load_jobs()
    
    def create_job(
        self,
        dataset_info: DatasetInfo,
        task_type: TaskType,
        class_hierarchy: ClassHierarchy,
        strategy: LabelingStrategy = LabelingStrategy.AI_ASSISTED,
        confidence_threshold: float = None,
        models_config: Dict = None
    ) -> LabelingJob:
        """
        Create a new labeling job.
        
        Args:
            dataset_info: Validated dataset information
            task_type: Type of labeling task
            class_hierarchy: Hierarchical class definitions
            strategy: Labeling strategy to use
            confidence_threshold: Minimum confidence for labels
            models_config: Model-specific configuration
            
        Returns:
            Created labeling job
        """
        # Generate job ID
        job_id = hashlib.md5(
            f"{dataset_info.id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        job = LabelingJob(
            id=job_id,
            dataset_id=dataset_info.id,
            task_type=task_type,
            strategy=strategy,
            class_hierarchy=class_hierarchy,
            confidence_threshold=confidence_threshold or settings.DEFAULT_CONFIDENCE_THRESHOLD,
            models_config=models_config or {},
            status="created",
            total_images=dataset_info.valid_images
        )
        
        self.active_jobs[job_id] = job
        self._save_job_metadata(job)
        self.logger.info(f"Created labeling job: {job_id}")
        
        return job
    
    async def run_job(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[LabelingProgress], None]] = None
    ) -> List[ImageLabels]:
        """
        Run a labeling job.
        
        Args:
            job_id: Job ID to run
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of image labels
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job not found: {job_id}")
        
        job = self.active_jobs[job_id]
        self._stop_flags[job_id] = False
        
        if job.status == "running":
            self.logger.warning(f"Job {job_id} is already running.")
            return []

        job.status = "running"
        job.started_at = datetime.now()
        self._save_job_metadata(job)
        
        self.logger.info(f"Starting labeling job: {job_id}")
        
        # Load dataset metadata
        metadata_path = settings.UPLOAD_DIR / f"{job.dataset_id}_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        images = metadata.get("images", [])
        results = []
        
        # Update total images if metadata changed or missing
        if job.total_images == 0:
            job.total_images = len(images)
            self._save_job_metadata(job)
        
        # Load appropriate models based on task
        # Load appropriate models based on task
        models_config = getattr(job, "models_config", {}) or {}
        self._load_models_for_task(job.task_type, models_config)
        
        # Process images in batches
        start_time = time.time()
        processed = 0
        failed = 0
        batch_size = settings.BATCH_SIZE
        
        for i in range(0, len(images), batch_size):
            # Check stop flag
            if self._stop_flags.get(job_id, False):
                self.logger.info(f"Job stopped: {job_id}")
                job.status = "stopped"
                break
            
            batch_images = images[i : i + batch_size]
            
            for img_data in batch_images:
                try:
                    image_info = ImageInfo(**img_data)
                    
                    # Process single image
                    labels = await self._process_image(
                        image_info,
                        job.task_type,
                        job.class_hierarchy,
                        job.confidence_threshold,
                        models_config,
                    )
                    
                    results.append(labels)
                    processed += 1
                    
                    # Update progress
                    job.processed_images = processed
                    job.progress = {
                        "current_image": image_info.filename,
                        "processed": processed,
                        "total": len(images),
                        "percentage": round(processed / len(images) * 100, 1),
                        "elapsed_seconds": int(time.time() - start_time)
                    }
                    
                    # Send progress update
                    if progress_callback:
                        progress = LabelingProgress(
                            job_id=job_id,
                            total_images=len(images),
                            processed_images=processed,
                            failed_images=failed,
                            current_image=image_info.filename,
                            status="running"
                        )
                        progress_callback(progress)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process image {img_data.get('filename')}: {e}")
                    failed += 1
                    job.failed_images = failed
            
            # Per-batch memory cleanup
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Yield control to event loop
            await asyncio.sleep(0.01)

        
        # Save results
        job.status = "completed" if not self._stop_flags.get(job_id, False) else "stopped"
        job.completed_at = datetime.now()
        self.job_results[job_id] = results
        
        # Save both metadata and results
        self._save_job_metadata(job)
        self._save_results(job_id, results)
        self.platform.record_metric("jobs.completed", 1, {"job_id": job_id, "status": job.status})
        self.platform.record_model_run(
            model_name="auto-labeler-pipeline",
            task=job.task_type.value,
            inputs={"dataset_id": job.dataset_id, "images": len(images), "threshold": job.confidence_threshold},
            outputs={"processed": processed, "failed": failed, "annotations": sum(len(r.annotations) for r in results)},
        )
        self.platform.record_audit_event("job.complete", "job", job_id, {"processed": processed, "failed": failed})
        
        self.logger.info(
            f"Job {job_id} finished: {processed} processed, {failed} failed"
        )
        
        return results
    
    @staticmethod
    def _compute_image_url(image_path_str: str) -> str:
        """Convert an absolute file path to a /uploads/ URL."""
        p = str(image_path_str).replace("\\", "/")
        if "uploads/" in p:
            return "/uploads/" + p.split("uploads/")[-1]
        # fallback: last two components
        parts = p.split("/")
        return "/uploads/" + "/".join(parts[-2:]) if len(parts) >= 2 else "/uploads/" + parts[-1]

    async def _process_image(
        self,
        image_info: ImageInfo,
        task_type: TaskType,
        class_hierarchy: ClassHierarchy,
        confidence_threshold: float,
        models_config: Optional[Dict[str, Any]] = None
    ) -> ImageLabels:
        """
        Process a single image.
        
        Args:
            image_info: Image information
            task_type: Type of task
            class_hierarchy: Class hierarchy
            confidence_threshold: Confidence threshold
            
        Returns:
            Image labels
        """
        start_time = time.time()
        
        # Load image
        image_path = Path(image_info.path)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        annotations = []
        models_config = models_config or {}
        selected_model_id = models_config.get("selected_model_id", "")
        prompt_first = selected_model_id in {"sam3_concept", "grounding_dino_15", "yolo_world_sam2"}
        prompt_classes = [c.name for c in class_hierarchy.classes]
        allow_model_download = bool(models_config.get("allow_model_download", settings.ALLOW_MODEL_DOWNLOADS))
        
        # Run detection based on task type
        if task_type == TaskType.OBJECT_DETECTION:
            annotations = self._detect_with_optional_profile(
                image,
                prompt_classes,
                confidence_threshold,
                selected_model_id,
                prompt_first,
                allow_model_download,
            )
            if annotations is None:
                # Use YOLOv26 for detection (NMS-free, end-to-end)
                annotations = self.model_manager.detect_objects_yolov26(
                    image,
                    conf_threshold=confidence_threshold
                )
            # Object detection should only return bounding boxes, not masks
            # Strip any segmentation data that may come from YOLOv26-seg model
            for ann in annotations:
                ann.segmentation = None
        
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            # YOLOv26 + SAM2 Pipeline for high-quality instance segmentation
            # Step 1: Detect objects with YOLOv26-seg (includes masks)
            detections = self._detect_with_optional_profile(
                image,
                prompt_classes,
                confidence_threshold,
                selected_model_id,
                prompt_first,
                allow_model_download,
            )
            if detections is None:
                detections = self.model_manager.detect_objects_yolov26(
                    image,
                    conf_threshold=confidence_threshold
                )
            
            # Step 2: Optionally refine masks with SAM2 for better boundary quality
            if detections:
                if selected_model_id == "sam3_concept" and any(det.segmentation for det in detections):
                    annotations = detections
                    self.logger.info(f"Instance segmentation (SAM3): {len(annotations)} objects with polygons")
                else:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bboxes = [d.bbox for d in detections if d.bbox]
                    
                    # Load SAM2 for high-quality mask refinement
                    if "sam2" not in self.model_manager.models:
                        self.model_manager.load_sam2()
                    
                    sam2_masks = self.model_manager.segment_with_sam2(image_rgb, bboxes)
                    
                    # Combine detector class info with SAM2 masks (better quality)
                    for det, sam2_mask in zip(detections, sam2_masks):
                        # Use SAM2 mask if available, otherwise keep detector mask
                        if sam2_mask.polygon:
                            det.segmentation = sam2_mask
                        annotations.append(det)
                        
                    self.logger.info(f"Instance segmentation (detector+SAM2): {len(annotations)} objects with polygons")
            else:
                annotations = detections
        
        elif task_type == TaskType.SEMANTIC_SEGMENTATION:
            # Hybrid approach: YOLO for known classes, SAM+CLIP for others
            # This provides better quality for common objects (via YOLO+SAM2)
            # and open-vocabulary support for others (via SAM2+CLIP)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            annotations = None
            if selected_model_id == "sam3_concept":
                annotations = self._detect_with_optional_profile(
                    image,
                    prompt_classes,
                    confidence_threshold,
                    selected_model_id,
                    True,
                    allow_model_download,
                )
            if annotations is None:
                annotations = self._process_image_semantic_hybrid(
                    image, 
                    image_rgb, 
                    class_hierarchy, 
                    confidence_threshold
                )
            
            self.logger.info(f"Semantic segmentation (Hybrid): {len(annotations)} annotations with polygons")


        
        # Map to user-defined classes if needed
        annotations = self._map_to_user_classes(annotations, class_hierarchy)
        
        # Calculate uncertainty for each annotation
        for ann in annotations:
            uncertainty = self.model_manager.estimate_uncertainty(ann, annotations)
            ann.attributes = {"uncertainty": uncertainty}
        
        # Set image IDs
        for i, ann in enumerate(annotations):
            ann.id = i
            ann.image_id = image_info.id
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return ImageLabels(
            image_id=image_info.id,
            image_url=self._compute_image_url(image_info.path),
            annotations=annotations,
            status="labeled",
            processed_at=datetime.now(),
            processing_time_ms=round(processing_time, 2)
        )

    def _detect_with_optional_profile(
        self,
        image: np.ndarray,
        classes: List[str],
        confidence_threshold: float,
        selected_model_id: str,
        prompt_first: bool,
        allow_model_download: bool,
    ) -> Optional[List[LabelAnnotation]]:
        """Try the selected advanced runtime, falling back to prompt detector."""
        if selected_model_id == "sam3_concept":
            try:
                return self.model_manager.detect_and_segment_sam3(
                    image,
                    classes=classes,
                    conf_threshold=confidence_threshold,
                    allow_download=allow_model_download,
                )
            except Exception as e:
                self.logger.warning(f"SAM3 profile unavailable: {e}. Falling back to YOLO-World/SAM2.")

        if selected_model_id == "grounding_dino_15":
            try:
                return self.model_manager.detect_objects_grounding_dino(
                    image,
                    classes=classes,
                    conf_threshold=confidence_threshold,
                    allow_download=allow_model_download,
                )
            except Exception as e:
                self.logger.warning(f"Grounding DINO profile unavailable: {e}. Falling back to YOLO-World/SAM2.")

        if prompt_first:
            try:
                return self.model_manager.detect_objects_yolo_world(
                    image,
                    classes=classes,
                    conf_threshold=confidence_threshold,
                )
            except Exception as e:
                self.logger.warning(f"YOLO-World prompt detector unavailable: {e}. Falling back to YOLO.")

        return None
    
    def _fuse_detections_and_masks(
        self,
        detections: List[LabelAnnotation],
        raw_masks: List[Dict[str, Any]],
        class_hierarchy: Optional[ClassHierarchy] = None
    ) -> List[LabelAnnotation]:
        """
        Fuse YOLO detections (class info) with SAM masks (geometry).
        Uses a robust containment and IoU based matching strategy.
        """
        fused = []
        if not raw_masks:
            return []

        # Identify "stuff" classes from hierarchy if possible
        stuff_classes = []
        if class_hierarchy:
            stuff_classes = [c for c in class_hierarchy.classes if any(keyword in c.name.lower() for keyword in ["road", "sky", "sidewalk", "building", "background"])]

        for i, mask_data in enumerate(raw_masks):
            mask_bbox = mask_data["bbox"]  # [x, y, w, h]
            mask_segmentation = mask_data["segmentation"] # Binary array (H, W) or RLE
            
            best_det = None
            max_match_score = 0.0
            
            # Optimization: Pre-filter detections that overlap even slightly with the mask bbox
            for det in detections:
                if not det.bbox: continue
                
                det_bbox = [det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height]
                iou = self._calculate_bbox_iou_simple(mask_bbox, det_bbox)
                
                if iou <= 0: continue
                
                # Calculate Containment: How much of the SAM mask is inside the YOLO box
                # det_bbox: [x, y, w, h]
                x1, y1 = int(det_bbox[0]), int(det_bbox[1])
                x2, y2 = int(det_bbox[0] + det_bbox[2]), int(det_bbox[1] + det_bbox[3])
                
                # Height/Width of image
                h, w = mask_segmentation.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Portion of mask inside YOLO box
                mask_in_box = mask_segmentation[y1:y2, x1:x2]
                contained_area = np.sum(mask_in_box)
                total_mask_area = mask_data["area"]
                
                containment = contained_area / total_mask_area if total_mask_area > 0 else 0
                
                # Heuristic Score: Combination of IoU and Containment
                # This favors masks that are mostly contained within a YOLO box
                # and have similar shapes (IoU).
                match_score = (0.3 * iou) + (0.7 * containment)
                
                # If the YOLO prediction is highly confident and the mask is mostly inside it, 
                # we have high trust in this match.
                match_score *= (0.5 + 0.5 * det.confidence)
                
                if match_score > max_match_score and (containment > 0.5 or iou > 0.3):
                    max_match_score = match_score
                    best_det = det
            
            # Create annotation if we found a match or if it's a large unmapped region (likely background/stuff)
            polygon = self.model_manager._mask_to_polygon(mask_segmentation)
            if not polygon:
                continue

            if best_det:
                ann = LabelAnnotation(
                    id=i,
                    image_id=best_det.image_id,
                    class_id=best_det.class_id,
                    class_name=best_det.class_name,
                    confidence=float(best_det.confidence * min(1.0, max_match_score + 0.2)),
                    bbox=BoundingBox(
                        x=float(mask_bbox[0]),
                        y=float(mask_bbox[1]),
                        width=float(mask_bbox[2]),
                        height=float(mask_bbox[3])
                    ),
                    segmentation=SegmentationMask(polygon=polygon),
                    area=float(mask_data["area"])
                )
                fused.append(ann)
            # NOTE: We do NOT auto-assign unmapped masks to "stuff" classes.
            # The user wants ONLY the classes they specified. Unmapped masks are skipped.

                
        return fused

    def _calculate_bbox_iou_simple(self, box1: List[float], box2: List[float]) -> float:
        """Simple IoU for [x, y, w, h] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        union = (box1[2] * box1[3]) + (box2[2] * box2[3]) - intersection
        
        return intersection / union if union > 0 else 0.0

    def _map_to_user_classes(
        self,
        annotations: List[LabelAnnotation],
        class_hierarchy: ClassHierarchy
    ) -> List[LabelAnnotation]:
        """
        Map model output classes to user-defined class hierarchy.
        
        Args:
            annotations: Model predictions
            class_hierarchy: User-defined classes
            
        Returns:
            Mapped annotations
        """
        # 1. Pre-process hierarchy for faster lookup
        # Direct name map
        class_map = {c.name.lower(): c for c in class_hierarchy.classes}
        
        # Alias/Sub-class map from attributes
        alias_map = {}
        for c in class_hierarchy.classes:
            if c.attributes:
                aliases = c.attributes.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                for alias in aliases:
                    alias_map[alias.lower()] = c
        
        # ID lookup map
        id_map = {c.id: c for c in class_hierarchy.classes}
        
        mapped_annotations = []
        for ann in annotations:
            if not ann.class_name:
                continue
                
            class_key = ann.class_name.lower()
            user_cls = None
            
            # --- MAPPING STRATEGY ---
            
            # 1. Try exact name match
            if class_key in class_map:
                user_cls = class_map[class_key]
                self.logger.debug(f"Mapped '{ann.class_name}' via exact match to '{user_cls.name}'")
                
            # 2. Try alias match
            if not user_cls and class_key in alias_map:
                user_cls = alias_map[class_key]
                self.logger.info(f"Mapped '{ann.class_name}' via alias match to '{user_cls.name}'")
                
            # 3. Try substring match (fuzzy)
            if not user_cls:
                for user_name, candidate_cls in class_map.items():
                    if user_name in class_key or class_key in user_name:
                        user_cls = candidate_cls
                        self.logger.info(f"Mapped '{ann.class_name}' via fuzzy match to '{user_cls.name}'")
                        break
            
            # 4. Hierarchical Fallback (Logic: if model predicted 'SUV', and we have 'Car' as parent)
            # This requires model-specific knowledge or a more complex ontology.
            # For now, we'll rely on aliases for this.
            
            # Apply mapping if found
            if user_cls:
                ann.class_id = user_cls.id
                ann.class_name = user_cls.name
                mapped_annotations.append(ann)
            else:
                self.logger.warning(f"Label Mapping Failed: Model class '{ann.class_name}' not found in user hierarchy or aliases.")
        
        return mapped_annotations
    
    
    def _load_models_for_task(self, task_type: TaskType, config: Dict[str, Any] = {}):
        """
        Load required models for the task (YOLOv26 + SAM2).
        Handles dynamic device and model selection from config.
        
        Args:
            task_type: Task type
            config: Job models_config (contains use_gpu, yolo_model, sam_model)
        """
        # 1. Set Device
        device = config.get("device")
        if device:
            self.model_manager.set_device(device)
        else:
            use_gpu = config.get("use_gpu", True)
            self.model_manager.set_device(use_gpu)
        
        # 2. Get Model Names
        yolo_model = config.get("yolo_model", None) # None = default/auto
        sam_model = config.get("sam_model", None)   # None = default/auto
        selected_model_id = config.get("selected_model_id", "")
        prompt_first = selected_model_id in {"sam3_concept", "grounding_dino_15", "yolo_world_sam2"} or config.get("use_yolo_world", False)
        allow_model_download = bool(config.get("allow_model_download", settings.ALLOW_MODEL_DOWNLOADS))
        if selected_model_id == "sam3_concept":
            try:
                self.model_manager.load_sam3(config.get("sam3_model"), allow_download=allow_model_download)
                self.logger.info("SAM3 runtime loaded for concept segmentation.")
            except Exception as e:
                self.logger.warning(f"SAM3 runtime unavailable: {e}. Using local fallback.")
        elif selected_model_id == "grounding_dino_15":
            try:
                self.model_manager.load_grounding_dino(config.get("grounding_dino_model_id"), allow_download=allow_model_download)
                self.logger.info("Grounding DINO runtime loaded for open-set detection.")
            except Exception as e:
                self.logger.warning(f"Grounding DINO runtime unavailable: {e}. Using local fallback.")
        elif selected_model_id == "dinov3_quality":
            try:
                self.model_manager.load_dinov3(config.get("dinov3_model_id"), allow_download=allow_model_download)
                self.logger.info("DINOv3 runtime loaded for dataset intelligence.")
            except Exception as e:
                self.logger.warning(f"DINOv3 runtime unavailable: {e}. Labeling will use detection fallback.")
        
        # 3. Load Models based on Task
        if task_type in [TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            if prompt_first:
                if "yolo_world" not in self.model_manager.models:
                    try:
                        self.logger.info("Loading YOLO-World for prompt-first detection profile...")
                        self.model_manager.load_yolo_world()
                    except Exception as e:
                        self.logger.warning(f"YOLO-World unavailable: {e}. YOLO fallback will be used.")
            if "yolov26" not in self.model_manager.models or yolo_model:
                self.logger.info(f"Loading YOLOv26 model: {yolo_model or 'default'}")
                self.model_manager.load_yolov26(yolo_model, for_segmentation=True)
        
        # For semantic, we try to use YOLO if possible (Hybrid), so we load it
        if task_type == TaskType.SEMANTIC_SEGMENTATION:
            if "sam2" not in self.model_manager.models or sam_model:
                self.logger.info(f"Loading SAM2 model: {sam_model or 'default'}")
                self.model_manager.load_sam2(sam_model)
            
            # Load YOLO too for Hybrid Semantic approach
            if "yolov26" not in self.model_manager.models:
                self.logger.info("Loading YOLOv26 for Hybrid Semantic Segmentation...")
                self.model_manager.load_yolov26(yolo_model, for_segmentation=True) # Use same YOLO model
            
            # Load CLIP for open-vocabulary (fallback if YOLO-World not used)
            if "clip" not in self.model_manager.models and "yolo_world" not in self.model_manager.models:
                self.logger.info("Loading CLIP for Semantic Segmentation...")
                self.model_manager.load_clip()
                
            # Load YOLO-World if desirable (e.g. for prompt-based)
            if config.get("use_yolo_world", True): # Enable by default for prompt support
                if "yolo_world" not in self.model_manager.models:
                     self.logger.info("Loading YOLO-World for Prompt-Based Detection...")
                     self.model_manager.load_yolo_world()

        # For Instance Seg, we need SAM too
        if task_type == TaskType.INSTANCE_SEGMENTATION:
            if "sam2" not in self.model_manager.models or sam_model:
                self.logger.info(f"Loading SAM2 model for refinement: {sam_model or 'default'}")
                self.model_manager.load_sam2(sam_model)
    
    def _process_image_semantic_hybrid(
        self,
        image: np.ndarray,
        image_rgb: np.ndarray,
        class_hierarchy: ClassHierarchy,
        confidence_threshold: float
    ) -> List[LabelAnnotation]:
        """
        Hybrid Semantic Segmentation Pipeline.
        1. Checks if user classes map to YOLO classes.
        2. If yes, uses YOLOv26+SAM2 for those classes.
        3. For remaining classes, uses YOLO-World (Prompt-Based) if available, else SAM2+CLIP.
        """
        annotations = []
        user_classes = {c.name: c for c in class_hierarchy.classes}
        user_class_names = set(user_classes.keys())
        
        # 1. YOLO Stage (Fixed Classes)
        yolo_anns = []
        yolo_covered_classes = set()
        
        if "yolov26" in self.model_manager.models:
            yolo_model = self.model_manager.models["yolov26"]
            if hasattr(yolo_model, 'names'):
                yolo_names = yolo_model.names
                # Check overlap
                common_classes = set()
                # yolo_names can be dict {0: 'person', ...}
                yolo_vals = yolo_names.values() if isinstance(yolo_names, dict) else yolo_names
                
                for c_name in user_class_names:
                    # 1. Direct match
                    if any(y_name.lower() == c_name.lower() for y_name in yolo_vals):
                        common_classes.add(c_name)
                    # 2. Alias match
                    elif c_name.lower() in self.ALIAS_MAP:
                        aliases = self.ALIAS_MAP[c_name.lower()]
                        if any(y_name.lower() in aliases for y_name in yolo_vals):
                            common_classes.add(c_name)
                
                if common_classes:
                    self.logger.info(f"Hybrid Semantic: Using YOLO for {len(common_classes)} classes: {common_classes}")
                    
                    # Run Detection
                    detections = self.model_manager.detect_objects_yolov26(
                        image, conf_threshold=confidence_threshold
                    )
                    
                    # Filter for common classes
                    relevant_dets = []
                    for det in detections:
                        # Find matching user class
                        matched_user_cls = None
                        d_name = det.class_name.lower()
                        
                        # Direct check
                        for c in class_hierarchy.classes:
                            if c.name.lower() == d_name:
                                matched_user_cls = c
                                break
                        
                        # Alias check if not found
                        if not matched_user_cls:
                            for u_name, aliases in self.ALIAS_MAP.items():
                                if d_name in aliases:
                                    # Does this u_name match a user class?
                                    for c in class_hierarchy.classes:
                                        if c.name.lower() == u_name:
                                            matched_user_cls = c
                                            break
                                    if matched_user_cls: break
                        if matched_user_cls:
                            # Update ID to match user schema
                            det.class_id = matched_user_cls.id
                            det.class_name = matched_user_cls.name
                            relevant_dets.append(det)
                    
                    if relevant_dets:
                        # Start with YOLO's masks (already present from v26-seg)
                        # Refine with SAM2 for perfect quality
                        bboxes = [d.bbox for d in relevant_dets if d.bbox]
                        if bboxes:
                            sam_masks = self.model_manager.segment_with_sam2(image_rgb, bboxes)
                            for det, sam_mask in zip(relevant_dets, sam_masks):
                                if sam_mask.polygon:
                                    det.segmentation = sam_mask
                        
                        yolo_anns = relevant_dets
                        yolo_covered_classes = {a.class_name for a in yolo_anns}
        
        # 2. Open Vocabulary Stage (YOLO-World or CLIP)
        remaining_classes = user_class_names - yolo_covered_classes
        clip_target_classes = list(remaining_classes)
        
        clip_anns = []
        if clip_target_classes:
            self.logger.info(f"Hybrid Semantic: Processing {len(clip_target_classes)} open-vocab classes")
            
            # Try YOLO-World first (Faster & Better for prompts)
            yolo_world_covered_classes = set()
            if self.model_manager.is_model_loaded("yolo_world"):
                self.logger.info(f"Using YOLO-World for classes: {clip_target_classes}")
                try:
                    yolo_world_dets = self.model_manager.detect_objects_yolo_world(
                        image, 
                        classes=clip_target_classes,
                        conf_threshold=confidence_threshold
                    )
                    
                    # Map back to user classes and generate masks with SAM2
                    if yolo_world_dets:
                        bboxes = [d.bbox for d in yolo_world_dets if d.bbox]
                        # Only run SAM2 if we have detections
                        if bboxes:
                            sam_masks = self.model_manager.segment_with_sam2(image_rgb, bboxes)
                            
                            for det, sam_mask in zip(yolo_world_dets, sam_masks):
                                matched_user_cls = None
                                for u_name, u_cls in user_classes.items():
                                    if u_name.lower() == det.class_name.lower():
                                        matched_user_cls = u_cls
                                        break
                                
                                if matched_user_cls:
                                    det.class_id = matched_user_cls.id
                                    det.class_name = matched_user_cls.name
                                    
                                    if sam_mask.polygon:
                                        det.segmentation = sam_mask
                                        clip_anns.append(det)
                                        yolo_world_covered_classes.add(det.class_name)
                        
                except Exception as e:
                    self.logger.error(f"YOLO-World failed: {e}. Falling back to CLIP.")

            # Fallback to CLIP for classes NOT covered by YOLO-World
            remaining_semantic_classes = [c for c in clip_target_classes if c not in yolo_world_covered_classes]
            
            if remaining_semantic_classes and self.model_manager.is_model_loaded("clip"):
                self.logger.info(f"Using SAM2+CLIP for remaining classes: {remaining_semantic_classes}")
                
                # Generate all masks first
                raw_masks = self.model_manager.generate_automatic_masks_sam2(image_rgb)
                all_labels = remaining_semantic_classes + ["background", "noise", "other"]
                
                for mask_data in raw_masks:
                    # Get bbox
                    mask_bbox = mask_data["bbox"] # [x, y, w, h]
                    mask_seg = mask_data["segmentation"]
                    
                    # Crop image for CLIP
                    x, y, w, h = [int(v) for v in mask_bbox]
                    x, y = max(0, x), max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w <= 0 or h <= 0: 
                        continue
                        
                    crop = image[y:y+h, x:x+w]
                    
                    # Classify with CLIP
                    best_label, score = self.model_manager.classify_with_clip(crop, all_labels)
                    
                    # If match found in target classes
                    if best_label in clip_target_classes and score > 0.2:
                        matched_cls = user_classes[best_label]
                        polygon = self.model_manager._mask_to_polygon(mask_seg)
                        
                        if polygon:
                            ann = LabelAnnotation(
                                id=0, # set later
                                image_id="",
                                class_id=matched_cls.id,
                                class_name=matched_cls.name,
                                confidence=float(score),
                                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                                segmentation=SegmentationMask(polygon=polygon),
                                area=float(mask_data["area"])
                            )
                            clip_anns.append(ann)

        # 3. Merge
        # Check constraints? e.g. overlaps?
        # For now, just concat. YOLO results typically better spatially.
        # We might want to remove CLIPanns that overlap heavily with YOLOanns
        
        final_anns = yolo_anns.copy()
        
        for c_ann in clip_anns:
            # Check overlap with existing
            is_duplicate = False
            c_bbox = [c_ann.bbox.x, c_ann.bbox.y, c_ann.bbox.width, c_ann.bbox.height]
            
            for y_ann in yolo_anns:
                y_bbox = [y_ann.bbox.x, y_ann.bbox.y, y_ann.bbox.width, y_ann.bbox.height]
                iou = self._calculate_bbox_iou_simple(c_bbox, y_bbox)
                if iou > 0.7: # High overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_anns.append(c_ann)
                
        # Renumber IDs
        for i, ann in enumerate(final_anns):
            ann.id = i
            
        return final_anns

    def _save_results(self, job_id: str, results: List[ImageLabels]):
        """Save labeling results to disk."""
        output_path = settings.OUTPUT_DIR / f"{job_id}_results.json"
        
        data = {
            "job_id": job_id,
            "results": [
                {
                    "image_id": r.image_id,
                    "image_url": r.image_url,
                    "annotations": [a.model_dump() for a in r.annotations],
                    "status": r.status,
                    "processed_at": r.processed_at.isoformat() if r.processed_at else None,
                    "processing_time_ms": r.processing_time_ms
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
    
    def stop_job(self, job_id: str):
        """Stop a running job."""
        self._stop_flags[job_id] = True
        self.logger.info(f"Stop signal sent for job: {job_id}")
        
        # Also ensure job status in memory and on disk is updated to 'stopped' immediately
        job = self.get_job(job_id)
        if job and job.status == "running":
            job.status = "stopped"
            self._save_job_metadata(job)
            self.logger.info(f"Job {job_id} status updated to 'stopped'")
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files."""
        # First stop the job if running
        self.stop_job(job_id)
        
        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        # Remove from results
        if job_id in self.job_results:
            del self.job_results[job_id]
        
        # Remove stop flag
        if job_id in self._stop_flags:
            del self._stop_flags[job_id]
        
        # Delete result files from disk
        results_path = settings.OUTPUT_DIR / f"{job_id}_results.json"
        if results_path.exists():
            results_path.unlink()
            self.logger.info(f"Deleted results file: {results_path}")
        
        # Delete job metadata file
        job_meta_path = settings.OUTPUT_DIR / f"{job_id}_job.json"
        if job_meta_path.exists():
            job_meta_path.unlink()
            self.logger.info(f"Deleted job metadata: {job_meta_path}")
        
        self.logger.info(f"Job deleted: {job_id}")
        return True
    
    def get_all_jobs(self) -> List[LabelingJob]:
        """Get all labeling jobs."""
        return list(self.active_jobs.values())

    def get_job(self, job_id: str) -> Optional[LabelingJob]:
        """Get job information."""
        return self.active_jobs.get(job_id)

    def update_results(self, job_id: str, results: List[ImageLabels]):
        """Update and save labeling results with class ID normalization."""
        # 1. Get or load job
        job = self.get_job(job_id)
        if not job:
            self.logger.error(f"Cannot update results: Job {job_id} not found")
            return

        # 2. Normalize Classes: Build name -> ID map from hierarchy
        # This handles cases where frontend and backend IDs diverge
        hierarchy = job.class_hierarchy
        name_to_id = {c.name.lower(): c.id for c in hierarchy.classes}
        max_id = max([c.id for c in hierarchy.classes] + [0])
        
        new_classes_added = False
        for img_res in results:
            for ann in img_res.annotations:
                if not ann.class_name:
                    continue
                
                name_lower = ann.class_name.lower()
                if name_lower in name_to_id:
                    # Sync ID to hierarchy
                    ann.class_id = name_to_id[name_lower]
                else:
                    # New class found in results! Add to hierarchy
                    from ..models.schemas import ClassDefinition
                    max_id += 1
                    new_cls = ClassDefinition(id=max_id, name=ann.class_name)
                    hierarchy.classes.append(new_cls)
                    name_to_id[name_lower] = max_id
                    ann.class_id = max_id
                    new_classes_added = True
                    self.logger.info(f"UpdateResults: Auto-added new class '{ann.class_name}' (id={max_id}) to job {job_id}")

        # 3. Save the normalized results and metadata
        self.job_results[job_id] = results
        self._save_results(job_id, results)
        
        job.processed_images = len(results)
        if new_classes_added:
            job.class_hierarchy = hierarchy
            
        self._save_job_metadata(job)
        if new_classes_added:
            self.logger.info(f"Updated job {job_id} metadata with {len(hierarchy.classes)} total classes")
    
    async def refine_image(self, job_id: str, image_id: str, prompt: str) -> Optional[LabelAnnotation]:
        """
        Manually trigger AI refinement for a specific object on an image using a prompt.
        Uses YOLO-World or CLIP + SAM2.
        """
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        # Load results
        results = self.get_results(job_id)
        if results is None:
            return None
            
        # Find the specific image result
        img_result = next((r for r in results if r.image_id == image_id), None)
        if not img_result:
            return None
            
        # 1. Get Image Path from dataset metadata
        metadata_path = settings.UPLOAD_DIR / f"{job.dataset_id}_metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        img_info = next((i for i in metadata.get("images", []) if i["id"] == image_id), None)
        if not img_info:
            return None
            
        img_path = img_info["path"]
        image = cv2.imread(img_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # 2. Run Prompt-Based Detection (YOLO-World)
            self.logger.info(f"Refining image {image_id} with prompt: {prompt}")
            
            if "yolo_world" not in self.model_manager.models:
                self.model_manager.load_yolo_world()
                
            detections = self.model_manager.detect_objects_yolo_world(
                image, 
                classes=[prompt],
                conf_threshold=0.05 # Even lower for manual refinement
            )
            
            if not detections:
                self.logger.warning(f"YOLO-World found nothing for '{prompt}', refinement aborted.")
                return None
                
            # Take best detection
            best_det = detections[0]
            
            # 3. Map to existing class or create new if needed
            matched_cls = next((c for c in job.class_hierarchy.classes if c.name.lower() == prompt.lower()), None)
            if matched_cls:
                best_det.class_id = matched_cls.id
                best_det.class_name = matched_cls.name
            else:
                best_det.class_name = prompt
                best_det.class_id = 999 
            
            # 4. Generate Mask with SAM2
            if "sam2" not in self.model_manager.models:
                self.model_manager.load_sam2()
                
            masks = self.model_manager.segment_with_sam2(image_rgb, [best_det.bbox])
            if masks and masks[0].polygon:
                best_det.segmentation = masks[0]
            
            # 4. Integrate into results
            all_ann_ids = [a.id for r in results for a in r.annotations]
            new_id = max(all_ann_ids + [0]) + 1
            best_det.id = new_id
            best_det.image_id = image_id
            
            img_result.annotations.append(best_det)
            
            # 5. Save results
            self.update_results(job_id, results)
            
            return best_det
            
        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return None

    def get_results(self, job_id: str) -> Optional[List[ImageLabels]]:
        """Get labeling results for a job."""
        # First check memory
        if job_id in self.job_results:
            return self.job_results[job_id]
        
        # Try to load from disk
        output_path = settings.OUTPUT_DIR / f"{job_id}_results.json"
        if output_path.exists():
            with open(output_path) as f:
                data = json.load(f)
            
            results = []
            for r in data.get("results", []):
                annotations = [LabelAnnotation(**a) for a in r.get("annotations", [])]
                results.append(ImageLabels(
                    image_id=r["image_id"],
                    image_url=r.get("image_url"),
                    annotations=annotations,
                    status=r["status"],
                    processing_time_ms=r.get("processing_time_ms")
                ))
            
            self.job_results[job_id] = results
            return results
        
        return None
    
    async def stream_progress(self, job_id: str) -> AsyncGenerator[LabelingProgress, None]:
        """
        Stream progress updates for a job.
        
        Yields:
            LabelingProgress updates
        """
        job = self.active_jobs.get(job_id)
        if not job:
            yield LabelingProgress(job_id=job_id, status="not_found")
            return
        
        last_processed = 0
        
        while job.status == "running":
            if job.processed_images > last_processed:
                last_processed = job.processed_images
                
                progress = LabelingProgress(
                    job_id=job_id,
                    total_images=job.total_images,
                    processed_images=job.processed_images,
                    failed_images=job.failed_images,
                    current_image=job.progress.get("current_image"),
                    status=job.status
                )
                
                yield progress
            
            await asyncio.sleep(0.5)
        
        # Final progress
        yield LabelingProgress(
            job_id=job_id,
            total_images=job.total_images,
            processed_images=job.processed_images,
            failed_images=job.failed_images,
            status=job.status
        )


    def _save_job_metadata(self, job: LabelingJob):
        """Save job metadata to disk."""
        meta_path = settings.OUTPUT_DIR / f"job_{job.id}_meta.json"
        with open(meta_path, 'w') as f:
            f.write(job.model_dump_json(indent=2))

    def _load_jobs(self):
        """Load all jobs from disk."""
        if not settings.OUTPUT_DIR.exists():
            return
            
        for meta_file in settings.OUTPUT_DIR.glob("job_*_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    job = LabelingJob(**data)
                    self.active_jobs[job.id] = job
            except Exception as e:
                self.logger.error(f"Failed to load job from {meta_file}: {e}")

# Global labeling service instance
labeling_service = LabelingService()


def get_labeling_service() -> LabelingService:
    """Get labeling service instance."""
    return labeling_service
