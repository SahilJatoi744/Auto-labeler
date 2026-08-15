# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
AI Model management service.
Handles model loading, inference, and confidence scoring.
Supports YOLOv8, SAM, and other pretrained models.
"""

import gc
import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# For PyTorch 2.6+ security restrictions
# Monkeypatch torch.load to default weights_only=False for trusted local models
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import (
    BoundingBox, ImageInfo, LabelAnnotation, SegmentationMask, 
    TaskType, ModelInfo
)

logger = get_logger("model_manager")


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_at_least(version: Optional[str], minimum: str) -> bool:
    if not version:
        return False

    def parts(value: str) -> List[int]:
        numeric = []
        for part in value.split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            numeric.append(int(digits or 0))
        return numeric

    current = parts(version)
    target = parts(minimum)
    length = max(len(current), len(target))
    current.extend([0] * (length - len(current)))
    target.extend([0] * (length - len(target)))
    return current >= target


def _hf_cache_present(model_id: str, filename: str = "config.json") -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        return bool(try_to_load_from_cache(model_id, filename))
    except Exception:
        return False


class ModelManager:
    """
    Manages AI models for automatic labeling.
    Handles model loading, inference, and result post-processing.
    """
    
    def __init__(self):
        # Initialize logger first before any other operations
        self.logger = get_logger("model_manager")
        self.device = self._get_device()
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.device_preference = "auto"
        
        # Track loaded models
        self._loaded_models: set = set()
        
        # Track download status
        self.download_status: Dict[str, Dict[str, Any]] = {
            "yolo": {"status": "not_downloaded", "progress": 0, "error": None},
            "yolov26": {"status": "not_downloaded", "progress": 0, "error": None},
            "sam": {"status": "not_downloaded", "progress": 0, "error": None},
            "sam2": {"status": "not_downloaded", "progress": 0, "error": None}
        }
        
        # Check initial status
        self._check_initial_status()
    
    def is_model_loaded(self, model_key: str) -> bool:
        """Check if a model is loaded and ready."""
        return model_key in self._loaded_models

    def _check_initial_status(self):
        """Check if models are already on disk."""
        # Legacy models
        if (settings.MODELS_DIR / settings.YOLOV8_MODEL).exists():
            self.download_status["yolo"]["status"] = "ready"
            self.download_status["yolo"]["progress"] = 100
            
        if (settings.MODELS_DIR / settings.SAM_MODEL).exists():
            self.download_status["sam"]["status"] = "ready"
            self.download_status["sam"]["progress"] = 100
        
        # New models (YOLOv26 and SAM2)
        if (settings.MODELS_DIR / settings.YOLOV26_MODEL).exists():
            self.download_status["yolov26"]["status"] = "ready"
            self.download_status["yolov26"]["progress"] = 100
            
        if (settings.MODELS_DIR / settings.SAM2_MODEL).exists():
            self.download_status["sam2"]["status"] = "ready"
            self.download_status["sam2"]["progress"] = 100

    
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if settings.USE_GPU and torch.cuda.is_available():
            device = torch.device(f"cuda:{settings.GPU_DEVICE}")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return device
        else:
            self.logger.info("Using CPU for inference")
            return torch.device("cpu")
    
    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available."""
        if not torch.cuda.is_available():
            return None
        
        return {
            "name": torch.cuda.get_device_name(settings.GPU_DEVICE),
            "index": settings.GPU_DEVICE,
            "memory_total_gb": torch.cuda.get_device_properties(settings.GPU_DEVICE).total_memory / 1e9,
            "memory_allocated_gb": torch.cuda.memory_allocated(settings.GPU_DEVICE) / 1e9,
            "memory_cached_gb": torch.cuda.memory_reserved(settings.GPU_DEVICE) / 1e9,
        }

    def set_device(self, device: str = "auto"):
        """
        Set the inference device (GPU/CPU) dynamically.
        Reloads models if device changes.
        
        Args:
            device: 'cuda', 'gpu', 'cpu', or 'auto' (auto-detect)
        """
        self.device_preference = device
        
        # Handle various device formats
        if isinstance(device, bool):
            # Legacy boolean support
            device = "cuda" if device else "cpu"
        elif device.lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.lower() == "gpu":
            device = "cuda"
        else:
            device = device.lower()
        
        new_device = torch.device(f"cuda:{settings.GPU_DEVICE}") if (device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
        
        if new_device != self.device:
            self.logger.info(f"Switching device from {self.device} to {new_device}")
            self.device = new_device
            
            # Move loaded models to new device
            for name, model in self.models.items():
                if hasattr(model, 'to'):
                    try:
                        model.to(self.device)
                        self.logger.info(f"Moved {name} to {self.device}")
                    except Exception as e:
                        self.logger.warning(f"Could not move {name} to {self.device}: {e}")
                
                # Check for SAM predictor which wraps a model
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    try:
                        model.model.to(self.device)
                    except:
                        pass
    
    def load_yolo(self, model_name: Optional[str] = None, target_key: str = "yolo") -> Any:
        """
        Load YOLOv8 model for object detection.
        
        Args:
            model_name: YOLO model variant (defaults to settings.YOLOV8_MODEL)
            target_key: Key to store the model under (default: "yolo")
            
        Returns:
            Loaded YOLO model
        """
        try:
            from ultralytics import YOLO
            
            # Use default from settings if not provided
            model_name = model_name or settings.YOLOV8_MODEL
            
            model_path = settings.MODELS_DIR / model_name
            
            # Download if not exists
            if not model_path.exists():
                self.logger.info(f"Downloading YOLO model: {model_name}")
                status_key = "yolov26" if target_key == "yolov26" else "yolo"
                self.download_model_task(status_key)
                if self.download_status[status_key]["status"] == "error":
                    raise Exception(self.download_status[status_key]["error"])
            
            self.logger.info(f"Loading YOLO model from: {model_path} (key: {target_key})")
            model = YOLO(str(model_path))
            
            # Move to device
            model.to(self.device)
            
            self.models[target_key] = model
            self._loaded_models.add(target_key)
            
            model_display_name = "YOLOv26" if target_key == "yolov26" else "YOLOv8"
            self.model_info[target_key] = ModelInfo(
                name=model_display_name,
                version="8.0",
                task_types=[TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION],
                description=f"Ultralytics {model_display_name} for object detection and instance segmentation",
                is_downloaded=True
            )
            
            self.logger.info(f"YOLO model loaded on {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def load_yolov26(self, model_name: Optional[str] = None, for_segmentation: bool = True) -> Any:
        """
        Load latest YOLO model for object detection or instance segmentation.
        Attempts YOLOv26 first, falls back to YOLOv11 or YOLOv8 if unavailable.
        
        Args:
            model_name: YOLO model variant (auto-detects best available)
            for_segmentation: True for segmentation model, False for detection only
            
        Returns:
            Loaded YOLO model
        """
        try:
            from ultralytics import YOLO
            import ultralytics
            
            version = getattr(ultralytics, '__version__', '0.0.0')
            version_parts = [int(x) for x in version.split('.')[:3]]
            version_num = version_parts[0] * 10000 + version_parts[1] * 100 + version_parts[2]
            
            # Determine best available model based on ultralytics version
            if model_name is None:
                if version_num >= 80400:  # 8.4.0+ supports YOLOv26
                    model_name = "yolo26x-seg.pt" if for_segmentation else "yolo26x.pt"
                    model_version = "26.0"
                elif version_num >= 80300:  # 8.3.0+ supports YOLOv11
                    model_name = "yolo11x-seg.pt" if for_segmentation else "yolo11x.pt"
                    model_version = "11.0"
                else:  # Fallback to YOLOv8
                    model_name = "yolov8x-seg.pt" if for_segmentation else "yolov8x.pt"
                    model_version = "8.0"
            else:
                # Extract version from model name
                if 'v26' in model_name or 'yolo26' in model_name:
                    model_version = "26.0"
                elif 'v11' in model_name or 'yolo11' in model_name:
                    model_version = "11.0"
                else:
                    model_version = "8.0"
            
            self.logger.info(f"Loading YOLO model: {model_name} (ultralytics v{version})")
            
            # Check if model exists in MODELS_DIR
            model_path = settings.MODELS_DIR / model_name
            if model_path.exists():
                self.logger.info(f"Found local model at {model_path}")
                model = YOLO(str(model_path))
            else:
                self.logger.info(f"Model not found locally, downloading {model_name}...")
                # ultralytics will automatically download to current dir
                model = YOLO(model_name)
                
                # Try to move to models dir for future use
                try:
                    import shutil
                    local_file = Path(model_name)
                    if local_file.exists():
                        shutil.move(str(local_file), str(model_path))
                        self.logger.info(f"Moved downloaded model to {model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to move model to models dir: {e}")
            
            # Move to device
            model.to(self.device)
            
            self.models["yolov26"] = model  # Keep key for compatibility
            self._loaded_models.add("yolov26")
            self.download_status["yolov26"]["status"] = "ready"
            self.download_status["yolov26"]["progress"] = 100
            
            self.model_info["yolov26"] = ModelInfo(
                name=f"YOLO v{model_version.split('.')[0]}",
                version=model_version,
                task_types=[TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION],
                description=f"Ultralytics YOLO for detection and instance segmentation",
                is_downloaded=True
            )
            
            self.logger.info(f"YOLO model loaded on {self.device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.download_status["yolov26"]["status"] = "error"
            self.download_status["yolov26"]["error"] = str(e)
            raise

    def load_yolo_world(self, model_name: str = "yolov8s-world.pt") -> Any:
        """
        Load YOLO-World model for open-vocabulary detection (Prompt based).
        """
        try:
            from ultralytics import YOLO
            
            # Check if model exists locally
            model_path = settings.MODELS_DIR / model_name
            if model_path.exists():
                self.logger.info(f"Found local YOLO-World model at {model_path}")
                model = YOLO(str(model_path))
            else:
                self.logger.info(f"Downloading YOLO-World model: {model_name}")
                model = YOLO(model_name)
                # Move to models dir
                try:
                    import shutil
                    local_file = Path(model_name)
                    if local_file.exists():
                        shutil.move(str(local_file), str(model_path))
                except Exception as e:
                    self.logger.warning(f"Failed to move model: {e}")
            
            model.to(self.device)
            
            self.models["yolo_world"] = model
            self._loaded_models.add("yolo_world")
            
            self.model_info["yolo_world"] = ModelInfo(
                name="YOLO-World",
                version="8.0",
                task_types=[TaskType.OBJECT_DETECTION],
                description="Real-time Open-Vocabulary Object Detection",
                is_downloaded=True
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO-World: {e}")
            # The original instruction had a typo here (raiself.download_status...).
            # Correcting to a syntactically valid and logical error handling.
            # Assuming a similar download_status mechanism for yolo_world.
            if "yolo_world" in self.download_status:
                self.download_status["yolo_world"]["status"] = "error"
                self.download_status["yolo_world"]["error"] = str(e)
            raise

    def load_clip(self):
        """Load CLIP model for zero-shot classification."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.logger.info("Loading CLIP model (openai/clip-vit-base-patch32)...")
            
            # Use a standard stable CLIP model
            model_name = "openai/clip-vit-base-patch32"
            
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            model.to(self.device)
            
            self.models["clip_processor"] = processor
            self.models["clip_model"] = model
            self._loaded_models.add("clip")
            
            self.model_info["clip"] = ModelInfo(
                name="CLIP",
                version="ViT-B/32",
                task_types=[TaskType.SEMANTIC_SEGMENTATION],
                description="OpenAI CLIP for zero-shot classification",
                is_downloaded=True
            )
            
            self.logger.info(f"CLIP model loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise

    def classify_with_clip(self, image: np.ndarray, candidate_labels: List[str]) -> Tuple[str, float]:
        """
        Classify an image crop using CLIP.
        
        Args:
            image: Image crop (numpy array, BGR or RGB)
            candidate_labels: List of class names to check
            
        Returns:
            Tuple of (best_label, confidence_score)
        """
        if "clip_model" not in self.models:
            self.load_clip()
            
        model = self.models["clip_model"]
        processor = self.models["clip_processor"]
        
        # Ensure image is RGB (PIL usually expects RGB)
        # Check if already RGB or BGR. If from cv2 it's BGR.
        # Minimal check: assume standard CV2 usage -> BGR
        import cv2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        inputs = processor(
            text=candidate_labels, 
            images=image_rgb, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        
        # Get best match
        best_idx = probs.argmax().item()
        best_score = probs[0, best_idx].item()
        best_label = candidate_labels[best_idx]
        
        return best_label, best_score

    
    def load_sam(self, model_type: str = "vit_h") -> Any:
        """
        Load Segment Anything Model (SAM) for segmentation.
        
        Args:
            model_type: SAM model variant (vit_h, vit_l, vit_b)
            
        Returns:
            Loaded SAM model
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            
            # Check if already loading or ready
            if self.download_status["sam"]["status"] == "downloading":
                self.logger.info("SAM model is already downloading...")
                return None

            model_file = f"sam_{model_type}.pth"
            model_path = settings.MODELS_DIR / model_file
            
            # Download if not exists
            if not model_path.exists():
                self.logger.info(f"Downloading SAM model: {model_type}")
                self.download_model_task("sam")
                if self.download_status["sam"]["status"] == "error":
                    raise Exception(self.download_status["sam"]["error"])
            
            self.logger.info(f"Loading SAM model from: {model_path}")
            sam = sam_model_registry[model_type](checkpoint=str(model_path))
            sam.to(device=self.device)
            
            predictor = SamPredictor(sam)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,            # Higher for better coverage
                pred_iou_thresh=0.88,          # Higher for quality
                stability_score_thresh=0.95,   # Higher for stability
                crop_n_layers=1,               # Multi-layer cropping for small objects
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,       # Filter tiny artifacts
            )
            
            self.models["sam"] = predictor
            self.models["sam_generator"] = mask_generator
            self._loaded_models.add("sam")
            
            self.model_info["sam"] = ModelInfo(
                name="SAM",
                version="1.0",
                task_types=[TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION],
                description="Meta AI Segment Anything Model",
                is_downloaded=True
            )
            
            self.logger.info(f"SAM model loaded on {self.device}")
            return predictor
            
        except Exception as e:
            self.logger.error(f"Failed to load SAM model: {e}")
            raise

    def load_sam2(self, model_name: Optional[str] = None) -> Any:
        """
        Load Segment Anything Model for high-quality segmentation.
        Attempts SAM2 via ultralytics, falls back to SAM1 via segment-anything.
        
        Args:
            model_name: SAM model variant (auto-detects best available)
            
        Returns:
            SAM model object
        """
        # Check if already loading
        if self.download_status["sam2"]["status"] == "downloading":
            self.logger.info("SAM model is already downloading...")
            return None
        
        # First, try using ultralytics SAM wrapper
        try:
            from ultralytics import SAM
            import ultralytics
            
            version = getattr(ultralytics, '__version__', '0.0.0')
            
            # Determine best available SAM model
            if model_name is None:
                # Use configured model
                model_name = settings.SAM2_MODEL
            
            self.logger.info(f"Loading SAM2 model: {model_name} (Ultralytics v{version})")
            
            # Check if model exists in MODELS_DIR
            model_path = settings.MODELS_DIR / model_name
            if model_path.exists():
                self.logger.info(f"Found local SAM2 model at {model_path}")
                sam_model = SAM(str(model_path))
            else:
                self.logger.info(f"Model not found locally, downloading {model_name}...")
                sam_model = SAM(model_name)
                
                # Try to move to models dir for future use
                try:
                    import shutil
                    local_file = Path(model_name)
                    if local_file.exists():
                        shutil.move(str(local_file), str(model_path))
                        self.logger.info(f"Moved downloaded model to {model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to move model to models dir: {e}")
            sam_model.to(self.device)
            
            self.models["sam2"] = sam_model
            self._loaded_models.add("sam2")
            self.download_status["sam2"]["status"] = "ready"
            self.download_status["sam2"]["progress"] = 100
            
            self.model_info["sam2"] = ModelInfo(
                name="SAM",
                version="1.0",
                task_types=[TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION],
                description="Meta AI Segment Anything Model via ultralytics",
                is_downloaded=True
            )
            
            self.logger.info(f"SAM model loaded on {self.device}")
            return sam_model
            
        except Exception as e:
            self.logger.warning(f"Ultralytics SAM failed: {e}, falling back to segment-anything")
        
        # Fallback: Use segment-anything library (SAM1)
        try:
            # Load SAM1 using the existing load_sam method
            self.logger.info("Loading SAM via segment-anything library...")
            predictor = self.load_sam()
            
            # Wrap the predictor to work with our sam2 interface
            self.models["sam2"] = predictor  # sam predictor
            self._loaded_models.add("sam2")
            self.download_status["sam2"]["status"] = "ready"
            self.download_status["sam2"]["progress"] = 100
            
            self.model_info["sam2"] = ModelInfo(
                name="SAM1",
                version="1.0",
                task_types=[TaskType.SEMANTIC_SEGMENTATION, TaskType.INSTANCE_SEGMENTATION],
                description="Meta AI Segment Anything Model (SAM1)",
                is_downloaded=True
            )
            
            return predictor
            
        except Exception as e2:
            self.logger.error(f"Failed to load SAM model: {e2}")
            self.download_status["sam2"]["status"] = "error"
            self.download_status["sam2"]["error"] = str(e2)
            raise
    
    def generate_automatic_masks(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for the entire image using SAM1.
        """
        if "sam_generator" not in self.models:
            self.load_sam()
            
        generator = self.models["sam_generator"]
        return generator.generate(image)

    def generate_automatic_masks_sam2(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for the entire image using SAM2.
        """
        if "sam2" not in self.models:
            self.load_sam2()
            
        sam_model = self.models["sam2"]
        
        # Check if it's an ultralytics model
        if hasattr(sam_model, 'info'):
             # Ultralytics SAM2 segment everything
             results = sam_model(image, verbose=False)
             masks_list = []
             if results and results[0].masks is not None:
                for i, mask in enumerate(results[0].masks.data):
                    masks_list.append({
                        "segmentation": mask.cpu().numpy(),
                        "bbox": results[0].boxes.xywh[i].cpu().numpy(),
                        "area": float(mask.sum()),
                        "point_coords": None,
                        "predicted_iou": float(results[0].boxes.conf[i].cpu().numpy())
                    })
             return masks_list
        else:
             # Fallback to SAM1 generator
             return self.generate_automatic_masks(image)
    
    def detect_objects_yolo(
        self,
        image: np.ndarray,
        conf_threshold: float = None,
        iou_threshold: float = None,
        class_filter: Optional[List[int]] = None
    ) -> List[LabelAnnotation]:
        """
        Run object detection with YOLO.
        
        Args:
            image: Input image (BGR format from cv2)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            class_filter: Optional list of class IDs to keep
            
        Returns:
            List of label annotations
        """
        if "yolo" not in self.models:
            self.load_yolo()
        
        conf = conf_threshold or settings.DEFAULT_CONFIDENCE_THRESHOLD
        iou = iou_threshold or settings.NMS_IOU_THRESHOLD
        
        model = self.models["yolo"]
        
        # Run inference
        results = model(
            image,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        
        annotations = []
        
        for result in results:
            boxes = result.boxes
            masks = getattr(result, 'masks', None)
            
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by class if specified
                if class_filter and class_id not in class_filter:
                    continue
                
                # Get class name from model
                # Get class name from model (robust to list or dict)
                if isinstance(result.names, dict):
                    class_name = result.names.get(class_id, f"class_{class_id}")
                elif isinstance(result.names, (list, tuple)) and class_id < len(result.names):
                    class_name = result.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                annotation = LabelAnnotation(
                    id=i,
                    image_id="",  # Will be set by caller
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1)
                    ),
                    area=float((x2 - x1) * (y2 - y1))
                )
                
                # Add segmentation mask if provided by YOLOv8-Seg
                if masks is not None:
                    try:
                        mask_data = masks.data[i].cpu().numpy()
                        polygon = self._mask_to_polygon(mask_data)
                        if polygon:
                            annotation.segmentation = SegmentationMask(polygon=polygon)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract mask for detection {i}: {e}")
                
                annotations.append(annotation)
        
        return annotations
    
    def segment_with_sam(
        self,
        image: np.ndarray,
        bounding_boxes: List[BoundingBox],
        multimask_output: bool = True  # Changed default to True for better quality
    ) -> List[SegmentationMask]:
        """
        Generate segmentation masks using SAM from bounding boxes.
        Uses multi-mask output and center point prompting for higher accuracy.
        
        Args:
            image: Input image (RGB format)
            bounding_boxes: List of bounding boxes
            multimask_output: Whether to output multiple masks per box
            
        Returns:
            List of segmentation masks
        """
        if "sam" not in self.models:
            self.load_sam()
        
        predictor = self.models["sam"]
        
        # Set image
        predictor.set_image(image)
        
        masks_list = []
        
        for bbox in bounding_boxes:
            # Convert bbox to SAM input format [x1, y1, x2, y2]
            input_box = np.array([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
            
            # Add center point as additional prompt for better guidance
            center_x = bbox.x + bbox.width / 2
            center_y = bbox.y + bbox.height / 2
            center_point = np.array([[center_x, center_y]])
            center_label = np.array([1])  # 1 = foreground
            
            # First pass: get initial mask with multi-mask output
            masks, scores, logits = predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                box=input_box[None, :],
                multimask_output=True  # Always get multiple masks first
            )
            
            # Get the best mask from initial prediction
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_logits = logits[best_mask_idx:best_mask_idx+1, :, :]  # Keep dims for refinement
            
            # Second pass: refine the best mask for higher quality
            refined_masks, refined_scores, _ = predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                box=input_box[None, :],
                mask_input=best_logits,  # Use logits from first pass for refinement
                multimask_output=False  # Single refined output
            )
            
            final_mask = refined_masks[0]
            
            # Convert mask to polygon
            polygon = self._mask_to_polygon(final_mask)
            
            mask_obj = SegmentationMask(
                polygon=polygon if polygon else None,
                mask_path=None
            )
            
            masks_list.append(mask_obj)
        
        # Clear image embedding to save memory
        predictor.reset_image()
        
        return masks_list

    def detect_objects_yolov26(
        self,
        image: np.ndarray,
        conf_threshold: float = None,
        class_filter: Optional[List[int]] = None
    ) -> List[LabelAnnotation]:
        """
        Run object detection/instance segmentation with YOLOv26.
        YOLOv26 is NMS-free with end-to-end predictions.
        
        Args:
            image: Input image (BGR format from cv2)
            conf_threshold: Confidence threshold
            class_filter: Optional list of class IDs to keep
            
        Returns:
            List of label annotations with polygons (if using seg model)
        """
        if "yolov26" not in self.models:
            self.load_yolov26()
        
        conf = conf_threshold or settings.DEFAULT_CONFIDENCE_THRESHOLD
        
        model = self.models["yolov26"]
        
        # Run inference - YOLOv26 doesn't need NMS (end2end=True by default)
        results = model(
            image,
            conf=conf,
            verbose=False,
        )
        
        annotations = []
        
        for result in results:
            boxes = result.boxes
            masks = getattr(result, 'masks', None)
            
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter by class if specified
                if class_filter and class_id not in class_filter:
                    continue
                
                # Get class name from model
                # Get class name from model (robust to list or dict)
                if isinstance(result.names, dict):
                    class_name = result.names.get(class_id, f"class_{class_id}")
                elif isinstance(result.names, (list, tuple)) and class_id < len(result.names):
                    class_name = result.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                annotation = LabelAnnotation(
                    id=i,
                    image_id="",  # Will be set by caller
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1)
                    ),
                    area=float((x2 - x1) * (y2 - y1))
                )
                
                # Add segmentation mask if provided by YOLOv26-Seg
                if masks is not None:
                    try:
                        mask_data = masks.data[i].cpu().numpy()
                        polygon = self._mask_to_polygon(mask_data)
                        if polygon:
                            annotation.segmentation = SegmentationMask(polygon=polygon)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract mask for detection {i}: {e}")
                
                annotations.append(annotation)
        
        self.logger.info(f"YOLOv26 detected {len(annotations)} objects")
        return annotations

    def segment_with_sam2(
        self,
        image: np.ndarray,
        bounding_boxes: List[BoundingBox]
    ) -> List[SegmentationMask]:
        """
        Generate high-quality segmentation masks using SAM from bounding boxes.
        Automatically handles both ultralytics SAM and segment-anything SAM1.
        
        Args:
            image: Input image (RGB format)
            bounding_boxes: List of bounding boxes
        Segment objects using SAM2.
        
        Args:
            image: RGB image array
            bounding_boxes: List of bounding boxes to segment
            
        Returns:
            List of SegmentationMask objects
        """
        if not bounding_boxes:
            return []
            
        if "sam2" not in self.models:
            self.load_sam2()
            
        sam_model = self.models["sam2"]
        masks_list = []
        
        try:
            # Convert BoundingBoxes to xyxy format
            input_boxes = [[b.x, b.y, b.x + b.width, b.y + b.height] for b in bounding_boxes]
            
            # Use vectorized approach if possible
            results = sam_model(
                image,
                bboxes=input_boxes,
                verbose=False
            )
            
            if results and hasattr(results[0], 'masks') and results[0].masks is not None:
                # Ensure we iterate over the masks corresponding to the input boxes
                # Ultralytics SAM returns masks in the order of input boxes
                for mask_data in results[0].masks.data:
                    mask_np = mask_data.cpu().numpy()
                    polygon = self._mask_to_polygon(mask_np)
                    
                    masks_list.append(SegmentationMask(
                        polygon=polygon if polygon else None,
                        mask_path=None
                    ))
            
            # Pad if missing some (though Ultralytics should return one per box if successful)
            while len(masks_list) < len(bounding_boxes):
                masks_list.append(SegmentationMask(polygon=None))
                    
        except Exception as e:
            self.logger.error(f"SAM2 segmentation failed: {e}. Falling back to SAM1 logic per box.")
            # Individual fallback
            for bbox in bounding_boxes:
                try:
                    ib = [[bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height]]
                    res = sam_model(image, bboxes=ib, verbose=False)
                    if res and res[0].masks is not None:
                        md = res[0].masks.data[0].cpu().numpy()
                        poly = self._mask_to_polygon(md)
                        masks_list.append(SegmentationMask(polygon=poly))
                    else:
                        masks_list.append(SegmentationMask(polygon=None))
                except:
                    masks_list.append(SegmentationMask(polygon=None))
                    
        self.logger.info(f"SAM2 generated {len(masks_list)} masks for {len(bounding_boxes)} boxes")
        return masks_list

    def detect_objects_yolo_world(
        self,
        image: np.ndarray,
        classes: List[str],
        conf_threshold: float = None,
        iou_threshold: float = None
    ) -> List[LabelAnnotation]:
        """
        Run object detection with YOLO-World using text prompts.
        """
        if "yolo_world" not in self.models:
            self.load_yolo_world()
        
        model = self.models["yolo_world"]
        
        # Set classes (prompts)
        # Note: YOLO-World set_classes method might vary by version, 
        # but typically it's model.set_classes(["class1", "class2"])
        try:
            model.set_classes(classes)
        except Exception as e:
            self.logger.warning(f"Failed to set classes for YOLO-World: {e}. Running without explicit set.")
        
        conf = conf_threshold or settings.DEFAULT_CONFIDENCE_THRESHOLD
        iou = iou_threshold or settings.NMS_IOU_THRESHOLD
        
        # Run inference
        results = model(
            image,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        
        annotations = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name from model (robust to list or dict)
                if isinstance(result.names, dict):
                    class_name = result.names.get(class_id, f"class_{class_id}")
                elif isinstance(result.names, (list, tuple)) and class_id < len(result.names):
                    class_name = result.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                annotation = LabelAnnotation(
                    id=i,
                    image_id="",
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1)
                    ),
                    area=float((x2 - x1) * (y2 - y1))
                )
                annotations.append(annotation)
        
        return annotations

    def generate_automatic_masks_sam2(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate masks for the entire image using SAM.
        Works with both ultralytics SAM and segment-anything SAM1.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            List of mask dictionaries with bbox, segmentation, and area
        """
        if "sam2" not in self.models:
            self.load_sam2()
        
        sam_model = self.models["sam2"]
        
        # Check if this is SAM1 (segment-anything) with generator available
        is_sam1 = hasattr(sam_model, 'set_image') or "sam_generator" in self.models
        
        if is_sam1 or "sam_generator" in self.models:
            # Use SAM1 automatic mask generator
            return self.generate_automatic_masks(image)
        
        # Use ultralytics SAM (automatic mode)
        results = sam_model(image, verbose=False)
        
        masks_data = []
        for result in results:
            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()
                    
                    coords = np.where(mask_np > 0)
                    if len(coords[0]) == 0:
                        continue
                    
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    masks_data.append({
                        "segmentation": mask_np,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": float(mask_np.sum())
                    })
        
        self.logger.info(f"SAM auto-generated {len(masks_data)} masks")
        return masks_data
    
    def _mask_to_polygon(self, mask: np.ndarray, simplify_factor: float = 0.002) -> Optional[List[List[float]]]:
        """
        Convert binary mask to polygon points with morphological cleaning.
        
        Args:
            mask: Binary mask array
            simplify_factor: How much to simplify the contour (lower = more detail)
            
        Returns:
            List of polygon contours
        """
        # Ensure mask is uint8
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close small holes
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        # Remove small noise
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,  # Only external contours for cleaner output
            cv2.CHAIN_APPROX_TC89_L1  # Better approximation than SIMPLE
        )
        
        polygons = []
        for contour in contours:
            # Filter very small contours (noise)
            if cv2.contourArea(contour) < 50:
                continue
            
            if len(contour) >= 3:  # Need at least 3 points for polygon
                # Adaptive simplification based on contour size
                epsilon = simplify_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Ensure we still have enough points after simplification
                if len(approx) >= 3:
                    # Convert to flat list [x1, y1, x2, y2, ...]
                    points = approx.reshape(-1, 2).tolist()
                    polygons.append([[float(p[0]), float(p[1])] for p in points])
        
        return polygons if polygons else None
    
    def calculate_confidence_scores(
        self,
        annotations: List[LabelAnnotation],
        method: str = "softmax"
    ) -> List[float]:
        """
        Calculate calibrated confidence scores.
        
        Args:
            annotations: List of annotations
            method: Calibration method (softmax, temperature, none)
            
        Returns:
            List of calibrated confidence scores
        """
        if not annotations:
            return []
        
        confidences = [ann.confidence for ann in annotations]
        
        if method == "softmax":
            # Apply temperature-scaled softmax
            temperature = 1.5  # Higher temperature = softer distribution
            conf_tensor = torch.tensor(confidences) / temperature
            calibrated = F.softmax(conf_tensor, dim=0).numpy()
            return calibrated.tolist()
        
        elif method == "temperature":
            temperature = 1.2
            return [min(1.0, conf / temperature) for conf in confidences]
        
        else:  # none
            return confidences
    
    def estimate_uncertainty(
        self,
        annotation: LabelAnnotation,
        all_annotations: List[LabelAnnotation]
    ) -> Dict[str, float]:
        """
        Estimate uncertainty for an annotation.
        
        Args:
            annotation: Single annotation
            all_annotations: All annotations in image (for context)
            
        Returns:
            Dictionary of uncertainty metrics
        """
        uncertainties = {
            "confidence_uncertainty": 1.0 - annotation.confidence,
            "overlap_uncertainty": 0.0,
            "size_uncertainty": 0.0
        }
        
        # Check for overlapping boxes (high overlap = uncertainty)
        if annotation.bbox:
            for other in all_annotations:
                if other.id == annotation.id:
                    continue
                if other.bbox:
                    iou = self._calculate_iou(annotation.bbox, other.bbox)
                    if iou > 0.3:  # Significant overlap
                        uncertainties["overlap_uncertainty"] = max(
                            uncertainties["overlap_uncertainty"],
                            iou
                        )
        
        # Size-based uncertainty (very small or very large objects)
        if annotation.bbox:
            area = annotation.bbox.area
            img_area = 1000000  # Approximate, should be actual image size
            area_ratio = area / img_area
            
            if area_ratio < 0.001:  # Very small
                uncertainties["size_uncertainty"] = 0.5
            elif area_ratio > 0.5:  # Very large
                uncertainties["size_uncertainty"] = 0.3
        
        # Overall uncertainty score
        uncertainties["total_uncertainty"] = (
            uncertainties["confidence_uncertainty"] * 0.5 +
            uncertainties["overlap_uncertainty"] * 0.3 +
            uncertainties["size_uncertainty"] * 0.2
        )
        
        return uncertainties
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def download_model_task(self, model_name: str):
        """
        Download a model in the background.
        This is a blocking call intended to be run in a thread.
        """
        try:
            model_name = model_name.lower()
            
            # Map model names to their status keys
            if model_name in ["yolo", "yolov8"]:
                status_key = "yolo"
            elif model_name in ["yolov26", "yolo26"]:
                status_key = "yolov26"
            elif model_name in ["sam", "sam1"]:
                status_key = "sam"
            elif model_name == "sam2":
                status_key = "sam2"
            else:
                self.logger.error(f"Unknown model: {model_name}")
                return
            
            if model_name in ["yolo", "yolov8"]:
                self.download_status["yolo"]["status"] = "downloading"
                self.download_status["yolo"]["progress"] = 10
                from ultralytics import YOLO
                # YOLO constructor with model name automatically downloads if not local
                _ = YOLO(settings.YOLOV8_MODEL)
                model_path = settings.MODELS_DIR / settings.YOLOV8_MODEL
                
                # If YOLO downloaded to CWD, move it to models dir
                cwd_path = Path(settings.YOLOV8_MODEL)
                if cwd_path.exists() and cwd_path != model_path:
                    import shutil
                    shutil.move(str(cwd_path), str(model_path))
                self.download_status["yolo"]["status"] = "ready"
                self.download_status["yolo"]["progress"] = 100
                
            elif model_name in ["yolov26", "yolo26"]:
                self.download_status["yolov26"]["status"] = "downloading"
                self.download_status["yolov26"]["progress"] = 10
                from ultralytics import YOLO
                # Download YOLOv26 model (YOLO auto-downloads from Ultralytics hub)
                model_variant = getattr(settings, 'YOLOV26_MODEL', 'yolov8x-seg.pt')
                _ = YOLO(model_variant)
                model_path = settings.MODELS_DIR / model_variant
                
                # If downloaded to CWD, move to models dir
                cwd_path = Path(model_variant)
                if cwd_path.exists() and cwd_path != model_path:
                    import shutil
                    shutil.move(str(cwd_path), str(model_path))
                self.download_status["yolov26"]["status"] = "ready"
                self.download_status["yolov26"]["progress"] = 100
                self.logger.info(f"YOLOv26 model downloaded: {model_variant}")
                
            elif model_name in ["sam", "sam1"]:
                self.download_status["sam"]["status"] = "downloading"
                self.download_status["sam"]["progress"] = 0
                model_path = settings.MODELS_DIR / settings.SAM_MODEL
                
                import urllib.request
                
                def report_progress(block_num, block_size, total_size):
                    readsofar = block_num * block_size
                    if total_size > 0:
                        percent = min(100, int(readsofar * 100 / total_size))
                        self.download_status["sam"]["progress"] = percent
                
                urllib.request.urlretrieve(
                    settings.SAM_MODEL_URL, 
                    str(model_path), 
                    reporthook=report_progress
                )
                
                self.download_status["sam"]["status"] = "ready"
                self.download_status["sam"]["progress"] = 100
                
            elif model_name == "sam2":
                self.download_status["sam2"]["status"] = "downloading"
                self.download_status["sam2"]["progress"] = 10
                
                # SAM2 is downloaded via ultralytics.SAM
                from ultralytics import SAM
                model_variant = getattr(settings, 'SAM2_MODEL', 'sam2_b.pt')
                _ = SAM(model_variant)
                
                self.download_status["sam2"]["status"] = "ready"
                self.download_status["sam2"]["progress"] = 100
                self.logger.info(f"SAM2 model downloaded: {model_variant}")
                
        except Exception as e:
            self.logger.error(f"Download failed for {model_name}: {e}")
            if model_name in self.download_status:
                self.download_status[model_name]["status"] = "error"
                self.download_status[model_name]["error"] = str(e)

    def get_download_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the current download status of all managed models."""
        return self.download_status

    def get_advanced_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Report optional local integration readiness without loading models."""
        ultralytics_version = _package_version("ultralytics")
        transformers_version = _package_version("transformers")
        sam3_path = settings.MODELS_DIR / settings.SAM3_MODEL

        sam3_package_ready = _version_at_least(ultralytics_version, "8.3.237")
        transformers_ready = transformers_version is not None
        grounding_cache_present = _hf_cache_present(settings.GROUNDING_DINO_MODEL_ID)
        dinov3_cache_present = _hf_cache_present(settings.DINOV3_MODEL_ID)

        return {
            "sam3": {
                "ready": bool(sam3_package_ready and sam3_path.exists()),
                "package": "ultralytics",
                "package_version": ultralytics_version,
                "minimum_version": "8.3.237",
                "package_ready": sam3_package_ready,
                "weight_file": settings.SAM3_MODEL,
                "weight_path": str(sam3_path),
                "weight_present": sam3_path.exists(),
                "install_hint": "pip install -U ultralytics && place approved sam3.pt in backend/models",
                "local_only": not settings.ALLOW_MODEL_DOWNLOADS,
            },
            "grounding_dino": {
                "ready": bool(transformers_ready and grounding_cache_present),
                "package": "transformers",
                "package_version": transformers_version,
                "model_id": settings.GROUNDING_DINO_MODEL_ID,
                "package_ready": transformers_ready,
                "cache_present": grounding_cache_present,
                "install_hint": "pip install -U transformers accelerate safetensors; pre-cache the Grounding DINO model for offline use",
                "local_only": not settings.ALLOW_MODEL_DOWNLOADS,
            },
            "dinov3": {
                "ready": bool(transformers_ready and dinov3_cache_present),
                "package": "transformers",
                "package_version": transformers_version,
                "model_id": settings.DINOV3_MODEL_ID,
                "package_ready": transformers_ready,
                "cache_present": dinov3_cache_present,
                "install_hint": "pip install -U transformers accelerate safetensors; pre-cache the DINOv3 model for offline feature extraction",
                "local_only": not settings.ALLOW_MODEL_DOWNLOADS,
            },
        }

    def prepare_advanced_model(
        self,
        model_key: str,
        allow_download: bool = False,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prepare an optional advanced model and return an actionable status.

        This intentionally avoids installing Python packages from inside the API
        process. It can verify readiness, load local models, and download gated
        weights only when the required package/token/access already exist.
        """
        model_key = model_key.lower().replace("-", "_")
        status = self.get_advanced_model_status()
        if model_key not in status:
            return {
                "model_key": model_key,
                "status": "blocked",
                "message": f"Unknown advanced model integration: {model_key}",
                "next_steps": ["Use one of: sam3, grounding_dino, dinov3."],
            }

        if model_key == "sam3":
            return self._prepare_sam3(allow_download=allow_download, token=token)
        if model_key == "grounding_dino":
            return self._prepare_transformers_model(
                model_key="grounding_dino",
                model_id=settings.GROUNDING_DINO_MODEL_ID,
                load_fn=lambda: self.load_grounding_dino(settings.GROUNDING_DINO_MODEL_ID, allow_download=allow_download),
                allow_download=allow_download,
            )
        if model_key == "dinov3":
            return self._prepare_transformers_model(
                model_key="dinov3",
                model_id=settings.DINOV3_MODEL_ID,
                load_fn=lambda: self.load_dinov3(settings.DINOV3_MODEL_ID, allow_download=allow_download),
                allow_download=allow_download,
            )

        return {
            "model_key": model_key,
            "status": "blocked",
            "message": f"No prepare handler for {model_key}.",
            "next_steps": ["Select another integration."],
        }

    def _prepare_sam3(self, allow_download: bool, token: Optional[str] = None) -> Dict[str, Any]:
        sam3_status = self.get_advanced_model_status()["sam3"]
        model_path = settings.MODELS_DIR / settings.SAM3_MODEL
        if model_path.exists() and sam3_status["package_ready"]:
            return {
                "model_key": "sam3",
                "status": "ready",
                "message": "SAM3 is ready locally.",
                "path": str(model_path),
                "next_steps": ["Select SAM 3 Concept Segmentation when creating an image labeling job."],
            }

        next_steps = [
            "Upgrade the backend venv to ultralytics >= 8.3.237.",
            "Request access to facebook/sam3 on Hugging Face.",
            "Download sam3.pt and place it in backend/models/sam3.pt.",
        ]

        if not sam3_status["package_ready"]:
            return {
                "model_key": "sam3",
                "status": "blocked",
                "message": f"SAM3 package support is not ready. Installed ultralytics: {sam3_status['package_version'] or 'missing'}.",
                "next_steps": next_steps,
            }

        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not allow_download and not settings.ALLOW_MODEL_DOWNLOADS:
            return {
                "model_key": "sam3",
                "status": "blocked",
                "message": "SAM3 weights are gated and are not present locally. Automatic download is disabled.",
                "path": str(model_path),
                "next_steps": next_steps,
            }
        if not hf_token:
            return {
                "model_key": "sam3",
                "status": "blocked",
                "message": "SAM3 download requires an approved Hugging Face token in HF_TOKEN or request payload.",
                "path": str(model_path),
                "next_steps": next_steps + ["Set HF_TOKEN after access is approved, then click Prepare again."],
            }

        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            return {
                "model_key": "sam3",
                "status": "blocked",
                "message": f"huggingface_hub is required to download SAM3 weights: {exc}",
                "next_steps": ["Install huggingface_hub in the backend venv.", *next_steps],
            }

        try:
            downloaded = hf_hub_download(
                repo_id="facebook/sam3",
                filename=settings.SAM3_MODEL,
                token=hf_token,
                local_dir=str(settings.MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            return {
                "model_key": "sam3",
                "status": "ready",
                "message": "SAM3 weights downloaded successfully.",
                "path": downloaded,
                "next_steps": ["Select SAM 3 Concept Segmentation when creating an image labeling job."],
            }
        except Exception as exc:
            return {
                "model_key": "sam3",
                "status": "blocked",
                "message": f"SAM3 download did not complete: {exc}",
                "path": str(model_path),
                "next_steps": next_steps,
            }

    def _prepare_transformers_model(
        self,
        model_key: str,
        model_id: str,
        load_fn: Any,
        allow_download: bool,
    ) -> Dict[str, Any]:
        if _package_version("transformers") is None:
            return {
                "model_key": model_key,
                "status": "blocked",
                "message": f"{model_key} requires transformers in the backend venv.",
                "next_steps": ["Install transformers, accelerate, and safetensors.", f"Pre-cache {model_id} or enable downloads."],
            }
        try:
            load_fn()
            return {
                "model_key": model_key,
                "status": "ready",
                "message": f"{model_key} loaded successfully.",
                "next_steps": ["Use the model profile from the labeling job model selector."],
            }
        except Exception as exc:
            return {
                "model_key": model_key,
                "status": "blocked",
                "message": f"{model_key} could not be loaded locally: {exc}",
                "next_steps": [f"Pre-cache {model_id} in Hugging Face cache.", "Or set ALLOW_MODEL_DOWNLOADS=true and click Prepare again."],
            }

    def load_sam3(self, model_name: Optional[str] = None, allow_download: bool = False) -> Any:
        """Load SAM3 semantic predictor if package and weights are available."""
        model_name = model_name or settings.SAM3_MODEL
        status = self.get_advanced_model_status()["sam3"]
        if not status["package_ready"]:
            raise RuntimeError(
                f"SAM3 requires ultralytics >= {status['minimum_version']} "
                f"(installed: {status['package_version'] or 'missing'})."
            )

        model_path = settings.MODELS_DIR / model_name
        if not model_path.exists() and not allow_download and not settings.ALLOW_MODEL_DOWNLOADS:
            raise RuntimeError(f"SAM3 weight file not found: {model_path}. Download approved sam3.pt and place it in backend/models.")

        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception as exc:
            raise RuntimeError(f"SAM3 predictor is not available in this ultralytics install: {exc}") from exc

        model_ref = str(model_path) if model_path.exists() else model_name
        predictor = SAM3SemanticPredictor(
            overrides={
                "conf": settings.DEFAULT_CONFIDENCE_THRESHOLD,
                "task": "segment",
                "mode": "predict",
                "model": model_ref,
                "half": self.device.type == "cuda",
                "verbose": False,
                "save": False,
            }
        )
        self.models["sam3"] = predictor
        self._loaded_models.add("sam3")
        self.model_info["sam3"] = ModelInfo(
            name="SAM3",
            version="3.0",
            task_types=[TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION],
            description="Meta SAM3 promptable concept segmentation via Ultralytics",
            is_downloaded=model_path.exists(),
        )
        return predictor

    def detect_and_segment_sam3(
        self,
        image: np.ndarray,
        classes: List[str],
        conf_threshold: float = None,
        allow_download: bool = False,
    ) -> List[LabelAnnotation]:
        """Run SAM3 concept segmentation for all requested classes."""
        if not classes:
            return []
        if "sam3" not in self.models:
            self.load_sam3(allow_download=allow_download)

        predictor = self.models["sam3"]
        conf = conf_threshold or settings.DEFAULT_CONFIDENCE_THRESHOLD
        try:
            predictor.args.conf = conf
        except Exception:
            pass

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
            cv2.imwrite(temp_path, image)
            predictor.set_image(temp_path)
            results = predictor(text=classes)
            annotations = self._ultralytics_results_to_annotations(results, default_classes=classes)
            self.logger.info(f"SAM3 generated {len(annotations)} annotations for {len(classes)} prompts")
            return annotations
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SAM3 inference failed: {exc}") from exc
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def load_grounding_dino(self, model_id: Optional[str] = None, allow_download: bool = False) -> Any:
        """Load Grounding DINO through Transformers if available."""
        model_id = model_id or settings.GROUNDING_DINO_MODEL_ID
        if _package_version("transformers") is None:
            raise RuntimeError("Grounding DINO requires the transformers package.")

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError(f"Grounding DINO transformers classes are unavailable: {exc}") from exc

        local_only = not (allow_download or settings.ALLOW_MODEL_DOWNLOADS)
        try:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=local_only)
        except Exception as exc:
            mode = "local cache" if local_only else "download/cache"
            raise RuntimeError(f"Could not load Grounding DINO from {mode}: {model_id}. {exc}") from exc

        model.to(self.device)
        model.eval()
        self.models["grounding_dino_processor"] = processor
        self.models["grounding_dino_model"] = model
        self._loaded_models.add("grounding_dino")
        self.model_info["grounding_dino"] = ModelInfo(
            name="Grounding DINO",
            version="1.x",
            task_types=[TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION],
            description="Open-set object detector loaded through Transformers",
            is_downloaded=True,
        )
        return model

    def detect_objects_grounding_dino(
        self,
        image: np.ndarray,
        classes: List[str],
        conf_threshold: float = None,
        allow_download: bool = False,
    ) -> List[LabelAnnotation]:
        """Run Grounding DINO open-set detection for the requested classes."""
        if not classes:
            return []
        if "grounding_dino" not in self._loaded_models:
            self.load_grounding_dino(allow_download=allow_download)

        processor = self.models["grounding_dino_processor"]
        model = self.models["grounding_dino_model"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        text = ". ".join(classes) + "."
        threshold = conf_threshold or settings.DEFAULT_CONFIDENCE_THRESHOLD

        inputs = processor(images=pil_image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        try:
            processed = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                box_threshold=threshold,
                text_threshold=0.25,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            processed = processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )[0]

        annotations: List[LabelAnnotation] = []
        labels = processed.get("labels", [])
        scores = processed.get("scores", [])
        boxes = processed.get("boxes", [])
        text_labels = processed.get("text_labels", [])
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box.detach().cpu().tolist()]
            label_text = text_labels[i] if i < len(text_labels) else None
            if not label_text:
                label_index = int(labels[i].detach().cpu().item()) if i < len(labels) and hasattr(labels[i], "detach") else 0
                label_text = classes[label_index] if label_index < len(classes) else classes[0]
            confidence = float(scores[i].detach().cpu().item()) if i < len(scores) and hasattr(scores[i], "detach") else threshold
            annotations.append(
                LabelAnnotation(
                    id=i,
                    image_id="",
                    class_id=i,
                    class_name=str(label_text).strip(". "),
                    confidence=confidence,
                    bbox=BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1)),
                    area=max(0.0, x2 - x1) * max(0.0, y2 - y1),
                )
            )
        return annotations

    def load_dinov3(self, model_id: Optional[str] = None, allow_download: bool = False) -> Any:
        """Load DINOv3 feature extractor through Transformers if available."""
        model_id = model_id or settings.DINOV3_MODEL_ID
        if _package_version("transformers") is None:
            raise RuntimeError("DINOv3 requires the transformers package.")

        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            raise RuntimeError(f"DINOv3 transformers classes are unavailable: {exc}") from exc

        local_only = not (allow_download or settings.ALLOW_MODEL_DOWNLOADS)
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_only)
            model = AutoModel.from_pretrained(model_id, local_files_only=local_only)
        except Exception as exc:
            mode = "local cache" if local_only else "download/cache"
            raise RuntimeError(f"Could not load DINOv3 from {mode}: {model_id}. {exc}") from exc

        model.to(self.device)
        model.eval()
        self.models["dinov3_processor"] = processor
        self.models["dinov3_model"] = model
        self._loaded_models.add("dinov3")
        self.model_info["dinov3"] = ModelInfo(
            name="DINOv3",
            version="3.0",
            task_types=[TaskType.SEMANTIC_SEGMENTATION],
            description="Vision foundation model for dataset intelligence and feature extraction",
            is_downloaded=True,
        )
        return model

    def extract_dinov3_embedding(self, image: np.ndarray, allow_download: bool = False) -> List[float]:
        """Extract a compact DINOv3 image embedding for local dataset intelligence."""
        if "dinov3" not in self._loaded_models:
            self.load_dinov3(allow_download=allow_download)
        processor = self.models["dinov3_processor"]
        model = self.models["dinov3_model"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(outputs, "pooler_output", None)
        if hidden is None:
            raise RuntimeError("DINOv3 output does not contain usable embeddings.")
        if hidden.ndim == 3:
            vector = hidden.mean(dim=1)
        else:
            vector = hidden
        vector = F.normalize(vector, dim=-1)
        return vector[0].detach().cpu().float().tolist()

    def _ultralytics_results_to_annotations(self, results: Any, default_classes: Optional[List[str]] = None) -> List[LabelAnnotation]:
        annotations: List[LabelAnnotation] = []
        default_classes = default_classes or []
        if results is None:
            return annotations
        result_list = results if isinstance(results, (list, tuple)) else [results]
        for result in result_list:
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            
            # If we have masks but no boxes (common in semantic/prompt segmentation like SAM3)
            if (boxes is None or len(boxes) == 0) and masks is not None and getattr(masks, "data", None) is not None:
                for i, mask in enumerate(masks.data):
                    try:
                        mask_np = mask.detach().cpu().numpy()
                        coords = np.where(mask_np > 0)
                        if len(coords[0]) == 0:
                            continue
                        
                        y_min, y_max = float(coords[0].min()), float(coords[0].max())
                        x_min, x_max = float(coords[1].min()), float(coords[1].max())
                        
                        # Extract class and confidence from masks if available, otherwise fallback
                        class_id = i
                        if hasattr(masks, "cls") and masks.cls is not None:
                            try:
                                class_id = int(masks.cls[i].detach().cpu().numpy())
                            except Exception:
                                pass
                        
                        confidence = 1.0
                        if hasattr(masks, "conf") and masks.conf is not None:
                            try:
                                confidence = float(masks.conf[i].detach().cpu().numpy())
                            except Exception:
                                pass
                        
                        if isinstance(getattr(result, "names", None), dict):
                            class_name = result.names.get(class_id, default_classes[min(class_id, len(default_classes) - 1)] if default_classes else f"class_{class_id}")
                        elif default_classes:
                            class_name = default_classes[min(class_id, len(default_classes) - 1)]
                        else:
                            class_name = f"class_{class_id}"
                            
                        annotation = LabelAnnotation(
                            id=len(annotations),
                            image_id="",
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=BoundingBox(
                                x=x_min,
                                y=y_min,
                                width=max(0.0, x_max - x_min),
                                height=max(0.0, y_max - y_min)
                            ),
                            area=max(0.0, x_max - x_min) * max(0.0, y_max - y_min),
                        )
                        
                        polygon = self._mask_to_polygon(mask_np)
                        if polygon:
                            annotation.segmentation = SegmentationMask(polygon=polygon)
                        
                        annotations.append(annotation)
                    except Exception as exc:
                        self.logger.warning(f"Failed to extract mask-only annotation {i}: {exc}")
                continue
                
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().numpy()]
                    confidence = float(box.conf[0].detach().cpu().numpy()) if getattr(box, "conf", None) is not None else 1.0
                    class_id = int(box.cls[0].detach().cpu().numpy()) if getattr(box, "cls", None) is not None else i
                except Exception:
                    continue
                if isinstance(getattr(result, "names", None), dict):
                    class_name = result.names.get(class_id, default_classes[min(class_id, len(default_classes) - 1)] if default_classes else f"class_{class_id}")
                elif default_classes:
                    class_name = default_classes[min(class_id, len(default_classes) - 1)]
                else:
                    class_name = f"class_{class_id}"

                annotation = LabelAnnotation(
                    id=len(annotations),
                    image_id="",
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1)),
                    area=max(0.0, x2 - x1) * max(0.0, y2 - y1),
                )
                if masks is not None and getattr(masks, "data", None) is not None and i < len(masks.data):
                    try:
                        mask_np = masks.data[i].detach().cpu().numpy()
                        polygon = self._mask_to_polygon(mask_np)
                        if polygon:
                            annotation.segmentation = SegmentationMask(polygon=polygon)
                    except Exception as exc:
                        self.logger.warning(f"Failed to extract advanced model mask {i}: {exc}")
                annotations.append(annotation)
        return annotations

    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            self._loaded_models.discard(model_name)
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info(f"Unloaded model: {model_name}")
    
    def unload_all(self):
        """Unload all models."""
        for name in list(self.models.keys()):
            self.unload_model(name)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._loaded_models)


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get model manager instance."""
    return model_manager
