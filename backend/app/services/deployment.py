# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Deployment Utilities.
Model export, quantization, caching, and batched inference for production.
"""

import gc
import hashlib
import json
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger("deployment")


class LRUCache:
    """Thread-safe LRU cache for inference results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_input(self, image: np.ndarray, params: Dict) -> str:
        """Create hash from image and parameters."""
        # Use image shape and sample pixels for fast hashing
        shape_hash = str(image.shape)
        sample_pixels = image[::50, ::50].tobytes()[:1000]  # Sample
        params_str = json.dumps(params, sort_keys=True)
        
        combined = f"{shape_hash}{sample_pixels}{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, image: np.ndarray, params: Dict) -> Optional[Any]:
        """Get cached result if exists."""
        key = self._hash_input(image, params)
        
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, image: np.ndarray, params: Dict, result: Any):
        """Add result to cache."""
        key = self._hash_input(image, params)
        
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest
                self.cache.popitem(last=False)
            self.cache[key] = result
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate
        }


class DeploymentService:
    """
    Service for model deployment utilities.
    
    Features:
    - ONNX/TensorRT export
    - FP16/INT8 quantization
    - Inference caching
    - Batched inference
    """
    
    def __init__(self):
        self.inference_cache = LRUCache(max_size=1000)
        self.batch_queue: List[Tuple[np.ndarray, Dict]] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else "cpu"
        self._exported_models: Dict[str, Path] = {}
    
    def export_yolo_onnx(
        self,
        model_name: str = "yolov8x-seg",
        output_path: Optional[Path] = None,
        quantize: str = "none",  # "none", "fp16", "int8"
        dynamic_batch: bool = True,
        imgsz: int = 640,
        opset: int = 12
    ) -> Path:
        """
        Export YOLO model to ONNX format.
        
        Args:
            model_name: YOLO model variant
            output_path: Output file path
            quantize: Quantization mode
            dynamic_batch: Enable dynamic batch size
            imgsz: Input image size
            opset: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"Exporting {model_name} to ONNX (quantize={quantize})...")
            
            # Load model
            model = YOLO(model_name)
            
            # Set output path
            if output_path is None:
                suffix = f"_{quantize}" if quantize != "none" else ""
                output_path = settings.MODELS_DIR / f"{model_name.replace('.pt', '')}{suffix}.onnx"
            
            # Export with quantization
            half = quantize == "fp16"
            int8 = quantize == "int8"
            
            export_path = model.export(
                format="onnx",
                imgsz=imgsz,
                half=half,
                int8=int8,
                dynamic=dynamic_batch,
                opset=opset,
                simplify=True
            )
            
            # Move to target location if different
            export_path = Path(export_path)
            if export_path != output_path:
                import shutil
                shutil.move(str(export_path), str(output_path))
            
            self._exported_models[f"{model_name}_onnx"] = output_path
            
            logger.info(f"ONNX export complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def export_yolo_tensorrt(
        self,
        model_name: str = "yolov8x-seg",
        output_path: Optional[Path] = None,
        half: bool = True,
        imgsz: int = 640,
        workspace: int = 4  # GB
    ) -> Path:
        """
        Export YOLO model to TensorRT format.
        
        Args:
            model_name: YOLO model variant
            output_path: Output file path
            half: Use FP16 precision
            imgsz: Input image size
            workspace: TensorRT workspace size in GB
            
        Returns:
            Path to exported TensorRT engine
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"Exporting {model_name} to TensorRT (half={half})...")
            
            model = YOLO(model_name)
            
            if output_path is None:
                suffix = "_fp16" if half else ""
                output_path = settings.MODELS_DIR / f"{model_name.replace('.pt', '')}{suffix}.engine"
            
            export_path = model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                workspace=workspace,
                dynamic=False  # TensorRT prefers static
            )
            
            export_path = Path(export_path)
            if export_path != output_path:
                import shutil
                shutil.move(str(export_path), str(output_path))
            
            self._exported_models[f"{model_name}_tensorrt"] = output_path
            
            logger.info(f"TensorRT export complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            raise
    
    def export_yolo_tflite(
        self,
        model_name: str = "yolov8x-seg",
        output_path: Optional[Path] = None,
        int8: bool = False,
        imgsz: int = 640
    ) -> Path:
        """
        Export YOLO model to TFLite format for edge devices.
        
        Args:
            model_name: YOLO model variant
            output_path: Output file path
            int8: Use INT8 quantization
            imgsz: Input image size
            
        Returns:
            Path to exported TFLite model
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"Exporting {model_name} to TFLite (int8={int8})...")
            
            model = YOLO(model_name)
            
            if output_path is None:
                suffix = "_int8" if int8 else ""
                output_path = settings.MODELS_DIR / f"{model_name.replace('.pt', '')}{suffix}.tflite"
            
            export_path = model.export(
                format="tflite",
                imgsz=imgsz,
                int8=int8
            )
            
            export_path = Path(export_path)
            if export_path != output_path:
                import shutil
                shutil.move(str(export_path), str(output_path))
            
            self._exported_models[f"{model_name}_tflite"] = output_path
            
            logger.info(f"TFLite export complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            raise
    
    def batch_inference(
        self,
        images: List[np.ndarray],
        model_func: callable,
        batch_size: int = 8,
        use_cache: bool = True,
        **kwargs
    ) -> List[Any]:
        """
        Run batched inference with optional caching.
        
        Args:
            images: List of input images
            model_func: Model inference function
            batch_size: Batch size for inference
            use_cache: Whether to use result caching
            **kwargs: Additional arguments for model_func
            
        Returns:
            List of results for each image
        """
        results = [None] * len(images)
        uncached_indices = []
        uncached_images = []
        
        # Check cache
        if use_cache:
            for i, img in enumerate(images):
                cached = self.inference_cache.get(img, kwargs)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_images.append(img)
        else:
            uncached_indices = list(range(len(images)))
            uncached_images = images
        
        if not uncached_images:
            logger.info("All results from cache")
            return results
        
        # Process in batches
        for batch_start in range(0, len(uncached_images), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_images))
            batch_images = uncached_images[batch_start:batch_end]
            
            # Run inference
            batch_results = model_func(batch_images, **kwargs)
            
            # Store results
            for i, (img, result) in enumerate(zip(batch_images, batch_results)):
                original_idx = uncached_indices[batch_start + i]
                results[original_idx] = result
                
                if use_cache:
                    self.inference_cache.put(img, kwargs, result)
        
        logger.info(
            f"Batch inference: {len(images)} images, "
            f"{len(images) - len(uncached_images)} cached, "
            f"{len(uncached_images)} computed"
        )
        
        return results
    
    def measure_inference_time(
        self,
        model_func: callable,
        sample_image: np.ndarray,
        warmup_runs: int = 3,
        test_runs: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """
        Measure inference time for a model.
        
        Args:
            model_func: Model inference function
            sample_image: Sample input image
            warmup_runs: Number of warmup runs
            test_runs: Number of test runs
            **kwargs: Additional arguments for model_func
            
        Returns:
            Timing statistics
        """
        # Warmup
        for _ in range(warmup_runs):
            _ = model_func(sample_image, **kwargs)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = model_func(sample_image, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times))
        }
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
        }
    
    def optimize_memory(self):
        """Free up GPU and CPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        logger.info("Memory optimized")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get inference cache statistics."""
        return self.inference_cache.stats()
    
    def clear_cache(self):
        """Clear the inference cache."""
        self.inference_cache.clear()
        logger.info("Inference cache cleared")
    
    def list_exported_models(self) -> Dict[str, str]:
        """List all exported models."""
        return {k: str(v) for k, v in self._exported_models.items()}
    
    def create_serving_config(
        self,
        model_name: str,
        model_path: Path,
        backend: str = "torchserve"
    ) -> Dict[str, Any]:
        """
        Create configuration for model serving.
        
        Args:
            model_name: Name for the served model
            model_path: Path to the model file
            backend: Serving backend (torchserve, triton)
            
        Returns:
            Configuration dictionary
        """
        if backend == "torchserve":
            return {
                "model_name": model_name,
                "model_path": str(model_path),
                "handler": "vision",
                "batch_size": 8,
                "max_batch_delay": 100,  # ms
                "response_timeout": 120,
                "inference_address": "http://0.0.0.0:8085"
            }
        
        elif backend == "triton":
            return {
                "name": model_name,
                "platform": "onnxruntime_onnx",
                "max_batch_size": 8,
                "input": [{
                    "name": "images",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, 3, 640, 640]
                }],
                "output": [{
                    "name": "output",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, -1, -1]
                }],
                "instance_group": [{
                    "count": 1,
                    "kind": "KIND_GPU"
                }]
            }
        
        else:
            raise ValueError(f"Unknown backend: {backend}")


# Global instance
_deployment_service: Optional[DeploymentService] = None


def get_deployment_service() -> DeploymentService:
    """Get the global deployment service instance."""
    global _deployment_service
    if _deployment_service is None:
        _deployment_service = DeploymentService()
    return _deployment_service
