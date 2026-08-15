# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Data preprocessing and validation service.
Handles dataset integrity checks, format normalization, and error logging.
"""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json

import cv2
import numpy as np
from PIL import Image

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import DatasetInfo, ImageInfo

logger = get_logger("preprocessor")


class DataPreprocessor:
    """
    Preprocessor for image datasets.
    Validates integrity, handles corrupted data, normalizes formats.
    """
    
    def __init__(self):
        self.supported_formats = set(settings.SUPPORTED_FORMATS)
        self.max_image_size = settings.MAX_IMAGE_SIZE
        self.logger = get_logger("preprocessor")
    
    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[ImageInfo], Optional[str]]:
        """
        Validate a single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, image_info, error_message)
        """
        try:
            # Check if file exists
            if not image_path.exists():
                return False, None, f"File not found: {image_path}"
            
            # Check extension
            ext = image_path.suffix.lower()
            if ext not in self.supported_formats:
                return False, None, f"Unsupported format: {ext}"
            
            # Try to open with PIL
            try:
                with Image.open(image_path) as img:
                    # Verify image can be loaded
                    img.verify()
                
                # Reopen to get actual data (verify closes the file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_type = img.format.lower() if img.format else ext.lstrip('.')
                    
                    # Check image dimensions
                    if width == 0 or height == 0:
                        return False, None, "Invalid image dimensions"
                    
                    # Check max size
                    max_dim = max(width, height)
                    if max_dim > self.max_image_size:
                        self.logger.warning(f"Image {image_path.name} exceeds max size: {max_dim}px")
                    
                    # Get file size
                    size_bytes = image_path.stat().st_size
                    
                    # Generate unique ID
                    image_id = hashlib.md5(str(image_path).encode()).hexdigest()[:16]
                    
                    info = ImageInfo(
                        id=image_id,
                        filename=image_path.name,
                        path=str(image_path),
                        width=width,
                        height=height,
                        format=format_type,
                        size_bytes=size_bytes,
                        status="valid"
                    )
                    
                    return True, info, None
                    
            except Exception as e:
                return False, None, f"Corrupted image: {str(e)}"
                
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"
    
    def process_dataset(
        self,
        dataset_path: Path,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        max_workers: Optional[int] = None
    ) -> DatasetInfo:
        """
        Process and validate an entire dataset.
        
        Args:
            dataset_path: Path to dataset directory
            dataset_name: Optional name for the dataset
            dataset_id: Optional ID (generated externally)
            max_workers: Number of parallel workers
            
        Returns:
            DatasetInfo with validation results
        """
        self.logger.info(f"Processing dataset: {dataset_path}")
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Use provided ID or generate one
        if dataset_id is None:
            dataset_id = hashlib.md5(str(dataset_path).encode()).hexdigest()[:12]
        
        if dataset_name is None:
            dataset_name = dataset_path.name
        
        # Find all image files
        image_files_set = set()
        for ext in self.supported_formats:
            # On Windows, rglob might be case-insensitive, causing duplicates if we look for .jpg and .JPG
            # We use resolve() to get canonical path and add to set
            for p in dataset_path.rglob(f"*{ext}"):
                image_files_set.add(p.resolve())
            for p in dataset_path.rglob(f"*{ext.upper()}"):
                image_files_set.add(p.resolve())
        
        image_files = list(image_files_set)
        total_images = len(image_files)
        self.logger.info(f"Found {total_images} potential images")
        
        if total_images == 0:
            return DatasetInfo(
                id=dataset_id,
                name=dataset_name,
                path=str(dataset_path),
                total_images=0,
                valid_images=0,
                corrupted_images=0,
                total_size_mb=0.0,
                formats={},
                status="empty"
            )
        
        # Validate images in parallel
        max_workers = max_workers or settings.MAX_WORKERS
        valid_images: List[ImageInfo] = []
        corrupted_count = 0
        format_counts: Dict[str, int] = {}
        total_size = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.validate_image, img_path): img_path 
                for img_path in image_files
            }
            
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    is_valid, info, error = future.result()
                    
                    if is_valid and info:
                        valid_images.append(info)
                        total_size += info.size_bytes
                        
                        # Count formats
                        fmt = info.format.lower()
                        format_counts[fmt] = format_counts.get(fmt, 0) + 1
                        
                    else:
                        corrupted_count += 1
                        self.logger.debug(f"Invalid image {img_path.name}: {error}")
                        
                except Exception as e:
                    corrupted_count += 1
                    self.logger.error(f"Error processing {img_path.name}: {e}")
        
        # Save valid image list to JSON
        metadata_path = settings.UPLOAD_DIR / f"{dataset_id}_metadata.json"
        metadata = {
            "dataset_id": dataset_id,
            "name": dataset_name,
            "path": str(dataset_path),
            "images": [img.model_dump() for img in valid_images],
            "total_images": total_images,
            "valid_images": len(valid_images),
            "corrupted_images": corrupted_count
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        dataset_info = DatasetInfo(
            id=dataset_id,
            name=dataset_name,
            path=str(dataset_path),
            total_images=total_images,
            valid_images=len(valid_images),
            corrupted_images=corrupted_count,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            formats=format_counts,
            status="ready"
        )
        
        self.logger.info(
            f"Dataset processed: {dataset_info.valid_images} valid, "
            f"{dataset_info.corrupted_images} corrupted, "
            f"{dataset_info.total_size_mb:.2f} MB"
        )
        
        return dataset_info
    
    def normalize_image(
        self,
        image_path: Path,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Load and normalize an image for model inference.
        
        Args:
            image_path: Path to image
            target_size: Optional (width, height) to resize to
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Normalized image as numpy array
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if target_size:
            if maintain_aspect:
                # Calculate new size maintaining aspect ratio
                h, w = img.shape[:2]
                target_w, target_h = target_size
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Pad to target size
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                img = cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            else:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        return img
    
    def detect_duplicates(
        self,
        image_paths: List[Path],
        hash_size: int = 16
    ) -> List[Set[str]]:
        """
        Detect duplicate images using perceptual hashing.
        
        Args:
            image_paths: List of image paths
            hash_size: Hash size for comparison
            
        Returns:
            List of sets, each containing duplicate image IDs
        """
        from imagehash import phash
        from PIL import Image
        
        hashes: Dict[str, str] = {}
        duplicates: List[Set[str]] = []
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    img_hash = str(phash(img, hash_size))
                    
                img_id = hashlib.md5(str(img_path).encode()).hexdigest()[:16]
                
                # Check for similar hashes
                for existing_hash, existing_id in hashes.items():
                    # Calculate hash difference
                    diff = sum(c1 != c2 for c1, c2 in zip(img_hash, existing_hash))
                    if diff <= 5:  # Threshold for similarity
                        # Find or create duplicate group
                        found = False
                        for group in duplicates:
                            if existing_id in group:
                                group.add(img_id)
                                found = True
                                break
                        if not found:
                            duplicates.append({existing_id, img_id})
                
                hashes[img_hash] = img_id
                
            except Exception as e:
                self.logger.warning(f"Could not hash {img_path}: {e}")
        
        return duplicates
    
    def get_image_statistics(self, dataset_info: DatasetInfo) -> Dict:
        """
        Calculate statistics for a dataset.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            Dictionary of statistics
        """
        metadata_path = settings.UPLOAD_DIR / f"{dataset_info.id}_metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        images = metadata.get("images", [])
        
        if not images:
            return {}
        
        widths = [img["width"] for img in images]
        heights = [img["height"] for img in images]
        sizes = [img["size_bytes"] for img in images]
        
        aspect_ratios = [w / h for w, h in zip(widths, heights)]
        
        stats = {
            "image_count": len(images),
            "width": {
                "min": min(widths),
                "max": max(widths),
                "mean": sum(widths) / len(widths),
                "median": sorted(widths)[len(widths) // 2]
            },
            "height": {
                "min": min(heights),
                "max": max(heights),
                "mean": sum(heights) / len(heights),
                "median": sorted(heights)[len(heights) // 2]
            },
            "aspect_ratio": {
                "min": round(min(aspect_ratios), 2),
                "max": round(max(aspect_ratios), 2),
                "mean": round(sum(aspect_ratios) / len(aspect_ratios), 2)
            },
            "file_size_mb": {
                "min": round(min(sizes) / (1024 * 1024), 2),
                "max": round(max(sizes) / (1024 * 1024), 2),
                "mean": round(sum(sizes) / len(sizes) / (1024 * 1024), 2),
                "total": round(sum(sizes) / (1024 * 1024), 2)
            }
        }
        
        return stats


# Global preprocessor instance
preprocessor = DataPreprocessor()


def get_preprocessor() -> DataPreprocessor:
    """Get preprocessor instance."""
    return preprocessor
