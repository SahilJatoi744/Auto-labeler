# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Export service for labeled datasets.
Supports COCO JSON, Pascal VOC XML, and YOLO formats.
"""

import json
import random
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.config import settings
from ..core.logging import get_logger
from ..models.schemas import (
    BoundingBox, ClassDefinition, ClassHierarchy, ExportConfig, ExportFormat,
    ExportResult, ImageInfo, ImageLabels, LabelAnnotation
)

logger = get_logger("exporter")


class ExportService:
    """
    Service for exporting labeled datasets in various formats.
    """
    
    def __init__(self):
        self.logger = get_logger("exporter")
    
    def _merge_classes_from_annotations(
        self,
        results: List[ImageLabels],
        class_hierarchy: ClassHierarchy
    ) -> ClassHierarchy:
        """
        Auto-discover classes from annotations that aren't in the hierarchy.
        This ensures manually added annotations with new classes are never dropped.
        """
        known_ids = {c.id for c in class_hierarchy.classes}
        known_names = {c.name.lower() for c in class_hierarchy.classes}
        max_id = max(known_ids) if known_ids else 0
        new_classes = []

        for img_labels in results:
            for ann in img_labels.annotations:
                name = ann.class_name or f"class_{ann.class_id}"
                name_lower = name.lower()
                
                # If ID is unknown AND Name is unknown, add as new class
                if ann.class_id not in known_ids and name_lower not in known_names:
                    max_id += 1
                    new_cls = ClassDefinition(id=ann.class_id, name=name)
                    new_classes.append(new_cls)
                    known_ids.add(ann.class_id)
                    known_names.add(name_lower)
                    self.logger.info(f"Export: auto-discovered class '{name}' (id={ann.class_id})")

        if new_classes:
            # Create a new hierarchy with merged classes
            merged = ClassHierarchy(
                classes=list(class_hierarchy.classes) + new_classes,
                version=class_hierarchy.version
            )
            self.logger.info(f"Export: merged {len(new_classes)} new classes into hierarchy")
            return merged

        return class_hierarchy

    def export_dataset(
        self,
        job_id: str,
        results: List[ImageLabels],
        class_hierarchy: ClassHierarchy,
        config: ExportConfig
    ) -> ExportResult:
        """
        Export labeled dataset in specified format.
        """
        export_id = f"{job_id}_{config.format.value}_{datetime.now().strftime('%Y%H%M%S')}"
        
        # Auto-discover classes from annotations (fixes missing manually-added labels)
        class_hierarchy = self._merge_classes_from_annotations(results, class_hierarchy)
        
        # Create output directory
        output_dir = Path(config.output_path) if config.output_path else settings.OUTPUT_DIR / export_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting to {config.format.value}: {output_dir}")
        
        # Split data if requested
        default_splits = {"train": 0.7, "val": 0.2, "test": 0.1}
        splits = self._split_dataset(results, config.split_ratios or default_splits)
        
        # Export in specified format
        file_paths = {}
        
        if config.format == ExportFormat.COCO:
            file_paths = self._export_coco(splits, class_hierarchy, output_dir, config)
        
        elif config.format == ExportFormat.PASCAL_VOC:
            file_paths = self._export_pascal_voc(splits, class_hierarchy, output_dir, config)
        
        elif config.format == ExportFormat.YOLO:
            file_paths = self._export_yolo(splits, class_hierarchy, output_dir, config)
        
        # Calculate statistics
        stats = self._calculate_statistics(results, splits)
        
        # Create ZIP archive
        archive_path = self._create_archive(output_dir, export_id)
        
        result = ExportResult(
            export_id=export_id,
            job_id=job_id,
            format=config.format,
            output_path=str(archive_path), # Return zip path
            file_paths=file_paths,
            statistics=stats,
            created_at=datetime.now().isoformat()
        )
        
        self.logger.info(f"Export completed: {export_id}")
        
        return result
    
    def _create_archive(self, source_dir: Path, base_name: str) -> Path:
        """Create a ZIP archive of the exported dataset."""
        archive_name = settings.OUTPUT_DIR / f"{base_name}"
        shutil.make_archive(str(archive_name), 'zip', source_dir)
        return Path(f"{str(archive_name)}.zip")

    def _split_dataset(
        self,
        results: List[ImageLabels],
        ratios: Dict[str, float]
    ) -> Dict[str, List[ImageLabels]]:
        """
        Split dataset into train/val/test sets.
        """
        # Shuffle results
        shuffled = results.copy()
        random.seed(42) # Deterministic shuffle for consistency
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_ratio = ratios.get("train", 0.7)
        val_ratio = ratios.get("val", 0.2)
        test_ratio = ratios.get("test", 0.1)
        
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        if total > 0:
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:]
        }
        
        return splits
    
    def _export_coco(
        self,
        splits: Dict[str, List[ImageLabels]],
        class_hierarchy: ClassHierarchy,
        output_dir: Path,
        config: ExportConfig
    ) -> Dict[str, str]:
        """
        Export in COCO JSON format.
        """
        file_paths = {}
        class_map = {c.id: i + 1 for i, c in enumerate(class_hierarchy.classes)}
        
        for split_name, results in splits.items():
            if not results:
                continue
            
            coco_data = {
                "info": {
                    "description": f"AutoLabeler Export - {split_name}",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "AutoLabeler",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Add categories
            name_to_id_map = {c.name.lower(): class_map[c.id] for c in class_hierarchy.classes if c.id in class_map}
            for cls in class_hierarchy.classes:
                coco_data["categories"].append({
                    "id": class_map[cls.id],
                    "name": cls.name,
                    "supercategory": self._get_parent_name(cls.parent_id, class_hierarchy) or "none"
                })
            
            annotation_id = 0
            
            for img_labels in results:
                img_path = self._find_image_path(img_labels.image_id)
                width, height = 0, 0
                if img_path:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except:
                        pass
                
                coco_data["images"].append({
                    "id": img_labels.image_id,
                    "file_name": img_path.name if img_path else f"{img_labels.image_id}.jpg",
                    "width": width,
                    "height": height
                })
                
                for ann in img_labels.annotations:
                    if config.min_confidence and ann.confidence < config.min_confidence:
                        continue
                    
                    category_id = class_map.get(ann.class_id)
                    if category_id is None and ann.class_name:
                        category_id = name_to_id_map.get(ann.class_name.lower())
                        
                    if category_id is None:
                        continue
                        
                    coco_ann = {
                        "id": annotation_id,
                        "image_id": img_labels.image_id,
                        "category_id": category_id,
                        "bbox": [
                            ann.bbox.x,
                            ann.bbox.y,
                            ann.bbox.width,
                            ann.bbox.height
                        ] if ann.bbox else [],
                        "area": ann.area or (ann.bbox.area if ann.bbox else 0),
                        "iscrowd": 0,
                        "score": round(ann.confidence, 4)
                    }
                    
                    # Add segmentation if available
                    if ann.segmentation and ann.segmentation.polygon:
                        # COCO expects a list of polygons (contours)
                        coco_ann["segmentation"] = ann.segmentation.polygon
                    
                    coco_data["annotations"].append(coco_ann)
                    annotation_id += 1
            
            output_file = output_dir / f"{split_name}_coco.json"
            with open(output_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            file_paths[split_name] = str(output_file)
        
        return file_paths
    
    def _export_pascal_voc(
        self,
        splits: Dict[str, List[ImageLabels]],
        class_hierarchy: ClassHierarchy,
        output_dir: Path,
        config: ExportConfig
    ) -> Dict[str, str]:
        """
        Export in Pascal VOC XML format.
        """
        file_paths = {}
        # Build class_names from hierarchy + auto-discover from annotations
        class_names = {c.id: c.name for c in class_hierarchy.classes}
        for img_labels_list in splits.values():
            for img_labels in img_labels_list:
                for ann in img_labels.annotations:
                    if ann.class_id not in class_names:
                        class_names[ann.class_id] = ann.class_name or f"class_{ann.class_id}"
        
        for split_name, results in splits.items():
            if not results:
                continue
            
            split_dir = output_dir / split_name / "Annotations"
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_labels in results:
                img_path = self._find_image_path(img_labels.image_id)
                width, height, depth = 640, 640, 3
                if img_path:
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            depth = len(img.getbands())
                        filename = img_path.name
                    except:
                        filename = f"{img_labels.image_id}.jpg"
                else:
                    filename = f"{img_labels.image_id}.jpg"
                
                annotation = ET.Element("annotation")
                ET.SubElement(annotation, "folder").text = split_name
                ET.SubElement(annotation, "filename").text = filename
                
                size = ET.SubElement(annotation, "size")
                ET.SubElement(size, "width").text = str(width)
                ET.SubElement(size, "height").text = str(height)
                ET.SubElement(size, "depth").text = str(depth)
                
                for ann in img_labels.annotations:
                    if config.min_confidence and ann.confidence < config.min_confidence:
                        continue
                    if not ann.bbox:
                        continue
                    
                    class_name = class_names.get(ann.class_id)
                    if not class_name and ann.class_name:
                        # Fallback to name match
                        class_name = next((name for nid, name in class_names.items() if name.lower() == ann.class_name.lower()), None)
                    
                    if not class_name:
                        continue

                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = class_name
                    
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(int(ann.bbox.x))
                    ET.SubElement(bndbox, "ymin").text = str(int(ann.bbox.y))
                    ET.SubElement(bndbox, "xmax").text = str(int(ann.bbox.x2))
                    ET.SubElement(bndbox, "ymax").text = str(int(ann.bbox.y2))
                    ET.SubElement(obj, "confidence").text = str(round(ann.confidence, 4))
                
                xml_path = split_dir / f"{Path(filename).stem}.xml"
                tree = ET.ElementTree(annotation)
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            
            file_paths[split_name] = str(split_dir)
        
        return file_paths
    
    def _export_yolo(
        self,
        splits: Dict[str, List[ImageLabels]],
        class_hierarchy: ClassHierarchy,
        output_dir: Path,
        config: ExportConfig
    ) -> Dict[str, str]:
        """
        Export in YOLO format.
        """
        file_paths = {}
        class_map = {c.id: i for i, c in enumerate(class_hierarchy.classes)}
        name_to_idx_map = {c.name.lower(): i for i, c in enumerate(class_hierarchy.classes)}
        
        for split_name, results in splits.items():
            if not results:
                continue
            
            labels_dir = output_dir / split_name / "labels"
            images_dir = output_dir / split_name / "images"
            labels_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            
            for img_labels in results:
                img_path = self._find_image_path(img_labels.image_id)
                if not img_path:
                    continue
                
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except:
                    continue
                
                shutil.copy2(img_path, images_dir / img_path.name)
                
                label_lines = []
                for ann in img_labels.annotations:
                    if config.min_confidence and ann.confidence < config.min_confidence:
                        continue
                    class_idx = class_map.get(ann.class_id)
                    if class_idx is None and ann.class_name:
                        class_idx = name_to_idx_map.get(ann.class_name.lower())
                        
                    if class_idx is None:
                        continue
                    
                    # Check for segmentation first
                    if ann.segmentation and ann.segmentation.polygon:
                        for poly in ann.segmentation.polygon:
                            coords = []
                            for pt in poly:
                                if len(pt) >= 2:
                                    x, y = pt[0], pt[1]
                                    coords.append(max(0, min(1, x / img_width)))
                                    coords.append(max(0, min(1, y / img_height)))
                            if coords:
                                label_lines.append(f"{class_idx} " + " ".join([f"{c:.6f}" for c in coords]))
                    elif ann.bbox:
                        x_center = (ann.bbox.x + ann.bbox.width / 2) / img_width
                        y_center = (ann.bbox.y + ann.bbox.height / 2) / img_height
                        w = ann.bbox.width / img_width
                        h = ann.bbox.height / img_height
                        label_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                with open(labels_dir / f"{img_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(label_lines))
            
            # YAML config
            yaml_content = f"path: .\ntrain: images\nval: images\nnc: {len(class_map)}\nnames: {[c.name for c in class_hierarchy.classes]}"
            with open(output_dir / split_name / "data.yaml", 'w') as f:
                f.write(yaml_content)
            
            file_paths[split_name] = str(output_dir / split_name)
            
        return file_paths
    
    def _find_image_path(self, image_id: str) -> Optional[Path]:
        """Find image path from image ID."""
        for metadata_file in settings.UPLOAD_DIR.glob("*_metadata.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                for img in data.get("images", []):
                    if img.get("id") == image_id:
                        p = Path(img.get("path", ""))
                        if p.exists(): return p
            except: pass
        return None
    
    def _get_parent_name(self, parent_id, hierarchy):
        if parent_id is None: return None
        for c in hierarchy.classes:
            if c.id == parent_id: return c.name
        return None
    
    def _calculate_statistics(self, results, splits):
        total_anns = sum(len(r.annotations) for r in results)
        return {
            "total_images": len(results),
            "total_annotations": total_anns,
            "splits": {k: {"images": len(v), "annotations": sum(len(r.annotations) for r in v)} for k, v in splits.items()}
        }

export_service = ExportService()

def get_export_service():
    return export_service
