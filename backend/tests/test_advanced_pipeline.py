# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Test suite for advanced pipeline components.
Tests the new services: semantic segmentation, panoptic fusion, active learning, and deployment.
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSemanticSegmentation:
    """Tests for DeepLabV3+ semantic segmentation service."""
    
    def test_service_initialization(self):
        """Test that the service can be initialized."""
        from app.services.semantic_segmentation import SemanticSegmentationService
        
        service = SemanticSegmentationService()
        assert service is not None
        assert service.is_loaded is False
        assert service.model_type == "deeplabv3+"
    
    def test_cityscapes_classes(self):
        """Test that Cityscapes class mapping is correct."""
        from app.services.semantic_segmentation import CITYSCAPES_CLASSES
        
        assert 0 in CITYSCAPES_CLASSES  # road
        assert CITYSCAPES_CLASSES[0] == "road"
        assert CITYSCAPES_CLASSES[1] == "sidewalk"
        assert CITYSCAPES_CLASSES[10] == "sky"
    
    def test_mask_to_polygon(self):
        """Test mask to polygon conversion."""
        from app.services.semantic_segmentation import SemanticSegmentationService
        
        service = SemanticSegmentationService()
        
        # Create a simple binary mask (filled rectangle)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        
        polygon = service._mask_to_polygon(mask)
        
        assert polygon is not None
        assert len(polygon) > 0
        # Polygon should have at least 4 points for a rectangle
        assert len(polygon[0]) >= 8  # 4 points x 2 coordinates


class TestPanopticFusion:
    """Tests for panoptic fusion engine."""
    
    def test_fusion_engine_initialization(self):
        """Test that the fusion engine can be initialized."""
        from app.services.panoptic_fusion import PanopticFusionEngine
        
        engine = PanopticFusionEngine()
        assert engine is not None
        assert engine.next_segment_id == 1
    
    def test_thing_stuff_classification(self):
        """Test thing vs stuff class classification."""
        from app.services.panoptic_fusion import PanopticFusionEngine
        
        engine = PanopticFusionEngine()
        
        assert "car" in engine.THING_CLASSES
        assert "person" in engine.THING_CLASSES
        assert "road" in engine.STUFF_CLASSES
        assert "sky" in engine.STUFF_CLASSES
    
    def test_iou_calculation(self):
        """Test bounding box IoU calculation."""
        from app.services.panoptic_fusion import PanopticFusionEngine
        from app.models.schemas import BoundingBox
        
        engine = PanopticFusionEngine()
        
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=50, y=50, width=100, height=100)
        
        iou = engine._calculate_bbox_iou(box1, box2)
        
        # Overlap area = 50*50 = 2500
        # Union = 100*100 + 100*100 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.143
        assert 0.1 < iou < 0.2
    
    def test_polygon_to_mask(self):
        """Test polygon to mask conversion."""
        from app.services.panoptic_fusion import PanopticFusionEngine
        
        engine = PanopticFusionEngine()
        
        # Simple square polygon
        polygon = [[10, 10, 90, 10, 90, 90, 10, 90]]
        mask = engine._polygon_to_mask(polygon, (100, 100))
        
        assert mask is not None
        assert mask.shape == (100, 100)
        assert mask.sum() > 0  # Should have some filled area


class TestActiveLearning:
    """Tests for active learning service."""
    
    def test_service_initialization(self):
        """Test that the service can be initialized."""
        from app.services.active_learning import ActiveLearningService
        
        service = ActiveLearningService()
        assert service is not None
        assert len(service.scored_samples) == 0
    
    def test_uncertainty_weights(self):
        """Test that uncertainty weights sum to ~1."""
        from app.services.active_learning import ActiveLearningService
        
        service = ActiveLearningService()
        total_weight = sum(service.UNCERTAINTY_WEIGHTS.values())
        
        assert 0.99 < total_weight < 1.01
    
    def test_sampling_strategies(self):
        """Test that all sampling strategies are defined."""
        from app.services.active_learning import SamplingStrategy
        
        assert SamplingStrategy.UNCERTAINTY.value == "uncertainty"
        assert SamplingStrategy.DIVERSITY.value == "diversity"
        assert SamplingStrategy.HYBRID.value == "hybrid"
        assert SamplingStrategy.RANDOM.value == "random"
    
    def test_detection_mask_disagreement(self):
        """Test detection-mask disagreement calculation."""
        from app.services.active_learning import ActiveLearningService
        from app.models.schemas import LabelAnnotation, BoundingBox, SegmentationMask
        
        service = ActiveLearningService()
        
        # Create annotation with matching bbox and mask
        # Use proper polygon format: list of [x,y] coordinates flattened
        ann = LabelAnnotation(
            id=1,
            image_id="test",
            class_id=0,
            class_name="car",
            confidence=0.9,
            bbox=BoundingBox(x=0, y=0, width=100, height=100),
            segmentation=SegmentationMask(
                polygon=[[10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]]  # 80x80 filled
            )
        )
        
        disagreement = service._compute_detection_mask_disagreement(ann)
        
        # Should return a valid score between 0 and 1
        assert 0 <= disagreement <= 1
    
    def test_size_uncertainty(self):
        """Test size-based uncertainty calculation."""
        from app.services.active_learning import ActiveLearningService
        from app.models.schemas import LabelAnnotation, BoundingBox
        
        service = ActiveLearningService()
        
        # Very small object
        small_ann = LabelAnnotation(
            id=1,
            image_id="test",
            class_id=0,
            class_name="car",
            confidence=0.9,
            bbox=BoundingBox(x=0, y=0, width=5, height=5)  # 25 pixels
        )
        
        small_unc = service._compute_size_uncertainty(small_ann, (1920, 1080))
        
        # Very large object
        large_ann = LabelAnnotation(
            id=2,
            image_id="test",
            class_id=0,
            class_name="car",
            confidence=0.9,
            bbox=BoundingBox(x=0, y=0, width=1800, height=1000)
        )
        
        large_unc = service._compute_size_uncertainty(large_ann, (1920, 1080))
        
        # Small objects should have higher uncertainty than medium, large should also be higher
        assert small_unc > 0.5  # Very small
        assert large_unc > 0.3  # Very large


class TestDeploymentService:
    """Tests for deployment utilities."""
    
    def test_service_initialization(self):
        """Test that the service can be initialized."""
        from app.services.deployment import DeploymentService
        
        service = DeploymentService()
        assert service is not None
        assert service.inference_cache is not None
    
    def test_lru_cache(self):
        """Test LRU cache functionality."""
        from app.services.deployment import LRUCache
        
        cache = LRUCache(max_size=3)
        
        # Create test images
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        img3 = np.random.rand(100, 100, 3).astype(np.float32)
        img4 = np.random.rand(100, 100, 3).astype(np.float32)
        
        params = {"model": "test"}
        
        # Add items
        cache.put(img1, params, "result1")
        cache.put(img2, params, "result2")
        cache.put(img3, params, "result3")
        
        assert len(cache.cache) == 3
        
        # Get existing item
        result = cache.get(img1, params)
        assert result == "result1"
        
        # Add new item (should evict oldest)
        cache.put(img4, params, "result4")
        assert len(cache.cache) == 3
    
    def test_cache_stats(self):
        """Test cache statistics."""
        from app.services.deployment import LRUCache
        
        cache = LRUCache(max_size=10)
        
        img = np.random.rand(50, 50, 3).astype(np.float32)
        params = {"test": True}
        
        # Miss
        cache.get(img, params)
        assert cache.misses == 1
        
        # Put and hit
        cache.put(img, params, "cached")
        cache.get(img, params)
        assert cache.hits == 1
        
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        from app.services.deployment import DeploymentService
        
        service = DeploymentService()
        info = service.get_gpu_memory_usage()
        
        # Should return dict with at least 'available' key
        assert "available" in info


class TestHITLSchemas:
    """Tests for HITL-related schemas."""
    
    def test_annotation_status_enum(self):
        """Test annotation status enum values."""
        from app.models.schemas import AnnotationStatus
        
        assert AnnotationStatus.AUTO.value == "auto"
        assert AnnotationStatus.FLAGGED.value == "flagged"
        assert AnnotationStatus.APPROVED.value == "approved"
        assert AnnotationStatus.CORRECTED.value == "corrected"
    
    def test_refinement_type_enum(self):
        """Test refinement type enum values."""
        from app.models.schemas import RefinementType
        
        assert RefinementType.POINT_ADD.value == "point_add"
        assert RefinementType.BBOX_ADJUST.value == "bbox_adjust"
    
    def test_refinement_request_model(self):
        """Test refinement request model."""
        from app.models.schemas import RefinementRequest, RefinementType, PointPrompt
        
        request = RefinementRequest(
            refinement_type=RefinementType.POINT_ADD,
            points=[PointPrompt(x=100, y=100, label=1)]
        )
        
        assert request.refinement_type == RefinementType.POINT_ADD
        assert len(request.points) == 1
        assert request.points[0].x == 100
    
    def test_uncertainty_details_model(self):
        """Test uncertainty details model."""
        from app.models.schemas import UncertaintyDetails
        
        details = UncertaintyDetails(
            total=0.6,
            confidence=0.3,
            detection_mask_disagreement=0.2,
            semantic_instance_disagreement=0.1,
            size=0.0,
            overlap=0.0
        )
        
        assert details.total == 0.6


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Advanced Pipeline Component Tests")
    print("=" * 60)
    
    test_classes = [
        TestSemanticSegmentation,
        TestPanopticFusion,
        TestActiveLearning,
        TestDeploymentService,
        TestHITLSchemas
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(instance, method_name)
            
            try:
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if failed_tests:
        print("\nFailed tests:")
        for cls, method, error in failed_tests:
            print(f"  - {cls}.{method}: {error}")
    else:
        print("\n🎉 All tests passed!")
    
    print("=" * 60)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
