# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import BoundingBox, ImageLabels, LabelAnnotation, TaskType
from app.services.intelligence import get_intelligence_service


class IntelligenceServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = get_intelligence_service()

    def test_recommends_current_image_models_for_segmentation(self):
        recommendations = self.service.recommend_models(
            task_type=TaskType.INSTANCE_SEGMENTATION.value,
            class_names=["yellow school bus", "person"],
            device="gpu",
            limit=5,
        )

        ids = [item["id"] for item in recommendations]
        self.assertIn("sam3_concept", ids)
        self.assertIn("yolo26_sam2_hybrid", ids)
        self.assertGreater(recommendations[0]["score"], recommendations[-1]["score"])
        self.assertTrue(all("why" in item for item in recommendations))

    def test_scores_annotation_quality_and_flags_review_issues(self):
        image = ImageLabels(
            image_id="image-1",
            annotations=[
                LabelAnnotation(
                    id=1,
                    image_id="image-1",
                    class_id=1,
                    class_name="car",
                    confidence=0.25,
                    bbox=BoundingBox(x=0, y=0, width=20, height=20),
                ),
                LabelAnnotation(
                    id=2,
                    image_id="image-1",
                    class_id=1,
                    class_name="car",
                    confidence=0.9,
                    bbox=BoundingBox(x=1, y=1, width=20, height=20),
                ),
            ],
        )

        score = self.service.score_image_quality(image, TaskType.INSTANCE_SEGMENTATION.value)

        self.assertLess(score["score"], 90)
        self.assertEqual(score["review_priority"], "high")
        issue_codes = {issue["code"] for issue in score["issues"]}
        self.assertIn("low_confidence", issue_codes)
        self.assertIn("missing_segmentation", issue_codes)
        self.assertIn("duplicate_overlap", issue_codes)

    def test_evaluates_job_quality_with_dataset_level_summary(self):
        results = [
            ImageLabels(
                image_id="image-1",
                annotations=[
                    LabelAnnotation(
                        id=1,
                        image_id="image-1",
                        class_id=1,
                        class_name="car",
                        confidence=0.9,
                        bbox=BoundingBox(x=0, y=0, width=20, height=20),
                    )
                ],
            ),
            ImageLabels(image_id="image-2", annotations=[]),
        ]

        report = self.service.evaluate_job_quality(
            job_id="job-1",
            task_type=TaskType.OBJECT_DETECTION.value,
            results=results,
            class_names=["car", "person"],
        )

        self.assertEqual(report["job_id"], "job-1")
        self.assertEqual(report["summary"]["image_count"], 2)
        self.assertEqual(report["summary"]["empty_images"], 1)
        self.assertIn("car", report["summary"]["class_distribution"])
        self.assertGreater(len(report["recommended_actions"]), 0)

    def test_summarizes_dataset_health(self):
        health = self.service.summarize_dataset_health(
            dataset_id="dataset-1",
            metadata={
                "dataset_id": "dataset-1",
                "total_images": 10,
                "valid_images": 8,
                "corrupted_images": 2,
                "formats": {"jpg": 8, "png": 2},
            },
            versions=[{"id": "v1"}],
            lineage=[{"id": "lin1"}],
        )

        self.assertEqual(health["dataset_id"], "dataset-1")
        self.assertLess(health["health_score"], 100)
        self.assertIn("recommendations", health)


if __name__ == "__main__":
    unittest.main()
