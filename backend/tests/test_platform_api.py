# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import asyncio
import gc
import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api import platform as platform_api
from app.core.config import settings
from app.models.schemas import ActiveLearningConfig, BoundingBox, ImageLabels, LabelAnnotation
from app.services.platform import PlatformService


class FakeLabelingService:
    def __init__(self):
        self.job = SimpleNamespace(
            id="job-1",
            dataset_id="dataset-1",
            task_type=SimpleNamespace(value="object_detection"),
            class_hierarchy=SimpleNamespace(classes=[
                SimpleNamespace(name="car"),
                SimpleNamespace(name="person"),
            ]),
        )
        self.results = [
            ImageLabels(
                image_id="image-1",
                annotations=[
                    LabelAnnotation(
                        id=1,
                        image_id="image-1",
                        class_id=1,
                        class_name="car",
                        confidence=0.2,
                        bbox=BoundingBox(x=0, y=0, width=20, height=30),
                    )
                ],
            )
        ]

    def get_job(self, job_id):
        return self.job if job_id == "job-1" else None

    def get_results(self, job_id):
        return self.results if job_id == "job-1" else None


class PlatformApiTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(__file__).parent.parent / "data" / f"platform_api_test_{uuid4().hex}"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.tmpdir / "platform.db"
        self.original_platform = platform_api.platform
        self.original_labeling = platform_api.labeling_service
        platform_api.platform = PlatformService(self.db_path)
        platform_api.labeling_service = FakeLabelingService()
        self.dataset_meta_path = settings.UPLOAD_DIR / "dataset-1_metadata.json"
        self.dataset_meta_path.write_text(
            json.dumps(
                {
                    "dataset_id": "dataset-1",
                    "name": "API test dataset",
                    "total_images": 2,
                    "valid_images": 1,
                    "corrupted_images": 1,
                    "formats": {"jpg": 2},
                }
            )
        )

    def tearDown(self):
        platform_api.platform = self.original_platform
        platform_api.labeling_service = self.original_labeling
        gc.collect()
        for _ in range(5):
            try:
                if self.db_path.exists():
                    self.db_path.unlink()
                if self.dataset_meta_path.exists():
                    self.dataset_meta_path.unlink()
                if self.tmpdir.exists():
                    self.tmpdir.rmdir()
                return
            except PermissionError:
                time.sleep(0.05)

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_project_dataset_lineage_audit_queue_and_model_endpoints(self):
        workspace = self.run_async(platform_api.create_workspace({"name": "Vision QA"}))
        project = self.run_async(platform_api.create_project({"workspace_id": workspace["id"], "name": "Road Images"}))
        version = self.run_async(
            platform_api.create_dataset_version(
                "dataset-1",
                {"project_id": project["id"], "version_name": "v1", "source": "upload", "manifest": {"images": 1}},
            )
        )
        platform_api.platform.record_lineage("dataset-1", version["id"], "preprocess", {}, {"valid_images": 1})
        queue_item = platform_api.platform.enqueue_job("labeling", {"job_id": "job-1"})
        platform_api.platform.record_model_run("yolo", "label", {"job": "job-1"}, {"annotations": 1})
        platform_api.platform.record_metric("jobs.completed", 1, {"job": "job-1"})

        self.assertEqual(len(self.run_async(platform_api.list_workspaces())), 1)
        self.assertEqual(len(self.run_async(platform_api.list_projects())), 1)
        self.assertEqual(len(self.run_async(platform_api.list_dataset_versions("dataset-1"))), 1)
        self.assertEqual(len(self.run_async(platform_api.list_lineage("dataset-1"))), 1)
        self.assertEqual(self.run_async(platform_api.claim_worker_job("worker-1"))["id"], queue_item["id"])
        self.assertEqual(len(self.run_async(platform_api.list_model_runs())), 1)
        self.assertEqual(len(self.run_async(platform_api.list_metrics())), 1)
        self.assertGreaterEqual(len(self.run_async(platform_api.list_audit_events())), 2)

    def test_review_active_learning_export_validation_and_preference_endpoints(self):
        validation = self.run_async(platform_api.validate_export("job-1"))
        selection = self.run_async(
            platform_api.active_learning_select(
                "job-1",
                ActiveLearningConfig(strategy="uncertainty", n_samples=5, min_uncertainty=0.0),
            )
        )
        preference = self.run_async(
            platform_api.create_preference_item(
                {
                    "project_id": "project-1",
                    "image_id": "image-1",
                    "prompt": "Which annotation is better?",
                    "candidates": [{"id": "candidate_a"}, {"id": "candidate_b"}],
                }
            )
        )
        vote = self.run_async(
            platform_api.record_preference_vote(
                preference["id"],
                {"selected_candidate_id": "candidate_b", "rationale": "better boundary"},
            )
        )

        self.assertTrue(validation["valid"])
        self.assertEqual(selection["selected_image_ids"], ["image-1"])
        self.assertEqual(len(self.run_async(platform_api.list_export_validations("job-1"))), 1)
        self.assertEqual(len(self.run_async(platform_api.list_preference_items("project-1"))), 1)
        self.assertEqual(vote["selected_candidate_id"], "candidate_b")
        self.assertEqual(len(self.run_async(platform_api.list_preference_votes(preference["id"]))), 1)

    def test_model_recommendations_quality_reports_dataset_health_and_worker_run(self):
        catalog = self.run_async(platform_api.get_model_catalog())
        integrations = self.run_async(platform_api.get_model_integration_status())
        prepare = self.run_async(platform_api.prepare_model_integration("sam3", {"allow_download": False}))
        recommendation = self.run_async(
            platform_api.recommend_models(
                {
                    "task_type": "instance_segmentation",
                    "class_names": ["car", "person"],
                    "device": "gpu",
                    "limit": 4,
                }
            )
        )
        report = self.run_async(platform_api.evaluate_job_quality("job-1"))
        scores = self.run_async(platform_api.list_quality_scores("job-1"))
        reports = self.run_async(platform_api.list_quality_reports("job-1"))
        health = self.run_async(platform_api.get_dataset_health("dataset-1"))

        self.assertGreaterEqual(len(catalog["models"]), 5)
        self.assertIn("sam3", integrations)
        self.assertEqual(prepare["model_key"], "sam3")
        self.assertIn("sam3_concept", [item["id"] for item in recommendation["recommendations"]])
        self.assertEqual(report["job_id"], "job-1")
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(reports), 1)
        self.assertEqual(health["dataset_id"], "dataset-1")

        platform_api.platform.enqueue_job("quality_evaluation", {"job_id": "job-1"})
        worker_result = self.run_async(platform_api.run_next_worker_job("worker-test"))
        self.assertEqual(worker_result["status"], "completed")
        self.assertEqual(worker_result["task_type"], "quality_evaluation")


if __name__ == "__main__":
    unittest.main()
