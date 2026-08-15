# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import unittest
from pathlib import Path
import sys
import gc
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.platform import PlatformService


class PlatformServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(__file__).parent.parent / "data" / f"platform_test_{uuid4().hex}"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.tmpdir / "platform.db"
        self.service = PlatformService(self.db_path)

    def tearDown(self):
        del self.service
        gc.collect()
        for _ in range(5):
            try:
                if self.db_path.exists():
                    self.db_path.unlink()
                if self.tmpdir.exists():
                    self.tmpdir.rmdir()
                return
            except PermissionError:
                time.sleep(0.05)

    def test_project_workspace_dataset_version_and_lineage_are_persisted(self):
        workspace = self.service.create_workspace("Vision QA")
        project = self.service.create_project(workspace["id"], "Road Images")
        version = self.service.create_dataset_version(
            dataset_id="dataset-1",
            project_id=project["id"],
            version_name="v1",
            source="upload",
            manifest={"images": 12},
        )
        event = self.service.record_lineage(
            dataset_id="dataset-1",
            version_id=version["id"],
            event_type="preprocess",
            inputs={"raw": "dataset-1"},
            outputs={"valid_images": 12},
        )

        self.assertEqual(project["workspace_id"], workspace["id"])
        self.assertEqual(version["dataset_id"], "dataset-1")
        self.assertEqual(event["event_type"], "preprocess")
        self.assertEqual(len(self.service.list_projects()), 1)
        self.assertEqual(len(self.service.list_dataset_versions("dataset-1")), 1)
        self.assertEqual(len(self.service.list_lineage("dataset-1")), 1)

    def test_audit_queue_model_gateway_observability_export_and_rlhf(self):
        audit = self.service.record_audit_event("dataset.upload", "dataset", "dataset-1", {"ok": True})
        queue_item = self.service.enqueue_job("labeling", {"job_id": "job-1"})
        claimed = self.service.claim_next_job("worker-1")
        completed = self.service.complete_queue_job(claimed["id"], {"status": "manual_start_required"})
        model_run = self.service.record_model_run("yolo", "label", {"dataset": "dataset-1"}, {"items": 2})
        metric = self.service.record_metric("jobs.completed", 1, {"project": "demo"})
        validation = self.service.validate_export(
            "job-1",
            [
                {
                    "image_id": "img-1",
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": "img-1",
                            "class_id": 1,
                            "class_name": "car",
                            "confidence": 0.9,
                            "bbox": {"x": 0, "y": 0, "width": 20, "height": 30},
                            "iscrowd": False,
                        }
                    ],
                }
            ],
        )
        pref = self.service.create_preference_item(
            project_id="project-1",
            image_id="image-1",
            prompt="Which mask is better?",
            candidates=[{"id": "a"}, {"id": "b"}],
        )
        pref_vote = self.service.record_preference_vote(pref["id"], "b", "cleaner mask")
        quality_report = self.service.save_evaluation_report(
            "job-1",
            {"summary": {"image_count": 1, "avg_quality_score": 98}},
        )
        quality_score = self.service.save_quality_score(
            "job-1",
            "img-1",
            98,
            [{"code": "ok", "severity": "info"}],
        )

        self.assertEqual(audit["action"], "dataset.upload")
        self.assertEqual(claimed["id"], queue_item["id"])
        self.assertEqual(claimed["status"], "running")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(model_run["model_name"], "yolo")
        self.assertEqual(metric["name"], "jobs.completed")
        self.assertTrue(validation["valid"])
        self.assertEqual(pref_vote["selected_candidate_id"], "b")
        self.assertEqual(quality_report["job_id"], "job-1")
        self.assertEqual(quality_score["image_id"], "img-1")
        self.assertEqual(len(self.service.list_evaluation_reports("job-1")), 1)
        self.assertEqual(len(self.service.list_quality_scores("job-1")), 1)
        self.assertEqual(len(self.service.list_preference_items("project-1")), 1)


if __name__ == "__main__":
    unittest.main()
