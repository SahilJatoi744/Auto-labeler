# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""Platform services for projects, governance, durable queues, and QA.

This module intentionally uses the Python standard library so the local app can
gain platform semantics without a dependency migration. The companion
``docs/postgres_schema.sql`` file defines the production Postgres shape.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from ..core.config import settings


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _loads(value: Optional[str]) -> Any:
    if not value:
        return {}
    return json.loads(value)


class PlatformService:
    """SQLite-backed local platform store.

    The service owns metadata that the current filesystem JSON model cannot
    represent cleanly: projects, lineage, audit events, queue state, model runs,
    metrics, export validations, and image preference/RLHF records.
    """

    def __init__(self, db_path: Optional[Path] = None):
        data_dir = settings.BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path or data_dir / "platform.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    project_id TEXT,
                    version_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lineage_events (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    version_id TEXT,
                    event_type TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_queue (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    locked_by TEXT,
                    locked_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_runs (
                    id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observability_metrics (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS export_validations (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    valid INTEGER NOT NULL,
                    issues_json TEXT NOT NULL,
                    statistics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preference_items (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    image_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preference_votes (
                    id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    selected_candidate_id TEXT NOT NULL,
                    rationale TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_reports (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quality_scores (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    issues_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _row(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        for key in list(data):
            if key.endswith("_json"):
                data[key[:-5]] = _loads(data.pop(key))
        if "valid" in data:
            data["valid"] = bool(data["valid"])
        return data

    def create_workspace(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        item = {"id": f"ws_{uuid4().hex[:12]}", "name": name, "description": description, "created_at": _now()}
        with self._session() as conn:
            conn.execute(
                "INSERT INTO workspaces (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (item["id"], item["name"], item["description"], item["created_at"]),
            )
        self.record_audit_event("workspace.create", "workspace", item["id"], {"name": name})
        return item

    def list_workspaces(self) -> List[Dict[str, Any]]:
        with self._session() as conn:
            return [self._row(r) for r in conn.execute("SELECT * FROM workspaces ORDER BY created_at DESC")]

    def create_project(self, workspace_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        item = {
            "id": f"prj_{uuid4().hex[:12]}",
            "workspace_id": workspace_id,
            "name": name,
            "description": description,
            "status": "active",
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO projects (id, workspace_id, name, description, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], workspace_id, name, description, item["status"], item["created_at"]),
            )
        self.record_audit_event("project.create", "project", item["id"], {"workspace_id": workspace_id, "name": name})
        return item

    def list_projects(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._session() as conn:
            if workspace_id:
                rows = conn.execute("SELECT * FROM projects WHERE workspace_id = ? ORDER BY created_at DESC", (workspace_id,))
            else:
                rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC")
            return [self._row(r) for r in rows]

    def create_dataset_version(
        self,
        dataset_id: str,
        project_id: Optional[str],
        version_name: str,
        source: str,
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        item = {
            "id": f"dsv_{uuid4().hex[:12]}",
            "dataset_id": dataset_id,
            "project_id": project_id,
            "version_name": version_name,
            "source": source,
            "manifest": manifest,
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO dataset_versions (id, dataset_id, project_id, version_name, source, manifest_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item["id"], dataset_id, project_id, version_name, source, _json(manifest), item["created_at"]),
            )
        self.record_audit_event("dataset.version.create", "dataset", dataset_id, {"version_id": item["id"]})
        return item

    def list_dataset_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,))
            return [self._row(r) for r in rows]

    def record_lineage(self, dataset_id: str, version_id: Optional[str], event_type: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "id": f"lin_{uuid4().hex[:12]}",
            "dataset_id": dataset_id,
            "version_id": version_id,
            "event_type": event_type,
            "inputs": inputs,
            "outputs": outputs,
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO lineage_events (id, dataset_id, version_id, event_type, inputs_json, outputs_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item["id"], dataset_id, version_id, event_type, _json(inputs), _json(outputs), item["created_at"]),
            )
        return item

    def list_lineage(self, dataset_id: str) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM lineage_events WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,))
            return [self._row(r) for r in rows]

    def record_audit_event(self, action: str, resource_type: str, resource_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "id": f"aud_{uuid4().hex[:12]}",
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata,
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO audit_events (id, action, resource_type, resource_id, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], action, resource_type, resource_id, _json(metadata), item["created_at"]),
            )
        return item

    def list_audit_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row(r) for r in rows]

    def enqueue_job(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "id": f"que_{uuid4().hex[:12]}",
            "task_type": task_type,
            "payload": payload,
            "status": "queued",
            "attempts": 0,
            "locked_by": None,
            "locked_at": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO worker_queue (id, task_type, payload_json, status, attempts, locked_by, locked_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item["id"], task_type, _json(payload), item["status"], 0, None, None, item["created_at"], item["updated_at"]),
            )
        return item

    def claim_next_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM worker_queue WHERE status = 'queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            now = _now()
            conn.execute(
                "UPDATE worker_queue SET status = 'running', attempts = attempts + 1, locked_by = ?, locked_at = ?, updated_at = ? WHERE id = ?",
                (worker_id, now, now, row["id"]),
            )
            claimed = conn.execute("SELECT * FROM worker_queue WHERE id = ?", (row["id"],)).fetchone()
            return self._row(claimed)

    def list_queue(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM worker_queue ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row(r) for r in rows]

    def complete_queue_job(self, queue_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        now = _now()
        with self._session() as conn:
            existing = conn.execute("SELECT * FROM worker_queue WHERE id = ?", (queue_id,)).fetchone()
            if not existing:
                raise ValueError(f"Queue job not found: {queue_id}")
            payload = _loads(existing["payload_json"])
            payload["result"] = result
            conn.execute(
                "UPDATE worker_queue SET status = 'completed', payload_json = ?, updated_at = ? WHERE id = ?",
                (_json(payload), now, queue_id),
            )
            row = conn.execute("SELECT * FROM worker_queue WHERE id = ?", (queue_id,)).fetchone()
            return self._row(row)

    def fail_queue_job(self, queue_id: str, error: str) -> Dict[str, Any]:
        now = _now()
        with self._session() as conn:
            existing = conn.execute("SELECT * FROM worker_queue WHERE id = ?", (queue_id,)).fetchone()
            if not existing:
                raise ValueError(f"Queue job not found: {queue_id}")
            payload = _loads(existing["payload_json"])
            payload["error"] = error
            conn.execute(
                "UPDATE worker_queue SET status = 'failed', payload_json = ?, updated_at = ? WHERE id = ?",
                (_json(payload), now, queue_id),
            )
            row = conn.execute("SELECT * FROM worker_queue WHERE id = ?", (queue_id,)).fetchone()
            return self._row(row)

    def record_model_run(self, model_name: str, task: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        item = {"id": f"mdl_{uuid4().hex[:12]}", "model_name": model_name, "task": task, "inputs": inputs, "outputs": outputs, "created_at": _now()}
        with self._session() as conn:
            conn.execute(
                "INSERT INTO model_runs (id, model_name, task, inputs_json, outputs_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], model_name, task, _json(inputs), _json(outputs), item["created_at"]),
            )
        return item

    def list_model_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM model_runs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row(r) for r in rows]

    def record_metric(self, name: str, value: float, labels: Dict[str, Any]) -> Dict[str, Any]:
        item = {"id": f"met_{uuid4().hex[:12]}", "name": name, "value": float(value), "labels": labels, "created_at": _now()}
        with self._session() as conn:
            conn.execute(
                "INSERT INTO observability_metrics (id, name, value, labels_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (item["id"], name, float(value), _json(labels), item["created_at"]),
            )
        return item

    def list_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM observability_metrics ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row(r) for r in rows]

    def validate_export(self, job_id: str, results: Iterable[Any]) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        image_count = 0
        annotation_count = 0
        class_counts: Dict[str, int] = {}
        for image in results:
            image_count += 1
            image_dict = image if isinstance(image, dict) else image.model_dump()
            annotations = image_dict.get("annotations", [])
            if not annotations:
                issues.append({"severity": "warning", "image_id": image_dict.get("image_id"), "message": "Image has no annotations"})
            for ann in annotations:
                annotation_count += 1
                bbox = ann.get("bbox")
                class_name = ann.get("class_name") or str(ann.get("class_id", "unknown"))
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if bbox and (bbox.get("width", 0) <= 0 or bbox.get("height", 0) <= 0):
                    issues.append({"severity": "error", "image_id": image_dict.get("image_id"), "message": "Annotation has invalid bbox"})
                if ann.get("confidence", 0) < 0 or ann.get("confidence", 0) > 1:
                    issues.append({"severity": "error", "image_id": image_dict.get("image_id"), "message": "Annotation confidence outside [0,1]"})
        valid = not any(issue["severity"] == "error" for issue in issues)
        statistics = {"images": image_count, "annotations": annotation_count, "classes": class_counts}
        item = {
            "id": f"val_{uuid4().hex[:12]}",
            "job_id": job_id,
            "valid": valid,
            "issues": issues,
            "statistics": statistics,
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO export_validations (id, job_id, valid, issues_json, statistics_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], job_id, 1 if valid else 0, _json(issues), _json(statistics), item["created_at"]),
            )
        return item

    def list_export_validations(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._session() as conn:
            if job_id:
                rows = conn.execute("SELECT * FROM export_validations WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            else:
                rows = conn.execute("SELECT * FROM export_validations ORDER BY created_at DESC LIMIT 100")
            return [self._row(r) for r in rows]

    def create_preference_item(self, project_id: Optional[str], image_id: str, prompt: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        item = {
            "id": f"pref_{uuid4().hex[:12]}",
            "project_id": project_id,
            "image_id": image_id,
            "prompt": prompt,
            "candidates": candidates,
            "status": "open",
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO preference_items (id, project_id, image_id, prompt, candidates_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item["id"], project_id, image_id, prompt, _json(candidates), item["status"], item["created_at"]),
            )
        return item

    def list_preference_items(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._session() as conn:
            if project_id:
                rows = conn.execute("SELECT * FROM preference_items WHERE project_id = ? ORDER BY created_at DESC", (project_id,))
            else:
                rows = conn.execute("SELECT * FROM preference_items ORDER BY created_at DESC LIMIT 100")
            return [self._row(r) for r in rows]

    def record_preference_vote(self, item_id: str, selected_candidate_id: str, rationale: Optional[str] = None) -> Dict[str, Any]:
        item = {"id": f"vote_{uuid4().hex[:12]}", "item_id": item_id, "selected_candidate_id": selected_candidate_id, "rationale": rationale, "created_at": _now()}
        with self._session() as conn:
            conn.execute(
                "INSERT INTO preference_votes (id, item_id, selected_candidate_id, rationale, created_at) VALUES (?, ?, ?, ?, ?)",
                (item["id"], item_id, selected_candidate_id, rationale, item["created_at"]),
            )
            conn.execute("UPDATE preference_items SET status = 'voted' WHERE id = ?", (item_id,))
        return item

    def list_preference_votes(self, item_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._session() as conn:
            if item_id:
                rows = conn.execute("SELECT * FROM preference_votes WHERE item_id = ? ORDER BY created_at DESC", (item_id,))
            else:
                rows = conn.execute("SELECT * FROM preference_votes ORDER BY created_at DESC LIMIT 100")
            return [self._row(r) for r in rows]

    def save_evaluation_report(self, job_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
        item = {"id": f"eval_{uuid4().hex[:12]}", "job_id": job_id, "report": report, "created_at": _now()}
        with self._session() as conn:
            conn.execute(
                "INSERT INTO evaluation_reports (id, job_id, report_json, created_at) VALUES (?, ?, ?, ?)",
                (item["id"], job_id, _json(report), item["created_at"]),
            )
        return item

    def list_evaluation_reports(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._session() as conn:
            if job_id:
                rows = conn.execute("SELECT * FROM evaluation_reports WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            else:
                rows = conn.execute("SELECT * FROM evaluation_reports ORDER BY created_at DESC LIMIT 100")
            return [self._row(r) for r in rows]

    def save_quality_score(self, job_id: str, image_id: str, score: float, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        item = {
            "id": f"qsc_{uuid4().hex[:12]}",
            "job_id": job_id,
            "image_id": image_id,
            "score": float(score),
            "issues": issues,
            "created_at": _now(),
        }
        with self._session() as conn:
            conn.execute(
                "INSERT INTO quality_scores (id, job_id, image_id, score, issues_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], job_id, image_id, float(score), _json(issues), item["created_at"]),
            )
        return item

    def list_quality_scores(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._session() as conn:
            if job_id:
                rows = conn.execute("SELECT * FROM quality_scores WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            else:
                rows = conn.execute("SELECT * FROM quality_scores ORDER BY created_at DESC LIMIT 100")
            return [self._row(r) for r in rows]


platform_service = PlatformService()


def get_platform_service() -> PlatformService:
    return platform_service
