# AutoLabeler Platform: Start-to-End Usage Guide

This guide walks through the full image annotation workflow, including the new platform features: projects, dataset versioning, lineage, review queues, active learning, image preference/RLHF records, audit events, durable worker records, model runs, observability, and export validation.

## 1. Start the Application

### Backend

```cmd
cd C:\Users\Sahil\Documents\Labeling_application\auto-labeler\backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend API docs are available at:

```text
http://localhost:8000/docs
```

### Frontend

```cmd
cd C:\Users\Sahil\Documents\Labeling_application\auto-labeler\frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

## 2. Confirm System Readiness

Open **Dashboard**.

Check:

- System status is healthy.
- Device mode is correct: Auto, GPU, or CPU.
- AI model readiness shows available/downloaded models.
- Existing labeling jobs are visible in Recent Labeling Jobs.

If no GPU is available, the app still works, but image labeling will be slower.

## 2.1 Optional Advanced Local Model Setup

The app runs locally with the existing YOLO / YOLO-World / SAM2 pipeline. Advanced 2026 model profiles are now wired as optional adapters and activate only when the local environment is ready.

### SAM 3 Concept Segmentation

Use this when you want text-prompted segmentation such as "yellow school bus" or "person wearing red shirt".

Requirements:

- `ultralytics >= 8.3.237`
- Approved `sam3.pt` weight file
- Place the file at:

```text
C:\Users\Sahil\Documents\Labeling_application\auto-labeler\backend\models\sam3.pt
```

SAM 3 weights are not automatically downloaded by Ultralytics. Request/download the weight from the official model page, then place it in `backend\models`.

### Grounding DINO

Use this for open-set detection before SAM2 mask refinement.

Requirements:

- `transformers`
- `accelerate`
- `safetensors`
- Local Hugging Face cache for `IDEA-Research/grounding-dino-base`, or set `ALLOW_MODEL_DOWNLOADS=true`.

### DINOv3

Use this for dataset intelligence and future feature-based quality workflows, not direct labeling.

Requirements:

- `transformers`
- Local Hugging Face cache for `facebook/dinov3-vits16-pretrain-lvd1689m`, or set `ALLOW_MODEL_DOWNLOADS=true`.

Check readiness in **Platform -> Intelligence -> Local Integration Status**.

## 3. Create a Workspace and Project

Open **Platform** from the sidebar.

In the **Projects** tab:

1. Enter a workspace name.
2. Enter a project name.
3. Click **Create Project**.

The project is stored in the local platform database and audit events are recorded automatically.

Production Postgres schema reference:

```text
docs/postgres_schema.sql
```

## 4. Upload an Image Dataset

Open **Upload Dataset**.

1. Enter an optional dataset name.
2. Upload a ZIP file containing images.
3. Wait for validation to complete.

Supported image formats:

- JPG / JPEG
- PNG
- BMP
- TIFF
- WebP

After upload, the system records:

- Dataset metadata.
- Dataset version `v1`.
- Lineage event for upload/preprocessing.
- Audit event.
- Observability metric.

## 5. Review Dataset Versioning and Lineage

Open **Platform → Projects**.

1. Select the uploaded dataset.
2. Review **Dataset Versioning**.
3. Review **Lineage**.

Use this to confirm how a dataset moved from upload into validated platform state.

## 6. Create an Image Labeling Job

Open **Labeling Job**.

1. Select the dataset.
2. Define classes manually or load a template.
3. Choose task type:
   - Object Detection
   - Instance Segmentation
   - Semantic Segmentation
4. Choose strategy:
   - AI-Assisted
   - Rule-Based
   - Hybrid
5. Select device:
   - Auto
   - GPU
   - CPU
6. Review the recommended model profile.
   - **SAM 3 Concept Segmentation** is recommended for open-vocabulary mask-first work when a SAM 3 runtime is installed.
   - **YOLO plus SAM2 Hybrid** is the strongest local default for production image labeling.
   - **YOLO-World plus SAM2** is useful for custom class prompts that are not covered by standard detector labels.
   - **DINOv3 Dataset Intelligence** is listed for dataset quality and active learning, not direct labeling.
7. Set confidence threshold.
8. Click **Create Labeling Job**.

The system records:

- Job metadata.
- Durable worker queue record.
- Audit event.
- Selected model profile and top model recommendations.

## 7. Run and Monitor the Job

Open **Dashboard**.

1. Find the created job.
2. Click **Start**.
3. Monitor progress.

When the job completes, the system records:

- Model gateway run metadata.
- Job completion metric.
- Audit event.
- Labeling results in local output storage.

## 8. Inspect and Correct Annotations

From **Dashboard**, click **View** on a completed job.

The Annotation Editor supports:

- Bounding boxes.
- Polygons.
- Mask visualization.
- Class reassignment.
- Annotation delete.
- Manual annotation creation.
- Undo / redo.
- Zoom / fit to screen.
- Prompt-based AI refinement.

Click **Save** after editing.

Saved edits update the job results and record audit metadata.

## 9. Use Review Queue and Active Learning

Open **Platform → Review**.

1. Select a completed job.
2. Click **Load Review Queue**.

The review queue displays flagged annotations based on uncertainty and confidence signals.

The active learning selection returns the most informative image IDs for human review. This is useful when you want to spend human effort on the samples most likely to improve label quality.

## 10. Use Dataset and Annotation Intelligence

Open **Platform -> Intelligence**.

Use this page to operate the image-only AI governance layer:

1. Review the **Model Catalog** to see local and optional model-gateway profiles.
2. Review **Local Integration Status** to see whether SAM 3, Grounding DINO, and DINOv3 can execute locally.
3. Select a job and click **Evaluate Quality** to run the annotation quality agent.
4. Click **Queue Quality Job** to add a durable worker task for later execution.
5. Select a dataset and click **Check Dataset Health** to inspect corruption, versioning, lineage, and readiness signals.
6. Click **Run Next Worker Job** to execute the next queued platform task.

Quality evaluation records:

- Per-job evaluation report.
- Per-image quality scores.
- Low-confidence, missing-segmentation, invalid-geometry, duplicate-overlap, and uncertainty issues.
- Observability metrics.
- Audit events.
- Model-gateway run metadata for the quality agent.

## 11. Create Image Preference / RLHF Items

Open **Platform → RLHF**.

Use this when you want preference-style feedback for image annotation quality, such as choosing between two candidate masks or label sets.

1. Enter an image ID.
2. Enter a preference prompt.
3. Click **Create Preference**.
4. In the preference queue, vote for Candidate A or Candidate B.

The system records:

- Preference item.
- Preference vote.
- Rationale.

This is image-focused RLHF/preference data. Text/audio/document/RLHF chat annotation workflows are intentionally not included.

## 12. Validate and Export

Open **Export**.

1. Select a completed labeling job.
2. Choose export format:
   - COCO JSON
   - Pascal VOC XML
   - YOLO
3. Set train/validation/test ratios.
4. Click **Export Dataset**.
5. Download the ZIP.

The backend automatically records export validation metadata.

You can also open **Platform → Review** and click **Validate Export** for the selected job.

## 13. Monitor Platform Operations

Open **Platform → Ops**.

This page shows:

- Durable Worker Queue
- Model Gateway Runs
- Observability Metrics
- Export Validations
- Audit Events

These records help debug and govern the image annotation pipeline.

## 14. Current Scope

Included:

- Image annotation.
- Projects/workspaces.
- Local platform metadata database.
- Postgres production schema.
- Durable worker records.
- Review queue.
- Active learning UX.
- Dataset versioning and lineage.
- Image preference/RLHF records.
- Audit events.
- Model gateway run records.
- Model catalog and recommendation profiles for SAM 3, YOLO plus SAM2, YOLO-World plus SAM2, Grounding DINO 1.5, DINOv3, and SAM2.
- Annotation quality agent.
- Dataset health scoring.
- Durable worker execution for quality and validation tasks.
- Observability metrics.
- Export validation.

Intentionally not included:

- Authentication, users, teams, RBAC.
- Embedding search.
- Video annotation.
- NLP annotation.
- Audio annotation.
- Document annotation.

## 15. Recommended Operating Loop

1. Create project.
2. Upload image dataset.
3. Confirm version and lineage.
4. Check dataset health.
5. Create labeling job with a recommended model profile.
6. Run auto-labeling.
7. Evaluate annotation quality.
8. Review uncertain and high-priority samples.
9. Correct labels in Annotation Editor.
10. Capture preference votes for hard cases.
11. Validate export.
12. Export dataset.
13. Review metrics, audit, worker, quality, and model run records.
