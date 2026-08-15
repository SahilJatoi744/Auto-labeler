# 🏷️ AutoLabeler - Complete Start-to-End Usage Guide

Welcome to **AutoLabeler**, a state-of-the-art, AI-powered automatic image dataset labeling platform. This application integrates cutting-edge Object Detection (YOLO, YOLO-World) and Instance/Semantic Segmentation (SAM, SAM 2, SAM 3 Concept Segmentation) to automate and refine your dataset annotation workflows.

This guide walks you through the entire lifecycle—from launching the servers to running automated quality audits, human-in-the-loop (HITL) reviews, active learning sampling, and final COCO/YOLO/Pascal VOC exporting.

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Quick Start (Launching the App)](#2-quick-start-launching-the-app)
3. [Start-to-End Platform Workflow](#3-start-to-end-platform-workflow)
   * [Step 1: Create a Project & Workspace](#step-1-create-a-project--workspace)
   * [Step 2: Upload your Image Dataset](#step-2-upload-your-image-dataset)
   * [Step 3: Monitor Model Readiness](#step-3-monitor-model-readiness)
   * [Step 4: Configure and Launch a Labeling Job](#step-4-configure-and-launch-a-labeling-job)
   * [Step 5: Review and Correct Annotations (HITL Editor)](#step-5-review-and-correct-annotations-hitl-editor)
   * [Step 6: AI Governance, active learning, and Quality Check](#step-6-ai-governance-active-learning-and-quality-check)
   * [Step 7: Capture Image Preferences (RLHF)](#step-7-capture-image-preferences-rlhf)
   * [Step 8: Validate and Export Your Dataset](#step-8-validate-and-export-your-dataset)
4. [Optional Advanced Model Integrations (SAM 3, Grounding DINO, DINOv3)](#4-optional-advanced-model-integrations)
5. [Automated Verification (E2E API Test)](#5-automated-verification-e2e-api-test)
6. [Performance Tuning & Troubleshooting](#6-performance-tuning--troubleshooting)

---

## 1. Prerequisites

Before running the application, make sure your system meets these requirements:
* **OS**: Windows 10 or 11
* **Runtime**: Python 3.10+ and Node.js 18+
* **GPU (Recommended)**: NVIDIA GPU with CUDA drivers configured (the system automatically falls back to CPU if no GPU is available).

---

## 2. Quick Start (Launching the App)

AutoLabeler consists of a fast **FastAPI backend** and a beautiful **Vite + React frontend**.

### Option A: Using Startup Scripts (Easiest)
Simply run the batch scripts created in the project root:
1. **Backend**: Double-click `start_backend.bat` (launches server on port `8000`)
2. **Frontend**: Double-click `start_frontend.bat` (launches web app on port `5173`)
3. **Control Center**: Run `QUICKSTART.bat` for quick configuration steps.

### Option B: Manual Startup

#### Terminal 1: Launch Backend
```cmd
cd backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
* **API Documentation**: Open your browser to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore all Swagger endpoints.

#### Terminal 2: Launch Frontend
```cmd
cd frontend
npm run dev
```
* **Web UI Interface**: Open your browser to [http://localhost:5173](http://localhost:5173).

---

## 3. Start-to-End Platform Workflow

Follow this complete loop to label, analyze, review, and export your image datasets.

### Step 1: Create a Project & Workspace
1. Select **Platform** from the sidebar navigation.
2. In the **Projects** tab, enter a **Workspace Name** and a **Project Name** (e.g., `autonomous_vehicles_v1`).
3. Click **Create Project**.
4. The system registers the project in the SQLite database and logs a system audit event.

### Step 2: Upload your Image Dataset
1. Go to the **Upload Dataset** page.
2. Enter an optional dataset name or description.
3. Drag & drop a `.zip` archive containing your images.
4. AutoLabeler automatically runs preprocessing:
   * Validates image headers and formats (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`).
   * Generates fast image hashes to detect and prevent duplicate uploads.
   * Tracks dataset versioning and registers a **Dataset Lineage** event.
5. Note the returned **Dataset ID** for your labeling jobs.

### Step 3: Monitor Model Readiness
1. On the **Dashboard**, check the **AI Model Readiness** widget.
2. The application supports standard auto-downloading for core weights (like `yolov8x-seg.pt` and `sam2_l.pt`).
3. Ensure the models show status **"Ready"** before initiating a labeling job.

### Step 4: Configure and Launch a Labeling Job
1. Navigate to the **Labeling Job** page.
2. Select your uploaded dataset from the dropdown.
3. **Define Classes**: Enter the names of objects you want to label (e.g., `car`, `person`, `traffic_light`). You can also customize display colors.
4. **Select Configuration**:
   * **Task Type**: Select *Object Detection* (bounding boxes), *Instance Segmentation* (polygons), or *Semantic Segmentation* (full-pixel mask).
   * **Strategy**: Select **AI-Assisted** (fastest, uses combined YOLO+SAM local pipelines).
   * **Device**: Select **Auto** (prioritizes CUDA GPU), **GPU**, or **CPU**.
   * **Confidence Threshold**: Set a threshold (e.g., `0.5`). Predictions below this value will be flagged for review, while predictions above the threshold are automatically accepted.
5. Click **Create Labeling Job**.
6. Switch back to the **Dashboard** and click **Start** on the new job. Monitor the processing progress bar in real-time.

### Step 5: Review and Correct Annotations (HITL Editor)
Once the job completes or while it's in progress, you can review and edit annotations:
1. Click **View** or **Human Review** for the completed job.
2. The interactive Canvas editor loads. You can:
   * View AI-generated bounding boxes, masks, and polygons overlayed on images.
   * Edit coordinates by dragging points, resizing boxes, or drawing new polygons.
   * Modify object classes, delete inaccurate boxes, or undo/redo edits.
   * Enter natural-language prompts to execute **interactive AI refinement** (e.g., typing "refine wheels").
3. Click **Save** to commit changes to the backend.

### Step 6: AI Governance, active learning, and Quality Check
Make sure your labels are of high scientific quality using the governance system:
1. Go to **Platform → Review**.
2. Click **Load Review Queue** for your job. This displays items flagged as high uncertainty.
3. Click **Active Learning Sampling** to automatically retrieve the most informative images—prioritizing human labeling where the model was least certain.
4. Navigate to **Platform → Intelligence**:
   * Click **Check Dataset Health** to score dataset corruption, deduplication, and format integrity.
   * Click **Evaluate Quality** to run the local annotation QA agent. The agent flags overlapping boxes, missing segmentation masks, geometry errors, and class mismatches.

### Step 7: Capture Image Preferences (RLHF)
For edge cases where two annotations are possible:
1. Open **Platform → RLHF**.
2. Enter an Image ID and preference prompt.
3. Click **Create Preference** to generate a side-by-side evaluation comparison.
4. Vote for **Candidate A** or **Candidate B** to record preference data, ideal for downstream fine-tuning and model alignment.

### Step 8: Validate and Export Your Dataset
1. Go to the **Export** page.
2. Select your completed labeling job.
3. Choose your desired output format:
   * **COCO JSON**: Standard format for most computer vision pipelines.
   * **YOLO**: Normalised `.txt` annotations, ready to train YOLO models immediately.
   * **Pascal VOC**: Structured `.xml` metadata.
4. Configure **Dataset Splits** (e.g., 70% Train, 15% Val, 15% Test).
5. Click **Export Dataset** to download a beautifully packaged `.zip` containing organized images and annotation folders.

---

## 4. Optional Advanced Model Integrations

AutoLabeler supports modular adapters for advanced computer vision workloads:

### SAM 3 Concept Segmentation
* **Use Case**: Segment images using open-vocabulary descriptions (e.g., "blue backpack").
* **Setup**: Place an approved `sam3.pt` weights file into `backend/models/sam3.pt`.

### Grounding DINO
* **Use Case**: Perform zero-shot object detection using text prompts.
* **Setup**: Ensure dependencies `transformers`, `accelerate`, and `safetensors` are installed. Toggle `ALLOW_MODEL_DOWNLOADS=true` in `config.py` to auto-cache model weights.

---

## 5. Running Automated Verification (E2E API Test)

To verify the entire backend pipeline (Upload ➔ AI Labeling ➔ Quality Check ➔ Export) programmatically without using the UI, run the End-to-End API test:

```cmd
backend\venv\Scripts\python.exe scripts\e2e_api_test.py
```

A successful run output will look like this:
```text
--- Starting API-Level E2E Test ---
Health Check: 200 - healthy
Uploading Dataset...
Upload Success: Dataset ID = 5b326397-cc3, Valid Images = 2
Creating Labeling Job...
Job Created: Job ID = 8e385b753b665a84
Starting Job...
Start Response: Job started
Monitoring Progress...
Status: completed | Processed: 2/2
Results found at: backend\outputs\8e385b753b665a84_results.json
Exporting results to COCO...
Export Success: backend\outputs\8e385b753b665a84_coco_2026181040.zip
```

---

## 6. Performance Tuning & Troubleshooting

### 🛑 Error: `[Errno 10048] address already in use`
* **Cause**: Another service is running on port 8000.
* **Fix**: Find and terminate the process:
  ```cmd
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```

### ⚡ CUDA Out of Memory (OOM)
* **Cause**: Your GPU ran out of VRAM processing high-resolution images.
* **Fix**: Edit `backend/app/core/config.py` and adjust memory settings:
  ```python
  BATCH_SIZE = 8       # Reduce from 16
  MAX_IMAGE_SIZE = 1024 # Reduce from 2048
  ```

### 🐢 Labeling is Slow
* **Fix**: Ensure your NVIDIA CUDA toolkit is installed. Check **Dashboard -> System Status** to confirm GPU acceleration status is active. If running on CPU, consider using smaller model weights in `config.py`.

### 📂 File Logs
Review real-time operations, warnings, and tracebacks inside:
* `backend/logs/autolabeler.log`
