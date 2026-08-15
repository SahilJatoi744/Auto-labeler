# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import requests
import time
import os
import json

BASE_URL = "http://localhost:8000/api/v1"
# Resolve path to test_dataset.zip dynamically relative to script location
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_dataset.zip"))

def test_flow():
    print("--- Starting API-Level E2E Test ---")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {resp.status_code} - {resp.json()['status']}")
    except Exception as e:
        print(f"Health Check failed: {e}")
        return

    # 2. Upload Dataset
    print("\nUploading Dataset...")
    with open(DATASET_PATH, "rb") as f:
        files = {"file": ("test_dataset.zip", f, "application/zip")}
        data = {"dataset_name": "API_Test_Dataset"}
        resp = requests.post(f"{BASE_URL}/datasets/upload", files=files, data=data)
    
    if resp.status_code != 200:
        print(f"Upload failed: {resp.status_code} - {resp.text}")
        return
    
    dataset_info = resp.json()
    dataset_id = dataset_info["id"]
    print(f"Upload Success: Dataset ID = {dataset_id}, Valid Images = {dataset_info['valid_images']}")

    # 3. Check Models Status
    resp = requests.get(f"{BASE_URL}/models/status")
    print(f"\nModels Status: {json.dumps(resp.json(), indent=2)}")

    # 4. Create Labeling Job
    print("\nCreating Labeling Job...")
    job_config = {
        "dataset_id": dataset_id,
        "task_type": "object_detection",
        "class_hierarchy": {
            "classes": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"}
            ]
        },
        "strategy": "ai_assisted",
        "confidence_threshold": 0.3
    }
    # Send as JSON body (JobCreate schema)
    resp = requests.post(f"{BASE_URL}/jobs", json=job_config)

    if resp.status_code != 200:
        print(f"Job creation failed: {resp.status_code} - {resp.text}")
        return
    
    job_info = resp.json()
    job_id = job_info["id"]
    print(f"Job Created: Job ID = {job_id}")

    # 5. Start Job
    print("\nStarting Job...")
    resp = requests.post(f"{BASE_URL}/jobs/{job_id}/start")
    print(f"Start Response: {resp.json()['message']}")

    # 6. Monitor Progress
    print("\nMonitoring Progress...")
    for _ in range(10):  # Poll up to 10 times
        time.sleep(2)
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/progress")
        prog = resp.json()
        print(f"Status: {prog['status']} | Processed: {prog['processed_images']}/{prog['total_images']}")
        if prog["status"] in ["completed", "failed", "stopped"]:
            break
    
    # 7. Check Results on Disk
    results_path = os.path.join(os.getcwd(), 'backend', 'outputs', f"{job_id}_results.json")
    if os.path.exists(results_path):
        print(f"\nResults found at: {results_path}")
        with open(results_path) as f:
            results = json.load(f)
            print(f"Number of annotated images: {len(results.get('results', []))}")
    else:
        print(f"\n[!] Results file NOT found at {results_path}")

    # 8. Export
    print("\nExporting results to COCO...")
    export_config = {
        "job_id": job_id,
        "format": "coco",
        "split_ratios": {"train": 0.8, "val": 0.2, "test": 0.0}
    }
    resp = requests.post(f"{BASE_URL}/export", json=export_config)
    if resp.status_code == 200:
        export_info = resp.json()
        print(f"Export Success: {export_info['output_path']}")
        print(f"Files: {json.dumps(export_info['file_paths'], indent=2)}")
    else:
        print(f"Export failed: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    test_flow()
