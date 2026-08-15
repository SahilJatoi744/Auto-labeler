# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import requests
import time
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_system_status():
    print("--- Testing System Status (Dynamic) ---")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        status = resp.json()
        
        print(f"Status: {status['status']}")
        print(f"CPU Usage: {status.get('cpu_usage')}%")
        print(f"Memory Usage: {status.get('memory_usage_percent')}%")
        print(f"Active Jobs count: {status.get('active_jobs')}")
        print(f"Models Ready: {status.get('models_loaded')}")
        
        # Verify new fields exist
        assert "cpu_usage" in status, "Missing cpu_usage in health check"
        assert "memory_usage_percent" in status, "Missing memory_usage_percent in health check"
        assert "active_jobs" in status, "Missing active_jobs in health check"
        
        print("SUCCESS: Dynamic metrics are being served.")
    except Exception as e:
        print(f"FAILURE: {e}")

def test_jobs_list():
    print("\n--- Testing Jobs List ---")
    try:
        resp = requests.get(f"{BASE_URL}/jobs")
        resp.raise_for_status()
        jobs = resp.json()
        print(f"Total jobs in DB: {len(jobs)}")
        
        if jobs:
            job_id = jobs[0]['id']
            print(f"Verifying results for first job: {job_id}")
            res_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/results")
            res_resp.raise_for_status()
            results_data = res_resp.json()
            
            # Check structure (compatibility)
            if isinstance(results_data, dict) and "results" in results_data:
                results = results_data["results"]
            else:
                results = results_data
                
            print(f"Job {job_id} has {len(results)} images processed.")
            
        print("SUCCESS: Jobs and Results endpoints are responsive.")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    print("Starting Comprehensive API Verification...")
    test_system_status()
    test_jobs_list()
    print("\nVerification Complete.")
