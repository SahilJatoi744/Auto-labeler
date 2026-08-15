# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_results_update():
    print("Testing HITL Results Update API...")
    
    # 1. Get existing jobs
    jobs_res = requests.get(f"{BASE_URL}/jobs")
    jobs = jobs_res.json()
    if not jobs:
        print("No jobs found to test with.")
        return
    
    job_id = jobs[0]['id']
    print(f"Using Job ID: {job_id}")
    
    # 2. Get results
    results_res = requests.get(f"{BASE_URL}/jobs/{job_id}/results")
    data = results_res.json()
    results = data['results']
    
    if not results:
        print("No results found in job. Cannot test update.")
        return
    
    print(f"Original first image annotations count: {len(results[0]['annotations'])}")
    
    # 3. Modify first result (Change first annotation name)
    original_results = json.loads(json.dumps(results)) # Deep copy
    if results[0]['annotations']:
        results[0]['annotations'][0]['class_name'] = "AUDITED_SUCCESS"
    
    # 4. Push update
    update_res = requests.put(f"{BASE_URL}/jobs/{job_id}/results", json=results)
    print(f"Update Status: {update_res.status_code}")
    
    if update_res.status_code == 200:
        # 5. Verify persistence
        verify_res = requests.get(f"{BASE_URL}/jobs/{job_id}/results")
        verify_data = verify_res.json()
        if verify_data['results'][0]['annotations'][0]['class_name'] == "AUDITED_SUCCESS":
            print("✅ Verification Successful: Changes persisted to backend!")
        else:
            print("❌ Verification Failed: Changes did not persist.")
    else:
        print(f"❌ API Error: {update_res.text}")

if __name__ == "__main__":
    try:
        test_results_update()
    except Exception as e:
        print(f"Connection error (is backend running?): {e}")
