# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_device_preference():
    print("--- Testing Global Device Preference ---")
    try:
        # 1. Get current status
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        status = resp.json()
        initial_pref = status.get('device_preference', 'auto')
        print(f"Initial Preference: {initial_pref}")
        
        # 2. Update preference to 'cpu'
        print("Setting preference to 'cpu'...")
        upd_resp = requests.post(f"{BASE_URL}/system/device?device=cpu")
        upd_resp.raise_for_status()
        
        # 3. Verify update
        resp = requests.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        status = resp.json()
        new_pref = status.get('device_preference')
        print(f"New Preference in Health: {new_pref}")
        
        assert new_pref == "cpu", f"Expected 'cpu', got '{new_pref}'"
        
        # 4. Reset to 'auto'
        print("Resetting preference to 'auto'...")
        requests.post(f"{BASE_URL}/system/device?device=auto")
        
        print("SUCCESS: Global device preference is working dynamically.")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_device_preference()
