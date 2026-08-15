# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Test script to diagnose image URL resolution issues.
Run with: python scripts/test_image_urls.py
Requires the backend to be running on port 8000.
"""
import sys
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

BACKEND = "http://127.0.0.1:8000"

def test_api_reachable():
    """Step 1: Can we reach the backend?"""
    try:
        resp = urllib.request.urlopen(f"{BACKEND}/", timeout=5)
        data = json.loads(resp.read())
        print(f"[PASS] Backend reachable: {data.get('name', '?')} v{data.get('version', '?')}")
        return True
    except Exception as e:
        print(f"[FAIL] Backend not reachable at {BACKEND}: {e}")
        return False

def get_jobs():
    """Step 2: List all jobs."""
    try:
        resp = urllib.request.urlopen(f"{BACKEND}/api/v1/jobs", timeout=10)
        jobs = json.loads(resp.read())
        print(f"\n[INFO] Found {len(jobs)} jobs:")
        for j in jobs:
            print(f"  - {j['id']}  task={j.get('task_type','?')}  status={j.get('status','?')}  dataset={j.get('dataset_id','?')}")
        return jobs
    except Exception as e:
        print(f"[FAIL] Cannot list jobs: {e}")
        return []

def test_job_results(job):
    """Step 3: Check results for a job, verify image_url is present and resolvable."""
    job_id = job['id']
    task_type = job.get('task_type', '?')
    print(f"\n{'='*60}")
    print(f"Testing job {job_id} (task={task_type}, dataset={job.get('dataset_id', '?')})")
    print(f"{'='*60}")

    try:
        resp = urllib.request.urlopen(f"{BACKEND}/api/v1/jobs/{job_id}/results", timeout=15)
        data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  [FAIL] HTTP {e.code} fetching results: {body[:200]}")
        return
    except Exception as e:
        print(f"  [FAIL] Error fetching results: {e}")
        return

    results = data.get("results", [])
    print(f"  [INFO] Total results: {len(results)}")

    if not results:
        print(f"  [WARN] No results for this job (may not have been run).")
        return

    # Check first 3 results
    for i, r in enumerate(results[:3]):
        image_id = r.get("image_id", "?")
        image_url = r.get("image_url")
        ann_count = len(r.get("annotations", []))
        print(f"\n  --- Result {i+1}: image_id={image_id}, annotations={ann_count} ---")
        
        if not image_url:
            print(f"  [FAIL] image_url is MISSING (None/empty) - this is the root cause!")
            print(f"         The frontend cannot display the image without image_url")
            # Show what fields ARE present
            print(f"         Fields present: {list(r.keys())}")
        else:
            print(f"  [OK]   image_url = {image_url}")
            
            # Try to fetch the image via backend
            full_url = f"{BACKEND}{image_url}"
            try:
                req = urllib.request.Request(full_url, method='HEAD')
                resp = urllib.request.urlopen(req, timeout=5)
                ct = resp.headers.get('Content-Type', '?')
                cl = resp.headers.get('Content-Length', '?')
                print(f"  [PASS] Image accessible: {ct}, {cl} bytes")
            except urllib.error.HTTPError as e:
                print(f"  [FAIL] Image NOT accessible: HTTP {e.code} at {full_url}")
            except Exception as e:
                print(f"  [FAIL] Image fetch error: {e}")

        # Show annotation sample
        for ann in r.get("annotations", [])[:2]:
            has_bbox = "bbox" in ann and ann["bbox"] is not None
            has_seg = "segmentation" in ann and ann.get("segmentation") is not None
            has_poly = has_seg and ann["segmentation"].get("polygon") is not None if has_seg else False
            print(f"    Ann: class={ann.get('class_name','?')} conf={ann.get('confidence',0):.2f} bbox={has_bbox} seg={has_seg} poly={has_poly}")

def main():
    print("=" * 60)
    print("Image URL Resolution Diagnostic")
    print("=" * 60)
    
    if not test_api_reachable():
        print("\nBackend is not running. Start it first:")
        print("  cd backend && .\\venv\\Scripts\\activate && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    jobs = get_jobs()
    if not jobs:
        print("\nNo jobs found. Nothing to test.")
        sys.exit(0)

    # Test completed jobs
    completed = [j for j in jobs if j.get("status") in ("completed", "stopped")]
    if not completed:
        print("\nNo completed jobs found. Testing all jobs.")
        completed = jobs

    for job in completed[:3]:  # Test up to 3
        test_job_results(job)

    print(f"\n{'='*60}")
    print("Diagnostic complete.")

if __name__ == "__main__":
    main()
