
import requests

BASE_URL = 'http://127.0.0.1:8000/api/v1'
job_id = 'eb03677d7a97a876'

res = requests.get(f'{BASE_URL}/jobs/{job_id}/results')
if res.status_code == 200:
    results = res.json()
    if not results.get('results'):
        print('Empty results!')
    else:
        img_url = results['results'][0].get('image_url')
        print('Image URL:', img_url)
        if img_url:
            img_res = requests.get(f'http://127.0.0.1:8000{img_url}')
            print('Image fetch status:', img_res.status_code)
            if img_res.status_code == 200:
                print('Image bytes length:', len(img_res.content))
        else:
            print('image_url is missing!')
else:
    print('Failed to get job:', res.status_code, res.text)

