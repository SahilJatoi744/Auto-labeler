
import requests
import time

BASE_URL = 'http://127.0.0.1:8000/api/v1'

img_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB\x82'
with open('dummy.png', 'wb') as f: f.write(img_content)
import zipfile
with zipfile.ZipFile('dummy.zip', 'w') as z: z.write('dummy.png')

with open('dummy.zip', 'rb') as f:
    res = requests.post(f'{BASE_URL}/datasets/upload', files={'file': ('dummy.zip', f, 'application/zip')})
ds_id = res.json()['id']

res = requests.post(f'{BASE_URL}/jobs', json={
    'dataset_id': ds_id,
    'task_type': 'object_detection',
    'strategy': 'ai_assisted',
    'classes': [{'id':1, 'name':'person'}],
    'class_hierarchy': {'classes': [{'id':1, 'name':'person'}]}
})
job_id = res.json()['id']

print('Waiting for job...', job_id)
requests.post(f'{BASE_URL}/jobs/{job_id}/start')

while True:
    time.sleep(1)
    res = requests.get(f'{BASE_URL}/jobs/{job_id}/results')
    if res.status_code == 200:
        results = res.json()
        print('Job done!')
        break
    elif res.status_code == 404:
        print('Still running...')
    else:
        print('Error:', res.text)
        break

if not results['results']:
    print('No results returned!')
else:
    img_url = results['results'][0]['image_url']
    print('Image URL:', img_url)

    res = requests.get(f'http://127.0.0.1:8000{img_url}')
    print('Image fetch status:', res.status_code)

