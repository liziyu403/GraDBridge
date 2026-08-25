import subprocess
from pathlib import Path

import requests
import torch


def attempt_download(file, repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", ''))
    if file.exists():
        return

    try:
        response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()
        assets = [x['name'] for x in response['assets']]
        tag = response['tag_name']
    except Exception:
        assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                  'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
        try:
            tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
        except Exception:
            tag = 'v5.0'

    if file.name not in assets:
        return

    url = f'https://github.com/{repo}/releases/download/{tag}/{file.name}'
    print(f'Downloading {url} to {file}...')
    try:
        torch.hub.download_url_to_file(url, file)
        assert file.exists() and file.stat().st_size > 1E6
    except Exception as e:
        file.unlink(missing_ok=True)
        print(f'Download error: {e}')
