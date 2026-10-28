"""Download pinned public data and reward-model files through hf-mirror.com."""
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = 'llamafactory/DPO-En-Zh-20k'
DATA_REVISION = '21ff425feb7cc16a45a2e91536ff237a6fd3779a'
REWARD_REPO = 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'
REWARD_REVISION = '8c14a4e9e6321deaf572544339b16b8d6bbe8886'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {'url': url, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
    partial = path.with_name(path.name + '.part')
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            response = requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                                    stream=True, timeout=(30, 120))
            response.raise_for_status()
            mode = 'ab' if offset and response.status_code == 206 else 'wb'
            with partial.open(mode) as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
            partial.replace(path)
            print(f'Downloaded {path.name}: {path.stat().st_size:,} bytes', flush=True)
            return {'url': url, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    manifest = {'endpoint': 'https://hf-mirror.com', 'dataset': DATA_REPO,
                'dataset_revision': DATA_REVISION, 'reward_model': REWARD_REPO,
                'reward_revision': REWARD_REVISION, 'files': {}}
    data_dir = ROOT / 'dataset/downloads/dpo_en_zh_20k'
    model_dir = ROOT / 'models/skywork_reward_0.6b'
    for name in ['README.md', 'dpo_zh.json']:
        path = data_dir / name
        manifest['files'][str(path.relative_to(ROOT))] = download(
            f'https://hf-mirror.com/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{name}', path)
    for name in ['README.md', 'config.json', 'tokenizer_config.json', 'tokenizer.json',
                 'special_tokens_map.json', 'chat_template.jinja', 'added_tokens.json',
                 'merges.txt', 'vocab.json', 'model.safetensors']:
        path = model_dir / name
        manifest['files'][str(path.relative_to(ROOT))] = download(
            f'https://hf-mirror.com/{REWARD_REPO}/resolve/{REWARD_REVISION}/{name}', path)
    (data_dir / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Download manifest:', data_dir / 'download_manifest.json', flush=True)


if __name__ == '__main__':
    main()
