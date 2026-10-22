#!/usr/bin/env python3
"""Download existing ElevenLabs stock previews; does not generate or spend credits."""
import concurrent.futures
import json
from pathlib import Path
import ssl
import subprocess
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    'worker_bella': 'hpp4J3VqNfWAUOO0d1Us',
    'worker_sarah': 'EXAVITQu4vr4xnSDxMaL',
    'robot_river': 'SAz9YHcvj6GT2YYXdXww',
    'robot_eric': 'cjVigY5qzO86Huf0OWal',
}


def download_one(item):
    name, voice_id = item
    voices = json.loads((ROOT / 'research/voices.json').read_text())['voices']
    voice = next(v for v in voices if v['voice_id'] == voice_id)
    if voice['category'] != 'premade':
        raise RuntimeError('Audition requires a stock premade voice')
    url = voice['preview_url']
    if urllib.parse.urlparse(url).hostname not in {
        'storage.googleapis.com', 'api.elevenlabs.io', 'api.us.elevenlabs.io'}:
        raise RuntimeError('Unexpected stock-preview host')
    target = ROOT / 'voices' / (name + '_stock_preview.mp3')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60,
                context=ssl.create_default_context(cafile='/etc/ssl/cert.pem')) as response:
            target.write_bytes(response.read())
    import imageio_ffmpeg
    short = target.with_name(name + '_audition.wav')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
        '-y', '-i', str(target), '-t', '7', '-ar', '44100', '-ac', '1', str(short)], check=True)
    return {'candidate': name, 'voice_id': voice_id, 'name': voice['name'],
            'category': voice['category'], 'labels': voice['labels'],
            'audition_file': str(short.relative_to(ROOT)), 'source': 'Existing stock preview',
            'generation_credits_used': 0}


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download_one, CANDIDATES.items()))
    (ROOT / 'voices/audition_manifest.json').write_text(json.dumps(results, indent=2) + '\n')
    for result in results:
        print(result['candidate'], result['audition_file'])
