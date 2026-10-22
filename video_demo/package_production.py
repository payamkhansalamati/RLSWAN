#!/usr/bin/env python3
"""Save the production archive and integrity manifest; no generation/API calls."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'context_aware_robot_production_package.zip'
MANIFEST = ROOT / 'asset_manifest.json'


def selected(path):
    relative = path.relative_to(ROOT)
    if path == ARCHIVE or path == MANIFEST:
        return False
    if any(part in {'.venv', '__pycache__', '.git'} for part in relative.parts):
        return False
    if relative.parts[:2] == ('audio', 'assembly'):
        return False
    if path.suffix == '.zip' or path.name in {'.asset_jobs.lock', '.DS_Store', '.env'}:
        return False
    if path.name.startswith('.env.') and path.name != '.env.example':
        return False
    return path.is_file()


def main():
    if ARCHIVE.exists():
        raise RuntimeError('Archive already exists. Preserve it and use a versioned archive name for a later edit.')
    key = os.environ.get('ELEVENLABS_API_KEY', '')
    env = ROOT.parent / '.env'
    if not key and env.is_file():
        for line in env.read_text().splitlines():
            if line.strip().startswith('ELEVENLABS_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
                break
    files = sorted(p for p in ROOT.rglob('*') if selected(p))
    entries = []
    for path in files:
        data = path.read_bytes()
        if key and key.encode() in data:
            raise RuntimeError('Credential detected in a selected package file; packaging stopped without exposing it.')
        entries.append({'path': str(path.relative_to(ROOT)), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'project': 'Context-aware robot concept film',
                'status': 'Saved on hold; no motion footage or final MP4',
                'source_media_preserved': True, 'secrets_included': False,
                'excluded': ['machine-specific virtual environment', 'Python caches',
                             'credentials', 'redundant audio working renders'],
                'files': entries}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('video_demo/source_clips/', '')
        for path in files + [MANIFEST]:
            archive.write(path, 'video_demo/' + str(path.relative_to(ROOT)))
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
        for entry in entries:
            data = archive.read('video_demo/' + entry['path'])
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise RuntimeError('Archived file differs from the saved source')
    print(f'Saved {len(entries)} files plus integrity manifest; verified archive: {ARCHIVE.name}')
    print(f'Archive size: {ARCHIVE.stat().st_size / (1024 * 1024):.1f} MiB. No API calls or source regeneration.')


if __name__ == '__main__':
    main()
