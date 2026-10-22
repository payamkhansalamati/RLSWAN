#!/usr/bin/env python3
"""Credential-safe ElevenLabs production client. Never logs headers or credentials."""
import argparse
import json
import os
import ssl
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def load_environment():
    path = PROJECT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.removeprefix('export ').split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
    key = os.environ.get('ELEVENLABS_API_KEY', '')
    if not key or key.startswith('your_'):
        raise RuntimeError('ELEVENLABS_API_KEY is not configured locally')
    return key


def request(path, payload=None):
    key = load_environment()
    headers = {'xi-api-key': key, 'Accept': 'application/json'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request('https://api.elevenlabs.io' + path, data=body, headers=headers)
    try:
        context = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
        with urllib.request.urlopen(req, timeout=120, context=context) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            result = json.loads(exc.read())
            detail = result.get('detail', {})
            message = detail.get('message', '') if isinstance(detail, dict) else str(detail)
        except Exception:
            message = 'No structured error detail'
        raise RuntimeError(f'HTTP {exc.code}: {message.replace(key, "[REDACTED]")[:500]}') from None


def save(name, data):
    target = ROOT / 'research' / (name + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + '\n')


def inspect():
    checks = {}
    for name, endpoint in [
        ('subscription', '/v1/user/subscription'),
        ('voices', '/v2/voices?page_size=100'),
        ('models', '/v1/models'),
        ('video_access', '/v1/flows/video?page_size=1'),
        ('image_access', '/v1/flows/image?page_size=1'),
    ]:
        try:
            data = request(endpoint)
            if name == 'subscription':
                safe = {k: data.get(k) for k in [
                    'tier', 'status', 'character_count', 'character_limit',
                    'can_extend_character_limit', 'allowed_to_extend_character_limit',
                    'max_credit_limit_extension', 'current_overage',
                    'currency', 'voice_limit', 'professional_voice_limit']}
                print(name, json.dumps(safe))
            elif name == 'voices':
                safe = {'voices': [{k: v.get(k) for k in [
                    'voice_id', 'name', 'category', 'labels', 'preview_url',
                    'is_owner', 'is_legacy', 'is_mixed']} for v in data.get('voices', [])],
                    'has_more': data.get('has_more')}
                print(name, 'saved', len(safe['voices']), 'entries')
            elif name == 'models':
                safe = [{k: m.get(k) for k in ['model_id', 'name',
                    'can_do_text_to_speech', 'token_cost_factor', 'model_rates']}
                    for m in data]
                print(name, 'saved', len(safe), 'entries')
            else:
                safe = {'accessible': True, 'generation_count': len(data.get('generations', []))}
                print(name, 'read-only listing accessible')
            save(name, safe)
            checks[name] = {'ok': True, 'endpoint': endpoint}
        except Exception as exc:
            message = str(exc).replace(load_environment(), '[REDACTED]')
            print(name, message)
            checks[name] = {'ok': False, 'endpoint': endpoint, 'error': message}
    save('account_checks', checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect'])
    args = parser.parse_args()
    inspect()
