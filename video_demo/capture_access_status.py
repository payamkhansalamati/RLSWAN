#!/usr/bin/env python3
"""Read-only plan/credit and API eligibility evidence; no generation calls."""
from datetime import datetime, timezone
import json
import ssl
import urllib.error
import urllib.request

from elevenlabs_client import ROOT, load_environment
from generate_assets import safe_subscription


def main():
    key = load_environment()
    context = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
    evidence = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                'video_or_image_generation_submitted': False, 'checks': []}
    for endpoint in ['/v1/flows/video?page_size=1', '/v1/flows/image?page_size=1']:
        url = 'https://api.elevenlabs.io' + endpoint
        record = {'method': 'GET', 'url': url, 'purpose': 'Read-only capability listing'}
        req = urllib.request.Request(url, headers={'xi-api-key': key})
        try:
            with urllib.request.urlopen(req, timeout=30, context=context) as response:
                record['http_status'] = response.status
        except urllib.error.HTTPError as error:
            record['http_status'] = error.code
            body = json.loads(error.read())
            detail = body.get('detail', {})
            if isinstance(detail, dict):
                record['error'] = {k: str(detail[k]).replace(key, '[REDACTED]')
                                   for k in ('status', 'message') if k in detail}
            else:
                record['error'] = {'message': str(detail).replace(key, '[REDACTED]')}
        evidence['checks'].append(record)
        print(record['method'], record['url'], record['http_status'], record.get('error', {}))
    evidence['subscription'] = safe_subscription()
    (ROOT / 'research/api_access_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('Included credits remaining:', evidence['subscription']['included_remaining'])


if __name__ == '__main__':
    main()
