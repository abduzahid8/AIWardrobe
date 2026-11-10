#!/usr/bin/env python3
"""Upload SHA-verified 46-photo lookbook into PRIVATE Supabase Storage.

Pure Python standard library. Checksum dry-run by default. Never exports
private media, keys or service tokens to public GitHub. Never starts training.

 python3 scripts/style_references/upload_private_reference_bundle.py \
   --archive /private/AIWardrobe_Style_References_46_Pilot.zip

 SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... python3 \
   scripts/style_references/upload_private_reference_bundle.py \
   --archive /private/AIWardrobe_Style_References_46_Pilot.zip \
   --upload --confirm-private-research --create-private-bucket
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DATASET = 'image-teaste-2026-11-10'
DEFAULT_BUCKET = 'stylist6-private-research'
MAX_IMAGE = 5 * 1024 * 1024
ALLOWED = re.compile(r'^images/[a-f0-9]{64}\.jpg$')


def validated_bundle(archive: Path):
    """Return validated [(reference_id, storage_key, JPEG bytes, SHA)]."""
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo('manifest.json')
        if info.file_size > 2 * 1024 * 1024:
            raise ValueError('manifest_too_large')
        manifest = json.loads(z.read(info).decode('utf-8'))
        if manifest.get('schema_version') != 2 or manifest.get('dataset_id') != DATASET:
            raise ValueError('wrong_dataset')
        if manifest.get('media_count') != 46 or len(manifest.get('images', [])) != 46:
            raise ValueError('expected_46_photos')
        if len({x.get('id') for x in manifest['images']}) != 46:
            raise ValueError('duplicate_reference_id')
        if manifest.get('published_or_uploaded') is not False:
            raise ValueError('unexpected_publication_state')
        archive_paths = {i.filename for i in z.infolist()}
        output, used = [], set()
        for row in manifest['images']:
            sha, path = row.get('sha256'), row.get('local_media_path')
            if not isinstance(path, str) or not ALLOWED.fullmatch(path):
                raise ValueError('unsafe_media_path')
            if not isinstance(sha, str) or path != f'images/{sha}.jpg' or sha in used:
                raise ValueError('mismatched_or_duplicate_media_sha')
            if path not in archive_paths:
                raise ValueError('missing_media')
            media_info = z.getinfo(path)
            if media_info.file_size > MAX_IMAGE or media_info.file_size < 4096:
                raise ValueError('media_size_invalid')
            raw = z.read(media_info)
            if not raw.startswith(b'\xff\xd8\xff') or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('media_sha_validation_failed')
            if row.get('training_approved') is not False:
                raise ValueError('training_requires_separate_approval')
            used.add(sha)
            output.append((row['id'], f'{DATASET}/{sha}.jpg', raw, sha))
        return output


class Storage:
    def __init__(self, url: str, secret: str):
        self.base = url.rstrip('/')
        if not re.fullmatch(r'https://[a-z0-9-]+\.supabase\.co', self.base):
            raise ValueError('invalid_supabase_url')
        if not secret or len(secret) < 20:
            raise ValueError('missing_service_role_key')
        self.secret = secret

    def request(self, method, resource, data=None, content_type='application/json'):
        url = self.base + '/storage/v1/' + resource
        body = json.dumps(data).encode() if isinstance(data, dict) else data
        headers = {'apikey': self.secret, 'Authorization': 'Bearer '+self.secret,
                   'Content-Type': content_type}
        if method == 'POST' and resource.startswith('object/'):
            headers['x-upsert'] = 'false'
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=35) as response:
                result = response.read(MAX_IMAGE + 1)
                if len(result) > MAX_IMAGE:
                    raise ValueError('oversized_server_response')
                return response.status, result
        except urllib.error.HTTPError as err:
            raise RuntimeError(f'storage_http_{err.code} on {method} {resource.split("/")[0]}') from None
        except urllib.error.URLError:
            raise RuntimeError(f'storage_network_error on {method} {resource.split("/")[0]}') from None

    def ensure_private_bucket(self, bucket, allow_create):
        q = urllib.parse.quote(bucket, safe='')
        try:
            _, response = self.request('GET', f'bucket/{q}')
        except RuntimeError as err:
            if 'storage_http_404' not in str(err) or not allow_create:
                raise
            self.request('POST', 'bucket', {
                'id': bucket, 'name': bucket, 'public': False,
                'file_size_limit': MAX_IMAGE, 'allowed_mime_types': ['image/jpeg']})
            _, response = self.request('GET', f'bucket/{q}')
        metadata = json.loads(response)
        if metadata.get('public') is not False or metadata.get('id') != bucket:
            raise ValueError('bucket_is_not_proven_private')

    def send_and_verify(self, bucket, key, data, sha):
        escaped = '/'.join(urllib.parse.quote(part, safe='') for part in key.split('/'))
        target = f'object/{urllib.parse.quote(bucket, safe="")}/{escaped}'
        try:
            self.request('POST', target, data, 'image/jpeg')
        except RuntimeError as err:
            if 'storage_http_409' not in str(err):
                raise
        _, downloaded = self.request('GET',
            'object/authenticated/'+urllib.parse.quote(bucket, safe='')+'/'+escaped)
        if hashlib.sha256(downloaded).hexdigest() != sha:
            raise RuntimeError('remote_photo_hash_mismatch')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--bucket', default=DEFAULT_BUCKET)
    p.add_argument('--upload', action='store_true')
    p.add_argument('--confirm-private-research', action='store_true')
    p.add_argument('--create-private-bucket', action='store_true')
    p.add_argument('--receipt', type=Path)
    args = p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.bucket):
        p.error('Invalid bucket name')
    rows = validated_bundle(args.archive)
    if not args.upload:
        print(json.dumps({'result':'verified_private_dry_run','photos':len(rows),
            'distinct_sha256':len({x[3] for x in rows}),
            'total_bytes':sum(len(x[2]) for x in rows),
            'uploaded':0,'model_trained':False}))
        return 0
    if not args.confirm_private_research:
        p.error('--confirm-private-research required')
    remote = Storage(os.getenv('SUPABASE_URL',''),
                     os.getenv('SUPABASE_SERVICE_ROLE_KEY',''))
    remote.ensure_private_bucket(args.bucket, args.create_private_bucket)
    verified = []
    for ref,key,raw,sha in rows:
        remote.send_and_verify(args.bucket, key, raw, sha)
        verified.append({'id':ref, 'bucket':args.bucket,'key':key,'sha256':sha})
        print(f'verified remote {ref} ({len(verified)}/{len(rows)})')
    receipt = {'dataset_id':DATASET,'storage_private':True,'photos':len(verified),
        'research_only':True,'training_approved':False,'files':verified}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
        os.chmod(args.receipt,0o600)
    print(json.dumps({'result':'uploaded_and_verified','bucket':args.bucket,
        'photos':len(verified),'training_approved':False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
