#!/usr/bin/env python3
"""Convert a user-provided flat-lay ZIP into a PRIVATE local Stylist 6 reference vault.

Usage: python3 scripts/style_references/import_flatlay_archive.py
  --source /private/image-teaste.zip
  --index benchmarks/stylist6/references/image_teaste_46.json
  --output /private/aiwardrobe-lookbook

No network calls, Supabase writes, model training, or public Git asset uploads.
Rights remain pending until these particular image copyrights are confirmed.
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from datetime import datetime, timezone

def import_archive(source, index_file, output):
    source, index_file, output = map(Path, (source,index_file,output))
    index=json.loads(index_file.read_text(encoding='utf-8'))
    rows=index.get('images',[])
    if index.get('schema_version')!=1 or index.get('dataset_id')!='image-teaste-2026-11-10':
        raise ValueError('Wrong reference index')
    if len(rows)!=46 or len({r['id'] for r in rows})!=46:
        raise ValueError('Expected 46 unique reviewed outfits')
    if output.resolve()==source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Source and output paths overlap')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    extracted=[]
    with zipfile.ZipFile(source) as z:
        names={}
        for info in z.infolist():
            name=info.filename.replace('\\','/')
            if info.is_dir():continue
            if name.startswith('/') or '..' in name.split('/') or ':' in name:
                raise ValueError('Unsafe archive member')
            if (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Symlink archive member')
            if info.file_size>35*1024*1024:
                raise ValueError('Oversized archive member')
            if re.fullmatch(r'image-teaste/photo_\d+_2026-11-10_09-23-10\.jpg',name):
                names[name]=info
        expected={'image-teaste/'+r['filename'] for r in rows}
        if set(names)!=expected:raise ValueError('Missing or unexpected reference images')
        fingerprints=set()
        for row in rows:
            member='image-teaste/'+row['filename']
            data=z.read(names[member])
            if not data.startswith(b'\xff\xd8\xff') or len(data)<5000:
                raise ValueError('Bad image magic/size')
            sha=hashlib.sha256(data).hexdigest()
            if sha in fingerprints:raise ValueError('Duplicated image requires review')
            fingerprints.add(sha)
            extracted.append((row,data,sha,member))
    output.mkdir(parents=True,exist_ok=True)
    (output/'images').mkdir(exist_ok=True)
    manifest=[]
    for row,data,sha,archive_path in extracted:
        local='images/'+sha+'.jpg'
        (output/local).write_bytes(data)
        manifest.append({**row,'media_sha256':sha,'local_media_path':local,
            'archive_path':archive_path,'bytes':len(data),
            'training_approved':False,'rights_confirmed':False})
    result={'schema_version':2,'dataset_id':index['dataset_id'],
        'source':'user_uploaded_flat_lay_reference_archive',
        'origin_verified':False,
        'rights_status':'check_each_photo_before_commercial_training_or_publishing',
        'usage':'isolated_research_only',
        'media_count':len(manifest),
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'images':manifest}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--index',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    r=import_archive(a.source,a.index,a.output)
    print(json.dumps({'count':r['media_count'],'status':'private_research_ready',
        'training_approved':False,'manifest':str(Path(a.output)/'manifest.json')}))
