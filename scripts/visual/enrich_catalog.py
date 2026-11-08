#!/usr/bin/env python3
"""Offline SigLIP2 image evidence + cached embeddings. No paid APIs.
Input: bounded authoritative catalog JSON export. Output: JSONL for guarded import.
Never writes catalog fields. Supply a pinned HF commit via --revision.
Similarity confidence is relative, not a calibrated probability. Weak evidence
is omitted and source-slot contradictions are quarantined by consumers.
"""
import argparse, hashlib, io, json, os, pathlib, time, subprocess
from urllib.parse import urlparse
import ipaddress, socket
import numpy as np
import requests
from PIL import Image, ImageOps
import torch
from transformers import AutoModel, AutoProcessor

MODEL='google/siglip2-base-patch16-224'
TAXONOMY='stylist5-facets-3'
from image_pixels import primary_hue, merge_primary_hue
# Descriptions refer only to visible traits. Fiber composition is deliberately absent.
GROUPS={
 'slot': {'top':'a shirt, polo or t-shirt','outerwear':'a jacket, coat or cardigan','bottom':'a pair of trousers or jeans','shoes':'a pair of shoes'},
 'subtype': {'dress-shirt':'a tailored dress shirt','shirt':'a casual buttoned shirt','polo':'a collared polo shirt','t-shirt':'a simple t-shirt','fine-knit':'a fine gauge knit base top','blazer':'a tailored blazer jacket','cardigan':'an openable knit cardigan','overshirt':'an overshirt jacket','coat':'a long coat','bomber':'a bomber jacket','hoodie':'a hooded sweatshirt','pullover':'a pullover sweater','jacket':'a casual jacket','tailored-trouser':'structured tailored trousers','chino':'cotton chinos','jeans':'denim jeans','athletic-trouser':'drawstring jogging trousers','cargo':'cargo trousers with side pockets','trouser':'simple trousers','oxford':'oxford lace-up dress shoes','derby':'derby lace-up dress shoes','loafer':'leather loafers','chelsea':'Chelsea boots','boot':'ankle boots','sneaker':'casual sneakers','sport-sneaker':'athletic running shoes','sandal':'open sandals','driver':'soft driving moccasins'},
 'color': {x:x+' colored clothing or shoes' for x in ['black','white','cream','beige','brown','navy','blue','grey','green','olive','burgundy','red','pink','purple','yellow','orange']},
 'volume': {'slim':'a slim fitted garment','regular':'a regular fit garment','wide':'a loose wide garment'},
 'bulk': {'light':'light thin clothing','heavy':'thick bulky clothing'},
 'length': {'cropped':'a cropped short garment','regular':'a regular length garment','long':'a long length garment'},
 'texture': {x:x+' visible fabric texture' for x in ['smooth','knit','ribbed','denim','suede','corduroy','tweed','technical']},
 'pattern': {'solid':'plain solid color clothing','stripe':'striped clothing','check':'checked plaid clothing','print':'printed graphic patterned clothing'},
 'patternProminence': {'subtle':'clothing with a subtle quiet small pattern','prominent':'clothing with a large bold visible pattern'},
 'formality': {'athletic':'athletic sportswear','casual':'casual everyday clothing','smart':'smart casual refined clothing','formal':'formal business dress clothing'},
 'soleProfile': {'dress':'shoes with thin elegant low profile soles','comfort':'shoes with thick cushioned comfort soles','sneaker':'shoes with sneaker rubber cupsoles','rugged':'boots with heavy lugged rugged soles'},
 'closure': {'openable':'clothing with a button or zip front opening','pullover':'pullover clothing with no front opening'},
 'collar': {'shirt':'a shirt collar','high':'a high neck collar','open':'an open neckline','crew':'a round crew neckline'},
 'footwearPurpose': {'walking':'casual city walking shoes','training':'athletic training or basketball shoes','trail':'rugged trail hiking shoes','dress':'formal dress shoes'},
 'accentHue': {x:x+' contrasting accents on clothing or shoes' for x in ['black','white','cream','beige','brown','navy','blue','grey','green','red']},
 'hemShape': {'straight':'a straight garment hem','tapered':'tapered trouser legs','barrel':'curved barrel shaped trouser legs','flared':'flared wide trouser hems'},
 'imageKind': {'product':'A studio product photograph showing clothing or shoes against a plain background.','diagram':'A clothing size chart or technical measurement diagram with numbers and lines.','promotion':'An advertisement banner dominated by promotional text and sale prices.'},
}

def public_url(url):
 p=urlparse(url)
 if p.scheme!='https' or not p.hostname or p.username or p.password or p.port not in (None,443):
  raise ValueError('image URL must be public HTTPS')
 for info in socket.getaddrinfo(p.hostname,443,type=socket.SOCK_STREAM):
  if not ipaddress.ip_address(info[4][0]).is_global: raise ValueError('non-public image host')
 return url

def fetch_image(url):
 # Validate every redirect; cap bytes before decoding, never send credentials.
 started_request=time.monotonic()
 for _ in range(4):
  if time.monotonic()-started_request>20: raise ValueError('image deadline')
  public_url(url)
  r=requests.get(url,timeout=(5,10),allow_redirects=False,stream=True)
  if r.is_redirect:
   from urllib.parse import urljoin
   url=urljoin(url,r.headers['Location']); r.close(); continue
  if not r.ok:
   status=r.status_code;r.close();raise ValueError('image HTTP '+str(status))
  if not r.headers.get('content-type','').startswith('image/'):
   r.close();raise ValueError('not an image')
  content=bytearray();started=time.monotonic()
  for chunk in r.iter_content(65536):
   if time.monotonic()-started>20: r.close();raise ValueError('image deadline')
   content.extend(chunk)
   if len(content)>12_000_000: r.close();raise ValueError('image too large')
  r.close()
  image=Image.open(io.BytesIO(content))
  if image.width*image.height>30_000_000: raise ValueError('image dimensions too large')
  image.load()
  return bytes(content),ImageOps.exif_transpose(image).convert('RGB')
 raise ValueError('redirect limit')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('output');ap.add_argument('--revision',required=True);ap.add_argument('--limit',type=int,default=32);ap.add_argument('--ocr-command',help='optional compiled image_text.swift binary');ap.add_argument('--resume',action='store_true',help='append missing or stale source revisions');ap.add_argument('--recheck-hours',type=float,default=24);ap.add_argument('--cache',default=os.path.expanduser('~/.cache/aiwardrobe/visual-v1'));args=ap.parse_args()
 if len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision): raise ValueError('pin a model commit SHA')
 if not 1<=args.limit<=256: raise ValueError('batch must contain 1..256 images')
 rows=json.load(open(args.input))
 if args.recheck_hours<0 or args.recheck_hours>168: raise ValueError('recheck interval must be 0..168 hours')
 existing=set()
 if args.resume and pathlib.Path(args.output).exists():
  for line in open(args.output):
   try:
    p=json.loads(line)
    observed=p['profile'].get('observedAt',pathlib.Path(args.output).stat().st_mtime)
    if time.time()-observed<args.recheck_hours*3600: existing.add((p['product_id'],p['profile']['sourceHash'],p['profile'].get('taxonomyRevision')))
   except (ValueError,KeyError): raise ValueError('invalid resume journal')
 if len(rows)>args.limit: raise ValueError('export exceeds explicit batch bound')
 torch.set_num_threads(2)
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 model=AutoModel.from_pretrained(MODEL,revision=args.revision).eval().to(device)
 processor=AutoProcessor.from_pretrained(MODEL,revision=args.revision)
 prompts=[label if name=='imageKind' else f'A product photograph of {label}.' for name,group in GROUPS.items() for label in group.values()]
 with torch.inference_mode():
  inputs=processor(text=prompts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
  text=model.get_text_features(**inputs);text=torch.nn.functional.normalize(text,dim=-1)
 cache=pathlib.Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
 done=0;start=time.monotonic()
 with open(args.output,'a' if args.resume else 'w') as output:
  os.chmod(args.output,0o600)
  for row in rows:
   if (row['id'],row['source_hash'],TAXONOMY) in existing: continue
   try:
    url=row['image_url'];source_hash=row['source_hash'];
    for attempt in range(2):
     try:
      raw,image=fetch_image(url);break
     except (requests.RequestException,ValueError,OSError):
      if attempt: raise
      time.sleep(0.25)
    sha=hashlib.sha256(raw).hexdigest()
    key=hashlib.sha256((sha+args.revision).encode()).hexdigest();file=cache/(key+'.json')
    cached=json.load(open(file)) if file.exists() else None
    with torch.inference_mode():
     if cached:
      vector=torch.tensor([cached['embedding']],device=device)
     else:
      inputs=processor(images=image,return_tensors='pt').to(device)
      vector=torch.nn.functional.normalize(model.get_image_features(**inputs),dim=-1)
     logits=(vector@text.T)*model.logit_scale.exp()
    attrs={};offset=0
    for name,group in GROUPS.items():
     values=list(group);probs=torch.softmax(logits[0,offset:offset+len(values)],dim=-1).cpu().numpy();offset+=len(values)
     order=np.argsort(probs)[::-1];score=float(probs[order[0]]);margin=float(probs[order[0]]-probs[order[1]])
     if score>=0.8 and margin>=0.1: attrs[name]={'value':values[order[0]],'confidence':round(score,5),'margin':round(margin,5),'evidence':'image-model'}
    # Shoe construction and cloth-only attributes never cross source slots.
    # Length/cloth weight are not calibrated by this model; leave them unknown.
    attrs.pop('length',None);attrs.pop('bulk',None)
    if attrs.get('pattern',{}).get('value')=='solid': attrs.pop('patternProminence',None)
    if row['slot']=='bottom':
     attrs.pop('closure',None);attrs.pop('collar',None)
    if row['slot']!='bottom': attrs.pop('hemShape',None)
    if row['slot']!='shoes':
     attrs.pop('soleProfile',None);attrs.pop('footwearPurpose',None)
    else:
     for name in ['volume','bulk','length','closure','collar']: attrs.pop(name,None)
    result={'attributes':attrs,'embedding':vector[0].cpu().tolist(),'imageQuality':{'width':image.width,'height':image.height}}
    file.write_text(json.dumps(result))
    attrs=dict(result['attributes']);kind=attrs.pop('imageKind',None)
    profile={'version':1,'sourceHash':source_hash,'imageUrl':url,'imageSha256':sha,'model':MODEL,'revision':args.revision,'taxonomyRevision':TAXONOMY,'observedAt':time.time(),'attributes':attrs,'imageQuality':result['imageQuality']}
    if args.ocr_command:
     image_file=cache/(sha+'.image');image_file.write_bytes(raw)
     try:
      geometry=json.loads(subprocess.check_output([args.ocr_command,str(image_file)],timeout=10))
      profile['imageQuality'].update(geometry)
      # OCR confirms substantial text; do not turn small garment labels into ads.
      if kind and kind['value'] in ['diagram','promotion'] and geometry['textCoverage']>=0.08 and geometry['textRegions']>=5:
       kind={**kind,'evidence':'image-ocr'}
     except (subprocess.SubprocessError,ValueError): pass
    if image.width<128 or image.height<128:
     kind={'value':'unusable','confidence':1,'margin':1,'evidence':'image-pixels'}
    observed=primary_hue(image)
    merge_primary_hue(profile['attributes'],observed)
    (cache/(sha+'.image')).write_bytes(raw)
    if kind: profile['imageKind']=kind
    output.write(json.dumps({'product_id':row['id'],'profile':profile,'embedding':result['embedding']})+'\n');output.flush();done+=1
    print(json.dumps({'done':done,'evidenceFields':len(attrs),'imageKind':kind,'seconds':round(time.monotonic()-start,2)}),flush=True)
   except Exception as e:

    with open(args.output+'.failures.jsonl','a') as failures: os.chmod(args.output+'.failures.jsonl',0o600);failures.write(json.dumps({'product_id':row['id'],'source_hash':row['source_hash'],'status':'unknown','reason':type(e).__name__,'attemptedAt':time.time()})+'\n')
    print(json.dumps({'status':'unknown','reason':type(e).__name__}),flush=True)
 print(json.dumps({'enriched':done,'requested':len(rows),'paidCalls':0,'seconds':round(time.monotonic()-start,2)}),flush=True)
if __name__=='__main__': main()
