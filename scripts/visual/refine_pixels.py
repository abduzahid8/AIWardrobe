"""Refine existing private journals from the SAME checksum-verified source photos.
Bounded threads; no new model inference. Writes new resumable per-batch journals.
"""
import sys,json,pathlib,concurrent.futures,hashlib,os,time
from enrich_catalog import fetch_image
from image_pixels import primary_hue, merge_primary_hue
root=pathlib.Path(sys.argv[1]);name=sys.argv[2]
rows=[json.loads(line) for line in open(root/(name+'.jsonl')) if line.strip()]
if len(rows)>256:raise ValueError('batch bound')
output=root/(name+'-pixels.jsonl');done=set()
if output.exists():done={json.loads(line)['product_id'] for line in open(output) if line.strip()}
cache=pathlib.Path(os.path.expanduser('~/.cache/aiwardrobe/visual-v1'))
def refine(row):
 try:
  p=row['profile'];image_file=cache/(p['imageSha256']+'.image')
  if image_file.exists():
   raw=image_file.read_bytes()
   from PIL import Image
   import io
   image=Image.open(io.BytesIO(raw)).convert('RGB')
  else:
   raw,image=fetch_image(p['imageUrl'])
  if hashlib.sha256(raw).hexdigest()!=p['imageSha256']:return row,False,'checksum_changed'
  image_file.write_bytes(raw)
  observed=primary_hue(image)
  merge_primary_hue(p['attributes'],observed)
  p['taxonomyRevision']='stylist5-facets-3';p['observedAt']=time.time()
  return row,bool(observed),None
 except Exception as error:return row,False,type(error).__name__
counts={'requested':len(rows),'refined':0,'pixelHues':0,'unavailable':0,'modelInferenceCalls':0}
with open(output,'a') as out:
 os.chmod(output,0o600)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
  for row,changed,error in executor.map(refine,[r for r in rows if r['product_id'] not in done]):
   out.write(json.dumps(row)+'\n');out.flush();counts['refined']+=1;counts['pixelHues']+=int(changed);counts['unavailable']+=int(error is not None)
print(json.dumps(counts))
