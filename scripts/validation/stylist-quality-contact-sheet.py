"""Real merchant images only. Missing downloads remain visibly missing."""
import json,urllib.request,io,textwrap,concurrent.futures,sys
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
root=Path('docs/reports/stylist-quality');cache=Path('/tmp/aiwardrobe-stylist-media');cache.mkdir(exist_ok=True)
version='baseline' if len(sys.argv)>1 and sys.argv[1]=='baseline' else 'after';prefix='baseline-' if version=='baseline' else '';report=json.loads((root/(version+'.json')).read_text());selected=[]
for n,c in enumerate(report['contexts'][:6]):
 looks=[(r['seed'],l) for r in c['rounds'] for l in r['looks']][:6]
 selected.append((n,c,looks))
products={i['id']:i for _,_,looks in selected for _,l in looks for i in l['items']}
def fetch(item):
 import hashlib
 path=cache/(hashlib.sha256(item['imageUrl'].encode()).hexdigest()+'.jpg')
 try:
  if not path.exists():
   req=urllib.request.Request(item['imageUrl'],headers={'User-Agent':'Mozilla/5.0'})
   with urllib.request.urlopen(req,timeout=20) as r:data=r.read(8*1024*1024)
   img=Image.open(io.BytesIO(data)).convert('RGB');img.thumbnail((500,600));img.save(path)
  return item['id'],str(path),None
 except Exception as e:return item['id'],None,type(e).__name__
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:images={k:(p,e) for k,p,e in pool.map(fetch,products.values())}
manifest=[]
for n,c,looks in selected:
 sheet=Image.new('RGB',(1200,6*330+55),'white');draw=ImageDraw.Draw(sheet)
 draw.text((12,12),f"ACTUAL CATALOG / {c['style']} + {c['occasion']} / known weather: {c['weather']}",fill='black')
 rows=[]
 for j,(seed,look) in enumerate(looks):
  y=55+j*330;draw.text((5,y),f"look {j+1}; seed {seed}; quality {look['quality']['score']:.3f}",fill='black')
  ordered=sorted(look['items'],key=lambda i:['top','outerwear','bottom','shoes'].index(i['macroCategory']))
  for k,item in enumerate(ordered):
   x=k*300;path,error=images[item['id']]
   if path:
    img=ImageOps.contain(Image.open(path),(290,230));sheet.paste(img,(x+(300-img.width)//2,y+23+(230-img.height)//2))
   else:draw.text((x+10,y+75),f'IMAGE UNAVAILABLE: {error}',fill='red')
   label=f"{item['macroCategory']}: {item['name']}"
   draw.multiline_text((x+5,y+255),'\n'.join(textwrap.wrap(label,43))[:210],fill='black')
  rows.append({'seed':seed,'items':[{'id':i['id'],'name':i['name'],'url':i['imageUrl'],'downloaded':bool(images[i['id']][0]),'error':images[i['id']][1]} for i in ordered]})
 out=root/f'{prefix}contact-{n+1}.jpg';sheet.save(out,quality=87)
 manifest.append({'file':out.name,'style':c['style'],'occasion':c['occasion'],'looks':rows})
(root/(prefix+'visual-manifest.json')).write_text(json.dumps(manifest,indent=2))
print(json.dumps({'looks':sum(len(x['looks']) for x in manifest),'uniqueImages':len(images),'downloaded':sum(bool(p) for p,e in images.values()),'failed':sum(not p for p,e in images.values())}))
