"""Private real-product visual QA sheets. Missing images remain labeled.
Safe bounded image transport is shared with offline enrichment; no AI calls.
"""
import sys,json,pathlib,hashlib,concurrent.futures,textwrap
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'visual'))
from enrich_catalog import fetch_image
from PIL import Image,ImageOps,ImageDraw
root=pathlib.Path(sys.argv[1]);data=json.load(open(root/'benchmark-private.json'))
cache=root/'review-images';cache.mkdir(exist_ok=True)
products={i['id']:i for scenario in data for run in scenario['runs'] for look in run['deck'] for i in look['items']}
def download(item):
 path=cache/(hashlib.sha256(item['imageUrl'].encode()).hexdigest()+'.jpg')
 if path.exists():return item['id'],path
 try:
  _,img=fetch_image(item['imageUrl']);img.thumbnail((480,600));img.save(path);return item['id'],path
 except Exception:return item['id'],None
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:paths=dict(executor.map(download,products.values()))
for n,scenario in enumerate(data):
 sheet=Image.new('RGB',(1800,1600),'#f1efe9');draw=ImageDraw.Draw(sheet)
 draw.text((25,15),scenario['style']+' / '+scenario['occasion']+' | fixed real catalog | weather unknown',fill='black')
 for r,run in enumerate(scenario['runs']):
  y0=45+r*770;draw.text((25,y0),run['metrics']['label'],fill='black')
  for j,look in enumerate(run['deck']):
   x0=j*600+15;draw.text((x0,y0+22),look['concept'],fill='black')
   for k,item in enumerate(look['items']):
    x=x0+(k%2)*285;y=y0+50+(k//2)*350
    p=paths[item['id']]
    if p:
     img=ImageOps.contain(Image.open(p).convert('RGB'),(275,280));sheet.paste(img,(x+(275-img.width)//2,y))
    else:draw.text((x+25,y+120),'IMAGE UNAVAILABLE',fill='red')
    title=item['name'].encode('ascii','replace').decode();caption='\n'.join(textwrap.wrap(title,38)[:2])+f"\n{item['price']:,.0f} {item['currency']}"
    draw.text((x,y+283),caption,fill='black')
 sheet.save(root/f'comparison-{n}.jpg',quality=88)
print(json.dumps({'sheets':len(data),'images':len(products),'unavailable':sum(p is None for p in paths.values()),'inferenceCalls':0}))
