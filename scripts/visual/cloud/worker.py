"""Authenticated cloud CPU worker. Durable progress lives in Supabase, not logs."""
import os,sys,time,uuid,json,requests,tempfile,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from enrich_catalog import CatalogEnricher,MODEL_REVISION
endpoint=os.environ['STYLIST_VISUAL_WORKER_URL']
credential=os.environ['STYLIST_VISUAL_WORKER_TOKEN']
worker=str(uuid.uuid4());session=requests.Session()
headers={'Authorization':'Bearer '+credential,'Content-Type':'application/json'}
def call(action,payload=None):
 for attempt in range(5):
  try:
   r=session.post(endpoint,headers=headers,json={'action':action,'worker':worker,'payload':payload or {}},timeout=(10,40))
   if r.status_code in (401,403):raise RuntimeError('worker authentication rejected')
   if r.status_code==400:raise RuntimeError('worker contract rejected')
   r.raise_for_status();return r.json()['data']
  except requests.RequestException:
   if attempt==4:raise RuntimeError('worker control plane unavailable') from None
   time.sleep(min(30,2**attempt))
# A push can start before the new control-plane isolate is ready. No inference
# or model download begins until scoped authentication and queue access succeed.
for attempt in range(20):
 try:call('seed');break
 except RuntimeError:
  if attempt==19:raise
  time.sleep(30)
if os.environ.get('STYLIST_VISUAL_STATE_ONLY')=='1':
 print(json.dumps(call('state')),flush=True);sys.exit(0)
started=time.monotonic();limit=int(os.environ.get('STYLIST_VISUAL_RUN_SECONDS','14400'))
enricher=None
while time.monotonic()-started<limit:
 rows=call('lease')
 if not rows:
  state=call('state');print(json.dumps({'progress':state}),flush=True)
  if state['remaining']==0:break
  time.sleep(20);continue
 if enricher is None:enricher=CatalogEnricher(MODEL_REVISION,cache='/tmp/stylist-visual-cache')
 results=[]
 for row in rows:
  try:
   result=enricher.enrich(row);result['source_hash']=row['source_hash'];result['status']='succeeded';results.append(result)
  except Exception as error:
   reason='download_unavailable' if isinstance(error,(requests.RequestException,ValueError,OSError)) else 'inference_failed'
   results.append({'product_id':row['id'],'source_hash':row['source_hash'],'status':'unknown','reason':reason})
  # Every completed image is durable immediately; a stopped runner cannot lose
  # the rest of a batch or force paid/local re-generation of successful images.
  call('complete',{'rows':[results.pop()]})
 print(json.dumps({'progress':call('state')}),flush=True)
print(json.dumps({'finished':call('state'),'modelRevision':MODEL_REVISION}),flush=True)
