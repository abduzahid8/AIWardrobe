/** Temporary diagnostic handler. Deploy only with fixed disposable fixture configuration. */
export const MEDIA_PROBE_PHASES = Object.freeze([2, 5, 10]);
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const validObjects = ids => ids === undefined || (Array.isArray(ids) && ids.length === 170 && new Set(ids).size === 170 && ids.every(id => typeof id === 'string' && UUID.test(id)));
const validConcurrency = value => value === undefined || value === 20 || value === 40;
const validExtension = value => value === undefined || value === 'png' || value === 'jpg';
const json = (value, status = 200) => new Response(JSON.stringify(value), {status, headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
const percentile = (values, fraction) => values.length ? [...values].sort((a,b)=>a-b)[Math.ceil(values.length*fraction)-1] : null;
const hash = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');

async function boundedBody(response, expectedBytes) {
  const reader=response.body?.getReader();
  if(!reader)throw new Error('Missing image body');
  const bytes=new Uint8Array(expectedBytes);let offset=0;
  try {
    for(;;){
      const {value,done}=await reader.read();if(done)break;
      if(offset+value.byteLength>expectedBytes){await reader.cancel();throw new Error('Excess image bytes');}
      bytes.set(value,offset);offset+=value.byteLength;
    }
    if(offset!==expectedBytes)throw new Error('Incomplete image body');
    return bytes.buffer;
  }finally{reader.releaseLock();}
}

export async function measureRemoteMedia(config, authorization, dependencies = {}) {
  if (!validObjects(config.objectIds) || !validExtension(config.thumbnailExtension) || !validConcurrency(config.concurrencyCap)) throw new Error('Invalid probe object cohort');
  const issuedObjects = new Set(); let cursor = 0;
  const concurrencyCap = config.concurrencyCap ?? 20;
  const request = dependencies.fetch || fetch;
  const now = dependencies.now || (()=>performance.now());
  const wallNow = dependencies.wallNow || dependencies.now || Date.now;
  const sleep = dependencies.sleep || (ms => new Promise(resolve=>setTimeout(resolve,ms)));
  const digest = dependencies.digest || hash;
  const stages = [];
  async function read() {
    const objectId = config.objectIds ? config.objectIds[cursor++] : config.objectId;
    issuedObjects.add(objectId);
    const url = `${config.projectUrl}/storage/v1/object/authenticated/wardrobe-images/${config.owner}/${objectId}/thumbnail.${config.thumbnailExtension || 'png'}`;
    const start = now(); let headersMs = null, status = null, phase = 'response', cache = {};
    try {
      const remaining = config.expiresAt-wallNow();
      if (remaining <= 0) throw new Error('ProbeExpired');
      const response = await request(url, {headers:{Authorization:authorization},signal:AbortSignal.timeout(Math.max(1,Math.min(15000,remaining)))});
      headersMs = now()-start; status = response.status;
      cache = Object.fromEntries(['cf-cache-status','x-cache','age','content-length','via'].map(name => [name,response.headers.get(name)?.slice(0,120) ?? null]));
      phase = 'body';
      const bytes = await boundedBody(response,config.bytes);
      phase = 'hash';
      const actualHash = await digest(bytes);
      phase = 'integrity';
      const ok = response.ok && bytes.byteLength === config.bytes && actualHash === config.sha256;
      return {ok,objectId,status,cache,phase,headersMs,bodyMs:now()-start-headersMs,ms:now()-start,bytes:bytes.byteLength};
    } catch (error) { return {ok:false,objectId,status,cache,phase,headersMs,ms:now()-start,error:error.name,bytes:0}; }
  }
  for (const rps of MEDIA_PROBE_PHASES) {
    const planned=rps*10, results=[],active=new Set();let dropped=0,peakActiveReads=0;
    const start=now();
    for(let index=0;index<planned;index++) {
      const delay=start+index*1000/rps-now();if(delay>0)await sleep(delay);
      if(wallNow()>=config.expiresAt)break;
      if(active.size>=concurrencyCap){dropped++;continue;}
      const job=read().then(result=>results.push(result));active.add(job);peakActiveReads=Math.max(peakActiveReads,active.size);job.finally(()=>active.delete(job));
    }
    await Promise.all(active);
    const stage={concurrencyCap,peakActiveReads,targetRps:rps,seconds:10,planned,completed:results.length,dropped,failures:results.filter(x=>!x.ok).length,elapsedMs:now()-start,
      p95Ms:percentile(results.map(x=>x.ms),.95),p95HeadersMs:percentile(results.map(x=>x.headersMs).filter(x=>x!==null),.95),
      p95BodyMs:percentile(results.filter(x=>x.ok).map(x=>x.bodyMs),.95),results,bytes:results.reduce((total,x)=>total+x.bytes,0)};
    stages.push(stage);
    if(stage.completed!==planned||stage.failures||stage.dropped)break;
  }
  return {scope:'Same-project remote Storage transport diagnostic; not 10000 DAU certification',region:config.region||null,concurrencyCap,stages,
    distinctIssuedObjects:issuedObjects.size,objectPattern:config.objectIds?'distinct fixture URLs':'repeated fixture URL',passed:stages.length===3&&stages.every(x=>x.completed===x.planned&&!x.dropped&&!x.failures)};
}

export function createRemoteMediaProbe(config, dependencies = {}) {
  const now=dependencies.now||Date.now, request=dependencies.fetch||fetch;
  const project=new URL(config.projectUrl);
  if(project.protocol!=='https:'||project.username||project.password||project.pathname!=='/'||project.search||project.hash
    ||!validConcurrency(config.concurrencyCap)||!validExtension(config.thumbnailExtension)||!UUID.test(config.owner)||!UUID.test(config.objectId)||!validObjects(config.objectIds)||!Number.isSafeInteger(config.bytes)||config.bytes<1||config.bytes>2*1024*1024
    ||!Number.isSafeInteger(config.expiresAt)||config.expiresAt>now()+600000||!/^[a-f0-9]{64}$/.test(config.sha256)||!config.anonKey)throw new Error('Invalid probe configuration');
  let used=false;
  return async requestIn => {
    if(requestIn.method!=='POST')return json({error:'Method not allowed'},405);
    if(now()>=config.expiresAt)return json({error:'Probe expired'},410);
    const authorization=requestIn.headers.get('Authorization');
    if(!authorization?.startsWith('Bearer '))return json({error:'Authentication required'},401);
    let user;
    try {
      const response=await request(`${config.projectUrl}/auth/v1/user`,{headers:{Authorization:authorization,apikey:config.anonKey},signal:AbortSignal.timeout(6000)});
      if(!response.ok)return json({error:'Identity verification failed'},response.status>=500?503:401);
      user=await response.json();
    }catch{return json({error:'Identity verification unavailable'},503);}
    if(user.id!==config.owner)return json({error:'Fixture owner required'},403);
    if(used)return json({error:'Probe already used in this isolate'},409);
    used=true;
    try{return json(await (dependencies.measure||measureRemoteMedia)(config,authorization,dependencies));}
    catch{return json({error:'Probe failed'},503);}
  };
}
