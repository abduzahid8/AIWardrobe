import { BudgetFxError, type BudgetFxSnapshot, parseCbuBudgetRates, uzbekDateKey, validateBudgetFxSnapshot } from './budgetFx.ts';

// Coalesce local calls; the service-only database lease coordinates cold isolates.
let refresh: Promise<BudgetFxSnapshot> | null = null;
async function bounded<T>(work: (signal: AbortSignal) => PromiseLike<T>, ms: number, parent?: AbortSignal): Promise<T> {
  const controller=new AbortController();
  const abort=()=>controller.abort();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let rejectAbort: (()=>void) | undefined;
  const cancelled=new Promise<never>((_,reject)=>{rejectAbort=()=>reject(new BudgetFxError());controller.signal.addEventListener('abort',rejectAbort,{once:true});});
  parent?.addEventListener('abort',abort,{once:true});
  if(parent?.aborted)controller.abort();
  timer=setTimeout(()=>controller.abort(),ms);
  try {return await Promise.race([Promise.resolve().then(()=>{if(controller.signal.aborted)throw new BudgetFxError();return work(controller.signal);}),cancelled]);}
  finally {clearTimeout(timer);parent?.removeEventListener('abort',abort);if(rejectAbort)controller.signal.removeEventListener('abort',rejectAbort);controller.abort();}
}
async function cachedSnapshot(admin:any,now:number,signal:AbortSignal):Promise<BudgetFxSnapshot|null>{
  const result:any=await bounded(s=>admin.from('budget_exchange_rates').select('snapshot').order('fetched_at',{ascending:false}).limit(1).abortSignal(s).maybeSingle(),5000,signal);
  if(result.error)throw new BudgetFxError();
  if(result.data?.snapshot){try{return validateBudgetFxSnapshot(result.data.snapshot,now)}catch{/* Expired values must never be published as fresh. */}}
  return null;
}
async function leaseRpc(admin:any,name:string,args:Record<string,unknown>,signal?:AbortSignal,ms=2000):Promise<any>{
  const result:any=await bounded(s=>admin.rpc(name,args).abortSignal(s),ms,signal);
  if(result.error)throw new BudgetFxError();return result.data;
}
async function waitForOwner(ms:number,signal:AbortSignal):Promise<void>{
  await bounded(s=>new Promise<void>((resolve,reject)=>{const timer=setTimeout(resolve,ms);s.addEventListener('abort',()=>{clearTimeout(timer);reject(new BudgetFxError())},{once:true});}),ms+100,signal);
}
export async function currentBudgetRates(admin: any, now = Date.now(), request: typeof fetch = fetch): Promise<BudgetFxSnapshot> {
  if(refresh)return refresh;
  const pending=(async()=>{
    let token:string|null=null,stage='cache_read';
    try{return await bounded(async signal=>{
      let delay=100;
      while(!signal.aborted){
        stage='cache_read';const cached=await cachedSnapshot(admin,now,signal);if(cached)return cached;
        stage='lease_claim';const candidate=crypto.randomUUID();
        const state=await leaseRpc(admin,'claim_budget_rate_refresh',{p_token:candidate},signal);
        if(state==='cooldown')throw new BudgetFxError();
        if(state==='busy'){await waitForOwner(delay,signal);delay=Math.min(1000,delay*2);continue;}
        if(state!=='acquired')throw new BudgetFxError();token=candidate;
        // Another owner can have published between our first read and claim.
        stage='cache_recheck';let snapshot=await cachedSnapshot(admin,now,signal);
        if(!snapshot){
          snapshot=await bounded(async upstreamSignal=>{
            stage='upstream_fetch';
            const url=`https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/${uzbekDateKey(now)}/`;
            const response=await request(url,{signal:upstreamSignal,redirect:'error',headers:{Accept:'application/json'}});
            if(upstreamSignal.aborted||!response.ok||Number(response.headers.get('content-length')||0)>262144)throw new BudgetFxError();
            stage='upstream_body';const reader=response.body?.getReader();if(!reader)throw new BudgetFxError();
            const chunks:Uint8Array[]=[];let size=0;
            try{for(;;){const {done,value}=await reader.read();if(upstreamSignal.aborted)throw new BudgetFxError();if(done)break;size+=value.byteLength;if(size>262144){await reader.cancel();throw new BudgetFxError();}chunks.push(value);}}
            finally{reader.releaseLock();}
            const bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}
            stage='upstream_validation';return parseCbuBudgetRates(JSON.parse(new TextDecoder().decode(bytes)),now);
          },8000,signal);
        }
        if(signal.aborted)throw new BudgetFxError();
        stage='fenced_publish';const saved=await leaseRpc(admin,'finish_budget_rate_refresh',{p_token:token,p_snapshot:snapshot},signal);
        if(saved!==true||signal.aborted)throw new BudgetFxError();token=null;return snapshot;
      }
      throw new BudgetFxError();
    },12000)}catch{
      console.warn('[budget-rates] refresh_failed',{stage});
      if(token){try{await leaseRpc(admin,'finish_budget_rate_refresh',{p_token:token,p_snapshot:null},undefined,1000)}catch{/* The database lease expires even when cleanup cannot reach it. */}}
      throw new BudgetFxError();
    }
  })();
  refresh=pending;try{return await pending}finally{if(refresh===pending)refresh=null;}
}

/** A client may name a trusted stored snapshot, never supply its conversion values. */
export async function budgetRatesById(admin: any, id: string, now = Date.now()): Promise<BudgetFxSnapshot> {
  if (typeof id !== 'string' || !id.startsWith('CBU:') || id.length > 200) throw new BudgetFxError();
  const controller = new AbortController(); let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const query = admin.from('budget_exchange_rates').select('snapshot').eq('id', id).abortSignal(controller.signal).maybeSingle();
    const result: any = await Promise.race([query, new Promise<never>((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new BudgetFxError()); }, 5000); })]);
    if (result.error || !result.data?.snapshot) throw new BudgetFxError();
    const snapshot = validateBudgetFxSnapshot(result.data.snapshot, now);
    if (snapshot.id !== id) throw new BudgetFxError();
    return snapshot;
  } finally { if (timer) clearTimeout(timer); }
}
