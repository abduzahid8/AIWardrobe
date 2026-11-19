import {withRequestDeadline} from './requestDeadline';

/**
 * A signed, already-paid backend may finish AFTER a native HTTP relay loses
 * its original response. Poll ONLY a caller-supplied READ of the exact same
 * request's RLS-protected completed result. This helper NEVER posts a new
 * generation or accepts an outfit lacking the reader's server validation.
 *
 * The grace window is bounded; an explicit screen cancellation ends both
 * the database query and any scheduled wait. A genuine PHOTO/source reject
 * must never call this helper (the caller checks transport-only failures).
 */
export async function pollSettledReadOnlyResult<T>(
  read: (signal:AbortSignal)=>Promise<T|null>,
  options: {signal?:AbortSignal; maxWaitMs?:number; intervalMs?:number;
    queryTimeoutMs?:number}={},
): Promise<T|null> {
  if(typeof read!=='function'||options.signal?.aborted)return null;
  const maxWaitMs=Number.isFinite(options.maxWaitMs)
    ?Math.max(50,Math.min(25000,options.maxWaitMs!)):20000;
  const intervalMs=Number.isFinite(options.intervalMs)
    ?Math.max(10,Math.min(5000,options.intervalMs!)):1800;
  const queryTimeoutMs=Number.isFinite(options.queryTimeoutMs)
    ?Math.max(20,Math.min(4000,options.queryTimeoutMs!)):2500;
  const deadline=Date.now()+maxWaitMs;
  const pause=(ms:number)=>new Promise<void>(resolve=>{
    if(options.signal?.aborted){resolve();return;}
    const finish=()=>{
      clearTimeout(timer);
      options.signal?.removeEventListener('abort',finish);
      resolve();
    };
    const timer=setTimeout(finish,ms);
    options.signal?.addEventListener('abort',finish,{once:true});
    if(options.signal?.aborted)finish();
  });
  while(!options.signal?.aborted){
    const remaining=deadline-Date.now();
    if(remaining<=0)return null;
    try{
      const result=await withRequestDeadline(read,
        Math.min(queryTimeoutMs,remaining),options.signal);
      if(options.signal?.aborted)return null;
      if(result!==null&&result!==undefined)return result;
    }catch{
      // A transient read error is NOT permission to invoke paid generation.
      if(options.signal?.aborted)return null;
    }
    const left=deadline-Date.now();
    if(left<=0)return null;
    await pause(Math.min(intervalMs,left));
  }
  return null;
}
