/**
 * Protected preview-only negative admission test. Sends an intentionally
 * invalid JWT via the separate Supabase staging proxy. This can NEVER use
 * paid NVIDIA inference or consume a user quota, as the original rate
 * limiter authenticates before rate counters and quota changes.
 * Remove this diagnostic route before production.
 */
export default {
  async fetch(req:Request):Promise<Response>{
    if(req.method!=='GET')return Response.json({status:'method_not_allowed'},{status:405});
    try{
      const key=process.env.SUPABASE_ANON_KEY||'';
      if(key.length<30)return Response.json({status:'missing_preview_config'},{status:503});
      const upstream=await fetch(
        'https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/aiw-stylist-node-staging',
        {
          method:'POST',redirect:'error',signal:AbortSignal.timeout(16000),
          headers:{
            authorization:'Bearer invalid-aiw-staging-auth-probe-no-user',
            apikey:key,
            'content-type':'application/json',
          },
          body:'{"style":"old_money","occasion":"work"}',
        }
      );
      const body=await upstream.json().catch(()=>({}));
      const originalHandlerReached=!!upstream.headers.get('server-timing')?.startsWith('aiw_node_cpu;dur=');
      const rejected=upstream.status===401 && body.error==='unauthorized';
      return Response.json({
        status:rejected&&originalHandlerReached?'ok':'unexpected_result',
        upstreamHTTP:upstream.status,
        originalHandlerReached,
        unauthorizedRejected:rejected,
        errorCode:typeof body.error==='string'?body.error.slice(0,80):null,
      },{
        status:rejected&&originalHandlerReached?200:503,
        headers:{'Cache-Control':'no-store'}
      });
    }catch{
      return Response.json({status:'connection_unavailable'},{status:503,headers:{'Cache-Control':'no-store'}});
    }
  }
};
