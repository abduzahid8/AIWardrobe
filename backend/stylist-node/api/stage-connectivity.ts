/**
 * Temporary protected Vercel preview check of Supabase -> Vercel HMAC
 * connectivity. No user data, credentials, or original generation involved.
 * Remove after transport is verified.
 */
export default {
  async fetch(request: Request): Promise<Response> {
    if (request.method !== 'GET') return Response.json({status:'method_not_allowed'},{status:405});
    try {
      const upstream=await fetch(
        'https://fyqpifmrsftsfqibhwhy.supabase.co/functions/v1/aiw-stylist-node-staging/health',
        {method:'GET',redirect:'error',signal:AbortSignal.timeout(14000)}
      );
      const data=await upstream.json().catch(()=>({}));
      const ok=upstream.status===200 && data.signedTransport===true && data.upstreamStatus===200;
      return Response.json({
        status:ok?'ok':'connection_failed',
        supabaseStagingHTTP:upstream.status,
        hmacVerified:ok,
        upstreamHTTP:Number.isInteger(data.upstreamStatus)?data.upstreamStatus:null,
      },{status:ok?200:503,headers:{'Cache-Control':'no-store'}});
    }catch{
      return Response.json({status:'connection_unavailable',hmacVerified:false},
        {status:503,headers:{'Cache-Control':'no-store'}});
    }
  },
};
