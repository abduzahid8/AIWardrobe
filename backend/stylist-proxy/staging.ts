/**
 * ISOLATED staging Edge function. Does NOT replace production generate-outfits.
 * Uses a pinned, single-deployment Vercel preview URL. That deployment is
 * reachable by the Supabase service, but its generation route requires the
 * existing shared HMAC and the original verified user JWT and paid quota.
 */
import { signStylistTransport, STYLIST_TRANSPORT_MAX_BODY_BYTES } from '../stylist-node/transport/signature.mjs';

const NODE_ORIGIN = 'https://aiwardrobe-stylist-compute-697pmz6vi-abduzahid8s-projects.vercel.app';
const CORS = {
  'Access-Control-Allow-Origin':'*',
  'Access-Control-Allow-Headers':'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods':'GET, POST, OPTIONS',
  'Access-Control-Expose-Headers':'server-timing',
  'Content-Type':'application/json',
};
const fail = (code:string,status:number) =>
  Response.json({success:false,outfits:[],code,error:code},{status,headers:CORS});

Deno.serve(async(request:Request):Promise<Response>=>{
  if(request.method==='OPTIONS') return new Response('ok',{headers:CORS});

  const secret = Deno.env.get('STYLIST_NODE_SHARED_SECRET');
  if (!secret || secret.length < 32) return fail('staging_transport_unconfigured',503);

  // A transport-only probe which never invokes original Stylist generation.
  if(request.method==='GET' && new URL(request.url).pathname.endsWith('/health')){
    const body='{"probe":"aiw-stylist-transport-v1"}';
    const authorization='Bearer aiw-transport-probe-no-user';
    try{
      const signature = await signStylistTransport({secret,body,authorization});
      const response = await fetch(NODE_ORIGIN+'/api/transport-health',{
        method:'POST',
        headers:{...signature,authorization,'content-type':'application/json'},
        body,redirect:'error',signal:AbortSignal.timeout(9000),
      });
      const success = response.status===200 && response.headers.get('content-type')?.includes('application/json');
      return Response.json(
        {status:success?'ok':'upstream_rejected',signedTransport:!!success,upstreamStatus:response.status},
        {status:success?200:503,headers:{...CORS,'Cache-Control':'no-store'}}
      );
    }catch{
      return fail('staging_transport_unavailable',503);
    }
  }
  if(request.method!=='POST')return fail('method_not_allowed',405);

  const authorization = request.headers.get('authorization')||'';
  if(!/^Bearer\s+\S+$/i.test(authorization))return fail('unauthorized',401);
  if(Number(request.headers.get('content-length')||0)>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  const body=await request.text();
  if(new TextEncoder().encode(body).byteLength>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  try{
    const signature=await signStylistTransport({secret,body,authorization});
    const response=await fetch(NODE_ORIGIN+'/api/generate-outfits',{
      method:'POST',redirect:'error',
      headers:{
        ...signature,authorization,
        apikey:request.headers.get('apikey')||'',
        'content-type':'application/json',
      },
      body,signal:AbortSignal.timeout(45000),
    });
    if(!response.headers.get('content-type')?.includes('application/json'))
      return fail('staging_compute_unavailable',502);
    const headers=new Headers(CORS);
    const timing=response.headers.get('server-timing');
    if(timing && /^aiw_node_cpu;dur=\d+(?:\.\d+)?, aiw_node_wall;dur=\d+(?:\.\d+)?$/.test(timing))
      headers.set('server-timing',timing);
    return new Response(response.body,{status:response.status,headers});
  }catch{
    return fail('staging_compute_unavailable',503);
  }
});
