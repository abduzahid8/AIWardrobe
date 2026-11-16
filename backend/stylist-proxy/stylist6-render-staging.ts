/** Isolated Render canary for Stylist 6. Never changes S5 or the
 * currently working Vercel-backed Supabase staging function. JWT verified
 * by Supabase + forwarded unchanged; HMAC covers exact body and JWT.
 * No retry: quota can be consumed once by the real cloud handler.
 */
import {signStylistTransport,STYLIST_TRANSPORT_MAX_BODY_BYTES} from '../stylist-node/transport/signature.mjs';

const RENDER_ORIGIN='https://aiwardrobe-stylist6-staging-compute.onrender.com';
const RENDER_PATH='/api/stylist6-generate-outfits';
const MAX_COMPUTE_RESPONSE_BYTES=2*1024*1024;
const CORS={
  'Access-Control-Allow-Origin':'*',
  'Access-Control-Allow-Headers':'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods':'POST, OPTIONS',
  'Access-Control-Expose-Headers':'server-timing',
  'Content-Type':'application/json',
};
const fail=(code:string,status:number)=>Response.json(
  {success:false,outfits:[],code,error:code}, {status,headers:CORS});

Deno.serve(async(request:Request):Promise<Response>=>{
  if(request.method==='OPTIONS')return new Response('ok',{headers:CORS});
  if(request.method!=='POST')return fail('method_not_allowed',405);
  const secret=Deno.env.get('STYLIST6_RENDER_NODE_SHARED_SECRET');
  if(!secret || secret.length<32)return fail('render_transport_unconfigured',503);
  const authorization=request.headers.get('authorization')||'';
  if(!/^Bearer\s+\S+$/i.test(authorization))return fail('unauthorized',401);
  if(Number(request.headers.get('content-length')||0)>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  const body=await request.text();
  if(new TextEncoder().encode(body).byteLength>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  const computeDeadline=AbortSignal.timeout(85000);
  try {
    const signature=await signStylistTransport({secret,body,authorization});
    const response=await fetch(RENDER_ORIGIN+RENDER_PATH,{
      method:'POST',redirect:'error',
      headers:{...signature,authorization,apikey:request.headers.get('apikey')||'',
        'content-type':'application/json'},
      body,signal:computeDeadline,
    });
    if(!response.headers.get('content-type')?.includes('application/json'))
      return fail('render_compute_unavailable',502);
    // Do NOT forward an upstream 200 with a still-pending streamed body.
    // On a slow Render instance the caller received 200 headers and then an
    // aborted/empty body after its 100s deadline. Buffer and validate the
    // complete, bounded JSON first; an interrupted body is an HTTP error.
    const announcedSize=Number(response.headers.get('content-length')||0);
    if(announcedSize>MAX_COMPUTE_RESPONSE_BYTES)return fail('render_response_too_large',502);
    const payload=await response.text(); // covered by computeDeadline, including body read
    if(new TextEncoder().encode(payload).byteLength>MAX_COMPUTE_RESPONSE_BYTES)
      return fail('render_response_too_large',502);
    let parsed:unknown;
    try{parsed=JSON.parse(payload);}catch{return fail('render_invalid_json',502);}
    if(!parsed || typeof parsed!=='object' || Array.isArray(parsed))
      return fail('render_invalid_json',502);
    const headers=new Headers(CORS);
    const timing=response.headers.get('server-timing');
    if(timing && /^aiw_node_cpu;dur=\d+(?:\.\d+)?, aiw_node_wall;dur=\d+(?:\.\d+)?$/.test(timing))
      headers.set('server-timing',timing);
    return new Response(payload,{status:response.status,headers});
  }catch(error){
    const name=error instanceof Error?error.name:'';
    return computeDeadline.aborted || name==='AbortError' || name==='TimeoutError'
      ? fail('render_compute_timeout',504)
      : fail('render_compute_unavailable',503);
  }
});
