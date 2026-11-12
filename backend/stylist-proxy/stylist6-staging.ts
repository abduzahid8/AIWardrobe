/**
 * ISOLATED Stylist 6 staging Edge function. Does NOT replace production
 * generate-outfits and does NOT touch the Stylist 5 staging proxy.
 *
 * Uses a pinned, single-deployment Vercel Preview URL serving the isolated
 * `stylist6-generate-outfits` route (truthful `stylist-6-*` versions only).
 * The proxy signs each request with the STYLIST 6 STAGING shared HMAC
 * (function secret STYLIST6_NODE_SHARED_SECRET, a staging-only value that
 * matches the Preview branch env — never a production secret) and forwards
 * the original verified user JWT untouched.
 *
 * Quota is enforced exactly once inside the compute route
 * (consume_edge_quota on the production outfit-generation bucket); this
 * proxy never debits and never retries (a retry would double-charge).
 *
 * DEPLOY (owner action, isolated function only):
 *   supabase functions deploy aiw-stylist6-node-staging --project-ref fyqpifmrsftsfqibhwhy
 *   supabase secrets set STYLIST6_NODE_SHARED_SECRET='<staging-only-64-hex>' \
 *     STYLIST_VERCEL_BYPASS_SECRET='<protection-bypass>' --project-ref fyqpifmrsftsfqibhwhy
 */
import { signStylistTransport, STYLIST_TRANSPORT_MAX_BODY_BYTES } from '../stylist-node/transport/signature.mjs';

// Pinned Stylist 6 staging deployment (branch codex/stylist6-visual-first-20261110).
// Isolated READY Preview pinned to commit f53505fa73eeb5cfcb0edebd784e5b95eb11eb6b.
// Includes source-role guard, personal-taste ranking and three qualified
// creative directions; no production routing or paid inference changes.
// Keep the JWT + HMAC transport and quotas unchanged; no production routing.
// Fail-closed idempotency + RLS migration included.
// Repoint ONLY to another verified Stylist 6 staging Preview, never to production.
const NODE_ORIGIN = 'https://aiwardrobe-stylist-compute-ovmvgsdkf-abduzahid8s-projects.vercel.app';
const NODE_ROUTE = '/api/stylist6-generate-outfits';
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

  // GET must not trigger even non-billable signed upstream work.
  if(request.method!=='POST')return fail('method_not_allowed',405);

  const secret = Deno.env.get('STYLIST6_NODE_SHARED_SECRET');
  if (!secret || secret.length < 32) return fail('staging_transport_unconfigured',503);


  const authorization = request.headers.get('authorization')||'';
  if(!/^Bearer\s+\S+$/i.test(authorization))return fail('unauthorized',401);
  if(Number(request.headers.get('content-length')||0)>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  const body=await request.text();
  if(new TextEncoder().encode(body).byteLength>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  try{
    // Optional Vercel deployment protection bypass is independent of the
    // HMAC transport and original user JWT; never forward to another host.
    const bypass = Deno.env.get('STYLIST_VERCEL_BYPASS_SECRET')?.trim();
    if (bypass && (!/^[a-zA-Z0-9_-]{32,128}$/.test(bypass) ||
        !new URL(NODE_ORIGIN).hostname.toLowerCase().endsWith('.vercel.app')))
      return fail('staging_transport_unconfigured',503);
    const signature=await signStylistTransport({secret,body,authorization});
    const response=await fetch(NODE_ORIGIN+NODE_ROUTE,{
      method:'POST',redirect:'error',
      headers:{
        ...signature,authorization,
        apikey:request.headers.get('apikey')||'',
        'content-type':'application/json',
        ...(bypass ? {'x-vercel-protection-bypass':bypass} : {}),
      },
      body,signal:AbortSignal.timeout(55000),
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
