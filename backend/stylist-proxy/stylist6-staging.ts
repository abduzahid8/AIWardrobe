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

// Pinned Stylist 6 Fluid Compute staging deployment (branch ai_fashion_stylist).
// Active READY Compute Preview 5c925fhk2 (dpl_7zEUvoW2bh4HSmW98c5EWjEDJeii; commit 822fd095; maxDuration: 300).
// Previous rollback targets: gibkm7rl5 (commit 3b2779d2), Render canary (stylist6-render-staging.ts).
// Runs Phase 11-13 Autonomous Visual Stylist (Act 1 Showroom + Act 2 Portfolio Editor +
// Parallel Visual Swaps) with 75s default compute budget and zero Render sleep cold starts.
// Production Stylist 5 remains untouched.
// Keep the JWT + HMAC transport and quotas unchanged; no production routing.
// Fail-closed idempotency + RLS migration included.
// Repoint ONLY to another verified Stylist 6 staging Preview, never to production.
const DEFAULT_NODE_ORIGIN = 'https://aiwardrobe-stylist-compute-5c925fhk2-abduzahid8s-projects.vercel.app';
const NODE_ROUTE = '/api/stylist6-generate-outfits';
const MAX_COMPUTE_RESPONSE_BYTES = 2 * 1024 * 1024;
const CORS = {
  'Access-Control-Allow-Origin':'*',
  'Access-Control-Allow-Headers':'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods':'GET, POST, OPTIONS',
  'Access-Control-Expose-Headers':'server-timing',
  'Content-Type':'application/json',
};
const fail = (code:string,status:number) =>
  Response.json({success:false,outfits:[],code,error:code},{status,headers:CORS});

function resolveNodeOrigin(): string | null {
  const raw = (Deno.env.get('STYLIST6_NODE_ORIGIN')?.trim() || DEFAULT_NODE_ORIGIN).replace(/\/+$/, '');
  try {
    const u = new URL(raw);
    if (u.protocol !== 'https:') return null;
    if (!u.hostname.toLowerCase().endsWith('.vercel.app')) return null;
    if (u.pathname && u.pathname !== '/') return null;
    return u.origin;
  } catch {
    return null;
  }
}

Deno.serve(async(request:Request):Promise<Response>=>{
  if(request.method==='OPTIONS') return new Response('ok',{headers:CORS});

  // GET must not trigger even non-billable signed upstream work.
  if(request.method!=='POST')return fail('method_not_allowed',405);

  const secret = Deno.env.get('STYLIST6_NODE_SHARED_SECRET');
  if (!secret || secret.length < 32) return fail('staging_transport_unconfigured',503);

  const nodeOrigin = resolveNodeOrigin();
  if (!nodeOrigin) return fail('staging_transport_unconfigured',503);

  const authorization = request.headers.get('authorization')||'';
  if(!/^Bearer\s+\S+$/i.test(authorization))return fail('unauthorized',401);
  if(Number(request.headers.get('content-length')||0)>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  const body=await request.text();
  if(new TextEncoder().encode(body).byteLength>STYLIST_TRANSPORT_MAX_BODY_BYTES)
    return fail('invalid_generation_request',413);
  const computeDeadline = AbortSignal.timeout(85000);
  try{
    // Optional Vercel deployment protection bypass is independent of the
    // HMAC transport and original user JWT; never forward to another host.
    const bypass = Deno.env.get('STYLIST_VERCEL_BYPASS_SECRET')?.trim();
    if (bypass && !/^[a-zA-Z0-9_-]{32,128}$/.test(bypass))
      return fail('staging_transport_unconfigured',503);
    const signature=await signStylistTransport({secret,body,authorization});
    const response=await fetch(nodeOrigin+NODE_ROUTE,{
      method:'POST',redirect:'error',
      headers:{
        ...signature,authorization,
        apikey:request.headers.get('apikey')||'',
        'content-type':'application/json',
        ...(bypass ? {'x-vercel-protection-bypass':bypass} : {}),
      },
      body,signal:computeDeadline,
    });
    if(!response.headers.get('content-type')?.includes('application/json'))
      return fail('staging_compute_unavailable',502);
    const announcedSize = Number(response.headers.get('content-length') || 0);
    if (announcedSize > MAX_COMPUTE_RESPONSE_BYTES) return fail('staging_response_too_large', 502);
    const payload = await response.text();
    if (new TextEncoder().encode(payload).byteLength > MAX_COMPUTE_RESPONSE_BYTES)
      return fail('staging_response_too_large', 502);
    let parsed: unknown;
    try { parsed = JSON.parse(payload); } catch { return fail('staging_invalid_json', 502); }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
      return fail('staging_invalid_json', 502);
    const headers=new Headers(CORS);
    const timing=response.headers.get('server-timing');
    if(timing && /^aiw_node_cpu;dur=\d+(?:\.\d+)?, aiw_node_wall;dur=\d+(?:\.\d+)?$/.test(timing))
      headers.set('server-timing',timing);
    return new Response(payload,{status:response.status,headers});
  }catch(error){
    const name = error instanceof Error ? error.name : '';
    return computeDeadline.aborted || name === 'AbortError' || name === 'TimeoutError'
      ? fail('staging_compute_timeout', 504)
      : fail('staging_compute_unavailable', 503);
  }
});
