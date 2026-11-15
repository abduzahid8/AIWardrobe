/**
 * Inactive until explicitly deployed as the existing "generate-outfits"
 * Supabase function. The mobile client keeps the same API URL and contract.
 *
 * Only transport is performed here: the Node worker retains the ORIGINAL
 * rate-limit, user authentication, subscription quota, catalog and styling
 * logic. Never meter requests a second time inside this proxy.
 */
import {
  signStylistTransport,
  STYLIST_TRANSPORT_MAX_BODY_BYTES,
} from '../stylist-node/transport/signature.mjs';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Expose-Headers': 'server-timing',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Content-Type': 'application/json',
};

function jsonError(code: string, status: number): Response {
  return new Response(JSON.stringify({ success: false, outfits: [], code, error: code }), {
    status, headers: corsHeaders,
  });
}

async function resolveVerifiedCohortUserId(authorization: string, apikey: string): Promise<string | null> {
  const supabaseUrl = Deno.env.get('SUPABASE_URL')?.trim();
  const anonKey = (Deno.env.get('SUPABASE_ANON_KEY') || apikey || '').trim();
  if (!supabaseUrl || !anonKey) return null;
  try {
    const authUrl = new URL('/auth/v1/user', supabaseUrl);
    if (authUrl.protocol !== 'https:') return null;
    const res = await fetch(authUrl, {
      method: 'GET',
      redirect: 'error',
      headers: { authorization, apikey: anonKey },
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) return null;
    const payload = await res.json();
    const id = typeof payload?.id === 'string' ? payload.id.trim() : '';
    return id || null;
  } catch {
    return null;
  }
}

async function inStylist6Cohort(userId: string): Promise<boolean> {
  const allowlist = (Deno.env.get('STYLIST6_ROLLOUT_USER_IDS') || '')
    .split(',')
    .map(entry => entry.trim())
    .filter(Boolean);
  if (allowlist.includes(userId)) return true;
  const percent = Number(Deno.env.get('STYLIST6_ROLLOUT_PERCENT') || 0);
  if (!Number.isFinite(percent) || percent <= 0) return false;
  if (percent >= 100) return true;
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`stylist6:${userId}`)));
  const bucket = ((digest[0] << 8) | digest[1]) % 100;
  return bucket < percent;
}

Deno.serve(async (request: Request): Promise<Response> => {
  if (request.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });
  if (request.method !== 'POST') return jsonError('method_not_allowed', 405);

  const authorization = request.headers.get('authorization') || '';
  if (!/^Bearer\s+\S+$/i.test(authorization)) return jsonError('unauthorized', 401);

  // Server-controlled engine selection: never trust caller-supplied engine flags.
  // Modes: 'stylist5' (default), 'stylist6' (100% Stylist 6), 'cohort' (verified user cohort).
  const rolloutMode = (Deno.env.get('STYLIST_ENGINE_ROLLOUT') || 'stylist5').trim().toLowerCase();
  let useStylist6 = rolloutMode === 'stylist6';
  if (rolloutMode === 'cohort') {
    const verifiedUserId = await resolveVerifiedCohortUserId(authorization, request.headers.get('apikey') || '');
    useStylist6 = Boolean(verifiedUserId && await inStylist6Cohort(verifiedUserId));
  }

  const endpoint = useStylist6
    ? Deno.env.get('STYLIST6_NODE_URL')
    : Deno.env.get('STYLIST_NODE_URL');
  const secret = useStylist6
    ? Deno.env.get('STYLIST6_NODE_SHARED_SECRET')
    : Deno.env.get('STYLIST_NODE_SHARED_SECRET');
  const expectedPath = useStylist6 ? '/api/stylist6-generate-outfits' : '/api/generate-outfits';
  if (!endpoint || !secret || secret.length < 32) return jsonError('stylist_compute_unconfigured', 503);

  let destination: URL;
  try {
    destination = new URL(endpoint);
    if (destination.protocol !== 'https:' || destination.username || destination.password ||
        destination.search || destination.hash ||
        destination.pathname !== expectedPath) throw new Error('invalid_target');
  } catch {
    return jsonError('stylist_compute_unconfigured', 503);
  }

  if (Number(request.headers.get('content-length') || 0) > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return jsonError('invalid_generation_request', 413);
  }
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return jsonError('invalid_generation_request', 413);
  }

  try {
    const signatureHeaders = await signStylistTransport({ secret, authorization, body });
    // Deployment Protection is independent of the application HMAC. Only
    // forward a separately configured automation bypass to *.vercel.app.
    // Redirects are forbidden so neither JWT nor HMAC can be sent elsewhere.
    const bypass = Deno.env.get('STYLIST_VERCEL_BYPASS_SECRET')?.trim();
    if (bypass && (!/^[a-zA-Z0-9_-]{32,128}$/.test(bypass) ||
        !destination.hostname.toLowerCase().endsWith('.vercel.app'))) {
      return jsonError('stylist_compute_unconfigured', 503);
    }
    const headers: Record<string, string> = {
      ...signatureHeaders,
      authorization,
      apikey: request.headers.get('apikey') || '',
      'content-type': 'application/json',
    };
    if (bypass) headers['x-vercel-protection-bypass'] = bypass;
    const upstream = await fetch(destination, {
      method: 'POST',
      headers,
      redirect: 'error',
      body,
      signal: AbortSignal.timeout(useStylist6 ? 55_000 : 45_000),
    });
    if (!upstream.headers.get('content-type')?.toLowerCase().includes('application/json')) {
      return jsonError('stylist_compute_unavailable', 502);
    }
    // Only whitelisted non-sensitive runtime timing is forwarded. No
    // provider details, tokens, product records, or internal debug headers.
    const responseHeaders = new Headers(corsHeaders);
    const timing = upstream.headers.get('server-timing');
    if (timing && /^aiw_node_cpu;dur=\d+(?:\.\d+)?, aiw_node_wall;dur=\d+(?:\.\d+)?$/.test(timing)) {
      responseHeaders.set('server-timing', timing);
    }
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch {
    // Never log JWTs, products, request bodies, HMAC signatures, or secrets.
    return jsonError('stylist_compute_unavailable', 503);
  }
});
