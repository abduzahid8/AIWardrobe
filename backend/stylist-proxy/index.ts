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

Deno.serve(async (request: Request): Promise<Response> => {
  if (request.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });
  if (request.method !== 'POST') return jsonError('method_not_allowed', 405);

  const authorization = request.headers.get('authorization') || '';
  if (!/^Bearer\s+\S+$/i.test(authorization)) return jsonError('unauthorized', 401);

  const endpoint = Deno.env.get('STYLIST_NODE_URL');
  const secret = Deno.env.get('STYLIST_NODE_SHARED_SECRET');
  if (!endpoint || !secret || secret.length < 32) return jsonError('stylist_compute_unconfigured', 503);

  let destination: URL;
  try {
    destination = new URL(endpoint);
    if (destination.protocol !== 'https:' || destination.username || destination.password ||
        destination.search || destination.hash ||
        destination.pathname !== '/api/generate-outfits') throw new Error('invalid_target');
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
    const upstream = await fetch(destination, {
      method: 'POST',
      headers: {
        ...signatureHeaders,
        authorization,
        apikey: request.headers.get('apikey') || '',
        'content-type': 'application/json',
      },
      body,
      signal: AbortSignal.timeout(45_000),
    });
    if (!upstream.headers.get('content-type')?.toLowerCase().includes('application/json')) {
      return jsonError('stylist_compute_unavailable', 502);
    }
    // Only whitelisted non-sensitive runtime timing is forwarded. No
    // provider details, tokens, product records, or internal debug headers.
    const headers = new Headers(corsHeaders);
    const timing = upstream.headers.get('server-timing');
    if (timing && /^aiw_node_cpu;dur=\d+(?:\.\d+)?, aiw_node_wall;dur=\d+(?:\.\d+)?$/.test(timing)) {
      headers.set('server-timing', timing);
    }
    return new Response(upstream.body, {
      status: upstream.status,
      headers,
    });
  } catch {
    // Never log JWTs, products, request bodies, HMAC signatures, or secrets.
    return jsonError('stylist_compute_unavailable', 503);
  }
});
