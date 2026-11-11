/**
 * Isolated Stylist 6 cloud staging route (Vercel Node.js).
 *
 * This is NOT the Stylist 5 compute route: it runs the Stylist 6 fashion
 * engine server-side and reports a truthful `stylist-6-*` generation
 * version. Access requires a short-lived HMAC signed by the isolated
 * Stylist 6 staging proxy, in addition to the original user's JWT.
 *
 * Nothing is metered here before the Stylist 6 handler runs; JWT
 * verification, rate limiting and consume_edge_quota happen inside the
 * handler exactly once per admitted request.
 */
import {
  verifyStylistTransport,
  STYLIST_TRANSPORT_MAX_BODY_BYTES,
} from '../transport/signature.mjs';
import { handleStylist6CloudGenerate, createCloudIo, STYLIST6_CLOUD_VERSION } from '../../stylist6/cloud_handler.mjs';
import * as supabaseJs from '@supabase/supabase-js';

function errorResponse(code: string, status: number): Response {
  return Response.json(
    { success: false, outfits: [], code, error: code, generationVersion: STYLIST6_CLOUD_VERSION },
    { status },
  );
}

function envOf(name: string): string {
  return process.env[name] || '';
}

export async function POST(request: Request): Promise<Response> {
  const secret = envOf('STYLIST_NODE_SHARED_SECRET');
  if (!secret || secret.length < 32) return errorResponse('stylist_compute_unconfigured', 503);

  const authorization = request.headers.get('authorization') || '';
  if (Number(request.headers.get('content-length') || 0) > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return errorResponse('invalid_generation_request', 413);
  }
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return errorResponse('invalid_generation_request', 413);
  }

  const authorized = await verifyStylistTransport({
    secret,
    body,
    authorization,
    timestamp: request.headers.get('x-aiw-timestamp'),
    nonce: request.headers.get('x-aiw-nonce'),
    signature: request.headers.get('x-aiw-signature'),
  });
  if (!authorized) return errorResponse('unauthorized', 401);

  const env = {
    STYLIST_NODE_SHARED_SECRET: secret,
    SUPABASE_URL: envOf('SUPABASE_URL'),
    SUPABASE_ANON_KEY: envOf('SUPABASE_ANON_KEY'),
    SUPABASE_SERVICE_ROLE_KEY: envOf('SUPABASE_SERVICE_ROLE_KEY'),
    NVIDIA_API_KEY: envOf('NVIDIA_API_KEY'),
    STYLIST6_REFERENCE_BUCKET: envOf('STYLIST6_REFERENCE_BUCKET'),
  };
  if (!env.SUPABASE_URL || !env.SUPABASE_ANON_KEY || !env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse('stylist_compute_unconfigured', 503);
  }

  const headers: Record<string, string> = { authorization };
  for (const name of ['x-aiw-timestamp', 'x-aiw-nonce', 'x-aiw-signature']) {
    const value = request.headers.get(name);
    if (value) headers[name] = value;
  }
  // Include cold-start module loading AND the Stylist 6 handler in timing.
  const cpuStart = process.cpuUsage();
  const wallStart = performance.now();
  try {
    const io = createCloudIo({
      url: env.SUPABASE_URL,
      anonKey: env.SUPABASE_ANON_KEY,
      serviceKey: env.SUPABASE_SERVICE_ROLE_KEY,
      supabaseJs,
    });
    const { status, body: responseBody } = await handleStylist6CloudGenerate({
      method: 'POST', headers, bodyText: body, env, io,
    });
    const cpu = process.cpuUsage(cpuStart);
    const cpuMs = (cpu.user + cpu.system) / 1000;
    const wallMs = performance.now() - wallStart;
    return Response.json(responseBody, {
      status,
      headers: {
        'server-timing': `aiw_node_cpu;dur=${cpuMs.toFixed(1)}, aiw_node_wall;dur=${wallMs.toFixed(1)}`,
      },
    });
  } catch {
    // Never leak backend internals, JWTs, product data or service credentials.
    return errorResponse('stylist_compute_unavailable', 503);
  }
}

export function GET(): Response {
  return errorResponse('method_not_allowed', 405);
}
