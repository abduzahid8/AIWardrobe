/**
 * Private Vercel Node.js execution of the SAME source as the Supabase
 * generate-outfits Edge Function. Access requires a short-lived HMAC signed
 * by the Supabase proxy, in addition to the original user's JWT.
 *
 * Nothing is metered here before the ORIGINAL handler runs; rate limiting and
 * consume_edge_quota remain in the original handler exactly once.
 */
import {
  verifyStylistTransport,
  STYLIST_TRANSPORT_MAX_BODY_BYTES,
} from '../../stylist-transport/signature.ts';

type EdgeHandler = (request: Request) => Promise<Response>;
type StylistGlobal = typeof globalThis & {
  Deno?: { env: { get: (name: string) => string | undefined } };
  __AIW_STYLIST_HANDLER__?: EdgeHandler;
};

let handlerPromise: Promise<EdgeHandler> | undefined;

function originalHandler(): Promise<EdgeHandler> {
  if (!handlerPromise) {
    handlerPromise = (async () => {
      const globals = globalThis as StylistGlobal;
      if (!globals.Deno) {
        globals.Deno = { env: { get: (name: string) => process.env[name] } };
      }
      // Generated during npm run build from the pinned production Edge source.
      // This module registers the original Deno.serve request handler.
      await import('../dist/stylist-bundle.mjs');
      const handler = globals.__AIW_STYLIST_HANDLER__;
      if (!handler) throw new Error('stylist_handler_unavailable');
      return handler;
    })().catch(error => {
      handlerPromise = undefined;
      throw error;
    });
  }
  return handlerPromise;
}

function errorResponse(code: string, status: number): Response {
  return Response.json({ success: false, outfits: [], code, error: code }, { status });
}

export async function POST(request: Request): Promise<Response> {
  const secret = process.env.STYLIST_NODE_SHARED_SECRET;
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
    secret, body, authorization,
    timestamp: request.headers.get('x-aiw-timestamp'),
    nonce: request.headers.get('x-aiw-nonce'),
    signature: request.headers.get('x-aiw-signature'),
  });
  if (!authorized) return errorResponse('unauthorized', 401);

  try {
    const originalRequest = new Request('https://stylist.internal/functions/v1/generate-outfits', {
      method: 'POST',
      headers: {
        authorization,
        apikey: request.headers.get('apikey') || '',
        'content-type': 'application/json',
        'x-client-info': request.headers.get('x-client-info') || 'aiwardrobe-stylist-proxy',
      },
      body,
    });
    const handler = await originalHandler();
    // Original handler performs: auth -> quota -> catalog -> full quality
    // checks -> bounded planning -> NVIDIA ranking -> response.
    return await handler(originalRequest);
  } catch {
    // Never leak backend internals, JWTs, product data or service credentials.
    return errorResponse('stylist_compute_unavailable', 503);
  }
}

export function GET(): Response {
  return errorResponse('method_not_allowed', 405);
}
