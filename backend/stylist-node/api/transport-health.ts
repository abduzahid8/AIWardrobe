/**
 * Transport-only preflight. This never calls the paid Stylist handler, reads
 * garments, or consumes user quota. A valid HMAC is still required.
 */
import { verifyStylistTransport } from '../transport/signature.mjs';

const error = (code: string, status: number) =>
  Response.json({ ok: false, code }, { status, headers: { 'Cache-Control': 'no-store' } });

export default {
  async fetch(request: Request): Promise<Response> {
    if (request.method !== 'POST') return error('method_not_allowed', 405);
    if (Number(request.headers.get('content-length') || 0) > 512) return error('invalid_request', 413);
    const secret = process.env.STYLIST_NODE_SHARED_SECRET || '';
    if (secret.length < 32) return error('compute_unconfigured', 503);
    const body = await request.text();
    if (new TextEncoder().encode(body).byteLength > 512 ||
        body !== '{"probe":"aiw-stylist-transport-v1"}') return error('invalid_request', 400);
    const verified = await verifyStylistTransport({
      secret, body,
      authorization: request.headers.get('authorization') || '',
      timestamp: request.headers.get('x-aiw-timestamp'),
      nonce: request.headers.get('x-aiw-nonce'),
      signature: request.headers.get('x-aiw-signature'),
    });
    if (!verified) return error('unauthorized', 401);
    return Response.json({ ok: true, transport: 'signed' }, {
      status: 200, headers: { 'Cache-Control': 'no-store' }
    });
  },
};
