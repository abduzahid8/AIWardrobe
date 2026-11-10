/**
 * Isolated single-request NVIDIA vision test for Stylist 6 (Preview only).
 *
 * POST-only, HMAC-signed (same transport as generate-outfits/transport-health).
 * Runs the TESTED Stylist 6 hosted critic (`provider:'nvidia'`) exactly once
 * against caller-supplied reference bytes + real product photo URLs, and
 * returns a redacted verdict. No quotas, no training, no persistence.
 *
 * Server-side only: reads NVIDIA_API_KEY from process env (never accepted
 * from the caller, never logged, never echoed). The caller must embed
 * explicit spend + image-rights approval INSIDE the signed body
 * ({confirmSpend:true, rightsConfirmed:true}); unsigned or unapproved calls
 * are rejected before any provider traffic. Exactly one provider POST per
 * admitted invocation; no retry loops anywhere (callers must not retry).
 *
 * Stateless-serverless caveat: exactly-once holds per admitted invocation.
 * Replay of a signed body within the 45s skew window re-executes once more
 * (nonces are not centrally stored); the runbook therefore mandates a single
 * curl with a fresh nonce and no retries.
 */

import { verifyStylistTransport } from '../transport/signature.mjs';
import { createHostedVisualCritic, HOSTED_VISUAL_CRITIC_VERSION } from '../../stylist6/visual_critic_hosted.mjs';

const PROTOCOL = 'aiw-stylist6-vision-test-v1';
const NVIDIA_ENDPOINT = 'https://integrate.api.nvidia.com/v1/chat/completions';
const MAX_BODY_BYTES = 4 * 1024 * 1024;
const MAX_REFERENCE_BYTES = 3 * 1024 * 1024;
const MAX_PRODUCTS = 4;

const error = (code: string, status: number) =>
  Response.json({ ok: false, code }, { status, headers: { 'Cache-Control': 'no-store' } });

function isHttps(value: unknown): value is string {
  try {
    const url = new URL(String(value));
    return url.protocol === 'https:' && !!url.hostname;
  } catch {
    return false;
  }
}

export default {
  async fetch(request: Request): Promise<Response> {
    if (request.method !== 'POST') return error('method_not_allowed', 405);
    if (Number(request.headers.get('content-length') || 0) > MAX_BODY_BYTES) {
      return error('invalid_request', 413);
    }
    const secret = process.env.STYLIST_NODE_SHARED_SECRET || '';
    if (secret.length < 32) return error('compute_unconfigured', 503);
    const body = await request.text();
    if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) return error('invalid_request', 413);

    const authorized = await verifyStylistTransport({
      secret, body,
      authorization: request.headers.get('authorization') || '',
      timestamp: request.headers.get('x-aiw-timestamp'),
      nonce: request.headers.get('x-aiw-nonce'),
      signature: request.headers.get('x-aiw-signature'),
    });
    if (!authorized) return error('unauthorized', 401);

    let payload: any;
    try {
      payload = JSON.parse(body);
    } catch {
      return error('invalid_request', 400);
    }
    if (payload?.protocol !== PROTOCOL) return error('invalid_request', 400);
    // Explicit spend + image-rights approval must be inside the SIGNED body.
    if (payload?.confirmSpend !== true || payload?.rightsConfirmed !== true || payload?.maxCalls !== 1) {
      return error('approval_required', 403);
    }
    const candidate = payload?.candidate;
    const products = Array.isArray(candidate?.products) ? candidate.products : [];
    const reference = candidate?.reference;
    if (!candidate || typeof candidate.id !== 'string' || !candidate.id ||
        typeof candidate.referenceId !== 'string' || !candidate.referenceId ||
        !reference || typeof reference.base64 !== 'string' || !reference.base64 ||
        products.length < 1 || products.length > MAX_PRODUCTS ||
        !products.every((p: any) => p && typeof p.id === 'string' && isHttps(p.imageUrl))) {
      return error('invalid_request', 400);
    }
    let refBytes: Buffer;
    try {
      refBytes = Buffer.from(reference.base64, 'base64');
    } catch {
      return error('invalid_request', 400);
    }
    if (refBytes.length < 1024 || refBytes.length > MAX_REFERENCE_BYTES ||
        refBytes[0] !== 0xff || refBytes[1] !== 0xd8) {
      return error('invalid_request', 400);
    }
    const apiKey = process.env.NVIDIA_API_KEY || '';
    if (apiKey.length < 20) return error('vision_unconfigured', 503);

    // Integrity: the submitted SHA must match the submitted bytes.
    const { createHash } = await import('node:crypto');
    const actualSha = createHash('sha256').update(refBytes).digest('hex');
    if (typeof reference.sha256 !== 'string' || reference.sha256.toLowerCase() !== actualSha) {
      return error('invalid_request', 400);
    }

    let providerCalls = 0;
    const started = Date.now();
    try {
      const critic = createHostedVisualCritic({
        apiKey,
        provider: 'nvidia',
        timeoutMs: 55000,
        resolveReference: async (referenceId: string) => {
          if (referenceId !== candidate.referenceId) return null;
          return { base64: reference.base64, mimeType: 'image/jpeg', sha256: actualSha };
        },
        fetchImpl: (async (url: string, init?: any) => {
          if (url === NVIDIA_ENDPOINT) {
            providerCalls += 1;
            if (providerCalls > 1) throw new Error('provider_call_budget_exhausted');
            return globalThis.fetch(url, init);
          }
          if (!isHttps(url)) throw new Error('product_photo_blocked');
          return globalThis.fetch(url, init);
        }) as typeof fetch,
      });
      const verdicts = await critic([{
        id: candidate.id,
        briefId: candidate.briefId || candidate.id,
        concept: candidate.concept || candidate.id,
        referenceId: candidate.referenceId,
        products: products.map((p: any) => ({ id: p.id, imageUrl: p.imageUrl, name: p.name || p.id })),
      }], { style: payload?.request?.style || 'unspecified', occasion: payload?.request?.occasion || 'unspecified' });
      const wallMs = Date.now() - started;
      const verdict = verdicts[0] || null;
      return Response.json({
        ok: true,
        protocol: PROTOCOL,
        model: verdict ? verdict.criticVersion : `${HOSTED_VISUAL_CRITIC_VERSION}/nvidia/unjudged`,
        providerCalls,
        wallMs,
        verdict: verdict ? {
          candidateId: verdict.candidateId,
          visualCohesion: verdict.visualCohesion,
          styleFit: verdict.styleFit,
          occasionFit: verdict.occasionFit,
          formalityFit: verdict.formalityFit,
          confidence: verdict.confidence,
          evidence: verdict.evidence,
          ...(verdict.reject === true ? { reject: true } : {}),
        } : null,
        abstained: !verdict,
        costEstimateUsd: null,
        note: 'single isolated test; no quota, no training, no persistence',
      }, { status: 200, headers: { 'Cache-Control': 'no-store' } });
    } catch (err) {
      return Response.json({
        ok: false, code: 'vision_failed', providerCalls,
        wallMs: Date.now() - started,
      }, { status: 502, headers: { 'Cache-Control': 'no-store' } });
    }
  },
};
