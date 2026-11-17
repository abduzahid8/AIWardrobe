/**
 * Single-use signed invoker for the isolated Preview NVIDIA vision test.
 *
 * Runs EXACTLY ONE POST with a fresh HMAC signature and exits. No retries,
 * no loops, no fallback URLs. Refuses to run unless every explicit approval
 * is present in the environment AND on the command line:
 *   PREVIEW_URL                      https://<preview-deployment>/api/stylist6-vision-test
 *   STYLIST_NODE_SHARED_SECRET       HMAC transport secret (never printed)
 *   NVIDIA_SPEND_APPROVED=1          explicit spend approval for ONE call
 *   NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1  explicit image-rights approval
 *   --confirm-spend                  typed confirmation flag
 *   --ref <photo.jpg>                verified reference bytes (SHA recorded)
 *   --products <u1,u2,...>           1-4 real product photo HTTPS URLs
 * Missing anything -> exit 2 with zero network traffic.
 *
 * Prints ONLY: HTTP status, providerCalls, wallMs, the redacted verdict
 * (scores + evidence), and the reference SHA. Secrets, request bodies and
 * image bytes are never printed. The receipt is written to --receipt
 * (mode 600) when given.
 *
 * Usage (owner shell holding the secrets; never paste secrets in chat):
 *   STYLIST_NODE_SHARED_SECRET=... NVIDIA_SPEND_APPROVED=1 \
 *   NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1 \
 *   node scripts/stylist6/nvidia_preview_invoke.mjs \
 *     --url https://<preview>/api/stylist6-vision-test \
 *     --ref ./image-teaste/photo_5_2026-11-10_09-23-10.jpg \
 *     --products https://.../top.jpg,https://.../bottom.jpg \
 *     --style old_money --occasion Work \
 *     --receipt /tmp/nvidia-preview-receipt.json --confirm-spend
 */

import fs from 'node:fs';
import { createHash } from 'node:crypto';

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => {
    if (!a.startsWith('--')) return [];
    const eq = a.indexOf('=');
    if (eq > 0) return [[a.slice(2, eq), a.slice(eq + 1)]];
    const next = all[i + 1];
    return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
  }),
);

function refuse(reason, extra = {}) {
  console.error(JSON.stringify({ refused: true, reason, ...extra, networkCalls: 0 }));
  process.exit(2);
}

const secret = process.env.STYLIST_NODE_SHARED_SECRET || '';
const url = args.url || process.env.PREVIEW_URL || '';
if (secret.length < 32) refuse('missing STYLIST_NODE_SHARED_SECRET (>=32 chars)');
if (!/^https:\/\//.test(url) || !url.includes('/api/stylist6-vision-test')) {
  refuse('missing --url https://<preview>/api/stylist6-vision-test');
}
if (process.env.NVIDIA_SPEND_APPROVED !== '1' || args['confirm-spend'] !== 'true') {
  refuse('need NVIDIA_SPEND_APPROVED=1 + --confirm-spend');
}
if (process.env.NVIDIA_REFERENCE_RIGHTS_CONFIRMED !== '1') {
  refuse('need NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1');
}
const refFile = args.ref || '';
const products = String(args.products || '').split(',').map((s) => s.trim()).filter(Boolean);
if (!refFile || !fs.existsSync(refFile)) refuse('missing --ref verified reference JPEG');
if (!products.length || products.length > 4) refuse('need 1-4 --products HTTPS URLs');
const refBytes = fs.readFileSync(refFile);
if (refBytes[0] !== 0xff || refBytes[1] !== 0xd8 || refBytes.length < 1024) {
  refuse('reference is not a readable JPEG');
}
const refSha = createHash('sha256').update(refBytes).digest('hex');

const { signStylistTransport } = await import('../../backend/stylist-node/transport/signature.mjs');

const body = JSON.stringify({
  protocol: 'aiw-stylist6-vision-test-v1',
  confirmSpend: true,
  rightsConfirmed: true,
  maxCalls: 1,
  candidate: {
    id: 'preview-oneshot-1',
    briefId: 'preview-oneshot-1',
    concept: 'preview isolation check',
    referenceId: 'rt_preview',
    reference: { base64: refBytes.toString('base64'), mimeType: 'image/jpeg', sha256: refSha },
    products: products.map((imageUrl, i) => ({ id: `preview-product-${i + 1}`, imageUrl })),
  },
  request: { style: args.style || 'old_money', occasion: args.occasion || 'Work' },
});
const authorization = 'Bearer preview-single-use-caller';
const headers = await signStylistTransport({ secret, body, authorization });

const started = Date.now();
const response = await globalThis.fetch(url, {
  method: 'POST',
  headers: { ...headers, authorization, 'content-type': 'application/json' },
  body,
  signal: AbortSignal.timeout(75000),
});
const wallMs = Date.now() - started;
let payload = null;
try {
  payload = await response.json();
} catch {
  refuse('preview did not return JSON', { httpStatus: response.status, wallMs });
}

// Redacted output only: scores + evidence + identity, never secrets/bytes.
const verdict = payload && payload.ok ? payload.verdict : null;
const out = {
  httpStatus: response.status,
  ok: !!(payload && payload.ok),
  model: payload ? payload.model : null,
  providerCalls: payload ? payload.providerCalls : null,
  wallMs,
  referenceSha256: refSha,
  products: products.length,
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
  code: payload && !payload.ok ? payload.code : null,
};
console.log(JSON.stringify(out, null, 1));
if (args.receipt) fs.writeFileSync(args.receipt, JSON.stringify(out, null, 1) + '\n', { mode: 0o600 });
if (!out.ok) process.exit(1);
