/**
 * Single authorized hosted-vision test for the Stylist 6 pilot.
 *
 * Makes EXACTLY ONE provider call (one candidate = reference photo + up to 4
 * product photos) and writes a private receipt. Refuses to run unless:
 *   1. NVIDIA_API_KEY is set (server-side, never committed or printed), AND
 *   2. NVIDIA_SPEND_APPROVED=1 and NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1, AND
 *   3. --confirm-spend is passed on the command line.
 * Missing any one of the three -> exit 2 with no network traffic.
 *
 * NVIDIA pricing/remaining hosted trial credits vary. This script makes AT
 * MOST ONE provider call and never retries it. It does NOT claim an enforced
 * USD billing ceiling; obtain owner approval for the actual account rates.
 *
 * Reference bytes must be the independently verified canonical images:
 * pass --ref-dir <image-teaste folder>. The receipt records the SHA-256
 * of exact bytes submitted. Use verify_image_teaste_folder.py first.
 *
 * Usage:
 *   NVIDIA_API_KEY=... NVIDIA_SPEND_APPROVED=1 node scripts/stylist6/vision_oneshot.mjs \
 *     --scenario S6-01 --ref-dir ./image-teaste \
 *     --receipt /private/vision-oneshot-receipt.json --confirm-spend
 *
 * NEVER run in CI. NEVER commit the receipt (may embed key-adjacent data).
 */

import fs from 'node:fs';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => {
    if (!a.startsWith('--')) return [];
    const eq = a.indexOf('=');
    if (eq > 0) return [[a.slice(2, eq), a.slice(eq + 1)]];
    const next = all[i + 1];
    return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
  }),
);

const key = process.env.NVIDIA_API_KEY || '';
const approved = process.env.NVIDIA_SPEND_APPROVED === '1';
const rightsConfirmed = process.env.NVIDIA_REFERENCE_RIGHTS_CONFIRMED === '1';
if (key.length < 20 || !approved || !rightsConfirmed || args['confirm-spend'] !== 'true') {
  console.error(JSON.stringify({
    refused: true,
    reason: 'need NVIDIA_API_KEY (>=20 chars) + NVIDIA_SPEND_APPROVED=1 + NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1 + --confirm-spend; zero provider calls made',
    hadKey: key.length >= 20,
    hadApproval: approved,
    rightsConfirmed,
    hadFlag: args['confirm-spend'] === 'true',
  }));
  process.exit(2);
}

const { createHostedVisualCritic, HOSTED_VISUAL_CRITIC_VERSION } = await import('../../backend/stylist6/visual_critic_hosted.mjs');
const { resolveImages } = await import('./image_access.mjs');
const { createHash } = await import('node:crypto');

const pilot = JSON.parse(fs.readFileSync(args.pilot || '/tmp/s6-pilot-20261110/pilot-private.json', 'utf8'));
const run = pilot.runs.find((r) => r.id === (args.scenario || 'S6-01'));
if (!run) throw new Error('scenario_not_in_private_pilot');
const look = (run.arms['reference-first']?.looks || [])[Number(args.look || 0)];
if (!look) throw new Error('reference_look_missing');
const refId = look.referenceId;
const refNum = String(Number(refId.replace('rt_', '')));
const refFile = (fs.readdirSync(args['ref-dir'] || '') || []).find((f) => f.startsWith(`photo_${refNum}_`));
if (!refFile) throw new Error('reference_file_missing_for_' + refId);
const refBytes = fs.readFileSync(path.join(args['ref-dir'], refFile));
if (refBytes[0] !== 0xff || refBytes[1] !== 0xd8) throw new Error('reference_not_jpeg');
const refSha = createHash('sha256').update(refBytes).digest('hex');
if (refBytes.length < 4096) throw new Error('reference_photo_too_small');

const imgDir = fs.mkdtempSync('/tmp/vision-oneshot-img-');
const resolved = await resolveImages({
  urls: look.items.map((i) => ({ id: i.id, imageUrl: i.imageUrl })),
  outDir: imgDir, auditPath: null,
});
const blocked = resolved.filter((r) => r.status !== 'fetched' && r.status !== 'cached');
if (blocked.length) throw new Error('product_photo_blocked:' + blocked.map((r) => r.id).join(','));
const photos = new Map(resolved.map((r) => [r.id, fs.readFileSync(r.file).toString('base64')]));

const events = [];
let providerCalls = 0;
const critic = createHostedVisualCritic({
  apiKey: key,
  provider: 'nvidia',
  onEvent: (e) => events.push(e),
  resolveReference: async () => ({ base64: refBytes.toString('base64'), mimeType: 'image/jpeg', sha256: refSha }),
  // Route product bytes through the already policy-resolved cache only.
  fetchImpl: async (url, init) => {
    if (url === 'https://integrate.api.nvidia.com/v1/chat/completions') {
      if (providerCalls >= 1) throw new Error('provider_call_budget_exhausted');
      providerCalls += 1;
      return globalThis.fetch(url, init);
    }
    const id = look.items.find((i) => i.imageUrl === url)?.id;
    const b64 = id ? photos.get(id) : null;
    if (!b64) return { ok: false, status: 403, arrayBuffer: async () => Buffer.alloc(0) };
    const buf = Buffer.from(b64, 'base64');
    return { ok: true, status: 200, arrayBuffer: async () => buf };
  },
});

const started = Date.now();
const out = await critic([{
  id: 'oneshot-1', briefId: look.briefId, concept: look.concept, referenceId: refId,
  products: look.items.map((i) => ({ id: i.id, imageUrl: i.imageUrl, name: i.name })),
}], { style: run.id });
const wallMs = Date.now() - started;

const receipt = {
  version: HOSTED_VISUAL_CRITIC_VERSION,
  scenario: run.id,
  referenceId: refId,
  referenceFile: refFile,
  referenceSha256: refSha,
  candidateItems: look.items.map((i) => i.id),
  providerCalls,
  wallMs,
  verdict: out[0] || null,
  events,
  providerCostUsd: null,
  note: 'NVIDIA Nemotron 3.5 VL: at most one authorized call, no retries; provider billing rate not verified',
};
if (args.receipt) fs.writeFileSync(args.receipt, JSON.stringify(receipt, null, 1) + '\n', { mode: 0o600 });
console.log(JSON.stringify({ ...receipt, verdict: out[0] ? 'judged' : 'abstained' }, null, 1));
