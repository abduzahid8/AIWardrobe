/**
 * Isolated Stylist 6 + NVIDIA Nemotron 3.5 Super VL pilot (NO production).
 *
 * Offline, single-process, no routes, no quota, no training:
 *   1. loads the committed snapshot catalog via the wearing-role adapter,
 *   2. builds reference-first briefs for a FIXED reference set
 *      (--refs rt_005,rt_008 — no relevance ranking),
 *   3. runs retrieval (shortlistRetrievalDeck),
 *   4. judges the shortlist with the NVIDIA hosted critic
 *      (reference photo + real product photos, structured verdicts),
 *   5. selects the final deck with proposeVisualFirstDeck semantics and
 *      writes a private receipt.
 *
 * Gates (same as vision_oneshot): NVIDIA_API_KEY (>=20 chars) +
 * NVIDIA_SPEND_APPROVED=1 + NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1 +
 * --confirm-spend. Missing any one -> exit 2 with zero provider traffic.
 * Provider calls are bounded by --max-visual (default 4, hard cap 8).
 * Reference bytes come from the verified ./image-teaste folder; the receipt
 * records each submitted SHA-256 for manifest confirmation.
 *
 * Usage:
 *   NVIDIA_API_KEY=... NVIDIA_SPEND_APPROVED=1 NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1 \
 *   node scripts/stylist6/stylist6_nvidia_pilot.mjs --refs rt_005,rt_008 \
 *     --out /tmp/s6-nvidia-iso --confirm-spend
 *
 * NEVER run in CI. NEVER commit receipts.
 */

import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const root = path.resolve(import.meta.dirname, '../..');
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
const okGate = key.length >= 20
  && process.env.NVIDIA_SPEND_APPROVED === '1'
  && process.env.NVIDIA_REFERENCE_RIGHTS_CONFIRMED === '1'
  && args['confirm-spend'] === 'true';
if (!okGate) {
  console.error(JSON.stringify({
    refused: true,
    reason: 'need NVIDIA_API_KEY + NVIDIA_SPEND_APPROVED=1 + NVIDIA_REFERENCE_RIGHTS_CONFIRMED=1 + --confirm-spend; zero provider calls made',
    hadKey: key.length >= 20,
    hadApproval: process.env.NVIDIA_SPEND_APPROVED === '1',
    rightsConfirmed: process.env.NVIDIA_REFERENCE_RIGHTS_CONFIRMED === '1',
    hadFlag: args['confirm-spend'] === 'true',
  }));
  process.exit(2);
}

const refIds = (args.refs || 'rt_005,rt_008').split(',').map((s) => s.trim()).filter(Boolean);
if (!refIds.length || refIds.length > 4) throw new Error('refs_must_be_1_to_4_reference_ids');
const maxVisual = Math.min(8, Math.max(1, Number(args['max-visual'] || 4)));
const outDir = args.out || '/tmp/s6-nvidia-iso';
fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });

const index = JSON.parse(fs.readFileSync(
  path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json'), 'utf8'));
const { adaptCatalog } = await import('../../backend/stylist6/catalog_adapter.mjs');
const { briefForReferenceId } = await import('../../backend/stylist6/lookbook_briefs.mjs');
const { shortlistRetrievalDeck, proposeVisualFirstDeck } = await import('../../backend/stylist6/prototype.mjs');
const { createHostedVisualCritic, HOSTED_VISUAL_CRITIC_VERSION } = await import('../../backend/stylist6/visual_critic_hosted.mjs');
const { resolveImages } = await import('./image_access.mjs');

const rows = [];
for (const file of fs.readdirSync(path.join(root, 'data'))) {
  if (!file.startsWith('shop_catalog_') || !file.endsWith('.json')) continue;
  rows.push(...JSON.parse(fs.readFileSync(path.join(root, 'data', file), 'utf8')));
}
const adapted = adaptCatalog(rows);
const briefs = refIds.map((id) => briefForReferenceId(index, id));

const request = { style: 'old_money', occasion: 'Work', sourceMode: 'catalog' };
const hardGate = async (items) =>
  new Set(items.map((i) => i.id)).size === items.length ? { ok: true } : { ok: false };
const priceToBudgetCurrency = (item, currency) => (item.currency === currency ? item.price : NaN);

// Verified reference bytes: ./image-teaste photo_N ↔ rt_NNN, SHA recorded.
const refBytesById = new Map();
for (const id of refIds) {
  const num = String(Number(id.replace('rt_', '')));
  const file = fs.readdirSync(path.join(root, 'image-teaste')).find((f) => f.startsWith(`photo_${num}_`));
  if (!file) throw new Error('reference_file_missing_for_' + id);
  const bytes = fs.readFileSync(path.join(root, 'image-teaste', file));
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) throw new Error('reference_not_jpeg:' + id);
  refBytesById.set(id, { bytes, sha256: createHash('sha256').update(bytes).digest('hex'), file });
}

const imgDir = fs.mkdtempSync(path.join(outDir, 'img-'));
const events = [];
let providerCalls = 0;

// Retrieval first (no provider traffic): the critic judges a slice of this.
const retrieval = await shortlistRetrievalDeck({
  request, catalog: adapted.items,
  creativeDirector: async () => briefs,
  hardGate, priceToBudgetCurrency,
  options: { briefLimit: 6, perRole: 12, maxVisual: 48, perBrief: 60 },
});
const shortlist = retrieval.shortlist.slice(0, maxVisual);

// Pre-resolve product photos through the image_access allowlist policy.
// Only these bytes may reach the model; blocked hosts abstain inside the
// critic (recorded in events) WITHOUT spending provider calls.
const urlToB64 = new Map();
{
  const seen = new Map();
  const urls = [];
  for (const c of shortlist) {
    for (const p of c.items) {
      if (!seen.has(p.imageUrl)) {
        seen.set(p.imageUrl, seen.size);
        urls.push({ id: `u${seen.get(p.imageUrl)}`, imageUrl: p.imageUrl });
      }
    }
  }
  const byId = new Map(urls.map((u) => [u.id, u.imageUrl]));
  const resolved = await resolveImages({ urls, outDir: imgDir, auditPath: path.join(outDir, 'image_audit.jsonl') });
  for (const r of resolved) {
    if ((r.status === 'fetched' || r.status === 'cached') && r.file) {
      urlToB64.set(byId.get(r.id), fs.readFileSync(r.file).toString('base64'));
    }
  }
}
const judgedCritic = createHostedVisualCritic({
  apiKey: key,
  provider: 'nvidia',
  timeoutMs: 90000,
  onEvent: (e) => events.push(e),
  resolveReference: async (referenceId) => {
    const r = refBytesById.get(referenceId);
    return r ? { base64: r.bytes.toString('base64'), mimeType: 'image/jpeg', sha256: r.sha256 } : null;
  },
  fetchImpl: async (url, init) => {
    if (url === 'https://integrate.api.nvidia.com/v1/chat/completions') {
      if (providerCalls >= maxVisual) throw new Error('provider_call_budget_exhausted');
      providerCalls += 1;
      return globalThis.fetch(url, init);
    }
    const b64 = urlToB64.get(url);
    if (!b64) return { ok: false, status: 403, arrayBuffer: async () => Buffer.alloc(0) };
    const buf = Buffer.from(b64, 'base64');
    return { ok: true, status: 200, arrayBuffer: async () => buf };
  },
});
const started = Date.now();
const judged = await judgedCritic(shortlist.map((c) => ({
  id: c.id, briefId: c.briefId, concept: c.brief.name,
  referenceId: c.brief.referenceId || null,
  products: c.items.map((i) => ({ id: i.id, imageUrl: i.imageUrl, name: i.name })),
})), request);
const wallMs = Date.now() - started;

const receipt = {
  tool: 'stylist6-nvidia-isolated-v1',
  criticVersion: HOSTED_VISUAL_CRITIC_VERSION,
  refs: refIds.map((id) => ({ id, sha256: refBytesById.get(id).sha256, file: refBytesById.get(id).file })),
  retrieval: {
    candidateCount: retrieval.diagnostics.candidateCount,
    shortlisted: retrieval.shortlist.length,
    judgedShortlist: shortlist.length,
  },
  providerCalls,
  maxVisual,
  wallMs,
  judgedCount: judged.length,
  abstentions: events.filter((e) => e.outcome === 'abstain'),
  verdicts: judged,
  policyResolvedBytes: urlToB64.size,
};
fs.writeFileSync(path.join(outDir, 'nvidia_isolated_receipt.json'), JSON.stringify(receipt, null, 1) + '\n', { mode: 0o600 });
console.log(JSON.stringify({
  refs: refIds, providerCalls, judged: judged.length,
  abstentions: receipt.abstentions.length,
  receipt: path.join(outDir, 'nvidia_isolated_receipt.json'),
}));
