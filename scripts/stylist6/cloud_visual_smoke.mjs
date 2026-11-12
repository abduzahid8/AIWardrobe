/**
 * LOCAL bounded real-inference smoke for the Stylist 6 cloud staging critic.
 *
 * Runs `handleStylist6CloudGenerate` IN-PROCESS (from the built
 * backend/stylist-node/dist/stylist6-bundle.mjs artifact) against the frozen
 * catalog snapshot `.private/catalog-uz-20261113.json`, with a mocked
 * Supabase layer ONLY for inventory/quota/idempotency (the same fake-io
 * pattern as backend/stylist6/cloud_handler.test.mjs). Everything else —
 * merchant image downloads, the transmission gate, the hosted NVIDIA critic,
 * verdict validation and deck selection — is the real production code path.
 *
 * Bounds (hard-coded, enforced):
 * - at most 2 scenarios, each run twice: enableVisual=false (deterministic
 *   baseline) then enableVisual=true (visual review);
 * - at most 8 NVIDIA provider calls in the whole process; the script refuses
 *   to dispatch further calls beyond the budget;
 * - merchant image bytes are cached on disk under `.private/` (mode 600
 *   files, mode 700 dir) so repeated runs fetch each URL at most once.
 *
 * The NVIDIA API key is read from `.env` and NEVER printed. Provider
 * responses are summarized (status, usage token counts, finish reason) —
 * never dumped. Every failure is reported honestly: a quota/region refusal
 * from NVIDIA fails the visual runs with their real codes while everything
 * else still completes.
 *
 * Usage: node scripts/stylist6/cloud_visual_smoke.mjs [--out .private/cloud-visual-smoke.json]
 */
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { signStylistTransport } from '../../backend/stylist-node/transport/signature.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PRIVATE_DIR = path.join(repoRoot, '.private');
const IMAGE_CACHE_DIR = path.join(PRIVATE_DIR, 'cloud-visual-smoke-images');
const CATALOG_PATH = path.join(PRIVATE_DIR, 'catalog-uz-20261113.json');

const args = Object.fromEntries(process.argv.slice(2).flatMap((a, i, all) => {
  if (!a.startsWith('--')) return [];
  const eq = a.indexOf('=');
  if (eq > 0) return [[a.slice(2), a.slice(eq + 1)]];
  const next = all[i + 1];
  return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
}));

const PROVIDER_CALL_BUDGET = 8;
const MAX_SCENARIOS = 2;
const NVIDIA_HOST = 'integrate.api.nvidia.com';
const IMAGE_FETCH_TIMEOUT_MS = 8000;

// ---- Configuration ---------------------------------------------------------

const dotenv = await import('dotenv');
const envFile = dotenv.parse(fs.readFileSync(path.join(repoRoot, '.env')));
const NVIDIA_API_KEY = envFile.NVIDIA_API_KEY || process.env.NVIDIA_API_KEY || '';
if (!NVIDIA_API_KEY.startsWith('nvapi-')) {
  console.error('NVIDIA_API_KEY missing or malformed in .env (never printed). The deterministic halves would still run; aborting because the visual halves would be dishonest.');
  process.exit(2);
}

const bundlePath = path.join(repoRoot, 'backend/stylist-node/dist/stylist6-bundle.mjs');
if (!fs.existsSync(bundlePath)) {
  console.error(`bundle missing at ${bundlePath} — run: (cd backend/stylist-node && npm run build)`);
  process.exit(2);
}
const bundleSource = fs.readFileSync(bundlePath, 'utf8');
const bundleSha256 = createHash('sha256').update(bundleSource).digest('hex');
const { handleStylist6CloudGenerate } = await import(bundlePath);

if (!fs.existsSync(CATALOG_PATH)) {
  console.error(`frozen catalog missing at ${CATALOG_PATH}`);
  process.exit(2);
}
const catalogRows = JSON.parse(fs.readFileSync(CATALOG_PATH, 'utf8'));

// ---- Mocked Supabase layer (inventory/quota only) --------------------------

const JWT = 'smoke-jwt-token';
const USER = 'smoke-user';
const claims = new Map();

const makeIo = (shopRows) => ({
  async verifyUser(jwt) {
    if (jwt !== JWT) throw new Error('stylist6_unauthorized');
    return USER;
  },
  async checkRate() { /* local smoke: no rate bucket */ },
  async debitQuota() { /* local smoke: no quota debit */ },
  async claimRequest(userId, requestId, fingerprint) {
    if (claims.has(requestId)) {
      const prior = claims.get(requestId);
      return { duplicate: true, conflict: prior.fingerprint !== fingerprint, prior, durable: true };
    }
    claims.set(requestId, { status: null, code: null, fingerprint });
    return { duplicate: false, prior: null, durable: true };
  },
  async finishRequest(userId, requestId, status, code) {
    claims.set(requestId, { ...claims.get(requestId), status, code });
    return true;
  },
  async fetchShopRows() { return shopRows; },
  async fetchWardrobeRows() { return []; },
  async fetchVisualProfiles() { return []; },
});

// ---- Bounded transport: disk-cached images + counted provider calls --------

fs.mkdirSync(IMAGE_CACHE_DIR, { recursive: true, mode: 0o700 });

let providerCallsTotal = 0;
const transportStats = { imageCacheHits: 0, imageFetches: 0, imageFetchFailures: 0, providerCalls: 0 };

async function smokeFetch(url, options = {}) {
  const parsed = new URL(String(url));
  if (parsed.hostname === NVIDIA_HOST) {
    if (providerCallsTotal >= PROVIDER_CALL_BUDGET) {
      throw new Error(`provider_call_budget_exhausted (${PROVIDER_CALL_BUDGET})`);
    }
    providerCallsTotal += 1;
    transportStats.providerCalls += 1;
    return fetch(url, options);
  }
  // Merchant (or first-party) image: disk-cached under .private, mode 600.
  const digest = createHash('sha256').update(String(url)).digest('hex');
  const file = path.join(IMAGE_CACHE_DIR, `${digest}.jpg`);
  if (fs.existsSync(file)) {
    transportStats.imageCacheHits += 1;
    return new Response(fs.readFileSync(file), { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
  }
  transportStats.imageFetches += 1;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), IMAGE_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal, redirect: 'follow' });
    if (!response.ok) {
      transportStats.imageFetchFailures += 1;
      void response.body?.cancel().catch(() => {});
      return response;
    }
    const bytes = Buffer.from(await response.arrayBuffer());
    if (bytes[0] === 0xff && bytes[1] === 0xd8) {
      fs.writeFileSync(file, bytes, { mode: 0o600 });
    }
    return new Response(bytes, { status: response.status, headers: { 'Content-Type': response.headers.get('content-type') || 'image/jpeg' } });
  } catch (error) {
    transportStats.imageFetchFailures += 1;
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

// ---- Scenario runner -------------------------------------------------------

const SECRET = `smoke-shared-secret-${'0'.repeat(32)}-local-only`;

const slimReceipt = (receipt) => ({
  model: receipt.model,
  requested: receipt.requested,
  serverDisabled: receipt.serverDisabled,
  providerCalls: receipt.providerCalls,
  imagesSent: receipt.imagesSent,
  cacheHits: receipt.cacheHits,
  retriedCalls: receipt.retriedCalls,
  validVerdicts: receipt.validVerdicts,
  abstentions: receipt.abstentions,
  totalTokens: receipt.totalTokens,
  events: (receipt.events || []).map((event) => ({
    outcome: event.outcome, dispatched: event.dispatched, reason: event.reason,
    httpStatus: event.httpStatus, latencyMs: event.latencyMs,
    ...(event.images != null ? { images: event.images } : {}),
    ...(event.cacheHit ? { cacheHit: true } : {}),
    ...(event.retried ? { retried: true } : {}),
    ...(event.usage ? { usage: event.usage } : {}),
  })),
});

const deckShape = (body) => ({
  status: body.status,
  visualStatus: body.visualStatus,
  visualVerdict: body.diagnostics?.visualVerdict ?? null,
  deckIds: body.outfits.map((look) => look.items.map((item) => item.id).sort().join('+')),
  lookIds: body.outfits.map((look) => look.id),
  scores: body.outfits.map((look) => Number((look.concept?.score ?? 0).toFixed(4))),
});

async function runScenario(name, request, { visual }) {
  const payload = { request: { ...request, enableVisual: visual, generationRequestId: `smoke-${name}-${visual ? 'visual' : 'baseline'}` } };
  const bodyText = JSON.stringify(payload);
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body: bodyText, authorization });
  const started = Date.now();
  const result = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, 'x-aiw-timestamp': signature['x-aiw-timestamp'], 'x-aiw-nonce': signature['x-aiw-nonce'], 'x-aiw-signature': signature['x-aiw-signature'] },
    bodyText,
    env: {
      STYLIST6_NODE_SHARED_SECRET: SECRET,
      // Deliberately NO STYLIST6_REFERENCE_BUCKET: this smoke proves the
      // reference-optional path (looks judged purely on product images).
      SUPABASE_URL: 'https://smoke-local.invalid',
      NVIDIA_API_KEY,
    },
    io: makeIo(catalogRows),
    fetchImpl: smokeFetch,
  });
  const wallMs = Date.now() - started;
  if (result.status !== 200) {
    return { wallMs, httpStatus: result.status, code: result.body.code, stage: result.body.stage };
  }
  const shape = deckShape(result.body);
  return {
    wallMs,
    httpStatus: result.status,
    ...shape,
    receipt: slimReceipt(result.body.visualReceipt || {}),
    generationRequestId: result.body.generationRequestId,
  };
}

const scenarios = [
  { name: 'A', style: 'old_money', occasion: 'Weekend', brief: 'relaxed old money weekend, no layer needed' },
  { name: 'B', style: 'casual', occasion: 'Weekend', brief: 'clean casual weekend' },
].slice(0, MAX_SCENARIOS);

const report = {
  ranAt: new Date().toISOString(),
  kind: 'cloud_visual_smoke',
  bundleSha256,
  bundle: 'backend/stylist-node/dist/stylist6-bundle.mjs',
  catalogRows: catalogRows.length,
  catalogPath: '.private/catalog-uz-20261113.json',
  keyPresent: true,
  referenceBucket: 'unset (reference-optional staging proof)',
  providerCallBudget: PROVIDER_CALL_BUDGET,
  imageCacheDir: path.relative(repoRoot, IMAGE_CACHE_DIR),
  scenarios: [],
};

for (const scenario of scenarios) {
  const request = {
    style: scenario.style,
    occasion: scenario.occasion,
    brief: scenario.brief,
    sourceMode: 'catalog',
    preferences: {},
    recentSignatures: [],
    excludeItemIds: [],
  };
  const entry = { name: scenario.name, style: scenario.style, occasion: scenario.occasion };
  try {
    entry.baseline = await runScenario(scenario.name, request, { visual: false });
    entry.visual = await runScenario(scenario.name, request, { visual: true });
    const comparable = entry.baseline.deckIds && entry.visual.deckIds;
    entry.selectionChanged = comparable
      ? JSON.stringify(entry.baseline.deckIds) !== JSON.stringify(entry.visual.deckIds)
      : null;
    if (entry.selectionChanged) {
      const index = entry.baseline.deckIds.findIndex((id, i) => entry.visual.deckIds[i] !== id);
      entry.firstDiffIndex = index;
      entry.changedFrom = entry.baseline.deckIds[index] ?? null;
      entry.changedTo = entry.visual.deckIds[index] ?? null;
    }
  } catch (error) {
    entry.error = { name: error?.name, message: error?.message };
  }
  report.scenarios.push(entry);
  console.error(`[smoke] scenario ${scenario.name} (${scenario.style}/${scenario.occasion}) baseline=${entry.baseline?.visualStatus ?? entry.error?.message} visual=${entry.visual?.visualStatus ?? entry.error?.message} changed=${entry.selectionChanged}`);
}

report.transport = {
  ...transportStats,
  providerCallsTotal,
  budgetUsed: providerCallsTotal,
  budgetRemaining: PROVIDER_CALL_BUDGET - providerCallsTotal,
};
report.imageCache = {
  files: fs.readdirSync(IMAGE_CACHE_DIR).length,
  bytes: fs.readdirSync(IMAGE_CACHE_DIR).reduce((sum, f) => sum + fs.statSync(path.join(IMAGE_CACHE_DIR, f)).size, 0),
};

const outPath = args.out
  ? path.resolve(repoRoot, args.out)
  : path.join(PRIVATE_DIR, `cloud-visual-smoke-${new Date().toISOString().slice(0, 10)}.json`);
fs.writeFileSync(outPath, JSON.stringify(report, null, 1) + '\n', { mode: 0o600 });

console.log(JSON.stringify(report, null, 1));
console.error(`[smoke] report written to ${outPath} (mode 600)`);
