// Frozen catalog comparison. Default is offline; --free-tier-v5 enables text ranking only.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { NVIDIA_VISION_MODEL } from '../../backend/stylist6/visual_critic_hosted.mjs';

const [organizer, mode] = process.argv.slice(2);
if (!organizer || (mode && mode !== '--free-tier-v5')) throw new Error('expected organizer path and optional --free-tier-v5');
const root = path.resolve(import.meta.dirname, '../..');
const here = fs.realpathSync(organizer);
const rankLive = mode === '--free-tier-v5';
if (fs.existsSync(path.join(here, 'replay.json'))) throw new Error('use_fresh_replay_output_directory');
const apiKey = process.env.NVIDIA_API_KEY;
if (rankLive && !apiKey?.startsWith('nvapi-')) throw new Error('trial_credential_unavailable');
const trialFetch = globalThis.fetch;
const liveEvents = [];
let stopLive = false;
let nextDispatch = 0;
const repoRequire = createRequire(path.join(root, 'package.json'));
const read = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const sha = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const local = (file) => import(pathToFileURL(path.join(root, 'backend/stylist6', file)));
globalThis.fetch = () => { throw new Error('Network disabled for this evaluation'); };

async function bundleV5() {
  let esbuild;
  try {
    esbuild = repoRequire('esbuild');
  } catch {
    throw new Error('esbuild_unavailable_for_v5_baseline');
  }
  const entryPath = path.join(root, 'backend/supabase/functions/generate-outfits/index.ts');
  if (sha(entryPath) !== 'f90ced33ca3b0034c61cd2dfa289b189728dc75398cd72329f0b7c1854efc5e1') {
    throw new Error('v5_entry_changed');
  }
  const original = fs.readFileSync(entryPath, 'utf8');
  const payload = original.match(/const used=new Set\(plan\.candidates[\s\S]+?(?=            const ranking=)/)?.[0];
  if (!payload) throw new Error('v5_rank_payload_missing');
  const source = original.replace(/^import \{ serve \} from .*$/m, 'const serve=()=>{};')
    .replace(/^import \{ createClient \} from .*$/m, 'const createClient=()=>{throw Error("network forbidden")};');
  const entry = source
    + '\nimport {outfitStylingDistance} from \'../../../../src/domain/outfitVisualSignature.ts\';'
    + '\nexport {createOutfitPlan,outfitConcepts,stylingContext,prepareGarment,describeGarment,'
    + 'assessOutfitQuality,styleJudgeReason,fourSlotOutfitReason,outfitStructureReason,validateOutfitAgainstStyle,'
    + 'validateOutfitCompatibility,rejectForStyleOccasion,rejectForClimate,rejectForRequestIntent,'
    + 'rejectForActivity,retainBudgetFloorCandidates,scoreGarmentForContext,garmentContextReason,'
    + 'visualAdmissionReason,applyVisualFashion,inferShopColor,isFourSlotCandidate,isNonOutfitItem,'
    + 'outfitStylingDistance,rankNvidiaLooks};'
    + '\nexport function frozenRankPayload(plan,wardrobeItems,prompt,occasion,weather,preferences,selectionStyle){'
    + payload + '\nreturn {system,selectionUser};}';
  const built = await esbuild.build({
    stdin: {
      loader: 'ts', contents: entry, resolveDir: path.join(root, 'backend/supabase/functions/generate-outfits'),
      sourcefile: 'v207-pilot.ts',
    },
    bundle: true, format: 'cjs', platform: 'node', target: 'node22', write: false,
  });
  const module = { exports: {} };
  new Function('module', 'exports', 'require', 'Deno', built.outputFiles[0].text)(
    module, module.exports,
    (name) => name.startsWith('https:') ? { createClient: () => { throw Error('network forbidden'); } } : repoRequire(name),
    { env: { get: () => undefined } },
  );
  return module.exports;
}

async function runV5(api, snapshotRows, scenario, slotById) {
  const pool = snapshotRows
    .filter((r) => r.is_active && typeof r.price === 'number' && r.price > 0
      && String(r.image_url || '').startsWith('https://') && String(r.source_url || '').startsWith('https://'))
    .filter((r) => !scenario.budget || r.currency === scenario.budget.currency)
    .flatMap((r) => {
      const slot = slotById.get(r.id);
      if (!slot) return [];
      return [{
        ...r, id: 'shop_' + r.id, isShopItem: true, isWardrobe: false,
        macroCategory: slot, type: r.garment_type,
        imageUrl: r.image_url, shopUrl: r.source_url,
        color: api.inferShopColor(r),
      }];
    });
  const preferences = { favoriteColors: scenario.palette };
  const occasion = `${scenario.occasion} ${scenario.brief}`;
  const context = api.stylingContext({
    style: scenario.style, occasion: scenario.occasion, request: scenario.brief, preferences, sourceMode: 'catalog',
  });
  // No cached intelligence descriptors offline: prepareGarment falls back to
  // native describeGarment text analysis (same as uncached production rows).
  const items = pool.map((i) => api.prepareGarment(i));
  const gates = {
    visual: (i) => api.visualAdmissionReason(i),
    context: (i) => api.garmentContextReason(i, context),
    structure: (i) => !api.isFourSlotCandidate(i),
    accessory: (i) => api.isNonOutfitItem(i),
    style: (i) => api.rejectForStyleOccasion(i, scenario.style, scenario.occasion),
    climate: (i) => api.rejectForClimate(i, null),
    intent: (i) => api.rejectForRequestIntent(i, scenario.occasion, null),
    activity: (i) => api.rejectForActivity(i, scenario.occasion),
  };
  const eligible = items.filter((i) => !gates.visual(i) && !gates.context(i) && !gates.structure(i)
    && !gates.accessory(i) && !gates.style(i) && !gates.climate(i) && !gates.intent(i) && !gates.activity(i));
  const kept = api.retainBudgetFloorCandidates(eligible, eligible, (i) => i.price, () => true, undefined, 24,
    (i) => api.scoreGarmentForContext(i, context), context);
  const map = new Map(kept.map((i) => [i.id, i]));
  const evaluate = (parts) => {
    if (scenario.budget && parts.reduce((sum, i) => sum + i.price, 0) > scenario.budget.amount) {
      return { valid: false, score: 0 };
    }
    if (api.fourSlotOutfitReason(parts, true)) return { valid: false, score: 0 };
    if (!api.validateOutfitAgainstStyle({ items: parts }, scenario.style, map, true, preferences).ok
      || !api.validateOutfitCompatibility({ items: parts }, map, scenario.style).ok) {
      return { valid: false, score: 0 };
    }
    const a = api.assessOutfitQuality(parts, scenario.style, occasion, null, null, preferences, context);
    return {
      valid: a.acceptance.passed
        && !api.styleJudgeReason(parts, scenario.style, occasion, null, null, preferences, a, context),
      score: a.score,
    };
  };
  const plan = api.createOutfitPlan(kept, context, evaluate,
    { requiredIds: [], target: 3, priority: () => [], cost: (i) => i.price, maxEvaluations: 256, maxSyncMs: 600 });
  let ranking = null;
  if (rankLive && !stopLive && plan.candidates.length) {
    const lookIds = plan.candidates.map((look) => look.id);
    const { system, selectionUser } = api.frozenRankPayload(plan, kept, scenario.brief, scenario.occasion, null, preferences, scenario.style);
    await delay(Math.max(0, nextDispatch - performance.now()));
    let httpStatus;
    const started = performance.now();
    ranking = await api.rankNvidiaLooks(apiKey, NVIDIA_VISION_MODEL, system, selectionUser, lookIds, async (url, init) => {
      if (url !== 'https://integrate.api.nvidia.com/v1/chat/completions' || init.method !== 'POST'
          || liveEvents.length >= 12 || !JSON.parse(init.body).messages.every(({ content }) => typeof content === 'string')) {
        throw new Error('text_trial_dispatch_blocked');
      }
      nextDispatch = performance.now() + 3500;
      const response = await trialFetch(url, { ...init, redirect: 'error' });
      httpStatus = response.status;
      if ([401, 402, 403, 429].includes(httpStatus)) stopLive = true;
      return response;
    });
    const event = { id: scenario.id, httpStatus, latencyMs: performance.now() - started,
      ...ranking.attempt, rankedIds: ranking.ids };
    liveEvents.push(event);
    fs.appendFileSync(path.join(here, 'v5-provider-events.ndjson'), JSON.stringify(event) + '\n', { flush: true });
  }
  const ranks = new Map((ranking?.ids || []).map((id, index) => [id, index]));
  const selected = ranking?.ids ? plan.select(ranks) : plan.seed;
  const format = (looks) => looks.map((look) => look.items.map((item) => item.id.slice(5)));
  return { deck: format(selected.length ? selected : plan.seed), seed: format(plan.seed),
    mode: ranking?.ids ? ranking.attempt.status === 'repaired_hybrid' ? 'hybrid' : 'hosted_ranked' : 'deterministic_fallback' };
}

const scenarios = read(path.join(here, 'scenarios.json'));
const inventory = read(path.join(here, 'inventory.json'));
const index = read(path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json'));
const { adaptCatalog } = await local('catalog_adapter.mjs');
const { briefForReferenceId, referenceFirstBriefs } = await local('lookbook_briefs.mjs');
const { shortlistRetrievalDeck } = await local('prototype.mjs');

for (const item of inventory) {
  if (sha(path.join(here, item.photo)) !== item.photoSha256) throw Error('Photo hash mismatch: ' + item.row.id);
}
const catalog = adaptCatalog(inventory.map((entry) => entry.row)).items;
const v5 = await bundleV5();
if (rankLive) fs.writeFileSync(path.join(here, 'v5-provider-events.ndjson'), '', { flag: 'wx', mode: 0o600, flush: true });
const results = [];
const timing = [];
for (const scenario of scenarios) {
  const exact = briefForReferenceId(index, scenario.reference);
  const request = { ...scenario, sourceMode: 'catalog' };
  const pool = catalog.filter((item) => !scenario.budget || item.currency === scenario.budget.currency);
  const slotById = new Map(pool.map((item) => [item.id, item.macroCategory]));
  const rows = inventory.map((entry) => entry.row).filter((item) => slotById.has(item.id));
  const catalogRows = new Map(rows.map((row) => [row.id, row]));
  const briefs = [exact, ...referenceFirstBriefs(index, request, 8).filter((brief) => brief.id !== exact.id)]
    .slice(0, 8);
  const v6Started = performance.now();
  const retrieval = await shortlistRetrievalDeck({
    request, catalog: pool, creativeDirector: async () => briefs,
    hardGate: async (items) => ({
      ok: items.every(({ id }) => catalogRows.has(id)) &&
        !v5.outfitStructureReason(items, true) &&
        v5.validateOutfitCompatibility({ items }, catalogRows, scenario.style).ok,
    }),
    priceToBudgetCurrency: (item, currency) => item.currency === currency ? item.price : NaN,
    options: { maxVisual: 72, perRole: 24, perBrief: 120 },
  });
  const v6Ms = performance.now() - v6Started;
  const v5Started = performance.now();
  const control = await runV5(v5, rows, { ...scenario, palette: exact.palette }, slotById);
  const baseline = control.deck;
  const v5Ms = performance.now() - v5Started;
  timing.push({id:scenario.id,v5Ms,v6Ms});
  const v6Shortlist = retrieval.shortlist.filter((candidate) => candidate.briefId === exact.id);
  const v6Retrieval = v6Shortlist.slice(0, 3).map((candidate) => candidate.items.map((item) => item.id));
  for (const deck of [baseline, v6Retrieval]) {
    for (const look of deck) {
      const items = look.map((id) => pool.find((item) => item.id === id));
      if (items.some((item) => !item) || new Set(items.map((item) => item.macroCategory)).size !== 4
          || items.length !== 4
          || (scenario.budget && items.reduce((sum, item) => sum + item.price, 0) > scenario.budget.amount)) {
        throw Error('Hard constraint violation: ' + scenario.id);
      }
    }
  }
  results.push({
    id: scenario.id, poolSize: pool.length, baseline, v5Seed: control.seed, v5Mode: control.mode, v6Retrieval,
    v6Shortlist: v6Shortlist.map((c) => ({ id: c.id, items: c.items.map((i) => i.id) })),
    v6Hosted: null, hostedStatus: 'NOT_RUN',
  });
  console.log(scenario.id, pool.length, baseline.length, v6Retrieval.length);
}
const report = {
  repoCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
  v5EntrySha256: sha(path.join(root, 'backend/supabase/functions/generate-outfits/index.ts')),
  inventorySha256: sha(path.join(here, 'inventory.json')),
  scenariosSha256: sha(path.join(here, 'scenarios.json')),
  admitted: catalog.map(({ id, macroCategory, subtype }) => ({ id, macroCategory, subtype })),
  v5ProviderAttempts: liveEvents.length, results,
};
fs.writeFileSync(path.join(here, 'replay.json'), JSON.stringify(report, null, 2) + '\n');

const cases = results.map((row, n) => {
  const candidates = [];
  const seen = new Set();
  const options = row.v6Shortlist;
  for (const { id, items } of options) {
    const identity = [...items].sort().join('|');
    if (seen.has(identity)) continue;
    seen.add(identity);
    candidates.push({ id: row.id + '-' + String(candidates.length + 1).padStart(2, '0'), origin: id, items });
    if (candidates.length === 4) break;
  }
  if (candidates.length !== 4) throw Error('Insufficient distinct candidates for ' + row.id);
  return { scenario: scenarios[n], candidates };
});
fs.writeFileSync(path.join(here, 'paired_critic_queue.json'), JSON.stringify({
  status: 'NOT_RUN', providerCalls: 0, inventorySha256: report.inventorySha256,
  scenariosSha256: report.scenariosSha256, cases,
}, null, 2) + '\n');

const percentiles = (field) => {
  const values = timing.map((row) => row[field]).sort((a, b) => a - b);
  return {p50:values[Math.ceil(values.length * .5)-1], p95:values[Math.ceil(values.length * .95)-1]};
};
fs.writeFileSync(path.join(here, '../offline-planning-timing.json'), JSON.stringify({
  scope: rankLive ? 'Local V5 planner plus text ranker and V6 retrieval.' : 'Local V5 planner and V6 retrieval only.',
  node:process.version, v5Ms:percentiles('v5Ms'), v6Ms:percentiles('v6Ms'), timing
}, null, 2) + '\n');
