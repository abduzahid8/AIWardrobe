// Frozen catalog comparison. Default is offline; --free-tier-v5 enables text ranking only.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { bundleV5, runV5 } from './v5_control.mjs';
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
const read = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const sha = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const local = (file) => import(pathToFileURL(path.join(root, 'backend/stylist6', file)));
globalThis.fetch = () => { throw new Error('Network disabled for this evaluation'); };

async function rankLooks({ api, plan, kept, scenario, preferences, weather }) {
  if (stopLive) return null;
  const lookIds = plan.candidates.map((look) => look.id);
  const { system, selectionUser } = api.frozenRankPayload(plan, kept, scenario.brief, scenario.occasion, weather, preferences, scenario.style);
  await delay(Math.max(0, nextDispatch - performance.now()));
  let httpStatus;
  const started = performance.now();
  const ranking = await api.rankNvidiaLooks(apiKey, NVIDIA_VISION_MODEL, system, selectionUser, lookIds, async (url, init) => {
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
  return ranking;
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
  const control = await runV5(v5, rows, { ...scenario, palette: exact.palette }, slotById, rankLive ? rankLooks : undefined);
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
