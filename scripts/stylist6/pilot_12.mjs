/**
 * Stylist 6 twelve-scenario pilot (offline, deterministic, no paid calls).
 *
 * Compares, on the SAME committed catalog snapshot:
 *   A. V5 baseline  — real `generate-outfits` v207 domain kernel
 *      (stylingContext/outfitConcepts/gates/createOutfitPlan), bundled
 *      locally with network forbidden.
 *   B. Creative-first S6 — prototype engine with deterministic house briefs.
 *   C. Reference-first S6 — prototype engine with lookbook briefs from the
 *      46-reference blueprint index.
 *
 * Arms B and C use `shortlistRetrievalDeck`: every deterministic stage up
 * to the hosted visual critic. Visual verdicts are NOT_RUN (no vision key
 * and no authorized reference bytes in this environment); the emitted decks
 * are retrieval shortlists for BLIND HUMAN review, never visually approved
 * looks. Budgets are evaluated in native currency only — no FX is invented.
 *
 * Usage:
 *   node scripts/stylist6/pilot_12.mjs --out /tmp/s6-pilot-20261110 \
 *     [--scenarios S6-01,S6-02,...] [--skip-v5]
 *
 * Private output (mode 700, never committed): full decks with product links.
 * Committed output: benchmarks/stylist6/pilot_12_report.json (metrics only).
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
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

const DEFAULT_SCENARIOS = [
  'S6-01', 'S6-02', 'S6-03', 'S6-04', 'S6-05',
  'S6-06', 'S6-07', 'S6-08',
  'S6-11', 'S6-12',
  'S6-B01', 'S6-B03',
];

const outDir = args.out || path.join('/tmp', 's6-pilot-20261110');
const scenarioIds = (args.scenarios || DEFAULT_SCENARIOS.join(',')).split(',').map((s) => s.trim());
const skipV5 = args['skip-v5'] === 'true';

const scenarios = JSON.parse(fs.readFileSync(path.join(root, 'benchmarks/stylist6/scenarios.json'), 'utf8'));
const index = JSON.parse(fs.readFileSync(
  path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json'), 'utf8'));

const byId = new Map(scenarios.scenarios.map((s) => [s.id, s]));
for (const id of scenarioIds) {
  if (!byId.has(id)) throw new Error(`unknown_scenario:${id}`);
}

function shaFile(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

async function loadSnapshotCatalog() {
  const dir = path.join(root, 'data');
  const rows = [];
  for (const file of fs.readdirSync(dir)) {
    if (!file.startsWith('shop_catalog_') || !file.endsWith('.json')) continue;
    const parsed = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
    rows.push(...parsed);
  }
  const h = crypto.createHash('sha256');
  for (const r of rows) h.update(String(r.id));
  return { rows, snapshotSha: h.digest('hex').slice(0, 16), files: fs.readdirSync(dir).length };
}

// ---- shared gates (native currency only; missing FX is a rejection) ----
const hardGate = async (items) => {
  const ids = items.map((i) => i.id);
  if (new Set(ids).size !== ids.length) return { ok: false, reason: 'duplicate_item' };
  return { ok: true };
};
const priceToBudgetCurrency = (item, currency) =>
  item.currency === currency ? item.price : NaN;

// ---- Arm C: reference-first briefs ----
const poolOptions = (request) => request.budget
  // Strict budgets starve score-ranked shortlists; widen the retrieval pool
  // (same engine, same hard filter) so compliant quartets can form.
  ? { briefLimit: 8, perRole: 24, maxVisual: 48, perBrief: 120 }
  : { briefLimit: 8, perRole: 12, maxVisual: 48, perBrief: 60 };

async function runReferenceFirst(stylist6, catalogItems, request) {
  const { referenceFirstBriefs } = await import('../../backend/stylist6/lookbook_briefs.mjs');
  const briefs = referenceFirstBriefs(index, { style: request.style, occasion: request.occasion }, 8);
  const t0 = Date.now();
  const deck = await stylist6.shortlistRetrievalDeck({
    request: { ...request, sourceMode: 'catalog' },
    catalog: catalogItems,
    creativeDirector: async () => briefs,
    hardGate, priceToBudgetCurrency,
    options: poolOptions(request),
  });
  return { ...deck, wallMs: Date.now() - t0 };
}

// ---- Arm B: creative-first house briefs (no reference material) ----
function houseBriefs(style) {
  const formal = ['old_money', 'classic'].includes(style);
  const topA = formal ? ['shirt', 'dress-shirt'] : ['polo', 'shirt'];
  const topB = formal ? ['fine-knit'] : ['t-shirt', 'polo'];
  const layerA = formal ? ['blazer'] : ['overshirt', 'jacket'];
  const layerB = formal ? ['cardigan', 'coat'] : ['bomber', 'jacket'];
  const botA = ['tailored-trouser', 'trouser'];
  const botB = formal ? ['tailored-trouser'] : ['chino', 'jeans'];
  const shoeA = formal ? ['loafer'] : ['minimal-sneaker', 'sneaker'];
  const shoeB = formal ? ['derby', 'loafer'] : ['sneaker', 'minimal-sneaker'];
  const palettes = formal
    ? [['navy', 'white', 'brown'], ['grey', 'ivory', 'black'], ['camel', 'cream', 'brown'],
       ['charcoal', 'sky', 'navy'], ['olive', 'cream', 'brown'], ['burgundy', 'grey', 'navy'],
       ['tan', 'white', 'brown'], ['black', 'grey', 'white']]
    : [['beige', 'white', 'olive'], ['navy', 'grey', 'white'], ['brown', 'cream', 'tan'],
       ['black', 'grey', 'white'], ['olive', 'sand', 'brown'], ['sky', 'navy', 'white'],
       ['stone', 'brown', 'beige'], ['charcoal', 'black', 'grey']];
  const layers = formal
    ? ['required', 'required', 'required', 'required', 'absent', 'required', 'required', 'absent']
    : ['absent', 'optional', 'required', 'absent', 'required', 'absent', 'required', 'optional'];
  const names = formal
    ? ['Boardroom tailoring', 'Heritage knit layering', 'Evening polish', 'Clubhouse classic',
       'Shirt-and-trouser ease', 'Overcoat formality', 'Suede-trim tradition', 'Weekend refinement']
    : ['Off-duty ease', 'Soft utility', 'City casual', 'Weekend minimal',
       'Layered comfort', 'Staple denim', 'Smart leisure', 'Travel relaxed'];
  return names.map((name, i) => ({
    id: `house_${i + 1}`,
    name,
    roles: {
      top: i % 2 ? topB : topA,
      outerwear: i % 2 ? layerB : layerA,
      bottom: i % 2 ? botB : botA,
      shoes: i % 2 ? shoeB : shoeA,
    },
    layerPolicy: layers[i],
    palette: palettes[i],
    silhouette: ['balanced', 'relaxed', 'tailored', 'soft'][i % 4],
  }));
}

async function runCreativeFirst(stylist6, catalogItems, request) {
  const briefs = houseBriefs(request.style);
  const t0 = Date.now();
  const deck = await stylist6.shortlistRetrievalDeck({
    request: { ...request, sourceMode: 'catalog' },
    catalog: catalogItems,
    creativeDirector: async () => briefs,
    hardGate, priceToBudgetCurrency,
    options: poolOptions(request),
  });
  return { ...deck, wallMs: Date.now() - t0 };
}

// ---- Arm A: real V5 v207 domain kernel, in-memory, network forbidden ----
async function bundleV5() {
  let esbuild;
  try {
    esbuild = require('esbuild');
  } catch {
    throw new Error('esbuild_unavailable_for_v5_baseline');
  }
  const source = fs.readFileSync(
    path.join(root, 'backend/supabase/functions/generate-outfits/index.ts'), 'utf8')
    .replace(/^import \{ serve \} from .*$/m, 'const serve=()=>{};')
    .replace(/^import \{ createClient \} from .*$/m, 'const createClient=()=>{throw Error("network forbidden")};');
  const entry = source
    + '\nimport {outfitStylingDistance} from \'../../../../src/domain/outfitVisualSignature.ts\';'
    + '\nexport {createOutfitPlan,outfitConcepts,stylingContext,prepareGarment,describeGarment,'
    + 'assessOutfitQuality,styleJudgeReason,fourSlotOutfitReason,validateOutfitAgainstStyle,'
    + 'validateOutfitCompatibility,rejectForStyleOccasion,rejectForClimate,rejectForRequestIntent,'
    + 'rejectForActivity,retainBudgetFloorCandidates,scoreGarmentForContext,garmentContextReason,'
    + 'visualAdmissionReason,applyVisualFashion,inferShopColor,isFourSlotCandidate,isNonOutfitItem,'
    + 'outfitStylingDistance};';
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
    (name) => name.startsWith('https:') ? { createClient: () => { throw Error('network forbidden'); } } : require(name),
    { env: { get: () => undefined } },
  );
  return module.exports;
}

// Production V5 reads slots from catalog_garment_intelligence (unavailable
// offline), so the V5 arm reuses the same transparent pilot slot projection
// for POOL CONSTRUCTION ONLY. All V5 gates, descriptors and the planner run
// natively on the raw snapshot rows. Rows the adapter cannot project are
// excluded from the V5 pool and counted.
async function runV5(api, snapshotRows, scenario, slotById) {
  const t0 = Date.now();
  let unprojected = 0;
  const pool = snapshotRows
    .filter((r) => r.is_active && typeof r.price === 'number' && r.price > 0
      && String(r.image_url || '').startsWith('https://') && String(r.source_url || '').startsWith('https://'))
    .filter((r) => !scenario.budget || r.currency === scenario.budget.currency)
    .flatMap((r) => {
      const slot = slotById.get(r.id);
      if (!slot) { unprojected += 1; return []; }
      return [{
        ...r, id: 'shop_' + r.id, isShopItem: true, isWardrobe: false,
        macroCategory: slot, type: r.garment_type,
        imageUrl: r.image_url, shopUrl: r.source_url,
        color: api.inferShopColor(r),
      }];
    });
  const context = api.stylingContext({ style: scenario.style, occasion: scenario.occasion, sourceMode: 'catalog' });
  const concepts = api.outfitConcepts(context).slice(0, 3);
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
  const rejections = Object.fromEntries(
    Object.entries(gates).map(([name, check]) => [name, items.filter(check).length]));
  const eligible = items.filter((i) => !gates.visual(i) && !gates.context(i) && !gates.structure(i)
    && !gates.accessory(i) && !gates.style(i) && !gates.climate(i) && !gates.intent(i) && !gates.activity(i));
  const kept = api.retainBudgetFloorCandidates(eligible, eligible, () => 0, () => true, undefined, 24,
    (i) => api.scoreGarmentForContext(i, context), context);
  const map = new Map(kept.map((i) => [i.id, i]));
  const evaluate = (parts) => {
    if (api.fourSlotOutfitReason(parts, true)) return { valid: false, score: 0 };
    if (!api.validateOutfitAgainstStyle({ items: parts }, scenario.style, map, true, {}).ok
      || !api.validateOutfitCompatibility({ items: parts }, map, scenario.style).ok) {
      return { valid: false, score: 0 };
    }
    const a = api.assessOutfitQuality(parts, scenario.style, scenario.occasion, null, null, {}, context);
    return {
      valid: a.acceptance.passed
        && !api.styleJudgeReason(parts, scenario.style, scenario.occasion, null, null, {}, a, context),
      score: a.score,
    };
  };
  const plan = api.createOutfitPlan(kept, context, evaluate,
    { requiredIds: [], target: 3, priority: () => [], cost: (i) => i.price, maxEvaluations: 256, maxSyncMs: 600 });
  const wallMs = Date.now() - t0;
  return {
    engine: 'generate-outfits-v207-kernel',
    wallMs,
    deck: plan.seed.map((look) => ({
      concept: look.concept?.id || 'v5',
      conceptScore: look.conceptScore ?? null,
      quality: look.quality ?? null,
      items: look.items.map((i) => ({
        id: i.id, name: i.name, price: i.price, currency: i.currency,
        imageUrl: i.imageUrl, shopUrl: i.shopUrl, macroCategory: i.macroCategory,
      })),
    })),
    diagnostics: {
      ...plan.diagnostics, eligible: eligible.length, poolSize: kept.length, rejections,
      unprojectedPoolRows: unprojected, poolSlotSource: 'pilot_adapter_projection_v1',
      concepts: concepts.map((c) => c.id),
    },
  };
}

function priceTotals(looks) {
  return looks.map((l) => {
    const byCurrency = {};
    for (const i of l.items) byCurrency[i.currency] = (byCurrency[i.currency] || 0) + i.price;
    const currencies = Object.keys(byCurrency);
    return {
      singleCurrency: currencies.length === 1,
      byCurrency,
      amount: currencies.length === 1 ? byCurrency[currencies[0]] : null,
      currency: currencies.length === 1 ? currencies[0] : null,
    };
  });
}

function normalizeDeck(deck, arm) {
  // First three shortlist entries = top retrieval pick of first three briefs (round-robin).
  return deck.shortlist.slice(0, 3).map((c) => ({
    briefId: c.briefId,
    referenceId: c.brief.referenceId || null,
    concept: c.brief.name,
    seedScore: c.seedScore,
    items: c.items.map((i) => ({
      id: i.id, name: i.name, price: i.price, currency: i.currency,
      imageUrl: i.imageUrl, shopUrl: i.shopUrl, macroCategory: i.macroCategory,
      subtype: i.subtype, projection: i.projection?.kind || null,
    })),
  }));
}

const strictDistinct = (looks) => {
  const ids = looks.flatMap((l) => l.items.map((i) => i.id));
  return new Set(ids).size === ids.length;
};

async function main() {
  fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });
  const gitSha = (() => {
    try {
      return require('child_process').execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
    } catch { return 'unknown'; }
  })();

  const { rows, snapshotSha } = await loadSnapshotCatalog();
  const { adaptCatalog } = await import('../../backend/stylist6/catalog_adapter.mjs');
  const adapted = adaptCatalog(rows);
  const stylist6 = await import('../../backend/stylist6/prototype.mjs');

  let v5api = null;
  let v5error = null;
  if (!skipV5) {
    try {
      v5api = await bundleV5();
    } catch (error) {
      v5error = error instanceof Error ? error.message : String(error);
    }
  }

  const v207sha = shaFile(path.join(root, 'backend/supabase/functions/generate-outfits/index.ts'));
  const report = {
    pilot: 'stylist6-12-scenario-pilot',
    generatedAt: new Date().toISOString(),
    gitSha,
    provenance: {
      v207sha256: v207sha,
      stylist6prototype: stylist6.STYLIST6_VERSION || 'stylist6-visual-first-prototype-1',
      catalogAdapter: adapted.adapterVersion,
      catalogSnapshotSha: snapshotSha,
      catalogRows: rows.length,
      catalogAdmitted: adapted.items.length,
      catalogSkipped: adapted.skipped,
      referenceIndex: index.dataset_id,
      visualVerdict: 'NOT_RUN',
      visualNotRunReason: 'no hosted vision key and no authorized reference bytes in this environment; '
        + 'decks are retrieval shortlists for blind human review',
    },
    scenarios: [],
  };
  const privateOut = { ...report, runs: [] };

  for (const id of scenarioIds) {
    const scenario = byId.get(id);
    const request = {
      style: scenario.style, occasion: scenario.occasion,
      ...(scenario.budget ? { budget: scenario.budget } : {}),
    };
    console.log(`scenario ${id} ${scenario.style}/${scenario.occasion}${scenario.budget ? ` budget=${scenario.budget.amount}${scenario.budget.currency}` : ''}`);
    const entry = { id, style: scenario.style, occasion: scenario.occasion, budget: scenario.budget || null, arms: {} };
    const priv = { id, arms: {} };
    // All three arms shop the SAME inventory slice. Budget scenarios use the
    // native-currency slice for every arm (no FX is invented anywhere); the
    // engine-level hard budget filter then applies identically.
    const armCatalog = scenario.budget
      ? adapted.items.filter((i) => i.currency === scenario.budget.currency)
      : adapted.items;
    entry.inventorySlice = scenario.budget
      ? { currency: scenario.budget.currency, items: armCatalog.length } : { currency: null, items: armCatalog.length };

    const [refDeck, creDeck] = await Promise.all([
      runReferenceFirst(stylist6, armCatalog, request),
      runCreativeFirst(stylist6, armCatalog, request),
    ]);
    for (const [arm, deck] of [['reference-first', refDeck], ['creative-first', creDeck]]) {
      const looks = normalizeDeck(deck, arm);
      const totals = priceTotals(looks);
      const budgetOk = scenario.budget
        ? totals.map((t) => t.singleCurrency && t.currency === scenario.budget.currency
          ? t.amount <= scenario.budget.amount : false)
        : null;
      entry.arms[arm] = {
        looks: looks.length,
        distinctInventory: strictDistinct(looks),
        shortlisted: deck.diagnostics.shortlisted,
        candidateCount: deck.diagnostics.candidateCount,
        admittedCatalogItems: deck.diagnostics.admittedCatalogItems,
        rejections: deck.diagnostics.rejections,
        wallMs: deck.wallMs,
        visualVerdict: 'NOT_RUN',
        totals,
        budgetOk,
      };
      priv.arms[arm] = { looks, wallMs: deck.wallMs };
    }

    if (v5api) {
      try {
        const slotById = new Map(adapted.items.map((i) => [i.id, i.macroCategory]));
        const v5rows = scenario.budget ? rows.filter((r) => r.currency === scenario.budget.currency) : rows;
        const v5 = await runV5(v5api, v5rows, scenario, slotById);
        // Same hard budget rule for the control: over-cap V5 looks are
        // dropped from the compliant deck (the in-harness planner only
        // value-scores cost; production enforces the cap at the edge).
        // Non-compliant proposals are counted, never shown as valid.
        const rawTotals = priceTotals(v5.deck);
        const rawBudgetOk = scenario.budget
          ? rawTotals.map((t) => t.singleCurrency && t.currency === scenario.budget.currency
            ? t.amount <= scenario.budget.amount : false)
          : null;
        if (scenario.budget) {
          v5.deck = v5.deck.filter((_, i) => rawBudgetOk[i]);
          v5.diagnostics.budgetNonCompliantDropped = rawBudgetOk.filter((x) => !x).length;
        }
        const totals = priceTotals(v5.deck);
        const budgetOk = scenario.budget ? totals.map(() => true) : null;
        entry.arms['v5-baseline'] = {
          looks: v5.deck.length,
          distinctInventory: strictDistinct(v5.deck),
          wallMs: v5.wallMs,
          visualVerdict: 'N/A-control',
          totals,
          budgetOk,
          diagnostics: v5.diagnostics,
        };
        priv.arms['v5-baseline'] = { looks: v5.deck, wallMs: v5.wallMs };
      } catch (error) {
        entry.arms['v5-baseline'] = { error: error instanceof Error ? error.message : String(error) };
      }
    } else {
      entry.arms['v5-baseline'] = { error: v5error || 'skipped' };
    }

    report.scenarios.push(entry);
    privateOut.runs.push({ id, ...priv });
  }

  fs.writeFileSync(path.join(outDir, 'pilot-private.json'), JSON.stringify(privateOut, null, 2), { mode: 0o600 });
  fs.writeFileSync(
    path.join(root, 'benchmarks/stylist6/pilot_12_report.json'),
    JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({
    scenarios: report.scenarios.length,
    outDir, reportFile: 'benchmarks/stylist6/pilot_12_report.json',
    v5: v5api ? 'ran' : `NOT_RUN(${v5error || 'skipped'})`,
  }));
}

await main();
