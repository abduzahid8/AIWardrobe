// Actual pinned V5 planner and validation used by isolated replay tools.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { hasColdLayerIntent, resolveOutfitComposition, resolveStylistWeather } from '../../src/domain/outfitComposition.ts';

const root = path.resolve(import.meta.dirname, '../..');
const repoRequire = createRequire(path.join(root, 'package.json'));
const sha = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');

export async function bundleV5() {
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

export async function runV5(api, snapshotRows, scenario, slotById, rankLooks) {
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
        visualFashion: r.visual_fashion || undefined, visualSourceHash: r.visual_source_hash || undefined,
      }];
    });
  const preferences = scenario.preferences ?? { favoriteColors: scenario.palette };
  const occasion = `${scenario.occasion} ${scenario.brief}`;
  const weather = resolveStylistWeather(scenario.brief, scenario.weather);
  const optionalLayer = scenario.optionalLayer === true;
  const contextualReplay = optionalLayer || !!weather;
  const requireLayer = !optionalLayer || hasColdLayerIntent(weather) || /\bformal\b|black[ -]?tie/i.test(scenario.occasion);
  const omitLayer = !requireLayer && !resolveOutfitComposition(scenario.style, weather, scenario.brief, optionalLayer).layered;
  const context = api.stylingContext({
    style: scenario.style, occasion: scenario.occasion, request: scenario.brief, preferences, weather,
    optionalLayer, sourceMode: 'catalog', ...(contextualReplay && scenario.budget ? { budget: scenario.budget } : {}),
  });
  // The V5 descriptor merges a current image profile with text evidence
  // using the same implementation as the deployed generation kernel.
  const items = pool.map((i) => api.prepareGarment(i));
  const gates = {
    visual: (i) => api.visualAdmissionReason(i),
    context: (i) => api.garmentContextReason(i, context),
    structure: (i) => !api.isFourSlotCandidate(i),
    accessory: (i) => api.isNonOutfitItem(i),
    style: (i) => api.rejectForStyleOccasion(i, scenario.style, scenario.occasion),
    climate: (i) => api.rejectForClimate(i, weather),
    intent: (i) => api.rejectForRequestIntent(i, contextualReplay ? occasion : scenario.occasion, weather),
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
    const layered = parts.some((item) => item.macroCategory === 'outerwear');
    if ((requireLayer && !layered) || (omitLayer && layered)
        || api.fourSlotOutfitReason(parts, requireLayer)) return { valid: false, score: 0 };
    if (!api.validateOutfitAgainstStyle({ items: parts }, scenario.style, map, layered, preferences).ok
      || !api.validateOutfitCompatibility({ items: parts }, map, scenario.style).ok) {
      return { valid: false, score: 0 };
    }
    const a = api.assessOutfitQuality(parts, scenario.style, occasion, weather, null, preferences, context);
    return {
      valid: a.acceptance.passed
        && !api.styleJudgeReason(parts, scenario.style, occasion, weather, null, preferences, a, context),
      score: a.score,
    };
  };
  const plan = api.createOutfitPlan(kept, context, evaluate,
    { requiredIds: [], target: 3, priority: () => [], cost: (i) => i.price, budget: contextualReplay ? scenario.budget?.amount : undefined, maxEvaluations: 256, maxSyncMs: 600 });
  const ranking = rankLooks && plan.candidates.length ? await rankLooks({ api, plan, kept, scenario, preferences, weather }) : null;
  const ranks = new Map((ranking?.ids || []).map((id, index) => [id, index]));
  const selected = ranking?.ids ? plan.select(ranks) : plan.seed;
  const format = (looks) => looks.map((look) => look.items.map((item) => item.id.slice(5)));
  return { deck: format(selected.length ? selected : plan.seed), seed: format(plan.seed),
    diagnostics: plan.diagnostics,
    mode: ranking?.ids ? ranking.attempt.status === 'repaired_hybrid' ? 'hybrid' : 'hosted_ranked' : rankLooks ? 'deterministic_fallback' : 'deterministic' };
}
