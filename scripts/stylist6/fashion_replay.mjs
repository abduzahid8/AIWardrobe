// Reproducible frozen-catalog fashion comparison. No image requests or inference.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { bundleV5, runV5 } from './v5_control.mjs';
import { adaptCatalog, baseProductId } from '../../backend/stylist6/catalog_adapter.mjs';
import { fashionStylingContext, recommendFashionDeck } from '../../backend/stylist6/fashion_deck.mjs';
import { referenceFirstBriefs } from '../../backend/stylist6/lookbook_briefs.mjs';
import { shortlistRetrievalDeck } from '../../backend/stylist6/prototype.mjs';
import { describeGarment, prepareGarment } from '../../src/domain/garmentDescriptor.ts';
import { stylingContext } from '../../src/domain/stylingContext.ts';
import { outfitCoherence } from '../../src/domain/outfitCoherence.ts';
import { outfitStylingDistance } from '../../src/domain/outfitVisualSignature.ts';
import { garmentContextReason } from '../../src/domain/garmentContext.ts';
import { hasColdLayerIntent, resolveOutfitComposition, resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { rejectForClimate } from '../../src/domain/occasionRules.ts';

const [snapshot, destination] = process.argv.slice(2);
if (!snapshot || !destination) throw new Error('expected catalog snapshot and output path');
const root = path.resolve(import.meta.dirname, '../..');
const bytes = fs.readFileSync(snapshot);
const raw = JSON.parse(bytes.toString('utf8'));
if (!Array.isArray(raw) || !raw.length || raw.some(({ currency }) => currency !== 'UZS')) throw new Error('expected native UZS catalog');
const { items } = adaptCatalog(raw);
const byId = new Map(items.map((item) => [item.id, prepareGarment(item)]));
const slotById = new Map(items.map((item) => [item.id, item.macroCategory]));
if (byId.size !== items.length) throw new Error('duplicate_catalog_sku');
const index = JSON.parse(fs.readFileSync(path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json')));
const base = JSON.parse(fs.readFileSync(path.join(root, 'benchmarks/stylist6/scenarios.json'))).scenarios;
const temperatures = [23, 28, 17, 9, 21];
const preferences = [
  { favoriteColors: ['navy', 'cream'] },
  { avoidColors: ['black'], fitPreference: 'relaxed' },
  { dislikedPatterns: ['graphic'], fitPreference: 'regular' },
];
const scenarios = base.map((scenario, i) => ({
  id: scenario.id, style: scenario.style, occasion: scenario.occasion, budget: scenario.budget,
  weather: { temp: temperatures[i % temperatures.length] },
  preferences: preferences[i % preferences.length], sourceMode: 'catalog', optionalLayer: true, brief: '',
})).concat([
  { id: 'S6-F36', style: 'streetwear', occasion: 'Weekend', weather: { temp: 16 }, preferences: { fitPreference: 'relaxed' } },
  { id: 'S6-F37', style: 'streetwear', occasion: 'Casual', weather: { temp: 29 }, preferences: { avoidColors: ['black'] }, budget: { amount: 4_000_000, currency: 'UZS' } },
  { id: 'S6-F38', style: 'classic', occasion: 'Formal dinner', weather: { temp: 20 }, preferences: { dislikedPatterns: ['print'] }, budget: { amount: 12_000_000, currency: 'UZS' } },
  { id: 'S6-F39', style: 'smart_casual', occasion: 'Date night', weather: { temp: 27 }, preferences: { favoriteColors: ['brown'] }, brief: 'No outerwear' },
].map((scenario) => ({ sourceMode: 'catalog', optionalLayer: true, brief: '', ...scenario })));

globalThis.fetch = () => { throw new Error('fashion replay is offline'); };
const v5 = await bundleV5();
const average = (values) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
const summarizeLook = (ids, context, reference) => {
  const parts = ids.map((id) => byId.get(id));
  if (parts.some((item) => !item)) throw new Error('look_not_in_catalog');
  const judge = outfitCoherence(parts, context);
  const descriptors = parts.map(describeGarment);
  const refColors = new Set((reference?.palette || []).map((color) => color.replaceAll('_', '-')));
  const knownColors = descriptors.filter(({ color }) => !['unknown', 'neutral', ''].includes(color));
  const spend = parts.reduce((sum, item) => sum + item.price, 0);
  const physical = new Set(ids.map(baseProductId));
  return {
    ids, spend, layered: parts.some(({ macroCategory }) => macroCategory === 'outerwear'),
    structure: v5.outfitStructureReason(parts, false),
    feasibleBudget: !context.budget || (parts.every((item) => item.currency === context.budget.currency) && spend <= context.budget.amount + .000001),
    uniqueProducts: physical.size === ids.length,
    coherence: judge.score, coherent: judge.valid, reasons: judge.reasons,
    dimensions: judge.dimensions, evidence: judge.evidence,
    imageSupportedItems: descriptors.filter((d) => d.evidence.some((entry) => entry.startsWith('image:'))).length,
    referencePaletteFit: reference ? average(knownColors.map(({ color }) => Number(refColors.has(color)))) : null,
  };
};
const results = [];
for (const scenario of scenarios) {
  const weather = resolveStylistWeather(scenario.brief, scenario.weather);
  const context = stylingContext({ ...scenario, weather, request: scenario.brief });
  const refs = referenceFirstBriefs(index, { ...scenario, season: weather.temp >= 25 ? 'summer' : weather.temp < 12 ? 'winter' : 'autumn' });
  // Metadata reference is an input common to both engines, only when the
  // reference shortlist contains the requested style and occasion.
  const reference = refs.find(({ referenceId }) => {
    const ref = index.images.find((entry) => entry.id === referenceId);
    return ref.styles.includes(scenario.style) && ref.occasions.includes(scenario.occasion.toLowerCase().replaceAll(' ', '_'));
  });
  const input = { ...scenario, weather, referenceId: reference?.referenceId };
  const auditContext = fashionStylingContext(input);
  const pool = scenario.budget ? items.filter((item) => item.currency === scenario.budget.currency) : items;
  const requireLayer = hasColdLayerIntent(weather) || context.dressCode === 'formal';
  const omitLayer = !requireLayer && !resolveOutfitComposition(context.style, weather, context.request, true).layered;
  const hardGate = (look) => {
    const prepared = look.map(({ id }) => byId.get(id));
    if (prepared.some((item) => !item)) return { ok: false };
    const layered = prepared.some((item) => item.macroCategory === 'outerwear');
    return { ok: (!requireLayer || layered) && (!omitLayer || !layered) &&
      !v5.outfitStructureReason(prepared, false) &&
      prepared.every((item) => !garmentContextReason(item, context) && !rejectForClimate(item, weather)) };
  };
  const priceToBudgetCurrency = (item, currency) => item.currency === currency ? item.price : NaN;
  const started = performance.now();
  const improved = await recommendFashionDeck({
    request: input, catalog: pool, referenceBriefs: refs, hardGate, priceToBudgetCurrency,
  });
  const improvedMs = performance.now() - started;
  const before = await shortlistRetrievalDeck({
    request: input, catalog: pool, creativeDirector: async () => refs, hardGate, priceToBudgetCurrency,
    options: { perRole: 8, maxVisual: 48, perBrief: 100 },
  });
  const controlStarted = performance.now();
  const control = await runV5(v5, raw, input, slotById);
  const controlMs = performance.now() - controlStarted;
  const arms = {
    stylist5: control.deck.map((ids) => summarizeLook(ids, auditContext, reference)),
    stylist6Previous: before.shortlist.slice(0, 3).map(({ items }) => summarizeLook(items.map(({ id }) => id), auditContext, reference)),
    stylist6Fashion: improved.outfits.map((look) => ({
      ...summarizeLook(look.items.map(({ id }) => id), auditContext, reference), description: look.description,
      referenceMetadata: look.fashion.referenceMetadata, rankScore: look.fashion.score,
    })),
  };
  const metrics = Object.fromEntries(Object.entries(arms).map(([name, looks]) => {
    const pairs = looks.flatMap((a, i) => looks.slice(i + 1).map((b) =>
      outfitStylingDistance(a.ids.map((id) => byId.get(id)), b.ids.map((id) => byId.get(id)))));
    return [name, { looks: looks.length, coherence: average(looks.map((look) => look.coherence)),
      coherentLooks: looks.filter((look) => look.coherent).length, stylingDistance: average(pairs),
      uniqueSkus: new Set(looks.flatMap(({ ids }) => ids.map(baseProductId))).size,
      budgetViolations: looks.filter((look) => !look.feasibleBudget).length,
      constraintViolations: looks.filter((look) => look.structure || !look.uniqueProducts || !look.feasibleBudget ||
        (requireLayer && !look.layered) || (omitLayer && look.layered) ||
        !hardGate(look.ids.map((id) => byId.get(id))).ok).length,
      referencePaletteFit: average(looks.map((look) => look.referencePaletteFit).filter(Number.isFinite)),
    }];
  }));
  results.push({ scenario: input, reference, arms, metrics, improvedMs, controlMs,
    v5CpuRisk: control.diagnostics.cpuRisk, fashionDiagnostics: improved.diagnostics });
  console.log(scenario.id, Object.values(metrics).map(({ looks }) => looks).join('/'));
}
const catalog = Object.fromEntries(items.map((item) => [item.id, {
  id: item.id, name: item.name, brand: item.brand, imageUrl: item.imageUrl, shopUrl: item.shopUrl,
  price: item.price, currency: item.currency, slot: item.macroCategory,
}]));
const report = {
  scope: 'Frozen catalog, actual V5 deterministic kernel, previous V6 retrieval, current V6 fashion ranker',
  snapshotSha256: createHash('sha256').update(bytes).digest('hex'), independentHumanVotes: 0, providerCalls: 0,
  catalog, results,
};
fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
