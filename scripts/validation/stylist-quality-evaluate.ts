import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { layerCompatibility } from '../../src/domain/layerCompatibility';
import {
  outfitVisualSignature,
  recentStylingPenalty,
} from '../../src/domain/outfitVisualSignature';
import { baselineModule } from './stylist-quality-baseline';
import { garmentContextReason, scoreGarmentForContext } from '../../src/domain/garmentContext';
import { rejectForStyleOccasion, rejectForClimate } from '../../src/domain/occasionRules';
/** Offline deterministic replay on GET-only public catalog snapshot; no providers. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { validatedQuartets } from '../../src/domain/validatedQuartets';
import { stylingContext } from '../../src/domain/stylingContext';
import { outfitChoices as currentChoices } from '../../src/domain/outfitChoices';
import {
  fourSlotOutfitReason as currentStructure,
  isDressGarment,
} from '../../src/domain/outfitStructure';
import {
  styleJudgeReason as currentJudge,
  assessOutfitQuality as currentQuality,
  spreadQualityFamilies,
} from '../../src/domain/occasionRules';
import { rankItemsForStyle, normalizeStyleId } from '../../src/domain/styleInference';
import {
  assembleDiverseDeck,
  buildOutfitFingerprint,
  garmentArchetypeToken,
  visualSimilarity,
  sharedVisualFamilies,
} from '../../src/domain/outfitDiversity';
const snapshot = JSON.parse(
  gunzipSync(readFileSync('docs/reports/stylist-quality/catalog-snapshot.json.gz')).toString('utf8')
);
const rows = snapshot.rows.map((r: any) => ({
  ...r,
  id: `shop_${r.id}`,
  isShopItem: true,
  imageUrl: r.image_url,
  type: r.garment_type,
  macroCategory:
    r.category === 'shoes' || r.garment_type === 'shoes'
      ? 'shoes'
      : r.category === 'bottoms' || r.garment_type === 'lower_body'
        ? 'bottom'
        : /\b(blazer|jacket|coat|cardigan|overshirt|sweater|pullover|hoodie|bomber)\b/i.test(r.name)
          ? 'outerwear'
          : 'top',
}));
const contexts = [
  ['minimalist', 'Date night'],
  ['semi_classic', 'Formal'],
  ['old_money', 'Date night'],
  ['casual', 'Travel'],
  ['minimalist', 'Work'],
  ['casual', 'Weekend'],
  ['casual', 'Travel', { temp: 3, condition: 'rain' }],
] as const;
const hash = (s: string) => {
  let n = 2166136261;
  for (const c of s) n = Math.imul(n ^ c.charCodeAt(0), 16777619);
  return n >>> 0;
};
const report: any = {
  version: process.argv[2] || 'baseline',
  catalogObservedAt: snapshot.observedAt,
  activeVisible: rows.length,
  kernelSha256: createHash('sha256')
    .update(
      process.argv[2] === 'baseline'
        ? readFileSync('scripts/validation/stylist-quality-baseline.json', 'utf8')
        : [
            'validatedQuartets',
            'garmentDescriptor',
            'garmentContext',
            'layerCompatibility',
            'outfitCoherence',
            'occasionRules',
            'outfitDiversity',
            'styleInference',
            'stylingContext',
            'stylingProfiles',
            'outfitVisualSignature',
          ]
            .map((name) => readFileSync(`src/domain/${name}.ts`, 'utf8'))
            .join('\n')
    )
    .digest('hex'),
  catalogSha256: createHash('sha256').update(JSON.stringify(snapshot.rows)).digest('hex'),
  method:
    'Offline replay: frozen pre-mission kernel vs current shared pair search/context/judge/deck on identical actual catalog snapshot and seeds. Current prefilter before truncation is an intended algorithm change. No provider calls or production latency. Full snapshot is evaluation-only, never downloaded to phone.',
  contexts: [],
};
const baseline = report.version === 'baseline';
const styleJudgeReason = baseline ? baselineModule('occasionRules').styleJudgeReason : currentJudge;
const assessOutfitQuality = baseline
  ? baselineModule('occasionRules').assessOutfitQuality
  : currentQuality;
const fourSlotOutfitReason = baseline
  ? baselineModule('outfitStructure').fourSlotOutfitReason
  : currentStructure;
const outfitChoices = baseline ? baselineModule('outfitChoices').outfitChoices : currentChoices;
for (const [style, occasion, weather] of contexts) {
  const rounds: any[] = [];
  const recentLookSignatures: string[] = [];
  let rejected = 0;
  const reasons: Record<string, number> = {};
  for (let round = 0; round < 10; round++) {
    const started = performance.now();
    const eligible = rows.filter((r: any) => {
      // Admission uses existing style/occasion gate through complete quartet validation below.
      return r.is_active && r.imageUrl && r.source_url && Number(r.price) > 0;
    });
    const ranked: any[] = rankItemsForStyle<any>(eligible, normalizeStyleId(style));
    const eligibleCounts: Record<string, number> = {};
    const pools = ['top', 'outerwear', 'bottom', 'shoes'].map((slot) => {
      const group = ranked.filter(
        (r: any) =>
          r.macroCategory === slot &&
          (report.version === 'baseline' ||
            (!garmentContextReason(r, stylingContext({ style, occasion, weather })) &&
              !rejectForStyleOccasion(r, style, occasion) &&
              !rejectForClimate(r, weather)))
      );
      eligibleCounts[slot] = group.length;
      // Seed only ties within a fixed quality band; identical for before/after.
      const preferred =
        report.version === 'baseline'
          ? group
          : group
              .map((item: any, index: number) => ({
                item,
                index,
                score: scoreGarmentForContext(item, stylingContext({ style, occasion, weather })),
              }))
              .sort((a: any, b: any) => b.score - a.score || a.index - b.index)
              .map((e: any) => e.item);
      const spread = spreadQualityFamilies(preferred, (r: any) => garmentArchetypeToken(r));
      const offset = group.length
        ? hash(`${style}:${occasion}:${round}:${slot}`) % Math.min(24, group.length)
        : 0;
      return [...spread.slice(offset), ...spread.slice(0, offset)].slice(0, 24);
    });
    const accepts = (parts: any[]) => {
      const reason =
        fourSlotOutfitReason(parts) || styleJudgeReason(parts, style, occasion, weather);
      if (reason) {
        rejected++;
        reasons[reason] = (reasons[reason] || 0) + 1;
        return false;
      }
      return true;
    };
    const choices =
      report.version === 'baseline'
        ? outfitChoices(pools.flat(), true, accepts, isDressGarment, [], 36, 6000, 1)
        : validatedQuartets(pools.flat(), stylingContext({ style, occasion, weather }), accepts, {
            maximum: 36,
            searchLimit: 6000,
          });
    const deck = assembleDiverseDeck({
      candidates: choices,
      itemsOf: (c: any) => c.items,
      qualityOf: (c: any) =>
        Math.max(
          0,
          assessOutfitQuality(c.items, style, occasion, weather).score -
            (baseline ? 0 : recentStylingPenalty(c.items, recentLookSignatures))
        ),
      target: 3,
      usedKeys: new Set(),
      poolItems: pools.flat(),
    }).deck;
    if (!baseline)
      recentLookSignatures.push(...deck.map((c: any) => outfitVisualSignature(c.items)));
    rounds.push({
      seed: round,
      eligibleCounts,
      candidateCounts: Object.fromEntries(
        pools.map((p, i) => [['top', 'layer', 'bottom', 'shoes'][i], p.length])
      ),
      elapsedMs: performance.now() - started,
      looks: deck.map((c: any) => ({
        items: c.items,
        quality: assessOutfitQuality(c.items, style, occasion, weather),
        descriptors: c.items.map(describeGarment),
        whyAdmitted: {
          structure: fourSlotOutfitReason(c.items),
          context: styleJudgeReason(c.items, style, occasion, weather),
          source: 'active public catalog',
          budget: 'uncapped; no conversion performed',
        },
        fingerprint: buildOutfitFingerprint(c.items),
      })),
    });
  }
  const looks = rounds.flatMap((r) => r.looks);
  const exact = new Set(
    looks.map((l) =>
      l.items
        .map((i: any) => i.id)
        .sort()
        .join('|')
    )
  );
  let similar = 0,
    total = 0;
  for (let i = 0; i < looks.length; i++)
    for (let j = i + 1; j < looks.length; j++) {
      total++;
      if (visualSimilarity(looks[i].fingerprint, looks[j].fingerprint) > 0.65) similar++;
    }
  const currentErrors = looks
    .map((l) => currentJudge(l.items, style, occasion, weather))
    .filter(Boolean);
  let layersInvalid = 0;
  for (const l of looks) {
    const top = l.items.find((i: any) => i.macroCategory === 'top'),
      layer = l.items.find((i: any) => i.macroCategory === 'outerwear');
    if (
      !top ||
      !layer ||
      !layerCompatibility(top, layer, stylingContext({ style, occasion, weather })).valid
    )
      layersInvalid++;
  }
  const duplicateDecks = rounds.filter(
    (r) =>
      new Set(
        r.looks.map((l: any) =>
          l.items
            .map((i: any) => i.id)
            .sort()
            .join('|')
        )
      ).size !== r.looks.length
  ).length;
  let familyPairs = 0;
  for (let i = 0; i < looks.length; i++)
    for (let j = i + 1; j < looks.length; j++)
      if (sharedVisualFamilies(looks[i].fingerprint, looks[j].fingerprint) >= 3) familyPairs++;
  const archetypes = new Set(
    looks.flatMap((l) =>
      l.items
        .filter((i: any) => i.macroCategory === 'shoes')
        .map((i: any) => describeGarment(i).subtype)
    )
  );
  report.contexts.push({
    style,
    occasion,
    weather: weather || null,
    emitted: looks.length,
    currentContextErrors: currentErrors.length,
    currentErrorReasons: [...new Set(currentErrors)],
    layerCompatibilityErrors: layersInvalid,
    fourSlotErrors: looks.filter((l) => currentStructure(l.items)).length,
    sourceErrors: looks.filter((l) => l.items.some((i: any) => !i.isShopItem)).length,
    withinDeckExactDuplicateCount: duplicateDecks,
    shoeArchetypes: [...archetypes],
    familyPairOverlap: total ? familyPairs / total : null,
    missing: 30 - looks.length,
    uniqueExact: exact.size,
    semanticPairOverlap: total ? similar / total : null,
    rejected,
    topRejections: Object.entries(reasons)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15),
    rounds,
  });
}
mkdirSync('docs/reports/stylist-quality', { recursive: true });
writeFileSync(
  `docs/reports/stylist-quality/${report.version}.json`,
  JSON.stringify(report, null, 2)
);
console.log(
  JSON.stringify(
    report.contexts.map(({ rounds, ...c }: any) => c),
    null,
    2
  )
);
