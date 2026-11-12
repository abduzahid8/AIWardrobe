// Experimental catalog/wardrobe recommendation path. Callers own inventory,
// permission and provider admission. Local styling never requires image upload.
import { shortlistRetrievalDeck, evaluateVisualShortlist } from './prototype.mjs';
import { baseProductId } from './catalog_adapter.mjs';
import { describeGarment, prepareGarment } from '../../src/domain/garmentDescriptor.ts';
import { currentVisual, visualAdmissionReason } from '../../src/domain/visualFashion.ts';
import { outfitCoherence } from '../../src/domain/outfitCoherence.ts';
import { garmentContextReason, scoreGarmentForContext } from '../../src/domain/garmentContext.ts';
import { lookConcept, outfitConcepts } from '../../src/domain/outfitConcepts.ts';
import { isDateNightOccasion, stylingContext } from '../../src/domain/stylingContext.ts';
import { hasColdLayerIntent, resolveOutfitComposition, resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { outfitStylingFeatures, recentStylingPenalty, stylingFeatureDistance } from '../../src/domain/outfitVisualSignature.ts';
import { colorTokensOf } from '../../src/domain/outfitDiversity.ts';
import { personalTaste, evaluatePersonalItem, evaluatePersonalLook } from './personal_taste.mjs';
import { isNonOutfitItem, outfitStructureReason } from '../../src/domain/outfitStructure.ts';
import { rejectForClimate, rejectForStyleOccasion } from '../../src/domain/occasionRules.ts';
import { groundedLookExplanation } from '../../src/domain/lookExplanation.ts';

const slots = ['top', 'outerwear', 'bottom', 'shoes'];
const normalized = (value) => String(value || '').toLowerCase().replace(/[\s-]+/g, '_');
const known = (value) => !!value && value !== 'unknown' && value !== 'neutral';
const image = (item) => item.imageUrl || item.image_url;
const sameProduct = (a, b) => baseProductId(a.id) === baseProductId(b.id) ||
  (!!image(a) && image(a) === image(b)) ||
  (!!currentVisual(a)?.imageSha256 && currentVisual(a).imageSha256 === currentVisual(b)?.imageSha256);
const overlaps = (a, b) => a.items.filter((item) => b.items.some((other) => sameProduct(item, other))).length;
const SEARCH_DECK_LIMIT = 20;
const TIE_MARGIN = 0.025;
// A selected look this far below the shortlist best is decisively off the
// pace (10x the near-tie margin), not a close alternative worth showing.
const EXCLUSION_QUALITY_GAP = 0.25;
// Minimum per-currency samples before an unbudgeted premium dampener may
// apply. Below this there is no evidence for what "expensive" means here.
const PRICE_SAMPLE_MINIMUM = 8;

/**
 * Per slot-and-currency typical-price bands over the admitted pool. Pure
 * evidence about this catalog — no FX, no invented budget. Exported for
 * tests.
 */
export function buildPriceProfile(catalog) {
  const byBand = new Map();
  for (const item of catalog || []) {
    if (!item || !/^(top|outerwear|bottom|shoes)$/.test(item.macroCategory || '')
      || !Number.isFinite(item.price) || item.price <= 0 || !/^[A-Z]{3}$/.test(item.currency || '')) continue;
    const key = `${item.currency}|${item.macroCategory}`;
    if (!byBand.has(key)) byBand.set(key, []);
    byBand.get(key).push(item.price);
  }
  const profile = {};
  for (const [key, prices] of byBand) {
    if (prices.length < PRICE_SAMPLE_MINIMUM) continue;
    prices.sort((a, b) => a - b);
    profile[key] = { median: prices[Math.floor((prices.length - 1) / 2)], count: prices.length };
  }
  return profile;
}

/**
 * Soft, bounded dampener for unbudgeted requests: items priced above their
 * slot's typical price in the same currency cost up to 0.06 score in total.
 * It breaks near-ties toward representative prices but never rejects —
 * without a budget there is nothing to fail. A look is flagged only when an
 * item costs more than twice its slot median. Exported for tests.
 */
export function pricePenaltyForLook(items, profile) {
  let penalty = 0, flagged = 0;
  for (const item of items || []) {
    const band = item && profile ? profile[`${item.currency}|${item.macroCategory}`] : null;
    if (!band || !(item.price > band.median)) continue;
    penalty += 0.06 * Math.min(1, (item.price - band.median) / band.median);
    if (item.price > 2 * band.median) flagged += 1;
  }
  return { penalty: Math.min(0.06, penalty), flagged };
}

function referenceAffinity(items, reference) {
  if (!reference) return null;
  const descriptions = items.map(describeGarment);
  const roles = descriptions.map((d, index) => {
    const expected = reference.roles[items[index].macroCategory] || [];
    return expected.some((subtype) => normalized(subtype) === normalized(d.subtype)) ? 1 : 0;
  });
  const colors = descriptions.map(({ color }) => color).filter(known);
  const palette = new Set(reference.palette.map(normalized));
  const paletteFit = colors.length ? colors.filter((color) => palette.has(normalized(color))).length / colors.length : null;
  const slotMatches = descriptions.flatMap(({ color }, index) => {
    const notes = reference.garmentNotes?.[items[index].macroCategory];
    const expected = colorTokensOf({ name: notes });
    return known(color) && expected.length ? [Number(expected.includes(color))] : [];
  });
  const layered = items.some((item) => item.macroCategory === 'outerwear');
  const layerFit = reference.layerPolicy === 'optional' || layered === (reference.layerPolicy === 'required');
  return {
    roleFit: roles.reduce((a, b) => a + b, 0) / roles.length,
    paletteFit,
    slotColorFit: slotMatches.length ? slotMatches.reduce((a, b) => a + b, 0) / slotMatches.length : null,
    layerFit,
  };
}

/**
 * Date-night register (backend ranker only): shorts and athletic bottoms
 * read daytime-casual under smart/formal evening dress codes, even when the
 * catalog files them as bottoms. Casual dress codes and daytime wear keep
 * them — this is a context rejection, never a garment ban.
 */
function rejectForUnsuitableDateNight(descriptor, context) {
  if (descriptor.slot !== 'bottom') return false;
  if (context.dressCode !== 'smart' && context.dressCode !== 'formal') return false;
  if (!isDateNightOccasion(context.occasion) && !isDateNightOccasion(context.request)) return false;
  return descriptor.subtype === 'shorts' || descriptor.athletic === true;
}

/**
 * Single-currency preference for one look: all-shown-currency decks total
 * honestly and shop simply. Tie-break scale only — never overrides fit.
 * Exported for tests.
 */
export function singleCurrencyBonus(items) {
  const currencies = new Set(
    (items || []).map((item) => item?.currency).filter(Boolean).map(String));
  return currencies.size === 1 ? 0.02 : 0;
}

/**
 * Exclusion quality floor: when previously displayed products were excluded
 * and a selected look scores decisively below the shortlist best, prefer a
 * smaller honest deck over a substantially worse outfit. Never empties the
 * deck — the top selected look is kept and the limit is disclosed.
 * Exported for tests.
 */
export function applyExclusionQualityFloor(selected, best, exclusionsApplied) {
  if (!exclusionsApplied || !Number.isFinite(best) || !selected.length) return { kept: selected, limited: false };
  const kept = selected.filter((look) => (look.fashion?.score ?? -Infinity) >= best - EXCLUSION_QUALITY_GAP);
  if (kept.length === selected.length) return { kept: selected, limited: false };
  if (!kept.length) {
    const top = selected.reduce((a, b) => (b.fashion?.score ?? -Infinity) > (a.fashion?.score ?? -Infinity) ? b : a);
    return { kept: [top], limited: true };
  }
  return { kept, limited: true };
}

function makeFashionScorer(context, reference, recentSignatures, extras = {}) {
  const contextScores = new WeakMap(), personalScores = new WeakMap(), coherent = new Map();
  const taste = personalTaste(context.preferences);
  const lockedIds = new Set((extras.requiredIds || []).map(String));
  return {
    prepare(item) {
      if (visualAdmissionReason(item) || isNonOutfitItem(item)) return null;
      const prepared = prepareGarment(item.color ? item : { ...item, color: undefined }), descriptor = describeGarment(prepared);
      if (descriptor.slot !== item.macroCategory || descriptor.productKind === 'complete-suit' ||
          garmentContextReason(prepared, context) || rejectForClimate(prepared, context.weather) ||
          rejectForStyleOccasion(prepared, context.style, context.occasion) ||
          rejectForUnsuitableDateNight(descriptor, context)) return null;
      const personal = evaluatePersonalItem(prepared, descriptor, taste);
      // An explicitly locked owned item is a direct user instruction. Brand
      // avoidance must not silently discard that anchor. All other garments
      // respect opted-in brand exclusions before ranking.
      if (personal.blocked && !lockedIds.has(String(prepared.id))) return null;
      contextScores.set(prepared, scoreGarmentForContext(prepared, context));
      personalScores.set(prepared, personal.score);
      return prepared;
    },
    family(item) {
      const d = describeGarment(item);
      return [d.subtype, d.color, d.volume, d.pattern, d.texture].join('|');
    },
    rankItem(item, brief) {
      // Brief-led seeds: the brief's roles decide which items represent this
      // concept; context fit orders within that space. A context-led seed
      // (0.45 for every brief) builds the same pool for every concept and
      // the decks converge on one set of heroes.
      const d = describeGarment(item);
      return (brief.roles[item.macroCategory].includes(normalized(item.subtype)) ||
          brief.roles[item.macroCategory].includes(normalized(d.subtype)) ? 0.55 : 0)
        + 0.30 * contextScores.get(item)
        + (brief.palette.includes(normalized(d.color)) ? 0.15 : 0)
        + 0.70 * (personalScores.get(item) || 0);
    },
    rankLook(items) {
      const identity = JSON.stringify(items.map(({ id }) => id).sort());
      if (coherent.has(identity)) return coherent.get(identity);
      const assessment = outfitCoherence(items, context);
      if (!assessment.valid || outfitStructureReason(items, false) ||
          items.some((item, index) => items.slice(0, index).some((other) => sameProduct(item, other)))) {
        coherent.set(identity, null);
        return null;
      }
      const { dimensions } = assessment;
      // The shared judge exposes competing-pattern evidence before its color
      // pass can replace the palette dimension. Keep that penalty in S6.
      const palette = assessment.evidence.includes('competing_upper_patterns')
        ? Math.min(dimensions.palette, 0.55) : dimensions.palette;
      const unlayered = !items.some((item) => item.macroCategory === 'outerwear');
      // outfitCoherence score is the mean of its 7 dimensions. With no
      // outerwear there is no layer relationship to judge, so where the
      // request permits an unlayered look the neutral prior is excluded from
      // both the mean and the min() — otherwise every three-piece look pays
      // a systematic ~0.05 tax against layered looks before style is weighed.
      const layerApplies = !(unlayered && extras.layerOptional);
      const compatibility = layerApplies
        ? (assessment.score * 7 - dimensions.palette + palette) / 7
        : (assessment.score * 7 - dimensions.layer - dimensions.palette + palette) / 6;
      const weakLink = layerApplies
        ? Math.min(dimensions.layer, dimensions.footwear, dimensions.proportion, palette)
        : Math.min(dimensions.footwear, dimensions.proportion, palette);
      const affinity = referenceAffinity(items, reference);
      const referenceFit = affinity
        ? .55 * affinity.roleFit + .15 * (affinity.paletteFit ?? .5)
          + .15 * (affinity.slotColorFit ?? affinity.paletteFit ?? .5) + .15 * Number(affinity.layerFit) : .5;
      // Unbudgeted requests get a soft, evidence-based premium dampener —
      // never a reject. Budgeted requests already hard-filter in admit().
      const pricing = extras.budgeted || !extras.priceProfile ? { penalty: 0, flagged: 0 }
        : pricePenaltyForLook(items, extras.priceProfile);
      const personal = evaluatePersonalLook(
        items.map((item) => ({ item, descriptor: describeGarment(item) })), taste);
      const result = {
        score: .68 * compatibility + .22 * weakLink + .10 * referenceFit +
          personal.adjustment - recentStylingPenalty(items, recentSignatures) - pricing.penalty,
        dimensions: { ...dimensions, palette },
        evidence: pricing.flagged ? [...assessment.evidence, 'unbudgeted_premium_selection_dampened'] : assessment.evidence,
        referenceMetadata: affinity,
        imageSupportedItems: items.filter((item) =>
          describeGarment(item).evidence.some((entry) => entry.startsWith('image:'))).length,
        personalization: {
          adjustment: personal.adjustment,
          matchedItems: personal.matchedItems,
          matches: personal.matches,
        },
      };
      coherent.set(identity, result);
      return result;
    },
  };
}

function chooseDeck(shortlist, context, excluded = new Set()) {
  const pool = shortlist.filter(({ id }) => !excluded.has(id))
    .sort((a, b) => b.fashion.score - a.fashion.score || a.id.localeCompare(b.id)).slice(0, SEARCH_DECK_LIMIT);
  const features = pool.map(({ items }) => outfitStylingFeatures(items, context.anchorId));
  const pairs = pool.map((a, i) => pool.map((b, j) => {
    const shared = overlaps(a, b);
    if (Math.max(a.items.length, b.items.length) - shared < 2) return null;
    return .18 * stylingFeatureDistance(features[i], features[j]) - .05 * shared;
  }));
  let best = [], utility = -Infinity;
  // Tie-break toward distinct styling concepts: looks assembled from
  // different briefs represent different ideas even when they share a hero
  // item. Bounded well below the coherence scale so it never overrides fit.
  const briefBonus = (indices) => {
    if (indices.length < 2) return 0;
    return 0.025 * (new Set(indices.map((i) => pool[i].briefId)).size - 1);
  };
  const consider = (indices, score) => {
    // Single-currency looks total honestly and shop simply. Per-look bonus
    // so one mixed look cannot hide inside an otherwise consistent deck.
    const total = score + briefBonus(indices)
      + indices.reduce((sum, i) => sum + singleCurrencyBonus(pool[i].items), 0);
    if (indices.length > best.length || (indices.length === best.length && total > utility)) {
      best = indices;
      utility = total;
    }
  };
  for (let i = 0; i < pool.length; i++) {
    consider([i], pool[i].fashion.score);
    for (let j = i + 1; j < pool.length; j++) {
      if (pairs[i][j] === null) continue;
      const two = pool[i].fashion.score + pool[j].fashion.score + pairs[i][j];
      consider([i, j], two);
      for (let k = j + 1; k < pool.length; k++) {
        if (pairs[i][k] === null || pairs[j][k] === null) continue;
        consider([i, j, k], two + pool[k].fashion.score + pairs[i][k] + pairs[j][k]);
      }
    }
  }
  return best.map((index) => pool[index]);
}

export function fashionStylingContext(request) {
  const weather = resolveStylistWeather(request.brief, request.weather);
  const context = stylingContext({ ...request, weather, request: request.brief, optionalLayer: request.optionalLayer === true });
  // The shared context treated an unqualified wedding as casual. Keep an
  // explicit beach/casual wedding or dress code distinct from that default.
  const dressCode = ['formal', 'smart', 'casual', 'athletic'].includes(request.dressCode) ? request.dressCode : null;
  const wedding = /\bwedding\b/i.test(context.occasion);
  const casualWedding = /\b(?:casual|beach)\b/i.test(`${context.occasion} ${context.request}`);
  if (dressCode) return { ...context, dressCode };
  return wedding && !casualWedding ? { ...context, dressCode: 'formal' } : context;
}

export async function recommendFashionDeck({
  request, catalog, referenceBriefs = [], hardGate, priceToBudgetCurrency, visualCritic, canJudgeOutfit,
}) {
  const context = fashionStylingContext(request);
  const weather = context.weather;
  const reference = referenceBriefs.find(({ referenceId }) => referenceId === request.referenceId);
  const forceLayer = !context.optionalLayer || hasColdLayerIntent(weather) || context.dressCode === 'formal';
  const composition = resolveOutfitComposition(context.style, weather, context.request, context.optionalLayer);
  const noLayer = !forceLayer && !composition.layered;
  const requiredIds = [...new Set([...(request.requiredIds || []), ...(context.anchorId ? [context.anchorId] : [])])];
  const adaptLayer = (brief) => {
    if (noLayer) return { ...brief, layerPolicy: 'absent' };
    if (forceLayer && brief.roles.outerwear.length) return { ...brief, layerPolicy: 'required' };
    if (forceLayer) return null;
    return brief;
  };
  const house = outfitConcepts({ ...context, optionalLayer: noLayer }).map((concept) => ({
    id: concept.id, name: concept.title, palette: [...concept.palette], silhouette: concept.silhouette,
    roles: Object.fromEntries(slots.map((slot, index) => [slot, [...(concept.slots[index] || [])].slice(0, 10)])),
    layerPolicy: concept.layerRole === 'absent' ? 'absent' : context.optionalLayer && !forceLayer ? 'optional' : 'required',
  }));
  const references = [reference, ...referenceBriefs.filter((brief) => brief !== reference)].filter(Boolean);
  const briefs = [...house, ...references].map(adaptLayer).filter(Boolean).slice(0, 10);
  if (!briefs.length) return { status: 'no_feasible_looks', outfits: [], diagnostics: { visualVerdict: 'NOT_RUN' } };
  const priceProfile = buildPriceProfile(catalog);
  const retrieval = await shortlistRetrievalDeck({
    request: { ...request, requiredIds }, catalog, hardGate, priceToBudgetCurrency,
    creativeDirector: async () => briefs,
    options: { fashion: makeFashionScorer(context, reference, Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : [], {
      priceProfile, budgeted: !!request.budget, layerOptional: context.optionalLayer === true && !forceLayer,
      requiredIds,
    }), perRole: 8, perBrief: 100, maxVisual: 72 },
  });
  const layerMix = { three: 0, four: 0 };
  for (const candidate of retrieval.shortlist) layerMix[candidate.items.length === 3 ? 'three' : 'four'] += 1;
  const ordered = retrieval.shortlist.map((c) => c.fashion?.score).filter(Number.isFinite).sort((a, b) => b - a);
  let selected = chooseDeck(retrieval.shortlist, context);
  const floor = applyExclusionQualityFloor(
    selected, ordered[0] ?? NaN, (retrieval.diagnostics.exclusions?.requested || 0) > 0);
  selected = floor.kept;
  let visualVerdict = 'NOT_RUN', visualWinner;
  const permitted = async (look) => {
    try { return await canJudgeOutfit(look, reference.referenceId) === true; } catch { return false; }
  };
  if (selected.length >= 1 && reference && typeof visualCritic === 'function' && typeof canJudgeOutfit === 'function' && await permitted(selected[0])) {
    const lead = selected[0];
    const alternatives = retrieval.shortlist.filter((candidate) =>
      !selected.includes(candidate) && Math.abs(candidate.fashion.score - lead.fashion.score) <= TIE_MARGIN &&
      selected.slice(1).every((look) => Math.max(look.items.length, candidate.items.length) - overlaps(look, candidate) >= 2))
      .sort((a, b) => b.fashion.score - a.fashion.score);
    let judged = false;
    for (const alternate of alternatives) {
      if (!await permitted(alternate)) continue;
      const result = await evaluateVisualShortlist({ ...retrieval, shortlist: [lead, alternate] }, request, visualCritic, { target: 1 });
      visualVerdict = result.diagnostics.visualValid === 2 ? 'JUDGED' : 'ABSTAINED';
      const rejected = new Set(result.diagnostics.rejectedCandidateIds);
      if (visualVerdict === 'JUDGED' && result.outfits[0]) {
        visualWinner = result.outfits[0];
        if (visualWinner.id === alternate.id) selected = [alternate, ...selected.slice(1)];
      } else if (rejected.has(lead.id)) {
        selected = chooseDeck(retrieval.shortlist, context, rejected);
      }
      judged = true;
      break;
    }
    if (!judged) {
      // No close alternative: confirm the lead alone in one bounded
      // single-candidate batch. The strongest finalist still gets a real
      // visual verdict (confirm or veto) instead of the critic never
      // running; at most one provider batch is spent here.
      const result = await evaluateVisualShortlist({ ...retrieval, shortlist: [lead] }, request, visualCritic, { target: 1 });
      visualVerdict = result.diagnostics.visualValid === 1 ? 'JUDGED' : 'ABSTAINED';
      const rejected = new Set(result.diagnostics.rejectedCandidateIds);
      if (visualVerdict === 'JUDGED' && result.outfits[0]) {
        visualWinner = result.outfits[0];
      } else if (rejected.has(lead.id)) {
        selected = chooseDeck(retrieval.shortlist, context, rejected);
      }
    }
  }
  return {
    status: selected.length === 3 ? 'complete' : selected.length ? 'partial' : 'no_feasible_looks',
    outfits: selected.map((look) => {
      const { id, title } = lookConcept(look.items, context).concept;
      return {
        id: look.id, items: look.items, description: groundedLookExplanation(look.items, context),
        concept: { id, title, referenceId: reference?.referenceId ?? null },
        fashion: look.fashion,
        ...(visualWinner?.id === look.id ? { visualEvidence: visualWinner.evidence } : {}),
      };
    }),
    diagnostics: { ...retrieval.diagnostics, visualVerdict,
      shortlistMix: { ...layerMix, best: ordered[0] ?? null, top20cutoff: ordered[19] ?? null },
      ...(floor.limited ? { exclusionQualityLimited: true } : {}) },
  };
}
