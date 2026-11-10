// Experimental catalog/wardrobe recommendation path. Callers own inventory,
// permission and provider admission. Local styling never requires image upload.
import { shortlistRetrievalDeck, evaluateVisualShortlist } from './prototype.mjs';
import { baseProductId } from './catalog_adapter.mjs';
import { describeGarment, prepareGarment } from '../../src/domain/garmentDescriptor.ts';
import { currentVisual, visualAdmissionReason } from '../../src/domain/visualFashion.ts';
import { outfitCoherence } from '../../src/domain/outfitCoherence.ts';
import { garmentContextReason, scoreGarmentForContext } from '../../src/domain/garmentContext.ts';
import { lookConcept, outfitConcepts } from '../../src/domain/outfitConcepts.ts';
import { stylingContext } from '../../src/domain/stylingContext.ts';
import { hasColdLayerIntent, resolveOutfitComposition, resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { outfitStylingFeatures, recentStylingPenalty, stylingFeatureDistance } from '../../src/domain/outfitVisualSignature.ts';
import { colorTokensOf, patternOf } from '../../src/domain/outfitDiversity.ts';
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

function makeFashionScorer(context, reference, recentSignatures) {
  const contextScores = new WeakMap(), coherent = new Map();
  const likedPatterns = new Set((context.preferences?.likedPatterns || []).map((pattern) => patternOf({ pattern })));
  return {
    prepare(item) {
      if (visualAdmissionReason(item) || isNonOutfitItem(item)) return null;
      const prepared = prepareGarment(item.color ? item : { ...item, color: undefined }), descriptor = describeGarment(prepared);
      if (descriptor.slot !== item.macroCategory || descriptor.productKind === 'complete-suit' ||
          garmentContextReason(prepared, context) || rejectForClimate(prepared, context.weather) ||
          rejectForStyleOccasion(prepared, context.style, context.occasion)) return null;
      contextScores.set(prepared, scoreGarmentForContext(prepared, context) + (known(descriptor.pattern) && likedPatterns.has(descriptor.pattern) ? .035 : 0));
      return prepared;
    },
    family(item) {
      const d = describeGarment(item);
      return [d.subtype, d.color, d.volume, d.pattern, d.texture].join('|');
    },
    rankItem(item, brief) {
      const d = describeGarment(item);
      return (brief.roles[item.macroCategory].includes(normalized(item.subtype)) ||
          brief.roles[item.macroCategory].includes(normalized(d.subtype)) ? 0.40 : 0)
        + 0.45 * contextScores.get(item)
        + (brief.palette.includes(normalized(d.color)) ? 0.15 : 0);
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
      const compatibility = (assessment.score * 7 - dimensions.palette + palette) / 7;
      const weakLink = Math.min(dimensions.layer, dimensions.footwear, dimensions.proportion, palette);
      const affinity = referenceAffinity(items, reference);
      const referenceFit = affinity
        ? .55 * affinity.roleFit + .15 * (affinity.paletteFit ?? .5)
          + .15 * (affinity.slotColorFit ?? affinity.paletteFit ?? .5) + .15 * Number(affinity.layerFit) : .5;
      const result = {
        score: .68 * compatibility + .22 * weakLink + .10 * referenceFit - recentStylingPenalty(items, recentSignatures),
        dimensions: { ...dimensions, palette },
        evidence: assessment.evidence,
        referenceMetadata: affinity,
        imageSupportedItems: items.filter((item) =>
          describeGarment(item).evidence.some((entry) => entry.startsWith('image:'))).length,
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
    return .12 * stylingFeatureDistance(features[i], features[j]) - .035 * shared;
  }));
  let best = [], utility = -Infinity;
  const consider = (indices, score) => {
    if (indices.length > best.length || (indices.length === best.length && score > utility)) {
      best = indices;
      utility = score;
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
  const retrieval = await shortlistRetrievalDeck({
    request: { ...request, requiredIds }, catalog, hardGate, priceToBudgetCurrency,
    creativeDirector: async () => briefs,
    options: { fashion: makeFashionScorer(context, reference, Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : []), perRole: 8, perBrief: 100, maxVisual: 72 },
  });
  let selected = chooseDeck(retrieval.shortlist, context), visualVerdict = 'NOT_RUN', visualWinner;
  const permitted = async (look) => {
    try { return await canJudgeOutfit(look, reference.referenceId) === true; } catch { return false; }
  };
  if (selected.length === 3 && reference && typeof visualCritic === 'function' && typeof canJudgeOutfit === 'function' && await permitted(selected[0])) {
    const lead = selected[0];
    const alternatives = retrieval.shortlist.filter((candidate) =>
      !selected.includes(candidate) && Math.abs(candidate.fashion.score - lead.fashion.score) <= TIE_MARGIN &&
      selected.slice(1).every((look) => Math.max(look.items.length, candidate.items.length) - overlaps(look, candidate) >= 2))
      .sort((a, b) => b.fashion.score - a.fashion.score);
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
      break;
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
    diagnostics: { ...retrieval.diagnostics, visualVerdict },
  };
}
