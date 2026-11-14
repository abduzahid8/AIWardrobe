// Experimental catalog/wardrobe recommendation path. Callers own inventory,
// permission and provider admission. Local styling never requires image upload.
import { shortlistRetrievalDeck, evaluateVisualShortlist } from './prototype.mjs';
import { baseProductId } from './catalog_adapter.mjs';
import { describeGarment, prepareGarment } from '../../src/domain/garmentDescriptor.ts';
import { currentVisual, visualAdmissionReason } from '../../src/domain/visualFashion.ts';
import { outfitCoherence } from '../../src/domain/outfitCoherence.ts';
import { garmentContextReason, scoreGarmentForContext } from '../../src/domain/garmentContext.ts';
import { outfitConcepts } from '../../src/domain/outfitConcepts.ts';
import { isDateNightOccasion, stylingContext } from '../../src/domain/stylingContext.ts';
import { hasColdLayerIntent, resolveOutfitComposition, resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { outfitStylingFeatures, recentStylingPenalty, stylingFeatureDistance } from '../../src/domain/outfitVisualSignature.ts';
import { colorTokensOf } from '../../src/domain/outfitDiversity.ts';
import { personalTaste, evaluatePersonalItem, evaluatePersonalLook } from './personal_taste.mjs';
import {
  FEEDBACK_TASTE_VERSION, evaluateFeedbackItem, evaluateFeedbackLook,
} from './feedback_taste.mjs';
import { DIRECTION_VERSION, directionDifference, selectCreativeDirectionDeck, labelCreativeDirections } from './creative_directions.mjs';
import { EDITORIAL_QUALITY_VERSION, evaluateEditorialQuality } from './editorial_quality.mjs';
import { isNonOutfitItem, outfitStructureReason } from '../../src/domain/outfitStructure.ts';
import { rejectForClimate, rejectForStyleOccasion } from '../../src/domain/occasionRules.ts';
import { editorialDeckCopy, DECK_COPY_VERSION } from './deck_editorial_copy.mjs';
import { editorialRegisterVeto, EDITORIAL_REGISTER_VERSION } from './editorial_register.mjs';

const slots = ['top', 'outerwear', 'bottom', 'shoes'];
const normalized = (value) => String(value || '').toLowerCase().replace(/[\s-]+/g, '_');
const known = (value) => !!value && value !== 'unknown' && value !== 'neutral';
const image = (item) => item.imageUrl || item.image_url;
const sameProduct = (a, b) => baseProductId(a.id) === baseProductId(b.id) ||
  (!!image(a) && image(a) === image(b)) ||
  (!!currentVisual(a)?.imageSha256 && currentVisual(a).imageSha256 === currentVisual(b)?.imageSha256);
const overlaps = (a, b) => a.items.filter((item) => b.items.some((other) => sameProduct(item, other))).length;
const TIE_MARGIN = 0.025;
// The hosted critic receives only one spare slot after the three intended
// directions. The fourth must be a *different viable fashion direction*,
// not just the most similarly scored clone of the lead. Keep it inside the
// same .16 deterministic quality window used for creative-direction planning.
const VISUAL_CHALLENGER_QUALITY_WINDOW = 0.16;
// A selected look this far below the shortlist best is decisively off the
// pace (10x the near-tie margin), not a close alternative worth showing.
const EXCLUSION_QUALITY_GAP = 0.25;
// Minimum per-currency samples before an unbudgeted premium dampener may
// apply. Below this there is no evidence for what "expensive" means here.
const PRICE_SAMPLE_MINIMUM = 8;
// Explicit thumbs-feedback retrieval seed. The saturated per-item tally from
// feedback_taste.mjs (±1) is weighted like the strongest single personal-
// taste item seed (0.70 * personal cap 0.12 ≈ 0.08): enough to reorder
// near-tie candidates when alternatives exist, never a block or a gate — a
// disliked item can still appear when the inventory genuinely cannot supply
// another eligible piece.
const FEEDBACK_ITEM_SEED_WEIGHT = 0.08;

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
 * Within [0, 0.06], ultra-luxury multiples (>2x median) continue scaling
 * logarithmically toward 10x instead of hitting a flat plateau at 2x.
 * It breaks near-ties toward representative prices but never rejects —
 * without a budget there is nothing to fail. A look is flagged only when an
 * item costs more than twice its slot median. Exported for tests.
 */
export function pricePenaltyForLook(items, profile) {
  let penalty = 0, flagged = 0;
  for (const item of items || []) {
    const band = item && profile ? profile[`${item.currency}|${item.macroCategory}`] : null;
    if (!band || !(item.price > band.median)) continue;
    const ratio = item.price / band.median;
    if (ratio <= 2) {
      penalty += 0.03 * (ratio - 1);
    } else {
      flagged += 1;
      penalty += 0.03 + 0.03 * Math.min(1, Math.log10(ratio / 2) / Math.log10(5));
    }
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
 * Date-night register for NEW CLOTHING. "Discovery" may explore silhouettes
 * but cannot turn a smart date into resort / training wear. The shared visual
 * descriptor currently treats any "leather trainers" as minimal sneakers;
 * an explicit retro/training silhouette is still athletic/casual evidence.
 * Only scope these exclusions to smart/formal date nights. Casual occasions,
 * daywear and an explicit casual dress code retain their original freedom.
 */
export function rejectForUnsuitableDateNight(descriptor, context, item) {
  if (context.dressCode !== 'smart' && context.dressCode !== 'formal') return false;
  if (!isDateNightOccasion(context.occasion) && !isDateNightOccasion(context.request)) return false;
  const name = String(item?.name || '').toLowerCase();
  if (descriptor.slot === 'bottom') {
    // Feed taxonomy can label "Blue denim shorts" as jeans because denim
    // matches before shorts. Physical short inseams remain inappropriate
    // regardless of that imperfect subtype ordering.
    return descriptor.subtype === 'shorts' || descriptor.athletic === true ||
      /\b(shorts?|bermudas?)\b/.test(name);
  }
  if (descriptor.slot === 'top') {
    // A known resort towelling tee is daytime leisure, not an elevated date
    // base. Do not prohibit every cotton T-shirt or every linen shirt.
    return descriptor.subtype === 't-shirt' &&
      /\b(resort|towell?ing|terry|beachwear)\b/.test(name);
  }
  if (descriptor.slot === 'shoes') {
    if (descriptor.athletic || descriptor.subtype === 'sport-sneaker') return true;
    if (/\b(retro|running|training|trail|basketball|skate)\b/.test(name) &&
      /\b(sneakers?|trainers?)\b/.test(name)) return true;
    // "Leather" alone is not evidence of a minimal shoe. A deliberately
    // sleek low-top remains eligible, subject to whole-look suitability.
    if (['sneaker', 'minimal-sneaker'].includes(descriptor.subtype) &&
      !/\b(minimal|clean|plain|sleek|low[ -]?top)\b/.test(name)) return true;
  }
  return false;
}

/**
 * Catalog-only quiet-smart layer guard. A known leisure/technical sport shell
 * has a different visual register from classic office and old-money evening
 * tailoring. The name must explicitly identify the leisure/training purpose:
 * technical fabric alone is not evidence of an unsuitable garment.
 *
 * This is not a global jacket ban: casual/travel, user-directed sportswear,
 * wardrobe items, and normal minimalist layering are unaffected.
 */
export function rejectQuietSmartLeisureLayer(descriptor, context, item) {
  if (context.sourceMode !== 'catalog' ||
      !['classic','old_money','semi_classic'].includes(context.style) ||
      !['smart','formal'].includes(context.dressCode) ||
      descriptor.slot !== 'outerwear' ||
      !/\b(work|office|business|date|dinner|evening)\b/i.test(
        `${context.occasion || ''} ${context.request || ''}`)) return false;
  // A user explicitly choosing sporty outerwear outranks stylist aesthetics,
  // without waiving the existing physical safety or budget gates.
  if (/\b(technical leisure|sporty jacket|track jacket|training jacket|reversible resort jacket)\b/i
    .test(context.request || '')) return false;
  const name = String(item?.name || '');
  return /\b(technical[ -]?leisure|track jacket|training jacket|running jacket|sport[s]? jacket)\b/i.test(name) ||
    // Owner iOS screenshot: a bright zip-front REVER-SIBLE RESORT shell
    // replaced tailoring in a semi-classic romantic-date outfit. Both
    // words must be explicit SKU evidence: an ordinary resort blazer,
    // reversible *tailored* jacket or plain cotton jacket is NOT banned.
    /\breversible\s+resort\s+(?:jacket|blouson)\b/i.test(name);
}

/**
 * Quiet-style fit is a product-level gate for a smart outfit, not a blanket
 * ban on denim or comfort shoes. These are identifiable mismatches from
 * actual catalog metadata/validated descriptors, before shortlist ranking:
 *
 *  - Old Money date night should not default to jeans, unless requested.
 *  - Classic/Semi-Classic/Old Money smart work and Old Money smart dates
 *    should not default to a visibly athletic, chunky or rugged sole.
 *
 * Casual, travel, athletic, user-requested items, and all owned wardrobe
 * modes are outside this catalog-only editorial rule.
 */
export function rejectPolishedContextMismatch(descriptor, context, item) {
  if (context.sourceMode !== 'catalog' || context.dressCode !== 'smart') return false;
  const quiet = ['classic', 'semi_classic', 'old_money'].includes(context.style);
  if (!quiet) return false;
  const dateNight = isDateNightOccasion(context.occasion);
  const work = /\b(work|office|business)\b/i.test(String(context.occasion || ''));
  const userBrief = String(context.request || '');
  if (context.style === 'old_money' && dateNight && descriptor.slot === 'bottom' &&
      descriptor.subtype === 'jeans' && !/\b(jeans?|denim)\b/i.test(userBrief)) return true;
  if (descriptor.slot !== 'shoes' || (!work && !dateNight)) return false;
  if (/\b(chunky|lug[ -]?sole|rugged|sport[ -]?sole|comfort[ -]?sole)\b/i.test(userBrief))
    return false;
  // No rejection on unknown soles: never infer actual shoe construction
  // from leather, price, brand or store alone.
  return descriptor.casualSole === true;
}

/** A smart date outfit needs polish even when every individual SKU is valid. */
export function rejectUnpolishedDateNightLook(items, context) {
  if (context.dressCode !== 'smart' && context.dressCode !== 'formal') return false;
  if (!isDateNightOccasion(context.occasion) && !isDateNightOccasion(context.request)) return false;
  const top = items.find(item => item.macroCategory === 'top');
  const shoe = items.find(item => item.macroCategory === 'shoes');
  const hasLayer = items.some(item => item.macroCategory === 'outerwear');
  if (!top || !shoe || hasLayer) return false;
  const base = describeGarment(top), footwear = describeGarment(shoe);
  return base.subtype === 't-shirt' &&
    ['sneaker','minimal-sneaker','sport-sneaker'].includes(footwear.subtype);
}

/**
 * Footwear-register tie-back for elevated occasions (bounded, never a gate).
 *
 * Deck reviews repeatedly show trainer-only footwear losing the shoe-register
 * read at quiet styles worn to dressed occasions, even though equally
 * coherent dress shoes exist in the same catalog. The elevated register is a
 * property of the PAIR: a quiet/elevated style (constant table below) at an
 * occasion whose register expects shoes (constant table below — the
 * smart-work set the shared context already recognises, plus the conference /
 * gallery / dinner evening occasions from the reviews). Casual styles,
 * travel, weekends and athletic contexts are deliberately untouched, and
 * warm-weather layering behavior is out of scope here.
 *
 * The penalty is a tiny bounded score adjustment (FOOTWEAR_REGISTER_PENALTY,
 * about half the 0.025 near-tie margin): it reorders near-ties toward dress
 * shoes, never rejects, and is skipped entirely unless the retrieval's
 * admitted shoe pool demonstrably contains a non-trainer option that passed
 * every hard gate for this context — when the only feasible shoes are
 * trainers, nothing fires. A sleek minimal/low-top sneaker keeps the quiet
 * register, mirroring the documented smart date-night boundary.
 */
export const FOOTWEAR_REGISTER_PENALTY = 0.024;
const ELEVATED_REGISTER_STYLES = ['old_money', 'classic', 'semi_classic', 'business_casual', 'smart_casual'];
const ELEVATED_REGISTER_OCCASIONS =
  /\b(conference|gallery|exhibition|museum|dinner|date(?:[\s_-]?night)?|cocktail|opera|theat(?:er|re)|wedding|business|meeting|work|office)\b/i;
const SLEEK_SNEAKER = /\b(minimal|clean|plain|sleek|low[ -]?top)\b/i;

export function elevatedFootwearRegisterContext(context) {
  return ELEVATED_REGISTER_STYLES.includes(context?.style) &&
    ELEVATED_REGISTER_OCCASIONS.test(String(context?.occasion || ''));
}

export function trainerFootwearDescriptor(descriptor, item) {
  if (descriptor?.slot !== 'shoes') return false;
  if (descriptor.athletic === true ||
      ['sport-sneaker', 'sneaker'].includes(descriptor.subtype)) return true;
  // The shared descriptor classifies any "leather trainers" listing as a
  // minimal sneaker; without minimal/clean/low-top evidence that listing is
  // still a generic trainer (same boundary as the smart date-night gate).
  return descriptor.subtype === 'minimal-sneaker' &&
    !SLEEK_SNEAKER.test(String(item?.name || ''));
}

export function footwearRegisterPenaltyForLook(items, descriptors, context, nonTrainerShoeFeasible) {
  if (nonTrainerShoeFeasible !== true || !elevatedFootwearRegisterContext(context)) return 0;
  const shoes = (descriptors || []).flatMap((descriptor, index) =>
    descriptor?.slot === 'shoes' && items?.[index]?.macroCategory === 'shoes'
      ? [[descriptor, items[index]]] : []);
  return shoes.length > 0 &&
    shoes.every(([descriptor, item]) => trainerFootwearDescriptor(descriptor, item))
    ? FOOTWEAR_REGISTER_PENALTY : 0;
}

/**
 * Bounded polish tie-break for elevated catalog contexts: henleys in classic/old-money
 * business tailoring and sleeveless tank tops as a primary top on elevated occasions
 * yield to collared shirts, fine knits, or knit polos unless user-requested.
 */
export const ELEVATED_BASE_POLISH_PENALTY = 0.045;
export function elevatedBasePolishPenaltyForLook(items, descriptors, context) {
  if (context?.sourceMode !== 'catalog' || !context?.style || !context?.occasion) return 0;
  const occasion = String(context.occasion || '');
  const elevatedOccasion = ELEVATED_REGISTER_OCCASIONS.test(occasion);
  if (!elevatedOccasion) return 0;
  const elevatedStyle = ELEVATED_REGISTER_STYLES.includes(context.style) || context.style === 'minimalist';
  if (!elevatedStyle) return 0;
  const userReq = String(context?.request || '');
  const tops = (items || []).filter((item) => item?.macroCategory === 'top');
  for (const top of tops) {
    const name = String(top?.name || '');
    if (/\btank[\s-]*top\b/i.test(name) && !/\btank\b/i.test(userReq)) {
      return ELEVATED_BASE_POLISH_PENALTY;
    }
    const formalStyle = ['old_money', 'classic', 'business_casual', 'semi_classic'].includes(context.style);
    if (formalStyle && /\bhenley\b/i.test(name) && !/\bhenley\b/i.test(userReq) &&
        /\b(work|office|business|meeting|conference|wedding|dinner|gallery)\b/i.test(occasion)) {
      return ELEVATED_BASE_POLISH_PENALTY;
    }
  }
  const hasEspadrille = (descriptors || []).some((d, idx) =>
    d?.slot === 'shoes' && items?.[idx]?.macroCategory === 'shoes' &&
    (d.subtype === 'espadrille' || d.subtype === 'boat-shoe'));
  if (hasEspadrille && !/\b(espadrilles?|boat)\b/i.test(userReq) &&
      /\b(work|office|business|meeting|conference|dinner|date(?:[\s_-]?night)?|cocktail|opera|theat(?:er|re)|wedding|gallery)\b/i.test(occasion)) {
    return ELEVATED_BASE_POLISH_PENALTY;
  }
  return 0;
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
 * Merchant-offer price freshness honesty (gate F). A stale or untimestamped
 * merchant offer must not outrank an equally coherent candidate whose price
 * was verifiably seen recently, but staleness is evidence about the OFFER,
 * never about the garment: the penalty is bounded at 0.035 total — about one
 * tie margin beyond the 0.025 near-tie threshold and ~5% of the 0.68
 * coherence weight — so it reorders near-ties toward verifiable recency and
 * can never rescue an incoherent look or override any hard gate.
 *
 * Per selected item: stale 0.0125, unknown 0.007, fresh/recent 0, summed
 * then capped at PRICE_FRESHNESS_PENALTY_MAX. Only shoppable pieces pay:
 * owned wardrobe items carry no merchant offer and no staleness claim.
 * Items that predate the freshness contract (no `priceFreshness` field) pay
 * nothing — the penalty is never invented from missing data the producer
 * never claimed. Exported for tests.
 */
export const PRICE_FRESHNESS_PENALTY_MAX = 0.035;
const PRICE_FRESHNESS_TIER_PENALTY = { fresh: 0, recent: 0, stale: 0.0125, unknown: 0.007 };

export function priceFreshnessPenaltyForLook(items) {
  let penalty = 0;
  for (const item of items || []) {
    if (item?.isShopItem !== true) continue;
    penalty += PRICE_FRESHNESS_TIER_PENALTY[item?.priceFreshness] || 0;
  }
  return Math.min(PRICE_FRESHNESS_PENALTY_MAX, penalty);
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
  const feedbackScores = new WeakMap();
  const taste = personalTaste(context.preferences);
  const feedbackTaste = extras.feedbackTaste?.active ? extras.feedbackTaste : null;
  const lockedIds = new Set((extras.requiredIds || []).map(String));
  // Evidence about this retrieval's admitted shoe pool, collected during
  // prepare() before any look is ranked: the footwear-register tie-back may
  // only fire when at least one non-trainer shoe survived every admission
  // gate for this context, so a trainer-only feasible pool pays nothing.
  const shoePool = { trainer: 0, nonTrainer: 0 };
  return {
    prepare(item) {
      if (visualAdmissionReason(item) || isNonOutfitItem(item)) return null;
      const prepared = prepareGarment(item.color ? item : { ...item, color: undefined }), descriptor = describeGarment(prepared);
      if (descriptor.slot !== item.macroCategory || descriptor.productKind === 'complete-suit' ||
          garmentContextReason(prepared, context) || rejectForClimate(prepared, context.weather) ||
          rejectForStyleOccasion(prepared, context.style, context.occasion) ||
          rejectForUnsuitableDateNight(descriptor, context, prepared) ||
          rejectQuietSmartLeisureLayer(descriptor, context, prepared) ||
          rejectPolishedContextMismatch(descriptor, context, prepared)) return null;
      if (descriptor.slot === 'shoes') {
        shoePool[trainerFootwearDescriptor(descriptor, prepared) ? 'trainer' : 'nonTrainer'] += 1;
      }
      const personal = evaluatePersonalItem(prepared, descriptor, taste);
      const feedback = evaluateFeedbackItem(prepared, feedbackTaste);
      // An explicitly locked owned item is a direct user instruction. Brand
      // avoidance must not silently discard that anchor. All other garments
      // respect opted-in brand exclusions before ranking.
      if (personal.blocked && !lockedIds.has(String(prepared.id))) return null;
      contextScores.set(prepared, scoreGarmentForContext(prepared, context));
      personalScores.set(prepared, personal.score);
      feedbackScores.set(prepared, feedback.score);
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
        + 0.70 * (personalScores.get(item) || 0)
        + FEEDBACK_ITEM_SEED_WEIGHT * (feedbackScores.get(item) || 0);
    },
    rankLook(items, brief) {
      const identity = JSON.stringify(items.map(({ id }) => id).sort());
      if (coherent.has(identity)) return coherent.get(identity);
      const assessment = outfitCoherence(items, context);
      if (!assessment.valid || rejectUnpolishedDateNightLook(items, context) || outfitStructureReason(items, false) ||
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
      // Bounded merchant-offer staleness dampener (gate F): reorders near-ties
      // toward verifiably recent price evidence, never a gate.
      const freshness = priceFreshnessPenaltyForLook(items);
      const descriptions = items.map(describeGarment);
      // Elevated-register footwear tie-back: a tiny, evidence-checked nudge
      // toward dress shoes at quiet styles on dressed occasions, only when
      // non-trainer shoes demonstrably passed admission for this context.
      const registerPenalty = footwearRegisterPenaltyForLook(
        items, descriptions, context, shoePool.nonTrainer > 0);
      const basePolishPenalty = elevatedBasePolishPenaltyForLook(items, descriptions, context);
      const personal = evaluatePersonalLook(
        items.map((item, index) => ({ item, descriptor: descriptions[index] })), taste);
      // Explicit-feedback taste (gate D): a capped, decayed adjustment from
      // this user's own thumbs rows, applied AFTER every hard gate. The
      // candidate's generating brief id is the native concept slug the deck
      // displays, which is exactly the concept id feedback is written under.
      const feedback = evaluateFeedbackLook(
        items, brief?.id, feedbackTaste,
        // The client already persists this versioned semantic five-field
        // vector for every rated outfit. Transfer an explicit preference to
        // similarly styled replacement SKUs instead of overfitting to ids.
        feedbackTaste?.active ? outfitStylingFeatures(items, context.anchorId) : undefined,
      );
      // This phase is explicitly for New Clothing. Do not perturb owned
      // wardrobe or mixed-source ranking until their source-mix and anchor
      // acceptance gates have dedicated editorial calibration.
      const editorial = context.sourceMode === 'catalog'
        ? evaluateEditorialQuality({
            dimensions: { ...dimensions, palette },
            descriptors: descriptions,
            context,
            hasLayer: !unlayered,
          })
        : { score: null, adjustment: 0, evidence: [], version: EDITORIAL_QUALITY_VERSION };
      const result = {
        score: .68 * compatibility + .22 * weakLink + .10 * referenceFit +
          personal.adjustment + editorial.adjustment + feedback.adjustment -
          recentStylingPenalty(items, recentSignatures) - pricing.penalty - freshness -
          registerPenalty - basePolishPenalty,
        dimensions: { ...dimensions, palette },
        evidence: pricing.flagged || freshness > 0 || registerPenalty > 0 || basePolishPenalty > 0
          ? [...assessment.evidence,
            ...(pricing.flagged ? ['unbudgeted_premium_selection_dampened'] : []),
            ...(freshness > 0 ? ['stale_price_evidence_dampened'] : []),
            ...(registerPenalty > 0 ? ['trainer_footwear_register_dampened'] : []),
            ...(basePolishPenalty > 0 ? ['casual_base_elevated_context_dampened'] : [])]
          : assessment.evidence,
        referenceMetadata: affinity,
        editorial,
        imageSupportedItems: items.filter((item) =>
          describeGarment(item).evidence.some((entry) => entry.startsWith('image:'))).length,
        personalization: {
          adjustment: personal.adjustment,
          matchedItems: personal.matchedItems,
          matches: personal.matches,
          // Present only when explicit feedback actually influences this look.
          ...(feedback.adjustment !== 0 || feedback.matches.length ? {
            feedback: {
              version: FEEDBACK_TASTE_VERSION,
              adjustment: feedback.adjustment,
              likedItems: feedback.likedItems,
              dislikedItems: feedback.dislikedItems,
              matches: feedback.matches,
            },
          } : {}),
        },
      };
      coherent.set(identity, result);
      return result;
    },
  };
}

// Keep the exact same budget / coherence / identity-admitted candidates.
// Selection, rather than cosmetic labels, now owns three styling directions.
function creativeMetrics(context) {
  return {
    featuresOf: (look) => outfitStylingFeatures(look.items, context.anchorId),
    featureDistance: stylingFeatureDistance,
    overlap: overlaps,
    singleCurrencyBonus,
    requiredIds: context.requiredIds || [],
  };
}

function chooseDeck(shortlist, context, excluded = new Set()) {
  return selectCreativeDirectionDeck(
    shortlist.filter(({ id }) => !excluded.has(id)),
    creativeMetrics(context),
  ).looks;
}

/**
 * One gentle step beyond the quiz choice, no extra onboarding questions.
 * Neighbor concepts are suggestions, never independent style overrides:
 * every SKU still passes the ORIGINAL requested style, occasion, dress code,
 * weather, and budget hard gates.
 */
export function adjacentCreativeStyles(style, dressCode) {
  if (dressCode === 'formal' || dressCode === 'athletic') return [];
  const neighbors = {
    old_money: ['semi_classic', 'minimalist'],
    classic: ['semi_classic', 'minimalist'],
    semi_classic: ['old_money', 'minimalist'],
    minimalist: ['semi_classic', 'casual'],
    casual: ['minimalist', 'semi_classic'],
    business_casual: ['minimalist', 'old_money'],
    smart_casual: ['semi_classic', 'minimalist'],
  };
  return neighbors[style] || [];
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
  // Optional prebuilt model from feedback_taste.mjs (explicit thumbs rows).
  // Absent/inert model = byte-identical legacy ranking (back-compat).
  feedbackTaste,
  // Cloud staging can opt into reviewing all three displayed looks in one
  // bounded provider batch. The isolated local / offline paths stay unchanged.
  visualDeckReview = false,
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
  const neighborHouse = adjacentCreativeStyles(context.style, context.dressCode)
    .flatMap((style) => outfitConcepts({ ...context, style, optionalLayer: noLayer })
      .slice(0, 1).map((concept) => ({
        id: `adjacent-${style}-${concept.id}`,
        name: concept.title,
        palette: [...concept.palette],
        silhouette: concept.silhouette,
        roles: Object.fromEntries(slots.map((slot,index) =>
          [slot,[...(concept.slots[index] || [])].slice(0,10)])),
        layerPolicy: concept.layerRole === 'absent' ? 'absent'
          : context.optionalLayer && !forceLayer ? 'optional' : 'required',
      })));
  const references = [reference, ...referenceBriefs.filter((brief) => brief !== reference)].filter(Boolean);
  const briefs = [...house, ...neighborHouse, ...references].map(adaptLayer).filter(Boolean).slice(0, 10);
  if (!briefs.length) return { status: 'no_feasible_looks', outfits: [], diagnostics: { visualVerdict: 'NOT_RUN' } };
  const priceProfile = buildPriceProfile(catalog);
  const retrieval = await shortlistRetrievalDeck({
    request: { ...request, requiredIds }, catalog, hardGate, priceToBudgetCurrency,
    creativeDirector: async () => briefs,
    options: { fashion: makeFashionScorer(context, reference, Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : [], {
      priceProfile, budgeted: !!request.budget, layerOptional: context.optionalLayer === true && !forceLayer,
      requiredIds, feedbackTaste,
    }), perRole: 8, perBrief: 100, maxVisual: 72 },
  });
  // A single cross-slot veto applies to both selection and hosted review.
  // Never swap a pinned user item without asking; show fewer looks instead.
  const editorialVetoes = {};
  const selectable = retrieval.shortlist.filter((look) => {
    const reason = editorialRegisterVeto(look, context);
    if (!reason) return true;
    editorialVetoes[reason] = (editorialVetoes[reason] || 0) + 1;
    return false;
  });
  const layerMix = { three: 0, four: 0 };
  for (const candidate of selectable) layerMix[candidate.items.length === 3 ? 'three' : 'four'] += 1;
  const ordered = selectable.map((c) => c.fashion?.score).filter(Number.isFinite).sort((a, b) => b - a);
  const deckContext = { ...context, requiredIds };
  let selected = chooseDeck(selectable, deckContext);
  const floor = applyExclusionQualityFloor(
    selected, ordered[0] ?? NaN, (retrieval.diagnostics.exclusions?.requested || 0) > 0);
  selected = floor.kept;
  let visualVerdict = 'NOT_RUN', visualWinner;
  const visualEvidenceById = new Map();
  let visualReviewCoverage = null;
  // Reference-optional review: a deck with no matching reference brief is
  // still visually judged on its product images alone (the critic marks such
  // verdicts referenceless and never claims a reference comparison). With a
  // reference, behavior is unchanged.
  const permitted = async (look) => {
    try { return await canJudgeOutfit(look, reference?.referenceId ?? null) === true; } catch { return false; }
  };
  if (visualDeckReview && selected.length && typeof visualCritic === 'function' &&
      typeof canJudgeOutfit === 'function') {
    // Budgeted, source-verified finalist set: no model can nominate a SKU or
    // skip the original hard gates. Judge all authorised displayed looks, not
    // only the lead; fill remaining slots up to the 4-call cap so a single
    // visual rejection or unresolved CDN host still yields a full 3-look deck.
    const candidates = [];
    for (const look of selected) {
      if (await permitted(look)) candidates.push(look);
    }
    const lead = selected[0];
    if (candidates.length < 4) {
      const metrics = creativeMetrics(deckContext);
      const leadScore = lead.fashion?.score ?? -Infinity;
      const extras = selectable
        .filter(look => !candidates.some(chosen => chosen.id === look.id) &&
          Number.isFinite(look.fashion?.score) &&
          look.fashion.score >= leadScore - VISUAL_CHALLENGER_QUALITY_WINDOW &&
          (candidates.length ? candidates : selected).every(other =>
            Math.max(other.items.length, look.items.length) - overlaps(other, look) >= 2))
        .map(look => ({
          look,
          difference: Math.min(...(candidates.length ? candidates : selected).map(other =>
            directionDifference(other, look, metrics.featuresOf, metrics.featureDistance).distance)),
        }))
        .sort((a, b) => b.difference - a.difference ||
          b.look.fashion.score - a.look.fashion.score ||
          String(a.look.id).localeCompare(String(b.look.id)))
        .map(({ look }) => look);
      for (const alternate of extras) {
        if (candidates.length >= 4) break;
        if (await permitted(alternate)) candidates.push(alternate);
      }
    }
    if (candidates.length) {
      const reviewed = await evaluateVisualShortlist(
        { ...retrieval, shortlist: candidates }, request, visualCritic,
        { target: Math.min(3, candidates.length), minNovelty: 0.12, requiredIds, requireDistinctArchitectures: true },
      );
      visualVerdict = reviewed.diagnostics.visualValid > 0 ? 'JUDGED' : 'ABSTAINED';
      const rejected = new Set(reviewed.diagnostics.rejectedCandidateIds);
      // The hosted critic has already applied visual quality, confidence,
      // explicit veto and cross-look novelty gates to its <=4 candidates.
      // Its approved *deck*, not just its lead, is the authoritative result.
      // Previously the fashion caller ignored that verdict deck and showed
      // original items (or a fresh, unjudged replacement after a veto).
      // That made NVIDIA_APPLIED true while the user's actual third look
      // could have failed visual evaluation. Never launder that evidence.
      //
      // A successful partial review yields an honestly partial deck. If all
      // returned verdicts fail the visual quality gate, return no looks:
      // server reports quality failure rather than reinstating an unsafe
      // deterministic outfit. If the provider returns no valid verdict at
      // all, retain deterministic recommendations with ABSTAINED status,
      // so a media/network outage does not become a false fashion veto.
      const byId = new Map(candidates.map(look => [look.id, look]));
      for (const approved of reviewed.outfits) {
        if (Array.isArray(approved.evidence)) {
          visualEvidenceById.set(approved.id, approved.evidence);
        }
      }
      if (reviewed.diagnostics.visualValid > 0) {
        selected = reviewed.outfits.map(({ id }) => byId.get(id)).filter(Boolean);
      }
      visualReviewCoverage = {
        mode: 'displayed_deck',
        attempted: reviewed.diagnostics.visualAttempted,
        valid: reviewed.diagnostics.visualValid,
        rejected: rejected.size,
        approved: reviewed.diagnostics.visualApproved,
        // Only an approved, actually displayed look has review evidence.
        displayedEvidence: selected.filter(look => visualEvidenceById.has(look.id)).length,
        fallbackUnreviewed: reviewed.diagnostics.visualValid === 0,
      };
    }
  } else if (selected.length >= 1 && typeof visualCritic === 'function' && typeof canJudgeOutfit === 'function' && await permitted(selected[0])) {
    const lead = selected[0];
    const alternatives = selectable.filter((candidate) =>
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
        selected = chooseDeck(selectable, context, rejected);
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
        selected = chooseDeck(selectable, context, rejected);
      }
    }
  }
  const directions = labelCreativeDirections(selected, creativeMetrics(context));
  const copy = editorialDeckCopy(selected, context, directions);
  return {
    status: selected.length === 3 ? 'complete' : selected.length ? 'partial' : 'no_feasible_looks',
    outfits: selected.map((look, index) => {
      return {
        id: look.id, items: look.items, description: copy[index].description,
        concept: { ...copy[index].concept, referenceId: reference?.referenceId ?? null },
        direction: directions[index],
        fashion: look.fashion,
        ...(visualEvidenceById.has(look.id)
          ? { visualEvidence: visualEvidenceById.get(look.id) }
          : visualWinner?.id === look.id ? { visualEvidence: visualWinner.evidence } : {}),
      };
    }),
    diagnostics: { ...retrieval.diagnostics, editorialRegister: { version: EDITORIAL_REGISTER_VERSION, rejected: editorialVetoes, eligible: selectable.length }, visualVerdict,
      directionVersion: DIRECTION_VERSION,
      editorialQualityVersion: EDITORIAL_QUALITY_VERSION,
      deckCopyVersion: DECK_COPY_VERSION,
      feedbackTasteVersion: FEEDBACK_TASTE_VERSION,
      feedbackTasteActive: !!(feedbackTaste && feedbackTaste.active),
      ...(visualReviewCoverage ? { visualReviewCoverage } : {}),
      directions: directions.map((d) => d.id),
      qualifiedDirections: directions.filter((d) => d.qualified).length,
      neighborBriefs: neighborHouse.length,
      shortlistMix: { ...layerMix, best: ordered[0] ?? null, top20cutoff: ordered[19] ?? null },
      ...(floor.limited ? { exclusionQualityLimited: true } : {}) },
  };
}
