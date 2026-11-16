// Experimental catalog/wardrobe recommendation path. Callers own inventory,
// permission and provider admission. Local styling never requires image upload.
import { shortlistRetrievalDeck, evaluateVisualShortlist, independentOutfitDirection } from './prototype.mjs';
import { MAX_VISUAL_CANDIDATES } from './visual_critic_hosted.mjs';
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
import { CONCEPT_FIDELITY_VERSION, evaluateConceptFidelity, verifiedPaletteMatch } from './concept_fidelity.mjs';
import { FASHION_FINISH_VERSION, evaluateFashionFinish } from './fashion_finish.mjs';
import { isNonOutfitItem, outfitStructureReason } from '../../src/domain/outfitStructure.ts';
import { rejectForClimate, rejectForStyleOccasion, rejectForRequestIntent, isWorkoutOccasion } from '../../src/domain/occasionRules.ts';
import { editorialDeckCopy, DECK_COPY_VERSION } from './deck_editorial_copy.mjs';
import { editorialRegisterVeto, EDITORIAL_REGISTER_VERSION } from './editorial_register.mjs';

const slots = ['top', 'outerwear', 'bottom', 'shoes'];

/** Guard against calling fashion-casual products workout equipment. A source
 * description or generic 'athletic fit' tag is not enough: the PRODUCT NAME
 * must support the intended physical activity, and hard contradictions win.
 * This is Stylist 6 retrieval admission, before price/visual ranking; it is
 * not a post-hoc change to a user's pinned SKU or a fabricated outfit.
 */
export function workoutItemTruthReason(item, context) {
  if (!isWorkoutOccasion(context?.occasion)) return null;
  const name=String(item?.name||'').toLowerCase().replace(/[-_]+/g,' ');
  const slot=String(item?.macroCategory||'');
  if (!['top','outerwear','bottom','shoes'].includes(slot)) return 'invalid_workout_slot';
  if (slot==='top') {
    if (/\b(polos?|linen|cashmere|merino|wool|flannel|oxford|poplin|dress|suit)\b/.test(name)) return 'non_workout_top';
    if (/\b(?:shirts?|button[ -]?down)\b/.test(name) && !/\b(?:t[ -]?shirts?|tee)\b/.test(name)) return 'non_workout_shirt';
    if (!/\b(?:training|workout|running|gym|sport(?:s)?wear|activewear|athletic|performance|technical|tech|quick[ -]?dry|drymove|moisture[ -]?wick)\b/.test(name)) return 'unverified_workout_top';
  }
  if (slot==='bottom') {
    if (/\b(?:jeans?|denim|chinos?|suit|tailored|wool|flannel|formal|dress|pleated|cargo)\b/.test(name)) return 'non_workout_bottom';
    if (!/\b(?:joggers?|sweatpants?|track[ -]?pants?|training|workout|running|gym|sport(?:s)?wear|activewear|athletic|technical|tech|quick[ -]?dry|drymove|shorts?)\b/.test(name)) return 'unverified_workout_bottom';
  }
  if (slot==='shoes') {
    if (/\b(?:loafer|derby|oxford|chelsea|formal|dress|suede)\b/.test(name)) return 'non_workout_footwear';
    if (!/\b(?:training|running|gym|workout|athletic|sports?|trail)\b/.test(name)) return 'unverified_workout_footwear';
  }
  if (slot==='outerwear') {
    if (/\b(?:blazer|wool|cardigan|suit|dress|denim)\b/.test(name)) return 'non_workout_layer';
    if (!/\b(?:training|running|gym|workout|activewear|athletic|performance|technical|tech|sports?|track|windbreaker)\b/.test(name)) return 'unverified_workout_layer';
  }
  return null;
}


/** Prevent a sleeveless vest from counting as the only winter coat in a
 * catalog outfit at below-5°C temperatures. This narrows only Stylist 6
 * catalog retrieval; owned garments and pinned user anchors are untouched.
 * Existing insulation/boot climate gates still validate the complete look.
 */
export function freezingCatalogVestReason(item, context) {
  if (context?.sourceMode !== 'catalog' || item?.macroCategory !== 'outerwear' ||
      !Number.isFinite(context?.weather?.temp) || context.weather.temp >= 5) return null;
  const label = `${item.name || ''} ${item.subtype || ''} ${item.fashion_subtype || ''}`.toLowerCase();
  return /\b(?:vests?|gilets?|waistcoats?)\b|жилет|безрукавк/.test(label)
    ? 'freezing_requires_sleeved_outerwear' : null;
}

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
export function buildPriceProfile(catalog, serverValidatedFxRates = null) {
  // Cross-currency comparisons are enabled ONLY from the authenticated
  // cloud's already-validated CBU reference-rate snapshot. No invented FX,
  // no multiplication of USD prices against UZS without proof.
  const fx=serverValidatedFxRates && serverValidatedFxRates.UZS===1 &&
    ['USD','EUR','GBP'].every(c=>Number.isFinite(serverValidatedFxRates[c]) &&
      serverValidatedFxRates[c] > 0 && serverValidatedFxRates[c] <= 1e12)
    ? Object.freeze({...serverValidatedFxRates}) : null;
  const byBand = new Map();
  for (const item of catalog || []) {
    if (!item || !/^(top|outerwear|bottom|shoes)$/.test(item.macroCategory || '')
      || !Number.isFinite(item.price) || item.price <= 0 || !/^[A-Z]{3}$/.test(item.currency || '')) continue;
    const key = `${item.currency}|${item.macroCategory}`;
    if (!byBand.has(key)) byBand.set(key, []);
    byBand.get(key).push(item.price);
    if(fx && Number.isFinite(fx[item.currency])){
      const normalizedKey=`CBU_REFERENCE_UZS|${item.macroCategory}`;
      if(!byBand.has(normalizedKey))byBand.set(normalizedKey,[]);
      const normalizedPrice=item.price*fx[item.currency];
      if(Number.isFinite(normalizedPrice))byBand.get(normalizedKey).push(normalizedPrice);
    }
  }
  const profile = {};
  if(fx)Object.defineProperty(profile,'_verifiedFx',{value:fx,enumerable:false});
  for (const [key, prices] of byBand) {
    if (prices.length < PRICE_SAMPLE_MINIMUM) continue;
    prices.sort((a, b) => a - b);
    // A catalog dominated by luxury inventory can have a 6.5m UZS
    // 'median' trouser even when perfectly good 850k trousers are offered.
    // Keep median for provenance; use the observed lower quartile as the
    // representative accessible-price anchor for unbudgeted requests.
    profile[key] = { median: prices[Math.floor((prices.length - 1) / 2)],
      accessible: prices[Math.floor((prices.length - 1) / 4)], count: prices.length };
  }
  return profile;
}

/** No invented cap when the user did not request a budget, but a 10–20x
 * catalog-price outlier should not routinely beat an equally stylish item.
 * Compare observed same-slot prices; validated CBU reference rates enable
 * cross-currency comparisons. Without that snapshot use native currencies.
 * The 0.18 score ceiling keeps this a soft ranking signal, not a luxury ban.
 */
export const UNBUDGETED_PRICE_PENALTY_MAX = 0.18;
/** A populated currency cohort can itself be entirely luxury-biased:
 * the 2026-11-16 Render Old Money run selected a €950 Aurélien layer even
 * while affordable alternative layers existed in UZS. Compare the native
 * lower-quartile anchor with the verified CBU-normalized all-currency anchor
 * and use the better-supported accessible reference if materially lower.
 * Only a server-validated snapshot permits this comparison; the user's
 * merchant price, currency label and explicit luxury choices stay intact.
 */
export function observedPriceComparison(item, profile){
  if(!item || !profile || !Number.isFinite(item.price) || item.price<=0)return null;
  const native=profile[`${item.currency}|${item.macroCategory}`];
  const fx=profile._verifiedFx;
  const reference=fx && Number.isFinite(fx[item.currency])
    ? profile[`CBU_REFERENCE_UZS|${item.macroCategory}`] : null;
  if(reference?.count>=12){
    const amount=item.price*fx[item.currency];
    const nativeAnchor=native?.accessible > 0 ? native.accessible : native?.median;
    const referenceAnchor=reference.accessible > 0 ? reference.accessible : reference.median;
    // A 25% gap prevents minor FX/tax/merchant variation from dominating a
    // useful local cohort. If currency cohorts are sparse, the combined
    // sample remains the only observed price evidence.
    if(Number.isFinite(amount) && referenceAnchor>0 &&
        (!nativeAnchor || referenceAnchor*1.25 < nativeAnchor*fx[item.currency]))
      return {value:amount,band:reference,source:'verified_cbu_reference'};
  }
  return native ? {value:item.price,band:native,source:'same_currency'} : null;
}
export function pricePenaltyForLook(items, profile) {
  let penalty = 0, flagged = 0;
  for (const item of items || []) {
    const comparator=observedPriceComparison(item,profile);
    const anchor=comparator?.band?.accessible > 0 ? comparator.band.accessible : comparator?.band?.median;
    if (!anchor || !(comparator.value > anchor)) continue;
    const ratio = comparator.value / anchor;
    if (ratio <= 2) {
      penalty += 0.035 * (ratio - 1);
    } else {
      flagged += 1;
      // A 3x item incurs a mild penalty; a 10x luxury piece receives a
      // substantially stronger one. Never reject or fabricate a user budget.
      penalty += 0.035 + 0.05 * Math.log2(ratio / 2);
    }
  }
  return { penalty: Math.min(UNBUDGETED_PRICE_PENALTY_MAX, penalty), flagged };
}

/** Small per-item seed adjustment before the 8-piece role candidate cap.
 * Without this, a 7m trouser survives every role shortlist solely because
 * the catalog's first page is mostly luxury, so later outfit-price penalties
 * have no lower-priced alternatives to choose. Source-quoted prices only,
 * same slot/currency evidence; never an invented overall spending limit.
 */
export function unbudgetedPriceSeedAdjustment(item,profile) {
  const comparator=observedPriceComparison(item,profile);
  const anchor=comparator?.band?.accessible > 0 ? comparator.band.accessible : comparator?.band?.median;
  if(!anchor)return 0;
  const ratio=comparator.value/anchor;
  return ratio<=1 ? 0.065 : -Math.min(0.26,0.085*Math.log2(ratio));
}

/** Explicit requests for luxury/premium styling are not cost-optimized.
 * A style such as Old Money by itself is NOT a request to spend 9m UZS on
 * one sweater; price sensitivity should remain active by default.
 */
export function explicitlyRequestsLuxury(context) {
  const request = `${context?.request || ''} ${context?.brief || ''}`;
  return /\b(?:luxury|premium|designer|high[ -]?end|couture)\b|люкс|премиальн|дорог(?:ой|ие|ую)|дизайнерск/i.test(request);
}

/** Reject only unrequested extreme source-price outliers when a genuinely
 * populated slot/currency cohort proves comparable accessible stock exists.
 * This is NOT a hard user budget, exchange-rate guess, or luxury brand ban.
 * Explicit luxury instructions, favorite houses and owned/locked pieces win.
 * The default catalog should not quietly turn a 900k trouser request into
 * an 8.6m trouser simply because an editorial model likes the photograph.
 */
export function extremeUnrequestedCatalogPriceReason(item, profile, context) {
  if(context?.sourceMode !== 'catalog' || context?.budget ||
      explicitlyRequestsLuxury(context) || !Number.isFinite(item?.price) || item.price <= 0)
    return null;
  const comparator=observedPriceComparison(item,profile);
  const cohort=comparator?.band;
  if(!cohort || cohort.count<12 || !(cohort.accessible>0))return null;
  if(comparator.value < cohort.accessible*6)return null;
  const brand=String(item.brand||'').trim().toLowerCase();
  if(brand && Array.isArray(context?.preferences?.favoriteBrands) &&
      context.preferences.favoriteBrands.some(x=>String(x||'').trim().toLowerCase()===brand))return null;
  if(brand && String(context?.request||'').toLowerCase().includes(brand))return null;
  return 'extreme_unrequested_catalog_price_outlier';
}

function referenceAffinity(items, reference) {
  if (!reference) return null;
  const descriptions = items.map(describeGarment);
  const roles = descriptions.map((d, index) => {
    const expected = reference.roles[items[index].macroCategory] || [];
    return expected.some((subtype) => normalized(subtype) === normalized(d.subtype)) ? 1 : 0;
  });
  // Use the source's verified hue tokens (navy, cream, white...) instead of
  // the descriptor's coarse tone bins (cool-dark, light, mid...). Otherwise
  // paletteFit and slotColorFit were systematically 0 even for a perfect match.
  const paletteMatches = items.map(item => verifiedPaletteMatch(item,reference.palette))
    .filter(value => value !== null);
  const paletteFit = paletteMatches.length
    ? paletteMatches.filter(Boolean).length / paletteMatches.length : null;
  const slotMatches = items.flatMap(item => {
    const notes = reference.garmentNotes?.[item.macroCategory];
    const expected = colorTokensOf({ name: notes });
    if (!expected.length) return [];
    const matched = verifiedPaletteMatch(item, expected);
    return matched === null ? [] : [Number(matched)];
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
 * High-confidence catalog register: a holiday/towelling shirt or an avant-garde
 * asymmetric jacket can look individually handsome in a photo but cannot be
 * recommended as an unrequested CLASSIC OFFICE piece. This is a retrieval
 * admission rule (before ranking and NVIDIA), so a coherent replacement is
 * actually available to compete instead of wasting the critic's top-4 slots.
 * It never edits merchant metadata and does not apply to owned items, leisure
 * occasions or an explicitly requested statement garment.
 */
export function rejectUnrequestedOfficeRegisterPiece(item, context) {
  if (context?.sourceMode !== 'catalog' ||
      !['classic','old_money','semi_classic','business_casual'].includes(context?.style) ||
      !/\b(work|office|business|meeting|conference)\b/i.test(String(context?.occasion || ''))) return null;
  const name = String(item?.name || '');
  const request = String(context?.request || '');
  if (item?.macroCategory === 'top') {
    const explicitResortTop =
      /\b(?:resort|beach|aloha|hawaiian|holiday)\b.{0,35}\b(?:shirt|tee|polo|top)\b/i.test(name) ||
      /\b(?:terry(?:[ -]?towell(?:ing)?)?|towell?ing|towel[ -]?cloth|beachwear)\b.{0,35}\b(?:shirt|tee|polo|top)\b/i.test(name);
    if (explicitResortTop && !/\b(?:resort|beach|aloha|hawaiian|terry|towell?ing|towel[ -]?cloth)\b/i.test(request))
      return 'unrequested_resort_top_in_polished_office';
  }
  if (item?.macroCategory === 'outerwear' && context.style === 'classic') {
    // Verified source image: "Icon guru-collar wool-blend jacket with
    // off-centre fastening" reads avant-garde, not traditional office.
    const statement = /\b(?:guru[ -]?collar|off[ -]?cent(?:er|re)|asymmetri(?:c|cal))\b/i.test(name);
    if (statement && !/\b(?:guru[ -]?collar|asymmetri(?:c|cal)|off[ -]?cent(?:er|re)|statement[ -]?jacket)\b/i.test(request))
      return 'unrequested_statement_jacket_in_classic_office';
  }
  return null;
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
  const dressedQuiet = ['classic','old_money','semi_classic'].includes(context.style) &&
    ['smart','formal'].includes(context.dressCode) &&
    /\b(work|office|business|date|dinner|evening)\b/i.test(
      `${context.occasion || ''} ${context.request || ''}`);
  // Actual owner iOS QA, v51: Old Money / Casual still admitted the
  // Navy Technical Leisure Blazer. Casual is a dress-code level, NOT a
  // request for sportswear. A known leisure shell contradicts a quiet
  // heritage capsule even on weekends, unless explicitly asked for.
  const oldMoneyWeekend = context.style === 'old_money' &&
    /\b(casual|weekend|brunch|outing)\b|повседнев|выходн|прогулк/i.test(
      `${context.occasion || ''} ${context.request || ''}`);
  if (context.sourceMode !== 'catalog' ||
      descriptor.slot !== 'outerwear' || !(dressedQuiet || oldMoneyWeekend)) return false;
  // A user explicitly choosing sporty outerwear outranks stylist aesthetics,
  // without waiving the existing physical safety or budget gates.
  if (/\b(technical leisure|sporty jacket|track jacket|training jacket|reversible resort jacket|denim leisure blazer)\b/i
    .test(context.request || '')) return false;
  const name = String(item?.name || '');
  return /\b(technical[ -]?leisure|track jacket|training jacket|running jacket|sport[s]? jacket)\b/i.test(name) ||
    // Live staging v52: Aurélien's literal "Dark Blue Denim Leisure Blazer"
    // looked like an elevated second layer to the generic descriptor, even
    // though the merchant explicitly identifies denim leisure, not tailoring.
    (oldMoneyWeekend && /\bdenim\s+leisure\s+(?:blazer|jacket)\b/i.test(name)) ||
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

export const FASHION_QUALITY_VERSION = 's6-fashion-quality-v1';

const OFFICE_INTENT = /\b(work|office|business|meeting|conference)\b|работ|офис|делов|встреч|\b(ofis|ish|biznes|uchrashuv)\b/i;

const QUIET_OFFICE_STYLES = new Set(['classic','old_money','semi_classic','business_casual']);
const QUIET_LIFESTYLE_STYLES = new Set(['classic','old_money','semi_classic']);
const CASUAL_INTENT = /\b(casual|weekend|date|dinner|brunch|outing|evening|weekend)\b|повседнев|выходн|ужин|прогулк|свидан/i;
function fashionQualityApplies(context) {
  const style = String(context?.style || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const occasion = String(context?.occasion || '');
  const req = String(context?.request || '');
  const brief = String(context?.brief || '');
  const office = OFFICE_INTENT.test(occasion) || OFFICE_INTENT.test(req) || OFFICE_INTENT.test(brief);
  const quietLifestyle = QUIET_LIFESTYLE_STYLES.has(style) &&
    CASUAL_INTENT.test(`${occasion} ${req} ${brief}`);
  return context?.sourceMode === 'catalog' && QUIET_OFFICE_STYLES.has(style) &&
    (office || quietLifestyle) &&
    !(Array.isArray(context?.requiredIds) && context.requiredIds.length);
}
const inactiveFashionQuality = () => ({ version:FASHION_QUALITY_VERSION,active:false,
  paletteHarmony:null,shoeAppropriateness:null,silhouetteBalance:null,
  textureCoordination:null,boringOfficePenalty:0,score:null,adjustment:0 });

const QUIET_PALETTE = new Set([
  'navy', 'blue', 'grey', 'gray', 'charcoal', 'black', 'white',
  'cream', 'beige', 'tan', 'brown', 'camel', 'olive', 'khaki',
  'burgundy', 'forest_green', 'taupe', 'ecru', 'stone', 'sand',
  'light', 'mid', 'cool-dark', 'warm-dark', 'dark',
  'light-blue', 'dark-neutral', 'indigo',
]);

function itemColorToken(item) {
  const explicit = typeof item?.color === 'string' ? colorTokensOf({ color:item.color }) : [];
  if (explicit.length === 1) return explicit[0];
  if (explicit.length > 1) return null;
  const fromName = colorTokensOf({ name:item?.name || item?.title || '' });
  // A coarse "earth", "dark" or "light" bin is not an exact color observation.
  return fromName.length === 1 ? fromName[0] : null;
}

function computePaletteHarmony(items, descriptors, context, coherencePalette) {
  const colors = items.map((item, i) => itemColorToken(item, descriptors[i])).filter(Boolean);
  const base = Number.isFinite(coherencePalette) ? coherencePalette : 0.75;
  const style = String(context?.style || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const occasion = String(context?.occasion || '').toLowerCase();
  const req = String(context?.request || '').toLowerCase();
  const brief = String(context?.brief || '').toLowerCase();
  const quiet = ['classic', 'old_money', 'semi_classic', 'business_casual', 'smart_casual'].includes(style) ||
    OFFICE_INTENT.test(occasion) || OFFICE_INTENT.test(req) || OFFICE_INTENT.test(brief);

  let bonus = 0;
  if (quiet && colors.length > 0) {
    const quietMatchCount = colors.filter(c => QUIET_PALETTE.has(c)).length;
    if (quietMatchCount === colors.length) {
      bonus += 0.12;
    } else if (quietMatchCount / colors.length >= 0.75) {
      bonus += 0.06;
    }
  }

  const clashing = items.some(item => {
    const text = `${item.name || item.title || ''} ${item.color || ''}`.toLowerCase();
    return /\b(neon|fluorescent|electric[ -]blue|hot[ -]pink|bright[ -]yellow)\b/.test(text);
  });
  if (clashing && quiet) {
    bonus -= 0.25;
  }

  const uniqueHues = new Set(colors);
  if (uniqueHues.size > 3 && quiet) {
    bonus -= 0.08;
  }

  return Math.max(0, Math.min(1, base + bonus));
}

function itemSubtype(item, descriptor) {
  if (descriptor?.subtype && descriptor.subtype !== 'unknown' && descriptor.subtype !== 'none') {
    return descriptor.subtype.toLowerCase().replace(/_/g, '-');
  }
  const explicit = item?.subtype || item?.fashion_subtype ||
    (typeof item?.type === 'string' && item.type !== item?.macroCategory ? item.type : '');
  if (explicit) return String(explicit).toLowerCase().replace(/_/g, '-');
  if (item) {
    const desc = describeGarment(item);
    if (desc?.subtype && desc.subtype !== 'unknown' && desc.subtype !== 'none') {
      return desc.subtype.toLowerCase().replace(/_/g, '-');
    }
  }
  return '';
}

function computeShoeAppropriateness(items, descriptors, context) {
  const shoeDesc = descriptors.find(d => d.slot === 'shoes');
  const shoeItem = items.find(i => i.macroCategory === 'shoes');
  if (!shoeDesc && !shoeItem) return 0.5;

  const style = String(context?.style || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const occasion = String(context?.occasion || '').toLowerCase();
  const req = String(context?.request || '').toLowerCase();
  const brief = String(context?.brief || '').toLowerCase();
  const isWork = OFFICE_INTENT.test(occasion) || OFFICE_INTENT.test(req) || OFFICE_INTENT.test(brief);
  const isQuietLifestyle = ['classic', 'old_money', 'semi_classic'].includes(style) &&
    !isWork && CASUAL_INTENT.test(`${occasion} ${req} ${brief}`);
  const isQuietOrWork = ['classic', 'old_money', 'semi_classic', 'business_casual'].includes(style) ||
    isWork ||
    /\b(work|office|business|meeting|conference|formal)\b/i.test(`${occasion} ${req} ${brief}`);

  const sub = itemSubtype(shoeItem, shoeDesc);
  const text = `${shoeItem?.name || shoeItem?.title || ''} ${shoeItem?.type || ''}`.toLowerCase();

  if (isQuietOrWork) {
    // Explicit shoe preferences outrank this purely editorial prior.
    if (/\b(sneakers?|trainers?)\b|кроссовк|кед/i.test(`${req} ${brief}`) &&
        /^(sport-sneaker|sneaker|minimal-sneaker)s?$/i.test(sub)) return 0.90;
    if (/^(?:.*[_-])?(loafer|oxford|monk(?:[_-]?strap)?|brogue|dress[_-]?shoe|chelsea(?:[_-]?boot)?|dress[_-]?boot)s?$|^(?:.*[_-])?derb(?:y|ies|ys)$/i.test(sub) ||
        /\b(loafers?|derb(?:y|ies|ys)|oxfords?|monk[ -]?straps?|chelsea(?:[ -]?boots?)?|brogues?|dress[ -]shoes?|dress[ -]boots?)\b|дерби|оксфорд|туфл|броги|монк|\btufli\b/i.test(text)) {
      if (shoeDesc?.soleProfile === 'dress' || /\b(leather|blake|welt|calfskin)\b|кож[а-я]*|замш[а-я]*|\b(charm|teri)\b/i.test(text)) {
        return 1.0;
      }
      return 0.95;
    }
    if (sub === 'minimal-sneaker' || SLEEK_SNEAKER.test(text)) {
      // Old Money weekend can genuinely use a clean low-profile sneaker.
      // Running, retro/chunky and trail shoes remain less appropriate.
      return isQuietLifestyle ? (shoeDesc?.casualSole ? 0.77 : 0.84)
        : shoeDesc?.casualSole ? 0.60 : 0.68;
    }
    if (/^(sport-sneaker|sneaker|hiking-boot)s?$/i.test(sub) ||
        shoeDesc?.athletic || shoeDesc?.casualSole ||
        /\b(running|training|athletic|trail|sports?|retro|chunky)\b|кроссовк|кед|бегов/i.test(text)) {
      return 0.20;
    }
    if (/^(espadrille|boat-shoe|sandal|driver)s?$/i.test(sub) ||
        /\b(espadrilles?|boat[ -]shoes?|sandals?|drivers?|slides?)\b|сандал|шлеп/i.test(text)) {
      return 0.35;
    }
    return 0.60;
  }

  if (/^(sport-sneaker|sneaker|minimal-sneaker)s?$/i.test(sub)) {
    return 0.95;
  }
  return 0.80;
}

function computeSilhouetteBalance(items, descriptors, context, coherenceProportion) {
  const base = Number.isFinite(coherenceProportion) ? coherenceProportion : 0.80;
  const topDesc = descriptors.find(d => d.slot === 'top');
  const botDesc = descriptors.find(d => d.slot === 'bottom');
  const outerDesc = descriptors.find(d => d.slot === 'outerwear');
  const botItem = items.find(i => i.macroCategory === 'bottom');
  const outerItem = items.find(i => i.macroCategory === 'outerwear');

  let bonus = 0;
  if (topDesc && botDesc) {
    const topVol = topDesc.volume;
    const botVol = botDesc.volume;
    if (topVol && botVol && topVol !== 'unknown' && botVol !== 'unknown') {
      if (topVol === botVol) bonus += 0.05;
      else if ((topVol === 'slim' && botVol === 'regular') ||
               (topVol === 'regular' && botVol === 'wide')) bonus += 0.05;
      else if (topVol === 'wide' && botVol === 'slim') bonus -= 0.05;
    }
  }

  const botSub = itemSubtype(botItem, botDesc);
  const outerSub = itemSubtype(outerItem, outerDesc);

  if (botSub && /^(tailored[_-]?(?:trouser|pants?)|trouser|pants?|slacks?)s?$/i.test(botSub)) {
    bonus += 0.05;
  }
  if (outerSub && /^(blazer|cardigan|coat)s?$/i.test(outerSub)) {
    bonus += 0.05;
  }

  return Math.max(0, Math.min(1, base + bonus));
}

function computeTextureCoordination(items, descriptors, context, coherenceTexture) {
  const base = Number.isFinite(coherenceTexture) ? coherenceTexture : 0.78;
  let bonus = 0;
  const hasKnit = descriptors.some((d, idx) => {
    const itSub = itemSubtype(items[idx], d);
    const itTex = (items[idx]?.texture || '').toLowerCase();
    const itText = `${items[idx]?.name || items[idx]?.title || ''} ${items[idx]?.type || ''}`.toLowerCase();
    return /^(fine-knit|sweater|cardigan|knit-trouser)s?$/i.test(d.subtype) ||
      /^(fine-knit|sweater|cardigan|knit-trouser)s?$/i.test(itSub) ||
      /^(knit|fine-knit)$/i.test(d.texture) ||
      /^(knit|fine-knit)$/i.test(itTex) ||
      /\b(merino|cashmere|wool|knit|ribbed)\b/i.test(`${d.evidence?.join(' ') || ''}`) ||
      /\b(merino|cashmere|wool|knit|ribbed)\b|шерст|кашемир|меринос|трикотаж|вязаный/i.test(itText);
  });
  const hasTailored = descriptors.some((d, idx) => {
    const itSub = itemSubtype(items[idx], d);
    const itTex = (items[idx]?.texture || '').toLowerCase();
    const itText = `${items[idx]?.name || items[idx]?.title || ''} ${items[idx]?.type || ''}`.toLowerCase();
    return /^(blazer|tailored[_-]?(?:trouser|pants?)|suit[_-]pants?)s?$/i.test(d.subtype) ||
      /^(blazer|tailored[_-]?(?:trouser|pants?)|suit[_-]pants?)s?$/i.test(itSub) ||
      /^(flannel|wool|tweed|suiting)$/i.test(d.texture) ||
      /^(flannel|wool|tweed|suiting)$/i.test(itTex) ||
      /\b(flannel|wool|tweed|suiting)\b|фланел|шерст|твид|костюмн/i.test(itText);
  });
  const hasLeather = descriptors.some((d, idx) => {
    const itSub = itemSubtype(items[idx], d);
    const itTex = (items[idx]?.texture || '').toLowerCase();
    const itText = `${items[idx]?.name || items[idx]?.title || ''} ${items[idx]?.type || ''}`.toLowerCase();
    return (d.slot === 'shoes' || items[idx]?.macroCategory === 'shoes') && (
      /^(?:.*[_-])?(loafer|oxford|monk(?:[_-]?strap)?|brogue|dress[_-]?shoe|chelsea(?:[_-]?boot)?|dress[_-]?boot)s?$|^(?:.*[_-])?derb(?:y|ies|ys)$/i.test(d.subtype) ||
      /^(?:.*[_-])?(loafer|oxford|monk(?:[_-]?strap)?|brogue|dress[_-]?shoe|chelsea(?:[_-]?boot)?|dress[_-]?boot)s?$|^(?:.*[_-])?derb(?:y|ies|ys)$/i.test(itSub) ||
      /^(leather|suede)$/i.test(d.texture) ||
      /^(leather|suede)$/i.test(itTex) ||
      /\b(leather|suede|calfskin)\b|кож[а-я]*|замш[а-я]*|\b(charm|teri)\b/i.test(itText)
    );
  });

  if (hasKnit && hasTailored) bonus += 0.10;
  if (hasTailored && hasLeather) bonus += 0.05;

  return Math.max(0, Math.min(1, base + bonus));
}

function computeBoringOfficePenalty(items, descriptors, context) {
  const topDesc = descriptors.find(d => d.slot === 'top');
  const botDesc = descriptors.find(d => d.slot === 'bottom');
  const outerDesc = descriptors.find(d => d.slot === 'outerwear');
  const topItem = items.find(i => i.macroCategory === 'top');
  const botItem = items.find(i => i.macroCategory === 'bottom');
  const outerItem = items.find(i => i.macroCategory === 'outerwear');

  const outerSub = itemSubtype(outerItem, outerDesc);

  const hasOuter = Boolean(outerItem && outerSub !== 'none') ||
    Boolean(outerSub && outerSub !== 'unknown' && outerSub !== 'none');
  if (hasOuter) {
    return 0;
  }

  const style = String(context?.style || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const occasion = String(context?.occasion || '').toLowerCase();
  const req = String(context?.request || '').toLowerCase();
  const brief = String(context?.brief || '').toLowerCase();
  const isWork = OFFICE_INTENT.test(occasion) || OFFICE_INTENT.test(req) || OFFICE_INTENT.test(brief) ||
    /\b(work|office|business|meeting)\b/i.test(`${occasion} ${req} ${brief}`);

  if (!isWork) return 0; // weekend simplicity is not an office-quality defect

  const temp = Number(context?.weather?.temp ?? context?.weather?.temperature);
  const intentionallySimple = /\b(no jacket|no blazer|no layer|without (?:a )?jacket|just a shirt)\b|без пиджак|без куртк|без слоя/i.test(`${req} ${brief}`);
  if ((Number.isFinite(temp) && temp >= 25) || intentionallySimple) return 0;
  const topSub = itemSubtype(topItem, topDesc);
  const botSub = itemSubtype(botItem, botDesc);
  const isShirt = /^(shirt|dress[_-]?shirt|button[_-]?down|oxford([_-]?shirt)?)s?$/i.test(topSub);
  const isTrouser = /^(trouser|tailored[_-]?(?:trouser|pants?)|suit[_-]pants?|chino|pants?|slacks?)s?$/i.test(botSub);

  if (isShirt && isTrouser) {
    const interestingTexture = descriptors.some((d, idx) => {
      const itTex = (items[idx]?.texture || '').toLowerCase();
      const itPat = (items[idx]?.pattern || '').toLowerCase();
      const itText = `${items[idx]?.name || items[idx]?.title || ''} ${items[idx]?.type || ''}`.toLowerCase();
      return /^(fine-knit|knit|flannel|tweed|corduroy|linen)$/i.test(d.texture) ||
        /^(fine-knit|knit|flannel|tweed|corduroy|linen)$/i.test(itTex) ||
        ['stripe', 'check'].includes(d.pattern) ||
        ['stripe', 'check'].includes(itPat) ||
        /\b(flannel|tweed|corduroy|linen|stripe|check|plaid)\b|фланел|твид|вельвет|лен|льн|полоск|клетк/i.test(itText);
    });
    if (!interestingTexture) {
      return 0.10;
    }
    return 0.05;
  }
  return 0;
}

/**
 * Fashion-quality evaluator: bounded whole-look styling assessment
 * incorporating palette harmony, shoe appropriateness, silhouette balance,
 * texture coordination, and boring office penalty.
 */
export function evaluateFashionQuality(lookOrItems, context = {}) {
  if (!fashionQualityApplies(context)) return inactiveFashionQuality();
  const rawItems = Array.isArray(lookOrItems) ? lookOrItems : (lookOrItems?.items || []);
  const items = Array.isArray(rawItems) ? rawItems.filter(Boolean) : [];
  const look = (lookOrItems && !Array.isArray(lookOrItems)) ? lookOrItems : { items };
  const descriptors = items.map(describeGarment);

  const coherencePalette = look?.fashion?.dimensions?.palette;
  const coherenceProportion = look?.fashion?.dimensions?.proportion;
  const coherenceTexture = look?.fashion?.dimensions?.texture;

  const paletteHarmony = computePaletteHarmony(items, descriptors, context, coherencePalette);
  const shoeAppropriateness = computeShoeAppropriateness(items, descriptors, context);
  const silhouetteBalance = computeSilhouetteBalance(items, descriptors, context, coherenceProportion);
  const textureCoordination = computeTextureCoordination(items, descriptors, context, coherenceTexture);
  const boringOfficePenalty = computeBoringOfficePenalty(items, descriptors, context);

  const score = Math.max(0, Math.min(1,
    0.25 * paletteHarmony +
    0.30 * shoeAppropriateness +
    0.20 * silhouetteBalance +
    0.25 * textureCoordination -
    boringOfficePenalty
  ));

  const adjustment = Math.max(-0.08, Math.min(0.06, (score - 0.75) * 0.25));

  return {
    paletteHarmony: Number(paletteHarmony.toFixed(4)),
    shoeAppropriateness: Number(shoeAppropriateness.toFixed(4)),
    silhouetteBalance: Number(silhouetteBalance.toFixed(4)),
    textureCoordination: Number(textureCoordination.toFixed(4)),
    boringOfficePenalty: Number(boringOfficePenalty.toFixed(4)),
    score: Number(score.toFixed(4)),
    adjustment: Number(adjustment.toFixed(4)),
    active: true,
    version: FASHION_QUALITY_VERSION,
  };
}

/**
 * Bounded fashion-quality reranker for final look selection.
 * Reranks candidate looks by evaluating fashion quality signals and applying
 * bounded quality adjustments.
 */
export function rerankFashionCandidates(candidates, context = {}) {
  if (!Array.isArray(candidates)) return [];
  if (!fashionQualityApplies(context)) return candidates;
  for (const look of candidates) {
    if (!look) continue;
    if (!look.fashionQuality) {
      look.fashionQuality = look.fashion?.fashionQuality || evaluateFashionQuality(look, context);
    }
    const quality = look.fashionQuality;
    if (look.fashion) {
      look.fashion.fashionQuality = quality;
      if (Number.isFinite(look.fashion.score) && !look.fashion.qualityAdjustmentApplied && Number.isFinite(quality?.adjustment)) {
        look.fashion.score = Number((look.fashion.score + quality.adjustment).toFixed(4));
        look.fashion.qualityAdjustmentApplied = true;
      }
    } else if (Number.isFinite(look.seedScore) && !look.qualityAdjustmentApplied && Number.isFinite(quality?.adjustment)) {
      look.seedScore = Number((look.seedScore + quality.adjustment).toFixed(4));
      look.qualityAdjustmentApplied = true;
    }
  }
  return candidates.sort((a, b) => {
    const sa = Number.isFinite(a?.fashion?.score) ? a.fashion.score : (Number.isFinite(a?.seedScore) ? a.seedScore : -Infinity);
    const sb = Number.isFinite(b?.fashion?.score) ? b.fashion.score : (Number.isFinite(b?.seedScore) ? b.seedScore : -Infinity);
    if (sb !== sa) return sb - sa;
    return String(a?.id || '').localeCompare(String(b?.id || ''));
  });
}

/** A generic Old Money / Casual chip is not a summer-holiday brief.
 * The owner iOS QA surfaced terry resort shirts and two SHORTS looks at
 * unknown weather (November). Prefer versatile tailoring/knits until there
 * is explicit warm-weather evidence. Do not override summer or a deliberate
 * shorts request, and do not affect owned/anchored garments.
 */
export function rejectUnrequestedRivieraPiece(descriptor, context, item) {
  if(context?.sourceMode !== 'catalog' || context?.style !== 'old_money' ||
    !/\b(casual|weekend|outing|brunch)\b|повседнев|выходн|прогулк/i.test(
      `${context?.occasion || ''} ${context?.request || ''}`)) return null;
  const brief=String(context.request || '');
  const explicitSummer=/\b(summer|beach|resort|coastal|riviera|holiday|vacation|shorts?|bermudas?)\b|летн|пляж|курорт|шорт/i.test(brief);
  const warmWeather=Number.isFinite(context?.weather?.temp) && context.weather.temp>=24;
  if(explicitSummer || warmWeather) return null;
  const name=String(item?.name || '');
  if(descriptor.slot==='bottom' && (descriptor.subtype==='shorts' ||
    // The 2026-11-16 owner Render iOS test surfaced Zara's literal
    // "Pleated Relaxed Fit Culotte Trousers" as calf-exposing shorts,
    // even though its inferred subtype was 'tailored-trouser'.
    /\b(shorts?|bermudas?|culottes?)\b/i.test(name)))
    return 'unknown_weather_unrequested_short_bottom';
  if(descriptor.slot==='top' &&
    /\b(terry|towell?ing|beachwear|resort(?:[ -]wear)?)\b/i.test(name) &&
    /\b(shirts?|tees?|t-shirts?|polos?|tops?)\b/i.test(name))
    return 'unknown_weather_resort_terry_top';
  return null;
}

/** An Old Money concept is not permission to show a generic sports hoodie.
 * User-requested hoodies, locked garments and owned wardrobe remain intact.
 */
export function quietLuxuryHoodieReason(item, descriptor, context) {
  const style = String(context?.style || '').toLowerCase().replace(/[\s-]+/g,'_');
  if(context?.sourceMode !== 'catalog' || style !== 'old_money' ||
      item?.macroCategory !== 'outerwear') return null;
  const requested = String(context?.request || '');
  // Explicit user intent wins. A hooded item owned by the user is already
  // handled by the source-mode check above; never silently replace it.
  if(/\bhood(?:ie|ies|y|ed)\b|\bscuba\b|худи|капюшон/i.test(requested)) return null;
  const title = String(item?.name || '');
  // Live Render iOS (2026-11-16): H&M "Loose-Fit Scuba Hooded Jacket"
  // is classified as subtype=jacket, and "hooded" bypassed the historical
  // hoodie-only filter. Catch visibly hooded outerwear from merchant titles
  // without banning ordinary, unhooded jackets and blazers.
  return descriptor?.subtype === 'hoodie' ||
    /\bhood(?:ies|ie|y|ed)\b|худи|с капюшоном/i.test(title)
    ? 'quiet_luxury_hoodie_mismatch' : null;
}

/** A basic jersey tee can be styled for Old Money, but is not the
 * strongest default hero when collared shirts and fine knitwear exist.
 * A bounded ranking preference, NOT a hidden exclusion or user budget.
 * Deliberately requested tees, refined fabrics and other styles are intact.
 */
export function quietLuxuryBasicTeePenaltyForLook(items, context) {
  if (context?.sourceMode !== 'catalog' || context?.style !== 'old_money' ||
      !/\b(casual|weekend|outing|brunch)\b|повседнев|выходн|прогулк/i.test(
        `${context.occasion || ''} ${context.request || ''}`) ||
      /\b(t[- ]?shirts?|tees?|jersey tee)\b|футболк/i.test(String(context.request || ''))) return 0;
  const top=(items || []).find(item=>item?.macroCategory==='top');
  if (!top) return 0;
  const title=String(top.name || '');
  if (describeGarment(top).subtype!=='t-shirt' && !/\b(t[- ]?shirts?|tees?)\b/i.test(title)) return 0;
  // Known knit/silk/merino construction can be a refined jersey base.
  if (/\b(merino|cashmere|silk|fine[ -]?knit|ribbed knit)\b/i.test(title)) return 0;
  const layer=(items || []).find(item=>item?.macroCategory==='outerwear');
  const refinedLayer=layer && /^(blazer|cardigan|coat)$/.test(describeGarment(layer).subtype);
  return refinedLayer ? 0.025 : 0.065;
}

/** Do not market a hoodie as a cardigan simply because the reference photo
 * requested a cardigan and the rest of the pieces look pleasant together.
 * Named garment directions must be substantiated by the actual product.
 */
export function namedCardiganBriefMismatch(items, brief, context, requiredIds = []) {
  if(context?.sourceMode !== 'catalog' || requiredIds?.length ||
      !/\bcardigan\b/i.test(String(brief?.name || '')) ||
      !brief?.roles?.outerwear?.includes('cardigan')) return false;
  const layer = (items || []).find(item => item?.macroCategory === 'outerwear');
  return !layer || describeGarment(layer).subtype !== 'cardigan';
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
      if (workoutItemTruthReason(prepared, context) ||
          freezingCatalogVestReason(prepared, context) ||
          (!lockedIds.has(String(prepared.id)) &&
            rejectUnrequestedOfficeRegisterPiece(prepared, context)) ||
          rejectForRequestIntent(prepared, context.request, context.weather) ||
          descriptor.slot !== item.macroCategory || descriptor.productKind === 'complete-suit' ||
          garmentContextReason(prepared, context) || rejectForClimate(prepared, context.weather) ||
          rejectForStyleOccasion(prepared, context.style, context.occasion) ||
          rejectForUnsuitableDateNight(descriptor, context, prepared) ||
          rejectQuietSmartLeisureLayer(descriptor, context, prepared) ||
          quietLuxuryHoodieReason(prepared, descriptor, context) ||
          rejectUnrequestedRivieraPiece(descriptor, context, prepared) ||
          (!lockedIds.has(String(prepared.id)) &&
            extremeUnrequestedCatalogPriceReason(prepared, extras.priceProfile, context)) ||
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
        // Preserve proven coarse-palette seeds (light, dark, earth) and only
        // add a small exact-hue preference when those seeds do not apply.
        + (brief.palette.includes(normalized(d.color)) ? 0.15 :
          (!extras.budgeted && !extras.requiredIds?.length &&
            verifiedPaletteMatch(item, brief.palette) === true ? 0.02 : 0))
        + 0.70 * (personalScores.get(item) || 0)
        + FEEDBACK_ITEM_SEED_WEIGHT * (feedbackScores.get(item) || 0)
        + (context.sourceMode === 'catalog' && !extras.budgeted &&
          !lockedIds.has(String(item.id)) && !explicitlyRequestsLuxury(context)
          ? unbudgetedPriceSeedAdjustment(item,extras.priceProfile) : 0);
    },
    rankLook(items, brief) {
      if (namedCardiganBriefMismatch(items, brief, context, extras.requiredIds || [])) return null;
      const identity = JSON.stringify([brief?.id || null, ...items.map(({ id }) => id).sort()]);
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
      const pricing = extras.budgeted || !extras.priceProfile || explicitlyRequestsLuxury(context)
        ? { penalty: 0, flagged: 0 }
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
      const quietTeePenalty = quietLuxuryBasicTeePenaltyForLook(items, context);
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
      // A look also needs to look like the concept that nominated it.
      // Previously ranking ignored its generating brief, so an identical
      // generic capsule scored the same for every distinct fashion direction.
      const conceptFidelity = context.sourceMode === 'catalog' && !extras.requiredIds?.length
        ? evaluateConceptFidelity(items, brief, { requiredIds: extras.requiredIds })
        : { version: CONCEPT_FIDELITY_VERSION, score:null, adjustment:0,
          roleFit:null, paletteFit:null, mismatchedSlots:[] };
      const finish = context.sourceMode === 'catalog'
        ? evaluateFashionFinish({ items, descriptors:descriptions, context,
            requiredIds:extras.requiredIds || [] })
        : { version:FASHION_FINISH_VERSION,active:false,adjustment:0,evidence:[] };
      const fashionQuality = evaluateFashionQuality(
        { items, fashion: { dimensions: { ...dimensions, palette } } },
        { ...context, requiredIds:extras.requiredIds || [] }
      );
      const result = {
        score: .68 * compatibility + .22 * weakLink + .10 * referenceFit +
          personal.adjustment + editorial.adjustment + feedback.adjustment + finish.adjustment +
          // This was computed and published in diagnostics but silently
          // omitted from ranking. With reference-led NVIDIA judging, the
          // shortlist must prefer garments that actually match the brief.
          conceptFidelity.adjustment + fashionQuality.adjustment -
          recentStylingPenalty(items, recentSignatures) - pricing.penalty - freshness -
          registerPenalty - basePolishPenalty - quietTeePenalty,
        dimensions: { ...dimensions, palette },
        unbudgetedPricePenalty: pricing.penalty,
        evidence: pricing.flagged || freshness > 0 || registerPenalty > 0 || basePolishPenalty > 0 || quietTeePenalty > 0
          ? [...assessment.evidence,
            ...(pricing.flagged ? ['unbudgeted_premium_selection_dampened'] : []),
            ...(freshness > 0 ? ['stale_price_evidence_dampened'] : []),
            ...(registerPenalty > 0 ? ['trainer_footwear_register_dampened'] : []),
            ...(basePolishPenalty > 0 ? ['casual_base_elevated_context_dampened'] : []),
            ...(quietTeePenalty > 0 ? ['quiet_luxury_basic_tee_dampened'] : [])]
          : assessment.evidence,
        referenceMetadata: affinity,
        editorial,
        finish,
        fashionQuality,
        qualityAdjustmentApplied: Boolean(fashionQuality.active),
        conceptFidelity,
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

/**
 * Maximum size of a pairwise independent outfit deck. Deliberately bounded
 * to the <=8 garment-group candidates chosen for one NVIDIA review. A
 * different SKU, jacket color or title alone does not prove a new direction:
 * use the same structural rule as the final model-approved selection.
 */
export function visualArchitectureCapacity(looks, requiredIds = []) {
  if (!Array.isArray(looks) || looks.length > MAX_VISUAL_CANDIDATES) return 0;
  let maximum = 0;
  const walk = (index, selected) => {
    if (selected.length > maximum) maximum = selected.length;
    if (index === looks.length || maximum === 3) return;
    const candidate = looks[index];
    if (selected.every(other => independentOutfitDirection(candidate, other, requiredIds))) {
      walk(index + 1, [...selected, candidate]);
    }
    walk(index + 1, selected);
  };
  walk(0, []);
  return Math.min(3, maximum);
}

/** Number of different 3-look sets the approved-photo selector could form
 * if ALL candidates passed the visual quality gate. Maximizing this before
 * inference creates actual replacement options, not a row of clones. Bounded
 * at eight candidates (56 triples) and uses the final pairwise rule unchanged.
 */
export function visualRescueTripleCount(looks, requiredIds = []) {
  if (!Array.isArray(looks) || looks.length > MAX_VISUAL_CANDIDATES) return 0;
  let count = 0;
  for (let a = 0; a < looks.length; a++) {
    for (let b = a + 1; b < looks.length; b++) {
      if (!independentOutfitDirection(looks[a], looks[b], requiredIds)) continue;
      for (let c = b + 1; c < looks.length; c++) {
        if (independentOutfitDirection(looks[a], looks[c], requiredIds) &&
            independentOutfitDirection(looks[b], looks[c], requiredIds)) count++;
      }
    }
  }
  return count;
}

/**
 * Eight visual assessments must have a chance to yield three genuinely
 * different looks. If the three initially chosen cards contain structural
 * twins, pick the spare which unlocks the biggest independent subset; after
 * that prefer whole-look fashion quality over maximal novelty. All options
 * have already passed exact SKU/occasion/budget admission and the quality
 * window. NVIDIA retains the ONLY authority to approve displayed looks.
 */
export function rankVisualChallengerPool(entries) {
  return [...entries]
    .sort((a,b) => (b.rescueGain || 0) - (a.rescueGain || 0) ||
      (b.capacityGain || 0) - (a.capacityGain || 0) ||
      (b.unseenTop || 0) - (a.unseenTop || 0) ||
      b.look.fashion.score - a.look.fashion.score ||
      b.difference - a.difference || String(a.look.id).localeCompare(String(b.look.id)))
    .map(({look}) => look);
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

// The broad Ivy/preppy concept family is correct for weekends, but a Work
// intent needs office-weight garment roles *before* candidate retrieval.
// Otherwise seed retrieval overweights casual plaid/gingham shirts and heavy
// leisure footwear; a later NVIDIA veto has only four slots to fix them.
// This adjusts concept *preferences*, never budget, stock, anchors, or the
// visual quality gate. Every final SKU still passes the original hard gates.
const OLD_MONEY_OFFICE_ROLES = Object.freeze({
  'refined-ivy': [
    ['dress-shirt'], ['blazer', 'cardigan'], ['chino', 'tailored-trouser'], ['loafer', 'derby'],
  ],
  'textured-tailoring': [
    ['dress-shirt', 'fine-knit'], ['blazer', 'coat'], ['tailored-trouser', 'trouser'], ['derby', 'loafer', 'monk'],
  ],
  'smart-knit-layering': [
    ['dress-shirt', 'fine-knit'], ['cardigan'], ['tailored-trouser', 'chino'], ['loafer', 'chelsea'],
  ],
});
export function refineQuietOfficeConcept(concept, context) {
  // The source-bound catalog has genuine alternatives for an editorial
  // preference; wardrobe/mixed modes must never imply an owned SKU can be
  // substituted just because the office formula prefers another subtype.
  const style = String(context?.style || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  if (context?.sourceMode !== 'catalog' || style !== 'old_money' ||
      (!OFFICE_INTENT.test(String(context?.occasion || '')) &&
       !OFFICE_INTENT.test(String(context?.request || '')) &&
       !OFFICE_INTENT.test(String(context?.brief || ''))) ||
      !OLD_MONEY_OFFICE_ROLES[concept?.id]) return concept;
  const roles = OLD_MONEY_OFFICE_ROLES[concept.id];
  return { ...concept, slots: roles.map(slot => [...slot]) };
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
  // Optional trusted inventory-grounded AI directions; still subject to the
  // exact same retrieval, price, wardrobe, weather and visual checks.
  aiCreativeBriefs = [],
  // Optional prebuilt model from feedback_taste.mjs (explicit thumbs rows).
  // Absent/inert model = byte-identical legacy ranking (back-compat).
  feedbackTaste,
  // Passed by the trusted cloud handler only after validating the official
  // CBU snapshot; the app never provides these rates to the ranker.
  valueFxRates,
  // Cloud staging can opt into reviewing all three displayed looks in one
  // bounded provider batch. The isolated local / offline paths stay unchanged.
  visualDeckReview = false,
  // Optional in canary only: the third unreviewed merchant look must actually
  // load every product image, instead of displaying a 403/404/blank tile.
  canDisplayOutfit,
  // Only for isolated staging: after exactly two NVIDIA-approved looks,
  // permit ONE additional source-verified, independently styled finalist
  // that was NOT sent to NVIDIA. It is explicitly NOT_JUDGED in the response;
  // never reintroduce a model-vetoed or timed-out candidate, or invent evidence.
  allowUnreviewedVisualFill = false,
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
  const house = outfitConcepts({ ...context, optionalLayer: noLayer })
    .map((concept) => refineQuietOfficeConcept(concept, context))
    .map((concept) => ({
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
  // Let a bounded AI Creative Director design the FIRST three architectures,
  // while keeping the established house references as safe fallback choices.
  // These are *plans*, never invented items or guaranteed approved looks.
  // The retrieval kernel canonicalizes brief IDs (`ai-director-1` ->
  // `ai_director_1`). Compare canonical IDs or AI candidates become invisible.
  const canonicalBriefId = value => String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const aiPlans = Array.isArray(aiCreativeBriefs) ? aiCreativeBriefs.slice(0, 3).filter(
    brief => brief?.generatedBy === 's6-creative-director-v1' &&
      brief?.roles && slots.every(slot => Array.isArray(brief.roles[slot]))
  ) : [];
  const briefs = [...aiPlans, ...house, ...neighborHouse, ...references]
    .map(adaptLayer).filter(Boolean).slice(0, 10);
  if (!briefs.length) return { status: 'no_feasible_looks', outfits: [], diagnostics: { visualVerdict: 'NOT_RUN' } };
  const priceProfile = buildPriceProfile(catalog,valueFxRates);
  const retrieval = await shortlistRetrievalDeck({
    request: { ...request, requiredIds }, catalog, hardGate, priceToBudgetCurrency,
    creativeDirector: async () => briefs,
    options: { fashion: makeFashionScorer(context, reference, Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : [], {
      priceProfile, budgeted: !!request.budget, layerOptional: context.optionalLayer === true && !forceLayer,
      requiredIds, feedbackTaste,
    }), perRole: 8, perBrief: 100, maxVisual: 72 },
  });
  // A single cross-slot veto applies to selection AND visual challengers.
  // The pre-reviewed critic must never fill a rejected outfit from another
  // path after an NVIDIA verdict. Retain exact SKU and feasibility admission.
  const editorialVetoes = {};
  const selectable = retrieval.shortlist.filter((look) => {
    const reason = editorialRegisterVeto(look, context);
    if (!reason) return true;
    editorialVetoes[reason] = (editorialVetoes[reason] || 0) + 1;
    return false;
  });
  const deckContext = { ...context, requiredIds };
  rerankFashionCandidates(selectable, deckContext);
  const layerMix = { three: 0, four: 0 };
  for (const candidate of selectable) layerMix[candidate.items.length === 3 ? 'three' : 'four'] += 1;
  const ordered = selectable.map((c) => c.fashion?.score).filter(Number.isFinite).sort((a, b) => b - a);
  let selected = chooseDeck(selectable, deckContext);
  const floor = applyExclusionQualityFloor(
    selected, ordered[0] ?? NaN, (retrieval.diagnostics.exclusions?.requested || 0) > 0);
  selected = floor.kept;
  let visualVerdict = 'NOT_RUN', visualWinner;
  const visualEvidenceById = new Map();
  const visualReferenceModeById = new Map();
  let visualReviewCoverage = null;
  let visualQualityDecisions = null;
  // Reference-optional review: a deck with no matching reference brief is
  // still visually judged on its product images alone (the critic marks such
  // verdicts referenceless and never claims a reference comparison). With a
  // reference, behavior is unchanged.
  const permitted = async (look) => {
    try { return await canJudgeOutfit(look, reference?.referenceId ?? null) === true; } catch { return false; }
  };
  if (visualDeckReview && selected.length && typeof visualCritic === 'function' &&
      typeof canJudgeOutfit === 'function') {
    // Budgeted and source-verified finalist pool. Expand beyond the three
    // displayed directions: one timed-out NVIDIA call and one real fashion
    // rejection must not force a two-look deck when better alternatives exist.
    // The critic still owns every approval and enforces an eight-call paid cap.
    const candidates = [];
    for (const look of selected) {
      if (await permitted(look)) candidates.push(look);
    }
    const metrics = creativeMetrics(deckContext);
    const leadScore = selected[0].fashion?.score ?? -Infinity;
    // A Creative Director is not useful if its inventory-grounded designs
    // are never shown to the visual critic. Reserve at most one challenge per
    // accepted AI plan BEFORE ordinary lookalike challengers consume the
    // eight-call NVIDIA limit. All candidates have passed exact SKU, hard
    // budget/occasion/retail admission and must also pass photo permission.
    // The AI choices enter critique, NOT the displayed deck. Only the
    // independent visual gate decides whether a lower-ranked but novel plan
    // deserves to displace a familiar safe house combination.
    const aiBriefIds = new Set(aiPlans.map(plan => canonicalBriefId(plan.id)));
    for (const plan of aiPlans) {
      if (candidates.length >= MAX_VISUAL_CANDIDATES) break;
      const planned = selectable.filter(look =>
        look.briefId === canonicalBriefId(plan.id) && !candidates.some(other => other.id === look.id) &&
        Number.isFinite(look.fashion?.score))
        .sort((a,b) => b.fashion.score - a.fashion.score);
      for (const look of planned.slice(0, 6)) {
        if (await permitted(look)) { candidates.push(look); break; }
      }
    }

    // Recompute marginal direction gain after EVERY alternate. The previous
    // one-time four-candidate sort could fill its only spare slot with a
    // redundant outfit, while all genuinely distinct challengers were hidden.
    while (candidates.length < MAX_VISUAL_CANDIDATES) {
      const current = candidates.length ? candidates : selected;
      const capacity = visualArchitectureCapacity(current, requiredIds);
      const rescueTriples = visualRescueTripleCount(current, requiredIds);
      const pinned = new Set(requiredIds.map(String));
      const usedTopIds = new Set(current.flatMap(({ items }) => (items || [])
        .filter(it => it.macroCategory === 'top' && !pinned.has(String(it.id)))
        .map(it => String(it.id))));
      const extras = rankVisualChallengerPool(selectable
        .filter(look => !candidates.some(chosen => chosen.id === look.id) &&
          Number.isFinite(look.fashion?.score) &&
          look.fashion.score >= leadScore - VISUAL_CHALLENGER_QUALITY_WINDOW &&
          // A challenger only needs to be materially different from SOME
          // existing direction. Near substitutes can replace a visually
          // rejected garment; the selector still requires final pairwise
          // independence, and the reviewer never approves an unjudged card.
          current.some(other => Math.max(other.items.length, look.items.length) - overlaps(other, look) >= 2))
        .map(look => ({
          look,
          capacityGain: visualArchitectureCapacity([...current, look], requiredIds) - capacity,
          rescueGain: visualRescueTripleCount([...current, look], requiredIds) - rescueTriples,
          unseenTop: look.items.some(it => it.macroCategory === 'top' &&
            !pinned.has(String(it.id)) && !usedTopIds.has(String(it.id))) ? 1 : 0,
          difference: Math.min(...current.map(other =>
            directionDifference(other, look, metrics.featuresOf, metrics.featureDistance).distance)),
        }))
      );
      let alternate = null;
      for (const contender of extras) {
        if (await permitted(contender)) { alternate = contender; break; }
      }
      if (!alternate) break;
      candidates.push(alternate);
    }
    if (candidates.length) {
      const reviewed = await evaluateVisualShortlist(
        { ...retrieval, shortlist: candidates }, request, visualCritic,
        { target: Math.min(3, candidates.length), minNovelty: 0.12, requiredIds,
          requireDistinctArchitectures: true,
          // Among already model-approved sets, a true signature/refresh/
          // discovery triple may outrank a near-tie with an "alternative"
          // third. This changes the chosen garments, not merely card labels,
          // without weakening any NVIDIA or editorial qualification gate.
          directionDeckBonus: (looks) => {
            if(looks.length!==3)return 0;
            const directions=labelCreativeDirections(looks,creativeMetrics(deckContext));
            return directions.every((direction)=>direction.qualified) ? 0.12 : 0;
          },
          preferValue: request.sourceMode === 'catalog' && !request.budget &&
            !requiredIds.length && !explicitlyRequestsLuxury(context) },
      );
      visualQualityDecisions = Array.isArray(reviewed.diagnostics.visualQualityDecisions)
        ? reviewed.diagnostics.visualQualityDecisions.slice(0, MAX_VISUAL_CANDIDATES) : [];
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
          if (approved.referenceMode) visualReferenceModeById.set(approved.id, approved.referenceMode);
        }
      }
      if (reviewed.diagnostics.visualValid > 0) {
        selected = reviewed.outfits.map(({ id }) => byId.get(id)).filter(Boolean);
      }
      let unreviewedFillAdded = 0;
      if (allowUnreviewedVisualFill && request.sourceMode === 'catalog' &&
          requiredIds.length === 0 && selected.length === 2 &&
          (reviewed.diagnostics.visualApproved ?? 0) >= 2) {
        // All cards in `selectable` passed the source, image, identity,
        // price/budget, weather, style and merchant gates already. The only
        // eligible third is UNREVIEWED, never a photo/model failure from the
        // batch, and it must be structurally distinct from BOTH approved
        // cards. This is a disclosed ranking fallback, not an AI approval.
        const judgedIds = new Set(candidates.map(look => look.id));
        const ranked = [...selectable].sort((a,b) =>
          (b.fashion?.score ?? -Infinity) - (a.fashion?.score ?? -Infinity));
        const minScore = (selected[0].fashion?.score ?? -Infinity) - VISUAL_CHALLENGER_QUALITY_WINDOW;
        // One unjudged fallback, NEVER an image-inaccessible fallback.
        // Check at most twelve structurally eligible alternatives to bound
        // CDN requests/latency and leave NVIDIA's eight paid calls untouched.
        let inspected=0;
        for(const next of ranked){
          if(judgedIds.has(next.id) || !Number.isFinite(next.fashion?.score) ||
              next.fashion.score < minScore ||
              !selected.every(other => independentOutfitDirection(next,other,requiredIds) &&
                Math.max(next.items.length,other.items.length) - overlaps(next,other) >= 2))continue;
          if(++inspected>12)break;
          if(typeof canDisplayOutfit==='function'){
            let available=false;
            try {available=await canDisplayOutfit(next)===true;}catch{available=false;}
            if(!available)continue;
          }
          selected=[...selected,next];unreviewedFillAdded=1;break;
        }
      }
      visualReviewCoverage = {
        mode: 'displayed_deck',
        attempted: reviewed.diagnostics.visualAttempted,
        valid: reviewed.diagnostics.visualValid,
        rejected: rejected.size,
        approved: reviewed.diagnostics.visualApproved,
        // Only an approved, actually displayed look has review evidence.
        displayedEvidence: selected.filter(look => visualEvidenceById.has(look.id)).length,
        aiCreativeDirectorJudged: candidates.filter(look => aiBriefIds.has(look.briefId)).length,
        aiCreativeDirectorDisplayed: selected.filter(look => aiBriefIds.has(look.briefId)).length,
        fallbackUnreviewed: reviewed.diagnostics.visualValid === 0 || unreviewedFillAdded > 0,
        unreviewedFillAdded,
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
        selected = chooseDeck(selectable, deckContext, rejected);
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
        selected = chooseDeck(selectable, deckContext, rejected);
      }
    }
  }
  const directions = labelCreativeDirections(selected, creativeMetrics(deckContext));
  const copy = editorialDeckCopy(selected, context, directions);
  return {
    status: selected.length === 3 ? 'complete' : selected.length ? 'partial' : 'no_feasible_looks',
    outfits: selected.map((look, index) => {
      return {
        id: look.id, items: look.items, description: copy[index].description,
        concept: { ...copy[index].concept, referenceId: reference?.referenceId ?? null },
        direction: directions[index],
        fashion: look.fashion,
        ...(look.fashionQuality || look.fashion?.fashionQuality ? {
          fashionQuality: look.fashionQuality || look.fashion?.fashionQuality,
        } : {}),
        ...(visualReferenceModeById.has(look.id)
          ? { visualReferenceMode: visualReferenceModeById.get(look.id) } : {}),
        ...(visualEvidenceById.has(look.id)
          ? { visualEvidence: visualEvidenceById.get(look.id) }
          : visualWinner?.id === look.id ? { visualEvidence: visualWinner.evidence } : {}),
      };
    }),
    diagnostics: { ...retrieval.diagnostics, editorialRegister: { version: EDITORIAL_REGISTER_VERSION, rejected: editorialVetoes, eligible: selectable.length }, visualVerdict,
      directionVersion: DIRECTION_VERSION,
      editorialQualityVersion: EDITORIAL_QUALITY_VERSION,
      fashionQualityVersion: FASHION_QUALITY_VERSION,
      fashionFinishVersion: FASHION_FINISH_VERSION,
      conceptFidelityVersion: CONCEPT_FIDELITY_VERSION,
      deckCopyVersion: DECK_COPY_VERSION,
      feedbackTasteVersion: FEEDBACK_TASTE_VERSION,
      feedbackTasteActive: !!(feedbackTaste && feedbackTaste.active),
      ...(visualReviewCoverage ? { visualReviewCoverage } : {}),
      ...(visualQualityDecisions ? { visualQualityDecisions } : {}),
      directions: directions.map((d) => d.id),
      qualifiedDirections: directions.filter((d) => d.qualified).length,
      aiCreativeDirectorBriefs: aiPlans.length,
      aiCreativeDirectorEligible: selectable.filter(look => aiPlans.some(plan => look.briefId === canonicalBriefId(plan.id))).length,
      aiCreativeDirectorDisplayed: selected.filter(look => aiPlans.some(plan => look.briefId === canonicalBriefId(plan.id))).length,
      neighborBriefs: neighborHouse.length,
      shortlistMix: { ...layerMix, best: ordered[0] ?? null, top20cutoff: ordered[19] ?? null },
      ...(floor.limited ? { exclusionQualityLimited: true } : {}) },
  };
}
