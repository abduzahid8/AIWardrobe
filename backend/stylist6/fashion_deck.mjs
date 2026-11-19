// Experimental catalog/wardrobe recommendation path. Callers own inventory,
// permission and provider admission. Local styling never requires image upload.
import { shortlistRetrievalDeck, evaluateVisualShortlist, selectVisualDeck, independentOutfitDirection, outerwearArchetype } from './prototype.mjs';
import { sourceBoundCompositionPool, sourceVerifyComposedOutfits } from './sku_composer_ai.mjs';
import {trainPhotoFeedbackTeacher,photoTaughtRescueOrder} from './photo_feedback_teacher.mjs';
import {anonymousPhotoJournalAllowed} from './photo_reward_journal.mjs';
import { MAX_VISUAL_CANDIDATES, verifiedAvoidColors } from './visual_critic_hosted.mjs';
import { baseProductId } from './catalog_adapter.mjs';
import { describeGarment, prepareGarment } from '../../src/domain/garmentDescriptor.ts';
import { currentVisual, confidentVisual, visualAdmissionReason } from '../../src/domain/visualFashion.ts';
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
import { deliberatelyRelaxedIntent } from './editorial_intent.mjs';

const slots = ['top', 'outerwear', 'bottom', 'shoes'];

/** Early staging PHOTO admission for the SAME four-real-SKU catalog contract
 * already enforced at cloud publication. Avoid wasting paid vision capacity
 * on an inherently unpublishable three-piece candidate; never fabricate one.
 */
export function eligibleCatalogFourItemPhotoCandidate(look){
  const items=look?.items;
  return Array.isArray(items)&&items.length===4&&
    slots.every(slot=>items.filter(item=>item?.macroCategory===slot).length===1)&&
    items.every(item=>item?.isShopItem===true&&typeof item.id==='string'&&item.id.length>0)&&
    new Set(items.map(item=>item.id)).size===4;
}


/**
 * Learned PHOTO-history creative audition of WHOLE garment relationships.
 *
 * The primary source-bound AI already chose real TOP/LAYER/BOTTOM/SHOES sets.
 * When (and ONLY when) its cross-request, same-scene PHOTO outcome teacher
 * generalizes on independent held-out examples, consider NEW source-product
 * pairings from those four verified item pools. The proposals are NOT
 * product recommendations: source/budget/weather/merchant feasibility,
 * image-usage authorization and the independent whole-outfit PHOTO critic
 * must all separately approve them. No rule about which garment goes with
 * another is encoded here; the prior learns the relationship.
 *
 * All photographed input is already in the existing trusted item catalog;
 * nothing (image URL, pixel, merchant identifier, model weight) is uploaded,
 * persisted or sent to another provider by this pure function.
 */
export function photoHistoryRecompositionCandidates(
  initial,scoreFeatures,{limit=8,featuresOf=items=>outfitStylingFeatures(items),
    sourceMenu=[]}={}
){
  if(!Array.isArray(initial)||initial.length<3||initial.length>8||
      typeof scoreFeatures!=='function'||typeof featuresOf!=='function'||
      !Number.isInteger(limit)||limit<1||limit>16)return [];
  const authorized=initial.filter(eligibleCatalogFourItemPhotoCandidate);
  if(authorized.length<3)return [];
  const original=new Set(authorized.map(look=>
    look.items.map(item=>baseProductId(String(item.id))).sort().join('|')));
  const proposals=new Map();
  const roleOf=look=>Object.fromEntries(
    slots.map(slot=>[slot,look.items.find(item=>item.macroCategory===slot)]));
  const roles=authorized.map(roleOf);
  const score=items=>{
    try{
      const value=scoreFeatures(featuresOf(items));
      return Number.isFinite(value)&&value>=-1&&value<=1?value:null;
    }catch{return null}
  };
  // A complete outfit inherits two coherent actual SKU relationships
  // from each independent artist direction, then the learned full-outfit
  // reward decides whether the NEW four-way combination looks promising.
  // This is bounded pairing search, not a fixed costume template.
  const halves=[
    ['top','outerwear'],['top','bottom'],['top','shoes'],
    ['outerwear','bottom'],['outerwear','shoes'],['bottom','shoes'],
  ];
  const consider=(items,{positivePhotoSignal=false}={})=>{
    if(items.some(item=>!item||item.isShopItem!==true)||
        items.some((item,i)=>items.slice(0,i).some(other=>
          sameProduct(item,other))))return;
    const key=items.map(item=>baseProductId(String(item.id))).sort().join('|');
    if(original.has(key)||proposals.has(key))return;
    const estimatedPhotoQuality=score(items);
    // A completely UNSEEN four-SKU combination can have a neutral score
    // because none of its features appeared in training. Don't spend a
    // PHOTO slot pretending this zero is a learned positive hypothesis.
    // Previously invented by the actual model source drafts remain legal.
    if(estimatedPhotoQuality===null||
        (positivePhotoSignal&&estimatedPhotoQuality<=0))return;
    proposals.set(key,{items,estimatedPhotoQuality});
  };
  for(let a=0;a<roles.length;a++)for(let b=0;b<roles.length;b++){
    if(a===b)continue;
    for(const half of halves){
      const split=new Set(half);
      consider(slots.map(slot=>(split.has(slot)?roles[a]:roles[b])[slot]));
    }
  }
  // The old learned creator could only RECOMBINE the six original AI looks,
  // and thus could never discover a wonderful source garment the first
  // artist skipped entirely. Search a tiny, deliberately bounded subset
  // of the 48 ALREADY ADMITTED authentic P-token MENU (max 5 per role,
  // 5^4=625 full looks). The trained four-way PHOTO relation — not a
  // garment/subtype rule — selects promising whole outfits. Every candidate
  // STILL passes the original independent stock, weather, budget, merchant
  // and photo-permission gates AFTER this lightweight exploration.
  //
  // The source menu is PRIVATE and never transmitted, logged or persisted
  // by this inference-only scorer. No rights to send original photo pixels
  // are implied by the existing source item or image-derived features.
  if(Array.isArray(sourceMenu)&&sourceMenu.length<=48){
    const choices=slots.map(slot=>{
      const pool=[...new Map(sourceMenu.filter(entry=>
        entry?.slot===slot&&entry.item?.macroCategory===slot&&
        entry.item?.isShopItem===true&&
        typeof entry.item.id==='string')
        .map(entry=>[baseProductId(String(entry.item.id)),entry.item])).values()];
      if(pool.length<=5)return pool;
      return [...new Set([0,Math.floor((pool.length-1)/4),
        Math.floor((pool.length-1)/2),
        Math.floor((pool.length-1)*3/4),pool.length-1])]
        .map(i=>pool[i]);
    });
    if(choices.every(pool=>pool.length>=3)){
      for(const top of choices[0])for(const outerwear of choices[1])
        for(const bottom of choices[2])for(const shoes of choices[3])
          consider([top,outerwear,bottom,shoes],{positivePhotoSignal:true});
    }
  }
  const ranked=[...proposals.values()].sort((a,b)=>
    b.estimatedPhotoQuality-a.estimatedPhotoQuality||
    a.items.map(x=>x.id).join('|').localeCompare(b.items.map(x=>x.id).join('|')));
  // The PHOTO teacher predicts WHOLE-LOOK quality, but our actual product
  // promises require a beautiful three-LOOK portfolio. Purely selecting
  // individual top-N scores can spend both remaining PHOTO slots on tiny
  // variations of ONE strong top/coat, which can never coexist in the final
  // independently shoppable deck. The first hypothesis retains the model's
  // full artistic freedom and maximum predicted quality. For subsequent
  // hypotheses, use the best positively PHOTO-supported real look which is
  // product-independent from the alternatives already selected, when one
  // exists. This is a set-level selection objective, not a rule against
  // colors, silhouettes, garment categories or any original source SKU.
  //
  // If the permitted inventory has no such option, the next highest scored
  // overlap remains available as a legitimate replacement PHOTO audition.
  const selected=[],remaining=[...ranked];
  while(selected.length<limit&&remaining.length){
    let index=0;
    if(selected.length){
      const independent=remaining.findIndex(candidate=>
        candidate.estimatedPhotoQuality>0&&
        selected.every(previous=>
          candidate.items.every(item=>
            previous.items.every(other=>!sameProduct(item,other)))));
      if(independent>=0)index=independent;
    }
    selected.push(remaining.splice(index,1)[0]);
  }
  return selected;
}

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

/** Count actual source-hash-verified, confidently IMAGE-observed permitted
 * garments in a complete outfit. A merchant name is not photographic proof:
 * blue suit pants may photograph as navy, and a catalog title may omit color.
 * Used to PRIORITIZE scarce hosted vision calls, never bypass its own verdict.
 * Unknown items remain eligible in the ordinary fallback path; locked user
 * garment choices must not be filtered by preferences.
 */
/** User-pinned garments are an explicit instruction, not an unknown
 * merchant offer that has to use up the remaining photo-color evidence
 * quota. A complete three-piece outfit with one pinned owned hero has only
 * TWO independently selected pieces to certify; requiring >=3 is impossible.
 */
export function sufficientPhotographicPaletteEvidence(evidence) {
  return !!evidence && !evidence.blocked && evidence.verified >= Math.min(3,evidence.total) &&
    evidence.verified >= evidence.total-1;
}

export function photographicPaletteEvidence(look, preferences, requiredIds = []) {
  const excluded=new Set(verifiedAvoidColors(preferences?.avoidColors));
  const preferred=new Set(verifiedAvoidColors(preferences?.favoriteColors));
  const pinned=new Set((requiredIds||[]).map(String));
  let total=0, verified=0, preferredCount=0, blocked=0;
  for(const item of look?.items||[]){
    if(pinned.has(String(item.id)))continue;
    total++;
    const observation=currentVisual(item)?.attributes?.color;
    if(!confidentVisual(observation) || typeof observation.value!=='string')continue;
    const color=String(observation.value).toLowerCase();
    const normalized=color==='gray'||color==='charcoal'?'grey':color;
    if(excluded.has(normalized))blocked++;
    else {verified++;if(preferred.has(normalized))preferredCount++;}
  }
  return {total,verified,preferred:preferredCount,blocked,
    // A certified complete outfit outranks uncertain images; a photographer's
    // verified favorite color breaks near-ties, never supersedes style.
    score:blocked?-1:(total?verified/total + .15*preferredCount/total:0)};
}

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
  // Photo Director owns the aesthetic decision in the isolated catalog canary.
  // Keep actual owner exclusions, activity, weather, provenance and formal
  // safety. Older clients and wardrobe/mixed modes retain old behavior.
  const modelOwnsTaste = extras.photoDirectorEnabled === true && context.sourceMode === 'catalog';
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
          rejectForRequestIntent(prepared, context.request, context.weather) ||
          descriptor.slot !== item.macroCategory || descriptor.productKind === 'complete-suit' ||
          garmentContextReason(prepared, context,
            {deferStylistic:modelOwnsTaste}) ||
          rejectForClimate(prepared, context.weather) ||
          (!lockedIds.has(String(prepared.id)) &&
            extremeUnrequestedCatalogPriceReason(prepared, extras.priceProfile, context)) ||
          (!modelOwnsTaste && (
            (!lockedIds.has(String(prepared.id)) &&
              rejectUnrequestedOfficeRegisterPiece(prepared, context)) ||
            rejectForStyleOccasion(prepared, context.style, context.occasion) ||
            rejectForUnsuitableDateNight(descriptor, context, prepared) ||
            rejectQuietSmartLeisureLayer(descriptor, context, prepared) ||
            quietLuxuryHoodieReason(prepared, descriptor, context) ||
            rejectUnrequestedRivieraPiece(descriptor, context, prepared) ||
            rejectPolishedContextMismatch(descriptor, context, prepared)
          ))) return null;
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
      if (!modelOwnsTaste && namedCardiganBriefMismatch(items, brief, context, extras.requiredIds || [])) return null;
      const identity = JSON.stringify([brief?.id || null, ...items.map(({ id }) => id).sort()]);
      if (coherent.has(identity)) return coherent.get(identity);
      const assessment = outfitCoherence(items, context);
      // Leave whole-outfit AESTHETIC clashes to the visual fashion judge in
      // model-led mode. Physical winter/summer mismatch and structural
      // defects still cannot enter the paid model slate.
      const criticalWeatherConflict=assessment.reasons.includes('insulated_layer_with_summer_footwear');
      if ((!modelOwnsTaste && (!assessment.valid || rejectUnpolishedDateNightLook(items, context))) ||
          (modelOwnsTaste && criticalWeatherConflict) ||
          outfitStructureReason(items, false) ||
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
export function visualProductUniqueTripleCount(looks, requiredIds = []) {
  if (!Array.isArray(looks) || looks.length > MAX_VISUAL_CANDIDATES) return 0;
  const pinned=new Set((requiredIds||[]).map(String));
  const shares=(a,b)=>(a?.items||[]).some(item=>
    !pinned.has(String(item.id)) && (b?.items||[]).some(other=>
      !pinned.has(String(other.id)) && sameProduct(item,other)));
  let count=0;
  for(let a=0;a<looks.length;a++)for(let b=a+1;b<looks.length;b++){
    if(shares(looks[a],looks[b]) || !independentOutfitDirection(looks[a],looks[b],requiredIds))continue;
    for(let c=b+1;c<looks.length;c++){
      if(!shares(looks[a],looks[c]) && !shares(looks[b],looks[c]) &&
        independentOutfitDirection(looks[a],looks[c],requiredIds) &&
        independentOutfitDirection(looks[b],looks[c],requiredIds))count++;
    }
  }
  return count;
}

/**
 * Put at least one independent, product-unique TRIPLE into the expensive
 * photo-review slate whenever the admitted catalog can support it.
 *
 * The Fashion Director's six ranked choices are strongly preferred, but its
 * output alone is not proof that those choices can coexist as three distinct
 * outfits (its original prompt could not see repeated SKUs). This bounded
 * optimization only assembles already admissible looks; the actual photo
 * critic retains sole authority to approve them. No hidden third-card fill.
 */
export function chooseIndependentPhotoReviewTrio(looks, modelChoices = [], requiredIds = [], metrics) {
  if (!Array.isArray(looks) || !Array.isArray(modelChoices)) return [];
  const pool=looks.slice(0,72).filter(look=>Array.isArray(look?.items) &&
    Number.isFinite(look.fashion?.score));
  if(pool.length<3)return [];
  const preferred=new Map(modelChoices.map((look,i)=>[look.id,i]));
  const pinned=new Set((requiredIds||[]).map(String));
  const eligible=(a,b)=>{
    if(!independentOutfitDirection(a,b,requiredIds))return false;
    if(a.items.some(x=>!pinned.has(String(x.id)) && b.items.some(y=>
      !pinned.has(String(y.id)) && sameProduct(x,y))))return false;
    if(metrics && directionDifference(a,b,metrics.featuresOf,metrics.featureDistance).distance<0.12)
      return false;
    return true;
  };
  const edges=Array.from({length:pool.length},()=>new Uint8Array(pool.length));
  for(let i=0;i<pool.length;i++)for(let j=i+1;j<pool.length;j++)
    edges[i][j]=eligible(pool[i],pool[j])?1:0;
  let winner=null,best=-Infinity;
  for(let i=0;i<pool.length;i++)for(let j=i+1;j<pool.length;j++){
    if(!edges[i][j])continue;
    for(let k=j+1;k<pool.length;k++){
      if(!edges[i][k]||!edges[j][k])continue;
      const trio=[pool[i],pool[j],pool[k]];
      const modelBonus=trio.reduce((sum,look)=>sum+(preferred.has(look.id)?
        0.22+0.06*(1-preferred.get(look.id)/Math.max(1,modelChoices.length)):0),0);
      const archetypes=new Set(trio.map(x=>outerwearArchetype(x,requiredIds)).filter(Boolean));
      const quality=trio.reduce((sum,x)=>sum+x.fashion.score,0);
      const utility=quality+modelBonus+0.12*archetypes.size;
      if(utility>best){best=utility;winner=trio;}
    }
  }
  return winner||[];
}

/**
 * Give the FIRST expensive original-photo evaluations a viable three-look
 * product-independent opportunity, rather than sampling three seductive
 * variations of one original outfit and learning too late that even three
 * approvals cannot form three different shopping cards.
 *
 * Only reorders fully SOURCE/price/weather admitted model creations and
 * approved-to-be-photographed alternatives; no extra item, category ban,
 * photo approval, score threshold, or provider invocation is introduced.
 * A single AI direction never becomes a forced anchor. If the existing
 * first three are already independent, retain the designer's exact order.
 * If no independent trio exists, retain the truthfully partial audition.
 */
export function prioritizeIndependentPhotoAuditions(looks,requiredIds=[],metrics){
  if(!Array.isArray(looks)||looks.length<3||
      looks.length>MAX_VISUAL_CANDIDATES)return looks;
  if(visualProductUniqueTripleCount(looks.slice(0,3),requiredIds)>0)return looks;
  const core=chooseIndependentPhotoReviewTrio(
    looks,looks.slice(0,3),requiredIds,metrics);
  if(core.length!==3||visualProductUniqueTripleCount(core,requiredIds)===0)
    return looks;
  const selected=new Set(core.map(look=>look.id));
  return [...core,...looks.filter(look=>!selected.has(look.id))];
}

/**
 * Build a PHOTO-REVIEW PORTFOLIO, not six stylistically identical model
 * favorites. When three independent whole looks exist, keep them as
 * foundation; then spend remaining scarce NVIDIA slots on strong,
 * product-unique and structurally distinct alternatives. Every input still
 * comes from the server-vetted shortlist; no look is approved here.
 */
export function assembleIndependentVisualPortfolio(
  looks, modelChoices = [], requiredIds = [], metrics, limit = MAX_VISUAL_CANDIDATES,
) {
  if(!Array.isArray(looks)||!Array.isArray(modelChoices)||
      !Number.isInteger(limit)||limit<3||limit>MAX_VISUAL_CANDIDATES)return [];
  const pool=looks.slice(0,72).filter(x=>Array.isArray(x?.items)&&
    Number.isFinite(x.fashion?.score));
  const core=chooseIndependentPhotoReviewTrio(pool,modelChoices,requiredIds,metrics);
  const selected=[...core], ids=new Set(selected.map(x=>x.id));
  const nominated=new Set(modelChoices.map(x=>x.id));
  const pinned=new Set((requiredIds||[]).map(String));
  const sharesProduct=(a,b)=>a.items.some(item=>!pinned.has(String(item.id)) &&
    b.items.some(other=>!pinned.has(String(other.id))&&sameProduct(item,other)));
  // Skip weak one-off concepts before expensive vision when strong, fully
  // documented alternatives exist in the same current inventory.
  const ceiling=Math.max(...pool.map(x=>x.fashion.score));
  while(selected.length<limit){
    let next=null,best=-Infinity;
    const currentTriples=visualProductUniqueTripleCount(selected,requiredIds);
    for(let pass=0;pass<3&&!next;pass++){
      for(const look of pool){
        if(ids.has(look.id)||look.fashion.score < ceiling-(pass===2?0.25:0.20))continue;
        const shared=selected.filter(x=>sharesProduct(x,look)).length;
        const independent=selected.filter(x=>
          independentOutfitDirection(x,look,requiredIds)).length;
        const minDiff=metrics&&selected.length?Math.min(...selected.map(x=>
          directionDifference(x,look,metrics.featuresOf,metrics.featureDistance).distance)):1;
        if(pass===0 && (shared>0||independent!==selected.length||minDiff<0.12))continue;
        if(pass===1 && (shared>0||independent===0))continue;
        if(pass===2 && selected.length&&independent===0)continue;
        const layer=outerwearArchetype(look,requiredIds);
        const existingLayers=new Set(selected.map(x=>outerwearArchetype(x,requiredIds)));
        const confidentPhotos=look.items.filter(x=>currentVisual(x)).length/
          Math.max(1,look.items.length);
        const hypothetical=[...selected,look];
        // Optimize actual independently combinable TRIPLES rather than
        // incidental color/role novelty. The previous selection spent paid
        // review slots on 5 individually approved training outfits but had
        // only 2 mutually compatible complete cards after an NVIDIA veto.
        // A marginal three-look combination provides a real replacement
        // option. A robust portfolio should also retain triples when any
        // one candidate is rejected by photo judgement.
        const tripleGain=visualProductUniqueTripleCount(
          hypothetical,requiredIds)-currentTriples;
        const afterOneVeto=visualPortfolioSingleVetoResilience(
          hypothetical,requiredIds);
        const resilience=0.028*Math.min(14,tripleGain)+
          0.011*Math.min(12,afterOneVeto);
        const score=look.fashion.score +
          (nominated.has(look.id)?0.07:0) +
          (layer&&!existingLayers.has(layer)?0.065:0) +
          0.035*confidentPhotos +
          0.025*independent + resilience -
          0.4*shared;
        if(score>best){best=score;next=look;}
      }
    }
    if(!next)break;
    selected.push(next);ids.add(next.id);
  }
  // If the catalog genuinely lacks an independent triple, preserve the
  // model shortlist for an honest partial visual review; no fake 3-card fill.
  if(!core.length)return modelChoices.slice(0,limit);
  return selected;
}

/** Formal tuxedo/suit requests constrain permissible garment architecture.
 * If every genuinely formal option has the same shirt/blazer/trouser/oxford
 * roles, the ordinary pre-photo architecture test is correctly inconclusive,
 * NOT proof that the photographed styles are identical. Reserve 8 independent
 * REAL merchandise sets for the photo critic to inspect; the final three
 * still need actual-photo evidence of a jacket shift plus a second core
 * apparel color shift.
 * No extra provider calls, no SKU invention, no unreviewed third card.
 */
export function assembleFormalPhotoProbePortfolio(
  looks, modelChoices = [], requiredIds = [], limit = MAX_VISUAL_CANDIDATES,
) {
  if(!Array.isArray(looks)||!Array.isArray(modelChoices)||
      !Number.isInteger(limit)||limit<3||limit>MAX_VISUAL_CANDIDATES)return [];
  const pinned=new Set((requiredIds||[]).map(String));
  const pool=looks.slice(0,72).filter(x=>Array.isArray(x?.items)&&
    Number.isFinite(x.fashion?.score) && ['top','outerwear','bottom','shoes'].every(slot=>
      x.items.some(item=>item.macroCategory===slot)));
  const disjoint=(a,b)=>!(a.items||[]).some(item=>
    !pinned.has(String(item.id)) && (b.items||[]).some(other=>
      !pinned.has(String(other.id))&&sameProduct(item,other)));
  const chosenByAI=new Map(modelChoices.map((look,i)=>[look.id,i]));
  // Only trustworthy current-image observations (or clearly sourced catalog
  // color fields) GUIDE pre-review variety. They never APPROVE a direction.
  const color=(look,slot)=>{
    const item=look.items.find(x=>x.macroCategory===slot);
    if(!item)return null;
    const profile=currentVisual(item),attr=profile?.attributes?.color;
    if(confidentVisual(attr)&&typeof attr.value==='string'&&attr.value!=='unknown')
      return attr.value;
    const raw=String(item.color||'').toLowerCase();
    return raw && raw!=='unknown'&&raw!=='neutral'?raw:null;
  };
  const photoPaletteBonus=trio=>{
    let bonus=0;
    for(const slot of ['outerwear','bottom','top']){
      const colors=trio.map(x=>color(x,slot)).filter(Boolean);
      bonus+=(slot==='top'?0.04:0.075)*new Set(colors).size;
    }
    return bonus;
  };
  let core=null,best=-Infinity;
  // <=72 candidates => <=60k triples, bounded without N+1 SKU searches.
  for(let i=0;i<pool.length;i++)for(let j=i+1;j<pool.length;j++){
    if(!disjoint(pool[i],pool[j]))continue;
    for(let k=j+1;k<pool.length;k++){
      if(!disjoint(pool[i],pool[k])||!disjoint(pool[j],pool[k]))continue;
      const triple=[pool[i],pool[j],pool[k]];
      const utility=triple.reduce((sum,look)=>sum+look.fashion.score+
        (chosenByAI.has(look.id)?0.09:0),0)+photoPaletteBonus(triple);
      if(utility>best){best=utility;core=triple;}
    }
  }
  if(!core)return []; // Never claim availability when exact SKU choice is thin.
  const result=[...core],ids=new Set(core.map(x=>x.id));
  while(result.length<limit){
    let next=null,highest=-Infinity;
    for(const candidate of pool){
      if(ids.has(candidate.id)||!result.every(x=>disjoint(x,candidate)))continue;
      const newPhotoPalette=photoPaletteBonus([...result,candidate])-
        photoPaletteBonus(result);
      const novelty=0.04*['outerwear','bottom'].filter(slot=>{
        const c=color(candidate,slot);
        return c&&!result.some(x=>color(x,slot)===c);
      }).length;
      const score=candidate.fashion.score+newPhotoPalette+novelty+
        (chosenByAI.has(candidate.id)?0.09:0);
      if(score>highest){highest=score;next=candidate;}
    }
    if(!next)break;
    result.push(next);ids.add(next.id);
  }
  // Fill remaining paid critique slots with plausible near substitutes only
  // if the independent SKU inventory itself is genuinely exhausted. The
  // final display still forbids reuse unless the user pinned an anchor.
  for(const look of modelChoices){
    if(result.length>=limit)break;
    if(!ids.has(look.id)&&pool.some(x=>x.id===look.id)){
      result.push(look);ids.add(look.id);
    }
  }
  return result;
}

/** Choose a real, SECOND set of formal ensembles only after the whole-look
 * photo critic has failed to approve three from the first wave. Each new
 * outfit uses different actual merchandise from the APPROVED first wave;
 * all remain unreviewed until the independent NVIDIA follow-up.
 * This is a shortlist composition step, not a styling verdict.
 */
export function chooseFormalPhotoRescueCandidates(
  pool, firstBatch, firstApproved, requiredIds=[], limit=4,
) {
  if(!Array.isArray(pool)||!Array.isArray(firstBatch)||!Array.isArray(firstApproved)||
    !Number.isInteger(limit)||limit<1||limit>4)return [];
  const judged=new Set(firstBatch.map(x=>x?.id));
  const pinned=new Set((requiredIds||[]).map(String));
  const shares=(a,b)=>(a?.items||[]).some(item=>
    !pinned.has(String(item.id))&&(b?.items||[]).some(other=>
      !pinned.has(String(other.id))&&sameProduct(item,other)));
  const eligible=pool.filter(x=>x?.id&&!judged.has(x.id)&&
    Number.isFinite(x?.fashion?.score) && Array.isArray(x?.items)&&
    firstApproved.every(a=>!shares(x,a)));
  // A new source-verified portfolio of actual suits (not a synthetic third).
  const probe=assembleFormalPhotoProbePortfolio(eligible,[],requiredIds,Math.max(3,limit));
  if(probe.length>=1)return probe.slice(0,limit);
  // Sparse real inventory: one or two independent outfits can still improve
  // the final collection; never reintroduce a rejected first-wave garment.
  const chosen=[];
  for(const look of eligible.sort((a,b)=>b.fashion.score-a.fashion.score)){
    if(chosen.every(other=>!shares(look,other))){
      chosen.push(look);
      if(chosen.length>=limit)break;
    }
  }
  return chosen;
}

/** Enumerate EXACTLY feasible complete decks from the independent NVIDIA
 * verdicts before asking the final AI to make an artistic portfolio decision.
 * The AI sees only opaque L- and F-tokens. Every 3/2/1-look menu entry has
 * already passed actual-photo quality, permitted SKU uniqueness, wardrobe
 * anchors, visual novelty and (when applicable) photographed formal-palette
 * proof. Do not ask the model for impossible combinations then reject it.
 */
export function verifiedPhotoEditorialPortfolios(
  approvedLooks,critiques,reviewOptions,{triples=24,pairs=12,singles=4}={},
){
  if(!Array.isArray(approvedLooks)||!Array.isArray(critiques)||
     approvedLooks.length>MAX_VISUAL_CANDIDATES)return [];
  const quality=new Map(critiques.map(r=>[r.candidateId,
    (Number(r.visualCohesion)||0)+(Number(r.styleFit)||0)+
    (Number(r.occasionFit)||0)+(Number(r.formalityFit)||0)]));
  // At most C(8,3)+C(8,2)+C(8,1)=92 exact short-list replays.
  // These are local pure portfolio checks, NOT extra NVIDIA inferences.
  const groups={3:[],2:[],1:[]};
  for(let i=0;i<approvedLooks.length;i++){
    groups[1].push([i]);
    for(let j=i+1;j<approvedLooks.length;j++){
      groups[2].push([i,j]);
      for(let k=j+1;k<approvedLooks.length;k++)groups[3].push([i,j,k]);
    }
  }
  const candidates=[];
  for(const size of [3,2,1])for(const inds of groups[size]){
    const shortlist=inds.map(n=>approvedLooks[n]);
    const verified=selectVisualDeck({
      generationVersion:'s6-photo-editor-proof',shortlist,
      diagnostics:{},
    },critiques,{...reviewOptions,target:shortlist.length,
      onCritiques:undefined});
    if(verified.outfits.length!==shortlist.length||
      !shortlist.every(x=>verified.outfits.some(y=>y.id===x.id)))continue;
    candidates.push({
      size,
      indices:verified.outfits.map(x=>approvedLooks.findIndex(y=>y.id===x.id)),
      strength:shortlist.reduce((sum,x)=>sum+(quality.get(x.id)||0),0)/
        shortlist.length,
    });
  }
  const menu=[];
  for(const size of [3,2,1]){
    const cap=size===3?triples:size===2?pairs:singles;
    const choice=candidates.filter(c=>c.size===size)
      .sort((a,b)=>b.strength-a.strength||
        a.indices.join(',').localeCompare(b.indices.join(',')))
      .slice(0,cap);
    for(const c of choice){
      menu.push({
        portfolioToken:'F'+String(menu.length).padStart(2,'0'),
        orderedLooks:c.indices.map(n=>'L'+String(n).padStart(2,'0')),
        size:c.size,
      });
    }
  }
  return menu;
}

/** How many independent triples survive the worst *single* photo veto?
 * Zero means the eight-candidate slate still has a fragile one-SKU bottleneck.
 * Uses the EXACT final independent-product deck feasibility semantics.
 */
export function visualPortfolioSingleVetoResilience(looks, requiredIds = []) {
  if(!Array.isArray(looks)||looks.length<4||looks.length>MAX_VISUAL_CANDIDATES)return 0;
  return Math.min(...looks.map((_,i)=>visualProductUniqueTripleCount(
    looks.filter((_,j)=>j!==i),requiredIds)));
}

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
    .sort((a,b) => (b.uniqueTripleGain || 0) - (a.uniqueTripleGain || 0) ||
      (b.rescueGain || 0) - (a.rescueGain || 0) ||
      (b.capacityGain || 0) - (a.capacityGain || 0) ||
      (b.outerwearGain || 0) - (a.outerwearGain || 0) ||
      (b.unseenTop || 0) - (a.unseenTop || 0) ||
      // A validated PHOTO reward prior can prefer the more visually
      // promising COMPLETE garment relationship among otherwise equally
      // feasible, equally distinct options; never overrides real PHOTO veto.
      (Number.isFinite(b.photoHistoryScore)?b.photoHistoryScore:0) -
        (Number.isFinite(a.photoHistoryScore)?a.photoHistoryScore:0) ||
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

/** A polished catalog dinner at 12–17°C needs an intentional outer layer.
 * The weather policy already treats <18°C as cool, but the cloud's default
 * `optionalLayer:true` + the narrower `hasColdLayerIntent(<12)` let the
 * house concepts still nominate bare polo/trousers/boots at 13°C. Respect
 * an explicit request for no jacket and warmer/relaxed scenarios. This is
 * a structural shortlist policy, not fabricated weather or a visual verdict.
 */
export function polishedEveningLayerRequired(context) {
  if(context?.sourceMode!=='catalog')return false;
  const temperature=context.weather?.temp;
  if(!Number.isFinite(temperature)||temperature>=18)return false;
  const style=String(context.style||'').trim().toLowerCase().replace(/[\s-]+/g,'_');
  if(!['semi_classic','classic','old_money','business_casual'].includes(style))return false;
  if(!/\b(date|dinner|romantic|cocktail|evening)\b/i.test(String(context.occasion||'')))return false;
  const text=String(context.request||'');
  if(deliberatelyRelaxedIntent(text))return false;
  if(/\b(?:no|without|skip)\s+(?:(?:an?|any|extra|additional)\s+){0,3}(?:layers?|outerwear|jacket|blazer|coat|cardigan)\b/i.test(text) ||
    /без\s+(?:(?:лишнего|дополнительного)\s+)?(?:пиджака|куртки|пальто|кардигана|верхней\s+одежды|слоёв|слоя)/i.test(text))return false;
  return true;
}

/**
 * Present a diverse, provenance-safe cross-brief sample to the model-led
 * Fashion Director. The model sees ANONYMOUS tokens and attributes observed
 * in current source-hash-verified photos, not merchant names or raw IDs.
 */
export function directorSlatePreview(looks, aiPlans = [], { includePublicCatalogNames = false } = {}) {
  const canon=x=>String(x||'').trim().toLowerCase().replace(/[\s-]+/g,'_');
  const aiKeys=(Array.isArray(aiPlans)?aiPlans:[]).map(p=>canon(p.id));
  const groups=new Map();
  for(const look of Array.isArray(looks)?looks:[]){
    if(!Number.isFinite(look?.fashion?.score)||!Array.isArray(look.items))continue;
    const key=String(look.briefId||'unplanned');
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(look);
  }
  const keys=[...new Set([...aiKeys,...groups.keys()])].filter(k=>groups.has(k));
  const chosen=[],used=new Set();
  for(let round=0;round<9&&chosen.length<28;round++){
    for(const key of keys){
      const arr=groups.get(key)||[];
      const diverse=arr.find(look=>!used.has(look.id)&&chosen.every(prev=>
        overlaps(prev,look)<Math.min(prev.items.length,look.items.length)-1));
      const look=diverse||arr.find(x=>!used.has(x.id));
      if(!look)continue;
      chosen.push(look);used.add(look.id);
      if(chosen.length>=28)break;
    }
  }
  const attrs=['color','subtype','formality','texture','pattern','volume',
    'length','bulk','soleProfile','footwearPurpose','collar'];
  const opaqueProducts=new Map();
  return chosen.map((look,index)=>({
    token:'L'+String(index).padStart(2,'0'),
    direction:aiKeys.includes(String(look.briefId||''))?'ai':'house',
    qualityHint:look.fashion.score,
    items:look.items.map(item=>{
      const profile=currentVisual(item),photo={};
      for(const name of attrs){
        const o=profile?.attributes?.[name];
        if(confidentVisual(o)&&typeof o.value==='string')photo[name]=o.value;
      }
      // Anonymous per-request product identity lets the model recognize when
      // two complete outfits reuse the same item, without revealing its SKU.
      const key=baseProductId(String(item.id));
      if(!opaqueProducts.has(key))opaqueProducts.set(key,
        'P'+String(opaqueProducts.size).padStart(3,'0'));
      // Catalog-only public product labels convey purpose (resort, technical,
      // dress) that photo color/slot vectors cannot establish. Explicitly
      // opt-in only for sourceMode=catalog; no user-owned item names go out.
      const merchantLabel=includePublicCatalogNames && typeof item.name==='string'
        ? item.name.replace(/[<>{}\u0000-\u001f]/g,' ').trim().slice(0,64):'';
      return {slot:item.macroCategory,subtype:item.subtype,
        itemToken:opaqueProducts.get(key),photo,
        ...(merchantLabel?{merchantLabel}:{})};
    }),
    look,
  }));
}

export async function recommendFashionDeck({
  request, catalog, referenceBriefs = [], hardGate, priceToBudgetCurrency, visualCritic, canJudgeOutfit,
  // Optional trusted inventory-grounded AI directions; still subject to the
  // exact same retrieval, price, wardrobe, weather and visual checks.
  aiCreativeBriefs = [],
  // Server-only exact licensed-photo product IDs, source-hash+rights checked
  // against the CURRENT merchant catalog. Never a client styling control.
  photoFirstPriorityIds=null,
  // Server-only AI chooser of REAL outfit finalists (no invented SKU access).
  selectDirectorFinalists,
  // Optional prebuilt model from feedback_taste.mjs (explicit thumbs rows).
  // Absent/inert model = byte-identical legacy ranking (back-compat).
  feedbackTaste,
  // Staging-only, OPTIONAL private PHOTO reward recorder. Records only
  // authenticated independent original-photo examples in a privacy-scrubbed
  // server callback. No effect on ranking, any jury verdict or S5.
  photoRewardSink=null,
  // Optional HELD-OUT calibrated PHOTO history. Only the isolated staging
  // server can supply a pure scoreFeatures() callback. It predicts which
  // authentic SOURCE-feasible candidate to audition with the independent
  // PHOTO judge, never an actual approval or source product requirement.
  photoHistory=null,
  // Passed by the trusted cloud handler only after validating the official
  // CBU snapshot; the app never provides these rates to the ranker.
  valueFxRates,
  // Cloud staging can opt into reviewing all three displayed looks in one
  // bounded provider batch. The isolated local / offline paths stay unchanged.
  visualDeckReview = false,
  // Trusted server-only inference budget: fewer provider calls improve cloud
  // latency and cost while each displayed look still requires its OWN full
  // photographic verdict. Local QA keeps the eight-call ceiling by default.
  visualCandidateLimit = MAX_VISUAL_CANDIDATES,
  // Optional in canary only: the third unreviewed merchant look must actually
  // load every product image, instead of displaying a 403/404/blank tile.
  canDisplayOutfit,
  // Only for isolated staging: after exactly two NVIDIA-approved looks,
  // permit ONE additional source-verified, independently styled finalist
  // that was NOT sent to NVIDIA. It is explicitly NOT_JUDGED in the response;
  // never reintroduce a model-vetoed or timed-out candidate, or invent evidence.
  allowUnreviewedVisualFill = false,
  // Server-owned REAL staging time left, never a client-provided budget.
  // A second paid photo wave is forbidden unless the Render compute still
  // has enough time to return a valid response (not a deadline 504).
  photoRescueRemainingMs = null,
  // Model-level whole-deck EDITORIAL decision after independent garment
  // photo approval. It cannot introduce unapproved merchandise. Any
  // cross-look selection still passes the unchanged selectVisualDeck gates.
  selectPhotoApprovedPortfolio = null,
  // Isolated STAGING-only: the AI creates novel COMPLETE outfits by choosing
  // actual approved SKU tokens, not by choosing the old ranker's prebuilt cards.
  selectCatalogCompositions = null,
  // A second photo-critic-informed CREATIVE SKU pass for insufficient
  // casual/date portfolios (isolated 72s staging only). The server is the
  // sole authority for this paid inference callback and remaining deadline.
  selectCatalogRescueCompositions = null,
}) {
  const reviewedLimit = Number.isInteger(visualCandidateLimit)
    ? Math.max(3,Math.min(MAX_VISUAL_CANDIDATES,visualCandidateLimit)) : MAX_VISUAL_CANDIDATES;
  const context = fashionStylingContext(request);
  const modelLed = context.sourceMode === 'catalog' &&
    visualDeckReview && typeof visualCritic === 'function' &&
    typeof canJudgeOutfit === 'function' && typeof selectDirectorFinalists === 'function';
  // The isolated 72-second catalog publisher requires four real SKUs. Never
  // pay a whole-look PHOTO verdict for three garments which the cloud release
  // boundary must discard. Compose a suitable REAL layer, not a fake fourth.
  const catalogFourItemIntent = modelLed &&
    typeof selectCatalogCompositions === 'function';
  const weather = context.weather;
  const reference = referenceBriefs.find(({ referenceId }) => referenceId === request.referenceId);
  const forceLayer = !context.optionalLayer || hasColdLayerIntent(weather) ||
    context.dressCode === 'formal' || polishedEveningLayerRequired(context);
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
  // Reuse this EXACT scorer for AI-selected new combinations. Its WeakMaps
  // contain the already prepared real catalog items from this retrieval,
  // and its structure/weather constraints cannot be overridden by the LLM.
  const fashionScorer=makeFashionScorer(context,reference,
    Array.isArray(request.recentSignatures)?request.recentSignatures.slice(-32):[],{
      priceProfile,budgeted:!!request.budget,
      layerOptional:context.optionalLayer===true&&!forceLayer,
      requiredIds,feedbackTaste,photoDirectorEnabled:modelLed,
    });
  // THE AI FASHION DESIGNER STARTS BEFORE THE LEGACY LOOK RANKER.
  // No finished formula-look is a prerequisite for an independent designer
  // to select real, photographed garments. Ranker retrieval runs concurrently
  // solely as a source-safe fallback and to populate the proven hard gate
  // scorer. The early AI's returned SKU selections are NEVER displayed until
  // the same independent full-look hard gate and NVIDIA PHOTO judge approve.
  let parallelComposition=null;
  let sourceVerifiedAiFirst=[];
  if(modelLed&&typeof selectCatalogCompositions==='function'&&
      request.sourceMode==='catalog'&&!requiredIds.length){
    const excludedBases=new Set(
      (Array.isArray(request.excludedIds)?request.excludedIds:[])
        .map(id=>baseProductId(String(id))));
    const prepared=Array.isArray(catalog)?catalog.slice(0,650)
      .filter(item=>!excludedBases.has(baseProductId(String(item.id))))
      .map(item=>fashionScorer.prepare(item)).filter(Boolean):[];
    // When the first visual artist is legally enabled, ensure that even
    // source-authorized items OUTSIDE the 650-item ranker window reach the
    // actual artist's limited 48-P-token menu. All original SKU feasibility
    // checks and catalogue product identities remain unchanged.
    const pictured=photoFirstPriorityIds instanceof Set&&Array.isArray(catalog)
      ?catalog.filter(item=>photoFirstPriorityIds.has(String(item.id))&&
        !excludedBases.has(baseProductId(String(item.id))))
        .map(item=>fashionScorer.prepare(item)).filter(Boolean):[];
    const visualSeed=pictured.length
      ?[...new Map([...pictured,...prepared].map(x=>
        [baseProductId(String(x.id)),x])).values()].slice(0,650)
      :prepared;
    const pool=sourceBoundCompositionPool([],{
      additionalItems:visualSeed,inventoryFirst:true,
      explorationSlots:4,directions:aiPlans,feedbackTaste,
      photoFirstPriorityIds,
    });
    const counts=Object.fromEntries(slots.map(slot=>
      [slot,pool.filter(p=>p.slot===slot).length]));
    const enough=['top','bottom','shoes',...(forceLayer?['outerwear']:[])]
      .every(slot=>counts[slot]>=6);
    if(enough){
      const contextPrompt={
        style:context.style,occasion:context.occasion,brief:context.request,
        weatherCelsius:Number.isFinite(context.weather?.temp)?context.weather.temp:null,
        preferences:context.preferences,
      };
      // Invoke the bounded provider request synchronously: its first await
      // yields while the ranker computes. Never leave a dangling rejection.
      let promise;
      try{
        // Start provider I/O NOW, before any retrieval kernel execution.
        promise=Promise.resolve(selectCatalogCompositions({
          pool,directions:aiPlans,outfitCount:6,feedbackTaste,context:contextPrompt,
          layerRequired:forceLayer||catalogFourItemIntent,
             allowLayer:!noLayer||catalogFourItemIntent,parallel:true,
        })).catch(()=>({status:'provider_unavailable',looks:[],providerCalls:1}));
      }catch{
        promise=Promise.resolve({status:'provider_unavailable',
          looks:[],providerCalls:1});
      }
      // IMPORTANT: the legacy retrieval is CPU-heavy and synchronous for
      // ~20-26s in LIVE Render requests. Letting it run while awaiting a
      // network AI promise blocks the Node event loop; an otherwise-fast
      // NVIDIA answer is incorrectly timed out before callbacks can run.
      // AI is the PRIMARY designer. Await its independently submitted
      // source-bound outfit design FIRST, before any legacy CPU work.
      const initial=await promise;
      parallelComposition={pool,promise:Promise.resolve(initial)};
      if(initial?.status==='composed'&&Array.isArray(initial.looks)){
        const admitted=await sourceVerifyComposedOutfits({
          looks:initial.looks,request:{...request,requiredIds},requiredIds,
          hardGate,priceToBudgetCurrency,scorer:fashionScorer,
          sourceMode:request.sourceMode,creativePlans:aiPlans,
          // The 6-look AI slate is a PHOTO audition, not six final cards.
          // First three directions remain source-disjoint. Optional later
          // alternatives may remix admitted clothes; the FINAL photo deck
          // still enforces strict no-product-reuse on displayed looks.
          allowVariantProductReuse:true,
        });
        // Source/budget/physical admission, original item identities and
        // transmission permission are all mandatory. This is NOT photo
        // approval. Every surviving whole look still goes through NVIDIA.
        const palette=context.preferences;
        for(const look of admitted){
          if(catalogFourItemIntent && !eligibleCatalogFourItemPhotoCandidate(look))continue;
          if(verifiedAvoidColors(palette?.avoidColors).length&&
            photographicPaletteEvidence(look,palette,requiredIds).blocked)continue;
          try{
            if(await canJudgeOutfit(look,reference?.referenceId??null)===true)
              sourceVerifiedAiFirst.push(look);
          }catch{}
        }
      }
    }
  }
  const aiFirstFastTrack=sourceVerifiedAiFirst.length>=3;
  // The photo teacher formerly controlled only a late tie-break on
  // ranker-prebuilt challengers, while AI-first skipped retrieval ENTIRELY.
  // Consequently historical real PHOTO taste did not help design the FIRST
  // round's extra candidate relationships. Learn from verified whole-outfit
  // photo outcomes and construct a FEW fresh alternatives directly from the
  // AI's already source-admitted garment decisions, without 20s of ranker
  // CPU or a new model call. A separately verified full-image critic still
  // has ABSOLUTE publication veto over every proposed combination.
  let photoHistoryCreativeAuditions=0;
  if(aiFirstFastTrack&&request.sourceMode==='catalog'&&!requiredIds.length&&
      photoHistory?.status==='heldout_photo_validated'&&
      typeof photoHistory.scoreFeatures==='function'){
    const alternateParts=photoHistoryRecompositionCandidates(
      sourceVerifiedAiFirst,photoHistory.scoreFeatures,{
        limit:8,featuresOf:items=>outfitStylingFeatures(items,context.anchorId),
        sourceMenu:parallelComposition?.pool||[],
      });
    const knownIds=new Set(sourceVerifiedAiFirst.map(x=>x.id));
    const space=Math.max(0,MAX_VISUAL_CANDIDATES-sourceVerifiedAiFirst.length);
    for(const proposal of alternateParts){
      if(photoHistoryCreativeAuditions>=space)break;
      const admitted=await sourceVerifyComposedOutfits({
        looks:[proposal.items],request:{...request,requiredIds},requiredIds,
        hardGate,priceToBudgetCurrency,scorer:fashionScorer,
        sourceMode:request.sourceMode,
      });
      const candidate=admitted[0];
      if(!candidate||knownIds.has(candidate.id)||
        (catalogFourItemIntent&&!eligibleCatalogFourItemPhotoCandidate(candidate)))
        continue;
      if(verifiedAvoidColors(context.preferences?.avoidColors).length&&
        photographicPaletteEvidence(candidate,context.preferences,requiredIds).blocked)
        continue;
      try{
        if(await canJudgeOutfit(candidate,reference?.referenceId??null)!==true)
          continue;
      }catch{continue}
      knownIds.add(candidate.id);
      sourceVerifiedAiFirst.push(candidate);
      photoHistoryCreativeAuditions++;
    }
  }
  // The proven legacy retrieval is only a FALLBACK if the independent AI
  // did not construct at least three complete, source-admitted looks.
  // It must never slow a successful AI fashion design or consume the CPU
  // while the model's network request is still in flight.
  const retrieval=aiFirstFastTrack
    ?{shortlist:sourceVerifiedAiFirst,diagnostics:{
      retrievalMode:'source_verified_ai_first',
      candidateCount:sourceVerifiedAiFirst.length,
      exclusions:{requested:0},
    }}
    :await shortlistRetrievalDeck({
      request: { ...request, requiredIds }, catalog, hardGate, priceToBudgetCurrency,
      creativeDirector: async () => briefs,
      options: { fashion: fashionScorer,
        perRole: verifiedAvoidColors(context.preferences?.avoidColors).length?12:8,
        perBrief: 100, maxVisual: 72 },
    });
  // A single cross-slot veto applies to selection AND visual challengers.
  // The pre-reviewed critic must never fill a rejected outfit from another
  // path after an NVIDIA verdict. Retain exact SKU and feasibility admission.
  const editorialVetoes = {};
  // The new photo-backed AI path owns taste and occasion/editorial choices,
  // rather than accumulating product-title regexes. Keep the hard physical
  // weather gate, and restore ALL legacy fashion gates if AI abstains.
  let selectable = retrieval.shortlist.filter((look) => {
    const reason = editorialRegisterVeto(look, context);
    if (!reason || (modelLed && reason !== 'cool_weather_sleeveless_single_layer')) return true;
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
  let modelFashionSelection = {status:'not_enabled',candidatesSeen:0,chosen:0};
  let composerPool=null;
  // Reference-optional review: a deck with no matching reference brief is
  // still visually judged on its product images alone (the critic marks such
  // verdicts referenceless and never claims a reference comparison). With a
  // reference, behavior is unchanged.
  const permitted = async (look) => {
    if(catalogFourItemIntent && !eligibleCatalogFourItemPhotoCandidate(look))return false;
    try { return await canJudgeOutfit(look, reference?.referenceId ?? null) === true; } catch { return false; }
  };
  if (visualDeckReview && (selected.length || modelLed) && typeof visualCritic === 'function' &&
      typeof canJudgeOutfit === 'function') {
    // Budgeted and source-verified finalist pool. Expand beyond the three
    // displayed directions: one timed-out NVIDIA call and one real fashion
    // rejection must not force a two-look deck when better alternatives exist.
    // The critic still owns every approval and enforces an eight-call paid cap.
    const candidates = [];
    // Under explicit color exclusions the deterministic top-3 may carry
    // UNKNOWN PHOTO COLORS: old code paid NVIDIA for all three, only to veto
    // them as grey/navy. Prefer image-grounded safe candidates BEFORE paid
    // vision. Unknown is not a known bad product, it can still be a fallback
    // if there is insufficient verified merchandise.
    const explicitColorPreferences=context.sourceMode==='catalog' &&
      verifiedAvoidColors(context.preferences?.avoidColors).length>0;
    let modelChosen=false;
    const skuCompositionReceipt={
      status:'not_run',proposed:0,admitted:0,previewProducts:0,
      exploratoryProducts:0,parallel:!!parallelComposition,
      visualMenuStatus:'not_run',aiFirst:!!parallelComposition,
      retrievalSkipped:aiFirstFastTrack,
      // Number of freshly SOURCE-VERIFIED in-memory creative pairings
      // from HELD-OUT validated original PHOTO critic relationships.
      // None counts as approved until the critic actually inspects photos.
      photoHistoryCreativeAuditions,
    };
    if(modelLed){
      const reviewable=[];
      for(const look of selectable)if(await permitted(look))reviewable.push(look);
      const preview=directorSlatePreview(reviewable,aiPlans,{
        includePublicCatalogNames:request.sourceMode==='catalog',
      });
      modelFashionSelection={status:'insufficient_candidates',candidatesSeen:preview.length,chosen:0};
      // PRIMARY AI composition path: the model is given 12 different real
      // merchandise pieces per role, and CREATES six independent candidate
      // ensembles for the same eight-photo critic budget. This gives AI
      // fashion decisions priority over ranker-built filler; the separate
      // photo editor still displays no more than three proven looks.
      // Previously it merely selected ranker-prebuilt L cards and rejected
      // 7/8 in real Classic/Work because the combinations were already bad.
      // Every new combination is rerun through the REAL source/budget/weather
      // hard gate, the existing scorer and the independent NVIDIA photo judge.
      // If the composer abstains, the existing model-led selector is retained.
      if(typeof selectCatalogCompositions==='function'&&
          request.sourceMode==='catalog'&&!requiredIds.length&&
          (reviewable.length>=6||parallelComposition!==null)){
        // Context/owner admissions are enforced by the SAME retrieval
        // preparer before exposing any additional real SKU to the composer.
        // The old prebuilt-look shortlist remains a quality seed, NOT a
        // restriction to one dozen nearly identical office shirts.
        let pool=parallelComposition?.pool;
        if(!pool){
          const excludedBases=new Set(
            (Array.isArray(request.excludedIds)?request.excludedIds:[])
              .map(id=>baseProductId(String(id))));
          const extra=Array.isArray(catalog)?catalog.slice(0,650)
            .filter(item=>!excludedBases.has(baseProductId(String(item.id))))
            .map(item=>fashionScorer.prepare(item)).filter(Boolean):[];
          pool=sourceBoundCompositionPool(reviewable,{
            additionalItems:extra,explorationSlots:4,directions:aiPlans,feedbackTaste,
          });
        }
        composerPool=pool;
        skuCompositionReceipt.previewProducts=pool.length;
        skuCompositionReceipt.designDirections=aiPlans.length;
        skuCompositionReceipt.exploratoryProducts=pool.filter(p=>
          !reviewable.some(look=>look.items.some(item=>item.id===p.item.id))).length;
        try{
          const composition=parallelComposition
            ?await parallelComposition.promise
            :await selectCatalogCompositions({
              pool,directions:aiPlans,outfitCount:6,feedbackTaste,context:{
                style:context.style,occasion:context.occasion,brief:context.request,
                weatherCelsius:Number.isFinite(context.weather?.temp)?context.weather.temp:null,
                preferences:context.preferences,
              },layerRequired:forceLayer||catalogFourItemIntent,
             allowLayer:!noLayer||catalogFourItemIntent,
            });
          skuCompositionReceipt.status=String(composition?.status||'unavailable').slice(0,40);
          skuCompositionReceipt.studio=
            ['nemotron','cross_lab_kimi','nemotron_fallback',
              'nemotron_plus_kimi_initial','cross_lab_kimi_initial_fallback',
              'nemotron_plus_kimi_parallel','nemotron_parallel',
              'cross_lab_kimi_parallel']
              .includes(composition?.providerFamily)
              ?composition.providerFamily:'not_run';
          skuCompositionReceipt.visualMenuStatus=
            typeof composition?.visualMenuStatus==='string'
              ?composition.visualMenuStatus.slice(0,40):'not_run';
          skuCompositionReceipt.proposed=Array.isArray(composition?.looks)
            ?composition.looks.length:0;
          modelFashionSelection={
            status:String(composition?.status||'unavailable').slice(0,40),
            candidatesSeen:pool.length,
            chosen:0,
            providerCalls:Math.min(2,Math.max(0,Number(composition?.providerCalls)||0)),
            compositionMode:true,
          };
          if(composition?.status==='composed'&&Array.isArray(composition.looks)){
            const admitted=await sourceVerifyComposedOutfits({
              looks:composition.looks,
              request:{...request,requiredIds},requiredIds,
              hardGate,priceToBudgetCurrency,scorer:fashionScorer,
              sourceMode:request.sourceMode,creativePlans:aiPlans,
              allowVariantProductReuse:true,
            });
            for(const look of admitted){
              if(explicitColorPreferences&&
                photographicPaletteEvidence(look,context.preferences,requiredIds).blocked)continue;
              if(await permitted(look))candidates.push(look);
            }
            skuCompositionReceipt.admitted=candidates.length;
            modelFashionSelection.chosen=candidates.length;
            // One surviving composition isn't enough to ignore existing
            // director's six alternatives. Fall back to ordinary choices.
            modelChosen=candidates.length>=2;
            if(!modelChosen)candidates.length=0;
            else modelFashionSelection.status='composed';
          }
        }catch{
          skuCompositionReceipt.status='composition_unavailable';
          modelFashionSelection={status:'composition_unavailable',
            candidatesSeen:pool.length,chosen:0,compositionMode:true};
        }
      }
      if(!modelChosen&&preview.length>=6){
        try{
          const decision=await selectDirectorFinalists({
            options:preview.map(({look,...safe})=>safe),
            context:{
              style:context.style,occasion:context.occasion,brief:context.request,
              weatherCelsius:Number.isFinite(context.weather?.temp)?context.weather.temp:null,
              preferences:context.preferences,
            },
          });
          modelFashionSelection={
            status:String(decision?.status||'no_decision').slice(0,40),
            candidatesSeen:preview.length,chosen:0,
            providerCalls:Math.min(1,Math.max(0,Number(decision?.providerCalls)||0)),
            ...(Number.isFinite(decision?.timeLeftMs)?
              {timeLeftMs:Math.round(decision.timeLeftMs)}:{}),
            ...(Number.isFinite(decision?.timeoutMs)?
              {timeoutMs:Math.round(decision.timeoutMs)}:{}),
          };
          if(decision?.status==='selected'&&Array.isArray(decision.tokens)){
            const byToken=new Map(preview.map(x=>[x.token,x.look]));
            for(const token of decision.tokens.slice(0,6)){
              const look=byToken.get(token);
              if(!look||candidates.some(x=>x.id===look.id))continue;
              if(explicitColorPreferences&&
                photographicPaletteEvidence(look,context.preferences,requiredIds).blocked)continue;
              if(await permitted(look))candidates.push(look);
            }
            modelChosen=candidates.length>=3;
            modelFashionSelection.chosen=candidates.length;
            if(modelChosen){
              const original=[...candidates];
              const metrics=creativeMetrics(deckContext);
              const portfolio=assembleIndependentVisualPortfolio(
                reviewable,original,requiredIds,metrics,reviewedLimit);
              const core=portfolio.slice(0,3);
              modelFashionSelection.uniqueTripleBefore=
                visualProductUniqueTripleCount(original,requiredIds)>0;
              modelFashionSelection.uniqueTripleAvailable=
                core.length===3&&visualProductUniqueTripleCount(core,requiredIds)>0;
              if(modelFashionSelection.uniqueTripleAvailable){
                candidates.splice(0,candidates.length,...portfolio);
                modelFashionSelection.coreModelChoices=
                  core.filter(x=>original.some(y=>y.id===x.id)).length;
                modelFashionSelection.portfolioSize=candidates.length;
                modelFashionSelection.portfolioTriples=
                  visualProductUniqueTripleCount(candidates,requiredIds);
                modelFashionSelection.portfolioSingleVetoResilience=
                  visualPortfolioSingleVetoResilience(candidates,requiredIds);
              } else if(context.dressCode==='formal'&&request.sourceMode==='catalog'){
                // A strict formal dress code can force identical four ROLE
                // names even when two genuine suit palettes are visually
                // different. Photograph product-independent complete outfits
                // and let NVIDIA verify their real jacket/trouser color shifts
                // before calling them separate tailored directions.
                const probe=assembleFormalPhotoProbePortfolio(
                  reviewable,original,requiredIds,reviewedLimit);
                if(probe.length>=3){
                  candidates.splice(0,candidates.length,...probe);
                  modelFashionSelection.formalPhotoProbe=true;
                  modelFashionSelection.portfolioSize=candidates.length;
                  modelFashionSelection.coreModelChoices=
                    probe.slice(0,3).filter(x=>original.some(y=>y.id===x.id)).length;
                }
              }
            }
          }
        }catch{
          modelFashionSelection={status:'selection_unavailable',
            candidatesSeen:preview.length,chosen:0};
        }
      }
      if(!modelChosen){
        // A timed-out AI may have admitted garments that older title-only
        // filters excluded. Reconstruct a complete legacy fashion scorer over
        // the real pre-approved shortlist before choosing any fallback. Never
        // use an AI-deferred outfit as a merely "deterministic" approval.
        candidates.length=0;
        const recoveryScorer=makeFashionScorer(context,reference,
          Array.isArray(request.recentSignatures)?request.recentSignatures.slice(-32):[],{
            priceProfile,budgeted:!!request.budget,
            layerOptional:context.optionalLayer===true&&!forceLayer,
            requiredIds,feedbackTaste,
            photoDirectorEnabled:false,
          });
        selectable=selectable.flatMap(look=>{
          const prepared=look.items.map(item=>recoveryScorer.prepare(item));
          if(prepared.some(x=>!x))return [];
          const quality=recoveryScorer.rankLook(prepared,look.brief);
          if(!quality)return [];
          return [{...look,items:prepared,fashion:quality,seedScore:quality.score}];
        }).filter(look=>!editorialRegisterVeto(look,context));
        rerankFashionCandidates(selectable,deckContext);
        selected=chooseDeck(selectable,deckContext);
        if(modelFashionSelection.status==='selected')
          modelFashionSelection.status='insufficient_admitted_selection';
      }
    }
    // IMPORTANT: the AI-first creative PHOTO learner already made additional
    // source-authorized, full-look proposals before legacy retrieval. The
    // normal model branch reconstructs its original six outfit drafts from
    // `parallelComposition.promise`; previously this SILENTLY discarded
    // every photo-history remix despite logging that they had been created.
    // Only source-verified alternatives which can fit the same eight-photo
    // budget are forwarded to the *independent* original-image jury. They
    // are not approved for display and never replace a user-pinned item.
    let photoHistoryReachedPhoto=0;
    if(modelLed&&modelChosen&&aiFirstFastTrack&&
        request.sourceMode==='catalog'&&!requiredIds.length&&
        photoHistoryCreativeAuditions>0){
      const already=new Set(candidates.map(look=>look.id));
      const learned=sourceVerifiedAiFirst.slice(-photoHistoryCreativeAuditions);
      for(const look of learned){
        if(candidates.length>=reviewedLimit)break;
        if(!look?.id||already.has(look.id)||
            !eligibleCatalogFourItemPhotoCandidate(look))continue;
        if(explicitColorPreferences&&
            photographicPaletteEvidence(look,context.preferences,requiredIds).blocked)
          continue;
        if(!await permitted(look))continue;
        already.add(look.id);
        candidates.push(look);
        photoHistoryReachedPhoto++;
      }
    }
    // Count source-admitted history remixes and those ACTUALLY promoted to
    // the original-photo shortlist separately. Neither is a beauty verdict.
    skuCompositionReceipt.photoHistoryReachedPhoto=photoHistoryReachedPhoto;
    if(modelLed)modelFashionSelection.skuCompositionReceipt=skuCompositionReceipt;
    if(!modelChosen)for(const look of selected) {
      if (explicitColorPreferences) {
        const evidence=photographicPaletteEvidence(look,context.preferences,requiredIds);
        if (!sufficientPhotographicPaletteEvidence(evidence)) continue;
      }
      if (await permitted(look)) candidates.push(look);
    }
    const metrics = creativeMetrics(deckContext);
    const leadScore = selected[0]?.fashion?.score ?? -Infinity;
    // Strongly photographed, PERSONALIZED catalog alternatives are reserved
    // before weakly evidenced lookalikes can exhaust 8 paid NVIDIA calls.
    // Only complete photos whose color intelligence matches the CURRENT
    // image+source hash count as verified. The provider still decides which
    // of these candidates actually looks good and matches the occasion.
    const photoSafe=explicitColorPreferences ? selectable
      .filter(look=>!candidates.some(other=>other.id===look.id) &&
        Number.isFinite(look.fashion?.score) &&
        look.fashion.score>=leadScore-VISUAL_CHALLENGER_QUALITY_WINDOW)
      .map(look=>({look,evidence:photographicPaletteEvidence(look,context.preferences,requiredIds)}))
      .filter(({evidence})=>sufficientPhotographicPaletteEvidence(evidence))
      .sort((a,b)=>b.evidence.score-a.evidence.score ||
        (b.look.fashion?.score??0)-(a.look.fashion?.score??0)) : [];
    let photoReserved=0;
    // The AI commissions THREE independent source-backed designs, never just
    // two. Keep PHOTO-review capacity for every actually available plan;
    // the original reviewer may STILL reject any or all of those choices.
    // This allocates scarce review ATTENTION, not artificial approval.
    const plannedReserve=Math.max(2,Math.min(3,aiPlans.length));
    const photoSafeReserve=Math.min(5,Math.max(0,
      reviewedLimit-candidates.length-plannedReserve));
    // Prefer SKUs not already present anywhere in the model-review batch.
    // Only on the second pass accept a materially different shared-cardigan
    // alternative as a last resort: it can still be useful if the original
    // candidate is vetoed. The displayed deck remains PRODUCT-unique.
    for(let pass=0;pass<2&&photoReserved<photoSafeReserve;pass++)for(const {look} of photoSafe){
      if(photoReserved>=photoSafeReserve || candidates.length>=reviewedLimit)break;
      if(candidates.some(other=>other.id===look.id))continue;
      if(candidates.some(other=>Math.max(other.items.length,look.items.length)-overlaps(other,look)<2))continue;
      if(pass===0 && candidates.some(other=>overlaps(other,look)>0 ||
        !independentOutfitDirection(look,other,requiredIds)))continue;
      if(await permitted(look)){candidates.push(look);photoReserved++;}
    }

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
      if (candidates.length >= reviewedLimit) break;
      const planned = selectable.filter(look =>
        look.briefId === canonicalBriefId(plan.id) && !candidates.some(other => other.id === look.id) &&
        Number.isFinite(look.fashion?.score) &&
        (!explicitColorPreferences || !photographicPaletteEvidence(look,context.preferences,requiredIds).blocked))
        .sort((a,b) =>
          // AI plans have multiple actual SKU executions. Prefer the one
          // that adds genuinely NEW products instead of using the same
          // popular cardigan the first two outfits already contain.
          candidates.reduce((n,chosen)=>n+overlaps(chosen,a),0) -
          candidates.reduce((n,chosen)=>n+overlaps(chosen,b),0) ||
          (explicitColorPreferences ?
            photographicPaletteEvidence(b,context.preferences,requiredIds).score -
            photographicPaletteEvidence(a,context.preferences,requiredIds).score : 0) ||
          b.fashion.score - a.fashion.score);
      for (const look of planned.slice(0, 6)) {
        if (await permitted(look)) { candidates.push(look); break; }
      }
    }

    // Recompute marginal direction gain after EVERY alternate. The previous
    // one-time four-candidate sort could fill its only spare slot with a
    // redundant outfit, while all genuinely distinct challengers were hidden.
    while (candidates.length < reviewedLimit) {
      const current = candidates.length ? candidates : selected;
      const capacity = visualArchitectureCapacity(current, requiredIds);
      const rescueTriples = visualRescueTripleCount(current, requiredIds);
      const uniqueTriples=visualProductUniqueTripleCount(current,requiredIds);
      const pinned = new Set(requiredIds.map(String));
      const existingLayerArchetypes=new Set(current.map(look =>
        outerwearArchetype(look,requiredIds)).filter(Boolean));
      const usedTopIds = new Set(current.flatMap(({ items }) => (items || [])
        .filter(it => it.macroCategory === 'top' && !pinned.has(String(it.id)))
        .map(it => String(it.id))));
      const extras = rankVisualChallengerPool(selectable
        .filter(look => !candidates.some(chosen => chosen.id === look.id) &&
          Number.isFinite(look.fashion?.score) &&
          (!explicitColorPreferences ||
            !photographicPaletteEvidence(look,context.preferences,requiredIds).blocked) &&
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
          uniqueTripleGain: visualProductUniqueTripleCount([...current,look],requiredIds) - uniqueTriples,
          outerwearGain: existingLayerArchetypes.size &&
            outerwearArchetype(look,requiredIds) &&
            !existingLayerArchetypes.has(outerwearArchetype(look,requiredIds)) ? 1 : 0,
          unseenTop: look.items.some(it => it.macroCategory === 'top' &&
            !pinned.has(String(it.id)) && !usedTopIds.has(String(it.id))) ? 1 : 0,
          difference: Math.min(...current.map(other =>
            directionDifference(other, look, metrics.featuresOf, metrics.featureDistance).distance)),
          // Learned statistical photo HISTORY is used only as a challenger
          // tie-break AFTER actual physical feasibility, inventory,
          // meaningful novelty and independent 3-direction capacity.
          // No experience or unreliable held-out comparisons => null.
          photoHistoryScore: typeof photoHistory?.scoreFeatures==='function'
            ?(()=>{try{
                const value=photoHistory.scoreFeatures(
                  outfitStylingFeatures(look.items,context.anchorId));
                return Number.isFinite(value)&&value>=-1&&value<=1?value:null;
              }catch{return null}})():null,
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
      // The autonomous model owns the aesthetics and SKU choices, yet its
      // first three photographed drafts may share clothes or be structural
      // twins. Rearrange ONLY already-source-admitted options so early PHOTO
      // assessments can actually yield three independent approved cards.
      // Shared-SKU learned remixes remain valuable as rejected-look rescue
      // options, but cannot crowd out a feasible full independent trio.
      const auditionCoreBefore=visualProductUniqueTripleCount(
        candidates.slice(0,3),requiredIds);
      if(modelChosen&&modelLed&&request.sourceMode==='catalog'&&
          candidates.length>=3&&auditionCoreBefore===0){
        const arranged=prioritizeIndependentPhotoAuditions(
          candidates,requiredIds,metrics);
        const changed=arranged.slice(0,3).some((look,i)=>
          look?.id!==candidates[i]?.id);
        if(changed){
          candidates.splice(0,candidates.length,...arranged);
          modelFashionSelection.photoAuditionCoreRecovered=true;
        }
      }
      if(modelLed&&modelChosen&&request.sourceMode==='catalog'){
        // This is the exact FIRST wave passed to the independent PHOTO
        // critic, not a count of already approved or displayed looks.
        skuCompositionReceipt.photoAuditionCoreFeasible=
          visualProductUniqueTripleCount(
            candidates.slice(0,3),requiredIds)>0;
      }
      let primaryCritiques=[];
      const reviewOptions={
        target:Math.min(3,candidates.length), minNovelty:0.12, requiredIds,
          requireDistinctArchitectures: true,
          // Server-owned FORMAL-only alternative: the critic must positively
          // identify different blazer AND shirt/trouser photos. No other occasion
          // gets a softer architecture rule, and all photo quality floors stay.
          allowFormalPhotoPaletteDirections: context.dressCode==='formal'&&
            request.sourceMode==='catalog',
          requireNoProductReuse: request.sourceMode==='catalog',
          // Whole-outfit quality is conjunctive: correct colors cannot excuse
          // an unsuitable dress code or occasion. These are per-axis NVIDIA
          // verdict scores, never inferred user attributes or SKU heuristics.
          // A SECOND arbitrary stricter score floor on AI-selected looks
          // defeated the Fashion Director: NVIDIA could give an elegant
          // outfit 0.78 style, 0.75 occasion and 0.76 formality (all with a
          // positive whole-look verdict), yet the model-led path alone
          // vetoed it over a 0.02 margin. Keep one calibrated quality gate
          // for every mode, plus the photographic judge's explicit reject.
          minStyleFit: 0.72,
          minOccasionFit: 0.75,
          minFormalityFit: 0.70,
          avoidColors: context.preferences?.avoidColors || [],
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
            !requiredIds.length && !explicitlyRequestsLuxury(context),
          onCritiques:rows=>{primaryCritiques=rows;},
      };
      // Truthful telemetry to distinguish a genuinely trained and applied
      // early photo-history decision from a disabled/unreliable prior.
      modelFashionSelection.photoHistoryStatus=String(
        photoHistory?.status||'not_run').slice(0,42);
      modelFashionSelection.photoHistoryExamples=Number.isSafeInteger(
        photoHistory?.examples)?photoHistory.examples:0;
      modelFashionSelection.photoHistoryHoldoutWins=Number.isSafeInteger(
        photoHistory?.holdoutWins)?photoHistory.holdoutWins:0;
      modelFashionSelection.photoHistoryHoldoutPairs=Number.isSafeInteger(
        photoHistory?.holdoutPairs)?photoHistory.holdoutPairs:0;
      modelFashionSelection.photoHistoryPriorConsidered=
        typeof photoHistory?.scoreFeatures==='function';
      let reviewed=await evaluateVisualShortlist(
        {...retrieval,shortlist:candidates},request,visualCritic,reviewOptions);
      // Preserve the EXACT FIRST independent photo wave before source-SKU
      // rescue and portfolio editors add more candidates. The server-only
      // journal may use these actual critique outcomes, but never anything
      // unjudged or displayed merely because it was proposed.
      const photoJournalLooks=candidates.slice(0,MAX_VISUAL_CANDIDATES);
      const photoJournalDecisions=
        (reviewed.diagnostics.visualQualityDecisions||[]).slice(0,MAX_VISUAL_CANDIDATES);
      // Train an EPHEMERAL, in-request photographed outfit teacher from the
      // independently judged 4-item source looks; do not mix AI photo scores
      // with owner thumbs or promote them to actual aesthetic approval.
      // When the critic examples cannot predict a matched HELD-OUT photo
      // comparison this learner ABSTAINS and selection stays unchanged.
      const photoFeaturesOf=items=>outfitStylingFeatures(items,context.anchorId);
      const photoTeacher=trainPhotoFeedbackTeacher({
        looks:candidates,
        decisions:reviewed.diagnostics.visualQualityDecisions||[],
        critiques:primaryCritiques,featuresOf:photoFeaturesOf,
      });
      modelFashionSelection.photoTeacherStatus=photoTeacher.status;
      modelFashionSelection.photoTeacherExamples=photoTeacher.examples;
      modelFashionSelection.photoTeacherPositive=photoTeacher.positive;
      modelFashionSelection.photoTeacherNegative=photoTeacher.negative;
      let rescueAttempted=0;
      let secondaryCritiques=[];
      // THE FASHION-LEARNING LOOP: The first set of real product photographs
      // sometimes leaves only one poor date look or two generic tee/jeans
      // casual looks. A professional stylist must not stop there or fabricate
      // a third: use the independent model's real refusal reasons to create
      // NEW source-backed combinations, then independently re-judge the new
      // merchandise photos. This is not a numeric boost or a product regex.
      //
      // One bounded extra LLM call, <=3 real looks, <=3 additional photo
      // calls, only on 72s Render staging with a live deadline; never S5.
      if(context.sourceMode==='catalog'&&context.dressCode!=='formal'&&
          modelLed&&typeof selectCatalogRescueCompositions==='function'&&
          Array.isArray(composerPool)&&composerPool.length>=12&&
          typeof photoRescueRemainingMs==='function'&&
          reviewed.outfits.length<3&&
          reviewed.diagnostics.visualValid>=3&&
          photoRescueRemainingMs()>=19000){
        const firstApprovedIds=new Set(
          (reviewed.diagnostics.visualQualityDecisions||[])
            .filter(row=>row.reason==='approved').map(row=>row.candidateId));
        const approvedFirst=candidates.filter(x=>firstApprovedIds.has(x.id));
        const firstProductKeys=new Set(approvedFirst.flatMap(x=>
          (x.items||[]).map(it=>baseProductId(String(it.id)))));
        // Exclude already APPROVED real garments from the NEW proposal pool.
        // The photo judge may still reject the new choices, and if it does
        // the verified original partial deck remains intact.
        const freshPool=composerPool.filter(p=>
          !firstProductKeys.has(baseProductId(String(p.item?.id))));
        // Actual authenticated iOS Work evidence: 11 original-product PHOTO
        // critiques yielded FIVE approved looks but still ZERO feasible
        // independent THREE-look portfolios (the final editor chose two
        // near-identical shirt/blazer/loafer architectures). The former
        // rescue explored genuinely UNSEEN merchandise only when ZERO
        // photographs were approved. That is the wrong creative stopping
        // condition: the goal is a complete, beautiful, diverse COLLECTION,
        // not one or two individually acceptable photo cards. Let a second
        // PHOTO-informed author see different authentic merchandise any
        // time the first judged collection cannot fulfill three looks.
        // Keep original owner constraints and PHOTO vetoes unchanged.
        const needsNewCreativeInventory=reviewed.outfits.length<3;
        const initialSkuKeys=new Set(composerPool.map(p=>
          baseProductId(String(p.item?.id))));
        const excludedKeys=new Set(
          (Array.isArray(request.excludedIds)?request.excludedIds:[])
            .map(id=>baseProductId(String(id))));
        const unexplored=needsNewCreativeInventory&&Array.isArray(catalog)
          ?catalog.filter(item=>item?.isShopItem===true&&
              !initialSkuKeys.has(baseProductId(String(item.id)))&&
              !excludedKeys.has(baseProductId(String(item.id))))
            .map(item=>fashionScorer.prepare(item)).filter(Boolean):[];
        const unfamiliarPool=unexplored.length
          ?sourceBoundCompositionPool([],{additionalItems:unexplored,
              inventoryFirst:true,explorationSlots:4,directions:[],feedbackTaste})
          :[];
        const totallyNew=slots.every(slot=>unfamiliarPool.filter(p=>
          p.slot===slot).length>=3);
        const creativeRetryPool=needsNewCreativeInventory&&totallyNew
          ?unfamiliarPool:freshPool;
        modelFashionSelection.photoRevisionInventory=totallyNew
          ?'unseen_source_skus':'source_fallback';
        // Transfer concrete photographic evidence from the exact FAILED
        // outfit to the autonomous composer. Generic reason codes alone do
        // not tell a stylist whether the shoe shape, proportion or fabric
        // relationship should change. The composer further anonymizes and
        // bounds every observation; real source identity never leaves here.
        const decisions=(reviewed.diagnostics.visualQualityDecisions||[]);
        const reasonById=new Map(decisions.map(d=>[d.candidateId,d.reason]));
        const photoById=new Map(primaryCritiques.map(row=>[row.candidateId,row]));
        const rejectedOutfits=candidates.flatMap(look=>{
          const reason=reasonById.get(look.id),photo=photoById.get(look.id);
          if(!reason||reason==='approved'||!photo)return [];
          return [{
            reason,
            roles:Object.fromEntries(slots.map(slot=>[slot,
              String(look.items.find(it=>it.macroCategory===slot)?.subtype||'')
                .slice(0,32)])),
            // Trusted request-local source identities are converted to
            // allowed opaque P-tokens by the composer. The raw IDs are
            // NEVER included in an external model payload or logged.
            sourceItemIds:Object.fromEntries(slots.map(slot=>[slot,
              String(look.items.find(it=>it.macroCategory===slot)?.id||'')
                .slice(0,220)])),
            observations:photo.slotObservations,
            colors:photo.slotColors,
            relationships:photo.relationships,
            scores:Object.fromEntries(
              ['visualCohesion','styleFit','occasionFit','formalityFit','confidence']
                .map(axis=>[axis,photo[axis]])),
          }];
        }).slice(0,4);
        // CONTRASTIVE PHOTO LEARNING, not a product-category rule.
        // This first rescue actually runs on real iPhone attempts, unlike
        // the often-deadline-gated last portfolio revision. It previously
        // saw failures but not WHAT WAS BEAUTIFUL in the accepted real
        // photographs. Give the independent creative AI BOTH examples so
        // it can compare proportions, materials and shoe-to-trouser harmony
        // and form its own different source-SKU decision. The existing
        // groundedPhotoRevisionFeedback gateway strips untrusted identity,
        // URLs, extra fields and unsupported image claims before inference.
        const approvedPhotoEvidence=approvedFirst.slice(0,2).flatMap(look=>{
          const photo=photoById.get(look.id);
          if(!photo||photo.reject===true)return [];
          return [{
            roles:Object.fromEntries(slots.map(slot=>[slot,
              String(look.items?.find(it=>it.macroCategory===slot)?.subtype||'')
                .slice(0,32)])),
            observations:photo.slotObservations,
            colors:photo.slotColors,
            relationships:photo.relationships,
            scores:Object.fromEntries(
              ['visualCohesion','styleFit','occasionFit','formalityFit','confidence']
                .map(axis=>[axis,photo[axis]])),
          }];
        });
        const feedback={
          reasons:decisions.filter(row=>row.reason!=='approved')
            .map(row=>row.reason).slice(0,8),
          approvedArchitectures:approvedFirst.slice(0,3).map(look=>
            Object.fromEntries(slots.map(slot=>[slot,
              String(look.items.find(it=>it.macroCategory===slot)?.subtype||'')
                .slice(0,32)]))),
          approvedPhotoEvidence,
          rejectedOutfits,
        };
        let retry={status:'unavailable',looks:[]};
        try{
          retry=await selectCatalogRescueCompositions({
            pool:creativeRetryPool,critiqueFeedback:feedback,directions:aiPlans,feedbackTaste,
            context:{style:context.style,occasion:context.occasion,
              brief:context.request,preferences:context.preferences,
              weatherCelsius:Number.isFinite(context.weather?.temp)
                ?context.weather.temp:null},
            layerRequired:forceLayer||catalogFourItemIntent,
             allowLayer:!noLayer||catalogFourItemIntent,
          });
        }catch{/* always keep the already judged original deck */}
        modelFashionSelection.photoRevisionStatus=
          String(retry?.status||'unavailable').slice(0,32);
        modelFashionSelection.photoRevisionStudioCalls=
          Number.isSafeInteger(retry?.providerCalls)?retry.providerCalls:0;
        modelFashionSelection.photoRevisionArtist=
          ['nemotron','nemotron_fallback','cross_lab_kimi',
              'nemotron_fast_photo_revision',
              'parallel_kimi_nemotron_photo_revision']
            .includes(retry?.providerFamily)?retry.providerFamily:'not_run';
        if(retry?.status==='composed'&&Array.isArray(retry.looks)&&
            photoRescueRemainingMs()>=14000){
          const admitted=await sourceVerifyComposedOutfits({
            looks:retry.looks,request:{...request,requiredIds},requiredIds,
            hardGate,priceToBudgetCurrency,scorer:fashionScorer,
            sourceMode:request.sourceMode,
          });
          // CONTRASTIVE photo-based decision, not an extra garment rule:
          // the trained teacher may compare AI-authored new outfits against
          // already source-feasible but NEVER PHOTOGRAPHED retrieval options.
          // It only orders the next expensive PHOTO audition; the real critic
          // must independently approve EVERY displayed 4-SKU outfit.
          // Without a held-out-validated teacher preserve original author
          // order and avoid any surprising behavior in sparse catalogs.
          const unseen=photoTeacher.usable&&!requiredIds.length
            ?selectable.filter(look=>eligibleCatalogFourItemPhotoCandidate(look)&&
                !candidates.some(x=>x.id===look.id)&&
                look.items.every(it=>!firstProductKeys.has(baseProductId(String(it.id)))))
              .slice(0,48)
            :[];
          const options=photoTeacher.usable
            ?photoTaughtRescueOrder([...admitted,...unseen],
              photoTeacher,photoFeaturesOf)
            :admitted;
          const extras=[],takenProducts=new Set(firstProductKeys);
          for(const look of options){
            if(extras.length>=3||photoRescueRemainingMs()<13500)break;
            if(candidates.some(x=>x.id===look.id))continue;
            if(photoTeacher.usable&&!requiredIds.length&&
                look.items.some(it=>takenProducts.has(baseProductId(String(it.id)))))
              continue;
            if(explicitColorPreferences&&
              photographicPaletteEvidence(look,context.preferences,requiredIds).blocked)continue;
            if(await permitted(look)){
              extras.push(look);
              if(photoTeacher.usable)
                for(const it of look.items)takenProducts.add(baseProductId(String(it.id)));
            }
          }
          modelFashionSelection.photoTeacherSelectionApplied=
            photoTeacher.usable&&(
              extras.some(x=>!admitted.some(y=>y.id===x.id))||
              extras.some((x,i)=>admitted[i]?.id!==x.id));
          modelFashionSelection.photoRevisionAdmitted=extras.length;
          if(extras.length>0&&photoRescueRemainingMs()>=13500){
            const firstDiagnostics=reviewed.diagnostics;
            const allowedMs=Math.max(1000,Math.min(11500,
              Math.floor(photoRescueRemainingMs()-3500)));
            let revisionCritiques=[];
            const revision=await evaluateVisualShortlist(
              {...retrieval,shortlist:extras},request,visualCritic,{
                ...reviewOptions,target:Math.min(3,extras.length),
                criticSignal:AbortSignal.timeout(allowedMs),
                onCritiques:rows=>{revisionCritiques=rows;},
              });
            rescueAttempted=revision.diagnostics.visualAttempted;
            secondaryCritiques=revisionCritiques;
            const keepFirst=approvedFirst.slice(
              0,Math.max(0,MAX_VISUAL_CANDIDATES-extras.length));
            const keepIds=new Set(keepFirst.map(x=>x.id));
            const allCritiques=[
              ...primaryCritiques.filter(row=>keepIds.has(row.candidateId)),
              ...revisionCritiques,
            ];
            const combined=selectVisualDeck({
              ...retrieval,shortlist:[...keepFirst,...extras],
            },allCritiques,reviewOptions);
            const summary={
              visualAttempted:firstDiagnostics.visualAttempted+
                revision.diagnostics.visualAttempted,
              visualValid:firstDiagnostics.visualValid+
                revision.diagnostics.visualValid,
              visualApproved:firstDiagnostics.visualApproved+
                revision.diagnostics.visualApproved,
              visualAbstained:firstDiagnostics.visualAbstained+
                revision.diagnostics.visualAbstained,
              visualRejected:firstDiagnostics.visualRejected+
                revision.diagnostics.visualRejected,
              rejectedCandidateIds:[...new Set([
                ...(firstDiagnostics.rejectedCandidateIds||[]),
                ...(revision.diagnostics.rejectedCandidateIds||[]),
              ])],
              visualQualityDecisions:[
                ...(firstDiagnostics.visualQualityDecisions||[]),
                ...(revision.diagnostics.visualQualityDecisions||[]),
              ],
              criticFailed:firstDiagnostics.criticFailed||
                revision.diagnostics.criticFailed,
            };
            reviewed=combined.outfits.length>=reviewed.outfits.length
              ?{...combined,diagnostics:{...combined.diagnostics,...summary}}
              :{...reviewed,diagnostics:{...firstDiagnostics,...summary}};
            candidates.push(...extras);
            modelFashionSelection.photoRevisionAttempted=rescueAttempted;
            modelFashionSelection.photoRevisionApproved=
              revision.diagnostics.visualApproved;
          }
        }
      }
      // The first photo review is NOT allowed to define the last available
      // style choice. When seven of eight real formal outfits fail NVIDIA,
      // sample up to four NEW, entirely source-approved real merchandise sets
      // and have the same independent critic inspect their photographs.
      //
      // This correction is Render staging ONLY and runs only if the server
      // still owns a bounded photo time budget; a deadline failure must
      // return the original verified two rather than an HTTP 504. Every final
      // displayed look MUST come from a real passed critic verdict.
      if(context.sourceMode==='catalog'&&context.dressCode==='formal'&&
          modelLed&&typeof photoRescueRemainingMs==='function'&&
          reviewed.outfits.length<3&&
          reviewed.diagnostics.visualValid>=4&&
          reviewed.diagnostics.visualApproved>0&&
          photoRescueRemainingMs()>=17500){
        const approvedIds=new Set(
          (reviewed.diagnostics.visualQualityDecisions||[])
            .filter(row=>row.reason==='approved').map(row=>row.candidateId));
        const approvedFirst=candidates.filter(look=>approvedIds.has(look.id));
        const unreviewed=selectable.filter(look=>
          !candidates.some(x=>x.id===look.id)&&
          (!explicitColorPreferences||
            !photographicPaletteEvidence(look,context.preferences,requiredIds).blocked));
        let extra=chooseFormalPhotoRescueCandidates(
          unreviewed,[],approvedFirst,requiredIds,4);
        // The helper's supplied pool has already excluded ALL judged IDs.
        // The actual reviewer still separately gates every image host.
        const permittedExtra=[];
        for(const look of extra){
          if(photoRescueRemainingMs()<15000)break;
          if(await permitted(look))permittedExtra.push(look);
        }
        extra=permittedExtra;
        if(extra.length>0&&photoRescueRemainingMs()>=14500){
          const allowedMs=Math.max(1000,Math.min(14000,
            Math.floor(photoRescueRemainingMs()-3500)));
          let rescueCritiques=[];
          const firstDiagnostics=reviewed.diagnostics;
          const rescue=await evaluateVisualShortlist(
            {...retrieval,shortlist:extra},request,visualCritic,{
              ...reviewOptions,target:Math.min(3,extra.length),
              criticSignal:AbortSignal.timeout(allowedMs),
              onCritiques:rows=>{rescueCritiques=rows;},
            });
          rescueAttempted=rescue.diagnostics.visualAttempted;
          secondaryCritiques=rescueCritiques;
          const orderedApproved=[...reviewed.outfits.map(x=>
            candidates.find(c=>c.id===x.id)).filter(Boolean),
            ...approvedFirst].filter((x,i,arr)=>arr.findIndex(y=>y.id===x.id)===i)
            .slice(0,MAX_VISUAL_CANDIDATES-extra.length);
          const judged=[...orderedApproved,...extra];
          const keepIds=new Set(orderedApproved.map(x=>x.id));
          const combinedCritiques=[
            ...primaryCritiques.filter(row=>keepIds.has(row.candidateId)),
            ...rescueCritiques,
          ];
          // SelectVisualDeck is still constrained to EIGHT source identities
          // per execution, and EVERY member has a genuine photo verdict.
          const combined=selectVisualDeck({
            ...retrieval,shortlist:judged,
          },combinedCritiques,reviewOptions);
          const summary={
            visualAttempted:firstDiagnostics.visualAttempted+
              rescue.diagnostics.visualAttempted,
            visualValid:firstDiagnostics.visualValid+
              rescue.diagnostics.visualValid,
            visualApproved:firstDiagnostics.visualApproved+
              rescue.diagnostics.visualApproved,
            visualAbstained:firstDiagnostics.visualAbstained+
              rescue.diagnostics.visualAbstained,
            visualRejected:firstDiagnostics.visualRejected+
              rescue.diagnostics.visualRejected,
            rejectedCandidateIds:[...new Set([
              ...(firstDiagnostics.rejectedCandidateIds||[]),
              ...(rescue.diagnostics.rejectedCandidateIds||[]),
            ])],
            visualQualityDecisions:[
              ...(firstDiagnostics.visualQualityDecisions||[]),
              ...(rescue.diagnostics.visualQualityDecisions||[]),
            ],
            criticFailed:firstDiagnostics.criticFailed||rescue.diagnostics.criticFailed,
          };
          // If provider latency/poor choices made the second wave worse,
          // NEVER discard an already verified original 1-2 outfit deck.
          reviewed=combined.outfits.length>=reviewed.outfits.length
            ? {...combined,diagnostics:{...combined.diagnostics,...summary}}
            : {...reviewed,diagnostics:{...firstDiagnostics,...summary}};
          candidates.push(...extra);
          modelFashionSelection.photoRescueAttempted=rescueAttempted;
          modelFashionSelection.photoRescueApproved=rescue.diagnostics.visualApproved;
        }
      }
      // THIRD artistic decision: independently photographed outfits are
      // each individually acceptable, but three similar blazer uniforms
      // do NOT automatically make a professionally curated collection.
      // Let a separate AI director choose the best *whole portfolio* using
      // the critic's actual per-slot PHOTO observations and relationships,
      // not catalog keyword heuristics or invented products. This advisor
      // can return fewer, more worthwhile looks, or select three compelling
      // candidates from a larger pool of already approved looks.
      // If unavailable, keep the original independently approved deck.
      if(modelLed&&typeof selectPhotoApprovedPortfolio==='function'&&
          reviewed.diagnostics.visualValid>=3){
        const allCritiques=[...primaryCritiques,...secondaryCritiques];
        const approvedIds=new Set(
          (reviewed.diagnostics.visualQualityDecisions||[])
            .filter(row=>row.reason==='approved').map(row=>row.candidateId));
        const approvedLooks=candidates.filter(c=>approvedIds.has(c.id))
          .slice(0,MAX_VISUAL_CANDIDATES);
        // A REAL iPhone Classic/Work paid run at 84ba6a8 had 2/9 PHOTO
        // approvals and ~40 seconds of engine budget left. The independent
        // whole-collection fashion editor was NEVER invoked, because this
        // artificial >=3 gate short-circuited it. Exactly two approved
        // photographed looks are a VALID editorial comparison: ask the AI
        // whether they are truly different and what artistic direction the
        // third must take. Any NEW SKU idea STILL needs real photo approval.
        if(approvedLooks.length>=2){
          // The menu is a real, finite, ALREADY VERIFIED set of cohesive
          // portfolios. This is an existence proof, not a new taste veto:
          // the AI now exercises its own fashion judgment among feasible
          // 3/2/1-look collections without producing an impossible trio.
          const verifiedPortfolios=verifiedPhotoEditorialPortfolios(
            approvedLooks,allCritiques,reviewOptions);
          let decision;
          try{
            decision=await selectPhotoApprovedPortfolio({
              looks:approvedLooks,critiques:allCritiques,
              portfolios:verifiedPortfolios,creativePlans:aiPlans,
              context:{style:context.style,occasion:context.occasion,
                brief:context.request,preferences:context.preferences},
            });
          }catch{decision={status:'unavailable'};}
          const editorial={
            status:String(decision?.status||'not_available').slice(0,32),
            approvedSeen:approvedLooks.length,
            eligibleTriples:verifiedPortfolios.filter(p=>p.size===3).length,
            eligiblePairs:verifiedPortfolios.filter(p=>p.size===2).length,
            eligibleSingles:verifiedPortfolios.filter(p=>p.size===1).length,
            chosen:0,
            visualJury:['applied','not_run','image_unavailable',
              'invalid_selection','timeout','provider_unavailable','unavailable']
              .includes(decision?.visualJury)?decision.visualJury:'not_run',
            independentJury:['not_run','agreed','narrowed',
              'disagreed','unavailable','standalone',
              'arbitrated_second','arbitrated_primary'].includes(decision?.independentJury)
              ?decision.independentJury:'not_run',
            independentVoteSize:Number.isInteger(decision?.independentVoteSize)
              ?Math.max(1,Math.min(3,decision.independentVoteSize)):null,
            visualMediaStatus:['ready','not_run','invalid_input',
              'unauthorized_or_missing','deadline','source_missing',
              'photo_processing','oversized']
              .includes(decision?.visualMediaStatus)
              ?decision.visualMediaStatus:'not_run',
          };
          if(decision?.status==='selected'&&Array.isArray(decision.tokens)){
            const chosen=[...new Set(decision.tokens)]
              .map(t=>/^L[0-9]{2}$/.test(t)?
                approvedLooks[Number(t.slice(1))]:null).filter(Boolean)
              .slice(0,3);
            if(chosen.length===decision.tokens.length){
              // Replay ORIGINAL NVIDIA verdicts against precisely the
              // editor-chosen candidates. Invalid novelty, duplicate SKUs
              // or other structural failures CANNOT be approved by the AI.
              const replay=selectVisualDeck(
                {...retrieval,shortlist:chosen},allCritiques,
                {...reviewOptions,target:chosen.length,
                  onCritiques:undefined});
              if(replay.outfits.length===chosen.length){
                const byId=new Map(replay.outfits.map(x=>[x.id,x]));
                reviewed={...reviewed,
                  status:replay.status,
                  outfits:chosen.map(x=>byId.get(x.id))};
                editorial.chosen=chosen.length;
                editorial.status='applied';
              } else editorial.status='invalid_combination';
            } else editorial.status='invalid_selection';
          }
          // SECOND TRUE CREATIVE DECISION, ONLY on isolated 72s staging.
          // The independent photographic COLLECTION editor may identify
          // exactly what novel silhouette is MISSING after selecting only
          // two of >=3 already NVIDIA-approved near-clones. The original
          // source SKU artist gets that image-observed gap, proposes unused
          // SKUs and must earn NEW independent photographic approval.
          // No title regex bans, no relaxed quality floor, no weak padding.
          // If any step fails or time runs out, retain the fully photo-
          // verified original 1–2 chosen outfits completely unchanged.
          editorial.feedbackRevision='not_run';
          // Real 120B VLM JSON_OBJECT often chooses the correct verified
          // two-look F-token but omits the optional free-text gap. The model
          // has STILL seen their true source-product JPEGs and decided these
          // two are worth keeping; the independent photo critic also supplied
          // exact garment-color/material/relationship observations.
          // Do not let a missing JSON field BLOCK a real next creative
          // decision: pass only those originally photographed observations
          // to the next AI SKU designer. A TEXT-only editor cannot trigger
          // this evidence fallback or pretend to have seen the pictures.
          const explicitEditorialGap=typeof decision?.creativeGap==='string'&&
            decision.creativeGap.length>=18;
          const actualPhotoEditorialDecision=
            decision?.visualJury==='applied'&&
            decision?.visualMediaStatus==='ready';
          if(editorial.status==='applied'&&editorial.chosen<3&&
              (explicitEditorialGap||actualPhotoEditorialDecision)&&
              request.sourceMode==='catalog'&&
              Array.isArray(composerPool)&&
              typeof selectCatalogRescueCompositions==='function'&&
              typeof photoRescueRemainingMs==='function'&&
              photoRescueRemainingMs()>=22500){
            const remainingPhotoQuota=Math.max(0,
              MAX_VISUAL_CANDIDATES+3-
              (reviewed.diagnostics.visualAttempted||0));
            const photoQuota=Math.min(3,remainingPhotoQuota);
            const alreadyApprovedProductKeys=new Set(approvedLooks.flatMap(
              look=>(look.items||[]).map(it=>baseProductId(String(it.id)))));
            const freshPool=composerPool.filter(token=>
              !alreadyApprovedProductKeys.has(
                baseProductId(String(token.item?.id))));
            // The *first* 120B designer only saw up to 12 source SKUs/slot.
            // Real iPhone Date night repeatedly made cardigan + chinos and
            // a second critique of the SAME 48 already-overused products
            // approved zero replacements. The entire already-discovered
            // ~400-SKU trusted inventory is available here without NEW DB
            // calls. Let a photo-critic-informed SECOND independent creator
            // explore previously UNSEEN merchandise (new textures, shapes,
            // layers and shoe constructions), not recirculate stale tokens.
            // Always re-prepare genuine source rows and re-apply original
            // SKU/stock/budget/merchant/physical/PHOTO gates afterwards.
            const initialComposerProducts=new Set(composerPool.map(token=>
              baseProductId(String(token.item?.id))));
            const forbiddenSourceIds=new Set(
              (Array.isArray(request.excludedIds)?request.excludedIds:[])
                .map(id=>baseProductId(String(id))));
            const unseenPrepared=Array.isArray(catalog)?catalog
              .filter(item=>item?.isShopItem===true&&
                !forbiddenSourceIds.has(baseProductId(String(item.id)))&&
                !initialComposerProducts.has(baseProductId(String(item.id)))&&
                !alreadyApprovedProductKeys.has(baseProductId(String(item.id))))
              .map(item=>fashionScorer.prepare(item)).filter(Boolean):[];
            // PHOTO-informed second creative decision: do not seed the
            // unseen SKU palette with the original Director concepts that
            // just produced the repetitive first audition. Present diverse,
            // already source-prepared garment architectures; new fashion
            // choices and whole-look photo verdicts remain model-led.
            const unseenPool=sourceBoundCompositionPool([],{
              additionalItems:unseenPrepared,inventoryFirst:true,
              explorationSlots:4,directions:[],feedbackTaste,
            });
            // Never replace a viable original correction menu with a sparse
            // sample. This uses no new provider/SQL requests and does not
            // change ANY safety/quality floor or paid-photo budget.
            const unseenFeasible=slots.every(slot=>
              unseenPool.filter(p=>p.slot===slot).length>=3);
            const correctionPool=unseenFeasible?unseenPool:freshPool;
            editorial.feedbackInventory=unseenFeasible
              ?'unseen_source_skus':'original_source_skus';
            const enoughFresh=slots.every(slot=>
              correctionPool.filter(p=>p.slot===slot).length>=3);
            if(photoQuota>0&&enoughFresh){
              editorial.feedbackRevision='attempted';
              // selectVisualDeck returns display-ready look shells with
              // concept/evidence but WITHOUT the original brief+fashion.
              // Second portfolio feasibility replays MUST use the exact
              // initial approved SOURCE candidates, never display shells.
              const originalChosen=reviewed.outfits.flatMap(look=>{
                const source=candidates.find(x=>x.id===look.id);
                return source?[source]:[];
              });
              const reasons=(reviewed.diagnostics.visualQualityDecisions||[])
                .filter(row=>row.reason!=='approved').map(row=>row.reason).slice(0,6);
              const selectedArchitectures=originalChosen.slice(0,2).map(look=>
                Object.fromEntries(slots.map(slot=>[slot,
                  String(look.items?.find(it=>it.macroCategory===slot)?.subtype||'')
                    .slice(0,32)])));
              // No untrusted raw photo text/IDs are shipped to the second
              // designer: the dedicated SKU composer whitelists slots,
              // colors, scores and bounded photographic observations.
              const approvedPhotoEvidence=originalChosen.slice(0,2).flatMap(look=>{
                const photo=allCritiques.find(x=>
                  x.candidateId===look.id&&x.reject!==true);
                if(!photo)return [];
                return [{
                  roles:Object.fromEntries(slots.map(slot=>[slot,
                    String(look.items?.find(it=>it.macroCategory===slot)?.subtype||'')
                      .slice(0,32)])),
                  observations:photo.slotObservations,
                  colors:photo.slotColors,
                  relationships:photo.relationships,
                  scores:Object.fromEntries(
                    ['visualCohesion','styleFit','occasionFit',
                      'formalityFit','confidence'].map(axis=>[axis,photo[axis]])),
                }];
              });
              // The final PHOTO collection editor also starts a SECOND
              // real-SKU creative repair. Live Work/Date iOS had 0/3
              // approvals in this round: it previously saw approved looks
              // and generic rejection codes, but NOT the photographic
              // relationships that made other actual four-item candidates
              // fail. Give this autonomous creator the same bounded,
              // anonymized independent-photo evidence as the first repair.
              // These observations are descriptive DATA, not a SKU veto or
              // aesthetic rule; groundedPhotoRevisionFeedback sanitizes
              // every field before any provider sees it.
              const rejectedReasonById=new Map(
                (reviewed.diagnostics.visualQualityDecisions||[])
                  .filter(row=>row.reason&&row.reason!=='approved')
                  .map(row=>[row.candidateId,row.reason]));
              const rejectedPhotoEvidence=candidates.flatMap(look=>{
                const reason=rejectedReasonById.get(look.id);
                const photo=allCritiques.find(x=>x.candidateId===look.id);
                if(!reason||!photo)return [];
                return [{
                  reason,
                  roles:Object.fromEntries(slots.map(slot=>[slot,
                    String(look.items?.find(it=>it.macroCategory===slot)?.subtype||'')
                      .slice(0,32)])),
                  observations:photo.slotObservations,
                  colors:photo.slotColors,
                  relationships:photo.relationships,
                  scores:Object.fromEntries(
                    ['visualCohesion','styleFit','occasionFit',
                      'formalityFit','confidence'].map(axis=>[axis,photo[axis]])),
                }];
              }).slice(0,4);
              editorial.feedbackSource=explicitEditorialGap
                ?'visual_editor_gap':'approved_photo_evidence';
              let retry=null;
              try{
                retry=await selectCatalogRescueCompositions({
                  pool:correctionPool,
                  critiqueFeedback:{reasons,
                    editorPhotoGap:explicitEditorialGap
                      ?decision.creativeGap:undefined,
                    approvedArchitectures:selectedArchitectures,
                    approvedPhotoEvidence,
                    rejectedOutfits:rejectedPhotoEvidence},
                  directions:aiPlans,feedbackTaste,
                  context:{style:context.style,occasion:context.occasion,
                    brief:context.request,preferences:context.preferences,
                    weatherCelsius:Number.isFinite(context.weather?.temp)
                      ?context.weather.temp:null},
                  layerRequired:forceLayer||catalogFourItemIntent,
             allowLayer:!noLayer||catalogFourItemIntent,
                });
              }catch{/* preserve original PHOTO approved set */}
              editorial.feedbackRevisionStatus=
                String(retry?.status||'unavailable').slice(0,32);
              editorial.feedbackRevisionStudioCalls=
                Number.isSafeInteger(retry?.providerCalls)?retry.providerCalls:0;
              editorial.feedbackRevisionArtist=
                ['nemotron','nemotron_fallback','cross_lab_kimi',
              'nemotron_fast_photo_revision',
              'parallel_kimi_nemotron_photo_revision']
                  .includes(retry?.providerFamily)?retry.providerFamily:'not_run';
              if(retry?.status==='composed'&&Array.isArray(retry.looks)&&
                  photoRescueRemainingMs()>=16000){
                const validated=await sourceVerifyComposedOutfits({
                  looks:retry.looks,request:{...request,requiredIds},
                  requiredIds,hardGate,priceToBudgetCurrency,
                  scorer:fashionScorer,sourceMode:request.sourceMode,
                });
                const extras=[];
                for(const look of validated){
                  if(extras.length>=photoQuota||photoRescueRemainingMs()<15000)break;
                  if(candidates.some(c=>c.id===look.id))continue;
                  if(look.items.some(it=>alreadyApprovedProductKeys.has(
                       baseProductId(String(it.id)))))continue;
                  if(explicitColorPreferences&&
                    photographicPaletteEvidence(look,context.preferences,
                      requiredIds).blocked)continue;
                  if(await permitted(look))extras.push(look);
                }
                editorial.feedbackRevisionAdmitted=extras.length;
                if(extras.length>0&&photoRescueRemainingMs()>=14500){
                  const allowedMs=Math.max(1000,Math.min(8200,
                    Math.floor(photoRescueRemainingMs()-8300)));
                  let extraCritiques=[];
                  let revised=null;
                  try{
                    revised=await evaluateVisualShortlist(
                      {...retrieval,shortlist:extras},request,visualCritic,{
                        ...reviewOptions,target:extras.length,
                        criticSignal:AbortSignal.timeout(allowedMs),
                        onCritiques:rows=>{extraCritiques=rows;},
                      });
                  }catch{/* independent photo veto: original deck remains */}
                  editorial.feedbackRevisionAttempted=
                    revised?.diagnostics?.visualAttempted||0;
                  editorial.feedbackRevisionApproved=
                    revised?.diagnostics?.visualApproved||0;
                  if(revised?.outfits?.length>0&&
                      photoRescueRemainingMs()>=7300){
                    const acceptedExtraIds=new Set(
                      (revised.diagnostics.visualQualityDecisions||[])
                        .filter(row=>row.reason==='approved')
                        .map(row=>row.candidateId));
                    const approvedExtra=extras.filter(x=>
                      acceptedExtraIds.has(x.id));
                    // Reopen the original full source/photo feasibility menu:
                    // the new VLM editor can select only ORIGINAL approved
                    // looks + independently photo-accepted alternatives.
                    const allApproved=[...originalChosen,...approvedExtra]
                      .slice(0,MAX_VISUAL_CANDIDATES);
                    const allCritiquesRevised=[
                      ...allCritiques,...extraCritiques,
                    ];
                    const newMenus=verifiedPhotoEditorialPortfolios(
                      allApproved,allCritiquesRevised,reviewOptions);
                    if(approvedExtra.length>0&&newMenus.length>0){
                      let revisedDecision=null;
                      try{
                        revisedDecision=await selectPhotoApprovedPortfolio({
                          looks:allApproved,critiques:allCritiquesRevised,
                          portfolios:newMenus,
                          context:{style:context.style,occasion:context.occasion,
                            brief:context.request,preferences:context.preferences},
                        });
                      }catch{/* original editorial deck remains */}
                      if(revisedDecision?.status==='selected'&&
                        Array.isArray(revisedDecision.tokens)){
                        const rescored=[...new Set(revisedDecision.tokens)]
                          .map(token=>/^L[0-9]{2}$/.test(token)
                            ?allApproved[Number(token.slice(1))]:null)
                          .filter(Boolean).slice(0,3);
                        if(rescored.length===revisedDecision.tokens.length&&
                          rescored.some(look=>approvedExtra.some(x=>x.id===look.id))){
                          const replay=selectVisualDeck(
                            {...retrieval,shortlist:rescored},
                            allCritiquesRevised,
                            {...reviewOptions,target:rescored.length,
                              onCritiques:undefined});
                          if(replay.outfits.length===rescored.length){
                            const replayById=new Map(
                              replay.outfits.map(x=>[x.id,x]));
                            const first=reviewed.diagnostics;
                            reviewed={...reviewed,status:replay.status,
                              outfits:rescored.map(x=>replayById.get(x.id)),
                              diagnostics:{...first,
                                visualAttempted:first.visualAttempted+
                                  (revised.diagnostics.visualAttempted||0),
                                visualValid:first.visualValid+
                                  (revised.diagnostics.visualValid||0),
                                visualApproved:first.visualApproved+
                                  (revised.diagnostics.visualApproved||0),
                                visualRejected:first.visualRejected+
                                  (revised.diagnostics.visualRejected||0),
                                visualAbstained:first.visualAbstained+
                                  (revised.diagnostics.visualAbstained||0),
                                rejectedCandidateIds:[...new Set([
                                  ...(first.rejectedCandidateIds||[]),
                                  ...(revised.diagnostics.rejectedCandidateIds||[]),
                                ])],
                                visualQualityDecisions:[
                                  ...(first.visualQualityDecisions||[]),
                                  ...(revised.diagnostics.visualQualityDecisions||[]),
                                ],
                              }};
                            candidates.push(...extras);
                            editorial.chosen=rescored.length;
                            editorial.feedbackRevision='applied';
                            // The SECOND real PHOTO portfolio decision is
                            // the one the customer actually sees. Its different-
                            // lab jury (not the earlier F-token vote) must
                            // determine the emitted monitoring receipt.
                            if(['not_run','agreed','narrowed','disagreed',
                                'unavailable','standalone',
                                'arbitrated_second','arbitrated_primary']
                                .includes(revisedDecision.independentJury)){
                              editorial.independentJury=
                                revisedDecision.independentJury;
                              editorial.independentVoteSize=
                                Number.isInteger(revisedDecision.independentVoteSize)
                                ?Math.max(1,Math.min(3,revisedDecision.independentVoteSize))
                                :null;
                            }
                            editorial.status='applied';
                          }
                        }
                      }
                    }
                  }
                }
              }
            }else editorial.feedbackRevision='insufficient_safe_capacity';
          }
          modelFashionSelection.photoEditorial=editorial;
        }
      }
      visualQualityDecisions = Array.isArray(reviewed.diagnostics.visualQualityDecisions)
        ? reviewed.diagnostics.visualQualityDecisions.slice(
            0,reviewedLimit+rescueAttempted) : [];
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
      // Optional EVIDENCE collection only; it NEVER affects this request's
      // fashion outcome and is never sent to the user. Personal-wardrobe
      // anchors are excluded. For real catalog PHOTO tests with personalized
      // preferences, the server HMAC-isolates the ENTIRE preference context
      // before persistence: a photo-quality verdict on someone's avoided
      // colors must not become an unconditional general fashion rule.
      // Human thumbs are learned separately, never from these PHOTO labels.
      const journalEligible=anonymousPhotoJournalAllowed({
        sourceMode:context.sourceMode,requiredIds,preferences:context.preferences});
      modelFashionSelection.photoJournalScope=typeof photoRewardSink!=='function'
        ?'not_enabled':!primaryCritiques.length?'no_photo_evidence'
          :!journalEligible?'personal_context_excluded':'eligible';
      if(typeof photoRewardSink==='function'&&primaryCritiques.length>0&&
          journalEligible){
        try{await photoRewardSink({
          looks:photoJournalLooks,decisions:photoJournalDecisions,
          critiques:primaryCritiques,
          context:{style:context.style,occasion:context.occasion,
            preferences:context.preferences},
          featuresOf:items=>outfitStylingFeatures(items),
        });}catch{/* Nonessential training IO must NEVER fail a paid deck. */}
      }
      visualReviewCoverage = {
        mode: 'displayed_deck',
        inferenceCandidateCap: reviewedLimit+rescueAttempted,
        attempted: reviewed.diagnostics.visualAttempted,
        valid: reviewed.diagnostics.visualValid,
        rejected: rejected.size,
        approved: reviewed.diagnostics.visualApproved,
        // Only an approved, actually displayed look has review evidence.
        displayedEvidence: selected.filter(look => visualEvidenceById.has(look.id)).length,
        aiCreativeDirectorJudged: candidates.filter(look => aiBriefIds.has(look.briefId)).length,
        // Only server-generated art-slot numbers, NEVER SKU ids, user data,
        // private photos or raw AI fashion text in the safe audit receipt.
        aiCreativePlansJudged:aiPlans.flatMap((plan,index)=>
          candidates.some(look=>look.briefId===canonicalBriefId(plan.id))
            ?[index+1]:[]),
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
      feedbackTasteLearnedStatus: feedbackTaste?.learnedPreference?.status || 'untrained',
      modelFashionSelection,
      ...(visualReviewCoverage ? { visualReviewCoverage } : {}),
      ...(visualQualityDecisions ? { visualQualityDecisions } : {}),
      directions: directions.map((d) => d.id),
      qualifiedDirections: directions.filter((d) => d.qualified).length,
      // Measurable actual displayed architecture, not cosmetic direction
      // labels. Unknown/pinned layer evidence remains null.
      displayedLayerArchitectures: selected.map(look=>outerwearArchetype(look,requiredIds)),
      distinctLayerArchitectures: new Set(selected.map(look=>
        outerwearArchetype(look,requiredIds)).filter(Boolean)).size,
      aiCreativeDirectorBriefs: aiPlans.length,
      aiCreativeDirectorEligible: selectable.filter(look => aiPlans.some(plan => look.briefId === canonicalBriefId(plan.id))).length,
      aiCreativePlansEligible:aiPlans.flatMap((plan,index)=>
        selectable.some(look=>look.briefId===canonicalBriefId(plan.id))
          ?[index+1]:[]),
      aiCreativeDirectorDisplayed: selected.filter(look => aiPlans.some(plan => look.briefId === canonicalBriefId(plan.id))).length,
      neighborBriefs: neighborHouse.length,
      shortlistMix: { ...layerMix, best: ordered[0] ?? null, top20cutoff: ordered[19] ?? null },
      ...(floor.limited ? { exclusionQualityLimited: true } : {}) },
  };
}
