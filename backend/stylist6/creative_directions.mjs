/**
 * Stylist 6: three intentional, bounded directions from ALREADY feasible looks.
 *
 * The selected style/occasion is the anchor, not a prison. We do not ask for a
 * longer onboarding form and never relax hard gates, budget, provenance, or
 * coherence to manufacture novelty. If the catalogue lacks structural variety,
 * show an honestly labelled alternative rather than claiming a new direction.
 *
 * This module is pure: the caller supplies previously validated looks,
 * canonical product overlap and evidence-based styling features.
 */
import { editorialDirectionEligible } from './editorial_quality.mjs';

export const DIRECTION_VERSION = 's6-three-directions-editorial-v4';
const MAX_POOL = 72;
const QUALITY_WINDOW = 0.16;
const EPS = 1e-9;
const SLOT_NAMES = ['top', 'outerwear', 'bottom', 'shoes'];
// Deck brand-monotony tie-break. Three cards that all draw from the same
// house read like one capsule repeated, which is exactly the "duplicate
// upper_body / same brand everywhere" complaint a varied catalog should
// answer. Same bounded-magnitude convention as the footwear-variety
// tie-break below: a look whose brands were all already drawn earlier in
// the deck pays at most -0.035 (scaled by the share of its brands that are
// repeats), and introducing a brand the deck has not used yet earns the
// small +0.012 nudge. The worst-case deck-level swing (2 x 0.035 across the
// refresh+discovery pair) therefore matches the existing footwear
// convention's deck-level swing: enough to decide a near-tie, never enough
// to override a clear quality winner, and a constant (no-op) offset on a
// single-brand catalog.
const BRAND_REPEAT_PENALTY_MAX = 0.035;
const BRAND_NOVELTY_BONUS = 0.012;
const normalizeBrand = (value) => String(value || '').trim().toLowerCase();

const known = (x) => typeof x === 'string' && !!x &&
  x !== 'unknown' && x !== 'neutral' && x !== 'missing';
const delta = (a, b) => known(a) && known(b) && a !== b;

function score(look) {
  return Number.isFinite(look?.fashion?.score) ? look.fashion.score : -Infinity;
}

function baselineOrder(a, b) {
  return score(b) - score(a) || String(a.id).localeCompare(String(b.id));
}

export function directionDifference(a, b, featuresOf, featureDistance) {
  const left = featuresOf(a), right = featuresOf(b);
  const changedSlots = [], structuralSlots = [], paletteSlots = [];
  for (let i = 0; i < SLOT_NAMES.length; i++) {
    const x = left?.[i], y = right?.[i];
    // Both empty: e.g. an optional layer absent, or a locked anchor.
    if (!x && !y) continue;
    // One missing: a genuine layer change. Caller excludes fixed anchors
    // from both feature vectors, so no anchor earns novelty credit.
    if (!x || !y) {
      changedSlots.push(SLOT_NAMES[i]);
      structuralSlots.push(SLOT_NAMES[i]);
      continue;
    }
    const structural = [0, 1, 3, 4].some(k => delta(x[k], y[k]));
    const palette = delta(x[2], y[2]);
    if (structural || palette) changedSlots.push(SLOT_NAMES[i]);
    if (structural) structuralSlots.push(SLOT_NAMES[i]);
    if (palette) paletteSlots.push(SLOT_NAMES[i]);
  }
  const measured = featureDistance(left, right);
  // Layered/unlayered changes were omitted by the shared paired-feature
  // distance when one slot is empty; credit only the verified slot swap.
  const layerChanged = structuralSlots.includes('outerwear') && (!left?.[1] || !right?.[1]);
  const distance = Math.max(Number.isFinite(measured) ? measured : 0, layerChanged ? 0.3 : 0);
  return {
    distance: Math.max(0, Math.min(1, distance)),
    changedSlots, structuralSlots, paletteSlots,
  };
}

const refreshWorthShowing = (diff) =>
  diff.distance >= 0.12 && diff.changedSlots.length >= 1;

const discoveryWorthShowing = (base, previous) =>
  base.distance >= 0.24 &&
  (base.structuralSlots.length >= 1 || base.paletteSlots.length >= 3) &&
  (!previous || previous.distance >= 0.16);

function allowedTogether(a, b, overlap) {
  // Existing production boundary: two different purchased/owned products
  // outside an optional pinned anchor, including aliases and photo matches.
  const shared = overlap(a, b);
  return Math.max(a.items.length, b.items.length) - shared >= 2;
}

/**
 * A deck with the same shoes in every card can feel like one outfit repeated.
 * Favor a different actual shoe SKU only within the existing quality window;
 * never invent a new pair, fail a valid sparse-catalog deck, or override an
 * explicitly pinned user anchor. Canonical overlap is caller-supplied so
 * catalog aliases/photos count as the same product.
 */
export function footwearVarietyAdjustment(candidate, chosen, overlap) {
  const shoe = candidate?.items?.filter(item => item?.macroCategory === 'shoes') || [];
  if (!shoe.length || typeof overlap !== 'function') return 0;
  const earlier = (chosen || []).flatMap(look =>
    (look?.items || []).filter(item => item?.macroCategory === 'shoes'));
  if (!earlier.length) return 0;
  return earlier.some(item => overlap({items:shoe},{items:[item]}) > 0) ? -0.035 : 0.012;
}

/**
 * Owner iOS live fashion QA, 2026-11-13: a deck can change tops and
 * jackets yet recycle the same trousers AND shoes in multiple cards.
 * This is a bounded *tie-break*, not a feasibility rule. The existing
 * 2-SKU difference hard gate is unchanged. A single repeated shoe or
 * trouser incurs NO extra penalty (sparse catalog / pinned anchors).
 * Canonical overlap is delegated to the caller (aliases/photos match).
 */
export function lowerBodyPairRepeatAdjustment(candidate, chosen, overlap) {
  if (!Array.isArray(candidate?.items) || typeof overlap !== 'function') return 0;
  const of = (look, slot) => look?.items?.find(item=>item?.macroCategory===slot) || null;
  const bottom = of(candidate, 'bottom'), shoe = of(candidate, 'shoes');
  if (!bottom || !shoe) return 0;
  for (const old of chosen || []) {
    const prevBottom = of(old, 'bottom'), prevShoe = of(old, 'shoes');
    if (!prevBottom || !prevShoe) continue;
    if (overlap({items:[bottom]},{items:[prevBottom]})>0 &&
        overlap({items:[shoe]},{items:[prevShoe]})>0) return -0.025;
  }
  return 0;
}

/**
 * Deck brand-monotony tie-break. Brand evidence comes only from catalog
 * metadata (item.brand): a look with no known brands stays neutral, so a
 * sparse or brand-less catalog can never be penalized on invented claims.
 * Proportional to the overlap between this look's brand set and the union
 * of the brand sets already chosen for the deck: full repeat -0.035, partial
 * repeat scaled by the repeat share, all-new brand set +0.012. Exported for
 * tests.
 */
export function brandVarietyAdjustment(candidate, chosen) {
  const brands = new Set((candidate?.items || [])
    .map((item) => normalizeBrand(item?.brand)).filter(Boolean));
  if (!brands.size) return 0;
  const earlier = new Set((chosen || []).flatMap((look) =>
    (look?.items || []).map((item) => normalizeBrand(item?.brand))).filter(Boolean));
  if (!earlier.size) return 0;
  const repeats = [...brands].filter((brand) => earlier.has(brand)).length;
  return repeats ? -BRAND_REPEAT_PENALTY_MAX * repeats / brands.size : BRAND_NOVELTY_BONUS;
}

function pick(pool, chosen, overlap, predicate, rank) {
  let best = null, utility = -Infinity;
  for (const look of pool) {
    if (chosen.some(other => other.id === look.id ||
      !allowedTogether(other, look, overlap))) continue;
    const evidence = chosen.map(other => rank.diff(other, look));
    if (!predicate(look, evidence)) continue;
    const value = rank.value(look, evidence) +
      footwearVarietyAdjustment(look, chosen, overlap) +
      lowerBodyPairRepeatAdjustment(look, chosen, overlap) +
      brandVarietyAdjustment(look, chosen);
    if (value > utility + EPS || (Math.abs(value - utility) <= EPS &&
      String(look.id).localeCompare(String(best?.id || '')) < 0)) {
      best = look; utility = value;
    }
  }
  return best;
}

/**
 * Select signature -> refreshed interpretation -> considered discovery.
 * All candidates have already passed hard SKU/stock-evidence/budget/coherence
 * admission. The bounded score window prevents novelty from dominating fit.
 * Fallbacks keep valid looks but never receive deceptive discovery labels.
 */
export function selectCreativeDirectionDeck(shortlist, {
  featuresOf, featureDistance, overlap, singleCurrencyBonus = () => 0,
} = {}) {
  if (typeof featuresOf !== 'function' || typeof featureDistance !== 'function' ||
    typeof overlap !== 'function') throw new TypeError('creative_direction_metrics_required');

  const pool = (Array.isArray(shortlist) ? shortlist : [])
    .filter(x => x && Array.isArray(x.items) && Number.isFinite(x.fashion?.score))
    .sort(baselineOrder).slice(0, MAX_POOL);
  if (!pool.length) return {looks:[], directions:[], diagnostics:{directionVersion:DIRECTION_VERSION, qualified:0, alternatives:0}};
  const eligible = pool.filter(x => score(x) >= score(pool[0]) - QUALITY_WINDOW);
  const featureCache = new Map(eligible.map(look => [look.id, featuresOf(look)]));
  const features = look => featureCache.get(look.id) || featuresOf(look);
  const diff = (a,b) => directionDifference(a,b,features,featureDistance);
  const bonus = look => singleCurrencyBonus(look.items);
  const differentBrief = (a,b) => a?.briefId !== b?.briefId ? 0.012 : 0;
  const signature = eligible[0];
  const chosen = [signature];
  const refreshedRank = {
    diff,
    value: (look, [d]) => score(look) + bonus(look) +
      0.11 * d.distance - 0.06 * Math.abs(d.distance - 0.35) -
      0.20 * Math.max(0, d.distance - 0.42) +
      differentBrief(signature, look),
  };
  const discoveredRank = {
    diff,
    value: (look, [d1,d2]) => score(look) + bonus(look) +
      .16 * Math.min(d1.distance,d2.distance) +
      .025 * Math.min(d1.structuralSlots.length,2) +
      differentBrief(signature,look),
  };
  // Plan the second and third directions JOINTLY. Greedy selection would
  // spend the only distinctive silhouette as "refresh" and leave three
  // visually similar results. The bounded 72x72 search preserves quality.
  let plannedPair = null, bestPairValue = -Infinity;
  for (const fresh of eligible) {
    if (fresh.id === signature.id || !allowedTogether(signature, fresh, overlap)) continue;
    const firstDiff = diff(signature, fresh);
    if (!refreshWorthShowing(firstDiff) || !editorialDirectionEligible(signature, fresh, 'refresh')) continue;
    const discovered = pick(eligible, [signature, fresh], overlap,
      (look, [d1,d2]) => discoveryWorthShowing(d1,d2) && editorialDirectionEligible(signature, look, 'discovery'),
      discoveredRank);
    if (!discovered) continue;
    const joint = refreshedRank.value(fresh, [firstDiff]) +
      footwearVarietyAdjustment(fresh, [signature], overlap) +
      lowerBodyPairRepeatAdjustment(fresh, [signature], overlap) +
      brandVarietyAdjustment(fresh, [signature]) +
      discoveredRank.value(discovered, [diff(signature,discovered),diff(fresh,discovered)]) +
      footwearVarietyAdjustment(discovered, [signature,fresh], overlap) +
      lowerBodyPairRepeatAdjustment(discovered, [signature,fresh], overlap) +
      brandVarietyAdjustment(discovered, [signature,fresh]);
    if (joint > bestPairValue + EPS || (Math.abs(joint-bestPairValue)<=EPS &&
      `${fresh.id}|${discovered.id}`.localeCompare(
        plannedPair ? `${plannedPair[0].id}|${plannedPair[1].id}` : '') < 0)) {
      plannedPair = [fresh,discovered]; bestPairValue = joint;
    }
  }
  if (plannedPair) chosen.push(...plannedPair);
  else {
    const refreshed = pick(eligible, chosen, overlap,
      (look, distances) => refreshWorthShowing(distances[0]) && editorialDirectionEligible(signature, look, 'refresh'), refreshedRank);
    if (refreshed) chosen.push(refreshed);
    else {
      const fallback = pick(eligible, chosen, overlap, (look) => editorialDirectionEligible(signature, look, 'alternative'), {
        diff,
        value: (look, [d]) => score(look) + bonus(look) + .07 * d.distance,
      });
      if (fallback) chosen.push(fallback);
    }
    if (chosen.length === 2) {
      const discovered = pick(eligible, chosen, overlap,
        (look, [d1,d2]) => discoveryWorthShowing(d1,d2) && editorialDirectionEligible(signature, look, 'discovery'), discoveredRank);
      if (discovered) chosen.push(discovered);
      else {
        const fallback = pick(eligible, chosen, overlap, (look) => editorialDirectionEligible(signature, look, 'alternative'), {
          diff,
          value: (look, d) => score(look) + bonus(look) +
            .11 * Math.min(...d.map(x => x.distance)),
        });
        if (fallback) chosen.push(fallback);
      }
    }
  }
  const directions = labelCreativeDirections(chosen,{featuresOf: features,featureDistance});
  return {
    looks:chosen, directions,
    diagnostics:{
      directionVersion:DIRECTION_VERSION,
      qualified:directions.filter(d=>d.qualified).length,
      alternatives:directions.filter(d=>d.id==='alternative').length,
      candidateCount:pool.length,
      qualityEligible:eligible.length,
      selected:directions.map(d=>d.id),
      // Measurable output QA: lower-body combinations repeated across
      // rendered looks, regardless of whether sparse inventory forced them.
      repeatedLowerPairs: chosen.slice(1).filter((look,index)=>
        lowerBodyPairRepeatAdjustment(look,chosen.slice(0,index+1),overlap)<0).length,
    },
  };
}

/** Re-evaluate only the final displayed order after an optional visual swap. */
export function labelCreativeDirections(looks, {featuresOf, featureDistance}) {
  return (looks || []).map((look, index) => {
    if (index === 0) return {
      id:'signature',qualified:true,
      evidence:{changedSlots:[],structuralSlots:[],paletteSlots:[],distance:0},
    };
    const fromSignature = directionDifference(looks[0], look, featuresOf, featureDistance);
    const fromRefresh = index >= 2
      ? directionDifference(looks[1], look, featuresOf, featureDistance)
      : null;
    const qualified = index === 1
      ? refreshWorthShowing(fromSignature) && editorialDirectionEligible(looks[0], look, 'refresh')
      : discoveryWorthShowing(fromSignature, fromRefresh) &&
        editorialDirectionEligible(looks[0], look, 'discovery');
    return {
      id:qualified ? (index === 1 ? 'refresh' : 'discovery') : 'alternative',
      qualified,
      evidence:{
        changedSlots:fromSignature.changedSlots,
        structuralSlots:fromSignature.structuralSlots,
        paletteSlots:fromSignature.paletteSlots,
        distance:Number(fromSignature.distance.toFixed(3)),
      },
    };
  });
}
