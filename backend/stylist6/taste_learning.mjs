/**
 * Stylist 6 — Unified Personal Taste Learning Engine.
 *
 * Unifies explicit look feedback (with surgical slot-level attribution:
 * `targetSlot`, `feedbackTags`, `swappedFromSku`) and implicit behavioral
 * signals (`saved_outfits`, `wear_logs`, `brand_clicks`) with exponential
 * recency decay, plus optional bounded 768-d visual taste centroids.
 *
 * Design invariants (aligned with STYLIST6_NORTH_STAR & ADR):
 * 1. Fixes the blunt `itemSum / 4` penalty bug: marking `wrong_shoes` or
 *    swapping a single SKU ONLY downweights that shoe SKU / slot subtype /
 *    slot color without unfairly penalizing the coat, top, or trousers.
 * 2. When scoring a whole look, explicitly rejected/targeted slot items carry
 *    full slot salience rather than being diluted 4x, while innocent sibling
 *    pieces remain completely unpenalized.
 * 3. Optional 768-d visual embedding cosine affinity is strictly bounded and
 *    is NEVER used alone as a fashion quality or coherence gate.
 * 4. Pure, deterministic (via injectable `nowMs`), zero I/O, zero PII in
 *    prompt summaries, and resilient to malformed or hostile payloads.
 */

export const TASTE_LEARNING_VERSION = 's6-unified-taste-learning-v1';
export const TASTE_EMBEDDING_DIM = 768;

// Exponential recency half-lives (days) by signal reliability.
export const SIGNAL_HALF_LIFE_DAYS = Object.freeze({
  explicit: 30,
  wearLog: 60,
  savedOutfit: 45,
  brandClick: 21,
});

// Relative signal strength hierarchy: explicit > wear_log > saved_outfit > brand_click.
export const SIGNAL_WEIGHTS = Object.freeze({
  explicit: 1.0,
  wearLog: 0.65,
  savedOutfit: 0.55,
  brandClick: 0.22,
});

// Bounded score envelopes — meaningful enough to shift recommendations,
// strictly bounded so taste can never override hard gates or quality floors.
export const MAX_ITEM_TASTE_SCORE = 0.35;
export const MAX_LOOK_TASTE_SCORE = 0.25;
export const MAX_VISUAL_ITEM_AFFINITY = 0.08;
export const MAX_VISUAL_LOOK_AFFINITY = 0.06;

export const UNIFIED_FEEDBACK_TAGS = Object.freeze([
  'wrong_shoes',
  'wrong_color',
  'too_formal',
  'too_casual',
  'not_my_fit',
  'over_budget',
  'love_palette',
  'love_silhouette',
  'slot_swapped',
]);

const DAY_MS = 86_400_000;
const MAX_AGE_DAYS = 365;
const MAX_EXPLICIT_ROWS = 120;
const MAX_IMPLICIT_ROWS = 200;
const MAX_ITEM_ID_LEN = 128;
const MAX_TOKEN_LEN = 64;

const SLOT_INDEX = Object.freeze({ top: 0, outerwear: 1, bottom: 2, shoes: 3 });
const SLOT_ALIASES = Object.freeze({
  top: 'top',
  tops: 'top',
  upper_body: 'top',
  outerwear: 'outerwear',
  layer: 'outerwear',
  jacket: 'outerwear',
  coat: 'outerwear',
  bottom: 'bottom',
  bottoms: 'bottom',
  lower_body: 'bottom',
  shoes: 'shoes',
  shoe: 'shoes',
  footwear: 'shoes',
  accessory: 'accessory',
  accessories: 'accessory',
  dress: 'dress',
  dresses: 'dress',
});

const TAG_SET = new Set(UNIFIED_FEEDBACK_TAGS);
const TAG_ALIASES = Object.freeze({
  wrong_shoe: 'wrong_shoes',
  wrong_footwear: 'wrong_shoes',
  footwear: 'wrong_shoes',
  great_fit: 'love_silhouette',
  love_fit: 'love_silhouette',
  wrong_fit: 'not_my_fit',
  bad_fit: 'not_my_fit',
  too_expensive: 'over_budget',
  swapped: 'slot_swapped',
});
const TAG_IMPLIES_SLOT = Object.freeze({
  wrong_shoes: 'shoes',
});

const COLOR_ALIASES = Object.freeze({
  gray: 'grey',
  '#0a1931': 'navy',
  midnight: 'navy',
  'midnight blue': 'navy',
  'dark blue': 'navy',
  'off-white': 'cream',
  'off white': 'cream',
  ecru: 'cream',
  oatmeal: 'beige',
  sand: 'beige',
  khaki: 'beige',
  tan: 'camel',
  cognac: 'brown',
  espresso: 'brown',
  chocolate: 'brown',
  bordeaux: 'burgundy',
  wine: 'burgundy',
  maroon: 'burgundy',
  slate: 'charcoal',
});

const FORMALITY_ORDER = Object.freeze({
  athletic: 0,
  casual: 1,
  smart_casual: 2,
  smart: 2.5,
  business_casual: 2.5,
  tailored: 3,
  formal: 4,
  black_tie: 5,
});

const DIRECTION_TRANSFER = Object.freeze({
  signature: 1.0,
  refresh: 0.5,
  discovery: 0.5,
  alternative: 0.25,
});

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const round4 = (value) => Math.round(value * 10_000) / 10_000;

function normToken(value) {
  if (typeof value !== 'string') return '';
  return value.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, MAX_TOKEN_LEN);
}

function normKey(value) {
  return normToken(value).replace(/[\s-]+/g, '_');
}

function cleanItemId(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim().slice(0, MAX_ITEM_ID_LEN);
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeSlot(value) {
  const key = normKey(value);
  return SLOT_ALIASES[key] || null;
}

function normalizeColor(value) {
  const raw = normToken(value);
  if (!raw || raw === 'unknown' || raw === 'none' || raw === 'missing' || raw === 'neutral') return null;
  return COLOR_ALIASES[raw] || raw;
}

function normalizeFormality(value) {
  const key = normKey(value);
  if (!key || key === 'unknown' || key === 'none') return null;
  if (key in FORMALITY_ORDER) return key;
  if (key.includes('formal') || key.includes('dressy')) return 'formal';
  if (key.includes('smart')) return 'smart_casual';
  if (key.includes('casual') || key.includes('relaxed')) return 'casual';
  if (key.includes('tailor') || key.includes('business')) return 'tailored';
  return null;
}

function normalizeSilhouette(value) {
  const key = normKey(value);
  if (!key || key === 'unknown' || key === 'none' || key === 'missing') return null;
  if (key === 'fitted' || key === 'skinny' || key === 'tailored_slim') return 'slim';
  if (key === 'loose' || key === 'baggy' || key === 'boxy' || key === 'wide_leg') return 'relaxed';
  if (key === 'regular' || key === 'straight' || key === 'classic' || key === 'balanced') return 'regular';
  if (key === 'oversized') return 'oversized';
  if (key === 'slim' || key === 'relaxed' || key === 'wide' || key === 'cropped' || key === 'structured') return key;
  return key;
}

function normalizeSubtype(value) {
  const raw = normToken(value).replace(/\s+/g, '-');
  if (!raw || raw === 'unknown' || raw === 'none' || raw === 'missing') return null;
  return raw;
}

function normalizeBrand(value) {
  const raw = normToken(value);
  if (!raw || raw === 'unknown' || raw === 'none' || raw === 'unbranded') return null;
  return raw;
}

function normalizeTags(raw) {
  if (!Array.isArray(raw)) return [];
  const seen = new Set();
  const out = [];
  for (const entry of raw.slice(0, 12)) {
    const key = normKey(entry);
    const canonical = TAG_SET.has(key) ? key : TAG_ALIASES[key] || null;
    if (canonical && !seen.has(canonical)) {
      seen.add(canonical);
      out.push(canonical);
    }
  }
  return out;
}

function parseTimestampMs(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.getTime();
  if (typeof value === 'string' && value.length >= 4) {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function recencyDecay(createdMs, nowMs, halfLifeDays) {
  if (createdMs === null) return 0;
  const ageDays = Math.max(0, (nowMs - createdMs) / DAY_MS);
  if (ageDays > MAX_AGE_DAYS) return 0;
  return 0.5 ** (ageDays / halfLifeDays);
}

function rowField(row, snake, camel) {
  if (!row || typeof row !== 'object' || Array.isArray(row)) return undefined;
  return row[snake] !== undefined ? row[snake] : row[camel];
}

function lookupCatalog(catalogById, id) {
  if (!catalogById || !id) return null;
  if (catalogById instanceof Map) return catalogById.get(id) || null;
  if (typeof catalogById === 'object') return catalogById[id] || null;
  return null;
}

function lookupEmbedding(embeddingsByImageSha, item) {
  if (!item) return null;
  const inline = item.embedding || item.embedding_vec;
  if (isValidEmbedding(inline)) return inline;
  if (!embeddingsByImageSha) return null;
  const keys = [
    item.imageSha256,
    item.image_sha256,
    item.visual_image_sha256,
    item.visual_fashion?.imageSha256,
    item.id,
  ].filter((k) => typeof k === 'string' && k.length > 0);
  for (const key of keys) {
    const vec = embeddingsByImageSha instanceof Map
      ? embeddingsByImageSha.get(key)
      : embeddingsByImageSha[key];
    if (isValidEmbedding(vec)) return vec;
  }
  return null;
}

function isValidEmbedding(vec) {
  if (!vec || (vec.length !== TASTE_EMBEDDING_DIM)) return false;
  if (!Array.isArray(vec) && !ArrayBuffer.isView(vec)) return false;
  let normSq = 0;
  for (let i = 0; i < TASTE_EMBEDDING_DIM; i++) {
    const v = vec[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) return false;
    normSq += v * v;
  }
  return normSq > 1e-8;
}

function normalizeVector(vec) {
  let normSq = 0;
  for (let i = 0; i < TASTE_EMBEDDING_DIM; i++) normSq += vec[i] * vec[i];
  const norm = Math.sqrt(normSq);
  if (!Number.isFinite(norm) || norm <= 1e-8) return null;
  const out = new Float32Array(TASTE_EMBEDDING_DIM);
  for (let i = 0; i < TASTE_EMBEDDING_DIM; i++) out[i] = vec[i] / norm;
  return out;
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== TASTE_EMBEDDING_DIM || b.length !== TASTE_EMBEDDING_DIM) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < TASTE_EMBEDDING_DIM; i++) {
    const av = a[i];
    const bv = b[i];
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  const denom = Math.sqrt(normA * normB);
  if (!Number.isFinite(denom) || denom <= 1e-8) return 0;
  return clamp(dot / denom, -1, 1);
}

function extractItemFacets(item, slotHint = null) {
  if (!item || typeof item !== 'object') {
    return { id: null, slot: normalizeSlot(slotHint), subtype: null, color: null, brand: null, formality: null, silhouette: null, texture: null, price: null };
  }
  const visualAttrs = item.visual_fashion?.attributes || item.visualFashion?.attributes || {};
  const intelDesc = item.garment_intelligence?.descriptor || item.fashion_intelligence?.descriptor || {};

  const pickVisual = (field) => {
    const obs = visualAttrs[field];
    if (obs && typeof obs === 'object' && typeof obs.value === 'string') {
      const conf = typeof obs.confidence === 'number' ? obs.confidence : 1;
      if (conf >= 0.75) return obs.value;
    }
    return null;
  };

  const slot = normalizeSlot(
    slotHint
      || item.slot
      || item.macroCategory
      || item.fashion_slot
      || pickVisual('slot')
      || item.category
      || item.garment_type,
  );
  const subtype = normalizeSubtype(
    item.subtype || item.fashion_subtype || pickVisual('subtype') || intelDesc.subtype || item.sub_category || item.type,
  );
  const color = normalizeColor(
    pickVisual('color') || item.color || item.primary_color || intelDesc.color,
  );
  const brand = normalizeBrand(item.brand);
  const formality = normalizeFormality(
    item.formality || item.fashion_formality || pickVisual('formality') || intelDesc.formality,
  );
  const silhouette = normalizeSilhouette(
    item.volume || item.fit || item.silhouette || pickVisual('volume') || intelDesc.volume,
  );
  const texture = normToken(pickVisual('texture') || item.texture || item.material || intelDesc.texture) || null;
  const price = typeof item.price === 'number' && Number.isFinite(item.price) && item.price > 0 ? item.price : null;

  return {
    id: cleanItemId(item.id || item.item_id || item.product_id),
    slot,
    subtype,
    color,
    brand,
    formality,
    silhouette,
    texture: texture && texture !== 'unknown' ? texture : null,
    price,
  };
}

function addWeighted(map, key, delta) {
  if (!key || !Number.isFinite(delta) || delta === 0) return;
  map.set(key, (map.get(key) || 0) + delta);
}

function saturateMap(rawMap, scale = 1.0) {
  const out = new Map();
  for (const [key, sum] of rawMap) {
    // Smooth tanh saturation prevents hostile repetition from blowing up weights
    // while keeping opposing signals cancelling cleanly.
    const val = Math.tanh(sum / scale);
    if (Math.abs(val) >= 0.005) out.set(key, round4(val));
  }
  return out;
}

function accumulateVector(acc, vec, weight) {
  if (!vec || !Number.isFinite(weight) || weight <= 0) return;
  const unit = normalizeVector(vec);
  if (!unit) return;
  for (let i = 0; i < TASTE_EMBEDDING_DIM; i++) {
    acc.sum[i] += unit[i] * weight;
  }
  acc.totalWeight += weight;
  acc.count += 1;
}

function extractSlotFeatureTuple(lookSignature, slot) {
  const idx = slot ? SLOT_INDEX[slot] : undefined;
  if (idx === undefined) return null;
  const features = lookSignature?.features;
  if (!Array.isArray(features) || features.length !== 4) return null;
  const tuple = features[idx];
  if (!Array.isArray(tuple) || tuple.length !== 5) return null;
  return {
    subtype: normalizeSubtype(tuple[0]),
    silhouette: normalizeSilhouette(tuple[1]),
    color: normalizeColor(tuple[2]),
    texture: normToken(tuple[3]) || null,
    pattern: normToken(tuple[4]) || null,
  };
}

function resolveFeedbackSlotTarget(row, rawItemIds, effectiveSlot, swappedSku, catalogById) {
  if (swappedSku) {
    return { sku: swappedSku, item: lookupCatalog(catalogById, swappedSku) };
  }
  const slotItems = rowField(row, 'slot_items', 'slotItems');
  if (slotItems && typeof slotItems === 'object' && !Array.isArray(slotItems) && effectiveSlot) {
    const id = cleanItemId(slotItems[effectiveSlot]);
    if (id) return { sku: id, item: lookupCatalog(catalogById, id) };
  }
  if (Array.isArray(rawItemIds) && rawItemIds.length > 0) {
    const ids = rawItemIds.slice(0, 8).map(cleanItemId).filter(Boolean);
    if (ids.length === 1) {
      return { sku: ids[0], item: lookupCatalog(catalogById, ids[0]) };
    }
    // Prefer matching slot via catalog lookup when catalogById is supplied
    if (effectiveSlot && catalogById) {
      for (const id of ids) {
        const catItem = lookupCatalog(catalogById, id);
        if (catItem && extractItemFacets(catItem).slot === effectiveSlot) {
          return { sku: id, item: catItem };
        }
      }
    }
    // Positional fallback for canonical 4-slot order [top, outerwear, bottom, shoes]
    const slotIdx = effectiveSlot ? SLOT_INDEX[effectiveSlot] : undefined;
    if (slotIdx !== undefined && ids.length === 4) {
      const id = ids[slotIdx];
      return { sku: id, item: lookupCatalog(catalogById, id) };
    }
  }
  return { sku: null, item: null };
}

/**
 * Builds a unified, recency-decayed personal taste profile from explicit
 * feedback (with slot attribution) and implicit behavioral signals.
 */
export function buildUnifiedTasteProfile({
  explicitFeedback = [],
  savedOutfits = [],
  wearLogs = [],
  brandClicks = [],
  catalogById = null,
  embeddingsByImageSha = null,
  nowMs = Date.now(),
} = {}) {
  const now = typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : Date.now();

  const rawSku = new Map();
  const rawTargetedSku = new Map();
  const rawSlotSubtype = new Map();
  const rawSlotColor = new Map();
  const rawColor = new Map();
  const rawBrand = new Map();
  const rawFormality = new Map();
  const rawSilhouette = new Map();
  const rawConceptPair = new Map();
  const rawTags = new Map();

  const posCentroidAcc = { sum: new Float64Array(TASTE_EMBEDDING_DIM), totalWeight: 0, count: 0 };
  const negCentroidAcc = { sum: new Float64Array(TASTE_EMBEDDING_DIM), totalWeight: 0, count: 0 };

  let formalityShiftSum = 0;
  let formalityShiftWeight = 0;
  let overBudgetSignal = 0;
  const priceSamples = [];
  const signalCounts = { explicit: 0, saved: 0, worn: 0, clicks: 0 };

  // Helper to apply item facets + optional embedding into preference accumulators
  const recordItemFacets = (facets, itemObj, weight, {
    includeSku = true,
    includeBrand = true,
    includeColor = true,
    includeGlobalColor = true,
    includeSilhouette = true,
    includeFormality = true,
    includeSlotSubtype = true,
    includeEmbedding = true,
    targeted = false,
    skuWeightScale = 1.0,
  } = {}) => {
    if (!Number.isFinite(weight) || weight === 0) return;
    if (includeSku && facets.id) {
      addWeighted(rawSku, facets.id, weight * skuWeightScale);
      if (targeted) addWeighted(rawTargetedSku, facets.id, weight * skuWeightScale);
    }
    if (includeSlotSubtype && facets.slot && facets.subtype) {
      addWeighted(rawSlotSubtype, `${facets.slot}|${facets.subtype}`, weight);
    }
    if (includeColor && facets.color) {
      if (facets.slot) addWeighted(rawSlotColor, `${facets.slot}|${facets.color}`, weight);
      if (includeGlobalColor) addWeighted(rawColor, facets.color, weight * 0.75);
    }
    if (includeBrand && facets.brand) {
      addWeighted(rawBrand, facets.brand, weight * 0.85);
    }
    if (includeFormality && facets.formality) {
      addWeighted(rawFormality, facets.formality, weight * 0.75);
    }
    if (includeSilhouette && facets.silhouette) {
      addWeighted(rawSilhouette, facets.silhouette, weight * 0.8);
    }
    if (includeEmbedding && itemObj) {
      const emb = lookupEmbedding(embeddingsByImageSha, itemObj);
      if (emb) {
        if (weight > 0) accumulateVector(posCentroidAcc, emb, weight);
        else accumulateVector(negCentroidAcc, emb, Math.abs(weight));
      }
    }
  };

  // 1. Explicit look feedback (stylist_look_feedback)
  if (Array.isArray(explicitFeedback)) {
    for (const row of explicitFeedback.slice(0, MAX_EXPLICIT_ROWS)) {
      if (!row || typeof row !== 'object') continue;
      const rating = row.rating === 1 ? 1 : row.rating === -1 ? -1 : null;
      const created = parseTimestampMs(rowField(row, 'created_at', 'createdAt'));
      if (rating === null || created === null) continue;
      const decay = recencyDecay(created, now, SIGNAL_HALF_LIFE_DAYS.explicit);
      if (decay <= 0) continue;

      const baseWeight = rating * decay * SIGNAL_WEIGHTS.explicit;
      const rawItemIds = rowField(row, 'item_ids', 'itemIds');
      const explicitSlot = normalizeSlot(rowField(row, 'target_slot', 'targetSlot'));
      const tags = normalizeTags(rowField(row, 'feedback_tags', 'feedbackTags'));
      const swappedSku = cleanItemId(rowField(row, 'swapped_from_sku', 'swappedFromSku'));
      const lookSig = rowField(row, 'look_signature', 'lookSignature');
      const impliedSlot = explicitSlot || tags.map((t) => TAG_IMPLIES_SLOT[t]).find(Boolean) || null;

      let rowContributed = false;
      for (const tag of tags) {
        addWeighted(rawTags, tag, decay);
        rowContributed = true;
      }

      // Handle formality / budget global preference shifts from explicit tags
      if (tags.includes('too_formal')) {
        formalityShiftSum -= decay;
        formalityShiftWeight += decay;
        addWeighted(rawFormality, 'formal', -decay * 0.9);
        addWeighted(rawFormality, 'casual', decay * 0.45);
        addWeighted(rawFormality, 'smart_casual', decay * 0.35);
      }
      if (tags.includes('too_casual')) {
        formalityShiftSum += decay;
        formalityShiftWeight += decay;
        addWeighted(rawFormality, 'casual', -decay * 0.9);
        addWeighted(rawFormality, 'tailored', decay * 0.45);
        addWeighted(rawFormality, 'smart_casual', decay * 0.4);
      }
      if (tags.includes('over_budget')) {
        overBudgetSignal += decay;
      }

      // Case A: Explicit swapped-away SKU (always a targeted negative signal on ONLY that SKU)
      if (swappedSku) {
        const swappedItem = lookupCatalog(catalogById, swappedSku);
        const swappedFacets = swappedItem
          ? extractItemFacets(swappedItem, impliedSlot)
          : {
            id: swappedSku,
            slot: impliedSlot,
            subtype: null,
            color: null,
            brand: null,
            formality: null,
            silhouette: null,
            texture: null,
            price: null,
          };
        recordItemFacets(swappedFacets, swappedItem, -decay * SIGNAL_WEIGHTS.explicit, {
          targeted: true,
          includeGlobalColor: false, // keep color penalty scoped to the swapped slot!
          includeBrand: false,
          includeFormality: false,
          includeSilhouette: tags.includes('not_my_fit'),
        });
        rowContributed = true;
      }

      const isSlotScopedNegative = rating === -1 && Boolean(impliedSlot);
      const isOnlyGlobalStyleComplaint = rating === -1
        && !impliedSlot
        && !swappedSku
        && tags.length > 0
        && tags.every((t) => ['too_formal', 'too_casual', 'wrong_color', 'not_my_fit', 'over_budget'].includes(t));

      // Case B: Slot-scoped negative feedback (e.g., `wrong_shoes` or `targetSlot: 'shoes'`)
      // Fixes the blunt itemSum/4 bug: ONLY downweights the offending slot's SKU,
      // subtype, color, and embedding; NEVER penalizes the coat/top/trouser!
      if (isSlotScopedNegative) {
        const target = resolveFeedbackSlotTarget(row, rawItemIds, impliedSlot, swappedSku, catalogById);
        const sigTuple = extractSlotFeatureTuple(lookSig, impliedSlot);
        if (target.sku && target.sku !== swappedSku) {
          const itemFacets = target.item
            ? extractItemFacets(target.item, impliedSlot)
            : {
              id: target.sku,
              slot: impliedSlot,
              subtype: sigTuple?.subtype || null,
              color: sigTuple?.color || null,
              brand: null,
              formality: null,
              silhouette: sigTuple?.silhouette || null,
              texture: sigTuple?.texture || null,
              price: null,
            };
          // Merge look_signature tuple if catalog lacked color/subtype
          if (!itemFacets.subtype && sigTuple?.subtype) itemFacets.subtype = sigTuple.subtype;
          if (!itemFacets.color && sigTuple?.color) itemFacets.color = sigTuple.color;
          if (!itemFacets.silhouette && sigTuple?.silhouette) itemFacets.silhouette = sigTuple.silhouette;
          recordItemFacets(itemFacets, target.item, baseWeight, {
            targeted: true,
            includeGlobalColor: false, // Do NOT ban brown coats because of wrong brown shoes!
            includeBrand: false,
            includeFormality: false,
            includeSilhouette: tags.includes('not_my_fit'),
          });
          rowContributed = true;
        } else if (sigTuple) {
          if (sigTuple.subtype) addWeighted(rawSlotSubtype, `${impliedSlot}|${sigTuple.subtype}`, baseWeight);
          if (sigTuple.color) addWeighted(rawSlotColor, `${impliedSlot}|${sigTuple.color}`, baseWeight);
          rowContributed = true;
        }
      } else if (isOnlyGlobalStyleComplaint) {
        // Attribute specifically to the complained facet (color, fit, formality, budget)
        // instead of indiscriminately banning all 4 SKUs.
        const ids = Array.isArray(rawItemIds) ? rawItemIds.slice(0, 8).map(cleanItemId).filter(Boolean) : [];
        const resolvedItems = ids.map((id) => ({ id, item: lookupCatalog(catalogById, id) }));
        if (tags.includes('wrong_color')) {
          for (const { item } of resolvedItems) {
            if (!item) continue;
            const f = extractItemFacets(item);
            if (f.color) addWeighted(rawColor, f.color, baseWeight * 0.9);
          }
          if (Array.isArray(lookSig?.features)) {
            for (const tuple of lookSig.features) {
              const c = normalizeColor(tuple?.[2]);
              if (c) addWeighted(rawColor, c, baseWeight * 0.5);
            }
          }
        }
        if (tags.includes('not_my_fit')) {
          for (const { item } of resolvedItems) {
            if (!item) continue;
            const f = extractItemFacets(item);
            if (f.silhouette) addWeighted(rawSilhouette, f.silhouette, baseWeight * 0.9);
          }
          if (Array.isArray(lookSig?.features)) {
            for (const tuple of lookSig.features) {
              const s = normalizeSilhouette(tuple?.[1]);
              if (s) addWeighted(rawSilhouette, s, baseWeight * 0.5);
            }
          }
        }
        rowContributed = true;
      } else {
        // Whole-look explicit feedback (positive or untargeted negative)
        const ids = Array.isArray(rawItemIds) ? rawItemIds.slice(0, 8).map(cleanItemId).filter(Boolean) : [];
        const slotOrder = ['top', 'outerwear', 'bottom', 'shoes'];
        const paletteBoost = tags.includes('love_palette') && rating === 1 ? 1.4 : 1.0;
        const silhouetteBoost = tags.includes('love_silhouette') && rating === 1 ? 1.4 : 1.0;

        for (let i = 0; i < ids.length; i++) {
          const id = ids[i];
          if (id === swappedSku) continue;
          const itemObj = lookupCatalog(catalogById, id);
          const posSlot = ids.length === 4 ? slotOrder[i] : null;
          const facets = itemObj
            ? extractItemFacets(itemObj, posSlot)
            : { id, slot: posSlot, subtype: null, color: null, brand: null, formality: null, silhouette: null, texture: null, price: null };
          const sigTuple = facets.slot ? extractSlotFeatureTuple(lookSig, facets.slot) : null;
          if (!facets.subtype && sigTuple?.subtype) facets.subtype = sigTuple.subtype;
          if (!facets.color && sigTuple?.color) facets.color = sigTuple.color;
          if (!facets.silhouette && sigTuple?.silhouette) facets.silhouette = sigTuple.silhouette;

          recordItemFacets(facets, itemObj, baseWeight * 0.65, {
            skuWeightScale: 1.0,
            includeColor: true,
            includeGlobalColor: true,
            includeSilhouette: true,
          });
          if (paletteBoost > 1 && facets.color) {
            addWeighted(rawColor, facets.color, decay * 0.45);
          }
          if (silhouetteBoost > 1 && facets.silhouette) {
            addWeighted(rawSilhouette, facets.silhouette, decay * 0.45);
          }
          rowContributed = true;
        }

        // Concept + direction transfer (only for whole-look ratings)
        const concept = normToken(rowField(row, 'concept_id', 'conceptId'));
        const direction = normToken(rowField(row, 'direction', 'directionLabel') || lookSig?.direction);
        if (concept && direction && DIRECTION_TRANSFER[direction] !== undefined) {
          addWeighted(rawConceptPair, `${concept}|${direction}`, baseWeight);
          rowContributed = true;
        }
      }

      if (rowContributed) signalCounts.explicit += 1;
    }
  }

  // 2. Implicit Signal: Saved Outfits (saved_outfits)
  if (Array.isArray(savedOutfits)) {
    for (const row of savedOutfits.slice(0, MAX_IMPLICIT_ROWS)) {
      if (!row || typeof row !== 'object') continue;
      const created = parseTimestampMs(
        rowField(row, 'updated_at', 'updatedAt') || rowField(row, 'created_at', 'createdAt') || row.date,
      );
      const decay = created !== null ? recencyDecay(created, now, SIGNAL_HALF_LIFE_DAYS.savedOutfit) : 0.5;
      if (decay <= 0) continue;
      const weight = decay * SIGNAL_WEIGHTS.savedOutfit;

      const rawItems = Array.isArray(row.items)
        ? row.items
        : Array.isArray(rowField(row, 'item_ids', 'itemIds'))
          ? rowField(row, 'item_ids', 'itemIds')
          : [];
      let contributed = false;
      for (const entry of rawItems.slice(0, 8)) {
        const id = typeof entry === 'string' ? cleanItemId(entry) : cleanItemId(entry?.id || entry?.item_id || entry?.product_id);
        const catItem = id ? lookupCatalog(catalogById, id) : null;
        const mergedObj = catItem || (typeof entry === 'object' ? entry : null);
        if (!mergedObj && !id) continue;
        const facets = extractItemFacets(mergedObj || { id });
        if (!facets.id && id) facets.id = id;
        recordItemFacets(facets, mergedObj, weight * 0.55, { skuWeightScale: 0.7 });
        if (facets.price) priceSamples.push(facets.price);
        contributed = true;
      }
      if (contributed) signalCounts.saved += 1;
    }
  }

  // 3. Implicit Signal: Wear Logs (wear_logs)
  if (Array.isArray(wearLogs)) {
    for (const row of wearLogs.slice(0, MAX_IMPLICIT_ROWS)) {
      if (!row || typeof row !== 'object') continue;
      const created = parseTimestampMs(row.date || rowField(row, 'created_at', 'createdAt'));
      const decay = created !== null ? recencyDecay(created, now, SIGNAL_HALF_LIFE_DAYS.wearLog) : 0.5;
      if (decay <= 0) continue;
      const weight = decay * SIGNAL_WEIGHTS.wearLog;

      const rawItems = Array.isArray(rowField(row, 'item_ids', 'itemIds'))
        ? rowField(row, 'item_ids', 'itemIds')
        : Array.isArray(row.items)
          ? row.items
          : [];
      let contributed = false;
      for (const entry of rawItems.slice(0, 8)) {
        const id = typeof entry === 'string' ? cleanItemId(entry) : cleanItemId(entry?.id || entry?.item_id || entry?.product_id);
        const catItem = id ? lookupCatalog(catalogById, id) : null;
        const mergedObj = catItem || (typeof entry === 'object' ? entry : null);
        if (!mergedObj && !id) continue;
        const facets = extractItemFacets(mergedObj || { id });
        if (!facets.id && id) facets.id = id;
        recordItemFacets(facets, mergedObj, weight * 0.65, { skuWeightScale: 0.8 });
        if (facets.price) priceSamples.push(facets.price);
        contributed = true;
      }
      if (contributed) signalCounts.worn += 1;
    }
  }

  // 4. Implicit Signal: Brand & Product Clicks (brand_clicks)
  if (Array.isArray(brandClicks)) {
    for (const row of brandClicks.slice(0, MAX_IMPLICIT_ROWS)) {
      if (!row || typeof row !== 'object') continue;
      const created = parseTimestampMs(
        rowField(row, 'clicked_at', 'clickedAt') || rowField(row, 'created_at', 'createdAt'),
      );
      const decay = created !== null ? recencyDecay(created, now, SIGNAL_HALF_LIFE_DAYS.brandClick) : 0.5;
      if (decay <= 0) continue;
      const weight = decay * SIGNAL_WEIGHTS.brandClick;

      const id = cleanItemId(rowField(row, 'item_id', 'itemId') || row.product_id || row.id);
      const brand = normalizeBrand(row.brand);
      const catItem = id ? lookupCatalog(catalogById, id) : null;
      let contributed = false;

      if (brand) {
        addWeighted(rawBrand, brand, weight);
        contributed = true;
      }
      if (catItem) {
        const facets = extractItemFacets(catItem);
        recordItemFacets(facets, catItem, weight * 0.6, {
          includeBrand: !brand, // avoid double-counting brand
          skuWeightScale: 0.45,
        });
        contributed = true;
      } else if (id) {
        addWeighted(rawSku, id, weight * 0.35);
        contributed = true;
      }
      const clickPrice = typeof row.price === 'number' && Number.isFinite(row.price) && row.price > 0
        ? row.price
        : null;
      if (clickPrice) priceSamples.push(clickPrice);
      if (contributed) signalCounts.clicks += 1;
    }
  }

  const skuAffinity = saturateMap(rawSku, 1.15);
  const targetedSkuAffinity = saturateMap(rawTargetedSku, 1.0);
  const slotSubtypeAffinity = saturateMap(rawSlotSubtype, 1.25);
  const slotColorAffinity = saturateMap(rawSlotColor, 1.25);
  const colorAffinity = saturateMap(rawColor, 1.5);
  const brandAffinity = saturateMap(rawBrand, 1.5);
  const formalityAffinity = saturateMap(rawFormality, 1.25);
  const silhouetteAffinity = saturateMap(rawSilhouette, 1.25);
  const conceptPairAffinity = saturateMap(rawConceptPair, 1.1);

  const positiveCentroid = posCentroidAcc.count > 0 ? normalizeVector(posCentroidAcc.sum) : null;
  const negativeCentroid = negCentroidAcc.count > 0 ? normalizeVector(negCentroidAcc.sum) : null;

  const formalityShift = formalityShiftWeight > 0
    ? round4(clamp(formalityShiftSum / Math.max(1, formalityShiftWeight), -1, 1))
    : 0;
  const priceSensitivity = round4(clamp(Math.tanh(overBudgetSignal / 1.5), 0, 1));
  const medianReferencePrice = priceSamples.length >= 3
    ? [...priceSamples].sort((a, b) => a - b)[Math.floor(priceSamples.length / 2)]
    : null;

  const totalSignals = signalCounts.explicit + signalCounts.saved + signalCounts.worn + signalCounts.clicks;
  const active = totalSignals > 0 && (
    skuAffinity.size > 0
    || slotSubtypeAffinity.size > 0
    || slotColorAffinity.size > 0
    || colorAffinity.size > 0
    || brandAffinity.size > 0
    || formalityAffinity.size > 0
    || silhouetteAffinity.size > 0
    || conceptPairAffinity.size > 0
    || Boolean(positiveCentroid)
    || Boolean(negativeCentroid)
    || Math.abs(formalityShift) > 0
    || priceSensitivity > 0
  );

  return {
    version: TASTE_LEARNING_VERSION,
    active,
    builtAtMs: now,
    signalCounts,
    skuAffinity,
    targetedSkuAffinity,
    slotSubtypeAffinity,
    slotColorAffinity,
    colorAffinity,
    brandAffinity,
    formalityAffinity,
    silhouetteAffinity,
    conceptPairAffinity,
    tagCounts: rawTags,
    formalityShift,
    priceSensitivity,
    medianReferencePrice,
    visualCentroids: {
      dim: TASTE_EMBEDDING_DIM,
      positiveCentroid,
      negativeCentroid,
      positiveCount: posCentroidAcc.count,
      negativeCount: negCentroidAcc.count,
      boundedMaxItemAffinity: MAX_VISUAL_ITEM_AFFINITY,
      boundedMaxLookAffinity: MAX_VISUAL_LOOK_AFFINITY,
      cannotOverrideQualityGate: true,
    },
    embeddingsByImageSha: embeddingsByImageSha || null,
  };
}

/**
 * Scores a single catalog/wardrobe item's personalized taste affinity in
 * `[-MAX_ITEM_TASTE_SCORE, +MAX_ITEM_TASTE_SCORE]`.
 *
 * Explicitly targeted negative SKUs (e.g. `wrong_shoes`, `swappedFromSku`)
 * carry strong negative affinity with `targetedPenalty: true`, while sibling
 * slots from that same feedback event remain untouched (`score: 0`).
 */
export function scoreItemTasteAffinity(item, slot, tasteProfile) {
  const neutral = {
    score: 0,
    targetedPenalty: false,
    slot: normalizeSlot(slot || item?.slot || item?.macroCategory),
    components: {
      sku: 0,
      slotSubtype: 0,
      slotColor: 0,
      color: 0,
      brand: 0,
      formality: 0,
      silhouette: 0,
      priceSensitivity: 0,
      visualEmbedding: 0,
    },
    reasons: [],
  };
  if (!tasteProfile?.active || !item || typeof item !== 'object') return neutral;

  const facets = extractItemFacets(item, slot);
  const reasons = [];

  // 1. Exact SKU affinity (distinguishing targeted slot penalties from general ratings)
  const targetedVal = facets.id ? (tasteProfile.targetedSkuAffinity?.get(facets.id) || 0) : 0;
  const generalSkuVal = facets.id ? (tasteProfile.skuAffinity?.get(facets.id) || 0) : 0;
  const targetedPenalty = targetedVal < -0.15;
  const skuComp = targetedPenalty
    ? targetedVal * 0.28
    : generalSkuVal * 0.14;
  if (targetedPenalty) reasons.push(`targeted_slot_reject:${facets.slot || 'item'}`);
  else if (generalSkuVal > 0.2) reasons.push('liked_sku');
  else if (generalSkuVal < -0.2) reasons.push('disliked_sku');

  // 2. Slot-scoped subtype & color affinity (e.g., dislikes chunky sneakers in shoes, not coats)
  const slotSubtypeVal = facets.slot && facets.subtype
    ? (tasteProfile.slotSubtypeAffinity?.get(`${facets.slot}|${facets.subtype}`) || 0)
    : 0;
  const slotSubtypeComp = slotSubtypeVal * 0.11;
  if (Math.abs(slotSubtypeVal) >= 0.25) {
    reasons.push(`${slotSubtypeVal > 0 ? 'liked' : 'disliked'}_slot_subtype:${facets.slot}:${facets.subtype}`);
  }

  const slotColorVal = facets.slot && facets.color
    ? (tasteProfile.slotColorAffinity?.get(`${facets.slot}|${facets.color}`) || 0)
    : 0;
  const slotColorComp = slotColorVal * 0.08;
  if (Math.abs(slotColorVal) >= 0.25) {
    reasons.push(`${slotColorVal > 0 ? 'liked' : 'disliked'}_slot_color:${facets.slot}:${facets.color}`);
  }

  // 3. Global palette, brand, formality, silhouette affinity
  const colorVal = facets.color ? (tasteProfile.colorAffinity?.get(facets.color) || 0) : 0;
  const colorComp = colorVal * 0.08;
  if (Math.abs(colorVal) >= 0.25) {
    reasons.push(`${colorVal > 0 ? 'liked' : 'disliked'}_color:${facets.color}`);
  }

  const brandVal = facets.brand ? (tasteProfile.brandAffinity?.get(facets.brand) || 0) : 0;
  const brandComp = brandVal * 0.07;
  if (Math.abs(brandVal) >= 0.25) {
    reasons.push(`${brandVal > 0 ? 'preferred' : 'avoided'}_brand:${facets.brand}`);
  }

  let formalityComp = 0;
  if (facets.formality) {
    const directFormality = tasteProfile.formalityAffinity?.get(facets.formality) || 0;
    formalityComp += directFormality * 0.07;
    if (tasteProfile.formalityShift !== 0 && facets.formality in FORMALITY_ORDER) {
      // Shift > 0 means user wants smarter/more formal; < 0 means more casual.
      const centeredLevel = (FORMALITY_ORDER[facets.formality] - 2.0) / 2.0; // approx [-1, +1]
      formalityComp += clamp(centeredLevel * tasteProfile.formalityShift * 0.06, -0.06, 0.06);
    }
    if (Math.abs(formalityComp) >= 0.025) {
      reasons.push(`${formalityComp > 0 ? 'aligned' : 'mismatched'}_formality:${facets.formality}`);
    }
  }

  const silhouetteVal = facets.silhouette ? (tasteProfile.silhouetteAffinity?.get(facets.silhouette) || 0) : 0;
  const silhouetteComp = silhouetteVal * 0.07;
  if (Math.abs(silhouetteVal) >= 0.25) {
    reasons.push(`${silhouetteVal > 0 ? 'preferred' : 'disliked'}_silhouette:${facets.silhouette}`);
  }

  // 4. Soft price sensitivity (when user tagged `over_budget` and we have reference prices)
  let priceComp = 0;
  if (tasteProfile.priceSensitivity > 0.15 && facets.price && tasteProfile.medianReferencePrice) {
    const ratio = facets.price / tasteProfile.medianReferencePrice;
    if (ratio > 1.35) {
      priceComp = -clamp((ratio - 1.35) * 0.05 * tasteProfile.priceSensitivity, 0, 0.05);
      reasons.push('price_above_comfort_band');
    } else if (ratio <= 0.95) {
      priceComp = clamp(0.02 * tasteProfile.priceSensitivity, 0, 0.02);
    }
  }

  // 5. Optional 768-d visual taste centroid cosine affinity (strictly bounded per ADR!)
  let visualComp = 0;
  const centroids = tasteProfile.visualCentroids;
  if (centroids && (centroids.positiveCentroid || centroids.negativeCentroid)) {
    const emb = lookupEmbedding(tasteProfile.embeddingsByImageSha, item);
    if (emb) {
      const posSim = centroids.positiveCentroid ? cosineSimilarity(emb, centroids.positiveCentroid) : 0;
      const negSim = centroids.negativeCentroid ? cosineSimilarity(emb, centroids.negativeCentroid) : 0;
      const netSim = centroids.positiveCentroid && centroids.negativeCentroid
        ? (posSim - negSim)
        : centroids.positiveCentroid
          ? (posSim - 0.15) // gently center baseline positive cosine
          : -(negSim - 0.15);
      visualComp = clamp(netSim * MAX_VISUAL_ITEM_AFFINITY, -MAX_VISUAL_ITEM_AFFINITY, MAX_VISUAL_ITEM_AFFINITY);
      if (Math.abs(visualComp) >= 0.02) {
        reasons.push(visualComp > 0 ? 'visual_centroid_match' : 'visual_centroid_avoid');
      }
    }
  }

  const rawTotal = skuComp
    + slotSubtypeComp
    + slotColorComp
    + colorComp
    + brandComp
    + formalityComp
    + silhouetteComp
    + priceComp
    + visualComp;

  const score = round4(clamp(rawTotal, -MAX_ITEM_TASTE_SCORE, MAX_ITEM_TASTE_SCORE));
  return {
    score,
    targetedPenalty,
    slot: facets.slot,
    components: {
      sku: round4(skuComp),
      slotSubtype: round4(slotSubtypeComp),
      slotColor: round4(slotColorComp),
      color: round4(colorComp),
      brand: round4(brandComp),
      formality: round4(formalityComp),
      silhouette: round4(silhouetteComp),
      priceSensitivity: round4(priceComp),
      visualEmbedding: round4(visualComp),
    },
    reasons: reasons.slice(0, 8),
  };
}

function extractLookItems(look) {
  if (Array.isArray(look)) {
    return look.filter(Boolean).map((item) => ({ item, slot: normalizeSlot(item.slot || item.macroCategory) }));
  }
  if (!look || typeof look !== 'object') return [];
  if (Array.isArray(look.items)) {
    return look.items.filter(Boolean).map((item) => ({ item, slot: normalizeSlot(item.slot || item.macroCategory) }));
  }
  const out = [];
  for (const slot of ['top', 'outerwear', 'bottom', 'shoes', 'dress', 'accessory']) {
    if (look[slot] && typeof look[slot] === 'object') {
      out.push({ item: look[slot], slot });
    }
  }
  return out;
}

/**
 * Scores a complete look against the user's unified taste profile.
 * Fixes the blunt `itemSum / 4` dilution bug: targeted slot penalties
 * (e.g., re-introducing a `wrong_shoes` SKU) preserve full salience while
 * innocent pieces stay unpenalized.
 */
export function scoreLookTasteAffinity(look, tasteProfile) {
  const inert = {
    active: false,
    score: 0,
    perSlot: {},
    targetedPenalties: [],
    visualScore: 0,
    conceptScore: 0,
    reasons: [],
  };
  if (!tasteProfile?.active) return inert;
  const entries = extractLookItems(look);
  if (entries.length === 0) return inert;

  const perSlot = {};
  const targetedPenalties = [];
  const reasonSet = new Set();
  let normalSum = 0;
  let normalCount = 0;
  let targetedPenaltySum = 0;
  let visualSum = 0;
  let visualCount = 0;

  for (const { item, slot } of entries.slice(0, 8)) {
    const res = scoreItemTasteAffinity(item, slot, tasteProfile);
    const slotKey = res.slot || slot || cleanItemId(item?.id) || `slot_${normalCount + targetedPenalties.length}`;
    perSlot[slotKey] = res;
    for (const r of res.reasons) reasonSet.add(r);

    // Track visual component separately so we can enforce the ADR look cap cleanly.
    const nonVisualScore = res.score - res.components.visualEmbedding;
    if (res.components.visualEmbedding !== 0) {
      visualSum += res.components.visualEmbedding;
      visualCount += 1;
    }

    if (res.targetedPenalty) {
      targetedPenalties.push({ slot: slotKey, productId: cleanItemId(item?.id), score: res.score });
      // Full targeted slot salience — NEVER diluted by 1/4!
      targetedPenaltySum += nonVisualScore;
    } else {
      normalSum += nonVisualScore;
      normalCount += 1;
    }
  }

  const normalMean = normalCount > 0 ? normalSum / normalCount : 0;
  const visualScore = round4(
    clamp(
      visualCount > 0 ? visualSum / visualCount : 0,
      -MAX_VISUAL_LOOK_AFFINITY,
      MAX_VISUAL_LOOK_AFFINITY,
    ),
  );

  // Concept + direction transfer when look carries conceptId
  let conceptScore = 0;
  const conceptId = normToken(look?.conceptId || look?.concept_id);
  if (conceptId && tasteProfile.conceptPairAffinity?.size > 0) {
    let rawConcept = 0;
    for (const [key, val] of tasteProfile.conceptPairAffinity) {
      if (!key.startsWith(`${conceptId}|`)) continue;
      const dir = key.slice(conceptId.length + 1);
      rawConcept += (DIRECTION_TRANSFER[dir] || 0) * val;
    }
    conceptScore = round4(clamp(rawConcept * 0.05, -0.05, 0.05));
    if (Math.abs(conceptScore) >= 0.015) {
      reasonSet.add(conceptScore > 0 ? 'liked_direction_concept' : 'disliked_direction_concept');
    }
  }

  const combined = normalMean + targetedPenaltySum + visualScore + conceptScore;
  const score = round4(clamp(combined, -MAX_LOOK_TASTE_SCORE, MAX_LOOK_TASTE_SCORE));

  return {
    active: true,
    score,
    perSlot,
    targetedPenalties,
    visualScore,
    conceptScore,
    reasons: [...reasonSet].slice(0, 10),
  };
}

function sanitizePromptWord(value) {
  return String(value || '')
    .replace(/https?:\/\/\S+|\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b/gi, '')
    .replace(/[\u0000-\u001f<>{}\[\]`"'\\]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 32);
}

function topEntries(map, { positive = true, minAbs = 0.2, limit = 5 } = {}) {
  if (!(map instanceof Map)) return [];
  return [...map.entries()]
    .filter(([, val]) => (positive ? val >= minAbs : val <= -minAbs))
    .sort((a, b) => (positive ? b[1] - a[1] : a[1] - b[1]) || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([k]) => sanitizePromptWord(k))
    .filter(Boolean);
}

/**
 * Generates an anonymous, zero-PII, injection-safe summary and prompt block
 * for the Creative Director and SKU Composer. Never emits user IDs, raw SKU
 * IDs, timestamps, URLs, or raw feedback strings.
 */
export function summarizeTasteForPrompt(tasteProfile) {
  const empty = {
    active: false,
    preferredColors: [],
    avoidedColors: [],
    preferredBrands: [],
    preferredSilhouettes: [],
    avoidedSilhouettes: [],
    formalityShift: 'balanced',
    priceSensitivity: 'neutral',
    slotPreferences: {},
    signalCounts: { explicit: 0, saved: 0, worn: 0, clicks: 0 },
    promptBlock: '',
  };
  if (!tasteProfile?.active) return empty;

  const preferredColors = topEntries(tasteProfile.colorAffinity, { positive: true, minAbs: 0.2, limit: 5 });
  const avoidedColors = topEntries(tasteProfile.colorAffinity, { positive: false, minAbs: 0.2, limit: 4 });
  const preferredBrands = topEntries(tasteProfile.brandAffinity, { positive: true, minAbs: 0.2, limit: 5 });
  const preferredSilhouettes = topEntries(tasteProfile.silhouetteAffinity, { positive: true, minAbs: 0.2, limit: 3 });
  const avoidedSilhouettes = topEntries(tasteProfile.silhouetteAffinity, { positive: false, minAbs: 0.2, limit: 3 });

  const formalityShift = tasteProfile.formalityShift >= 0.25
    ? 'more_formal'
    : tasteProfile.formalityShift <= -0.25
      ? 'more_casual'
      : 'balanced';
  const priceSensitivity = tasteProfile.priceSensitivity >= 0.25 ? 'value_conscious' : 'neutral';

  const slotPreferences = {};
  for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) {
    const likedSubtypes = [];
    const dislikedSubtypes = [];
    const likedColors = [];
    const dislikedColors = [];
    const prefix = `${slot}|`;
    for (const [k, v] of tasteProfile.slotSubtypeAffinity || []) {
      if (!k.startsWith(prefix)) continue;
      const val = sanitizePromptWord(k.slice(prefix.length));
      if (!val) continue;
      if (v >= 0.25) likedSubtypes.push([val, v]);
      else if (v <= -0.25) dislikedSubtypes.push([val, v]);
    }
    for (const [k, v] of tasteProfile.slotColorAffinity || []) {
      if (!k.startsWith(prefix)) continue;
      const val = sanitizePromptWord(k.slice(prefix.length));
      if (!val) continue;
      if (v >= 0.25) likedColors.push([val, v]);
      else if (v <= -0.25) dislikedColors.push([val, v]);
    }
    const sortDesc = (arr) => arr.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 3).map((x) => x[0]);
    const sortAsc = (arr) => arr.sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0])).slice(0, 3).map((x) => x[0]);
    const ls = sortDesc(likedSubtypes);
    const ds = sortAsc(dislikedSubtypes);
    const lc = sortDesc(likedColors);
    const dc = sortAsc(dislikedColors);
    if (ls.length || ds.length || lc.length || dc.length) {
      slotPreferences[slot] = {
        likedSubtypes: ls,
        dislikedSubtypes: ds,
        likedColors: lc,
        dislikedColors: dc,
      };
    }
  }

  const lines = [];
  if (preferredColors.length) lines.push(`Colors gravitated toward: ${preferredColors.join(', ')}.`);
  if (avoidedColors.length) lines.push(`Colors recently avoided: ${avoidedColors.join(', ')}.`);
  if (preferredSilhouettes.length) lines.push(`Preferred silhouettes/fits: ${preferredSilhouettes.join(', ')}.`);
  if (avoidedSilhouettes.length) lines.push(`Avoided silhouettes/fits: ${avoidedSilhouettes.join(', ')}.`);
  if (formalityShift === 'more_formal') lines.push('Formality preference shift: prefers slightly smarter/more tailored polish.');
  if (formalityShift === 'more_casual') lines.push('Formality preference shift: prefers slightly more relaxed/less formal ease.');
  if (preferredBrands.length) lines.push(`Frequently saved/clicked brand aesthetics: ${preferredBrands.join(', ')}.`);
  if (priceSensitivity === 'value_conscious') lines.push('Budget sensitivity: prioritize strong value-for-money pieces well inside budget.');

  for (const [slot, prefs] of Object.entries(slotPreferences)) {
    const parts = [];
    if (prefs.likedSubtypes.length) parts.push(`favor ${prefs.likedSubtypes.join('/')}`);
    if (prefs.dislikedSubtypes.length) parts.push(`avoid ${prefs.dislikedSubtypes.join('/')}`);
    if (prefs.likedColors.length) parts.push(`liked hues ${prefs.likedColors.join('/')}`);
    if (prefs.dislikedColors.length) parts.push(`avoided hues ${prefs.dislikedColors.join('/')}`);
    if (parts.length) lines.push(`Slot guidance (${slot}): ${parts.join('; ')}.`);
  }

  const promptBlock = lines.length
    ? `LEARNED PERSONAL TASTE (soft preference signals; never override occasion, weather, budget, or quality gates):\n- ${lines.join('\n- ')}`.slice(0, 900)
    : '';

  return {
    active: true,
    preferredColors,
    avoidedColors,
    preferredBrands,
    preferredSilhouettes,
    avoidedSilhouettes,
    formalityShift,
    priceSensitivity,
    slotPreferences,
    signalCounts: { ...tasteProfile.signalCounts },
    promptBlock,
  };
}
