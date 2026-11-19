/**
 * Stylist 6 — Catalog Visual Quality Flywheel.
 *
 * Turns independent Visual Critic outcomes into a self-healing catalog quality
 * signal using Bayesian + Wilson lower-bound shrinkage:
 * - Unseen SKUs start neutral at ~0.65 (no cold-start starvation).
 * - Consistently photo-approved, high-editorial SKUs climb cleanly toward 0.95+.
 * - Chronic photo-defect, mislabeled-category, or consistently rejected SKUs
 *   automatically downrank and quarantine without manual regex whac-a-mole.
 *
 * Quarantine rules:
 * - `photoDefects >= 3` -> quarantined (`chronic_photo_defects`)
 * - `categoryMismatches >= 2` -> quarantined (`category_mismatch`)
 * - `reviews >= 5` AND Wilson lower bound `< 0.25` -> quarantined (`low_wilson_approval`)
 */

export const QUALITY_FLYWHEEL_VERSION = 's6-catalog-quality-flywheel-v1';
export const DEFAULT_NEUTRAL_QUALITY_SCORE = 0.65;

export const QUARANTINE_THRESHOLDS = Object.freeze({
  maxPhotoDefects: 3,
  maxCategoryMismatches: 2,
  minReviewsForWilson: 5,
  minWilsonScore: 0.25,
  wilsonZ: 1.28155, // 90% one-sided / 80% two-sided confidence bound
});

const BAYESIAN_PRIOR_MEAN = DEFAULT_NEUTRAL_QUALITY_SCORE;
const BAYESIAN_PRIOR_WEIGHT = 4;
const EDITORIAL_PRIOR_WEIGHT = 2;

const PHOTO_DEFECT_CATEGORIES = new Set([
  'photo_defect',
  'broken_photo',
  'broken_image',
  'blurry_photo',
  'blurry_image',
  'model_photo',
  'mannequin_photo',
  'watermark',
  'unreadable_image',
  'low_res_photo',
  'corrupt_image',
  'cropped_photo',
  'media_defect',
]);

const CATEGORY_MISMATCH_CATEGORIES = new Set([
  'category_mismatch',
  'slot_mismatch',
  'wrong_slot',
  'wrong_category',
  'mislabeled_category',
  'mislabeled_garment',
  'swimwear_mismatch',
  'multi_item_bundle',
  'non_garment',
]);

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const round4 = (value) => Math.round(value * 10_000) / 10_000;

function nonNegInt(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
}

function nonNegNum(value) {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : 0;
}

function cleanId(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim().slice(0, 128);
  return trimmed.length > 0 ? trimmed : null;
}

function normToken(value) {
  if (typeof value !== 'string') return '';
  return value.normalize('NFKC').trim().toLowerCase().replace(/[\s-]+/g, '_');
}

/**
 * Normalizes an editorial score to [0, 1]. Supports both 0..10 scale (used by
 * 6-axis Visual Critic) and 0..1 scale transparently.
 */
function normalizeEditorialUnit(score) {
  const n = Number(score);
  if (!Number.isFinite(n) || n < 0) return null;
  if (n > 1.0) return clamp(n / 10, 0, 1);
  return clamp(n, 0, 1);
}

/**
 * Normalizes raw snake_case (Postgres `catalog_visual_quality`) or camelCase
 * stats into a canonical stats record.
 */
export function normalizeQualityStats(raw) {
  if (!raw || typeof raw !== 'object') {
    return {
      productId: null,
      approvals: 0,
      rejections: 0,
      photoDefects: 0,
      categoryMismatches: 0,
      editorialScoreSum: 0,
      editorialScoreCount: 0,
      avgEditorialScore: null,
      explicitQuarantined: false,
      explicitQuarantineReason: null,
      updatedAt: null,
    };
  }
  const productId = cleanId(raw.productId ?? raw.product_id ?? raw.id);
  const approvals = nonNegInt(raw.approvals);
  const rejections = nonNegInt(raw.rejections);
  const photoDefects = nonNegInt(raw.photoDefects ?? raw.photo_defects);
  const categoryMismatches = nonNegInt(raw.categoryMismatches ?? raw.category_mismatches);

  let editorialScoreSum = nonNegNum(raw.editorialScoreSum ?? raw.editorial_score_sum);
  let editorialScoreCount = nonNegInt(raw.editorialScoreCount ?? raw.editorial_score_count);

  let avgEditorialScore = null;
  if (editorialScoreCount > 0) {
    avgEditorialScore = round4(editorialScoreSum / editorialScoreCount);
  } else if (raw.avgEditorialScore !== undefined && raw.avgEditorialScore !== null) {
    const direct = Number(raw.avgEditorialScore);
    if (Number.isFinite(direct) && direct >= 0) {
      avgEditorialScore = round4(direct);
      editorialScoreCount = Math.max(1, approvals + rejections);
      editorialScoreSum = round4(direct * editorialScoreCount);
    }
  } else if (raw.avg_editorial_score !== undefined && raw.avg_editorial_score !== null) {
    const direct = Number(raw.avg_editorial_score);
    if (Number.isFinite(direct) && direct >= 0) {
      avgEditorialScore = round4(direct);
      editorialScoreCount = Math.max(1, approvals + rejections);
      editorialScoreSum = round4(direct * editorialScoreCount);
    }
  }

  return {
    productId,
    approvals,
    rejections,
    photoDefects,
    categoryMismatches,
    editorialScoreSum,
    editorialScoreCount,
    avgEditorialScore,
    explicitQuarantined: raw.quarantined === true,
    explicitQuarantineReason: typeof (raw.quarantineReason ?? raw.quarantine_reason) === 'string'
      ? (raw.quarantineReason ?? raw.quarantine_reason)
      : null,
    updatedAt: raw.updatedAt ?? raw.updated_at ?? null,
  };
}

/**
 * Wilson score interval lower bound for binomial approval rate.
 * Returns null when `total <= 0`.
 */
export function computeWilsonLowerBound(
  approvals,
  total,
  z = QUARANTINE_THRESHOLDS.wilsonZ,
) {
  if (!Number.isFinite(total) || total <= 0) return null;
  const k = clamp(Number(approvals) || 0, 0, total);
  const p = k / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const center = p + z2 / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total);
  return round4(clamp((center - margin) / denom, 0, 1));
}

/**
 * Evaluates quarantine eligibility with explicit reason and Wilson diagnostics.
 */
export function getQuarantineStatus(stats) {
  const s = normalizeQualityStats(stats);
  const reviewCount = s.approvals + s.rejections;
  const wilsonScore = reviewCount > 0
    ? computeWilsonLowerBound(s.approvals, reviewCount, QUARANTINE_THRESHOLDS.wilsonZ)
    : null;

  if (s.photoDefects >= QUARANTINE_THRESHOLDS.maxPhotoDefects) {
    return {
      quarantined: true,
      reason: 'chronic_photo_defects',
      wilsonScore,
      reviewCount,
    };
  }
  if (s.categoryMismatches >= QUARANTINE_THRESHOLDS.maxCategoryMismatches) {
    return {
      quarantined: true,
      reason: 'category_mismatch',
      wilsonScore,
      reviewCount,
    };
  }
  if (
    reviewCount >= QUARANTINE_THRESHOLDS.minReviewsForWilson
    && wilsonScore !== null
    && wilsonScore < QUARANTINE_THRESHOLDS.minWilsonScore
  ) {
    return {
      quarantined: true,
      reason: 'low_wilson_approval',
      wilsonScore,
      reviewCount,
    };
  }
  // Honor explicit manual/persisted quarantine flag when no fresh counts clear it
  if (s.explicitQuarantined && reviewCount === 0 && s.photoDefects === 0 && s.categoryMismatches === 0) {
    return {
      quarantined: true,
      reason: s.explicitQuarantineReason || 'manual_quarantine',
      wilsonScore,
      reviewCount,
    };
  }
  return {
    quarantined: false,
    reason: null,
    wilsonScore,
    reviewCount,
  };
}

/**
 * Returns `true` if a SKU must be quarantined from recommendation pools.
 */
export function isSkuQuarantined(stats) {
  return getQuarantineStatus(stats).quarantined;
}

/**
 * Computes a Bayesian/Wilson-shrunk catalog quality score in `[0, 1]`.
 * New SKUs with zero reviews start cleanly at `0.65`.
 */
export function computeCatalogQualityScore(stats) {
  const s = normalizeQualityStats(stats);
  const n = s.approvals + s.rejections;

  // Cold start: neutral 0.65 minus any direct defect penalties
  const bayesianApproval = (s.approvals + BAYESIAN_PRIOR_WEIGHT * BAYESIAN_PRIOR_MEAN)
    / (n + BAYESIAN_PRIOR_WEIGHT);
  const wilson = n > 0
    ? computeWilsonLowerBound(s.approvals, n, QUARANTINE_THRESHOLDS.wilsonZ)
    : BAYESIAN_PRIOR_MEAN;

  // Smoothly shift confidence weight from Bayesian prior to Wilson lower bound
  // as sample size increases.
  const wilsonWeight = n > 0 ? n / (n + 6) : 0;
  let score = (1 - wilsonWeight) * bayesianApproval + wilsonWeight * wilson;

  // Blend normalized editorial quality score (shrunk toward 0.65 at low sample size)
  if (s.avgEditorialScore !== null && s.editorialScoreCount > 0) {
    const unitEditorial = normalizeEditorialUnit(s.avgEditorialScore);
    if (unitEditorial !== null) {
      const shrunkEditorial = (
        unitEditorial * s.editorialScoreCount
        + BAYESIAN_PRIOR_MEAN * EDITORIAL_PRIOR_WEIGHT
      ) / (s.editorialScoreCount + EDITORIAL_PRIOR_WEIGHT);
      score = 0.75 * score + 0.25 * shrunkEditorial;
    }
  }

  // Explicit defect penalties ensure defective photos / wrong categories drop immediately
  const defectPenalty = s.photoDefects * 0.12 + s.categoryMismatches * 0.18;
  score -= defectPenalty;

  return round4(clamp(score, 0, 1));
}

function toPersistedRecord(stats, nowIso) {
  const qualityScore = computeCatalogQualityScore(stats);
  const q = getQuarantineStatus(stats);
  const avgEditorialScore = stats.editorialScoreCount > 0
    ? round4(stats.editorialScoreSum / stats.editorialScoreCount)
    : null;
  return {
    productId: stats.productId,
    product_id: stats.productId,
    approvals: stats.approvals,
    rejections: stats.rejections,
    photoDefects: stats.photoDefects,
    photo_defects: stats.photoDefects,
    categoryMismatches: stats.categoryMismatches,
    category_mismatches: stats.categoryMismatches,
    editorialScoreSum: round4(stats.editorialScoreSum),
    editorial_score_sum: round4(stats.editorialScoreSum),
    editorialScoreCount: stats.editorialScoreCount,
    editorial_score_count: stats.editorialScoreCount,
    avgEditorialScore,
    qualityScore,
    quality_score: qualityScore,
    quarantined: q.quarantined,
    quarantineReason: q.reason,
    quarantine_reason: q.reason,
    wilsonScore: q.wilsonScore,
    reviewCount: q.reviewCount,
    updatedAt: nowIso,
    updated_at: nowIso,
  };
}

function adaptStore(store) {
  if (!store) {
    const map = new Map();
    return adaptStore(map);
  }
  if (store instanceof Map) {
    return {
      async getMany(ids) {
        const out = new Map();
        for (const id of ids) {
          if (store.has(id)) out.set(id, store.get(id));
        }
        return out;
      },
      async upsertMany(records) {
        for (const rec of records) store.set(rec.productId, rec);
      },
      async listAll() {
        return [...store.values()];
      },
    };
  }
  // Custom store adapter `{ get, set, getMany?, upsertMany?, listAll? }`
  if (typeof store.getMany === 'function' && typeof store.upsertMany === 'function') {
    return store;
  }
  if (typeof store.get === 'function' && typeof store.set === 'function') {
    return {
      async getMany(ids) {
        const out = new Map();
        for (const id of ids) {
          const val = await store.get(id);
          if (val) out.set(id, val);
        }
        return out;
      },
      async upsertMany(records) {
        for (const rec of records) {
          await store.set(rec.productId, rec);
        }
      },
      async listAll() {
        if (typeof store.values === 'function') return [...(await store.values())];
        if (typeof store.listAll === 'function') return await store.listAll();
        return [];
      },
    };
  }
  // Supabase client (`supabase.from('catalog_visual_quality')`)
  if (typeof store.from === 'function') {
    return {
      async getMany(ids) {
        if (!ids.length) return new Map();
        const { data, error } = await store
          .from('catalog_visual_quality')
          .select('*')
          .in('product_id', ids);
        if (error) throw new Error(`catalog_quality_read_failed:${error.message || error}`);
        const out = new Map();
        for (const row of data || []) {
          if (row?.product_id) out.set(row.product_id, row);
        }
        return out;
      },
      async upsertMany(records) {
        if (!records.length) return;
        const rows = records.map((r) => ({
          product_id: r.productId,
          approvals: r.approvals,
          rejections: r.rejections,
          photo_defects: r.photoDefects,
          category_mismatches: r.categoryMismatches,
          editorial_score_sum: r.editorialScoreSum,
          editorial_score_count: r.editorialScoreCount,
          quality_score: r.qualityScore,
          quarantined: r.quarantined,
          quarantine_reason: r.quarantineReason,
          updated_at: r.updatedAt,
        }));
        const { error } = await store
          .from('catalog_visual_quality')
          .upsert(rows, { onConflict: 'product_id' });
        if (error) throw new Error(`catalog_quality_write_failed:${error.message || error}`);
      },
      async listAll() {
        const { data, error } = await store
          .from('catalog_visual_quality')
          .select('*');
        if (error) throw new Error(`catalog_quality_list_failed:${error.message || error}`);
        return Array.isArray(data) ? data : [];
      },
    };
  }
  throw new TypeError('catalog_quality_flywheel: unsupported store adapter');
}

function resolveOutcomeUpdates(outcome) {
  if (!outcome || typeof outcome !== 'object') return new Map();
  const deltas = new Map();
  const ensure = (id) => {
    const cleaned = cleanId(id);
    if (!cleaned) return null;
    if (!deltas.has(cleaned)) {
      deltas.set(cleaned, {
        approvals: 0,
        rejections: 0,
        photoDefects: 0,
        categoryMismatches: 0,
        editorialScoreSum: 0,
        editorialScoreCount: 0,
      });
    }
    return deltas.get(cleaned);
  };

  // Extract all candidate SKU ids and slot-to-SKU mapping
  const allIds = [];
  const slotToId = new Map();
  if (Array.isArray(outcome.items)) {
    for (const entry of outcome.items) {
      if (!entry) continue;
      const id = cleanId(typeof entry === 'string' ? entry : entry.id || entry.productId || entry.product_id);
      if (!id) continue;
      allIds.push(id);
      const slot = normToken(entry.slot || entry.macroCategory || entry.fashion_slot);
      if (slot) slotToId.set(slot, id);
    }
  }
  if (Array.isArray(outcome.itemIds)) {
    for (const rawId of outcome.itemIds) {
      const id = cleanId(rawId);
      if (id && !allIds.includes(id)) allIds.push(id);
    }
  }
  if (outcome.productId || outcome.product_id) {
    const id = cleanId(outcome.productId || outcome.product_id);
    if (id && !allIds.includes(id)) allIds.push(id);
  }

  const defectCat = normToken(outcome.defectCategory || outcome.defect_category || outcome.reason);
  const isPhotoDefect = PHOTO_DEFECT_CATEGORIES.has(defectCat) || outcome.photoDefect === true;
  const isCategoryMismatch = CATEGORY_MISMATCH_CATEGORIES.has(defectCat) || outcome.categoryMismatch === true;

  const explicitPhotoDefectIds = Array.isArray(outcome.photoDefectIds)
    ? outcome.photoDefectIds.map(cleanId).filter(Boolean)
    : [];
  const explicitCategoryMismatchIds = Array.isArray(outcome.categoryMismatchIds)
    ? outcome.categoryMismatchIds.map(cleanId).filter(Boolean)
    : [];

  for (const id of explicitPhotoDefectIds) {
    const d = ensure(id);
    if (d) d.photoDefects += 1;
  }
  for (const id of explicitCategoryMismatchIds) {
    const d = ensure(id);
    if (d) d.categoryMismatches += 1;
  }

  const rawEditorial = outcome.editorialScore ?? outcome.editorial_score;
  const editorialNumeric = typeof rawEditorial === 'number' && Number.isFinite(rawEditorial) && rawEditorial >= 0
    ? rawEditorial
    : null;

  if (outcome.approved === true) {
    for (const id of allIds) {
      const d = ensure(id);
      if (!d) continue;
      d.approvals += 1;
      if (editorialNumeric !== null) {
        d.editorialScoreSum += editorialNumeric;
        d.editorialScoreCount += 1;
      }
    }
  } else if (outcome.approved === false) {
    const weakestSlot = normToken(outcome.weakestSlot || outcome.weakest_slot);
    const blamedId = cleanId(outcome.weakestProductId || outcome.weakest_product_id)
      || (weakestSlot ? slotToId.get(weakestSlot) : null);

    // Surgical attribution: if the Visual Critic pinpointed a single weakest slot
    // or specific defect SKU(s), attribute the rejection/defect ONLY to them so
    // innocent co-occurring pieces are not unfairly punished!
    if (blamedId) {
      const d = ensure(blamedId);
      if (d) {
        d.rejections += 1;
        if (isPhotoDefect && !explicitPhotoDefectIds.includes(blamedId)) d.photoDefects += 1;
        if (isCategoryMismatch && !explicitCategoryMismatchIds.includes(blamedId)) d.categoryMismatches += 1;
        if (editorialNumeric !== null) {
          d.editorialScoreSum += editorialNumeric;
          d.editorialScoreCount += 1;
        }
      }
    } else if (explicitPhotoDefectIds.length > 0 || explicitCategoryMismatchIds.length > 0) {
      for (const id of new Set([...explicitPhotoDefectIds, ...explicitCategoryMismatchIds])) {
        const d = ensure(id);
        if (d) d.rejections += 1;
      }
    } else {
      // Whole-look rejection without slot attribution
      for (const id of allIds) {
        const d = ensure(id);
        if (!d) continue;
        d.rejections += 1;
        if (isPhotoDefect) d.photoDefects += 1;
        if (isCategoryMismatch) d.categoryMismatches += 1;
        if (editorialNumeric !== null) {
          d.editorialScoreSum += editorialNumeric;
          d.editorialScoreCount += 1;
        }
      }
    }
  }

  return deltas;
}

/**
 * Records one (or an array of) Visual Critic outcome(s) into `store`,
 * updating Bayesian/Wilson quality scores and quarantine flags atomically.
 * Returns the updated array of SKU quality records.
 */
export async function recordVisualCriticOutcome(store, outcome, nowIso = new Date().toISOString()) {
  const adapter = adaptStore(store);
  const outcomes = Array.isArray(outcome) ? outcome : [outcome];

  const mergedDeltas = new Map();
  for (const single of outcomes) {
    const deltas = resolveOutcomeUpdates(single);
    for (const [id, d] of deltas) {
      const acc = mergedDeltas.get(id) || {
        approvals: 0,
        rejections: 0,
        photoDefects: 0,
        categoryMismatches: 0,
        editorialScoreSum: 0,
        editorialScoreCount: 0,
      };
      acc.approvals += d.approvals;
      acc.rejections += d.rejections;
      acc.photoDefects += d.photoDefects;
      acc.categoryMismatches += d.categoryMismatches;
      acc.editorialScoreSum += d.editorialScoreSum;
      acc.editorialScoreCount += d.editorialScoreCount;
      mergedDeltas.set(id, acc);
    }
  }

  const ids = [...mergedDeltas.keys()];
  if (ids.length === 0) return [];

  const existingMap = await adapter.getMany(ids);
  const updatedRecords = [];

  for (const id of ids) {
    const prev = normalizeQualityStats(existingMap.get(id) || { productId: id });
    const d = mergedDeltas.get(id);
    const nextStats = {
      productId: id,
      approvals: prev.approvals + d.approvals,
      rejections: prev.rejections + d.rejections,
      photoDefects: prev.photoDefects + d.photoDefects,
      categoryMismatches: prev.categoryMismatches + d.categoryMismatches,
      editorialScoreSum: prev.editorialScoreSum + d.editorialScoreSum,
      editorialScoreCount: prev.editorialScoreCount + d.editorialScoreCount,
      avgEditorialScore: null,
      explicitQuarantined: prev.explicitQuarantined,
      explicitQuarantineReason: prev.explicitQuarantineReason,
    };
    updatedRecords.push(toPersistedRecord(nextStats, nowIso));
  }

  await adapter.upsertMany(updatedRecords);
  return updatedRecords;
}

/**
 * Builds a fast O(1) lookup `Map<productId, QualityEntry>` from DB rows
 * (`public.catalog_visual_quality`) or JS stats objects.
 */
export function buildQualityLookupMap(rows) {
  const map = new Map();
  if (!rows) return map;
  const iterable = rows instanceof Map ? rows.values() : Array.isArray(rows) ? rows : [];
  for (const row of iterable) {
    const s = normalizeQualityStats(row);
    if (!s.productId) continue;
    const record = toPersistedRecord(s, s.updatedAt || null);
    map.set(s.productId, record);
  }
  return map;
}

/**
 * Creates a store-backed Catalog Quality Flywheel controller.
 * Defaults to an in-memory `Map` when `store` is omitted.
 */
export function createQualityFlywheel(store = new Map()) {
  const adapter = adaptStore(store);
  return {
    version: QUALITY_FLYWHEEL_VERSION,

    async recordOutcome(outcome, nowIso) {
      return recordVisualCriticOutcome(adapter, outcome, nowIso);
    },

    async getQuality(productId) {
      const id = cleanId(productId);
      if (!id) return null;
      const existing = await adapter.getMany([id]);
      const row = existing.get(id);
      if (!row) {
        return toPersistedRecord(normalizeQualityStats({ productId: id }), null);
      }
      return toPersistedRecord(normalizeQualityStats(row), row.updatedAt ?? row.updated_at ?? null);
    },

    async isQuarantined(productId) {
      const rec = await this.getQuality(productId);
      return Boolean(rec?.quarantined);
    },

    async getLookupMap(productIds = null) {
      if (Array.isArray(productIds) && productIds.length > 0) {
        const cleanIds = [...new Set(productIds.map(cleanId).filter(Boolean))];
        const found = await adapter.getMany(cleanIds);
        return buildQualityLookupMap(found);
      }
      const all = await adapter.listAll();
      return buildQualityLookupMap(all);
    },

    async listQuarantined() {
      const map = await this.getLookupMap();
      return [...map.values()].filter((entry) => entry.quarantined);
    },

    async stats() {
      const map = await this.getLookupMap();
      const entries = [...map.values()];
      const quarantined = entries.filter((e) => e.quarantined).length;
      const avgQuality = entries.length > 0
        ? round4(entries.reduce((sum, e) => sum + e.qualityScore, 0) / entries.length)
        : DEFAULT_NEUTRAL_QUALITY_SCORE;
      return {
        version: QUALITY_FLYWHEEL_VERSION,
        trackedSkus: entries.length,
        quarantinedSkus: quarantined,
        healthySkus: entries.length - quarantined,
        avgQualityScore: avgQuality,
      };
    },
  };
}
