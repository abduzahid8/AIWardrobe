/**
 * wardrobeHealth.ts — Wardrobe Health + lifecycle + cost-per-wear +
 * underused detection + redundancy map + gap graph (Phase 3G–3L).
 *
 * No single vanity score: `computeWardrobeHealth` returns per-dimension
 * grades. Lifecycle/CPW never fabricate prices or wears. Redundancy never
 * recommends selling — it reports. All functions are deterministic and
 * bounded (slice caps, no O(N^2) on render paths beyond 500-item cap).
 */

import type {
  CostPerWear,
  ItemLifecycle,
  RedundantRole,
  UnderusedItem,
  WardrobeGap,
  WardrobeHealth,
  WardrobeHealthDimension,
} from '../../types/styleOS';
import type { OutfitMemoryEntry } from '../../types/styleOS';
import { daysSince, roleKeyOfView, toWardrobeItemView, type WardrobeItemView } from './wardrobeViews';
import type { ClothingItem } from '../../types/domain';

export type { WardrobeItemView };
export { toWardrobeItemView };

const ITEM_CAP = 500;
const MIN_OWNERSHIP_DAYS = 30;
const UNDERUSED_MAX_WEARS = 2;

type Ctx = { tempC?: number; occasion?: string; season?: string };

// ---------- Wardrobe Health (3G) ----------

function grade(score: number): WardrobeHealth['dimensions'][number]['grade'] {
  if (score >= 0.8) return 'strong';
  if (score >= 0.6) return 'healthy';
  if (score >= 0.4) return 'needs_attention';
  if (score >= 0) return 'gap';
  return 'unknown';
}

function dim(
  key: string,
  score: number,
  evidence: string[],
  confidence: WardrobeHealthDimension['confidence'] = 'medium',
): WardrobeHealthDimension {
  return { key, grade: score < 0 ? 'unknown' : grade(score), score, evidence, confidence };
}

export function computeWardrobeHealth(
  items: readonly ClothingItem[],
  memory: readonly OutfitMemoryEntry[] = [],
  fitConfidenceByItem: Record<string, 'low' | 'medium' | 'high'> = {},
  now = Date.now(),
): WardrobeHealth {
  const views = items.slice(0, ITEM_CAP).map(toWardrobeItemView);
  const userId = items[0]?.userId || 'unknown';
  const empty: WardrobeHealth = {
    userId,
    computedAt: new Date(now).toISOString(),
    itemCount: views.length,
    dimensions: [],
    summary: views.length === 0 ? ['Wardrobe is empty; add items to assess health'] : [],
  };
  if (!views.length) {
    empty.dimensions = ['coverage', 'versatility', 'occasion_coverage'].map((k) =>
      dim(k, -1, ['No items to assess']),
    );
    return empty;
  }
  const cats = new Set(views.map((v) => v.category));
  const core = ['top', 'bottom', 'shoes'].filter((c) => cats.has(c)).length;
  const coverage = dim('coverage', core / 3, [
    `${cats.size} categories present; core ${core}/3 (top/bottom/shoes)`,
  ]);
  const roles = new Map<string, number>();
  for (const v of views) roles.set(roleKeyOfView(v), (roles.get(roleKeyOfView(v)) || 0) + 1);
  const dupRoles = [...roles.values()].filter((n) => n >= 3).length;
  const redundancy = dim(
    'redundancy',
    roles.size === 0 ? -1 : Math.max(0, 1 - dupRoles / Math.max(1, roles.size)),
    [dupRoles ? `${dupRoles} role(s) with 3+ items` : 'No heavily duplicated roles'],
  );
  const worn = views.filter((v) => v.wearCount > 0).length;
  const versatility = dim('versatility', worn / views.length, [
    `${worn}/${views.length} items worn at least once`,
  ]);
  const fitKnown = views.filter((v) => fitConfidenceByItem[v.id] && fitConfidenceByItem[v.id] !== 'low').length;
  const fit = dim(
    'fit_confidence',
    views.length ? fitKnown / views.length : -1,
    [`${fitKnown}/${views.length} items with usable fit confidence`],
    fitKnown >= 3 ? 'medium' : 'low',
  );
  const totalWears = views.reduce((n, v) => n + v.wearCount, 0);
  const avg = totalWears / views.length;
  const unusedShare = views.filter((v) => v.wearCount === 0).length / views.length;
  const usage = dim('usage_balance', Math.max(0, 1 - unusedShare), [
    `${Math.round(unusedShare * 100)}% never worn; avg ${avg.toFixed(1)} wears/item`,
  ]);
  const damaged = views.filter((v) => v.condition === 'damaged' || v.condition === 'worn').length;
  const condition = dim('condition', Math.max(0, 1 - damaged / views.length), [
    damaged ? `${damaged} item(s) worn/damaged` : 'No damaged items reported',
  ]);
  const occasions = ['casual', 'work', 'formal', 'sport', 'date', 'travel'];
  const coveredOcc = occasions.filter((o) => views.some((v) => v.occasions.includes(o))).length;
  const occasion = dim('occasion_coverage', coveredOcc / occasions.length, [
    `${coveredOcc}/${occasions.length} occasions covered`,
  ]);
  const seasons = ['spring', 'summer', 'fall', 'winter'];
  const coveredSeasons = seasons.filter((s) => views.some((v) => v.seasons.includes(s))).length;
  const seasonal = dim('seasonal_balance', coveredSeasons / seasons.length, [
    `${coveredSeasons}/4 seasons covered`,
  ]);
  const shoes = views.filter((v) => v.category === 'shoes').length;
  const footwear = dim('footwear_coverage', Math.min(1, shoes / 3), [`${shoes} shoe(s)`]);
  const outerwear = views.filter((v) => v.category === 'outerwear').length;
  const outer = dim('outerwear_coverage', Math.min(1, outerwear / 2), [`${outerwear} outerwear piece(s)`]);
  const colors = new Set(views.map((v) => v.primaryColor));
  const neutrals = views.filter((v) =>
    ['black', 'white', 'grey', 'gray', 'navy', 'beige', 'brown'].includes(v.primaryColor),
  ).length;
  const coherence = dim('color_coherence', views.length ? neutrals / views.length : -1, [
    `${colors.size} distinct colors; ${Math.round((neutrals / Math.max(1, views.length)) * 100)}% neutrals`,
  ]);
  const climate = dim(
    'climate_coverage',
    (seasonal.score + outer.score) / 2,
    ['Derived from seasonal balance + outerwear coverage'],
    'low',
  );
  const missing = dim('missing_pieces', -1, ['Run gap detection for missing-piece detail'], 'low');
  const shopping = dim('shopping_efficiency', -1, ['Derived from purchase outcomes when available'], 'low');
  const dimensions = [
    coverage, redundancy, versatility, fit, usage, condition, missing,
    occasion, climate, coherence, footwear, outer, seasonal, shopping,
  ];
  const weak = dimensions
    .filter((d) => d.grade === 'gap' || d.grade === 'needs_attention')
    .map((d) => `${d.key}: ${d.grade}`);
  return {
    userId,
    computedAt: new Date(now).toISOString(),
    itemCount: views.length,
    dimensions,
    summary: weak.length ? [`Needs attention: ${weak.join(', ')}`] : ['Wardrobe looks balanced'],
  };
}

// ---------- Lifecycle (3H) ----------

export function defaultLifecycle(itemId: string, userId: string): ItemLifecycle {
  return {
    itemId, userId, acquiredAt: null, acquisitionPrice: null, currency: null,
    source: null, condition: null, repairNeeded: false, lastWornAt: null,
    wearCount: 0, status: 'active', satisfaction: null, fitStatus: 'unknown',
    replacementCandidate: false,
  };
}

/** Merge a partial lifecycle update; wearCount only from confirmed wears. */
export function updateLifecycle(
  prev: ItemLifecycle,
  patch: Partial<ItemLifecycle>,
  confirmedWears?: number,
): ItemLifecycle {
  return {
    ...prev,
    ...patch,
    itemId: prev.itemId,
    userId: prev.userId,
    wearCount: confirmedWears ?? patch.wearCount ?? prev.wearCount,
  };
}

// ---------- Cost per wear (3I) ----------

export function costPerWear(
  itemId: string,
  lifecycle: Pick<ItemLifecycle, 'acquisitionPrice' | 'currency'>,
  confirmedWearCount: number,
  estimatedPrice?: { amount: number; currency: string },
): CostPerWear {
  const price = lifecycle.acquisitionPrice;
  if (price !== null && price !== undefined && price > 0) {
    return {
      itemId,
      wearCount: confirmedWearCount,
      acquisitionPrice: price,
      currency: lifecycle.currency || null,
      costPerWear: confirmedWearCount > 0 ? Math.round((price / confirmedWearCount) * 100) / 100 : null,
      estimated: false,
      explanation:
        confirmedWearCount > 0
          ? `Confirmed: ${price} / ${confirmedWearCount} confirmed wears`
          : 'Confirmed price; no confirmed wears yet',
    };
  }
  if (estimatedPrice) {
    return {
      itemId,
      wearCount: confirmedWearCount,
      acquisitionPrice: estimatedPrice.amount,
      currency: estimatedPrice.currency,
      costPerWear:
        confirmedWearCount > 0 ? Math.round((estimatedPrice.amount / confirmedWearCount) * 100) / 100 : null,
      estimated: true,
      explanation: 'ESTIMATED price — acquisition price was never recorded',
    };
  }
  return {
    itemId,
    wearCount: confirmedWearCount,
    acquisitionPrice: null,
    currency: null,
    costPerWear: null,
    estimated: false,
    explanation: 'No acquisition price recorded; cost per wear unknown',
  };
}

// ---------- Underused detection (3J) ----------

export function detectUnderused(
  items: readonly ClothingItem[],
  lifecycles: Record<string, ItemLifecycle> = {},
  memory: readonly OutfitMemoryEntry[] = [],
  now = Date.now(),
): UnderusedItem[] {
  const wornItemIds = new Set(memory.filter((m) => m.confirmed).flatMap((m) => m.itemIds));
  const out: UnderusedItem[] = [];
  for (const item of items.slice(0, ITEM_CAP)) {
    const lc = lifecycles[item.id];
    if (lc && (lc.status === 'archived' || lc.status === 'storage' || lc.status === 'sell' || lc.status === 'donate'))
      continue;
    const owned = daysSince(lc?.acquiredAt || item.createdAt, now);
    if (owned === null || owned < MIN_OWNERSHIP_DAYS) continue;
    const wears = lc?.wearCount ?? item.wearCount ?? 0;
    if (wears > UNDERUSED_MAX_WEARS) continue;
    const reasons: UnderusedItem['reasons'] = [];
    const evidence: string[] = [`Owned ${owned} days, ${wears} confirmed wears`];
    // duplicate role: 2+ items share the role key.
    const role = roleKeyOfView(toWardrobeItemView(item));
    const roleCount = items.filter(
      (o) => o.id !== item.id && roleKeyOfView(toWardrobeItemView(o)) === role,
    ).length;
    if (roleCount >= 1) {
      reasons.push('duplicate_role');
      evidence.push(`${roleCount + 1} items share role ${role}`);
    }
    if (lc?.fitStatus === 'tight' || lc?.fitStatus === 'loose' || lc?.fitStatus === 'needs_alteration') {
      reasons.push('fit_issue');
      evidence.push(`Fit status: ${lc.fitStatus}`);
    }
    if (lc?.condition === 'damaged' || lc?.condition === 'worn' || lc?.repairNeeded) {
      reasons.push('condition_issue');
      evidence.push('Condition needs attention');
    }
    if (lc?.satisfaction !== null && lc?.satisfaction !== undefined && lc.satisfaction <= 2) {
      reasons.push('style_mismatch');
      evidence.push(`Satisfaction ${lc.satisfaction}/5`);
    }
    if (!wornItemIds.has(item.id) && wears === 0) {
      reasons.push('hard_to_combine');
      evidence.push('Never worn in confirmed memory');
    }
    if (!reasons.length) reasons.push('unknown');
    const suggested: UnderusedItem['suggestedActions'] = [];
    if (reasons.includes('condition_issue') || reasons.includes('fit_issue')) suggested.push('repair');
    if (reasons.includes('duplicate_role') || reasons.includes('style_mismatch')) suggested.push('sell', 'donate');
    if (reasons.includes('hard_to_combine')) suggested.push('style_it', 'find_complement');
    if (!suggested.length) suggested.push('keep');
    out.push({
      itemId: item.id,
      daysOwned: owned,
      wearCount: wears,
      reasons,
      evidence,
      suggestedActions: [...new Set(suggested)],
    });
  }
  return out.sort((a, b) => b.daysOwned - a.daysOwned).slice(0, 100);
}

// ---------- Redundancy map (3K) ----------

export function redundancyMap(items: readonly ClothingItem[]): RedundantRole[] {
  const groups = new Map<string, ClothingItem[]>();
  for (const item of items.slice(0, ITEM_CAP)) {
    const key = roleKeyOfView(toWardrobeItemView(item));
    const list = groups.get(key) || [];
    list.push(item);
    groups.set(key, list);
  }
  const out: RedundantRole[] = [];
  for (const [roleKey, group] of groups) {
    if (group.length < 2) continue;
    const total = group.reduce((n, g) => n + (g.wearCount || 0), 0);
    const wearDistribution = group.map((g) => ({
      itemId: g.id,
      wears: g.wearCount || 0,
      share: total > 0 ? Math.round(((g.wearCount || 0) / total) * 100) / 100 : 0,
    }));
    const favorite = [...group].sort((a, b) => (b.wearCount || 0) - (a.wearCount || 0))[0];
    const underused = group.filter((g) => (g.wearCount || 0) === 0).map((g) => g.id);
    out.push({
      roleKey,
      label: roleKey.replace(/:/g, ' · '),
      itemIds: group.map((g) => g.id),
      wearDistribution,
      favoriteItemId: (favorite.wearCount || 0) > 0 ? favorite.id : null,
      underusedItemIds: underused,
      duplicateRisk: group.length >= 4 ? 'high' : group.length === 3 ? 'medium' : 'low',
      note: 'Report only — never an automatic instruction to sell.',
    });
  }
  return out.sort((a, b) => b.itemIds.length - a.itemIds.length).slice(0, 100);
}

// ---------- Gap graph (3L) ----------

const OCCASION_CATEGORY: Record<string, string[]> = {
  work: ['top', 'bottom', 'shoes'],
  formal: ['dress', 'shoes', 'outerwear'],
  sport: ['top', 'bottom', 'shoes'],
  travel: ['top', 'bottom', 'shoes', 'outerwear'],
};

export function detectGaps(
  userId: string,
  items: readonly ClothingItem[],
  contexts: readonly string[] = ['casual', 'work', 'formal', 'sport', 'travel'],
  now = Date.now(),
): WardrobeGap[] {
  void now;
  const views = items.slice(0, ITEM_CAP).map(toWardrobeItemView);
  const gaps: WardrobeGap[] = [];
  let n = 0;
  for (const ctx of contexts.slice(0, 10)) {
    const needed = OCCASION_CATEGORY[ctx] || ['top', 'bottom', 'shoes'];
    for (const cat of needed) {
      const has = views.some((v) => v.category === cat && (v.occasions.includes(ctx) || v.occasions.length === 0));
      if (!has) {
        n++;
        gaps.push({
          id: `gap_${ctx}_${cat}`,
          type: `missing_${cat}_for_${ctx}`,
          label: `No ${cat} for ${ctx}`,
          blockedContexts: [ctx],
          blockedOutfitCount: 1,
          expectedUnlocks: 2,
          urgency: ctx === 'work' || ctx === 'formal' ? 'high' : 'medium',
          confidence: views.length >= 5 ? 'medium' : 'low',
          candidateCategories: [cat],
          candidateAttributes: { occasion: ctx },
        });
      }
    }
  }
  void n;
  return gaps.slice(0, 40);
}

/** Rank companions for an item from confirmed co-wear memory. */
export function validatedCompanions(
  itemId: string,
  memory: readonly OutfitMemoryEntry[],
  limit = 8,
): Array<{ itemId: string; wears: number; positive: number }> {
  const counts = new Map<string, { wears: number; positive: number }>();
  for (const m of memory) {
    if (!m.confirmed || !m.itemIds.includes(itemId)) continue;
    for (const other of m.itemIds) {
      if (other === itemId) continue;
      const c = counts.get(other) || { wears: 0, positive: 0 };
      c.wears++;
      if (m.rating === 'loved' || m.rating === 'good') c.positive++;
      counts.set(other, c);
    }
  }
  return [...counts.entries()]
    .map(([id, c]) => ({ itemId: id, ...c }))
    .sort((a, b) => b.positive - a.positive || b.wears - a.wears)
    .slice(0, limit);
}

export type { Ctx };
