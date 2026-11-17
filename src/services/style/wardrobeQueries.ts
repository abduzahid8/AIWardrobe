/**
 * wardrobeQueries.ts — Wardrobe query engine (Phase 3 §34).
 *
 * Reusable deterministic queries. All logic lives here, outside UI.
 * Every query is bounded (caps) and deterministic (stable sort with id
 * tiebreak) so repeated calls return identical results.
 */

import type {
  NoBuyDecision,
  OutfitMemoryEntry,
  PlannedPurchase,
  RedundantRole,
  UnderusedItem,
  WardrobeGap,
} from '../../types/styleOS';
import type { ClothingItem } from '../../types/domain';
import { detectGaps, detectUnderused, redundancyMap, validatedCompanions } from './wardrobeHealth';
import type { ItemLifecycle } from '../../types/styleOS';

export interface QueryCtx {
  userId: string;
  items: readonly ClothingItem[];
  memory?: readonly OutfitMemoryEntry[];
  lifecycles?: Record<string, ItemLifecycle>;
  now?: number;
}

const byIdTiebreak = <T>(a: { id: string } & T, b: { id: string } & T, cmp: number) =>
  cmp !== 0 ? cmp : a.id.localeCompare(b.id);

/** Best items for a context: occasion match, then wears, then recency. */
export function getBestItemsForContext(
  ctx: QueryCtx,
  contextId: string,
  limit = 20,
): ClothingItem[] {
  const mem = ctx.memory || [];
  const score = (item: ClothingItem): number => {
    let s = 0;
    if (item.occasions?.includes(contextId as never)) s += 10;
    if (item.occasions?.length === 0) s += 2;
    s += Math.min(5, item.wearCount || 0);
    if (item.isFavorite) s += 3;
    const bad = mem.some(
      (m) => m.contextId === contextId && m.itemIds.includes(item.id) && m.rating === 'didnt_work',
    );
    if (bad) s -= 8;
    return s;
  };
  return [...ctx.items]
    .filter((i) => i.userId === ctx.userId)
    .sort((a, b) => byIdTiebreak(a, b, score(b) - score(a)))
    .slice(0, Math.min(50, Math.max(1, limit)));
}

export function getUnderusedItems(ctx: QueryCtx, limit = 30): UnderusedItem[] {
  return detectUnderused(ctx.items, ctx.lifecycles, ctx.memory, ctx.now).slice(0, limit);
}

export function getRedundantRoles(ctx: QueryCtx): RedundantRole[] {
  return redundancyMap(ctx.items.filter((i) => i.userId === ctx.userId));
}

export function getTopWardrobeGaps(
  ctx: QueryCtx,
  contexts?: string[],
  limit = 10,
): WardrobeGap[] {
  const order = { high: 0, medium: 1, low: 2 } as const;
  return detectGaps(ctx.userId, ctx.items.filter((i) => i.userId === ctx.userId), contexts, ctx.now)
    .sort((a, b) => order[a.urgency] - order[b.urgency] || b.expectedUnlocks - a.expectedUnlocks)
    .slice(0, limit);
}

export function getValidatedCompanions(
  ctx: QueryCtx,
  itemId: string,
  limit = 8,
): Array<{ item: ClothingItem | null; wears: number; positive: number }> {
  const byId = new Map(ctx.items.map((i) => [i.id, i]));
  return validatedCompanions(itemId, ctx.memory || [], limit).map((c) => ({
    item: byId.get(c.itemId) || null,
    wears: c.wears,
    positive: c.positive,
  }));
}

export function getSubstitutes(ctx: QueryCtx, itemId: string, limit = 5): ClothingItem[] {
  const target = ctx.items.find((i) => i.id === itemId);
  if (!target) return [];
  return ctx.items
    .filter(
      (i) =>
        i.id !== itemId &&
        i.category === target.category &&
        i.subCategory === target.subCategory,
    )
    .sort((a, b) => byIdTiebreak(a, b, (b.wearCount || 0) - (a.wearCount || 0)))
    .slice(0, limit);
}

/** High-confidence outfits: confirmed, positively rated, worn 2+ times. */
export function getHighConfidenceOutfits(ctx: QueryCtx): OutfitMemoryEntry[] {
  const byCombo = new Map<string, OutfitMemoryEntry[]>();
  for (const m of ctx.memory || []) {
    if (!m.confirmed || m.userId !== ctx.userId) continue;
    const list = byCombo.get(m.combinationKey) || [];
    list.push(m);
    byCombo.set(m.combinationKey, list);
  }
  const out: OutfitMemoryEntry[] = [];
  for (const list of byCombo.values()) {
    if (list.length >= 2 && list.some((m) => m.rating === 'loved' || m.rating === 'good'))
      out.push(list.sort((a, b) => b.wornDate.localeCompare(a.wornDate))[0]);
  }
  return out.slice(0, 30);
}

export function getNextPurchaseCandidates(
  ctx: QueryCtx,
  gaps: readonly WardrobeGap[],
  limit = 5,
): PlannedPurchase[] {
  return gaps.slice(0, limit).map((g, i) => ({
    productId: `gap-candidate:${g.id}`,
    label: g.label,
    priority: i + 1,
    expectedUnlocks: g.expectedUnlocks,
    confidence: g.confidence,
    evidence: [`Blocks ${g.blockedContexts.join(', ')}`, `~${g.expectedUnlocks} expected unlocks`],
    tradeoffs: ['Verify fit evidence before buying', 'Check duplicates in the same role'],
  }));
}

/** Evidence rows supporting restraint for a candidate category. */
export function getNoBuyEvidence(
  ctx: QueryCtx,
  category: string,
  lifecycles?: Record<string, ItemLifecycle>,
): NoBuyDecision {
  const owned = ctx.items.filter((i) => i.category === category);
  const neverWorn = owned.filter((i) => (lifecycles?.[i.id]?.wearCount ?? i.wearCount ?? 0) === 0);
  if (owned.length >= 4) {
    return {
      recommendation: 'SKIP_DUPLICATE',
      confidence: 'high',
      evidence: [`Own ${owned.length} in ${category}`, `${neverWorn.length} never worn`],
      explanation: `You already own ${owned.length} ${category} pieces — wear what you own first.`,
    };
  }
  if (neverWorn.length >= 2) {
    return {
      recommendation: 'SKIP_LOW_VALUE',
      confidence: 'medium',
      evidence: [`${neverWorn.length} unworn ${category} pieces owned`],
      explanation: 'Unworn pieces in this category suggest low value from another purchase.',
    };
  }
  return {
    recommendation: 'BUY_IF_NEEDED',
    confidence: 'low',
    evidence: [`Own ${owned.length} in ${category}`],
    explanation: 'No strong restraint evidence; buy only if a real gap is confirmed.',
  };
}
