/**
 * twin.ts — Digital wardrobe twin aggregation + queries (4O, §41).
 *
 * The twin is an aggregated VIEW, not a new source of truth: every field
 * carries provenance, conflicts resolve user > structured > visual, and
 * user corrections always win. Queries are bounded and cacheable by
 * caller (pure functions over supplied rows).
 */
import type { TwinCondition, VisualEvidence, WardrobeTwinItem } from '../../types/visualOS';
import { canonicalVisualValue } from './visualEvidence';
import { cosineSimilarity } from '../style/semantic';

export interface TwinSources {
  itemId: string;
  identity: Record<string, string>;
  visual: readonly VisualEvidence[];
  conditionEvidence: readonly VisualEvidence[];
  fitSummary?: string | null;
  styleSummary?: string | null;
  lifecycleSummary?: string | null;
  usageSummary?: string | null;
  companionIds?: string[];
  substituteIds?: string[];
}

const CONDITION_RANK: Record<string, number> = {
  excellent: 0, good: 1, worn: 2, needs_attention: 3, repair: 4, replace_candidate: 5, unknown: 6,
};

export function aggregateTwinItem(src: TwinSources): WardrobeTwinItem {
  const provenance: Record<string, string> = { identity: 'wardrobe-metadata' };
  const visualByAttr = new Map<string, VisualEvidence[]>();
  for (const v of src.visual) {
    const list = visualByAttr.get(v.attributeType) || [];
    list.push(v);
    visualByAttr.set(v.attributeType, list);
  }
  const visualFeatures: Record<string, string[]> = {};
  for (const [attr, rows] of visualByAttr) {
    const canon = canonicalVisualValue(rows);
    if (canon && canon.value !== 'unknown') {
      visualFeatures[attr] = [canon.value];
      provenance[`visual:${attr}`] = `${canon.source}${canon.modelVersion ? `@${canon.modelVersion}` : ''}${canon.userConfirmed ? '+user-confirmed' : ''}`;
    }
  }
  let condition: TwinCondition = 'unknown';
  const condRows = src.conditionEvidence.filter((r) => !r.userRejected && !r.supersededBy);
  if (condRows.length) {
    const canon = canonicalVisualValue(condRows);
    const v = (canon?.value || 'unknown') as TwinCondition;
    condition = v in CONDITION_RANK ? v : 'unknown';
    provenance.condition = `${canon?.source || 'none'}`;
  }
  return {
    itemId: src.itemId,
    identity: { ...src.identity },
    visualFeatures,
    condition,
    fitSummary: src.fitSummary ?? null,
    styleSummary: src.styleSummary ?? null,
    lifecycleSummary: src.lifecycleSummary ?? null,
    usageSummary: src.usageSummary ?? null,
    companions: [...(src.companionIds || [])].slice(0, 12),
    substitutes: [...(src.substituteIds || [])].slice(0, 12),
    provenance,
  };
}

/** Twin queries (§41) — pure, bounded. */
export function twinById(twin: readonly WardrobeTwinItem[], itemId: string): WardrobeTwinItem | null {
  return twin.find((t) => t.itemId === itemId) || null;
}

export function conditionRisks(twin: readonly WardrobeTwinItem[]): WardrobeTwinItem[] {
  return twin
    .filter((t) => ['needs_attention', 'repair', 'replace_candidate'].includes(t.condition))
    .slice(0, 50);
}

export function itemsByVisualRole(
  twin: readonly WardrobeTwinItem[],
  attr: string,
  value: string,
  limit = 20,
): WardrobeTwinItem[] {
  const v = value.toLowerCase();
  return twin.filter((t) => (t.visualFeatures[attr] || []).some((x) => x.toLowerCase() === v)).slice(0, limit);
}

export function bestVisualCompanions(
  twin: readonly WardrobeTwinItem[],
  itemId: string,
  embeddings: Record<string, readonly number[]> = {},
  limit = 8,
): Array<{ itemId: string; score: number }> {
  const self = embeddings[itemId];
  const out: Array<{ itemId: string; score: number }> = [];
  if (self) {
    for (const [id, emb] of Object.entries(embeddings)) {
      if (id === itemId) continue;
      out.push({ itemId: id, score: cosineSimilarity(self, emb) });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, limit);
  }
  const me = twinById(twin, itemId);
  if (!me) return [];
  return (me.companions || []).slice(0, limit).map((id) => ({ itemId: id, score: 0.5 }));
}

export function closestOwnedAlternatives(
  twin: readonly WardrobeTwinItem[],
  attrs: Record<string, string>,
  limit = 5,
): WardrobeTwinItem[] {
  const keys = Object.keys(attrs);
  return [...twin]
    .map((t) => ({
      t,
      hits: keys.filter((k) => (t.visualFeatures[k] || []).includes(attrs[k]) || t.identity[k] === attrs[k]).length,
    }))
    .filter((r) => r.hits > 0)
    .sort((a, b) => b.hits - a.hits || a.t.itemId.localeCompare(b.t.itemId))
    .slice(0, limit)
    .map((r) => r.t);
}
