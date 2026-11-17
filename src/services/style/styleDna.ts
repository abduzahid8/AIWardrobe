/**
 * styleDna.ts — StyleDNA aggregation + evolution signals (Phase 3B/3C).
 *
 * Aggregates StyleEvidence into per-dimension values with confidence,
 * evidence count, provenance, recency and contradiction. Inferred
 * preferences never become hard constraints here — see `confidence`
 * gating: only high-confidence dimensions constrain ranking, and
 * overrides always win (resolved in contextualProfiles.ts).
 */

import type {
  StyleDimension,
  StyleDNA,
  StyleEvidence,
  StyleEvolutionSignal,
} from '../../types/styleOS';
import { evidenceWeight } from './styleEvidence';

/** Dimensions tracked by the aggregator. */
export const STYLE_DIMENSIONS = [
  'preferred_colors',
  'avoided_colors',
  'color_combinations',
  'silhouettes',
  'fit_intent',
  'pattern_tolerance',
  'materials',
  'formality_range',
  'minimal_vs_expressive',
  'classic_vs_trend',
  'fitted_vs_relaxed',
  'monochrome_vs_contrast',
  'footwear',
  'layering',
  'brand_affinity',
  'price_sensitivity',
  'novelty_tolerance',
  'repetition_tolerance',
  'occasion_bias',
] as const;

const MIN_HIGH = 4;
const MIN_MEDIUM = 1.5;

function dimensionConfidence(
  weight: number,
  contradiction: number,
): StyleDimension['confidence'] {
  if (contradiction > 0.45 || weight < 0.6) return 'low';
  if (weight >= MIN_HIGH && contradiction < 0.2) return 'high';
  if (weight >= MIN_MEDIUM) return 'medium';
  return 'emerging';
}

export function aggregateStyleDNA(
  userId: string,
  rows: readonly StyleEvidence[],
  now = Date.now(),
): StyleDNA {
  const mine = rows.filter((r) => r && r.userId === userId);
  // dimension -> value -> {pos, neg, provenance, latest}
  const acc = new Map<
    string,
    Map<string, { pos: number; neg: number; provenance: Set<string>; latest: string | null }>
  >();
  for (const row of mine) {
    const dim = entityToDimension(row.entityType, row.entityKey);
    if (!dim) continue;
    const value = valueOf(row.entityKey);
    const w = evidenceWeight(row, now);
    if (w <= 0) continue;
    if (!acc.has(dim)) acc.set(dim, new Map());
    const bucket = acc.get(dim)!;
    if (!bucket.has(value))
      bucket.set(value, { pos: 0, neg: 0, provenance: new Set(), latest: null });
    const cell = bucket.get(value)!;
    if (row.polarity === 'negative') cell.neg += w;
    else if (row.polarity === 'positive') cell.pos += w;
    else {
      cell.pos += w * 0.5;
      cell.neg += w * 0.5;
    }
    cell.provenance.add(row.sourceType);
    if (!cell.latest || row.observedAt > cell.latest) cell.latest = row.observedAt;
  }
  const dimensions: Record<string, StyleDimension> = {};
  const emerging: string[] = [];
  for (const dim of STYLE_DIMENSIONS) {
    const bucket = acc.get(dim);
    if (!bucket || bucket.size === 0) {
      emerging.push(dim);
      continue;
    }
    const ranked = [...bucket.entries()].sort(
      (a, b) => b[1].pos - b[1].neg - (a[1].pos - a[1].neg),
    );
    const [value, cell] = ranked[0];
    const total = cell.pos + cell.neg;
    const contradiction =
      total > 0 ? Math.min(1, cell.neg / total + rivalShare(ranked)) : 1;
    const net = Math.max(0, cell.pos - cell.neg);
    const confidence = dimensionConfidence(net, contradiction);
    dimensions[dim] = {
      value,
      confidence,
      evidenceCount: Math.round(total * 10) / 10,
      provenance: [...cell.provenance].sort(),
      recency: cell.latest,
      contradiction: Math.round(contradiction * 100) / 100,
    };
    if (confidence === 'low' || confidence === 'emerging') emerging.push(dim);
  }
  return {
    version: 1,
    userId,
    computedAt: new Date(now).toISOString(),
    evidenceCount: mine.length,
    dimensions,
    emerging,
  };
}

function rivalShare(
  ranked: Array<[string, { pos: number; neg: number }]>,
): number {
  if (ranked.length < 2) return 0;
  const top = Math.max(0, ranked[0][1].pos - ranked[0][1].neg);
  const second = Math.max(0, ranked[1][1].pos - ranked[1][1].neg);
  if (top <= 0) return 0.5;
  return Math.min(0.5, second / (top + second));
}

/** Map (entityType, entityKey "facet:value") to a DNA dimension. */
export function entityToDimension(
  entityType: StyleEvidence['entityType'],
  entityKey: string,
): string | null {
  const [facet] = entityKey.split(':');
  switch (entityType) {
    case 'color':
      return facet === 'avoid' ? 'avoided_colors' : 'preferred_colors';
    case 'silhouette':
      return 'silhouettes';
    case 'material':
      return 'materials';
    case 'pattern':
      return 'pattern_tolerance';
    case 'brand':
      return 'brand_affinity';
    case 'style_tag':
      return styleTagDimension(facet, entityKey);
    case 'visual_family':
    case 'garment':
    case 'outfit':
    case 'product':
      return 'occasion_bias';
    default:
      return null;
  }
}

function styleTagDimension(facet: string, key: string): string {
  if (facet === 'formality') return 'formality_range';
  if (facet === 'fit') return key.includes('relaxed') || key.includes('fitted') ? 'fitted_vs_relaxed' : 'fit_intent';
  if (facet === 'footwear') return 'footwear';
  if (facet === 'combo') return 'color_combinations';
  if (facet === 'contrast') return 'monochrome_vs_contrast';
  if (facet === 'expression') return 'minimal_vs_expressive';
  if (facet === 'trend') return 'classic_vs_trend';
  if (facet === 'layer') return 'layering';
  if (facet === 'price') return 'price_sensitivity';
  if (facet === 'novelty') return 'novelty_tolerance';
  if (facet === 'repeat') return 'repetition_tolerance';
  return 'occasion_bias';
}

function valueOf(entityKey: string): string {
  const idx = entityKey.indexOf(':');
  return idx >= 0 ? entityKey.slice(idx + 1) : entityKey;
}

/**
 * Evolution signals (§3C): compare baseline evidence (old) vs recent window.
 * A claim requires minRecentWeight and a clear margin — otherwise no signal.
 */
export function detectStyleEvolution(
  userId: string,
  rows: readonly StyleEvidence[],
  options: { recentDays?: number; minWeight?: number } = {},
  now = Date.now(),
): StyleEvolutionSignal[] {
  const recentMs = (options.recentDays ?? 45) * 86_400_000;
  const minWeight = options.minWeight ?? 1.2;
  const mine = rows.filter((r) => r && r.userId === userId);
  const dims = new Set(
    mine.map((r) => entityToDimension(r.entityType, r.entityKey)).filter(Boolean) as string[],
  );
  const signals: StyleEvolutionSignal[] = [];
  for (const dim of dims) {
    const inDim = mine.filter((r) => entityToDimension(r.entityType, r.entityKey) === dim);
    const base = aggregateStyleDNA(userId, inDim.filter((r) => Date.parse(r.observedAt) < now - recentMs), now);
    const recent = aggregateStyleDNA(userId, inDim.filter((r) => Date.parse(r.observedAt) >= now - recentMs), now);
    const b = base.dimensions[dim];
    const r = recent.dimensions[dim];
    if (!r || r.value === b?.value) continue;
    if (r.evidenceCount < minWeight) continue;
    if (r.confidence === 'low' || r.confidence === 'emerging') continue;
    const from = new Date(now - recentMs).toISOString();
    signals.push({
      dimension: dim,
      previousValue: b?.value ?? null,
      emergingValue: r.value,
      confidence: r.confidence === 'high' ? 'high' : 'medium',
      evidenceWindow: { from, to: new Date(now).toISOString(), count: Math.round(r.evidenceCount) },
      changeStrength: Math.round(Math.min(1, r.evidenceCount / (minWeight * 2)) * 100) / 100,
    });
  }
  return signals.sort((a, b) => b.changeStrength - a.changeStrength).slice(0, 12);
}
