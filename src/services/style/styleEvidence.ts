/**
 * styleEvidence.ts — StyleEvidence model (Phase 3A + decay 3C).
 *
 * Canonical evidence with polarity, context, confidence, expiry and
 * provenance. Decay: exponential half-life per source family; explicit
 * `expiresAt` wins. Permanent signals require declared/manual origin —
 * inferred one-off clicks never become permanent.
 */

import type { StyleEvidence, StyleSourceType } from '../../types/styleOS';

export const STYLE_EVIDENCE_VERSION = 1;
const DAY_MS = 86_400_000;

/** Half-life in days per source family. Shorter = faster decay. */
const HALF_LIFE_DAYS: Record<StyleSourceType, number> = {
  declared_preference: 730,
  outfit_worn: 180,
  outfit_saved: 120,
  outfit_liked: 120,
  outfit_disliked: 180,
  product_saved: 90,
  product_skipped: 60,
  product_bought: 365,
  product_returned: 365,
  manual_feedback: 730,
  wardrobe_item: 365,
  reference_outfit: 365,
  moodboard: 180,
  social_import: 90,
};

/** Sources allowed to carry permanent (non-expiring) weight. */
const PERMANENT_SOURCES: ReadonlySet<StyleSourceType> = new Set([
  'declared_preference',
  'manual_feedback',
  'reference_outfit',
]);

export interface StyleEvidenceInput {
  userId: string;
  sourceType: StyleSourceType;
  entityType: StyleEvidence['entityType'];
  entityKey: string;
  polarity?: StyleEvidence['polarity'];
  strength?: number;
  context?: Record<string, string>;
  confidence?: number;
  observedAt?: string;
  expiresAt?: string | null;
  provenance: string;
}

let evidenceSeq = 0;

export function createStyleEvidence(input: StyleEvidenceInput): StyleEvidence {
  const observedAt = input.observedAt || new Date().toISOString();
  const clamp = (n: number) => (Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0);
  return {
    id: `sev_${Date.now().toString(36)}_${(evidenceSeq++).toString(36)}`,
    userId: input.userId,
    sourceType: input.sourceType,
    entityType: input.entityType,
    entityKey: input.entityKey.trim().toLowerCase(),
    polarity: input.polarity || 'positive',
    strength: clamp(input.strength ?? 0.5),
    context: input.context ? { ...input.context } : undefined,
    confidence: clamp(input.confidence ?? 0.5),
    observedAt,
    expiresAt: input.expiresAt ?? null,
    provenance: input.provenance,
  };
}

/** Decayed weight of one evidence row at `now`. 0 when expired/invalid. */
export function evidenceWeight(row: StyleEvidence, now = Date.now()): number {
  if (!row || row.strength <= 0 || row.confidence <= 0) return 0;
  if (row.expiresAt && Date.parse(row.expiresAt) <= now) return 0;
  const observed = Date.parse(row.observedAt);
  if (!Number.isFinite(observed) || observed > now + DAY_MS) return 0;
  const halfLife = HALF_LIFE_DAYS[row.sourceType] || 120;
  const ageDays = Math.max(0, (now - observed) / DAY_MS);
  return row.strength * row.confidence * Math.pow(0.5, ageDays / halfLife);
}

/** True when this row may contribute permanent (non-decaying) signal. */
export function isPermanentCandidate(row: StyleEvidence): boolean {
  return PERMANENT_SOURCES.has(row.sourceType) && !row.expiresAt && row.confidence >= 0.7;
}

/** Keep the newest row per (entityKey, sourceType, context) — bounded dedupe. */
export function dedupeStyleEvidence(
  rows: readonly StyleEvidence[],
  now = Date.now(),
  cap = 2000,
): StyleEvidence[] {
  const latest = new Map<string, StyleEvidence>();
  for (const row of rows.slice(-cap * 2)) {
    if (!row || !row.entityKey || !row.userId) continue;
    const key = JSON.stringify([
      row.entityKey,
      row.sourceType,
      row.polarity,
      row.context ? Object.keys(row.context).sort().map((k) => `${k}=${row.context?.[k]}`) : null,
    ]);
    const prior = latest.get(key);
    if (!prior || Date.parse(row.observedAt) >= Date.parse(prior.observedAt)) latest.set(key, row);
  }
  return [...latest.values()]
    .sort((a, b) => evidenceWeight(b, now) - evidenceWeight(a, now))
    .slice(0, cap);
}

/** Split rows into baseline / recent / override / contextual layers (§3C). */
export function layerStyleEvidence(
  rows: readonly StyleEvidence[],
  options: { recentDays?: number; contextId?: string } = {},
): Record<'baseline' | 'recent' | 'override' | 'contextual', StyleEvidence[]> {
  const recentMs = (options.recentDays ?? 45) * DAY_MS;
  const now = Date.now();
  const out = { baseline: [] as StyleEvidence[], recent: [] as StyleEvidence[], override: [] as StyleEvidence[], contextual: [] as StyleEvidence[] };
  for (const row of rows) {
    if (row.sourceType === 'declared_preference' || row.sourceType === 'manual_feedback') {
      out.override.push(row);
      continue;
    }
    if (options.contextId && row.context?.contextId === options.contextId) {
      out.contextual.push(row);
      continue;
    }
    if (now - Date.parse(row.observedAt) <= recentMs) out.recent.push(row);
    else out.baseline.push(row);
  }
  return out;
}
