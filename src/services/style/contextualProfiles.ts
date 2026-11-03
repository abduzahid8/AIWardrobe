/**
 * contextualProfiles.ts — Contextual style profiles (Phase 3D).
 *
 * Context profile = global StyleDNA + context-scoped evidence + current
 * constraints. Explicit overrides win; weak inferred context signals never
 * override a strong global dimension. No sensitive identity inference.
 */

import type {
  ContextualStyleProfile,
  StyleContextId,
  StyleDimension,
  StyleDNA,
  StyleEvidence,
} from '../../types/styleOS';
import { aggregateStyleDNA } from './styleDna';
import { evidenceWeight } from './styleEvidence';

export const BUILT_IN_CONTEXTS: Array<{ id: StyleContextId; label: string }> = [
  { id: 'university', label: 'University' },
  { id: 'office', label: 'Office' },
  { id: 'casual_weekend', label: 'Casual weekend' },
  { id: 'dinner_date', label: 'Dinner / date' },
  { id: 'formal_event', label: 'Formal event' },
  { id: 'travel', label: 'Travel' },
  { id: 'warm_weather', label: 'Warm weather' },
  { id: 'rainy_day', label: 'Rainy day' },
  { id: 'gym', label: 'Gym' },
];

/** Hard constraints per context applied at query time (never stored as DNA). */
export const CONTEXT_CONSTRAINTS: Record<string, string[]> = {
  gym: ['activity:gym', 'flexible fit preferred'],
  formal_event: ['formality:high'],
  rainy_day: ['weather:water-resistant preferred'],
  warm_weather: ['climate:breathable preferred'],
  office: ['formality:smart minimum'],
};

export function buildContextualProfile(
  contextId: StyleContextId,
  label: string | undefined,
  global: StyleDNA,
  rows: readonly StyleEvidence[],
  now = Date.now(),
): ContextualStyleProfile {
  const scoped = rows.filter(
    (r) =>
      r &&
      r.userId === global.userId &&
      (r.context?.contextId === contextId || r.context?.occasion === contextId),
  );
  const scopedDna = aggregateStyleDNA(global.userId, scoped, now);
  const dimensions: Record<string, StyleDimension> = {};
  let scopedWins = 0;
  for (const [dim, base] of Object.entries(global.dimensions)) {
    const local = scopedDna.dimensions[dim];
    // Local override only with medium+ confidence and real weight.
    if (
      local &&
      (local.confidence === 'high' || local.confidence === 'medium') &&
      local.evidenceCount >= 1
    ) {
      // Explicit manual/declared provenance always wins over inference.
      const explicit = local.provenance.some((p) =>
        ['declared_preference', 'manual_feedback'].includes(p),
      );
      const baseStrong = base.confidence === 'high' && !explicit;
      dimensions[dim] = baseStrong ? base : local;
      if (dimensions[dim] === local) scopedWins++;
    } else {
      dimensions[dim] = base;
    }
  }
  // Context-only dimensions (no global signal yet) surface as-is.
  for (const [dim, local] of Object.entries(scopedDna.dimensions)) {
    if (!dimensions[dim]) dimensions[dim] = local;
  }
  const totalWeight = scoped.reduce((n, r) => n + evidenceWeight(r, now), 0);
  return {
    contextId,
    label: label || contextLabel(contextId),
    dimensions,
    evidenceCount: scoped.length,
    confidence:
      scopedWins >= 2 && totalWeight >= 3
        ? 'high'
        : scoped.length >= 2 && totalWeight >= 1
          ? 'medium'
          : 'low',
  };
}

export function contextLabel(contextId: string): string {
  return (
    BUILT_IN_CONTEXTS.find((c) => c.id === contextId)?.label ||
    contextId.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

/** Current constraints for a context (weather/dress-code hints merged). */
export function contextConstraints(
  contextId: string,
  hints: { dressCode?: string | null; climateHint?: string | null } = {},
): string[] {
  const base = [...(CONTEXT_CONSTRAINTS[contextId] || [])];
  if (hints.dressCode) base.push(`dress-code:${hints.dressCode}`);
  if (hints.climateHint) base.push(`climate:${hints.climateHint}`);
  return [...new Set(base)];
}
