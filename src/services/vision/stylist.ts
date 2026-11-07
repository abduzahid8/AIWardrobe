/**
 * stylist.ts — Autonomous stylist: tool layer, plan, orchestration,
 * daily/weekly planning, visual reranking (4U–4W, §50–53).
 *
 * NOT a giant prompt: intent → minimal tool selection → evidence →
 * deterministic validation → result + evidence summary. Hard validators
 * stay authoritative; model-flavored reasoning only explains. Boundaries
 * (§53) enforced: no purchase/delete/merge/upload/notify without explicit
 * user action — the orchestrator only READS and PROPOSES.
 */
import type {
  StylistPlan,
  StylistResult,
  StylistToolName,
} from '../../types/visualOS';
import { scoreCompatibility, type CompatItem } from './compatibility';

let seq = 0;

export type IntentKind =
  | 'daily_outfit' | 'weekly_plan' | 'diagnose' | 'transform'
  | 'recreate' | 'compare' | 'shopping_advice' | 'unknown';

export function classifyIntent(text: string): { kind: IntentKind; tools: StylistToolName[]; constraints: string[] } {
  const t = text.toLowerCase();
  const constraints: string[] = [];
  if (/\btomorrow\b|\btoday\b|\bdaily\b/.test(t)) {
    return { kind: 'daily_outfit', tools: ['wardrobe.query', 'planner.getEvents', 'planner.getWeatherContext', 'outfit.generate', 'outfit.validate', 'style.getProfile'], constraints };
  }
  if (/\bweek\b/.test(t)) {
    return { kind: 'weekly_plan', tools: ['wardrobe.query', 'planner.getEvents', 'planner.getWeatherContext', 'planner.getWardrobeGoals', 'outfit.generate', 'outfit.validate'], constraints };
  }
  if (/wrong|why.*(not|bad)|fix/.test(t)) {
    return { kind: 'diagnose', tools: ['visual.parseOutfit', 'outfit.diagnose', 'style.getProfile'], constraints };
  }
  if (/formal|casual|cleaner|warmer|minimal|expressive/.test(t)) {
    return { kind: 'transform', tools: ['outfit.transform', 'outfit.validate', 'fit.evaluate'], constraints };
  }
  if (/recreate|pinterest|instagram|reference|inspiration/.test(t)) {
    return { kind: 'recreate', tools: ['visual.parseOutfit', 'visual.recreateLook', 'outfit.validate'], constraints };
  }
  if (/better|compare|which/.test(t)) {
    return { kind: 'compare', tools: ['outfit.diagnose', 'style.getProfile'], constraints };
  }
  if (/buy|worth|cheap|price/.test(t)) {
    return { kind: 'shopping_advice', tools: ['shopping.buyCheck', 'price.evaluate', 'shopping.findMissingPiece'], constraints };
  }
  return { kind: 'unknown', tools: ['wardrobe.query', 'style.getProfile'], constraints };
}

export function buildStylistPlan(
  intentText: string,
  candidateCount: number,
  validationStages: string[] = ['hard-gates', 'visual-compatibility', 'diversity'],
): StylistPlan {
  const { kind, tools, constraints } = classifyIntent(intentText);
  return {
    id: `splan_${Date.now().toString(36)}_${(seq++).toString(36)}`,
    intent: kind,
    requestedConstraints: constraints,
    toolsSelected: tools, // minimal set only — never the full catalog
    candidateCounts: { retrieved: Math.min(candidateCount, 60) },
    validationStages,
    fallbackPath: 'structured-metadata-only',
    evidenceSummary: [],
    createdAt: new Date().toISOString(),
  };
}

/** Deterministic-first orchestration: validators gate, reasoning explains. */
export function orchestrateLocal(input: {
  plan: StylistPlan;
  candidates: CompatItem[][];
  hardValidate: (items: CompatItem[]) => { ok: boolean; reasons: string[] };
  context?: { occasion?: string; tempC?: number };
}): StylistResult & { plan: StylistPlan } {
  const fallbacksUsed: string[] = [];
  if (!input.candidates.length) {
    return {
      planId: input.plan.id, outfitItemIds: null,
      explanation: ['No candidates survived retrieval.'],
      evidence: ['Empty candidate set'],
      confidence: 'low', fallbacksUsed: ['structured-metadata-only'],
      plan: { ...input.plan, evidenceSummary: ['Empty candidate set'] },
    };
  }
  // Deterministic generation → hard validation (authority) → visual rerank.
  const valid = input.candidates.filter((c) => input.hardValidate(c).ok).slice(0, 20);
  if (!valid.length) {
    const reasons = input.hardValidate(input.candidates[0]).reasons;
    return {
      planId: input.plan.id, outfitItemIds: null,
      explanation: ['Candidates failed hard validation; no visual override applied.'],
      evidence: reasons, confidence: 'low',
      fallbacksUsed: ['validation-blocked'],
      plan: { ...input.plan, evidenceSummary: reasons },
    };
  }
  const ranked = valid
    .map((items) => ({ items, comp: scoreCompatibility(items, input.context || {}) }))
    .sort((a, b) => {
      const sa = a.comp.dimensions.reduce((n, d) => n + d.score, 0);
      const sb = b.comp.dimensions.reduce((n, d) => n + d.score, 0);
      return sb - sa;
    });
  const best = ranked[0];
  if (!best) {
    return {
      planId: input.plan.id, outfitItemIds: null,
      explanation: ['Ranking produced no result.'],
      evidence: ['Empty ranking'], confidence: 'low',
      fallbacksUsed: ['structured-metadata-only'],
      plan: { ...input.plan, evidenceSummary: ['Empty ranking'] },
    };
  }
  const evidence = [
    `Used ${input.plan.candidateCounts.retrieved} wardrobe candidates, kept ${valid.length} after hard gates.`,
    ...best.comp.strengths.slice(0, 2),
  ];
  return {
    planId: input.plan.id,
    outfitItemIds: best.items.map((i) => i.id),
    backupItemIds: ranked[1] ? ranked[1].items.map((i) => i.id) : undefined,
    explanation: [
      `Selected after deterministic validation then visual rerank (${ranked.length} valid).`,
      ...best.comp.improvements.slice(0, 1),
    ],
    evidence,
    confidence: best.comp.confidence,
    fallbacksUsed,
    plan: { ...input.plan, evidenceSummary: evidence },
  };
}

/** Daily stylist domain (§50): local only, no scheduled notifications. */
export function dailyStylist(input: {
  plan: StylistPlan;
  candidates: CompatItem[][];
  hardValidate: (items: CompatItem[]) => { ok: boolean; reasons: string[] };
  context: { occasion?: string; tempC?: number; eventLabel?: string };
  recentlyWornIds?: string[];
}): StylistResult & { preparationNotes: string[] } {
  const recent = new Set(input.recentlyWornIds || []);
  const fresh = input.candidates.filter((c) => !c.some((i) => recent.has(i.id)));
  const pool = fresh.length ? fresh : input.candidates;
  const res = orchestrateLocal({ plan: input.plan, candidates: pool, hardValidate: input.hardValidate, context: input.context });
  const notes = [
    input.context.eventLabel ? `Prepared for ${input.context.eventLabel}.` : 'No calendar event attached.',
    fresh.length ? 'Avoids recently worn pieces.' : 'All options reuse recent pieces; laundry may be pending.',
  ];
  return { ...res, preparationNotes: notes };
}

/** Weekly planner (§51): 5–7 days, footwear reuse, repetition limits. */
export function weeklyPlan(input: {
  days: Array<{ label: string; occasion?: string; tempC?: number }>;
  candidates: CompatItem[][];
  hardValidate: (items: CompatItem[], day: { occasion?: string; tempC?: number }) => { ok: boolean; reasons: string[] };
  maxRepeats?: number;
}): Array<{ day: string; itemIds: string[] | null; note: string }> {
  const used = new Map<string, number>();
  const maxRep = input.maxRepeats ?? 2;
  return input.days.slice(0, 7).map((day) => {
    const pool = input.candidates.filter(
      (c) => input.hardValidate(c, day).ok && c.every((i) => (used.get(i.id) || 0) < maxRep),
    );
    const pick = pool[0] || input.candidates.find((c) => input.hardValidate(c, day).ok) || null;
    if (pick) for (const i of pick) used.set(i.id, (used.get(i.id) || 0) + 1);
    return {
      day: day.label,
      itemIds: pick ? pick.map((i) => i.id) : null,
      note: pick ? `For ${day.occasion || 'general wear'}.` : 'No valid outfit for this day\u2019s constraints.',
    };
  });
}
