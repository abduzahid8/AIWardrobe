/**
 * wardrobeAgent.ts — Agentic tasks, proactive recommendations, fatigue
 * control, outcomes, novelty modes (Phase 3X–3Z, §35–38).
 *
 * The agent may analyze, rank, plan and prepare LOCAL actions only. It must
 * NOT purchase, deploy, mutate remotes, or create remote automations —
 * every produced task/suggestion is local data. Sparse surfacing with
 * cooldown + dismissal history prevents spam.
 */

import type {
  AgentTask,
  FatigueRecord,
  NoveltyMode,
  ProactiveSuggestion,
  RecommendationOutcome,
  StyleEvidence,
  SuggestionExplanation,
  WardrobeGoal,
} from '../../types/styleOS';
import { createStyleEvidence } from './styleEvidence';

const DAY_MS = 86_400_000;
let taskSeq = 0;
let suggSeq = 0;

const TASK_TITLES: Record<string, string> = {
  plan_week: 'Plan outfits for next week',
  build_capsule: 'Build a travel capsule',
  find_shoe: 'Find one missing shoe',
  review_underused: 'Review underused items',
  check_watches: 'Check price watches',
  prepare_event: 'Prepare for event',
  optimize_budget: 'Optimize shopping budget',
};

export function createAgentTask(
  userId: string,
  kind: AgentTask['kind'],
  payload: Record<string, string> = {},
  title?: string,
): AgentTask {
  return {
    id: `atask_${Date.now().toString(36)}_${(taskSeq++).toString(36)}`,
    kind,
    title: title || TASK_TITLES[kind] || kind,
    status: 'pending',
    payload: { userId, ...payload },
    resultSummary: null,
    evidence: [],
  };
}

export function completeAgentTask(
  task: AgentTask,
  resultSummary: string,
  evidence: string[] = [],
): AgentTask {
  return { ...task, status: 'done', resultSummary, evidence: [...task.evidence, ...evidence] };
}

// ---------- Proactive recommendations (3Y) ----------

export interface SuggestionInput {
  kind: string;
  title: string;
  detail: string;
  urgency: ProactiveSuggestion['urgency'];
  confidence: ProactiveSuggestion['confidence'];
  expectedValue: string;
  evidence: string[];
  tradeoffs?: string[];
  goalIds?: string[];
}

const COOLDOWN_DAYS: Record<ProactiveSuggestion['urgency'], number> = {
  high: 2,
  medium: 7,
  low: 14,
};

/** Max suggestions per week to prevent spam. */
export const MAX_SUGGESTIONS_PER_WEEK = 5;

/**
 * Filter candidate suggestions by cooldown, fatigue, goal relevance and the
 * weekly cap. Returns the sparse set worth surfacing.
 */
export function selectSuggestions(
  candidates: readonly SuggestionInput[],
  fatigue: Record<string, FatigueRecord>,
  goals: readonly WardrobeGoal[],
  recentSurfaced: readonly { createdAt: string }[],
  now = Date.now(),
): ProactiveSuggestion[] {
  const weekAgo = now - 7 * DAY_MS;
  const recentCount = recentSurfaced.filter((s) => Date.parse(s.createdAt) >= weekAgo).length;
  const budget = Math.max(0, MAX_SUGGESTIONS_PER_WEEK - recentCount);
  if (budget === 0) return [];
  const activeGoals = new Set(goals.filter((g) => g.status === 'active').map((g) => g.type));
  const scored = candidates
    .map((c) => {
      const f = fatigue[c.kind];
      if (f?.snoozedUntil && Date.parse(f.snoozedUntil) > now) return null;
      if (f?.lastSurfacedAt) {
        const cool = (COOLDOWN_DAYS[c.urgency] || 7) * DAY_MS;
        if (now - Date.parse(f.lastSurfacedAt) < cool) return null;
      }
      const dismissRate = f && f.surfacedCount7d > 0 ? f.dismissedCount7d / f.surfacedCount7d : 0;
      if (dismissRate >= 0.8 && f!.surfacedCount7d >= 3) return null;
      const urgencyW = c.urgency === 'high' ? 3 : c.urgency === 'medium' ? 2 : 1;
      const confW = c.confidence === 'high' ? 3 : c.confidence === 'medium' ? 2 : 1;
      const goalW = (c.goalIds || []).some((g) => activeGoals.has(g)) ? 2 : 1;
      return { c, score: urgencyW * confW * goalW - dismissRate * 4 };
    })
    .filter(Boolean) as Array<{ c: SuggestionInput; score: number }>;
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, budget)
    .filter((s) => s.score > 0)
    .map(({ c }) => ({
      id: `psug_${Date.now().toString(36)}_${(suggSeq++).toString(36)}`,
      kind: c.kind,
      title: c.title,
      detail: c.detail,
      urgency: c.urgency,
      confidence: c.confidence,
      expectedValue: c.expectedValue,
      evidence: [...c.evidence],
      tradeoffs: c.tradeoffs || [],
      createdAt: new Date(now).toISOString(),
    }));
}

// ---------- Fatigue control (3Z) ----------

export function updateFatigue(
  prev: FatigueRecord | undefined,
  topicKey: string,
  promptType: string,
  event: 'surfaced' | 'accepted' | 'dismissed' | 'snoozed',
  snoozeDays = 7,
  now = Date.now(),
): FatigueRecord {
  const base: FatigueRecord = prev || {
    topicKey,
    lastSurfacedAt: null,
    surfacedCount7d: 0,
    dismissedCount7d: 0,
    acceptedCount7d: 0,
    promptType,
  };
  switch (event) {
    case 'surfaced':
      return { ...base, lastSurfacedAt: new Date(now).toISOString(), surfacedCount7d: base.surfacedCount7d + 1, promptType };
    case 'accepted':
      return { ...base, acceptedCount7d: base.acceptedCount7d + 1, promptType };
    case 'dismissed':
      return { ...base, dismissedCount7d: base.dismissedCount7d + 1, promptType };
    case 'snoozed':
      return { ...base, snoozedUntil: new Date(now + snoozeDays * DAY_MS).toISOString(), promptType };
  }
}

// ---------- Outcomes (§36) + calibration ----------

export function summarizeOutcomes(
  outcomes: readonly RecommendationOutcome[],
): Record<RecommendationOutcome['outcome'], number> & { calibration: string[] } {
  const counts = {
    accepted: 0, ignored: 0, dismissed: 0, purchased: 0,
    returned: 0, worn_repeatedly: 0, rated_poorly: 0,
  } as Record<RecommendationOutcome['outcome'], number>;
  for (const o of outcomes) counts[o.outcome]++;
  const calibration: string[] = [];
  if (counts.returned >= 2) calibration.push('High return rate — tighten fit/price gating');
  if (counts.rated_poorly >= 3) calibration.push('Poor ratings — reduce similar suggestions');
  if (counts.worn_repeatedly >= 3) calibration.push('Repeated wears — reinforce this pattern');
  if (counts.dismissed > counts.accepted * 3 && counts.dismissed >= 4)
    calibration.push('Dismissal-heavy topic — extend cooldown');
  return { ...counts, calibration };
}

// ---------- Novelty (§37) + manual control (§38) ----------

/**
 * Experimental feedback is quarantined: a disliked experimental outfit
 * produces weak, expiring evidence — never a permanent global dislike.
 */
export function noveltyFeedbackEvidence(
  userId: string,
  outfitId: string,
  liked: boolean,
  mode: NoveltyMode,
  contextId?: string,
): StyleEvidence[] {
  if (mode !== 'experimental') {
    return [
      createStyleEvidence({
        userId,
        sourceType: liked ? 'outfit_liked' : 'outfit_disliked',
        entityType: 'outfit',
        entityKey: `outfit:${outfitId}`,
        polarity: liked ? 'positive' : 'negative',
        strength: 0.7,
        confidence: 0.7,
        context: contextId ? { contextId } : undefined,
        provenance: `novelty:${mode} feedback`,
      }),
    ];
  }
  if (liked) {
    return [
      createStyleEvidence({
        userId,
        sourceType: 'outfit_liked',
        entityType: 'outfit',
        entityKey: `outfit:${outfitId}`,
        polarity: 'positive',
        strength: 0.5,
        confidence: 0.5,
        context: contextId ? { contextId, novelty: 'experimental' } : { novelty: 'experimental' },
        provenance: 'novelty:experimental liked — context-scoped',
      }),
    ];
  }
  // Disliked experiment: weak + expires in 30 days, context-scoped.
  return [
    createStyleEvidence({
      userId,
      sourceType: 'outfit_disliked',
      entityType: 'outfit',
      entityKey: `outfit:${outfitId}`,
      polarity: 'negative',
      strength: 0.25,
      confidence: 0.4,
      context: contextId ? { contextId, novelty: 'experimental' } : { novelty: 'experimental' },
      expiresAt: new Date(Date.now() + 30 * DAY_MS).toISOString(),
      provenance: 'novelty:experimental disliked — quarantined, expires',
    }),
  ];
}

/** Manual overrides (§38): explicit preference outranks inference. */
export function manualOverrideEvidence(
  userId: string,
  kind: 'like_this' | 'stop_recommending' | 'want_more_color' | 'ignore_style_for_event' | 'changing_style',
  entityKey: string,
  contextId?: string,
): StyleEvidence {
  const polarity = kind === 'stop_recommending' ? 'negative' : 'positive';
  return createStyleEvidence({
    userId,
    sourceType: kind === 'changing_style' ? 'declared_preference' : 'manual_feedback',
    entityType: kind === 'want_more_color' ? 'color' : 'style_tag',
    entityKey,
    polarity,
    strength: 1,
    confidence: 0.95,
    context: contextId ? { contextId, manual: kind } : { manual: kind },
    provenance: `manual override: ${kind}`,
  });
}

// ---------- Explainability (§29) ----------

export function explain(
  whyNow: string,
  evidence: string[],
  changeIfFollowed: string,
  tradeoffs: string[],
  confidence: SuggestionExplanation['confidence'],
): SuggestionExplanation {
  return { whyNow, evidence: [...evidence], changeIfFollowed, tradeoffs: [...tradeoffs], confidence };
}

/** Sanitized analytics event (§39): topic + outcome only, no ids/photos. */
export function sanitizedOutcomeEvent(outcome: RecommendationOutcome): Record<string, string> {
  return { outcome: outcome.outcome, day: outcome.at.slice(0, 10) };
}
