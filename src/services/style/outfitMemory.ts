/**
 * outfitMemory.ts — Outfit feedback loop (3E) + OutfitMemory (3F).
 *
 * Feedback is optional, one-tap, dismissible, cooldown-controlled and
 * non-blocking: `shouldPromptForFeedback` gates prompting. Memory tracks
 * worn/planned with repeat counts; repetition is NOT scored as bad —
 * callers decide via `repetitionVerdict` and their own variety preference.
 */

import type {
  OutfitFeedback,
  OutfitMemoryEntry,
  OutfitRating,
} from '../../types/styleOS';

export const FEEDBACK_COOLDOWN_DAYS = 7;
const DAY_MS = 86_400_000;

let feedbackSeq = 0;

export function createOutfitFeedback(input: {
  userId: string;
  outfitId: string;
  rating: OutfitRating;
  reasons?: OutfitFeedback['reasons'];
  contextId?: string;
  createdAt?: string;
}): OutfitFeedback {
  return {
    id: `ofb_${Date.now().toString(36)}_${(feedbackSeq++).toString(36)}`,
    userId: input.userId,
    outfitId: input.outfitId,
    rating: input.rating,
    reasons: input.reasons ? [...input.reasons] : undefined,
    contextId: input.contextId,
    createdAt: input.createdAt || new Date().toISOString(),
  };
}

/** Prompt at most once per outfit per cooldown window; never block UI. */
export function shouldPromptForFeedback(
  outfitId: string,
  history: readonly OutfitFeedback[],
  now = Date.now(),
): boolean {
  const prior = history
    .filter((f) => f && f.outfitId === outfitId)
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt))[0];
  if (!prior) return true;
  return now - Date.parse(prior.createdAt) > FEEDBACK_COOLDOWN_DAYS * DAY_MS;
}

export function combinationKey(itemIds: readonly string[]): string {
  return [...itemIds].sort().join('+');
}

/** Record a confirmed wear; repeatCount increments on same combination. */
export function recordWear(
  memory: readonly OutfitMemoryEntry[],
  input: {
    userId: string;
    outfitId: string;
    itemIds: string[];
    wornDate?: string;
    contextId?: string | null;
    rating?: OutfitRating | null;
    weatherTemp?: number | null;
    weatherCondition?: string | null;
    plannedDate?: string | null;
    note?: string | null;
  },
): OutfitMemoryEntry[] {
  const key = combinationKey(input.itemIds);
  const wornDate = input.wornDate || new Date().toISOString().slice(0, 10);
  const sameCombo = memory.filter((m) => m.userId === input.userId && m.combinationKey === key);
  const repeatCount = sameCombo.length;
  const lastWorn =
    sameCombo
      .map((m) => m.wornDate)
      .sort()
      .at(-1) || null;
  const entry: OutfitMemoryEntry = {
    outfitId: input.outfitId,
    userId: input.userId,
    itemIds: [...input.itemIds],
    wornDate,
    plannedDate: input.plannedDate ?? null,
    contextId: input.contextId ?? null,
    rating: input.rating ?? null,
    weatherTemp: input.weatherTemp ?? null,
    weatherCondition: input.weatherCondition ?? null,
    repeatCount: repeatCount + 1,
    lastWorn,
    combinationKey: key,
    planned: !!input.plannedDate,
    confirmed: true,
    note: input.note ?? null,
  };
  return [...memory, entry].slice(-2000);
}

export interface RepetitionVerdict {
  outfitId: string;
  combinationKey: string;
  timesWorn: number;
  daysSinceLastWorn: number | null;
  verdict: 'fresh' | 'proven_repeat' | 'consider_variety';
  evidence: string[];
}

/**
 * Neutral repetition readout. `varietySeeking` comes from DNA
 * (repetition_tolerance); high variety-seeking nudges toward variety,
 * otherwise repeats of loved outfits are endorsed.
 */
export function repetitionVerdict(
  outfitId: string,
  memory: readonly OutfitMemoryEntry[],
  options: { varietySeeking?: 'low' | 'medium' | 'high'; recentDays?: number } = {},
  now = Date.now(),
): RepetitionVerdict | null {
  const rows = memory.filter((m) => m.outfitId === outfitId && m.confirmed);
  if (!rows.length) return null;
  const last = rows.map((r) => r.wornDate).sort().at(-1)!;
  const daysSince = Math.max(0, Math.round((now - Date.parse(last)) / DAY_MS));
  const loved = rows.filter((r) => r.rating === 'loved' || r.rating === 'good').length;
  const variety = options.varietySeeking || 'medium';
  const recentWindow = (options.recentDays ?? 14) * DAY_MS;
  const recentWears = rows.filter((r) => now - Date.parse(r.wornDate) <= recentWindow).length;
  let verdict: RepetitionVerdict['verdict'] = 'proven_repeat';
  if (recentWears >= 2 && variety !== 'low') verdict = 'consider_variety';
  else if (rows.length === 1 && daysSince > 30) verdict = 'fresh';
  return {
    outfitId,
    combinationKey: rows[0].combinationKey,
    timesWorn: rows.length,
    daysSinceLastWorn: daysSince,
    verdict,
    evidence: [
      `Worn ${rows.length} time${rows.length === 1 ? '' : 's'}; last worn ${last}`,
      loved ? `${loved} positive rating${loved === 1 ? '' : 's'}` : 'No positive rating recorded',
      variety === 'low'
        ? 'Repetition is welcome per tolerance'
        : 'Variety preference considered',
    ],
  };
}

/** Proven combinations: worn 2+ times with a positive rating. */
export function provenCombinations(
  memory: readonly OutfitMemoryEntry[],
  minWears = 2,
): OutfitMemoryEntry[] {
  const byCombo = new Map<string, OutfitMemoryEntry[]>();
  for (const m of memory) {
    if (!m.confirmed) continue;
    const list = byCombo.get(m.combinationKey) || [];
    list.push(m);
    byCombo.set(m.combinationKey, list);
  }
  const out: OutfitMemoryEntry[] = [];
  for (const list of byCombo.values()) {
    if (list.length >= minWears && list.some((m) => m.rating === 'loved' || m.rating === 'good'))
      out.push(list.sort((a, b) => b.wornDate.localeCompare(a.wornDate))[0]);
  }
  return out.slice(0, 50);
}

/** Garment ids worn in the last `days` (for packing/planning reuse). */
export function recentlyWornItemIds(
  memory: readonly OutfitMemoryEntry[],
  days = 14,
  now = Date.now(),
): string[] {
  const ids = new Set<string>();
  for (const m of memory) {
    if (!m.confirmed) continue;
    if (now - Date.parse(m.wornDate) <= days * DAY_MS) for (const id of m.itemIds) ids.add(id);
  }
  return [...ids];
}
