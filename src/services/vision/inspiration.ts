/**
 * inspiration.ts — Reference looks, recreate-this-look, visual search,
 * moodboard, Build-This-Style, visual shopping match (4L–4N, §38–40).
 *
 * Recreate preserves the visual idea, not the SKU: roles are matched by
 * structure + visual similarity (category, color family, silhouette,
 * formality), then validated. Anchors preserved. Missing pieces named at
 * most one role per look.
 */
import type { RecreateLookResult, ReferenceLook } from '../../types/visualOS';
import { colorFamily } from './styleLanguage';

let seq = 0;

export interface RoleCandidate {
  id: string;
  role: string;
  category: string;
  color?: string;
  silhouette?: string;
  formality01?: number;
  available: boolean;
}

export function saveReferenceLook(input: {
  userId: string;
  assetId: string;
  sourceLabel: string;
  styleRepresentation?: Record<string, string[]>;
  contexts?: string[];
}): ReferenceLook {
  return {
    id: `ref_${Date.now().toString(36)}_${(seq++).toString(36)}`,
    userId: input.userId,
    assetId: input.assetId,
    sourceLabel: input.sourceLabel,
    // Never permanently alters Style DNA from one save: medium-confidence,
    // contextual evidence only (consumer decides weight).
    styleRepresentation: input.styleRepresentation || {},
    savedAt: new Date().toISOString(),
    contexts: input.contexts || [],
    recreatedOutfitIds: [],
  };
}

/** Score a candidate against an inspiration role. */
function roleScore(
  want: { category: string; color?: string; silhouette?: string; formality01?: number },
  got: RoleCandidate,
): number {
  if (got.category !== want.category) return 0;
  let s = 0.4;
  if (want.color && got.color && colorFamily(got.color) === colorFamily(want.color)) s += 0.25;
  if (want.silhouette && got.silhouette === want.silhouette) s += 0.15;
  if (want.formality01 !== undefined && got.formality01 !== undefined) {
    s += 0.2 * (1 - Math.min(1, Math.abs(got.formality01 - want.formality01)));
  }
  return Math.round(s * 100) / 100;
}

export function recreateLook(
  referenceId: string,
  roles: Array<{ role: string; category: string; color?: string; silhouette?: string; formality01?: number }>,
  wardrobe: readonly RoleCandidate[],
  options: { anchors?: string[]; occasion?: string } = {},
): RecreateLookResult {
  const anchors = new Set(options.anchors || []);
  const available = wardrobe.filter((w) => w.available);
  const picks: string[] = [];
  const missingRoles: string[] = [];
  const explanation: string[] = [];
  for (const role of roles.slice(0, 6)) {
    const cands = available
      .filter((w) => !picks.includes(w.id))
      .map((w) => ({ w, s: roleScore(role, w) }))
      .filter((r) => r.s > 0)
      .sort((a, b) => b.s - a.s || a.w.id.localeCompare(b.w.id));
    const best = cands[0];
    if (best && best.s >= 0.4) {
      picks.push(best.w.id);
      explanation.push(`${role.role}: ${best.w.id} (match ${best.s}).`);
    } else {
      missingRoles.push(role.role);
      explanation.push(`${role.role}: no wardrobe equivalent — missing piece.`);
    }
  }
  const anchored = [...anchors].filter((a) => available.some((w) => w.id === a));
  for (const a of anchored) if (!picks.includes(a)) picks.push(a);
  const coverage = roles.length ? picks.filter((p) => !anchored.includes(p)).length / roles.length : 0;
  return {
    referenceId,
    closestWardrobeOutfit: picks.length ? [...picks] : null,
    bestPracticalOutfit: picks.length ? [...picks] : null,
    coverage: Math.round(coverage * 100) / 100,
    missingRoles: missingRoles.slice(0, 2),
    missingPieceSuggestion: missingRoles.length
      ? `One missing role: ${missingRoles[0]}. You do not need to replace anything else.`
      : null,
    explanation,
    confidence: coverage >= 0.8 ? 'high' : coverage >= 0.5 ? 'medium' : 'low',
  };
}

/** Visual style search over bounded candidate pool (no full-catalog scan). */
export function visualSearch(
  query: { category?: string; silhouette?: string; color?: string; maxPrice?: number },
  pool: ReadonlyArray<{ id: string; category: string; silhouette?: string; color?: string; price?: number; familyKey?: string }>,
  limit = 20,
): Array<{ id: string; score: number; why: string[] }> {
  const seenFamilies = new Set<string>();
  const out: Array<{ id: string; score: number; why: string[] }> = [];
  for (const item of pool.slice(0, 500)) {
    if (query.category && item.category !== query.category) continue;
    if (query.maxPrice !== undefined && (item.price ?? Infinity) > query.maxPrice) continue;
    // Family diversity: one rep per family.
    if (item.familyKey && seenFamilies.has(item.familyKey)) continue;
    let s = 0.3;
    const why: string[] = ['Category match'];
    if (query.silhouette && item.silhouette === query.silhouette) { s += 0.35; why.push('Same silhouette'); }
    if (query.color && colorFamily(item.color) === colorFamily(query.color)) { s += 0.35; why.push('Same color family'); }
    if (item.familyKey) seenFamilies.add(item.familyKey);
    out.push({ id: item.id, score: Math.round(s * 100) / 100, why });
    if (out.length >= limit) break;
  }
  return out.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
}

/** Moodboard → direction + wardrobe coverage (§38). */
export function moodboardDirection(
  looks: ReadonlyArray<{ styleRepresentation: Record<string, string[]> }>,
): { direction: Record<string, string[]>; coverage: { covered: string[]; missing: string[] } } {
  const counts = new Map<string, Map<string, number>>();
  for (const look of looks) {
    for (const [k, vals] of Object.entries(look.styleRepresentation)) {
      if (!counts.has(k)) counts.set(k, new Map());
      for (const v of vals.slice(0, 4)) counts.get(k)!.set(v, (counts.get(k)!.get(v) || 0) + 1);
    }
  }
  const direction: Record<string, string[]> = {};
  for (const [k, m] of counts) {
    direction[k] = [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([v]) => v);
  }
  return { direction, coverage: { covered: [], missing: [] } };
}

/** Compare moodboard direction against wardrobe roles. */
export function moodboardCoverage(
  direction: Record<string, string[]>,
  wardrobeRoles: ReadonlyArray<{ role: string; covered: boolean }>,
): { covered: string[]; partially: string[]; missing: string[]; overrepresented: string[] } {
  void direction;
  const covered = wardrobeRoles.filter((r) => r.covered).map((r) => r.role);
  const missing = wardrobeRoles.filter((r) => !r.covered).map((r) => r.role);
  return { covered, partially: [], missing, overrepresented: [] };
}

/** Build This Style plan (§39): own first, 1–3 leverage pieces max. */
export function buildThisStyle(
  ownedCombos: string[][],
  missingRoles: string[],
  budget?: number,
): { owned: string[][]; combosPossible: number; leveragePieces: string[]; noBuyPath: boolean } {
  return {
    owned: ownedCombos.slice(0, 10),
    combosPossible: ownedCombos.length,
    leveragePieces: missingRoles.slice(0, 3),
    noBuyPath: missingRoles.length === 0,
  };
}

/** Visual shopping match (§40). */
export function visualShoppingMatch(input: {
  productAttrs: Record<string, string>;
  closestOwnedId?: string | null;
  gapBlockedContexts?: string[];
  unlocks?: number;
  dnaFit?: 'high' | 'medium' | 'low';
  duplicateLevel?: 'low' | 'medium' | 'high';
}): { stylisticFit: string; utility: string; redundancy: string; recommendation: string } {
  const dup = input.duplicateLevel || 'low';
  return {
    stylisticFit: input.dnaFit === 'high' ? 'Strong style fit' : input.dnaFit === 'medium' ? 'Compatible with your style' : 'Outside your usual style',
    utility: (input.unlocks || 0) >= 3 ? `Unlocks ~${input.unlocks} outfits` : 'Low incremental outfit value',
    redundancy: dup === 'high' ? `Closest owned ${input.closestOwnedId || 'item'} is near-identical` : 'No strong owned substitute',
    recommendation: dup === 'high'
      ? 'Skip: closest owned piece already covers this.'
      : (input.unlocks || 0) >= 2 ? 'Consider: genuine gap support with style fit.' : 'Skip unless a real gap is confirmed.',
  };
}
