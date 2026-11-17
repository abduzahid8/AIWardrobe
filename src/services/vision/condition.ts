/**
 * condition.ts — Garment condition intelligence + repair/replace/keep (4P, 4Q).
 *
 * Never claims damage that cannot be reliably seen: visual hints become
 * `needs_attention` at most; `repair`/`replace_candidate` require user
 * confirmation or strong multi-signal evidence. Age alone never disposes.
 */
import type { LifecycleDecisionResult, TwinCondition } from '../../types/visualOS';

const VISUAL_HINTS = ['pilling', 'fading', 'stain', 'tear', 'sole wear', 'shape loss', 'fraying'];

export function conditionFromHints(
  hints: readonly string[],
  userConfirmed: boolean,
  confirmedValue?: TwinCondition,
): { condition: TwinCondition; evidence: string[] } {
  if (userConfirmed && confirmedValue) {
    return { condition: confirmedValue, evidence: ['User-confirmed condition overrides model output'] };
  }
  const seen = hints.map((h) => h.toLowerCase()).filter((h) => VISUAL_HINTS.some((v) => h.includes(v)));
  if (!seen.length) return { condition: 'unknown', evidence: ['No reliable visual condition signal'] };
  const severe = seen.some((h) => h.includes('tear') || h.includes('stain'));
  return {
    condition: severe ? 'needs_attention' : 'worn',
    evidence: [`Visual hints: ${seen.join(', ')} (unconfirmed — user review required for repair/replace)`],
  };
}

export interface RepairInputs {
  itemId: string;
  condition: TwinCondition;
  wearCount: number;
  satisfaction?: number | null;
  fitOk?: boolean;
  costPerWear?: number | null;
  duplicateCount?: number;
  replacementAvailable?: boolean;
  repairCostKnown?: boolean;
}

export function repairReplaceKeep(input: RepairInputs): LifecycleDecisionResult {
  const evidence: string[] = [];
  if (input.condition === 'unknown' && (input.satisfaction === null || input.satisfaction === undefined) && input.wearCount === 0) {
    return { itemId: input.itemId, decision: 'INSUFFICIENT_EVIDENCE', evidence: ['No condition, usage, or satisfaction signal'], confidence: 'low' };
  }
  evidence.push(`Condition ${input.condition}; ${input.wearCount} wears`);
  if (input.condition === 'excellent' || input.condition === 'good') {
    if ((input.satisfaction ?? 3) >= 3) return { itemId: input.itemId, decision: 'KEEP', evidence, confidence: 'high' };
    return { itemId: input.itemId, decision: (input.duplicateCount || 0) >= 2 ? 'SELL' : 'DONATE', evidence: [...evidence, 'Good condition but low satisfaction'], confidence: 'medium' };
  }
  if (input.condition === 'repair' || (input.condition === 'needs_attention' && input.repairCostKnown)) {
    return { itemId: input.itemId, decision: 'REPAIR', evidence: [...evidence, 'Repair path viable'], confidence: 'medium' };
  }
  if (input.condition === 'replace_candidate' && input.replacementAvailable) {
    return { itemId: input.itemId, decision: 'REPLACE_WHEN_NEEDED', evidence: [...evidence, 'Replacement available'], confidence: 'medium' };
  }
  if (input.condition === 'needs_attention' || input.condition === 'worn') {
    if (input.fitOk === false) return { itemId: input.itemId, decision: 'ARCHIVE', evidence: [...evidence, 'Fit issue plus wear'], confidence: 'low' };
    return { itemId: input.itemId, decision: 'KEEP', evidence: [...evidence, 'Usable; monitor condition'], confidence: 'medium' };
  }
  return { itemId: input.itemId, decision: 'KEEP', evidence, confidence: 'low' };
}
