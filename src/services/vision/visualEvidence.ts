/**
 * visualEvidence.ts — VisualEvidence model + user correction loop (4A, §30).
 *
 * Priority: user > merchant structured > wardrobe metadata > deterministic
 * image analysis > model output. A user correction supersedes the inference
 * (supersededBy), wins as canonical, and becomes evaluation evidence.
 * Free-form tags are never over-normalized: low-confidence values stay
 * `candidate:` prefixed and unknown stays unknown.
 */
import type { VisualEvidence, VisualSource } from '../../types/visualOS';

let seq = 0;

const SOURCE_RANK: Record<VisualSource, number> = {
  user: 0,
  merchant: 1,
  metadata: 2,
  deterministic_image_analysis: 3,
  local_model: 4,
  remote_model: 5,
};

export function sourceRank(source: VisualSource): number {
  return SOURCE_RANK[source] ?? 9;
}

export function createVisualEvidence(input: {
  userId: string;
  sourceAssetId: string;
  entityType: VisualEvidence['entityType'];
  entityId?: string | null;
  attributeType: VisualEvidence['attributeType'];
  value: string;
  confidence: number;
  source: VisualSource;
  modelVersion?: string | null;
}): VisualEvidence {
  const c = Number.isFinite(input.confidence) ? Math.min(1, Math.max(0, input.confidence)) : 0;
  const raw = (input.value || '').trim().toLowerCase();
  return {
    id: `vev_${Date.now().toString(36)}_${(seq++).toString(36)}`,
    userId: input.userId,
    sourceAssetId: input.sourceAssetId,
    entityType: input.entityType,
    entityId: input.entityId ?? null,
    attributeType: input.attributeType,
    // Never invent precision: empty stays unknown.
    value: raw || 'unknown',
    confidence: c,
    source: input.source,
    modelVersion: input.modelVersion ?? null,
    createdAt: new Date().toISOString(),
    supersededBy: null,
    userConfirmed: input.source === 'user',
    userRejected: false,
  };
}

/** Canonical value: user correction > merchant > metadata > analysis > model. */
export function canonicalVisualValue(rows: readonly VisualEvidence[]): VisualEvidence | null {
  const live = rows.filter((r) => r && !r.userRejected && !r.supersededBy);
  if (!live.length) return null;
  return [...live].sort(
    (a, b) => sourceRank(a.source) - sourceRank(b.source) || b.confidence - a.confidence,
  )[0];
}

/** Apply a user correction: supersede prior inference, record winner. */
export function applyUserCorrection(
  rows: readonly VisualEvidence[],
  correction: { userId: string; sourceAssetId: string; entityType: VisualEvidence['entityType']; entityId?: string | null; attributeType: VisualEvidence['attributeType']; value: string },
): { rows: VisualEvidence[]; winner: VisualEvidence } {
  const winner = createVisualEvidence({ ...correction, confidence: 1, source: 'user' });
  const updated = rows.map((r) =>
    r.userId === correction.userId &&
    r.sourceAssetId === correction.sourceAssetId &&
    r.attributeType === correction.attributeType &&
    (r.entityId ?? null) === (correction.entityId ?? null) &&
    !r.supersededBy
      ? { ...r, supersededBy: winner.id }
      : r,
  );
  return { rows: [...updated, winner], winner };
}

/** Unconfirmed low-confidence model rows awaiting review. */
export function unconfirmedVisualEvidence(
  rows: readonly VisualEvidence[],
  threshold = 0.6,
): VisualEvidence[] {
  return rows.filter(
    (r) =>
      !r.userConfirmed && !r.userRejected && !r.supersededBy &&
      (r.source === 'local_model' || r.source === 'remote_model') &&
      r.confidence < threshold,
  ).slice(0, 100);
}
