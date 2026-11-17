/**
 * fingerprint.ts — Garment fingerprint + duplicate review (4E, §31).
 *
 * Fingerprint: category + colors + silhouette + pattern + known
 * brand/model + embedding abstraction + perceptual-hash hook. Output
 * verdicts; auto-merge NEVER happens on weak evidence — merges require
 * explicit confirmation and preserve original IDs via merge mapping.
 */
import type { FingerprintResult, FingerprintVerdict } from '../../types/visualOS';

export interface FingerprintInput {
  itemId: string;
  category: string;
  subCategory?: string;
  dominantColors: string[];
  silhouette?: string;
  pattern?: string;
  brand?: string;
  modelKey?: string;
  embedding?: readonly number[];
  perceptualHash?: string | null;
}

function norm(s?: string): string {
  return (s || '').trim().toLowerCase();
}

function cosine(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? d / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

function hamming(a: string, b: string): number {
  if (a.length !== b.length || !a.length) return Number.MAX_SAFE_INTEGER;
  let d = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  return d;
}

export function fingerprintVerdict(a: FingerprintInput, b: FingerprintInput): FingerprintResult {
  const evidence: string[] = [];
  let score = 0;
  if (norm(a.category) && norm(a.category) === norm(b.category)) { score += 0.2; evidence.push('Same category'); }
  else { return { verdict: 'DISTINCT', score: 0, evidence: ['Different categories'], requiresConfirmation: false }; }
  const colorOverlap = a.dominantColors.filter((c) => b.dominantColors.map(norm).includes(norm(c)));
  if (colorOverlap.length) { score += 0.2; evidence.push(`Shared color ${colorOverlap[0]}`); }
  if (norm(a.silhouette) && norm(a.silhouette) === norm(b.silhouette)) { score += 0.15; evidence.push('Same silhouette'); }
  if (norm(a.pattern) && norm(a.pattern) === norm(b.pattern)) { score += 0.1; evidence.push('Same pattern'); }
  if (norm(a.brand) && norm(a.brand) === norm(b.brand)) { score += 0.15; evidence.push('Same brand'); }
  if (norm(a.modelKey) && norm(a.modelKey) === norm(b.modelKey)) { score += 0.15; evidence.push('Same model key'); }
  if (a.embedding && b.embedding) {
    const sim = cosine(a.embedding, b.embedding);
    score += sim * 0.15;
    evidence.push(`Visual embedding similarity ${sim.toFixed(2)}`);
  }
  if (a.perceptualHash && b.perceptualHash && a.perceptualHash.length === b.perceptualHash.length) {
    const dist = hamming(a.perceptualHash, b.perceptualHash);
    if (dist <= 5) { score += 0.25; evidence.push(`Near-identical perceptual hash (distance ${dist})`); }
  }
  score = Math.min(1, Math.round(score * 100) / 100);
  let verdict: FingerprintVerdict = 'DISTINCT';
  if (score >= 0.85) verdict = 'SAME_ITEM_HIGH_CONFIDENCE';
  else if (score >= 0.6) verdict = 'SAME_ITEM_POSSIBLE';
  else if (score >= 0.4) verdict = a.subCategory && a.subCategory === b.subCategory ? 'SAME_FAMILY' : 'VISUALLY_SIMILAR';
  else if (score >= 0.25) verdict = 'VISUALLY_SIMILAR';
  return { verdict, score, evidence, requiresConfirmation: verdict !== 'DISTINCT' };
}

export interface MergeDecision {
  keepId: string;
  mergedId: string;
  relation: 'merge' | 'keep_separate' | 'same_family' | 'not_related';
  decidedAt: string;
}

/**
 * Duplicate review (§31): never auto-deletes. Merge preserves history via
 * explicit mapping {mergedId -> keepId}.
 */
export function reviewDuplicate(
  verdict: FingerprintResult,
  choice: MergeDecision['relation'],
  keepId: string,
  mergedId: string,
): MergeDecision | null {
  if (verdict.verdict === 'DISTINCT') return null;
  if (choice === 'merge' && verdict.verdict !== 'SAME_ITEM_HIGH_CONFIDENCE' && verdict.verdict !== 'SAME_ITEM_POSSIBLE') {
    return null; // weak evidence cannot merge
  }
  return { keepId, mergedId, relation: choice, decidedAt: new Date().toISOString() };
}
