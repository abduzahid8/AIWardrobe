/** Local beta evidence. No network, photos, body data or automatic collection. */
import { newPurchaseRecordId } from '../purchase/localId';
export const BETA_SCHEMA_VERSION = 1;
export const BETA_FEATURES = [
  'general',
  'outfit',
  'buycheck',
  'fit',
  'shopping',
  'visual',
  'subscription',
] as const;
export const ISSUE_TYPES = [
  'wrong_outfit',
  'wrong_fit',
  'duplicate',
  'irrelevant_shopping',
  'price_issue',
  'visual_analysis_wrong',
  'slow',
  'bug',
  'confusing_ux',
  'privacy_concern',
  'other',
] as const;
export const CORRECTION_FIELDS = [
  'garment_type',
  'color',
  'fit',
  'duplicate_group',
  'outfit_context',
  'product_family',
  'price_match',
] as const;
export type BetaFeature = (typeof BETA_FEATURES)[number];
export type IssueType = (typeof ISSUE_TYPES)[number];
export type CorrectionField = (typeof CORRECTION_FIELDS)[number];
export interface ResultDiagnostics {
  requestId?: string;
  traceId?: string;
  ruleVersion?: string;
  featureFlags?: Record<string, boolean>;
  confidence?: 'low' | 'medium' | 'high';
  candidateCount?: number;
  fallback?: 'none' | 'local' | 'cached' | 'unavailable';
}
// Identifiers must be generated opaque IDs, not URLs, emails or serialized data.
const opaque = (value: unknown): string | undefined =>
  typeof value === 'string' && /^[a-zA-Z0-9_-]{1,96}$/.test(value) ? value : undefined;
export function sanitizeResultDiagnostics(input: ResultDiagnostics): ResultDiagnostics {
  const clean: ResultDiagnostics = {};
  for (const key of ['requestId', 'traceId', 'ruleVersion'] as const) {
    const value = opaque(input[key]);
    if (value) clean[key] = value;
  }
  if (['low', 'medium', 'high'].includes(input.confidence || ''))
    clean.confidence = input.confidence;
  if (
    Number.isSafeInteger(input.candidateCount) &&
    input.candidateCount! >= 0 &&
    input.candidateCount! <= 100000
  )
    clean.candidateCount = input.candidateCount;
  if (['none', 'local', 'cached', 'unavailable'].includes(input.fallback || ''))
    clean.fallback = input.fallback;
  if (input.featureFlags) {
    const allowed = [
      'buycheck_v2',
      'fit_dna_v2',
      'visual_parser',
      'recreate_look',
      'new_retrieval',
      'new_ranking',
      'autonomous_stylist',
      'price_intelligence',
    ];
    clean.featureFlags = Object.fromEntries(
      Object.entries(input.featureFlags).filter(
        ([key, value]) => allowed.includes(key) && typeof value === 'boolean'
      )
    );
  }
  return clean;
}
export interface BetaFeedback {
  schemaVersion: 1;
  id: string;
  createdAt: string;
  appVersion: string;
  feature: BetaFeature;
  helpful: boolean;
  issueType: IssueType;
  comment?: string;
  diagnostics: ResultDiagnostics;
  correction?: { field: CorrectionField; before: string; after: string; revertedAt?: string };
}
export function makeBetaFeedback(input: {
  appVersion: string;
  feature: BetaFeature;
  helpful: boolean;
  issueType: IssueType;
  comment?: string;
  diagnostics?: ResultDiagnostics;
  correction?: { field: CorrectionField; before: string; after: string };
}): BetaFeedback {
  if (
    !BETA_FEATURES.includes(input.feature) ||
    !ISSUE_TYPES.includes(input.issueType) ||
    typeof input.helpful !== 'boolean'
  )
    throw new Error('Invalid feedback');
  const correction = input.correction;
  if (correction && (!CORRECTION_FIELDS.includes(correction.field) || !correction.after.trim()))
    throw new Error('Invalid correction');
  return {
    schemaVersion: BETA_SCHEMA_VERSION,
    id: newPurchaseRecordId('beta'),
    createdAt: new Date().toISOString(),
    appVersion: /^\d+\.\d+\.\d+$/.test(input.appVersion) ? input.appVersion : 'unknown',
    feature: input.feature,
    helpful: input.helpful,
    issueType: input.issueType,
    ...(input.comment?.trim() ? { comment: input.comment.trim().slice(0, 500) } : {}),
    diagnostics: sanitizeResultDiagnostics(input.diagnostics || {}),
    ...(correction
      ? {
          correction: {
            field: correction.field,
            before: correction.before.slice(0, 100),
            after: correction.after.trim().slice(0, 100),
          },
        }
      : {}),
  };
}
/** Undo preserves the original assertion for local audit. Never modifies shared catalog data. */
export function revertBetaCorrection(record: BetaFeedback): BetaFeedback {
  return record.correction
    ? { ...record, correction: { ...record.correction, revertedAt: new Date().toISOString() } }
    : record;
}
export interface BetaEvent {
  schemaVersion: 1;
  eventId: string;
  name: 'feedback_recorded' | 'correction_recorded' | 'correction_reverted';
  feature: BetaFeature;
  helpful: boolean;
  issueType: IssueType;
  diagnostics: ResultDiagnostics;
}
/** Free text/correction values remain in user-controlled export, never in analytics. */
export function feedbackEvent(record: BetaFeedback, reverted = false): BetaEvent {
  return {
    schemaVersion: 1,
    eventId: `${record.id}${reverted ? '-undo' : ''}`,
    name: reverted
      ? 'correction_reverted'
      : record.correction
        ? 'correction_recorded'
        : 'feedback_recorded',
    feature: record.feature,
    helpful: record.helpful,
    issueType: record.issueType,
    diagnostics: sanitizeResultDiagnostics(record.diagnostics),
  };
}
/** Optional sink adapter: bounded dedupe; failures do not affect the user action and can retry. */
export function createBetaEmitter(sink: (event: BetaEvent) => Promise<void>) {
  const sent = new Set<string>(),
    pending = new Set<string>();
  return async (event: BetaEvent): Promise<boolean> => {
    if (sent.has(event.eventId) || pending.has(event.eventId)) return false;
    pending.add(event.eventId);
    try {
      // Reconstruct the allowlist at runtime, even if a caller bypasses TypeScript.
      if (
        !BETA_FEATURES.includes(event.feature) ||
        !ISSUE_TYPES.includes(event.issueType) ||
        !['feedback_recorded', 'correction_recorded', 'correction_reverted'].includes(event.name) ||
        !opaque(event.eventId) ||
        typeof event.helpful !== 'boolean'
      )
        return false;
      await sink({
        schemaVersion: 1,
        eventId: event.eventId,
        name: event.name,
        feature: event.feature,
        helpful: event.helpful,
        issueType: event.issueType,
        diagnostics: sanitizeResultDiagnostics(event.diagnostics),
      });
      if (sent.size >= 200) sent.delete(sent.values().next().value!);
      sent.add(event.eventId);
      return true;
    } catch {
      return false;
    } finally {
      pending.delete(event.eventId);
    }
  };
}
