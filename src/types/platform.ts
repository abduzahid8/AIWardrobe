/**
 * platform.ts — Production AI platform domain types (Phase 5).
 *
 * Hardening wraps existing behavior; nothing here redefines product
 * semantics. All telemetry/analytics types exclude raw images, body data,
 * and private payloads by construction.
 */

export type DecisionEventName =
  | 'outfit_recommendation_generated' | 'outfit_recommendation_accepted'
  | 'outfit_recommendation_rejected' | 'buycheck_completed' | 'buycheck_followed'
  | 'buycheck_ignored' | 'fit_prediction_generated' | 'fit_feedback_received'
  | 'style_recommendation_generated' | 'reference_look_recreated'
  | 'wardrobe_gap_detected' | 'price_recommendation_generated'
  | 'no_buy_recommendation_generated' | 'proactive_recommendation_surfaced';

export interface DecisionEvent {
  eventName: DecisionEventName;
  timestamp: string;
  userId: string;
  sessionId?: string | null;
  requestId: string;
  feature: string;
  decisionType: string;
  inputEvidenceSummary: string[];
  modelProvider?: string | null;
  modelVersion?: string | null;
  rulesVersion: string;
  confidence?: number | null;
  candidateCount?: number | null;
  latencyMs?: number | null;
  fallbackUsed?: string | null;
  experimentAssignment?: string | null;
  outcome?: string | null;
  errorClass?: ErrorClass | null;
  schemaVersion: 1;
}

export type ErrorClass =
  | 'AUTH_REQUIRED' | 'NETWORK_UNAVAILABLE' | 'PROVIDER_TIMEOUT'
  | 'PROVIDER_RATE_LIMIT' | 'PROVIDER_MALFORMED' | 'VALIDATION_FAILED'
  | 'INSUFFICIENT_EVIDENCE' | 'CATALOG_UNAVAILABLE' | 'FIT_DATA_INCOMPLETE'
  | 'PRICE_HISTORY_UNAVAILABLE' | 'PERMISSION_REQUIRED' | 'FEATURE_DISABLED';

export interface RequestTrace {
  requestId: string;
  parentRequestId?: string | null;
  traceId: string;
  stepId: string;
  feature: string;
  startedAt: string;
}

export interface TraceStep {
  stepId: string;
  tool: string;
  rulesInvoked: string[];
  candidateCount?: number | null;
  provider?: string | null;
  validationOutcome?: 'pass' | 'fail' | 'skipped' | null;
  fallbackUsed?: string | null;
  latencyMs?: number | null;
}

export interface TraceSummary {
  traceId: string;
  requestId: string;
  feature: string;
  toolsUsed: string[];
  rulesInvoked: string[];
  totalCandidates?: number | null;
  provider?: string | null;
  fallbackPath?: string | null;
  errorClass?: ErrorClass | null;
}

export interface DecisionProvenance {
  rulesVersion: string;
  catalogVersion?: string | null;
  wardrobeSnapshotVersion?: string | null;
  styleProfileVersion?: string | null;
  fitProfileVersion?: string | null;
  modelProvider?: string | null;
  modelVersion?: string | null;
  schemaVersion?: string | null;
  experimentId?: string | null;
  createdAt: string;
}

export interface EvaluationCase {
  id: string;
  domain: string;
  input: unknown;
  invariants: string[];
  softExpectations?: string[];
  tags?: string[];
  severity: 'critical' | 'major' | 'minor';
  source: 'golden' | 'synthetic' | 'field';
}

export interface EvaluationResult {
  caseId: string;
  pass: boolean;
  violations: string[];
  scores?: Record<string, number> | null;
  latencyMs?: number | null;
  provider?: string | null;
  fallbackUsed?: string | null;
  rulesVersion: string;
}

export interface ExperimentVariant { name: string; weight: number }
export interface Experiment {
  id: string;
  description: string;
  status: 'draft' | 'running' | 'paused' | 'completed';
  assignmentUnit: 'user' | 'session' | 'request';
  variants: ExperimentVariant[];
  primaryMetric: string;
  secondaryMetrics?: string[];
  guardrails?: string[];
  rollbackRule?: string | null;
}

export type PlatformFlag =
  | 'fit_dna_v2' | 'buycheck_v2' | 'visual_parser' | 'recreate_look'
  | 'autonomous_stylist' | 'proactive_recommendations' | 'price_intelligence'
  | 'wardrobe_graph' | 'daily_stylist' | 'new_retrieval' | 'new_ranking';

export type ProviderState = 'healthy' | 'degraded' | 'unavailable';

export interface ProviderHealth {
  provider: string;
  state: ProviderState;
  successRate: number;
  timeoutRate: number;
  malformedRate: number;
  avgLatencyMs: number;
  recentFailures: number;
  circuitOpen: boolean;
  updatedAt: string;
}

export interface CostPolicy {
  feature: string;
  provider: string;
  estimatedUnitCost?: number | null;
  maxCalls: number;
  maxDailyPerUser?: number | null;
  cacheTtlMs: number;
  premiumRequired: boolean;
}

export interface LatencyBudget { feature: string; p50Ms: number; p95Ms: number; maxMs: number }

export interface CacheDescriptor {
  name: string;
  key: string;
  ttlMs: number;
  version: string;
  invalidationReason?: string | null;
  sizeBound: number;
}

export interface JobDefinition {
  name: string;
  idempotent: boolean;
  rateCapPerMin: number;
  maxRetries: number;
  description: string;
}

export interface PromptDefinition {
  id: string;
  version: string;
  purpose: string;
  inputSchema: Record<string, string>;
  outputSchema: Record<string, string>;
  providerCompatibility: string[];
  privacyClassification: 'public' | 'pseudonymous' | 'sensitive';
  maxTokens: number;
  maxImages: number;
  fallback: string;
}

export interface ReleaseStage {
  stage: number;
  label: string;
  successCriteria: string[];
  guardrails: string[];
  rollbackThreshold: string;
}

export interface BugSnapshot {
  appVersion: string;
  featureVersions: Record<string, string>;
  requestIds: string[];
  sanitizedInputs: Record<string, string>;
  flags: Record<string, boolean>;
  experiments: Record<string, string>;
  errors: Array<{ errorClass: ErrorClass; message: string }>;
  environment: string;
  capturedAt: string;
}
