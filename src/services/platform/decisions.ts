/**
 * decisions.ts — Decision events, trace identity, provenance,
 * privacy-safe observability + analytics registry (5A–5C, 5X–5Y).
 *
 * Nothing logged here can carry raw images, body data, private URLs, or
 * prompt payloads: `sanitizeEvent` enforces the allowlist and drops the
 * event if a prohibited pattern appears.
 */
import type {
  DecisionEvent, DecisionProvenance, ErrorClass, RequestTrace,
  TraceStep, TraceSummary,
} from '../../types/platform';

let seq = 0;
const rid = (p: string) => `${p}_${Date.now().toString(36)}_${(seq++).toString(36)}`;

export function newRequestId(): string { return rid('req'); }
export function newTraceId(): string { return rid('tr'); }

export function startTrace(feature: string, parentRequestId?: string | null): RequestTrace {
  return {
    requestId: newRequestId(), parentRequestId: parentRequestId ?? null,
    traceId: newTraceId(), stepId: 'root', feature, startedAt: new Date().toISOString(),
  };
}

export function childStep(trace: RequestTrace, tool: string): TraceStep {
  return {
    stepId: `${trace.stepId}.${tool}.${seq++}`, tool, rulesInvoked: [],
    candidateCount: null, provider: null, validationOutcome: null,
    fallbackUsed: null, latencyMs: null,
  };
}

export function summarizeTrace(
  trace: RequestTrace, steps: readonly TraceStep[], errorClass?: ErrorClass,
): TraceSummary {
  return {
    traceId: trace.traceId, requestId: trace.requestId, feature: trace.feature,
    toolsUsed: [...new Set(steps.map((s) => s.tool))],
    rulesInvoked: [...new Set(steps.flatMap((s) => s.rulesInvoked))],
    totalCandidates: steps.reduce((n, s) => n + (s.candidateCount || 0), 0) || null,
    provider: steps.map((s) => s.provider).find(Boolean) || null,
    fallbackPath: steps.map((s) => s.fallbackUsed).find(Boolean) || null,
    ...(errorClass ? { errorClass } : {}),
  };
}

export function provenance(input: Partial<DecisionProvenance> & { rulesVersion: string }): DecisionProvenance {
  return {
    catalogVersion: null, wardrobeSnapshotVersion: null, styleProfileVersion: null,
    fitProfileVersion: null, modelProvider: null, modelVersion: null,
    schemaVersion: null, experimentId: null, ...input, createdAt: new Date().toISOString(),
  };
}

const PROHIBITED = /data:image\/|base64|"(bust|waist|hips|weight|heightCm)"\s*:|https?:\/\/|file:\/\//;

export function sanitizeEvent(event: DecisionEvent): DecisionEvent | null {
  const s = JSON.stringify(event);
  if (s.length > 8000 || PROHIBITED.test(s)) return null;
  const allowed = ['eventName','timestamp','userId','sessionId','requestId','feature','decisionType','inputEvidenceSummary','modelProvider','modelVersion','rulesVersion','confidence','candidateCount','latencyMs','fallbackUsed','experimentAssignment','outcome','errorClass','schemaVersion'];
  return Object.fromEntries(Object.entries(event).filter(([key]) => allowed.includes(key))) as unknown as DecisionEvent;
}

export function buildDecisionEvent(input: Omit<DecisionEvent, 'timestamp' | 'schemaVersion'>): DecisionEvent | null {
  return sanitizeEvent({ ...input, timestamp: new Date().toISOString(), schemaVersion: 1 });
}

// ---------- Analytics registry (5Y) ----------

export interface RegistryEntry {
  name: string;
  purpose: string;
  allowedFields: string[];
  prohibitedFields: string[];
  owner: string;
  schemaVersion: number;
}

const REGISTRY: Record<string, RegistryEntry> = {
  outfit_generated: {
    name: 'outfit_generated', purpose: 'Outfit funnel', allowedFields: ['requestId', 'candidateCount', 'latencyMs', 'fallbackUsed', 'rulesVersion'],
    prohibitedFields: ['images', 'wardrobeContents', 'bodyProfile'], owner: 'outfit', schemaVersion: 1,
  },
  buycheck_completed: {
    name: 'buycheck_completed', purpose: 'BuyCheck funnel', allowedFields: ['requestId', 'verdict', 'confidence', 'latencyMs', 'rulesVersion'],
    prohibitedFields: ['images', 'priceHistory'], owner: 'shopping', schemaVersion: 1,
  },
  fit_feedback_received: {
    name: 'fit_feedback_received', purpose: 'Fit calibration', allowedFields: ['requestId', 'kept', 'returned', 'rulesVersion'],
    prohibitedFields: ['measurements', 'bodyProfile'], owner: 'fit', schemaVersion: 1,
  },
  proactive_surfaced: {
    name: 'proactive_surfaced', purpose: 'Proactive funnel', allowedFields: ['requestId', 'topic', 'latencyMs'],
    prohibitedFields: ['images', 'wardrobeContents'], owner: 'style', schemaVersion: 1,
  },
};

export function validateAnalyticsEvent(name: string, fields: Record<string, unknown>): { ok: boolean; violations: string[] } {
  const entry = REGISTRY[name];
  if (!entry) return { ok: false, violations: [`unregistered event ${name}`] };
  const violations = Object.keys(fields).filter(
    (f) => !entry.allowedFields.includes(f) || entry.prohibitedFields.includes(f),
  );
  return { ok: violations.length === 0, violations };
}

export function registryNames(): string[] { return Object.keys(REGISTRY); }
