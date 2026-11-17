/**
 * devtools.ts — Profiling helpers, bundle/startup safety, memory bounds,
 * accessibility checklist, debug panel model, decision inspector,
 * reproducible bug snapshots, load tests (5BA–5BG, 5BB–5BD, 5BH).
 *
 * Debug surfaces never include secrets or raw sensitive data.
 */
import type { BugSnapshot } from '../../types/platform';

export function timeIt<T>(feature: string, fn: () => T): { result: T; sample: { feature: string; durationMs: number; candidates: number; cacheHit: boolean } } {
  const t0 = Date.now();
  const result = fn();
  return { result, sample: { feature, durationMs: Date.now() - t0, candidates: 0, cacheHit: false } };
}

// ---------- Bundle/startup safety (5BB) ----------

const HEAVY_MODULES = ['openai', 'anthropic', '@google/generative-ai', 'vision-provider-sdk', 'model-sdk', 'image-processing-native', 'analytics-vendor-sdk', '@sentry/react-native'];

export function startupImportViolations(imports: readonly string[]): string[] {
  return imports.filter((i) => HEAVY_MODULES.some((h) => i.includes(h)));
}

// ---------- Memory safety (5BC) ----------

export function bounded<T>(arr: readonly T[], cap: number): T[] {
  return arr.slice(0, Math.max(0, cap));
}

export const MEMORY_CAPS = {
  imageCacheEntries: 50, eventQueue: 200, historyDepth: 2000,
  candidatePool: 500, telemetryLog: 500, outfitVersions: 10,
} as const;

// ---------- Accessibility checklist (5BD) ----------

export const A11Y_CHECKLIST = [
  'accessible labels on interactive elements',
  'screen reader names for icon-only buttons',
  '44pt minimum touch targets',
  'loading states announced',
  'cancel action available on long tasks',
  'empty states with guidance',
  'error recovery path',
] as const;

export function a11yReview(passed: Record<string, boolean>): string[] {
  return A11Y_CHECKLIST.filter((item) => !passed[item]);
}

// ---------- Debug panel model (5BE) ----------

export interface DebugPanelState {
  flags: Record<string, boolean>;
  experiment: Record<string, string>;
  provider: Record<string, string>;
  cache: Record<string, number>;
  versions: Record<string, string>;
  lastTrace: string | null;
  fallback: string | null;
  catalogSources: number;
  timings: Record<string, number>;
}

export function emptyDebugPanel(): DebugPanelState {
  return {
    flags: {}, experiment: {}, provider: {}, cache: {}, versions: {},
    lastTrace: null, fallback: null, catalogSources: 0, timings: {},
  };
}

const SECRET_KEYS = ['secret', 'token', 'apiKey', 'api_key', 'password', 'private'];

export function stripSecrets(state: Record<string, unknown>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(state)) {
    const lk = k.toLowerCase().replace(/[_-]/g, '');
    out[k] = SECRET_KEYS.some((s) => lk.includes(s.replace(/[_-]/g, ''))) ? '[redacted]' : String(v).slice(0, 200);
  }
  return out;
}

// ---------- Decision inspector (5BF) ----------

export interface InspectionResult {
  question: string;
  answer: string[];
  evidence: string[];
  rulesInvolved: string[];
}

export function inspectDecision(input: {
  question: 'why-ranked-first' | 'why-skip' | 'why-low-confidence' | 'why-filtered' | 'why-fallback';
  evidence: string[];
  rules: string[];
}): InspectionResult {
  const answers: Record<InspectionResult['question'], string> = {
    'why-ranked-first': 'Ranked first by deterministic validation, then compatibility score.',
    'why-skip': 'Skipped by a hard gate (budget, duplicate, fit, or context).',
    'why-low-confidence': 'Confidence capped by insufficient or conflicting evidence.',
    'why-filtered': 'Filtered by a category, weather, activity, or budget constraint.',
    'why-fallback': 'Primary path failed; deterministic fallback applied.',
  };
  return { question: input.question, answer: [answers[input.question]], evidence: [...input.evidence], rulesInvolved: [...input.rules] };
}

// ---------- Bug snapshot (5BG) ----------

export function buildBugSnapshot(input: {
  appVersion: string; featureVersions: Record<string, string>; requestIds: string[];
  inputs: Record<string, unknown>; flags: Record<string, boolean>; experiments: Record<string, string>;
  errors: Array<{ errorClass: BugSnapshot['errors'][number]['errorClass']; message: string }>; environment: string;
}): BugSnapshot {
  const sanitized: Record<string, string> = {};
  for (const [k, v] of Object.entries(input.inputs)) {
    const s = JSON.stringify(v);
    sanitized[k] = /data:image|waist|bust|https?:\/\/[^\s"]{60,}/.test(s) ? '[withheld-sensitive]' : s.slice(0, 500);
  }
  return {
    appVersion: input.appVersion, featureVersions: input.featureVersions,
    requestIds: input.requestIds.slice(0, 10), sanitizedInputs: sanitized,
    flags: input.flags, experiments: input.experiments,
    errors: input.errors.map((e) => ({ errorClass: e.errorClass, message: e.message.slice(0, 300) })),
    environment: input.environment, capturedAt: new Date().toISOString(),
  };
}

// ---------- Load tests (5BH) ----------

export interface LoadProbe { name: string; n: number; run: (i: number) => void }

export function runLoadProbe(probe: LoadProbe): { name: string; n: number; totalMs: number; avgMs: number } {
  const t0 = Date.now();
  for (let i = 0; i < probe.n; i++) probe.run(i);
  const totalMs = Date.now() - t0;
  return { name: probe.name, n: probe.n, totalMs, avgMs: totalMs / Math.max(1, probe.n) };
}
