/**
 * safety.ts — Feature flags, observability, shadow mode, privacy and
 * cost policy (§44–49).
 *
 * Flags default safely (all experimental OFF) with no remote config.
 * Telemetry is sanitized by construction: only identifiers, never raw
 * images, body data, private URLs, or prompt payloads. Shadow mode keeps
 * production rankings authoritative while new rankers are evaluated.
 */
import type { SanitizedTelemetry, VisualFlag } from '../../types/visualOS';

const DEFAULT_FLAGS: Record<VisualFlag, boolean> = {
  closet_scan: false,
  visual_duplicates: false,
  inspiration_recreate: false,
  outfit_diagnostics: false,
  visual_search: false,
  stylist_orchestration: false,
};

let overrides: Partial<Record<VisualFlag, boolean>> = {};

export function isVisualEnabled(flag: VisualFlag): boolean {
  return overrides[flag] ?? DEFAULT_FLAGS[flag] ?? false;
}

export function setVisualFlag(flag: VisualFlag, on: boolean): void {
  overrides[flag] = on;
}

export function resetVisualFlags(): void {
  overrides = {};
}

export function enabledVisualFlags(): VisualFlag[] {
  return (Object.keys(DEFAULT_FLAGS) as VisualFlag[]).filter(isVisualEnabled);
}

// ---------- Observability (§44) ----------

const eventLog: SanitizedTelemetry[] = [];

export function recordTelemetry(e: Omit<SanitizedTelemetry, 'day'> & { day?: string }): SanitizedTelemetry {
  const entry: SanitizedTelemetry = { ...e, day: e.day || new Date().toISOString().slice(0, 10) };
  eventLog.push(entry);
  if (eventLog.length > 500) eventLog.splice(0, eventLog.length - 500);
  return entry;
}

export function telemetryLog(): readonly SanitizedTelemetry[] {
  return eventLog;
}

/** Assert a payload is log-safe: no base64 blobs, urls, or body metrics. */
export function isLogSafe(value: unknown): boolean {
  const s = JSON.stringify(value);
  if (s.length > 4000) return false;
  if (/data:image\/|base64|https?:\/\/[^"]{40,}/.test(s)) return false;
  if (/"(bust|waist|hips|weight|heightCm|bodyMass)"\s*:\s*\d/.test(s)) return false;
  return true;
}

// ---------- Shadow mode (§45) ----------

export interface ShadowComparison<T> {
  production: T;
  candidate: T;
  agree: boolean;
  metric: string;
}

export function shadowCompare<T>(
  production: T,
  candidate: T,
  equals: (a: T, b: T) => boolean,
  metric: string,
): ShadowComparison<T> {
  // Production result returned untouched; candidate only measured.
  return { production, candidate, agree: equals(production, candidate), metric };
}

// ---------- Privacy: request sanitization (§47) ----------

export interface ProviderImageRequest {
  assetId: string;
  contentHash?: string | null;
  imageRef: string;
  task: string;
  maxResolution: number;
  includeMetadata: Record<string, string>;
}

const BANNED_META_KEYS = new Set([
  'face', 'faces', 'identity', 'location', 'gps', 'background_text',
  'body_shape', 'body_rating', 'demographic', 'age_estimate', 'skin_tone',
]);

export function sanitizeProviderRequest(input: {
  assetId: string;
  contentHash?: string | null;
  imageRef: string;
  task: string;
  maxResolution?: number;
  metadata?: Record<string, string>;
}): ProviderImageRequest | null {
  if (!input.imageRef || /data:image\//.test(input.imageRef) && input.imageRef.length > 200_000) return null;
  const includeMetadata: Record<string, string> = {};
  for (const [k, v] of Object.entries(input.metadata || {})) {
    if (!BANNED_META_KEYS.has(k.toLowerCase())) includeMetadata[k] = v;
  }
  return {
    assetId: input.assetId,
    contentHash: input.contentHash ?? null,
    imageRef: input.imageRef,
    task: input.task,
    maxResolution: Math.min(input.maxResolution || 1024, 1600),
    includeMetadata,
  };
}

// ---------- Cost policy (4Z) ----------

export interface ModelTaskPolicy {
  task: string;
  maxCalls: number;
  maxImages: number;
  maxResolution: number;
  cacheable: boolean;
  preferLocal: boolean;
  premiumEligible: boolean;
}

const POLICIES: Record<string, ModelTaskPolicy> = {
  'parse-garment': { task: 'parse-garment', maxCalls: 3, maxImages: 2, maxResolution: 1024, cacheable: true, preferLocal: true, premiumEligible: false },
  'parse-outfit': { task: 'parse-outfit', maxCalls: 2, maxImages: 1, maxResolution: 1024, cacheable: true, preferLocal: true, premiumEligible: false },
  recreate: { task: 'recreate', maxCalls: 2, maxImages: 1, maxResolution: 768, cacheable: true, preferLocal: true, premiumEligible: true },
  tryon: { task: 'tryon', maxCalls: 1, maxImages: 2, maxResolution: 1024, cacheable: false, preferLocal: false, premiumEligible: true },
};

export function taskPolicy(task: string): ModelTaskPolicy {
  return POLICIES[task] || { task, maxCalls: 1, maxImages: 1, maxResolution: 768, cacheable: true, preferLocal: true, premiumEligible: false };
}

/** Skip paid/provider calls when metadata already answers the question. */
export function metadataAlreadySufficient(knownFields: readonly string[], needed: readonly string[]): boolean {
  return needed.every((n) => knownFields.includes(n));
}
