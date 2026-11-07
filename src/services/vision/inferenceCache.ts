/**
 * inferenceCache.ts — Safe inference caching (§29).
 *
 * Key: asset identity/hash + task + model/provider version + schema
 * version. Invalidated on asset change, user correction, or material
 * schema/model change. Pure in-memory LRU with cap; no raw images stored.
 */
import type { InferenceCacheEntry } from '../../types/visualOS';

const MAX_ENTRIES = 300;
const SCHEMA_VERSION = 'v1';
const store = new Map<string, InferenceCacheEntry>();

export function cacheKey(input: {
  assetId: string;
  contentHash?: string | null;
  task: string;
  modelVersion: string;
}): string {
  return [input.contentHash || input.assetId, input.task, input.modelVersion, SCHEMA_VERSION].join('|');
}

export function getCached<T>(key: string): T | null {
  const hit = store.get(key);
  if (!hit) return null;
  store.delete(key);
  store.set(key, hit);
  return hit.result as T;
}

export function setCached(key: string, task: string, modelVersion: string, result: unknown): void {
  if (store.size >= MAX_ENTRIES) {
    const oldest = store.keys().next().value;
    if (oldest) store.delete(oldest);
  }
  store.set(key, { key, task, modelVersion, schemaVersion: SCHEMA_VERSION, result, cachedAt: new Date().toISOString() });
}

/** Invalidate every entry for an asset (asset changed / user corrected). */
export function invalidateAsset(assetId: string, contentHash?: string | null): number {
  let n = 0;
  for (const [k, v] of store) {
    void v;
    if (k.startsWith(`${contentHash || assetId}|`)) { store.delete(k); n++; }
  }
  return n;
}

export function cacheStats(): { size: number; schemaVersion: string } {
  return { size: store.size, schemaVersion: SCHEMA_VERSION };
}

export function clearInferenceCache(): void { store.clear(); }
