import type { CatalogCandidateRetrievalDiagnostics } from '../domain/catalogCandidates';
import { loadCatalogDiscovery, type CatalogDiscoveryQuery, type CatalogDiscoveryResult } from './catalogDiscovery';

/** Browse seeds are cacheable; an explicit generation owns a fresh replayable seed. */
let generationSequence = 0;
export function newOutfitGenerationSeed(context = 'outfit'): string {
  return `${context.slice(0, 48)}:${Date.now()}:${++generationSequence}`;
}

export interface OutfitCandidateRequest extends Omit<CatalogDiscoveryQuery, 'page' | 'limit'> {
  /** Share this seed across slots, retries, and double-tap deduplication. */
  seed: string;
  breadth?: number;
}
export interface OutfitCandidateResult extends CatalogDiscoveryResult {
  diagnostics: CatalogCandidateRetrievalDiagnostics;
}

/** One bounded mobile retrieval contract. Hard outfit/budget gates still run downstream.
 * A missing RPC is explicitly degraded: circular windows reach interior products,
 * but cannot promise database-wide relevance ranking. Never downloads the relation.
 */
export async function loadOutfitCandidates(
  request: OutfitCandidateRequest,
  signal?: AbortSignal,
): Promise<OutfitCandidateResult> {
  const { breadth = 200, ...query } = request;
  if (!Number.isFinite(breadth) || breadth < 1) throw new Error('invalid_candidate_breadth');
  const count = Math.min(300, Math.floor(breadth));
  const rows = new Map<string, CatalogDiscoveryResult['rows'][number]>();
  let strategy: CatalogDiscoveryResult['strategy'] = 'balanced-rpc';
  let hasMore = false;
  let pages = 0;
  for (let offset = 0; offset < count; offset += 100) {
    // Fixed page size prevents overlapping offsets on the final partial page.
    const result = await loadCatalogDiscovery({ ...query, limit: 100, page: pages }, signal);
    pages++;
    if (result.strategy !== 'balanced-rpc') strategy = result.strategy;
    for (const row of result.rows) rows.set(row.id, row);
    hasMore = result.hasMore;
    if (!hasMore) break;
  }
  const candidates = [...rows.values()].slice(0, count);
  return { rows: candidates, hasMore, strategy,
    diagnostics: { seed: request.seed, pages, degraded: strategy !== 'balanced-rpc', candidateCount: candidates.length } };
}
