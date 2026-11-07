import type { CatalogCandidateRpcRequest } from '../domain/catalogCandidates';
import { supabase } from '../../lib/supabase';
import { withRequestDeadline } from '../lib/requestDeadline';
import { hasProductOnlyCatalogProvenance } from '../utils/productOnlyCatalog';

export type DiscoverySlot =
  'all' | 'top' | 'bottom' | 'shoes' | 'outerwear' | 'dresses' | 'accessory' | 'outfits';
export interface CatalogDiscoveryQuery {
  slot?: DiscoverySlot;
  source?: string;
  search?: string;
  limit?: number;
  page?: number;
  refresh?: boolean;
  seed?: string;
  currency?: string;
  maxPrice?: number;
}
export type CatalogDiscoveryRow = Record<string, any>;
export interface CatalogDiscoveryResult {
  rows: CatalogDiscoveryRow[];
  hasMore: boolean;
  strategy: 'balanced-rpc' | 'catalog-window';
}
const cache = new Map<string, { value: CatalogDiscoveryResult; at: number }>();
const pending = new Map<string, Promise<CatalogDiscoveryResult>>();
const TTL = 300000;
export function catalogDiscoverySeed(): string {
  return String(Math.floor(Date.now() / TTL));
}
function remember<T>(map: Map<string, T>, key: string, value: T) {
  if (map.size >= 24) map.delete(map.keys().next().value!);
  map.set(key, value);
}
export function discoverySlotForCategory(category: string): DiscoverySlot {
  return (
    (
      {
        tops: 'top',
        upper_body: 'top',
        bottoms: 'bottom',
        lower_body: 'bottom',
        outfit: 'outfits',
      } as Record<string, DiscoverySlot>
    )[category] || (category as DiscoverySlot)
  );
}
/** A shared, bounded discovery read. The RPC ranks the full eligible catalog on
 * the database. A missing capability is an explicit retrieval failure; normal
 * generation never repeats legacy catalog scans.
 */
export async function loadCatalogDiscovery(
  query: CatalogDiscoveryQuery = {},
  signal?: AbortSignal
): Promise<CatalogDiscoveryResult> {
  const options = {
    slot: query.slot || 'all',
    source: query.source || 'all',
    search: query.search?.trim() || '',
    seed: query.seed || catalogDiscoverySeed(),
    limit: Math.max(1, Math.min(100, Math.floor(query.limit || 100))),
    page: Math.max(0, Math.floor(query.page || 0)),
    currency: query.currency,
    maxPrice: query.maxPrice,
  };
  if (
    !Number.isFinite(options.limit) ||
    !Number.isFinite(options.page) ||
    options.page > 5000 ||
    options.search.length > 256 ||
    options.seed.length > 128 ||
    (options.maxPrice !== undefined && (!options.currency || !Number.isFinite(options.maxPrice) || options.maxPrice < 0))
  )
    throw new Error('invalid_catalog_discovery');
  const key = JSON.stringify(options),
    hit = cache.get(key);
  if (signal?.aborted) throw Object.assign(new Error('cancelled'), { name: 'AbortError' });
  if (!query.refresh && hit && Date.now() - hit.at < TTL) return hit.value;
  let request = pending.get(key);
  if (!request) {
    request = withRequestDeadline(async (transportSignal) => {
      const {data,error}=await supabase.rpc('catalog_discovery_candidates',{
        p_slot:options.slot,p_source:options.source,p_search:options.search,p_seed:options.seed,
        p_limit:options.limit,p_offset:options.page*options.limit,p_currency:options.currency||null,
        p_max_price:options.maxPrice??null,
      } satisfies CatalogCandidateRpcRequest).abortSignal(transportSignal);
      if(error)throw Object.assign(new Error('catalog_lookup_unavailable'),{code:'catalog_lookup_unavailable',stage:'catalog_retrieval'});
      if(!Array.isArray(data))throw new Error('catalog_lookup_unavailable');
      const raw=data as CatalogDiscoveryRow[];
      return {rows:raw.filter(hasProductOnlyCatalogProvenance),hasMore:raw.length===options.limit,strategy:'balanced-rpc' as const};

    }, 15000)
      .then((value) => {
        remember(cache, key, { value, at: Date.now() });
        return value;
      })
      .finally(() => pending.delete(key));
    pending.set(key, request);
  }
  return signal ? withRequestDeadline(() => request!, 15000, signal) : request;
}
