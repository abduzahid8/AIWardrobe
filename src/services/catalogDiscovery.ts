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
const FIELDS =
  'id,name,brand,description,image_url,garment_type,category,price,currency,source,source_url,external_id';
const FILTERS: Partial<Record<DiscoverySlot, string>> = {
  top: 'garment_type.eq.upper_body,category.eq.tops',
  bottom: 'garment_type.eq.lower_body,category.eq.bottoms',
  shoes: 'garment_type.eq.shoes,category.eq.shoes',
  outerwear:
    'category.eq.outerwear,garment_type.eq.outerwear,name.ilike.%jacket%,name.ilike.%coat%,name.ilike.%blazer%,name.ilike.%cardigan%,name.ilike.%sweater%',
};
const cache = new Map<string, { value: CatalogDiscoveryResult; at: number }>();
const pending = new Map<string, Promise<CatalogDiscoveryResult>>();
const counts = new Map<string, { value: number; at: number }>();
let rpcMissingUntil = 0;
const TTL = 300000;
export function catalogDiscoverySeed(): string {
  return String(Math.floor(Date.now() / TTL));
}
function hash(value: string): number {
  let n = 2166136261;
  for (const char of value) n = Math.imul(n ^ char.codePointAt(0)!, 16777619);
  return n >>> 0;
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
 * the database. Before activation, counted circular windows can still reach
 * interior rows; they never invent a catalog count or download the table.
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
    (options.maxPrice !== undefined && (!Number.isFinite(options.maxPrice) || options.maxPrice < 0))
  )
    throw new Error('invalid_catalog_discovery');
  const key = JSON.stringify(options),
    hit = cache.get(key);
  if (signal?.aborted) throw Object.assign(new Error('cancelled'), { name: 'AbortError' });
  if (!query.refresh && hit && Date.now() - hit.at < TTL) return hit.value;
  let request = pending.get(key);
  if (!request) {
    request = withRequestDeadline(async (transportSignal) => {
      if (typeof supabase.rpc === 'function' && Date.now() >= rpcMissingUntil) {
        const { data, error } = await supabase
          .rpc('catalog_discovery_candidates', {
            p_slot: options.slot,
            p_source: options.source,
            p_search: options.search,
            p_seed: options.seed,
            p_limit: options.limit,
            p_offset: options.page * options.limit,
            p_currency: options.currency || null,
            p_max_price: options.maxPrice ?? null,
          } satisfies CatalogCandidateRpcRequest)
          .abortSignal(transportSignal);
        if (!error) {
          const raw = (data || []) as CatalogDiscoveryRow[];
          return {
            rows: raw.filter(hasProductOnlyCatalogProvenance),
            hasMore: raw.length === options.limit,
            strategy: 'balanced-rpc' as const,
          };
        }
        // Missing RPC is the expected pre-migration state. Do not mask auth,
        // permissions or transport errors as a missing schema capability.
        if (['PGRST202', '42883'].includes(error.code)) rpcMissingUntil = Date.now() + TTL;
        else throw error;
      }
      const base = (count = false) => {
        let q = supabase
          .from('shop_catalog')
          .select(count ? 'id' : FIELDS, count ? { count: 'exact', head: true } : undefined)
          .eq('is_active', true);
        const filter = FILTERS[options.slot];
        if (filter) q = q.or(filter);
        else if (options.slot !== 'all')
          q = q.eq('garment_type', options.slot === 'outfits' ? 'outfit' : options.slot);
        if (options.source !== 'all') q = q.eq('source', options.source);
        if (options.search) {
          const term = JSON.stringify(`%${options.search.replace(/[%_*\\]/g, (c) => `\\${c}`)}%`);
          q = q.or(`name.ilike.${term},brand.ilike.${term},description.ilike.${term}`);
        }
        if (options.currency) q = q.eq('currency', options.currency);
        if (options.maxPrice !== undefined) q = q.lte('price', options.maxPrice);
        return q;
      };
      const first = base();
      const countKey = JSON.stringify({
        ...options,
        seed: undefined,
        page: undefined,
        limit: undefined,
      });
      let total = query.refresh ? undefined : counts.get(countKey);
      // Older read adapters without range retain their bounded limit contract.
      if (typeof first.range === 'function' && (!total || Date.now() - total.at >= TTL)) {
        const result = await base(true).abortSignal(transportSignal);
        if (result.error) throw result.error;
        if (typeof result.count === 'number' && Number.isFinite(result.count)) {
          total = { value: Math.max(0, result.count), at: Date.now() };
          remember(counts, countKey, total);
        }
      }
      const size = options.limit;
      const remaining = total ? Math.max(0, total.value - options.page * size) : size;
      if (!remaining) return { rows: [], hasMore: false, strategy: 'catalog-window' as const };
      const take = Math.min(size, remaining);
      const start = total?.value
        ? ((hash(`${options.seed}:${options.slot}:${options.source}`) % total.value) +
            options.page * size) %
          total.value
        : options.page * size;
      const read = async (from: number, amount: number, prepared = base()) => {
        let q = prepared.order('id', { ascending: true });
        q = typeof q.range === 'function' ? q.range(from, from + amount - 1) : q.limit(amount);
        const response = await q.abortSignal(transportSignal);
        if (response.error) throw response.error;
        return (response.data || []) as CatalogDiscoveryRow[];
      };
      const headTake = total ? Math.min(take, total.value - start) : take;
      const head = await read(start, headTake, first);
      const tail = headTake < take ? await read(0, take - headTake) : [];
      const raw = [...head, ...tail];
      return {
        rows: raw.filter(hasProductOnlyCatalogProvenance),
        hasMore: total ? options.page * size + take < total.value : raw.length === size,
        strategy: 'catalog-window' as const,
      };
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
