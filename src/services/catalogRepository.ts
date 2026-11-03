import { supabase } from '../../lib/supabase';
import type { ShopCatalogItem } from '../types/shop';
import { loadCatalogDiscovery, discoverySlotForCategory } from './catalogDiscovery';
import { withRequestDeadline } from '../lib/requestDeadline';
import { hasProductOnlyCatalogProvenance } from '../utils/productOnlyCatalog';
import { catalogTitleColor } from '../utils/catalogIdentity';
import { buildCatalogFamilyViews, type CatalogProductFamilyView } from './catalogFamilyView';

export interface CatalogSearchQuery {
  query: string;
  category?: string;
  brand?: string;
  color?: string;
  priceMin?: number;
  priceMax?: number;
  currency?: string;
  limit?: number;
  page?: number;
  discovery?: boolean;
  seed?: string;
}
export interface CatalogSearchResult {
  items: ShopCatalogItem[];
  families: CatalogProductFamilyView[];
  hasMore: boolean;
  origin: 'live' | 'fixture';
}
export interface CatalogRepository {
  search(query: CatalogSearchQuery): Promise<CatalogSearchResult>;
  /** Only loaded families: a family key is never sent as a database row ID. */
  getFamily(key: string): Promise<CatalogProductFamilyView | null>;
  getVariant(id: string): Promise<ShopCatalogItem | null>;
}
const garmentType = (category?: string) =>
  ({ tops: 'upper_body', bottoms: 'lower_body', accessories: 'accessory' })[category || ''] ||
  category;
const literalPattern = (value: string) =>
  JSON.stringify(`%${value.replace(/[%_*\\]/g, (c) => `\\${c}`)}%`);
export function catalogRowToItem(row: Record<string, any>): ShopCatalogItem {
  return {
    id: row.id,
    name: row.name,
    brand: row.brand,
    price: Number(row.price),
    currency: row.currency || undefined,
    imageUrl: row.image_url,
    garmentType: row.garment_type,
    category: row.category,
    description: row.description || '',
    source: row.source,
    externalId: row.external_id,
    sourceUrl: row.source_url,
  };
}
const fields =
  'id,name,brand,price,currency,image_url,garment_type,description,source,source_url,external_id';

export function createFixtureCatalogRepository(
  load: () => readonly ShopCatalogItem[]
): CatalogRepository {
  const search = async (query: CatalogSearchQuery): Promise<CatalogSearchResult> => {
    const matches = load().filter((item) => {
      const text = `${item.name} ${item.brand} ${item.description || ''}`.toLowerCase();
      return (
        text.includes(query.query.trim().toLowerCase()) &&
        (!query.category || item.garmentType === garmentType(query.category)) &&
        (!query.brand || item.brand.toLowerCase() === query.brand.toLowerCase()) &&
        (!query.currency || item.currency === query.currency) &&
        (!query.color || catalogTitleColor(item) === query.color.toLowerCase()) &&
        (query.priceMin === undefined || item.price >= query.priceMin) &&
        (query.priceMax === undefined || item.price <= query.priceMax)
      );
    });
    const limit = Math.max(1, Math.min(100, Math.floor(query.limit || 20))),
      start = Math.max(0, query.page || 0) * limit;
    const items = matches.slice(start, start + limit);
    return {
      items,
      families: buildCatalogFamilyViews(items).families,
      hasMore: start + limit < matches.length,
      origin: 'fixture',
    };
  };
  return {
    search,
    getFamily: async (key) =>
      buildCatalogFamilyViews(load()).families.find((f) => f.familyKey === key) || null,
    getVariant: async (id) => load().find((item) => item.id === id) || null,
  };
}

export function createSupabaseCatalogRepository(fallback: CatalogRepository): CatalogRepository {
  const cache = new Map<string, { result: CatalogSearchResult; at: number }>();
  const pending = new Map<string, Promise<CatalogSearchResult>>();
  const search = async (query: CatalogSearchQuery): Promise<CatalogSearchResult> => {
    const key = JSON.stringify(query),
      cached = cache.get(key);
    if (cached && Date.now() - cached.at < 300000) return cached.result;
    const running = pending.get(key);
    if (running) return running;
    const request = withRequestDeadline(async (signal) => {
      const limit = Math.max(1, Math.min(100, Math.floor(query.limit || 20))),
        start = Math.max(0, query.page || 0) * limit;
      if (
        query.discovery !== false &&
        !query.brand &&
        !query.color &&
        query.priceMin === undefined
      ) {
        const page = await loadCatalogDiscovery(
          {
            slot: discoverySlotForCategory(query.category || 'all'),
            search: query.query,
            limit,
            page: query.page,
            seed: query.seed,
            currency: query.currency,
            maxPrice: query.priceMax,
          },
          signal
        );
        const items = page.rows.map(catalogRowToItem);
        return {
          items,
          families: buildCatalogFamilyViews(items).families,
          hasMore: page.hasMore,
          origin: 'live' as const,
        };
      }
      let db = supabase.from('shop_catalog').select(fields).eq('is_active', true);
      if (query.query.trim()) {
        const term = literalPattern(query.query.trim());
        db = db.or(`name.ilike.${term},brand.ilike.${term},description.ilike.${term}`);
      }
      if (query.category) db = db.eq('garment_type', garmentType(query.category));
      if (query.brand) db = db.eq('brand', query.brand);
      if (query.currency) db = db.eq('currency', query.currency);
      if (query.priceMin !== undefined) db = db.gte('price', query.priceMin);
      if (query.priceMax !== undefined) db = db.lte('price', query.priceMax);
      if (query.color) {
        const term = literalPattern(query.color);
        db = db.or(`name.ilike.${term},description.ilike.${term}`);
      }
      const { data, error } = await db
        .order('sort_order', { ascending: true })
        .order('id', { ascending: true })
        .range(start, start + limit - 1)
        .abortSignal(signal);
      if (error) throw error;
      const items = (data || [])
        .filter(hasProductOnlyCatalogProvenance)
        .map(catalogRowToItem)
        .filter((item) => !query.color || catalogTitleColor(item) === query.color.toLowerCase());
      return {
        items,
        families: buildCatalogFamilyViews(items).families,
        hasMore: (data || []).length === limit,
        origin: 'live' as const,
      };
    }, 15000)
      .then((result) => {
        if (cache.size >= 20) cache.delete(cache.keys().next().value!);
        cache.set(key, { result, at: Date.now() });
        return result;
      })
      .catch(() => fallback.search(query))
      .finally(() => pending.delete(key));
    pending.set(key, request);
    return request;
  };
  return {
    search,
    getFamily: async (key) => {
      const loaded = [...cache.values()]
        .filter((entry) => Date.now() - entry.at < 300000)
        .flatMap((entry) => entry.result.items);
      const family = buildCatalogFamilyViews(loaded).families.find((f) => f.familyKey === key);
      if (family) return family;
      return fallback.getFamily(key);
    },
    getVariant: async (id) => {
      try {
        return await withRequestDeadline(async (signal) => {
          const { data, error } = await supabase
            .from('shop_catalog')
            .select(fields)
            .eq('is_active', true)
            .eq('id', id)
            .limit(1)
            .abortSignal(signal);
          if (error) throw error;
          const row = data?.find(hasProductOnlyCatalogProvenance);
          return row ? catalogRowToItem(row) : null;
        }, 15000);
      } catch {
        return fallback.getVariant(id);
      }
    },
  };
}
