import { supabase } from '../../lib/supabase';
import { withRequestDeadline } from '../lib/requestDeadline';

const INVENTORY_TTL_MS = 5 * 60 * 1000;
type Inventory = Awaited<ReturnType<typeof fetchStylistCatalogInventory>>;
let cached: { rows: Inventory; fetchedAt: number } | undefined;
let pending: Promise<Inventory> | undefined;

/** Only recent, successfully checked live inventory can seed the first render. */
export function getCachedStylistCatalogInventory(): Inventory | undefined {
  return cached && Date.now() - cached.fetchedAt < INVENTORY_TTL_MS ? cached.rows : undefined;
}

/** Share Home prewarming and screen loads; closing one screen must not cancel other readers. */
export async function loadStylistCatalogInventory(cancelSignal?: AbortSignal, refresh = false): Promise<Inventory> {
  if (cancelSignal?.aborted) throw Object.assign(new Error('The request was cancelled.'), { name: 'AbortError' });
  const rows = refresh ? undefined : getCachedStylistCatalogInventory();
  if (rows !== undefined) return rows;
  if (!pending) {
    pending = fetchStylistCatalogInventory().then(rows => {
      cached = { rows, fetchedAt: Date.now() };
      return rows;
    }).finally(() => { pending = undefined; });
  }
  const request = pending;
  return cancelSignal ? withRequestDeadline(() => request, 15000, cancelSignal) : request;
}

/** Balanced live evidence: a generic first page may contain no shoes or layers. */
async function fetchStylistCatalogInventory() {
  const filters = [
    'and(or(garment_type.eq.upper_body,category.eq.tops),or(name.ilike.%shirt%,name.ilike.%polo%,name.ilike.%tee%))',
    'garment_type.eq.lower_body,category.eq.bottoms',
    'garment_type.eq.shoes,category.eq.shoes',
    'category.eq.outerwear,name.ilike.%jacket%,name.ilike.%coat%,name.ilike.%blazer%,name.ilike.%cardigan%,name.ilike.%sweater%',
  ];
  return withRequestDeadline(async signal => {
    const pages = await Promise.all(filters.map(filter => supabase.from('shop_catalog')
      .select('id,name,brand,description,image_url,garment_type,price,currency,source_url')
      .eq('is_active', true).or(filter).order('id', { ascending: true }).limit(100).abortSignal(signal)));
    const rows = new Map<string, NonNullable<typeof pages[number]['data']>[number]>();
    for (const page of pages) {
      if (page.error) throw page.error;
      for (const row of page.data || []) rows.set(row.id, row);
    }
    return [...rows.values()];
  }, 15000);
}
