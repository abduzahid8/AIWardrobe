import { supabase } from '../../lib/supabase';
import { loadCatalogDiscovery } from './catalogDiscovery';
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
export async function loadStylistCatalogInventory(
  cancelSignal?: AbortSignal,
  refresh = false
): Promise<Inventory> {
  if (cancelSignal?.aborted)
    throw Object.assign(new Error('The request was cancelled.'), { name: 'AbortError' });
  const rows = refresh ? undefined : getCachedStylistCatalogInventory();
  if (rows !== undefined) return rows;
  if (!pending) {
    pending = fetchStylistCatalogInventory(refresh)
      .then((rows) => {
        cached = { rows, fetchedAt: Date.now() };
        return rows;
      })
      .finally(() => {
        pending = undefined;
      });
  }
  const request = pending;
  return cancelSignal ? withRequestDeadline(() => request, 15000, cancelSignal) : request;
}

/** Balanced live evidence: a generic first page may contain no shoes or layers. */
let refreshRevision = 0;
async function fetchStylistCatalogInventory(refresh = false) {
  const seed = refresh ? `refresh:${Date.now()}:${++refreshRevision}` : undefined;
  return withRequestDeadline(async (signal) => {
    const pages = await Promise.all(
      (['top', 'bottom', 'shoes', 'outerwear'] as const).map((slot) =>
        loadCatalogDiscovery({ slot, limit: 100, seed, refresh }, signal)
      )
    );
    const rows = new Map<string, (typeof pages)[number]['rows'][number]>();
    for (const page of pages) {
      for (const row of page.rows) rows.set(row.id, row);
    }
    return [...rows.values()];
  }, 15000);
}
