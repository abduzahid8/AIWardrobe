/**
 * Read-only enrichment for Stylist 6's bounded catalog discovery.
 *
 * catalog_discovery_candidates() projects product identity/price but omits
 * shop_catalog.last_seen_at (the date of the last catalog feed import).
 * Without this, even a legitimately observed listing is misreported UNKNOWN.
 *
 * A bounded, service-role-only REST reread attaches that existing field to
 * the *same* active product identity. It does not change prices or stock,
 * invent merchant verification, update the DB, or accept client inventory.
 */
export const CATALOG_OBSERVATION_VERSION = 'catalog-observation-v1';
export const CATALOG_OBSERVATION_MAX_ROWS = 400;
export const CATALOG_OBSERVATION_MAX_IDS_PER_GET = 100;
export const CATALOG_OBSERVATION_MAX_QUERY_ID_BYTES = 5400;
export const CATALOG_OBSERVATION_TIMEOUT_MS = 2200;
const SAFE_ID = /^[A-Za-z0-9._-]{1,130}$/;

export function catalogObservationBatches(rows) {
  const ids = [];
  const seen = new Set();
  for (const row of rows || []) {
    const id = row?.id;
    if (typeof id !== 'string' || !SAFE_ID.test(id) || seen.has(id) ||
        Object.hasOwn(row, 'last_seen_at')) continue;
    seen.add(id);
    ids.push(id);
  }
  if (ids.length > CATALOG_OBSERVATION_MAX_ROWS) throw new Error('catalog_observation_limit_exceeded');
  const pages = [];
  let page = [], bytes = 0;
  for (const id of ids) {
    const size = encodeURIComponent(id).length + 1;
    if (page.length && (page.length >= CATALOG_OBSERVATION_MAX_IDS_PER_GET ||
        bytes + size > CATALOG_OBSERVATION_MAX_QUERY_ID_BYTES)) {
      pages.push(page);
      page = []; bytes = 0;
    }
    page.push(id); bytes += size;
  }
  if (page.length) pages.push(page);
  return pages;
}

function unchangedIdentity(source, current) {
  if (current?.is_active !== true) return false;
  for (const field of ['name', 'category', 'garment_type', 'currency', 'image_url', 'source_url']) {
    if (source[field] !== current[field]) return false;
  }
  return Number.isFinite(Number(source.price)) &&
    Number(source.price) === Number(current.price);
}

/**
 * Treat read failure as UNKNOWN age for the entire window rather than
 * fabricating a "recent" value. If the reread succeeds but the SKU changed
 * or was deleted/deactivated, exclude that stale discovery snapshot.
 * Successful HTTP responses are not themselves proof of checkout stock.
 */
export async function hydrateCatalogObservations({
  rows, url, serviceKey, fetchImpl = fetch, timeoutMs = CATALOG_OBSERVATION_TIMEOUT_MS,
}) {
  if (!Array.isArray(rows) || !rows.length) return Array.isArray(rows) ? rows : [];
  const batches = catalogObservationBatches(rows);
  if (!batches.length) return rows;
  const requested = new Set(batches.flat());
  let responseRows;
  try {
    const fetched = await Promise.all(batches.map(async (ids) => {
      const endpoint = new URL('/rest/v1/shop_catalog', url);
      endpoint.searchParams.set('select',
        'id,is_active,last_seen_at,name,category,garment_type,price,currency,image_url,source_url');
      endpoint.searchParams.set('id', 'in.(' + ids.join(',') + ')');
      const response = await fetchImpl(endpoint.toString(), {
        method: 'GET',
        headers: { apikey: serviceKey, Authorization: 'Bearer ' + serviceKey },
        redirect: 'error',
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) throw new Error('catalog_observation_unavailable');
      const data = await response.json();
      if (!Array.isArray(data) || data.length > ids.length) throw new Error('catalog_observation_invalid');
      const expected = new Set(ids), got = new Set();
      for (const value of data) {
        if (!expected.has(value?.id) || got.has(value.id)) throw new Error('catalog_observation_invalid');
        got.add(value.id);
      }
      return data;
    }));
    responseRows = new Map(fetched.flat().map(item => [item.id, item]));
  } catch {
    // No partial hydration: all rows remain in the honest UNKNOWN state.
    return rows;
  }
  return rows.flatMap(row => {
    if (!requested.has(row?.id)) return [row];
    const current = responseRows.get(row.id);
    if (!unchangedIdentity(row, current)) return [];
    const seen = current.last_seen_at;
    const valid = typeof seen === 'string' && Number.isFinite(Date.parse(seen));
    return [{ ...row, last_seen_at: valid ? seen : null }];
  });
}
