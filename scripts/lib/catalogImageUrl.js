'use strict';

/**
 * Plain-JS mirror of `normalizeCatalogImageUrl` in
 * `src/domain/catalogIdentity.ts` for Node ingestion scripts.
 *
 * Production Zara rows carry the construction `.jpg&w=1024` (a query
 * continuation glued to the extension without `?`). The CDN still serves an
 * image, so the safe repair is to restore the query separator; dropping the
 * whole source would silently shrink catalog coverage. Keep this rule in sync
 * with the TypeScript authority and its cross-check test.
 */
function normalizeCatalogImageUrl(url) {
  const value = typeof url === 'string' ? url.trim() : '';
  if (!value) return undefined;
  if (value.includes('?')) return value;
  return value.replace(/(\.(?:jpe?g|png|webp|avif|gif))&(?=[^&=]+=)/i, '$1?');
}

module.exports = { normalizeCatalogImageUrl };
