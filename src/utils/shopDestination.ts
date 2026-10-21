/**
 * Return the destination for a shop item.
 *
 * Catalog rows normally include the retailer's exact product URL. Older or
 * partially synced rows may not, so use a product-specific Google Shopping
 * search instead of leaving the Buy action without a destination.
 */
export interface ShopDestinationItem {
  brand?: string;
  name?: string;
  shopUrl?: string;
}

const isWebUrl = (value: string): boolean => /^https?:\/\//i.test(value);

function normalizeStoreUrl(value?: string): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) return undefined;
  if (isWebUrl(trimmed)) return trimmed;

  // Some imported feeds omit the scheme. Accept domain-like values while
  // keeping arbitrary strings out of Linking.openURL.
  if (!/\s/.test(trimmed) && /^[a-z0-9.-]+(?:\/|$)/i.test(trimmed) && trimmed.includes('.')) {
    return `https://${trimmed}`;
  }
  return undefined;
}

export function getShopSearchUrl(item: Pick<ShopDestinationItem, 'brand' | 'name'>): string {
  const query = [item.brand, item.name].filter(Boolean).join(' ').trim() || 'clothing';
  return `https://www.google.com/search?tbm=shop&q=${encodeURIComponent(query)}`;
}

export function getShopDestination(item: ShopDestinationItem): string {
  return normalizeStoreUrl(item.shopUrl) ?? getShopSearchUrl(item);
}
