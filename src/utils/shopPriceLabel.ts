/** Display the listed amount in its known currency, without guessing a unit. */
export function shopPriceLabel(item: {price?: number; currency?: string}, locale?: string): string | null {
  if (typeof item.price !== 'number' || !Number.isFinite(item.price) || item.price <= 0) return null;
  const currency=String(item.currency || '').trim().toUpperCase();
  if (!/^[A-Z]{3}$/.test(currency)) return null;
  try {
    return new Intl.NumberFormat(locale, {
      style: 'currency', currency,
    }).format(item.price);
  } catch { return null; }
}
