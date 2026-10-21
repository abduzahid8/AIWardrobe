/** Audit each requested outfit separately using verified UZS catalog prices. */
export function catalogOutfitSpends(outfits, rows) {
  if (!Array.isArray(outfits)) return [];
  return outfits.map(outfit => {
    if (!Array.isArray(outfit?.items) || !outfit.items.length) return null;
    let total = 0;
    for (const item of outfit.items) {
      const row = rows.get(String(item?.id || '').replace(/^shop_/, ''));
      if (!row || row.currency !== 'UZS' || !Number.isFinite(row.price) || row.price <= 0) return null;
      total += row.price;
    }
    return Number.isFinite(total) ? total : null;
  });
}
