/** Preserve accepted product identity and monetary metadata through Home rendering. */
export function dailyOutfitDisplayItems(outfit: { items?: unknown }) {
  if (!Array.isArray(outfit.items)) return [];
  return outfit.items.map(item => ({
    ...item,
    image: item.imageUrl || item.image_url || item.image || '',
    shopUrl: item.shopUrl || item.sourceUrl,
  }));
}
