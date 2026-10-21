// Normalize equivalent provider item references before strict outfit validation.
// This does not select, invent, or replace any wardrobe/catalog identifier.
export function normalizeModelOutfits(value: unknown): any[] {
  if (!Array.isArray(value)) return [];
  return value.filter(outfit => outfit && typeof outfit === 'object' && !Array.isArray(outfit)).map(outfit => ({
    ...outfit,
    items: Array.isArray(outfit.items) ? outfit.items.map((item: unknown) => {
      if (typeof item === 'string') return { id: item.trim() };
      if (!item || typeof item !== 'object' || Array.isArray(item)) return { id: '' };
      const row = item as Record<string, unknown>;
      const id = row.id ?? row.itemId ?? row.item_id;
      return { ...row, id: typeof id === 'string' ? id.trim() : '' };
    }) : [],
  }));
}
