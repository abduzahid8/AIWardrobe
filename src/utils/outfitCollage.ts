import type { Outfit } from '../types/domain';
import type { OutfitCollageItem } from '../../components/ui/OutfitCollagePreview';
import { SHOP_CATALOG_ITEMS, SHOP_OUTFITS } from '../../data/shopCatalogItems';
import { CLASSIC_MENS_ITEMS, INSPO_MENS_SHOP_ITEMS } from '../../data/inspoMensShopItems';

type ImageItem = { id?: string; imageUrl?: string; category?: string; name?: string };
type ShopItem = { id: string; imageUrl?: string; name?: string };

const isValidImage = (v: unknown): v is string =>
  typeof v === 'string' && v.length > 0 &&
  (v.startsWith('http') || v.startsWith('file://') || v.startsWith('data:'));

// Static fallback catalogs for legacy outfits whose shop IDs paged out of
// the live (non-persisted) shop store or came from a backup catalog.
const STATIC_FALLBACK_MAP = (() => {
  const map = new Map<string, ShopItem>();
  for (const item of [...SHOP_CATALOG_ITEMS, ...SHOP_OUTFITS, ...CLASSIC_MENS_ITEMS, ...INSPO_MENS_SHOP_ITEMS]) {
    if (!map.has(item.id)) map.set(item.id, item);
  }
  return map;
})();

/**
 * Resolve up to 4 collage pieces for a saved outfit.
 * Prefers the frozen snapshot captured at like-time (wardrobe + shop),
 * falls back to wardrobe + live shop catalog + static backup catalogs for
 * legacy outfits saved before snapshots existed.
 */
export function resolveOutfitCollageItems(
  outfit: Outfit,
  wardrobeItems: ImageItem[] = [],
  shopItems: ShopItem[] = [],
): OutfitCollageItem[] {
  if (Array.isArray(outfit.snapshotItems) && outfit.snapshotItems.length > 0) {
    return outfit.snapshotItems
      .filter((s) => isValidImage(s.image))
      .slice(0, 4)
      .map((s) => ({ id: s.id, image: s.image, category: s.category, name: s.name }));
  }

  const wardrobeMap = new Map(wardrobeItems.map((i) => [String(i.id), i]));
  const shopMap = new Map(shopItems.map((i) => [String(i.id), i]));

  const resolved: OutfitCollageItem[] = [];
  for (const rawId of outfit.itemIds) {
    const id = String(rawId);
    const w = wardrobeMap.get(id);
    if (w?.imageUrl && isValidImage(w.imageUrl)) {
      resolved.push({ id, image: w.imageUrl, category: w.category, name: w.name });
      continue;
    }
    const s = shopMap.get(id) ?? STATIC_FALLBACK_MAP.get(id);
    if (s?.imageUrl && isValidImage(s.imageUrl)) {
      resolved.push({ id, image: s.imageUrl, category: undefined, name: s.name });
      continue;
    }
    if (isValidImage(rawId)) {
      resolved.push({ id, image: rawId });
    }
    if (resolved.length >= 4) break;
  }
  return resolved.slice(0, 4);
}

export type PieceTarget =
  | { kind: 'wardrobe'; item: ImageItem & { id: string } }
  | { kind: 'shop'; item: ShopItem }
  | { kind: 'unknown' };

/**
 * Decide where tapping an outfit piece should go: a wardrobe item opens
 * its detail screen, a shop item jumps to the Shop tab focused on it.
 * Wardrobe wins on id collisions since owned items are actionable.
 */
export function resolvePieceTarget(
  piece: { id?: string | number },
  wardrobeItems: ImageItem[] = [],
  shopItems: ShopItem[] = [],
): PieceTarget {
  const id = String(piece?.id ?? '');
  if (!id) return { kind: 'unknown' };

  const owned = wardrobeItems.find((i) => String(i.id) === id);
  if (owned?.id) return { kind: 'wardrobe', item: owned as ImageItem & { id: string } };

  const live = shopItems.find((i) => String(i.id) === id);
  if (live) return { kind: 'shop', item: live };

  const fallback = STATIC_FALLBACK_MAP.get(id);
  if (fallback) return { kind: 'shop', item: fallback };

  return { kind: 'unknown' };
}
