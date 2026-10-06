import type { Outfit } from '../types/domain';
import type { OutfitCollageItem } from '../../components/ui/OutfitCollagePreview';

type ImageItem = { id?: string; imageUrl?: string; category?: string; name?: string };
type ShopItem = { id: string; imageUrl?: string; name?: string };

const isValidImage = (v: unknown): v is string =>
  typeof v === 'string' && v.length > 0 &&
  (v.startsWith('http') || v.startsWith('file://') || v.startsWith('data:'));

// Static fallback catalogs for legacy outfits whose shop IDs paged out of
// the live (non-persisted) shop store or came from a backup catalog.
// Lazily built on first miss — the previous IIFE merged ~2k bundled items
// at module import (startup parse + Map build on the critical path even
// when every outfit resolves from snapshots/live maps and never touches it).
let staticFallbackMap: Map<string, ShopItem> | null = null;
function getStaticFallbackMap(): Map<string, ShopItem> {
  if (!staticFallbackMap) {
    // require()d lazily — the ~120KB data modules must not parse/evaluate
    // on app startup (this util is imported by the eager Closet/Profile
    // tabs). First cache miss pays the cost, off the critical path.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const data = require('../../data/shopCatalogItems') as {
      SHOP_CATALOG_ITEMS: ShopItem[];
      SHOP_OUTFITS: ShopItem[];
    };
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const inspo = require('../../data/inspoMensShopItems') as {
      CLASSIC_MENS_ITEMS: ShopItem[];
      INSPO_MENS_SHOP_ITEMS: ShopItem[];
    };
    const map = new Map<string, ShopItem>();
    for (const item of [...data.SHOP_CATALOG_ITEMS, ...data.SHOP_OUTFITS, ...inspo.CLASSIC_MENS_ITEMS, ...inspo.INSPO_MENS_SHOP_ITEMS]) {
      if (!map.has(item.id)) map.set(item.id, item);
    }
    staticFallbackMap = map;
  }
  return staticFallbackMap;
}

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

  return resolveOutfitCollageItemsWithMaps(
    outfit,
    new Map(wardrobeItems.map((i) => [String(i.id), i])),
    new Map(shopItems.map((i) => [String(i.id), i])),
  );
}

/**
 * Map-taking variant for list rendering: build the lookup Maps ONCE per
 * data change and reuse across all outfits. Calling the array variant in a
 * `.map` over N outfits rebuilds both Maps per outfit — O(N × (W + S)) —
 * the main JS freeze behind Profile's first paint with a large wardrobe.
 */
export function resolveOutfitCollageItemsWithMaps(
  outfit: Outfit,
  wardrobeMap: Map<string, ImageItem>,
  shopMap: Map<string, ShopItem>,
): OutfitCollageItem[] {
  if (Array.isArray(outfit.snapshotItems) && outfit.snapshotItems.length > 0) {
    return outfit.snapshotItems
      .filter((s) => isValidImage(s.image))
      .slice(0, 4)
      .map((s) => ({ id: s.id, image: s.image, category: s.category, name: s.name }));
  }

  const resolved: OutfitCollageItem[] = [];
  for (const rawId of outfit.itemIds) {
    const id = String(rawId);
    const w = wardrobeMap.get(id);
    if (w?.imageUrl && isValidImage(w.imageUrl)) {
      resolved.push({ id, image: w.imageUrl, category: w.category, name: w.name });
      continue;
    }
    const s = shopMap.get(id) ?? getStaticFallbackMap().get(id);
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

  const fallback = getStaticFallbackMap().get(id);
  if (fallback) return { kind: 'shop', item: fallback };

  return { kind: 'unknown' };
}
