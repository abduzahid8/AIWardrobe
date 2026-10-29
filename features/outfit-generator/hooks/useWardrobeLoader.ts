/**
 * useWardrobeLoader — loads wardrobe or shop items for the outfit generator.
 */

import { useState, useEffect, useMemo, useSyncExternalStore } from 'react';
import useWardrobeStore from '../../../store/wardrobeStore';
import useAuthStore from '../../../store/auth';
import { authEvents } from '../../../store/authEvents';
import { useShopCatalog } from '../../../hooks/useShopCatalog';
import { getMacroCategory } from './useItemSelection';
import { detectSlotMismatch } from '../utils/occasionRules';
import type { WardrobeDisplayItem } from '../types';
import type { ShopCatalogItem } from '../../../src/types/shop';

const LIVE_SHOP_SOURCE = 'all';

export function mapShopCatalogGarmentTypeToCategory(
  garmentType?: ShopCatalogItem['garmentType'],
  name?: string,
): string {
  switch (garmentType) {
    case 'lower_body':
      return 'bottoms';
    case 'shoes':
      return 'shoes';
    case 'outfit':
      return 'outfits';
    case 'upper_body': {
      const macroCategory = getMacroCategory(name || '');
      return macroCategory === 'outerwear' ? 'outerwear' : 'tops';
    }
    default:
      return 'tops';
  }
}

export function mapShopCatalogItemToWardrobeDisplayItem(item: ShopCatalogItem): WardrobeDisplayItem {
  const category = mapShopCatalogGarmentTypeToCategory(item.garmentType, item.name);

  return {
    id: item.id,
    image: item.imageUrl,
    type: item.name,
    name: item.name,
    brand: item.brand,
    price: item.price,
    macroCategory: getMacroCategory(category, item.name),
    category,
    isShopItem: true,
    // Carry the retailer's direct product URL so Buy opens the product
    // page, not a Google search fallback.
    shopUrl: item.sourceUrl || undefined,
    sourceUrl: item.sourceUrl || undefined,
  };
}

const subscribeAuth = (notify: () => void) => {
  const login = authEvents.onLogin(notify), logout = authEvents.onLogout(notify);
  return () => { login(); logout(); };
};

export function useWardrobeLoader(source: 'wardrobe' | 'shop') {
  const storeItems = useWardrobeStore(state => state.items);
  const owner = useAuthStore(state => state.session?.user.id);
  const revision = useSyncExternalStore(subscribeAuth, authEvents.getRevision, authEvents.getRevision);
  const canRead = Boolean(owner && authEvents.canApplyToUser(owner));
  const [refreshing, setRefreshing] = useState(false);
  const { items: liveShopCatalog, loading: shopCatalogLoading } = useShopCatalog({
    source: LIVE_SHOP_SOURCE,
    enabled: source === 'shop',
  });
  const personalItems = useMemo(() => {
    if (!canRead) return [];
    return storeItems.filter(item => item.userId === owner && item.imageUrl).map(item => {
      const type = item.subCategory || item.category || 'Clothing';
      const name = item.name || type;
      const macro = getMacroCategory(item.category, type);
      return {
        id: item.id, image: item.imageUrl, type, name,
        color: item.primaryColor, brand: item.brand, category: item.category,
        macroCategory: detectSlotMismatch(macro, name, type) ?? macro,
      };
    });
  }, [storeItems, owner, canRead]);
  const shopItems = useMemo(() => liveShopCatalog.map(mapShopCatalogItemToWardrobeDisplayItem), [liveShopCatalog]);

  useEffect(() => {
    if (source !== 'wardrobe' || !owner || !authEvents.canApplyToUser(owner)) return;
    let active = true;
    setRefreshing(true);
    // Reuse the app's paginated, account-scoped sync with its TTL and shared
    // request. Store updates render directly instead of launching another fetch.
    void useWardrobeStore.getState().rehydrateFromCloud(false)
      .catch(() => { /* Retain current owner's cached choices while offline. */ })
      .finally(() => { if (active) setRefreshing(false); });
    return () => { active = false; };
  }, [source, owner, revision]);

  const wardrobeItems: WardrobeDisplayItem[] = source === 'shop' ? shopItems : personalItems;
  const loading = wardrobeItems.length === 0 && (source === 'shop' ? shopCatalogLoading : canRead && refreshing);
  return { wardrobeItems, loading };
}
