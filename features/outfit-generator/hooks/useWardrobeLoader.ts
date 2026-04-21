/**
 * useWardrobeLoader — loads wardrobe or shop items for the outfit generator.
 */

import { useState, useEffect, useMemo } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useWardrobeStore from '../../../store/wardrobeStore';
import { fetchWardrobeDisplayItems } from '../../../src/services/outfitGenerationService';
import { useShopCatalog } from '../../../hooks/useShopCatalog';
import { getMacroCategory } from './useItemSelection';
import type { WardrobeDisplayItem } from '../types';
import type { ShopCatalogItem } from '../../try-on/types';

const LIVE_SHOP_SOURCE = 'apify-zara-men';

// The shop catalog is fetched from Supabase via useShopCatalog().
// When that fetch returns zero items (first launch, bad network), we
// fall back to the empty array below rather than bundling dozens of
// local PNGs into the IPA like the old scratch implementation did.
const SHOP_ITEMS_FALLBACK: Array<{
  id: string;
  name: string;
  brand: string;
  price: number;
  image: string;
  type: string;
  category: string;
}> = [];

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
  };
}

export function useWardrobeLoader(source: 'wardrobe' | 'shop') {
  const storeItems = useWardrobeStore((state) => state.items);
  const [wardrobeItems, setWardrobeItems] = useState<WardrobeDisplayItem[]>([]);
  const [loading, setLoading] = useState(source === 'shop');
  const {
    items: liveShopCatalog,
    loading: shopCatalogLoading,
  } = useShopCatalog({
    source: LIVE_SHOP_SOURCE,
    enabled: source === 'shop',
  });
  const liveShopItems = useMemo(
    () => liveShopCatalog.map(mapShopCatalogItemToWardrobeDisplayItem),
    [liveShopCatalog],
  );

  useEffect(() => {
    if (source === 'shop') {
      if (shopCatalogLoading) {
        setLoading(true);
        setWardrobeItems([]);
        return;
      }

      setWardrobeItems(liveShopItems);
      setLoading(false);
      return;
    }

    loadItems();
  }, [liveShopItems, shopCatalogLoading, source, storeItems]);

  const loadItems = async () => {
    try {
      setLoading(true);
      if (source === 'shop') {
        setWardrobeItems(liveShopItems);
        return;
      }

      // Primary: fetch directly from Supabase DB
      const dbItems = await fetchWardrobeDisplayItems();
      if (dbItems.length > 0) {
        setWardrobeItems(dbItems.map(item => ({
          id: item.id,
          image: item.imageUrl,
          type: item.type,
          color: item.color,
          name: item.name,
          macroCategory: item.macroCategory,
          category: item.category,
        })));
        return;
      }

      // Fallback: merge store + AsyncStorage items
      const data = await AsyncStorage.getItem('myWardrobeItems');
      const localItems = data ? JSON.parse(data) : [];
      const normalizedStoreItems = storeItems.map((item) => ({
        id: item.id,
        image: item.imageUrl,
        type: item.subCategory || item.category,
        color: item.primaryColor,
        name: item.name,
      }));
      const mergedByImage = new Map<string, any>();
      [...localItems, ...normalizedStoreItems].forEach((item: any) => {
        const image = item?.image || item?.imageUrl;
        if (!image) return;
        mergedByImage.set(String(image), item);
      });
      const personalItems = Array.from(mergedByImage.values());
      const existingIds = new Set(personalItems.map((i: any) => String(i.id || i.uniqueId || i.image || '')));
      const mergedItems = [
        ...SHOP_ITEMS_FALLBACK.filter(s => !existingIds.has(String(s.id))),
        ...personalItems,
      ];
      const normalized: WardrobeDisplayItem[] = mergedItems.map((item: any, index: number) => {
        const resolvedType = item.type || item.itemType || item.subCategory || item.category || 'Clothing Piece';
        const resolvedName = item.name || resolvedType;
        const resolvedCategory = item.category || item.itemType || item.subCategory || '';

        return {
          ...item,
          id: String(item.id || item.uniqueId || `uniq_item_${index}_${resolvedType || 'unknown'}`),
          image: item.image || item.imageUrl,
          type: resolvedType,
          name: resolvedName,
          category: resolvedCategory,
          macroCategory:
            item.macroCategory ||
            getMacroCategory(resolvedCategory || resolvedType, resolvedName),
        };
      }).filter((item) => item.image);

      setWardrobeItems(normalized);
    } catch (e) {
      console.error('Failed to load wardrobe', e);
    } finally {
      setLoading(false);
    }
  };

  return { wardrobeItems, loading };
}
