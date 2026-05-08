/**
 * useHomeData - Centralized data fetching for HomeScreen
 * Replaces scattered useEffect chains with proper data management
 */

import { useState, useEffect, useCallback } from 'react';
import { useIsFocused } from '@react-navigation/native';
import useAuthStore from '../../../store/auth';
import useWardrobeStore from '../../../store/wardrobeStore';
import { useShopCatalog } from '../../../hooks/useShopCatalog';
import { createLogger } from '../../../src/utils/logger';

const logger = createLogger('useHomeData');

export function useHomeData() {
  const isFocused = useIsFocused();
  const { user, isAuthenticated } = useAuthStore();
  const wardrobeItems = useWardrobeStore((s) => s.items);
  const { items: shopItems, loading: shopLoading, refresh: refreshShop } = useShopCatalog();
  
  const [prompt, setPrompt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      await refreshShop();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [refreshShop]);

  return {
    isLoading: shopLoading,
    hasWardrobeItems: wardrobeItems.length > 0,
    wardrobeItemCount: wardrobeItems.length,
    shopItems,
    prompt,
    error,
    refresh,
    dismissPrompt: () => setPrompt(null),
  };
}
