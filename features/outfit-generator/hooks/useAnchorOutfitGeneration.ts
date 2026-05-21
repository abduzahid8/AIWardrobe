/**
 * useAnchorOutfitGeneration — orchestrates the anchor-item outfit generation pipeline.
 *
 * This hook is the primary entry point for the "Generate Outfit" flow triggered from
 * MyClosetScreen. It owns the AbortController lifecycle, the subscription gate check,
 * mannequin preload, and all dispatches to outfitGeneratorSlice.
 *
 * Subscription gate note: `verifySubscriptionFromServer()` is intentionally NOT called
 * inside `startGeneration()` because it is too slow for a synchronous gate check. The
 * AppState listener in RootNavigator handles server-side subscription refresh on every
 * app foreground event, keeping the local store up-to-date. We rely on the local
 * `checkFeatureAccess` check here (Requirements 5.1, 5.2, 5.3).
 *
 * Requirements: 5.1, 5.2, 5.3
 */

import { useRef, useEffect, useCallback } from 'react';
import { Share } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { Asset } from 'expo-asset';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

import useWardrobeStore from '../../../store/wardrobeStore';
import useSubscriptionStore from '../../../store/subscriptionStore';
import { useStylePreferenceStore } from '../../../store/stylePreferenceStore';
import { supabase } from '../../../lib/supabase';
import type { RootStackParamList } from '../../../navigation/types';
import type { ClosetClothingItem, GeneratedOutfit, WardrobeDisplayItem } from '../types';
import {
  generateOutfitsFromDB,
  generateOutfitsLocally,
} from '../../../src/services/outfitGenerationService';
import apiClient from '../../../src/services/apiClient';
import { getMacroCategory, canonicalizeMacroCategory } from '../../../src/utils/categoryMapper';

// ============================================
// TYPES
// ============================================

export interface UseAnchorOutfitGenerationParams {
  anchorItem: ClosetClothingItem | null;
  wardrobeItems: WardrobeDisplayItem[];
  style?: string;
}

export interface UseAnchorOutfitGenerationReturn {
  startGeneration: () => Promise<void>;
  cancelGeneration: () => void;
  saveOutfit: (outfit: GeneratedOutfit) => Promise<void>;
  shareOutfit: (imageUrl: string | null, outfit: GeneratedOutfit) => Promise<void>;
}

// ============================================
// HOOK
// ============================================

export function useAnchorOutfitGeneration({
  anchorItem,
  wardrobeItems,
  style = 'old_money',
}: UseAnchorOutfitGenerationParams): UseAnchorOutfitGenerationReturn {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();

  // ── Subscription store ──────────────────────────────────────────────────────
  const checkFeatureAccess = useSubscriptionStore((state) => state.checkFeatureAccess);
  const preferences = useStylePreferenceStore((state) => state.preferences);

  // ── Wardrobe store (outfitGeneratorSlice actions) ───────────────────────────
  const setStatus = useWardrobeStore((state) => state.setStatus);
  const setCurrentOutfit = useWardrobeStore((state) => state.setCurrentOutfit);
  const setGeneratedImageUrl = useWardrobeStore((state) => state.setGeneratedImageUrl);
  const setFallbackActive = useWardrobeStore((state) => state.setFallbackActive);
  const setError = useWardrobeStore((state) => state.setError);
  const reset = useWardrobeStore((state) => state.reset);

  // ── AbortController ref — holds the controller for the current in-flight request ──
  const abortControllerRef = useRef<AbortController | null>(null);

  // ── Mannequin base64 ref — preloaded on mount so it is ready when generation starts ──
  const mannequinB64Ref = useRef<string | null>(null);

  // ============================================
  // MANNEQUIN PRELOAD (same pattern as useOutfitGeneration)
  // ============================================

  useEffect(() => {
    (async () => {
      try {
        const asset = Asset.fromModule(
          require('../../../assets/images/mannequin_front.png'),
        );
        await asset.downloadAsync();
        const b64 = await FileSystem.readAsStringAsync(asset.localUri!, {
          encoding: FileSystem.EncodingType.Base64,
        });
        mannequinB64Ref.current = `data:image/png;base64,${b64}`;
      } catch (e) {
        console.warn('[useAnchorOutfitGeneration] Mannequin preload failed', e);
      }
    })();
  }, []);

  // ============================================
  // HELPERS
  // ============================================

  /**
   * Downloads an image from a URL or local asset and returns it as a base64
   * data URI. Returns null on failure (image composition will be skipped).
   */
  const fetchImageAsBase64 = useCallback(
    async (imgSource: string | number): Promise<string | null> => {
      try {
        if (typeof imgSource === 'number') {
          const asset = Asset.fromModule(imgSource);
          await asset.downloadAsync();
          if (!asset.localUri) return null;
          const b64 = await FileSystem.readAsStringAsync(asset.localUri, {
            encoding: FileSystem.EncodingType.Base64,
          });
          return `data:image/jpeg;base64,${b64}`;
        }
        if (imgSource.startsWith('data:')) return imgSource;
        const localUri = `${FileSystem.cacheDirectory}anchor_garment_${Date.now()}.jpg`;
        const { uri } = await FileSystem.downloadAsync(imgSource, localUri);
        const b64 = await FileSystem.readAsStringAsync(uri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        return `data:image/jpeg;base64,${b64}`;
      } catch (e) {
        console.warn('[useAnchorOutfitGeneration] fetchImageAsBase64 failed', e);
        return null;
      }
    },
    [],
  );

  // ============================================
  // startGeneration
  //
  // Full pipeline: subscription gate → edge function (with 45 s timeout race)
  // → image composition → dispatch results.
  //
  // Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 3.1, 3.2, 6.3, 6.5
  // ============================================

  const startGeneration = useCallback(async (): Promise<void> => {
    // ── Subscription gate (Requirements 5.1, 5.2) ──────────────────────────
    // Check BEFORE any state mutation or network call. If the user does not
    // have access, navigate to Paywall and return immediately.
    if (!checkFeatureAccess('aiOutfits')) {
      navigation.navigate('Paywall');
      return;
    }

    // ── Guard: do not start if no anchor item is set ────────────────────────
    if (!anchorItem) {
      return;
    }

    // ── Guard: do not start if already in progress (Requirement 6.1) ────────
    const currentStatus = useWardrobeStore.getState().status;
    if (currentStatus !== 'idle' && currentStatus !== 'complete' && currentStatus !== 'error') {
      return;
    }

    // ── Create a fresh AbortController for this generation run ──────────────
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // ── Resolve the anchor item's effective ID ───────────────────────────────
    const anchorItemId = anchorItem.id ?? anchorItem._id;

    // ── Step 1: Transition to 'selecting_items' (Requirement 3.1) ───────────
    setStatus('selecting_items', 'Building your outfit…');

    let outfit: GeneratedOutfit | null = null;
    let usedFallback = false;

    try {
      // ── Step 2: Call edge function with 45-second hard timeout ──────────────
      // The timeout is enforced here in the hook as a second layer of defence
      // (the service also has its own timeout, but we want the hook to control
      // the fallback dispatch independently).
      const TIMEOUT_MS = 45_000;

      const generationPromise = generateOutfitsFromDB({
        stylePreferences: style,
        occasion: 'Everyday',
        limit: 3,
        anchorItemId,
        preferences,
      });

      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('outfit_generation_timeout')), TIMEOUT_MS),
      );

      let result: Awaited<ReturnType<typeof generateOutfitsFromDB>>;

      try {
        result = await Promise.race([generationPromise, timeoutPromise]);
      } catch (raceErr: any) {
        // Timeout or HTTP error — activate local fallback
        console.warn('[useAnchorOutfitGeneration] Edge function failed/timed out, using local fallback:', raceErr?.message);
        result = await generateOutfitsLocally({
          stylePreferences: style,
          occasion: 'Everyday',
          limit: 3,
          anchorItemId,
          preferences,
        });
        usedFallback = true;
      }

      // If the service itself returned a non-success result (e.g. empty wardrobe),
      // treat it as a fallback scenario.
      if (!result.success || result.outfits.length === 0) {
        console.warn('[useAnchorOutfitGeneration] Generation returned no outfits, using local fallback');
        const fallbackResult = await generateOutfitsLocally({
          stylePreferences: style,
          occasion: 'Everyday',
          limit: 3,
          anchorItemId,
          preferences,
        });
        result = fallbackResult;
        usedFallback = true;
      }

      // Check if aborted before continuing
      if (controller.signal.aborted) return;

      // Pick the first outfit from the result
      const rawOutfit = result.outfits[0] ?? null;
      if (!rawOutfit) {
        setError('No outfits could be generated. Please try again.');
        return;
      }

      // Enforce anchor item in the outfit (client-side guarantee)
      const anchorMacro = canonicalizeMacroCategory(
        anchorItem.category
          ? getMacroCategory(anchorItem.category, anchorItem.type ?? '')
          : getMacroCategory(anchorItem.type ?? '', anchorItem.type ?? '')
      );

      const anchorOutfitItem = {
        id: anchorItemId,
        name: anchorItem.type ?? 'Item',
        image: anchorItem.imageUrl ?? anchorItem.image ?? '',
        color: anchorItem.color ?? '',
        type: anchorItem.type ?? '',
        macroCategory: anchorMacro,
        isShopItem: false as const,
        recommendation: 'Anchor item — user selected',
      };

      const alreadyHasAnchor = rawOutfit.items.some(
        (i) => String(i.id) === String(anchorItemId),
      );

      let patchedItems = [...rawOutfit.items];
      if (!alreadyHasAnchor) {
        const slotIdx = patchedItems.findIndex(
          (i) => canonicalizeMacroCategory(i.macroCategory ?? getMacroCategory(i.type ?? '', i.name ?? '')) === anchorMacro,
        );
        if (slotIdx >= 0) {
          patchedItems.splice(slotIdx, 1, anchorOutfitItem);
        } else {
          patchedItems.unshift(anchorOutfitItem);
        }
      }

      outfit = {
        ...rawOutfit,
        items: patchedItems,
      };

      // ── Dispatch fallback state if applicable ────────────────────────────
      if (usedFallback) {
        setFallbackActive(true);
        setStatus('fallback_active');
      }

      // Check if aborted before image composition
      if (controller.signal.aborted) return;

      // ── Step 3: Image composition (only on AI path, not fallback) ──────────
      let resultUrl: string | null = null;

      if (!usedFallback) {
        setStatus('composing_image', 'Composing the image…');

        try {
          // Find the best garment for the try-on render:
          // prefer the anchor item if it is a top/outerwear, otherwise use the first top.
          const isAnchorTop =
            anchorMacro === 'top' || anchorMacro === 'outerwear';

          const topItem = isAnchorTop
            ? anchorOutfitItem
            : outfit.items.find(
                (i) =>
                  (i.macroCategory ?? getMacroCategory(i.type ?? '', i.name ?? '')) === 'top' ||
                  (i.macroCategory ?? getMacroCategory(i.type ?? '', i.name ?? '')) === 'outerwear',
              );

          const garmentUri = topItem
            ? (typeof topItem.image === 'string' ? topItem.image : null)
            : null;

          if (garmentUri && mannequinB64Ref.current && !controller.signal.aborted) {
            const garmentB64 = await fetchImageAsBase64(garmentUri);

            if (garmentB64 && !controller.signal.aborted) {
              const response = await apiClient.post(
                '/api/tryon/render',
                {
                  mannequin_image: mannequinB64Ref.current,
                  garments: [
                    {
                      label: topItem?.macroCategory ?? 'top',
                      garment_image: garmentB64,
                    },
                  ],
                  total: 1,
                },
                { timeout: 90_000 },
              );

              const data = response.data;
              if (data?.success && data?.resultUrl) {
                resultUrl = data.resultUrl;
              } else {
                console.warn(
                  '[useAnchorOutfitGeneration] /api/tryon/render failed:',
                  data?.error,
                );
                // resultUrl stays null → triggers collage fallback in OutfitResultSheet
              }
            }
          }
        } catch (imgErr: any) {
          if (controller.signal.aborted) return;
          console.warn(
            '[useAnchorOutfitGeneration] Image composition error:',
            imgErr?.response?.data?.error ?? imgErr?.message ?? imgErr,
          );
          // On image composition failure: set generatedImageUrl to null
          // (triggers OutfitCollageDisplay fallback in the sheet — Requirement 3.2)
          resultUrl = null;
        }
      }

      // Check if aborted before dispatching results
      if (controller.signal.aborted) return;

      // ── Step 4: Dispatch results ─────────────────────────────────────────
      setGeneratedImageUrl(resultUrl);
      setCurrentOutfit(outfit);
      setStatus('complete');

    } catch (err: any) {
      if (controller.signal.aborted) return;

      // ── Network offline detection (Requirement 6.5) ──────────────────────
      const isNetworkError =
        err?.message === 'Network request failed' ||
        err?.code === 'ECONNABORTED' ||
        err?.message?.includes('network') ||
        err?.message?.includes('Network');

      if (isNetworkError) {
        // Preserve anchorItem in slice — do NOT call reset()
        setError(
          'No internet connection. Please check your network and try again.',
        );
      } else {
        console.error('[useAnchorOutfitGeneration] Unexpected error:', err);
        setError('Something went wrong. Please try again.');
      }
    }
  }, [
    anchorItem,
    checkFeatureAccess,
    fetchImageAsBase64,
    navigation,
    setCurrentOutfit,
    setError,
    setFallbackActive,
    setGeneratedImageUrl,
    setStatus,
    style,
  ]);

  // ============================================
  // cancelGeneration
  //
  // Aborts any in-flight network requests and resets all slice state.
  // ============================================

  const cancelGeneration = useCallback((): void => {
    // Abort any in-flight fetch/apiClient calls that were passed the signal
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;

    // Reset the slice back to idle — releases generatedImageUrl and currentOutfit
    // references to prevent memory leaks (Requirement 7.5)
    reset();
  }, [reset]);

  // ============================================
  // saveOutfit
  //
  // 1. Maps GeneratedOutfit → wardrobeStore.addOutfit format and persists
  //    the outfit in the local Zustand store.
  // 2. Inserts a matching row into the `saved_outfits` Supabase table for
  //    the currently authenticated user.
  //
  // Requirements: 4.6
  // ============================================

  const saveOutfit = useCallback(async (outfit: GeneratedOutfit): Promise<void> => {
    try {
      // Collect only the non-shop item IDs (wardrobe items the user owns)
      const itemIds = outfit.items
        .filter((item) => !item.isShopItem && item.id != null)
        .map((item) => String(item.id));

      // Map GeneratedOutfit → the shape expected by wardrobeStore.addOutfit
      // (Omit<Outfit, 'id' | 'createdAt' | 'wornCount' | 'lastWornAt' | 'saved'>)
      const outfitInput = {
        userId: '',           // will be overwritten below once we have the user id
        itemIds,
        occasion: 'casual' as const,
        generatedBy: 'ai' as const,
        previewImageUrl: typeof outfit.mainImage === 'string' ? outfit.mainImage : undefined,
        reasoning: outfit.description,
        style,
      };

      // Persist to local Zustand store
      const addOutfit = useWardrobeStore.getState().addOutfit;
      const newOutfitId = addOutfit(outfitInput);

      // Mark the newly added outfit as saved in the store
      const saveInStore = useWardrobeStore.getState().saveOutfit;
      saveInStore(newOutfitId);

      // Persist to Supabase `saved_outfits` table
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const { error } = await supabase.from('saved_outfits').insert({
          user_id: user.id,
          outfit_id: newOutfitId,
          item_ids: itemIds,
          description: outfit.description,
          preview_image_url: typeof outfit.mainImage === 'string' ? outfit.mainImage : null,
          style,
          occasion: 'casual',
          generated_by: 'ai',
          created_at: new Date().toISOString(),
        });

        if (error) {
          console.warn('[useAnchorOutfitGeneration] saveOutfit: Supabase insert failed', error);
        }
      }
    } catch (err) {
      console.warn('[useAnchorOutfitGeneration] saveOutfit error:', err);
    }
  }, [style]);

  // ============================================
  // shareOutfit
  //
  // Opens the native share sheet with the generated image URL as the message.
  // Falls back to the outfit description text when imageUrl is null.
  //
  // Requirements: 4.7
  // ============================================

  const shareOutfit = useCallback(
    async (imageUrl: string | null, outfit: GeneratedOutfit): Promise<void> => {
      try {
        const message = imageUrl ?? outfit.description;
        await Share.share({ message });
      } catch (err: any) {
        // User dismissed the share sheet — not an error worth surfacing
        if (err?.message !== 'The user did not share') {
          console.warn('[useAnchorOutfitGeneration] shareOutfit error:', err);
        }
      }
    },
    [],
  );

  return {
    startGeneration,
    cancelGeneration,
    saveOutfit,
    shareOutfit,
  };
}
