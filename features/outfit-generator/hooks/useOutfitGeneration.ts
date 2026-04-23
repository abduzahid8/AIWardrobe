/**
 * useOutfitGeneration — handles outfit generation, AI visuals, saving, and calendar logic.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { Alert } from 'react-native';
import * as Haptics from 'expo-haptics';
import * as FileSystem from 'expo-file-system/legacy';
import { Asset } from 'expo-asset';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../../../lib/supabase';
import useWardrobeStore from '../../../store/wardrobeStore';
import useAuthStore from '../../../store/auth';
import { useStylePreferenceStore } from '../../../store/stylePreferenceStore';
import { BASIC_CLOTHING_ITEMS } from '../../../data/basicClothingItems';
import { generateOutfitsFromDB } from '../../../src/services/outfitGenerationService';
import { fillMissingSlots, type OutfitSlotId } from '../../../src/services/shoppingService';
import { getMacroCategory } from './useItemSelection';
import { STYLE_PERSONALITY_MAP } from '../types';
import type { GeneratedOutfit, OutfitVisual, WardrobeDisplayItem } from '../types';
import { sanitizeGeneratedOutfitItemsDetailed } from '../utils/sanitizeGeneratedOutfit';
import {
  inferItemAttributes,
  rankItemsForStyle,
  normalizeStyleId,
  needsLayering,
} from '../utils/styleInference';

interface UseOutfitGenerationParams {
  source: 'wardrobe' | 'shop';
  calendarDate?: string;
  initialStyle?: string;
  wardrobeItems: WardrobeDisplayItem[];
  navigation: any;
  /** Anchor a specific wardrobe item so every generated outfit includes it. */
  baseItemId?: string;
}

export function useOutfitGeneration({
  source,
  calendarDate,
  initialStyle,
  wardrobeItems,
  navigation,
  baseItemId,
}: UseOutfitGenerationParams) {
  const { user } = useAuthStore();
  const stylePersonality = useStylePreferenceStore((state) => state.preferences.stylePersonality);
  const likeOutfit = useStylePreferenceStore((state) => state.likeOutfit);

  const [selectedStyle, setSelectedStyle] = useState(initialStyle || 'old_money');
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
  const [error, setError] = useState('');
  const [promptText, setPromptText] = useState('');
  const [outfitVisuals, setOutfitVisuals] = useState<Record<string, OutfitVisual>>({});
  const mannequinB64Ref = useRef<string | null>(null);

  // Preload mannequin image
  useEffect(() => {
    (async () => {
      try {
        const asset = Asset.fromModule(require('../../../assets/images/mannequin_front.png'));
        await asset.downloadAsync();
        const b64 = await FileSystem.readAsStringAsync(asset.localUri!, {
          encoding: FileSystem.EncodingType.Base64,
        });
        mannequinB64Ref.current = `data:image/png;base64,${b64}`;
      } catch (e) {
        console.warn('[useOutfitGeneration] Mannequin preload failed', e);
      }
    })();
  }, []);

  // Sync style from preferences (skip when caller provided an explicit initialStyle)
  useEffect(() => {
    if (initialStyle) return;
    if (stylePersonality && STYLE_PERSONALITY_MAP[stylePersonality]) {
      setSelectedStyle(STYLE_PERSONALITY_MAP[stylePersonality]);
    }
  }, [stylePersonality]);

  const fetchImageAsBase64 = async (imgSource: string | number): Promise<string | null> => {
    try {
      if (typeof imgSource === 'number') {
        const asset = Asset.fromModule(imgSource);
        await asset.downloadAsync();
        if (!asset.localUri) return null;
        const b64 = await FileSystem.readAsStringAsync(asset.localUri, { encoding: FileSystem.EncodingType.Base64 });
        return `data:image/jpeg;base64,${b64}`;
      }
      if (imgSource.startsWith('data:')) return imgSource;
      const localUri = `${FileSystem.cacheDirectory}garment_${Date.now()}.jpg`;
      const { uri } = await FileSystem.downloadAsync(imgSource, localUri);
      const b64 = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });
      return `data:image/jpeg;base64,${b64}`;
    } catch (e) {
      console.warn('[useOutfitGeneration] fetchImageAsBase64 failed', e);
      return null;
    }
  };

  const resolvePersistableImageUri = useCallback(async (imgSource?: string | number | null): Promise<string> => {
    if (imgSource == null) return '';

    if (typeof imgSource === 'string') {
      if (!imgSource) return '';
      if (imgSource.startsWith('basic_clothing_')) {
        const basicId = imgSource.replace('basic_clothing_', '');
        const basicItem = BASIC_CLOTHING_ITEMS.find(item => item.id === basicId);
        if (!basicItem) return '';
        const asset = Asset.fromModule(basicItem.image);
        await asset.downloadAsync();
        return asset.localUri || asset.uri || '';
      }
      return imgSource;
    }

    const asset = Asset.fromModule(imgSource);
    await asset.downloadAsync();
    return asset.localUri || asset.uri || '';
  }, []);

  const generateAIVisual = useCallback(async (outfit: GeneratedOutfit) => {
    const topItem = outfit.items.find((item) => {
      const macroCategory =
        item.macroCategory || getMacroCategory(item.type || '', item.name || item.type);
      return macroCategory === 'top' || macroCategory === 'outerwear';
    });
    const garmentUri = topItem?.image;
    if (!garmentUri) return;
    if (!mannequinB64Ref.current) return;

    setOutfitVisuals(prev => ({ ...prev, [outfit.id]: { loading: true, image: null } }));
    try {
      const garmentB64 = await fetchImageAsBase64(garmentUri);
      if (!garmentB64) throw new Error('garment base64 failed');

      const { data, error: fnErr } = await supabase.functions.invoke('mannequin-tryon', {
        body: {
          mannequin_image: mannequinB64Ref.current,
          garment_image: garmentB64,
          garment_type: 'upper_body',
        },
      });

      if (!fnErr && data?.success && data?.resultUrl) {
        setOutfitVisuals(prev => ({ ...prev, [outfit.id]: { loading: false, image: data.resultUrl } }));
      } else {
        console.warn('[useOutfitGeneration] mannequin-tryon failed:', fnErr || data?.error);
        setOutfitVisuals(prev => ({ ...prev, [outfit.id]: { loading: false, image: null } }));
      }
    } catch (e) {
      console.warn('[useOutfitGeneration] generateAIVisual error:', e);
      setOutfitVisuals(prev => ({ ...prev, [outfit.id]: { loading: false, image: null } }));
    }
  }, []);

  const enforceAnchorInOutfits = useCallback(
    (outfitsToPatch: GeneratedOutfit[]): GeneratedOutfit[] => {
      if (!baseItemId) return outfitsToPatch;
      const anchor = wardrobeItems.find((w) => String(w.id) === String(baseItemId));
      if (!anchor) return outfitsToPatch;

      const anchorMacro =
        anchor.macroCategory ||
        getMacroCategory(anchor.category || anchor.type || '', anchor.name || anchor.type);

      const anchorItem = {
        id: String(anchor.id),
        name: anchor.name || anchor.type || 'Item',
        image: (anchor.image || (anchor as any).imageUrl) as string | number,
        color: anchor.color,
        type: anchor.type,
        macroCategory: anchorMacro,
        brand: anchor.brand,
        isShopItem: anchor.isShopItem ?? false,
        price: anchor.price,
      };

      return outfitsToPatch.map((outfit) => {
        const alreadyHas = outfit.items.some((i) => String(i.id) === String(anchor.id));
        if (alreadyHas) return outfit;

        const matchIdx = outfit.items.findIndex((i) => {
          const m = i.macroCategory || getMacroCategory(i.type || '', i.name);
          return m === anchorMacro;
        });

        const patchedItems = [...outfit.items];
        if (matchIdx >= 0) {
          patchedItems.splice(matchIdx, 1, anchorItem);
        } else {
          patchedItems.unshift(anchorItem);
        }

        const anchorFirst = [
          anchorItem,
          ...patchedItems.filter((i) => String(i.id) !== String(anchor.id)),
        ].slice(0, 5);

        return {
          ...outfit,
          items: anchorFirst,
          mainImage:
            (typeof anchorItem.image === 'string' && anchorItem.image) ||
            outfit.mainImage,
        };
      });
    },
    [baseItemId, wardrobeItems]
  );

  /**
   * Walks a list of generated outfits and, for each outfit with missing
   * slots (e.g. layered style but user has no outerwear), fetches 1
   * style-matching shop item per missing slot and injects it as a
   * shop item in the outfit. Never throws — if shop lookup fails the
   * outfit is returned unchanged.
   */
  const autoFillMissingSlotsForOutfits = useCallback(
    async (outfits: GeneratedOutfit[], style: string): Promise<GeneratedOutfit[]> => {
      const anyMissing = outfits.some(
        o => Array.isArray(o.missingSlots) && o.missingSlots.length > 0,
      );
      if (!anyMissing) return outfits;

      return Promise.all(
        outfits.map(async (outfit) => {
          const slots = (outfit.missingSlots || []) as OutfitSlotId[];
          if (slots.length === 0) return outfit;
          try {
            const fills = await fillMissingSlots(slots, style);
            if (fills.length === 0) return outfit;
            const appended = [
              ...outfit.items,
              ...fills.map(f => ({
                id: f.id,
                name: f.name,
                image: f.image,
                type: f.type,
                macroCategory: f.macroCategory,
                color: f.color,
                brand: f.brand,
                price: f.price,
                shopUrl: f.shopUrl,
                isShopItem: true as const,
              })),
            ];
            const filledSlotIds = new Set(fills.map(f => f.missingSlot));
            const remainingMissing = slots.filter(s => !filledSlotIds.has(s));
            return {
              ...outfit,
              items: appended,
              shopItemCount: (outfit.shopItemCount || 0) + fills.length,
              missingSlots: remainingMissing,
            };
          } catch (err) {
            console.warn('[useOutfitGeneration] autoFillMissingSlotsForOutfits error:', err);
            return outfit;
          }
        }),
      );
    },
    [],
  );

  const generateOutfits = async (
    overrideStyle?: string,
    activeMode: 'auto' | 'manual' = 'auto',
    selectedItemIds: Set<string> = new Set(),
  ) => {
    const styleToUse = overrideStyle || selectedStyle;

    setLoading(true);
    setError('');
    setOutfits([]);
    setOutfitVisuals({});

    // Shop source: send shop items to the AI for intelligent outfit composition.
    // We pre-score every item against the requested style, drop items that
    // strongly clash (e.g. graphic hoodies when the user asked for Old Money),
    // and forward rich metadata (name, description, inferred styleTags,
    // materials, patterns) so the LLM can actually reason about fit.
    if (source === 'shop') {
      const baseItems = activeMode === 'manual' && selectedItemIds.size > 0
        ? wardrobeItems.filter(i => selectedItemIds.has(i.id))
        : wardrobeItems;

      if (baseItems.length === 0) {
        setError('No shop items available. Browse the shop to add items.');
        setLoading(false);
        return;
      }

      const normalizedStyle = normalizeStyleId(styleToUse);

      // In manual mode respect the user's picks verbatim; in auto mode
      // rank + filter by style so the candidate pool is on-aesthetic.
      const itemsForAI = activeMode === 'manual'
        ? baseItems
        : rankItemsForStyle(
            baseItems.map(i => ({
              ...i,
              name: i.name,
              description: (i as any).description,
            })),
            normalizedStyle,
            { minKeep: 16, dropThreshold: -2, perCategoryFloor: 4 },
          ).slice(0, 40);

      try {
        const shopPayload = itemsForAI.map(item => {
          const attrs = inferItemAttributes({
            name: item.name,
            description: (item as any).description,
            brand: item.brand,
            color: item.color,
            type: item.type,
            category: item.category,
            macroCategory: item.macroCategory,
          });
          return {
            id: item.id,
            name: item.name || item.type || 'Item',
            description: (item as any).description || '',
            type: item.type || 'clothing',
            category: item.category || 'Other',
            macroCategory: item.macroCategory || 'other',
            color: item.color || 'neutral',
            brand: item.brand || '',
            material: attrs.materials.join(', '),
            pattern: attrs.patterns.join(', '),
            styleTags: attrs.styleTags,
            formality: attrs.formality,
            imageUrl: typeof item.image === 'string' ? item.image : '',
            isShopItem: true,
            price: item.price,
          };
        });

        let shopPromptForAI = promptText.trim() || '';
        if (baseItemId) {
          const anchor = wardrobeItems.find(w => String(w.id) === String(baseItemId));
          if (anchor) {
            const anchorDesc = [anchor.color, anchor.name || anchor.type].filter(Boolean).join(' ');
            const hint = `ANCHOR ITEM: Build every outfit around item id="${anchor.id}" (${anchorDesc || 'the selected piece'}). Include this exact id in every outfit.`;
            shopPromptForAI = shopPromptForAI ? `${hint}\n\nUser request: ${shopPromptForAI}` : hint;
          }
        }
        const { data, error: fnErr } = await supabase.functions.invoke('generate-outfits', {
          body: {
            prompt: shopPromptForAI,
            stylePreferences: styleToUse,
            occasion: 'Everyday',
            limit: promptText.trim() ? 5 : 3,
            wardrobeItems: shopPayload,
          },
        });

        if (!fnErr && data?.success && Array.isArray(data.outfits) && data.outfits.length > 0) {
          const hasItems = data.outfits.some((o: any) => Array.isArray(o.items) && o.items.length > 0);
          if (hasItems) {
            const layeredForShop = Boolean(data.layered) || needsLayering(styleToUse, null, shopPromptForAI);
            const mappedOutfits: GeneratedOutfit[] = data.outfits.map((o: any) => {
              const rawItems = (o.items || []).map((item: any) => {
                const src = itemsForAI.find(w => w.id === item.id);
                const resolvedType = src?.type || item.type || 'clothing';
                const resolvedName = src?.name || item.name || resolvedType || 'Item';
                const resolvedMacroCategory =
                  src?.macroCategory ||
                  item.macroCategory ||
                  getMacroCategory(src?.category || resolvedType, resolvedName);
                return {
                  id: item.id,
                  name: resolvedName,
                  image: item.imageUrl || src?.image || '',
                  type: resolvedType,
                  macroCategory: resolvedMacroCategory,
                  color: item.color || src?.color || 'neutral',
                  isShopItem: true,
                  price: src?.price || item.price,
                  brand: item.brand || src?.brand || '',
                };
              });
              const { items: sanitizedItems, missingSlots } = sanitizeGeneratedOutfitItemsDetailed(
                rawItems,
                itemsForAI,
                { maxItems: 5, style: styleToUse, layered: layeredForShop, prompt: shopPromptForAI },
              );
              return {
                id: o.id || `shop_ai_${Date.now()}_${Math.random()}`,
                mainImage: sanitizedItems[0]?.image || o.items?.[0]?.imageUrl || o.items?.[0]?.image,
                matchScore: o.confidence ?? o.matchScore ?? 0.88,
                description: o.description || `A ${styleToUse.replace('_', ' ')} look from our shop collection.`,
                items: sanitizedItems,
                stylingTips: Array.isArray(o.stylingTips) ? o.stylingTips : ['Style to your preference'],
                wardrobeItemCount: 0,
                shopItemCount: sanitizedItems.length,
                missingSlots,
                layered: layeredForShop,
              };
            });
            const filled = await autoFillMissingSlotsForOutfits(mappedOutfits, styleToUse);
            setOutfits(enforceAnchorInOutfits(filled));
            setLoading(false);
            return;
          }
        }
        console.warn('[useOutfitGeneration] Shop AI failed, using local fallback', fnErr || data?.error);
      } catch (shopAiErr) {
        console.warn('[useOutfitGeneration] Shop AI call error, falling back:', shopAiErr);
      }

      // Local fallback for shop mode
      const tops = itemsForAI.filter(i => i.macroCategory === 'top' || i.macroCategory === 'outerwear' || i.category === 'tops');
      const bottoms = itemsForAI.filter(i => i.macroCategory === 'pants' || i.macroCategory === 'bottom' || i.category === 'bottoms');
      const shoes = itemsForAI.filter(i => i.macroCategory === 'shoes' || i.category === 'shoes');
      const fallbackLimit = promptText.trim() ? 5 : 3;
      const fallbackOutfits: GeneratedOutfit[] = [];
      for (let i = 0; i < fallbackLimit; i++) {
        const outfitItems = [];
        const top = tops[i % Math.max(tops.length, 1)];
        const bottom = bottoms[(i + 1) % Math.max(bottoms.length, 1)];
        const shoe = shoes[(i + 2) % Math.max(shoes.length, 1)];
        if (top) outfitItems.push(top);
        if (bottom) outfitItems.push(bottom);
        if (shoe) outfitItems.push(shoe);
        if (outfitItems.length > 0) {
          fallbackOutfits.push({
            id: `shop_local_${i}_${Date.now()}`,
            mainImage: outfitItems[0]?.image,
            matchScore: 0.75,
            description: `A ${styleToUse.replace('_', ' ')} look from the shop collection.`,
            items: outfitItems.map(item => ({
              id: item.id, name: item.name, image: item.image, type: item.type,
              macroCategory: item.macroCategory,
              color: item.color, isShopItem: true, price: item.price, brand: item.brand,
            })),
            stylingTips: [`Mix and match for a ${styleToUse.replace('_', ' ')} vibe.`],
            wardrobeItemCount: 0,
            shopItemCount: outfitItems.length,
          });
        }
      }
      setOutfits(enforceAnchorInOutfits(fallbackOutfits));
      setLoading(false);
      return;
    }

    // Minimum wardrobe requirement. For non-layered styles we still need
    // 3 tops / 3 pants / 3 shoes. For layered styles (old_money, business_casual,
    // streetwear or cold weather) we track base-top vs outerwear separately —
    // a missing layer is NOT a blocker; the generator will auto-suggest a
    // shop item to fill the gap.
    const layeredForGen = needsLayering(styleToUse, undefined, promptText);
    const MIN_REQUIRED = baseItemId ? 1 : 3;
    const countBaseTops = wardrobeItems.filter(i => {
      const mc = i.macroCategory || getMacroCategory(i.type || '');
      return mc === 'top';
    }).length;
    const countOuterwear = wardrobeItems.filter(i => {
      const mc = i.macroCategory || getMacroCategory(i.type || '');
      return mc === 'outerwear';
    }).length;
    const countTops = countBaseTops + countOuterwear;
    const countPants = wardrobeItems.filter(i => {
      const mc = i.macroCategory || getMacroCategory(i.type || '');
      return mc === 'bottom';
    }).length;
    const countShoes = wardrobeItems.filter(i => {
      const mc = i.macroCategory || getMacroCategory(i.type || '');
      return mc === 'shoes';
    }).length;

    // Block only when the wardrobe truly has no way to form an outfit. When
    // a required layer is missing for a layered style, we still generate and
    // fill the slot from the shop catalog downstream.
    if (countTops < MIN_REQUIRED || countPants < MIN_REQUIRED || countShoes < MIN_REQUIRED) {
      const missing: string[] = [];
      if (countTops < MIN_REQUIRED) missing.push(`${MIN_REQUIRED - countTops} more top${MIN_REQUIRED - countTops > 1 ? 's' : ''}`);
      if (countPants < MIN_REQUIRED) missing.push(`${MIN_REQUIRED - countPants} more pant${MIN_REQUIRED - countPants > 1 ? 's' : ''}`);
      if (countShoes < MIN_REQUIRED) missing.push(`${MIN_REQUIRED - countShoes} more shoe${MIN_REQUIRED - countShoes > 1 ? 's' : ''}`);
      setError(
        `Your wardrobe needs at least ${MIN_REQUIRED} top${MIN_REQUIRED > 1 ? 's' : ''}, ${MIN_REQUIRED} pant${MIN_REQUIRED > 1 ? 's' : ''}, and ${MIN_REQUIRED} shoe${MIN_REQUIRED > 1 ? 's' : ''} to generate outfits.\n\nAdd ${missing.join(', ')} to continue.`
      );
      setLoading(false);
      return;
    }

    if (layeredForGen) {
      console.log(
        `[useOutfitGeneration] layered=true style=${styleToUse} baseTops=${countBaseTops} outerwear=${countOuterwear} — shop auto-fill will cover any missing layer`,
      );
    }

    // Inject anchor item hint into the AI prompt so the model styles around the chosen piece.
    let promptForAI = promptText.trim() || undefined;
    if (baseItemId) {
      const anchor = wardrobeItems.find(w => String(w.id) === String(baseItemId));
      if (anchor) {
        const anchorDesc = [anchor.color, anchor.name || anchor.type].filter(Boolean).join(' ');
        const anchorHint = `ANCHOR ITEM: The user wants every outfit to be built around wardrobe item id="${anchor.id}" (${anchorDesc || 'the selected piece'}). You MUST include this exact id in the items array of every outfit you generate, and pick complementary items from the wardrobe that match its color, style, and material.`;
        promptForAI = promptForAI ? `${anchorHint}\n\nUser request: ${promptForAI}` : anchorHint;
      }
    }

    const selectedIds =
      activeMode === 'manual' && selectedItemIds.size > 0
        ? Array.from(selectedItemIds)
        : undefined;

    try {
      const result = await generateOutfitsFromDB({
        prompt: promptForAI,
        stylePreferences: styleToUse,
        occasion: 'Everyday',
        limit: promptText.trim() ? 5 : 3,
        selectedItemIds: selectedIds,
      });

      if (result.success && result.outfits.length > 0) {
        const wardrobeImageMap = new Map(
          wardrobeItems
            .filter(w => w.id && (w.image || (w as any).imageUrl))
            .map(w => [w.id, (w.image || (w as any).imageUrl) as string])
        );
        const layeredFromBackend = Boolean(result.layered) || layeredForGen;
        const mappedOutfits: GeneratedOutfit[] = result.outfits.map(outfit => {
          const rawItems = outfit.items.map(item => {
            const sourceItem = wardrobeItems.find(w => String(w.id) === String(item.id));
            const resolvedType = sourceItem?.type || item.type;
            const resolvedName = sourceItem?.name || item.name;
            const resolvedMacroCategory =
              sourceItem?.macroCategory ||
              item.macroCategory ||
              getMacroCategory(sourceItem?.category || resolvedType || '', resolvedName || resolvedType);
            const resolvedImage =
              (typeof item.imageUrl === 'string' && item.imageUrl) ||
              (typeof item.image === 'string' && item.image) ||
              wardrobeImageMap.get(item.id) ||
              '';
            return {
              id: item.id,
              name: resolvedName,
              image: resolvedImage,
              color: item.color,
              type: resolvedType,
              macroCategory: resolvedMacroCategory,
              isShopItem: item.isShopItem,
              price: item.price,
              brand: item.brand,
              shopUrl: item.shopUrl,
            };
          });
          const { items: sanitizedItems, missingSlots } = sanitizeGeneratedOutfitItemsDetailed(
            rawItems,
            wardrobeItems,
            { maxItems: 5, style: styleToUse, layered: layeredFromBackend, prompt: promptText },
          );
          return {
            id: outfit.id,
            mainImage: sanitizedItems[0]?.image || outfit.items[0]?.imageUrl || outfit.items[0]?.image,
            matchScore: outfit.matchScore,
            description: outfit.description,
            items: sanitizedItems,
            stylingTips: outfit.stylingTips,
            wardrobeItemCount: sanitizedItems.filter(i => !i.isShopItem).length,
            shopItemCount: sanitizedItems.filter(i => i.isShopItem).length,
            missingSlots,
            layered: layeredFromBackend,
          };
        });
        const filled = await autoFillMissingSlotsForOutfits(mappedOutfits, styleToUse);
        setOutfits(enforceAnchorInOutfits(filled));
      } else {
        setError(result.error || 'No outfits found. Add more items to your wardrobe!');
      }
    } catch (err: any) {
      console.error('Outfit generation error:', err);
      setError('Generation failed. Please try again.');
    }
    setLoading(false);
  };

  const styleToOccasion = (styleId?: string): string => {
    switch (styleId) {
      case 'business_casual': return 'work';
      case 'old_money': return 'formal';
      case 'streetwear':
      case 'minimalist':
      case 'y2k':
        return 'casual';
      default:
        return 'casual';
    }
  };

  const saveOutfit = async (outfit: GeneratedOutfit) => {
    const itemIds = outfit.items
      .map((item) => String(item.id || item.image))
      .filter(Boolean);
    if (itemIds.length === 0) {
      Alert.alert('Cannot Save', 'This outfit has no valid items to save.');
      return;
    }

    const occasion = styleToOccasion(selectedStyle);
    const store = useWardrobeStore.getState();
    const newOutfitId = store.addOutfit({
      userId: user?.id || 'guest',
      itemIds,
      occasion,
      generatedBy: 'ai',
      previewImageUrl: typeof outfit.mainImage === 'string' ? outfit.mainImage : undefined,
      reasoning: outfit.description,
      style: selectedStyle,
    });
    if (newOutfitId) {
      store.saveOutfit(newOutfitId);
      likeOutfit(newOutfitId, itemIds, occasion);
    }

    if (user?.id) {
      try {
        await supabase
          .from('saved_outfits')
          .insert({
            user_id: user.id,
            items: outfit.items.map((item) => ({
              id: String(item.id || item.image),
              type: item.type || 'Clothing Piece',
              image: item.image,
            })),
            date: new Date().toISOString().split('T')[0],
            occasion,
            season: 'All',
            name: `${selectedStyle} outfit`,
            caption: outfit.description,
            visibility: 'Everyone',
            is_ootd: false,
          });
      } catch (saveError) {
        console.error('Failed to sync saved outfit', saveError);
      }
    }

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    Alert.alert('Saved', 'Outfit saved to your closet.');
  };

  const addToCalendar = async (outfit: GeneratedOutfit) => {
    const dateKey = calendarDate || (() => {
      const today = new Date();
      return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    })();

    try {
      const existing = await AsyncStorage.getItem('outfitLogs');
      const logs = existing ? JSON.parse(existing) : {};
      const calendarItems = await Promise.all(
        outfit.items.slice(0, 6).map(async (item, index) => ({
          id: String(item.id || item.image || `${outfit.id}-${index}`),
          type: item.type || item.macroCategory || 'Clothing Piece',
          name: item.name || item.type || item.macroCategory || 'Clothing Piece',
          image: await resolvePersistableImageUri(item.image ?? (index === 0 ? outfit.mainImage : undefined)),
          color: item.color,
        }))
      );

      logs[dateKey] = {
        date: dateKey,
        items: calendarItems,
        occasion: styleToOccasion(selectedStyle),
      };
      await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert(
        'Added to Calendar',
        `Outfit logged for ${dateKey}.`,
        [{ text: 'OK', onPress: () => navigation.goBack() }]
      );
    } catch (err) {
      console.error('Failed to add outfit to calendar', err);
      Alert.alert('Error', 'Could not save outfit to calendar.');
    }
  };

  const clearOutfits = () => {
    setOutfits([]);
    setOutfitVisuals({});
  };

  return {
    selectedStyle,
    setSelectedStyle,
    loading,
    outfits,
    error,
    promptText,
    setPromptText,
    outfitVisuals,
    generateOutfits,
    generateAIVisual,
    saveOutfit,
    addToCalendar,
    clearOutfits,
  };
}
