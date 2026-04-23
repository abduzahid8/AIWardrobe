import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  Dimensions,
  StyleSheet,
  Animated,
  Platform,
  Modal,
  TextInput,
  Alert,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import * as Haptics from "expo-haptics";
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import { useShopCatalog } from '../hooks/useShopCatalog';
import { fillMissingSlots, type OutfitSlotId } from '../src/services/shoppingService';
import LiquidGlass2026Theme, { SpatialElevation } from '../constants/LiquidGlass2026Theme';
import * as Location from 'expo-location';
import Config from '../src/config/env';
import { createOutfitLog, type OutfitItem as CalendarOutfitItem, type OutfitLog } from '../features/calendar/types';
import AIThinkingAnimation from '../components/AIThinkingAnimation';
import { canonicalizeMacroCategory } from '../src/utils/categoryMapper';
import useWardrobeStore from '../store/wardrobeStore';
import { useTranslation } from 'react-i18next';

const { width, height } = Dimensions.get('window');

interface OutfitItem {
  id?: string;
  name: string;
  image: string;
  imageUrl?: string;
  type?: string;
  category?: string;
  macroCategory?: string;
  color?: string;
  brand?: string;
}

interface GeneratedOutfit {
  id: string;
  mainImage: string;
  matchScore: number;
  description: string;
  items: OutfitItem[];
  stylingTips?: string;
  weather?: { temp: number; condition: string; city?: string };
}

// ── Shared helpers: formal-layer + shorts rule ─────────────────────────
function isFormalLayer(item: { type?: string; name?: string; brand?: string; macroCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  const blob = `${item.type || ''} ${item.name || ''} ${item.brand || ''}`.toLowerCase();
  const macro = (item.macroCategory || '').toLowerCase();
  const isOuter = macro === 'outerwear' || /jacket|coat|blazer|vest|outerwear/.test(blob);
  if (!isOuter) return false;
  return /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/.test(blob);
}

function isShortsBottom(item: { type?: string; name?: string; macroCategory?: string; subCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  const blob = `${item.type || ''} ${item.name || ''} ${item.subCategory || ''}`.toLowerCase();
  const macro = (item.macroCategory || '').toLowerCase();
  const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt/.test(blob);
  if (!isBottom) return false;
  return /\b(shorts?|bermudas?)\b/.test(blob);
}

// Placeholder shoes item injected when no real shoes exist in the wardrobe.
const PLACEHOLDER_SHOES: OutfitItem = {
  id: 'placeholder_shoes',
  type: 'shoes',
  macroCategory: 'shoes',
  name: 'Shoes',
  image: 'placeholder_shoes',
  imageUrl: 'placeholder_shoes',
  color: 'neutral',
  brand: '',
};

const PLACEHOLDER_SHOES_IMAGE = require('../assets/images/basic_brown_loafers.png');
const PLACEHOLDER_PANTS_IMAGE = require('../assets/images/basic_brown_pants.png');
const PLACEHOLDER_TOP_IMAGE = require('../assets/images/basic_white_tshirt.png');
const PLACEHOLDER_LAYER_IMAGE = require('../assets/images/basic_zip_hoodie.png');

const getSlotImageSource = (item?: OutfitItem) => {
  if (item?.id === PLACEHOLDER_SHOES.id) return PLACEHOLDER_SHOES_IMAGE;
  const rawImage = typeof item?.image === 'string' && item.image ? item.image : item?.imageUrl;

  // Only http(s) / file / data URIs are renderable in a React Native <Image
  // source={{ uri }}>. Legacy placeholder strings like `basic_clothing_shoes`
  // / `placeholder_*` cannot be loaded and used to render as an empty tile
  // (the "shoes just space in card" bug). Map those back to a fallback
  // asset — loafers for the shoes slot, pants for bottom, etc. — so every
  // slot always shows an image.
  const isRenderableUri = typeof rawImage === 'string'
    && /^(https?:|file:|data:|asset:|content:)/i.test(rawImage);
  if (isRenderableUri) {
    return { uri: rawImage };
  }

  const macro = canonicalizeMacroCategory(item?.macroCategory || '');
  if (macro === 'shoes') return PLACEHOLDER_SHOES_IMAGE;
  if (macro === 'bottom') return PLACEHOLDER_PANTS_IMAGE;
  if (macro === 'outerwear') return PLACEHOLDER_LAYER_IMAGE;
  if (macro === 'top') return PLACEHOLDER_TOP_IMAGE;
  return PLACEHOLDER_TOP_IMAGE;
};

const getOfflineMacroCategory = (type: string, category?: string, name?: string) => {
  // First honor any canonical macroCategory alias present in either string
  // (e.g. shop_catalog's `upper_body` / `lower_body`). Falling back to
  // keyword matching afterwards.
  const aliasHit = canonicalizeMacroCategory(type) !== 'other'
    ? canonicalizeMacroCategory(type)
    : canonicalizeMacroCategory(category || '');
  if (aliasHit !== 'other') return aliasHit;

  const t = `${type || ''} ${category || ''} ${name || ''}`.toLowerCase();
  if (t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat/)) return 'outerwear';
  if (t.match(/shirt|t-shirt|tee|blouse|polo|tops?(?:\b)/)) return 'top';
  if (t.match(/pant|trouser|jeans?|bottom|shorts?|skirt|lower[_\s-]?body/)) return 'bottom';
  if (t.match(/shoe|sneaker|boot|loafer|sandal|footwear/)) return 'shoes';
  if (t.match(/dress|upper[_\s-]?body/)) return 'top';
  return 'top';
};

const aiStyles = [
  { id: 'old_money', label: 'Old Money', icon: 'diamond', desc: 'Classic, refined pieces with a subtle focus on pure luxury.' },
  { id: 'streetwear', label: 'Streetwear', icon: 'flash', desc: 'Edgy, oversized aesthetics blending comfort with high fashion.' },
  { id: 'minimalist', label: 'Minimalist', icon: 'remove', desc: 'Clean lines, neutral colors, and essential wardrobe staples.' },
  { id: 'y2k', label: 'Y2K', icon: 'sparkles', desc: 'Bold colors, nostalgic 2000s vibes, and striking accessories.' },
  { id: 'business_casual', label: 'Modern Professional', icon: 'briefcase', desc: 'Sharp, tailored looks perfect for the modern workplace.' },
];

function useDesignTokens() {
  const isDark = false;
  return {
    isDark,
    bg: LiquidGlass2026Theme.colors.background.primary,
    glass: 'rgba(255, 255, 255, 0.56)',
    glassStrong: 'rgba(255, 255, 255, 0.76)',
    heroGradient: ['#F2F7FC', '#ECF3FA', '#FAFCFF'] as readonly [string, string, string],
    panelHighlight: ['#FFFFFF', '#F6F8FA'] as readonly [string, string],
    textPrimary: LiquidGlass2026Theme.colors.text.primary,
    textSecondary: LiquidGlass2026Theme.colors.text.secondary,
    accent: '#2B5CE9',
    success: '#10B981',
    borderGlass: 'rgba(255, 255, 255, 0.5)',
  };
}

interface OutfitSlotGridProps {
  items: OutfitItem[];
  weather?: { temp: number; condition: string };
}

function OutfitSlotGrid({ items, weather }: OutfitSlotGridProps) {
  const D = useDesignTokens();

  const needsOuterwear = React.useMemo(() => {
    if (!weather) return true;
    const temp = weather.temp;
    const condition = weather.condition.toLowerCase();
    const isCold = temp < 18;
    const isRainy = /\b(rain|drizzle|shower|storm)\b/.test(condition);
    const isWindy = /\b(wind|gust|breezy)\b/.test(condition);
    return isCold || isRainy || isWindy;
  }, [weather]);

  const slotConfig = React.useMemo(() => {
    // Helper to classify items with priority. We canonicalize the item's
    // macroCategory FIRST so aliases like "upper_body" / "lower_body" /
    // "tops" / "footwear" (common when the AI mirrors shop_catalog's
    // garment_type column back) are correctly routed to layer / top / pants
    // / shoes slots. Previously these fell through every branch and the
    // item was silently dropped — producing the "layer shows but no base
    // top / pants missing / shoes empty" card the user reported.
    const classifyItem = (item: OutfitItem): string | null => {
      const type = (item.type || '').toLowerCase();
      const cat = (item.category || '').toLowerCase();
      const name = (item.name || '').toLowerCase();

      // 1. Canonical macroCategory (incl. aliases). Most trustworthy signal.
      const canonical = canonicalizeMacroCategory(item.macroCategory || '');
      if (canonical === 'outerwear') return 'layer';
      if (canonical === 'bottom') return 'pants';
      if (canonical === 'shoes') return 'shoes';
      if (canonical === 'top') return 'top';

      // 2. Fall back to the category field (may itself be an alias like
      //    "tops" / "upper_body").
      const catCanonical = canonicalizeMacroCategory(cat);
      if (catCanonical === 'outerwear') return 'layer';
      if (catCanonical === 'bottom') return 'pants';
      if (catCanonical === 'shoes') return 'shoes';
      if (catCanonical === 'top') return 'top';

      // 3. Last resort: keyword matching across type / category / name.
      const blob = `${type} ${cat} ${name}`;
      if (/\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear)\b/.test(blob)) return 'layer';
      if (/\b(pants?|trousers?|jeans?|shorts?|bottom|skirt|lower[_\s-]?body)\b/.test(blob)) return 'pants';
      if (/\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|footwear)\b/.test(blob)) return 'shoes';
      if (/\b(t-?shirt|shirt|polo|blouse|top|tee|dress|upper[_\s-]?body)\b/.test(blob)) return 'top';
      return null;
    };

    const classified = items.map(item => ({ item, type: classifyItem(item) }));
    console.log('[OutfitSlotGrid] items received:', items.length, 'classified:', classified.map(c => ({ id: c.item.id, name: c.item.name, macro: c.item.macroCategory, type: c.type })));
    console.log('[OutfitSlotGrid] needsOuterwear:', needsOuterwear, 'weather:', weather);

    const layerItems = classified.filter(c => c.type === 'layer').map(c => c.item);
    const topItems = classified.filter(c => c.type === 'top').map(c => c.item);
    const pantsItems = classified.filter(c => c.type === 'pants').map(c => c.item);
    const shoesItems = classified.filter(c => c.type === 'shoes').map(c => c.item);

    const layerItem = layerItems[0];
    const pantsItem = pantsItems[0];
    const shoesItem = shoesItems.find(item => {
      const rawImage = typeof item?.image === 'string' && item.image ? item.image : (item as any)?.imageUrl;
      return Boolean(rawImage && !/^placeholder_/i.test(String(rawImage)));
    }) || shoesItems[0];

    // Formal-layer + shorts rule: if bottom is shorts, drop a formal
    // outerwear layer from the display.
    const pantsIsShorts = pantsItem ? isShortsBottom(pantsItem) : false;
    const layerIsFormal = layerItem ? isFormalLayer(layerItem) : false;

    // Show Layer as a separate slot ONLY when we have both the outerwear AND
    // a real base top to pair with it. If the AI returned only a jacket/
    // blazer and no base shirt (e.g. when the server decided the look was
    // non-layered), we demote that single outerwear piece into the Top slot
    // so the card never renders an empty "Top" tile next to a filled
    // "Layer" tile (the exact bug in the user's screenshot).
    const canSplitLayerAndTop = needsOuterwear
      && !!layerItem
      && topItems.length > 0
      && !(pantsIsShorts && layerIsFormal);

    const slots: Array<{ key: string; label: string; item: OutfitItem | undefined }> = [];

    // ── Always produce exactly 4 slots for the 2×2 grid ────────────
    // 1. Main-Top / Layer (outerwear, or first top if no outerwear)
    // 2. Second-Top (base shirt/tee underneath the layer)
    // 3. Pants
    // 4. Shoes

    // Slot 1: Main-Top (Layer)
    const mainTopItem = canSplitLayerAndTop
      ? layerItem
      : (topItems[0] || layerItem);
    slots.push({ key: 'main-top', label: 'Main Top', item: mainTopItem });

    // Slot 2: Second-Top (base shirt/tee)
    const secondTopItem = canSplitLayerAndTop
      ? (topItems[0] || topItems[1])
      : (topItems[1] || topItems[0]);
    // Clone if same as mainTop to avoid duplicate id
    const resolvedSecondTop = secondTopItem && secondTopItem.id !== mainTopItem?.id
      ? secondTopItem
      : (mainTopItem ? { ...mainTopItem, id: `${mainTopItem.id || mainTopItem.name}_second_top` } : undefined);
    slots.push({ key: 'second-top', label: 'Second Top', item: resolvedSecondTop });

    // Slot 3: Pants — always show. If no bottom was classified, try
    // to find any unclassified item that could be pants, else use placeholder.
    const resolvedPants = pantsItem
      || classified.find(c => c.type === null && /\b(pant|trouser|jeans|short|skirt|bottom)\b/i.test(`${c.item.name} ${c.item.type} ${c.item.category}`))?.item;
    const pantsLabel = (resolvedPants ? isShortsBottom(resolvedPants) : false) ? 'Shorts' : 'Pants';
    slots.push({ key: 'pants', label: pantsLabel, item: resolvedPants });

    // Shoes: always show a shoes slot with a renderable image. If the
    // classified shoes item has no real image, use the placeholder.
    const resolvedShoes = shoesItem || PLACEHOLDER_SHOES;
    slots.push({ key: 'shoes', label: 'Shoes', item: resolvedShoes });

    console.log('[OutfitSlotGrid] FINAL SLOTS:', slots.map(s => ({ key: s.key, label: s.label, itemId: s.item?.id, itemName: s.item?.name, itemMacro: s.item?.macroCategory })));
    return slots;
  }, [items, needsOuterwear]);

  return (
    <View style={slotStyles.grid}>
      {slotConfig.map((slot, idx) => (
        <View key={slot.key} style={slotStyles.cell}>
          <Image
            source={getSlotImageSource(slot.item)}
            style={slotStyles.image}
            resizeMode="cover"
          />
          <View style={slotStyles.labelWrap}>
            <Text style={[slotStyles.label, { color: D.textSecondary }]}>{slot.label}</Text>
            <Text style={[slotStyles.name, { color: D.textPrimary }]} numberOfLines={1}>{slot.item?.name || 'Item'}</Text>
          </View>
        </View>
      ))}
    </View>
  );
}

// ── 100% OFFLINE OUTFIT GENERATOR (No Edge Function Required) ─────────────
async function generateOfflineOutfits(
  items: any[],
  extraItems: any[],
  style: string,
  weather: any,
  limit: number
): Promise<GeneratedOutfit[]> {
  // ── Fill missing macro-category slots from shop_catalog ──────────────
  // If the wardrobe has no shoes (or no outerwear/top/bottom), pull
  // matching items from the shop catalog so the outfit builder can
  // produce a complete outfit.
  const availableItems = [...items, ...(extraItems || [])];
  const itemMacros = new Set(availableItems.map((it: any) => (
    (it.macroCategory || getOfflineMacroCategory(it.type || '', it.category || '', it.name || '')).toLowerCase()
  )));
  const missingSlots: OutfitSlotId[] = [];
  if (!itemMacros.has('shoes')) missingSlots.push('shoes');
  if (!itemMacros.has('bottom')) missingSlots.push('bottom');
  if (!itemMacros.has('top')) missingSlots.push('top');
  const needsLayerFill = !weather || weather.temp < 18 || /\b(cold|rain|wind)\b/.test((weather?.condition || '').toLowerCase());
  if (needsLayerFill && !itemMacros.has('outerwear')) missingSlots.push('outerwear');
  if (needsLayerFill && availableItems.filter((it: any) => getOfflineMacroCategory(it.type || '', it.category || '', it.name || '') === 'top').length < 2) {
    missingSlots.push('top');
  }
  let shopFills: any[] = [];
  if (missingSlots.length > 0) {
    try {
      const fills = await fillMissingSlots(missingSlots, style);
      shopFills = fills.map(sf => ({
        ...sf,
        category: sf.macroCategory,
        image: sf.image || sf.macroCategory,
      }));
    } catch (_) {
      // Shop catalog unreachable — continue with wardrobe only.
    }
  }
  const allItems = [...availableItems, ...shopFills];

  const classified = allItems.map(item => {
    const type = (item.type || '').toLowerCase();
    const cat = (item.category || '').toLowerCase();
    const name = (item.name || '').toLowerCase();
    // Canonicalize macroCategory / category so aliases like upper_body /
    // lower_body / tops / footwear route to the correct slot instead of
    // falling through and being dropped.
    const canonical = canonicalizeMacroCategory(item.macroCategory || '');
    const catCanonical = canonical !== 'other' ? canonical : canonicalizeMacroCategory(cat);

    let category: string | null = null;
    if (catCanonical === 'outerwear' || /\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear)\b/.test(`${type} ${cat} ${name}`)) {
      category = 'layer';
    } else if (catCanonical === 'bottom' || /\b(pants?|trousers?|jeans?|shorts?|skirt|lower[_\s-]?body|bottom)\b/.test(`${type} ${cat} ${name}`)) {
      category = 'pants';
    } else if (catCanonical === 'shoes' || /\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|footwear)\b/.test(`${type} ${cat} ${name}`)) {
      category = 'shoes';
    } else if (catCanonical === 'top' || /\b(t-?shirt|shirt|polo|blouse|tee|top|dress|upper[_\s-]?body)\b/.test(`${type} ${cat} ${name}`)) {
      category = 'top';
    }
    return { item, category };
  });

  const layers = classified.filter(c => c.category === 'layer').map(c => c.item);
  const tops = classified.filter(c => c.category === 'top').map(c => c.item);
  const pants = classified.filter(c => c.category === 'pants').map(c => c.item);
  const shoes = classified.filter(c => c.category === 'shoes').map(c => c.item);

  // Determine if layering needed based on weather ONLY
  const needsLayer = !weather || weather.temp < 18 || /\b(cold|rain|wind)\b/.test((weather?.condition || '').toLowerCase());

  const casualLayers = layers.filter(l => !isFormalLayer(l));
  const nonShortsPants = pants.filter(p => !isShortsBottom(p));

  // For old_money, classic, and business_casual styles, completely exclude shorts
  // when formal outerwear is available. This prevents the illogical coat + shorts combination.
  const isFormalStyle = ['old_money', 'classic', 'business_casual'].includes(style);
  if (isFormalStyle && layers.some(l => isFormalLayer(l))) {
    // Filter out shorts entirely for formal styles when formal outerwear exists
    pants.length = 0;
    pants.push(...nonShortsPants);
  }

  const styleName = style.replace(/_/g, ' ');
  const descriptions = [
    `A curated ${styleName} look styled from your wardrobe.`,
    `An elegant ${styleName} ensemble with balanced proportions.`,
    `A refined ${styleName} outfit perfect for the occasion.`,
  ];
  const tips = [
    ['Pair with subtle accessories', 'Keep colors tonal for cohesion'],
    ['Add a leather belt and watch', 'Layer outerwear for depth'],
    ['Choose minimal jewelry', 'Match shoes to belt color'],
  ];

  const outfits: GeneratedOutfit[] = [];
  const targetItems = 4;

  for (let i = 0; i < limit; i++) {
    const outfitItems: any[] = [];

    const candidatePant = pants[i % Math.max(pants.length, 1)] || pants[0];
    const pantIsShorts = !!candidatePant && isShortsBottom(candidatePant);

    // 4-slot model: [outerwear/layer, baseTop, pants, shoes]
    // Slot 1: outerwear (main-top / layer)
    if (needsLayer && layers.length > 0) {
      let layer = layers[i % layers.length];
      if (pantIsShorts && isFormalLayer(layer)) {
        layer = casualLayers[i % Math.max(casualLayers.length, 1)] || layer;
      }
      outfitItems.push({ ...layer, macroCategory: 'outerwear' });
    }

    // Slot 2: base top (second-top / shirt / tee)
    if (tops.length > 0) {
      const top = tops[i % tops.length];
      outfitItems.push({ ...top, macroCategory: 'top' });
    }
    // When layering, ensure we have a second base top (clone if only 1 top)
    if (needsLayer && outfitItems.some(item => item.macroCategory === 'outerwear')) {
      const existingTop = outfitItems.find(item => item.macroCategory === 'top');
      const secondTop = tops.find((top) => top && top.id !== existingTop?.id)
        || tops[(i + 1) % Math.max(tops.length, 1)];
      if (secondTop && secondTop.id !== existingTop?.id) {
        // Replace the single top with a different second top
        const topIdx = outfitItems.findIndex(item => item.macroCategory === 'top');
        if (topIdx >= 0) outfitItems[topIdx] = { ...secondTop, macroCategory: 'top' };
        // Keep the first top as-is (already pushed)
      }
      // If only 1 top available, clone it for the second slot
      if (outfitItems.filter(item => item.macroCategory === 'top').length < 2 && existingTop) {
        outfitItems.push({ ...existingTop, id: `${existingTop.id || existingTop.name}_layered_copy`, macroCategory: 'top' });
      }
    }

    let finalPant = candidatePant;
    if (needsLayer && pantIsShorts && outfitItems[0] && isFormalLayer(outfitItems[0]) && nonShortsPants.length > 0) {
      finalPant = nonShortsPants[i % nonShortsPants.length];
    }
    if (finalPant) {
      outfitItems.push({ ...finalPant, macroCategory: 'bottom' });
    }

    if (shoes.length > 0) {
      const shoe = shoes[i % shoes.length];
      outfitItems.push({ ...shoe, macroCategory: 'shoes' });
    } else {
      outfitItems.push({ ...PLACEHOLDER_SHOES });
    }

    // Enforce exactly 4 items for the 2x2 grid
    // Priority: outerwear, top, bottom, shoes — drop extras from the end
    if (outfitItems.length > 4) {
      // Keep: outerwear, first top, bottom, shoes — drop second top if 5
      const keep = [] as any[];
      const ow = outfitItems.find(item => item.macroCategory === 'outerwear');
      if (ow) keep.push(ow);
      const tp = outfitItems.find(item => item.macroCategory === 'top');
      if (tp) keep.push(tp);
      const bt = outfitItems.find(item => item.macroCategory === 'bottom');
      if (bt) keep.push(bt);
      const sh = outfitItems.find(item => item.macroCategory === 'shoes');
      if (sh) keep.push(sh);
      outfitItems.length = 0;
      outfitItems.push(...keep);
    }
    const topCount = outfitItems.filter(item => item.macroCategory === 'top').length;
    const hasBottom = outfitItems.some(item => item.macroCategory === 'bottom');
    const hasShoes = outfitItems.some(item => item.macroCategory === 'shoes');
    const hasLayer = !needsLayer || outfitItems.some(item => item.macroCategory === 'outerwear');
    const hasRequiredTopCount = needsLayer ? topCount >= 1 : topCount >= 1;
    if (!hasRequiredTopCount || !hasBottom || !hasShoes || !hasLayer) continue;

    outfits.push({
      id: `offline_${Date.now()}_${i}`,
      mainImage: outfitItems[0]?.image || outfitItems[0]?.imageUrl || '',
      matchScore: 0.85 + (i * 0.03),
      description: descriptions[i % 3],
      items: outfitItems,
      stylingTips: tips[i % 3].join(' · '),
      weather,
    });
  }

  if (outfits.length === 0 && allItems.length >= 3) {
    const fallbackClassified = [...allItems].map((it: any) => {
      // Canonicalize any alias (upper_body / tops / footwear / …) before
      // we run macroCategory-equality lookups below.
      const fromMacro = canonicalizeMacroCategory(it.macroCategory || '');
      const canonical = fromMacro !== 'other'
        ? fromMacro
        : getOfflineMacroCategory(it.type || '', it.category || '', it.name || '');
      return { ...it, macroCategory: String(canonical).toLowerCase() };
    });
    const fallbackOuterwear = needsLayer
      ? fallbackClassified.find((it: any) => it.macroCategory === 'outerwear' && !(isShortsBottom(fallbackClassified.find((candidate: any) => candidate.macroCategory === 'bottom')) && isFormalLayer(it)))
      : undefined;
    const fallbackTop = fallbackClassified.find((it: any) => it.macroCategory === 'top');
    const fallbackSecondTop = needsLayer
      ? (fallbackClassified.find((it: any) => it.macroCategory === 'top' && it.id !== fallbackTop?.id)
        || (fallbackTop ? { ...fallbackTop, id: `${fallbackTop.id || fallbackTop.name}_layered_copy` } : undefined))
      : undefined;
    const fallbackBottom = fallbackClassified.find((it: any) => it.macroCategory === 'bottom');
    const fallbackShoes = fallbackClassified.find((it: any) => it.macroCategory === 'shoes') || { ...PLACEHOLDER_SHOES };
    // 4-slot model: [outerwear, baseTop, bottom, shoes]
    // Merge outerwear+top into slot1=main-top, keep secondTop as slot2
    const slot1 = fallbackOuterwear || fallbackTop;
    const slot2 = fallbackOuterwear ? (fallbackTop || fallbackSecondTop) : fallbackSecondTop;
    const fallbackItems = [slot1, slot2, fallbackBottom, fallbackShoes]
      .filter(Boolean)
      .map((it: any) => ({ ...it }));
    outfits.push({
      id: `offline_${Date.now()}_0`,
      mainImage: fallbackItems[0]?.image || fallbackItems[0]?.imageUrl || allItems[0]?.image || allItems[0]?.imageUrl || '',
      matchScore: 0.8,
      description: `A ${styleName} look from your wardrobe.`,
      items: fallbackItems,
      stylingTips: 'Mix and match to find your perfect style.',
      weather,
    });
  }

  return outfits;
}

const slotStyles = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 16,
    gap: 8,
    width: '100%',
  },
  cell: {
    width: '48%', // Use percentage to ensure 2 per row
    aspectRatio: 0.6, // Make cards slightly taller (width/height ratio)
    borderRadius: LiquidGlass2026Theme.radius.md,
    backgroundColor: 'rgba(255,255,255,0.4)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
    overflow: 'hidden',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  cellFullWidth: {
    width: '100%', // Full width for single item in last row
  },
  image: {
    width: '100%',
    height: '70%', // Smaller height for more compact cards
  },
  labelWrap: {
    padding: 6,
    flex: 1,
    justifyContent: 'center',
  },
  label: {
    ...LiquidGlass2026Theme.typography.scale.labelSmall,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    fontSize: 10,
  },
  name: {
    ...LiquidGlass2026Theme.typography.scale.bodySmall,
    fontWeight: '600',
    marginTop: 1,
    fontSize: 11,
  },
});

const AIOutfitGenerator = () => {
  const navigation = useNavigation();
  const route = useRoute<any>();
  const source = route.params?.source;
  const { t } = useTranslation();
  const [selectedStyle, setSelectedStyle] = useState('old_money');
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
  const [error, setError] = useState('');
  const [userPrompt, setUserPrompt] = useState('');
  const insets = useSafeAreaInsets();

  // ── Weather State ─────────────────────────────────────────────────
  const [weather, setWeather] = useState<{ temp: number; condition: string; icon?: string; city?: string } | undefined>(undefined);

  // ── Calendar Modal State ──────────────────────────────────────────
  const [calendarVisible, setCalendarVisible] = useState(false);
  const [calendarOutfit, setCalendarOutfit] = useState<GeneratedOutfit | null>(null);
  const [calendarDate, setCalendarDate] = useState<Date>(new Date());

  const D = useDesignTokens();

  // Wardrobe Items State
  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);

  // Use the live catalog from Supabase
  const { items: liveShopCatalog, loading: shopCatalogLoading } = useShopCatalog({
    enabled: true, // we always want it ready for fallback/shop injections
  });

  // Map the new catalog format into what the outfit generator expects
  // Key fix: use the item name to distinguish outerwear (blazer, coat, jacket)
  // from base tops (shirt, t-shirt, polo) even though they share garmentType 'upper_body'.
  // Without this, ALL upper_body items become 'top' and the 4-slot layering pipeline
  // can never find an outerwear piece → outfits collapse to 3 slots.
  const liveShopMapped = React.useMemo(() => liveShopCatalog.map((item) => {
    const nameStr = (item.name || '').toLowerCase();
    const descStr = (item.description || '').toLowerCase();
    const blob = `${nameStr} ${descStr}`;

    // Determine macroCategory from name/description keywords FIRST,
    // then fall back to garmentType.
    let macroCategory: string;
    if (/\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear|trench|peacoat)\b/.test(blob)) {
      macroCategory = 'outerwear';
    } else if (item.garmentType === 'upper_body') {
      macroCategory = 'top';
    } else if (item.garmentType === 'lower_body') {
      macroCategory = 'bottom';
    } else {
      macroCategory = 'shoes';
    }

    return {
      id: item.id,
      image: item.imageUrl,
      imageUrl: item.imageUrl,
      type: macroCategory === 'outerwear' ? 'outerwear' : item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : 'shoes',
      category: macroCategory === 'outerwear' ? 'outerwear' : item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : 'shoes',
      macroCategory,
      name: item.name,
      brand: item.brand,
      description: item.description || `${item.name} by ${item.brand}`,
    };
  }), [liveShopCatalog]);

  // Prevent setState after unmount
  const isMounted = React.useRef(true);
  const fallbackTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    isMounted.current = true;
    
    // We only want to load items once the live shop catalog has responded (so we have fallbacks/shop items)
    if (!shopCatalogLoading) {
      loadWardrobeItems();
    }
    
    return () => {
      isMounted.current = false;
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
    };
  }, [shopCatalogLoading, liveShopMapped]);

  const loadWardrobeItems = async () => {
    try {
      // ── Shop mode: use ONLY live Supabase shop catalog items.
      // Real Supabase UUIDs MUST be preserved so the edge function can enrich
      // items with imageUrl via its itemMap lookup. Do NOT mix in AsyncStorage.
      if (source === 'shop') {
        if (!isMounted.current) return;
        setWardrobeItems(
          liveShopMapped.map((item: any) => ({
            ...item,
            // Preserve the real UUID from shop_catalog — never overwrite with a fake id
            type: item.type || item.category || 'Clothing Piece',
            imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
            name: item.name || item.type || 'Clothing item',
            description: item.description || item.name || '',
          }))
        );
        return;
      }

      // ── Wardrobe mode: prefer Supabase clothing_items, fall back to AsyncStorage ──
      let items: any[] = [];

      try {
        const { data: sessionData } = await supabase.auth.getSession();
        const userId = sessionData?.session?.user?.id;

        if (userId) {
          const { data: remoteItems, error } = await supabase
            .from('clothing_items')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false });

          if (error) throw error;

          items = (remoteItems || []).map((item: any) => ({
            id: item.id,
            type: item.type || item.sub_category || item.category || 'Clothing Piece',
            category: item.category || item.type || 'tops',
            subCategory: item.sub_category || undefined,
            color: Array.isArray(item.color) ? item.color[0] : (item.primary_color || 'neutral'),
            brand: item.brand || '',
            name: item.name || item.type || 'Clothing item',
            description: item.description || item.name || '',
            image: item.image_url || '',
            imageUrl: item.image_url || '',
          }));
        }
      } catch (remoteError) {
        console.warn('[AIOutfitmaker] Failed to load Supabase wardrobe, falling back to local storage:', remoteError);
      }

      if (items.length === 0) {
        const data = await AsyncStorage.getItem('myWardrobeItems');
        items = data ? JSON.parse(data) : [];
      }

      // Normalise image/imageUrl so both fields are always populated
      items = items.map((i: any) => ({
        ...i,
        image: i.image || i.imageUrl,
        imageUrl: i.imageUrl || (typeof i.image === 'string' ? i.image : undefined),
      }));

      // Drop items with no usable image
      items = items.filter((i: any) => i && (i.imageUrl || typeof i.image === 'string'));

      // Only inject shop items when NOT in wardrobe-only mode.
      // When source is 'wardrobe', outfits must be created from the user's
      // own clothing only — no shop catalog items.
      if (source !== 'wardrobe') {
        const existingIds = new Set(items.map((i: any) => i.id));
        const newShopItems = liveShopMapped.filter(s => !existingIds.has(s.id));
        items = [...items, ...newShopItems];
      }

      if (!isMounted.current) return;
      setWardrobeItems(items.map((item: any, index: number) => ({
        ...item,
        // For personal wardrobe items that have no real id, generate a stable key.
        // For shop items merged here, preserve their real UUID.
        id: item.id || `local_item_${index}_${item.type || item.category || 'unknown'}`,
        type: item.type || item.category || 'Clothing Piece',
        imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : undefined) || '',
        name: item.name || item.type || 'Clothing item',
        description: item.description || item.name || '',
      })));
    } catch (e) {
      console.error('Failed to load wardrobe', e);
    }
  };

  // Backend-compatible macroCategory matching the edge function's slot model:
  // 'top' (base layer), 'outerwear' (jacket/blazer/sweater), 'bottom', 'shoes'
  const getBackendMacroCategory = (type: string, category?: string, name?: string) => {
    // Prefer canonical aliases (upper_body / lower_body / tops / footwear / …)
    // so items coming back from shop_catalog are slotted correctly.
    const aliasHit = canonicalizeMacroCategory(type) !== 'other'
      ? canonicalizeMacroCategory(type)
      : canonicalizeMacroCategory(category || '');
    if (aliasHit !== 'other') return aliasHit;

    const t = `${type || ''} ${category || ''} ${name || ''}`.toLowerCase();
    if (t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat/)) return 'outerwear';
    if (t.match(/shirt|t-shirt|tee|blouse|polo|tops?(?:\b)/)) return 'top';
    if (t.match(/pant|trouser|jeans?|bottom|shorts?|skirt|lower[_\s-]?body/)) return 'bottom';
    if (t.match(/shoe|sneaker|boot|loafer|sandal|footwear/)) return 'shoes';
    if (t.match(/dress|upper[_\s-]?body/)) return 'top';
    return 'top'; // fallback keeps the backend happy
  };

  // ── Weather Fetch ────────────────────────────────────────────────
  const fetchWeather = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return;
      const loc = await Location.getCurrentPositionAsync({});
      const { latitude, longitude } = loc.coords;
      const response = await fetch(
        `${Config.weather.baseUrl}/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${Config.weather.apiKey}`
      );
      const data = await response.json();
      if (data.main && data.weather) {
        setWeather({
          temp: Math.round(data.main.temp),
          condition: data.weather[0].description,
          icon: data.weather[0].icon,
          city: data.name,
        });
      }
    } catch (e) {
      console.warn('Weather fetch error', e);
    }
  };

  useEffect(() => {
    fetchWeather();
  }, []);

  const normalizeTo4Slots = (items: OutfitItem[], styleId?: string): OutfitItem[] => {
    const pool = source === 'shop'
      ? liveShopMapped
      : source === 'wardrobe'
        ? wardrobeItems
        : [...wardrobeItems, ...liveShopMapped];
    // Helper: prefer an item's own macroCategory (canonicalized) over the
    // keyword-based fallback. Keeps items whose type/category are raw
    // garmentType strings (`upper_body`, `lower_body`) from being dropped.
    const resolveMacroCategory = (it: any): string => {
      const ownCanonical = canonicalizeMacroCategory(it?.macroCategory || '');
      if (ownCanonical !== 'other') return ownCanonical;
      return getBackendMacroCategory(it?.type || '', it?.category || '', it?.name || '');
    };

    const candidatePool = pool.map((it: any) => ({
      id: it.id || it.imageUrl || it.name,
      name: it.name || it.type || 'Item',
      image: (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '',
      imageUrl: it.imageUrl || (typeof it.image === 'string' ? it.image : ''),
      type: it.type || it.category || 'top',
      macroCategory: resolveMacroCategory(it),
      color: it.color || 'neutral',
      brand: it.brand || '',
    }));

    const resolveImage = (it: any) => (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '';

    const incoming = (items || []).map((it) => ({
      ...it,
      image: it.image || it.imageUrl || '',
      macroCategory: resolveMacroCategory(it),
    }));
    // Align with `needsOuterwear` in OutfitSlotGrid + server defaults: when
    // weather is unavailable (Location denied) default to layered=TRUE, so
    // the outfit pipeline always expects a base top + outerwear. This
    // avoids the "layer shown but no base top" visual bug when weather is
    // missing.
    const needsLayer = !weather
      || (weather?.temp != null && weather.temp < 18)
      || /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test((weather?.condition || '').toLowerCase());
    const usedIds = new Set<string>();

    console.log('[normalizeTo4Slots] incoming items:', incoming.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory, image: (it.image || '').substring(0, 40) })));
    console.log('[normalizeTo4Slots] candidatePool size:', candidatePool.length, 'sample:', candidatePool.slice(0, 5).map((it: any) => ({ id: it.id, macro: it.macroCategory })));
    console.log('[normalizeTo4Slots] needsLayer:', needsLayer, 'weather:', weather);

    const pickCandidate = (predicate: (candidate: any) => boolean) => {
      const found = [...incoming, ...candidatePool].find((candidate: any) => {
        const key = String(candidate.id || '');
        return !usedIds.has(key) && predicate(candidate);
      });
      if (!found) return undefined;
      usedIds.add(String(found.id || ''));
      return { ...found, image: resolveImage(found) || found.image || '' };
    };

    // Prefer candidates whose image is a real URI (http / file / data / asset
    // / content). This lets a real wardrobe or shop_catalog item override
    // any server-side placeholder item that shares the same macroCategory
    // but only carries a legacy `basic_clothing_*` / empty string.
    const hasRealImage = (candidate: any): boolean => {
      const raw = typeof candidate?.image === 'string' && candidate.image
        ? candidate.image
        : candidate?.imageUrl;
      return typeof raw === 'string' && /^(https?:|file:|data:|asset:|content:)/i.test(raw);
    };

    const slotPantsRaw =
      pickCandidate((candidate: any) => candidate.macroCategory === 'bottom' && hasRealImage(candidate))
      || pickCandidate((candidate: any) => candidate.macroCategory === 'bottom')
      || pickCandidate((candidate: any) => candidate.macroCategory !== 'shoes' && candidate.macroCategory !== 'top' && candidate.macroCategory !== 'outerwear' && candidate.macroCategory !== 'other');
    // Force macroCategory to 'bottom' so OutfitSlotGrid.classifyItem never drops it
    const slotPants = slotPantsRaw ? { ...slotPantsRaw, macroCategory: 'bottom' as const } : undefined;
    const pantsIsShorts = slotPants ? isShortsBottom(slotPants) : false;

    const slotOuterwearRaw = needsLayer
      ? (pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)))
        || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !(pantsIsShorts && isFormalLayer(candidate)))
        || (!pantsIsShorts ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear') : undefined))
      : undefined;
    // Force macroCategory to 'outerwear' so OutfitSlotGrid.classifyItem never drops it
    const slotOuterwear = slotOuterwearRaw ? { ...slotOuterwearRaw, macroCategory: 'outerwear' as const } : undefined;

    const slotMainTopRaw =
      pickCandidate((candidate: any) => candidate.macroCategory === 'top' && hasRealImage(candidate))
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top')
      || pickCandidate((candidate: any) => candidate.macroCategory !== 'bottom' && candidate.macroCategory !== 'shoes' && candidate.macroCategory !== 'outerwear' && candidate.macroCategory !== 'other');
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotMainTop = slotMainTopRaw ? { ...slotMainTopRaw, macroCategory: 'top' as const } : undefined;

    // Pick shoes BEFORE second-top so the second-top fallback can't
    // accidentally consume a shoes item whose macroCategory resolved
    // to 'other' or an unrecognized alias.
    const slotShoesRaw =
      pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate))
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes')
      || { ...PLACEHOLDER_SHOES };
    // Force macroCategory to 'shoes' so OutfitSlotGrid.classifyItem never drops it
    const slotShoes = { ...slotShoesRaw, macroCategory: 'shoes' as const };

    const slotSecondTopRaw = needsLayer && slotOuterwear
      ? (pickCandidate((candidate: any) => candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate))
        || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id)
        || (slotMainTop ? { ...slotMainTop, id: `${slotMainTop.id || slotMainTop.name}_layered_copy` } : undefined))
      : undefined;
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotSecondTop = slotSecondTopRaw ? { ...slotSecondTopRaw, macroCategory: 'top' as const } : undefined;

    console.log('[normalizeTo4Slots] PICKS:', {
      slotOuterwear: slotOuterwear ? { id: slotOuterwear.id, name: slotOuterwear.name, macro: slotOuterwear.macroCategory, image: (slotOuterwear.image || '').substring(0, 40) } : null,
      slotMainTop: slotMainTop ? { id: slotMainTop.id, name: slotMainTop.name, macro: slotMainTop.macroCategory, image: (slotMainTop.image || '').substring(0, 40) } : null,
      slotSecondTop: slotSecondTop ? { id: slotSecondTop.id, name: slotSecondTop.name, macro: slotSecondTop.macroCategory, image: (slotSecondTop.image || '').substring(0, 40) } : null,
      slotPants: slotPants ? { id: slotPants.id, name: slotPants.name, macro: slotPants.macroCategory, image: (slotPants.image || '').substring(0, 40) } : null,
      slotShoes: slotShoes ? { id: slotShoes.id, name: slotShoes.name, macro: slotShoes.macroCategory, image: (slotShoes.image || '').substring(0, 40) } : null,
    });

    // Exactly 4 items for the 2x2 grid:
    // 1. outerwear = main-top (layer)   2. baseTop = second-top (shirt/tee)
    // 3. pants                        4. shoes
    // When no outerwear exists, mainTop fills slot 1 and secondTop fills slot 2.
    const slot1 = slotOuterwear || slotMainTop;           // main-top / layer
    const slot2 = slotMainTop && slotOuterwear             // second-top (base shirt)
      ? (slotMainTop.id !== slotOuterwear.id ? slotMainTop : slotSecondTop)
      : (slotOuterwear ? slotSecondTop : slotSecondTop || slotMainTop);
    const result = [slot1, slot2, slotPants, slotShoes].filter(Boolean) as OutfitItem[];
    console.log('[normalizeTo4Slots] RESULT (4 items):', result.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory })));
    return result;
  };

  const isCompleteNormalizedOutfit = (items: OutfitItem[]) => {
    const macros = items.map((item) => canonicalizeMacroCategory(item.macroCategory || ''));
    const topCount = macros.filter((macro) => macro === 'top').length;
    const hasBottom = macros.includes('bottom');
    const hasShoes = macros.includes('shoes');
    const hasOuterwear = macros.includes('outerwear');
    // A complete 4-slot outfit needs: top-like (top or outerwear), bottom, shoes.
    // With the new 4-item model: [outerwear|top, top, bottom, shoes]
    const hasTopLike = topCount >= 1 || hasOuterwear;
    return hasTopLike && hasBottom && hasShoes;
  };

  const getCalendarOccasion = (styleId?: string) => {
    switch (styleId) {
      case 'business_casual':
        return 'work';
      case 'old_money':
        return 'formal';
      case 'streetwear':
      case 'minimalist':
      case 'y2k':
        return 'casual';
      default:
        return 'casual';
    }
  };

  // ── Save Outfit to Closet ─────────────────────────────────────────
  const saveOutfitToCloset = async (outfit: GeneratedOutfit) => {
    const itemIds = outfit.items
      .map((item) => String(item.id || item.image))
      .filter(Boolean);
    if (itemIds.length === 0) {
      Alert.alert('Cannot Save', 'This outfit has no valid items to save.');
      return;
    }

    const store = useWardrobeStore.getState();
    const occasion = getCalendarOccasion(selectedStyle);
    store.addOutfit({
      userId: 'user',
      itemIds,
      occasion,
      generatedBy: 'ai',
      previewImageUrl: typeof outfit.mainImage === 'string' ? outfit.mainImage : undefined,
      reasoning: outfit.description,
      style: selectedStyle,
    });

    // Find the outfit we just added by matching itemIds
    const savedOutfit = useWardrobeStore.getState().outfits.find(
      (o) => o.itemIds.length === itemIds.length && o.itemIds.every((id) => itemIds.includes(id))
    );
    if (savedOutfit?.id) {
      store.saveOutfit(savedOutfit.id);
    }

    // Persist to Supabase if authenticated
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const userId = sessionData?.session?.user?.id;
      if (userId) {
        await supabase.from('saved_outfits').insert({
          user_id: userId,
          items: outfit.items.map((item) => ({
            id: String(item.id || item.image),
            type: item.type || 'Clothing Piece',
            image: item.image || item.imageUrl || '',
          })),
          date: new Date().toISOString().split('T')[0],
          occasion,
          season: 'All',
          name: `${selectedStyle} outfit`,
          caption: outfit.description,
          visibility: 'Everyone',
          is_ootd: false,
        });
      }
    } catch (saveError) {
      console.error('Failed to sync saved outfit', saveError);
    }

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    Alert.alert('Saved', 'Outfit saved to your closet.');
  };

  // ── Add to Calendar ─────────────────────────────────────────────
  const addToCalendar = async (outfit: GeneratedOutfit, date: Date) => {
    try {
      const dateStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
      const mappedItems: CalendarOutfitItem[] = outfit.items.map(it => ({
        id: it.id || it.name || `${Date.now()}`,
        type: it.type || 'top',
        image: it.image || it.imageUrl || '',
        color: it.color,
        name: it.name,
      }));
      const log = createOutfitLog(dateStr, mappedItems, getCalendarOccasion(selectedStyle));
      const raw = await AsyncStorage.getItem('outfitLogs');
      const logs: Record<string, OutfitLog> = raw ? JSON.parse(raw) : {};

      // Warn before overwriting an existing outfit for this date
      if (logs[dateStr]) {
        Alert.alert(
          'Outfit Already Exists',
          'You already have an outfit for this date. Replace it?',
          [
            { text: 'Cancel', style: 'cancel' },
            {
              text: 'Replace',
              style: 'destructive',
              onPress: async () => {
                logs[dateStr] = log;
                await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                setCalendarVisible(false);
                setCalendarOutfit(null);
              },
            },
          ]
        );
        return;
      }

      logs[dateStr] = log;
      await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setCalendarVisible(false);
      setCalendarOutfit(null);
    } catch (e) {
      console.error('Calendar save error', e);
    }
  };

  const generateOutfits = async (overrideStyle?: string) => {
    const styleToUse = overrideStyle || selectedStyle;

    if (!isMounted.current) return;
    setLoading(true);
    setError('');
    setOutfits([]);

    const selectedClothing: any[] = [];

    // Build a clean payload for the backend — strip non-serialisable fields
    // (require() numbers) and ensure every item has macroCategory + imageUrl.
    const payloadItems = wardrobeItems.map((item: any) => ({
      id: item.id,
      name: item.name || item.type || 'Clothing item',
      type: item.type || 'top',
      category: item.category || item.type || 'tops',
      color: item.color || 'neutral',
      brand: item.brand || '',
      description: item.description || item.name || '',
      imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
      macroCategory: getBackendMacroCategory(item.type || '', item.category || '', item.name || ''),
    }));

    try {
      // Generate outfits with selected items and style preferences.
      // NOTE: do NOT set useProvidedWardrobeOnly:true — the edge function's
      // enrichment step (step 8) needs to match item ids against itemMap to
      // attach real imageUrls. When wardrobeItems is provided the edge fn
      // already uses it as the candidate pool (step 3), so this is safe.
      // Fall back to a "cool" weather default when Location is denied /
      // unavailable. The server's `needsLayering()` returns FALSE when
      // weather is undefined, which makes it build 3-item non-layered
      // outfits. That's what produced the "layer + pants + shoes with no
      // base top" card in the user screenshot. Sending a cool default keeps
      // the server in layered mode so every outfit has base top + layer
      // + bottom + shoes.
      const resolvedWeather = weather ?? { temp: 15, condition: 'cool' };
      const { data, error: invokeError } = await supabase.functions.invoke('generate-outfits', {
        body: {
          stylePreferences: styleToUse,
          occasion: userPrompt || 'Any',
          selectedItemIds: [],
          wardrobeItems: payloadItems,
          weather: resolvedWeather,
          limit: 3,
          prompt: userPrompt,
        },
      });

      if (invokeError) throw invokeError;

      if (data && data.success && data.outfits && data.outfits.length > 0) {
        const cleanedOutfits = data.outfits.map((outfit: any, index: number) => {
          const normalized = normalizeTo4Slots(outfit.items || [], styleToUse);
          return {
            ...outfit,
            items: normalized,
            mainImage: normalized[0]?.image || outfit.mainImage || '',
            matchScore: outfit.confidence ?? outfit.matchScore ?? 0.9,
            stylingTips: Array.isArray(outfit.stylingTips)
              ? outfit.stylingTips.join(' · ')
              : (outfit.stylingTips || ''),
            weather,
          };
        }).filter((outfit: GeneratedOutfit) => isCompleteNormalizedOutfit(outfit.items));

        if (cleanedOutfits.length === 0) {
          throw new Error('No complete outfits returned from AI');
        }

        if (isMounted.current) {
          setOutfits(cleanedOutfits);
          setLoading(false);
        }
        return;
      }

      // Backend returned success=false or empty outfits — fall through to local fallback
      throw new Error(data?.error || 'No outfits returned from AI');
    } catch (err: any) {
      console.error('[AIOutfitmaker] Edge function failed, using offline generator:', err);

      // ENHANCED OFFLINE GENERATOR — Works 100% without internet/edge function
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
      fallbackTimer.current = setTimeout(async () => {
        if (!isMounted.current) return;

        const offlineOutfits = await generateOfflineOutfits(wardrobeItems, liveShopMapped, styleToUse, weather, 3);
        setOutfits(offlineOutfits);
        setLoading(false);
        setError('');
      }, 600);
    }
  };

  if (outfits.length > 0) {
    return (
      <View style={styles.container}>
        <LinearGradient colors={D.heroGradient} style={StyleSheet.absoluteFill} />
        <View style={[styles.orbTop, { backgroundColor: 'rgba(43,92,233,0.06)', top: insets.top + 40 }]} />
        <View style={[styles.orbBottom, { backgroundColor: 'rgba(236,72,153,0.05)' }]} />

        <SafeAreaView style={{ flex: 1 }} edges={['top', 'left', 'right']}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => setOutfits([])} style={styles.backButton}>
              <Ionicons name="chevron-back" size={26} color={D.textPrimary} />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>{t('outfitMaker.yourOutfits')}</Text>
            <View style={{ width: 40 }} />
          </View>

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 32 }}>
            {outfits.map((outfit, index) => (
              <Animated.View key={outfit.id} style={[styles.glassCard, { marginTop: index === 0 ? 8 : 24 }]}>
                <BlurView intensity={Platform.OS === 'ios' ? 48 : 90} tint="light" style={StyleSheet.absoluteFill} />
                <LinearGradient colors={['rgba(255,255,255,0.45)', 'rgba(255,255,255,0.15)']} style={StyleSheet.absoluteFill} />

                <View style={styles.cardHeaderRow}>
                  {outfit.weather ? (
                    <View style={styles.weatherChip}>
                      <Ionicons name="cloud-outline" size={14} color={D.textSecondary} />
                      <Text style={styles.weatherChipText}>{outfit.weather.temp}°C · {outfit.weather.condition}</Text>
                    </View>
                  ) : <View />}
                  <LinearGradient colors={[D.accent, '#5B7CF9']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.matchPill}>
                    <Text style={styles.matchPillText}>{Math.round(outfit.matchScore * 100)}% Match</Text>
                  </LinearGradient>
                </View>

                <OutfitSlotGrid items={outfit.items} weather={outfit.weather} />

                <Text style={styles.outfitDesc}>{outfit.description}</Text>

                <View style={styles.tipCard}>
                  <LinearGradient colors={D.panelHighlight} style={StyleSheet.absoluteFill} />
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <Ionicons name="sparkles-outline" size={16} color={D.accent} />
                    <Text style={styles.tipCardText}>{outfit.stylingTips}</Text>
                  </View>
                </View>

                <View style={styles.actionRow}>
                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={styles.primaryAction}
                    onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); setCalendarOutfit(outfit); setCalendarDate(new Date()); setCalendarVisible(true); }}
                  >
                    <LinearGradient colors={[D.accent, '#5B7CF9']} style={StyleSheet.absoluteFill} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} />
                    <Ionicons name="calendar-outline" size={18} color="#fff" />
                    <Text style={styles.primaryActionText}>{t('outfitMaker.addToCalendar')}</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={styles.secondaryAction}
                    onPress={() => saveOutfitToCloset(outfit)}
                  >
                    <Ionicons name="checkmark-circle-outline" size={18} color={D.textPrimary} />
                    <Text style={styles.secondaryActionText}>{t('outfitMaker.saveOutfit')}</Text>
                  </TouchableOpacity>
                </View>
              </Animated.View>
            ))}
          </ScrollView>
        </SafeAreaView>

        <Modal animationType="slide" transparent visible={calendarVisible} onRequestClose={() => setCalendarVisible(false)}>
          <View style={styles.modalOverlay}>
            <BlurView intensity={Platform.OS === 'ios' ? 60 : 100} tint="light" style={StyleSheet.absoluteFill} />
            <View style={styles.modalCard}>
              <Text style={styles.modalTitle}>{t('outfitMaker.saveToCalendar')}</Text>
              <Text style={styles.modalSubtitle}>{t('outfitMaker.pickDate')}</Text>
              <View style={styles.datePickerRow}>
                <TouchableOpacity onPress={() => setCalendarDate(new Date(calendarDate.getTime() - 86400000))}>
                  <Ionicons name="chevron-back" size={24} color={D.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.datePickerValue}>{calendarDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</Text>
                <TouchableOpacity onPress={() => setCalendarDate(new Date(calendarDate.getTime() + 86400000))}>
                  <Ionicons name="chevron-forward" size={24} color={D.textPrimary} />
                </TouchableOpacity>
              </View>
              <View style={styles.modalActions}>
                <TouchableOpacity style={styles.modalCancel} onPress={() => setCalendarVisible(false)}>
                  <Text style={styles.modalCancelText}>{t('common.cancel')}</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.modalConfirm}
                  onPress={() => calendarOutfit && addToCalendar(calendarOutfit, calendarDate)}
                >
                  <LinearGradient colors={[D.accent, '#5B7CF9']} style={StyleSheet.absoluteFill} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} />
                  <Text style={styles.modalConfirmText}>{t('common.save')}</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </View>
    );
  }

  // Show AI thinking animation when loading
  if (loading) {
    const selectedStyleLabel = aiStyles.find(s => s.id === selectedStyle)?.label || 'AI';
    // Map wardrobe items for the animation
    const animationItems = wardrobeItems.slice(0, 24).map((item: any) => ({
      id: item.id || String(Math.random()),
      image: item.imageUrl || (typeof item.image === 'string' ? item.image : ''),
      name: item.name || item.type,
      type: item.type,
      category: item.category,
    })).filter((item: any) => item.image);
    
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={LiquidGlass2026Theme.colors.gradients.liquidGlass}
          style={StyleSheet.absoluteFill}
        />
        <SafeAreaView edges={['top', 'left', 'right']} style={{ flex: 1 }}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => { setLoading(false); setOutfits([]); }} style={styles.backButton}>
              <Ionicons name="chevron-back" size={28} color={LiquidGlass2026Theme.colors.text.primary} />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>{t('outfitMaker.aiStylist')}</Text>
            <View style={{ width: 44 }} />
          </View>
          <AIThinkingAnimation styleName={selectedStyleLabel} clothingItems={animationItems} />
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Dynamic Background */}
      <LinearGradient
        colors={LiquidGlass2026Theme.colors.gradients.liquidGlass}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView
        edges={['top', 'left', 'right']}
        style={{ flex: 1 }}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={28} color={LiquidGlass2026Theme.colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{t('outfitMaker.aiStylist')}</Text>
          <Ionicons name="sparkles" size={24} color="#F59E0B" />
        </View>

        {/* Title Area */}
        <View style={{ paddingHorizontal: 20, paddingTop: 36, paddingBottom: 24 }}>
          <Text style={styles.sectionTitle}>
            Discover Your Vibe
          </Text>
          <Text style={styles.sectionSubtitle}>
            Tap a style card below and AI will instantly build a complete look from your wardrobe.
          </Text>
        </View>

        {/* User Situation/Occasion Prompt */}
        <View style={{ paddingHorizontal: 20, marginBottom: 16 }}>
          <View style={styles.promptCard}>
            <View style={styles.promptHeader}>
              <Ionicons name="chatbubble-outline" size={18} color={D.accent} />
              <Text style={styles.promptTitle}>{t('outfitMaker.whereGoing')}</Text>
            </View>
            <TextInput
              style={styles.promptInput}
              placeholder="e.g., Date night, job interview, casual brunch, beach vacation..."
              placeholderTextColor={D.textSecondary}
              value={userPrompt}
              onChangeText={setUserPrompt}
              multiline
              numberOfLines={2}
              textAlignVertical="top"
            />
          </View>
        </View>

        {/* Error Message */}
        {error ? (
          <View style={{ paddingHorizontal: 20, marginBottom: 16 }}>
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          </View>
        ) : null}

        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20 }}>
          {aiStyles.map((styleObj) => {
            return (
              <TouchableOpacity
                key={styleObj.id}
                activeOpacity={0.8}
                onPress={() => {
                  setSelectedStyle(styleObj.id);
                  generateOutfits(styleObj.id);
                }}
                style={styles.vibeCard}
              >
                <View style={styles.vibeCardInner}>
                  <View style={styles.vibeIconWrap}>
                    <Ionicons name={styleObj.icon as any} size={24} color={'#FFFFFF'} />
                  </View>
                  <View style={{ flex: 1, marginLeft: 20 }}>
                    <Text style={styles.vibeTitle}>{styleObj.label}</Text>
                    <Text style={styles.vibeDesc}>{styleObj.desc}</Text>
                  </View>
                  <View style={{ paddingLeft: 12 }}>
                    <Ionicons name="chevron-forward" size={24} color={'#4B5563'} />
                  </View>
                </View>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      </SafeAreaView>

    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlass2026Theme.colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  backButton: {
    marginRight: 16,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  headerTitle: {
    ...LiquidGlass2026Theme.typography.scale.headlineMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    flex: 1,
  },
  sectionTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
  },
  sectionSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    marginTop: 6,
    lineHeight: 22,
  },

  // Toggle Styles (Matching CreateAvatarScreen)
  viewToggleWrap: {
    flexDirection: "row",
    alignSelf: "center",
    borderRadius: 24,
    padding: 6,
    marginTop: 24,
    marginBottom: 8,
    overflow: "hidden",
    backgroundColor: "rgba(255,255,255,0.6)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.8)",
    zIndex: 10,
    width: width - 40,
  },
  viewToggleOption: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: 18,
  },
  viewToggleActive: {
    backgroundColor: "#fff",
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  viewToggleText: {
    fontSize: 15,
    fontWeight: "600",
    color: LiquidGlass2026Theme.colors.text.secondary,
  },
  viewToggleTextActive: {
    color: LiquidGlass2026Theme.colors.text.primary,
  },

  // User Prompt Card
  promptCard: {
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    borderWidth: 1,
    borderColor: 'rgba(43, 92, 233, 0.15)',
    padding: 16,
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  promptHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  promptTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: LiquidGlass2026Theme.colors.text.primary,
  },
  promptInput: {
    fontSize: 15,
    color: LiquidGlass2026Theme.colors.text.primary,
    lineHeight: 22,
    minHeight: 56,
    padding: 0,
  },

  // Auto-Mode Vibe Cards
  vibeCard: {
    marginBottom: 16,
    borderRadius: 24,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 12,
    elevation: 2,
  },
  vibeCardInner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    paddingRight: 24,
  },
  vibeIconWrap: {
    width: 64,
    height: 64,
    borderRadius: 18,
    backgroundColor: '#0A1931', // Dark Navy
    alignItems: 'center',
    justifyContent: 'center',
  },
  vibeTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0A1931',
    marginBottom: 8,
    letterSpacing: -0.3,
  },
  vibeDesc: {
    fontSize: 14,
    color: '#4B5563',
    lineHeight: 22,
  },

  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  gridItemWrap: {
    width: '33.33%',
    padding: 6,
  },
  gridItem: {
    aspectRatio: 1,
    backgroundColor: 'rgba(255,255,255,0.8)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  gridItemActive: {
    borderColor: LiquidGlass2026Theme.colors.accent.primary,
    borderWidth: 2,
    backgroundColor: 'rgba(20, 30, 50, 0.05)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  gridItemImage: {
    width: '90%',
    height: '90%',
  },
  checkBadge: {
    position: 'absolute',
    top: 6,
    right: 6,
    backgroundColor: LiquidGlass2026Theme.colors.accent.primary,
    borderRadius: LiquidGlass2026Theme.radius.full,
    padding: 3,
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
  errorBox: {
    backgroundColor: '#FEF2F2',
    padding: 16,
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderLeftWidth: 4,
    borderLeftColor: LiquidGlass2026Theme.colors.accent.error,
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  errorText: {
    color: LiquidGlass2026Theme.colors.accent.error,
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    fontWeight: '500',
  },
  // Results styles
  outfitCard: {
    marginBottom: 32,
    backgroundColor: LiquidGlass2026Theme.colors.background.elevated,
    borderRadius: LiquidGlass2026Theme.radius.card,
    overflow: 'hidden',
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  outfitImage: {
    width: '100%',
    height: 400,
  },
  matchBadge: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  matchBadgeText: {
    color: LiquidGlass2026Theme.colors.text.onDark,
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    fontWeight: '700',
  },
  outfitDesc: {
    ...LiquidGlass2026Theme.typography.scale.bodyLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    marginBottom: 16,
  },
  itemsLabel: {
    ...LiquidGlass2026Theme.typography.scale.titleSmall,
    color: LiquidGlass2026Theme.colors.text.secondary,
    marginBottom: 12,
  },
  includedItem: { // keeping old style around just in case
    marginRight: 12,
    width: 64,
    height: 64,
    borderRadius: LiquidGlass2026Theme.radius.md,
    backgroundColor: 'rgba(255,255,255,0.5)',
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  includedItemImage: {
    width: '85%',
    height: '85%',
  },
  collageContainer: {
    flexDirection: 'column',
    alignItems: 'flex-start',
    marginBottom: 16,
    gap: 12, // RN >= 0.71 standard gap
  },
  collageRow: {
    flexDirection: 'row',
    gap: 12,
  },
  collageItem: {
    width: (width - 92) / 2, // 40px outer padding + 40px inner padding + 12px gap = 92px
    height: (width - 92) / 2,
    borderRadius: LiquidGlass2026Theme.radius.md,
    backgroundColor: 'rgba(255,255,255,0.5)',
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  collageItemImage: {
    width: '100%',
    height: '100%',
  },

  // Category section styles for Build My Own
  categorySectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 12,
    gap: 8,
  },
  categorySectionTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    flex: 1,
  },
  categorySectionBadge: {
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    backgroundColor: 'rgba(255,255,255,0.7)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    overflow: 'hidden',
  },
  categoryGridItem: {
    width: (width - 80) / 3,
    height: (width - 80) / 3,
    backgroundColor: 'rgba(255,255,255,0.8)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  categoryGridItemImage: {
    width: '90%',
    height: '90%',
  },

  stylingTipsBox: {
    backgroundColor: 'rgba(255,255,255,0.7)',
    padding: 16,
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.glass,
    marginTop: 8,
  },
  stylingTipsText: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
  },
  wishlistButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    backgroundColor: 'rgba(255,255,255,0.5)',
    gap: 8,
  },
  wishlistButtonText: {
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },

  // ── Liquid Glass Result Card Styles ──────────────────────────────
  orbTop: {
    position: 'absolute',
    width: 220,
    height: 220,
    borderRadius: 110,
    left: -60,
  },
  orbBottom: {
    position: 'absolute',
    width: 280,
    height: 280,
    borderRadius: 140,
    right: -80,
    bottom: 100,
  },
  glassCard: {
    borderRadius: LiquidGlass2026Theme.radius.card,
    overflow: 'hidden',
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  cardHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  weatherChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(255,255,255,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  weatherChipText: {
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    fontWeight: '500',
  },
  matchPill: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: LiquidGlass2026Theme.radius.pill,
  },
  matchPillText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    fontWeight: '700',
  },
  slotGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  slotCell: {
    width: (width - 72) / 2,
    marginBottom: 12,
    borderRadius: LiquidGlass2026Theme.radius.md,
    backgroundColor: 'rgba(255,255,255,0.4)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
    overflow: 'hidden',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  slotImage: {
    width: '100%',
    height: (width - 72) / 2,
  },
  slotLabelWrap: {
    padding: 10,
  },
  slotLabel: {
    ...LiquidGlass2026Theme.typography.scale.labelSmall,
    color: LiquidGlass2026Theme.colors.text.secondary,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  slotName: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
    marginTop: 2,
  },
  tipCard: {
    borderRadius: LiquidGlass2026Theme.radius.md,
    overflow: 'hidden',
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
  },
  tipCardText: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    flex: 1,
  },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  primaryAction: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
  },
  primaryActionText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '700',
  },
  secondaryAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    backgroundColor: 'rgba(255,255,255,0.55)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  secondaryActionText: {
    color: LiquidGlass2026Theme.colors.text.primary,
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '600',
  },

  // ── Calendar Modal Styles ──────────────────────────────────────
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalCard: {
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    paddingHorizontal: 24,
    paddingTop: 28,
    paddingBottom: 36,
    marginHorizontal: 0,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
  modalTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    textAlign: 'center',
    marginBottom: 4,
  },
  modalSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: 20,
  },
  datePickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255,255,255,0.6)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    paddingVertical: 14,
    paddingHorizontal: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  datePickerValue: {
    ...LiquidGlass2026Theme.typography.scale.bodyLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  modalCancel: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    backgroundColor: 'rgba(255,255,255,0.6)',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  modalCancelText: {
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  modalConfirm: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
    alignItems: 'center',
  },
  modalConfirmText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '700',
  },
});

export default AIOutfitGenerator;
