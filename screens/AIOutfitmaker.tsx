import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { View, TouchableOpacity, ScrollView, Image, Dimensions, StyleSheet, Animated, Platform, Modal, TextInput, Alert, Linking } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import * as Haptics from "expo-haptics";
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import Reanimated, { FadeIn, FadeInUp, FadeOut, FadeOutUp } from 'react-native-reanimated';
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
import { scoreItemForStyle, normalizeStyleId } from '../features/outfit-generator/utils/styleInference';
import useWardrobeStore from '../store/wardrobeStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { useTranslation } from 'react-i18next';
import useAuthStore from '../store/auth';

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
  isWardrobe?: boolean;
  isShopItem?: boolean;
  shopUrl?: string;
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

const PLACEHOLDER_PANTS: OutfitItem = {
  id: 'placeholder_pants',
  type: 'pants',
  macroCategory: 'bottom',
  name: 'Pants',
  image: 'placeholder_pants',
  imageUrl: 'placeholder_pants',
  color: 'neutral',
  brand: '',
};

const PLACEHOLDER_SHOES_IMAGE = require('../assets/images/basic_brown_loafers.png');
const PLACEHOLDER_PANTS_IMAGE = require('../assets/images/basic_brown_pants.png');
const PLACEHOLDER_TOP_IMAGE = require('../assets/images/basic_white_tshirt.png');
const PLACEHOLDER_LAYER_IMAGE = require('../assets/images/basic_zip_hoodie.png');

const getSlotImageSource = (item?: OutfitItem) => {
  if (item?.id === PLACEHOLDER_SHOES.id) return PLACEHOLDER_SHOES_IMAGE;
  if (item?.id === PLACEHOLDER_PANTS.id) return PLACEHOLDER_PANTS_IMAGE;
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
  // Outerwear keywords win over a generic alias. A blazer stored under
  // category "top"/"tops" must still resolve to "outerwear" (main-top/layer),
  // otherwise the anchor gets locked into the wrong (second-top) slot.
  const t = `${type || ''} ${category || ''} ${name || ''}`.toLowerCase();
  if (t.match(/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat|bomber)\b/)) return 'outerwear';

  // Then honor any canonical macroCategory alias present in either string
  // (e.g. shop_catalog's `upper_body` / `lower_body`). Falling back to
  // keyword matching afterwards.
  const aliasHit = canonicalizeMacroCategory(type) !== 'other'
    ? canonicalizeMacroCategory(type)
    : canonicalizeMacroCategory(category || '');
  if (aliasHit !== 'other') return aliasHit;

  if (t.match(/\b(pant|trouser|jeans?|bottom|shorts?|skirt|lower[_\s-]?body)\b/)) return 'bottom';
  if (t.match(/\b(shoe|sneaker|boot|loafer|sandal|footwear)\b/)) return 'shoes';
  if (t.match(/\b(shirt|t-shirt|tee|blouse|polo|top|dress|upper[_\s-]?body)\b/)) return 'top';
  return 'top';
};

// ── AI styles — aligned 1:1 with Style Quiz personalities ──────────────
// Quiz ids: classic / semi_classic / minimalist / casual / old_money.
// `business_casual` is kept as a legacy alias of `classic` so old saved
// outfits / deep-links keep working.
const aiStyles = [
  { id: 'old_money', label: 'Old Money', icon: 'diamond', desc: 'Classic, refined pieces with a subtle focus on pure luxury.', quizId: 'old_money' },
  { id: 'semi_classic', label: 'Semi-Classic', icon: 'umbrella', desc: 'Refined everyday elegance — tailored touches with relaxed comfort.', quizId: 'semi_classic' },
  { id: 'minimalist', label: 'Minimalist', icon: 'remove', desc: 'Clean lines, neutral colors, and essential wardrobe staples.', quizId: 'minimalist' },
  { id: 'casual', label: 'Casual', icon: 'cafe', desc: 'Relaxed, effortless style with well-fitted basics and clean combos.', quizId: 'casual' },
  { id: 'classic', label: 'Classic', icon: 'briefcase', desc: 'Sharp, tailored looks perfect for the modern workplace.', quizId: 'classic' },
  { id: 'business_casual', label: 'Modern Professional', icon: 'briefcase', desc: 'Sharp, tailored looks perfect for the modern workplace.', quizId: 'classic' },
];

/** Map a Style Quiz personality id to an AI generator style id. */
function mapQuizStyleToAiStyle(quizId?: string | null): string {
  if (!quizId) return 'old_money';
  const key = String(quizId).toLowerCase().replace(/[\s-]+/g, '_');
  if (key === 'semi-classic' || key === 'semi_classic') return 'semi_classic';
  if (key === 'old_money' || key === 'oldmoney') return 'old_money';
  if (key === 'minimalist') return 'minimalist';
  if (key === 'casual') return 'casual';
  if (key === 'classic' || key === 'business_casual') return 'classic';
  return 'old_money';
}

// ── Occasion presets for the "Where are you going?" step ───────────────
// ids match COLLECTION_CATEGORIES / calendar occasions.
const OCCASION_PRESETS = [
  { id: 'work', label: 'Work', icon: 'briefcase-outline', emoji: '💼' },
  { id: 'casual', label: 'Casual', icon: 'cafe-outline', emoji: '☕' },
  { id: 'date', label: 'Date night', icon: 'heart-outline', emoji: '💕' },
  { id: 'formal', label: 'Formal', icon: 'diamond-outline', emoji: '🤵' },
  { id: 'sport', label: 'Sport', icon: 'fitness-outline', emoji: '🏃' },
  { id: 'travel', label: 'Travel', icon: 'airplane-outline', emoji: '✈️' },
] as const;

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

const getAiStyleLabel = (id: string, t: (key: string) => string) => {
  const style = aiStyles.find(s => s.id === id);
  return style?.label || id;
};

const getAiStyleDesc = (id: string, t: (key: string) => string) => {
  const style = aiStyles.find(s => s.id === id);
  return style?.desc || '';
};

interface OutfitSlotGridProps {
  items: OutfitItem[];
  weather?: { temp: number; condition: string };
}

function OutfitSlotGrid({ items, weather }: OutfitSlotGridProps) {
  const D = useDesignTokens();
  const { t } = useTranslation();

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

    // Slot 3: Pants — always show pants, never shorts. Prefer a real
    // non-shorts bottom; if only shorts exist or nothing matches, fall
    // back to the PLACEHOLDER_PANTS tile so the user always sees pants.
    const nonShortsPantsItem = pantsItems.find(p => !isShortsBottom(p));
    const resolvedPants = nonShortsPantsItem
      || classified.find(c => c.type === null && /\b(pant|trouser|jeans|skirt|bottom)\b/i.test(`${c.item.name} ${c.item.type} ${c.item.category}`) && !/\bshorts?\b/i.test(`${c.item.name} ${c.item.type} ${c.item.category}`))?.item
      || PLACEHOLDER_PANTS;
    slots.push({ key: 'pants', label: 'Pants', item: resolvedPants });

    // Shoes: always show a shoes slot with a renderable image. If the
    // classified shoes item has no real image, use the placeholder.
    const resolvedShoes = shoesItem || PLACEHOLDER_SHOES;
    slots.push({ key: 'shoes', label: 'Shoes', item: resolvedShoes });

    console.log('[OutfitSlotGrid] FINAL SLOTS:', slots.map(s => ({ key: s.key, label: s.label, itemId: s.item?.id, itemName: s.item?.name, itemMacro: s.item?.macroCategory })));
    return slots;
  }, [items, needsOuterwear]);

  const handleBuy = useCallback(async (url?: string) => {
    if (url) {
      try { await Linking.openURL(url); } catch (_) {}
    }
  }, []);

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
            <ScaledText style={[slotStyles.label, { color: D.textSecondary }]}>{slot.label}</ScaledText>
            <ScaledText style={[slotStyles.name, { color: D.textPrimary }]} numberOfLines={1}>{slot.item?.name || 'Item'}</ScaledText>
            {slot.item?.shopUrl ? (
              <TouchableOpacity
                style={slotStyles.buyButton}
                onPress={() => handleBuy(slot.item?.shopUrl)}
                activeOpacity={0.75}
              >
                <Ionicons name="cart-outline" size={11} color="#FFFFFF" />
                <ScaledText style={slotStyles.buyButtonText}>{t('outfitMaker.buy', 'Buy')}</ScaledText>
              </TouchableOpacity>
            ) : null}
          </View>
        </View>
      ))}
    </View>
  );
}

// ── Shuffle helper for outfit variety ─────────────────────────────────────
function shuffleArray<T>(array: T[]): T[] {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const OFFLINE_MOCK_SHOP_CATALOG = [
  {
    id: 'mock_top_1',
    name: 'Classic White T-Shirt',
    brand: 'Essential',
    type: 'top',
    category: 'tops',
    macroCategory: 'top',
    color: 'white',
    imageUrl: 'https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=500&q=80',
    description: 'A classic organic cotton white tee.',
  },
  {
    id: 'mock_top_2',
    name: 'Oxford Button-Down Shirt',
    brand: 'Classic',
    type: 'top',
    category: 'tops',
    macroCategory: 'top',
    color: 'blue',
    imageUrl: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=500&q=80',
    description: 'Light blue oxford cotton button-down shirt.',
  },
  {
    id: 'mock_bottom_1',
    name: 'Slim-Fit Chino Pants',
    brand: 'Essential',
    type: 'bottom',
    category: 'bottoms',
    macroCategory: 'bottom',
    color: 'beige',
    imageUrl: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=500&q=80',
    description: 'Slim-fit stretch chino pants in light beige.',
  },
  {
    id: 'mock_bottom_2',
    name: 'Dark Wash Denim Jeans',
    brand: 'Denim Co',
    type: 'bottom',
    category: 'bottoms',
    macroCategory: 'bottom',
    color: 'navy',
    imageUrl: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=500&q=80',
    description: 'Classic dark wash straight-leg jeans.',
  },
  {
    id: 'mock_outerwear_1',
    name: 'Unstructured Wool Blazer',
    brand: 'Tailored',
    type: 'outerwear',
    category: 'outerwear',
    macroCategory: 'outerwear',
    color: 'grey',
    imageUrl: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=500&q=80',
    description: 'Comfortable grey wool blazer for smart casual occasions.',
  },
  {
    id: 'mock_shoes_1',
    name: 'White Leather Sneakers',
    brand: 'Minimalist',
    type: 'shoes',
    category: 'shoes',
    macroCategory: 'shoes',
    color: 'white',
    imageUrl: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=500&q=80',
    description: 'Clean minimalist white leather low-top sneakers.',
  },
  {
    id: 'mock_shoes_2',
    name: 'Suede Penny Loafers',
    brand: 'Classic',
    type: 'shoes',
    category: 'shoes',
    macroCategory: 'shoes',
    color: 'brown',
    imageUrl: 'https://images.unsplash.com/photo-1533867617858-e7b97e060509?w=500&q=80',
    image: 'https://images.unsplash.com/photo-1533867617858-e7b97e060509?w=500&q=80',
    description: 'Handcrafted brown suede penny loafers.',
  }
];

// ── 100% OFFLINE OUTFIT GENERATOR (No Edge Function Required) ─────────────
async function generateOfflineOutfits(
  items: any[],
  extraItems: any[],
  style: string,
  weather: any,
  limit: number,
  t: any,
  forceWardrobeOnly: boolean = false
): Promise<GeneratedOutfit[]> {
  // ── Fill missing macro-category slots from shop_catalog ──────────────
  // If the wardrobe has no shoes (or no outerwear/top/bottom), pull
  // matching items from the shop catalog so the outfit builder can
  // produce a complete outfit.
  const availableItems = forceWardrobeOnly 
    ? items.filter((it: any) => it.isWardrobe)
    : [...items, ...(extraItems || [])];
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

    // Safety fallback: if database returned no shop fills for missing slots, use our local static mock catalog!
    if (shopFills.length < missingSlots.length) {
      const missingFills = missingSlots.filter(slot => !shopFills.some(sf => sf.macroCategory === slot));
      for (const slot of missingFills) {
        const mockMatch = OFFLINE_MOCK_SHOP_CATALOG.find(m => m.macroCategory === slot);
        if (mockMatch) {
          shopFills.push({
            ...mockMatch,
            id: `${mockMatch.id}_fallback`,
            isShopItem: true,
            recommendation: `Basic ${slot} suggested as fallback`,
          });
        }
      }
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

  // Shuffle each category to ensure variety even with limited items
  const layers = shuffleArray(classified.filter(c => c.category === 'layer').map(c => c.item));
  const tops = shuffleArray(classified.filter(c => c.category === 'top').map(c => c.item));
  const pants = shuffleArray(classified.filter(c => c.category === 'pants').map(c => c.item));
  const shoes = shuffleArray(classified.filter(c => c.category === 'shoes').map(c => c.item));

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
    t('outfitMaker.description1', { style: styleName }),
    t('outfitMaker.description2', { style: styleName }),
    t('outfitMaker.description3', { style: styleName }),
  ];
  const tips = [
    [t('outfitMaker.tip1_1'), t('outfitMaker.tip1_2')],
    [t('outfitMaker.tip2_1'), t('outfitMaker.tip2_2')],
    [t('outfitMaker.tip3_1'), t('outfitMaker.tip3_2')],
  ];

  const outfits: GeneratedOutfit[] = [];
  const targetItems = 4;

  for (let i = 0; i < limit; i++) {
    const outfitItems: any[] = [];
    
    // Use random offsets to ensure variety even with small wardrobes
    const randOffset1 = Math.floor(Math.random() * 100);
    const randOffset2 = Math.floor(Math.random() * 100);

    const candidatePant = pants[(i + randOffset1) % Math.max(pants.length, 1)] || pants[0];
    const pantIsShorts = !!candidatePant && isShortsBottom(candidatePant);

    // 4-slot model: [outerwear/layer, baseTop, pants, shoes]
    // Slot 1: outerwear (main-top / layer)
    if (needsLayer && layers.length > 0) {
      let layer = layers[(i + randOffset1) % layers.length];
      if (pantIsShorts && isFormalLayer(layer)) {
        layer = casualLayers[(i + randOffset1) % Math.max(casualLayers.length, 1)] || layer;
      }
      outfitItems.push({ ...layer, macroCategory: 'outerwear' });
    }

    // Slot 2: base top (second-top / shirt / tee)
    if (tops.length > 0) {
      const top = tops[(i + randOffset2) % tops.length];
      outfitItems.push({ ...top, macroCategory: 'top' });
    }
    // When layering, ensure we have a second base top (clone if only 1 top)
    if (needsLayer && outfitItems.some(item => item.macroCategory === 'outerwear')) {
      const existingTop = outfitItems.find(item => item.macroCategory === 'top');
      const secondTop = tops.find((top) => top && top.id !== existingTop?.id)
        || tops[(i + randOffset2 + 1) % Math.max(tops.length, 1)];
      if (secondTop && secondTop.id !== existingTop?.id) {
        // Replace the single top with a different second top
        const topIdx = outfitItems.findIndex(item => item.macroCategory === 'top');
        if (topIdx >= 0) outfitItems[topIdx] = { ...secondTop, macroCategory: 'top' };
        // Keep the first top as-is (already pushed)
      }
      // If only 1 top available, clone it for the second slot
      if (outfitItems.filter(item => item.macroCategory === 'top').length < 2 && existingTop) {
        outfitItems.push({ ...existingTop, id: `${existingTop.id || existingTop.name}_layered_copy_${i}_${Date.now()}`, macroCategory: 'top' });
      }
    }

    let finalPant = candidatePant;
    if (needsLayer && pantIsShorts && outfitItems[0] && isFormalLayer(outfitItems[0]) && nonShortsPants.length > 0) {
      finalPant = nonShortsPants[(i + randOffset1) % nonShortsPants.length];
    }
    if (finalPant) {
      outfitItems.push({ ...finalPant, macroCategory: 'bottom' });
    }

    if (shoes.length > 0) {
      const shoe = shoes[(i + randOffset2) % shoes.length];
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
      id: `offline_${Date.now()}_${i}_${Math.floor(Math.random() * 1000)}`,
      mainImage: outfitItems[0]?.image || outfitItems[0]?.imageUrl || '',
      matchScore: 0.85 + (Math.random() * 0.1), // Random variation in match score
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
  buyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    marginTop: 6,
    backgroundColor: '#0A1931',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: LiquidGlass2026Theme.radius.pill,
  },
  buyButtonText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
});

const parseCalendarDateParam = (value: unknown): Date | null => {
  if (typeof value !== 'string') return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim());
  if (!match) return null;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]), 12, 0, 0);
  return Number.isNaN(date.getTime()) ? null : date;
};

const formatPlanningDate = (date: Date): string =>
  date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

const AIOutfitGenerator = () => {
  const navigation = useNavigation();
  const route = useRoute<any>();
  const [currentSource, setCurrentSource] = useState(route.params?.source || 'wardrobe');
  // Anchor item passed from MyClosetScreen — when set, every generated outfit
  // must include this item locked into its slot.
  const baseItemId: string | undefined = route.params?.baseItemId;
  const baseItem: { id: string; imageUrl?: string; name?: string; type?: string; macroCategory?: string; color?: string; category?: string; brand?: string; description?: string } | undefined = route.params?.baseItem;
  // Calendar planning context — when navigated from OutfitCalendar's
  // "Plan Outfit", this holds the target day (YYYY-MM-DD). The AI flow
  // then acts as a travel/day planner: results default to saving back
  // to that day.
  const planningDateKey: string | undefined = typeof route.params?.calendarDate === 'string'
    ? route.params.calendarDate
    : undefined;
  const planningDate = React.useMemo(
    () => parseCalendarDateParam(planningDateKey),
    [planningDateKey],
  );
  const { t } = useTranslation();
  // ── Quiz-oriented defaults ──────────────────────────────────────────
  // When the user arrives from the Style Quiz, the generator orients on
  // their quiz personality instead of a hardcoded fallback.
  const quizPersonality = useStylePreferenceStore((s) => s.preferences.stylePersonality);
  const quizOccasions = useStylePreferenceStore((s) => s.preferences.primaryOccasions);
  const quizStyleDefault = useMemo(
    () => mapQuizStyleToAiStyle(
      typeof route.params?.initialStyle === 'string' && route.params.initialStyle
        ? route.params.initialStyle
        : quizPersonality,
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [quizPersonality],
  );
  const [selectedStyle, setSelectedStyle] = useState<string>(
    typeof route.params?.initialStyle === 'string' && route.params.initialStyle
      ? normalizeStyleId(route.params.initialStyle) === 'business_casual' ? 'classic' : route.params.initialStyle
      : quizStyleDefault,
  );
  // Keep in sync if the quiz completes while this screen is mounted.
  useEffect(() => {
    if (typeof route.params?.initialStyle !== 'string') {
      setSelectedStyle(quizStyleDefault);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quizStyleDefault]);
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
  const [error, setError] = useState('');
  const [userPrompt, setUserPrompt] = useState('');
  // ── New staged Create flow ──────────────────────────────────────────
  // Step 1 "create": source (wardrobe / new clothing) + style (quiz-oriented).
  // Step 2 "occasion": small "Where are you going?" input.
  // On Generate the whole setup block animates out (exit animation) and
  // the AI thinking view takes over on the SAME page.
  const [selectedOccasion, setSelectedOccasion] = useState<string | undefined>(
    Array.isArray(quizOccasions) && quizOccasions.length > 0 ? quizOccasions[0] : undefined,
  );
  // Occasion dropdown: 2 chips visible, rest behind a "More" toggle.
  const [showAllOccasions, setShowAllOccasions] = useState(false);
  // Style dropdown: same pattern — 2 rows visible, rest behind "More".
  const [showAllStyles, setShowAllStyles] = useState(false);
  const [collectionPickerVisible, setCollectionPickerVisible] = useState(false);
  const [pendingSaveOutfit, setPendingSaveOutfit] = useState<GeneratedOutfit | null>(null);
  const insets = useSafeAreaInsets();


  // ── Weather State ─────────────────────────────────────────────────
  const [weather, setWeather] = useState<{ temp: number; condition: string; icon?: string; city?: string } | undefined>(undefined);

  // ── Calendar Modal State ──────────────────────────────────────────
  const [calendarVisible, setCalendarVisible] = useState(false);
  const [calendarOutfit, setCalendarOutfit] = useState<GeneratedOutfit | null>(null);
  const [calendarDate, setCalendarDate] = useState<Date>(() => planningDate ?? new Date());

  // Keep the save-to-calendar picker in sync when arriving from
  // OutfitCalendar's "Plan Outfit" with a target day.
  useEffect(() => {
    if (planningDate) setCalendarDate(planningDate);
  }, [planningDateKey]);

  const D = useDesignTokens();

  // Wardrobe Items State
  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);

  // Use the live catalog from Supabase (all sources: Zara + Massimo Dutti)
  const { items: liveShopCatalog, loading: shopCatalogLoading } = useShopCatalog({
    enabled: true,
    source: 'all',
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
      shopUrl: item.sourceUrl,
      isShopItem: true,
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
  // baseItemId/baseItem included: anchor must survive source switches
  // (wardrobe ↔ shop) without losing the lock.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shopCatalogLoading, liveShopMapped, currentSource, baseItemId]);

  const loadWardrobeItems = async () => {
    try {
      // ── Shop mode: live Supabase shop catalog items PLUS the locked anchor.
      // Real Supabase UUIDs MUST be preserved so the edge function can enrich
      // items with imageUrl via its itemMap lookup. Do NOT mix in AsyncStorage.
      // The anchor is always prepended so "New clothing" still builds around it.
      if (currentSource === 'shop') {
        if (!isMounted.current) return;
        const shopOnly = liveShopMapped.map((item: any) => ({
          ...item,
          // Preserve the real UUID from shop_catalog — never overwrite with a fake id
          type: item.type || item.category || 'Clothing Piece',
          imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
          name: item.name || item.type || t('common.clothingItem'),
          description: item.description || item.name || '',
        }));
        if (!shopOnly.some((i: any) => String(i.id) === String(baseItemId)) && baseItemId) {
          // baseItem params can be dropped on re-render — rebuild from store.
          let anchorSrc: any = baseItem;
          if (!anchorSrc) {
            try {
              anchorSrc = useWardrobeStore.getState().items?.find((s: any) => String(s.id) === String(baseItemId));
              if (anchorSrc) {
                anchorSrc = {
                  type: anchorSrc.subCategory || anchorSrc.category,
                  category: anchorSrc.category,
                  name: anchorSrc.name || anchorSrc.subCategory,
                  color: anchorSrc.primaryColor,
                  brand: anchorSrc.brand,
                  description: (anchorSrc as any).description,
                  imageUrl: anchorSrc.imageUrl,
                };
              }
            } catch (_) { /* ignore */ }
          }
          if (anchorSrc) {
            const rawType = anchorSrc.type || (anchorSrc as any).category || '';
            const rawName = anchorSrc.name || anchorSrc.type || '';
            shopOnly.unshift({
              id: baseItemId,
              isWardrobe: true,
              type: rawType || 'Clothing Piece',
              category: (anchorSrc as any).category || rawType || 'tops',
              macroCategory: getBackendMacroCategory(rawType, (anchorSrc as any).category || '', rawName),
              color: anchorSrc.color || 'neutral',
              brand: (anchorSrc as any).brand || '',
              name: rawName || t('common.clothingItem'),
              description: (anchorSrc as any).description || rawName || '',
              image: anchorSrc.imageUrl || '',
              imageUrl: anchorSrc.imageUrl || '',
            });
          }
        }
        setWardrobeItems(shopOnly);
        return;
      }

      // ── Wardrobe mode: prefer Zustand useWardrobeStore items, then Supabase, then AsyncStorage ──
      let items: any[] = [];

      try {
        const storeItems = useWardrobeStore.getState().items;
        if (storeItems && storeItems.length > 0) {
          items = storeItems.map((item: any) => {
            const rawType = item.subCategory || item.category || '';
            const rawName = item.name || item.subCategory || '';
            const computedMacro = getBackendMacroCategory(rawType, item.category || '', rawName);
            return {
              id: item.id,
              isWardrobe: true,
              type: rawType || 'Clothing Piece',
              category: item.category || rawType || 'tops',
              macroCategory: computedMacro,
              subCategory: item.subCategory || undefined,
              color: item.primaryColor || 'neutral',
              brand: item.brand || '',
              name: rawName || t('common.clothingItem'),
              description: item.description || rawName || '',
              image: item.imageUrl || '',
              imageUrl: item.imageUrl || '',
            };
          });
        }
      } catch (storeError) {
        console.warn('[AIOutfitmaker] Failed to load Zustand store items:', storeError);
      }

      // Fall back to direct Supabase query if store was empty/failed
      if (items.length === 0) {
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

            items = (remoteItems || []).map((item: any) => {
              const rawType = item.type || item.sub_category || item.category || '';
              const rawName = item.name || item.type || '';
              const computedMacro = getBackendMacroCategory(rawType, item.category || '', rawName);
              return {
                id: item.id,
                isWardrobe: true,
                type: rawType || 'Clothing Piece',
                category: item.category || rawType || 'tops',
                macroCategory: computedMacro,
                subCategory: item.sub_category || undefined,
                color: Array.isArray(item.color) ? item.color[0] : (item.primary_color || 'neutral'),
                brand: item.brand || '',
                name: rawName || t('common.clothingItem'),
                description: item.description || rawName || '',
                image: item.image_url || '',
                imageUrl: item.image_url || '',
              };
            });
          }
        } catch (remoteError) {
          console.warn('[AIOutfitmaker] Failed to load Supabase wardrobe, falling back to local storage:', remoteError);
        }
      }

      // Fall back to AsyncStorage directly
      if (items.length === 0) {
        const data = await AsyncStorage.getItem('myWardrobeItems');
        items = data ? JSON.parse(data).map((it: any) => ({ ...it, isWardrobe: true })) : [];
      }

      // Normalise image/imageUrl so both fields are always populated
      items = items.map((i: any) => ({
        ...i,
        image: i.image || i.imageUrl,
        imageUrl: i.imageUrl || (typeof i.image === 'string' ? i.image : undefined),
      }));

      // Drop items with no usable image
      items = items.filter((i: any) => i && (i.imageUrl || typeof i.image === 'string'));

      // Prepend/inject the anchor item (baseItem) if provided but not in items pool.
      // This guarantees that "build around this item" always has the selected item,
      // even if the user has no other saved items or sync is lagging!
      if (baseItem && baseItemId && !items.some(i => i.id === baseItemId)) {
        console.log('[AIOutfitmaker] Injected baseItem into wardrobe pool:', baseItemId);
        const rawType = baseItem.type || baseItem.category || '';
        const rawName = baseItem.name || baseItem.type || '';
        const computedMacro = getBackendMacroCategory(rawType, baseItem.category || '', rawName);
        items.unshift({
          id: baseItemId,
          isWardrobe: true,
          type: rawType || 'Clothing Piece',
          category: baseItem.category || rawType || 'tops',
          macroCategory: computedMacro,
          subCategory: undefined,
          color: baseItem.color || 'neutral',
          brand: baseItem.brand || '',
          name: rawName || t('common.clothingItem'),
          description: baseItem.description || rawName || '',
          image: baseItem.imageUrl || '',
          imageUrl: baseItem.imageUrl || '',
        });
      }

      // Always inject shop items but mark them.
      // When source is 'wardrobe', outfits will be a mix of owned and shop items.
      const existingIds = new Set(items.map((i: any) => i.id));
      const newShopItems = liveShopMapped.filter(s => !existingIds.has(s.id));
      items = [...items, ...newShopItems.map(it => ({ ...it, isWardrobe: false }))];

      if (!isMounted.current) return;
      setWardrobeItems(items.map((item: any, index: number) => ({
        ...item,
        // For personal wardrobe items that have no real id, generate a stable key.
        // For shop items merged here, preserve their real UUID.
        id: item.id || `local_item_${index}_${item.type || item.category || 'unknown'}`,
        isWardrobe: item.isWardrobe ?? false,
        type: item.type || item.category || 'Clothing Piece',
        imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : undefined) || '',
        name: item.name || item.type || t('common.clothingItem'),
        description: item.description || item.name || '',
      })));
    } catch (e) {
      console.warn('Failed to load wardrobe', e);
    }
  };

  // Backend-compatible macroCategory matching the edge function's slot model:
  // 'top' (base layer), 'outerwear' (jacket/blazer/sweater), 'bottom', 'shoes'
  const getBackendMacroCategory = (type: string, category?: string, name?: string) => {
    // Outerwear keywords take precedence over a generic alias: a blazer
    // stored as category "top"/"tops" is still a layer (main-top slot),
    // not a base top. Without this the anchor locks into second-top.
    const t = `${type || ''} ${category || ''} ${name || ''}`.toLowerCase();
    if (t.match(/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat|bomber)\b/)) return 'outerwear';

    // Prefer canonical aliases (upper_body / lower_body / tops / footwear / …)
    // so items coming back from shop_catalog are slotted correctly.
    const aliasHit = canonicalizeMacroCategory(type) !== 'other'
      ? canonicalizeMacroCategory(type)
      : canonicalizeMacroCategory(category || '');
    if (aliasHit !== 'other') return aliasHit;

    if (t.match(/\b(pant|trouser|jeans?|bottom|shorts?|skirt|lower[_\s-]?body)\b/)) return 'bottom';
    if (t.match(/\b(shoe|sneaker|boot|loafer|sandal|footwear)\b/)) return 'shoes';
    if (t.match(/\b(shirt|t-shirt|tee|blouse|polo|top|dress|upper[_\s-]?body)\b/)) return 'top';
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

  const candidatePool = useMemo(() => {
    // In shop mode wardrobeItems already = shop catalog + locked anchor
    // (see loadWardrobeItems). Using liveShopMapped alone would drop the
    // anchor from fallback picks and lose the lock.
    const pool = currentSource === 'shop'
      ? wardrobeItems
      : currentSource === 'wardrobe'
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

    return pool.map((it: any) => ({
      id: it.id || it.imageUrl || it.name,
      name: it.name || it.type || 'Item',
      image: (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '',
      imageUrl: it.imageUrl || (typeof it.image === 'string' ? it.image : ''),
      type: it.type || it.category || 'top',
      macroCategory: resolveMacroCategory(it),
      color: it.color || 'neutral',
      brand: it.brand || '',
      isWardrobe: it.isWardrobe ?? false,
      isShopItem: it.isShopItem ?? false,
      shopUrl: it.shopUrl || undefined,
    })).sort((a, b) => (b.isWardrobe ? 1 : 0) - (a.isWardrobe ? 1 : 0));
  }, [currentSource, liveShopMapped, wardrobeItems]);

  const normalizeTo4Slots = (items: OutfitItem[], styleId?: string): OutfitItem[] => {
    // ── Anchor resolution (single source of truth for locking) ──────────
    // The anchor MUST win its macro slot regardless of AI order. We compute
    // its true macro with outerwear-keyword priority (blazer stored as "top"
    // is still a layer), then pick anchor-first per slot + post-enforce.
    // Fallback: route params can be lost on re-render — if baseItem is gone
    // but baseItemId survives, rebuild it from pool/store so the lock holds.
    let effectiveBaseItem = baseItem;
    if (!effectiveBaseItem && baseItemId) {
      const found: any = [...(items || []), ...candidatePool].find(
        (i: any) => String(i?.id) === String(baseItemId),
      );
      if (found) {
        effectiveBaseItem = {
          id: baseItemId,
          imageUrl: found.imageUrl || (typeof found.image === 'string' ? found.image : undefined),
          name: found.name,
          type: found.type,
          macroCategory: found.macroCategory,
          color: found.color,
          category: found.category,
          brand: found.brand,
          description: found.description,
        } as any;
      }
    }
    let anchorMacro: string | null = null;
    let anchorOutfitItem: OutfitItem | null = null;
    if (effectiveBaseItem && baseItemId) {
      const recomputed = getBackendMacroCategory(
        effectiveBaseItem.type || '',
        (effectiveBaseItem as any).category || effectiveBaseItem.macroCategory || '',
        effectiveBaseItem.name || '',
      );
      const stored = canonicalizeMacroCategory(effectiveBaseItem.macroCategory || '');
      // Outerwear wins on conflict: a blazer mis-tagged as "top" upstream
      // must lock into the main-top/layer slot, not second-top.
      anchorMacro = recomputed === 'outerwear' ? 'outerwear' : (stored !== 'other' ? stored : recomputed);
      anchorOutfitItem = {
        id: baseItemId,
        name: effectiveBaseItem.name || effectiveBaseItem.type || 'Item',
        image: effectiveBaseItem.imageUrl || '',
        imageUrl: effectiveBaseItem.imageUrl || '',
        type: effectiveBaseItem.type || anchorMacro,
        macroCategory: anchorMacro,
        color: effectiveBaseItem.color || 'neutral',
        isWardrobe: true,
      };
      // Pre-inject so AI-order-independent picks still see it first.
      const alreadyPresent = items.some(i => String(i.id) === String(baseItemId));
      if (!alreadyPresent) {
        const slotIdx = items.findIndex(
          i => (canonicalizeMacroCategory(i.macroCategory || '') === anchorMacro
            || canonicalizeMacroCategory(getBackendMacroCategory(i.type || '', i.category || '', i.name || '')) === anchorMacro)
        );
        if (slotIdx >= 0) {
          items = [...items];
          (items as OutfitItem[])[slotIdx] = anchorOutfitItem;
        } else {
          items = [anchorOutfitItem, ...items];
        }
      }
    }
    let incomingItems = items;
    // Helper: prefer an item's own macroCategory (canonicalized) over the
    // keyword-based fallback. Keeps items whose type/category are raw
    // garmentType strings (`upper_body`, `lower_body`) from being dropped.
    const resolveMacroCategory = (it: any): string => {
      const ownCanonical = canonicalizeMacroCategory(it?.macroCategory || '');
      if (ownCanonical !== 'other') return ownCanonical;
      return getBackendMacroCategory(it?.type || '', it?.category || '', it?.name || '');
    };

    const resolveImage = (it: any) => (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '';

    const incoming = (incomingItems || []).map((it) => ({
      ...it,
      image: it.image || it.imageUrl || '',
      macroCategory: resolveMacroCategory(it),
    }));
    // A locked outerwear anchor forces layering even in warm weather —
    // otherwise its slot would never be picked and it would "fall" into top.
    const anchorForcesLayer = anchorMacro === 'outerwear';
    const needsLayer = anchorForcesLayer || !weather
      || (weather?.temp != null && weather.temp < 18)
      || /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test((weather?.condition || '').toLowerCase());
    const usedIds = new Set<string>();

    console.log('[normalizeTo4Slots] incoming items:', incoming.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory, image: (it.image || '').substring(0, 40) })));
    console.log('[normalizeTo4Slots] candidatePool size:', candidatePool.length, 'sample:', candidatePool.slice(0, 5).map((it: any) => ({ id: it.id, macro: it.macroCategory })));
    console.log('[normalizeTo4Slots] needsLayer:', needsLayer, 'weather:', weather, 'anchor:', anchorMacro, baseItemId);

    const pickCandidate = (predicate: (candidate: any) => boolean) => {
      const found = [...incoming, ...candidatePool].find((candidate: any) => {
        const key = String(candidate.id || '');
        return !usedIds.has(key) && predicate(candidate);
      });
      if (!found) return undefined;
      usedIds.add(String(found.id || ''));
      return { ...found, image: resolveImage(found) || found.image || '' };
    };

    // Anchor-first pick: when this slot IS the anchor's macro, the locked
    // item wins over AI order. Marks usedIds so it can't be double-picked.
    const pickAnchorForSlot = (slotMacro: string) => {
      if (!anchorOutfitItem || !anchorMacro || anchorMacro !== slotMacro) return undefined;
      const key = String(anchorOutfitItem.id || '');
      if (usedIds.has(key)) {
        // Already placed (e.g. outerwear anchor also matching top fallback) —
        // return the same object so caller keeps a single instance.
        return { ...anchorOutfitItem, image: resolveImage(anchorOutfitItem) || anchorOutfitItem.image || '' };
      }
      usedIds.add(key);
      return { ...anchorOutfitItem, image: resolveImage(anchorOutfitItem) || anchorOutfitItem.image || '' };
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

    // Prefer NON-shorts bottoms. Never promote a random non-shoe/top item
    // to the pants slot — that was producing the "flowy shirt labeled
    // SHORTS" bug. Fall back to PLACEHOLDER_PANTS when no real pants exist.
    const slotPantsRaw =
      pickAnchorForSlot('bottom')
      || pickCandidate((candidate: any) => candidate.macroCategory === 'bottom' && hasRealImage(candidate) && !isShortsBottom(candidate))
      || pickCandidate((candidate: any) => candidate.macroCategory === 'bottom' && !isShortsBottom(candidate))
      || { ...PLACEHOLDER_PANTS };
    // Force macroCategory to 'bottom' so OutfitSlotGrid.classifyItem never drops it
    const slotPants = { ...slotPantsRaw, macroCategory: 'bottom' as const };
    const pantsIsShorts = isShortsBottom(slotPantsRaw);

    const slotOuterwearRaw = needsLayer
      ? (pickAnchorForSlot('outerwear')
        || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)))
        || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !(pantsIsShorts && isFormalLayer(candidate)))
        || (!pantsIsShorts ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear') : undefined))
      : (anchorMacro === 'outerwear' ? pickAnchorForSlot('outerwear') : undefined);
    // Force macroCategory to 'outerwear' so OutfitSlotGrid.classifyItem never drops it
    const slotOuterwear = slotOuterwearRaw ? { ...slotOuterwearRaw, macroCategory: 'outerwear' as const } : undefined;

    const slotMainTopRaw =
      pickAnchorForSlot('top')
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && hasRealImage(candidate))
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top')
      || pickCandidate((candidate: any) => candidate.macroCategory !== 'bottom' && candidate.macroCategory !== 'shoes' && candidate.macroCategory !== 'outerwear' && candidate.macroCategory !== 'other');
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotMainTop = slotMainTopRaw ? { ...slotMainTopRaw, macroCategory: 'top' as const } : undefined;

    // Pick shoes BEFORE second-top so the second-top fallback can't
    // accidentally consume a shoes item whose macroCategory resolved
    // to 'other' or an unrecognized alias.
    const slotShoesRaw =
      pickAnchorForSlot('shoes')
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate))
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
      ? (String(slotMainTop.id) !== String(slotOuterwear.id) ? slotMainTop : slotSecondTop)
      : (slotOuterwear ? slotSecondTop : slotSecondTop || slotMainTop);
    const result = [slot1, slot2, slotPants, slotShoes].filter(Boolean) as OutfitItem[];
    // ── Post-enforce: anchor MUST be in the final outfit ────────────────
    // First-match picks above can still miss when AI order puts another item
    // of the same macro first and anchor-first was bypassed (e.g. placeholder
    // fallback). Replace the same-macro slot so every slot mapping stays
    // correct: outerwear→slot1(main-top), top→slot2(second-top),
    // bottom→slot3, shoes→slot4.
    if (anchorOutfitItem && anchorMacro) {
      const hasAnchor = result.some(i => String(i.id) === String(anchorOutfitItem.id));
      if (!hasAnchor) {
        const targetIdx =
          anchorMacro === 'outerwear' ? 0
          : anchorMacro === 'top' ? 1
          : anchorMacro === 'bottom' ? 2
          : anchorMacro === 'shoes' ? 3
          : result.findIndex(i => canonicalizeMacroCategory(i.macroCategory || '') === anchorMacro);
        if (targetIdx >= 0 && targetIdx < result.length) {
          result[targetIdx] = { ...anchorOutfitItem, macroCategory: anchorMacro as any };
        } else {
          const fallbackIdx = result.findIndex(i => canonicalizeMacroCategory(i.macroCategory || '') === anchorMacro);
          if (fallbackIdx >= 0) result[fallbackIdx] = { ...anchorOutfitItem, macroCategory: anchorMacro as any };
          else result.unshift({ ...anchorOutfitItem, macroCategory: anchorMacro as any });
        }
      }
    }
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
      case 'classic':
        return 'formal';
      case 'semi_classic':
      case 'minimalist':
      case 'casual':
        return 'casual';
      default:
        return 'casual';
    }
  };

  const COLLECTION_CATEGORIES: Array<{ id: string; label: string; icon: string }> = [
    { id: 'casual', label: t('outfitMaker.casual'), icon: 'sunny' },
    { id: 'work', label: t('outfitMaker.work'), icon: 'briefcase' },
    { id: 'formal', label: t('outfitMaker.formal'), icon: 'shirt' },
    { id: 'sport', label: t('outfitMaker.sport'), icon: 'fitness' },
    { id: 'date', label: t('outfitMaker.date'), icon: 'heart' },
    { id: 'travel', label: t('outfitMaker.travel'), icon: 'airplane' },
  ];

  // ── Save Outfit to Closet ─────────────────────────────────────────
  // When launched from the trip/calendar planner (planningDateKey), this
  // is a *connected* save: the outfit is stored in the closet AND linked
  // to the trip day (outfitLogs) so the Trips tab / packing list updates
  // and the user gets a visible confirmation + a way back.
  const persistOutfitToTripDay = async (outfit: GeneratedOutfit, dateKey: string, occasion: string) => {
    const mappedItems: CalendarOutfitItem[] = outfit.items.map((it) => ({
      id: it.id || it.name || `${Date.now()}`,
      type: it.type || 'top',
      image: it.image || it.imageUrl || '',
      color: it.color,
      name: it.name,
    }));
    const log = createOutfitLog(dateKey, mappedItems, occasion);
    const raw = await AsyncStorage.getItem('outfitLogs');
    const logs: Record<string, OutfitLog> = raw ? JSON.parse(raw) : {};
    const updated = Boolean(logs[dateKey]);
    logs[dateKey] = log;
    await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
    return { updated };
  };

  const saveOutfitToCloset = async (outfit: GeneratedOutfit, collectionCategory?: string) => {
    const itemIds = outfit.items
      .map((item) => String(item.id || item.image))
      .filter(Boolean);
    if (itemIds.length === 0) {
      Alert.alert(t('outfitMaker.cannotSave'), t('outfitMaker.noValidItems'));
      return;
    }

    const store = useWardrobeStore.getState();
    const occasion = collectionCategory || getCalendarOccasion(selectedStyle);
    // Connected save targets the trip day; standalone saves use today.
    const connectedDateKey = planningDateKey ?? null;
    const newOutfitId = store.addOutfit({
      userId: 'user',
      itemIds,
      occasion,
      generatedBy: 'ai',
      previewImageUrl: typeof outfit.mainImage === 'string' ? outfit.mainImage : undefined,
      reasoning: outfit.description,
      style: selectedStyle,
      snapshotItems: outfit.items
        .map((item: any) => ({
          id: String(item.id || item.image),
          image: String(item.imageUrl || item.image || ''),
          category: item.macroCategory || item.category,
          name: item.name,
        }))
        .filter((s: any) => s.image.length > 0)
        .slice(0, 4),
    } as any);

    if (newOutfitId) {
      store.saveOutfit(newOutfitId, collectionCategory);
    }

    // Also save to Supabase saved_outfits table
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      const userId = sessionData?.session?.user?.id;
      if (userId) {
        const { error: saveError } = await supabase.from('saved_outfits').insert({
          user_id: userId,
          items: outfit.items.map((item) => ({
            id: String(item.id || item.image),
            type: item.type || 'Clothing Piece',
            image: item.image,
            imageUrl: item.imageUrl,
            name: item.name,
          })),
          date: connectedDateKey ?? new Date().toISOString().split('T')[0],
          occasion,
          caption: outfit.description || '',
          visibility: 'Everyone',
          is_ootd: false,
        });
        if (saveError) {
          console.warn('[AIOutfitmaker] Failed to sync saved outfit to Supabase:', saveError);
        }
      }
    } catch (err) {
      console.warn('[AIOutfitmaker] Error syncing saved outfit:', err);
    }

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

    // Connected (trip planner) save: also link the outfit to the trip day
    // so Calendar + Trips packing list update immediately.
    if (connectedDateKey) {
      let updated = false;
      try {
        ({ updated } = await persistOutfitToTripDay(outfit, connectedDateKey, occasion));
      } catch (err) {
        console.warn('[AIOutfitmaker] Failed to link saved outfit to trip day:', err);
      }
      const dateLabel = planningDate ? formatPlanningDate(planningDate) : connectedDateKey;
      const goBackToPlanner = () => {
        if ((navigation as any)?.goBack) {
          (navigation as any).goBack();
        }
      };
      Alert.alert(
        updated
          ? t('outfitMaker.savedToTripUpdated', { defaultValue: 'Trip outfit updated for {{date}}', date: dateLabel })
          : t('outfitMaker.savedToTrip', { defaultValue: 'Saved to your trip · {{date}}', date: dateLabel }),
        t('outfitMaker.savedToTripSubtitle', {
          defaultValue: 'Saved to your closet and connected to your trip day calendar + packing list.',
        }),
        [
          { text: t('common.done', { defaultValue: 'Done' }), style: 'cancel', onPress: goBackToPlanner },
          {
            text: t('outfitMaker.viewTrip', { defaultValue: 'View Trip' }),
            onPress: goBackToPlanner,
          },
        ],
      );
      return;
    }

    Alert.alert(t('outfitMaker.saved'), t('outfitMaker.outfitSavedCloset'));
  };

  const handleSaveOutfitPress = (outfit: GeneratedOutfit) => {
    setPendingSaveOutfit(outfit);
    setCollectionPickerVisible(true);
  };

  const handleCollectionSelect = (collectionId: string) => {
    setCollectionPickerVisible(false);
    if (pendingSaveOutfit) {
      saveOutfitToCloset(pendingSaveOutfit, collectionId);
      setPendingSaveOutfit(null);
    }
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

      const finishSave = (savedDate: Date) => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        setCalendarVisible(false);
        setCalendarOutfit(null);
        const dateLabel = formatPlanningDate(savedDate);
        const goBackToPlanner = () => {
          // When planning a specific calendar/trip day, return to it so the
          // user immediately sees the planned outfit.
          if (planningDateKey && (navigation as any)?.goBack) {
            (navigation as any).goBack();
          }
        };
        // Visible confirmation — saving was previously silent (modal just
        // closed), which felt like "nothing happened".
        Alert.alert(
          t('outfitMaker.savedToCalendar', { defaultValue: 'Saved to {{date}}', date: dateLabel }),
          t('outfitMaker.savedToCalendarSubtitle', {
            defaultValue: 'Your outfit was added to your calendar and trip packing list.',
          }),
          [
            { text: t('common.done', { defaultValue: 'Done' }), style: 'cancel', onPress: goBackToPlanner },
            {
              text: t('outfitMaker.viewCalendar', { defaultValue: 'View Calendar' }),
              onPress: () => {
                if (planningDateKey && (navigation as any)?.goBack) {
                  (navigation as any).goBack();
                } else {
                  (navigation as any)?.navigate?.('Calendar');
                }
              },
            },
          ],
        );
      };

      // Warn before overwriting an existing outfit for this date
      if (logs[dateStr]) {
        Alert.alert(
          t('outfitMaker.outfitExists'),
          t('outfitMaker.replaceOutfit'),
          [
            { text: t('common.cancel'), style: 'cancel' },
            {
              text: t('outfitMaker.replace'),
              style: 'destructive',
              onPress: async () => {
                logs[dateStr] = log;
                await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
                finishSave(date);
              },
            },
          ]
        );
        return;
      }

      logs[dateStr] = log;
      await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
      finishSave(date);
    } catch (e) {
      console.error('Calendar save error', e);
    }
  };

  const openCalendarForOutfit = (outfit: GeneratedOutfit) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setCalendarOutfit(outfit);
    // Default to the planned day when arriving from OutfitCalendar.
    setCalendarDate(planningDate ?? new Date());
    setCalendarVisible(true);
  };

  const buildEffectivePrompt = useCallback(() => {
    const occasionLabel = OCCASION_PRESETS.find((o) => o.id === selectedOccasion)?.label;
    const trimmed = userPrompt.trim();
    if (occasionLabel && trimmed) return `${occasionLabel}: ${trimmed}`;
    if (occasionLabel) return occasionLabel;
    return trimmed;
  }, [selectedOccasion, userPrompt]);

  // ── iOS grouped checkmark row (shared by source / occasion / style) ──
  const renderCheckRow = (
    key: string,
    icon: string,
    label: string,
    selected: boolean,
    onPress: () => void,
    showDivider: boolean,
  ) => (
    <View key={key}>
      {showDivider && <View style={styles.rowDivider} />}
      <TouchableOpacity
        activeOpacity={0.7}
        onPress={() => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          onPress();
        }}
        style={styles.formRow}
      >
        <View style={styles.formRowIcon}>
          <Ionicons name={icon as any} size={17} color="#173A65" />
        </View>
        <ScaledText style={styles.formRowLabel}>{label}</ScaledText>
        <View style={{ flex: 1 }} />
        {selected && <Ionicons name="checkmark" size={22} color="#0A1931" />}
      </TouchableOpacity>
    </View>
  );

  // ── Generate (same page: setup animates out → thinking fades in) ──
  // Generating unmounts setup → thinking with FadeOutUp → FadeIn.
  const handleGeneratePress = useCallback((styleOverride?: string) => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    const styleToUse = styleOverride || selectedStyle;
    if (styleOverride) setSelectedStyle(styleOverride);
    generateOutfits(styleToUse);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStyle, selectedOccasion, userPrompt]);

  const generateOutfits = async (overrideStyle?: string) => {
    const styleToUse = overrideStyle || selectedStyle;
    const effectivePrompt = (() => {
      const occasionLabel = OCCASION_PRESETS.find((o) => o.id === selectedOccasion)?.label;
      const trimmed = userPrompt.trim();
      if (occasionLabel && trimmed) return `${occasionLabel}: ${trimmed}`;
      if (occasionLabel) return occasionLabel;
      return trimmed;
    })();

    if (!isMounted.current) return;
    setLoading(true);
    setError('');
    setOutfits([]);

    const selectedClothing: any[] = [];

    let effectiveSource = currentSource;

    if (currentSource === 'wardrobe') {
      const userOwnedItems = wardrobeItems.filter(i => i.isWardrobe === true);
      const MIN_REQUIRED = baseItemId ? 1 : 3;

      // Pre-compute macro category and score once per item to avoid redundant regex operations
      const userOwnedItemsWithMeta = userOwnedItems.map(i => {
        const mc = canonicalizeMacroCategory(i.macroCategory || getBackendMacroCategory(i.type || '', i.category || '', i.name || ''));
        const score = scoreItemForStyle({
          name: i.name,
          brand: i.brand,
          color: i.color,
          type: i.type,
          macroCategory: i.macroCategory,
        }, normalizeStyleId(styleToUse));
        return { item: i, mc, score };
      });

      const countBaseTops = userOwnedItemsWithMeta.filter(x => x.mc === 'top').length;
      const countOuterwear = userOwnedItemsWithMeta.filter(x => x.mc === 'outerwear').length;
      const countTops = countBaseTops + countOuterwear;
      const countPants = userOwnedItemsWithMeta.filter(x => x.mc === 'bottom').length;
      const countShoes = userOwnedItemsWithMeta.filter(x => x.mc === 'shoes').length;

      const suitableTops = userOwnedItemsWithMeta.filter(x => {
        if (baseItemId && String(x.item.id) === String(baseItemId)) return false;
        return (x.mc === 'top' || x.mc === 'outerwear') && x.score >= 0.35;
      }).map(x => x.item);

      const suitablePants = userOwnedItemsWithMeta.filter(x => {
        if (baseItemId && String(x.item.id) === String(baseItemId)) return false;
        return x.mc === 'bottom' && x.score >= 0.35;
      }).map(x => x.item);

      const suitableShoes = userOwnedItemsWithMeta.filter(x => {
        if (baseItemId && String(x.item.id) === String(baseItemId)) return false;
        return x.mc === 'shoes' && x.score >= 0.35;
      }).map(x => x.item);

      const isUnsuitable = suitableTops.length < MIN_REQUIRED || suitablePants.length < MIN_REQUIRED || suitableShoes.length < MIN_REQUIRED;

      if (countTops < MIN_REQUIRED || countPants < MIN_REQUIRED || countShoes < MIN_REQUIRED || isUnsuitable) {
        console.log(`[AIOutfitmaker] Wardrobe items are insufficient/unsuitable. Transitioning to shop mode.`);
        effectiveSource = 'shop';
        setCurrentSource('shop');
      }
    }

    // Build a clean payload for the backend — strip non-serialisable fields
    // (require() numbers) and ensure every item has macroCategory + imageUrl.
    let payloadItems = [];
    if (effectiveSource === 'shop') {
      payloadItems = wardrobeItems
        .filter((item: any) => !item.isWardrobe || (baseItemId && String(item.id) === String(baseItemId)))
        .map((item: any) => ({
          id: item.id,
          name: item.name || item.type || t('common.clothingItem'),
          type: item.type || 'top',
          category: item.category || item.type || 'tops',
          color: item.color || 'neutral',
          brand: item.brand || '',
          description: item.description || item.name || '',
          imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
          macroCategory: getBackendMacroCategory(item.type || '', item.category || '', item.name || ''),
        }));
    } else {
      payloadItems = wardrobeItems.map((item: any) => ({
        id: item.id,
        name: item.name || item.type || t('common.clothingItem'),
        type: item.type || 'top',
        category: item.category || item.type || 'tops',
        color: item.color || 'neutral',
        brand: item.brand || '',
        description: item.description || item.name || '',
        imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
        macroCategory: getBackendMacroCategory(item.type || '', item.category || '', item.name || ''),
      }));
    }

    let pureWardrobeOutfit: GeneratedOutfit | null = null;

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

      // ── Special Logic for My Wardrobe page ──
      // Requirement: first outfit must be pure wardrobe, others mixed.
      if (effectiveSource === 'wardrobe') {
        try {
          const offlinePure = await generateOfflineOutfits(
            wardrobeItems,
            liveShopMapped,
            styleToUse,
            resolvedWeather,
            1,
            t,
            true
          );
          if (offlinePure.length > 0) {
            pureWardrobeOutfit = {
              ...offlinePure[0],
              items: normalizeTo4Slots(offlinePure[0].items || [], styleToUse)
            };
          }
        } catch (pureErr) {
          console.warn('[AIOutfitmaker] Pure wardrobe generation failed', pureErr);
        }
      }

      const { data, error: invokeError } = await supabase.functions.invoke('generate-outfits', {
        body: {
          stylePreferences: styleToUse,
          occasion: effectivePrompt || selectedOccasion || userPrompt || 'Any',
          selectedItemIds: [],
          wardrobeItems: payloadItems,
          weather: resolvedWeather,
          limit: effectiveSource === 'wardrobe' ? 2 : 3, // request fewer if we already have a pure one
          prompt: effectivePrompt,
          // Lock the anchor item into every outfit when navigated from MyClosetScreen
          anchorItemId: baseItemId,
          // When source is 'shop', use only the provided shop catalog items
          // (Massimo Dutti etc.) and do NOT mix in user's own wardrobe items
          useProvidedWardrobeOnly: effectiveSource === 'shop',
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

        // Pad to 3 outfits when the backend returned fewer complete ones.
        // Use the offline builder to fill the remaining slots so the user
        // always sees three outfit variations.
        let finalOutfits = cleanedOutfits;
        if (finalOutfits.length < 3) {
          try {
            const offlinePad = await generateOfflineOutfits(
              wardrobeItems,
              liveShopMapped,
              styleToUse,
              resolvedWeather,
              3,
              t
            );
            const existingIds = new Set(finalOutfits.map((o: GeneratedOutfit) => o.id));
            for (const o of offlinePad) {
              if (finalOutfits.length >= 3) break;
              if (!existingIds.has(o.id)) {
                finalOutfits = [...finalOutfits, { ...o, items: normalizeTo4Slots(o.items || [], styleToUse) }]
                  .filter((out: GeneratedOutfit) => isCompleteNormalizedOutfit(out.items));
              }
            }
          } catch (padErr) {
            console.warn('[generateOutfits] offline pad failed', padErr);
          }
        }

        if (isMounted.current) {
          // NOTE: use finalOutfits (padded) — not cleanedOutfits — so the user
          // always sees 3 variations.
          const mergedOutfits = pureWardrobeOutfit ? [pureWardrobeOutfit, ...finalOutfits] : finalOutfits;
          setOutfits(mergedOutfits.slice(0, 3));
          setLoading(false);
        }
        return;
      }

      // Backend returned success=false or empty outfits — fall through to local fallback
      throw new Error(data?.error || 'No outfits returned from AI');
    } catch (err: any) {
      console.warn('[AIOutfitmaker] Edge function failed, using offline generator:', err);

      // ENHANCED OFFLINE GENERATOR — Works 100% without internet/edge function
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
      fallbackTimer.current = setTimeout(async () => {
        if (!isMounted.current) return;
        try {
          const offlineOutfits = await generateOfflineOutfits(wardrobeItems, liveShopMapped, styleToUse, weather, 3, t);

          // Anchor guarantee: the offline builder is anchor-blind, so every
          // fallback outfit MUST go through normalizeTo4Slots (anchor-first +
          // post-enforce) — otherwise the lock silently disappears when the
          // edge function fails. This was the "anchor missing entirely" bug.
          const normalizedOffline = (offlineOutfits || [])
            .map((o: GeneratedOutfit) => ({ ...o, items: normalizeTo4Slots(o.items || [], styleToUse) }))
            .filter((o: GeneratedOutfit) => isCompleteNormalizedOutfit(o.items));

          // Ensure first outfit is pure if source is wardrobe
          let finalOffline = normalizedOffline;
          if (effectiveSource === 'wardrobe' && pureWardrobeOutfit) {
            finalOffline = [pureWardrobeOutfit, ...normalizedOffline.slice(0, 2)];
          }

          if (isMounted.current) {
            setOutfits(finalOffline.length > 0 ? finalOffline : (pureWardrobeOutfit ? [pureWardrobeOutfit] : []));
            setLoading(false);
            setError(finalOffline.length === 0 && !pureWardrobeOutfit ? t('outfitMaker.noOutfitsError', { defaultValue: 'Could not generate outfits. Please add more items to your wardrobe.' }) : '');
          }
        } catch (offlineErr: any) {
          console.error('[AIOutfitmaker] Offline generator also failed:', offlineErr);
          if (isMounted.current) {
            setLoading(false);
            setError(t('outfitMaker.noOutfitsError', { defaultValue: 'Could not generate outfits. Please try again.' }));
          }
        }
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
              <Ionicons name="chevron-back" size={22} color="#0A1931" />
            </TouchableOpacity>
            <ScaledText style={styles.headerTitle}>{t('outfitMaker.yourOutfits')}</ScaledText>
            <View style={{ width: 44 }} />
          </View>

          {planningDate && (
            <View style={{ paddingHorizontal: 20, paddingBottom: 4 }}>
              <View style={styles.planningBanner}>
                <View style={styles.planningIconWrap}>
                  <Ionicons name="calendar-outline" size={20} color="#173A65" />
                </View>
                <View style={{ flex: 1 }}>
                  <ScaledText style={styles.planningTitle} minScale={0.65}>
                    {t('outfitMaker.planningFor', {
                      defaultValue: 'Planning for {{date}}',
                      date: formatPlanningDate(planningDate),
                    })}
                  </ScaledText>
                  <ScaledText style={styles.planningSubtitle} minScale={0.65}>
                    {t('outfitMaker.planningSubtitle', {
                      defaultValue: 'Generated looks will be saved back to this day',
                    })}
                  </ScaledText>
                </View>
              </View>
            </View>
          )}

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 32 }}>
            {outfits.map((outfit, index) => (
              <Animated.View key={outfit.id} style={[styles.glassCard, { marginTop: index === 0 ? 8 : 24 }]}>
                <BlurView intensity={Platform.OS === 'ios' ? 48 : 90} tint="light" style={StyleSheet.absoluteFill} />
                <LinearGradient colors={['rgba(255,255,255,0.45)', 'rgba(255,255,255,0.15)']} style={StyleSheet.absoluteFill} />

                <View style={styles.cardHeaderRow}>
                  {outfit.weather ? (
                    <View style={styles.weatherChip}>
                      <Ionicons name="cloud-outline" size={14} color={D.textSecondary} />
                      <ScaledText style={styles.weatherChipText}>{outfit.weather.temp}°C · {outfit.weather.condition}</ScaledText>
                    </View>
                  ) : <View />}
                  <LinearGradient colors={[D.accent, '#5B7CF9']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.matchPill}>
                    <ScaledText style={styles.matchPillText}>{Math.round(outfit.matchScore * 100)}% Match</ScaledText>
                  </LinearGradient>
                </View>

                <OutfitSlotGrid items={outfit.items} weather={outfit.weather} />

                <ScaledText style={styles.outfitDesc} minScale={0.65}>{outfit.description}</ScaledText>

                <View style={styles.tipCard}>
                  <LinearGradient colors={D.panelHighlight} style={StyleSheet.absoluteFill} />
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <Ionicons name="sparkles-outline" size={16} color={D.accent} />
                    <ScaledText style={styles.tipCardText}>{outfit.stylingTips}</ScaledText>
                  </View>
                </View>

                <View style={styles.actionRow}>
                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={styles.primaryAction}
                    onPress={() => openCalendarForOutfit(outfit)}
                  >
                    <LinearGradient colors={[D.accent, '#5B7CF9']} style={StyleSheet.absoluteFill} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} />
                    <Ionicons name="calendar-outline" size={18} color="#fff" />
                    <ScaledText style={styles.primaryActionText} minScale={0.65}>
                      {planningDate
                        ? t('outfitMaker.planForDate', {
                            defaultValue: 'Plan for {{date}}',
                            date: formatPlanningDate(planningDate),
                          })
                        : t('outfitMaker.addToCalendar')}
                    </ScaledText>
                  </TouchableOpacity>

                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={styles.secondaryAction}
                    onPress={() => handleSaveOutfitPress(outfit)}
                  >
                    <Ionicons name="checkmark-circle-outline" size={18} color={D.textPrimary} />
                    <ScaledText style={styles.secondaryActionText} minScale={0.65}>
                      {planningDate
                        ? t('outfitMaker.saveAndConnectTrip', {
                            defaultValue: 'Save + Connect to trip',
                          })
                        : t('outfitMaker.saveOutfit')}
                    </ScaledText>
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
              <ScaledText style={styles.modalTitle} minScale={0.65}>{t('outfitMaker.saveToCalendar')}</ScaledText>
              <ScaledText style={styles.modalSubtitle} minScale={0.65}>{t('outfitMaker.pickDate')}</ScaledText>
              <View style={styles.datePickerRow}>
                <TouchableOpacity onPress={() => setCalendarDate(new Date(calendarDate.getTime() - 86400000))}>
                  <Ionicons name="chevron-back" size={24} color={D.textPrimary} />
                </TouchableOpacity>
                <ScaledText style={styles.datePickerValue} minScale={0.6}>{calendarDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</ScaledText>
                <TouchableOpacity onPress={() => setCalendarDate(new Date(calendarDate.getTime() + 86400000))}>
                  <Ionicons name="chevron-forward" size={24} color={D.textPrimary} />
                </TouchableOpacity>
              </View>
              <View style={styles.modalActions}>
                <TouchableOpacity style={styles.modalCancel} onPress={() => setCalendarVisible(false)}>
                  <ScaledText style={styles.modalCancelText} minScale={0.65}>{t('common.cancel')}</ScaledText>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.modalConfirm}
                  onPress={() => calendarOutfit && addToCalendar(calendarOutfit, calendarDate)}
                >
                  <LinearGradient colors={[D.accent, '#5B7CF9']} style={StyleSheet.absoluteFill} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} />
                  <ScaledText style={styles.modalConfirmText} minScale={0.65}>{t('common.save')}</ScaledText>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>

        {/* Collection Category Picker Modal */}
        <Modal animationType="slide" transparent visible={collectionPickerVisible} onRequestClose={() => setCollectionPickerVisible(false)}>
          <View style={styles.modalOverlay}>
            <BlurView intensity={Platform.OS === 'ios' ? 60 : 100} tint="light" style={StyleSheet.absoluteFill} />
            <View style={styles.modalCard}>
              <ScaledText style={styles.modalTitle} minScale={0.65}>{t('outfitMaker.chooseCollection')}</ScaledText>
              <ScaledText style={styles.modalSubtitle} minScale={0.65}>{t('outfitMaker.collectionSubtitle')}</ScaledText>
              <View style={styles.collectionGrid}>
                {COLLECTION_CATEGORIES.map((cat) => (
                  <TouchableOpacity
                    key={cat.id}
                    style={styles.collectionOption}
                    onPress={() => handleCollectionSelect(cat.id)}
                    activeOpacity={0.7}
                  >
                    <LinearGradient
                      colors={[D.accent + '20', D.accent + '10']}
                      style={StyleSheet.absoluteFill}
                      start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
                    />
                    <Ionicons name={cat.icon as any} size={24} color={D.accent} />
                    <ScaledText style={styles.collectionOptionText}>{cat.label}</ScaledText>
                  </TouchableOpacity>
                ))}
              </View>
              <TouchableOpacity style={styles.modalCancel} onPress={() => setCollectionPickerVisible(false)}>
                <ScaledText style={styles.modalCancelText}>{t('common.cancel')}</ScaledText>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
      </View>
    );
  }

  // ── AI thinking view (same page, after setup animates out) ──────────
  // Shows HOW the AI is thinking: quiz style → occasion → source scan.
  if (loading) {
    const selectedStyleLabel = aiStyles.find(s => s.id === selectedStyle)?.label || 'AI';
    const occasionLabel = OCCASION_PRESETS.find((o) => o.id === selectedOccasion)?.label;
    const sourceLabel = currentSource === 'shop'
      ? t('outfitMaker.newClothing', { defaultValue: 'new clothing' })
      : t('outfitMaker.fromWardrobe', { defaultValue: 'your wardrobe' });
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
        {/* Full-screen: no header — animation bleeds edge-to-edge */}
        <SafeAreaView edges={['bottom', 'left', 'right']} style={{ flex: 1 }}>
          <Reanimated.View entering={FadeIn.duration(350)} style={{ flex: 1, overflow: 'hidden' }}>
            <View style={{ flex: 1, overflow: 'hidden' }}>
              <AIThinkingAnimation styleName={selectedStyleLabel} clothingItems={animationItems} />
            </View>
            {/* Floating cancel — overlays animation, takes no layout space */}
            <SafeAreaView
              edges={['top', 'left', 'right']}
              style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 30 }}
              pointerEvents="box-none"
            >
              <View style={{ paddingHorizontal: 16, paddingTop: 8, alignItems: 'flex-start' }} pointerEvents="box-none">
                <TouchableOpacity
                  onPress={() => { setLoading(false); setOutfits([]); }}
                  style={styles.backButton}
                  hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                  accessibilityRole="button"
                  accessibilityLabel={t('common.back')}
                >
                  <Ionicons name="chevron-back" size={22} color="#0A1931" />
                </TouchableOpacity>
              </View>
            </SafeAreaView>
            {/* Thought-process context line: style → occasion → source */}
            <View style={styles.thinkingContextWrap} pointerEvents="none">
              <View style={styles.thinkingContextPill}>
                <Ionicons name="person-outline" size={13} color="#173A65" />
                <ScaledText style={styles.thinkingContextText} numberOfLines={1}>
                  {selectedStyleLabel}
                  {occasionLabel ? ` · ${occasionLabel}` : ''}
                  {` · ${sourceLabel}`}
                </ScaledText>
              </View>
              {(() => {
                const p = buildEffectivePrompt();
                return p ? (
                  <ScaledText style={styles.thinkingPromptText} numberOfLines={2}>
                    “{p}”
                  </ScaledText>
                ) : null;
              })()}
            </View>
          </Reanimated.View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
        style={StyleSheet.absoluteFill}
        pointerEvents="none"
      />
      <View pointerEvents="none" style={styles.backgroundOrbTop} />
      <View pointerEvents="none" style={styles.backgroundOrbBottom} />

      <SafeAreaView
        edges={['top', 'left', 'right']}
        style={{ flex: 1 }}
      >
        {/* Header — same 3-part centered pattern as main tabs */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={22} color="#0A1931" />
          </TouchableOpacity>
          <ScaledText style={styles.headerTitle}>{t('outfitMaker.aiStylist')}</ScaledText>
          <View style={styles.headerRight}>
            <Ionicons name="sparkles" size={20} color="#F59E0B" />
          </View>
        </View>

        {/* Hero — live summary of the look being built */}
        <View style={styles.heroBlock}>
          <ScaledText style={styles.heroTitle}>
            {t('outfitMaker.newOutfit', { defaultValue: 'New outfit' })}
          </ScaledText>
          <ScaledText style={styles.heroSub} numberOfLines={1}>
            {aiStyles.find((s) => s.id === selectedStyle)?.label}
            {' · '}
            {currentSource === 'shop'
              ? t('outfitMaker.newShort', { defaultValue: 'New' })
              : t('outfitMaker.wardrobeShort', { defaultValue: 'Wardrobe' })}
            {' · '}
            {selectedOccasion
              ? OCCASION_PRESETS.find((o) => o.id === selectedOccasion)?.label
              : t('outfitMaker.pickOccasion', { defaultValue: 'Pick an occasion' })}
          </ScaledText>
        </View>

        {/* Planning context — shown when arriving from OutfitCalendar's Plan Outfit (travel/day planning) */}
        {planningDate && (
          <View style={{ paddingHorizontal: 20, marginBottom: 16 }}>
            <View style={styles.planningBanner}>
              <View style={styles.planningIconWrap}>
                <Ionicons name="calendar-outline" size={20} color="#173A65" />
              </View>
              <View style={{ flex: 1 }}>
                <ScaledText style={styles.planningTitle} minScale={0.65}>
                  {t('outfitMaker.planningFor', {
                    defaultValue: 'Planning for {{date}}',
                    date: formatPlanningDate(planningDate),
                  })}
                </ScaledText>
                <ScaledText style={styles.planningSubtitle} minScale={0.65}>
                  {t('outfitMaker.planningSubtitle', {
                    defaultValue: 'Generated looks will be saved back to this day',
                  })}
                </ScaledText>
              </View>
              <View style={styles.planningBadge}>
                <Ionicons name="airplane-outline" size={12} color="#173A65" />
                <ScaledText style={styles.planningBadgeText} minScale={0.65}>
                  {t('outfitMaker.tripDay', { defaultValue: 'Trip day' })}
                </ScaledText>
              </View>
            </View>
          </View>
        )}

        {/* Anchor item preview — shown when navigated from MyClosetScreen */}
        {baseItem && (
          <View style={{ paddingHorizontal: 20, marginBottom: 16 }}>
            <View style={styles.anchorBanner}>
              {baseItem.imageUrl ? (
                <Image
                  source={{ uri: baseItem.imageUrl }}
                  style={styles.anchorImage}
                  resizeMode="cover"
                />
              ) : (
                <View style={[styles.anchorImage, { backgroundColor: '#F3F6FA', alignItems: 'center', justifyContent: 'center' }]}>
                  <Ionicons name="shirt-outline" size={24} color="#C8D0DA" />
                </View>
              )}
              <View style={{ flex: 1, marginLeft: 12 }}>
                <ScaledText style={styles.anchorLabel} minScale={0.65}>{t('outfitMaker.anchorItem', { defaultValue: 'Anchor item' })}</ScaledText>
                <ScaledText style={styles.anchorName} numberOfLines={1} minScale={0.65}>
                  {baseItem.name || baseItem.type || t('common.clothingItem')}
                </ScaledText>
              </View>
              <View style={styles.anchorBadge}>
                <Ionicons name="lock-closed" size={12} color="#007AFF" />
                <ScaledText style={styles.anchorBadgeText} minScale={0.65}>{t('outfitMaker.locked', { defaultValue: 'Locked' })}</ScaledText>
              </View>
            </View>
          </View>
        )}

        {/* ── Single-page setup: source + style + occasion ───────────── */}
        <Reanimated.View
          key="setup-single"
          entering={FadeIn.duration(300)}
          exiting={FadeOutUp.duration(280)}
          style={{ flex: 1 }}
        >
          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 16 }}>
            {/* Source — compact segmented control */}
            <View style={styles.sourceSegment}>
              {(
                [
                  {
                    id: 'wardrobe',
                    icon: 'shirt-outline',
                    label: t('outfitMaker.fromWardrobe', { defaultValue: 'From wardrobe' }),
                  },
                  {
                    id: 'shop',
                    icon: 'bag-handle-outline',
                    label: t('outfitMaker.newClothing', { defaultValue: 'New clothing' }),
                  },
                ] as const
              ).map((opt) => {
                const on = currentSource === opt.id;
                return (
                  <TouchableOpacity
                    key={opt.id}
                    activeOpacity={0.85}
                    onPress={() => {
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                      setCurrentSource(opt.id);
                    }}
                    style={[styles.sourceSegBtn, on && styles.sourceSegBtnActive]}
                  >
                    <Ionicons name={opt.icon as any} size={18} color={on ? '#FFFFFF' : '#0A1931'} />
                    <ScaledText style={[styles.sourceSegText, on && styles.sourceSegTextActive]}>
                      {opt.label}
                    </ScaledText>
                  </TouchableOpacity>
                );
              })}
            </View>

              {/* ── Where are you going? — 2 visible, rest in dropdown ─── */}
              <ScaledText style={[styles.setupSectionLabel, { marginTop: 24 }]}>
                {t('outfitMaker.whereGoing')}
              </ScaledText>
              {(() => {
                const quizIds = (Array.isArray(quizOccasions) ? quizOccasions : []).filter((id) =>
                  OCCASION_PRESETS.some((o) => o.id === id),
                );
                const fillPool: string[] = [...quizIds];
                for (const o of OCCASION_PRESETS) {
                  if (!fillPool.includes(o.id as string)) fillPool.push(o.id as string);
                }
                // Selected occasion always stays visible.
                let visibleIds = fillPool.slice(0, 2);
                if (selectedOccasion && !visibleIds.includes(selectedOccasion)) {
                  visibleIds = [selectedOccasion, ...fillPool.filter((id) => id !== selectedOccasion)].slice(0, 2);
                }
                const hiddenOnes = OCCASION_PRESETS.filter((o) => !visibleIds.includes(o.id as string));
                const visibleList = OCCASION_PRESETS.filter((o) => visibleIds.includes(o.id as string));
                const renderOccTile = (occ: (typeof OCCASION_PRESETS)[number]) => {
                  const on = selectedOccasion === occ.id;
                  return (
                    <TouchableOpacity
                      key={occ.id}
                      activeOpacity={0.85}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setSelectedOccasion(on ? undefined : (occ.id as string));
                      }}
                      style={[styles.bigTile, on && styles.bigTileSelected]}
                    >
                      <View style={[styles.bigTileIcon, on && styles.bigTileIconActive]}>
                        <Ionicons name={occ.icon as any} size={20} color={on ? '#FFFFFF' : '#173A65'} />
                      </View>
                      <ScaledText style={[styles.bigTileLabel, on && styles.bigTileLabelActive]}>
                        {occ.label}
                      </ScaledText>
                      {on && (
                        <View style={styles.bigTileCheck}>
                          <Ionicons name="checkmark" size={12} color="#FFFFFF" />
                        </View>
                      )}
                    </TouchableOpacity>
                  );
                };
                return (
                  <View>
                    <View style={styles.tileRow}>
                      {visibleList.map(renderOccTile)}
                    </View>
                    <TouchableOpacity
                      activeOpacity={0.75}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setShowAllOccasions((v) => !v);
                      }}
                      style={styles.moreToggle}
                      accessibilityRole="button"
                      accessibilityLabel={t('outfitMaker.moreOccasions', { defaultValue: 'More occasions' })}
                    >
                      <ScaledText style={styles.moreToggleText}>
                        {showAllOccasions
                          ? t('outfitMaker.lessOccasions', { defaultValue: 'Less' })
                          : t('outfitMaker.moreCount', {
                              defaultValue: 'More ({{count}})',
                              count: hiddenOnes.length,
                            })}
                      </ScaledText>
                      <Ionicons
                        name={showAllOccasions ? 'chevron-up' : 'chevron-down'}
                        size={16}
                        color="#173A65"
                      />
                    </TouchableOpacity>
                    {showAllOccasions && (
                      <Reanimated.View entering={FadeInUp.duration(250)} exiting={FadeOut.duration(200)}>
                        <View style={styles.tileRow}>
                          {hiddenOnes.map(renderOccTile)}
                        </View>
                      </Reanimated.View>
                    )}
                  </View>
                );
              })()}

              {/* ── Your style — one featured card ─────────────────────── */}
              <ScaledText style={[styles.setupSectionLabel, { marginTop: 24 }]}>
                {t('outfitMaker.yourStyle', { defaultValue: 'Your style' })}
              </ScaledText>
              {(() => {
                const allStyles = aiStyles.filter((s) => s.id !== 'business_casual');
                // Quiz style first, then list order.
                const fillPool = [...allStyles].sort((a, b) => {
                  if (a.id === quizStyleDefault) return -1;
                  if (b.id === quizStyleDefault) return 1;
                  return 0;
                });
                // Selected style always stays visible.
                let visibleStyles = fillPool.slice(0, 1);
                if (selectedStyle && !visibleStyles.some((s) => s.id === selectedStyle)) {
                  visibleStyles = [
                    fillPool.find((s) => s.id === selectedStyle) || fillPool[0],
                    ...fillPool.filter((s) => s.id !== selectedStyle),
                  ].slice(0, 1);
                }
                const hiddenStyles = allStyles.filter((s) => !visibleStyles.some((v) => v.id === s.id));
                const activeStyle = visibleStyles[0] || fillPool[0];
                return (
                  <View>
                    {/* Featured style card — the one defining your look */}
                    <View style={styles.featureCard}>
                      <View style={styles.featureIcon}>
                        <Ionicons name={activeStyle.icon as any} size={26} color="#FFFFFF" />
                      </View>
                      <View style={{ flex: 1, marginLeft: 14 }}>
                        <ScaledText style={styles.featureName}>{activeStyle.label}</ScaledText>
                        {quizStyleDefault === activeStyle.id && quizPersonality ? (
                          <View style={[styles.quizBadge, { alignSelf: 'flex-start', marginTop: 6 }]}>
                            <Ionicons name="sparkles" size={11} color="#FFFFFF" />
                            <ScaledText style={styles.quizBadgeText}>
                              {t('outfitMaker.fromQuiz', { defaultValue: 'From quiz' })}
                            </ScaledText>
                          </View>
                        ) : null}
                      </View>
                      <View style={styles.featureRadio}>
                        <View style={styles.featureRadioDot} />
                      </View>
                    </View>
                    <TouchableOpacity
                      activeOpacity={0.75}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setShowAllStyles((v) => !v);
                      }}
                      style={styles.moreToggle}
                      accessibilityRole="button"
                      accessibilityLabel={t('outfitMaker.moreStyles', { defaultValue: 'More styles' })}
                    >
                      <ScaledText style={styles.moreToggleText}>
                        {showAllStyles
                          ? t('outfitMaker.lessOccasions', { defaultValue: 'Less' })
                          : t('outfitMaker.moreCount', {
                              defaultValue: 'More ({{count}})',
                              count: hiddenStyles.length,
                            })}
                      </ScaledText>
                      <Ionicons
                        name={showAllStyles ? 'chevron-up' : 'chevron-down'}
                        size={16}
                        color="#173A65"
                      />
                    </TouchableOpacity>
                    {showAllStyles && (
                      <Reanimated.View entering={FadeInUp.duration(250)} exiting={FadeOut.duration(200)}>
                        <View style={styles.formGroup}>
                          {hiddenStyles.map((styleObj, i) => {
                            const isSelected = selectedStyle === styleObj.id;
                            return renderCheckRow(
                              styleObj.id,
                              styleObj.icon,
                              styleObj.label,
                              isSelected,
                              () => setSelectedStyle(styleObj.id),
                              i > 0,
                            );
                          })}
                        </View>
                      </Reanimated.View>
                    )}
                  </View>
                );
              })()}

              {/* Details input — grouped text field, no header */}
              <View style={[styles.formGroup, { marginTop: 24 }]}>
                <TextInput
                  style={styles.detailsInput}
                  placeholder={t('outfitMaker.promptPlaceholderShort', {
                    defaultValue: 'Add details…',
                  })}
                  placeholderTextColor={D.textSecondary}
                  value={userPrompt}
                  onChangeText={setUserPrompt}
                  multiline
                  numberOfLines={3}
                  textAlignVertical="top"
                  returnKeyType="done"
                  onSubmitEditing={() => handleGeneratePress()}
                />
              </View>

              {error ? (
                <View style={[styles.errorBox, { marginTop: 16 }]}>
                  <ScaledText style={styles.errorText}>{error}</ScaledText>
                </View>
              ) : null}
            </ScrollView>
          </Reanimated.View>
          {/* Sticky bottom CTA */}
          <View style={styles.stickyFooter}>
            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => handleGeneratePress()}
              style={styles.generateButton}
            >
              <Ionicons name="sparkles" size={17} color="#FFF" />
              <ScaledText style={styles.generateButtonText}>
                {t('outfitMaker.generateOutfit', { defaultValue: 'Generate outfit' })}
              </ScaledText>
            </TouchableOpacity>
          </View>
      </SafeAreaView>

    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  // Same page background as the 4 main tab screens
  backgroundOrbTop: {
    position: 'absolute',
    top: -100,
    right: -80,
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: 'rgba(188, 210, 245, 0.42)',
  },
  backgroundOrbBottom: {
    position: 'absolute',
    left: -120,
    bottom: 140,
    width: 300,
    height: 300,
    borderRadius: 150,
    backgroundColor: 'rgba(216, 229, 252, 0.34)',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.82)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 3,
  },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '700',
    color: '#0A1931',
    letterSpacing: 0.3,
    textAlign: 'center',
  },
  headerRight: {
    width: 44,
    alignItems: 'flex-end',
    justifyContent: 'center',
  },
  sectionTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
  },
  sectionSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodySmall,
    color: LiquidGlass2026Theme.colors.text.secondary,
    marginTop: 4,
    lineHeight: 18,
  },

  // Anchor item banner (shown when navigated from MyClosetScreen)
  anchorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 122, 255, 0.07)',
    borderWidth: 1,
    borderColor: 'rgba(0, 122, 255, 0.2)',
    borderRadius: 16,
    padding: 12,
  },
  anchorImage: {
    width: 52,
    height: 52,
    borderRadius: 10,
    overflow: 'hidden',
  },
  anchorLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: '#007AFF',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 2,
  },
  anchorName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#0A1931',
  },
  anchorBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(0, 122, 255, 0.12)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  anchorBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#007AFF',
  },

  // Planning context banner (shown when arriving from OutfitCalendar Plan Outfit)
  planningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.88)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    borderRadius: 16,
    padding: 12,
    gap: 12,
  },
  planningIconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: 'rgba(23,58,101,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  planningTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#0A1931',
  },
  planningSubtitle: {
    fontSize: 12,
    fontWeight: '500',
    color: '#4B5563',
    marginTop: 2,
  },
  planningBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(23,58,101,0.08)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  planningBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#173A65',
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
  // Minimal details input — flat fill inside the sheet
  promptCardMinimal: {
    borderRadius: 20,
    backgroundColor: '#F4F7FB',
    borderWidth: 1,
    borderColor: 'transparent',
    paddingHorizontal: 18,
    paddingVertical: 14,
    marginTop: 0,
  },
  promptHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  promptTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: LiquidGlass2026Theme.colors.text.primary,
  },
  promptInput: {
    fontSize: 14,
    color: LiquidGlass2026Theme.colors.text.primary,
    lineHeight: 20,
    minHeight: 48,
    padding: 0,
  },
  promptSubmitButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  promptSubmitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginTop: 10,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 12,
    overflow: 'hidden',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
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
    padding: 14,
    paddingRight: 18,
  },
  vibeIconWrap: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: '#0A1931',
    alignItems: 'center',
    justifyContent: 'center',
  },
  vibeTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0A1931',
    marginBottom: 4,
    letterSpacing: -0.2,
  },
  vibeDesc: {
    fontSize: 13,
    color: '#4B5563',
    lineHeight: 18,
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
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    marginBottom: 12,
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
    ...LiquidGlass2026Theme.typography.scale.bodySmall,
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
    gap: 6,
    paddingVertical: 12,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
  },
  primaryActionText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '700',
    fontSize: 13,
  },
  secondaryAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    backgroundColor: 'rgba(255,255,255,0.55)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  secondaryActionText: {
    color: LiquidGlass2026Theme.colors.text.primary,
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontSize: 13,
    fontWeight: '600',
  },

  // ── Calendar Modal Styles ──────────────────────────────────────
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalCard: {
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 28,
    marginHorizontal: 0,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
  modalTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    textAlign: 'center',
    marginBottom: 2,
  },
  modalSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodySmall,
    color: LiquidGlass2026Theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: 16,
  },
  datePickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255,255,255,0.6)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  datePickerValue: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  modalCancel: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    backgroundColor: 'rgba(255,255,255,0.6)',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  modalCancelText: {
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontSize: 13,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  modalConfirm: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
    alignItems: 'center',
  },
  modalConfirmText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontSize: 13,
    fontWeight: '700',
  },
  collectionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    justifyContent: 'center',
    paddingVertical: 16,
  },
  collectionOption: {
    width: (Dimensions.get('window').width - 112) / 3 - 6,
    paddingVertical: 18,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
    gap: 6,
  },
  collectionOptionText: {
    fontSize: 13,
    fontWeight: '600',
    color: LiquidGlass2026Theme.colors.text.primary,
    textAlign: 'center',
  },

  // ── New staged Create flow ─────────────────────────────────────────
  stepDotsRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  stepDotWrap: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stepDot: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(10,25,49,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepDotActive: {
    backgroundColor: '#0A1931',
  },
  stepDotDone: {
    backgroundColor: '#10B981',
  },
  stepDotNum: {
    fontSize: 12,
    fontWeight: '700',
    color: '#64748B',
  },
  stepDotNumActive: {
    color: '#FFF',
  },
  stepDotLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#94A3B8',
  },
  stepDotLabelActive: {
    color: '#0A1931',
  },
  stepDotLine: {
    flex: 1,
    height: 1.5,
    backgroundColor: 'rgba(10,25,49,0.1)',
    marginHorizontal: 8,
  },
  setupSectionLabel: {
    fontSize: 13,
    fontWeight: '500',
    color: '#8E8E93',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 8,
    marginTop: 6,
  },
  // Hero header with live look summary
  heroBlock: {
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 18,
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: '#0A1931',
    letterSpacing: -0.8,
  },
  heroSub: {
    fontSize: 13,
    fontWeight: '500',
    color: '#8E8E93',
    marginTop: 4,
  },
  // Bold occasion tiles (2-up row, wraps to grid when expanded)
  tileRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  bigTile: {
    flexGrow: 1,
    flexBasis: '46%',
    minHeight: 88,
    backgroundColor: '#FFFFFF',
    borderRadius: 22,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.7)',
    padding: 14,
    justifyContent: 'space-between',
    gap: 10,
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 5,
  },
  bigTileSelected: {
    backgroundColor: '#0A1931',
    borderColor: '#0A1931',
    shadowOpacity: 0.22,
  },
  bigTileIcon: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: 'rgba(23,58,101,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  bigTileIconActive: {
    backgroundColor: 'rgba(255,255,255,0.16)',
  },
  bigTileLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: '#0A1931',
    letterSpacing: -0.2,
  },
  bigTileLabelActive: {
    color: '#FFFFFF',
  },
  bigTileCheck: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Featured style card
  featureCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0A1931',
    borderRadius: 24,
    padding: 16,
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 8,
  },
  featureIcon: {
    width: 56,
    height: 56,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.14)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureName: {
    fontSize: 18,
    fontWeight: '800',
    color: '#FFFFFF',
    letterSpacing: -0.3,
  },
  featureRadio: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureRadioDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#FFFFFF',
  },
  // iOS grouped-list section card
  formGroup: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(23,58,101,0.1)',
    overflow: 'hidden',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.06,
    shadowRadius: 16,
    elevation: 4,
  },
  formRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    minHeight: 54,
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  formRowIcon: {
    width: 30,
    height: 30,
    borderRadius: 9,
    backgroundColor: 'rgba(23,58,101,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  formRowLabel: {
    fontSize: 16,
    color: '#0A1931',
  },
  rowDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(23,58,101,0.1)',
    marginLeft: 58,
  },
  detailsInput: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    lineHeight: 22,
    minHeight: 88,
    color: '#0A1931',
  },
  setupSectionHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 0,
  },
  // One floating sheet holds the whole form; hairline dividers split sections
  sheetCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 28,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.7)',
    padding: 18,
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.1,
    shadowRadius: 24,
    elevation: 8,
  },
  sheetDivider: {
    height: 1,
    backgroundColor: '#EDF1F6',
    marginVertical: 20,
  },
  // Pinned bottom CTA bar
  stickyFooter: {
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 20,
    backgroundColor: 'rgba(255,255,255,0.94)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(23,58,101,0.08)',
  },
  sourceRow: {
    flexDirection: 'row',
    gap: 10,
  },
  sourceCard: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.7)',
    padding: 16,
    alignItems: 'flex-start',
    position: 'relative',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 5,
  },
  sourceCardSelected: {
    borderColor: '#173A65',
    borderWidth: 2,
    backgroundColor: '#FFFFFF',
    shadowOpacity: 0.14,
  },
  sourceIconWrap: {
    width: 52,
    height: 52,
    borderRadius: 17,
    backgroundColor: 'rgba(23,58,101,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  sourceIconWrapSelected: {
    backgroundColor: '#173A65',
  },
  sourceTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#0A1931',
    letterSpacing: -0.2,
  },
  sourceTitleSelected: {
    color: '#0A1931',
  },
  sourceCount: {
    fontSize: 12,
    color: '#64748B',
    marginTop: 2,
  },
  sourceCheck: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#173A65',
    alignItems: 'center',
    justifyContent: 'center',
  },
  quizBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#0A1931',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 99,
  },
  quizBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  vibeCardSelected: {
    borderColor: '#0A1931',
    borderWidth: 1.5,
    backgroundColor: 'rgba(10,25,49,0.04)',
  },
  vibeIconWrapSelected: {
    backgroundColor: '#2B5CE9',
  },
  quizDot: {
    backgroundColor: 'rgba(23,58,101,0.1)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
  },
  quizDotText: {
    fontSize: 9,
    fontWeight: '800',
    color: '#173A65',
    letterSpacing: 0.5,
  },
  radioOuter: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1.5,
    borderColor: '#CBD5E1',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
  },
  radioOuterSelected: {
    borderColor: '#0A1931',
  },
  radioInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#0A1931',
  },
  // Flat style rows: icon + name + radio (single tap, no dropdown)
  styleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#F4F7FB',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'transparent',
    paddingHorizontal: 14,
    paddingVertical: 13,
    marginBottom: 10,
  },
  styleRowSelected: {
    borderColor: '#0A1931',
    backgroundColor: '#0A1931',
    shadowOpacity: 0.2,
    shadowRadius: 16,
  },
  styleRowIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(23,58,101,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  styleRowIconSelected: {
    backgroundColor: 'rgba(255,255,255,0.16)',
  },
  styleRowTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0A1931',
  },
  styleRowTitleSelected: {
    fontWeight: '700',
    color: '#FFFFFF',
  },
  changeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 4,
    marginTop: 8,
    backgroundColor: 'rgba(43,92,233,0.1)',
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(43,92,233,0.25)',
  },
  changeButtonText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#2B5CE9',
  },
  styleDropdown: {
    marginTop: 10,
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    overflow: 'hidden',
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 16,
    elevation: 4,
  },
  styleDropdownItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 13,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },
  styleDropdownIcon: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: '#0A1931',
    alignItems: 'center',
    justifyContent: 'center',
  },
  styleDropdownTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#0A1931',
  },
  styleDropdownDesc: {
    fontSize: 12,
    color: '#64748B',
    lineHeight: 16,
    marginTop: 2,
  },
  continueButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 20,
    paddingVertical: 16,
    borderRadius: 26,
    overflow: 'hidden',
  },
  continueButtonText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#FFF',
  },
  recapPill: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(16,185,129,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(16,185,129,0.25)',
    borderRadius: 16,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginBottom: 18,
  },
  recapLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  recapText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#0A1931',
    flex: 1,
  },
  recapEdit: {
    fontSize: 13,
    fontWeight: '700',
    color: '#2B5CE9',
  },
  occasionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },
  occasionChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    flexGrow: 1,
    flexBasis: '46%',
    minHeight: 62,
    backgroundColor: '#F4F7FB',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'transparent',
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  occasionChipSelected: {
    backgroundColor: '#0A1931',
    borderColor: '#0A1931',
    shadowOpacity: 0.2,
    shadowRadius: 16,
  },
  occasionTileIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(23,58,101,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  occasionTileIconActive: {
    backgroundColor: 'rgba(255,255,255,0.16)',
  },
  occasionEmoji: {
    fontSize: 16,
  },
  occasionChipText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0A1931',
    letterSpacing: -0.1,
  },
  occasionChipTextSelected: {
    color: '#FFF',
  },
  occasionQuizDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#2B5CE9',
  },
  // Centered inline expander row inside a form group
  moreToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: 13,
  },
  // Segmented source control (main-tabs pattern)
  sourceSegment: {
    flexDirection: 'row',
    backgroundColor: '#F1F5FA',
    borderRadius: 99,
    padding: 4,
    gap: 4,
  },
  sourceSegBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    borderRadius: 99,
  },
  sourceSegBtnActive: {
    backgroundColor: '#0A1931',
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.22,
    shadowRadius: 12,
    elevation: 4,
  },
  sourceSegText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#0A1931',
  },
  sourceSegTextActive: {
    color: '#FFFFFF',
  },
  moreToggleText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#173A65',
  },
  // Compact occasion dropdown: bold box, 2 quick picks, rest expandable
  occasionBox: {
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#0A1931',
    padding: 12,
  },
  occasionQuickRow: {
    flexDirection: 'row',
    gap: 10,
  },
  occasionQuickBtn: {
    flex: 1,
    justifyContent: 'center',
  },
  occasionDivider: {
    height: 1,
    backgroundColor: '#F1F5F9',
    marginVertical: 10,
  },
  occasionMoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    paddingVertical: 4,
  },
  occasionMoreLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  occasionMoreText: {
    fontSize: 13,
    fontWeight: '700',
    color: '#2B5CE9',
  },
  occasionMoreSelected: {
    fontSize: 14,
    fontWeight: '700',
    color: '#0A1931',
  },
  occasionExpandedList: {
    marginTop: 6,
    gap: 2,
  },
  occasionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 9,
    paddingHorizontal: 4,
  },
  occasionActions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
    alignItems: 'center',
  },
  ghostBackButton: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
  },
  generateButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    minHeight: 58,
    paddingVertical: 16,
    borderRadius: 29,
    backgroundColor: '#0A1931',
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.25,
    shadowRadius: 20,
    elevation: 8,
  },
  generateButtonText: {
    fontSize: 17,
    fontWeight: '700',
    color: '#FFF',
    letterSpacing: -0.2,
  },
  thinkingContextWrap: {
    position: 'absolute',
    bottom: 62,
    left: 20,
    right: 20,
    alignItems: 'center',
    gap: 8,
  },
  thinkingContextPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.12)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    maxWidth: '100%',
  },
  thinkingContextText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#0A1931',
    flexShrink: 1,
  },
  thinkingPromptText: {
    fontSize: 13,
    fontStyle: 'italic',
    color: '#64748B',
    textAlign: 'center',
  },
});

export default AIOutfitGenerator;
