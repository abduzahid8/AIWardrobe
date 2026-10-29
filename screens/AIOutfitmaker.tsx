import {formatUzsPrice} from '../src/utils/uzsPrice';
import { useUzsDisplayRates, useUzsPrice } from '../hooks/useUzsPrice';
import { DISPLAY_DECK_SIZE, resolveOutfitComposition, resolveStylistWeather } from '../backend/supabase/functions/_shared/outfitComposition';
import { getBudgetRates } from '../src/services/budgetRates';
import { budgetFxContext, fitsBudgetAmount } from '../backend/supabase/functions/_shared/budgetFx';
import readinessLogger from '../src/utils/logger';
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { View, TouchableOpacity, ScrollView, Dimensions, StyleSheet, Animated, Platform, Modal, TextInput, Alert, Linking, KeyboardAvoidingView } from 'react-native';
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
import { withRequestDeadline } from '../src/lib/requestDeadline';
import { OutfitGenerationFailure, outfitGenerationFailure } from '../src/services/outfitGenerationFailure';
import { useShopCatalog } from '../hooks/useShopCatalog';
import { fillMissingSlots, type OutfitSlotId } from '../src/services/shoppingService';
import { getRecentShownIds, getRecentShownProductKeys, getRotationHistory, makeRotationScope, recordShownOutfits } from '../src/services/recentOutfitIds';
import { mergeShopSlices, collapseSimilarByName, collapseDuplicateProductImages, isTrustedProductImage, isBrandCatalogImage, productImageKey, preferMappedMacro, hasUsableProductName, inferCatalogColor, normalizeCatalogNameKey, classicCatalogOffset, garmentVarietyFamily, inferShopRowMacro } from '../src/utils/shopCatalogOrder';
import { passesStyleJudge, styleJudgeReason, rejectForStyleOccasion, rejectForClimate, humanizeProductName, SHORT_SLEEVE_RE, STRIP_SHORT_SLEEVE_RE } from '../features/outfit-generator/utils/occasionRules';
import LiquidGlass2026Theme, { SpatialElevation } from '../constants/LiquidGlass2026Theme';
import * as Location from 'expo-location';
import Config from '../src/config/env';
import { saveCloudOutfit } from '../src/services/savedOutfitService';
import { confirmedImageReplacements, replaceGeneratedOutfitImages, replaceSavedOutfitImages } from '../src/utils/confirmedOutfitMedia';
import { localDateKey, parseCalendarDate, shiftCalendarDay } from '../src/utils/calendarDate';
import { authEvents } from '../store/authEvents';
import { readCalendarLogs, updateCalendarLogs } from '../src/services/calendarStorage';
import { createOutfitLog, resolveCalendarOccasion, type OutfitItem as CalendarOutfitItem } from '../features/calendar/types';
import AIThinkingAnimation from '../components/AIThinkingAnimation';
import { canonicalizeMacroCategory } from '../src/utils/categoryMapper';
import { prioritizeUnseen } from '../features/outfit-generator/utils/deckDedupe';
import {
    analyzeDeckProductDiversity,
    assembleDiverseDeck,
    canonicalProductKey,
    diversitySlotOf,
    productIdentityKeys,
    type DiversitySlot,
} from '../features/outfit-generator/utils/outfitDiversity';
import { isShortsBottom, isFormalLayer, fabricFamily, isSameLayerBaseCategory, PLACEHOLDER_PANTS, PLACEHOLDER_SHOES, getBackendMacroCategory, isCompleteNormalizedOutfit, isFourSlotLook, REAL_IMAGE_RE, repairEdgeItems, createSlotNormalizer, ensureSlotFloors } from '../features/outfit-generator/utils/normalizeSlots';
import { parseBudgetFromText, stripBudgetFromText, outfitShopSpend, formatMoney, budgetCandidatesFromText, resolveShopQuote, withOutfitBudget } from '../src/utils/outfitBudget';
import { getShopDestination } from '../src/utils/shopDestination';
import { parseStylistIntent } from '../features/outfit-generator/utils/stylistIntent';
import { scoreItemForStyle, normalizeStyleId } from '../features/outfit-generator/utils/styleInference';
import useWardrobeStore from '../store/wardrobeStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { useTranslation } from 'react-i18next';
import useAuthStore from '../store/auth';
import { SHOP_CATALOG_ITEMS } from '../data/shopCatalogItems';
import { CLASSIC_MENS_ITEMS } from '../data/inspoMensShopItems';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';
import { Image, ExpoImage } from '../src/components/MediaImage';


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
  price?: number;
  currency?: string;
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

// ── Static catalog fallback, normalized ONCE ────────────────────────────
// candidatePool re-ran toMacro regexes over ~2k bundled items on every
// wardrobe change. Static data never changes — normalize once at module
// scope and reuse. Each entry carries its macro + precomputed image flag.
// Stylist rule-set version — printed on every Generate tap log so a pasted
// log proves exactly which vetoes/picks were live in that build.
export const STYLIST_RULES_VERSION = '2026-10-06A';
interface StaticPoolItem {
  id: string;
  name: string;
  image: string;
  imageUrl: string;
  type: string;
  macroCategory: string;
  color: string;
  brand: string;
  isWardrobe: boolean;
  isShopItem: boolean;
  shopUrl?: string;
  price?: number;
  currency?: string;
  _hasImage: boolean;
}
let staticPoolCache: StaticPoolItem[] | null = null;
// Slot mapping for every shop_catalog row. See inferShopRowMacro.
function shopRowMacro(garmentType: string, name: string, desc: string): string {
  return inferShopRowMacro(garmentType, name, desc);
}

function mapShopCatalogRow(item: {
  id: string;
  name?: string;
  brand?: string;
  description?: string;
  imageUrl?: string;
  garmentType?: string;
  sourceUrl?: string;
  price?: number;
  currency?: string;
}): {
  id: string;
  image: string;
  imageUrl: string;
  type: string;
  category: string;
  macroCategory: string;
  name: string;
  brand: string;
  description: string;
  shopUrl?: string;
  isShopItem: boolean;
  price?: number;
  currency?: string;
  color?: string;
} {
  const macroCategory = shopRowMacro(item.garmentType || '', item.name || '', item.description || '');
  // Humanize feed captions once here ("Front view of a ... - Image 0" never
  // reaches cards or the edge prompt). Keywords survive (only boilerplate is
  // stripped), so scoring is unaffected.
  const displayName = humanizeProductName(item.name) || item.name || 'Clothing Piece';
  return {
    id: item.id,
    image: item.imageUrl || '',
    imageUrl: item.imageUrl || '',
    // 'other' (accessories) keeps non-clothing type/category so downstream
    // keyword fallbacks can never re-derive it as shoes/tops.
    type: macroCategory === 'other' ? 'accessories' : macroCategory === 'outerwear' ? 'outerwear' : item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : 'shoes',
    category: macroCategory === 'other' ? 'accessories' : macroCategory === 'outerwear' ? 'outerwear' : item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : 'shoes',
    macroCategory,
    name: displayName,
    brand: item.brand || '',
    description: item.description || (item.name ? `${item.name} by ${item.brand || ''}` : ''),
    shopUrl: item.sourceUrl,
    isShopItem: true,
    // Infer from name/description (the table has no color column) so
    // color-aware rules don't judge blind. Mirrors edge inferShopColor.
    color: inferCatalogColor(item.name, item.description),
    price: typeof item.price === 'number' ? item.price : undefined,
    currency: item.currency,
  };
}

function getStaticPoolItems(): StaticPoolItem[] {
  if (staticPoolCache) return staticPoolCache;
  const out: StaticPoolItem[] = [];
  for (const s of [...(SHOP_CATALOG_ITEMS as any[]), ...(CLASSIC_MENS_ITEMS as any[])]) {
    const id = String((s as any)?.id || '');
    if (!id) continue;
    const macro = shopRowMacro((s as any).garmentType, (s as any).name || '', (s as any).description || '');
    const imageUrl = (s as any).imageUrl || '';
    out.push({
      id,
      name: (s as any).name || 'Shop item',
      image: imageUrl,
      imageUrl,
      type: macro,
      macroCategory: macro,
      color: inferCatalogColor((s as any).name, (s as any).description),
      brand: (s as any).brand || '',
      isWardrobe: false,
      isShopItem: true,
      shopUrl: (s as any).sourceUrl || undefined,
      price: typeof (s as any).price === 'number' ? (s as any).price : undefined,
      currency: (s as any).currency || undefined,
      _hasImage: REAL_IMAGE_RE.test(imageUrl),
    });
  }
  // Collapse same-name ingestions (classic-m shoe-16 vs shoe-17) so one
  // product can't occupy several elite slots.
  staticPoolCache = collapseSimilarByName(out);
  return staticPoolCache;
}

const PLACEHOLDER_SHOES_IMAGE = require('../assets/images/basic_brown_loafers.png');
const PLACEHOLDER_PANTS_IMAGE = require('../assets/images/basic_brown_pants.png');
const PLACEHOLDER_TOP_IMAGE = require('../assets/images/basic_white_tshirt.png');
const PLACEHOLDER_LAYER_IMAGE = require('../assets/images/basic_zip_hoodie.png');

const getSlotImageSource = (item?: OutfitItem): { uri: string } => {
  const raw = item?.imageUrl || (typeof item?.image === 'string' ? item.image : '');
  return { uri: /^(https?:|file:|data:|asset:|content:)/i.test(raw || '') ? raw! : '' };
};

const getOfflineMacroCategory = (type: string, category?: string, name?: string) => {
  // Outerwear keywords win over a generic alias. A blazer stored under
  // category "top"/"tops" must still resolve to "outerwear" (main-top/layer),
  // otherwise the anchor gets locked into the wrong (second-top) slot.
  const t = `${type || ''} ${category || ''} ${name || ''}`.toLowerCase();
  if (SHORT_SLEEVE_RE.test(t)) return 'top';
  if (t.match(/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat|bomber)\b/)) return 'outerwear';

  // Then honor any canonical macroCategory alias present in either string
  // (e.g. shop_catalog's `upper_body` / `lower_body`). Falling back to
  // keyword matching afterwards.
  const aliasHit = canonicalizeMacroCategory(type) !== 'other'
    ? canonicalizeMacroCategory(type)
    : canonicalizeMacroCategory(category || '');
  if (aliasHit !== 'other') return aliasHit;

  if (t.match(/\b(pant|trouser|jeans?|bottom|shorts?|skirt|lower[_\s-]?body)\b/) && !SHORT_SLEEVE_RE.test(t)) return 'bottom';
  if (t.match(/\b(shoe|sneaker|boot|loafer|sandal|footwear|trainer|derby|derbies|moccasin|driver|mule|clog)\b/)) return 'shoes';
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

/** Human-readable style name for user-facing copy ("old_money" -> "Old Money"). */
function styleLabelFor(styleId: string): string {
  const known = aiStyles.find((s) => s.id === styleId)?.label;
  if (known) return known;
  return String(styleId || 'casual')
    .split('_')
    .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(' ');
}

/** "An Old Money" / "A Casual" — description lead-in with correct article. */
function stylePhraseFor(styleId: string): string {
  const label = styleLabelFor(styleId);
  return `${/^[aeiou]/i.test(label) ? 'An' : 'A'} ${label}`;
}

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
  /** Shop catalog rows (with price/currency) so tiles can show listed prices. */
  priceCatalog?: any[];
  /** New-clothing results must give every displayed item a purchase action. */
  showBuyButtons?: boolean;
}

// Memoized slot cell with expo-image memory-disk caching. Previously each
// cell used RN <Image> (re-decoded on every slotConfig change) with an
// inline onPress closure — re-rendering + re-decoding all 4 tiles whenever
// any slot changed (flicker + dropped frames during generation streaming).
const SlotCell = React.memo(function SlotCell({
  slotKey,
  label,
  name,
  price,
  priceCurrency,
  imageSource,
  imageUri,
  itemId,
  buyUrl,
  showBuyButton,
  buyLabel,
  textSecondary,
  textPrimary,
  onBuy,
}: {
  slotKey: string;
  label: string;
  name: string;
  price?: number;
  priceCurrency?: string;
  imageSource: number | { uri: string };
  imageUri?: string;
  itemId?: string;
  buyUrl: string;
  showBuyButton: boolean;
  buyLabel: string;
  textSecondary: string;
  textPrimary: string;
  onBuy: (url: string) => void;
}) {
  const displayPrice = useUzsPrice(price, priceCurrency);
  const handleBuy = React.useCallback(() => onBuy(buyUrl), [onBuy, buyUrl]);
  const [imgFailed, setImgFailed] = React.useState(false);
  React.useEffect(() => { setImgFailed(false); }, [imageUri, slotKey]);
  return (
    <View key={slotKey} style={slotStyles.cell}>
      <ExpoImage
        source={imgFailed ? undefined : imageSource}
        placeholderContentFit="contain"
        style={[slotStyles.image, { backgroundColor: '#EEF3F8' }]}
        contentFit="cover"
        cachePolicy="memory-disk"
        recyclingKey={`${slotKey}-${itemId ?? ''}-${imageUri ?? ''}-${imgFailed ? 'fb' : 'src'}`}
        onError={() => setImgFailed(true)}
      />
      <View style={slotStyles.labelWrap}>
        <ScaledText style={[slotStyles.label, { color: textSecondary }]}>{label}</ScaledText>
        <ScaledText style={[slotStyles.name, { color: textPrimary }]} numberOfLines={1}>{name}</ScaledText>
        {typeof price === 'number' && price > 0 && priceCurrency ? (
          <ScaledText style={slotStyles.price} numberOfLines={1}>
            {displayPrice || '— UZS'}
          </ScaledText>
        ) : null}
        {showBuyButton ? (
          <TouchableOpacity
            style={slotStyles.buyButton}
            onPress={handleBuy}
            activeOpacity={0.75}
            accessibilityRole="link"
            accessibilityLabel={`${buyLabel} ${name}`}
          >
            <Ionicons name="cart-outline" size={11} color="#FFFFFF" />
            <ScaledText style={slotStyles.buyButtonText}>{buyLabel}</ScaledText>
          </TouchableOpacity>
        ) : null}
      </View>
    </View>
  );
});

function OutfitSlotGrid({ items, weather, priceCatalog, showBuyButtons = false }: OutfitSlotGridProps) {
  const D = useDesignTokens();
  const { t } = useTranslation();

  // Render the validated snapshot verbatim. Never replace shorts, recolor
  // garments, or add a stock product while rendering a result.
  const slotConfig = React.useMemo(() => items.map(item => {
    const macro = canonicalizeMacroCategory(item.macroCategory || item.type || '');
    const labels: Record<string, string> = { top: 'Top', outerwear: 'Layer', bottom: 'Bottom', shoes: 'Shoes' };
    const keys: Record<string, string> = { top: 'second-top', outerwear: 'main-top', bottom: 'pants', shoes: 'shoes' };
    return { key: `${keys[macro] || macro}-${item.id}`, label: labels[macro] || 'Item', item };
  }), [items]);

  const handleBuy = useCallback(async (url: string) => {
    try {
      await Linking.openURL(url);
    } catch (error) {
      console.warn('[stylist-buy] Could not open destination', error);
      Alert.alert('Could not open store', 'Please try again or open the product in your browser.');
    }
  }, []);

  const buyLabel = t('outfitMaker.buy', 'Buy');
  return (
    <View style={slotStyles.grid}>
      {slotConfig.filter((slot) => !!slot.item?.id && !!slot.item?.name).map((slot) => {
        const src = getSlotImageSource(slot.item);
        const quote = resolveShopQuote(slot.item, priceCatalog);
        const buyUrl = getShopDestination({
          brand: slot.item?.brand,
          name: slot.item?.name,
          shopUrl: slot.item?.shopUrl,
        });
        return (
          <SlotCell
            key={slot.key}
            slotKey={slot.key}
            label={slot.label}
            name={slot.item?.name || ''}
            price={quote?.price}
            priceCurrency={quote?.currency || undefined}
            imageSource={src}
            imageUri={typeof src === 'object' ? src.uri : undefined}
            itemId={slot.item?.id ? String(slot.item.id) : undefined}
            buyUrl={buyUrl}
            showBuyButton={showBuyButtons}
            buyLabel={buyLabel}
            textSecondary={D.textSecondary}
            textPrimary={D.textPrimary}
            onBuy={handleBuy}
          />
        );
      })}
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

// Seeded shuffle lives in features/outfit-generator/utils/deckDedupe.ts
// (shuffleWithSeed — shared + unit-tested). Slot ordering below goes
// through orderSlotCandidates from the same module.

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
  forceWardrobeOnly: boolean = false,
  seedOffset = 0
): Promise<GeneratedOutfit[]> {
  // A non-positive limit means "no outfits" — bail before the last-resort
  // single-outfit block below, which otherwise emits 1 look for limit 0.
  if ((limit ?? 3) <= 0) return [];
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
  const needsLayerFill = resolveOutfitComposition(style, weather, '').layered;
  if (needsLayerFill && !itemMacros.has('outerwear')) missingSlots.push('outerwear');
  if (needsLayerFill && availableItems.filter((it: any) => getOfflineMacroCategory(it.type || '', it.category || '', it.name || '') === 'top').length < 2) {
    missingSlots.push('top');
  }
  let shopFills: any[] = [];
  // Wardrobe-only (Supabase clothing) requests must never be padded with
  // shop_catalog or mock items — return what the wardrobe can build.
  if (forceWardrobeOnly) {
    if (missingSlots.length > 0) return [];
  } else if (missingSlots.length > 0) {
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
  // Sparse-slot supplement (shop mode only): pad each slot to ≥limit
  // choices from the bundled static catalog. Wardrobe-only requests skip
  // this — static classic-m-* items must never leak into Supabase outfits.
  const countMacro = (m: string) =>
    allItems.filter((it: any) => {
      const c = canonicalizeMacroCategory(it.macroCategory || '');
      const macro = c !== 'other' ? c : getOfflineMacroCategory(it.type || '', it.category || '', it.name || '');
      return String(macro).toLowerCase() === m;
    }).length;
  if (!forceWardrobeOnly && (countMacro('top') < limit || countMacro('bottom') < limit || countMacro('shoes') < limit || countMacro('outerwear') < Math.min(3, limit))) {
    const seenIds = new Set(allItems.map((it: any) => String(it.id || it.imageUrl || it.name)));
    // Same shared rule: explicit clothing types are trusted, so "belt loops"
    // can never demote real trousers to 'other' here either.
    const toMacro = (g: string, name: string, desc: string): string => shopRowMacro(g, name, desc);
    for (const s of [...(SHOP_CATALOG_ITEMS as any[]), ...(CLASSIC_MENS_ITEMS as any[])]) {
      const id = String((s as any)?.id || '');
      if (!id || seenIds.has(id)) continue;
      const macro = toMacro((s as any).garmentType, (s as any).name || '', (s as any).description || '');
      if (macro === 'other') continue;
      if (macro === 'top' && countMacro('top') >= limit) continue;
      if (macro === 'bottom' && countMacro('bottom') >= limit) continue;
      if (macro === 'shoes' && countMacro('shoes') >= limit) continue;
      if (macro === 'outerwear' && countMacro('outerwear') >= Math.min(3, limit)) continue;
      seenIds.add(id);
      allItems.push({
        id,
        name: (s as any).name || 'Shop item',
        brand: (s as any).brand || '',
        type: macro,
        category: macro === 'top' ? 'tops' : macro === 'bottom' ? 'bottoms' : macro,
        macroCategory: macro,
        color: 'neutral',
        image: (s as any).imageUrl || '',
        imageUrl: (s as any).imageUrl || '',
        description: (s as any).description || '',
        isShopItem: true,
      });
      if (countMacro('top') >= limit && countMacro('bottom') >= limit && countMacro('shoes') >= limit) break;
    }
  }

  const classified = allItems.map(item => {
    const type = (item.type || '').toLowerCase();
    const cat = (item.category || '').toLowerCase();
    const name = (item.name || '').toLowerCase();
    const blobAll = `${type} ${cat} ${name}`;
    // Underwear never enters any slot (live bug: boxers rendered as TOP).
    if (/\bboxers?|briefs?|underwears?|panties?|thongs?|lingerie\b/.test(blobAll)) {
      return { item, category: null };
    }
    // Canonicalize macroCategory / category so aliases like upper_body /
    // lower_body / tops / footwear route to the correct slot instead of
    // falling through and being dropped.
    const canonical = canonicalizeMacroCategory(item.macroCategory || '');
    const catCanonical = canonical !== 'other' ? canonical : canonicalizeMacroCategory(cat);

    // Name-first overrides: feed mistags T-shirts as bottoms and short-sleeve
    // knits as outerwear. Trust distinctive names over stored macros.
    const isShoeName = /\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|footwear|trainers?|derbys?|derbies|moccasins?|drivers?|mules?|clogs?)\b/.test(blobAll);
    const isTopName = /\b(t-?shirts?|shirts?|polos?|blouses?|tees?|tanks?|henleys?|oxfords?|button[\s-]?downs?|overshirts?)\b/.test(blobAll);
    const isShortSleeveName = SHORT_SLEEVE_RE.test(blobAll);
    let category: string | null = null;
    if (isShoeName) {
      category = 'shoes';
    } else if (isShortSleeveName || isTopName) {
      category = 'top';
    } else if (catCanonical === 'outerwear' || /\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear)\b/.test(blobAll)) {
      category = 'layer';
    } else if (catCanonical === 'bottom' || /\b(pants?|trousers?|jeans?|shorts\b|skirt|lower[_\s-]?body|bottom)\b/.test(blobAll.replace(STRIP_SHORT_SLEEVE_RE, ' '))) {
      category = 'pants';
    } else if (catCanonical === 'shoes' || /\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|footwear|trainers?|derbys?|derbies|moccasins?|drivers?|mules?|clogs?)\b/.test(blobAll)) {
      category = 'shoes';
    } else if (catCanonical === 'top' || /\b(t-?shirt|shirt|polo|blouse|tee|top|dress|upper[_\s-]?body)\b/.test(blobAll)) {
      category = 'top';
    }
    return { item, category };
  }).filter(c => c.category !== null);

  // Shuffle each category to ensure variety even with limited items
  const layers = shuffleArray(classified.filter(c => c.category === 'layer').map(c => c.item));
  const tops = shuffleArray(classified.filter(c => c.category === 'top').map(c => c.item));
  const pants = shuffleArray(classified.filter(c => c.category === 'pants').map(c => c.item));
  const shoes = shuffleArray(classified.filter(c => c.category === 'shoes').map(c => c.item));

  // Determine if layering needed based on weather ONLY
  const needsLayer = resolveOutfitComposition(style, weather, '').layered;

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

  const styleName = styleLabelFor(style);
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
  const seenSigs = new Set<string>();
  const builtSets: Array<Set<string>> = [];
  // Deterministic diverse round-robin: every slot advances with i plus a
  // per-slot offset so 3 outfits never freeze on the same pieces.
  // Random offsets were causing collisions (same index twice) — the exact
  // "3 similar outfits" bug. seedOffset shifts every tap so re-taps differ.
  // `attempt` perturbs every slot rotation so a rejected combination is
  // never rebuilt identically on retry (previously the loop gave up after
  // limit*4 tries with a fixed rotation — thin pools collapsed to 1-2).
  const divIdx = (outfitIdx: number, slotOffset: number, len: number, attempt = 0): number => {
    if (len <= 0) return 0;
    return (outfitIdx + slotOffset + seedOffset * 5 + attempt * 7) % len;
  };

  const buildOfflineCandidate = (i: number, attempt: number): any[] | null => {
    const outfitItems: any[] = [];

    const candidatePant = pants[divIdx(i, 6, pants.length, attempt)] || pants[0];
    const pantIsShorts = !!candidatePant && isShortsBottom(candidatePant);

    // 4-slot model: [outerwear/layer, baseTop, pants, shoes]
    // Slot 1: outerwear (main-top / layer) — offset 0 rotates fastest
    if (needsLayer && layers.length > 0) {
      let layer = layers[divIdx(i, 0, layers.length, attempt)];
      if (pantIsShorts && isFormalLayer(layer)) {
        layer = casualLayers[divIdx(i, 0, casualLayers.length, attempt)] || layer;
      }
      outfitItems.push({ ...layer, macroCategory: 'outerwear' });
    }

    // Slot 2: base top — offset 3 so it rotates independently from outerwear
    if (tops.length > 0) {
      const top = tops[divIdx(i, 3, tops.length, attempt)];
      outfitItems.push({ ...top, macroCategory: 'top' });
    }
    // When layering, ensure we have a second base top (clone if only 1 top)
    if (needsLayer && outfitItems.some(item => item.macroCategory === 'outerwear')) {
      const existingTop = outfitItems.find(item => item.macroCategory === 'top');
      const secondTop = tops[divIdx(i + 1, 3, tops.length, attempt)]
        || tops[divIdx(i, 3, tops.length, attempt)];
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

    // Layer-vs-base repair: never pair two same-fabric tops.
    const _layer0 = outfitItems.find(item => item.macroCategory === 'outerwear');
    const _top0 = outfitItems.find(item => item.macroCategory === 'top');
    if (_layer0 && _top0 && isSameLayerBaseCategory(_layer0, _top0)) {
      const layerFam = fabricFamily(_layer0);
      const altTop = tops.filter((t: any) => t && String(t.id) !== String(_top0.id) && fabricFamily(t) !== layerFam)[divIdx(i, 3, Math.max(tops.filter((t: any, attempt) => t && fabricFamily(t) !== layerFam).length, 1))]
        || tops.find((t: any) => t && String(t.id) !== String(_top0.id) && fabricFamily(t) !== layerFam);
      if (altTop) {
        const topIdx = outfitItems.findIndex(item => item.macroCategory === 'top');
        if (topIdx >= 0) outfitItems[topIdx] = { ...altTop, macroCategory: 'top' };
      }
    }

    let finalPant = candidatePant;
    if (needsLayer && pantIsShorts && outfitItems[0] && isFormalLayer(outfitItems[0]) && nonShortsPants.length > 0) {
      finalPant = nonShortsPants[divIdx(i, 6, nonShortsPants.length, attempt)];
    }
    if (finalPant) {
      outfitItems.push({ ...finalPant, macroCategory: 'bottom' });
    }

    if (shoes.length > 0) {
      const shoe = shoes[divIdx(i, 9, shoes.length, attempt)];
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
    if (!hasRequiredTopCount || !hasBottom || !hasShoes || !hasLayer) return null;
    return outfitItems;
  };

  const keepOfflineCandidate = (outfitItems: any[], tag: number, relaxed: boolean): boolean => {
    // Reject exact + near-duplicates: any two looks may share at most 1
    // item (sharing 2+ reads as the same outfit with a different shirt).
    const sig = outfitItems.map((p: any) => String(p.id)).sort().join('|');
    if (!sig || seenSigs.has(sig)) return false;
    if (!relaxed) {
      const ids = outfitItems.map((p: any) => String(p.id));
      let maxShared = 0;
      for (const built of builtSets) {
        let shared = 0;
        for (const id of ids) if (built.has(id)) shared++;
        if (shared > maxShared) maxShared = shared;
      }
      if (builtSets.length > 0 && maxShared >= 2) return false;
    }
    seenSigs.add(sig);
    builtSets.push(new Set(outfitItems.map((p: any) => String(p.id))));

    outfits.push({
      id: `offline_${Date.now()}_${tag}_${Math.floor(Math.random() * 1000)}`,
      mainImage: outfitItems[0]?.image || outfitItems[0]?.imageUrl || '',
      matchScore: 0.85 + (Math.random() * 0.1), // Random variation in match score
      description: descriptions[outfits.length % 3],
      items: outfitItems,
      stylingTips: tips[outfits.length % 3].join(' · '),
      weather,
    });
    return true;
  };

  // Strict pass: decks must differ by ≥2 items. Rejections perturb the
  // attempt so retries explore new slot combinations.
  let offlineAttempt = 0;
  for (let i = 0, tag = 0; outfits.length < limit && i < limit * 8; i++, tag++) {
    const candidate = buildOfflineCandidate(i, offlineAttempt);
    if (!candidate) { offlineAttempt++; continue; }
    if (keepOfflineCandidate(candidate, tag, false)) offlineAttempt = 0;
    else offlineAttempt++;
  }

  // Relaxed pass: thin pools (a single shoes/bottom/outerwear choice) can
  // NEVER satisfy the strict rule — every second look shares 3 of 4 by
  // construction. Fill the remaining slots with exact-dupe-only rejection
  // so the user gets 3 looks instead of 1.
  if (outfits.length < limit) {
    for (let i = 0, tag = 1000; outfits.length < limit && i < limit * 8; i++, tag++) {
      const candidate = buildOfflineCandidate(i + limit * 8, offlineAttempt);
      if (!candidate) { offlineAttempt++; continue; }
      if (keepOfflineCandidate(candidate, tag, true)) offlineAttempt = 0;
      else offlineAttempt++;
    }
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
      description: forceWardrobeOnly
        ? `${stylePhraseFor(style)} look from your wardrobe.`
        : `${stylePhraseFor(style)} look curated from the shop.`,
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
  price: {
    ...LiquidGlass2026Theme.typography.scale.bodySmall,
    marginTop: 2,
    fontSize: 11,
    fontWeight: '700',
    color: '#1F6FEB',
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

const formatPlanningDate = (date: Date): string =>
  date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

const CLIENT_RUNTIME_MARKER = 'rotation-debug-555b157-v1';

const AIOutfitGenerator = () => {
  // eslint-disable-next-line no-console
  useEffect(() => { if (__DEV__) console.log('[client-runtime-marker]', CLIENT_RUNTIME_MARKER); }, []);
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  // ── Post-trial soft gate: page stays open, Generate → Paywall ──
  // free.aiOutfits = 0 → consumeOrPaywall denies + navigates to Paywall.
  // Trial-active (effectiveTier premium) and paid users pass through.
  const { consumeOrPaywall } = useSubscriptionGate();
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
    () => parseCalendarDate(planningDateKey),
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
  const displayRates = useUzsDisplayRates();
  const [userPrompt, setUserPrompt] = useState('');
  const promptRef = React.useRef(userPrompt);
  // Chip taps are the live dress-code. Stale comment text ("date night, old
  // money") must not override a later Casual/Formal chip — Maestro and
  // humans both hit this after a previous generate.
  const commentEditedRef = React.useRef(false);
  // Per-tap nonce so every Generate press rotates pool picks differently.
  // Without this, deterministic (i+offset)%len returns the SAME 3 outfits
  // on every tap — the "every time the same outfit" bug.
  const generationNonce = React.useRef(0);
  const generationRequest = React.useRef<AbortController | null>(null);
  useEffect(() => () => { generationRequest.current?.abort(); }, []);
  // ── New staged Create flow ──────────────────────────────────────────
  // Step 1 "create": source (wardrobe / new clothing) + style (quiz-oriented).
  // Step 2 "occasion": small "Where are you going?" input.
  // On Generate the whole setup block animates out (exit animation) and
  // the AI thinking view takes over on the SAME page.
  const [selectedOccasion, setSelectedOccasion] = useState<string | undefined>(
    Array.isArray(quizOccasions) && quizOccasions.length > 0 ? quizOccasions[0] : undefined,
  );
  // Occasion dropdown: 2 chips visible, rest behind a "More" toggle.
  // Collapsed by default — user expands explicitly via "More".
  const [showAllOccasions, setShowAllOccasions] = useState(false);
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
  const { items: liveShopCatalog, loading: shopCatalogLoading, error: shopCatalogError, refresh: refreshShopCatalog } = useShopCatalog({
    enabled: true,
    source: 'all',
  });

  // Surface catalog fetch failures in dev logs — an empty live catalog with
  // a hidden error is what silently degrades outfits to static items.
  useEffect(() => {
    if (__DEV__ && shopCatalogError) readinessLogger.debug(String('[AIOutfitmaker] shop_catalog fetch error:'), [shopCatalogError]);
  }, [shopCatalogError]);

  // Map the new catalog format into what the outfit generator expects
  // Key fix: use the shared shopRowMacro so a blazer stays a layer even
  // though it shares garmentType 'upper_body' with shirts — and so explicit
  // lower_body/shoes rows are never demoted by description keywords
  // ("belt loops" must not erase suit pants). Without the shared rule, ALL
  // upper_body items become 'top' and layering collapses to 3 slots.
  const liveShopMapped = React.useMemo(
    // Nameless feed rows render as raw ids on cards — drop them at the pool
    // root (wardrobe items are never filtered: user naming is sacred).
    () => liveShopCatalog
      .map((item) => mapShopCatalogRow(item as any))
      .filter((it: any) => hasUsableProductName(it.name, it.id)),
    [liveShopCatalog],
  );

  // Prevent setState after unmount
  const isMounted = React.useRef(true);
  const fallbackTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Cross-tap recency memory ──────────────────────────────────────
  // Item ids shown in recent decks. The next shop payload sorts these last
  // (prioritizeUnseen) so the edge's top-N trim drops them first — this is
  // what breaks the "same linen pants + penny loafers every tap" loop.
  // In-memory ref = instant effect on the very next tap; AsyncStorage = the
  // memory survives app restarts. Capped so the pool can never starve.
  const RECENT_SHOWN_KEY = 'aiRecentShownItemIds';
  const RECENT_SHOWN_CAP = 240;
  const recentShownIds = React.useRef<Set<string>>(new Set());
  const rememberShownIds = React.useCallback((outfits: Array<{ items?: Array<{ id?: string; name?: string; image?: string; imageUrl?: string }> }>) => {
    try {
      for (const o of outfits || []) {
        for (const it of o.items || []) {
          if (it?.id) recentShownIds.current.add(String(it.id));
          const nk = normalizeCatalogNameKey(it?.name);
          if (nk) recentShownIds.current.add(`__n:${nk}`);
          const img = productImageKey(it);
          if (img) recentShownIds.current.add(`__img:${img}`);
          const fam = garmentVarietyFamily(it);
          if (fam) recentShownIds.current.add(`__fam:${fam}`);
        }
      }
      while (recentShownIds.current.size > RECENT_SHOWN_CAP) {
        const oldest = recentShownIds.current.values().next().value;
        if (oldest === undefined) break;
        recentShownIds.current.delete(oldest);
      }
      AsyncStorage.setItem(RECENT_SHOWN_KEY, JSON.stringify([...recentShownIds.current])).catch(() => {});
    } catch (_) {
      // Recency is best-effort — never break generation.
    }
  }, []);

  useEffect(() => {
    AsyncStorage.getItem(RECENT_SHOWN_KEY)
      .then((raw) => {
        if (!raw || !isMounted.current) return;
        try {
          const ids = JSON.parse(raw);
          if (Array.isArray(ids)) recentShownIds.current = new Set(ids.map(String).slice(0, RECENT_SHOWN_CAP));
        } catch (_) {
          // Corrupt cache — start fresh.
        }
      })
      .catch(() => {});
  }, []);

  // ── Per-tap catalog variety ───────────────────────────────────────
  // The catalog hook only holds page 1 (first 100 rows by sort_order), so
  // without this every tap/style draws from the same 100 ids and the same
  // elite bottoms win forever. Each shop generation fetches one RANDOM
  // window from the full table and merges it into the pool (dedupe by id).
  // Bounded + silent-fallback: on timeout we proceed with the live page.
  const catalogTotalRef = React.useRef<number | null>(null);
  const RANDOM_SLICE_SIZE = 120;
  const withTimeout = <T,>(promise: PromiseLike<T>, ms: number, label: string): Promise<T> =>
    Promise.race([
      Promise.resolve(promise),
      new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`${label} timed out`)), ms)),
    ]);

  const fetchRandomShopSlice = async (opts?: {
    preferClassicBand?: boolean;
    salt?: number;
  }): Promise<Array<ReturnType<typeof mapShopCatalogRow>>> => {
    try {
      if (catalogTotalRef.current == null) {
        const { count } = await withTimeout(
          supabase.from('shop_catalog').select('id', { count: 'exact', head: true }).eq('is_active', true),
          4000,
          'shop count',
        ) as { count: number | null };
        catalogTotalRef.current = count && count > 0 ? count : 2308;
      }
      const total = catalogTotalRef.current;
      const mapRows = (data: any[] | null) =>
        (data ?? [])
          .map((row: any) =>
            mapShopCatalogRow({
              id: String(row.id),
              name: row.name,
              brand: row.brand,
              description: row.description,
              imageUrl: row.image_url,
              garmentType: row.garment_type,
              sourceUrl: row.source_url,
              // price/currency MUST survive this mapping: the random slice is
              // the generation pool, and without them budget caps compare
              // against 0 (every candidate is rejected -> placeholder
              // "Pants"/"Shoes" tiles) and tiles can never show a price.
              price: typeof row.price === 'number' ? row.price : undefined,
              currency: row.currency || undefined,
            }),
          )
          .filter((it: any) => hasUsableProductName(it.name, it.id));
      const fetchRange = async (start: number, size: number) => {
        const { data } = await withTimeout(
          supabase
            .from('shop_catalog')
            .select('id,brand,name,price,currency,image_url,garment_type,description,source_url')
            .eq('is_active', true)
            .order('sort_order', { ascending: true })
            .range(start, start + size - 1),
          6000,
          'shop slice',
        ) as { data: any[] | null };
        return mapRows(data);
      };
      // Classic / work / date must stay in the curated MD band (random Zara
      // windows around offset 1100 left no shirts). Rotate INSIDE that band
      // so taps are not frozen on the same 38 collapsed names at offset 0.
      if (opts?.preferClassicBand) {
        const salt = opts.salt ?? Date.now();
        const startA = classicCatalogOffset(total, RANDOM_SLICE_SIZE, salt);
        const startB = classicCatalogOffset(total, RANDOM_SLICE_SIZE, salt * 17 + 91);
        const [head, sliceA, sliceB] = await Promise.all([
          fetchRange(0, 80),
          fetchRange(startA, RANDOM_SLICE_SIZE),
          startB !== startA ? fetchRange(startB, RANDOM_SLICE_SIZE) : Promise.resolve([]),
        ]);
        const rows = mergeShopSlices(mergeShopSlices(head, sliceA), sliceB);
        if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] random slice: offset=${startA}+${startB} head=80 rows=${rows.length} total=${total}`), []);
        return rows;
      }
      const start = Math.floor(Math.random() * Math.max(1, total - RANDOM_SLICE_SIZE));
      const rows = await fetchRange(start, RANDOM_SLICE_SIZE);
      if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] random slice: offset=${start} rows=${rows.length} total=${total}`), []);
      return rows;
    } catch (err) {
      if (__DEV__) readinessLogger.debug(String('[AIOutfitmaker] random slice failed, using live page only:'), [(err as Error)?.message]);
      return [];
    }
  };

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
      // ── Shop mode: live Supabase shop catalog items ONLY.
      // Real Supabase UUIDs MUST be preserved so the edge function can enrich
      // items with imageUrl via its itemMap lookup. Do NOT mix in AsyncStorage.
      // "New clothing" means 100% new (Supabase shop_catalog) — a wardrobe
      // anchor must NEVER be injected here, otherwise every outfit gets built
      // around the user's own piece even though the UI shows New clothing.
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
        if (__DEV__) readinessLogger.debug(String('[AIOutfitmaker] Injected baseItem into wardrobe pool:'), [baseItemId]);
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

      // Wardrobe mode = Supabase clothing only. Do NOT merge shop catalog
      // here — mixing shop items is what produced classic-m-*/shop_*
      // outfits when the user asked for wardrobe outfits. Shop mode has
      // its own branch above and never reaches this code.

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


  // ── Weather Fetch ────────────────────────────────────────────────
  const fetchWeather = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setWeather(undefined);
        return;
      }
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
        return;
      }
    } catch (e) {
      console.warn('Weather fetch error', e);
    }
    setWeather(undefined);
  };

  useEffect(() => {
    fetchWeather();
  }, []);

  const candidatePool = useMemo(() => {
    // Wardrobe mode = Supabase clothing only (isWardrobe === true).
    // Shop mode merges live catalog. Mixing shop into the wardrobe pool is
    // what produced classic-m-*/shop_* outfits for wardrobe requests.
    const sourcePool =
      currentSource === 'wardrobe'
        ? (wardrobeItems || []).filter((it: any) => it?.isWardrobe !== false)
        : [...(wardrobeItems || []), ...(liveShopMapped || [])];
    const seen = new Set<string>();
    const pool: any[] = [];
    for (const it of sourcePool) {
      const key = String((it as any)?.id || (it as any)?.imageUrl || (it as any)?.name || '');
      if (!key || seen.has(key)) continue;
      seen.add(key);
      pool.push(it);
    }

    // Helper: prefer an item's own macroCategory (canonicalized) over the
    // keyword-based fallback. Keeps items whose type/category are raw
    // garmentType strings (`upper_body`, `lower_body`) from being dropped.
    const resolveMacroCategory = (it: any): string => {
      const ownCanonical = canonicalizeMacroCategory(it?.macroCategory || '');
      if (ownCanonical !== 'other') return ownCanonical;
      // Shop rows were classified upstream WITH the structured zone
      // (inferShopRowMacro): macroCategory 'other' = accessory or a
      // zone-vs-name conflict (md-uz-62899803 — lower_body zone + trousers
      // photo + stolen "suit blazer" name). Name-only resurrection here is
      // what put that trousers into the LAYER tile — keep shop 'other' dropped.
      if (it?.isShopItem) return 'other';
      return getBackendMacroCategory(it?.type || '', it?.category || '', it?.name || '');
    };

    const mapped = pool.map((it: any) => {
      const image = (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '';
      const imageUrl = it.imageUrl || (typeof it.image === 'string' ? it.image : '');
      return {
        id: it.id || it.imageUrl || it.name,
        name: it.name || it.type || 'Item',
        image,
        imageUrl,
        type: it.type || it.category || 'top',
        macroCategory: resolveMacroCategory(it),
        color: it.color || 'neutral',
        brand: it.brand || '',
        isWardrobe: it.isWardrobe ?? false,
        isShopItem: it.isShopItem ?? false,
        shopUrl: it.shopUrl || undefined,
        price: typeof it.price === 'number' ? it.price : undefined,
        currency: it.currency,
        // Precomputed once here — pickCandidate scanned this regex per
        // candidate per slot (~10 full-pool regex sweeps per outfit).
        _hasImage: REAL_IMAGE_RE.test(image || imageUrl),
      };
    }).sort((a, b) => (b.isWardrobe ? 1 : 0) - (a.isWardrobe ? 1 : 0));
    // Static fallback: shop mode only. Wardrobe mode must stay pure
    // Supabase clothing — padding with classic-m-* static items is what
    // produced non-wardrobe outfits. In wardrobe mode an empty pool stays
    // empty so the UI can ask the user to add items instead.
    if (currentSource !== 'wardrobe') {
      const macroCounts = new Map<string, number>();
      for (const x of mapped) {
        macroCounts.set(x.macroCategory, (macroCounts.get(x.macroCategory) ?? 0) + 1);
      }
      const countByMacro = (m: string) => macroCounts.get(m) ?? 0;
      if (countByMacro('top') < 6 || countByMacro('bottom') < 6 || countByMacro('shoes') < 6 || countByMacro('outerwear') < 3) {
        const seenStatic = new Set(mapped.map((x: any) => String(x.id)));
        for (const s of getStaticPoolItems()) {
          if (seenStatic.has(s.id)) continue;
          if (countByMacro(s.macroCategory) >= 8) continue;
          seenStatic.add(s.id);
          macroCounts.set(s.macroCategory, countByMacro(s.macroCategory) + 1);
          mapped.push({ ...s } as any);
        }
      }
    }
    return mapped;
  }, [currentSource, liveShopMapped, wardrobeItems]);

  const getCalendarOccasion = (styleId?: string) => resolveCalendarOccasion(selectedOccasion, styleId);

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
    let updated = false;
    await updateCalendarLogs(logs => {
      updated = Boolean(logs[dateKey]);
      logs[dateKey] = log;
    });
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

    const revision = authEvents.getRevision();
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
      savedFrom: 'ai-maker',
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

    let confirmedOutfit = outfit;
    const snapshotItems = outfit.items.map(item => ({ id: String(item.id || item.image), type: item.type || 'Clothing Piece',
      image: item.image, imageUrl: item.imageUrl, name: item.name }));
    try {
      const confirmedItems = await saveCloudOutfit({
        id: newOutfitId,
        items: snapshotItems,
        date: connectedDateKey ?? localDateKey(new Date()), occasion, caption: outfit.description || '',
      });
      if (revision !== authEvents.getRevision()) return;
      const replacements = confirmedImageReplacements(snapshotItems, confirmedItems);
      confirmedOutfit = replaceGeneratedOutfitImages(outfit, replacements);
      useWardrobeStore.setState(state => ({
        outfits: state.outfits.map(look => replaceSavedOutfitImages(look, replacements)),
        dailySuggestion: state.dailySuggestion ? { ...state.dailySuggestion,
          outfit: replaceSavedOutfitImages(state.dailySuggestion.outfit, replacements),
        } : state.dailySuggestion,
      }));
      setOutfits(current => current.map(look => replaceGeneratedOutfitImages(look, replacements)));
      store.saveOutfit(newOutfitId, collectionCategory, 'ai-maker');
    } catch {
      if (revision === authEvents.getRevision()) Alert.alert(t('common.error'), t('wardrobeItems.failedSaveChanges'));
      return;
    }

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

    // Connected (trip planner) save: also link the outfit to the trip day
    // so Calendar + Trips packing list update immediately.
    if (connectedDateKey) {
      let updated = false;
      try {
        ({ updated } = await persistOutfitToTripDay(confirmedOutfit, connectedDateKey, occasion));
      } catch (err) {
        console.warn('[AIOutfitmaker] Failed to link saved outfit to trip day:', err);
        Alert.alert(t('common.error'), t('calendar.persistenceError'));
        return;
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
    const revision = authEvents.getRevision();
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
      const logs = await readCalendarLogs();

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
                try {
                  await updateCalendarLogs(current => { current[dateStr] = log; }, revision);
                  finishSave(date);
                } catch {
                  Alert.alert(t('common.error'), t('calendar.persistenceError'));
                }
              },
            },
          ]
        );
        return;
      }

      await updateCalendarLogs(current => {
        if (current[dateStr]) throw new Error('Calendar date changed; confirm replacement again');
        current[dateStr] = log;
      }, revision);
      finishSave(date);
    } catch (e) {
      console.error('Calendar save error', e);
      Alert.alert(t('common.error'), t('calendar.persistenceError'));
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
        testID={`stylistStyle-${key}`}
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
  // Post-trial gate: consumeOrPaywall auto-redirects to Paywall when the
  // free quota is hit (free aiOutfits = 0 → always locked after trial).
  // NOTE: intentionally NOT useCallback-memoized. A memoized handler
  // closed over the first render's `currentSource`/`wardrobeItems`/`liveShopMapped`
  // (deps were [selectedStyle, selectedOccasion, userPrompt]), so toggling
  // "New clothing" still ran the wardrobe branch with an empty pool —
  // the "Wardrobe insufficient" logs while the UI showed New. A fresh
  // closure every render always sees the latest source + pools.
  const cancelOutfitGeneration = () => {
    generationRequest.current?.abort();
    generationRequest.current = null;
    generationNonce.current += 1;
    setLoading(false);
    setOutfits([]);
    setError('');
  };

  const handleGeneratePress = async (styleOverride?: string) => {
    const gateOwner = authEvents.getUserId(), gateRevision = authEvents.getRevision();
    if (!gateOwner || generationRequest.current) return;
    const controller = new AbortController();
    generationRequest.current = controller;
    const current = () => generationRequest.current === controller && !controller.signal.aborted &&
      gateOwner === authEvents.getUserId() && gateRevision === authEvents.getRevision() && isMounted.current;
    try {
      setLoading(true);
      setError('');
      let ok = false;
      try { ok = await consumeOrPaywall('aiOutfits'); } catch { ok = false; }
      if (!current() || !ok) return;
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      if (styleOverride) setSelectedStyle(styleOverride);
      await generateOutfits(styleOverride, controller.signal);
    } catch (err) {
      if (current()) setError(t(err instanceof OutfitGenerationFailure ? err.translationKey : 'outfitMaker.generateFailed'));
    } finally {
      // A cancelled request must not clear a newer retry's loading state.
      if (generationRequest.current === controller) {
        generationRequest.current = null;
        if (isMounted.current) setLoading(false);
      }
    }
  };

  const generateOutfits = async (overrideStyle?: string, cancelSignal?: AbortSignal) => {
    const generationOwner = authEvents.getUserId(), generationRevision = authEvents.getRevision();
    if (!generationOwner) return;
    let generationAccessToken: string | undefined;
    let recencyScope = 'anonymous';
    const promptText = String(promptRef.current || userPrompt || '');
    const commentIntent = parseStylistIntent(promptText);
    const useCommentIntent = commentEditedRef.current;
    const occasionToUse = (useCommentIntent && commentIntent.occasion) || selectedOccasion;
    const styleToUse = overrideStyle || (useCommentIntent && commentIntent.style) || selectedStyle;
    if (useCommentIntent && commentIntent.style && commentIntent.style !== selectedStyle && !overrideStyle) {
      setSelectedStyle(commentIntent.style);
    }
    if (useCommentIntent && commentIntent.occasion && commentIntent.occasion !== selectedOccasion) {
      setSelectedOccasion(commentIntent.occasion);
    }
    const effectivePrompt = (() => {
      const occasionLabel = OCCASION_PRESETS.find((o) => o.id === occasionToUse)?.label;
      const trimmed = promptText.trim();
      if (occasionLabel && trimmed) return `${occasionLabel}: ${trimmed}`;
      if (occasionLabel) return occasionLabel;
      return trimmed;
    })();
    const rawShopBudget = commentIntent.budget || parseBudgetFromText(promptText);
    // Preserve the cap in its original currency. A request-scoped reference
    // snapshot is loaded below only when monetary comparisons need conversion.
    const edgePrompt = stripBudgetFromText(effectivePrompt);
    let shopBudget = rawShopBudget;
    const fitsCap = (items: any[]) =>
      !shopBudget || fitsBudgetAmount(outfitShopSpend(items as any[], liveShopMapped, shopBudget?.currency, shopBudget?.fx), shopBudget.amount);
    const resolvedWeather = resolveStylistWeather(promptText, weather);

    if (!isMounted.current) return;
    generationNonce.current += 1;
    // Snapshot the tap's nonce — concurrent double-taps must NOT share one
    // live ref (both runs would read the final value and pick identically).
    // Every normalize/offline call below takes tapNonce explicitly.
    const tapNonce = generationNonce.current;
    const isCurrentGeneration = () => !cancelSignal?.aborted && isMounted.current && generationNonce.current === tapNonce &&
      authEvents.getUserId() === generationOwner && authEvents.getRevision() === generationRevision;
    setLoading(true);
    setError('');
    setOutfits([]);

    const selectedClothing: any[] = [];

    const effectiveSource = currentSource;
    if (__DEV__) {
      const liveCount = (liveShopMapped || []).length;
      const wardrobeInPool = (wardrobeItems || []).filter((i: any) => i.isWardrobe === true).length;
      readinessLogger.debug(String(`[AIOutfitmaker] generate tap: rules=${STYLIST_RULES_VERSION} source=${effectiveSource} wardrobeItems=${(wardrobeItems || []).length} (owned=${wardrobeInPool}) liveShop=${liveCount} pool=${candidatePool.length} style=${styleToUse} occasion=${occasionToUse} weather=${JSON.stringify(resolvedWeather)} budget=${JSON.stringify(shopBudget)}`), []);
      if (effectiveSource === 'shop' && wardrobeInPool > 0) readinessLogger.debug(String('[AIOutfitmaker] VIOLATION: wardrobe items inside shop pool!'), []);
    }

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

      // Wardrobe mode = Supabase clothing only. Never silently switch to
      // shop/static — that produced classic-m-*/shop_* outfits for wardrobe
      // requests. Insufficient wardrobes surface "add items" below instead.
      if (countTops < MIN_REQUIRED || countPants < MIN_REQUIRED || countShoes < MIN_REQUIRED || isUnsuitable) {
        if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] Wardrobe insufficient (tops=${countTops} pants=${countPants} shoes=${countShoes}). Staying in wardrobe mode — Supabase clothing only.`), []);
      }
    }

    // Build a clean payload for the backend — strip non-serialisable fields
    // (require() numbers) and ensure every item has macroCategory + imageUrl.
    let payloadItems = [];
    // Per-tap normalize pool: memoized first page + random slice, collapsed.
    // normalizeTo4Slots MUST draw from this (not the frozen memo), otherwise
    // the random slice only reaches the edge while finals keep recycling the
    // same 100 pool items. Wardrobe taps keep the memoized pool (purity).
    let genPool: any[] = candidatePool;
    if (effectiveSource === 'shop') {
      const toPayload = (item: any) => ({
        id: item.id,
        name: item.name || item.type || t('common.clothingItem'),
        type: item.type || 'top',
        category: item.category || item.type || 'tops',
        color: item.color || 'neutral',
        brand: item.brand || '',
        description: item.description || item.name || '',
        imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
        isShopItem: true,
        price: typeof item.price === 'number' ? item.price : undefined,
        currency: item.currency,
        // Trust the identity-first mapped macro: recomputing from
        // type/category aliases would flip mistagged tees (type "bottoms")
        // back to 'bottom' on the way out.
        macroCategory: preferMappedMacro(
          item.macroCategory,
          getBackendMacroCategory(item.type || '', item.category || '', item.name || ''),
        ),
      });
      // A fresh random window per tap: without this the pool is always page
      // 1 (same 100 ids → same elite bottoms win every style). Merged with
      // dedupe-by-id, so overlap with the live page is free.
      const classicOrWork = /old_money|classic|business_casual/.test(String(styleToUse || ''))
        || occasionToUse === 'work' || occasionToUse === 'date' || occasionToUse === 'formal';
      const randomSlice = await fetchRandomShopSlice({ preferClassicBand: classicOrWork, salt: tapNonce });
      const poolBase = collapseSimilarByName(mergeShopSlices(
        wardrobeItems.filter((item: any) => item.isWardrobe !== true),
        randomSlice,
      ));
      // Keep SKU variants in the pick pool so cards can rotate; name-collapse
      // is only for the edge payload (LLM trim). Family skip handles clones.
      genPool = collapseDuplicateProductImages(mergeShopSlices(candidatePool, randomSlice))
        .filter((it: any) => isTrustedProductImage(it.imageUrl || it.image));
      if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] shop pool: live=${wardrobeItems.length} slice=${randomSlice.length} merged=${poolBase.length} genPool=${genPool.length}`), []);
      payloadItems = poolBase
        .map(toPayload)
        // Accessories can never fill a slot — drop them so they don't eat
        // the edge's top-N trim budget (the first catalog page is 28% dead
        // weight otherwise).
        .filter((item: any) => {
          const m = canonicalizeMacroCategory(item.macroCategory);
          if (!(m === 'top' || m === 'bottom' || m === 'shoes' || m === 'outerwear')) return false;
          return rejectForStyleOccasion({ ...item, macroCategory: m }, styleToUse, occasionToUse) === null
            && rejectForClimate({ ...item, macroCategory: m }, resolvedWeather) === null
            && isTrustedProductImage(item.imageUrl || item.image);
        });
      const tasteKept = payloadItems.length;
      const staticTaste = getStaticPoolItems().map((s: any) => toPayload({ ...s, image: s.imageUrl }));
      const rotatedStatic = staticTaste.map((_, i, arr) => arr[(i + tapNonce * 7) % arr.length]);
      payloadItems = ensureSlotFloors(payloadItems, rotatedStatic, styleToUse, occasionToUse, undefined, resolvedWeather);
      payloadItems = collapseDuplicateProductImages(payloadItems);
      genPool = collapseDuplicateProductImages(mergeShopSlices(
        mergeShopSlices(rotatedStatic, payloadItems),
        genPool,
      )).filter((it: any) => isBrandCatalogImage(it.imageUrl || it.image) || isTrustedProductImage(it.imageUrl || it.image));
      if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] taste-filtered shop payload: ${tasteKept} items for ${styleToUse}/${occasionToUse}; floored=${payloadItems.length}`), []);
      // Cross-tap recency: recently shown items sort last, so the edge trim
      // prefers fresh pieces. Deprioritized, never deleted (no starvation).
      if (recentShownIds.current.size > 0) {
        const recencyKeys = (p: any) => {
          const keys = [String(p.id)];
          const nk = normalizeCatalogNameKey(p?.name);
          if (nk) keys.push(`__n:${nk}`);
          const img = productImageKey(p);
          if (img) keys.push(`__img:${img}`);
          const fam = garmentVarietyFamily(p);
          if (fam) keys.push(`__fam:${fam}`);
          return keys;
        };
        const before = payloadItems.map((p: any) => String(p.id));
        payloadItems = prioritizeUnseen(payloadItems, recentShownIds.current, recencyKeys);
        genPool = prioritizeUnseen(genPool, recentShownIds.current, recencyKeys);
        if (__DEV__) {
          const moved = before.filter((id) => recentShownIds.current.has(id)).length;
          readinessLogger.debug(String(`[AIOutfitmaker] recency: deprioritized ${moved} recently-shown item(s) in shop payload`), []);
        }
      }
      // Shop mode ("New clothing"): live Supabase catalog only.
      // STRICT: new clothing must come ONLY from new clothing (shop).
      // No wardrobe fallback, no bundled fallback (static pools are empty
      // since the live-exclusive migration). Empty live catalog fails fast
      // with a clear message instead of calling the edge with an empty pool.
      // NOTE: the fallback below is a no-op while static pools stay empty —
      // kept so a future curated seed can't silently reintroduce non-live rows
      // without passing through the shop-only filter above.
      if (payloadItems.length === 0) {
        const staticPool = getStaticPoolItems();
        if (__DEV__) readinessLogger.debug(String('[AIOutfitmaker] live shop catalog empty — static fallback pool:'), [staticPool.length]);
        payloadItems = staticPool.map((s: any) => toPayload({ ...s, image: s.imageUrl }));
      }

    } else {
      // Wardrobe mode: Supabase clothing only — drop anything not owned,
      // so shop/static items can never leak into the edge candidate pool.
      payloadItems = wardrobeItems
        .filter((item: any) => item.isWardrobe !== false)
        .map((item: any) => ({
          id: item.id,
          name: item.name || item.type || t('common.clothingItem'),
          type: item.type || 'top',
          category: item.category || item.type || 'tops',
          color: item.color || 'neutral',
          brand: item.brand || '',
          description: item.description || item.name || '',
          imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
          // Same rule as shop mode: a mapped clothing slot is authoritative.
          macroCategory: preferMappedMacro(
            item.macroCategory,
            getBackendMacroCategory(item.type || '', item.category || '', item.name || ''),
          ),
        }));
      // Fail fast with a clear message instead of letting the edge fall
      // back to shop/static and showing non-wardrobe outfits.
      if (payloadItems.length === 0) {
        if (isMounted.current) {
          setError(t('outfitMaker.noOutfitsError', { defaultValue: 'No clothing items found in your wardrobe. Add items first!' }));
          setLoading(false);
        }
        return;
      }
    }

    try {
      const budgetCandidates = budgetCandidatesFromText(promptText);
      const mixedLimits = new Set(budgetCandidates.map(b => b.currency)).size > 1;
      const crossCurrencyShop = shopBudget && effectiveSource === 'shop' && liveShopMapped.some((item: any) =>
        Number(item.price) > 0 && ['USD','EUR','GBP','UZS'].includes(String(item.currency || '').trim().toUpperCase()) &&
        String(item.currency || '').trim().toUpperCase() !== shopBudget!.currency);
      if (mixedLimits || crossCurrencyShop) {
        let fx;
        try { fx = budgetFxContext(await getBudgetRates()); }
        catch {
          if (mixedLimits) throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable');
          // Let the server find catalog prices in the requested currency.
          // No conversion or guessed reference is used in this mode.
        }
        if (!isCurrentGeneration()) return;
        shopBudget = parseBudgetFromText(promptText, fx);
      }
      try {
        const { data: sess } = await supabase.auth.getSession();
        if (!isCurrentGeneration()) return;
        if (sess.session?.user.id !== generationOwner || !sess.session?.access_token) throw new Error('generation_session_unavailable');
        generationAccessToken = sess.session.access_token;
        recencyScope = makeRotationScope(generationOwner, effectiveSource, styleToUse, occasionToUse);
        if (__DEV__) readinessLogger.debug(String('[AIOutfitmaker] edge payload:'), [{
          effectiveSource,
          wardrobeState: (wardrobeItems || []).length,
          owned: (wardrobeItems || []).filter((i: any) => i.isWardrobe !== false).length,
          liveShopMapped: (liveShopMapped || []).length,
          payloadItems: (payloadItems || []).length,
          weather: resolvedWeather,
          hasSession: !!sess?.session,
          useProvidedWardrobeOnly: effectiveSource === 'shop',
        }]);
      } catch (_) {
        if (!isCurrentGeneration()) return;
        throw new Error('generation_session_unavailable');
      }
      if (effectiveSource !== 'shop' && (payloadItems || []).length === 0) {
        if (isMounted.current) {
          setError(t('outfitMaker.noOutfitsError', { defaultValue: 'No clothing items found in your wardrobe. Add items first!' }));
          setLoading(false);
        }
        return;
      }

      const [excludeItemIds, excludeProductKeys, rotationHistory] = await Promise.all([
        getRecentShownIds(recencyScope),
        getRecentShownProductKeys(recencyScope),
        getRotationHistory(recencyScope),
      ]);
      if (!isCurrentGeneration() || !generationAccessToken) return;
      const generationRequestId = `${Date.now()}-${tapNonce}-${Math.random().toString(36).slice(2)}`;
      if (__DEV__) readinessLogger.debug('[rotation-client] GENERATE START', {
        generationRequestId,
        tapNonce,
        style: styleToUse,
        occasion: occasionToUse,
        source: effectiveSource,
        rotationScope: recencyScope,
        previousDeckProductKeys: rotationHistory.previousDeckProductKeys,
        olderRecentProductKeysCount: rotationHistory.olderRecentProductKeys.length,
      });
      const EDGE_TIMEOUT_MS = 120000;
      const { data, error: invokeError } = await withRequestDeadline(async (signal) => supabase.functions.invoke('generate-outfits', {
        signal,
        headers: {Authorization: `Bearer ${generationAccessToken}`},
        body: {
          stylePreferences: styleToUse,
          occasion: occasionToUse || 'Any',
          selectedItemIds: [],
          preferences: withOutfitBudget(useStylePreferenceStore.getState().preferences, shopBudget),
          wardrobeItems: effectiveSource === 'shop' ? [] : payloadItems,
          catalogOnly: effectiveSource === 'shop',
          weather: resolvedWeather,
          limit: 5,
          // The edge asks for a few extra candidates but the user sees three
          // cards. Capacity must be judged against the DISPLAYED deck size,
          // otherwise a 5-candidate request would mark a 3-pair shoe slot
          // "thin" and license repeating shoes on the three visible cards.
          deckTarget: 3,
          prompt: edgePrompt,
          // A locked owned piece remains the anchor; the other slots use the catalog.
          anchorItemId: baseItemId,
          // When source is 'shop', use only the provided shop catalog items
          // (Massimo Dutti etc.) and do NOT mix in user's own wardrobe items
          useProvidedWardrobeOnly: effectiveSource === 'shop',
          // Wardrobe mode: tell the edge to build ONLY from Supabase
          // clothing_items + provided wardrobe items — no shop_catalog
          // supplement, no static fill.
          wardrobeOnly: effectiveSource === 'wardrobe',
          // Per-tap salt: the edge fallback rotation is otherwise a pure
          // function of style+occasion, rebuilding identical decks.
          deckSalt: tapNonce,
          // Recently-shown window → edge builds the candidate pool without
          // pieces the user just saw, so each tap surfaces new clothing.
          excludeItemIds,
          excludeProductKeys,
          previousDeckProductKeys: rotationHistory.previousDeckProductKeys,
          olderRecentProductKeys: rotationHistory.olderRecentProductKeys,
          rotationScope: recencyScope,
          generationRequestId,
          clientPathId: 'AIOutfitmaker-direct',
          clientRuntimeMarker: CLIENT_RUNTIME_MARKER,
        },
      }), EDGE_TIMEOUT_MS, cancelSignal);

      if (!isCurrentGeneration()) return;
      if (invokeError) {
        // eslint-disable-next-line no-console
        if (__DEV__) console.log('[stylist-http-request]', { generationRequestId, clientPathId: 'AIOutfitmaker-direct' });
        throw await outfitGenerationFailure(data, invokeError);
      }
      if (__DEV__) readinessLogger.debug('[rotation-client] RESPONSE', {
        generationRequestId,
        source: data?.source,
        provider: data?.provider,
        firstOutfitIds: Array.isArray(data?.outfits?.[0]?.items)
          ? data.outfits[0].items.map((item: any) => String(item?.id || ''))
          : [],
      });

      if (shopBudget?.fx && data?.success && data?.budgetReference?.id !== shopBudget.fx.snapshot.id) throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable');
      if (data && data.success && data.outfits && data.outfits.length > 0) {
        // Feed the rotation window BEFORE normalizing — the next tap must
        // exclude these exact ids even if the deck is edited downstream.

        const slotDeckUsed = new Set<string>();
        const wrapLook = (outfit: any, items: any[], idx: number): GeneratedOutfit => {
          const description =
            effectiveSource === 'shop'
              ? `${stylePhraseFor(styleToUse)} look curated from the shop.`
              : outfit.description;
          const judgeOk = passesStyleJudge(items as any[], styleToUse, `${occasionToUse} ${effectivePrompt || promptText || ''}`, resolvedWeather, baseItemId);
          const baseScore: number = outfit.confidence ?? outfit.matchScore ?? 0.9;
          return {
            ...outfit,
            id: outfit.id || `look_${tapNonce}_${idx}`,
            description,
            items,
            mainImage: items[0]?.image || items[0]?.imageUrl || outfit.mainImage || '',
            matchScore: judgeOk ? baseScore : Math.min(baseScore, 0.69),
            stylingTips: (() => {
              const raw = Array.isArray(outfit.stylingTips)
                ? outfit.stylingTips.join(' · ')
                : (outfit.stylingTips || '');
              if (raw.trim()) return raw;
              // Edge sometimes returns stylingTips: [] — an empty tip box on
              // the card. Fall back to the same i18n tip sets the offline
              // builder uses, cycled per card index.
              const fallbackSets = [
                ['outfitMaker.tip1_1', 'outfitMaker.tip1_2'],
                ['outfitMaker.tip2_1', 'outfitMaker.tip2_2'],
                ['outfitMaker.tip3_1', 'outfitMaker.tip3_2'],
              ];
              return fallbackSets[idx % fallbackSets.length].map((k) => t(k)).join(' · ');
            })(),
            weather: resolvedWeather,
          };
        };
        const cleanedOutfits = data.outfits.map((outfit: any, i: number) => {
          const repaired = repairEdgeItems(outfit.items || [], true);
          const filled = repaired;
          return wrapLook(outfit, filled, i);
        }).filter((outfit: GeneratedOutfit) =>
          isFourSlotLook(outfit.items, true),
        );

        // Taste judge: drop cards carrying pieces rejected for this style ×
        // occasion. Never restore rejected looks to pad the deck — fill
        // missing cards from the pool instead.
        const judged = cleanedOutfits.filter((outfit: GeneratedOutfit) =>
          passesStyleJudge(outfit.items as any[], styleToUse, `${occasionToUse} ${effectivePrompt || promptText || ''}`, resolvedWeather, baseItemId),
        );
        if (__DEV__ && judged.length !== cleanedOutfits.length) {
          readinessLogger.debug(String(`[AIOutfitmaker] style judge: kept ${judged.length}/${cleanedOutfits.length} for ${styleToUse}/${occasionToUse}`), []);
          cleanedOutfits.forEach((o: GeneratedOutfit, i: number) => {
            if (!judged.includes(o)) {
              readinessLogger.debug(String(`[AIOutfitmaker] style judge: drop edge #${i} reason="${styleJudgeReason(o.items as any[], styleToUse, `${occasionToUse} ${effectivePrompt || promptText || ''}`, resolvedWeather) || '?'}" items (${(o.items as any[]).map((x: any) => String(x.name || '?').slice(0, 24)).join(' | ')})`), []);
            }
          });
        }
        // Keep EVERY validated look: the diversity gate below picks the best
        // COMBINATION of 3, so trimming to 3 here would throw away exactly
        // the alternatives the stylist needs.
        const judgedOutfits = judged;

        let withinBudget = shopBudget
          ? judgedOutfits.filter((outfit: GeneratedOutfit) => fitsCap(outfit.items as any[]))
          : judgedOutfits;
        // Never restore an over-cap or unpriced look after budget validation.
        if (withinBudget.length === 0 && judgedOutfits.length > 0) {
          if (__DEV__ && shopBudget) {
            readinessLogger.debug(String('[AIOutfitmaker] no judged outfits satisfy the verified budget'), [{
              budget: shopBudget,
              spends: judgedOutfits.map((o: GeneratedOutfit) => outfitShopSpend(o.items as any[], liveShopMapped, shopBudget?.currency, shopBudget?.fx)),
            }]);
          }
          withinBudget = [];
        }
        if (__DEV__ && shopBudget) {
          readinessLogger.debug(String('[AIOutfitmaker] budget'), [shopBudget, 'spends', judgedOutfits.map((o: GeneratedOutfit) => outfitShopSpend(o.items as any[], liveShopMapped, shopBudget?.currency, shopBudget?.fx)), 'kept', withinBudget.length]);
        }
        const budgetedOutfits = withinBudget;

        // Diversity gate: no non-anchor product may repeat across looks
        // (same SKU, normalized name, or normalized image). Same archetype
        // and same garment family are allowed when the products differ.
        // Near-duplicate edge looks are replaced from the remaining pool
        // before anything is relaxed. Accepted looks seed slotDeckUsed, so
        // fills rotate every slot. Uniqueness is relaxed only when the
        // inventory genuinely cannot supply enough distinct products.
        // Product identity: id + normalized name + normalized image. Garment
        // family is NOT identity — a black loafer and a brown loafer are two
        // products, not a duplicate.
        const productKeysOf = (x: any): string[] => productIdentityKeys(x);
        const anchorKeys = baseItemId
          ? new Set<string>([
              String(baseItemId),
              ...[...budgetedOutfits.flatMap((o: GeneratedOutfit) => o.items as any[]), ...genPool]
                .filter((x: any) => String(x?.id ?? '') === String(baseItemId))
                .flatMap((x: any) => productKeysOf(x)),
            ])
          : undefined;
        // The anchor pins one slot of the card. That slot is excluded from the
        // VISUAL comparison: repeating the anchor is the user's explicit
        // choice, repeating the rest of the formula is not.
        const anchorSlot = ((): DiversitySlot | null => {
          if (!baseItemId) return null;
          const anchorItem = [...budgetedOutfits.flatMap((o: GeneratedOutfit) => o.items as any[]), ...genPool]
            .find((x: any) => String(x?.id ?? '') === String(baseItemId));
          return anchorItem ? diversitySlotOf(anchorItem) : null;
        })();
        if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] fill start: judged=${judgedOutfits.length} budgeted=${budgetedOutfits.length}`), []);
        // The edge's capacity verdict (active / strict / forced-repeat slots)
        // as reported by the authoritative valid pool.
        const edgeDiversityCapacity = ((): { activeSlots?: DiversitySlot[]; strictSlots?: DiversitySlot[]; forcedRepeatSlots?: DiversitySlot[] } | undefined => {
          const raw = (data as any)?.diversity;
          if (!raw || typeof raw !== 'object') return undefined;
          const slots = (value: unknown): DiversitySlot[] | undefined =>
            Array.isArray(value) ? value.filter((slot): slot is DiversitySlot => typeof slot === 'string') : undefined;
          const active = slots(raw.activeSlots);
          const strict = slots(raw.strictSlots);
          const forced = slots(raw.forcedRepeatSlots);
          if (!active && !strict && !forced) return undefined;
          return { activeSlots: active, strictSlots: strict, forcedRepeatSlots: forced };
        })();
        // The client renders 4-slot cards (layer + top + bottom + shoes), so
        // every card slot participates in the composition.
        const clientDiversitySlots: DiversitySlot[] = ['top', 'outerwear', 'bottom', 'shoes'];
        const {
          deck: finalOutfits,
          relaxed: deckRelaxed,
          relaxReason: deckRelaxReason,
          archetypes: deckArchetypes,
          similarities: deckSimilarities,
          inventory: deckInventory,
        } = assembleDiverseDeck<GeneratedOutfit>({
          candidates: budgetedOutfits,
          itemsOf: (o) => o.items as any[],
          qualityOf: (o) => Number(o.matchScore ?? 0.9),
          keysOf: productKeysOf,
          anchorKeys,
          anchorSlot,
          // Every card the client renders is a 4-slot look
          // (layer + top + bottom + shoes), so all four slots participate.
          activeSlots: clientDiversitySlots,
          // The user sees three cards.
          target: 3,
          // The edge owns the valid pool and therefore the capacity verdict.
          // Re-deriving it from the truncated client pool would reject the
          // edge's legitimate forced repeat and cost the user cards.
          capacity: edgeDiversityCapacity,
          usedKeys: slotDeckUsed,
          // The VALID pool (style / occasion / climate filtered) decides
          // whether the inventory is genuinely thin. A catalogue full of
          // alternatives must never fall back to three colourways of one
          // formula, so relaxation is gated on this assessment.
          poolItems: payloadItems,
          debug: __DEV__
            ? (message: string, data?: unknown) => readinessLogger.debug(String(`[stylist-diversity] ${message}`), data ? [data] : [])
            : undefined,
        });
        const deckProducts = analyzeDeckProductDiversity(
          finalOutfits.map((outfit) => outfit.items as any[]),
          { anchorKeys, forcedSlots: deckInventory?.forcedRepeatSlots },
        );
        if (__DEV__) {
          readinessLogger.debug(String('[AIOutfitmaker] stylist deck'), [{
            archetypes: deckArchetypes,
            similarities: deckSimilarities.map((s) => `${s.a + 1}<->${s.b + 1}=${s.score.toFixed(2)}`),
            relaxed: deckRelaxed,
            reason: deckRelaxReason,
            uniqueProductsPerSlot: deckInventory?.uniqueProductsPerSlot,
            maxDistinctProducts: deckInventory?.maxDistinctProducts,
            supportsUniqueProducts: deckInventory?.supportsUniqueProducts,
            nonAnchorProductReuseCount: deckProducts.nonAnchorProductReuseCount,
            illegalProductReuseCount: deckProducts.illegalProductReuseCount,
            forcedRepeatProducts: deckProducts.forcedRepeatProducts,
            anchorRepeatProducts: deckProducts.anchorRepeatProducts,
            repeatedProductKeys: deckProducts.repeatedProductKeys,
            anchorRepetitionCount: deckProducts.anchorRepetitionCount,
          }]);
          const decision = deckProducts.illegalProductReuseCount > 0
            ? 'REPEATED_PRODUCTS'
            : (deckProducts.forcedRepeatProducts.length > 0
              ? 'FORCED_SLOT_REUSE'
              : 'UNIQUE_PRODUCTS');
          readinessLogger.debug(String('[stylist-product-diversity]'), [JSON.stringify({
            decision,
            nonAnchorProductReuseCount: deckProducts.nonAnchorProductReuseCount,
            illegalProductReuseCount: deckProducts.illegalProductReuseCount,
            forcedRepeatProducts: deckProducts.forcedRepeatProducts,
            anchorRepeatProducts: deckProducts.anchorRepeatProducts,
            repeatedProductKeys: deckProducts.repeatedProductKeys,
            repeatedCanonicalKeys: deckProducts.repeatedCanonicalKeys,
            repeatedProducts: deckProducts.repeatedProducts,
            anchorRepetitionCount: deckProducts.anchorRepetitionCount,
            lookProductKeys: deckProducts.lookProductKeys,
            lookCanonicalProducts: finalOutfits.map((outfit) =>
              (outfit.items as any[]).map((item) => canonicalProductKey(item)),
            ),
            looks: finalOutfits.map((outfit) => (outfit.items as any[]).map((item) => String(item?.id ?? ''))),
            relaxReason: deckRelaxReason,
            supportsUniqueProducts: deckInventory?.supportsUniqueProducts,
            activeSlots: deckInventory?.activeSlots,
            strictSlots: deckInventory?.strictSlots,
            forcedRepeatSlots: deckInventory?.forcedRepeatSlots,
            uniqueProductsPerSlot: deckInventory?.uniqueProductsPerSlot,
            deckSize: finalOutfits.length,
            deckTarget: 3,
            hasQualitySignal: true,
            // End-to-end marker: the edge shipped its capacity verdict, so
            // the client did NOT have to re-derive it from a truncated pool.
            edgeCapacity: edgeDiversityCapacity ? {
              activeSlots: edgeDiversityCapacity.activeSlots,
              strictSlots: edgeDiversityCapacity.strictSlots,
              forcedRepeatSlots: edgeDiversityCapacity.forcedRepeatSlots,
            } : null,
          })]);
        }
        if (__DEV__ && deckRelaxed) readinessLogger.debug(String(`[AIOutfitmaker] deck relaxed (${deckRelaxReason}) after strict diverse alternatives were exhausted`), []);
        if (__DEV__) readinessLogger.debug(String(`[AIOutfitmaker] nonce=${generationNonce.current} edge=${cleanedOutfits.length} shown=${finalOutfits.length}`), []);

        if (finalOutfits.length !== DISPLAY_DECK_SIZE) {
          // eslint-disable-next-line no-console
          if (__DEV__) console.log('[stylist-deck-rejected]', { generationRequestId, source: data.source, provider: data.provider, layered: data.layered, received: data.outfits.length, cleaned: cleanedOutfits.length, judged: judgedOutfits.length, budgeted: budgetedOutfits.length, selected: finalOutfits.length, reasons: cleanedOutfits.map((o: GeneratedOutfit) => styleJudgeReason(o.items as any[], styleToUse, `${occasionToUse} ${effectivePrompt || promptText || ''}`, resolvedWeather)), items: data.outfits.map((o: any) => o.items?.map((x: any) => ({ id: x.id, name: x.name, type: x.type, category: x.category, macroCategory: x.macroCategory }))) });
          throw new Error('no_valid_outfits_for_request');
        }

        if (!isCurrentGeneration()) return;
        if (isMounted.current) {
          // Only validated, in-cap looks may become visible.
          const inCap = finalOutfits.filter((o) => fitsCap(o.items as any[]));
          const overCap = finalOutfits.filter((o) => !fitsCap(o.items as any[]));
          const shownOutfits = inCap.slice(0, DISPLAY_DECK_SIZE);
          if (shownOutfits.length !== DISPLAY_DECK_SIZE) throw new Error('no_outfits_within_budget');
          setOutfits(
            shownOutfits
              .map((o) => {
                // Base description first (shop source always rewrites it),
                // THEN the budget line — overwriting after the append is what
                // silently dropped "· 2 400 000 UZS / 3 150 000 UZS".
                const baseDescription = effectiveSource === 'shop'
                  ? `${stylePhraseFor(styleToUse)} look curated from the shop.`
                  : o.description;
                return shopBudget
                  ? { ...o, description: `${baseDescription} · ${formatUzsPrice(outfitShopSpend(o.items as any[], liveShopMapped, shopBudget?.currency, shopBudget?.fx), shopBudget.currency, displayRates) || '— UZS'} / ${formatUzsPrice(shopBudget.amount, shopBudget.currency, displayRates) || '— UZS'}${shopBudget.fx ? ` · ${t('outfitMaker.referenceRate', {date: shopBudget.fx.snapshot.asOf})}` : ''}` }
                  : { ...o, description: baseDescription };
              }),
          );
          if (__DEV__) readinessLogger.debug('[rotation-client] DISPLAY', {
            generationRequestId,
            firstOutfitIds: shownOutfits[0]?.items?.map((item) => String(item.id)) || [],
          });
          recordShownOutfits(shownOutfits, recencyScope);
          rememberShownIds(shownOutfits);
          setLoading(false);
        }
        return;
      }

      if (__DEV__ && data?.debug) readinessLogger.debug(String('[AIOutfitmaker] edge debug:'), [data.debug]);
      const edgeErr: any = await outfitGenerationFailure(data);
      if (data?.debug) edgeErr.debug = data.debug;
      throw edgeErr;
    } catch (err: any) {
      console.warn('[AIOutfitmaker] Edge function failed:', err, err?.stack || '', err?.debug ? JSON.stringify(err.debug) : '');
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
      if (!isCurrentGeneration()) return;
      if (isMounted.current) {
        setOutfits([]);
        setLoading(false);
        setError(t(err instanceof OutfitGenerationFailure ? err.translationKey : 'outfitMaker.generateFailed', {
          defaultValue: 'Could not create a complete outfit that meets your request. Try again or add suitable items.',
        }));
      }
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
            <TouchableOpacity
              onPress={() => setOutfits([])}
              style={styles.backButton}
              testID="aiResultsBack"
              accessibilityLabel={t('common.back')}
              accessibilityRole="button"
            >
              <Ionicons name="chevron-back" size={22} color="#0A1931" />
            </TouchableOpacity>
            <ScaledText style={styles.headerTitle}>{t('outfitMaker.yourOutfits')}</ScaledText>
            <View style={styles.headerRight} />
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
              <Animated.View key={outfit.id} testID={`stylistResult-${index}`} style={[styles.glassCard, { marginTop: index === 0 ? 8 : 24 }]}>
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
                    <ScaledText style={styles.matchPillText}>{t('outfitMaker.aiCurated')}</ScaledText>
                  </LinearGradient>
                </View>

                <OutfitSlotGrid
                  items={outfit.items}
                  weather={outfit.weather}
                  priceCatalog={liveShopMapped}
                  showBuyButtons={currentSource === 'shop'}
                />

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
                <TouchableOpacity onPress={() => setCalendarDate(current => shiftCalendarDay(current, -1))}>
                  <Ionicons name="chevron-back" size={24} color={D.textPrimary} />
                </TouchableOpacity>
                <ScaledText style={styles.datePickerValue} minScale={0.6}>{calendarDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</ScaledText>
                <TouchableOpacity onPress={() => setCalendarDate(current => shiftCalendarDay(current, 1))}>
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
              <AIThinkingAnimation styleName={selectedStyleLabel} clothingItems={animationItems} source={currentSource} />
            </View>
            {/* Floating cancel — overlays animation, takes no layout space */}
            <SafeAreaView
              edges={['top', 'left', 'right']}
              style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 30 }}
              pointerEvents="box-none"
            >
              <View style={{ paddingHorizontal: 16, paddingTop: 8, alignItems: 'flex-start' }} pointerEvents="box-none">
                <TouchableOpacity
                  onPress={cancelOutfitGeneration}
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

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={{ flex: 1 }}
      >
      <SafeAreaView
        edges={['top', 'left', 'right']}
        style={{ flex: 1 }}
      >
        {/* Header — same 3-part centered pattern as main tabs */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={styles.backButton}
            testID="aiSetupBack"
            accessibilityLabel={t('common.back')}
            accessibilityRole="button"
          >
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
            {' · '}
            {weather ? `${weather.temp}°C` : 'Weather unavailable'}
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
                      testID={`stylistOccasion-${occ.id}`}
                      activeOpacity={0.85}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setSelectedOccasion(on ? undefined : (occ.id as string));
                        commentEditedRef.current = false;
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
                      testID="stylistMoreOccasions"
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
                    <TouchableOpacity
                      testID={`stylistStyle-${activeStyle.id}`}
                      activeOpacity={0.9}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setSelectedStyle(activeStyle.id);
                        commentEditedRef.current = false;
                      }}
                      style={styles.featureCard}
                    >
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
                    </TouchableOpacity>
                    <TouchableOpacity
                      activeOpacity={0.75}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setShowAllStyles((v) => !v);
                      }}
                      style={styles.moreToggle}
                      testID="stylistMoreStyles"
                      accessibilityRole="button"
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
                              () => {
                                setSelectedStyle(styleObj.id);
                                commentEditedRef.current = false;
                              },
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
                  testID="stylistCommentInput"
                  style={styles.detailsInput}
                  placeholder={t('outfitMaker.promptPlaceholderShort', {
                    defaultValue: "Where you're going, style, budget (1 500 000 UZS)…",
                  })}
                  placeholderTextColor={D.textSecondary}
                  value={userPrompt}
                  onChangeText={(text) => {
                    commentEditedRef.current = true;
                    promptRef.current = text;
                    setUserPrompt(text);
                  }}
                  onChange={(e) => {
                    const nativeText = (e as any)?.nativeEvent?.text;
                    if (typeof nativeText === 'string') promptRef.current = nativeText;
                  }}
                  multiline
                  numberOfLines={3}
                  textAlignVertical="top"
                  returnKeyType="done"
                  onSubmitEditing={() => handleGeneratePress()}
                />
              </View>


              {currentSource === 'shop' && !shopCatalogLoading && liveShopMapped.length === 0 ? (
                <View style={[styles.errorBox, { marginTop: 12 }]}>
                  <ScaledText style={styles.errorText}>
                    {shopCatalogError
                      ? `Live catalog failed: ${shopCatalogError}. New clothing comes only from the live shop catalog — no fallback.`
                      : 'Live catalog returned 0 items. New clothing comes only from the live shop catalog — no fallback.'}
                  </ScaledText>
                  <TouchableOpacity
                    activeOpacity={0.7}
                    onPress={() => refreshShopCatalog()}
                    style={{ marginTop: 8, alignSelf: 'flex-start', paddingHorizontal: 14, paddingVertical: 8, borderRadius: 16, backgroundColor: '#0A1931' }}
                  >
                    <ScaledText style={{ color: '#FFF', fontWeight: '700', fontSize: 13 }}>Retry live catalog</ScaledText>
                  </TouchableOpacity>
                </View>
              ) : null}
            </ScrollView>
          </Reanimated.View>
          {/* Keep failures visible beside the retry action, outside scrolling choices. */}
          <View style={styles.stickyFooter}>
            {error ? <View accessibilityRole="alert" accessibilityLiveRegion="assertive" style={[styles.errorBox, { marginBottom: 12 }]}>
              <ScaledText style={styles.errorText}>{error}</ScaledText>
            </View> : null}
            <TouchableOpacity
              testID="stylistGenerate"
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
      </KeyboardAvoidingView>

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
