import {validateOutfitGenerationRequest,OutfitRequestError} from '../src/domain/outfitGenerationRequest';
import {outfitDescription,outfitConceptTitle,distinctStylingTips,resolveSelectionSource,selectionBadgeKey} from '../src/services/stylingCopy';
import { recordLookFeedback, canRecordLookFeedback } from '../src/services/stylistFeedback';
import { prepareStylistResponse } from '../features/outfit-generator/utils/stylistResponse';
import {StylistRuntimeDiagnostic, clientRevision} from '../src/components/StylistRuntimeDiagnostic';
import {STYLIST6_DEV_ENABLED, invokeStylist6Bridge, invokeStylist6Cloud, buildStylist6BridgeBody, acceptStylist6DevDeck, acceptStylist6CloudDeck, type DevEngine} from '../src/services/stylist6Dev';
import {loadStylistCatalogInventory} from '../src/services/stylistCatalogInventory';
import {displayedRotation} from '../src/domain/outfitRotation';
import {outfitCoherence} from '../src/domain/outfitCoherence';
import {recentLookSimilarity} from '../src/domain/outfitVisualSignature';
import {stylingCopyParts} from '../src/services/stylingCopy';
import {stylingContext} from '../src/domain/stylingContext';
import { BetaFeedbackButton } from '../components/BetaFeedbackButton';
import { stylistCalendarItems } from '../features/calendar/outfitSlots';
import {formatUzsPrice} from '../src/utils/uzsPrice';
import { useUzsDisplayRates } from '../hooks/useUzsPrice';
import { resolveStylistWeather } from '../src/domain/outfitComposition';
import { getBudgetRates } from '../src/services/budgetRates';
import { budgetFxContext, fitsBudgetAmount, type BudgetFxSnapshot } from '../src/domain/budgetFx';
import { parseBudgetControl, strictestOutfitBudget } from '../src/domain/outfitBudgetSelection';
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
import { RequestTimeoutError } from '../src/lib/requestDeadline';
import { OutfitGenerationFailure, outfitGenerationFailure } from '../src/services/outfitGenerationFailure';
import { useShopCatalog } from '../hooks/useShopCatalog';
import { recordShownOutfits, rotationScope, generationRequestId, getRotationHistory, recordDisplayedDeck } from '../src/services/recentOutfitIds';
import { collapseSimilarByName, isTrustedProductImage, isBrandCatalogImage, preferMappedMacro, hasUsableProductName, inferCatalogColor, inferShopRowMacro } from '../src/utils/shopCatalogOrder';
import { passesStyleJudge, humanizeProductName, SHORT_SLEEVE_RE, STRIP_SHORT_SLEEVE_RE } from '../features/outfit-generator/utils/occasionRules';
import LiquidGlass2026Theme, { SpatialElevation } from '../constants/LiquidGlass2026Theme';
import * as Location from 'expo-location';
import Config from '../src/config/env';
import { saveCloudOutfit } from '../src/services/savedOutfitService';
import { confirmedImageReplacements, replaceGeneratedOutfitImages, replaceSavedOutfitImages } from '../src/utils/confirmedOutfitMedia';
import { localDateKey, parseCalendarDate, shiftCalendarDay } from '../src/utils/calendarDate';
import { authEvents } from '../store/authEvents';
import { readCalendarLogs, updateCalendarLogs } from '../src/services/calendarStorage';
import { createOutfitLog, resolveCalendarOccasion } from '../features/calendar/types';
import AIThinkingAnimation from '../components/AIThinkingAnimation';
import { canonicalizeMacroCategory } from '../src/utils/categoryMapper';
import {
    analyzeDeckProductDiversity,
    assembleDiverseDeck,
    canonicalProductKey,
    diversitySlotOf,
    productIdentityKeys,
    type DiversitySlot,
} from '../features/outfit-generator/utils/outfitDiversity';
import { isShortsBottom, isFormalLayer, fabricFamily, isSameLayerBaseCategory, PLACEHOLDER_PANTS, PLACEHOLDER_SHOES, getBackendMacroCategory, isCompleteNormalizedOutfit, isFourSlotLook, REAL_IMAGE_RE, createSlotNormalizer, ensureSlotFloors } from '../features/outfit-generator/utils/normalizeSlots';
import { parseBudgetFromText, stripBudgetFromText, outfitShopSpend, outfitTotalQuote, formatMoney, budgetCandidatesFromText, budgetInCurrency, resolveShopQuote, withOutfitBudget, mixedCurrencyPriced } from '../src/utils/outfitBudget';
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
  compositionProtocol?: 'optional-layer-v1';
  explanationVersion?: number;
  concept?: {id:string;title:string;story:string;score?:number};
  /** Catalog-grounded directions, not a cosmetic "three styles" claim. */
  direction?: {id:'signature'|'refresh'|'discovery'|'alternative';qualified:boolean; evidence?:{
    changedSlots?: string[];structuralSlots?:string[];paletteSlots?:string[];distance?:number;
  }};
  /** Only positive matches against actual selected SKU evidence. No raw profile. */
  fashionDiagnostics?: {personalization?: {
    adjustment?: number;
    matchedItems?: number;
    matches?: {type:string;value:string;productId:string}[];
  }};
  selectionSource?: string;
  generationRequestId?:string;
  engineVersion?:string;
  id: string;
  mainImage: string;
  matchScore: number;
  description: string;
  items: OutfitItem[];
  stylingTips?: string;
  /** Styling inputs captured at generation time so result copy can be
   * re-localized when the app language changes without a new request. */
  stylingStyle?: string;
  stylingOccasion?: string;
  stylingRequest?: string;
  /** True when tips came from the server and must not be re-derived. */
  serverTips?: boolean;
  weather?: { temp: number; condition: string; city?: string };
}

// ── Static catalog fallback, normalized ONCE ────────────────────────────
// candidatePool re-ran toMacro regexes over ~2k bundled items on every
// wardrobe change. Static data never changes — normalize once at module
// scope and reuse. Each entry carries its macro + precomputed image flag.
// Stylist rule-set version — printed on every Generate tap log so a pasted
// log proves exactly which vetoes/picks were live in that build.
export const STYLIST_RULES_VERSION = 'stylist-quality-v1';
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

type TranslateFn = (key: string, opts?: any) => string;
const getAiStyleLabel = (id: string, t: TranslateFn) => {
  const style = aiStyles.find(s => s.id === id);
  return t(`outfitMaker.style_${id}`, { defaultValue: style?.label || id });
};

const getAiStyleDesc = (id: string, t: TranslateFn) => {
  const style = aiStyles.find(s => s.id === id);
  return t(`outfitMaker.styleDesc_${id}`, { defaultValue: style?.desc || '' });
};

/** Localized occasion preset name. `date` renders as "Date night" via dateNight. */
const getOccasionLabel = (id: string, t: TranslateFn) => {
  const preset = OCCASION_PRESETS.find(o => o.id === id);
  const key = id === 'date' ? 'dateNight' : id;
  return t(`outfitMaker.${key}`, { defaultValue: preset?.label || id });
};

/** Single-currency card total from the same quote resolver the tiles use.
 * Hidden whenever any shop piece is unquoted or currencies are mixed. */
function outfitTotalDisplay(items: any[], catalog: any[] | undefined, rates: BudgetFxSnapshot | null,
  budget?: ReturnType<typeof parseBudgetFromText>): { label: string; converted: boolean } | null {
  const total = outfitTotalQuote(items as any[], catalog as any[]);
  if (!total?.currency) {
    if (!budget) return null;
    const spend = outfitShopSpend(items as any[], catalog as any[], budget.currency, budget.fx);
    return Number.isFinite(spend) ? {label: `${formatMoney(spend, budget.currency)} / ${formatMoney(budget.amount, budget.currency)}`, converted: !!budget.fx} : null;
  }
  // UZS conversion only when display rates are actually loaded; otherwise the
  // exact native-currency total stands, never a relabelled guess.
  const label = formatUzsPrice(total.price, total.currency, rates) || formatMoney(total.price, total.currency);
  return label ? { label, converted: false } : null;
}

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
  // Cards always show the original listing price and currency. Never convert
  // per-card prices: a converted guess would mislabel the listing, and
  // mixed-currency outfits have no valid combined total.
  const displayPrice = typeof price === 'number' && price > 0 && priceCurrency
    ? formatMoney(price, priceCurrency)
    : null;
  const handleBuy = React.useCallback(() => onBuy(buyUrl), [onBuy, buyUrl]);
  const [imgFailed, setImgFailed] = React.useState(false);
  React.useEffect(() => { setImgFailed(false); }, [imageUri, slotKey]);
  return (
    <View key={slotKey} style={slotStyles.cell}>
      <ExpoImage
        source={imgFailed ? undefined : imageSource}
        placeholderContentFit="contain"
        style={[slotStyles.image, { backgroundColor: '#EEF3F8' }]}
        contentFit="contain"
        cachePolicy="memory-disk"
        recyclingKey={`${slotKey}-${itemId ?? ''}-${imageUri ?? ''}-${imgFailed ? 'fb' : 'src'}`}
        onError={() => setImgFailed(true)}
      />
      <View style={slotStyles.labelWrap}>
        <ScaledText style={[slotStyles.label, { color: textSecondary }]}>{label}</ScaledText>
        <ScaledText style={[slotStyles.name, { color: textPrimary }]} numberOfLines={1}>{name}</ScaledText>
        {typeof price === 'number' && price > 0 && priceCurrency ? (
          <ScaledText style={slotStyles.price} numberOfLines={1}>
            {displayPrice || '—'}
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
    const labels: Record<string, string> = { top: t('calendar.top'), outerwear: t('calendar.layer'), bottom: t('calendar.bottom'), shoes: t('calendar.shoes') };
    const keys: Record<string, string> = { top: 'second-top', outerwear: 'main-top', bottom: 'pants', shoes: 'shoes' };
    return { key: `${keys[macro] || macro}-${item.id}`, label: labels[macro] || t('common.item'), item };
  }), [items, t]);

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
            showBuyButton={showBuyButtons && slot.item.isWardrobe !== true}
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

// Generation and offline fallback share the deterministic domain kernel.

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

const AIOutfitGenerator = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  // ── Post-trial soft gate: page stays open, Generate → Paywall ──
  // free.aiOutfits = 0 → consumeOrPaywall denies + navigates to Paywall.
  // Trial-active (effectiveTier premium) and paid users pass through.
  const { requireFeature, consumeOrPaywall } = useSubscriptionGate();
  const [lookRatings,setLookRatings]=useState<Record<string,number>>({});
  const [feedbackPending,setFeedbackPending]=useState<string|null>(null);
  const submitLookFeedback=async(look:GeneratedOutfit,rating:-1|1)=>{
    const key=`${look.generationRequestId}:${look.id}`;
    if(feedbackPending)return;
    setFeedbackPending(key);
    try {await recordLookFeedback(look,rating);setLookRatings(previous=>({...previous,[key]:rating}));}
    catch {Alert.alert(t('common.error'),t('outfitMaker.feedbackUnavailable'));}
    finally {setFeedbackPending(null);}
  };
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
  const { t, i18n } = useTranslation();
  // ── Quiz-oriented defaults ──────────────────────────────────────────
  // When the user arrives from the Style Quiz, the generator orients on
  // their quiz personality instead of a hardcoded fallback.
  const quizPersonality = useStylePreferenceStore((s) => s.preferences.stylePersonality);
  const stylistPreferences = useStylePreferenceStore((s) => s.preferences);
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
  // Switching languages re-renders static labels via t(), but stored result
  // copy (descriptions, tips) was localized at generation time. Re-derive it
  // from the stored garments so the result screen refreshes without a new
  // paid generation request. Concept titles already resolve at render time.
  const resultLanguage = i18n.language;
  useEffect(() => {
    setOutfits(current => {
      if (!current.length) return current;
      let changed = false;
      const next = current.map(look => {
        const items = (look.items || []) as any[];
        if (!items.length || !look.stylingStyle) return look;
        const ctx = stylingContext({
          style: look.stylingStyle,
          occasion: look.stylingOccasion || '',
          request: look.stylingRequest || '',
          weather: look.weather,
        });
        const description = outfitDescription(look, items, ctx);
        const tips = look.serverTips
          ? (look.stylingTips || '')
          : distinctStylingTips(description, stylingCopyParts(items, ctx).tips).join(' ');
        if (description === look.description && tips === (look.stylingTips || '')) return look;
        changed = true;
        return { ...look, description, stylingTips: tips };
      });
      return changed ? next : current;
    });
  }, [resultLanguage]);
  const [resultBudget, setResultBudget] = useState<ReturnType<typeof parseBudgetFromText>>(null);
  const [inventoryNote, setInventoryNote] = useState('');
  const [error, setError] = useState('');
  // Development-only engine choice. The picker renders only when
  // STYLIST6_DEV_ENABLED, so release builds can never select Stylist 6.
  const [devEngine, setDevEngine] = useState<DevEngine>(STYLIST6_DEV_ENABLED ? 'stylist6cloud' : 'stylist5');
  // Dev-only opt-in: hosted NVIDIA visual review for Stylist 6 (local
  // backend only, bounded to one small batch per deck). Off by default so
  // ordinary taps stay fast and free-tier calls stay bounded.
  const [nvidiaVisual, setNvidiaVisual] = useState(false);
  // Actual engine that produced the visible deck (or null before/after).
  const [lastEngineVersion, setLastEngineVersion] = useState<string | null>(null);
  const [lastEngineDetail, setLastEngineDetail] = useState<string | null>(null);
  // Dev-only: real failure code of the last failed generation.
  const [lastFailureCode, setLastFailureCode] = useState<string | null>(null);
  const displayRates = useUzsDisplayRates();
  const [userPrompt, setUserPrompt] = useState('');
  const [budgetAmount, setBudgetAmount] = useState('');
  const [budgetCurrency, setBudgetCurrency] = useState<'UZS' | 'USD' | 'EUR' | 'GBP'>('UZS');
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
  const { items: liveShopCatalog, loading: shopCatalogLoading, error: shopCatalogError } = useShopCatalog({
    enabled: true,
    source: 'all',
    discovery: true,
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

  useEffect(() => {
    isMounted.current = true;
    
    let active = true;
    if (currentSource === 'wardrobe') loadWardrobeItems(() => active);
    else setWardrobeItems([]);
    
    return () => {
      active = false;
      isMounted.current = false;
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
    };
  // baseItemId/baseItem included: anchor must survive source switches
  // (wardrobe ↔ shop) without losing the lock.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSource, baseItemId]);

  const loadWardrobeItems = async (isActive: () => boolean) => {
    try {
      // New clothing comes from the Edge catalog search; no local catalog
      // page needs to enter the generation pool.
      if (currentSource === 'shop') {
        if (!isMounted.current || !isActive()) return;
        setWardrobeItems([]);
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

      if (!isMounted.current || !isActive()) return;
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
    if (currentSource !== 'wardrobe') return [];
    const sourcePool = (wardrobeItems || []).filter((it: any) => it?.isWardrobe === true);
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
        description: it.description, material: it.material, pattern: it.pattern,
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
    return mapped;
  }, [currentSource, wardrobeItems]);

  const availableStyles = aiStyles.filter(style => style.id !== 'business_casual');
  const availableOccasions = OCCASION_PRESETS;

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
    const mappedItems = stylistCalendarItems(outfit.items,outfit.compositionProtocol);
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
      compositionProtocol:outfit.compositionProtocol,
      userId: useAuthStore.getState().user?.id || '',
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

    if(!newOutfitId){Alert.alert(t('outfitMaker.cannotSave'),t('outfitMaker.noValidItems'));return;}
    let confirmedOutfit = outfit;
    const snapshotItems = outfit.items.map(item => ({ id: String(item.id || item.image), type: item.type || 'Clothing Piece', category: item.category, macroCategory: item.macroCategory,
      image: item.image, imageUrl: item.imageUrl, name: item.name }));
    try {
      const confirmedItems = await saveCloudOutfit({
        id: newOutfitId,
        items: snapshotItems,compositionProtocol:outfit.compositionProtocol,
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
      const mappedItems = stylistCalendarItems(outfit.items,outfit.compositionProtocol);
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
      try { ok = requireFeature('aiOutfits'); } catch { ok = false; }
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
    // Rotation history for the catch-block closest-match fallback.
    let fallbackHistory: Awaited<ReturnType<typeof getRotationHistory>> | undefined;
    let displayedScope: string | undefined;
    let displayedRequestId: string | undefined;
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
    let selectedBudget: ReturnType<typeof parseBudgetControl>;
    try { selectedBudget = parseBudgetControl(budgetAmount, budgetCurrency); }
    catch {
      setError(t('outfitMaker.invalidBudget', {defaultValue:'Enter a valid budget amount.'}));
      setLoading(false);
      return;
    }
    const textBudget = commentIntent.budget || parseBudgetFromText(promptText);
    const rawShopBudget = selectedBudget || textBudget;
    // Preserve the cap in its original currency. A request-scoped reference
    // snapshot is loaded below only when monetary comparisons need conversion.
    const edgePrompt = stripBudgetFromText(effectivePrompt);
    let shopBudget: ReturnType<typeof parseBudgetFromText> = rawShopBudget;
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
    setLastEngineVersion(null);
    setLastEngineDetail(null);
    setLastFailureCode(null);

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
    let payloadItems: any[] = [];
    // Per-tap normalize pool: memoized first page + random slice, collapsed.
    // normalizeTo4Slots MUST draw from this (not the frozen memo), otherwise
    // the random slice only reaches the edge while finals keep recycling the
    // same 100 pool items. Wardrobe taps keep the memoized pool (purity).
    let genPool: any[] = candidatePool;
    let alternativeItems: any[] = [];

    if (effectiveSource === 'shop') {
      // Shop mode has one authority: the authenticated Edge catalog search.
      // Sending a local first page creates stale inventory, duplicate work and
      // a large request without giving the server additional products.
      payloadItems = [];
      genPool = [];
      alternativeItems = [];
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

    if (effectiveSource === 'wardrobe') alternativeItems = payloadItems;

    try {
      const budgetCandidates = budgetCandidatesFromText(promptText);
      const mixedLimits = new Set([...budgetCandidates.map(b => b.currency), ...(selectedBudget ? [selectedBudget.currency] : [])]).size > 1;
      // The Edge may discover a different currency beyond the client's first
      // catalog page. Pin one FX snapshot for every budgeted shop request so
      // the local safety check can validate the actual returned products.
      // The local Stylist 6 ranker enforces a same-currency budget itself
      // (pool + admit filter), so a single-currency S6 budget needs no
      // conversion and must not depend on the deployed rates function. The
      // cloud S6 engine enforces the identical same-currency filter
      // server-side. Multi-currency limits and the hosted edge keep the
      // fail-closed fetch.
      const s6SameCurrencyBudget = STYLIST6_DEV_ENABLED
        && (devEngine === 'stylist6' || devEngine === 'stylist6cloud') && !mixedLimits;
      if ((mixedLimits || (shopBudget && effectiveSource === 'shop')) && !s6SameCurrencyBudget) {
        let fx;
        try { fx = budgetFxContext(await getBudgetRates()); }
        catch { throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable'); }
        if (!isCurrentGeneration()) return;
        shopBudget = (() => {
          const parsed = parseBudgetFromText(promptText, fx);
          const selected = strictestOutfitBudget(selectedBudget, parsed, (budget, currency) =>
            budgetInCurrency({...budget, ...(fx ? {fx} : {})}, currency));
          return selected ? {...selected, ...(fx ? {fx} : {})} : null;
        })();
      } else {
        const selected = strictestOutfitBudget(selectedBudget, textBudget, (budget, currency) =>
          budgetInCurrency(budget, currency));
        shopBudget = selected;
      }
      try {
        const { data: sess } = await supabase.auth.getSession();
        if (!isCurrentGeneration()) return;
        if (sess.session?.user.id !== generationOwner || !sess.session?.access_token) throw new Error('generation_session_unavailable');
        generationAccessToken = sess.session.access_token;
        recencyScope = generationOwner;
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

      const scope = rotationScope(recencyScope, effectiveSource, styleToUse, occasionToUse || 'Any');
      const requestId = generationRequestId();
      displayedScope = scope;
      displayedRequestId = requestId;
      // Hoisted for the catch-block fallback below: the closest-match
      // alternative must also prefer looks unlike recently shown ones.
      fallbackHistory = await getRotationHistory(scope);
      const history = fallbackHistory;
      const rotation = displayedRotation(history, baseItemId);
      if (__DEV__) void 0;
      const excludeProductKeys = [...new Set([...history.olderRecentProductKeys,...history.previousDeckProductKeys])];
      // Older hosted functions also understand raw IDs. Never send an empty
      // list while the persisted displayed-product window is populated.
      const excludeItemIds = excludeProductKeys.filter(key => key.startsWith('id:')).map(key => key.slice(3)).filter(id=>id.length<=200).slice(-150);
      if (!isCurrentGeneration() || !generationAccessToken) return;
      const EDGE_TIMEOUT_MS = 120000;
      const requestBody = {
          compositionProtocol: 'optional-layer-v1',
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
          allowPartialResults: true,
          prompt: edgePrompt,
          // A locked owned piece remains the anchor; the other slots use the catalog.
          anchorItemId: baseItemId,
          // Catalog mode excludes the user's wardrobe; the server fetches
          // real shop items directly from its current catalog.
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
          rotationScope: scope,
          generationRequestId: requestId,
          ...history,
        };
      try { validateOutfitGenerationRequest(requestBody); }
      catch (error) {
        if (error instanceof OutfitRequestError) {
          readinessLogger.info('stylist_admission', {field:error.field,reason:error.reason});
          throw new OutfitGenerationFailure('outfitMaker.invalidRequest','invalid_generation_request','admission');
        }
        throw error;
      }
      // Development Stylist 6 path: the real local ranker over live catalog
      // rows via the loopback bridge. Ranking failures surface below and
      // are never replaced with a Stylist 5 deck.
      const s6Active = STYLIST6_DEV_ENABLED && devEngine === 'stylist6';
      const s6CloudActive = STYLIST6_DEV_ENABLED && devEngine === 'stylist6cloud';
      let data: any;
      let invokeError: any;
      if (s6CloudActive) {
        // Isolated Stylist 6 cloud staging: authenticated proxy -> signed
        // compute -> server-side engine. No client inventory is sent; the
        // server fetches shop and wardrobe rows itself. Failures surface
        // intact and are never replaced with a Stylist 5 deck.
        const cloudBody = {
          request: {
            style: styleToUse,
            occasion: occasionToUse || 'Any',
            brief: edgePrompt,
            weather: resolvedWeather?.temp != null
              ? {temp: resolvedWeather.temp, condition: resolvedWeather.condition || 'clear'}
              : undefined,
            budget: shopBudget ? {amount: shopBudget.amount, currency: shopBudget.currency} : undefined,
            preferences: useStylePreferenceStore.getState().preferences,
            recentSignatures: history.recentLookSignatures,
            excludeItemIds,
            requiredIds: baseItemId ? [baseItemId] : [],
            // baseItemId is always an owned piece (locked from the closet).
            // A locked owned anchor with catalog clothing mirrors the S5
            // semantic "the locked piece stays; the other slots use the
            // catalog" — that is exactly the server's mixed pool. Without an
            // anchor, shop mode stays pure catalog and wardrobe mode stays
            // wardrobe-only.
            sourceMode: (currentSource === 'wardrobe'
              ? 'wardrobe'
              : baseItemId ? 'mixed' : 'catalog') as 'wardrobe' | 'catalog' | 'mixed',
            enableVisual: nvidiaVisual === true,
            generationRequestId: requestId,
          },
        };
        try {
          data = await withRequestDeadline((signal) => invokeStylist6Cloud(cloudBody, {signal}), EDGE_TIMEOUT_MS, cancelSignal);
        } catch (error) { invokeError = error; }
      } else if (s6Active) {
        const catalogRows = await loadStylistCatalogInventory(cancelSignal);
        if (!isCurrentGeneration()) return;
        const bridgeBody = buildStylist6BridgeBody({
          request: {
            style: styleToUse,
            occasion: occasionToUse || 'Any',
            brief: edgePrompt,
            weather: resolvedWeather?.temp != null
              ? {temp: resolvedWeather.temp, condition: resolvedWeather.condition || 'clear'}
              : undefined,
            budget: shopBudget ? {amount: shopBudget.amount, currency: shopBudget.currency} : undefined,
            preferences: useStylePreferenceStore.getState().preferences,
            recentSignatures: history.recentLookSignatures,
            excludeItemIds,
            enableVisual: nvidiaVisual === true,
          },
          catalogRows: catalogRows as unknown as Record<string, unknown>[],
        });
        try {
          data = await withRequestDeadline((signal) => invokeStylist6Bridge(bridgeBody, {signal}), EDGE_TIMEOUT_MS, cancelSignal);
        } catch (error) { invokeError = error; }
      } else {
        ({ data, error: invokeError } = await withRequestDeadline((signal) => supabase.functions.invoke('generate-outfits', {
          signal,
          headers: {Authorization: `Bearer ${generationAccessToken}`},
          body: requestBody,
        }), EDGE_TIMEOUT_MS, cancelSignal));
      }

      if (!isCurrentGeneration()) return;
      if (__DEV__) void 0;
      // The dev bridge throws classified Stylist 6 failures; rethrow them
      // intact so their real code reaches the screen. Never translate a
      // Stylist 6 failure through the Stylist 5 code map.
      if (invokeError) {
        if (invokeError instanceof OutfitGenerationFailure) throw invokeError;
        throw await outfitGenerationFailure(data, invokeError);
      }
      // The server may return HTTP 200 with a partial deck. That is NOT a
      // successful three-look acceptance: preserve the real engine identity
      // and surface an explicit client-side completeness failure.
      if (s6CloudActive && data?.success === true &&
          (data.status !== 'complete' || !Array.isArray(data.outfits) || data.outfits.length !== 3)) {
        throw new OutfitGenerationFailure('outfitMaker.noMatchingClothing', 'stylist6_incomplete_cloud_deck', 'stylist6_render');
      }

      // The FX-snapshot equality guarantee only exists for engines that
      // actually consume a rates snapshot (the hosted S5 edge). The S6 cloud
      // engine enforces same-currency budgets server-side and never consumes
      // FX, so there is no snapshot to compare and the check must not fire.
      if (shopBudget?.fx && !s6CloudActive && data?.success && data?.budgetReference?.id !== shopBudget.fx.snapshot.id) throw new OutfitGenerationFailure('outfitMaker.budgetRatesUnavailable');
      if (data && data.success && data.outfits && data.outfits.length > 0) {
        // Feed the rotation window BEFORE normalizing — the next tap must
        // exclude these exact ids even if the deck is edited downstream.

        const slotDeckUsed = new Set<string>();
        const wrapLook = (outfit: any, items: any[], idx: number): GeneratedOutfit => {
          const copy = stylingCopyParts(items, stylingContext({style:styleToUse,occasion:occasionToUse,request:effectivePrompt||promptText,weather:resolvedWeather}));
          const description = outfitDescription(outfit, items, stylingContext({
            style: styleToUse, occasion: occasionToUse,
            request: effectivePrompt || promptText, weather: resolvedWeather,
          }));
          const judgeOk = /^stylist-5-/.test(String(data.generationVersion || '')) || passesStyleJudge(items as any[], styleToUse, `${occasionToUse} ${effectivePrompt || promptText || ''}`, resolvedWeather, baseItemId,useStylePreferenceStore.getState().preferences);
          // Hosted confidence is not a score from this client's occasion judge.
          const baseScore = outfitCoherence(items, stylingContext({style:styleToUse, occasion:occasionToUse, request:effectivePrompt||promptText, weather:resolvedWeather})).score;
          return {
            ...outfit,
            // Truthful provenance: only genuine provider acceptance is 'ai';
            // a repaired ordering is 'hybrid'; everything else is fallback.
            // The card pill shows the AI badge for 'ai' only.
            selectionSource:resolveSelectionSource(data.source),
            generationRequestId:data.generationRequestId,engineVersion:data.generationVersion,
            id: outfit.id || `look_${tapNonce}_${idx}`,
            description,
            stylingStyle: styleToUse,
            stylingOccasion: `${occasionToUse} ${effectivePrompt || promptText || ''}`,
            stylingRequest: effectivePrompt || promptText || '',
            serverTips: Array.isArray(outfit.stylingTips) && outfit.stylingTips.length > 0,
            items,
            mainImage: items[0]?.image || items[0]?.imageUrl || outfit.mainImage || '',
            matchScore: judgeOk ? baseScore : Math.min(baseScore, 0.69),
            stylingTips: (() => {
              const serverTips = Array.isArray(outfit.stylingTips) ? outfit.stylingTips : [];
              // Server v205 sends no localized tips; fall back to the
              // localized, evidence-grounded tips. Restated advice is
              // suppressed below: never render the description as its own tip.
              const tips = serverTips.length ? serverTips : copy.tips;
              // The screen bypasses stylingCopyParts, so restated advice is
              // suppressed here: never render the description as its own tip.
              return distinctStylingTips(description, tips).join(' ');
            })(),
            weather: resolvedWeather,
          };
        };
        // The Stylist 6 dev deck is ranked by the bridge over catalog rows, so
        // it flows through the shop pipeline with its exact plan intact: the
        // shared Stylist 5 taste gates (base-shirt-only tops, style-judge
        // repair, closest-match recovery) must not rewrite it. Only the
        // rendering contract is enforced; budgets still apply downstream.
        const pipelineSource = s6Active ? 'shop' : effectiveSource;
        const s6Acceptance = s6Active ? acceptStylist6DevDeck(data.outfits) : null;
        // The cloud deck is ranked server-side by the Stylist 6 engine over
        // server inventory. Like the local bridge deck, it flows through the
        // pipeline with its exact plan intact: the shared Stylist 5 taste
        // gates (base-shirt-only repair, style-judge rewrite, closest-match
        // recovery) must not rewrite or replace it. Only the rendering
        // contract is enforced — source-aware, because wardrobe and mixed
        // decks legitimately carry owned pieces — and budgets still apply
        // downstream.
        const s6CloudAcceptance = s6CloudActive
          ? acceptStylist6CloudDeck(data.outfits, (data as any)?.sourceMode)
          : null;
        const prepared = (s6Acceptance || s6CloudAcceptance)
          ? {
            cleaned: (s6Acceptance ?? s6CloudAcceptance)!.accepted,
            accepted: (s6Acceptance ?? s6CloudAcceptance)!.accepted,
            recovered: [] as any[],
            rejections: (s6Acceptance ?? s6CloudAcceptance)!.rejections,
          }
          : prepareStylistResponse(data.outfits, alternativeItems, {
          anchorId: baseItemId, style: styleToUse, occasion: `${occasionToUse} ${effectivePrompt || promptText || ''}`,
          weather: resolvedWeather, source: pipelineSource, accepts: fitsCap,
          preferences: useStylePreferenceStore.getState().preferences,
          recentLookSignatures: history.recentLookSignatures,
          history,
          includeReserve: !data.generationVersion,
          optionalLayer:data.compositionProtocol==='optional-layer-v1',
          immutablePlan: /^stylist-[345]-/.test(String(data.generationVersion || '')),
        });
        const cleanedOutfits = prepared.cleaned;
        const recovered = prepared.recovered.map((look, index) =>
          wrapLook({id: `recovered_${tapNonce}_${index}`, matchScore: look.quality}, look.items, index));
        const judgedOutfits = [...prepared.accepted.map((look, index) => wrapLook(look, look.items, index)), ...recovered];
        if (__DEV__ && Object.keys(prepared.rejections).length) readinessLogger.debug('[AIOutfitmaker] response validation', [prepared.rejections]);

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
          deck: assembledOutfits,
          relaxed: deckRelaxed,
          relaxReason: deckRelaxReason,
          archetypes: deckArchetypes,
          similarities: deckSimilarities,
          inventory: deckInventory,
        } = s6Active || s6CloudActive || /^stylist-5-/.test(String(data.generationVersion || ''))
          ? {deck:budgetedOutfits,relaxed:false,relaxReason:undefined,
             archetypes:[] as string[],similarities:[] as {a:number;b:number;score:number}[],
             inventory:undefined as ReturnType<typeof assembleDiverseDeck<GeneratedOutfit>>['inventory'] | undefined}
          : assembleDiverseDeck<GeneratedOutfit>({
          candidates: budgetedOutfits,
          itemsOf: (o) => o.items as any[],
          qualityOf: (o) => Number(o.matchScore ?? 0.9),
          recencyOf: o => recentLookSimilarity(o.items, history.recentLookSignatures),
          priorityOfDeck: deck => rotation.priority(deck.flatMap(look => look.items)),
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
          capacity: recovered.length ? undefined : edgeDiversityCapacity,
          usedKeys: slotDeckUsed,
          // The VALID pool (style / occasion / climate filtered) decides
          // whether the inventory is genuinely thin. A catalogue full of
          // alternatives must never fall back to three colourways of one
          // formula, so relaxation is gated on this assessment.
          poolItems: [...budgetedOutfits.flatMap(look => look.items)],
          debug: __DEV__
            ? (message: string, data?: unknown) => readinessLogger.debug(String(`[stylist-diversity] ${message}`), data ? [data] : [])
            : undefined,
        });
        const finalOutfits = assembledOutfits;
        const deckProducts = analyzeDeckProductDiversity(
          finalOutfits.map((outfit) => outfit.items as any[]),
          { anchorKeys, forcedSlots: edgeDiversityCapacity?.forcedRepeatSlots || deckInventory?.forcedRepeatSlots },
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

        if (finalOutfits.length === 0) {
          if (s6Active || s6CloudActive) {
            throw new OutfitGenerationFailure(
              'outfitMaker.noMatchingClothing',
              s6CloudActive ? 'stylist6_client_deck_rejected' : 'stylist6_deck_empty',
              s6CloudActive ? 'stylist6_render' : 'stylist6_dev');
          }
          throw new OutfitGenerationFailure('outfitMaker.noMatchingClothing');
        }

        if (!isCurrentGeneration()) return;
        if (isMounted.current) {
          // Only validated, in-cap looks may become visible.
          const inCap = finalOutfits.filter((o) => fitsCap(o.items as any[]));
          const overCap = finalOutfits.filter((o) => !fitsCap(o.items as any[]));
          const shownOutfits = inCap.slice(0, 3);
          if (shownOutfits.length !== 3 && !(/^stylist-5-/.test(String(data.generationVersion||'')) && data.inventoryReason==='distinct_editorial_inventory_limited' && shownOutfits.length>0)) {
            if (s6CloudActive) throw new OutfitGenerationFailure('outfitMaker.noMatchingClothing', 'stylist6_client_deck_rejected', 'stylist6_render');
            throw new Error('incomplete_outfit_deck');
          }
          const previousKeys = new Set(history.previousDeckProductKeys);
          const previousShoeKeys = new Set(history.previousShoeProductKeys ?? history.previousDeckProductKeys);
          const previousDeckOverlapBySlot = Object.fromEntries(['outerwear','top','bottom','shoes'].map(slot=>[slot,
            shownOutfits.flatMap(o=>o.items as any[]).filter(item=>item.macroCategory===slot && String(item.id)!==String(baseItemId||'')
              && productIdentityKeys(item).some(key=>(slot==='shoes'?previousShoeKeys:previousKeys).has(key))).length]));
          // Edge overlap describes its original looks. A locally recomposed
          // quartet has different products and is outside that audit.
          // Repair may adjust labels, but must never reintroduce a product
          // that the authoritative backend deck successfully avoided.
          for (const [slot,count] of Object.entries(previousDeckOverlapBySlot)) {
            // The dev bridge ships its exact deck; rotation-overlap audit
            // applies to recomposed edge decks only.
            if (!s6Active && !s6CloudActive && data.generationVersion && !/^stylist-5-/.test(String(data.generationVersion)) && count > Number(data.previousDeckOverlapBySlot?.[slot] ?? 0)) throw new Error('client_rotation_overlap');
          }

          // Stylist 6 Cloud has already performed the authoritative entitlement
          // check and debited edge:aiOutfits in the server's single admission.
          // Do not run a second client quota-consumption RPC after a paid
          // success: it can reject a valid deck and increments a separate
          // aiOutfits counter. Keep the session/nonce guard for every engine.
          if ((!s6CloudActive && !(await consumeOrPaywall('aiOutfits', {cancelSignal}))) || !isCurrentGeneration()) return;
          setResultBudget(shopBudget);
          setInventoryNote(data.inventoryReason==='distinct_editorial_inventory_limited' ? t('outfitMaker.distinctInventoryLimited') : '');
          // Record the ACTUAL engine that produced this deck for dev display.
          setLastEngineVersion(String(data.generationVersion || 'unknown'));
          setLastEngineDetail(typeof data?.visualStatus === 'string' ? String(data.visualStatus) : (data?.diagnostics?.visualVerdict ? `visual ${String(data.diagnostics.visualVerdict)}` : (s6Active ? 'local bridge' : null)));
          setOutfits(shownOutfits);
          // History persistence must not turn already rendered success into
          // an error (or charge a use for a screen cleared by a cache failure).
          try { await recordDisplayedDeck(shownOutfits, scope, requestId); }
          catch { console.warn('[AIOutfitmaker] Could not persist displayed outfit history'); }
          if (!isCurrentGeneration()) return;
          recordShownOutfits(shownOutfits, recencyScope);
          if (__DEV__) void 0;
          setLoading(false);
        }
        return;
      }

      if (__DEV__ && data?.debug) readinessLogger.debug(String('[AIOutfitmaker] edge debug:'), [data.debug]);
      const edgeErr: any = await outfitGenerationFailure(data);
      if (data?.debug) edgeErr.debug = data.debug;
      throw edgeErr;
    } catch (err: any) {
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
      if (!isCurrentGeneration()) return;
      if (err instanceof OutfitGenerationFailure && err.requiresSubscription) {
        setOutfits([]);
        setLoading(false);
        setError(t(err.translationKey));
        navigation.navigate('Paywall');
        return;
      }
      const noMatch = err instanceof OutfitGenerationFailure
        ? ['outfitMaker.noMatchingClothing', 'outfitMaker.aiUnavailable'].includes(err.translationKey)
        : ['No complete outfits returned from AI', 'incomplete_outfit_deck', 'client_rotation_overlap'].includes(err?.message);
      // Server failure stays visible. Do not silently replace a failed new
      // generation with local/previous suggestions labeled as generated.
      const failure = err instanceof RequestTimeoutError
        ? new OutfitGenerationFailure('outfitMaker.requestTimeout', 'request_timeout', 'transport')
        : noMatch && !(err instanceof OutfitGenerationFailure)
        ? new OutfitGenerationFailure('outfitMaker.noMatchingClothing') : err;
      const messageKey = noMatch && occasionToUse === 'formal' ? 'outfitMaker.formalInventoryShortfall'
        : failure instanceof OutfitGenerationFailure ? failure.translationKey : 'outfitMaker.generateFailed';
      console.warn('[AIOutfitmaker] No outfit available', { source: effectiveSource, messageKey: failure instanceof OutfitGenerationFailure ? failure.translationKey : 'outfitMaker.generateFailed' });
      if (isMounted.current) {
        setOutfits([]);
        setLoading(false);
        setLastFailureCode(failure instanceof OutfitGenerationFailure
          ? String(failure.code || failure.translationKey || failure.message || 'unknown')
          : String((failure as Error)?.message || 'unknown'));
        setError(t(messageKey, {
          defaultValue: 'Could not create an outfit right now. Please try again.',
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

          {__DEV__ && process.env.EXPO_PUBLIC_STYLIST_DIAGNOSTICS === 'true' && <StylistRuntimeDiagnostic engine={lastEngineVersion || undefined} detail={lastEngineDetail || undefined} />}
          {STYLIST6_DEV_ENABLED && lastEngineVersion ? (
            <View style={{ paddingHorizontal: 20, paddingBottom: 4 }}>
              <ScaledText testID="stylistEngineBanner" style={{fontSize: 11, textAlign: 'center', color: '#173A65', fontWeight: '700'}}>
                {`Engine ${lastEngineVersion}${lastEngineDetail ? ` · ${lastEngineDetail}` : ''}`}
              </ScaledText>
            </View>
          ) : null}
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
          {resultBudget && (
            <View testID="activeBudgetStatus" style={{ paddingHorizontal: 20, paddingBottom: 4 }}>
              <ScaledText style={styles.outfitDesc} minScale={0.65}>
                {t('outfitMaker.activeBudget', {
                  defaultValue: 'Active budget: {{amount}}. Every shown look is checked against this cap.',
                  amount: formatMoney(resultBudget.amount, resultBudget.currency),
                })}
              </ScaledText>
            </View>
          )}

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 32 }}>
            {!!inventoryNote && <ScaledText style={styles.outfitDesc}>{inventoryNote}</ScaledText>}
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
                    <ScaledText style={styles.matchPillText}>{t(selectionBadgeKey(outfit.selectionSource))}</ScaledText>
                  </LinearGradient>
                </View>
                {STYLIST6_DEV_ENABLED && outfit.engineVersion ? (
                  <ScaledText testID={`stylistEngine-${index}`} style={{fontSize: 10, color: '#526174', marginTop: 6}}>
                    {outfit.engineVersion}
                  </ScaledText>
                ) : null}

                {outfit.direction && ['signature','refresh','discovery','alternative'].includes(outfit.direction.id) && (
                  <View testID={`stylistDirection-${index}`} style={{marginBottom:9}}>
                    <View style={{flexDirection:'row',alignItems:'center',gap:6,marginBottom:3}}>
                      <Ionicons name={outfit.direction.id==='discovery'?'compass-outline':'sparkles-outline'} size={15} color={D.accent}/>
                      <ScaledText style={{color:D.accent,fontSize:12,fontWeight:'700'}}>
                        {t(`outfitMaker.direction_${outfit.direction.id}`)}
                      </ScaledText>
                    </View>
                    <ScaledText style={[styles.outfitDesc,{fontSize:12,marginBottom:0}]} minScale={0.75}>
                      {t(`outfitMaker.directionExplain_${outfit.direction.id}`)}
                    </ScaledText>
                  </View>
                )}
                {outfit.concept && <ScaledText style={[styles.outfitDesc,{fontWeight:'700',fontSize:20,marginBottom:12}]}>
                  {outfitConceptTitle(outfit.concept.id, outfit.concept.title)}
                </ScaledText>}
                {(() => {
                  const evidence = outfit.fashionDiagnostics?.personalization?.matches;
                  if (!Array.isArray(evidence) || !evidence.length) return null;
                  const selectedIds = new Set(outfit.items.map((item) => String(item.id)));
                  const acceptedTypes = new Set(['favorite_color','favorite_brand','liked_pattern','preferred_fit']);
                  const details = [...new Set(evidence
                    .filter((entry) => entry && acceptedTypes.has(entry.type) &&
                      selectedIds.has(String(entry.productId)) &&
                      typeof entry.value === 'string' && entry.value.trim().length > 0)
                    .map((entry) => entry.type === 'preferred_fit'
                      ? t(entry.value === 'wide' ? 'outfitMaker.personalizedFitWide' : 'outfitMaker.personalizedFitSlim')
                      : entry.value.trim()))].slice(0, 3);
                  if (!details.length) return null;
                  return (
                    <View testID={`personalStylistReason-${index}`} style={{flexDirection:'row',alignItems:'center',gap:8,marginBottom:12}}>
                      <Ionicons name="sparkles-outline" size={15} color={D.accent} />
                      <ScaledText style={[styles.outfitDesc,{fontSize:13,flex:1,marginBottom:0}]}>
                        {t('outfitMaker.personalizedForYou')}: {details.join(' · ')}
                      </ScaledText>
                    </View>
                  );
                })()}
                <OutfitSlotGrid
                  items={outfit.items}
                  weather={outfit.weather}
                  priceCatalog={liveShopMapped}
                  showBuyButtons={currentSource === 'shop'}
                />

                {(() => {
                  const total = outfitTotalDisplay(outfit.items as any[], liveShopMapped, displayRates, resultBudget);
                  if (total) {
                    return (
                    <View style={styles.priceRow}>
                      <ScaledText style={styles.priceRowLabel}>{t('outfitMaker.outfitTotal')}</ScaledText>
                      <View style={{alignItems:'flex-end', flex:1}}>
                        <ScaledText style={styles.priceRowValue} minScale={0.7}>{total.label}</ScaledText>
                        {total.converted && resultBudget?.fx && <ScaledText style={styles.priceRowLabel}>
                          {t('outfitMaker.referenceRate', {date: resultBudget.fx.snapshot.asOf})}
                        </ScaledText>}
                      </View>
                    </View>
                    );
                  }
                  // A combined total across currencies would be invented
                  // arithmetic: say so explicitly instead of hiding it.
                  return mixedCurrencyPriced(outfit.items as any[], liveShopMapped) ? (
                    <View style={styles.priceRow}>
                      <ScaledText style={styles.priceRowLabel}>{t('outfitMaker.outfitTotalUnavailable', {defaultValue: 'Total unavailable — items are listed in different currencies'})}</ScaledText>
                    </View>
                  ) : null;
                })()}

                <ScaledText style={styles.outfitDesc} minScale={0.65}>{outfit.description}</ScaledText>

                {!!outfit.stylingTips && <View style={styles.tipCard}>
                  <LinearGradient colors={D.panelHighlight} style={StyleSheet.absoluteFill} />
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <Ionicons name="sparkles-outline" size={16} color={D.accent} />
                    <ScaledText style={styles.tipCardText}>{outfit.stylingTips}</ScaledText>
                  </View>
                </View>}

                {canRecordLookFeedback(outfit) && (
                  <View style={{flexDirection:'row',alignItems:'center',gap:16,marginTop:12}}>
                    <ScaledText>{lookRatings[`${outfit.generationRequestId}:${outfit.id}`] ? t('outfitMaker.feedbackSaved') : t('outfitMaker.feedbackPrompt')}</ScaledText>
                    {([1,-1] as const).map(rating=>(
                      <TouchableOpacity key={rating} accessibilityRole="button" accessibilityLabel={t(rating===1?'outfitMaker.feedbackLike':'outfitMaker.feedbackDislike')}
                        disabled={feedbackPending!==null} onPress={()=>void submitLookFeedback(outfit,rating)}
                        style={{padding:10,opacity:feedbackPending?0.5:1}}>
                        <Ionicons name={rating===1?'thumbs-up-outline':'thumbs-down-outline'} size={22}
                          color={lookRatings[`${outfit.generationRequestId}:${outfit.id}`]===rating?D.accent:D.textSecondary}/>
                      </TouchableOpacity>
                    ))}
                  </View>
                )}
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
                <BetaFeedbackButton feature="outfit" />
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
    const selectedStyleLabel = getAiStyleLabel(selectedStyle, t);
    const occasionLabel = selectedOccasion ? getOccasionLabel(selectedOccasion, t) : undefined;
    const sourceLabel = currentSource === 'shop'
      ? t('outfitMaker.newClothing', { defaultValue: 'New clothing' })
      : t('outfitMaker.fromWardrobe', { defaultValue: 'From wardrobe' });
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

        {__DEV__ && process.env.EXPO_PUBLIC_STYLIST_DIAGNOSTICS === 'true' && <StylistRuntimeDiagnostic />}
        {/* Hero — live summary of the look being built */}
        <View style={styles.heroBlock}>
          <ScaledText style={styles.heroTitle}>
            {t('outfitMaker.newOutfit', { defaultValue: 'New outfit' })}
          </ScaledText>
          <ScaledText style={styles.heroSub} numberOfLines={1}>
            {getAiStyleLabel(selectedStyle, t)}
            {' · '}
            {currentSource === 'shop'
              ? t('outfitMaker.newShort', { defaultValue: 'New' })
              : t('outfitMaker.wardrobeShort', { defaultValue: 'Wardrobe' })}
            {' · '}
            {selectedOccasion
              ? getOccasionLabel(selectedOccasion, t)
              : t('outfitMaker.pickOccasion', { defaultValue: 'Pick an occasion' })}
            {' · '}
            {weather ? `${weather.temp}°C` : t('outfitMaker.weatherUnavailable', { defaultValue: 'Weather unavailable' })}
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
                    testID={`stylistSource-${opt.id}`}
                    accessibilityRole="button"
                    accessibilityLabel={opt.label}
                    accessibilityState={{selected:on}}
                    activeOpacity={0.85}
                    onPress={() => {
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                      setError('');
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
                  availableOccasions.some((o) => o.id === id),
                );
                const fillPool: string[] = [...quizIds];
                for (const o of availableOccasions) {
                  if (!fillPool.includes(o.id as string)) fillPool.push(o.id as string);
                }
                // Selected occasion always stays visible.
                let visibleIds = fillPool.slice(0, 2);
                if (selectedOccasion && !visibleIds.includes(selectedOccasion)) {
                  visibleIds = [selectedOccasion, ...fillPool.filter((id) => id !== selectedOccasion)].slice(0, 2);
                }
                const hiddenOnes = availableOccasions.filter((o) => !visibleIds.includes(o.id as string));
                const visibleList = availableOccasions.filter((o) => visibleIds.includes(o.id as string));
                const renderOccTile = (occ: (typeof OCCASION_PRESETS)[number]) => {
                  const on = selectedOccasion === occ.id;
                  return (
                    <TouchableOpacity
                      key={occ.id}
                      testID={`stylistOccasion-${occ.id}`}
                      activeOpacity={0.85}
                      onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setError('');
                        setSelectedOccasion(on ? undefined : (occ.id as string));
                        commentEditedRef.current = false;
                      }}
                      style={[styles.bigTile, on && styles.bigTileSelected]}
                    >
                      <View style={[styles.bigTileIcon, on && styles.bigTileIconActive]}>
                        <Ionicons name={occ.icon as any} size={20} color={on ? '#FFFFFF' : '#173A65'} />
                      </View>
                      <ScaledText style={[styles.bigTileLabel, on && styles.bigTileLabelActive]}>
                        {getOccasionLabel(occ.id, t)}
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
                    {hiddenOnes.length > 0 && <TouchableOpacity
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
                    </TouchableOpacity>}
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
                const allStyles = availableStyles;
                if (!allStyles.length) return null;
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
                        setError('');
                        setSelectedStyle(activeStyle.id);
                        commentEditedRef.current = false;
                      }}
                      style={styles.featureCard}
                    >
                      <View style={styles.featureIcon}>
                        <Ionicons name={activeStyle.icon as any} size={26} color="#FFFFFF" />
                      </View>
                      <View style={{ flex: 1, marginLeft: 14 }}>
                        <ScaledText style={styles.featureName}>{getAiStyleLabel(activeStyle.id, t)}</ScaledText>
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
                    {hiddenStyles.length > 0 && <TouchableOpacity
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
                    </TouchableOpacity>}
                    {showAllStyles && (
                      <Reanimated.View entering={FadeInUp.duration(250)} exiting={FadeOut.duration(200)}>
                        <View style={styles.formGroup}>
                          {hiddenStyles.map((styleObj, i) => {
                            const isSelected = selectedStyle === styleObj.id;
                            return renderCheckRow(
                              styleObj.id,
                              styleObj.icon,
                              getAiStyleLabel(styleObj.id, t),
                              isSelected,
                              () => {
                                setError('');
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
                    defaultValue: "Where you're going and the style you want…",
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

              {currentSource === 'shop' && (
                <View style={[styles.formGroup, {marginTop: 12, padding: 14}]}>
                  <ScaledText style={{color: D.textPrimary, fontWeight: '700', marginBottom: 4}}>
                    {t('outfitMaker.optionalBudget', {defaultValue: 'Budget for one complete outfit (optional)'})}
                  </ScaledText>
                  <TextInput
                    testID="stylistBudgetAmount"
                    accessibilityLabel={t('outfitMaker.budgetAmount', {defaultValue: 'Maximum outfit spend'})}
                    style={[styles.detailsInput, {minHeight: 48, paddingVertical: 8}]}
                    placeholder={t('outfitMaker.noBudget', {defaultValue: 'No budget set'})}
                    placeholderTextColor={D.textSecondary}
                    keyboardType="decimal-pad"
                    value={budgetAmount}
                    onChangeText={setBudgetAmount}
                  />
                  <View style={{flexDirection:'row', flexWrap:'wrap', gap:6}}>
                    {(['UZS','USD','EUR','GBP'] as const).map(currency => (
                      <TouchableOpacity key={currency} testID={`stylistBudgetCurrency-${currency}`}
                        accessibilityRole="radio" accessibilityState={{checked:budgetCurrency===currency}}
                        accessibilityLabel={`${t('outfitMaker.budgetCurrency', {defaultValue:'Budget currency'})}: ${currency}`}
                        onPress={() => setBudgetCurrency(currency)}
                        style={{paddingHorizontal:12,paddingVertical:7,borderRadius:10,
                          backgroundColor:budgetCurrency===currency?D.accent:'rgba(23,58,101,0.08)'}}>
                        <ScaledText style={{color:budgetCurrency===currency?'#FFF':D.textPrimary,fontWeight:'700'}}>{currency}</ScaledText>
                      </TouchableOpacity>
                    ))}
                  </View>
                </View>
              )}


            {STYLIST6_DEV_ENABLED ? (
              <View style={[styles.formGroup, {marginTop: 12, padding: 14}]}>
                <ScaledText style={{color: D.textPrimary, fontWeight: '700', marginBottom: 4}}>
                  {'Engine (dev only)'}
                </ScaledText>
                <View style={{flexDirection:'row', gap:6}}>
                  {(['stylist5','stylist6','stylist6cloud'] as const).map(engine => (
                    <TouchableOpacity key={engine} testID={`stylistEnginePick-${engine}`}
                      accessibilityRole="radio" accessibilityState={{checked:devEngine===engine}}
                      accessibilityLabel={engine === 'stylist5' ? 'Stylist 5' : engine === 'stylist6' ? 'Stylist 6 local' : 'Stylist 6 cloud staging'}
                      onPress={() => setDevEngine(engine)}
                      style={{paddingHorizontal:12,paddingVertical:7,borderRadius:10,
                        backgroundColor:devEngine===engine?D.accent:'rgba(23,58,101,0.08)'}}>
                      <ScaledText style={{color:devEngine===engine?'#FFF':D.textPrimary,fontWeight:'700'}}>
                        {engine === 'stylist5' ? 'Stylist 5' : engine === 'stylist6' ? 'Stylist 6 (local)' : 'Stylist 6 (cloud)'}
                      </ScaledText>
                    </TouchableOpacity>
                  ))}
                </View>
                <ScaledText style={{color: D.textSecondary, fontSize: 12, marginTop: 6}}>
                  {'Stylist 6 runs the local ranker over the live catalog. Failures are shown, never replaced.'}
                </ScaledText>
                <TouchableOpacity testID="stylistNvidiaToggle"
                  accessibilityRole="checkbox" accessibilityState={{checked: nvidiaVisual, disabled: devEngine !== 'stylist6' && devEngine !== 'stylist6cloud'}}
                  accessibilityLabel="NVIDIA visual review (Stylist 6 dev only)"
                  onPress={() => setNvidiaVisual((v) => !v)}
                  style={{flexDirection:'row', alignItems:'center', gap:8, marginTop:10,
                    paddingHorizontal:12, paddingVertical:8, borderRadius:10,
                    backgroundColor: nvidiaVisual ? 'rgba(23,58,101,0.12)' : 'rgba(23,58,101,0.05)',
                    opacity: devEngine === 'stylist6' || devEngine === 'stylist6cloud' ? 1 : 0.45}}>
                  <View style={{width:18, height:18, borderRadius:5, borderWidth:1.5, borderColor: D.accent,
                    backgroundColor: nvidiaVisual ? D.accent : 'transparent', alignItems:'center', justifyContent:'center'}}>
                    {nvidiaVisual ? <Ionicons name="checkmark" size={13} color="#FFF" /> : null}
                  </View>
                  <View style={{flex:1}}>
                    <ScaledText style={{color: D.textPrimary, fontWeight:'700', fontSize:13}}>
                      {'NVIDIA visual review'}
                    </ScaledText>
                    <ScaledText style={{color: D.textSecondary, fontSize: 11}}>
                      {'Hosted Nemotron check on the dev bridge only. Stays off for Stylist 5.'}
                    </ScaledText>
                  </View>
                </TouchableOpacity>
              </View>
            ) : null}

            </ScrollView>
          </Reanimated.View>
          {/* Keep failures visible beside the retry action, outside scrolling choices. */}
          <View style={styles.stickyFooter}>
            {error ? <View accessibilityRole="alert" accessibilityLiveRegion="assertive" style={[styles.errorBox, { marginBottom: 12 }]}>
              <ScaledText style={styles.errorText}>{error}</ScaledText>
              {STYLIST6_DEV_ENABLED && lastFailureCode ? (
                <ScaledText testID="stylistFailureCode" style={[styles.errorText, { marginTop: 4, fontSize: 12 }]}>
                  {`code: ${lastFailureCode}`}
                </ScaledText>
              ) : null}
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
  priceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 10,
    marginBottom: 4,
  },
  priceRowLabel: {
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
  },
  priceRowValue: {
    ...LiquidGlass2026Theme.typography.scale.titleSmall,
    color: LiquidGlass2026Theme.colors.text.primary,
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
