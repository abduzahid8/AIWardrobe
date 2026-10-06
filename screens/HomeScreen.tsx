/**
 * HomeScreen - 2026 Redesign
 * Features: Bento Grid layout, Liquid Glass aesthetics, GenUI-ready components
 * Based on 2026 Digital Experience Report guidelines
 */

import React, { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { View, ScrollView, Dimensions, StyleSheet, ActivityIndicator, TouchableOpacity, Alert } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { useIsFocused, useFocusEffect } from "@react-navigation/native";
import { useAppNavigation } from '../hooks/useAppNavigation';
import AsyncStorage from "@react-native-async-storage/async-storage";
import useAuthStore from '../store/auth';
import { LinearGradient } from "expo-linear-gradient";
import { Image as ExpoImage } from 'expo-image';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as Location from 'expo-location';
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeInDown,
} from 'react-native-reanimated';

// 2026 Design System
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import {
  LiquidGlassCard,
  OutfitSwipeStack,
} from '../components/ui';
import type { OutfitSwipeStackHandle, SwipeStackCard } from '../components/ui';
import { useAccessibility } from '../hooks/useAccessibility';
import Config from '../src/config/env';

// Core loop components

import StreakBadge from '../components/StreakBadge';
import useWardrobeStore from '../store/wardrobeStore';
import { useShallow } from 'zustand/shallow';
import useSubscriptionStore, { FREE_HOME_SWIPES_PER_DAY } from '../store/subscriptionStore';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';
import { CachedImage } from '../components/ui/CachedImage';

import { supabase } from '../lib/supabase';
import { quickSuggest } from '../src/services/suggestionEngine';
import { useDailyAIOutfit } from '../hooks/useDailyAIOutfit';
import { useShopCatalog } from '../hooks/useShopCatalog';
import useShopCatalogStore from '../store/shopCatalogStore';
import TrialCountdownBanner from '../components/TrialCountdownBanner';
import { OutfitShareModal } from '../components/OutfitShareModal';
import { OutfitShopSheet } from '../components/OutfitShopSheet';
import type { ShopCatalogItem } from '../src/types/shop';
import type { ClothingCategory, Occasion } from '../src/types/domain';
import { createLogger } from '../src/utils/logger';
import { useTranslation } from 'react-i18next';
import { perfMark, perfMeasure, perfAction, perfScreenReady } from '../src/utils/perf';
import {
  buildCatalogSimilarityKey,
  normalizeCatalogImageKey,
  normalizeCatalogNameKey,
  getCatalogShoeFamily,
} from '../src/utils/shopCatalogOrder';

type EssentialSlot = 'outerwear' | 'shirts' | 'knitwear' | 'tees' | 'bottoms' | 'shoes';
type EssentialAddState = 'adding' | 'added';

const ESSENTIALS_REQUIREMENTS: Record<EssentialSlot, number> = {
  outerwear: 2,  // blazers, jackets, coats
  shirts: 2,     // dress shirts, casual shirts
  knitwear: 2,   // sweaters, cardigans
  tees: 2,       // t-shirts, polos
  bottoms: 3,    // pants, chinos, jeans, shorts
  shoes: 3,      // sneakers, loafers, boots
};

const ESSENTIALS_LIMIT = Object.values(ESSENTIALS_REQUIREMENTS).reduce(
  (sum, count) => sum + count,
  0
);

// Keywords for better classification
const OUTERWEAR_KEYWORDS = [
  'blazer', 'jacket', 'coat', 'overcoat', 'suit', 'trench', 'parka',
  'bomber', 'puffer', 'vest', 'waistcoat', 'cardigan', 'hoodie'
];

const SHIRT_KEYWORDS = [
  'shirt', 'oxford', 'dress shirt', 'broadcloth', 'poplin', 'linen shirt',
  'flannel', 'chambray', 'button-down', 'button down'
];

const KNITWEAR_KEYWORDS = [
  'sweater', 'knit', 'pullover', 'turtleneck', 'crewneck', 'v-neck',
  'cashmere', 'merino', 'wool', 'chunky', 'cable', 'cardigan'
];

const TEE_KEYWORDS = [
  't-shirt', 'tshirt', 'tee', 'polo', 'henley', 'tank', 'top'
];

const BOTTOM_KEYWORDS = [
  'pants', 'trousers', 'chinos', 'jeans', 'shorts', 'joggers',
  'slacks', 'cargos', 'corduroy'
];

const SHOE_KEYWORDS = [
  'shoes', 'sneakers', 'loafers', 'boots', 'oxfords', 'derby',
  'trainers', 'slip-on', 'mules'
];

const classifyUpperBodyItem = (item: ShopCatalogItem): 'outerwear' | 'shirts' | 'knitwear' | 'tees' | null => {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();

  // Check in priority order
  if (OUTERWEAR_KEYWORDS.some((k) => text.includes(k))) return 'outerwear';
  if (SHIRT_KEYWORDS.some((k) => text.includes(k))) return 'shirts';
  if (KNITWEAR_KEYWORDS.some((k) => text.includes(k))) return 'knitwear';
  if (TEE_KEYWORDS.some((k) => text.includes(k))) return 'tees';

  // Default classification based on common patterns
  if (text.includes('blazer') || text.includes('jacket') || text.includes('coat')) return 'outerwear';
  if (text.includes('shirt')) return 'shirts';
  if (text.includes('sweater') || text.includes('knit')) return 'knitwear';

  return 'tees'; // default
};

const classifyBottomItem = (item: ShopCatalogItem): 'bottoms' | null => {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  if (BOTTOM_KEYWORDS.some((k) => text.includes(k))) return 'bottoms';
  if (item.garmentType === 'lower_body') return 'bottoms';
  return null;
};

const classifyShoeItem = (item: ShopCatalogItem): 'shoes' | null => {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  if (SHOE_KEYWORDS.some((k) => text.includes(k))) return 'shoes';
  if (item.garmentType === 'shoes') return 'shoes';
  return null;
};

/**
 * Deterministic seeded PRNG (mulberry32).
 * Returns a pseudo-random number in [0, 1) for a given 32-bit seed integer.
 * Using a stable daily seed (new Date().toDateString()) ensures the shuffle
 * produces the same order for the entire day, keeping useMemo results stable
 * across renders within the same day (fixes Defect 1.1).
 */
const seededRandom = (seed: number): () => number => {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** Convert a string seed to a 32-bit integer via a simple hash. */
const hashStrToInt = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (Math.imul(31, hash) + str.charCodeAt(i)) | 0;
  }
  return hash;
};

export const selectEssentialShoppingMix = (items: ShopCatalogItem[], dailySeed: string): ShopCatalogItem[] => {
  const pickedCounts: Record<EssentialSlot, number> = {
    outerwear: 0,
    shirts: 0,
    knitwear: 0,
    tees: 0,
    bottoms: 0,
    shoes: 0,
  };

  const selected: ShopCatalogItem[] = [];
  const selectedIds = new Set<string>();
  // Cross-source / near-duplicate guards. Catalog `id` alone is not enough:
  // the same shoe from `manual` + `apify-massimodutti` + `apify-zara-men-uz`
  // (or two static variants like classic-m-shoe-16/17 "Leather loafers with
  // penny strap") has different ids but the same name/image. Without these
  // sets the `shoes: 3` quota fills with 3 visually identical loafers.
  const seenImages = new Set<string>();
  const seenNames = new Set<string>();
  const seenSimilarityKeys = new Set<string>();
  const seenShoeFamilies = new Set<string>();

  // Shuffle items for daily variety using a stable seed so the result is
  // identical across all renders within the same calendar day.
  // Fisher–Yates O(n) — the old `.sort(() => rand() - 0.5)` was O(n log n)
  // with a biased comparator, run on every catalog refresh.
  const rand = seededRandom(hashStrToInt(dailySeed));
  const shuffled = [...items];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const getSlot = (item: ShopCatalogItem): EssentialSlot | null => {
    if (item.garmentType === 'upper_body') {
      return classifyUpperBodyItem(item);
    } else if (item.garmentType === 'lower_body') {
      return classifyBottomItem(item);
    } else if (item.garmentType === 'shoes') {
      return classifyShoeItem(item);
    }
    return null;
  };

  const tryPick = (item: ShopCatalogItem, strictVariety: boolean): boolean => {
    if (!item || selectedIds.has(item.id)) return false;

    const slot = getSlot(item);
    if (!slot) return false;
    if (pickedCounts[slot] >= ESSENTIALS_REQUIREMENTS[slot]) return false;

    // Exact + near-duplicate guards (apply in both passes).
    const imageKey = normalizeCatalogImageKey((item as any).imageUrl);
    if (imageKey && seenImages.has(imageKey)) return false;
    const nameKey = normalizeCatalogNameKey((item as any).name);
    if (nameKey && seenNames.has(nameKey)) return false;
    const similarityKey = buildCatalogSimilarityKey({
      id: item.id,
      name: (item as any).name ?? '',
      garmentType: (item as any).garmentType,
    });
    if (seenSimilarityKeys.has(similarityKey)) return false;

    // Strict pass: at most one shoe family (loafer/sneaker/boot/…) so the
    // 3-shoe quota can't fill with 3 loafers that have slightly different
    // names ("oiled leather penny loafer" vs "leather loafer penny strap").
    if (strictVariety && slot === 'shoes') {
      const family = getCatalogShoeFamily((item as any).name);
      if (seenShoeFamilies.has(family)) return false;
    }

    selected.push(item);
    selectedIds.add(item.id);
    if (imageKey) seenImages.add(imageKey);
    if (nameKey) seenNames.add(nameKey);
    seenSimilarityKeys.add(similarityKey);
    if (slot === 'shoes') seenShoeFamilies.add(getCatalogShoeFamily((item as any).name));
    pickedCounts[slot] += 1;
    return true;
  };

  // Pass 1 (strict): enforce shoe-family variety.
  for (const item of shuffled) {
    if (selected.length >= ESSENTIALS_LIMIT) break;
    tryPick(item, true);
  }
  // Pass 2 (relaxed fallback): if the catalog has only loafers, still fill
  // the quota rather than showing an incomplete grid — but exact duplicates
  // (same image/name/similarity) stay blocked.
  if (selected.length < ESSENTIALS_LIMIT) {
    for (const item of shuffled) {
      if (selected.length >= ESSENTIALS_LIMIT) break;
      if (selectedIds.has(item.id)) continue;
      tryPick(item, false);
    }
  }

  return selected;
};

const garmentTypeToCategory = (
  garmentType: ShopCatalogItem['garmentType']
): ClothingCategory => {
  switch (garmentType) {
    case 'upper_body':
      return 'top';
    case 'lower_body':
      return 'bottom';
    case 'shoes':
      return 'shoes';
    case 'dresses':
      return 'dress';
    default:
      return 'other';
  }
};

const logger = createLogger('HomeScreen');

const { width: SCREEN_WIDTH } = Dimensions.get("window");

// ── Stable Zustand selectors ─────────────────────────────────────────────────
// Defined at module scope so the function reference never changes between
// renders. React 18's useSyncExternalStore warns "getSnapshot should be cached"
// when the selector is an inline arrow function that returns a new object on
// every call. Primitive-returning selectors are fine inline, but object-
// returning ones must use useShallow (Zustand v5) or be module-level.
const selectAddItem = (state: any) => state.addItem;
// Select only the username string (primitive) — selecting the full user object
// would return a new reference on every render, triggering the getSnapshot warning.
const selectUsername = (state: any) => state.user?.username as string | undefined;

// Pre-compiled RegExp constants for shop item name classification.
// Defined at module scope so they are compiled once, not on every render.
const BOTTOM_NAME_RE    = /\b(pants?|trousers?|jeans?|chinos?|shorts?|skirts?|slacks?|joggers?|sweatpants?|bermudas?|cargos?|leggings?)\b/i;
const SHORTS_NAME_RE    = /\b(shorts?|bermudas?|cargo\s*shorts?)\b/i;
const TOP_NAME_RE       = /\b(shirts?|tees?|t-shirts?|tshirts?|polos?|blouses?|tops?|tanks?|sleeveless)\b/i;
const OUTERWEAR_NAME_RE = /\b(jackets?|coats?|blazers?|cardigans?|sweaters?|hoodies?|puffers?|bombers?|vests?|outerwear|trench(?:es)?|peacoats?|suits?)\b/i;
const SHOE_NAME_RE      = /\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|mules?)\b/i;

const isBottomName    = (name: string) => typeof name === 'string' && BOTTOM_NAME_RE.test(name);
const isShortsName    = (name: string) => typeof name === 'string' && SHORTS_NAME_RE.test(name);
const isTopName       = (name: string) => typeof name === 'string' && TOP_NAME_RE.test(name);
const isOuterwearName = (name: string) => typeof name === 'string' && OUTERWEAR_NAME_RE.test(name);
const isShoeName      = (name: string) => {
  if (!name || typeof name !== 'string') return false;
  const lower = name.toLowerCase();
  let res = false;
  if (lower.includes('oxford')) {
    if (lower.includes('shirt')) res = false;
    else if (lower.includes('shoe') || lower.includes('flat') || lower.includes('lace') || lower.includes('brogue') || lower.includes('derby') || lower.includes('loafer')) res = true;
    else if (/\boxfords\b/i.test(lower)) res = true;
    else res = false;
  } else {
    res = SHOE_NAME_RE.test(name);
  }
  return res;
};

// Theme shortcuts
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

/**
 * Team Collaboration is always 100% shop — no wardrobe anchor.
 * Keep in sync with `isShopOnlyDeck` in hooks/useDailyAIOutfit.ts and
 * src/services/outfitGenerationService.ts.
 */
const isShopOnlyDeck = (style?: string | null, occasion?: string | null): boolean => {
  const s = (style || '').toLowerCase().replace(/[\s-]+/g, '_');
  if (s === 'business_casual') return true;
  const o = (occasion || '').toLowerCase();
  if (o.includes('team') && o.includes('collab')) return true;
  if (o.trim() === 'team collaboration') return true;
  return false;
};

interface WeatherData {
  temp: number;
  description: string;
  icon: string;
  city: string;
}

interface DailyOutfitSectionProps {
  occasion: string;
  style: string;
  weather: WeatherData | null;
  weatherForAI: { temp: number; condition: string } | null;
  shopTops: ShopCatalogItem[];
  shopBottoms: ShopCatalogItem[];
  shopShoes: ShopCatalogItem[];
  catalogEssentials: ShopCatalogItem[];
  needsOuterwear: boolean;
  isReducedMotionEnabled: boolean;
  navigation: any;
  t: any;
  userId?: string;
  /**
   * False when Home is not the focused tab. Pauses the outfit engine
   * (edge-function call + fallback queries + combinatorial rebuild) so it
   * never saturates the JS thread / network while the user is browsing
   * another tab — that collision was the freeze behind tab switches in the
   * first seconds after launch. Re-enabling serves the daily cache instantly.
   */
  enabled?: boolean;
}

// actionIconButton renders a 32x32 tap target, below the 44x44 minimum
// recommended by Apple's HIG — hitSlop pads the touchable area without
// changing the visible icon layout.
const ACTION_ICON_HIT_SLOP = { top: 8, bottom: 8, left: 8, right: 8 };

// Buy-sheet helpers: collage items carry brand/price/shopUrl through so the
// "Buy" bottom sheet can show each piece with where to buy it.

const toCamelCase = (str: string) => {
  return str
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase();
    })
    .replace(/[\s\-_]+/g, '');
};

const DailyOutfitSection = React.memo(({
  occasion,
  style,
  weather,
  weatherForAI,
  shopTops,
  shopBottoms,
  shopShoes,
  catalogEssentials,
  needsOuterwear,
  isReducedMotionEnabled,
  navigation,
  t,
  userId,
  enabled = true,
}: DailyOutfitSectionProps) => {
  const [stackIndex, setStackIndex] = useState(0);
  const stackRef = useRef<OutfitSwipeStackHandle>(null);
  const [shareModalVisible, setShareModalVisible] = useState(false);
  const [shopSheetVisible, setShopSheetVisible] = useState(false);
  // Snapshot of the outfit that was on screen when Buy was tapped — the sheet
  // must keep showing THAT outfit even if the deck rebuilds (catalog refresh,
  // regenerate) while it's open.
  const [shopSheetItems, setShopSheetItems] = useState<any[]>([]);

  const addOutfit = useWardrobeStore((s) => s.addOutfit);
  const saveOutfit = useWardrobeStore((s) => s.saveOutfit);
  const dislikeOutfitItems = useWardrobeStore((s) => s.dislikeOutfit);

  // ── Post-trial swipe gate: free/expired users get 3 swipes/day on Home ──
  // NOTE: swipe handlers below must NEVER await the network. requireFeature
  // is a synchronous local-quota check; the server increment runs in the
  // background via consume(). Awaiting consumeOrPaywall (Supabase RPC)
  // before setStackIndex left the deck blank until the round-trip returned —
  // a freeze on every swipe, on every Home section.
  const { consumeOrPaywall, consume, requireFeature, getRemaining, canUseNow } = useSubscriptionGate();
  const effectiveTier = useSubscriptionStore((s) => s.effectiveTier);
  const isTrialExpired = useSubscriptionStore((s) => s.isTrialExpired);

  // Load the daily outfits for this style and occasion
  const dailyAI = useDailyAIOutfit({
    style,
    occasion,
    weather: weatherForAI,
    variants: 15,
    enabled,
  });

  // Tracks the `dailyAI.outfits` batch (by reference) we've already asked to
  // be refilled, so the effect below fires regenerate() at most once per
  // batch even if it re-runs on a stale pre-reset `stackIndex` (see below).
  const refillRequestedForRef = useRef<unknown>(null);

  // Reset the swipe deck to the top whenever a fresh batch of outfits arrives
  // (initial load or a "regenerate" tap) — otherwise a stale stackIndex from
  // the previous batch could point past the end of the new one.
  //
  // Keyed on the `outfits` ARRAY REFERENCE, not a content hash of its ids:
  // the local rule-based fallback (used whenever the AI edge function comes
  // back empty) is deterministic and can return the exact same 15 ids every
  // call, so a content-based key would never change and this reset would
  // never fire again — leaving stackIndex stuck past the end forever.
  // `setOutfits` always produces a new array instance, so the reference
  // reliably changes on every load, cached or not, identical content or not.
  useEffect(() => {
    setStackIndex(0);
    refillRequestedForRef.current = null;
  }, [dailyAI.outfits]);

  // Once the user swipes through every AI look in today's batch, quietly
  // prepare a fresh one instead of dead-ending the deck — the stack shows a
  // brief "preparing new looks" state (below) while this is in flight, and
  // the reset effect above snaps stackIndex back to 0 as soon as it lands.
  //
  // `refillRequestedForRef` caps this to ONE regenerate() call per distinct
  // `dailyAI.outfits` batch. Without it, this effect can still observe a
  // stale (pre-reset) `stackIndex` on the very render a new batch lands —
  // the reset effect above runs in the same pass but its setState hasn't
  // committed yet — and would otherwise fire a second, redundant regenerate.
  useEffect(() => {
    if (
      dailyAI.outfits.length > 0
      && stackIndex >= dailyAI.outfits.length
      && !dailyAI.loading
      && refillRequestedForRef.current !== dailyAI.outfits
    ) {
      // Expired-trial / free users must NOT auto-regenerate past their
      // 3-swipe limit — the deck dead-ends into a Paywall CTA instead.
      if (effectiveTier === 'free' || isTrialExpired) {
        if (!canUseNow('homeSwipes')) {
          logger.debug(`Deck exhausted + free swipe quota hit — holding paywall state (${style})`);
          return;
        }
      }
      refillRequestedForRef.current = dailyAI.outfits;
      logger.debug(`Deck exhausted, preparing a fresh ${style} batch`);
      dailyAI.regenerate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stackIndex, dailyAI.outfits, dailyAI.loading, effectiveTier, isTrialExpired]);

  // Determine occasion from the style prop
  const deriveOccasion = useMemo((): Occasion => {
    const lower = occasion.toLowerCase();
    if (lower.includes('team') || lower.includes('work') || lower.includes('office')) return 'work';
    if (lower.includes('dinner') || lower.includes('night') || lower.includes('date')) return 'date';
    if (lower.includes('formal') || lower.includes('wedding') || lower.includes('gala')) return 'formal';
    if (lower.includes('gym') || lower.includes('sport') || lower.includes('workout')) return 'sport';
    if (lower.includes('travel') || lower.includes('airport') || lower.includes('flight')) return 'travel';
    return 'casual';
  }, [occasion]);

  // Every occasion card now renders through the same LiquidGlassCard path —
  // isOldMoney only picks the occasion-flavored copy and empty-state icon,
  // it no longer swaps in a separate "dinner" theme (that theme had drifted
  // to be visually identical to the default card except for a stray italic
  // subtitle and a different pager-dot color, which read as a bug, not a
  // design choice).
  const isOldMoney = style === 'old_money';
  const suggestionTextStyle = styles.premiumSuggestionText;
  const buyButtonStyle = styles.createAvatarButton;
  const emptyIconName = isOldMoney ? 'wine-outline' : 'shirt-outline';

  // Helper classifiers inside component
  const isOuterwear = (name: string) => isOuterwearName(name);

  // Pre-compiled list of curated combinations for client-side fallback
  const curatedCombinations = useMemo(() => {
    if (shopTops.length === 0 || shopBottoms.length === 0) return [];
    
    const realOuterwear = shopTops.filter(t => isOuterwear(t.name));
    const realTops = shopTops.filter(t => !isOuterwear(t.name));
    const topsPool = realTops.length > 0 ? realTops : shopTops;

    const results = [];
    const count = 3;
    
    for (let i = 0; i < count; i++) {
      const seed = style.length + i;
      results.push({
        id: i + 1,
        mainTop: topsPool[seed % topsPool.length],
        mainBottom: shopBottoms[(seed + 1) % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[(seed + 2) % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[(seed + 3) % shopShoes.length] : null,
      });
    }
    
    return results;
  }, [shopTops, shopBottoms, shopShoes, style]);

  const mapLegacyOutfitItemsForCollage = useCallback((c: any) => {
    const allOuterwear = catalogEssentials.filter(t => isOuterwear(t.name));
    const safeOuterwear = allOuterwear.length > 0 ? allOuterwear : catalogEssentials;
    const allShoes = shopShoes.length > 0
      ? shopShoes
      : catalogEssentials.filter(t => t.garmentType === 'shoes' || t.name.toLowerCase().includes('shoe') || t.name.toLowerCase().includes('sneaker'));

    const shopByMacro: Record<string, ShopCatalogItem[]> = {
      top: shopTops.filter(t => !isOuterwear(t.name)),
      bottom: shopBottoms.length > 0 ? shopBottoms : catalogEssentials,
      shoes: allShoes,
      outerwear: safeOuterwear,
    };

    const items: any[] = [];
    const filledSlots = new Set<string>();

    if (c?.outerLayer) {
      items.push({ id: `legacy_outer_${c.id}`, image: c.outerLayer.imageUrl || c.outerLayer.image, type: c.outerLayer.type || c.outerLayer.name || 'Outerwear', name: c.outerLayer.name || 'Outerwear', macroCategory: 'outerwear', brand: c.outerLayer.brand, price: c.outerLayer.price, currency: c.outerLayer.currency, shopUrl: c.outerLayer.sourceUrl || c.outerLayer.shopUrl });
      filledSlots.add('outerwear');
    }
    if (c?.mainTop) {
      const topName = c.mainTop.name || 'Top';
      const forcedTopType = isTopName(topName) && !isOuterwearName(topName) ? topName : `${topName} Shirt`;
      items.push({ id: `legacy_top_${c.id}`, image: c.mainTop.imageUrl || c.mainTop.image, type: forcedTopType, name: topName, macroCategory: 'top', brand: c.mainTop.brand, price: c.mainTop.price, currency: c.mainTop.currency, shopUrl: c.mainTop.sourceUrl || c.mainTop.shopUrl });
      filledSlots.add('top');
    }
    if (c?.mainBottom) {
      const btmName = c.mainBottom.name || 'Pants';
      const forcedBtmType = isBottomName(btmName) ? btmName : `${btmName} Pants`;
      items.push({ id: `legacy_btm_${c.id}`, image: c.mainBottom.imageUrl || c.mainBottom.image, type: forcedBtmType, name: btmName, macroCategory: 'bottom', brand: c.mainBottom.brand, price: c.mainBottom.price, currency: c.mainBottom.currency, shopUrl: c.mainBottom.sourceUrl || c.mainBottom.shopUrl });
      filledSlots.add('bottom');
    }
    if (c?.shoes && shopByMacro.shoes.length > 0) {
      const shoeName = c.shoes.name || 'Shoes';
      const forcedShoeType = isShoeName(shoeName) ? shoeName : `${shoeName} Shoe`;
      items.push({ id: `legacy_shoe_${c.id}`, image: c.shoes.imageUrl || c.shoes.image, type: forcedShoeType, name: shoeName, macroCategory: 'shoes', brand: c.shoes.brand, price: c.shoes.price, currency: c.shoes.currency, shopUrl: c.shoes.sourceUrl || c.shoes.shopUrl });
      filledSlots.add('shoes');
    }

    if (needsOuterwear && !filledSlots.has('outerwear')) {
      const fallbackItems = shopByMacro['outerwear'] || [];
      const realOuter = fallbackItems.filter(i => isOuterwear(i.name));
      if (realOuter.length > 0) {
        const shopItem = realOuter[Math.floor(Math.random() * realOuter.length)];
        items.push({
          id: `legacy_fill_outerwear_${c.id}`,
          image: shopItem.imageUrl,
          type: shopItem.name,
          name: shopItem.name,
          macroCategory: 'outerwear',
          brand: shopItem.brand,
          price: shopItem.price,
          currency: shopItem.currency,
          shopUrl: shopItem.sourceUrl,
        });
        filledSlots.add('outerwear');
      }
    }

    const mandatorySlots = ['top', 'bottom', 'shoes'];
    mandatorySlots.forEach(slot => {
      if (!filledSlots.has(slot)) {
        const fallbackItems = shopByMacro[slot] || [];
        if (fallbackItems.length > 0) {
          const shopItem = fallbackItems[Math.floor(Math.random() * fallbackItems.length)];
          items.push({
            id: `legacy_fill_${slot}_${c.id}`,
            image: shopItem.imageUrl,
            type: shopItem.name,
            name: shopItem.name,
            macroCategory: slot,
            brand: shopItem.brand,
            price: shopItem.price,
            currency: shopItem.currency,
            shopUrl: shopItem.sourceUrl,
          });
        }
      }
    });

    // Curated fallback combinations are sourced from the shop catalog.
    return items.map((item) => ({ ...item, isShopItem: true }));
  }, [catalogEssentials, shopTops, shopBottoms, shopShoes, needsOuterwear]);

  // Snapshot the shop pools once per outfits batch. The live catalog arrives
  // in stages (instant static seed → background Supabase refresh with
  // different rows), and remapping every card on each refresh swapped tile
  // images under already-mounted cards — the "appears, disappears, then the
  // final image appears seconds later" flash. A mounted deck keeps the pools
  // it was born with; the next batch (or regenerate) picks up fresh rows.
  // The one exception: a batch born while the catalog was still completely
  // empty upgrades to the first non-empty pools so it never stays blank.
  const poolSnapshotRef = useRef<{ tops: ShopCatalogItem[]; bottoms: ShopCatalogItem[]; shoes: ShopCatalogItem[] } | null>(null);
  const outfitsSnapshotRef = useRef<unknown>(null);
  {
    const livePools = { tops: shopTops, bottoms: shopBottoms, shoes: shopShoes };
    const liveEmpty = livePools.tops.length === 0 && livePools.bottoms.length === 0 && livePools.shoes.length === 0;
    const snap = poolSnapshotRef.current;
    const snapEmpty = !snap || (snap.tops.length === 0 && snap.bottoms.length === 0 && snap.shoes.length === 0);
    if (outfitsSnapshotRef.current !== dailyAI.outfits || (snapEmpty && !liveEmpty)) {
      outfitsSnapshotRef.current = dailyAI.outfits;
      poolSnapshotRef.current = livePools;
    }
  }
  const frozenPools = poolSnapshotRef.current ?? { tops: shopTops, bottoms: shopBottoms, shoes: shopShoes };

  const mapAiOutfitItemsForCollage = useCallback((outfit: any, outfitIndex = 0) => {
    const shopByMacro: Record<string, ShopCatalogItem[]> = {
      top: frozenPools.tops.filter(t => !isOuterwear(t.name)),
      bottom: frozenPools.bottoms,
      shoes: frozenPools.shoes,
      outerwear: frozenPools.tops.filter(t => isOuterwear(t.name)),
    };
    // Team Collaboration is shop-only — every item in every outfit must come
    // from shopping, and every outfit combination must be unique.
    const shopOnly = isShopOnlyDeck(style, occasion);

    // Track ids already used in THIS outfit so two slots never share one
    // garment (duplicate React keys collapsed two tiles into one — a 4-piece
    // outfit rendering as 2-3 pieces).
    const usedIds = new Set<string>();
    // Peek-only: finds the first unused pick WITHOUT marking it used, so
    // computing a candidate never wastes pool entries. Call `claimId` only
    // for the pick that actually renders.
    const peekUniqueShopPick = (pool: ShopCatalogItem[], startIndex: number): ShopCatalogItem | undefined => {
      if (pool.length === 0) return undefined;
      for (let offset = 0; offset < pool.length; offset++) {
        const candidate = pool[(startIndex + offset) % pool.length];
        if (!usedIds.has(String(candidate.id))) return candidate;
      }
      return undefined;
    };
    const claimId = (id: unknown) => { if (id != null) usedIds.add(String(id)); };

    const mappedRaw = (outfit.items || []).map((item: any, index: number) => {
      let img = item.imageUrl || item.image_url || item.image || '';
      let finalItem = { ...item };

      const macro = (item.macroCategory || '').toLowerCase();
      const macroNormalized = macro === 'upper_body' ? 'top' : macro === 'lower_body' ? 'bottom' : macro;
      const shopItems = shopByMacro[macroNormalized] || [];

      // Keep real wardrobe pieces intact — except on shop-only decks where
      // wardrobe pieces are never allowed. The old implementation replaced
      // every AI-selected item with a random catalog product from the same
      // category, which made a wardrobe outfit look like an advert. Only
      // resolve through the catalog when the item is explicitly a shop pick,
      // or use it as a visual fallback when the original item has no image.
      const exactShopItem = shopItems.find((candidate) => String(candidate.id) === String(item.id));
      // Shop-only decks must ALWAYS resolve to a real catalog row. The old
      // code kept the AI's original URL when `isShopItem === true`, even if
      // the id was hallucinated (no exact catalog match) — that URL then
      // 404'd in <Piece/> and rendered as a blank placeholder tile.
      const needsShopSwap = shopOnly && shopItems.length > 0 && (
        item.isShopItem !== true || !exactShopItem || !img
      );
      // Deterministic rotation (no Math.random): the same batch always maps
      // to the same visuals, and different outfits rotate onto different shop
      // picks instead of randomly colliding onto identical combinations.
      // Peek (no claim) so an unused candidate never wastes a pool entry.
      let shopItem: ShopCatalogItem | undefined;
      if (exactShopItem && !usedIds.has(String(exactShopItem.id)) && (!shopOnly || img)) {
        shopItem = exactShopItem;
      } else if (needsShopSwap) {
        shopItem = peekUniqueShopPick(shopItems, outfitIndex + index);
      } else if (!img && shopItems.length > 0) {
        shopItem = peekUniqueShopPick(shopItems, outfitIndex + index);
      }
      if (shopItem && (item.isShopItem === true || !img || needsShopSwap)) {
        finalItem = {
          ...item,
          id: shopItem.id,
          imageUrl: shopItem.imageUrl,
          image: shopItem.imageUrl,
          name: shopItem.name,
          type: shopItem.name,
          macroCategory: macroNormalized,
          isShopItem: true,
          brand: shopItem.brand ?? item.brand,
          price: shopItem.price ?? item.price,
          currency: (shopItem as any).currency ?? (item as any).currency,
          shopUrl: shopItem.sourceUrl || (item as any).shopUrl,
        };
        img = typeof shopItem.imageUrl === 'string' ? shopItem.imageUrl : img;
        claimId(shopItem.id);
      } else if (finalItem.id) {
        // Reserve wardrobe ids too so a later shop fill can't collide with
        // an id already on screen.
        claimId(finalItem.id);
      }

      return {
        id: finalItem.id || `home_ai_${outfitIndex}_${index}`,
        image: img,
        type: finalItem.type || finalItem.name || 'Item',
        name: finalItem.name || finalItem.type || 'Item',
        macroCategory: finalItem.macroCategory,
        isShopItem: finalItem.isShopItem === true,
        brand: finalItem.brand,
        price: finalItem.price,
        currency: finalItem.currency,
        shopUrl: finalItem.shopUrl || finalItem.sourceUrl,
      };
    });
    // Shop-only decks must never render a wardrobe tile — drop it here so
    // the mandatory-slot fill below replaces it with a shop pick.
    const mapped = mappedRaw.filter((ci: any) => {
      if (shopOnly && ci.isShopItem !== true) return false;
      return true;
    });

    // Recompute filled slots AFTER the shop-only filter — otherwise a dropped
    // wardrobe tile would still count as "filled" and the slot would never be
    // replaced with a shop pick, leaving incomplete/duplicated cards.
    const filledSlots = new Set<string>();
    mapped.forEach((ci: any) => {
      if (ci.macroCategory) filledSlots.add(String(ci.macroCategory).toLowerCase());
    });

    const mandatorySlots = ['top', 'bottom', 'shoes'];
    mandatorySlots.forEach((slot, slotIdx) => {
      if (!filledSlots.has(slot)) {
        const fallbackItems = shopByMacro[slot] || [];
        if (fallbackItems.length > 0) {
          const shopItem = peekUniqueShopPick(fallbackItems, outfitIndex + slotIdx)
            ?? fallbackItems[(outfitIndex + slotIdx) % fallbackItems.length];
          if (shopItem && !usedIds.has(String(shopItem.id))) {
            claimId(shopItem.id);
            mapped.push({
              id: shopItem.id,
              image: shopItem.imageUrl,
              type: shopItem.name,
              name: shopItem.name,
              macroCategory: slot,
              isShopItem: true,
              brand: shopItem.brand,
              price: shopItem.price,
              currency: (shopItem as any).currency,
              shopUrl: shopItem.sourceUrl,
            });
            filledSlots.add(slot);
          }
        }
      }
    });

    // Last-defense styling rule (render layer): suits never pair with shorts,
    // and formal decks never contain shorts at all. Upstream rebuilds already
    // enforce this, but an edge-function outfit or a stale cache entry can
    // still arrive with a blazer + shorts combination — swap the shorts for
    // trousers/chinos here so it can never reach the card.
    const SHORTS_LIKE_RE = /\b(shorts?|bermudas?|trunks?|swim\w*|swimwear|board\s*shorts?|bathing\s*suit)\b/i;
    const FORMAL_LAYER_LIKE_RE = /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/i;
    const isShortsLike = (ci: any): boolean => {
      if (String(ci?.macroCategory || '').toLowerCase() !== 'bottom') return false;
      return SHORTS_LIKE_RE.test(`${ci?.type || ''} ${ci?.name || ''}`);
    };
    const isFormalLayerLike = (ci: any): boolean =>
      FORMAL_LAYER_LIKE_RE.test(`${ci?.type || ''} ${ci?.name || ''}`);
    const styleKey = String(style || '').toLowerCase().replace(/[\s-]+/g, '_');
    const isFormalDeck = styleKey === 'business_casual' || styleKey === 'old_money' || styleKey === 'classic' || styleKey === 'formal';
    const shortsIdx = mapped.findIndex(isShortsLike);
    if (shortsIdx >= 0 && (isFormalDeck || mapped.some(isFormalLayerLike))) {
      const replacementPool = (shopByMacro.bottom || []).filter(
        (b) => !usedIds.has(String(b.id)) && !SHORTS_LIKE_RE.test(`${b.name || ''}`),
      );
      const replacement = replacementPool.length > 0
        ? replacementPool[outfitIndex % replacementPool.length]
        : undefined;
      if (replacement) {
        claimId(replacement.id);
        mapped[shortsIdx] = {
          id: replacement.id,
          image: replacement.imageUrl,
          type: replacement.name,
          name: replacement.name,
          macroCategory: 'bottom',
          isShopItem: true,
          brand: replacement.brand,
          price: replacement.price,
          currency: (replacement as any).currency,
          shopUrl: replacement.sourceUrl,
        };
      } else {
        // No trousers available — drop the shorts rather than showing a
        // suit + shorts combination. A 3-piece card beats a broken 4-piece.
        mapped.splice(shortsIdx, 1);
      }
    }

    return mapped;
    // NOTE: intentionally depends on the per-batch frozen pools, not the
    // live catalog arrays — see the snapshot above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frozenPools, style, occasion]);

  const collageItemsList = useMemo(() => {
    return dailyAI.outfits.map((o, idx) => mapAiOutfitItemsForCollage(o, idx));
  }, [dailyAI.outfits, mapAiOutfitItemsForCollage]);

  // Built once per actual data change rather than on every render — the
  // OutfitSwipeStack's image-prefetch effect depends on `cards`, and an
  // unstable reference there made it re-issue Image.prefetch() for the SAME
  // uri on every unrelated re-render (weather ticks, wardrobe-store updates,
  // etc.), which could flicker whichever piece was already on screen.
  const deckData = useMemo(() => {
    const aiOutfits = dailyAI.outfits;
    const useAI = aiOutfits.length > 0;

    type DailyOutfitCard = { id: string; outfit?: any; legacy?: any; index: number };
    const data: DailyOutfitCard[] = useAI
      // The backend can repeat an outfit id across variants. A guaranteed
      // unique deck key is required here because the maps below and React's
      // card reconciliation both use it; duplicate ids made multiple swipes
      // appear to show the same outfit.
      ? aiOutfits.map((o, i) => ({ id: `${o.id || 'ai'}-${style}-${i}`, outfit: o, index: i }))
      : curatedCombinations.map((c) => ({ id: String(c.id), legacy: c, index: 0 }));

    // Full-fidelity collage items per card (image + name + macroCategory) —
    // used for the share sheet and the Try On handoff. The swipe stack
    // itself only needs the trimmed {id, image, category} shape below.
    const collageItemsByCardKey = new Map<string, any[]>();
    data.forEach((d) => {
      const items = d.outfit
        ? (collageItemsList[d.index] ?? mapAiOutfitItemsForCollage(d.outfit, d.index))
        : mapLegacyOutfitItemsForCollage(d.legacy);
      collageItemsByCardKey.set(d.id, items);
    });

    // Item IDs actually saved/disliked when the user decides on a card — the
    // AI outfit's own items for AI-generated looks, or the underlying shop
    // items for a curated combo when the AI batch hasn't loaded yet.
    const itemIdsByCardKey = new Map<string, string[]>();
    data.forEach((d) => {
      if (d.outfit) {
        itemIdsByCardKey.set(
          d.id,
          (d.outfit.items || []).filter((it: any) => it.id).map((it: any) => String(it.id)),
        );
      } else {
        const c = d.legacy;
        itemIdsByCardKey.set(
          d.id,
          [c?.mainTop?.id, c?.mainBottom?.id, c?.outerLayer?.id, c?.shoes?.id]
            .filter((v: any) => v !== undefined && v !== null)
            .map(String),
        );
      }
    });

    const cards: SwipeStackCard[] = [];
    const seenCardSigs = new Set<string>();
    data.forEach((d) => {
      const collageItems = collageItemsByCardKey.get(d.id) ?? [];
      // Never show the same combination twice — signature by sorted item ids.
      const sig = collageItems.map((ci: any) => String(ci.id)).sort().join('|');
      if (sig && seenCardSigs.has(sig)) return;
      if (sig) seenCardSigs.add(sig);
      const shopCount = collageItems.filter((ci: any) => ci.isShopItem === true).length;
      const wardrobeCount = collageItems.length - shopCount;
      // Shop-only decks (Team Collaboration) never show "0 yours · N suggested".
      const caption = wardrobeCount === 0 && shopCount > 0
        ? `${shopCount} suggested`
        : shopCount > 0
          ? t('home.outfitSources', { wardrobeCount, shopCount })
          : t('home.fromYourWardrobe', { count: wardrobeCount || collageItems.length });
      cards.push({
        key: d.id,
        items: collageItems.map((ci: any) => ({
          id: ci.id,
          image: ci.image,
          category: ci.macroCategory,
          name: ci.name,
          isShopItem: ci.isShopItem === true,
        })),
        wardrobeCount,
        shopCount,
        caption,
      });
    });

    return { cards, collageItemsByCardKey, itemIdsByCardKey };
  }, [dailyAI.outfits, curatedCombinations, collageItemsList, mapLegacyOutfitItemsForCollage, mapAiOutfitItemsForCollage, style, t]);

  const getOccasionTranslation = (key: string) => {
    const camel = toCamelCase(key);
    if (t(`home.${camel}`) !== `home.${camel}`) return t(`home.${camel}`);
    if (t(`home.${key}`) !== `home.${key}`) return t(`home.${key}`);
    if (t(key) !== key) return t(key);
    if (t(camel) !== camel) return t(camel);
    return key;
  };

  const getStyleTranslation = (key: string) => {
    const camel = toCamelCase(key);
    if (t(`home.${camel}`) !== `home.${camel}`) return t(`home.${camel}`);
    if (t(`home.${key}`) !== `home.${key}`) return t(`home.${key}`);
    if (t(key) !== key) return t(key);
    if (t(camel) !== camel) return t(camel);
    return key.replace(/_/g, ' ');
  };

  const titleText = getOccasionTranslation(occasion);
  const subtitleText = `${getStyleTranslation(style)} ›`;

  if (dailyAI.loading && dailyAI.outfits.length === 0) {
    return (
      <View style={styles.premiumSection}>
        <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
          <ScaledText style={styles.premiumHeaderTitle}>{titleText}</ScaledText>
          <ScaledText style={styles.premiumHeaderSubtitle}>{subtitleText}</ScaledText>
        </View>
        <View style={{ paddingHorizontal: spacing.screenPadding }}>
          <LiquidGlassCard
            variant="light"
            style={styles.premiumCard}
            contentStyle={[styles.premiumCardContent, { minHeight: 340, justifyContent: 'center' }]}
          >
            <ActivityIndicator size="small" color={colors.accent.primary} />
            <ScaledText style={[styles.premiumSuggestionText, { marginTop: spacing.sm }]}>
              {isOldMoney ? "Styling tonight's looks…" : "Styling today's looks…"}
            </ScaledText>
          </LiquidGlassCard>
        </View>
      </View>
    );
  }

  const { cards, collageItemsByCardKey, itemIdsByCardKey } = deckData;

  const topCard = cards[stackIndex];
  const hasTopCard = !!topCard;
  const topCollageItems = topCard ? (collageItemsByCardKey.get(topCard.key) ?? []) : [];
  const topOutfitForShare = topCard ? { id: topCard.key, items: topCollageItems, occasion, style } : null;
  // True once the user has swiped through every card in today's batch —
  // distinct from `cards.length === 0`, which means generation hasn't
  // produced anything at all yet (a real failure, not "all caught up").
  const stackExhausted = cards.length > 0 && !hasTopCard;

  const handleSwipeRight = (card: SwipeStackCard): boolean => {
    // Synchronous gate only: when the local quota is exhausted this opens
    // the Paywall and returns false. The server-authoritative increment
    // happens in the background AFTER the deck advances (below), so the
    // gesture never waits on a network round-trip.
    // The boolean return drives the stack's rollback: on `false` the card
    // springs back instead of staying flown off (which would desync the
    // visual deck from `stackIndex` and leave Buy/share reading a stale card).
    if (!requireFeature('homeSwipes')) return false;
    const itemIds = itemIdsByCardKey.get(card.key) ?? [];
    if (itemIds.length > 0) {
      // Freeze the exact pieces shown on the card (wardrobe + shop images)
      // so Collections / Saved Looks render the full outfit later even if
      // shop items page out of the live catalog or were never in wardrobe.
      const collageSnapshot = (collageItemsByCardKey.get(card.key) ?? [])
        .filter((ci: any) => typeof ci?.image === 'string' && ci.image.length > 0)
        .slice(0, 4)
        .map((ci: any) => ({
          id: String(ci.id ?? ''),
          image: String(ci.image),
          category: ci.macroCategory ?? ci.category,
          name: ci.name,
        }));
      const outfitId = addOutfit({
        userId: userId ?? '',
        itemIds,
        occasion: deriveOccasion,
        generatedBy: 'ai',
        snapshotItems: collageSnapshot,
        savedFrom: 'home',
      } as any);
      // Home likes → Closet → Collections (Saved). savedFrom='home' keeps
      // them out of Profile → Saved Looks (which is for single-clothing saves).
      saveOutfit(outfitId, 'saved', 'home');
    }
    setStackIndex((i) => i + 1);
    // Server-authoritative swipe count, fire-and-forget: mirrors the new
    // usage into the local cache on return, so the next swipe's synchronous
    // requireFeature above stays accurate. Never awaited — awaiting it here
    // froze the deck blank until the RPC round-trip finished.
    void consume('homeSwipes').catch(() => {});
    return true;
  };

  const handleSwipeLeft = (card: SwipeStackCard): boolean => {
    // Same 3-swipe/day gate, same optimistic pattern as swipe-right.
    if (!requireFeature('homeSwipes')) return false;
    const itemIds = itemIdsByCardKey.get(card.key) ?? [];
    if (itemIds.length > 0) dislikeOutfitItems(itemIds);
    setStackIndex((i) => i + 1);
    void consume('homeSwipes').catch(() => {});
    return true;
  };

  // Remaining free swipes today (-1 = unlimited). Used for the counter
  // pill and the locked empty-state below.
  const swipesRemaining = getRemaining('homeSwipes');
  const isSwipeLocked = !canUseNow('homeSwipes');

  // While a fresh batch is in flight — either because the deck just ran out
  // (auto-triggered above) or the user tapped "Try again" after a failure —
  // show a brief loading state instead of a dead-end "no more" card.
  // When the free 3-swipe quota is hit, dead-end into a Paywall CTA instead.
  const emptyStateContent = isSwipeLocked ? (
    <>
      <Ionicons name="lock-closed-outline" size={44} color={colors.accent.primary} />
      <ScaledText style={[suggestionTextStyle, { marginTop: spacing.md, textAlign: 'center', fontWeight: '700' }]}>
        {t('home.swipeLimitTitle', "You've used your 3 free swipes")}
      </ScaledText>
      <ScaledText style={[suggestionTextStyle, { marginTop: spacing.xs, textAlign: 'center' }]}>
        {t('home.swipeLimitBody', 'Upgrade to Pro for unlimited swipes.')}
      </ScaledText>
      <TouchableOpacity
        style={[styles.createAvatarButton, { marginTop: spacing.md }]}
        onPress={() => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
          (navigation as any)?.navigate?.('Paywall');
        }}
        accessibilityLabel={t('home.unlockUnlimited', 'Unlock unlimited')}
        accessibilityRole="button"
      >
        <ScaledText style={styles.createAvatarText}>{t('home.unlockUnlimited', 'Unlock unlimited')}</ScaledText>
      </TouchableOpacity>
    </>
  ) : stackExhausted || dailyAI.loading ? (
    <>
      <ActivityIndicator size="small" color={colors.accent.primary} />
      <ScaledText style={[suggestionTextStyle, { marginTop: spacing.md, textAlign: 'center' }]}>
        Preparing new looks…
      </ScaledText>
    </>
  ) : (
    <>
      <Ionicons name={emptyIconName} size={44} color={colors.text.tertiary} />
      <ScaledText style={[suggestionTextStyle, { marginTop: spacing.md, textAlign: 'center' }]}>
        No looks available right now.
      </ScaledText>
      <TouchableOpacity
        style={[styles.createAvatarButton, { marginTop: spacing.md }]}
        onPress={async () => {
          // Same AI-generation gate as the refresh button — expired users
          // go to Paywall instead of regenerating for free.
          const ok = await consumeOrPaywall('aiOutfits');
          if (!ok) return;
          logger.debug(`Regenerate ${style} daily outfits (empty state)`);
          dailyAI.regenerate();
        }}
        accessibilityLabel={t('home.tryAgain')}
        accessibilityRole="button"
      >
        <ScaledText style={styles.createAvatarText}>{t('home.tryAgain')}</ScaledText>
      </TouchableOpacity>
    </>
  );

  return (
    <View style={styles.premiumSection}>
      <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
        <ScaledText style={styles.premiumHeaderTitle}>{titleText}</ScaledText>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <ScaledText style={styles.premiumHeaderSubtitle}>{subtitleText}</ScaledText>
          {/* Free/expired users: show remaining swipes (3/day). Paid: unlimited. */}
          {swipesRemaining !== -1 && (
            <TouchableOpacity
              onPress={() => {
                if (isSwipeLocked) (navigation as any)?.navigate?.('Paywall');
              }}
              activeOpacity={0.7}
              style={{
                flexDirection: 'row',
                alignItems: 'center',
                gap: 4,
                paddingHorizontal: 8,
                paddingVertical: 4,
                borderRadius: 12,
                backgroundColor: isSwipeLocked ? '#7C3AED' : 'rgba(124,58,237,0.10)',
                borderWidth: 1,
                borderColor: 'rgba(124,58,237,0.25)',
              }}
              accessibilityLabel={t('home.swipesRemaining', '{{count}} swipes left', { count: swipesRemaining })}
            >
              <Ionicons
                name={isSwipeLocked ? 'lock-closed' : 'layers-outline'}
                size={11}
                color={isSwipeLocked ? '#FFF' : '#7C3AED'}
              />
              <ScaledText style={{ fontSize: 11, fontWeight: '800', color: isSwipeLocked ? '#FFF' : '#7C3AED' }}>
                {isSwipeLocked
                  ? t('home.pro', 'PRO')
                  : t('home.swipesLeft', '{{count}}/{{total}} free', { count: swipesRemaining, total: FREE_HOME_SWIPES_PER_DAY })}
              </ScaledText>
            </TouchableOpacity>
          )}
        </View>
      </View>

      <View style={{ paddingHorizontal: spacing.screenPadding }}>
        <OutfitSwipeStack
          ref={stackRef}
          cards={cards}
          activeIndex={stackIndex}
          onSwipeRight={handleSwipeRight}
          onSwipeLeft={handleSwipeLeft}
          reducedMotion={isReducedMotionEnabled}
          height={390}
          emptyState={emptyStateContent}
        />
      </View>

      {hasTopCard && (
        <View style={styles.swipeHintRow}>
          <Ionicons name="arrow-back" size={12} color={colors.text.tertiary} />
          <ScaledText style={styles.swipeHintText}>{t('home.swipeHint')}</ScaledText>
          <Ionicons name="arrow-forward" size={12} color={colors.text.tertiary} />
        </View>
      )}

      {/* Action Footer */}
      <View style={styles.premiumFooter}>
        <View style={styles.premiumActionIcons}>
          <TouchableOpacity
            style={styles.actionIconButton}
            hitSlop={ACTION_ICON_HIT_SLOP}
            disabled={!hasTopCard}
            onPress={() => {
              // Pre-check the swipe quota BEFORE starting the fly-off
              // animation: a blocked swipe would otherwise fly the card away
              // while `stackIndex` stays, desyncing the deck (Buy reads stale).
              if (!canUseNow('homeSwipes')) { requireFeature('homeSwipes'); return; }
              stackRef.current?.swipeRight();
            }}
            accessibilityLabel={t('home.saveOutfit', 'Save outfit')}
            accessibilityRole="button"
          >
            <Ionicons name="heart-outline" size={24} color={hasTopCard ? colors.text.primary : colors.text.tertiary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionIconButton}
            hitSlop={ACTION_ICON_HIT_SLOP}
            onPress={async () => {
              // Regenerating the deck costs an AI generation — gate it the
              // same as Create Outfit for expired-trial users.
              const ok = await consumeOrPaywall('aiOutfits');
              if (!ok) return;
              logger.debug(`Regenerate ${style} daily outfits`);
              dailyAI.regenerate();
            }}
            accessibilityLabel={t('home.tryAgain')}
            accessibilityRole="button"
          >
            <Ionicons name="refresh-outline" size={22} color={colors.text.primary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionIconButton}
            hitSlop={ACTION_ICON_HIT_SLOP}
            disabled={!hasTopCard}
            onPress={() => {
              // Same pre-check as save: never animate a swipe the gate will reject.
              if (!canUseNow('homeSwipes')) { requireFeature('homeSwipes'); return; }
              stackRef.current?.swipeLeft();
            }}
            accessibilityLabel={t('home.dislikeOutfit', 'Dislike outfit')}
            accessibilityRole="button"
          >
            <Ionicons name="close" size={24} color={hasTopCard ? colors.text.primary : colors.text.tertiary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionIconButton}
            hitSlop={ACTION_ICON_HIT_SLOP}
            disabled={!topOutfitForShare}
            onPress={() => {
              if (!topOutfitForShare) return;
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              setShareModalVisible(true);
            }}
            accessibilityLabel={t('home.shareOutfit', 'Share outfit')}
            accessibilityRole="button"
          >
            <Ionicons name="paper-plane-outline" size={22} color={colors.text.primary} />
          </TouchableOpacity>
        </View>
        <TouchableOpacity
          style={[buyButtonStyle, !hasTopCard && styles.createAvatarButtonDisabled]}
          disabled={!hasTopCard}
          onPress={() => {
            logger.debug('Buy button pressed');
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            setShopSheetItems(topCollageItems);
            setShopSheetVisible(true);
          }}
          accessibilityLabel={t('home.buy', 'Buy')}
          accessibilityRole="button"
        >
          <ScaledText style={styles.createAvatarText}>{t('home.buy', 'Buy')}</ScaledText>
        </TouchableOpacity>
      </View>
      {topOutfitForShare && (
        <OutfitShareModal
          visible={shareModalVisible}
          onClose={() => setShareModalVisible(false)}
          outfit={topOutfitForShare}
        />
      )}
      <OutfitShopSheet
        visible={shopSheetVisible}
        onClose={() => setShopSheetVisible(false)}
        items={shopSheetItems}
        occasion={occasion}
        style={style}
      />
    </View>
  );
});

// ============================================
// MAIN COMPONENT
// ============================================

const HomeScreen = () => {
  const navigation = useAppNavigation();
  const isFocused = useIsFocused();
  const { isReducedMotionEnabled } = useAccessibility();
  const { t } = useTranslation();

  // ── Performance timing ────────────────────────────────────────────────────
  // Record mount time so we can measure how long until the screen is ready.
  React.useEffect(() => {
    perfMark('HomeScreen:mount');
    // Screen is interactive immediately after mount — weather loads async
    // in the background and doesn't block user interaction.
    perfMeasure('HomeScreen:mount');
    perfScreenReady('Home');
    return () => {
      logger.debug('[PERF] 🏠 HomeScreen unmounted');
    };
  }, []);

  // Initialize the player without auto-playing — playback is started lazily
  // in the useFocusEffect below, only when the screen is actually visible.
  // This prevents the video from loading into memory on every Home tab visit
  // before the screen is even rendered (fixes Defect 2.2).
  const player = useVideoPlayer(require("../assets/videos/nux_men_o.mp4"), (player) => {
    player.loop = true;
    player.muted = true;
    // Do NOT call player.play() here — play is deferred to useFocusEffect
  });

  // Use useFocusEffect for reliable play/pause tied to tab focus lifecycle.
  // This fires correctly when the tab navigator shows/hides the screen,
  // whereas useIsFocused + useEffect can miss rapid tab switches.
  // The cleanup (blur) callback pauses the video and releases the decoder
  // buffer so video memory is not held while the user is on other tabs.
  useFocusEffect(
    useCallback(() => {
      let active = true;

      // Defer video playback past the tab-switch animation to avoid lag
      // during the transition. Plain setTimeout — InteractionManager is
      // deprecated (startup warning) and behaves the same here.
      const timer = setTimeout(() => {
        if (!active) return;
        try {
          player.play();
        } catch (error) {
          logger.warn('Unable to play hero video', error);
        }
      }, 350);

      return () => {
        active = false;
        clearTimeout(timer);
        // Pause on blur — releases decoder resources while off-screen
        try {
          player.pause();
        } catch (error) {
          logger.warn('Unable to pause hero video', error);
        }
      };
    }, [player])
  );

  // State
  const [userName, setUserName] = useState("User");

  const [weather, setWeather] = useState<WeatherData | null>(null);
  // No OpenWeather key → the widget is disabled: start unloaded so the
  // first frame never flashes a spinner that then vanishes (the key is
  // empty in this build, so the fetch below could only 401 and return
  // null, which read as "something loading then disappeared").
  const [loadingWeather, setLoadingWeather] = useState(() => !!Config.weather.apiKey);
  const [greeting, setGreeting] = useState(t('common.goodMorning'));

  // Wardrobe store data for core loop — useShallow handles snapshot caching
  // correctly in Zustand v5, preventing the React 18 "getSnapshot should be
  // cached" warning that fires when a selector returns a new object each call.
  const { items, wearLogs } = useWardrobeStore(
    useShallow((state) => ({ items: state.items, wearLogs: state.wearLogs }))
  );

  // Daily AI outfits — one batch per category, regenerates once per calendar day.
  // Each Home section feeds its own category in so the AI styles its variants
  // to match the section title ("Team Collaboration / Business Casual", etc.).
  // Memoize weatherForAI so it only changes when the actual temp/condition
  // values change — passing a new object literal on every render would cause
  // useDailyAIOutfit's `run` callback to be recreated every render, which
  // triggers the generation effect → setLoading(true) → re-render → loop.
  const weatherForAI = useMemo(
    () => weather ? { temp: weather.temp, condition: weather.description } : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [weather?.temp, weather?.description],
  );

  // Dynamic home screen occasions editable from admin panel / Supabase
  const [dynamicOccasions, setDynamicOccasions] = useState<any[]>([]);

  useEffect(() => {
    if (!isFocused) return;

    let active = true;
    const fetchOccasions = async () => {
      try {
        const { data, error } = await supabase
          .from('home_occasions')
          .select('*')
          .eq('is_active', true)
          .order('sort_order', { ascending: true });

        if (active) {
          if (!error && data && data.length > 0) {
            setDynamicOccasions(data);
          } else {
            // Default occasions fallback
            setDynamicOccasions([
              { id: 'default-1', occasion: 'Team Collaboration', style: 'business_casual' },
              { id: 'default-2', occasion: 'Night-Time Dinner', style: 'old_money' }
            ]);
          }
        }
      } catch (err) {
        if (active) {
          setDynamicOccasions([
            { id: 'default-1', occasion: 'Team Collaboration', style: 'business_casual' },
            { id: 'default-2', occasion: 'Night-Time Dinner', style: 'old_money' }
          ]);
        }
      }
    };

    fetchOccasions();
    return () => {
      active = false;
    };
  }, [isFocused]);

  // Today's Look Suggestion
  const todaysOutfit = useMemo(() => {
    if (items.length >= 3) {
      // Use quickSuggest for a real outfit
      const engineWeather = weatherForAI
        ? { temp: weatherForAI.temp, condition: weatherForAI.condition }
        : undefined;
      return quickSuggest(items, wearLogs, engineWeather);
    }
    return null;
  }, [items, wearLogs, weatherForAI]);

  // Determine greeting based on time
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting(t('aiHub.goodMorning'));
    else if (hour < 18) setGreeting(t('aiHub.goodAfternoon'));
    else setGreeting(t('aiHub.goodEvening'));
  }, [t]);

  // Read username from Supabase auth store (no JWT decode needed)
  const authUsername = useAuthStore(selectUsername);
  const userId = useAuthStore((state) => state.user?.id);
  // Gate the outfit engine: never run edge-function + fallback queries +
  // combinatorial rebuild for an anonymous/stale user (delete -> create gap)
  // or while the wardrobe rehydrate is still in flight. Both conditions
  // caused the thundering herd that froze fresh accounts — the decks now
  // wait for a real userId + settled wardrobe and then serve the daily
  // cache instantly.
  const isWardrobeLoading = useWardrobeStore((state) => state.isLoading);
  const dailyDecksEnabled = isFocused && !!userId && !isWardrobeLoading;
  // Defer deck DATA past first paint: each DailyOutfitSection fires an
  // edge-function + combinatorial rebuild on enable. Enabling the moment the
  // wardrobe settles competes with hero/video first paint (freeze). The
  // sections render skeletons while disabled, so no layout jump — the fetch
  // simply starts ~600ms later, after the user sees content.
  const [decksReady, setDecksReady] = useState(false);
  useEffect(() => {
    if (!dailyDecksEnabled) {
      setDecksReady(false);
      return;
    }
    const timer = setTimeout(() => setDecksReady(true), 600);
    return () => clearTimeout(timer);
  }, [dailyDecksEnabled]);
  const decksFetchEnabled = dailyDecksEnabled && decksReady;
  useEffect(() => {
    if (authUsername) {
      setUserName(authUsername);
    }
  }, [authUsername]);

  // Fetch weather with a 30-min AsyncStorage cache. Previously every mount ran
  // permission → GPS → network serially on the critical path; now a fresh
  // cache paints instantly and skips GPS entirely (the common case — most
  // Home visits are minutes apart). Stale/missing cache refreshes in background.
  useEffect(() => {
    // Weather disabled (no API key) — skip the location permission prompt
    // and the network call entirely; both could only fail.
    if (!Config.weather.apiKey) {
      setLoadingWeather(false);
      return;
    }
    const controller = new AbortController();
    const { signal } = controller;
    const WEATHER_CACHE_KEY = '@home_weather_cache';
    const WEATHER_TTL_MS = 30 * 60 * 1000;

    const readCache = async (): Promise<boolean> => {
      try {
        const raw = await AsyncStorage.getItem(WEATHER_CACHE_KEY);
        if (!raw || signal.aborted) return false;
        const cached = JSON.parse(raw) as { data: WeatherData; ts: number };
        if (cached?.data && Date.now() - cached.ts < WEATHER_TTL_MS) {
          setWeather(cached.data);
          setLoadingWeather(false);
          return true;
        }
      } catch {
        // Corrupt cache — fall through to network.
      }
      return false;
    };

    const fetchWeather = async () => {
      try {
        // Instant paint from cache; skip the GPS chain when fresh.
        if (await readCache()) return;

        // Location acquisition — environmental failures (no GPS fix,
        // airplane mode, tunnel, slow first fix) are NORMAL, not crashes.
        // They must never reach logger.error (redbox in dev + Sentry
        // exception per cold start). Bail silently to no-weather UI.
        let coords: { latitude: number; longitude: number } | null = null;
        try {
          const { status } = await Location.requestForegroundPermissionsAsync();
          // Bail out if the component unmounted while waiting for permission
          if (signal.aborted) return;

          if (status !== 'granted') {
            if (!signal.aborted) setLoadingWeather(false);
            return;
          }

          const location = await Location.getCurrentPositionAsync({});
          // Bail out if the component unmounted while waiting for GPS
          if (signal.aborted) return;

          coords = location.coords;
        } catch {
          if (!signal.aborted) setLoadingWeather(false);
          return;
        }
        if (!coords) {
          if (!signal.aborted) setLoadingWeather(false);
          return;
        }

        const { latitude, longitude } = coords;

        const response = await fetch(
          `${Config.weather.baseUrl}/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${Config.weather.apiKey}`,
          { signal }
        );
        const data = await response.json();

        if (!signal.aborted && data.main && data.weather) {
          const fresh: WeatherData = {
            temp: Math.round(data.main.temp),
            description: data.weather[0].description,
            icon: data.weather[0].icon,
            city: data.name,
          };
          setWeather(fresh);
          AsyncStorage.setItem(
            WEATHER_CACHE_KEY,
            JSON.stringify({ data: fresh, ts: Date.now() }),
          ).catch(() => {});
        }
      } catch (error) {
        // AbortError is expected when the component unmounts — don't log it
        if (signal.aborted) return;
        logger.error('Weather fetch error', error);
      } finally {
        if (!signal.aborted) {
          setLoadingWeather(false);
        }
      }
    };

    fetchWeather();

    // Cancel in-flight GPS and network requests on unmount
    return () => controller.abort();
  }, []);

  // Wardrobe Essentials — sourced from all shop_catalog sources for maximum variety
  // Use a plain object instead of Set so adding one item doesn't invalidate the
  // entire essentials grid (Set creates a new reference on every add).
  const [essentialAddStates, setEssentialAddStates] = useState<Record<string, EssentialAddState>>({});
  const addItem = useWardrobeStore(selectAddItem);

  const {
    items: catalogEssentials,
    loading: essentialsLoading,
    error: essentialsError,
  } = useShopCatalog({ enabled: true, source: 'all' });

  // Sync the fetched catalog into the shared store so other screens (e.g.
  // ProfileScreen) can read it without firing their own Supabase query
  // (fixes Defect 1.5).
  // IMPORTANT: We only sync `items`, NOT `loading`. Syncing loading caused an
  // infinite loop: setLoading(true) → store update → re-render → essentialsLoading
  // changes → effect re-runs → setLoading(false) → store update → re-render → loop.
  // ProfileScreen only reads items from the store, so loading sync is unnecessary.
  useEffect(() => {
    if (!essentialsLoading && catalogEssentials.length > 0) {
      useShopCatalogStore.getState().setItems(catalogEssentials);
    }
  }, [catalogEssentials, essentialsLoading]);

  // NOTE: shoes are already included in catalogEssentials (source: 'all').
  // A second useShopCatalog call for shoes was removed to avoid a redundant
  // Supabase query that competed with the main fetch on every mount.

  const essentialsItems = useMemo(
    () => selectEssentialShoppingMix(catalogEssentials, new Date().toDateString()),
    [catalogEssentials]
  );

  // Owned-product lookup so "Add" buttons stay in "Added" state even after
  // remount (essentialAddStates is in-memory only) and so two near-duplicate
  // cards for the same product can't both be added.
  const ownedEssentialKeys = useMemo(() => {
    const images = new Set<string>();
    const names = new Set<string>();
    for (const w of items ?? []) {
      const imageKey = normalizeCatalogImageKey((w as any)?.imageUrl);
      if (imageKey) images.add(imageKey);
      const nameKey = normalizeCatalogNameKey(`${(w as any)?.brand ?? ''} ${(w as any)?.name ?? ''}`);
      if (nameKey) names.add(nameKey);
    }
    return { images, names };
  }, [items]);

  // Shop items filtered by category directly from real catalog
  // Filter strictly using name keywords to prevent mislabeled DB entries from being misplaced
  // (isBottomName, isShortsName, isTopName, isOuterwearName, isShoeName are module-level constants)

  const shopTops = useMemo(() => catalogEssentials.filter(item =>
    item.garmentType === 'upper_body' && !isBottomName(item.name) && !isShoeName(item.name)
  ), [catalogEssentials]);
  const shopBottoms = useMemo(() => catalogEssentials.filter(item =>
    item.garmentType === 'lower_body' && !isTopName(item.name) && !isOuterwearName(item.name) && !isShoeName(item.name) && !isShortsName(item.name)
  ), [catalogEssentials]);
  const shopShoes = useMemo(() =>
    catalogEssentials.filter(item =>
      item.garmentType === 'shoes' || isShoeName(item.name)
    )
  , [catalogEssentials]);

  const handleAddToWardrobe = useCallback(async (item: ShopCatalogItem) => {
    if (essentialAddStates[item.id]) return;
    // Idempotency: don't insert the same catalog product twice when the grid
    // contains near-duplicates or the user double-taps after remount (the
    // in-memory essentialAddStates map is lost on remount/day change).
    // Match by normalized image URL first, then by brand+name.
    try {
      const existing = useWardrobeStore.getState().items ?? [];
      const incomingImageKey = normalizeCatalogImageKey((item as any).imageUrl);
      const incomingNameKey = normalizeCatalogNameKey(`${(item as any).brand ?? ''} ${(item as any).name ?? ''}`);
      const alreadyOwned = existing.some((w: any) => {
        const wImageKey = normalizeCatalogImageKey(w?.imageUrl);
        if (incomingImageKey && wImageKey && wImageKey === incomingImageKey) return true;
        const wNameKey = normalizeCatalogNameKey(`${w?.brand ?? ''} ${w?.name ?? ''}`);
        if (incomingNameKey && wNameKey && wNameKey === incomingNameKey) return true;
        return false;
      });
      if (alreadyOwned) {
        setEssentialAddStates(prev => ({ ...prev, [item.id]: 'added' }));
        return;
      }
    } catch { /* non-fatal — fall through to normal add */ }
    logger.info('Adding catalog essential to wardrobe', item.name);
    const donePerf = perfAction(`addToWardrobe:${item.name}`);

    setEssentialAddStates(prev => ({ ...prev, [item.id]: 'adding' }));

    const imageUrl = typeof item.imageUrl === 'string' ? item.imageUrl : '';

    const subCategory =
      item.garmentType === 'upper_body'
        ? (classifyUpperBodyItem(item) ?? 'tees')
        : item.garmentType === 'lower_body'
          ? 'pants'
          : item.garmentType;

    try {
      await addItem({
        userId: '',
        imageUrl,
        category: garmentTypeToCategory(item.garmentType),
        subCategory,
        primaryColor: '',
        colorHex: '',
        pattern: 'solid',
        material: '',
        brand: item.brand,
        name: item.name,
        seasons: [],
        occasions: [],
      });
      donePerf();
      setEssentialAddStates(prev => ({ ...prev, [item.id]: 'added' }));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      logger.info('Catalog essential added to wardrobe', item.name);
    } catch (err) {
      donePerf();
      logger.error('Failed to add item, reverting', err);
      setEssentialAddStates(prev => {
        const next = { ...prev };
        delete next[item.id];
        return next;
      });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      Alert.alert(t('home.addEssentialFailed'), t('home.addEssentialFailedMessage'));
    }
  }, [addItem, essentialAddStates, t]);

  // ============================================
  // RENDER COMPONENTS
  // ============================================

  // Weather widget with Liquid Glass - Minimalist.
  // No loading visuals by design: the widget stays invisible until weather
  // data arrives, and stays invisible if it never does.
  const renderWeatherWidget = useCallback(() => {
    if (!Config.weather.apiKey) return null;
    if (loadingWeather) return null;
    if (!weather) return null;

    return (
      <Animated.View entering={isReducedMotionEnabled ? undefined : FadeInDown.delay(100).duration(400)}>
        <View style={styles.weatherWidget}>
          <View style={styles.weatherContent}>
            <ExpoImage
              source={{ uri: `https://openweathermap.org/img/wn/${weather.icon}@2x.png` }}
              style={styles.weatherIcon as any}
              contentFit="contain"
              cachePolicy="memory-disk"
              accessibilityLabel={`Weather icon: ${weather.description}`}
            />
            <View style={styles.weatherInfo}>
              <ScaledText style={styles.weatherTemp}>{weather.temp}°C</ScaledText>
              <ScaledText style={styles.weatherDesc}>{weather.description}</ScaledText>
            </View>
            <View style={styles.weatherSuggestion}>
              <Ionicons name="shirt-outline" size={16} color={colors.text.secondary} />
              <ScaledText style={styles.suggestionText}>
                {weather.temp > 25 ? t('home.wearLight') :
                  weather.temp > 15 ? t('home.useLayers') : t('home.dressWarm')}
              </ScaledText>
            </View>
          </View>
        </View>
      </Animated.View>
    );
  }, [loadingWeather, weather, isReducedMotionEnabled, t]);

  const { requireFeature: requireCreateOutfit } = useSubscriptionGate();

  // Today's Look - Hero card
  const renderTodaysLook = useCallback(() => {
    return (
      <Animated.View
        entering={isReducedMotionEnabled ? undefined : FadeInDown.delay(200).duration(500)}
        style={styles.heroSection}
      >
        <LiquidGlassCard
          variant="opaque"
          elevation="floating"
          style={styles.heroCard}
          contentStyle={styles.heroContent}
        >
          <VideoView
            style={StyleSheet.absoluteFill}
            player={player}
            allowsPictureInPicture={false}
            contentFit="cover"
            pointerEvents="none"
          />
          {/* Overlay with info */}
          <LinearGradient
            colors={['transparent', 'rgba(0,0,0,0.8)']}
            style={styles.heroOverlay}
          >
            <View style={styles.heroInfo}>
              <ScaledText style={styles.heroTitle}>{t('home.todaysLook')}</ScaledText>
            </View>
          </LinearGradient>
        </LiquidGlassCard>

        <TouchableOpacity
          style={styles.createOutfitButton}
          onPress={() => {
            // Post-trial: Create Outfit is Pro-only (free aiOutfits = 0).
            // requireFeature auto-navigates to Paywall when locked.
            if (!requireCreateOutfit('aiOutfits')) return;
            logger.debug('Navigating to AI outfit maker with wardrobe source');
            navigation.navigate('AIOutfit', { source: 'wardrobe' });
          }}
          accessibilityLabel={t('home.createOutfitFromWardrobe')}
          accessibilityRole="button"
        >
          <ScaledText style={styles.createOutfitText}>{t('home.createOutfit')}</ScaledText>
        </TouchableOpacity>
      </Animated.View>
    );
  }, [items, wearLogs, todaysOutfit, isReducedMotionEnabled, t, navigation, requireCreateOutfit]);

  // ── AI outfit → tiles (2-per-row grid) ──────────────────────────────────
  // Cold / rainy / windy season: 4 tiles (outerwear + top + bottom + shoes)
  // Warm season: 3 tiles (top + bottom + shoes). Shoes always present.
  const needsOuterwear = React.useMemo(() => {
    if (!weather) return true;
    const t = weather.temp;
    const c = (weather.description || '').toLowerCase();
    if (t < 18) return true;
    return /\b(rain|drizzle|shower|storm|wind|gust|breezy|snow|sleet|hail)\b/.test(c);
  }, [weather]);

  // Wardrobe Essentials — one cohesive, curated closet builder.
  const renderEssentials = useCallback(() => {
    const showSkeleton = essentialsLoading && essentialsItems.length === 0;
    const showEmpty = !essentialsLoading && essentialsItems.length === 0;
    const addedCount = Object.values(essentialAddStates).filter((state) => state === 'added').length;

    return (
      <View style={styles.essentialsSection}>
        <View style={styles.essentialsHeader}>
          <View style={styles.essentialsHeadingCopy}>
            <ScaledText style={styles.essentialsEyebrow}>{t('home.essentialsCurated')}</ScaledText>
            <ScaledText style={styles.essentialsTitle} accessibilityRole="header">{t('home.wardrobeEssentials')}</ScaledText>
            <ScaledText style={styles.essentialsSubtitle}>{t('home.essentialsSubtitle')}</ScaledText>
          </View>
          {addedCount > 0 ? (
            <View style={styles.essentialsAddedCount}>
              <Ionicons name="checkmark" size={13} color="#FFFFFF" />
              <ScaledText style={styles.essentialsAddedCountText}>
                {t('home.essentialsAddedCount', { count: addedCount })}
              </ScaledText>
            </View>
          ) : null}
        </View>

        {showSkeleton ? (
          <View style={styles.essentialsLoadingBlock}>
            <ActivityIndicator size="small" color={colors.accent.primary} />
            <ScaledText style={styles.essentialsLoadingText}>{t('home.loadingEssentials')}</ScaledText>
          </View>
        ) : showEmpty ? (
          <View style={styles.essentialsEmptyBlock}>
            <Ionicons
              name={essentialsError ? 'alert-circle-outline' : 'shirt-outline'}
              size={22}
              color={colors.text.tertiary}
            />
            <ScaledText style={styles.essentialsEmptyText}>
              {essentialsError
                ? 'Could not load essentials. Pull to refresh.'
                : 'No essentials available yet.'}
            </ScaledText>
          </View>
        ) : (
          <View style={styles.gridContainer}>
            {essentialsItems.map((item) => {
              const addState = essentialAddStates[item.id];
              const isAdding = addState === 'adding';
              // Treat wardrobe-owned products as added even after remount.
              const imageKey = normalizeCatalogImageKey((item as any).imageUrl);
              const nameKey = normalizeCatalogNameKey(`${(item as any).brand ?? ''} ${(item as any).name ?? ''}`);
              const isOwned = (imageKey && ownedEssentialKeys.images.has(imageKey)) ||
                (nameKey && ownedEssentialKeys.names.has(nameKey));
              const isAdded = addState === 'added' || isOwned;

              return (
                <LiquidGlassCard
                  key={item.id}
                  style={styles.gridItem}
                  contentStyle={styles.gridItemContent}
                  variant="light"
                  radius={16}
                >
                  <View style={styles.gridImageWrapper}>
                    <CachedImage uri={typeof item.imageUrl === 'string' ? item.imageUrl : ''} style={styles.gridImage} contentFit="contain" fadeIn={false} />
                    <View style={styles.gridImageOverlay} pointerEvents="none">
                      {isAdded ? (
                        <View style={styles.itemAddedBadge}>
                          <Ionicons name="checkmark" size={14} color="#FFFFFF" />
                        </View>
                      ) : null}
                    </View>
                  </View>
                  <View style={styles.gridItemInfo}>
                    {item.brand ? (
                      <ScaledText style={styles.essentialItemBrand} numberOfLines={1}>{item.brand}</ScaledText>
                    ) : null}
                    <ScaledText style={styles.essentialItemName} numberOfLines={2}>{item.name}</ScaledText>
                  </View>
                  <TouchableOpacity
                    style={[styles.addButton, isAdded && styles.addedButton]}
                    onPress={() => {
                      if (!isAdded && !isAdding) {
                        logger.debug('Add button pressed', item.name);
                        handleAddToWardrobe(item);
                      }
                    }}
                    disabled={isAdded || isAdding}
                    accessibilityLabel={
                      isAdded
                        ? t('home.itemAddedToWardrobe', { itemName: item.name })
                        : t('home.addItemToWardrobe', { itemName: item.name })
                    }
                    accessibilityRole="button"
                    accessibilityState={{ disabled: isAdded || isAdding, busy: isAdding }}
                  >
                    {isAdding ? (
                      <ActivityIndicator size="small" color={colors.text.primary} />
                    ) : (
                      <Ionicons
                        name={isAdded ? 'checkmark' : 'add'}
                        size={17}
                        color={isAdded ? '#FFF' : colors.text.primary}
                      />
                    )}
                    <ScaledText style={[styles.addButtonText, isAdded && styles.addedButtonText]}>
                      {isAdding ? t('home.addingToWardrobe') : isAdded ? t('common.added') : t('home.addToCloset')}
                    </ScaledText>
                  </TouchableOpacity>
                </LiquidGlassCard>
              );
            })}
          </View>
        )}
      </View>
    );
  }, [essentialAddStates, essentialsLoading, essentialsError, essentialsItems, handleAddToWardrobe, ownedEssentialKeys, t]);







  // Weekly Planner - Horizontal Row
  const renderWeeklyPlanner = useCallback(() => {
    if (items.length < 8) return null;

    const days = [];
    const now = new Date();
    const startOfWeek = new Date(now);
    startOfWeek.setDate(now.getDate() - now.getDay());

    for (let i = 0; i < 7; i++) {
      const d = new Date(startOfWeek);
      d.setDate(startOfWeek.getDate() + i);
      days.push({
        name: ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][d.getDay()],
        date: d.getDate(),
        isToday: d.toDateString() === now.toDateString(),
      });
    }

    return (
      <Animated.View
        entering={isReducedMotionEnabled ? undefined : FadeInDown.delay(180).duration(400)}
        style={styles.plannerSection}
      >
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.plannerScroll}>
          {days.map((day, idx) => (
            <View key={idx} style={[styles.plannerDayItem, day.isToday && styles.plannerDayToday]}>
              <ScaledText style={styles.plannerDayName}>{day.name}</ScaledText>
              <View style={[styles.plannerDateCircle, day.isToday && styles.plannerDateCircleToday]}>
                <ScaledText style={[styles.plannerDateText, day.isToday && styles.plannerDateTextToday]}>{day.date}</ScaledText>
              </View>
              <View style={styles.plannerOutfitContainer}>
                {/* Outfits: Top, Pants, Shoes */}
                <MaterialCommunityIcons
                  name={"tshirt-crew" as any}
                  size={14}
                  color={day.isToday ? colors.text.primary : colors.text.tertiary}
                  style={{ opacity: 0.3 }}
                />
                <MaterialCommunityIcons
                  name={"hanger" as any}
                  size={14}
                  color={day.isToday ? colors.text.primary : colors.text.tertiary}
                  style={{ opacity: 0.3, marginTop: -2 }}
                />
                <MaterialCommunityIcons
                  name={"shoe-sneaker" as any}
                  size={14}
                  color={day.isToday ? colors.text.primary : colors.text.tertiary}
                  style={{ opacity: 0.3, marginTop: -2 }}
                />
              </View>
              {day.isToday && <View style={styles.plannerIndicator} />}
            </View>
          ))}
        </ScrollView>
      </Animated.View>
    );
  }, [items.length, isReducedMotionEnabled]);

  // ============================================
  // MAIN RENDER
  // ============================================

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
        style={StyleSheet.absoluteFill}
        pointerEvents="none"
      />
      <View pointerEvents="none" style={styles.backgroundOrbTop} />
      <View pointerEvents="none" style={styles.backgroundOrbBottom} />
      <SafeAreaView style={styles.safeArea}>
        <ScrollView
          style={styles.scrollView}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.scrollContent}
        >
          {/* Header */}
          <View style={styles.headerSection}>
            <ScaledText style={styles.appTitleText} accessibilityRole="header">{t('home.aiWardrobe')}</ScaledText>
          </View>

          {/* Trial Countdown Banner — visible only during active 7-day trial */}
          <TrialCountdownBanner />

          {/* Weekly Planner (Swapped position with Greeting) */}
          {renderWeeklyPlanner()}

          {/* Weather Widget */}
          {renderWeatherWidget()}

          {/* Greeting Row (Swapped position with Planner) */}
          <View style={[styles.greetingSection, { marginBottom: spacing.md }]}>
            <ScaledText style={styles.greetingText} numberOfLines={2}>
              {greeting}, {userName}
            </ScaledText>
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: spacing.sm }}>
              <StreakBadge variant="inline" />
              <TouchableOpacity
                style={styles.buzzerButton}
                onPress={() => {
                  logger.debug('[IPAD-DEBUG] Calendar button pressed');
                  navigation.navigate('Calendar');
                }}
                onPressIn={() => logger.debug('[IPAD-DEBUG] Calendar button touch start')}
                onPressOut={() => logger.debug('[IPAD-DEBUG] Calendar button touch end')}
                accessibilityLabel={t('home.openCalendar')}
              >
                <Ionicons name="calendar-outline" size={24} color={colors.text.primary} />
                <View style={styles.buzzerDot} />
              </TouchableOpacity>
            </View>
          </View>




          {/* Today's Look Hero */}
          {renderTodaysLook()}

          {/* Wardrobe Essentials Grid OR Dynamic Occasions */}
          {items.length < 5 ? (
            renderEssentials()
          ) : (
            dynamicOccasions.map((item) => (
              <DailyOutfitSection
                key={item.id}
                occasion={item.occasion}
                style={item.style}
                weather={weather}
                weatherForAI={weatherForAI}
                shopTops={shopTops}
                shopBottoms={shopBottoms}
                shopShoes={shopShoes}
                catalogEssentials={catalogEssentials}
                needsOuterwear={needsOuterwear}
                isReducedMotionEnabled={isReducedMotionEnabled}
                navigation={navigation}
                t={t}
                userId={userId}
                enabled={decksFetchEnabled}
              />
            ))
          )}

          {/* Bottom spacing for tab bar */}
          <View style={{ height: 120 }} />
        </ScrollView>
      </SafeAreaView>

    </View>
  );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
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
  stylistFAB: {
    position: 'absolute',
    bottom: 100,
    left: '50%',
    transform: [{ translateX: -58 }], // half of approximate button width
    borderRadius: 28,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 12,
    elevation: 8,
  },
  stylistFABGlass: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    backgroundColor: 'rgba(255,255,255,0.72)',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  stylistFABText: {
    color: colors.text.primary,
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  safeArea: {
    flex: 1,
  },



  // Scroll
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: 0,
    paddingBottom: 100,
  },
  plannerSection: {
    marginBottom: spacing.md,
  },
  plannerScroll: {
    paddingHorizontal: spacing.screenPadding,
    gap: 16,
  },
  plannerDayItem: {
    alignItems: 'center',
    width: 48,
    position: 'relative',
  },
  plannerDayToday: {
    // scale up or highlight
  },
  plannerDayName: {
    ...typography.scale.labelSmall,
    fontSize: 10,
    color: colors.text.tertiary,
    marginBottom: 4,
  },
  plannerDateCircle: {
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  plannerDateCircleToday: {
    backgroundColor: '#F5F5F5', // Light gray highlight like second photo
  },
  plannerDateText: {
    ...typography.scale.bodySmall,
    fontSize: 14,
    fontWeight: '700',
    color: colors.text.primary,
  },
  plannerDateTextToday: {
    color: colors.text.primary,
  },
  plannerOutfitContainer: {
    marginTop: 4,
    height: 48, // Increased for triple icons
    justifyContent: 'center',
    alignItems: 'center',
  },
  plannerIndicator: {
    position: 'absolute',
    bottom: -12,
    height: 3,
    width: '100%',
    backgroundColor: colors.text.primary,
    borderRadius: 1.5,
  },

  // Premium Outfit Section
  premiumSection: {
    marginBottom: spacing.xxl,
  },
  premiumHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: spacing.md,
  },
  premiumHeaderTitle: {
    ...typography.scale.titleLarge,
    color: colors.text.primary,
    fontWeight: '700',
  },
  premiumHeaderSubtitle: {
    ...typography.scale.bodyMedium,
    color: colors.text.tertiary,
  },
  premiumCard: {
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.06)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.06,
    shadowRadius: 18,
    elevation: 4,
  },
  premiumCardContent: {
    padding: spacing.md,
    alignItems: 'center',
  },
  outfitGridMain: {
    flexDirection: 'row',
    width: '100%',
    height: 300,
    gap: spacing.md,
  },
  outfitGridLeft: {
    flex: 1.2,
    gap: spacing.sm,
  },
  outfitGridRight: {
    flex: 1,
    gap: spacing.sm,
  },
  outfitLargeImage: {
    flex: 1,
    width: '100%',
  },
  outfitSmallBox: {
    flex: 1,
    backgroundColor: '#FBFCFF',
    borderRadius: radius.md,
    padding: spacing.xs,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.06)',
  },
  outfitSmallImage: {
    width: '100%',
    height: '100%',
  },
  premiumSuggestionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: spacing.md,
    opacity: 0.6,
  },
  premiumSuggestionText: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
  },
  premiumFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: spacing.sm,
    paddingHorizontal: spacing.screenPadding,
  },
  premiumActionIcons: {
    flexDirection: 'row',
    gap: 8,
  },
  actionIconButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.84)',
    borderWidth: 1,
    borderColor: 'rgba(23,58,101,0.09)',
  },
  swipeHintRow: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'center',
    gap: 7,
    marginTop: spacing.md,
  },
  swipeHintText: {
    ...typography.scale.labelSmall,
    color: colors.text.tertiary,
    textTransform: 'none',
    letterSpacing: 0,
  },
  createAvatarButton: {
    backgroundColor: '#173A65',
    minHeight: 46,
    justifyContent: 'center',
    paddingHorizontal: spacing.lg + 2,
    paddingVertical: spacing.sm + 2,
    borderRadius: 23,
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 4,
  },
  createAvatarText: {
    color: '#FFF',
    fontWeight: '700',
    fontSize: 15,
  },
  createAvatarButtonDisabled: {
    opacity: 0.35,
  },
  // Header & Greeting
  headerSection: {
    paddingTop: 0,
    marginBottom: 0,
  },
  appTitleText: {
    ...typography.scale.headlineMedium,
    color: colors.text.primary,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  greetingSection: {
    marginHorizontal: spacing.screenPadding,
    paddingHorizontal: 18,
    paddingVertical: 14,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderRadius: 26,
    backgroundColor: 'rgba(255,255,255,0.88)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 16,
    elevation: 4,
  },
  greetingText: {
    ...typography.scale.titleMedium,
    color: colors.text.secondary,
    fontWeight: '500',
    flex: 1,
  },
  buzzerButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.92)',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.06,
    shadowRadius: 10,
    elevation: 3,
  },
  buzzerDot: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#0A1931', // Using dark blue for active state
    borderWidth: 1.5,
    borderColor: colors.background.secondary,
  },

  // Weather
  weatherWidget: {
    marginHorizontal: spacing.screenPadding,
    marginBottom: spacing.xxl, // Increased again per request
    borderRadius: 24,
    backgroundColor: 'rgba(255,255,255,0.88)',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.06,
    shadowRadius: 16,
    elevation: 4,
    padding: 14,
  },
  weatherContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 0, // Removed padding
  },
  weatherIcon: {
    width: 56,
    height: 56,
  },
  weatherInfo: {
    marginLeft: spacing.sm,
    flex: 1,
  },
  weatherTemp: {
    ...typography.scale.headlineMedium,
    color: colors.text.primary,
    fontWeight: '700',
  },
  weatherDesc: {
    ...typography.scale.bodyMedium,
    color: colors.text.secondary,
    textTransform: 'capitalize',
  },

  weatherSuggestion: {
    backgroundColor: colors.accent.primary + '15',
    paddingHorizontal: spacing.sm + 4,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
  },

  suggestionText: {
    ...typography.scale.bodySmall,
    color: colors.text.primary,
    fontWeight: '600',
  },

  // Hero Card
  heroSection: {
    paddingHorizontal: spacing.screenPadding,
    marginBottom: spacing.xxl, // Increased again per request
  },
  heroCard: {
    height: 360, // Increased from 320 (Little bigger)
    overflow: 'hidden',
    borderRadius: 32,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.72)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 8,
  },
  heroContent: {
    padding: 0,
    flex: 1,
  },
  heroVideo: {
    flex: 1,
    width: '100%',
  },
  heroOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    padding: spacing.md,
    paddingTop: spacing.xxl,
  },
  heroInfo: {},
  heroTitle: {
    ...typography.scale.titleLarge,
    color: '#FFF',
    fontWeight: '700',
  },

  heroButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    gap: spacing.xs,
  },

  createOutfitButton: {
    backgroundColor: '#173A65',
    paddingVertical: 15,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.md,
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 5,
  },
  createOutfitText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },

  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
  },
  demoOutfitContainer: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  demoOutfitGrid: {
    flex: 1,
    flexDirection: 'row',
  },
  demoOutfitImage1: {
    flex: 1,
    height: '100%',
    resizeMode: 'cover',
  },
  demoOutfitImage2: {
    flex: 1,
    height: '100%',
    resizeMode: 'cover',
  },
  realOutfitContainer: {
    flex: 1,
    padding: spacing.sm,
    justifyContent: 'center',
  },
  realOutfitGrid: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  realOutfitItem: {
    width: '45%',
    aspectRatio: 1,
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderRadius: radius.md,
    overflow: 'hidden',
  },
  realOutfitImage: {
    width: '100%',
    height: '100%',
  },
  placeholderIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.glass.frosted,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  placeholderText: {
    ...typography.scale.bodyMedium,
    color: colors.text.secondary,
  },

  // Stats


  // Quick Actions
  quickActionsSection: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    ...typography.scale.titleLarge,
    color: colors.text.primary,
    fontWeight: '600',
    paddingHorizontal: spacing.screenPadding,
    marginBottom: spacing.md,
  },
  actionCard: {
    flex: 1,
    height: '100%',
  },
  actionCardContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.md,
  },
  actionIconContainer: {
    width: 48,
    height: 48,
    borderRadius: radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  actionTitle: {
    ...typography.scale.titleSmall,
    color: colors.text.primary,
    fontWeight: '600',
    marginBottom: spacing.xs,
  },

  // Essentials Grid
  essentialsSection: {
    paddingHorizontal: spacing.screenPadding,
    marginBottom: spacing.xxl,
  },
  essentialsHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  essentialsHeadingCopy: {
    flex: 1,
    paddingRight: spacing.md,
  },
  essentialsEyebrow: {
    ...typography.scale.labelSmall,
    color: '#315C8C',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
    marginBottom: 5,
  },
  essentialsTitle: {
    ...typography.scale.titleLarge,
    color: colors.text.primary,
    fontWeight: '700',
    letterSpacing: -0.4,
  },
  essentialsSubtitle: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
    lineHeight: 18,
    marginTop: 4,
    maxWidth: 290,
  },
  essentialsAddedCount: {
    minWidth: 38,
    minHeight: 30,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: radius.pill,
    backgroundColor: '#173A65',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 3,
    marginTop: 18,
  },
  essentialsAddedCountText: {
    ...typography.scale.labelSmall,
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 10,
  },
  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
  gridItem: {
    width: (SCREEN_WIDTH - (spacing.screenPadding * 2) - spacing.md) / 2,
    aspectRatio: 0.68,
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderColor: 'rgba(23,58,101,0.08)',
  },
  gridItemContent: {
    padding: 0,
    overflow: 'hidden',
  },
  gridImageWrapper: {
    width: '100%',
    height: '60%',
    overflow: 'hidden',
    backgroundColor: '#F4F1EC',
  },
  gridImage: {
    width: '100%',
    height: '100%',
  },
  gridImageOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    padding: spacing.sm,
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'flex-end',
  },
  itemAddedBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#173A65',
    alignItems: 'center',
    justifyContent: 'center',
  },
  gridItemInfo: {
    flex: 1,
    paddingHorizontal: spacing.sm,
    paddingTop: 9,
    paddingBottom: 4,
  },
  essentialItemName: {
    ...typography.scale.bodySmall,
    color: colors.text.primary,
    fontWeight: '600',
    fontSize: 13,
    lineHeight: 17,
    marginTop: 3,
  },
  essentialItemBrand: {
    ...typography.scale.labelSmall,
    color: '#6C7A8D',
    fontSize: 9,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 36,
    backgroundColor: '#F1F5FA',
    borderWidth: 1,
    borderColor: 'rgba(23,58,101,0.08)',
    paddingVertical: 8,
    paddingHorizontal: spacing.sm,
    borderRadius: radius.pill,
    gap: 4,
    marginHorizontal: spacing.sm,
    marginBottom: spacing.sm,
    marginTop: 0,
  },
  addedButton: {
    backgroundColor: '#173A65',
    borderColor: '#173A65',
  },
  addButtonText: {
    ...typography.scale.labelSmall,
    color: colors.text.primary,
    fontWeight: '700',
    fontSize: 10,
    letterSpacing: 0.25,
  },
  addedButtonText: {
    color: '#FFF',
  },
  essentialsLoadingBlock: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xl,
    gap: spacing.sm,
  },
  essentialsLoadingText: {
    ...typography.scale.bodySmall,
    color: colors.text.tertiary,
  },
  essentialsEmptyBlock: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xl,
    gap: spacing.sm,
  },
  essentialsEmptyText: {
    ...typography.scale.bodySmall,
    color: colors.text.tertiary,
    textAlign: 'center',
    paddingHorizontal: spacing.screenPadding,
  },


  // Scan CTA
  scanCTASection: {
    paddingHorizontal: spacing.screenPadding,
    marginBottom: spacing.lg,
  },
  scanButton: {
    marginTop: spacing.xs,
  },
});

export default HomeScreen;
