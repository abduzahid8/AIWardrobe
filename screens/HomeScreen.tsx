/**
 * HomeScreen - 2026 Redesign
 * Features: Bento Grid layout, Liquid Glass aesthetics, GenUI-ready components
 * Based on 2026 Digital Experience Report guidelines
 */

import React, { useState, useEffect, useMemo, useRef, useCallback } from "react";
import {
  View,
  Text,
  ScrollView,
  Image,
  Dimensions,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  FlatList,
  Alert,
  InteractionManager,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { useIsFocused, useFocusEffect } from "@react-navigation/native";
import { useAppNavigation } from '../hooks/useAppNavigation';
import AsyncStorage from "@react-native-async-storage/async-storage";
import useAuthStore from '../store/auth';
import { LinearGradient } from "expo-linear-gradient";
import { useVideoPlayer, VideoView } from 'expo-video';
import * as Location from 'expo-location';
import Animated, {
  FadeInDown,
} from 'react-native-reanimated';

// 2026 Design System
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import {
  LiquidGlassCard,
  FrostedGlassCard,
} from '../components/ui';
import { useAccessibility } from '../hooks/useAccessibility';
import Config from '../src/config/env';

// Core loop components

import StreakBadge from '../components/StreakBadge';
import useWardrobeStore from '../store/wardrobeStore';
import { useShallow } from 'zustand/shallow';
import useSubscriptionStore from '../store/subscriptionStore';
import useAppContextStore from '../src/store/contextStore';
import { CachedImage } from '../components/ui/CachedImage';
import type { ContextualPrompt } from '../src/services/contextualPromptService';
import {
  getContextualPrompt,
  markPromptShown,
  dismissPrompt,
} from '../src/services/contextualPromptService';

import { Swipeable } from 'react-native-gesture-handler';
import { BlurView } from 'expo-blur';

import { supabase } from '../lib/supabase';
import { quickSuggest } from '../src/services/suggestionEngine';
import { useDailyAIOutfit } from '../hooks/useDailyAIOutfit';
import { useShopCatalog } from '../hooks/useShopCatalog';
import useShopCatalogStore from '../store/shopCatalogStore';
import OutfitCollageDisplay from '../features/outfit-generator/components/OutfitCollageDisplay';
import TrialCountdownBanner from '../components/TrialCountdownBanner';
import type { ShopCatalogItem } from '../features/try-on/types';
import type { ClothingCategory } from '../src/types/domain';
import { createLogger } from '../src/utils/logger';
import { useTranslation } from 'react-i18next';
import { useAdminGuard } from '../hooks/useAdminGuard';
import { perfMark, perfMeasure, perfAction, perfScreenReady } from '../src/utils/perf';

type EssentialSlot = 'outerwear' | 'shirts' | 'knitwear' | 'tees' | 'bottoms' | 'shoes';

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

const selectEssentialShoppingMix = (items: ShopCatalogItem[], dailySeed: string): ShopCatalogItem[] => {
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

  // Shuffle items for daily variety using a stable seed so the result is
  // identical across all renders within the same calendar day.
  const rand = seededRandom(hashStrToInt(dailySeed));
  const shuffled = [...items].sort(() => rand() - 0.5);

  for (const item of shuffled) {
    if (selectedIds.has(item.id)) continue;

    let slot: EssentialSlot | null = null;

    if (item.garmentType === 'upper_body') {
      slot = classifyUpperBodyItem(item);
    } else if (item.garmentType === 'lower_body') {
      slot = classifyBottomItem(item);
    } else if (item.garmentType === 'shoes') {
      slot = classifyShoeItem(item);
    }

    if (!slot) continue;
    if (pickedCounts[slot] >= ESSENTIALS_REQUIREMENTS[slot]) continue;

    selected.push(item);
    selectedIds.add(item.id);
    pickedCounts[slot] += 1;

    if (selected.length >= ESSENTIALS_LIMIT) break;
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
const selectIsPremium = (state: any) => state.isPremium;
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
}

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
}: DailyOutfitSectionProps) => {
  const [currentOutfitIndex, setCurrentOutfitIndex] = useState(0);
  const outfitFlatListRef = useRef<FlatList>(null);

  // Load the daily outfits for this style and occasion
  const dailyAI = useDailyAIOutfit({
    style,
    occasion,
    weather: weatherForAI,
    variants: 3,
  });

  const isOldMoney = style === 'old_money';
  const cardStyle = isOldMoney ? styles.dinnerCard : styles.premiumCard;
  const suggestionTextStyle = isOldMoney ? styles.dinnerSuggestionText : styles.premiumSuggestionText;
  const tryOnButtonStyle = isOldMoney ? styles.dinnerAvatarButton : styles.createAvatarButton;
  const pagerBarActiveStyle = isOldMoney ? styles.dinnerPagerBarActive : styles.pagerBarActive;
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
      items.push({ id: `legacy_outer_${c.id}`, image: c.outerLayer.imageUrl || c.outerLayer.image, type: c.outerLayer.type || c.outerLayer.name || 'Outerwear', name: c.outerLayer.name || 'Outerwear', macroCategory: 'outerwear' });
      filledSlots.add('outerwear');
    }
    if (c?.mainTop) {
      const topName = c.mainTop.name || 'Top';
      const forcedTopType = isTopName(topName) && !isOuterwearName(topName) ? topName : `${topName} Shirt`;
      items.push({ id: `legacy_top_${c.id}`, image: c.mainTop.imageUrl || c.mainTop.image, type: forcedTopType, name: topName, macroCategory: 'top' });
      filledSlots.add('top');
    }
    if (c?.mainBottom) {
      const btmName = c.mainBottom.name || 'Pants';
      const forcedBtmType = isBottomName(btmName) ? btmName : `${btmName} Pants`;
      items.push({ id: `legacy_btm_${c.id}`, image: c.mainBottom.imageUrl || c.mainBottom.image, type: forcedBtmType, name: btmName, macroCategory: 'bottom' });
      filledSlots.add('bottom');
    }
    if (c?.shoes && shopByMacro.shoes.length > 0) {
      const shoeName = c.shoes.name || 'Shoes';
      const forcedShoeType = isShoeName(shoeName) ? shoeName : `${shoeName} Shoe`;
      items.push({ id: `legacy_shoe_${c.id}`, image: c.shoes.imageUrl || c.shoes.image, type: forcedShoeType, name: shoeName, macroCategory: 'shoes' });
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
          });
        }
      }
    });

    return items;
  }, [catalogEssentials, shopTops, shopBottoms, shopShoes, needsOuterwear]);

  const mapAiOutfitItemsForCollage = useCallback((outfit: any) => {
    const shopByMacro: Record<string, ShopCatalogItem[]> = {
      top: shopTops.filter(t => !isOuterwear(t.name)),
      bottom: shopBottoms,
      shoes: shopShoes,
      outerwear: shopTops.filter(t => isOuterwear(t.name)),
    };

    const filledSlots = new Set<string>();
    const mapped = (outfit.items || []).map((item: any, index: number) => {
      let img = item.imageUrl || item.image_url || item.image || '';
      let finalItem = { ...item };

      const macro = (item.macroCategory || '').toLowerCase();
      const macroNormalized = macro === 'upper_body' ? 'top' : macro === 'lower_body' ? 'bottom' : macro;
      const shopItems = shopByMacro[macroNormalized] || [];

      if (shopItems.length > 0) {
        const shopItem = shopItems[Math.floor(Math.random() * shopItems.length)];
        finalItem = {
          ...item,
          id: shopItem.id,
          imageUrl: shopItem.imageUrl,
          image: shopItem.imageUrl,
          name: shopItem.name,
          type: shopItem.name,
          macroCategory: macroNormalized,
        };
        img = shopItem.imageUrl;
      }

      if (finalItem.macroCategory) {
        filledSlots.add(finalItem.macroCategory.toLowerCase());
      }

      return {
        id: finalItem.id || `home_ai_${index}`,
        image: img,
        type: finalItem.type || finalItem.name || 'Item',
        name: finalItem.name || finalItem.type || 'Item',
        macroCategory: finalItem.macroCategory,
      };
    });

    const mandatorySlots = ['top', 'bottom', 'shoes'];
    mandatorySlots.forEach(slot => {
      if (!filledSlots.has(slot)) {
        const fallbackItems = shopByMacro[slot] || [];
        if (fallbackItems.length > 0) {
          const shopItem = fallbackItems[Math.floor(Math.random() * fallbackItems.length)];
          mapped.push({
            id: shopItem.id,
            image: shopItem.imageUrl,
            type: shopItem.name,
            name: shopItem.name,
            macroCategory: slot,
          });
        }
      }
    });

    return mapped;
  }, [shopTops, shopBottoms, shopShoes]);

  const collageItemsList = useMemo(() => {
    return dailyAI.outfits.map(o => mapAiOutfitItemsForCollage(o));
  }, [dailyAI.outfits, mapAiOutfitItemsForCollage]);

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
          <Text style={styles.premiumHeaderTitle}>{titleText}</Text>
          <Text style={isOldMoney ? styles.dinnerHeaderSubtitle : styles.premiumHeaderSubtitle}>{subtitleText}</Text>
        </View>
        <View style={{ paddingHorizontal: spacing.screenPadding }}>
          {isOldMoney ? (
            <View style={[styles.dinnerCard, { minHeight: 340, alignItems: 'center', justifyContent: 'center' }]}>
              <ActivityIndicator size="small" color={colors.accent.primary} />
              <Text style={[styles.dinnerSuggestionText, { marginTop: spacing.sm }]}>
                Styling tonight&apos;s looks…
              </Text>
            </View>
          ) : (
            <LiquidGlassCard
              variant="light"
              style={styles.premiumCard}
              contentStyle={[styles.premiumCardContent, { minHeight: 340, justifyContent: 'center' }]}
            >
              <ActivityIndicator size="small" color={colors.accent.primary} />
              <Text style={[styles.premiumSuggestionText, { marginTop: spacing.sm }]}>
                Styling today&apos;s looks…
              </Text>
            </LiquidGlassCard>
          )}
        </View>
      </View>
    );
  }

  const aiOutfits = dailyAI.outfits;
  const useAI = aiOutfits.length > 0;

  const data = useAI
    ? aiOutfits.map((o, i) => ({ id: o.id || `ai-${style}-${i}`, outfit: o, index: i }))
    : curatedCombinations.map((c) => ({ id: String(c.id), legacy: c, index: 0 }));

  const renderOutfitItem = ({ item }: { item: { id: string; outfit?: any; legacy?: any; index: number } }) => {
    const collageItems = item.outfit
      ? (collageItemsList[item.index] ?? mapAiOutfitItemsForCollage(item.outfit))
      : mapLegacyOutfitItemsForCollage(item.legacy) as any;
    const hasOuter = collageItems.some((ci: any) => ci.macroCategory === 'outerwear');

    if (isOldMoney) {
      return (
        <View style={{ width: SCREEN_WIDTH, paddingHorizontal: spacing.screenPadding }}>
          <View style={styles.dinnerCard}>
            <OutfitCollageDisplay
              items={collageItems}
              height={300}
              needsOuterwear={needsOuterwear && hasOuter}
            />

            <View style={styles.premiumSuggestionInfo}>
              <Text style={styles.dinnerSuggestionText}>
                {`${collageItems.length} shop items suggested`}
              </Text>
              <Ionicons name="bag-outline" size={16} color="rgba(255,255,255,0.4)" />
            </View>

            <View style={styles.premiumPager}>
              {data.map((_, index) => (
                <View key={index} style={[styles.pagerBar, index === currentOutfitIndex && styles.dinnerPagerBarActive]} />
              ))}
            </View>
          </View>
        </View>
      );
    }

    return (
      <View style={{ width: SCREEN_WIDTH, paddingHorizontal: spacing.screenPadding }}>
        <LiquidGlassCard
          variant="light"
          style={styles.premiumCard}
          contentStyle={styles.premiumCardContent}
        >
          <OutfitCollageDisplay
            items={collageItems}
            height={300}
            needsOuterwear={needsOuterwear && hasOuter}
          />

          <View style={styles.premiumSuggestionInfo}>
            <Text style={styles.premiumSuggestionText}>
              {`${collageItems.length} shop items suggested`}
            </Text>
            <Ionicons name="bag-outline" size={16} color={colors.text.tertiary} />
          </View>

          <View style={styles.premiumPager}>
            {data.map((_, index) => (
              <View key={index} style={[styles.pagerBar, index === currentOutfitIndex && styles.pagerBarActive]} />
            ))}
          </View>
        </LiquidGlassCard>
      </View>
    );
  };

  return (
    <View style={styles.premiumSection}>
      <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
        <Text style={styles.premiumHeaderTitle}>{titleText}</Text>
        <Text style={isOldMoney ? styles.dinnerHeaderSubtitle : styles.premiumHeaderSubtitle}>{subtitleText}</Text>
      </View>

      {data.length === 0 ? (
        <View style={{ paddingHorizontal: spacing.screenPadding }}>
          {isOldMoney ? (
            <View style={[styles.dinnerCard, { minHeight: 300, alignItems: 'center', justifyContent: 'center' }]}>
              <Ionicons name={emptyIconName} size={48} color="rgba(255,255,255,0.6)" />
              <Text style={[suggestionTextStyle, { marginTop: spacing.md, textAlign: 'center' }]}>
                No looks available right now.
              </Text>
              <TouchableOpacity
                style={[styles.createAvatarButton, { marginTop: spacing.md }]}
                onPress={() => {
                  logger.debug(`Regenerate ${style} daily outfits (empty state)`);
                  dailyAI.regenerate();
                }}
                accessibilityLabel={t('home.tryAgain')}
                accessibilityRole="button"
              >
                <Text style={styles.createAvatarText}>{t('home.tryAgain')}</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <LiquidGlassCard
              variant="light"
              style={styles.premiumCard}
              contentStyle={[styles.premiumCardContent, { minHeight: 300, justifyContent: 'center', alignItems: 'center' }]}
            >
              <Ionicons name={emptyIconName} size={48} color={colors.text.tertiary} />
              <Text style={[suggestionTextStyle, { marginTop: spacing.md, textAlign: 'center' }]}>
                No looks available right now.
              </Text>
              <TouchableOpacity
                style={[styles.createAvatarButton, { marginTop: spacing.md }]}
                onPress={() => {
                  logger.debug(`Regenerate ${style} daily outfits (empty state)`);
                  dailyAI.regenerate();
                }}
                accessibilityLabel={t('home.tryAgain')}
                accessibilityRole="button"
              >
                <Text style={styles.createAvatarText}>{t('home.tryAgain')}</Text>
              </TouchableOpacity>
            </LiquidGlassCard>
          )}
        </View>
      ) : (
        <FlatList
          ref={outfitFlatListRef}
          data={data}
          renderItem={renderOutfitItem}
          keyExtractor={(item) => item.id}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          onMomentumScrollEnd={(event) => {
            const index = Math.round(event.nativeEvent.contentOffset.x / SCREEN_WIDTH);
            setCurrentOutfitIndex(index);
          }}
          decelerationRate="fast"
          scrollEventThrottle={16}
        />
      )}

      {/* Action Footer */}
      <View style={styles.premiumFooter}>
        <View style={styles.premiumActionIcons}>
          <TouchableOpacity style={styles.actionIconButton}>
            <Ionicons name="heart-outline" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionIconButton}
            onPress={() => {
              logger.debug(`Regenerate ${style} daily outfits`);
              dailyAI.regenerate();
            }}
            accessibilityLabel={t('home.tryAgain')}
            accessibilityRole="button"
          >
            <Ionicons name="refresh-outline" size={22} color={colors.text.primary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionIconButton}>
            <Ionicons name="thumbs-down-outline" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionIconButton}>
            <Ionicons name="paper-plane-outline" size={22} color={colors.text.primary} />
          </TouchableOpacity>
        </View>
        <TouchableOpacity
          style={tryOnButtonStyle}
          onPress={() => {
            logger.debug('Try On button pressed');
            navigation.navigate('AITryOn');
          }}
        >
          <Text style={styles.createAvatarText}>{t('home.tryOn')}</Text>
        </TouchableOpacity>
      </View>
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
  const { isAdmin } = useAdminGuard();

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

      // Defer video playback to avoid lag during screen transition
      InteractionManager.runAfterInteractions(() => {
        if (!active) return;
        try {
          player.play();
        } catch (error) {
          logger.warn('Unable to play hero video', error);
        }
      });

      return () => {
        active = false;
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
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [greeting, setGreeting] = useState(t('common.goodMorning'));
  const [showHiddenGems, setShowHiddenGems] = useState(true);
  const [currentOutfitIndex, setCurrentOutfitIndex] = useState(0);
  const outfitFlatListRef = useRef<FlatList>(null);
  const [currentDinnerOutfitIndex, setCurrentDinnerOutfitIndex] = useState(0);
  const dinnerOutfitFlatListRef = useRef<FlatList>(null);

  // Wardrobe store data for core loop — useShallow handles snapshot caching
  // correctly in Zustand v5, preventing the React 18 "getSnapshot should be
  // cached" warning that fires when a selector returns a new object each call.
  const { items, wearLogs, streak } = useWardrobeStore(
    useShallow((state) => ({ items: state.items, wearLogs: state.wearLogs, streak: state.streak }))
  );

  // Contextual prompt
  const [activePrompt, setActivePrompt] = useState<ContextualPrompt | null>(null);

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
  const [loadingOccasions, setLoadingOccasions] = useState(true);

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
      } finally {
        if (active) {
          setLoadingOccasions(false);
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

  // Sync context for AI Assistant
  useEffect(() => {
    useAppContextStore.getState().setContext(weather, todaysOutfit);
  }, [weather, todaysOutfit]);

  // Memoize utilization so wearLogs.flatMap isn't called inside the effect on every focus
  const closetUtilization = useMemo(() => {
    if (items.length === 0) return 0;
    const wornIds = new Set(wearLogs.flatMap(l => l.itemIds));
    return Math.round((wornIds.size / items.length) * 100);
  }, [items.length, wearLogs.length]); // length deps are sufficient for utilization

  useEffect(() => {
    if (!isFocused) return;

    getContextualPrompt(items, wearLogs, streak, closetUtilization)
      .then(prompt => setActivePrompt(prompt))
      .catch(() => { });
  }, [isFocused, items.length, wearLogs.length, streak, closetUtilization]);

  // Determine greeting based on time
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting(t('aiHub.goodMorning'));
    else if (hour < 18) setGreeting(t('aiHub.goodAfternoon'));
    else setGreeting(t('aiHub.goodEvening'));
  }, [t]);

  // Read username from Supabase auth store (no JWT decode needed)
  const authUsername = useAuthStore(selectUsername);
  useEffect(() => {
    if (authUsername) {
      setUserName(authUsername);
    }
  }, [authUsername]);

  useEffect(() => {
    const loadSavedVideo = async () => {
      try {
        const savedVideo = await AsyncStorage.getItem('lastWardrobeVideo');
        if (savedVideo) {
          setVideoUri(savedVideo);
        }
      } catch (error) {
        logger.error('Error loading saved video', error);
      }
    };
    loadSavedVideo();
  }, []);

  // Fetch weather in a dedicated useEffect with an AbortController so the
  // GPS and network calls are non-blocking and are cancelled if the component
  // unmounts before they complete (fixes Defects 2.1 and 5.3).
  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    const fetchWeather = async () => {
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

        const { latitude, longitude } = location.coords;

        const response = await fetch(
          `${Config.weather.baseUrl}/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${Config.weather.apiKey}`,
          { signal }
        );
        const data = await response.json();

        if (!signal.aborted && data.main && data.weather) {
          setWeather({
            temp: Math.round(data.main.temp),
            description: data.weather[0].description,
            icon: data.weather[0].icon,
            city: data.name,
          });
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
  const [addedItemIds, setAddedItemIds] = useState<Record<string, boolean>>({});
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

  // Stable primitive keys derived from sorted item IDs.
  // These strings only change when the actual set of items changes, not on
  // every render when useShopCatalog returns a new array reference.
  // Used as memo dependencies for outfitCombinations / dinnerOutfitCombinations
  // so those memos are not invalidated by reference churn (fixes Defect 1.2).
  const catalogEssentialsKey = useMemo(
    () => catalogEssentials.map(i => i.id).sort().join(','),
    [catalogEssentials],
  );
  const shopTopsKey = useMemo(
    () => shopTops.map(i => i.id).sort().join(','),
    [shopTops],
  );
  const shopBottomsKey = useMemo(
    () => shopBottoms.map(i => i.id).sort().join(','),
    [shopBottoms],
  );
  const shopShoesKey = useMemo(
    () => shopShoes.map(i => i.id).sort().join(','),
    [shopShoes],
  );

  // Refs that always hold the latest arrays so the outfit memos can read
  // current data without listing the arrays themselves as dependencies.
  const shopTopsRef    = useRef(shopTops);
  const shopBottomsRef = useRef(shopBottoms);
  const shopShoesRef   = useRef(shopShoes);
  shopTopsRef.current    = shopTops;
  shopBottomsRef.current = shopBottoms;
  shopShoesRef.current   = shopShoes;


  // User's wardrobe categories (lowercase) — used to determine which outfit slots are truly "suggested"
  const userCategories = useMemo(
    () => new Set(items.map(i => (i.category as string).toLowerCase())),
    [items]
  );

  // Night-Time Dinner outfit combinations (elegant classic)
  // Safely index into fetched shop arrays using modulo, with fallbacks to avoid empty states.
  // Dependencies are stable string keys (sorted ID joins) rather than array references so
  // this memo only re-runs when the actual item set changes (fixes Defect 1.2).
  const dinnerOutfitCombinations = useMemo(() => {
    const tops    = shopTopsRef.current;
    const bottoms = shopBottomsRef.current;
    const shoes   = shopShoesRef.current;

    if (!catalogEssentialsKey) return []; // Only fail if catalog is completely empty
    if (tops.length === 0 || bottoms.length === 0) return [];

    // Only use REAL outerwear (jackets, coats, blazers, suits, etc.)
    const isOuterwear = (name: string) => isOuterwearName(name);
    const realOuterwear = tops.filter(t => isOuterwear(t.name));
    const realTops = tops.filter(t => !isOuterwear(t.name));
    const topsPool = realTops.length > 0 ? realTops : tops;

    return [
      {
        id: 1,
        mainTop: topsPool[2 % topsPool.length],
        mainBottom: bottoms[0 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[0 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[1 % shoes.length] : null,
      },
      {
        id: 2,
        mainTop: topsPool[3 % topsPool.length],
        mainBottom: bottoms[3 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[1 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[2 % shoes.length] : null,
      },
      {
        id: 3,
        mainTop: topsPool[1 % topsPool.length],
        mainBottom: bottoms[1 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[0 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[0 % shoes.length] : null,
      },
    ];
  // Stable string keys: only re-run when the actual item IDs change, not on
  // every render when useShopCatalog returns a new array reference.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalogEssentialsKey, shopTopsKey, shopBottomsKey, shopShoesKey]);

  const outfitCombinations = useMemo(() => {
    const tops    = shopTopsRef.current;
    const bottoms = shopBottomsRef.current;
    const shoes   = shopShoesRef.current;

    if (!catalogEssentialsKey) return [];
    if (tops.length === 0 || bottoms.length === 0) return [];

    const isOuterwear = (name: string) => isOuterwearName(name);
    const realOuterwear = tops.filter(t => isOuterwear(t.name));
    const realTops = tops.filter(t => !isOuterwear(t.name));
    const topsPool = realTops.length > 0 ? realTops : tops;

    return [
      {
        id: 1,
        mainTop: topsPool[0 % topsPool.length],
        mainBottom: bottoms[0 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[0 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[0 % shoes.length] : null,
      },
      {
        id: 2,
        mainTop: topsPool[1 % topsPool.length],
        mainBottom: bottoms[1 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[1 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[1 % shoes.length] : null,
      },
      {
        id: 3,
        mainTop: topsPool[2 % topsPool.length],
        mainBottom: bottoms[2 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[2 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[2 % shoes.length] : null,
      },
      {
        id: 4,
        mainTop: topsPool[3 % topsPool.length],
        mainBottom: bottoms[3 % bottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[3 % realOuterwear.length] : null,
        shoes: shoes.length > 0 ? shoes[1 % shoes.length] : null,
      },
    ];
  // Stable string keys: only re-run when the actual item IDs change, not on
  // every render when useShopCatalog returns a new array reference.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catalogEssentialsKey, shopTopsKey, shopBottomsKey, shopShoesKey]);

  const handleAddToWardrobe = async (item: ShopCatalogItem) => {
    if (addedItemIds[item.id] === true) return;
    logger.info('Adding catalog essential to wardrobe', item.name);
    const donePerf = perfAction(`addToWardrobe:${item.name}`);

    setAddedItemIds(prev => ({ ...prev, [item.id]: true }));

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
      logger.info('Catalog essential added to wardrobe', item.name);
    } catch (err) {
      donePerf();
      logger.error('Failed to add item, reverting', err);
      setAddedItemIds(prev => {
        const next = { ...prev };
        delete next[item.id];
        return next;
      });
    }
  };

  // ============================================
  // RENDER COMPONENTS
  // ============================================

  // Weather widget with Liquid Glass - Minimalist
  const renderWeatherWidget = useCallback(() => {
    if (loadingWeather) {
      return (
        <FrostedGlassCard style={styles.weatherWidget}>
          <ActivityIndicator size="small" color={colors.accent.primary} />
        </FrostedGlassCard>
      );
    }

    if (!weather) return null;

    return (
      <Animated.View entering={isReducedMotionEnabled ? undefined : FadeInDown.delay(100).duration(400)}>
        <View style={styles.weatherWidget}>
          <View style={styles.weatherContent}>
            <Image
              source={{ uri: `https://openweathermap.org/img/wn/${weather.icon}@2x.png` }}
              style={styles.weatherIcon as any}
              accessibilityLabel={`Weather icon: ${weather.description}`}
            />
            <View style={styles.weatherInfo}>
              <Text style={styles.weatherTemp}>{weather.temp}°C</Text>
              <Text style={styles.weatherDesc}>{weather.description}</Text>
            </View>
            <View style={styles.weatherSuggestion}>
              <Ionicons name="shirt-outline" size={16} color={colors.text.secondary} />
              <Text style={styles.suggestionText}>
                {weather.temp > 25 ? t('home.wearLight') :
                  weather.temp > 15 ? t('home.useLayers') : t('home.dressWarm')}
              </Text>
            </View>
          </View>
        </View>
      </Animated.View>
    );
  }, [loadingWeather, weather, isReducedMotionEnabled, t]);

  const isPremium = useSubscriptionStore(selectIsPremium);
  const isUnlocked = isPremium;

  // Today's Look - Hero card
  const renderTodaysLook = useCallback(() => {
    const hasEnoughItems = items.length >= 3;

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
            allowsFullscreen={false}
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
              <Text style={styles.heroTitle}>{t('home.todaysLook')}</Text>
            </View>
          </LinearGradient>
        </LiquidGlassCard>

        <TouchableOpacity
          style={styles.createOutfitButton}
          onPress={() => {
            logger.debug('Navigating to AI outfit maker with shop source');
            navigation.navigate('AIOutfit', { source: 'shop' });
          }}
          accessibilityLabel={t('home.createOutfitFromShopItems')}
          accessibilityRole="button"
        >
          <Text style={styles.createOutfitText}>{t('home.createOutfit')}</Text>
        </TouchableOpacity>

        {/* Unified Nudge Section (Prompts & Home Cards) */}
        {activePrompt && (
          <Animated.View
            entering={FadeInDown.delay(300).duration(500)}
            style={styles.hiddenGemsCard}
          >
            <View style={[styles.hiddenGemsAccent, { backgroundColor: activePrompt.color || '#F39C12' }]} />
            <View style={styles.hiddenGemsContent}>
              <View style={styles.hiddenGemsHeader}>
                <View style={styles.hiddenGemsTitleContainer}>
                  <Ionicons name={activePrompt.icon as any} size={18} color={activePrompt.color || '#E67E22'} />
                  <Text style={styles.hiddenGemsTitle}>{activePrompt.title}</Text>
                </View>
                <TouchableOpacity
                  onPress={() => {
                    logger.debug('Dismissing prompt', activePrompt.id);
                    dismissPrompt(activePrompt.id);
                    setActivePrompt(null);
                  }}
                  hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                >
                  <Ionicons name="close" size={20} color={colors.text.tertiary} />
                </TouchableOpacity>
              </View>

              <Text style={styles.hiddenGemsText}>
                {activePrompt.message}
              </Text>

              <TouchableOpacity
                style={[styles.viewAnalyticsButton, { backgroundColor: activePrompt.color || '#F39C12' }]}
                onPress={() => {
                  logger.debug('Prompt action pressed', { route: activePrompt.action.route, params: activePrompt.action.params });
                  markPromptShown();
                  setActivePrompt(null);
                  navigation.navigate(activePrompt.action.route as any, activePrompt.action.params as any);
                }}
              >
                <Text style={styles.viewAnalyticsText}>{activePrompt.action.label.toUpperCase()}</Text>
                <Ionicons name="arrow-forward" size={16} color="#FFF" />
              </TouchableOpacity>
            </View>
          </Animated.View>
        )}
      </Animated.View>
    );
  }, [items, wearLogs, todaysOutfit, activePrompt, isReducedMotionEnabled, t, navigation, markPromptShown, setActivePrompt]);

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

  // Convert a legacy curated outfit combination into the OutfitItem[] shape
  // that `OutfitCollageDisplay` expects, tagging each piece with a
  // macroCategory so the collage's slot logic keeps one item per category
  // (and drops outerwear in warm weather via the `needsOuterwear` prop).

  // Wardrobe Essentials Grid — Supabase-backed catalog picks
  const renderEssentials = useCallback(() => {
    const showSkeleton = essentialsLoading && essentialsItems.length === 0;
    const showEmpty = !essentialsLoading && essentialsItems.length === 0;

    return (
      <View style={styles.essentialsSection}>
        <Text style={styles.sectionTitle} accessibilityRole="header">{t('home.wardrobeEssentials')}</Text>

        {showSkeleton ? (
          <View style={styles.essentialsLoadingBlock}>
            <ActivityIndicator size="small" color={colors.accent.primary} />
            <Text style={styles.essentialsLoadingText}>{t('home.loadingEssentials')}</Text>
          </View>
        ) : showEmpty ? (
          <View style={styles.essentialsEmptyBlock}>
            <Ionicons
              name={essentialsError ? 'alert-circle-outline' : 'shirt-outline'}
              size={22}
              color={colors.text.tertiary}
            />
            <Text style={styles.essentialsEmptyText}>
              {essentialsError
                ? 'Could not load essentials. Pull to refresh.'
                : 'No essentials available yet.'}
            </Text>
          </View>
        ) : (
          <View style={styles.gridContainer}>
            {essentialsItems.map((item) => {
              const isAdded = addedItemIds[item.id] === true;
              const imageSrc =
                typeof item.imageUrl === 'string' ? { uri: item.imageUrl } : item.imageUrl;

              return (
                <LiquidGlassCard
                  key={item.id}
                  style={styles.gridItem}
                  contentStyle={styles.gridItemContent}
                  variant="light"
                >
                  <CachedImage uri={typeof item.imageUrl === 'string' ? item.imageUrl : ''} style={styles.gridImage} contentFit="cover" fadeIn={false} />
                  <Text style={styles.essentialItemName} numberOfLines={2}>{item.name}</Text>
                  <View style={styles.gridActions}>
                    <TouchableOpacity
                      style={[styles.addButton, isAdded && styles.addedButton]}
                      onPress={() => {
                        if (!isAdded) {
                          logger.debug('Add button pressed', item.name);
                          handleAddToWardrobe(item);
                        }
                      }}
                      accessibilityLabel={
                        isAdded
                          ? t('home.itemAddedToWardrobe', { itemName: item.name })
                          : t('home.addItemToWardrobe', { itemName: item.name })
                      }
                      accessibilityRole="button"
                    >
                      <Ionicons
                        name={isAdded ? 'checkmark' : 'add'}
                        size={20}
                        color={isAdded ? '#FFF' : colors.text.primary}
                      />
                      <Text style={[styles.addButtonText, isAdded && styles.addedButtonText]}>
                        {isAdded ? t('common.added') : t('common.add')}
                      </Text>
                    </TouchableOpacity>
                  </View>
                </LiquidGlassCard>
              );
            })}
          </View>
        )}
      </View>
    );
  }, [essentialsLoading, essentialsItems, addedItemIds, t, handleAddToWardrobe]);







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
              <Text style={styles.plannerDayName}>{day.name}</Text>
              <View style={[styles.plannerDateCircle, day.isToday && styles.plannerDateCircleToday]}>
                <Text style={[styles.plannerDateText, day.isToday && styles.plannerDateTextToday]}>{day.date}</Text>
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
            <Text style={styles.appTitleText} accessibilityRole="header">{t('home.aiWardrobe')}</Text>
          </View>

          {/* Trial Countdown Banner — visible only during active 7-day trial */}
          <TrialCountdownBanner />

          {/* Weekly Planner (Swapped position with Greeting) */}
          {renderWeeklyPlanner()}

          {/* Weather Widget */}
          {renderWeatherWidget()}

          {/* Greeting Row (Swapped position with Planner) */}
          <View style={[styles.greetingSection, { marginBottom: spacing.md }]}>
            <Text style={styles.greetingText} numberOfLines={2}>
              {greeting}, {userName}
            </Text>
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
    paddingBottom: spacing.sm,
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
    marginBottom: spacing.sm,
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
  premiumPager: {
    flexDirection: 'row',
    gap: 6,
    marginTop: spacing.lg,
    marginBottom: spacing.xs,
  },
  pagerBar: {
    width: 12,
    height: 3,
    backgroundColor: colors.text.tertiary,
    opacity: 0.3,
    borderRadius: 2,
  },
  pagerBarActive: {
    backgroundColor: colors.text.primary,
    opacity: 1,
  },
  premiumFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: spacing.md,
    paddingHorizontal: spacing.screenPadding,
  },
  premiumActionIcons: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  actionIconButton: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  createAvatarButton: {
    backgroundColor: '#173A65',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    borderRadius: 22,
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
  // Night-Time Dinner styles
  dinnerCard: {
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.92)',
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.06)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.06,
    shadowRadius: 18,
    elevation: 4,
  },
  dinnerLargeImage: {
    flex: 1,
    width: '100%',
    tintColor: undefined,
  },
  dinnerSmallBox: {
    flex: 1,
    backgroundColor: '#FBFCFF',
    borderRadius: radius.md,
    padding: spacing.xs,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.06)',
  },
  dinnerBadgeRow: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    marginTop: spacing.sm,
    width: '100%',
  },
  dinnerBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(212,175,55,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(212,175,55,0.35)',
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  dinnerBadgeText: {
    color: '#D4AF37',
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.4,
  },
  dinnerSuggestionText: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
  },
  dinnerHeaderSubtitle: {
    ...typography.scale.bodyMedium,
    color: colors.text.tertiary,
    fontStyle: 'italic',
  },
  dinnerPagerBarActive: {
    backgroundColor: '#000000',
    opacity: 1,
  },
  dinnerAvatarButton: {
    backgroundColor: '#173A65',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: '#173A65',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 4,
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
    marginBottom: spacing.xxl, // Increased again per request
  },
  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
  gridItem: {
    width: (SCREEN_WIDTH - (spacing.screenPadding * 2) - spacing.md) / 2,
    aspectRatio: 0.8,
    borderRadius: 24,
  },
  gridItemContent: {
    padding: spacing.sm,
    justifyContent: 'space-between',
  },
  gridImage: {
    width: '100%',
    height: '75%',
    borderRadius: radius.md,
  },
  gridActions: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: spacing.sm,
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F4F7FD',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.08)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: radius.pill,
    gap: 4,
    width: '100%',
  },
  addedButton: {
    backgroundColor: '#173A65',
    borderColor: '#173A65',
  },
  addButtonText: {
    ...typography.scale.labelSmall,
    color: colors.text.primary,
    fontWeight: '600',
  },
  addedButtonText: {
    color: '#FFF',
  },
  essentialItemName: {
    ...typography.scale.bodySmall,
    color: colors.text.primary,
    fontWeight: '600',
    marginTop: spacing.xs,
    marginBottom: 2,
    textAlign: 'center',
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

  // Contextual Prompt
  promptCard: {
    marginHorizontal: spacing.screenPadding,
    marginBottom: spacing.lg,
    backgroundColor: colors.glass.frosted,
    borderRadius: radius.xl,
    overflow: 'hidden',
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: colors.border.glass,
  },
  promptAccent: {
    width: 4,
  },
  promptBody: {
    flex: 1,
    padding: spacing.md,
    gap: spacing.sm,
  },
  promptHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  promptTitle: {
    ...typography.scale.titleSmall,
    color: colors.text.primary,
    fontWeight: '700',
    flex: 1,
  },
  promptMessage: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
    lineHeight: 18,
  },
  promptAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: radius.pill,
    alignSelf: 'flex-start',
  },
  promptActionText: {
    ...typography.scale.labelSmall,
    color: '#FFF',
    fontWeight: '700',
  },

  // Hidden Gems Styles — Match Screenshot
  hiddenGemsCard: {
    marginTop: spacing.lg,
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderRadius: 28,
    flexDirection: 'row',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(24,58,103,0.06)',
    shadowColor: '#173A65',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 4,
  },
  hiddenGemsAccent: {
    width: 6,
    backgroundColor: '#F39C12', // Orange accent from screenshot
  },
  hiddenGemsContent: {
    flex: 1,
    padding: spacing.md,
  },
  hiddenGemsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  hiddenGemsTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  hiddenGemsTitle: {
    ...typography.scale.titleSmall,
    color: '#0A1931',
    fontWeight: '700',
  },
  hiddenGemsText: {
    ...typography.scale.bodyMedium,
    color: colors.text.secondary,
    lineHeight: 20,
    marginBottom: spacing.md,
    marginTop: 4,
  },
  viewAnalyticsButton: {
    backgroundColor: '#F39C12',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: radius.pill,
    alignSelf: 'flex-start',
    gap: spacing.xs,
  },
  viewAnalyticsText: {
    color: '#FFF',
    fontWeight: '700',
    fontSize: 14,
  },
});

export default HomeScreen;
