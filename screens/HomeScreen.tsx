/**
 * HomeScreen - 2026 Redesign
 * Features: Bento Grid layout, Liquid Glass aesthetics, GenUI-ready components
 * Based on 2026 Digital Experience Report guidelines
 */

import React, { useState, useEffect, useMemo, useRef } from "react";
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
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { useIsFocused } from "@react-navigation/native";
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
import OutfitCollageDisplay from '../features/outfit-generator/components/OutfitCollageDisplay';
import TrialCountdownBanner from '../components/TrialCountdownBanner';
import type { ShopCatalogItem } from '../features/try-on/types';
import type { ClothingCategory } from '../src/types/domain';
import { createLogger } from '../src/utils/logger';
import { useTranslation } from 'react-i18next';
import { useAdminGuard } from '../hooks/useAdminGuard';

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

const selectEssentialShoppingMix = (items: ShopCatalogItem[]): ShopCatalogItem[] => {
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
  
  // Shuffle items for variety on each load
  const shuffled = [...items].sort(() => Math.random() - 0.5);

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

// Theme shortcuts
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

interface WeatherData {
  temp: number;
  description: string;
  icon: string;
  city: string;
}

// ============================================
// MAIN COMPONENT
// ============================================

const HomeScreen = () => {
  logger.debug('Component rendering');
  const navigation = useAppNavigation();
  const isFocused = useIsFocused();
  const { isReducedMotionEnabled } = useAccessibility();
  const { t } = useTranslation();
  const { isAdmin } = useAdminGuard();

  const player = useVideoPlayer(require("../assets/videos/nux_men_o.mp4"), (player) => {
    player.loop = true;
    player.muted = true;
    player.play();
  });

  useEffect(() => {
    if (isFocused) {
      player.play();
    } else {
      player.pause();
    }
    return () => { player.pause(); };
  }, [isFocused, player]);

  useEffect(() => {
    return () => { try { player.release?.(); } catch {} };
  }, []);

  // State
  const [userName, setUserName] = useState("User");

  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [greeting, setGreeting] = useState('Good morning');
  const [showHiddenGems, setShowHiddenGems] = useState(true);
  const [currentOutfitIndex, setCurrentOutfitIndex] = useState(0);
  const outfitFlatListRef = useRef<FlatList>(null);
  const [currentDinnerOutfitIndex, setCurrentDinnerOutfitIndex] = useState(0);
  const dinnerOutfitFlatListRef = useRef<FlatList>(null);

  // Wardrobe store data for core loop
  const items = useWardrobeStore((state) => state.items);
  const wearLogs = useWardrobeStore((state) => state.wearLogs);
  const streak = useWardrobeStore((state) => state.streak);

  // Contextual prompt
  const [activePrompt, setActivePrompt] = useState<ContextualPrompt | null>(null);

  // Daily AI outfits — one batch per category, regenerates once per calendar day.
  // Each Home section feeds its own category in so the AI styles its variants
  // to match the section title ("Team Collaboration / Business Casual", etc.).
  const weatherForAI = weather
    ? { temp: weather.temp, condition: weather.description }
    : null;

  const dailyBusinessCasual = useDailyAIOutfit({
    style: 'business_casual',
    occasion: 'Team Collaboration',
    weather: weatherForAI,
    variants: 3,
  });

  const dailyOldMoney = useDailyAIOutfit({
    style: 'old_money',
    occasion: 'Night-Time Dinner',
    weather: weatherForAI,
    variants: 3,
  });

  // Today's Look Suggestion
  const todaysOutfit = useMemo(() => {
    if (items.length >= 3) {
      // Use quickSuggest for a real outfit
      const engineWeather = weather ? { temp: weather.temp, condition: weather.description } : undefined;
      return quickSuggest(items, wearLogs, engineWeather);
    }
    return null;
  }, [items, wearLogs, weather]);

  // Sync context for AI Assistant
  useEffect(() => {
    useAppContextStore.getState().setContext(weather, todaysOutfit);
  }, [weather, todaysOutfit]);

  useEffect(() => {
    if (!isFocused) return;
    const utilization = items.length > 0
      ? Math.round((new Set(wearLogs.flatMap(l => l.itemIds)).size / items.length) * 100)
      : 0;

    getContextualPrompt(items, wearLogs, streak, utilization)
      .then(prompt => setActivePrompt(prompt))
      .catch(() => { });
  }, [isFocused, items.length, wearLogs.length, streak]);

  // Determine greeting based on time
  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting(t('aiHub.goodMorning'));
    else if (hour < 18) setGreeting(t('aiHub.goodAfternoon'));
    else setGreeting(t('aiHub.goodEvening'));
  }, [t]);

  // Read username from Supabase auth store (no JWT decode needed)
  const authUser = useAuthStore(s => s.user);
  useEffect(() => {
    if (authUser?.username) {
      setUserName(authUser.username);
    }
  }, [authUser?.username]);

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
    fetchWeather();
  }, []);

  const fetchWeather = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setLoadingWeather(false);
        return;
      }

      const location = await Location.getCurrentPositionAsync({});
      const { latitude, longitude } = location.coords;

      const response = await fetch(
        `${Config.weather.baseUrl}/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${Config.weather.apiKey}`
      );
      const data = await response.json();

      if (data.main && data.weather) {
        setWeather({
          temp: Math.round(data.main.temp),
          description: data.weather[0].description,
          icon: data.weather[0].icon,
          city: data.name,
        });
      }
    } catch (error) {
      logger.error('Weather fetch error', error);
    } finally {
      setLoadingWeather(false);
    }
  };

  // Wardrobe Essentials — sourced from all shop_catalog sources for maximum variety
  const [addedItems, setAddedItems] = useState<Set<string>>(new Set());
  const addItem = useWardrobeStore((state) => state.addItem);

  const {
    items: catalogEssentials,
    loading: essentialsLoading,
    error: essentialsError,
  } = useShopCatalog({ enabled: true, source: 'all' });

  // Dedicated shoes fetch from all sources for more variety
  const { items: catalogShoes } = useShopCatalog({
    enabled: true,
    source: 'all',
    category: 'shoes',
  });

  const essentialsItems = useMemo(
    () => selectEssentialShoppingMix(catalogEssentials),
    [catalogEssentials]
  );

  // Shop items filtered by category directly from real catalog
  // Filter strictly using name keywords to prevent mislabeled DB entries from being misplaced
  const isBottomName = (name: string) => /\b(pants?|trousers?|jeans?|chinos?|shorts?|skirts?|slacks?|joggers?|sweatpants?|bermudas?|cargos?|leggings?)\b/i.test(name);
  const isShortsName = (name: string) => /\b(shorts?|bermudas?|cargo\s*shorts?)\b/i.test(name);
  const isTopName = (name: string) => /\b(shirts?|tees?|t-shirts?|tshirts?|polos?|blouses?|tops?|tanks?|sleeveless)\b/i.test(name);
  const isOuterwearName = (name: string) => /\b(jackets?|coats?|blazers?|cardigans?|sweaters?|hoodies?|puffers?|bombers?|vests?|outerwear|trench(?:es)?|peacoats?|suits?)\b/i.test(name);
  const isShoeName = (name: string) => /\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|mules?|oxfords?)\b/i.test(name);

  const shopTops = useMemo(() => catalogEssentials.filter(item =>
    item.garmentType === 'upper_body' && !isBottomName(item.name) && !isShoeName(item.name)
  ), [catalogEssentials]);
  const shopBottoms = useMemo(() => catalogEssentials.filter(item =>
    item.garmentType === 'lower_body' && !isTopName(item.name) && !isOuterwearName(item.name) && !isShoeName(item.name) && !isShortsName(item.name)
  ), [catalogEssentials]);
  const shopShoes = useMemo(() => {
    const fromMixed = catalogEssentials.filter(item =>
      item.garmentType === 'shoes' || isShoeName(item.name)
    );
    const fromDedicated = catalogShoes.filter(item =>
      item.garmentType === 'shoes' || isShoeName(item.name)
    );
    const seen = new Set<string>();
    const merged: typeof fromMixed = [];
    for (const it of [...fromMixed, ...fromDedicated]) {
      if (!seen.has(it.id)) {
        seen.add(it.id);
        merged.push(it);
      }
    }
    return merged;
  }, [catalogEssentials, catalogShoes]);

  // Debug: log shop catalog fetch status
  if (__DEV__) {
    console.log('[HomeScreen] Shop catalog status:', {
      total: catalogEssentials.length,
      tops: shopTops.length,
      bottoms: shopBottoms.length,
      shoes: shopShoes.length,
      loading: essentialsLoading,
      error: essentialsError,
    });
  }

  // User's wardrobe categories (lowercase) — used to determine which outfit slots are truly "suggested"
  const userCategories = useMemo(
    () => new Set(items.map(i => (i.category as string).toLowerCase())),
    [items]
  );

  // Night-Time Dinner outfit combinations (elegant classic)
  // Safely index into fetched shop arrays using modulo, with fallbacks to avoid empty states
  const dinnerOutfitCombinations = useMemo(() => {
    if (!catalogEssentials.length) return []; // Only fail if catalog is completely empty
    if (shopTops.length === 0 || shopBottoms.length === 0) return [];
    
    // Only use REAL outerwear (jackets, coats, blazers, suits, etc.)
    const isOuterwear = (name: string) => isOuterwearName(name);
    const realOuterwear = shopTops.filter(t => isOuterwear(t.name));
    const realTops = shopTops.filter(t => !isOuterwear(t.name));
    const topsPool = realTops.length > 0 ? realTops : shopTops;
    
    return [
      {
        id: 1,
        mainTop: topsPool[2 % topsPool.length],
        mainBottom: shopBottoms[0 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[0 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[1 % shopShoes.length] : null,
      },
      {
        id: 2,
        mainTop: topsPool[3 % topsPool.length],
        mainBottom: shopBottoms[3 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[1 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[2 % shopShoes.length] : null,
      },
      {
        id: 3,
        mainTop: topsPool[1 % topsPool.length],
        mainBottom: shopBottoms[1 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[0 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[0 % shopShoes.length] : null,
      },
    ];
  }, [catalogEssentials, shopTops, shopBottoms, shopShoes]);

  const outfitCombinations = useMemo(() => {
    if (!catalogEssentials.length) return [];
    if (shopTops.length === 0 || shopBottoms.length === 0) return [];

    const isOuterwear = (name: string) => isOuterwearName(name);
    const realOuterwear = shopTops.filter(t => isOuterwear(t.name));
    const realTops = shopTops.filter(t => !isOuterwear(t.name));
    const topsPool = realTops.length > 0 ? realTops : shopTops;

    return [
      {
        id: 1,
        mainTop: topsPool[0 % topsPool.length],
        mainBottom: shopBottoms[0 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[0 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[0 % shopShoes.length] : null,
      },
      {
        id: 2,
        mainTop: topsPool[1 % topsPool.length],
        mainBottom: shopBottoms[1 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[1 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[1 % shopShoes.length] : null,
      },
      {
        id: 3,
        mainTop: topsPool[2 % topsPool.length],
        mainBottom: shopBottoms[2 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[2 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[2 % shopShoes.length] : null,
      },
      {
        id: 4,
        mainTop: topsPool[3 % topsPool.length],
        mainBottom: shopBottoms[3 % shopBottoms.length],
        outerLayer: realOuterwear.length > 0 ? realOuterwear[3 % realOuterwear.length] : null,
        shoes: shopShoes.length > 0 ? shopShoes[1 % shopShoes.length] : null,
      },
    ];
  }, [catalogEssentials, shopTops, shopBottoms, shopShoes]);

  const handleAddToWardrobe = async (item: ShopCatalogItem) => {
    if (addedItems.has(item.id)) return;
    logger.info('Adding catalog essential to wardrobe', item.name);

    setAddedItems(prev => new Set(prev).add(item.id));

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
      logger.info('Catalog essential added to wardrobe', item.name);
    } catch (err) {
      logger.error('Failed to add item, reverting', err);
      setAddedItems(prev => {
        const next = new Set(prev);
        next.delete(item.id);
        return next;
      });
    }
  };

  // ============================================
  // RENDER COMPONENTS
  // ============================================

  // Weather widget with Liquid Glass - Minimalist
  const renderWeatherWidget = () => {
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
  };

  const { isPremium } = useSubscriptionStore(s => ({ isPremium: s.isPremium }));
  const isUnlocked = isPremium;

  // Today's Look - Hero card
  const renderTodaysLook = () => {
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
  };

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
  const mapLegacyOutfitItemsForCollage = (c: any) => {
    // Build shop catalog lookup by macro category for client-side fallback
    const isOuterwear = (name: string) => isOuterwearName(name);
    
    // Some Zara items might not have "shoes" as garmentType, but we know they are shoes
    const allOuterwear = catalogEssentials.filter(t => isOuterwear(t.name));
    const safeOuterwear = allOuterwear.length > 0 ? allOuterwear : catalogEssentials;
    
    // If Zara has exactly 0 shoes on this page, fall back to ANY shoe in the database
    // Note: The UI requires shoes. If Zara has none, we must show a placeholder
    // or a non-Zara shoe. We use placeholders instead of putting shirts on feet.
    const allShoes = shopShoes.length > 0
      ? shopShoes
      : catalogEssentials.filter(t => t.garmentType === 'shoes' || t.name.toLowerCase().includes('shoe') || t.name.toLowerCase().includes('sneaker'));
    
    const shopByMacro: Record<string, ShopCatalogItem[]> = {
      top: shopTops.filter(t => !isOuterwear(t.name)),
      bottom: shopBottoms.length > 0 ? shopBottoms : catalogEssentials,
      shoes: allShoes, // We'll handle empty shoes below
      outerwear: safeOuterwear,
    };

    const items: Array<{ id: string; image: any; type: string; name: string; macroCategory: string }> = [];
    const filledSlots = new Set<string>();

    if (c?.outerLayer) {
      items.push({ id: `legacy_outer_${c.id}`, image: c.outerLayer.imageUrl || c.outerLayer.image, type: c.outerLayer.type || c.outerLayer.name || 'Outerwear', name: c.outerLayer.name || 'Outerwear', macroCategory: 'outerwear' });
      filledSlots.add('outerwear');
    }
    if (c?.mainTop) {
      const topName = c.mainTop.name || 'Top';
      // Force 'shirt' keyword in type so OutfitCollageDisplay classifier places it in the top slot
      const forcedTopType = isTopName(topName) && !isOuterwearName(topName) ? topName : `${topName} Shirt`;
      items.push({ id: `legacy_top_${c.id}`,   image: c.mainTop.imageUrl || c.mainTop.image,    type: forcedTopType, name: topName, macroCategory: 'top' });
      filledSlots.add('top');
    }
    if (c?.mainBottom) {
      const btmName = c.mainBottom.name || 'Pants';
      // Force 'pant/trouser' keyword in type so OutfitCollageDisplay classifier places it in the bottom slot
      const forcedBtmType = isBottomName(btmName) ? btmName : `${btmName} Pants`;
      items.push({ id: `legacy_btm_${c.id}`,   image: c.mainBottom.imageUrl || c.mainBottom.image, type: forcedBtmType, name: btmName, macroCategory: 'bottom' });
      filledSlots.add('bottom');
    }
    if (c?.shoes && shopByMacro.shoes.length > 0) {
      const shoeName = c.shoes.name || 'Shoes';
      // Force 'shoe' keyword in type so OutfitCollageDisplay classifier places it in the shoes slot
      const forcedShoeType = isShoeName(shoeName) ? shoeName : `${shoeName} Shoe`;
      items.push({ id: `legacy_shoe_${c.id}`,  image: c.shoes.imageUrl || c.shoes.image,      type: forcedShoeType,     name: shoeName,    macroCategory: 'shoes' });
      filledSlots.add('shoes');
    }

    // Force outerwear if weather requires it AND outerwear exists in shop catalog
    if (needsOuterwear && !filledSlots.has('outerwear')) {
      const fallbackItems = shopByMacro['outerwear'] || [];
      // Only add if we have REAL outerwear (not just tops falling back)
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

    // Fill missing mandatory slots (top, bottom, shoes)
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
            macroCategory: slot, // FORCE to correct slot
          });
        } else if (slot === 'shoes') {
          // Zara page had 0 shoes. Emit a placeholder so the Collage slot renders correctly
          // instead of grabbing a shirt and marking it as a "shoe".
          items.push({
            id: `legacy_placeholder_shoes_${c.id}`,
            image: null,
            type: 'Shoes',
            name: 'Shoes (Not in stock)',
            macroCategory: 'shoes',
          });
        }
      }
    });

    return items;
  };

  const mapAiOutfitItemsForCollage = (outfit: {
    items: Array<{
      id?: string;
      imageUrl?: string;
      image?: string | number;
      image_url?: string;
      type?: string;
      name?: string;
      macroCategory?: string;
    }>;
  }) => {
    // Build shop catalog lookup by macro category for client-side fallback
    const isOuterwear = (name: string) => isOuterwearName(name);
    
    const shopByMacro: Record<string, ShopCatalogItem[]> = {
      top: shopTops.filter(t => !isOuterwear(t.name)),
      bottom: shopBottoms,
      shoes: shopShoes,
      outerwear: shopTops.filter(t => isOuterwear(t.name)),
    };

    // 1. Process existing items and track which slots are filled
    const filledSlots = new Set<string>();
    const mapped = (outfit.items || []).map((item, index) => {
      // Handle multiple possible image field names from different sources
      let img = item.imageUrl || item.image_url || item.image || '';
      let finalItem = { ...item };
      
      const macro = (item.macroCategory || '').toLowerCase();
      const macroNormalized = macro === 'upper_body' ? 'top' : macro === 'lower_body' ? 'bottom' : macro;
      const shopItems = shopByMacro[macroNormalized] || [];
      
      // AGGRESSIVE FALLBACK: Replace ALL AI-generated items with shop catalog items
      // to ensure they all have valid images that work (shop images are confirmed to work)
      if (shopItems.length > 0) {
        const shopItem = shopItems[Math.floor(Math.random() * shopItems.length)];
        if (__DEV__) {
          console.log(`[mapAiOutfitItemsForCollage] Replacing ${item.id} (${macro}) with shop item ${shopItem.id}`);
        }
        finalItem = {
          ...item,
          id: shopItem.id,
          imageUrl: shopItem.imageUrl,
          image: shopItem.imageUrl,
          name: shopItem.name,
          type: shopItem.name, // Force type to name so classifier matches
          macroCategory: macroNormalized, // Ensure it's canonical
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

    // 2. Fill missing mandatory slots (top, bottom, shoes) from shop catalog
    const mandatorySlots = ['top', 'bottom', 'shoes'];
    mandatorySlots.forEach(slot => {
      if (!filledSlots.has(slot)) {
        const fallbackItems = shopByMacro[slot] || [];
        if (fallbackItems.length > 0) {
          const shopItem = fallbackItems[Math.floor(Math.random() * fallbackItems.length)];
          if (__DEV__) {
            console.log(`[mapAiOutfitItemsForCollage] Filling MISSING slot (${slot}) with shop item ${shopItem.id}`);
          }
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

    // Debug: log image availability for each item so we can trace
    // where images are being lost in the pipeline.
    if (__DEV__) {
      const imgSummary = mapped.map(m => ({
        id: String(m.id).slice(0, 8),
        hasImage: typeof m.image === 'string' ? m.image.length > 0 : Boolean(m.image),
        imgLen: typeof m.image === 'string' ? m.image.length : typeof m.image,
        macro: m.macroCategory,
      }));
      console.log('[mapAiOutfitItemsForCollage]', imgSummary);
    }
    return mapped;
  };

  // Team Collaboration — Business Casual (regenerates daily via AI)
  const renderPremiumOutfitSuggestion = () => {
    const loading = dailyBusinessCasual.loading && dailyBusinessCasual.outfits.length === 0;

    // Skeleton while the first-ever batch is loading.
    if (loading) {
      return (
        <View style={styles.premiumSection}>
          <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
            <Text style={styles.premiumHeaderTitle}>{t('home.teamCollaboration')}</Text>
            <Text style={styles.premiumHeaderSubtitle}>{t('home.businessCasualArrow')}</Text>
          </View>
          <View style={{ paddingHorizontal: spacing.screenPadding }}>
            <LiquidGlassCard
              variant="light"
              style={styles.premiumCard}
              contentStyle={[styles.premiumCardContent, { minHeight: 340, justifyContent: 'center' }]}
            >
              <ActivityIndicator size="small" color={colors.accent.primary} />
              <Text style={[styles.premiumSuggestionText, { marginTop: spacing.sm }]}>
                Styling today&apos;s business-casual looks…
              </Text>
            </LiquidGlassCard>
          </View>
        </View>
      );
    }

    const aiOutfits = dailyBusinessCasual.outfits;
  
    // Use AI outfits when available (they generate fresh combinations on regenerate)
    // Fall back to static shop catalog combinations on cold starts or network failures
    const useAI = aiOutfits.length > 0;
    
    if (__DEV__) {
      console.log('[renderPremiumOutfitSuggestion] useAI:', useAI, 'aiOutfits:', aiOutfits.length);
    }
    
    const data = useAI
      ? aiOutfits.map((o, i) => ({ id: o.id || `ai-bc-${i}`, outfit: o }))
      : outfitCombinations.map((c) => ({ id: String(c.id), legacy: c }));

    const renderOutfitItem = ({ item }: { item: { id: string; outfit?: any; legacy?: any } }) => {
      const collageItems = item.outfit 
        ? mapAiOutfitItemsForCollage(item.outfit) 
        : mapLegacyOutfitItemsForCollage(item.legacy) as any;
      const hasOuter = collageItems.some((ci: any) => ci.macroCategory === 'outerwear');

      return (
        <View style={{ width: SCREEN_WIDTH, paddingHorizontal: spacing.screenPadding}}>
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
          <Text style={styles.premiumHeaderTitle}>{t('home.teamCollaboration')}</Text>
          <Text style={styles.premiumHeaderSubtitle}>{t('home.businessCasualArrow')}</Text>
        </View>

        {data.length === 0 ? (
          <View style={{ paddingHorizontal: spacing.screenPadding }}>
            <LiquidGlassCard
              variant="light"
              style={styles.premiumCard}
              contentStyle={[styles.premiumCardContent, { minHeight: 300, justifyContent: 'center', alignItems: 'center' }]}
            >
              <Ionicons name="shirt-outline" size={48} color={colors.text.tertiary} />
              <Text style={[styles.premiumSuggestionText, { marginTop: spacing.md, textAlign: 'center' }]}>
                No business-casual looks available right now.
              </Text>
              <TouchableOpacity
                style={[styles.createAvatarButton, { marginTop: spacing.md }]}
                onPress={() => {
                  logger.debug('Regenerate business-casual daily outfits (empty state)');
                  dailyBusinessCasual.regenerate();
                }}
                accessibilityLabel={t('home.regenerateBusinessCasual')}
                accessibilityRole="button"
              >
                <Text style={styles.createAvatarText}>{t('home.tryAgain')}</Text>
              </TouchableOpacity>
            </LiquidGlassCard>
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
                logger.debug('Regenerate business-casual daily outfits');
                dailyBusinessCasual.regenerate();
              }}
              accessibilityLabel={t('home.regenerateBusinessCasual')}
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
            style={styles.createAvatarButton}
            onPress={() => {
              if (isAdmin) {
                logger.debug('Try On button pressed');
                navigation.navigate('AITryOn');
              } else {
                Alert.alert(t('common.comingSoon'));
              }
            }}
          >
            <Text style={styles.createAvatarText}>{t('home.tryOn')}</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Night-Time Dinner — Elegant Classic / Old Money (regenerates daily via AI)
  const renderNightDinnerSection = () => {
    const loading = dailyOldMoney.loading && dailyOldMoney.outfits.length === 0;

    if (loading) {
      return (
        <View style={styles.premiumSection}>
          <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
            <Text style={styles.premiumHeaderTitle}>{t('home.nightTimeDinner')}</Text>
            <Text style={styles.dinnerHeaderSubtitle}>{t('home.nightTimeDinnerSubtitle')}</Text>
          </View>
          <View style={{ paddingHorizontal: spacing.screenPadding }}>
            <View style={[styles.dinnerCard, { minHeight: 340, alignItems: 'center', justifyContent: 'center' }]}>
              <ActivityIndicator size="small" color={colors.accent.primary} />
              <Text style={[styles.dinnerSuggestionText, { marginTop: spacing.sm }]}>
                Styling tonight&apos;s elegant looks…
              </Text>
            </View>
          </View>
        </View>
      );
    }

    const aiOutfits = dailyOldMoney.outfits;
    const useAI = aiOutfits.length > 0;
    const data = useAI
      ? aiOutfits.map((o, i) => ({ id: o.id || `ai-om-${i}`, outfit: o }))
      : dinnerOutfitCombinations.map((c) => ({ id: String(c.id), legacy: c }));

    const renderDinnerItem = ({ item }: { item: { id: string; outfit?: any; legacy?: any } }) => {
      const collageItems = item.outfit 
        ? mapAiOutfitItemsForCollage(item.outfit)
        : mapLegacyOutfitItemsForCollage(item.legacy) as any;
      const hasOuter = collageItems.some((ci: any) => ci.macroCategory === 'outerwear');

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
                <View key={index} style={[styles.pagerBar, index === currentDinnerOutfitIndex && styles.dinnerPagerBarActive]} />
              ))}
            </View>
          </View>
        </View>
      );
    };

    return (
      <View style={styles.premiumSection}>
        <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
          <Text style={styles.premiumHeaderTitle}>{t('home.nightTimeDinner')}</Text>
          <Text style={styles.dinnerHeaderSubtitle}>{t('home.nightTimeDinnerSubtitle')}</Text>
        </View>

        {data.length === 0 ? (
          <View style={{ paddingHorizontal: spacing.screenPadding }}>
            <View style={[styles.dinnerCard, { minHeight: 300, alignItems: 'center', justifyContent: 'center' }]}>
              <Ionicons name="wine-outline" size={48} color="rgba(255,255,255,0.6)" />
              <Text style={[styles.dinnerSuggestionText, { marginTop: spacing.md, textAlign: 'center' }]}>
                No night-time looks available right now.
              </Text>
              <TouchableOpacity
                style={[styles.createAvatarButton, { marginTop: spacing.md }]}
                onPress={() => {
                  logger.debug('Regenerate old-money daily outfits (empty state)');
                  dailyOldMoney.regenerate();
                }}
                accessibilityLabel={t('home.regenerateDinnerOutfits')}
                accessibilityRole="button"
              >
                <Text style={styles.createAvatarText}>{t('home.tryAgain')}</Text>
              </TouchableOpacity>
            </View>
          </View>
        ) : (
          <FlatList
            ref={dinnerOutfitFlatListRef}
            data={data}
            renderItem={renderDinnerItem}
            keyExtractor={(item) => item.id}
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            onMomentumScrollEnd={(event) => {
              const index = Math.round(event.nativeEvent.contentOffset.x / SCREEN_WIDTH);
              setCurrentDinnerOutfitIndex(index);
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
                logger.debug('Regenerate old-money daily outfits');
                dailyOldMoney.regenerate();
              }}
              accessibilityLabel={t('home.regenerateDinnerOutfits')}
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
            style={styles.dinnerAvatarButton}
            onPress={() => {
              if (isAdmin) {
                navigation.navigate('AITryOn');
              } else {
                Alert.alert(t('common.comingSoon'));
              }
            }}
          >
            <Text style={styles.createAvatarText}>{t('home.tryOn')}</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Wardrobe Essentials Grid — Supabase-backed catalog picks
  const renderEssentials = () => {
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
              const isAdded = addedItems.has(item.id);
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
                  <Text style={styles.essentialItemName} numberOfLines={1}>{item.name}</Text>
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
  };







  // Weekly Planner - Horizontal Row
  const renderWeeklyPlanner = () => {
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
  };

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
            <Text style={styles.greetingText} numberOfLines={1}>
              {greeting}, {userName}
            </Text>
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: spacing.sm }}>
              <StreakBadge variant="inline" />
              <TouchableOpacity
                style={styles.buzzerButton}
                onPress={() => {
                  console.log('[IPAD-DEBUG] Calendar button pressed');
                  navigation.navigate('Calendar');
                }}
                onPressIn={() => console.log('[IPAD-DEBUG] Calendar button touch start')}
                onPressOut={() => console.log('[IPAD-DEBUG] Calendar button touch end')}
                accessibilityLabel={t('home.openCalendar')}
              >
                <Ionicons name="calendar-outline" size={24} color={colors.text.primary} />
                <View style={styles.buzzerDot} />
              </TouchableOpacity>
            </View>
          </View>




          {/* Today's Look Hero */}
          {renderTodaysLook()}

          {/* Wardrobe Essentials Grid OR Team Collaboration (Conditional) */}
          {items.length < 5 ? renderEssentials() : renderPremiumOutfitSuggestion()}

          {/* Night-Time Dinner — Elegant Classic */}
          {items.length >= 5 && renderNightDinnerSection()}

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