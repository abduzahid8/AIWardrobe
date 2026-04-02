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
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { useNavigation, useIsFocused } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { jwtDecode } from "jwt-decode";
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
import useAppContextStore from '../src/store/contextStore';
import type { ContextualPrompt } from '../src/services/contextualPromptService';
import {
  getContextualPrompt,
  markPromptShown,
  dismissPrompt,
} from '../src/services/contextualPromptService';

import { quickSuggest } from '../src/services/suggestionEngine';
import { INSPO_SHOP_ITEMS } from '../data/inspoShopItems';
import { BASIC_CLOTHING_ITEMS, type BasicClothingItem } from '../data/basicClothingItems';
import AIOutfitCreatorModal from '../components/AIOutfitCreatorModal';
import { NavigationMenu } from '../src/components/NavigationMenu';

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
  console.log('[HomeScreen] Component rendering');
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  const { isReducedMotionEnabled } = useAccessibility();

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
  }, [isFocused, player]);

  // State
  const [userName, setUserName] = useState("User");

  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [greeting, setGreeting] = useState('Good morning');
  const [showAICreator, setShowAICreator] = useState(false);
  const [showHiddenGems, setShowHiddenGems] = useState(true);
  const [showNavMenu, setShowNavMenu] = useState(false);
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
    if (hour < 12) setGreeting('Good morning');
    else if (hour < 18) setGreeting('Good afternoon');
    else setGreeting('Good evening');
  }, []);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = await AsyncStorage.getItem("userToken");
        if (token) {
          const decoded = jwtDecode<{ name?: string; username?: string }>(token);
          setUserName(decoded.name || decoded.username || "User");
        }



        const savedVideo = await AsyncStorage.getItem('lastWardrobeVideo');
        if (savedVideo) {
          setVideoUri(savedVideo);
        }
      } catch (error) {
        console.log("Error fetching user:", error);
      }
    };
    fetchUserData();
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
      console.log('Weather fetch error:', error);
    } finally {
      setLoadingWeather(false);
    }
  };

  // Basic Wardrobe Suggestions — from basic_clothing/
  const [addedItems, setAddedItems] = useState<Set<string>>(new Set());
  const addItem = useWardrobeStore((state) => state.addItem);

  // Shop items filtered by category
  const shopTops = useMemo(() => INSPO_SHOP_ITEMS.filter(item => item.category === 'tops'), []);
  const shopBottoms = useMemo(() => INSPO_SHOP_ITEMS.filter(item => item.category === 'bottoms'), []);
  const shopShoes = useMemo(() => INSPO_SHOP_ITEMS.filter(item => item.category === 'shoes'), []);

  // Night-Time Dinner outfit combinations (elegant classic)
  // Each outfit: 1 shirt/top + 1 bottom + 1 outer layer (jacket) + 1 shoes
  const dinnerOutfitCombinations = useMemo(() => [
    {
      id: 1,
      mainTop: shopTops[2],      // Ribbed Knit Top
      mainBottom: shopBottoms[0], // Wide Leg Trousers
      outerLayer: shopTops[0],   // Oversized Blazer
      shoes: shopShoes[1],       // Brown Loafers
    },
    {
      id: 2,
      mainTop: shopTops[3],       // Satin Mini Dress
      mainBottom: shopBottoms[3], // High Waist Trousers
      outerLayer: shopTops[1],    // Structured Jacket
      shoes: shopShoes[2] || shopShoes[0],
    },
    {
      id: 3,
      mainTop: shopTops[2],      // Ribbed Knit Top
      mainBottom: shopBottoms[1], // Slim Fit Jeans
      outerLayer: shopTops[0],   // Oversized Blazer
      shoes: shopShoes[0],
    },
  ], [shopTops, shopBottoms, shopShoes]);

  // Create multiple outfit combinations
  // Each outfit: 1 shirt/top + 1 bottom + 1 outer layer (jacket) + 1 shoes
  const outfitCombinations = useMemo(() => [
    {
      id: 1,
      mainTop: shopTops[2],       // Ribbed Knit Top
      mainBottom: shopBottoms[0], // Wide Leg Trousers
      outerLayer: shopTops[0],    // Oversized Blazer
      shoes: shopShoes[0],
    },
    {
      id: 2,
      mainTop: shopTops[3],       // Satin Mini Dress
      mainBottom: shopBottoms[1], // Slim Fit Jeans
      outerLayer: shopTops[1],    // Structured Jacket
      shoes: shopShoes[1],
    },
    {
      id: 3,
      mainTop: shopTops[2],       // Ribbed Knit Top
      mainBottom: shopBottoms[2], // Brown Pants
      outerLayer: shopTops[0],    // Oversized Blazer
      shoes: shopShoes[2],
    },
    {
      id: 4,
      mainTop: shopTops[3],       // Satin Mini Dress
      mainBottom: shopBottoms[3], // High Waist Trousers
      outerLayer: shopTops[1],    // Structured Jacket
      shoes: shopShoes[1],
    },
  ], [shopTops, shopBottoms, shopShoes]);

  const handleAddToWardrobe = async (item: BasicClothingItem) => {
    if (addedItems.has(item.id)) return;
    console.log('[HomeScreen] Adding basic clothing item to wardrobe:', item.name);

    // Optimistic UI update
    setAddedItems(prev => new Set(prev).add(item.id));

    try {
      await addItem({
        userId: '',
        imageUrl: `basic_clothing_${item.id}`,
        category: item.category,
        subCategory: item.subCategory,
        primaryColor: item.primaryColor,
        colorHex: item.colorHex,
        pattern: item.pattern,
        material: item.material,
        name: item.name,
        seasons: item.seasons,
        occasions: item.occasions,
      });
      console.log('[HomeScreen] Basic clothing item added to wardrobe:', item.name);
    } catch (err) {
      console.log('[HomeScreen] Failed to add item, reverting:', err);
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
                {weather.temp > 25 ? 'Wear light' :
                  weather.temp > 15 ? 'Use layers' : 'Dress warm'}
              </Text>
            </View>
          </View>
        </View>
      </Animated.View>
    );
  };

  const isUnlocked = true; // Force unlocked for demo

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
          />
          {/* Overlay with info */}
          <LinearGradient
            colors={['transparent', 'rgba(0,0,0,0.8)']}
            style={styles.heroOverlay}
          >
            <View style={styles.heroInfo}>
              <Text style={styles.heroTitle}>Today&apos;s Look</Text>
            </View>
          </LinearGradient>
        </LiquidGlassCard>

        <TouchableOpacity
              style={styles.createOutfitButton}
              onPress={() => {
                console.log('[HomeScreen] Create outfit button pressed - hasEnoughItems:', hasEnoughItems);
                if (hasEnoughItems) {
                  console.log('[HomeScreen] Opening AI creator modal');
                  setShowAICreator(true);
                } else {
                  console.log('[HomeScreen] Navigating to WardrobeVideo');
                  (navigation as any).navigate('WardrobeVideo');
                }
              }}
              accessibilityLabel={hasEnoughItems ? 'Create outfit with AI' : 'Add items to closet'}
              accessibilityRole="button"
            >
              <Text style={styles.createOutfitText}>{hasEnoughItems ? 'Create outfit' : 'Digitize Closet First'}</Text>
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
                    console.log('[HomeScreen] Dismissing prompt:', activePrompt.id);
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
                  console.log('[HomeScreen] Prompt action pressed:', activePrompt.action.route, activePrompt.action.params);
                  markPromptShown();
                  setActivePrompt(null);
                  (navigation as any).navigate(activePrompt.action.route, activePrompt.action.params);
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

  // Premium Outfit Suggestion (Shown when items >= 5)
  const renderPremiumOutfitSuggestion = () => {
    const renderOutfitItem = ({ item }: { item: typeof outfitCombinations[0] }) => (
      <View style={{ width: SCREEN_WIDTH, paddingHorizontal: spacing.screenPadding}}>
        <LiquidGlassCard
          variant="light"
          style={styles.premiumCard}
          contentStyle={styles.premiumCardContent}
        >
          <View style={styles.outfitGridMain}>
          {/* Left Column - Top + Bottom */}
          <View style={styles.outfitGridLeft}>
            {item.mainTop && <Image source={item.mainTop.image} style={styles.outfitLargeImage as any} resizeMode="contain" />}
            {item.mainBottom && <Image source={item.mainBottom.image} style={styles.outfitLargeImage as any} resizeMode="contain" />}
          </View>

          {/* Right Column - Outer Layer + Shoes */}
          <View style={styles.outfitGridRight}>
            <View style={styles.outfitSmallBox}>
              {item.outerLayer && <Image source={item.outerLayer.image} style={styles.outfitSmallImage as any} resizeMode="contain" />}
            </View>
            <View style={styles.outfitSmallBox}>
              {item.shoes && <Image source={item.shoes.image} style={styles.outfitSmallImage as any} resizeMode="contain" />}
            </View>
          </View>
        </View>

        <View style={styles.premiumSuggestionInfo}>
          <Text style={styles.premiumSuggestionText}>2 items suggested</Text>
          <Ionicons name="information-circle-outline" size={16} color={colors.text.tertiary} />
        </View>

        <View style={styles.premiumPager}>
          {outfitCombinations.map((_, index) => (
            <View key={index} style={[styles.pagerBar, index === currentOutfitIndex && styles.pagerBarActive]} />
          ))}
        </View>
      </LiquidGlassCard>
      </View>
    );

    return (
      <View style={styles.premiumSection}>
        <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
          <Text style={styles.premiumHeaderTitle}>Team Collaboration</Text>
          <Text style={styles.premiumHeaderSubtitle}>Business casual ›</Text>
        </View>

        <FlatList
          ref={outfitFlatListRef}
          data={outfitCombinations}
          renderItem={renderOutfitItem}
          keyExtractor={(item) => item.id.toString()}
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

        {/* Action Footer */}
        <View style={styles.premiumFooter}>
          <View style={styles.premiumActionIcons}>
            <TouchableOpacity style={styles.actionIconButton}>
              <Ionicons name="heart-outline" size={24} color={colors.text.primary} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionIconButton}>
              <Ionicons name="pencil-outline" size={22} color={colors.text.primary} />
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
              console.log('[HomeScreen] Create avatar button pressed');
              (navigation as any).navigate('CreateAvatar');
            }}
          >
            <Text style={styles.createAvatarText}>Create Avatar</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Night-Time Dinner — Elegant Classic
  const renderNightDinnerSection = () => {
    const renderDinnerItem = ({ item }: { item: typeof dinnerOutfitCombinations[0] }) => (
      <View style={{ width: SCREEN_WIDTH, paddingHorizontal: spacing.screenPadding }}>
        <View style={styles.dinnerCard}>
          <View style={styles.outfitGridMain}>
            {/* Left Column - Top + Bottom */}
            <View style={styles.outfitGridLeft}>
              {item.mainTop && <Image source={item.mainTop.image} style={styles.dinnerLargeImage as any} resizeMode="contain" />}
              {item.mainBottom && <Image source={item.mainBottom.image} style={styles.dinnerLargeImage as any} resizeMode="contain" />}
            </View>
            {/* Right Column - Outer Layer + Shoes */}
            <View style={styles.outfitGridRight}>
              <View style={styles.dinnerSmallBox}>
                {item.outerLayer && <Image source={item.outerLayer.image} style={styles.outfitSmallImage as any} resizeMode="contain" />}
              </View>
              <View style={styles.dinnerSmallBox}>
                {item.shoes && <Image source={item.shoes.image} style={styles.outfitSmallImage as any} resizeMode="contain" />}
              </View>
            </View>
          </View>

          {/* Occasion badge */}
          <View style={styles.dinnerBadgeRow}>
            <View style={styles.dinnerBadge}>
              <Ionicons name="wine" size={13} color="#D4AF37" />
              <Text style={styles.dinnerBadgeText}>Black Tie Optional</Text>
            </View>
          </View>

          <View style={styles.premiumSuggestionInfo}>
            <Text style={styles.dinnerSuggestionText}>3 items suggested</Text>
            <Ionicons name="information-circle-outline" size={16} color="rgba(255,255,255,0.4)" />
          </View>

          <View style={styles.premiumPager}>
            {dinnerOutfitCombinations.map((_, index) => (
              <View key={index} style={[styles.pagerBar, index === currentDinnerOutfitIndex && styles.dinnerPagerBarActive]} />
            ))}
          </View>
        </View>
      </View>
    );

    return (
      <View style={styles.premiumSection}>
        <View style={[styles.premiumHeader, { paddingHorizontal: spacing.screenPadding }]}>
          <Text style={styles.premiumHeaderTitle}>Team Collaboration</Text>
          <Text style={styles.dinnerHeaderSubtitle}>Night-time dinner ›</Text>
        </View>

        <FlatList
          ref={dinnerOutfitFlatListRef}
          data={dinnerOutfitCombinations}
          renderItem={renderDinnerItem}
          keyExtractor={(item) => item.id.toString()}
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

        {/* Action Footer */}
        <View style={styles.premiumFooter}>
          <View style={styles.premiumActionIcons}>
            <TouchableOpacity style={styles.actionIconButton}>
              <Ionicons name="heart-outline" size={24} color={colors.text.primary} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionIconButton}>
              <Ionicons name="pencil-outline" size={22} color={colors.text.primary} />
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
              (navigation as any).navigate('CreateAvatar');
            }}
          >
            <Text style={styles.createAvatarText}>Try On</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // Wardrobe Essentials Grid — basic clothing from basic_clothing/
  const renderEssentials = () => (
    <View style={styles.essentialsSection}>
      <Text style={styles.sectionTitle} accessibilityRole="header">Wardrobe Essentials</Text>
      <View style={styles.gridContainer}>
        {BASIC_CLOTHING_ITEMS.map((item) => {
          const isAdded = addedItems.has(item.id);
          return (
            <LiquidGlassCard
              key={item.id}
              style={styles.gridItem}
              contentStyle={styles.gridItemContent}
              variant="light"
            >
              <Image
                source={item.image}
                style={styles.gridImage as any}
                resizeMode="cover"
              />
              <Text style={styles.essentialItemName} numberOfLines={1}>{item.name}</Text>
              <View style={styles.gridActions}>
                <TouchableOpacity
                  style={[styles.addButton, isAdded && styles.addedButton]}
                  onPress={() => {
                    if (!isAdded) {
                      console.log('[HomeScreen] Add button pressed for:', item.name);
                      handleAddToWardrobe(item);
                    }
                  }}
                  accessibilityLabel={isAdded ? `${item.name} added to wardrobe` : `Add ${item.name} to wardrobe`}
                  accessibilityRole="button"
                >
                  <Ionicons
                    name={isAdded ? "checkmark" : "add"}
                    size={20}
                    color={isAdded ? "#FFF" : colors.text.primary}
                  />
                  <Text style={[styles.addButtonText, isAdded && styles.addedButtonText]}>
                    {isAdded ? "Added" : "Add"}
                  </Text>
                </TouchableOpacity>
              </View>
            </LiquidGlassCard>
          );
        })}
      </View>
    </View>
  );







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
      <SafeAreaView style={styles.safeArea}>
        <ScrollView
          style={styles.scrollView}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.scrollContent}
        >
          {/* Header */}
          <View style={styles.headerSection}>
            <Text style={styles.appTitleText} accessibilityRole="header">AIWardrobe</Text>
          </View>

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
                  console.log('[HomeScreen] Calendar button pressed');
                  (navigation as any).navigate('Calendar');
                }}
                accessibilityLabel="Open calendar"
              >
                <Ionicons name="calendar-outline" size={24} color={colors.text.primary} />
                <View style={styles.buzzerDot} />
              </TouchableOpacity>
            </View>
          </View>




          {/* Today's Look Hero */}
          {renderTodaysLook()}

          {/* Wardrobe Essentials Grid OR Premium Outfit (Conditional) */}
          {items.length < 5 ? renderEssentials() : renderPremiumOutfitSuggestion()}

          {/* Night-Time Dinner — Elegant Classic */}
          {items.length >= 5 && renderNightDinnerSection()}

          {/* Bottom spacing for tab bar */}
          <View style={{ height: 120 }} />
        </ScrollView>
      </SafeAreaView>

      {/* Navigation Menu */}
      <NavigationMenu visible={showNavMenu} onClose={() => setShowNavMenu(false)} />

      {/* AI Outfit Creator Modal */}
      <AIOutfitCreatorModal
        visible={showAICreator}
        onClose={() => {
          console.log('[HomeScreen] AI creator modal closed');
          setShowAICreator(false);
        }}
      />
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
    borderRadius: radius.xl,
    backgroundColor: '#F7F7F7',
    borderWidth: 0,
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
    backgroundColor: '#FFF',
    borderRadius: radius.md,
    padding: spacing.xs,
    justifyContent: 'center',
    alignItems: 'center',
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
    backgroundColor: '#000',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.pill,
  },
  createAvatarText: {
    color: '#FFF',
    fontWeight: '700',
    fontSize: 15,
  },
  // Night-Time Dinner styles
  dinnerCard: {
    borderRadius: radius.xl,
    backgroundColor: '#F7F7F7',
    padding: spacing.md,
    alignItems: 'center',
  },
  dinnerLargeImage: {
    flex: 1,
    width: '100%',
    tintColor: undefined,
  },
  dinnerSmallBox: {
    flex: 1,
    backgroundColor: '#FFF',
    borderRadius: radius.md,
    padding: spacing.xs,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(212,175,55,0.2)',
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
    color: 'rgba(255,255,255,0.45)',
  },
  dinnerHeaderSubtitle: {
    ...typography.scale.bodyMedium,
    color: '#8B6914',
    fontStyle: 'italic',
  },
  dinnerPagerBarActive: {
    backgroundColor: '#D4AF37',
    opacity: 1,
  },
  dinnerAvatarButton: {
    backgroundColor: '#1A1A1A',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm + 2,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: '#D4AF37',
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
    paddingHorizontal: spacing.screenPadding,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
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
    backgroundColor: colors.background.secondary,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
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
    backgroundColor: '#0A1931',
    paddingVertical: 14,
    borderRadius: radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.md,
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
    backgroundColor: colors.background.tertiary,
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: radius.pill,
    gap: 4,
    width: '100%',
  },
  addedButton: {
    backgroundColor: '#0A1931', // Success color (Dark blue)
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
    backgroundColor: '#FFF',
    borderRadius: radius.xl,
    flexDirection: 'row',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#F0F0F0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 3,
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