/**
 * HomeScreen - 2026 Redesign
 * Features: Bento Grid layout, Liquid Glass aesthetics, GenUI-ready components
 * Based on 2026 Digital Experience Report guidelines
 */

import React, { useState, useEffect, useMemo } from "react";
import {
  View,
  Text,
  ScrollView,
  Image,
  Dimensions,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation, useIsFocused } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { jwtDecode } from "jwt-decode";
import { LinearGradient } from "expo-linear-gradient";
import { Video, ResizeMode } from 'expo-av';
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

import { mpants, mshirts, pants, shoes, tops } from '../images';

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
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  const { isReducedMotionEnabled } = useAccessibility();

  // State
  const [userName, setUserName] = useState("User");

  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [greeting, setGreeting] = useState('Good morning');

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

  // Basic Wardrobe Suggestions
  const basicWardrobe = useMemo(() => {
    // Curate a mix of essentials
    return [
      ...mshirts.slice(0, 2),
      ...mpants.slice(0, 2),
      ...tops.slice(0, 2),
      ...pants.slice(0, 2),
      ...shoes.slice(0, 2),
    ].map(item => ({ ...item, added: false }));
  }, []);

  const [suggestions, setSuggestions] = useState(basicWardrobe);

  const handleAddToWardrobe = async (item: any) => {
    // Optimistic update
    setSuggestions(prev => prev.map(i =>
      i.image === item.image ? { ...i, added: true } : i
    ));

    // Simulate API/Storage
    // In a real app, we'd add to AsyncStorage/Database here
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
              style={styles.weatherIcon}
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
  const renderTodaysLook = () => (
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
        {isUnlocked && videoUri ? (
          <Video
            source={{ uri: videoUri }}
            style={styles.heroVideo}
            resizeMode={ResizeMode.COVER}
            shouldPlay={isFocused}
            isLooping
            isMuted
          />
        ) : isUnlocked ? (
          <Video
            source={require("../assets/videos/nux_men_o.mp4")}
            style={styles.heroVideo}
            resizeMode={ResizeMode.COVER}
            shouldPlay={isFocused}
            isLooping
            isMuted
          />
        ) : (
          <View style={styles.placeholderContainer}>
            <View style={styles.placeholderIcon}>
              <Ionicons name="shirt-outline" size={48} color={colors.text.tertiary} />
            </View>
            <Text style={styles.placeholderText}>Scan your wardrobe to see it here</Text>
          </View>
        )}

        {/* Overlay with info */}
        <LinearGradient
          colors={['transparent', 'rgba(0,0,0,0.7)']}
          style={styles.heroOverlay}
        >
          <View style={styles.heroInfo}>
            <Text style={styles.heroTitle}>Today&apos;s Look</Text>
            {/* Subtitle removed for simplicity */}
          </View>
          <TouchableOpacity
            style={styles.heroButton}
            onPress={() => (navigation as any).navigate('OutfitSwipe')}
            accessibilityLabel="Get new outfit suggestion"
          >
            <Ionicons name="refresh" size={20} color="#FFF" />
          </TouchableOpacity>
        </LinearGradient>
      </LiquidGlassCard>

      <TouchableOpacity
        style={styles.createOutfitButton}
        onPress={() => console.log('Create outfit')}
      >
        <Text style={styles.createOutfitText}>Create outfit</Text>
      </TouchableOpacity>
    </Animated.View>
  );

  // Wardrobe Essentials Grid
  const renderEssentials = () => (
    <View style={styles.essentialsSection}>
      <Text style={styles.sectionTitle}>Wardrobe Essentials</Text>
      <View style={styles.gridContainer}>
        {suggestions.map((item, index) => (
          <LiquidGlassCard
            key={index}
            style={styles.gridItem}
            contentStyle={styles.gridItemContent}
            variant="light"
          >
            <Image
              source={{ uri: item.image }}
              style={styles.gridImage}
              resizeMode="contain"
            />
            <View style={styles.gridActions}>
              <TouchableOpacity
                style={[styles.addButton, item.added && styles.addedButton]}
                onPress={() => !item.added && handleAddToWardrobe(item)}
              >
                <Ionicons
                  name={item.added ? "checkmark" : "add"}
                  size={20}
                  color={item.added ? "#FFF" : colors.text.primary}
                />
                <Text style={[styles.addButtonText, item.added && styles.addedButtonText]}>
                  {item.added ? "Added" : "Add"}
                </Text>
              </TouchableOpacity>
            </View>
          </LiquidGlassCard>
        ))}
      </View>
    </View>
  );







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
          {/* Greeting */}
          <View style={styles.headerSection}>
            <Text style={styles.appTitleText}>AIWardrobe</Text>
            <View style={styles.greetingSection}>
              <Text style={styles.greetingText} numberOfLines={1}>
                {greeting}, {userName}
              </Text>
              <TouchableOpacity
                style={styles.buzzerButton}
                onPress={() => {
                  // TODO: Navigate to calendar or open calendar modal
                  console.log("Calendar buzzer pressed");
                }}
                accessibilityLabel="Open calendar"
              >
                <Ionicons name="calendar-outline" size={24} color={colors.text.primary} />
                <View style={styles.buzzerDot} />
              </TouchableOpacity>
            </View>
          </View>

          {/* Weather Widget */}
          {renderWeatherWidget()}

          {/* Today's Look Hero */}
          {renderTodaysLook()}

          {/* Wardrobe Essentials Grid */}
          {renderEssentials()}

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
  safeArea: {
    flex: 1,
  },



  // Scroll
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: spacing.sm,
  },

  // Header & Greeting
  headerSection: {
    paddingTop: spacing.sm,
    marginBottom: spacing.xs,
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
    width: '100%',
    height: '100%',
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