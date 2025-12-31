import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  ScrollView,
  Image,
  Dimensions,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { jwtDecode } from "jwt-decode";
import { LinearGradient } from "expo-linear-gradient";
import { Video, ResizeMode } from 'expo-av';
import * as Location from 'expo-location';
import { BlurView } from 'expo-blur';
import { TahoeIconButton, TahoeActionCard, TahoeButton } from '../components/TahoeButton';
import AppColors from '../constants/AppColors';

const { width } = Dimensions.get("window");

// Weather API Key
const WEATHER_API_KEY = "acec1d31ef3e181c0ca471ac4db642ff";

// Use unified AppColors
const TAHOE = {
  background: AppColors.surface,
  surface: AppColors.background,
  glass: AppColors.glass,
  glassBorder: 'rgba(255, 255, 255, 0.5)',
  primary: AppColors.accent,
  secondary: AppColors.textSecondary,
  text: AppColors.text,
  textSecondary: AppColors.textSecondary,
  accent: AppColors.accent,
  success: AppColors.success,
  gradientStart: AppColors.accent,
  gradientEnd: '#5856D6',
};

interface WeatherData {
  temp: number;
  description: string;
  icon: string;
  city: string;
}

const HomeScreen = () => {
  const navigation = useNavigation();
  const [userName, setUserName] = useState("User");
  const [wardrobeCount, setWardrobeCount] = useState(0);
  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loadingWeather, setLoadingWeather] = useState(true);
  const [videoUri, setVideoUri] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = await AsyncStorage.getItem("userToken");
        if (token) {
          const decoded: any = jwtDecode(token);
          setUserName(decoded.name || decoded.username || "User");
        }

        const wardrobeData = await AsyncStorage.getItem('myWardrobeItems');
        if (wardrobeData) {
          const items = JSON.parse(wardrobeData);
          setWardrobeCount(items.length);
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
        `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${WEATHER_API_KEY}`
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

  const unlockThreshold = 5;
  const isUnlocked = wardrobeCount >= unlockThreshold;
  const progress = Math.min(wardrobeCount / unlockThreshold, 1);

  const quickActions = [
    {
      icon: 'sparkles' as const,
      title: 'AI Stylist',
      subtitle: 'Get outfit ideas',
      color: '#5856D6',
      screen: 'AIChat',
    },
    {
      icon: 'airplane' as const,
      title: 'Trip Planner',
      subtitle: 'Pack smart',
      color: '#FF2D55',
      screen: 'TripPlanner',
    },
    {
      icon: 'shirt' as const,
      title: 'Try On',
      subtitle: 'Virtual fitting',
      color: '#FF9500',
      screen: 'AITryOn',
    },
    {
      icon: 'stats-chart' as const,
      title: 'Analytics',
      subtitle: 'See insights',
      color: '#34C759',
      screen: 'WardrobeAnalytics',
    },
  ];

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        {/* Header with glass effect */}
        <BlurView intensity={Platform.OS === 'ios' ? 80 : 100} tint="light" style={styles.header}>
          <TahoeIconButton
            icon="person-circle-outline"
            onPress={() => (navigation as any).navigate("Profile")}
            color={TAHOE.text}
          />

          <Text style={styles.logo}>AIWardrobe</Text>

          <TahoeIconButton
            icon="notifications-outline"
            onPress={() => (navigation as any).navigate("Profile")}
            color={TAHOE.text}
          />
        </BlurView>

        <ScrollView
          style={styles.scrollView}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.scrollContent}
        >
          {/* Welcome Message & Weather */}
          <View style={styles.headerSection}>
            <Text style={styles.welcome}>Welcome, {userName}!</Text>

            {/* Weather Widget with glass effect */}
            {loadingWeather ? (
              <BlurView intensity={60} tint="light" style={styles.weatherWidget}>
                <ActivityIndicator size="small" color={TAHOE.primary} />
              </BlurView>
            ) : weather ? (
              <BlurView intensity={60} tint="light" style={styles.weatherWidget}>
                <Image
                  source={{ uri: `https://openweathermap.org/img/wn/${weather.icon}@2x.png` }}
                  style={styles.weatherIcon}
                />
                <View style={styles.weatherInfo}>
                  <Text style={styles.weatherTemp}>{weather.temp}°C</Text>
                  <Text style={styles.weatherDesc}>{weather.description}</Text>
                  <Text style={styles.weatherCity}>{weather.city}</Text>
                </View>
              </BlurView>
            ) : null}
          </View>

          {/* Main Outfit/Video Display */}
          <View style={styles.outfitContainer}>
            {isUnlocked && videoUri ? (
              <Video
                source={{ uri: videoUri }}
                style={styles.outfitVideo}
                resizeMode={ResizeMode.CONTAIN}
                shouldPlay={false}
                isLooping
                useNativeControls
              />
            ) : isUnlocked ? (
              <Image
                source={{ uri: "https://i.pinimg.com/736x/2e/3d/d1/2e3dd14ac81b207ee6d86bc99ef576eb.jpg" }}
                style={styles.outfitImage}
                resizeMode="contain"
              />
            ) : (
              <View style={styles.placeholderContainer}>
                <View style={styles.placeholderIcon}>
                  <Ionicons name="shirt-outline" size={60} color={TAHOE.secondary} />
                </View>
                <Text style={styles.placeholderText}>Scan your wardrobe to see it here</Text>
              </View>
            )}
          </View>

          {/* Progress Section */}
          <View style={styles.progressSection}>
            {!isUnlocked ? (
              <>
                <Text style={styles.progressTitle}>
                  Add {unlockThreshold} items to unlock personalized daily looks
                </Text>
                <View style={styles.progressBarContainer}>
                  <View style={styles.progressBarBackground}>
                    <LinearGradient
                      colors={[TAHOE.gradientStart, TAHOE.gradientEnd]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 0 }}
                      style={[styles.progressBarFill, { width: `${progress * 100}%` }]}
                    />
                  </View>
                  <Text style={styles.progressText}>{wardrobeCount}/{unlockThreshold} items</Text>
                </View>

                <TahoeButton
                  title="Scan Your Wardrobe"
                  icon="add-circle-outline"
                  variant="gradient"
                  fullWidth
                  haptic="medium"
                  onPress={() => (navigation as any).navigate('WardrobeVideo')}
                  style={styles.scanButton}
                />
              </>
            ) : (
              <BlurView intensity={60} tint="light" style={styles.unlockedSection}>
                <View style={styles.unlockedBadge}>
                  <Ionicons name="checkmark-circle" size={24} color={TAHOE.success} />
                  <Text style={styles.unlockedText}>Daily looks unlocked! 🎉</Text>
                </View>
                <Text style={styles.unlockedSubtext}>
                  You have {wardrobeCount} items in your wardrobe
                </Text>
              </BlurView>
            )}
          </View>

          {/* Quick Actions with Tahoe Action Cards */}
          <View style={styles.quickActions}>
            <Text style={styles.sectionTitle}>Quick Actions</Text>
            <View style={styles.actionGrid}>
              {quickActions.map((action, index) => (
                <TahoeActionCard
                  key={action.screen}
                  icon={action.icon}
                  title={action.title}
                  subtitle={action.subtitle}
                  iconColor={action.color}
                  onPress={() => (navigation as any).navigate(action.screen)}
                  style={styles.actionCardItem}
                />
              ))}
            </View>
          </View>

          {/* Extra spacing at bottom */}
          <View style={{ height: 100 }} />
        </ScrollView>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: TAHOE.background,
  },
  safeArea: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    borderBottomWidth: 0.5,
    borderBottomColor: 'rgba(0, 0, 0, 0.1)',
  },
  logo: {
    fontSize: 20,
    fontWeight: '700',
    color: TAHOE.text,
    letterSpacing: -0.5,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  headerSection: {
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 16,
  },
  welcome: {
    fontSize: 28,
    fontWeight: '700',
    color: TAHOE.text,
    marginBottom: 16,
    letterSpacing: -0.5,
  },
  weatherWidget: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 18,
    borderRadius: 16,
    backgroundColor: TAHOE.glass,
    borderWidth: 1,
    borderColor: TAHOE.glassBorder,
    overflow: 'hidden',
  },
  weatherIcon: {
    width: 50,
    height: 50,
  },
  weatherInfo: {
    marginLeft: 12,
  },
  weatherTemp: {
    fontSize: 22,
    fontWeight: '700',
    color: TAHOE.text,
  },
  weatherDesc: {
    fontSize: 14,
    color: TAHOE.textSecondary,
    textTransform: 'capitalize',
  },
  weatherCity: {
    fontSize: 12,
    color: TAHOE.secondary,
    marginTop: 2,
  },
  outfitContainer: {
    width: width - 40,
    height: 380,
    marginHorizontal: 20,
    backgroundColor: TAHOE.surface,
    borderRadius: 24,
    overflow: 'hidden',
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 24,
    elevation: 8,
  },
  outfitImage: {
    width: '100%',
    height: '100%',
  },
  outfitVideo: {
    width: '100%',
    height: '100%',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: TAHOE.background,
  },
  placeholderIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: TAHOE.glass,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  placeholderText: {
    fontSize: 16,
    color: TAHOE.textSecondary,
    fontWeight: '500',
  },
  progressSection: {
    paddingHorizontal: 20,
    marginBottom: 32,
  },
  progressTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: TAHOE.textSecondary,
    marginBottom: 16,
    textAlign: 'center',
  },
  progressBarContainer: {
    marginBottom: 20,
  },
  progressBarBackground: {
    height: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.06)',
    borderRadius: 5,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 5,
  },
  progressText: {
    fontSize: 14,
    color: TAHOE.textSecondary,
    fontWeight: '600',
    textAlign: 'center',
  },
  scanButton: {
    marginTop: 4,
  },
  unlockedSection: {
    alignItems: 'center',
    paddingVertical: 20,
    paddingHorizontal: 24,
    borderRadius: 20,
    backgroundColor: TAHOE.glass,
    borderWidth: 1,
    borderColor: TAHOE.glassBorder,
    overflow: 'hidden',
  },
  unlockedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  unlockedText: {
    fontSize: 18,
    fontWeight: '600',
    color: TAHOE.success,
  },
  unlockedSubtext: {
    fontSize: 14,
    color: TAHOE.textSecondary,
  },
  quickActions: {
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: TAHOE.text,
    marginBottom: 16,
    letterSpacing: -0.3,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 14,
  },
  actionCardItem: {
    width: (width - 54) / 2,
  },
});

export default HomeScreen;