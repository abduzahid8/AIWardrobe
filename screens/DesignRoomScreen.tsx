import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  ScrollView,
  Image,
  Alert,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy';
import axios from 'axios';
import moment from 'moment';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
  withSpring,
  FadeInUp,
  FadeInDown,
  Easing,
} from 'react-native-reanimated';
import AppColors from '../constants/AppColors';
import { CelebrityClothingCard } from '../components/ui';

const { width, height } = Dimensions.get('window');

// Use unified AppColors (Light Theme to match other pages)
const ALTA = {
  bg: AppColors.background,
  surface: AppColors.surface,
  surfaceLight: AppColors.surfaceSecondary,
  primary: AppColors.accent,
  accent: AppColors.vip,
  glow: AppColors.accent,
  textPrimary: AppColors.text,
  textSecondary: AppColors.textSecondary,
  textMuted: AppColors.textMuted,
};

// AliceVision API URL
const ALICEVISION_URL = 'http://172.20.10.5:5050';

// Detected clothing item interface
interface DetectedClothingItem {
  category: string;
  specificType?: string;
  primaryColor: string;
  colorHex: string;
  confidence: number;
  bbox: number[];
  cutoutImage: string;
  attributes?: {
    primaryColor?: string;
    colorPalette?: string[];
    pattern?: { type: string; confidence: number };
    material?: { type: string; confidence: number };
  };
}

// Floating Shirt Icon with Glow
const FloatingShirtIcon = () => {
  const floatY = useSharedValue(0);
  const glowOpacity = useSharedValue(0.5);

  React.useEffect(() => {
    floatY.value = withRepeat(
      withSequence(
        withTiming(-12, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) }),
        withTiming(0, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) })
      ),
      -1,
      true
    );

    glowOpacity.value = withRepeat(
      withSequence(
        withTiming(0.7, { duration: 1500 }),
        withTiming(0.3, { duration: 1500 })
      ),
      -1,
      true
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: floatY.value }],
  }));

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  return (
    <Animated.View style={[styles.emptyIconContainer, animatedStyle]}>
      <Animated.View style={[styles.iconGlow, glowStyle]} />
      <LinearGradient
        colors={[ALTA.surface, ALTA.surfaceLight]}
        style={styles.iconCircle}
      >
        <Ionicons name="shirt-outline" size={48} color={ALTA.primary} />
      </LinearGradient>
    </Animated.View>
  );
};

// Toolbar Button
const ToolbarButton = ({ icon, label, onPress, isActive = false }: {
  icon: string;
  label: string;
  onPress: () => void;
  isActive?: boolean;
}) => {
  const scale = useSharedValue(1);

  const animStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <TouchableOpacity
      onPressIn={() => { scale.value = withSpring(0.9); }}
      onPressOut={() => { scale.value = withSpring(1); }}
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
      activeOpacity={1}
    >
      <Animated.View style={[
        styles.toolbarButton,
        isActive && styles.toolbarButtonActive,
        animStyle
      ]}>
        <Ionicons
          name={icon as any}
          size={22}
          color={isActive ? '#fff' : ALTA.textSecondary}
        />
      </Animated.View>
      <Text style={[styles.toolbarLabel, isActive && styles.toolbarLabelActive]}>
        {label}
      </Text>
    </TouchableOpacity>
  );
};

const DesignRoomScreen = () => {
  const navigation = useNavigation();
  const [clothingItems, setClothingItems] = useState<any[]>([]);

  // Celebrity outfit recognition state
  const [detectedItems, setDetectedItems] = useState<DetectedClothingItem[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState('');
  const [savedItems, setSavedItems] = useState<Set<number>>(new Set());

  const isEmpty = clothingItems.length === 0 && detectedItems.length === 0;
  const currentDate = moment().format('MMMM Do');

  const handleAdd = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    (navigation as any).navigate('AddItem');
  };

  const handleScanVideo = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    (navigation as any).navigate('WardrobeVideo');
  };

  const handleAI = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    (navigation as any).navigate('AIOutfit');
  };

  const handleNext = () => {
    if (!isEmpty) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      (navigation as any).navigate('OutfitPreview');
    }
  };

  // Celebrity Outfit Upload Handler
  const handleUploadCelebrity = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    // Request permission
    if (Platform.OS !== 'web') {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please grant photo library access to upload photos.');
        return;
      }
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        analyzeOutfit(result.assets[0].uri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  // Analyze outfit with AI
  const analyzeOutfit = async (imageUri: string) => {
    setIsAnalyzing(true);
    setAnalysisProgress('📸 Processing image...');
    setDetectedItems([]);
    setSavedItems(new Set());

    try {
      // Read image as base64
      setAnalysisProgress('🔄 Preparing image...');
      const base64 = await FileSystem.readAsStringAsync(imageUri, {
        encoding: 'base64',
      });

      // Call AliceVision segment-all API
      setAnalysisProgress('🔍 AI detecting clothing items...');

      const response = await axios.post(
        `${ALICEVISION_URL}/segment-all`,
        {
          image: base64,
          add_white_background: true
        },
        { timeout: 120000 } // 2 minute timeout
      );

      if (response.data.success && response.data.items?.length > 0) {
        setAnalysisProgress(`✅ Found ${response.data.items.length} clothing items!`);

        // Map items to our format
        const items: DetectedClothingItem[] = response.data.items.map((item: any) => ({
          category: item.category,
          specificType: item.specificType,
          primaryColor: item.primaryColor,
          colorHex: item.colorHex,
          confidence: item.confidence,
          bbox: item.bbox,
          cutoutImage: item.cutoutImage,
          attributes: item.attributes,
        }));

        setDetectedItems(items);
        console.log(`✅ Detected ${items.length} items:`, items.map(i => i.specificType || i.category));

        setTimeout(() => setAnalysisProgress(''), 2000);
      } else {
        throw new Error('No clothing items detected');
      }
    } catch (error: any) {
      console.error('Analysis failed:', error);
      setAnalysisProgress('');

      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        Alert.alert('Timeout', 'AI analysis took too long. Please try a smaller image.');
      } else if (error.code === 'ERR_NETWORK') {
        Alert.alert('Connection Error', 'Cannot connect to AI service. Make sure the server is running.');
      } else {
        Alert.alert('Analysis Failed', error.message || 'Could not detect clothing items. Try a clearer photo.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Save single item to wardrobe
  const handleSaveItem = async (item: DetectedClothingItem, index: number) => {
    try {
      const AsyncStorage = require('@react-native-async-storage/async-storage').default;

      // Get existing items
      const existingData = await AsyncStorage.getItem('myWardrobeItems');
      const existingItems = existingData ? JSON.parse(existingData) : [];

      // Create new item
      const newItem = {
        id: `celebrity_${Date.now()}_${index}`,
        type: item.specificType || item.category,
        color: item.primaryColor,
        colorHex: item.colorHex,
        style: 'Celebrity Look',
        description: `${item.primaryColor} ${item.specificType || item.category}`,
        material: item.attributes?.material?.type || 'Unknown',
        pattern: item.attributes?.pattern?.type || 'Solid',
        season: 'All Seasons',
        image: item.cutoutImage,
        source: 'celebrity_scan',
        createdAt: new Date().toISOString()
      };

      // Save
      const allItems = [newItem, ...existingItems];
      await AsyncStorage.setItem('myWardrobeItems', JSON.stringify(allItems));

      // Mark as saved
      setSavedItems(prev => new Set([...prev, index]));

      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      console.log(`✅ Saved ${item.specificType || item.category} to wardrobe`);

    } catch (error) {
      console.error('Failed to save item:', error);
      Alert.alert('Error', 'Failed to save item to wardrobe');
    }
  };

  // Save all items
  const handleSaveAll = async () => {
    for (let i = 0; i < detectedItems.length; i++) {
      if (!savedItems.has(i)) {
        await handleSaveItem(detectedItems[i], i);
      }
    }
    Alert.alert('Saved! 🎉', `${detectedItems.length} items saved to your wardrobe!`);
  };

  // Clear results
  const handleClearResults = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setDetectedItems([]);
    setSavedItems(new Set());
  };

  return (
    <View style={styles.container}>
      {/* Dark Background */}
      <LinearGradient
        colors={[ALTA.bg, '#050810']}
        style={StyleSheet.absoluteFill}
      />

      {/* Subtle glow */}
      <View style={styles.topGlow}>
        <LinearGradient
          colors={[`${ALTA.accent}12`, 'transparent']}
          style={styles.topGlowGradient}
        />
      </View>

      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <Animated.View
          entering={FadeInDown.delay(50).springify()}
          style={styles.header}
        >
          <View style={styles.headerLeft}>
            <Text style={styles.headerTitle}>Style Room</Text>
            <Text style={styles.headerDate}>{currentDate}</Text>
          </View>

          {detectedItems.length > 0 ? (
            <View style={styles.headerActions}>
              <TouchableOpacity
                style={styles.clearButton}
                onPress={handleClearResults}
              >
                <Ionicons name="close" size={18} color={ALTA.textSecondary} />
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.saveAllButton}
                onPress={handleSaveAll}
              >
                <Text style={styles.saveAllText}>Save All</Text>
                <Ionicons name="checkmark-circle" size={16} color="#fff" />
              </TouchableOpacity>
            </View>
          ) : (
            <TouchableOpacity
              style={[
                styles.nextButton,
                isEmpty && styles.nextButtonDisabled
              ]}
              onPress={handleNext}
              disabled={isEmpty}
            >
              <Text style={[
                styles.nextButtonText,
                isEmpty && styles.nextButtonTextDisabled
              ]}>
                Preview
              </Text>
              <Ionicons
                name="arrow-forward"
                size={16}
                color={isEmpty ? ALTA.textMuted : '#fff'}
              />
            </TouchableOpacity>
          )}
        </Animated.View>

        {/* Canvas Area */}
        <View style={styles.canvasArea}>
          {isAnalyzing ? (
            // Loading State
            <Animated.View
              entering={FadeInUp.springify()}
              style={styles.loadingState}
            >
              <ActivityIndicator size="large" color={ALTA.primary} />
              <Text style={styles.loadingText}>{analysisProgress}</Text>
              <Text style={styles.loadingSubtext}>
                Detecting clothing with AI...
              </Text>
            </Animated.View>
          ) : detectedItems.length > 0 ? (
            // Results Grid
            <ScrollView
              contentContainerStyle={styles.resultsGrid}
              showsVerticalScrollIndicator={false}
            >
              <Text style={styles.resultsTitle}>
                🌟 Celebrity Look ({detectedItems.length} items)
              </Text>
              <View style={styles.cardsContainer}>
                {detectedItems.map((item, index) => (
                  <CelebrityClothingCard
                    key={index}
                    imageUri={item.cutoutImage}
                    clothingType={item.category}
                    specificType={item.specificType}
                    color={item.primaryColor}
                    colorHex={item.colorHex}
                    material={item.attributes?.material?.type}
                    pattern={item.attributes?.pattern?.type}
                    confidence={item.confidence}
                    attributes={item.attributes}
                    onSave={() => handleSaveItem(item, index)}
                    isSaved={savedItems.has(index)}
                    index={index}
                  />
                ))}
              </View>
            </ScrollView>
          ) : (
            // Empty State
            <Animated.View
              entering={FadeInUp.delay(150).springify()}
              style={styles.emptyState}
            >
              <FloatingShirtIcon />
              <Text style={styles.emptyTitle}>Create Your Outfit</Text>
              <Text style={styles.emptyText}>
                Add clothes from your wardrobe or{'\n'}scan new items to get started
              </Text>

              {/* Quick Actions */}
              <View style={styles.quickActions}>
                <TouchableOpacity
                  style={[styles.quickActionBtn, styles.celebrityBtn]}
                  onPress={handleUploadCelebrity}
                >
                  <Ionicons name="star" size={20} color="#FFD700" />
                  <Text style={styles.quickActionText}>Celebrity Look</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.quickActionBtn}
                  onPress={handleScanVideo}
                >
                  <Ionicons name="camera-outline" size={20} color={ALTA.primary} />
                  <Text style={styles.quickActionText}>Scan Wardrobe</Text>
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                style={styles.aiSuggestBtn}
                onPress={handleAI}
              >
                <Ionicons name="sparkles-outline" size={20} color={ALTA.accent} />
                <Text style={styles.aiSuggestText}>AI Suggest Outfit</Text>
              </TouchableOpacity>
            </Animated.View>
          )}
        </View>

        {/* Bottom Toolbar */}
        <Animated.View
          entering={FadeInUp.delay(200).springify()}
          style={styles.toolbar}
        >
          <View style={styles.toolbarContainer}>
            <ToolbarButton
              icon="add"
              label="Add"
              onPress={handleAdd}
            />
            <ToolbarButton
              icon="star"
              label="Celebrity"
              onPress={handleUploadCelebrity}
              isActive={detectedItems.length > 0}
            />
            <ToolbarButton
              icon="sparkles"
              label="AI"
              onPress={handleAI}
            />
          </View>

          {/* FAB */}
          <TouchableOpacity
            style={styles.fab}
            onPress={handleUploadCelebrity}
          >
            <LinearGradient
              colors={[ALTA.primary, ALTA.accent]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.fabGradient}
            >
              <Ionicons name="camera" size={28} color="#fff" />
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: ALTA.bg,
  },
  safeArea: {
    flex: 1,
  },
  topGlow: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: height * 0.35,
  },
  topGlowGradient: {
    flex: 1,
    borderBottomLeftRadius: 180,
    borderBottomRightRadius: 180,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  headerLeft: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 26,
    fontWeight: '700',
    color: ALTA.textPrimary,
  },
  headerDate: {
    fontSize: 14,
    color: ALTA.textSecondary,
    marginTop: 2,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  clearButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: ALTA.surfaceLight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  saveAllButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: ALTA.primary,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 20,
    gap: 6,
  },
  saveAllText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  nextButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: ALTA.primary,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    gap: 6,
  },
  nextButtonDisabled: {
    backgroundColor: ALTA.surfaceLight,
  },
  nextButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  nextButtonTextDisabled: {
    color: ALTA.textMuted,
  },

  // Canvas
  canvasArea: {
    flex: 1,
    marginHorizontal: 16,
    marginVertical: 8,
    borderRadius: 20,
    backgroundColor: ALTA.surface,
    borderWidth: 1,
    borderColor: ALTA.surfaceLight,
  },
  canvasContent: {
    flex: 1,
    padding: 16,
  },
  canvasItem: {},

  // Loading State
  loadingState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  loadingText: {
    fontSize: 16,
    fontWeight: '600',
    color: ALTA.textPrimary,
    marginTop: 16,
  },
  loadingSubtext: {
    fontSize: 14,
    color: ALTA.textSecondary,
    marginTop: 4,
  },

  // Results
  resultsGrid: {
    padding: 16,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: ALTA.textPrimary,
    marginBottom: 16,
    textAlign: 'center',
  },
  cardsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },

  // Empty State
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  emptyIconContainer: {
    width: 100,
    height: 100,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  iconGlow: {
    position: 'absolute',
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: ALTA.glow,
  },
  iconCircle: {
    width: 90,
    height: 90,
    borderRadius: 45,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: ALTA.surfaceLight,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: ALTA.textPrimary,
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: ALTA.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
  },
  quickActions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 24,
  },
  quickActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: ALTA.surfaceLight,
  },
  celebrityBtn: {
    backgroundColor: 'rgba(255, 215, 0, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(255, 215, 0, 0.3)',
  },
  quickActionText: {
    fontSize: 14,
    fontWeight: '500',
    color: ALTA.textPrimary,
  },
  aiSuggestBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: ALTA.surfaceLight,
  },
  aiSuggestText: {
    fontSize: 14,
    fontWeight: '500',
    color: ALTA.textSecondary,
  },

  // Toolbar
  toolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  toolbarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: ALTA.surface,
    borderRadius: 16,
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: ALTA.surfaceLight,
    gap: 4,
  },
  toolbarButton: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: ALTA.surfaceLight,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toolbarButtonActive: {
    backgroundColor: ALTA.primary,
  },
  toolbarLabel: {
    fontSize: 10,
    color: ALTA.textMuted,
    textAlign: 'center',
    marginTop: 4,
  },
  toolbarLabelActive: {
    color: ALTA.primary,
    fontWeight: '600',
  },
  fab: {
    width: 56,
    height: 56,
    borderRadius: 28,
    overflow: 'hidden',
  },
  fabGradient: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default DesignRoomScreen;
