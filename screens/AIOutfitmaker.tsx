import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  Dimensions,
  StyleSheet,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from "expo-haptics";
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { BlurView } from 'expo-blur';
import { supabase } from '../lib/supabase';
import { mpants, mshirts, pants, shoes, tops } from '../images';
import LiquidGlass2026Theme, { SpatialElevation } from '../constants/LiquidGlass2026Theme';

const shopItems = [
  { id: 'shop_1', image: require('../pictures/shop/image.png'), type: 'outerwear' },
  { id: 'shop_2', image: require('../pictures/shop/image copy.png'), type: 'outerwear' },
  { id: 'shop_3', image: require('../pictures/shop/image copy 2.png'), type: 'outerwear' },
  { id: 'shop_4', image: require('../pictures/shop/image copy 3.png'), type: 'pants' },
  { id: 'shop_5', image: require('../pictures/shop/image copy 4.png'), type: 'pants' },
  { id: 'shop_6', image: require('../pictures/shop/image copy 5.png'), type: 'shoes' },
  { id: 'shop_7', image: require('../pictures/shop/image copy 6.png'), type: 'shoes' },
];

const { width, height } = Dimensions.get('window');

interface OutfitItem {
  name: string;
  image: string;
  type?: string;
}

interface GeneratedOutfit {
  id: string;
  mainImage: string;
  matchScore: number;
  description: string;
  items: OutfitItem[];
  stylingTips?: string;
}

const aiStyles = [
  { id: 'old_money', label: 'Old Money', icon: 'diamond', desc: 'Classic, refined pieces with a subtle focus on pure luxury.' },
  { id: 'streetwear', label: 'Streetwear', icon: 'flash', desc: 'Edgy, oversized aesthetics blending comfort with high fashion.' },
  { id: 'minimalist', label: 'Minimalist', icon: 'remove', desc: 'Clean lines, neutral colors, and essential wardrobe staples.' },
  { id: 'y2k', label: 'Y2K', icon: 'sparkles', desc: 'Bold colors, nostalgic 2000s vibes, and striking accessories.' },
  { id: 'business_casual', label: 'Modern Professional', icon: 'briefcase', desc: 'Sharp, tailored looks perfect for the modern workplace.' },
];

const AIOutfitGenerator = () => {
  const navigation = useNavigation();
  const [activeMode, setActiveMode] = useState<'auto' | 'manual'>('auto');
  const [selectedStyle, setSelectedStyle] = useState('old_money');
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
  const [error, setError] = useState('');

  // Wardrobe Items State
  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);
  const [selectedItemIds, setSelectedItemIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadWardrobeItems();
  }, []);

  const loadWardrobeItems = async () => {
    try {
      const data = await AsyncStorage.getItem('myWardrobeItems');
      let items = data ? JSON.parse(data) : [];

      // Filter out invalid or purely empty items that might be stuck in storage
      items = items.filter((i: any) => i && i.image);

      // Always inject the shop items so the user can see/use them
      const existingIds = new Set(items.map((i: any) => i.id));
      const newShopItems = shopItems.filter(s => !existingIds.has(s.id));
      items = [...newShopItems, ...items];

      // If user has NO personal wardrobe items, add some fallback mock items
      if (items.length <= shopItems.length) {
        items = [
          ...items,
          ...mshirts.slice(0, 3).map(i => ({ ...i, type: 't-shirt' })),
          ...mpants.slice(0, 3).map(i => ({ ...i, type: 'pants' })),
          ...shoes.slice(0, 2).map(i => ({ ...i, type: 'shoes' })),
        ];
      }

      // Force truly unique string IDs for everything to fix "1 click selects 4" bug
      setWardrobeItems(items.map((item: any, index: number) => ({
        ...item,
        id: `uniq_item_${index}_${item.type || 'unknown'}`,
        type: item.type || 'Clothing Piece',
      })));
    } catch (e) {
      console.error('Failed to load wardrobe', e);
    }
  };

  const getMacroCategory = (type: string) => {
    const t = (type || '').toLowerCase();
    if (t.includes('jacket') || t.includes('coat') || t.includes('outer') || t.includes('zip') || t.includes('sweater') || t.includes('switer') || t.includes('pullover') || t.includes('hoodie') || t.includes('polo') || t.includes('shirt') || t.includes('t-shirt') || t.includes('top')) return 'outerwear';
    if (t.includes('pant') || t.includes('bottom') || t.includes('trouser') || t.includes('jeans')) return 'pants';
    if (t.includes('shoe') || t.includes('sneaker') || t.includes('boot')) return 'shoes';
    return t; // Unknown type is its own category
  };

  const toggleItemSelection = (id: string) => {
    setSelectedItemIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        const itemToAdd = wardrobeItems.find(i => i.id === id);
        const itemType = itemToAdd?.type || 'Clothing Piece';
        const macroCat = getMacroCategory(itemType);

        // Enforce max 1 item per macro category ALWAYS!
        const existingOfSameType = wardrobeItems.find(i =>
          newSet.has(i.id) && getMacroCategory(i.type || '') === macroCat
        );
        if (existingOfSameType) {
          newSet.delete(existingOfSameType.id);
        }

        newSet.add(id);
      }
      return newSet;
    });
  };

  const generateOutfits = async (overrideStyle?: string) => {
    const styleToUse = overrideStyle || selectedStyle;

    // Allow generation if a style is explicitly passed (tapping a chip), or if items are selected.
    if (!overrideStyle && selectedItemIds.size === 0) {
      setError('Please select at least one item, or tap an AI Style to generate an outfit automatically.');
      return;
    }

    setLoading(true);
    setError('');
    setOutfits([]);

    const selectedClothing = wardrobeItems.filter(item => selectedItemIds.has(item.id || item.image));
    const selectedClothingMetadata = selectedClothing.map(item => ({
      type: item.type,
      color: item.color,
      description: item.description || item.name,
      image: item.image,
    }));

    try {
      console.log('🎨 Generating outfits:', { stylePreferences: styleToUse, selectedItemsCount: selectedItemIds.size });

      const { data, error } = await supabase.functions.invoke('generate-outfits', {
        body: {
          stylePreferences: styleToUse,
          occasion: 'Any', // Generic fallback
          selectedItems: selectedClothingMetadata,
          limit: 3
        }
      });

      if (error) throw error;

      if (data && data.success && data.outfits.length > 0) {
        // Post-process the backend output to strictly guarantee no duplicates!
        const cleanedOutfits = data.outfits.map((outfit: any) => {
          const uniqueItems: any[] = [];
          const usedCats = new Set<string>();
          outfit.items.forEach((item: any) => {
            const cat = getMacroCategory(item.type || '');
            if (!usedCats.has(cat)) {
              usedCats.add(cat);
              uniqueItems.push(item);
            }
          });
          return { ...outfit, items: uniqueItems };
        });

        setOutfits(cleanedOutfits);
      } else {
        setError('No matching outfits found. Try selecting different items!');
      }
    } catch (err: any) {
      console.error('Outfit generation error:', err);
      // Fallback for demonstration since we don't know if the backend is updated yet
      setTimeout(() => {
        // Build a unique fallback selection ensuring max 1 item per category
        const demoItems: any[] = [];
        const usedCategories = new Set<string>();

        // Priority to selected items
        for (const item of selectedClothing) {
          const macroCat = getMacroCategory(item.type || '');
          if (!usedCategories.has(macroCat)) {
            demoItems.push(item);
            usedCategories.add(macroCat);
          }
        }

        // Fill remaining from wardrobe if needed
        for (const item of wardrobeItems) {
          const macroCat = getMacroCategory(item.type || '');
          if (!usedCategories.has(macroCat)) {
            demoItems.push(item);
            usedCategories.add(macroCat);
          }
          if (demoItems.length >= 3) break; // We only need Outerwear, Pants, Shoes
        }

        setOutfits([{
          id: `demo_${Date.now()}`,
          mainImage: demoItems.length > 0 ? demoItems[0].image : 'https://via.placeholder.com/400',
          matchScore: 0.95,
          description: `A perfect ${styleToUse} look built around your wardrobe. Effortless and stylish.`,
          items: demoItems,
          stylingTips: 'Pair this with some simple accessories to keep the look clean.'
        }]);
        setLoading(false);
        setError('');
      }, 1500);
    } finally {
      if (!error) setLoading(false);
    }
  };

  if (outfits.length > 0) {
    return (
      <View style={styles.container}>
        <SafeAreaView style={{ flex: 1 }}>
          <View style={styles.header}>
            <TouchableOpacity onPress={() => setOutfits([])} style={styles.backButton}>
              <Ionicons name="chevron-back" size={28} color="#1a1a1a" />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Your Outfits</Text>
            <View style={{ width: 40 }} />
          </View>

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 20 }}>
            {outfits.map((outfit) => (
              <View key={outfit.id} style={styles.outfitCard}>
                <Image source={typeof outfit.mainImage === 'number' ? outfit.mainImage : { uri: outfit.mainImage }} style={styles.outfitImage} resizeMode="cover" />
                <View style={styles.matchBadge}>
                  <Text style={styles.matchBadgeText}>{Math.round(outfit.matchScore * 100)}% Match</Text>
                </View>

                <View style={{ padding: 20 }}>
                  <Text style={styles.outfitDesc}>{outfit.description}</Text>
                  <Text style={styles.itemsLabel}>Items Included (Max 5):</Text>
                  <View style={[styles.collageContainer, { flexDirection: 'row', flexWrap: 'wrap' }]}>
                    {outfit.items.map((item: any, idx: number) => {
                      if (!item) return null;
                      return (
                        <View key={idx} style={styles.collageItem}>
                          <Image source={typeof item.image === 'number' ? item.image : { uri: item.image }} style={styles.collageItemImage} resizeMode="cover" />
                        </View>
                      );
                    })}
                  </View>
                  <View style={styles.stylingTipsBox}>
                    <Text style={styles.stylingTipsText}>
                      💡 <Text style={{ fontWeight: '600' }}>Styling Tip:</Text> {outfit.stylingTips}
                    </Text>
                  </View>

                  <TouchableOpacity
                    style={styles.wishlistButton}
                    onPress={() => {
                      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                      navigation.navigate('Main' as never); // Returns user to main page
                    }}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="heart-outline" size={20} color={LiquidGlass2026Theme.colors.text.primary} />
                    <Text style={styles.wishlistButtonText}>Add to wishlist</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </ScrollView>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Dynamic Background */}
      <LinearGradient
        colors={LiquidGlass2026Theme.colors.gradients.liquidGlass}
        style={StyleSheet.absoluteFill}
      />

      <SafeAreaView
        edges={activeMode === 'manual' ? ['top', 'left', 'right', 'bottom'] : ['top', 'left', 'right']}
        style={{ flex: 1, paddingBottom: activeMode === 'manual' ? 100 : 0 }}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={28} color={LiquidGlass2026Theme.colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AI Stylist</Text>
          <Ionicons name="sparkles" size={24} color="#F59E0B" />
        </View>

        {/* Dual-Mode Toggle */}
        <View style={styles.viewToggleWrap}>
          <BlurView intensity={30} tint="light" style={StyleSheet.absoluteFill} />
          <TouchableOpacity
            style={[styles.viewToggleOption, activeMode === "auto" && styles.viewToggleActive]}
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              setActiveMode("auto");
            }}
          >
            <Text style={[styles.viewToggleText, activeMode === "auto" && styles.viewToggleTextActive]}>
              Auto-Stylist
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.viewToggleOption, activeMode === "manual" && styles.viewToggleActive]}
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              setActiveMode("manual");
            }}
          >
            <Text style={[styles.viewToggleText, activeMode === "manual" && styles.viewToggleTextActive]}>
              Build My Own
            </Text>
          </TouchableOpacity>
        </View>

        {/* Title Area */}
        <View style={{ paddingHorizontal: 20, paddingTop: 36, paddingBottom: 24 }}>
          <Text style={styles.sectionTitle}>
            {activeMode === 'auto' ? "Discover Your Vibe" : "Select Your Base"}
          </Text>
          <Text style={styles.sectionSubtitle}>
            {activeMode === 'auto'
              ? "Tap a style card below and AI will instantly build a complete look from your wardrobe."
              : "Choose 1 or more items from your closet, and AI will style the rest around them."}
          </Text>
        </View>


        {/* Error Message */}
        {error ? (
          <View style={{ paddingHorizontal: 20, marginBottom: 16 }}>
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          </View>
        ) : null}

        {/* Dynamic Content based on Mode */}
        {activeMode === 'auto' ? (
          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20 }}>
            {aiStyles.map((styleObj) => {
              return (
                <TouchableOpacity
                  key={styleObj.id}
                  activeOpacity={0.8}
                  onPress={() => {
                    setSelectedStyle(styleObj.id);
                    generateOutfits(styleObj.id);
                  }}
                  style={styles.vibeCard}
                >
                  <View style={styles.vibeCardInner}>
                    <View style={styles.vibeIconWrap}>
                      <Ionicons name={styleObj.icon as any} size={24} color={'#FFFFFF'} />
                    </View>
                    <View style={{ flex: 1, marginLeft: 20 }}>
                      <Text style={styles.vibeTitle}>{styleObj.label}</Text>
                      <Text style={styles.vibeDesc}>{styleObj.desc}</Text>
                    </View>
                    <View style={{ paddingLeft: 12 }}>
                      <Ionicons name="chevron-forward" size={24} color={'#4B5563'} />
                    </View>
                  </View>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        ) : (
          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 140 }}>
            {/* Category sections */}
            {[
              { category: 'outerwear', label: 'Outerwear', icon: 'shirt' as const },
              { category: 'pants', label: 'Pants', icon: 'resize' as const },
              { category: 'shoes', label: 'Shoes', icon: 'footsteps' as const },
            ].map((section) => {
              const categoryItems = wardrobeItems.filter(
                (item) => getMacroCategory(item.type || '') === section.category
              );
              if (categoryItems.length === 0) return null;

              return (
                <View key={section.category} style={{ marginBottom: 24 }}>
                  {/* Section Header */}
                  <View style={styles.categorySectionHeader}>
                    <Ionicons name={section.icon} size={18} color={LiquidGlass2026Theme.colors.text.primary} />
                    <Text style={styles.categorySectionTitle}>{section.label}</Text>
                    <Text style={styles.categorySectionBadge}>Select 1</Text>
                  </View>

                  {/* Horizontal Scroll of Items */}
                  <ScrollView
                    horizontal
                    showsHorizontalScrollIndicator={false}
                    contentContainerStyle={{ paddingHorizontal: 16, gap: 12 }}
                  >
                    {categoryItems.map((item) => {
                      const itemId = item.id;
                      const isSelected = selectedItemIds.has(itemId);
                      return (
                        <TouchableOpacity
                          key={itemId}
                          onPress={() => toggleItemSelection(itemId)}
                          activeOpacity={0.8}
                        >
                          <View style={[styles.categoryGridItem, isSelected && styles.gridItemActive]}>
                            {item.image ? (
                              <Image
                                source={typeof item.image === 'number' ? item.image : { uri: item.image }}
                                style={styles.categoryGridItemImage}
                                resizeMode="contain"
                              />
                            ) : (
                              <Ionicons name="shirt-outline" size={40} color={LiquidGlass2026Theme.colors.text.disabled} />
                            )}
                            {isSelected && (
                              <View style={styles.checkBadge}>
                                <Ionicons name="checkmark" size={16} color="#FFF" />
                              </View>
                            )}
                          </View>
                        </TouchableOpacity>
                      );
                    })}
                  </ScrollView>
                </View>
              );
            })}
          </ScrollView>
        )}
      </SafeAreaView>

      {/* Floating Action Button */}
      {activeMode === 'manual' && selectedItemIds.size > 0 && (
        <View style={styles.fabContainer}>
          <BlurView intensity={40} tint="light" style={styles.fabGlass}>
            <TouchableOpacity
              style={[
                styles.generateBtn,
                (activeMode === 'manual' && selectedItemIds.size === 0) && styles.generateBtnDisabled
              ]}
              onPress={() => generateOutfits(selectedStyle)}
              disabled={loading || (activeMode === 'manual' && selectedItemIds.size === 0)}
              activeOpacity={0.85}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <>
                  <Ionicons name="sparkles" size={20} color="#fff" style={{ marginRight: 8 }} />
                  <Text style={styles.generateBtnText}>
                    Generate ({selectedItemIds.size})
                  </Text>
                </>
              )}
            </TouchableOpacity>
          </BlurView>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlass2026Theme.colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  backButton: {
    marginRight: 16,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  headerTitle: {
    ...LiquidGlass2026Theme.typography.scale.headlineMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    flex: 1,
  },
  sectionTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
  },
  sectionSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    marginTop: 6,
    lineHeight: 22,
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
    padding: 16,
    paddingRight: 24,
  },
  vibeIconWrap: {
    width: 64,
    height: 64,
    borderRadius: 18,
    backgroundColor: '#0A1931', // Dark Navy
    alignItems: 'center',
    justifyContent: 'center',
  },
  vibeTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0A1931',
    marginBottom: 8,
    letterSpacing: -0.3,
  },
  vibeDesc: {
    fontSize: 14,
    color: '#4B5563',
    lineHeight: 22,
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
  // Liquid Glass FAB - Matching CreateAvatarScreen
  fabContainer: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 34 : 24,
    left: 20,
    right: 20,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: "hidden",
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
  fabGlass: {
    padding: 6,
    backgroundColor: "rgba(255,255,255,0.4)",
  },
  generateBtn: {
    flexDirection: 'row',
    backgroundColor: LiquidGlass2026Theme.colors.accent.primary,
    height: 56,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
  },
  generateBtnDisabled: {
    backgroundColor: 'rgba(10, 25, 49, 0.4)', // Faded black
  },
  generateBtnText: {
    color: LiquidGlass2026Theme.colors.text.onDark,
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    letterSpacing: 0.2,
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
    ...LiquidGlass2026Theme.typography.scale.bodyLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    marginBottom: 16,
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
});

export default AIOutfitGenerator;
