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
  Alert,
  TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from "expo-haptics";
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { BlurView } from 'expo-blur';
import { supabase } from '../lib/supabase';
import LiquidGlass2026Theme, { SpatialElevation } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import useAuthStore from '../store/auth';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { type ShopItem } from '../src/services/ai/mixedOutfitService';
import { generateOutfitsFromDB, fetchWardrobeDisplayItems } from '../src/services/outfitGenerationService';
import { INSPO_SHOP_ITEMS } from '../data/inspoShopItems';

const shopItems: ShopItem[] = [
  { id: 'shop_1', name: 'Mini Dress', brand: 'ZARA', price: 59.90, image: require('../pictures/shop/image.png'), type: 'outerwear', category: 'tops' },
  { id: 'shop_2', name: 'Oversized Blazer', brand: 'ZARA', price: 129.00, image: require('../pictures/shop/image copy.png'), type: 'outerwear', category: 'tops' },
  { id: 'shop_3', name: 'Wide Leg Trousers', brand: 'ZARA', price: 89.90, image: require('../pictures/shop/image copy 2.png'), type: 'pants', category: 'bottoms' },
  { id: 'shop_4', name: 'Structured Jacket', brand: 'ZARA', price: 69.90, image: require('../pictures/shop/image copy 3.png'), type: 'outerwear', category: 'tops' },
  { id: 'shop_5', name: 'Slim Fit Jeans', brand: 'ZARA', price: 15.90, image: require('../pictures/shop/image copy 4.png'), type: 'pants', category: 'bottoms' },
  { id: 'shop_6', name: 'Ribbed Knit Top', brand: 'ZARA', price: 35.90, image: require('../pictures/shop/image copy 5.png'), type: 'top', category: 'tops' },
  { id: 'shop_7', name: 'Leather Ankle Boots', brand: 'ZARA', price: 99.90, image: require('../pictures/shop/image copy 6.png'), type: 'shoes', category: 'shoes' },
  { id: 'shop_8', name: 'Brown Trousers', brand: 'ZARA', price: 45.90, image: require('../pictures/shop/Brown-pants-with_line.png'), type: 'pants', category: 'bottoms' },
  { id: 'shop_9', name: 'High Waist Trousers', brand: 'ZARA', price: 55.90, image: require('../pictures/shop/highweist_trousers_whte.png'), type: 'pants', category: 'bottoms' },
  { id: 'shop_10', name: 'Brown Loafers', brand: 'Loro Piana', price: 850.00, image: require('../pictures/shop/Brown_loafers.png.png'), type: 'shoes', category: 'shoes' },
  { id: 'shop_11', name: 'Grey Loafers', brand: 'Loro Piana', price: 850.00, image: require('../pictures/shop/Grey_loafers_loropiana.png'), type: 'shoes', category: 'shoes' },
];

const { width, height } = Dimensions.get('window');

interface OutfitItem {
  name?: string;
  image?: string | number;
  id?: string | number;
  color?: string;
  type?: string;
  isShopItem?: boolean;
  price?: number;
  brand?: string;
  shopUrl?: string;
}

interface GeneratedOutfit {
  id: string;
  mainImage?: string | number;
  matchScore: number;
  description: string;
  items: OutfitItem[];
  stylingTips?: string | string[];
  wardrobeItemCount?: number;
  shopItemCount?: number;
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
  const { user } = useAuthStore();
  const storeItems = useWardrobeStore((state) => state.items);
  const stylePersonality = useStylePreferenceStore((state) => state.preferences.stylePersonality);
  const likeOutfit = useStylePreferenceStore((state) => state.likeOutfit);
  const [activeMode, setActiveMode] = useState<'auto' | 'manual'>('auto');
  const [selectedStyle, setSelectedStyle] = useState('old_money');
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
  const [error, setError] = useState('');
  const [promptText, setPromptText] = useState('');

  // Wardrobe Items State
  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);
  const [selectedItemIds, setSelectedItemIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadWardrobeItems();
  }, [storeItems]);

  useEffect(() => {
    const styleMap: Record<string, string> = {
      classic: 'old_money',
      trendy: 'streetwear',
      minimalist: 'minimalist',
      bohemian: 'y2k',
      edgy: 'streetwear',
      romantic: 'old_money',
      sporty: 'streetwear',
    };
    if (stylePersonality && styleMap[stylePersonality]) {
      setSelectedStyle(styleMap[stylePersonality]);
    }
  }, [stylePersonality]);

  const loadWardrobeItems = async () => {
    try {
      // Primary: fetch directly from Supabase DB
      const dbItems = await fetchWardrobeDisplayItems();
      if (dbItems.length > 0) {
        setWardrobeItems(dbItems.map(item => ({
          id: item.id,
          image: item.imageUrl,
          type: item.type,
          color: item.color,
          name: item.name,
          macroCategory: item.macroCategory,
        })));
        return;
      }
      // Fallback: merge store + AsyncStorage items
      const data = await AsyncStorage.getItem('myWardrobeItems');
      const localItems = data ? JSON.parse(data) : [];
      const normalizedStoreItems = storeItems.map((item) => ({
        id: item.id,
        image: item.imageUrl,
        type: item.subCategory || item.category,
        color: item.primaryColor,
        name: item.name,
      }));
      const mergedByImage = new Map<string, any>();
      [...localItems, ...normalizedStoreItems].forEach((item: any) => {
        const image = item?.image || item?.imageUrl;
        if (!image) return;
        mergedByImage.set(String(image), item);
      });
      const personalItems = Array.from(mergedByImage.values());
      const existingIds = new Set(personalItems.map((i: any) => String(i.id || i.uniqueId || i.image || '')));
      const mergedItems = [
        ...shopItems.filter(s => !existingIds.has(String(s.id))),
        ...personalItems,
      ];
      const normalized = mergedItems.map((item: any, index: number) => ({
        ...item,
        id: String(item.id || item.uniqueId || `uniq_item_${index}_${item.type || 'unknown'}`),
        image: item.image || item.imageUrl,
        type: item.type || item.itemType || item.subCategory || item.category || 'Clothing Piece',
      })).filter((item) => item.image);
      setWardrobeItems(normalized);
    } catch (e) {
      console.error('Failed to load wardrobe', e);
    }
  };

  const getMacroCategory = (type: string) => {
    const t = (type || '').toLowerCase();
    if (t.includes('jacket') || t.includes('coat') || t.includes('outer') || t.includes('zip') || t.includes('sweater') || t.includes('switer') || t.includes('pullover') || t.includes('hoodie') || t.includes('cardigan') || t.includes('vest') || t.includes('puffer')) return 'outerwear';
    if (t.includes('polo') || t.includes('shirt') || t.includes('t-shirt') || t.includes('tee') || t.includes('blouse') || t.includes('top')) return 'top';
    if (t.includes('pant') || t.includes('bottom') || t.includes('trouser') || t.includes('jeans')) return 'pants';
    if (t.includes('shoe') || t.includes('sneaker') || t.includes('boot') || t.includes('loafer')) return 'shoes';
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

    setLoading(true);
    setError('');
    setOutfits([]);

    const selectedIds =
      activeMode === 'manual' && selectedItemIds.size > 0
        ? Array.from(selectedItemIds)
        : undefined;

    try {
      const result = await generateOutfitsFromDB({
        prompt: promptText.trim() || undefined,
        stylePreferences: styleToUse,
        occasion: 'Everyday',
        limit: 3,
        selectedItemIds: selectedIds,
      });

      if (result.success && result.outfits.length > 0) {
        const mappedOutfits: GeneratedOutfit[] = result.outfits.map(outfit => ({
          id: outfit.id,
          mainImage: outfit.items[0]?.imageUrl || outfit.items[0]?.image,
          matchScore: outfit.matchScore,
          description: outfit.description,
          items: outfit.items.map(item => ({
            id: item.id,
            name: item.name,
            image: item.imageUrl || item.image,
            color: item.color,
            type: item.type,
            isShopItem: item.isShopItem,
            price: item.price,
            brand: item.brand,
            shopUrl: item.shopUrl,
          })),
          stylingTips: outfit.stylingTips,
          wardrobeItemCount: outfit.items.filter(i => !i.isShopItem).length,
          shopItemCount: outfit.items.filter(i => i.isShopItem).length,
        }));
        setOutfits(mappedOutfits);
      } else {
        setError(result.error || 'No outfits found. Add more items to your wardrobe!');
      }
    } catch (err: any) {
      console.error('Outfit generation error:', err);
      setError('Generation failed. Please try again.');
    }
    setLoading(false);
  };

  const saveOutfit = async (outfit: GeneratedOutfit) => {
    const itemIds = outfit.items
      .map((item) => String(item.id || item.image))
      .filter(Boolean);
    if (itemIds.length === 0) {
      Alert.alert('Cannot Save', 'This outfit has no valid items to save.');
      return;
    }

    const store = useWardrobeStore.getState();
    store.addOutfit({
      userId: user?.id || 'guest',
      itemIds,
      occasion: 'casual',
      generatedBy: 'ai',
      previewImageUrl: typeof outfit.mainImage === 'string' ? outfit.mainImage : undefined,
      reasoning: outfit.description,
      style: selectedStyle,
    });
    const latestOutfit = useWardrobeStore.getState().outfits[0];
    if (latestOutfit?.id) {
      store.saveOutfit(latestOutfit.id);
      likeOutfit(latestOutfit.id, itemIds, 'casual');
    }

    try {
      const today = new Date();
      const dateKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
      const existing = await AsyncStorage.getItem('outfitLogs');
      const logs = existing ? JSON.parse(existing) : {};
      logs[dateKey] = {
        date: dateKey,
        items: outfit.items.slice(0, 6).map((item) => ({
          id: String(item.id || item.image),
          type: item.type || 'Clothing Piece',
          image: item.image,
          color: item.color,
        })),
        occasion: 'casual',
      };
      await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
    } catch (calendarError) {
      console.error('Failed to schedule outfit log', calendarError);
    }

    if (user?.id) {
      try {
        await supabase
          .from('saved_outfits')
          .insert({
            user_id: user.id,
            items: outfit.items.map((item) => ({
              id: String(item.id || item.image),
              type: item.type || 'Clothing Piece',
              image: item.image,
            })),
            date: new Date().toISOString().split('T')[0],
            occasion: 'casual',
            season: 'All',
            name: `${selectedStyle} outfit`,
            caption: outfit.description,
            visibility: 'Everyone',
            is_ootd: false,
          });
      } catch (saveError) {
        console.error('Failed to sync saved outfit', saveError);
      }
    }

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    Alert.alert('Saved', 'Outfit saved to your closet and calendar.');
  };

  const OutfitCollageDisplay = ({ items }: { items: OutfitItem[] }) => {
    const top = items.find(i => getMacroCategory(i.type || '') === 'top') ||
                 items.find(i => getMacroCategory(i.type || '') === 'outerwear');
    const pants = items.find(i => getMacroCategory(i.type || '') === 'pants');
    const shoes = items.find(i => getMacroCategory(i.type || '') === 'shoes');
    const hasTop = !!top;
    const hasPants = !!pants;
    const hasShoes = !!shoes;

    if (!hasTop && !hasPants && !hasShoes) {
      const first = items[0];
      if (!first?.image) return null;
      return (
        <Image
          source={typeof first.image === 'number' ? first.image : { uri: first.image }}
          style={styles.outfitImage}
          resizeMode="cover"
        />
      );
    }

    return (
      <View style={styles.outfitCollage}>
        {/* Left Column: Top + Pants */}
        <View style={styles.collageColumn}>
          {hasTop && (
            <View style={styles.collageSlotHalf}>
              <Image
                source={typeof top!.image === 'number' ? top!.image : { uri: top!.image }}
                style={styles.collageSlotImage}
                resizeMode="contain"
              />
              <View style={styles.collageSlotLabel}>
                <Text style={styles.collageSlotLabelText}>Top</Text>
              </View>
            </View>
          )}
          {hasPants && (
            <View style={styles.collageSlotHalf}>
              <Image
                source={typeof pants!.image === 'number' ? pants!.image : { uri: pants!.image }}
                style={styles.collageSlotImage}
                resizeMode="contain"
              />
              <View style={styles.collageSlotLabel}>
                <Text style={styles.collageSlotLabelText}>Pants</Text>
              </View>
            </View>
          )}
        </View>

        {/* Right Column: Empty top + Shoes bottom */}
        <View style={styles.collageColumn}>
          <View style={styles.collageSlotHalf} />
          {hasShoes && (
            <View style={styles.collageSlotHalf}>
              <Image
                source={typeof shoes!.image === 'number' ? shoes!.image : { uri: shoes!.image }}
                style={styles.collageSlotImage}
                resizeMode="contain"
              />
              <View style={styles.collageSlotLabel}>
                <Text style={styles.collageSlotLabelText}>Shoes</Text>
              </View>
            </View>
          )}
        </View>
      </View>
    );
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
                <OutfitCollageDisplay items={outfit.items} />
                <View style={styles.matchBadge}>
                  <Text style={styles.matchBadgeText}>{Math.round((outfit.matchScore || 0.78) * 100)}% Match</Text>
                </View>

                <View style={{ padding: 20 }}>
                  <Text style={styles.outfitDesc}>{outfit.description}</Text>
                  <Text style={styles.itemsLabel}>Items in this outfit:</Text>
                  <ScrollView
                    horizontal
                    showsHorizontalScrollIndicator={false}
                    contentContainerStyle={{ paddingHorizontal: 2, paddingVertical: 4, marginBottom: 16 }}
                  >
                    {outfit.items.length === 0 && (
                      <Text style={{ color: '#999', fontStyle: 'italic' }}>No items available</Text>
                    )}
                    {outfit.items.map((item: OutfitItem, idx: number) => {
                      console.log(`Rendering item ${idx}:`, item.id, item.type, item.image ? 'has image' : 'no image');
                      const categoryLabel = (() => {
                        const cat = getMacroCategory(item.type || '');
                        if (cat === 'top') return 'Top';
                        if (cat === 'outerwear') return 'Outerwear';
                        if (cat === 'pants') return 'Pants';
                        if (cat === 'shoes') return 'Shoes';
                        return item.type || 'Item';
                      })();
                      return (
                        <View key={idx} style={styles.individualItemCard}>
                          {item.image ? (
                            <Image
                              source={typeof item.image === 'number' ? item.image : { uri: item.image }}
                              style={styles.individualItemImage}
                              resizeMode="contain"
                            />
                          ) : (
                            <View style={[styles.individualItemImage, { backgroundColor: '#f0f0f0', alignItems: 'center', justifyContent: 'center' }]}>
                              <Ionicons name="shirt-outline" size={40} color="#ccc" />
                            </View>
                          )}
                          <Text style={styles.individualItemType} numberOfLines={1}>
                            {categoryLabel}
                          </Text>
                          {item.isShopItem && item.price && (
                            <Text style={{ fontSize: 11, color: '#0A1931', fontWeight: '600' }}>
                              ${item.price}
                            </Text>
                          )}
                          {item.color ? (
                            <View style={styles.individualItemColorRow}>
                              <View style={[styles.individualItemColorDot, { backgroundColor: item.color }]} />
                              <Text style={styles.individualItemColorText} numberOfLines={1}>{item.color}</Text>
                            </View>
                          ) : null}
                        </View>
                      );
                    })}
                  </ScrollView>
                  <View style={styles.stylingTipsBox}>
                    <Text style={styles.stylingTipsText}>
                      💡 <Text style={{ fontWeight: '600' }}>Styling Tip:</Text>{' '}
                      {Array.isArray(outfit.stylingTips) ? outfit.stylingTips[0] : outfit.stylingTips}
                    </Text>
                  </View>

                  <TouchableOpacity
                    style={styles.wishlistButton}
                    onPress={() => saveOutfit(outfit)}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="bookmark-outline" size={20} color={LiquidGlass2026Theme.colors.text.primary} />
                    <Text style={styles.wishlistButtonText}>Save outfit</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.wishlistButton}
                    onPress={() => navigation.navigate('AITryOn' as never)}
                    activeOpacity={0.8}
                  >
                    <Ionicons name="sparkles-outline" size={20} color={LiquidGlass2026Theme.colors.text.primary} />
                    <Text style={styles.wishlistButtonText}>Validate in try-on</Text>
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
        <View style={{ paddingHorizontal: 20, paddingTop: 36, paddingBottom: 16 }}>
          <Text style={styles.sectionTitle}>
            {activeMode === 'auto' ? "Discover Your Vibe" : "Select Your Base"}
          </Text>
          <Text style={styles.sectionSubtitle}>
            {activeMode === 'auto'
              ? "Tap a style card below and AI will instantly build a complete look from your wardrobe."
              : "Choose 1 or more items from your closet, and AI will style the rest around them."}
          </Text>
        </View>

        {/* AI Prompt Input */}
        <View style={styles.promptContainer}>
          <Ionicons name="sparkles-outline" size={18} color="#6B7280" style={{ marginRight: 10 }} />
          <TextInput
            style={styles.promptInput}
            placeholder="Describe the vibe... (e.g. beach trip, business meeting)"
            placeholderTextColor="#9CA3AF"
            value={promptText}
            onChangeText={setPromptText}
            returnKeyType="done"
            multiline={false}
            maxLength={120}
          />
          {promptText.length > 0 && (
            <TouchableOpacity onPress={() => setPromptText('')} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Ionicons name="close-circle" size={18} color="#9CA3AF" />
            </TouchableOpacity>
          )}
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
              { category: 'top', label: 'Tops', icon: 'shirt' as const },
              { category: 'outerwear', label: 'Outerwear', icon: 'shirt' as const },
              { category: 'pants', label: 'Bottoms', icon: 'resize' as const },
              { category: 'shoes', label: 'Shoes', icon: 'footsteps' as const },
            ].map((section) => {
              const categoryItems = wardrobeItems.filter((item) => {
                const mc = item.macroCategory || getMacroCategory(item.type || '');
                if (section.category === 'pants') return mc === 'pants' || mc === 'bottom';
                if (section.category === 'top') return mc === 'top';
                return mc === section.category;
              });
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
            {wardrobeItems.length === 0 ? (
              <View style={styles.emptyManualState}>
                <Ionicons name="shirt-outline" size={42} color={LiquidGlass2026Theme.colors.text.secondary} />
                <Text style={styles.emptyManualTitle}>Your wardrobe is empty</Text>
                <Text style={styles.emptyManualSubtitle}>Scan your wardrobe to start building outfits.</Text>
                <TouchableOpacity
                  style={styles.emptyManualCta}
                  onPress={() => navigation.navigate('WardrobeVideo' as never)}
                  activeOpacity={0.85}
                >
                  <Text style={styles.emptyManualCtaText}>Scan wardrobe</Text>
                </TouchableOpacity>
              </View>
            ) : null}
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

  // Outfit collage display styles
  outfitCollage: {
    width: '100%',
    height: 400,
    backgroundColor: '#F7F8FA',
    overflow: 'hidden',
    flexDirection: 'row',
  },
  collageColumn: {
    flex: 1,
    flexDirection: 'column',
  },
  collageSlotHalf: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    backgroundColor: '#F7F8FA',
  },
  collageSlotImage: {
    width: '85%',
    height: '85%',
  },
  collageSlotLabel: {
    position: 'absolute',
    bottom: 10,
    left: 12,
    backgroundColor: 'rgba(10,25,49,0.62)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
  },
  collageSlotLabelText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  // Individual item cards below collage
  individualItemCard: {
    width: 120,
    marginRight: 14,
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    overflow: 'hidden',
    alignItems: 'center',
    paddingBottom: 12,
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  individualItemImage: {
    width: 120,
    height: 120,
    borderRadius: 18,
  },
  individualItemType: {
    fontSize: 12,
    color: '#1a1a2e',
    fontWeight: '700',
    marginTop: 8,
    paddingHorizontal: 8,
    textAlign: 'center',
    textTransform: 'capitalize',
    letterSpacing: 0.1,
  },
  individualItemColorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    gap: 5,
  },
  individualItemColorDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.1)',
  },
  individualItemColorText: {
    fontSize: 11,
    color: '#6B7280',
    textTransform: 'capitalize',
    maxWidth: 80,
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
  promptContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 20,
    marginBottom: 20,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: 'rgba(255,255,255,0.75)',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: 'rgba(200,200,210,0.6)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  promptInput: {
    flex: 1,
    fontSize: 14,
    color: '#1a1a2e',
    paddingVertical: 0,
  },
  emptyManualState: {
    marginTop: 24,
    marginHorizontal: 20,
    alignItems: 'center',
    padding: 24,
    borderRadius: LiquidGlass2026Theme.radius.card,
    backgroundColor: 'rgba(255,255,255,0.7)',
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  emptyManualTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    marginTop: 12,
  },
  emptyManualSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    marginTop: 8,
    textAlign: 'center',
  },
  emptyManualCta: {
    marginTop: 16,
    backgroundColor: LiquidGlass2026Theme.colors.accent.primary,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  emptyManualCtaText: {
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    color: LiquidGlass2026Theme.colors.text.onDark,
    fontWeight: '600',
  },
});

export default AIOutfitGenerator;
