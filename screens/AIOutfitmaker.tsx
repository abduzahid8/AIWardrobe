import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Image,
  Dimensions,
  StyleSheet,
  Animated,
  Platform,
  Modal,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute } from '@react-navigation/native';
import * as Haptics from "expo-haptics";
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import { useShopCatalog } from '../hooks/useShopCatalog';
import LiquidGlass2026Theme, { SpatialElevation } from '../constants/LiquidGlass2026Theme';
import * as Location from 'expo-location';
import Config from '../src/config/env';
import { createOutfitLog, type OutfitItem as CalendarOutfitItem, type OutfitLog } from '../features/calendar/types';

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
}

interface GeneratedOutfit {
  id: string;
  mainImage: string;
  matchScore: number;
  description: string;
  items: OutfitItem[];
  stylingTips?: string;
  weather?: { temp: number; condition: string; city?: string };
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
  const route = useRoute<any>();
  const source = route.params?.source;
  const [selectedStyle, setSelectedStyle] = useState('old_money');
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
  const [error, setError] = useState('');
  const insets = useSafeAreaInsets();

  // ── Weather State ─────────────────────────────────────────────────
  const [weather, setWeather] = useState<{ temp: number; condition: string; icon?: string; city?: string } | undefined>(undefined);

  // ── Calendar Modal State ──────────────────────────────────────────
  const [calendarVisible, setCalendarVisible] = useState(false);
  const [calendarOutfit, setCalendarOutfit] = useState<GeneratedOutfit | null>(null);
  const [calendarDate, setCalendarDate] = useState<Date>(new Date());

  // ── Design Tokens (Liquid Glass prestige style) ──────────────────
  function useDesignTokens() {
    const isDark = false; // keep light theme consistent with existing
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
  const D = useDesignTokens();

  // Wardrobe Items State
  const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);

  // Use the live catalog from Supabase
  const { items: liveShopCatalog, loading: shopCatalogLoading } = useShopCatalog({
    enabled: true, // we always want it ready for fallback/shop injections
  });

  // Map the new catalog format into what the outfit generator expects
  const liveShopMapped = React.useMemo(() => liveShopCatalog.map((item) => ({
    id: item.id,
    image: item.imageUrl,
    imageUrl: item.imageUrl,
    type: item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : 'shoes',
    category: item.garmentType === 'upper_body' ? 'tops' : item.garmentType === 'lower_body' ? 'bottoms' : 'shoes',
    name: item.name,
    brand: item.brand,
    description: item.description || `${item.name} by ${item.brand}`,
  })), [liveShopCatalog]);

  // Prevent setState after unmount
  const isMounted = React.useRef(true);
  const fallbackTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    isMounted.current = true;
    
    // We only want to load items once the live shop catalog has responded (so we have fallbacks/shop items)
    if (!shopCatalogLoading) {
      loadWardrobeItems();
    }
    
    return () => {
      isMounted.current = false;
      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
    };
  }, [shopCatalogLoading, liveShopMapped]);

  const loadWardrobeItems = async () => {
    try {
      // ── Shop mode: use ONLY live Supabase shop catalog items.
      // Real Supabase UUIDs MUST be preserved so the edge function can enrich
      // items with imageUrl via its itemMap lookup. Do NOT mix in AsyncStorage.
      if (source === 'shop') {
        if (!isMounted.current) return;
        setWardrobeItems(
          liveShopMapped.map((item: any) => ({
            ...item,
            // Preserve the real UUID from shop_catalog — never overwrite with a fake id
            type: item.type || item.category || 'Clothing Piece',
            imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
            name: item.name || item.type || 'Clothing item',
            description: item.description || item.name || '',
          }))
        );
        return;
      }

      // ── Wardrobe mode: merge AsyncStorage + shop items ──────────────────
      const data = await AsyncStorage.getItem('myWardrobeItems');
      let items = data ? JSON.parse(data) : [];

      // Normalise image/imageUrl so both fields are always populated
      items = items.map((i: any) => ({
        ...i,
        image: i.image || i.imageUrl,
        imageUrl: i.imageUrl || (typeof i.image === 'string' ? i.image : undefined),
      }));

      // Drop items with no usable image
      items = items.filter((i: any) => i && (i.imageUrl || typeof i.image === 'string'));

      const existingIds = new Set(items.map((i: any) => i.id));
      const newShopItems = liveShopMapped.filter(s => !existingIds.has(s.id));
      items = [...newShopItems, ...items];

      if (!isMounted.current) return;
      setWardrobeItems(items.map((item: any, index: number) => ({
        ...item,
        // For personal wardrobe items that have no real id, generate a stable key.
        // For shop items merged here, preserve their real UUID.
        id: item.id || `local_item_${index}_${item.type || item.category || 'unknown'}`,
        type: item.type || item.category || 'Clothing Piece',
        imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : undefined) || '',
        name: item.name || item.type || 'Clothing item',
        description: item.description || item.name || '',
      })));
    } catch (e) {
      console.error('Failed to load wardrobe', e);
    }
  };

  // Client-side 3-slot display category (used for UI grouping & selection logic)
  const getMacroCategory = (type: string) => {
    const t = (type || '').toLowerCase();
    if (t.includes('jacket') || t.includes('coat') || t.includes('outer') || t.includes('zip') || t.includes('sweater') || t.includes('pullover') || t.includes('hoodie') || t.includes('polo') || t.includes('shirt') || t.includes('t-shirt') || t.includes('top')) return 'outerwear';
    if (t.includes('pant') || t.includes('bottom') || t.includes('trouser') || t.includes('jeans')) return 'pants';
    if (t.includes('shoe') || t.includes('sneaker') || t.includes('boot')) return 'shoes';
    return t;
  };

  // Backend-compatible macroCategory matching the edge function's slot model:
  // 'top' (base layer), 'outerwear' (jacket/blazer/sweater), 'bottom', 'shoes'
  const getBackendMacroCategory = (type: string, category?: string) => {
    const t = `${type || ''} ${category || ''}`.toLowerCase();
    if (t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear/)) return 'outerwear';
    if (t.match(/shirt|t-shirt|tee|blouse|polo|tops?(?:\b)/)) return 'top';
    if (t.match(/pant|trouser|jeans?|bottom|shorts?|skirt/)) return 'bottom';
    if (t.match(/shoe|sneaker|boot|loafer|sandal/)) return 'shoes';
    if (t.match(/dress/)) return 'top';
    return 'top'; // fallback keeps the backend happy
  };

  // ── Weather Fetch ────────────────────────────────────────────────
  const fetchWeather = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return;
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
      }
    } catch (e) {
      console.warn('Weather fetch error', e);
    }
  };

  useEffect(() => {
    fetchWeather();
  }, []);

  // ── 4-Slot Outfit Normalisation ──────────────────────────────────
  // Ensures every outfit has: main-top, second-top (layer), pants, shoes
  const normalizeTo4Slots = (items: OutfitItem[], styleId?: string): OutfitItem[] => {
    const pool = source === 'shop'
      ? liveShopMapped
      : [...wardrobeItems, ...liveShopMapped];
    const candidatePool = pool.map((it: any) => ({
      id: it.id || it.imageUrl || it.name,
      name: it.name || it.type || 'Item',
      image: (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '',
      imageUrl: it.imageUrl || (typeof it.image === 'string' ? it.image : ''),
      type: it.type || it.category || 'top',
      macroCategory: getBackendMacroCategory(it.type || '', it.category || ''),
      color: it.color || 'neutral',
      brand: it.brand || '',
    }));

    const resolveImage = (it: any) => (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '';

    // Step 1: place incoming items into buckets by macroCategory
    const incoming = (items || []).map((it) => ({
      ...it,
      image: it.image || it.imageUrl || '',
      macroCategory: getBackendMacroCategory(it.type || '', it.category || ''),
    }));

    const topBase = incoming.find(i => i.macroCategory === 'top') || incoming.find(i => ['top', 'outerwear'].includes(i.macroCategory || ''));
    const outerLayer = incoming.find(i => i.macroCategory === 'outerwear') || incoming.find(i => ['jacket', 'coat', 'blazer', 'hoodie', 'cardigan', 'sweater'].some(k => (i.type || '').toLowerCase().includes(k)));
    const bottoms = incoming.find(i => i.macroCategory === 'bottom');
    const shoes = incoming.find(i => i.macroCategory === 'shoes');

    const needsLayer = (weather?.temp != null && weather.temp < 18) ||
      /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test((weather?.condition || '').toLowerCase());
    const styleNeedsLayer = styleId === 'old_money' || styleId === 'business_casual' || styleId === 'streetwear';

    const pickFromPool = (cat: string) => {
      const found = candidatePool.find(c => c.macroCategory === cat);
      if (found) return { ...found, image: found.image || found.imageUrl || '' };
      const fallback = candidatePool[0];
      return fallback ? { ...fallback, image: fallback.image || fallback.imageUrl || '' } : { id: 'fallback', name: 'Item', image: '', type: 'top', macroCategory: 'top' };
    };

    const slotMainTop = topBase || pickFromPool('top');
    const slotPants = bottoms || pickFromPool('bottom');
    const slotShoes = shoes || pickFromPool('shoes');

    // For second-top, if weather or style demands layering, pick outerwear; otherwise pick another top
    let slotSecondTop = outerLayer;
    if (!slotSecondTop) {
      if (needsLayer || styleNeedsLayer) {
        slotSecondTop = pickFromPool('outerwear');
      } else {
        slotSecondTop = candidatePool.find(c => c.macroCategory === 'top' && c.id !== slotMainTop.id) || slotMainTop;
      }
    }

    return [
      { ...slotMainTop, image: resolveImage(slotMainTop) || slotMainTop.image || '' },
      { ...slotSecondTop, image: resolveImage(slotSecondTop) || slotSecondTop.image || '' },
      { ...slotPants, image: resolveImage(slotPants) || slotPants.image || '' },
      { ...slotShoes, image: resolveImage(slotShoes) || slotShoes.image || '' },
    ];
  };

  // ── Add to Calendar ─────────────────────────────────────────────
  const addToCalendar = async (outfit: GeneratedOutfit, date: Date) => {
    try {
      const dateStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
      const mappedItems: CalendarOutfitItem[] = outfit.items.map(it => ({
        id: it.id || it.name || `${Date.now()}`,
        type: it.type || 'top',
        image: it.image || it.imageUrl || '',
        color: it.color,
        name: it.name,
      }));
      const log = createOutfitLog(dateStr, mappedItems, (selectedStyle as any) || 'casual');
      const raw = await AsyncStorage.getItem('outfitLogs');
      const logs: Record<string, OutfitLog> = raw ? JSON.parse(raw) : {};
      logs[dateStr] = log;
      await AsyncStorage.setItem('outfitLogs', JSON.stringify(logs));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setCalendarVisible(false);
      setCalendarOutfit(null);
    } catch (e) {
      console.error('Calendar save error', e);
    }
  };

  const generateOutfits = async (overrideStyle?: string) => {
    const styleToUse = overrideStyle || selectedStyle;

    if (!isMounted.current) return;
    setLoading(true);
    setError('');
    setOutfits([]);

    const selectedClothing: any[] = [];

    // Build a clean payload for the backend — strip non-serialisable fields
    // (require() numbers) and ensure every item has macroCategory + imageUrl.
    const payloadItems = wardrobeItems.map((item: any) => ({
      id: item.id,
      name: item.name || item.type || 'Clothing item',
      type: item.type || 'top',
      category: item.category || item.type || 'tops',
      color: item.color || 'neutral',
      brand: item.brand || '',
      description: item.description || item.name || '',
      imageUrl: item.imageUrl || (typeof item.image === 'string' ? item.image : '') || '',
      macroCategory: getBackendMacroCategory(item.type || '', item.category || ''),
    }));

    try {
      // Generate outfits with selected items and style preferences.
      // NOTE: do NOT set useProvidedWardrobeOnly:true — the edge function's
      // enrichment step (step 8) needs to match item ids against itemMap to
      // attach real imageUrls. When wardrobeItems is provided the edge fn
      // already uses it as the candidate pool (step 3), so this is safe.
      const { data, error: invokeError } = await supabase.functions.invoke('generate-outfits', {
        body: {
          stylePreferences: styleToUse,
          occasion: 'Any',
          selectedItemIds: [],
          wardrobeItems: payloadItems,
          weather: weather ?? undefined,
          limit: 3,
        },
      });

      if (invokeError) throw invokeError;

      if (data && data.success && data.outfits && data.outfits.length > 0) {
        // Post-process backend output: normalize to strict 4-slot structure
        const cleanedOutfits = data.outfits.map((outfit: any, index: number) => {
          const normalized = normalizeTo4Slots(outfit.items || [], styleToUse);
          return {
            ...outfit,
            items: normalized,
            mainImage: normalized[0]?.image || outfit.mainImage || '',
            matchScore: outfit.confidence ?? outfit.matchScore ?? 0.9,
            stylingTips: Array.isArray(outfit.stylingTips)
              ? outfit.stylingTips.join(' · ')
              : (outfit.stylingTips || ''),
            weather,
          };
        });

        if (isMounted.current) {
          setOutfits(cleanedOutfits);
          setLoading(false);
        }
        return;
      }

      // Backend returned success=false or empty outfits — fall through to local fallback
      throw new Error(data?.error || 'No outfits returned from AI');
    } catch (err: any) {
      console.error('[AIOutfitmaker] Outfit generation error:', err);

      // Local fallback — normalize wardrobe into 4 slots
      const demoItems = normalizeTo4Slots(wardrobeItems.slice(0, 12), styleToUse);
      const firstImageUrl = demoItems[0]?.image || '';

      if (fallbackTimer.current) clearTimeout(fallbackTimer.current);
      fallbackTimer.current = setTimeout(() => {
        if (!isMounted.current) return;
        setOutfits([{
          id: `demo_${Date.now()}`,
          mainImage: firstImageUrl,
          matchScore: 0.92,
          description: `A curated ${styleToUse.replace(/_/g, ' ')} look styled from your wardrobe.`,
          items: demoItems,
          stylingTips: 'Pair this with some simple accessories to complete the look.',
          weather,
        }]);
        setLoading(false);
        setError('');
      }, 800);
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
            <TouchableOpacity onPress={() => setOutfits([])} style={styles.backButton}>
              <Ionicons name="chevron-back" size={26} color={D.textPrimary} />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Your Outfits</Text>
            <View style={{ width: 40 }} />
          </View>

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 32 }}>
            {outfits.map((outfit, index) => (
              <Animated.View key={outfit.id} style={[styles.glassCard, { marginTop: index === 0 ? 8 : 24 }]}>
                <BlurView intensity={Platform.OS === 'ios' ? 48 : 90} tint="light" style={StyleSheet.absoluteFill} />
                <LinearGradient colors={['rgba(255,255,255,0.45)', 'rgba(255,255,255,0.15)']} style={StyleSheet.absoluteFill} />

                <View style={styles.cardHeaderRow}>
                  {outfit.weather ? (
                    <View style={styles.weatherChip}>
                      <Ionicons name="cloud-outline" size={14} color={D.textSecondary} />
                      <Text style={styles.weatherChipText}>{outfit.weather.temp}°C · {outfit.weather.condition}</Text>
                    </View>
                  ) : <View />}
                  <LinearGradient colors={[D.accent, '#5B7CF9']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.matchPill}>
                    <Text style={styles.matchPillText}>{Math.round(outfit.matchScore * 100)}% Match</Text>
                  </LinearGradient>
                </View>

                <View style={styles.slotGrid}>
                  {outfit.items.slice(0, 4).map((item, idx) => {
                    const slotLabels = ['Main Top', 'Layer', 'Pants', 'Shoes'];
                    return (
                      <View key={`${outfit.id}-${idx}`} style={styles.slotCell}>
                        <Image
                          source={typeof item.image === 'string' && item.image ? { uri: item.image } : require('../assets/images/basic_cardigan.png')}
                          style={styles.slotImage}
                          resizeMode="cover"
                        />
                        <View style={styles.slotLabelWrap}>
                          <Text style={styles.slotLabel}>{slotLabels[idx]}</Text>
                          <Text style={styles.slotName} numberOfLines={1}>{item.name || 'Item'}</Text>
                        </View>
                      </View>
                    );
                  })}
                </View>

                <Text style={styles.outfitDesc}>{outfit.description}</Text>

                <View style={styles.tipCard}>
                  <LinearGradient colors={D.panelHighlight} style={StyleSheet.absoluteFill} />
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <Ionicons name="sparkles-outline" size={16} color={D.accent} />
                    <Text style={styles.tipCardText}>{outfit.stylingTips}</Text>
                  </View>
                </View>

                <View style={styles.actionRow}>
                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={styles.primaryAction}
                    onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); setCalendarOutfit(outfit); setCalendarDate(new Date()); setCalendarVisible(true); }}
                  >
                    <LinearGradient colors={[D.accent, '#5B7CF9']} style={StyleSheet.absoluteFill} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} />
                    <Ionicons name="calendar-outline" size={18} color="#fff" />
                    <Text style={styles.primaryActionText}>Add to Calendar</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    activeOpacity={0.85}
                    style={styles.secondaryAction}
                    onPress={() => { Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success); navigation.navigate('Main' as never); }}
                  >
                    <Ionicons name="heart-outline" size={18} color={D.textPrimary} />
                    <Text style={styles.secondaryActionText}>Wishlist</Text>
                  </TouchableOpacity>
                </View>
              </Animated.View>
            ))}
          </ScrollView>
        </SafeAreaView>

        <Modal animationType="slide" transparent visible={calendarVisible} onRequestClose={() => setCalendarVisible(false)}>
          <View style={styles.modalOverlay}>
            <BlurView intensity={Platform.OS === 'ios' ? 60 : 100} tint="light" style={StyleSheet.absoluteFill} />
            <View style={styles.modalCard}>
              <Text style={styles.modalTitle}>Save to Calendar</Text>
              <Text style={styles.modalSubtitle}>Pick a date for this outfit</Text>
              <View style={styles.datePickerRow}>
                <TouchableOpacity onPress={() => setCalendarDate(new Date(calendarDate.getTime() - 86400000))}>
                  <Ionicons name="chevron-back" size={24} color={D.textPrimary} />
                </TouchableOpacity>
                <Text style={styles.datePickerValue}>{calendarDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}</Text>
                <TouchableOpacity onPress={() => setCalendarDate(new Date(calendarDate.getTime() + 86400000))}>
                  <Ionicons name="chevron-forward" size={24} color={D.textPrimary} />
                </TouchableOpacity>
              </View>
              <View style={styles.modalActions}>
                <TouchableOpacity style={styles.modalCancel} onPress={() => setCalendarVisible(false)}>
                  <Text style={styles.modalCancelText}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.modalConfirm}
                  onPress={() => calendarOutfit && addToCalendar(calendarOutfit, calendarDate)}
                >
                  <LinearGradient colors={[D.accent, '#5B7CF9']} style={StyleSheet.absoluteFill} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} />
                  <Text style={styles.modalConfirmText}>Save</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
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
        edges={['top', 'left', 'right']}
        style={{ flex: 1 }}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={28} color={LiquidGlass2026Theme.colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>AI Stylist</Text>
          <Ionicons name="sparkles" size={24} color="#F59E0B" />
        </View>

        {/* Title Area */}
        <View style={{ paddingHorizontal: 20, paddingTop: 36, paddingBottom: 24 }}>
          <Text style={styles.sectionTitle}>
            Discover Your Vibe
          </Text>
          <Text style={styles.sectionSubtitle}>
            Tap a style card below and AI will instantly build a complete look from your wardrobe.
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
      </SafeAreaView>

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
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
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
    gap: 8,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
  },
  primaryActionText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '700',
  },
  secondaryAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    backgroundColor: 'rgba(255,255,255,0.55)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  secondaryActionText: {
    color: LiquidGlass2026Theme.colors.text.primary,
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '600',
  },

  // ── Calendar Modal Styles ──────────────────────────────────────
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalCard: {
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    paddingHorizontal: 24,
    paddingTop: 28,
    paddingBottom: 36,
    marginHorizontal: 0,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.5)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
  modalTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    textAlign: 'center',
    marginBottom: 4,
  },
  modalSubtitle: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: 20,
  },
  datePickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255,255,255,0.6)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    paddingVertical: 14,
    paddingHorizontal: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  datePickerValue: {
    ...LiquidGlass2026Theme.typography.scale.bodyLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  modalCancel: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    backgroundColor: 'rgba(255,255,255,0.6)',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  modalCancelText: {
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  modalConfirm: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
    alignItems: 'center',
  },
  modalConfirmText: {
    color: '#FFFFFF',
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    fontWeight: '700',
  },
});

export default AIOutfitGenerator;
