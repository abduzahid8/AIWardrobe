/**
 * ProfileScreen — iOS liquid glass rebuild
 */

import React, { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ViewStyle,
} from 'react-native';
import { BlurView } from 'expo-blur';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Animated, {
  Extrapolation,
  FadeIn,
  FadeInDown,
  FadeOut,
  SlideInUp,
  SlideOutDown,
  interpolate,
  useAnimatedScrollHandler,
  useAnimatedStyle,
  useSharedValue,
} from 'react-native-reanimated';
import { useAppNavigation } from '../hooks/useAppNavigation';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import useWardrobeStore from '../store/wardrobeStore';
import useTryOnLooksStore from '../store/tryOnLooksStore';
import { useShopCatalog } from '../hooks/useShopCatalog';
import { useTheme } from '../src/theme/ThemeContext';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';
import useSubscriptionStore from '../store/subscriptionStore';

const { width, height } = Dimensions.get('window');
const HERO_HEIGHT = Math.min(height * 0.43, 390);
const LOOK_CARD_GAP = 14;
const LOOK_CARD_WIDTH = (width - 40 - LOOK_CARD_GAP) / 2;

type IconName = keyof typeof Ionicons.glyphMap;
type BlurTint = 'light' | 'dark';
type ImageSrc = string | number;

interface SavedOutfit {
  _id: string;
  date?: string;
  occasion?: string;
  itemImages: ImageSrc[];
  image?: ImageSrc;
  isTryOn?: boolean;
  tryOnGarmentName?: string;
  tryOnGarmentBrand?: string;
}

function titleCase(value: string) {
  if (!value) return value;
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

function useDesignTokens() {
  const { colors, isDark } = useTheme();

  return {
    isDark,
    tint: (isDark ? 'dark' : 'light') as BlurTint,
    bg: colors.background,
    glass: isDark ? 'rgba(17, 20, 30, 0.58)' : 'rgba(255, 255, 255, 0.56)',
    glassStrong: isDark ? 'rgba(14, 16, 26, 0.76)' : 'rgba(255, 255, 255, 0.76)',
    glassMuted: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.30)',
    glassBorder: isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(255, 255, 255, 0.68)',
    glassBorderSoft: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(10, 25, 49, 0.06)',
    separator: isDark ? 'rgba(255, 255, 255, 0.10)' : 'rgba(10, 25, 49, 0.08)',
    text: colors.text.primary,
    textSub: colors.text.secondary,
    textMute: colors.text.muted,
    accent: isDark ? '#A8C0DA' : '#12385F',
    accentStart: isDark ? '#446B95' : '#2A537F',
    accentEnd: isDark ? '#1C3654' : '#0D2743',
    accentSoft: isDark ? 'rgba(126, 162, 201, 0.18)' : 'rgba(18, 56, 95, 0.12)',
    accentSoftStrong: isDark ? 'rgba(126, 162, 201, 0.28)' : 'rgba(42, 83, 127, 0.20)',
    danger: colors.error,
    success: colors.success,
    heroGradient: (isDark
      ? ['#142338', '#0D1828', '#070C15']
      : ['#F2F7FC', '#ECF3FA', '#FAFCFF']) as readonly [string, string, string],
    panelHighlight: (isDark
      ? ['rgba(255,255,255,0.12)', 'rgba(255,255,255,0.02)']
      : ['rgba(255,255,255,0.82)', 'rgba(255,255,255,0.16)']) as readonly [string, string],
    buttonHighlight: (isDark
      ? ['rgba(255,255,255,0.08)', 'rgba(255,255,255,0.02)']
      : ['rgba(255,255,255,0.70)', 'rgba(255,255,255,0.12)']) as readonly [string, string],
    orbPrimary: isDark ? 'rgba(68, 107, 149, 0.30)' : 'rgba(42, 83, 127, 0.18)',
    orbSecondary: isDark ? 'rgba(122, 161, 203, 0.18)' : 'rgba(121, 158, 198, 0.14)',
    orbWarm: isDark ? 'rgba(166, 191, 219, 0.12)' : 'rgba(188, 210, 231, 0.22)',
    shadow: isDark ? '#000000' : '#4C6076',
    overlay: isDark ? 'rgba(7, 9, 16, 0.74)' : 'rgba(16, 18, 28, 0.34)',
    previewOverlay: 'rgba(5, 7, 15, 0.58)',
    white: '#FFFFFF',
    fieldText: isDark ? '#FFFFFF' : '#0F172A',
    darkBackdrop: 'rgba(6, 9, 18, 0.86)',
    footerGradient: ['rgba(8, 10, 18, 0)', 'rgba(8, 10, 18, 0.58)', 'rgba(8, 10, 18, 0.92)'] as readonly [string, string, string],
  };
}

type DTokens = ReturnType<typeof useDesignTokens>;

const ProfileScreen = () => {
  const D = useDesignTokens();
  const styles = useMemo(() => createStyles(D), [D]);
  const navigation = useAppNavigation();
  const insets = useSafeAreaInsets();
  const { requireFeature: requireSubFeature } = useSubscriptionGate();

  const { user, logout, deleteAccount, fetchUser } = useAuthStore();
  const wardrobeItems = useWardrobeStore((state) => state.items);
  const storeOutfits = useWardrobeStore((state) => state.outfits);
  const wearLogs = useWardrobeStore((state) => state.wearLogs);
  const tryOnLooks = useTryOnLooksStore((state) => state.looks);
  const removeTryOnLook = useTryOnLooksStore((state) => state.removeLook);

  const { effectiveTier, isTrialActive } = useSubscriptionStore();
  const { items: liveShopCatalog } = useShopCatalog();

  const [activeTab, setActiveTab] = useState<'looks' | 'trips'>('looks');
  const [showEditProfile, setShowEditProfile] = useState(false);
  const [editName, setEditName] = useState('');
  const [editAvatar, setEditAvatar] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [cloudOutfits, setCloudOutfits] = useState<SavedOutfit[]>([]);
  const [loading, setLoading] = useState(false);
  const [previewLook, setPreviewLook] = useState<SavedOutfit | null>(null);

  const scrollY = useSharedValue(0);
  const scrollHandler = useAnimatedScrollHandler((event) => {
    scrollY.value = event.contentOffset.y;
  });

  const triggerLightHaptic = () => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const userName = user?.username || 'Your Name';
  const userAvatar: string | undefined =
    typeof user?.profile_image === 'string' && user.profile_image.length > 0
      ? user.profile_image
      : undefined;
  const userEmail = user?.email || '';
  
  // Use effectiveTier for logic (trial counts as Pro access)
  const isPro = effectiveTier !== 'free';
  const planLabel = isTrialActive 
    ? 'Free Trial' 
    : isPro 
      ? titleCase(effectiveTier) 
      : 'Free';
      
  const topPlanLabel = isTrialActive
    ? '7-Day Free Trial'
    : isPro 
      ? `${planLabel} Plan` 
      : 'Free Plan';

  const localOutfits = useMemo<SavedOutfit[]>(() => {
    const shopMap = new Map(liveShopCatalog.map((item) => [item.id, item.imageUrl as ImageSrc]));
    const isValidUri = (value?: string) =>
      typeof value === 'string' &&
      (value.startsWith('http') || value.startsWith('file://') || value.startsWith('data:'));

    const resolveImageSrc = (id: string): ImageSrc | undefined => {
      const wardrobeItem = wardrobeItems.find((item) => item.id === id);
      if (wardrobeItem?.imageUrl) return wardrobeItem.imageUrl;

      const shopItem = shopMap.get(id);
      if (shopItem !== undefined) return shopItem;

      if (isValidUri(id)) return id;
      return undefined;
    };

    return storeOutfits
      .filter((outfit) => outfit.saved)
      .map((outfit) => {
        const coverImage =
          typeof outfit.previewImageUrl === 'string' && isValidUri(outfit.previewImageUrl)
            ? outfit.previewImageUrl
            : undefined;

        const itemImages = outfit.itemIds
          .map(resolveImageSrc)
          .filter((src): src is ImageSrc => src !== undefined);

        return {
          _id: outfit.id,
          date: outfit.createdAt?.split('T')[0],
          occasion: typeof outfit.occasion === 'string' ? outfit.occasion : undefined,
          image: coverImage ?? itemImages[0],
          itemImages,
        };
      });
  }, [storeOutfits, wardrobeItems]);

  const fetchOutfits = useCallback(async () => {
    if (!user?.id) return;

    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('saved_outfits')
        .select('id, date, occasion, items, created_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) return;

      if (data) {
        setCloudOutfits(
          data.map((outfit) => {
            const rawItems = Array.isArray(outfit.items)
              ? (outfit.items as Array<{ image?: unknown }>)
              : [];
            const itemImages: ImageSrc[] = rawItems.flatMap((item) =>
              typeof item?.image === 'string' && item.image.startsWith('http') ? [item.image] : []
            );

            return {
              _id: outfit.id,
              date: outfit.date || outfit.created_at?.split('T')[0],
              occasion: outfit.occasion,
              itemImages,
              image: itemImages[0],
            };
          })
        );
      }
    } catch {
      // Keep local data visible if the cloud fetch fails.
    } finally {
      setLoading(false);
    }
  }, [user?.id]);

  useFocusEffect(
    useCallback(() => {
      void fetchOutfits();
    }, [fetchOutfits])
  );

  const outfits = useMemo(() => {
    const localIds = new Set(localOutfits.map((outfit) => outfit._id));
    const cloudOnly = cloudOutfits.filter((outfit) => !localIds.has(outfit._id));

    const tryOnMapped: SavedOutfit[] = tryOnLooks.map((look) => ({
      _id: look.id,
      date: look.savedAt.split('T')[0],
      occasion: 'Virtual Try-On',
      image: look.resultUrl,
      itemImages: [],
      isTryOn: true,
      tryOnGarmentName: look.garmentName,
      tryOnGarmentBrand: look.garmentBrand,
    }));

    return [...tryOnMapped, ...localOutfits, ...cloudOnly];
  }, [localOutfits, cloudOutfits, tryOnLooks]);

  const streak = useMemo(() => {
    if (wearLogs.length === 0) return 0;

    const dates = [...new Set(wearLogs.map((log) => log.date))].sort().reverse();
    const today = new Date().toISOString().split('T')[0];
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

    if (dates[0] !== today && dates[0] !== yesterday) return 0;

    let nextStreak = 1;
    for (let index = 1; index < dates.length; index += 1) {
      const diff =
        (new Date(dates[index - 1]).getTime() - new Date(dates[index]).getTime()) / 86400000;

      if (Math.round(diff) === 1) {
        nextStreak += 1;
      } else {
        break;
      }
    }

    return nextStreak;
  }, [wearLogs]);

  const stats = useMemo(
    () => [
      { icon: 'shirt-outline' as IconName, label: 'Items', value: wardrobeItems.length },
      { icon: 'sparkles-outline' as IconName, label: 'Looks', value: outfits.length },
      { icon: 'flame-outline' as IconName, label: 'Streak', value: streak || '0' },
      { icon: 'checkmark-done-outline' as IconName, label: 'Wears', value: wearLogs.length },
    ],
    [wardrobeItems.length, outfits.length, streak, wearLogs.length]
  );

  const heroAnimStyle = useAnimatedStyle(() => ({
    transform: [
      {
        translateY: interpolate(
          scrollY.value,
          [0, HERO_HEIGHT],
          [0, -HERO_HEIGHT * 0.18],
          Extrapolation.CLAMP
        ),
      },
      {
        scale: interpolate(scrollY.value, [-100, 0, HERO_HEIGHT], [1.03, 1, 0.985], Extrapolation.CLAMP),
      },
    ],
    opacity: interpolate(scrollY.value, [0, HERO_HEIGHT * 0.62], [1, 0], Extrapolation.CLAMP),
  }));

  const stickyHeaderStyle = useAnimatedStyle(() => ({
    opacity: interpolate(scrollY.value, [HERO_HEIGHT * 0.72, HERO_HEIGHT * 0.92], [0, 1], Extrapolation.CLAMP),
    transform: [
      {
        translateY: interpolate(scrollY.value, [HERO_HEIGHT * 0.72, HERO_HEIGHT * 0.92], [-14, 0], Extrapolation.CLAMP),
      },
    ],
  }));

  const openEditProfile = useCallback(() => {
    setEditName(userName);
    setEditAvatar(null);
    setShowEditProfile(true);
    triggerLightHaptic();
  }, [userName]);

  const pickAvatar = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Allow photo library access to change your avatar.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setEditAvatar(result.assets[0].uri);
    }
  };

  const handleSaveProfile = async () => {
    if (!user?.id) return;

    setSaving(true);
    try {
      let avatarUrl = user.profile_image;

      if (editAvatar) {
        const ext = editAvatar.split('.').pop() || 'jpg';
        const fileName = `${user.id}/avatar_${Date.now()}.${ext}`;
        const blob = await (await fetch(editAvatar)).blob();
        const { error: uploadError } = await supabase.storage
          .from('avatars')
          .upload(fileName, blob, { contentType: `image/${ext}`, upsert: true });

        if (!uploadError) {
          const { data } = supabase.storage.from('avatars').getPublicUrl(fileName);
          avatarUrl = data.publicUrl;
        }
      }

      const updates: Record<string, unknown> = {};
      if (editName.trim() && editName.trim() !== userName) updates.username = editName.trim();
      if (avatarUrl !== user.profile_image) updates.profile_image = avatarUrl;

      if (Object.keys(updates).length > 0) {
        const { error } = await supabase.from('profiles').update(updates).eq('id', user.id);
        if (error) throw error;

        await fetchUser();
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }

      setShowEditProfile(false);
    } catch (error: unknown) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to update profile.'));
    } finally {
      setSaving(false);
    }
  };

  const handleLogout = () => {
    Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Sign Out',
        style: 'destructive',
        onPress: async () => {
          await AsyncStorage.removeItem('userToken');
          logout();
        },
      },
    ]);
  };

  const handleDeleteAccount = () => {
    Alert.alert('Delete Account', 'Permanently delete your account and all associated data?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () =>
          Alert.alert('Confirm Delete', 'This action cannot be undone.', [
            { text: 'Keep Account', style: 'cancel' },
            {
              text: 'Delete Everything',
              style: 'destructive',
              onPress: async () => {
                try {
                  await deleteAccount();
                  await AsyncStorage.removeItem('userToken');
                } catch (error: unknown) {
                  Alert.alert('Error', getErrorMessage(error, 'Failed to delete account.'));
                }
              },
            },
          ]),
      },
    ]);
  };

  const GlassPanel = ({
    children,
    style,
    radius = 30,
    intensity = 42,
  }: {
    children: React.ReactNode;
    style?: ViewStyle | ViewStyle[];
    radius?: number;
    intensity?: number;
  }) => (
    <View style={[styles.glassShadow, { borderRadius: radius }, style]}>
      <View style={[styles.glassPanel, { borderRadius: radius }]}>
        <BlurView
          intensity={Platform.OS === 'ios' ? intensity : 100}
          tint={D.tint}
          style={StyleSheet.absoluteFillObject}
        />
        <LinearGradient
          colors={D.panelHighlight}
          start={{ x: 0.08, y: 0 }}
          end={{ x: 0.92, y: 1 }}
          style={StyleSheet.absoluteFillObject}
        />
        <View style={[styles.glassScrim, { borderRadius: radius }]} />
        <View style={styles.glassContent}>{children}</View>
      </View>
    </View>
  );

  const SecondaryGlassButton = ({
    label,
    icon,
    onPress,
    danger = false,
    style,
  }: {
    label: string;
    icon: IconName;
    onPress: () => void;
    danger?: boolean;
    style?: ViewStyle | ViewStyle[];
  }) => (
    <TouchableOpacity
      activeOpacity={0.85}
      style={style}
      onPress={() => {
        triggerLightHaptic();
        onPress();
      }}
    >
      <View style={styles.inlineGlassButton}>
        <BlurView
          intensity={Platform.OS === 'ios' ? 34 : 100}
          tint={D.tint}
          style={StyleSheet.absoluteFillObject}
        />
        <LinearGradient
          colors={D.buttonHighlight}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFillObject}
        />
        <View style={styles.inlineGlassButtonContent}>
          <Ionicons name={icon} size={16} color={danger ? D.danger : D.text} />
          <Text style={[styles.inlineGlassButtonText, danger && { color: D.danger }]}>{label}</Text>
        </View>
      </View>
    </TouchableOpacity>
  );

  const PrimaryGradientButton = ({
    label,
    icon,
    onPress,
    style,
  }: {
    label: string;
    icon: IconName;
    onPress: () => void;
    style?: ViewStyle | ViewStyle[];
  }) => (
    <TouchableOpacity
      activeOpacity={0.85}
      style={style}
      onPress={() => {
        triggerLightHaptic();
        onPress();
      }}
    >
      <LinearGradient
        colors={[D.accentStart, D.accentEnd]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.primaryButton}
      >
        <Ionicons name={icon} size={16} color={D.white} />
        <Text style={styles.primaryButtonText}>{label}</Text>
      </LinearGradient>
    </TouchableOpacity>
  );

  const MenuRow = ({
    icon,
    label,
    onPress,
    trailing,
    danger = false,
  }: {
    icon: IconName;
    label: string;
    onPress: () => void;
    trailing?: string;
    danger?: boolean;
  }) => (
    <Pressable
      style={({ pressed }) => [styles.menuRow, pressed && styles.menuRowPressed]}
      onPress={() => {
        triggerLightHaptic();
        onPress();
      }}
    >
      <View style={[styles.menuLeading, trailing ? styles.menuLeadingWithTrailing : styles.menuLeadingCompact]}>
        <View style={[styles.menuIconWrap, danger && styles.menuIconWrapDanger]}>
          <Ionicons name={icon} size={17} color={danger ? D.danger : D.textSub} />
        </View>

        <View style={styles.menuCopy}>
          <Text style={[styles.menuTitle, danger && { color: D.danger }]} numberOfLines={1}>
            {label}
          </Text>
        </View>
      </View>

      <View style={styles.menuTrailing}>
        {trailing ? (
          <Text style={[styles.menuTrailingText, danger && { color: D.danger }]} numberOfLines={1}>
            {trailing}
          </Text>
        ) : null}
        <Ionicons name="chevron-forward" size={16} color={danger ? D.danger : D.textSub} />
      </View>
    </Pressable>
  );

  const renderLookCard = (outfit: SavedOutfit) => {
    const cardTitle = outfit.isTryOn
      ? `${outfit.tryOnGarmentBrand ? `${outfit.tryOnGarmentBrand} · ` : ''}${
          outfit.tryOnGarmentName || 'AI Look'
        }`
      : outfit.occasion || 'Saved Look';

    return (
      <TouchableOpacity
        key={outfit._id}
        style={styles.lookCardShadow}
        activeOpacity={0.88}
        onPress={() => {
          triggerLightHaptic();
          setPreviewLook(outfit);
        }}
      >
        <View style={styles.lookCard}>
          {outfit.image !== undefined ? (
            <Image
              source={typeof outfit.image === 'number' ? outfit.image : { uri: outfit.image as string }}
              style={styles.lookImage}
            />
          ) : outfit.itemImages.length > 0 ? (
            outfit.itemImages.length === 1 ? (
              <Image
                source={
                  typeof outfit.itemImages[0] === 'number'
                    ? outfit.itemImages[0]
                    : { uri: outfit.itemImages[0] as string }
                }
                style={styles.lookImage}
              />
            ) : (
              <View style={styles.collageGrid}>
                {outfit.itemImages.slice(0, 4).map((src, index) => (
                  <Image
                    key={`${outfit._id}-${index}`}
                    source={typeof src === 'number' ? src : { uri: src as string }}
                    style={styles.collageCell}
                  />
                ))}
              </View>
            )
          ) : (
            <View style={styles.lookEmpty}>
              <Ionicons name="shirt-outline" size={30} color="rgba(255,255,255,0.8)" />
            </View>
          )}

          <BlurView
            intensity={Platform.OS === 'ios' ? 32 : 100}
            tint="dark"
            style={styles.lookFooterBlur}
          />
          <LinearGradient
            colors={D.footerGradient}
            start={{ x: 0.5, y: 0 }}
            end={{ x: 0.5, y: 1 }}
            style={styles.lookFooterGradient}
          />

          <View style={styles.lookFooter}>
            <Text style={styles.lookTitle} numberOfLines={1}>
              {cardTitle}
            </Text>
            {outfit.date ? (
              <Text style={styles.lookDate} numberOfLines={1}>
                {outfit.date}
              </Text>
            ) : null}
          </View>

          {outfit.isTryOn ? (
            <View style={styles.lookBadge}>
              <Ionicons name="sparkles" size={10} color={D.white} />
              <Text style={styles.lookBadgeText}>AI</Text>
            </View>
          ) : null}
        </View>
      </TouchableOpacity>
    );
  };

  const renderLooksContent = () => {
    if (loading) {
      return (
        <GlassPanel style={styles.contentPanel}>
          <View style={styles.loadingBlock}>
            <ActivityIndicator color={D.accent} />
            <Text style={styles.loadingText}>Loading your looks...</Text>
          </View>
        </GlassPanel>
      );
    }

    if (outfits.length === 0) {
      return (
        <GlassPanel style={styles.contentPanel}>
          <View style={styles.emptyState}>
            <View style={styles.emptyIconWrap}>
              <LinearGradient
                colors={['rgba(255,255,255,0.94)', D.accentSoftStrong]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.emptyIconGradient}
              >
                <Ionicons name="sparkles-outline" size={28} color={D.accent} />
              </LinearGradient>
            </View>
            <Text style={styles.emptyTitle}>No saved looks</Text>
            <Text style={styles.emptySubtitle}>
              Create looks with AI or save your favorite combinations from your wardrobe.
            </Text>
            <PrimaryGradientButton
              label="New Look"
              icon="sparkles"
              onPress={() => navigation.navigate('AIOutfit')}
              style={styles.emptyAction}
            />
          </View>
        </GlassPanel>
      );
    }

    return <View style={styles.lookGrid}>{outfits.map(renderLookCard)}</View>;
  };

  const openTripPlanner = () => {
    // Pro / Max only — Free users see the paywall.
    if (!requireSubFeature('tripPlanner')) return;
    navigation.navigate('Calendar');
  };

  const renderTripsContent = () => (
    <GlassPanel style={styles.contentPanel}>
      <View style={styles.emptyState}>
        <View style={styles.emptyIconWrap}>
          <LinearGradient
            colors={['rgba(255,255,255,0.94)', D.accentSoftStrong]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.emptyIconGradient}
          >
            <Ionicons name="airplane-outline" size={28} color={D.accent} />
          </LinearGradient>
        </View>
        <Text style={styles.emptyTitle}>Plan trip outfits</Text>
        <Text style={styles.emptySubtitle}>
          Use the calendar to pick a travel date, generate an outfit with AI, and save it back to that day.
        </Text>
        <PrimaryGradientButton
          label="Open Calendar"
          icon="calendar-outline"
          onPress={openTripPlanner}
          style={styles.emptyAction}
        />
      </View>

      <View style={styles.tripFlowList}>
        <View style={styles.tripFlowStep}>
          <View style={styles.tripFlowIconWrap}>
            <Ionicons name="calendar-outline" size={16} color={D.accent} />
          </View>
          <View style={styles.tripFlowCopy}>
            <Text style={styles.tripFlowTitle}>1. Pick your travel day</Text>
            <Text style={styles.tripFlowSubtitle}>
              Select the trip date you want to plan for inside the outfit calendar.
            </Text>
          </View>
        </View>

        <View style={styles.tripFlowStep}>
          <View style={styles.tripFlowIconWrap}>
            <Ionicons name="sparkles-outline" size={16} color={D.accent} />
          </View>
          <View style={styles.tripFlowCopy}>
            <Text style={styles.tripFlowTitle}>2. Create outfit with AI</Text>
            <Text style={styles.tripFlowSubtitle}>
              Tap `Create Outfit with AI` on that day to open the generator with the date attached.
            </Text>
          </View>
        </View>

        <View style={styles.tripFlowStep}>
          <View style={styles.tripFlowIconWrap}>
            <Ionicons name="checkmark-done-outline" size={16} color={D.accent} />
          </View>
          <View style={styles.tripFlowCopy}>
            <Text style={styles.tripFlowTitle}>3. Save it to your plan</Text>
            <Text style={styles.tripFlowSubtitle}>
              In the results view, add the outfit to calendar and it will be saved on that selected date.
            </Text>
          </View>
        </View>
      </View>
    </GlassPanel>
  );

  return (
    <View style={styles.container}>
      <StatusBar
        translucent
        backgroundColor="transparent"
        barStyle={D.isDark ? 'light-content' : 'dark-content'}
      />

      <LinearGradient
        colors={D.heroGradient}
        start={{ x: 0.2, y: 0 }}
        end={{ x: 0.9, y: 1 }}
        style={styles.backgroundGradient}
      />
      <View style={[styles.orbLarge, { backgroundColor: D.orbPrimary }]} />
      <View style={[styles.orbMedium, { backgroundColor: D.orbSecondary }]} />
      <View style={[styles.orbSmall, { backgroundColor: D.orbWarm }]} />

      <Animated.View style={[styles.stickyHeaderWrap, stickyHeaderStyle, { top: insets.top + 4 }]}>
        <View style={styles.stickyHeaderPanel}>
          <BlurView
            intensity={Platform.OS === 'ios' ? 40 : 100}
            tint={D.tint}
            style={StyleSheet.absoluteFillObject}
          />
          <LinearGradient
            colors={D.buttonHighlight}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={StyleSheet.absoluteFillObject}
          />
          <View style={styles.stickyHeaderContent}>
            <Text style={styles.stickyHeaderTitle}>{userName}</Text>
            <View style={styles.stickyPlanChip}>
              <Ionicons name={isPro ? 'diamond-outline' : 'sparkles-outline'} size={11} color={D.textSub} />
              <Text style={styles.stickyPlanChipText}>{planLabel}</Text>
            </View>
          </View>
        </View>
      </Animated.View>

      <Animated.ScrollView
        onScroll={scrollHandler}
        scrollEventThrottle={16}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{
          paddingTop: insets.top + 16,
          paddingBottom: 120 + insets.bottom,
        }}
      >
        <Animated.View style={[styles.heroWrap, heroAnimStyle]}>
          <GlassPanel style={styles.heroCard} intensity={52} radius={34}>
            <View style={styles.heroTopRow}>
              <View style={styles.profileChip}>
                <Ionicons name="person-circle-outline" size={14} color={D.textSub} />
                <Text style={styles.profileChipText}>Account</Text>
              </View>

              <SecondaryGlassButton label="Edit" icon="create-outline" onPress={openEditProfile} />
            </View>

            <View style={styles.avatarWrapper}>
              <LinearGradient
                colors={['rgba(255,255,255,0.96)', D.accentSoftStrong]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.avatarHalo}
              >
                <LinearGradient
                  colors={[D.accentStart, D.accentEnd]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.avatarRing}
                >
                  {userAvatar ? (
                    <Image source={{ uri: userAvatar }} style={styles.avatarImage} />
                  ) : (
                    <View style={[styles.avatarImage, styles.avatarPlaceholder]}>
                      <Ionicons name="person" size={46} color={D.white} />
                    </View>
                  )}
                </LinearGradient>
              </LinearGradient>

              <TouchableOpacity style={styles.avatarCamera} activeOpacity={0.86} onPress={openEditProfile}>
                <BlurView
                  intensity={Platform.OS === 'ios' ? 36 : 100}
                  tint={D.tint}
                  style={StyleSheet.absoluteFillObject}
                />
                <LinearGradient
                  colors={[D.accentStart, D.accentEnd]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={StyleSheet.absoluteFillObject}
                />
                <Ionicons name="camera" size={12} color={D.white} />
              </TouchableOpacity>
            </View>

            <View style={styles.identityBlock}>
              <Text style={styles.heroName}>{userName}</Text>
              {userEmail ? <Text style={styles.heroEmail}>{userEmail}</Text> : null}

              <View style={styles.membershipBadge}>
                <Ionicons
                  name={isPro ? 'diamond-outline' : 'sparkles-outline'}
                  size={13}
                  color={isPro ? D.white : D.accent}
                />
                {isPro ? (
                  <LinearGradient
                    colors={[D.accentStart, D.accentEnd]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.membershipBadgeFill}
                  >
                    <Text style={styles.membershipBadgeText}>{topPlanLabel}</Text>
                  </LinearGradient>
                ) : (
                  <View style={styles.membershipBadgeGlass}>
                    <Text style={styles.membershipBadgeSubtleText}>{topPlanLabel}</Text>
                  </View>
                )}
              </View>
            </View>

            <View style={styles.heroActions}>
              {isPro ? (
                <PrimaryGradientButton
                  label="Manage Plan"
                  icon="diamond-outline"
                  onPress={() => navigation.navigate('Paywall')}
                  style={styles.heroActionPrimary}
                />
              ) : (
                <PrimaryGradientButton
                  label="Go Pro"
                  icon="rocket-outline"
                  onPress={() => navigation.navigate('Paywall')}
                  style={styles.heroActionPrimary}
                />
              )}

              <SecondaryGlassButton
                label="Insights"
                icon="stats-chart-outline"
                onPress={() => navigation.navigate('WardrobeAnalytics')}
                style={styles.heroActionSecondary}
              />
            </View>
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(40).duration(350)}>
          <GlassPanel style={styles.statsCard} radius={30}>
            <View style={styles.statsRow}>
              {stats.map((item, index) => (
                <React.Fragment key={item.label}>
                  <View style={styles.statCell}>
                    <View style={styles.statIconWrap}>
                      <Ionicons name={item.icon} size={15} color={D.textSub} />
                    </View>
                    <Text style={styles.statValue}>{item.value}</Text>
                    <Text style={styles.statLabel}>{item.label}</Text>
                  </View>
                  {index < stats.length - 1 ? <View style={styles.statDivider} /> : null}
                </React.Fragment>
              ))}
            </View>
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(75).duration(350)}>
          <GlassPanel style={styles.segmentedCard} radius={28}>
            <View style={styles.segmentedRow}>
              {(['looks', 'trips'] as const).map((tab) => {
                const isActive = activeTab === tab;
                return (
                  <Pressable
                    key={tab}
                    style={[styles.segmentedButton, isActive && styles.segmentedButtonActive]}
                    onPress={() => {
                      triggerLightHaptic();
                      setActiveTab(tab);
                    }}
                  >
                    {isActive ? (
                      <LinearGradient
                        colors={[D.accentStart, D.accentEnd]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={StyleSheet.absoluteFillObject}
                      />
                    ) : null}
                    <Text style={[styles.segmentedText, isActive && styles.segmentedTextActive]}>
                      {titleCase(tab)}
                    </Text>
                    <View style={[styles.segmentedBadge, isActive && styles.segmentedBadgeActive]}>
                      <Text style={[styles.segmentedBadgeText, isActive && styles.segmentedBadgeTextActive]}>
                        {tab === 'looks' ? outfits.length : 0}
                      </Text>
                    </View>
                  </Pressable>
                );
              })}
            </View>
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(100).duration(350)} style={styles.sectionBlock}>
          <View style={styles.sectionHeaderRow}>
            <View>
              <Text style={styles.sectionEyebrow}>{activeTab === 'looks' ? 'STYLE' : 'TRAVEL'}</Text>
              <Text style={styles.sectionTitle}>{activeTab === 'looks' ? 'Saved Looks' : 'Trips & Packing'}</Text>
            </View>
            {activeTab === 'looks' ? (
              <SecondaryGlassButton
                label="New"
                icon="add-outline"
                onPress={() => navigation.navigate('AIOutfit')}
              />
            ) : (
              <SecondaryGlassButton
                label="Open Calendar"
                icon="calendar-outline"
                onPress={openTripPlanner}
              />
            )}
          </View>

          {activeTab === 'looks' ? renderLooksContent() : renderTripsContent()}
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(130).duration(350)} style={styles.sectionBlock}>
          <Text style={styles.groupHeading}>ACCOUNT</Text>
          <GlassPanel radius={28}>
            <MenuRow
              icon="create-outline"
              label="Edit Profile"
              onPress={openEditProfile}
            />
            <View style={styles.menuSeparator} />
            <MenuRow
              icon="stats-chart-outline"
              label="Wardrobe Analytics"
              onPress={() => navigation.navigate('WardrobeAnalytics')}
            />
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(155).duration(350)} style={styles.sectionBlock}>
          <Text style={styles.groupHeading}>MEMBERSHIP</Text>
          <GlassPanel radius={28}>
            <MenuRow
              icon="diamond-outline"
              label="Subscription"
              trailing={planLabel}
              onPress={() => navigation.navigate('Paywall')}
            />
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(180).duration(350)} style={styles.sectionBlock}>
          <Text style={styles.groupHeading}>PREFERENCES</Text>
          <GlassPanel radius={28}>
            <MenuRow
              icon="notifications-outline"
              label="Notifications"
              onPress={() => Alert.alert('Coming soon', 'Notification settings are not available yet.')}
            />
            <View style={styles.menuSeparator} />
            <MenuRow
              icon="shield-checkmark-outline"
              label="Privacy Policy"
              onPress={() => navigation.navigate('PrivacyPolicy')}
            />
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(205).duration(350)} style={styles.sectionBlock}>
          <Text style={styles.groupHeading}>SESSION</Text>
          <GlassPanel radius={28}>
            <MenuRow
              icon="log-out-outline"
              label="Sign Out"
              onPress={handleLogout}
              danger
            />
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(230).duration(350)} style={styles.deleteAccountWrap}>
          <SecondaryGlassButton
            label="Delete Account"
            icon="trash-outline"
            onPress={handleDeleteAccount}
            danger
            style={styles.deleteButton}
          />
        </Animated.View>
      </Animated.ScrollView>

      <Modal
        visible={showEditProfile}
        transparent
        animationType="none"
        onRequestClose={() => setShowEditProfile(false)}
      >
        <KeyboardAvoidingView
          style={styles.modalRoot}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        >
          <Animated.View entering={FadeIn.duration(180)} exiting={FadeOut.duration(180)} style={styles.modalRoot}>
            <Pressable style={styles.modalBackdrop} onPress={() => setShowEditProfile(false)} />
          </Animated.View>

          <Animated.View
            entering={SlideInUp.springify().damping(22).stiffness(240).mass(0.85)}
            exiting={SlideOutDown.springify().damping(22).stiffness(240).mass(0.85)}
            style={[styles.sheetOuter, { paddingBottom: insets.bottom + 18 }]}
          >
            <View style={styles.sheetCard}>
              <BlurView
                intensity={Platform.OS === 'ios' ? 46 : 100}
                tint={D.tint}
                style={StyleSheet.absoluteFillObject}
              />
              <LinearGradient
                colors={D.panelHighlight}
                start={{ x: 0.1, y: 0 }}
                end={{ x: 0.9, y: 1 }}
                style={StyleSheet.absoluteFillObject}
              />
              <View style={styles.sheetScrim} />

              <View style={styles.sheetHandle} />
              <Text style={styles.sheetTitle}>Edit Account</Text>
              <Text style={styles.sheetSubtitle}>Update your photo, name, and account details.</Text>

              <TouchableOpacity style={styles.sheetAvatarWrap} activeOpacity={0.85} onPress={pickAvatar}>
                <LinearGradient
                  colors={['rgba(255,255,255,0.98)', D.accentSoftStrong]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.sheetAvatarHalo}
                >
                  {editAvatar || userAvatar ? (
                    <Image
                      source={{ uri: (editAvatar || userAvatar) as string }}
                      style={styles.sheetAvatar}
                    />
                  ) : (
                    <View style={[styles.sheetAvatar, styles.sheetAvatarPlaceholder]}>
                      <Ionicons name="person" size={52} color={D.accent} />
                    </View>
                  )}
                </LinearGradient>
                <View style={styles.sheetAvatarBadge}>
                  <LinearGradient
                    colors={[D.accentStart, D.accentEnd]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={StyleSheet.absoluteFillObject}
                  />
                  <Ionicons name="camera" size={13} color={D.white} />
                </View>
              </TouchableOpacity>

              <Text style={styles.fieldLabel}>Name</Text>
              <View style={styles.fieldCard}>
                <BlurView
                  intensity={Platform.OS === 'ios' ? 30 : 100}
                  tint={D.tint}
                  style={StyleSheet.absoluteFillObject}
                />
                <View style={styles.fieldScrim} />
                <TextInput
                  style={styles.fieldInput}
                  value={editName}
                  onChangeText={setEditName}
                  placeholder="Enter username"
                  placeholderTextColor={D.textMute}
                  autoCapitalize="none"
                  autoCorrect={false}
                />
              </View>

              <Text style={styles.fieldLabel}>Email</Text>
              <View style={styles.fieldCard}>
                <BlurView
                  intensity={Platform.OS === 'ios' ? 30 : 100}
                  tint={D.tint}
                  style={StyleSheet.absoluteFillObject}
                />
                <View style={styles.fieldScrim} />
                <View style={styles.readonlyRow}>
                  <Ionicons name="mail-outline" size={16} color={D.textSub} />
                  <Text style={styles.readonlyValue}>{userEmail || 'No email available'}</Text>
                </View>
              </View>

              <View style={styles.sheetActions}>
                <PrimaryGradientButton
                  label={saving ? 'Saving' : 'Save'}
                  icon={saving ? 'time-outline' : 'checkmark-outline'}
                  onPress={handleSaveProfile}
                  style={styles.sheetPrimaryAction}
                />
                <SecondaryGlassButton
                  label="Cancel"
                  icon="close-outline"
                  onPress={() => setShowEditProfile(false)}
                  style={styles.sheetSecondaryAction}
                />
              </View>

              {saving ? (
                <View style={styles.savingRow}>
                  <ActivityIndicator size="small" color={D.accent} />
                  <Text style={styles.savingText}>Saving updates...</Text>
                </View>
              ) : null}
            </View>
          </Animated.View>
        </KeyboardAvoidingView>
      </Modal>

      {previewLook ? (
        <Modal transparent animationType="fade" visible onRequestClose={() => setPreviewLook(null)}>
          <View style={styles.previewOverlay}>
            <Pressable style={StyleSheet.absoluteFillObject} onPress={() => setPreviewLook(null)} />

            <Animated.View entering={FadeIn.duration(220)} style={styles.previewShell}>
              <View style={styles.previewCard}>
                {previewLook.image !== undefined ? (
                  <Image
                    source={
                      typeof previewLook.image === 'number'
                        ? previewLook.image
                        : { uri: previewLook.image as string }
                    }
                    style={styles.previewImage}
                    resizeMode="cover"
                  />
                ) : previewLook.itemImages.length > 0 ? (
                  <View style={styles.previewCollage}>
                    {previewLook.itemImages.slice(0, 4).map((src, index) => (
                      <Image
                        key={`${previewLook._id}-${index}`}
                        source={typeof src === 'number' ? src : { uri: src as string }}
                        style={styles.previewCollageCell}
                      />
                    ))}
                  </View>
                ) : (
                  <View style={styles.previewFallback}>
                    <Ionicons name="shirt-outline" size={38} color="rgba(255,255,255,0.85)" />
                  </View>
                )}

                <BlurView
                  intensity={Platform.OS === 'ios' ? 38 : 100}
                  tint="dark"
                  style={styles.previewFooterBlur}
                />
                <LinearGradient
                  colors={D.footerGradient}
                  start={{ x: 0.5, y: 0 }}
                  end={{ x: 0.5, y: 1 }}
                  style={styles.previewFooterGradient}
                />

                <View style={styles.previewInfo}>
                  {previewLook.isTryOn ? (
                    <View style={styles.previewBadgeRow}>
                      <Ionicons name="sparkles" size={13} color={D.white} />
                      <Text style={styles.previewBadgeText}>AI Try-On</Text>
                    </View>
                  ) : null}
                  <Text style={styles.previewTitle} numberOfLines={2}>
                    {previewLook.isTryOn
                      ? `${previewLook.tryOnGarmentBrand ? `${previewLook.tryOnGarmentBrand} · ` : ''}${
                          previewLook.tryOnGarmentName || 'AI Look'
                        }`
                      : previewLook.occasion || 'Saved Look'}
                  </Text>
                  {previewLook.date ? <Text style={styles.previewDate}>{previewLook.date}</Text> : null}
                </View>

                <View style={styles.previewActions}>
                  {previewLook.isTryOn ? (
                    <SecondaryGlassButton
                      label="Try Again"
                      icon="sparkles-outline"
                      onPress={() => {
                        navigation.navigate('AITryOn');
                        setPreviewLook(null);
                      }}
                      style={styles.previewActionButton}
                    />
                  ) : null}

                  <SecondaryGlassButton
                    label={previewLook.isTryOn ? 'Remove' : 'Close'}
                    icon={previewLook.isTryOn ? 'trash-outline' : 'close-outline'}
                    onPress={() => {
                      if (previewLook.isTryOn) {
                        Alert.alert('Remove Look', 'Remove this AI try-on look?', [
                          { text: 'Cancel', style: 'cancel' },
                          {
                            text: 'Remove',
                            style: 'destructive',
                            onPress: () => {
                              removeTryOnLook(previewLook._id);
                              setPreviewLook(null);
                            },
                          },
                        ]);
                        return;
                      }

                      setPreviewLook(null);
                    }}
                    danger={previewLook.isTryOn}
                    style={styles.previewActionButton}
                  />
                </View>
              </View>
            </Animated.View>
          </View>
        </Modal>
      ) : null}
    </View>
  );
};

const createStyles = (D: DTokens) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: D.bg,
    },
    backgroundGradient: {
      ...StyleSheet.absoluteFillObject,
    },
    orbLarge: {
      position: 'absolute',
      width: 260,
      height: 260,
      borderRadius: 130,
      top: -20,
      right: -70,
      opacity: 0.95,
    },
    orbMedium: {
      position: 'absolute',
      width: 220,
      height: 220,
      borderRadius: 110,
      top: 190,
      left: -90,
      opacity: 0.8,
    },
    orbSmall: {
      position: 'absolute',
      width: 180,
      height: 180,
      borderRadius: 90,
      bottom: 180,
      right: -50,
      opacity: 0.6,
    },
    glassShadow: {
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 18 },
      shadowOpacity: D.isDark ? 0.42 : 0.16,
      shadowRadius: 28,
      elevation: 14,
    },
    glassPanel: {
      overflow: 'hidden',
      borderWidth: 1,
      borderColor: D.glassBorder,
      backgroundColor: D.glass,
    },
    glassScrim: {
      ...StyleSheet.absoluteFillObject,
      backgroundColor: D.glassStrong,
    },
    glassContent: {
      overflow: 'hidden',
    },
    stickyHeaderWrap: {
      position: 'absolute',
      left: 16,
      right: 16,
      zIndex: 100,
    },
    stickyHeaderPanel: {
      overflow: 'hidden',
      borderRadius: 20,
      borderWidth: 1,
      borderColor: D.glassBorder,
      backgroundColor: D.glassStrong,
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 14 },
      shadowOpacity: D.isDark ? 0.36 : 0.12,
      shadowRadius: 20,
      elevation: 12,
    },
    stickyHeaderContent: {
      paddingHorizontal: 16,
      paddingVertical: 10,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    stickyHeaderTitle: {
      fontSize: 15,
      fontWeight: '700',
      color: D.text,
      letterSpacing: -0.3,
    },
    stickyPlanChip: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      paddingHorizontal: 10,
      paddingVertical: 5,
      borderRadius: 999,
      backgroundColor: D.glassMuted,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    stickyPlanChipText: {
      fontSize: 11,
      fontWeight: '600',
      color: D.textSub,
    },
    heroWrap: {
      height: HERO_HEIGHT,
      justifyContent: 'flex-end',
      paddingHorizontal: 20,
      paddingBottom: 0,
    },
    heroCard: {
      minHeight: 300,
    },
    heroTopRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 16,
      paddingTop: 14,
    },
    profileChip: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      paddingHorizontal: 11,
      paddingVertical: 7,
      borderRadius: 999,
      backgroundColor: D.glassMuted,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    profileChipText: {
      fontSize: 11,
      fontWeight: '600',
      color: D.textSub,
    },
    avatarWrapper: {
      width: 118,
      height: 118,
      alignSelf: 'center',
      alignItems: 'center',
      justifyContent: 'center',
      marginTop: 14,
    },
    avatarHalo: {
      width: 118,
      height: 118,
      borderRadius: 59,
      alignItems: 'center',
      justifyContent: 'center',
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 14 },
      shadowOpacity: D.isDark ? 0.28 : 0.10,
      shadowRadius: 22,
      elevation: 10,
    },
    avatarRing: {
      width: 104,
      height: 104,
      borderRadius: 52,
      alignItems: 'center',
      justifyContent: 'center',
      padding: 3,
    },
    avatarImage: {
      width: 98,
      height: 98,
      borderRadius: 49,
      borderWidth: 1.5,
      borderColor: 'rgba(255,255,255,0.72)',
    },
    avatarPlaceholder: {
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: 'rgba(255,255,255,0.14)',
    },
    avatarCamera: {
      position: 'absolute',
      right: 2,
      bottom: 6,
      width: 30,
      height: 30,
      borderRadius: 15,
      overflow: 'hidden',
      alignItems: 'center',
      justifyContent: 'center',
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.52)',
    },
    identityBlock: {
      alignItems: 'center',
      paddingHorizontal: 24,
      marginTop: 12,
    },
    heroName: {
      fontSize: 28,
      fontWeight: '800',
      color: D.text,
      letterSpacing: -0.7,
    },
    heroEmail: {
      marginTop: 3,
      fontSize: 14,
      color: D.textSub,
      textAlign: 'center',
    },
    membershipBadge: {
      marginTop: 10,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      minHeight: 34,
    },
    membershipBadgeFill: {
      paddingHorizontal: 12,
      paddingVertical: 7,
      borderRadius: 999,
    },
    membershipBadgeText: {
      fontSize: 12,
      fontWeight: '700',
      color: D.white,
    },
    membershipBadgeGlass: {
      paddingHorizontal: 12,
      paddingVertical: 7,
      borderRadius: 999,
      backgroundColor: D.glassMuted,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    membershipBadgeSubtleText: {
      fontSize: 12,
      fontWeight: '700',
      color: D.accent,
    },
    heroActions: {
      flexDirection: 'row',
      gap: 10,
      paddingHorizontal: 16,
      paddingTop: 14,
      paddingBottom: 16,
    },
    heroActionPrimary: {
      flex: 1.05,
    },
    heroActionSecondary: {
      flex: 0.85,
    },
    primaryButton: {
      minHeight: 46,
      borderRadius: 999,
      paddingHorizontal: 16,
      paddingVertical: 12,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 8,
      shadowColor: D.accent,
      shadowOffset: { width: 0, height: 12 },
      shadowOpacity: 0.26,
      shadowRadius: 20,
      elevation: 10,
    },
    primaryButtonText: {
      fontSize: 14,
      fontWeight: '700',
      color: D.white,
      letterSpacing: -0.2,
    },
    inlineGlassButton: {
      minHeight: 40,
      borderRadius: 999,
      overflow: 'hidden',
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
      backgroundColor: D.glassMuted,
    },
    inlineGlassButtonContent: {
      minHeight: 40,
      paddingHorizontal: 14,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 8,
    },
    inlineGlassButtonText: {
      fontSize: 13,
      fontWeight: '600',
      color: D.text,
    },
    statsCard: {
      marginTop: 18,
      marginHorizontal: 20,
    },
    statsRow: {
      flexDirection: 'row',
      alignItems: 'stretch',
      paddingVertical: 16,
      paddingHorizontal: 12,
    },
    statCell: {
      flex: 1,
      alignItems: 'center',
      paddingHorizontal: 6,
    },
    statIconWrap: {
      width: 34,
      height: 34,
      borderRadius: 17,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.glassMuted,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
      marginBottom: 10,
    },
    statValue: {
      fontSize: 24,
      fontWeight: '800',
      color: D.text,
      letterSpacing: -0.7,
    },
    statLabel: {
      marginTop: 4,
      fontSize: 11,
      fontWeight: '600',
      color: D.textMute,
      textTransform: 'uppercase',
      letterSpacing: 0.7,
    },
    statDivider: {
      width: 1,
      backgroundColor: D.separator,
      marginVertical: 10,
    },
    segmentedCard: {
      marginHorizontal: 20,
      marginTop: 14,
    },
    segmentedRow: {
      flexDirection: 'row',
      padding: 6,
      gap: 8,
    },
    segmentedButton: {
      flex: 1,
      minHeight: 48,
      borderRadius: 999,
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'row',
      gap: 8,
      overflow: 'hidden',
    },
    segmentedButtonActive: {
      shadowColor: D.accent,
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.24,
      shadowRadius: 18,
      elevation: 8,
    },
    segmentedText: {
      fontSize: 15,
      fontWeight: '600',
      color: D.textSub,
    },
    segmentedTextActive: {
      color: D.white,
    },
    segmentedBadge: {
      minWidth: 22,
      height: 22,
      borderRadius: 11,
      alignItems: 'center',
      justifyContent: 'center',
      paddingHorizontal: 6,
      backgroundColor: D.glassMuted,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    segmentedBadgeActive: {
      backgroundColor: 'rgba(255,255,255,0.18)',
      borderColor: 'rgba(255,255,255,0.22)',
    },
    segmentedBadgeText: {
      fontSize: 11,
      fontWeight: '700',
      color: D.textSub,
    },
    segmentedBadgeTextActive: {
      color: D.white,
    },
    sectionBlock: {
      marginTop: 16,
      paddingHorizontal: 20,
    },
    sectionHeaderRow: {
      flexDirection: 'row',
      alignItems: 'flex-end',
      justifyContent: 'space-between',
      gap: 16,
      marginBottom: 14,
    },
    sectionEyebrow: {
      fontSize: 11,
      fontWeight: '700',
      color: D.textMute,
      letterSpacing: 1.1,
      marginBottom: 4,
    },
    sectionTitle: {
      fontSize: 22,
      fontWeight: '800',
      color: D.text,
      letterSpacing: -0.6,
    },
    contentPanel: {
      padding: 18,
    },
    loadingBlock: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 34,
      gap: 12,
    },
    loadingText: {
      fontSize: 14,
      color: D.textSub,
    },
    emptyState: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 24,
      paddingHorizontal: 16,
    },
    emptyIconWrap: {
      width: 76,
      height: 76,
      borderRadius: 38,
      marginBottom: 18,
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 12 },
      shadowOpacity: D.isDark ? 0.28 : 0.1,
      shadowRadius: 20,
      elevation: 8,
    },
    emptyIconGradient: {
      width: '100%',
      height: '100%',
      borderRadius: 38,
      alignItems: 'center',
      justifyContent: 'center',
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.58)',
    },
    emptyTitle: {
      fontSize: 22,
      fontWeight: '800',
      color: D.text,
      letterSpacing: -0.5,
    },
    emptySubtitle: {
      marginTop: 10,
      fontSize: 14,
      lineHeight: 21,
      textAlign: 'center',
      color: D.textSub,
      maxWidth: 280,
    },
    emptyAction: {
      marginTop: 22,
      minWidth: 160,
    },
    tripFlowList: {
      marginTop: 6,
      gap: 12,
    },
    tripFlowStep: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      gap: 12,
      padding: 14,
      borderRadius: 20,
      backgroundColor: D.isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.46)',
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    tripFlowIconWrap: {
      width: 32,
      height: 32,
      borderRadius: 16,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.accentSoft,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    tripFlowCopy: {
      flex: 1,
      gap: 4,
    },
    tripFlowTitle: {
      fontSize: 14,
      fontWeight: '700',
      color: D.text,
      letterSpacing: -0.2,
    },
    tripFlowSubtitle: {
      fontSize: 13,
      lineHeight: 19,
      color: D.textSub,
    },
    lookGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: LOOK_CARD_GAP,
    },
    lookCardShadow: {
      width: LOOK_CARD_WIDTH,
      borderRadius: 26,
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 18 },
      shadowOpacity: D.isDark ? 0.35 : 0.14,
      shadowRadius: 24,
      elevation: 12,
    },
    lookCard: {
      width: LOOK_CARD_WIDTH,
      height: LOOK_CARD_WIDTH * 1.36,
      borderRadius: 26,
      overflow: 'hidden',
      backgroundColor: D.glassStrong,
      borderWidth: 1,
      borderColor: D.glassBorder,
    },
    lookImage: {
      width: '100%',
      height: '100%',
      resizeMode: 'cover',
    },
    collageGrid: {
      flex: 1,
      flexDirection: 'row',
      flexWrap: 'wrap',
    },
    collageCell: {
      width: '50%',
      height: '50%',
      resizeMode: 'cover',
    },
    lookEmpty: {
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.darkBackdrop,
    },
    lookFooterBlur: {
      position: 'absolute',
      left: 0,
      right: 0,
      bottom: 0,
      height: 98,
    },
    lookFooterGradient: {
      position: 'absolute',
      left: 0,
      right: 0,
      bottom: 0,
      height: 112,
    },
    lookFooter: {
      position: 'absolute',
      left: 14,
      right: 14,
      bottom: 14,
    },
    lookTitle: {
      fontSize: 14,
      fontWeight: '700',
      color: D.white,
      letterSpacing: -0.2,
    },
    lookDate: {
      marginTop: 4,
      fontSize: 12,
      color: 'rgba(255,255,255,0.72)',
    },
    lookBadge: {
      position: 'absolute',
      top: 12,
      right: 12,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 4,
      paddingHorizontal: 10,
      paddingVertical: 6,
      borderRadius: 999,
      backgroundColor: D.accentStart,
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.24)',
    },
    lookBadgeText: {
      fontSize: 11,
      fontWeight: '700',
      color: D.white,
    },
    groupHeading: {
      marginBottom: 8,
      marginLeft: 2,
      fontSize: 12,
      fontWeight: '600',
      color: D.textMute,
      letterSpacing: 2.2,
    },
    menuRow: {
      position: 'relative',
      flexDirection: 'row',
      alignItems: 'center',
      height: 56,
      paddingLeft: 16,
      paddingRight: 20,
    },
    menuRowPressed: {
      opacity: 0.72,
    },
    menuLeading: {
      flex: 1,
      minWidth: 0,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 18,
    },
    menuLeadingCompact: {
      paddingRight: 32,
    },
    menuLeadingWithTrailing: {
      paddingRight: 88,
    },
    menuIconWrap: {
      width: 34,
      height: 34,
      borderRadius: 17,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.isDark ? 'rgba(255,255,255,0.06)' : 'rgba(13,39,67,0.04)',
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    menuIconWrapDanger: {
      backgroundColor: 'rgba(255,69,58,0.06)',
      borderColor: 'rgba(255,69,58,0.12)',
    },
    menuCopy: {
      flex: 1,
      minWidth: 0,
    },
    menuTitle: {
      fontSize: 15.5,
      fontWeight: '600',
      color: D.text,
      letterSpacing: -0.2,
    },
    menuTrailing: {
      position: 'absolute',
      top: 0,
      right: 20,
      bottom: 0,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'flex-end',
      gap: 6,
      minWidth: 24,
    },
    menuTrailingText: {
      fontSize: 13,
      fontWeight: '500',
      color: D.textSub,
      textAlign: 'right',
    },
    menuSeparator: {
      height: 1,
      marginLeft: 64,
      backgroundColor: D.separator,
    },
    deleteAccountWrap: {
      paddingHorizontal: 20,
      marginTop: 6,
    },
    deleteButton: {
      alignSelf: 'center',
      minWidth: 168,
    },
    modalRoot: {
      flex: 1,
      justifyContent: 'flex-end',
    },
    modalBackdrop: {
      flex: 1,
      backgroundColor: D.overlay,
    },
    sheetOuter: {
      paddingHorizontal: 12,
    },
    sheetCard: {
      overflow: 'hidden',
      borderTopLeftRadius: 34,
      borderTopRightRadius: 34,
      borderWidth: 1,
      borderColor: D.glassBorder,
      backgroundColor: D.glassStrong,
      paddingHorizontal: 20,
      paddingTop: 12,
      paddingBottom: 6,
    },
    sheetScrim: {
      ...StyleSheet.absoluteFillObject,
      backgroundColor: D.glassStrong,
    },
    sheetHandle: {
      width: 38,
      height: 5,
      borderRadius: 999,
      backgroundColor: D.separator,
      alignSelf: 'center',
      marginBottom: 18,
    },
    sheetTitle: {
      fontSize: 24,
      fontWeight: '800',
      color: D.text,
      textAlign: 'center',
      letterSpacing: -0.7,
    },
    sheetSubtitle: {
      marginTop: 6,
      fontSize: 14,
      lineHeight: 20,
      color: D.textSub,
      textAlign: 'center',
      paddingHorizontal: 18,
    },
    sheetAvatarWrap: {
      width: 112,
      height: 112,
      borderRadius: 56,
      alignSelf: 'center',
      marginTop: 20,
      marginBottom: 18,
    },
    sheetAvatarHalo: {
      width: '100%',
      height: '100%',
      borderRadius: 56,
      alignItems: 'center',
      justifyContent: 'center',
      padding: 4,
    },
    sheetAvatar: {
      width: 104,
      height: 104,
      borderRadius: 52,
    },
    sheetAvatarPlaceholder: {
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.accentSoft,
    },
    sheetAvatarBadge: {
      position: 'absolute',
      bottom: 6,
      right: 6,
      width: 32,
      height: 32,
      borderRadius: 16,
      overflow: 'hidden',
      alignItems: 'center',
      justifyContent: 'center',
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.52)',
    },
    fieldLabel: {
      marginBottom: 8,
      marginLeft: 4,
      fontSize: 11,
      fontWeight: '700',
      color: D.textMute,
      textTransform: 'uppercase',
      letterSpacing: 1.05,
    },
    fieldCard: {
      minHeight: 58,
      borderRadius: 20,
      overflow: 'hidden',
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
      backgroundColor: D.glassMuted,
      marginBottom: 16,
    },
    fieldScrim: {
      ...StyleSheet.absoluteFillObject,
      backgroundColor: D.glassMuted,
    },
    fieldInput: {
      minHeight: 58,
      paddingHorizontal: 18,
      fontSize: 16,
      color: D.fieldText,
    },
    readonlyRow: {
      minHeight: 58,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 10,
      paddingHorizontal: 18,
    },
    readonlyValue: {
      flex: 1,
      fontSize: 15,
      color: D.textSub,
    },
    sheetActions: {
      flexDirection: 'row',
      gap: 12,
      marginTop: 6,
    },
    sheetPrimaryAction: {
      flex: 1.1,
    },
    sheetSecondaryAction: {
      flex: 0.9,
    },
    savingRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 10,
      paddingTop: 14,
      paddingBottom: 4,
    },
    savingText: {
      fontSize: 13,
      color: D.textSub,
    },
    previewOverlay: {
      flex: 1,
      justifyContent: 'center',
      paddingHorizontal: 20,
      backgroundColor: D.previewOverlay,
    },
    previewShell: {
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 22 },
      shadowOpacity: D.isDark ? 0.46 : 0.2,
      shadowRadius: 32,
      elevation: 18,
    },
    previewCard: {
      borderRadius: 30,
      overflow: 'hidden',
      backgroundColor: D.glassStrong,
      borderWidth: 1,
      borderColor: D.glassBorder,
    },
    previewImage: {
      width: '100%',
      height: 390,
    },
    previewCollage: {
      width: '100%',
      height: 390,
      flexDirection: 'row',
      flexWrap: 'wrap',
    },
    previewCollageCell: {
      width: '50%',
      height: '50%',
    },
    previewFallback: {
      width: '100%',
      height: 390,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.darkBackdrop,
    },
    previewFooterBlur: {
      position: 'absolute',
      left: 0,
      right: 0,
      bottom: 92,
      height: 130,
    },
    previewFooterGradient: {
      position: 'absolute',
      left: 0,
      right: 0,
      bottom: 92,
      height: 150,
    },
    previewInfo: {
      position: 'absolute',
      left: 18,
      right: 18,
      bottom: 108,
    },
    previewBadgeRow: {
      alignSelf: 'flex-start',
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      paddingHorizontal: 10,
      paddingVertical: 6,
      borderRadius: 999,
      backgroundColor: 'rgba(255,255,255,0.16)',
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.16)',
      marginBottom: 10,
    },
    previewBadgeText: {
      fontSize: 11,
      fontWeight: '700',
      color: D.white,
      letterSpacing: 0.3,
    },
    previewTitle: {
      fontSize: 22,
      fontWeight: '800',
      color: D.white,
      letterSpacing: -0.6,
    },
    previewDate: {
      marginTop: 6,
      fontSize: 13,
      color: 'rgba(255,255,255,0.76)',
    },
    previewActions: {
      flexDirection: 'row',
      gap: 12,
      padding: 16,
      backgroundColor: D.glassStrong,
    },
    previewActionButton: {
      flex: 1,
    },
  });

export default React.memo(ProfileScreen);
