import { AccountNameError, assertHandleAvailable, displayNameOf, profileNameUpdates } from '../src/services/accountNames';
import readinessLogger from '../src/utils/logger';
/**
 * ProfileScreen — iOS liquid glass rebuild
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Dimensions, FlatList, KeyboardAvoidingView, Linking, Modal, Platform, Pressable, ScrollView, StatusBar, StyleSheet, TextInput, TouchableOpacity, View, ViewStyle } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { BlurView } from 'expo-blur';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useShallow } from 'zustand/shallow';
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
import { AvatarUploadError, avatarRequestHeaders, uploadProfileAvatar } from '../src/services/profileAvatar';
import useAuthStore from '../store/auth';
import useWardrobeStore from '../store/wardrobeStore';
import useShopCatalogStore from '../store/shopCatalogStore';
import useSavedShopStore from '../store/savedShopStore';
import { CachedImage } from '../components/ui/CachedImage';

import { OutfitCollagePreview } from '../components/ui/OutfitCollagePreview';
import { resolveOutfitCollageItemsWithMaps, resolvePieceTarget } from '../src/utils/outfitCollage';
import { useTheme } from '../src/theme/ThemeContext';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';
import { useAdminGuard } from '../hooks/useAdminGuard';
import useSubscriptionStore from '../store/subscriptionStore';
import useLanguageStore, { LANGUAGE_NAMES } from '../store/languageStore';
import LanguageSwitcher from '../components/LanguageSwitcher';
import { useTranslation } from 'react-i18next';
import { iapService } from '../src/services/iapService';
import { analyticsService } from '../src/services/analyticsService';
import { perfMark, perfMeasure, perfAction, perfScreenReady } from '../src/utils/perf';
import { listCloudOutfits } from '../src/services/savedOutfitService';
import { authEvents } from '../store/authEvents';
import { calculateStreak } from '../store/slices/helpers';
import { CalendarReadNotice } from '../features/calendar/components/CalendarReadNotice';
import { readCalendarLogs } from '../src/services/calendarStorage';
import { OCCASIONS, type OutfitLog } from '../features/calendar/types';
import { Image, ExpoImage } from '../src/components/MediaImage';


const { height } = Dimensions.get('window');
const HERO_HEIGHT = Math.min(height * 0.43, 390);
const LOOK_CARD_GAP = 14;
// Horizontal Saved Looks carousel: fixed editorial card so the full
// 3-4 piece collage is visible without vertical scrolling.
const LOOK_ROW_CARD_WIDTH = 240;

type IconName = keyof typeof Ionicons.glyphMap;
type BlurTint = 'light' | 'dark';
type ImageSrc = string | number;

interface SavedOutfit {
  _id: string;
  date?: string;
  occasion?: string;
  itemImages: ImageSrc[];
  image?: ImageSrc;
  collageItems?: { id: string | number; image: ImageSrc; category?: string; name?: string }[];
  /** Liked shop catalog item (Inspo → Shop heart). Rendered as a single-image card. */
  isShop?: boolean;
  shopItemId?: string;
  shopBrand?: string;
  shopName?: string;
  shopPrice?: number;
  shopCurrency?: string;
  /** Single wardrobe clothing save (Closet → Save new clothing). Rendered as a single-image card. */
  isWardrobe?: boolean;
  wardrobeItemId?: string;
  wardrobeName?: string;
  wardrobeCategory?: string;
}

function titleCase(value: string) {
  if (!value) return value;
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function humanizePieceLabel(item: { id?: unknown; image?: unknown; category?: unknown; name?: unknown }, index: number) {
  const macro = typeof item.category === 'string' ? item.category.toLowerCase() : '';
  const map: Record<string, string> = {
    top: 'Top',
    upper_body: 'Top',
    bottom: 'Bottom',
    lower_body: 'Bottom',
    shoes: 'Shoes',
    outerwear: 'Outerwear',
    dress: 'Dress',
    accessory: 'Accessory',
  };
  if (macro && map[macro]) return map[macro];
  const name = typeof item.name === 'string' ? item.name : '';
  if (name && name.length <= 18) return name;
  return `Piece ${index + 1}`;
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

// ── Trip plan (connected to Outfit Calendar logs) ──────────────────────
// A calendar day saved from the AI generator (Plan Outfit) is an
// `OutfitLog` under an owner-scoped `outfitLogs::` AsyncStorage key. The Trips tab
// reads that same bucket so a saved trip outfit actually shows up here.

interface TripPlanItem {
  id: string;
  image: string;
  name?: string;
  type?: string;
}

interface TripPlanDay {
  date: string;
  occasion: string;
  items: TripPlanItem[];
}

const EMPTY_TRIP_PLAN_DAYS: TripPlanDay[] = [];

function toTripPlanDay(log: OutfitLog): TripPlanDay {
  return {
    date: log.date,
    occasion: log.occasion,
    items: [...log.items].map((item) => ({
      id: String(item.id || item.image),
      image: String(item.image || ''),
      name: item.name,
      type: item.type,
    })),
  };
}

function formatTripDate(dateKey: string) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateKey);
  if (!match) return dateKey;
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]))
    .toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function getTripOccasionMeta(occasionId: string) {
  return OCCASIONS.find((o) => o.id === occasionId)
    ?? { id: occasionId, label: titleCase(occasionId), icon: '✓', color: '#64748B' };
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
    overlay: isDark ? 'rgba(3, 5, 10, 0.72)' : 'rgba(10, 20, 38, 0.55)',
    previewOverlay: 'rgba(5, 7, 15, 0.58)',
    // Solid surfaces for bottom sheets / modals — must stay fully opaque so
    // the profile content behind never bleeds through (readability fix).
    sheet: isDark ? '#151B2E' : '#FFFFFF',
    sheetBorder: isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(10, 25, 49, 0.08)',
    field: isDark ? '#222A44' : '#F1F4F8',
    fieldBorder: isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(10, 25, 49, 0.08)',
    white: '#FFFFFF',
    fieldText: isDark ? '#FFFFFF' : '#0F172A',
    darkBackdrop: 'rgba(6, 9, 18, 0.86)',
    footerGradient: ['rgba(8, 10, 18, 0)', 'rgba(8, 10, 18, 0.58)', 'rgba(8, 10, 18, 0.92)'] as readonly [string, string, string],
  };
}

type DTokens = ReturnType<typeof useDesignTokens>;

// ── Module-scope leaf components (stable types) ─────────────────────────
// These were previously defined INSIDE ProfileScreen, so every parent
// render created new component TYPES → React unmounted/remounted the entire
// subtree (BlurView + LinearGradient native views, shadows) on every
// setState. Hoisted here they reconcile by props instead — the fix for
// Profile's scroll/tap jank. Each pulls its own tokens/styles via hooks
// (memoized on isDark) so call sites need zero changes.
function useLeafAssets() {
  const D = useDesignTokens();
  // Parent memoizes on D.isDark only — mirror that here.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const styles = useMemo(() => createStyles(D), [D.isDark]);
  return { D, styles };
}

const hapticLight = () => {
  void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
};

const GlassPanel = React.memo(function GlassPanel({
  children,
  style,
  radius = 30,
  intensity = 42,
}: {
  children: React.ReactNode;
  style?: ViewStyle | ViewStyle[];
  radius?: number;
  intensity?: number;
}) {
  const { D, styles } = useLeafAssets();
  return (
    <View style={[styles.glassShadow, { borderRadius: radius }, style]}>
      <View style={[styles.glassPanel, { borderRadius: radius }]}>
        <BlurView
          intensity={Platform.OS === 'ios' ? intensity : 100}
          tint={D.tint}
          style={StyleSheet.absoluteFill}
        />
        <LinearGradient
          colors={D.panelHighlight}
          start={{ x: 0.08, y: 0 }}
          end={{ x: 0.92, y: 1 }}
          style={StyleSheet.absoluteFill}
        />
        <View style={[styles.glassScrim, { borderRadius: radius }]} />
        <View style={styles.glassContent}>{children}</View>
      </View>
    </View>
  );
});

const SecondaryGlassButton = React.memo(function SecondaryGlassButton({
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
}) {
  const { D, styles } = useLeafAssets();
  const handlePress = useCallback(() => {
    hapticLight();
    onPress();
  }, [onPress]);
  return (
    <TouchableOpacity activeOpacity={0.85} style={style} onPress={handlePress}>
      <View style={styles.inlineGlassButton}>
        <BlurView
          intensity={Platform.OS === 'ios' ? 34 : 100}
          tint={D.tint}
          style={StyleSheet.absoluteFill}
        />
        <LinearGradient
          colors={D.buttonHighlight}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFill}
        />
        <View style={styles.inlineGlassButtonContent}>
          <Ionicons name={icon} size={16} color={danger ? D.danger : D.text} />
          <ScaledText style={[styles.inlineGlassButtonText, danger && { color: D.danger }]}>{label}</ScaledText>
        </View>
      </View>
    </TouchableOpacity>
  );
});

const PrimaryGradientButton = React.memo(function PrimaryGradientButton({
  label,
  icon,
  onPress,
  style,
}: {
  label: string;
  icon: IconName;
  onPress: () => void;
  style?: ViewStyle | ViewStyle[];
}) {
  const { D, styles } = useLeafAssets();
  const handlePress = useCallback(() => {
    hapticLight();
    onPress();
  }, [onPress]);
  return (
    <TouchableOpacity activeOpacity={0.85} style={style} onPress={handlePress}>
      <LinearGradient
        colors={[D.accentStart, D.accentEnd]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.primaryButton}
      >
        <Ionicons name={icon} size={16} color={D.white} />
        <ScaledText style={styles.primaryButtonText}>{label}</ScaledText>
      </LinearGradient>
    </TouchableOpacity>
  );
});

const MenuRow = React.memo(function MenuRow({
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
}) {
  const { D, styles } = useLeafAssets();
  const handlePress = useCallback(() => {
    hapticLight();
    onPress();
  }, [onPress]);
  const pressStyle = useCallback(
    ({ pressed }: { pressed: boolean }) => [styles.menuRow, pressed && styles.menuRowPressed],
    [styles],
  );
  return (
    <Pressable style={pressStyle} onPress={handlePress}>
      <View style={[styles.menuLeading, trailing ? styles.menuLeadingWithTrailing : styles.menuLeadingCompact]}>
        <View style={[styles.menuIconWrap, danger && styles.menuIconWrapDanger]}>
          <Ionicons name={icon} size={17} color={danger ? D.danger : D.textSub} />
        </View>
        <View style={styles.menuCopy}>
          <ScaledText style={[styles.menuTitle, danger && { color: D.danger }]} numberOfLines={1}>
            {label}
          </ScaledText>
        </View>
      </View>
      <View style={styles.menuTrailing}>
        {trailing ? (
          <ScaledText style={[styles.menuTrailingText, danger && { color: D.danger }]} numberOfLines={1}>
            {trailing}
          </ScaledText>
        ) : null}
        <Ionicons name="chevron-forward" size={16} color={danger ? D.danger : D.textSub} />
      </View>
    </Pressable>
  );
});

// ── Memoized look card + virtualized row ──────────────────────────────
// Previously renderLookCard was an inline closure returning a TouchableOpacity
// with inline onPress per card, RN <Image> (no disk cache), and the row was a
// horizontal ScrollView .map — ALL cards (collage = up to 4 images + BlurView
// footer each) mounted in one commit, and 6→1000 after 600ms released the
// rest at once (first-visit freeze). Now: stable type, memoized on _id,
// expo-image memory-disk, FlatList virtualization below.
const LookCard = React.memo(function LookCard({
  outfit,
  onPreview,
}: {
  outfit: SavedOutfit;
  onPreview: (outfit: SavedOutfit) => void;
}) {
  const { D, styles } = useLeafAssets();
  const { t } = useTranslation();
  const handlePress = useCallback(() => {
    hapticLight();
    onPreview(outfit);
  }, [onPreview, outfit]);
  const cardTitle = outfit.isShop
    ? `${outfit.shopBrand ? `${outfit.shopBrand} · ` : ''}${outfit.shopName || 'Saved item'}`
    : outfit.isWardrobe
        ? outfit.wardrobeName || outfit.occasion || 'Saved item'
        : outfit.occasion || 'Saved Look';
  return (
    <TouchableOpacity
      style={[styles.lookCardShadow, { width: LOOK_ROW_CARD_WIDTH }]}
      activeOpacity={0.88}
      onPress={handlePress}
    >
      <View style={styles.lookCard}>
        {outfit.image !== undefined ? (
          <ExpoImage
            source={typeof outfit.image === 'number' ? outfit.image : { uri: outfit.image as string }}
            style={styles.lookImage}
            contentFit="cover"
            cachePolicy="memory-disk"
            recyclingKey={typeof outfit.image === 'string' ? outfit.image : undefined}
          />
        ) : (
          <OutfitCollagePreview
            items={
              outfit.collageItems && outfit.collageItems.length > 0
                ? outfit.collageItems
                : outfit.itemImages.map((src, index) => ({ id: index, image: src }))
            }
            backgroundColor="#F4F6FA"
            placeholderIconColor="rgba(15,23,42,0.22)"
            placeholderIconSize={30}
            footerInset={64}
          />
        )}
        <View style={styles.lookFooterBar}>
          <BlurView
            intensity={Platform.OS === 'ios' ? 48 : 100}
            tint="light"
            style={StyleSheet.absoluteFill}
          />
          <View style={styles.lookFooterScrim} />
          <View style={styles.lookFooterContent}>
            <View style={styles.lookFooterText}>
              <ScaledText style={styles.lookTitle} numberOfLines={1}>
                {cardTitle}
              </ScaledText>
              {outfit.date ? (
                <ScaledText style={styles.lookDate} numberOfLines={1}>
                  {outfit.date}
                </ScaledText>
              ) : null}
            </View>
            <View style={styles.lookFooterAction}>
              <Ionicons name="arrow-forward" size={14} color="#FFFFFF" />
            </View>
          </View>
        </View>
        {outfit.isShop ? (
          <View style={styles.lookBadge}>
            <Ionicons name="bag-outline" size={10} color={D.white} />
            <ScaledText style={styles.lookBadgeText}>
              {t('profile.shopBadge', { defaultValue: 'Shop' })}
            </ScaledText>
          </View>
        ) : null}
        {outfit.isWardrobe ? (
          <View style={styles.lookBadge}>
            <Ionicons name="shirt-outline" size={10} color={D.white} />
            <ScaledText style={styles.lookBadgeText}>
              {t('profile.closetBadge', { defaultValue: 'Closet' })}
            </ScaledText>
          </View>
        ) : null}
      </View>
    </TouchableOpacity>
  );
});

const ProfileScreen = () => {
  const D = useDesignTokens();
  // D is a new object reference every render; depend only on the primitive
  // that actually changes the styles (isDark) to avoid recreating 150+ rules.
  const styles = useMemo(() => createStyles(D), [D.isDark]); // eslint-disable-line react-hooks/exhaustive-deps
  const navigation = useAppNavigation();
  const insets = useSafeAreaInsets();
  const { requireFeature: requireSubFeature } = useSubscriptionGate();
  const { isAdmin: isAdminUser } = useAdminGuard();
  const { t } = useTranslation();

  // ── Performance timing ────────────────────────────────────────────────────
  React.useEffect(() => {
    perfMark('ProfileScreen:mount');
    // Profile screen is ready immediately (data comes from store)
    perfMeasure('ProfileScreen:mount');
    perfScreenReady('Profile');
    return () => {
      if (__DEV__) readinessLogger.debug(String('[PERF] 👤 ProfileScreen unmounted'), []);
    };
  }, []);

  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const deleteAccount = useAuthStore((s) => s.deleteAccount);
  const fetchUser = useAuthStore((s) => s.fetchUser);
  const { items: wardrobeItems, outfits: storeOutfits, wearLogs } = useWardrobeStore(
    useShallow((state) => ({ items: state.items, outfits: state.outfits, wearLogs: state.wearLogs }))
  );
  const effectiveTier = useSubscriptionStore((s) => s.effectiveTier);
  const isTrialActive = useSubscriptionStore((s) => s.isTrialActive);
  // Read catalog from the shared store populated by HomeScreen — no new
  // Supabase query is fired here (fixes Defect 1.5).
  const liveShopCatalog = useShopCatalogStore((state) => state.items);
  // Liked shop items (Inspo → Shop heart). Shared with InspoScreen via
  // savedShopStore so a like instantly appears in Saved Looks.
  const savedShopItems = useSavedShopStore((state) => state.items);
  const hydrateSavedShop = useSavedShopStore((state) => state.hydrate);
  const currentLanguage = useLanguageStore((s) => s.currentLanguage);

  const [activeTab, setActiveTab] = useState<'looks' | 'trips'>('looks');
  const [showEditProfile, setShowEditProfile] = useState(false);
  const [showLanguageModal, setShowLanguageModal] = useState(false);
  const [analyticsEnabled, setAnalyticsEnabled] = useState(true);
  const [editName, setEditName] = useState('');
  const [editHandle, setEditHandle] = useState('');
  const profileEditMounted = useRef(true);
  useEffect(() => { profileEditMounted.current = true; return () => { profileEditMounted.current = false; }; }, []);
  useEffect(() => { setSaving(false); setShowEditProfile(false); }, [user?.id]);
  const [editAvatar, setEditAvatar] = useState<string | null>(null);
  const [editAvatarMimeType, setEditAvatarMimeType] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [cloudOutfits, setCloudOutfits] = useState<SavedOutfit[]>([]);
  const [, setLoading] = useState(false);
  const [previewLook, setPreviewLook] = useState<SavedOutfit | null>(null);
  // Progressive looks: the horizontal row mounts a collage (up to 4 images)
  // + a BlurView per card. Mounting ALL saved looks in the first frame is
  // the commit storm behind Profile's first-visit freeze — paint the first
  // few, release more after paint settles, then paginate on scroll
  // (onEndReached). Previously 6→1000 at 600ms released everything at once.
  const [visibleLookCount, setVisibleLookCount] = useState(6);
  useEffect(() => {
    const timer = setTimeout(() => setVisibleLookCount(14), 600);
    return () => clearTimeout(timer);
  }, []);

  // Skip re-fetching if the user just switched tabs momentarily (fixes Defect 4.5).
  const OUTFITS_REFRESH_TTL_MS = 60_000; // 60 seconds
  // Initialize to Date.now() so the very first visit never fires a fetch —
  // localOutfits (derived from storeOutfits) already shows store data, and
  // the store fast-path in fetchOutfits handles the empty-store case.
  // After 60s the TTL expires and the next focus triggers a fresh cloud fetch.
  const lastOutfitFetchRef = useRef<number>(Date.now());

  const scrollY = useSharedValue(0);
  const scrollHandler = useAnimatedScrollHandler((event) => {
    scrollY.value = event.contentOffset.y;
  });

  const triggerLightHaptic = () => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  useFocusEffect(
    useCallback(() => {
      setAnalyticsEnabled(analyticsService.getEnabled());
      // Refresh shop likes — InspoScreen writes here on every heart tap.
      hydrateSavedShop();
    }, [hydrateSavedShop]),
  );

  const userName = displayNameOf(user) || t('profile.name');
  const userAvatar: string | undefined =
    typeof user?.profile_image === 'string' && user.profile_image.length > 0
      ? user.profile_image
      : undefined;
  const avatarAccessToken = useAuthStore((s) => s.session?.access_token);
  const privateAvatarHeaders = useMemo(() => avatarRequestHeaders(userAvatar, avatarAccessToken), [userAvatar, avatarAccessToken]);
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
    const isValidUri = (value?: string) =>
      typeof value === 'string' &&
      (value.startsWith('http') || value.startsWith('file://') || value.startsWith('data:'));

    // Build lookup Maps ONCE — resolving per outfit with the array variant
    // rebuilds both Maps for every saved outfit (O(outfits × wardrobe)),
    // the main JS freeze on Profile's first paint.
    const wardrobeMap = new Map(wardrobeItems.map((i) => [String(i.id), i]));
    const shopMap = new Map(liveShopCatalog.map((i) => [String(i.id), i]));

    return storeOutfits
      // Home swipe-likes (savedFrom='home') live in Closet → Collections (Saved),
      // not here. Profile Saved Looks is for single-clothing saves (wardrobe +
      // shop) + manual/AI-maker outfits. Legacy saves without savedFrom stay
      // visible in both for backward compat.
      .filter((outfit) => outfit.saved && (outfit as { savedFrom?: string }).savedFrom !== 'home')
      .map((outfit) => {
        const coverImage =
          typeof outfit.previewImageUrl === 'string' && isValidUri(outfit.previewImageUrl)
            ? outfit.previewImageUrl
            : undefined;

        // Full outfit: frozen snapshot first (wardrobe + shop pieces from
        // like-time), wardrobe + live catalog fallback for legacy saves.
        const collage = resolveOutfitCollageItemsWithMaps(outfit, wardrobeMap, shopMap);
        const itemImages = collage.map((c) => c.image as ImageSrc);

        return {
          _id: outfit.id,
          date: outfit.createdAt?.split('T')[0],
          occasion: typeof outfit.occasion === 'string' ? outfit.occasion : undefined,
          // previewImageUrl is just the first item's own image, not a real
          // composite of the outfit — only fall back to it when none of the
          // outfit's items resolved to an image, so multi-item outfits show
          // the itemImages collage below instead of a single garment.
          image: itemImages.length === 0 ? coverImage : undefined,
          itemImages,
          collageItems: collage.map((c, index) => ({
            id: c.id ?? index,
            image: c.image as ImageSrc,
            category: c.category,
            name: c.name,
          })),
        };
      });
  }, [storeOutfits, wardrobeItems, liveShopCatalog]);

  const fetchOutfits = useCallback(async (mounted?: { current: boolean }) => {
    if (!user?.id) return;

    const revision = authEvents.getRevision();
    const donePerf = perfAction('Profile:fetchOutfits');
    if (!mounted?.current) return;
    setLoading(true);
    try {
      const data = await listCloudOutfits();

      if (!mounted?.current || revision !== authEvents.getRevision()) { donePerf(); return; }

      if (data) {
        setCloudOutfits(
          data.map((outfit) => {
            const rawItems = Array.isArray(outfit.items)
              ? (outfit.items as Array<{ image?: unknown; imageUrl?: unknown }>)
              : [];
            const itemImages: ImageSrc[] = rawItems.flatMap(item => {
              const image = item?.imageUrl || item?.image;
              return typeof image === 'string' && image.startsWith('http') ? [image] : [];
            });

            return {
              _id: outfit.id,
              date: outfit.date || outfit.created_at?.split('T')[0],
              occasion: outfit.occasion ?? undefined,
              itemImages,
              // Only set a single cover image when there's just one item to
              // show — outfits with multiple items render the itemImages
              // collage instead (see renderLookCard).
              image: itemImages.length === 1 ? itemImages[0] : undefined,
            };
          })
        );
        donePerf();
      }
    } catch {
      donePerf();
      // Keep local data visible if the cloud fetch fails.
    } finally {
      if (mounted?.current && revision === authEvents.getRevision()) setLoading(false);
    }
  }, [user?.id]);

  useFocusEffect(
    useCallback(() => {
      const mounted = { current: true };
      const now = Date.now();
      if (now - lastOutfitFetchRef.current > OUTFITS_REFRESH_TTL_MS) {
        lastOutfitFetchRef.current = now;
        fetchOutfits(mounted);
      }
      return () => { mounted.current = false; };
    }, [fetchOutfits])
  );

  // New clothing saves (Closet → Save / ClothingDetailEditor addItem) —
  // each wardrobe item becomes a single-image Saved Look so "save new
  // clothing" lands on the Profile page as requested.
  const wardrobeMapped: SavedOutfit[] = useMemo(
    () =>
      [...wardrobeItems]
        .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''))
        .map((item) => {
          const image =
            typeof item.imageUrl === 'string' && item.imageUrl.length > 0
              ? item.imageUrl
              : undefined;
          const label = (
            typeof item.name === 'string' && item.name.length > 0
              ? item.name
              : item.subCategory || item.category || 'Saved item'
          ).slice(0, 40);
          return {
            _id: `wardrobe-${String(item.id)}`,
            date: typeof item.createdAt === 'string' ? item.createdAt.split('T')[0] : undefined,
            occasion: label,
            itemImages: [],
            image,
            isWardrobe: true,
            wardrobeItemId: String(item.id),
            wardrobeName: label,
            wardrobeCategory: typeof item.category === 'string' ? item.category : undefined,
          };
        }),
    [wardrobeItems],
  );

  const outfits = useMemo(() => {
    const localIds = new Set(localOutfits.map((outfit) => outfit._id));
    const cloudOnly = cloudOutfits.filter((outfit) => !localIds.has(outfit._id));

    // Liked shop clothing (Shop heart) — each liked product becomes a
    // single-image Saved Look so it survives even if the product leaves
    // the live catalog (we keep the like-time snapshot).
    const shopMapped: SavedOutfit[] = savedShopItems.map((item) => {
      const image =
        typeof item.imageUrl === 'string' && item.imageUrl.length > 0 ? item.imageUrl : undefined;
      const price =
        typeof item.price === 'number' && Number.isFinite(item.price) ? item.price : undefined;
      return {
        _id: `shop-${String(item.id)}`,
        date: price !== undefined ? `$${price.toFixed(0)}` : undefined,
        occasion: item.brand || item.name || 'Saved item',
        itemImages: [],
        image,
        isShop: true,
        shopItemId: String(item.id),
        shopBrand: item.brand,
        shopName: item.name,
        shopPrice: price,
        shopCurrency: item.currency,
      };
    });

    return [...wardrobeMapped, ...localOutfits, ...shopMapped, ...cloudOnly];
  }, [localOutfits, cloudOutfits, savedShopItems, wardrobeMapped]);

  // ── Connected trip plan ─────────────────────────────────────────────
  // Upcoming calendar days with a saved outfit (today + future). When
  // there is nothing upcoming, fall back to the 3 most recent past plans
  // so the tab never looks dead right after a trip.
  const [storedTripPlanDays, setTripPlanDays] = useState<TripPlanDay[]>([]);
  const [tripPlanOwner, setTripPlanOwner] = useState<string | null>(null);
  const tripPlanDays = tripPlanOwner === user?.id && user?.id === authEvents.getUserId()
    ? storedTripPlanDays : EMPTY_TRIP_PLAN_DAYS;
  const tripPlanRequest = useRef(0);
  const [tripPlanErrorOwner, setTripPlanErrorOwner] = useState<string | null>(null);
  const [tripPlanRecency, setTripPlanRecency] = useState<'upcoming' | 'recent'>('upcoming');
  // Re-read local calendar data on focus so a just-saved plan is visible.
  // Guard both the account and this focus lifetime before publishing private data.
  const loadTripPlans = useCallback(async (isActive: () => boolean) => {
    const owner = user?.id, revision = authEvents.getRevision();
    const request = ++tripPlanRequest.current;
    const current = () => isActive() && request === tripPlanRequest.current &&
      !!owner && owner === authEvents.getUserId() && revision === authEvents.getRevision();
    if (!current()) return;
    try {
      const logs = await readCalendarLogs();
      if (!current()) return;
      setTripPlanOwner(owner!);
      setTripPlanErrorOwner(null);
      if (!Object.keys(logs).length) {
        setTripPlanDays([]);
        return;
      }
      const entries = Object.values(logs).filter(
        (log): log is OutfitLog =>
          !!log && typeof log.date === 'string' && Array.isArray(log.items) && log.items.length > 0
      );
      const now = new Date();
      const todayKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      const upcoming = entries
        .filter((log) => log.date >= todayKey)
        .sort((a, b) => a.date.localeCompare(b.date));
      if (upcoming.length > 0) {
        setTripPlanRecency('upcoming');
        setTripPlanDays(upcoming.map(toTripPlanDay));
        return;
      }
      const recent = entries
        .sort((a, b) => b.date.localeCompare(a.date))
        .slice(0, 3);
      setTripPlanRecency('recent');
      setTripPlanDays(recent.map(toTripPlanDay));
    } catch {
      if (current()) setTripPlanErrorOwner(owner!);
    }
  }, [user?.id]);

  useFocusEffect(
    useCallback(() => {
      let active = true;
      const timer = setTimeout(() => { void loadTripPlans(() => active); }, 300);
      return () => { active = false; clearTimeout(timer); };
    }, [loadTripPlans]),
  );

  // Packing list aggregated from the visible trip days: unique pieces with
  // the number of days each piece is worn.
  const tripPackingList = useMemo(() => {
    const byId = new Map<string, TripPlanItem & { days: number }>();
    tripPlanDays.forEach((day) => {
      const seenInDay = new Set<string>();
      day.items.forEach((item) => {
        const key = String(item.id || item.image);
        if (!key) return;
        const existing = byId.get(key);
        if (existing) {
          if (!seenInDay.has(key)) {
            existing.days += 1;
            seenInDay.add(key);
          }
        } else {
          seenInDay.add(key);
          byId.set(key, { ...item, id: key, days: 1 });
        }
      });
    });
    return [...byId.values()].sort((a, b) => b.days - a.days);
  }, [tripPlanDays]);

  const tripPieceCount = useMemo(
    () => tripPlanDays.reduce((sum, day) => sum + day.items.length, 0),
    [tripPlanDays]
  );

  // Pieces for the open preview: full collage meta when available,
  // plain images otherwise (legacy cloud outfits).
  const previewPieces = useMemo(() => {
    if (!previewLook) return [];
    if (previewLook.collageItems && previewLook.collageItems.length > 0) {
      return previewLook.collageItems;
    }
    return previewLook.itemImages.map((src, index) => ({ id: index, image: src }));
  }, [previewLook]);

  const previewTitleText = useMemo(() => {
    if (!previewLook) return '';
    if (previewLook.isShop) {
      const brand = previewLook.shopBrand ? `${previewLook.shopBrand} · ` : '';
      return `${brand}${previewLook.shopName || 'Saved item'}`;
    }
    if (previewLook.isWardrobe) {
      return previewLook.wardrobeName || previewLook.occasion || 'Saved item';
    }
    return previewLook.occasion || 'Saved Look';
  }, [previewLook]);

  const openWardrobePreviewInCloset = useCallback(() => {
    if (!previewLook?.isWardrobe || !previewLook.wardrobeItemId) return;
    triggerLightHaptic();
    const target = wardrobeItems.find((i) => String(i.id) === String(previewLook.wardrobeItemId));
    setPreviewLook(null);
    if (target) {
      navigation.navigate('ClothingDetail', {
        itemId: String(target.id),
        fullItem: target,
      });
    } else {
      navigation.navigate('Main', { screen: 'Closet' });
    }
  }, [previewLook, wardrobeItems, navigation]);

  const openShopPreviewInShop = useCallback(() => {
    if (!previewLook?.isShop) return;
    triggerLightHaptic();
    const id = previewLook.shopItemId ?? '';
    setPreviewLook(null);
    navigation.navigate('Main', {
      screen: 'Inspo',
      params: {
        focusShopItemId: id,
        focusShopItemName: previewLook.shopName,
      },
    });
  }, [previewLook, navigation]);

  const buyShopPreview = useCallback(() => {
    if (!previewLook?.isShop) return;
    triggerLightHaptic();
    const query = encodeURIComponent(
      `${previewLook.shopBrand ?? ''} ${previewLook.shopName ?? ''}`.trim() || 'shop',
    );
    Linking.canOpenURL(`https://www.google.com/search?q=${query}`)
      .then((canOpen) => {
        if (canOpen) Linking.openURL(`https://www.google.com/search?q=${query}`);
      })
      .catch(() => {});
  }, [previewLook]);

  const removeShopPreview = useCallback(() => {
    if (!previewLook?.isShop || !previewLook.shopItemId) return;
    triggerLightHaptic();
    Alert.alert(
      t('profile.removeLook', { defaultValue: 'Remove look?' }),
      t('profile.removeShopSaved', {
        defaultValue: 'Remove this saved item from your looks?',
      }),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.remove'),
          style: 'destructive',
          onPress: async () => {
            const revision = authEvents.getRevision();
            try {
              await useSavedShopStore.getState().removeSave(previewLook.shopItemId as string);
              if (revision === authEvents.getRevision()) setPreviewLook(null);
            } catch {
              if (revision === authEvents.getRevision()) {
                Alert.alert(t('common.error'), t('inspo.savedItemsUnavailable'));
              }
            }
          },
        },
      ],
    );
  }, [previewLook, t]);

  // Navigation target per piece: wardrobe items open their detail screen,
  // shop items jump to the Shop tab focused on that product.
  const previewTargets = useMemo(
    () => previewPieces.map((piece) => resolvePieceTarget(piece, wardrobeItems, liveShopCatalog)),
    [previewPieces, wardrobeItems, liveShopCatalog],
  );

  const handlePiecePress = useCallback((index: number) => {
    const piece = previewPieces[index];
    if (!piece) return;
    const target = previewTargets[index];
    triggerLightHaptic();
    if (target?.kind === 'wardrobe') {
      const wardrobeId = String(target.item.id);
      setPreviewLook(null);
      navigation.navigate('ClothingDetail', {
        itemId: wardrobeId,
        fullItem: target.item,
      });
      return;
    }
    // Shop piece — or anything not owned: always land in Shop. If the id
    // exists in the catalog it gets lifted + ringed; otherwise Inspo falls
    // back to a name search so the tap is never dead.
    const pieceAny = piece as { id?: string | number; name?: unknown };
    setPreviewLook(null);
    navigation.navigate('Main', {
      screen: 'Inspo',
      params: {
        focusShopItemId: String(pieceAny.id ?? ''),
        focusShopItemName: typeof pieceAny.name === 'string' ? pieceAny.name : undefined,
      },
    });
  }, [previewPieces, previewTargets, navigation]);

  const streak = useMemo(() => calculateStreak(wearLogs), [wearLogs]);

  const stats = useMemo(
    () => [
      { icon: 'shirt-outline' as IconName, label: t('common.items'), value: wardrobeItems.length },
      { icon: 'sparkles-outline' as IconName, label: t('common.looks'), value: outfits.length },
      { icon: 'flame-outline' as IconName, label: t('common.streak'), value: streak || '0' },
      { icon: 'checkmark-done-outline' as IconName, label: t('common.wears'), value: wearLogs.length },
    ],
    [wardrobeItems.length, outfits.length, streak, wearLogs.length, t]
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
    setEditName(displayNameOf(user));
    setEditHandle(user?.handle || '');
    setEditAvatar(null);
    setEditAvatarMimeType(null);
    setShowEditProfile(true);
    triggerLightHaptic();
  }, [user]);

  const pickAvatar = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(t('profile.permissionNeeded'), t('profile.allowPhotoLibrary'));
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
      setEditAvatarMimeType(result.assets[0].mimeType || null);
    }
  };

  const handleSaveProfile = async () => {
    if (!user?.id || saving || user.id !== authEvents.getUserId()) return;
    const owner = user.id, revision = authEvents.getRevision();
    const stillCurrent = () => profileEditMounted.current && owner === authEvents.getUserId() && revision === authEvents.getRevision();
    setSaving(true);
    let uploadedAvatarPath: string | null = null;
    let profileSaved = false;
    try {
      const nameUpdates = profileNameUpdates(user, editName, editHandle);
      if (nameUpdates.handle) await assertHandleAvailable(nameUpdates.handle);
      if (!stillCurrent()) return;
      let avatarUrl = user.profile_image;

      if (editAvatar) {
        const uploaded = await uploadProfileAvatar(user.id, editAvatar, editAvatarMimeType);
        uploadedAvatarPath = uploaded.path;
        avatarUrl = uploaded.url;
        if (!stillCurrent()) { await supabase.storage.from('avatars').remove([uploaded.path]).catch(() => undefined); return; }
      }

      const updates: Record<string, unknown> = { ...nameUpdates };
      if (avatarUrl !== user.profile_image) updates.profile_image = avatarUrl;

      if (Object.keys(updates).length > 0) {
        const { error } = await supabase.from('profiles').update(updates).eq('id', user.id).select('id').single();
        if (error) throw error;
        profileSaved = true;

        await fetchUser();
        if (!stillCurrent()) return;
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }

      if (stillCurrent()) setShowEditProfile(false);
    } catch (error: unknown) {
      if (uploadedAvatarPath && !profileSaved) {
        await supabase.storage.from('avatars').remove([uploadedAvatarPath]).catch(() => undefined);
      }
      if (!stillCurrent()) return;
      const message = error instanceof AccountNameError
        ? t(error.code === 'username_exists' ? 'signUp.usernameAlreadyUse' : error.code === 'invalid_handle' ? 'signUp.usernameRequirements' : error.code === 'invalid_display_name' ? 'profile.displayNameRequirements' : 'profile.failedUpdateProfile')
        : (error as { code?: string })?.code === '23505'
          ? t('signUp.usernameAlreadyUse')
        : error instanceof AvatarUploadError
        ? t(error.code === 'invalid_size' ? 'profile.avatarInvalidSize' : 'profile.avatarUnsupportedFormat')
        : getErrorMessage(error, t('profile.failedUpdateProfile'));
      Alert.alert(t('common.error'), message);
    } finally {
      if (stillCurrent()) setSaving(false);
    }
  };

  const handleForceExpireTrial = async () => {
    try {
      // Set trial start to 8 days ago
      const eightDaysAgo = new Date();
      eightDaysAgo.setDate(eightDaysAgo.getDate() - 8);
      const dateStr = eightDaysAgo.toISOString();
      
      await AsyncStorage.setItem('trial_started_at_v1', dateStr);
      
      // Update local store state
      useSubscriptionStore.setState({ trialStartedAt: dateStr });
      await useSubscriptionStore.getState().initializeSubscription();
      
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert('Debug', 'Trial forced to expired state. Restart the app or navigate to see the gate.');
    } catch (error) {
      Alert.alert('Error', 'Failed to force expire trial');
    }
  };

  const handleResetTrial = async () => {
    try {
      await AsyncStorage.removeItem('trial_started_at_v1');
      await useSubscriptionStore.getState().initializeSubscription();
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert('Debug', 'Trial reset. It will re-initialize on next auth success.');
    } catch (error) {
      Alert.alert('Error', 'Failed to reset trial');
    }
  };

  const handleLogout = () => {
    Alert.alert(t('profile.signOut'), t('profile.areYouSure'), [
      { text: t('common.cancel'), style: 'cancel' },
      {
        text: t('profile.signOut'),
        style: 'destructive',
        onPress: async () => {
          const donePerf = perfAction('Profile:logout');
          try { await logout(); }
          catch { Alert.alert(t('common.error'), t('common.localDataCleanupFailed')); }
          finally { donePerf(); }
        },
      },
    ]);
  };

  const handleDeleteAccount = () => {
    Alert.alert(t('profile.deleteAccount'), t('profile.permanentlyDelete'), [
      { text: t('common.cancel'), style: 'cancel' },
      {
        text: t('common.delete'),
        style: 'destructive',
        onPress: () =>
          Alert.alert(t('profile.confirmDelete'), t('profile.cannotUndo'), [
            { text: t('profile.keepAccount'), style: 'cancel' },
            {
              text: t('profile.deleteEverything'),
              style: 'destructive',
              onPress: async () => {
                const revision = authEvents.getRevision();
                try {
                  await deleteAccount();
                } catch (error: unknown) {
                  if (revision !== authEvents.getRevision()) return;
                  Alert.alert(t('common.error'), getErrorMessage(error, t('profile.failedDeleteAccount')));
                }
              },
            },
          ]),
      },
    ]);
  };

  // Stable preview handler shared by all memoized LookCards.
  const handlePreviewLook = useCallback((outfit: SavedOutfit) => {
    triggerLightHaptic();
    setPreviewLook(outfit);
  }, []);

  const renderLookItem = useCallback(({ item }: { item: SavedOutfit }) => (
    <LookCard outfit={item} onPreview={handlePreviewLook} />
  ), [handlePreviewLook]);

  const lookKeyExtractor = useCallback((item: SavedOutfit) => item._id, []);
  const lookGetItemLayout = useCallback((_: ArrayLike<SavedOutfit> | null | undefined, index: number) => ({
    length: LOOK_ROW_CARD_WIDTH,
    offset: (LOOK_ROW_CARD_WIDTH + LOOK_CARD_GAP) * index,
    index,
  }), []);
  const handleLookEndReached = useCallback(() => {
    setVisibleLookCount((c) => (c >= 1000 ? c : Math.min(c + 12, 1000)));
  }, []);

  const savedProducts = useMemo(() => outfits.filter((item) => item.isShop), [outfits]);

  const renderLooksContent = () => {
    if (savedProducts.length === 0) {
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
            <ScaledText style={styles.emptyTitle}>{t('profile.noSavedProducts', { defaultValue: 'No saved items yet' })}</ScaledText>
            <ScaledText style={styles.emptySubtitle}>
              {t('profile.savedProductsHint', { defaultValue: 'Tap the heart on a shop item to save it here.' })}
            </ScaledText>
            <PrimaryGradientButton
              label={t('profile.browseShop', { defaultValue: 'Browse shop' })}
              icon="heart-outline"
              onPress={() => navigation.navigate('Main', { screen: 'Inspo', params: { openShop: true } })}
              style={styles.emptyAction}
            />
          </View>
        </GlassPanel>
      );
    }

    return (
      <FlatList
        data={savedProducts.slice(0, visibleLookCount)}
        renderItem={renderLookItem}
        keyExtractor={lookKeyExtractor}
        getItemLayout={lookGetItemLayout}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.lookRow}
        initialNumToRender={4}
        maxToRenderPerBatch={4}
        windowSize={3}
        updateCellsBatchingPeriod={50}
        removeClippedSubviews={Platform.OS === 'android'}
        onEndReached={handleLookEndReached}
        onEndReachedThreshold={0.5}
      />
    );
  };

  const openTripPlanner = (initialDate?: string) => {
    // Pro / Max only — Free users see the paywall.
    if (!requireSubFeature('tripPlanner')) return;
    triggerLightHaptic();
    navigation.navigate('Calendar', initialDate ? { initialDate } : undefined);
  };

  const renderTripsContent = () => (
    <GlassPanel style={styles.contentPanel}>
      {tripPlanErrorOwner === user?.id && user?.id === authEvents.getUserId() && <CalendarReadNotice onRetry={() => { void loadTripPlans(() => navigation.isFocused()); }} />}
      {tripPlanDays.length > 0 ? (
        <View style={styles.tripPlanList}>
          <View style={styles.tripPlanSummary}>
            <View style={styles.tripFlowIconWrap}>
              <Ionicons name="airplane-outline" size={16} color={D.accent} />
            </View>
            <View style={styles.tripFlowCopy}>
              <ScaledText style={styles.tripFlowTitle}>
                {tripPlanRecency === 'upcoming'
                  ? t('profile.tripPlanReady', { defaultValue: '{{count}} days planned', count: tripPlanDays.length })
                  : t('profile.tripRecentPlans', { defaultValue: 'Recent plans' })}
              </ScaledText>
              <ScaledText style={styles.tripFlowSubtitle}>
                {t('profile.tripPackingSummary', {
                  defaultValue: '{{pieces}} pieces · {{unique}} unique items to pack',
                  pieces: tripPieceCount,
                  unique: tripPackingList.length,
                })}
              </ScaledText>
            </View>
          </View>

          {tripPlanDays.map((day) => {
            const meta = getTripOccasionMeta(day.occasion);
            return (
              <TouchableOpacity
                key={day.date}
                style={styles.tripDayCard}
                activeOpacity={0.8}
                onPress={() => openTripPlanner(day.date)}
                accessibilityLabel={`Open trip day ${day.date} in calendar`}
              >
                <View style={styles.tripDayHeader}>
                  <ScaledText style={styles.tripDayDate}>{formatTripDate(day.date)}</ScaledText>
                  <View style={[styles.tripOccasionPill, { backgroundColor: `${meta.color}1A` }]}>
                    <ScaledText style={[styles.tripOccasionPillText, { color: meta.color }]}>
                      {meta.icon} {meta.label}
                    </ScaledText>
                  </View>
                </View>
                <View style={styles.tripDayThumbs}>
                  {day.items.slice(0, 4).map((item) => (
                    <CachedImage
                      key={`${day.date}-${item.id}`}
                      uri={item.image}
                      style={styles.tripDayThumb}
                      contentFit="cover"
                      showLoader={false}
                      fadeIn={false}
                    />
                  ))}
                  {day.items.length > 4 && (
                    <View style={[styles.tripDayThumb, styles.tripDayMore]}>
                      <ScaledText style={styles.tripDayMoreText}>+{day.items.length - 4}</ScaledText>
                    </View>
                  )}
                </View>
              </TouchableOpacity>
            );
          })}

          {tripPackingList.length > 0 && (
            <View style={styles.tripPackingCard}>
              <ScaledText style={styles.packingTitle}>
                {t('profile.packingList', { defaultValue: 'Packing list' })}
              </ScaledText>
              <View style={styles.packingGrid}>
                {tripPackingList.slice(0, 8).map((item) => (
                  <View key={item.id} style={styles.packingItem}>
                    <CachedImage
                      uri={item.image}
                      style={styles.packingThumb}
                      contentFit="cover"
                      showLoader={false}
                      fadeIn={false}
                    />
                    <ScaledText style={styles.packingName} numberOfLines={1}>
                      {item.name || item.type || t('profile.packingItem', { defaultValue: 'Item' })}
                    </ScaledText>
                    {item.days > 1 && (
                      <ScaledText style={styles.packingCount}>×{item.days}</ScaledText>
                    )}
                  </View>
                ))}
              </View>
              {tripPackingList.length > 8 && (
                <ScaledText style={styles.packingMore}>
                  {t('profile.packingMore', {
                    defaultValue: '+{{count}} more items',
                    count: tripPackingList.length - 8,
                  })}
                </ScaledText>
              )}
            </View>
          )}
        </View>
      ) : (
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
          <ScaledText style={styles.emptyTitle}>{t('profile.planTripOutfits')}</ScaledText>
          <ScaledText style={styles.emptySubtitle}>
            {t('profile.planTripDescription')}
          </ScaledText>
          <PrimaryGradientButton
            label={t('profile.openCalendar')}
            icon="calendar-outline"
            onPress={() => openTripPlanner()}
            style={styles.emptyAction}
          />
        </View>
      )}

      {/* Onboarding steps — only needed until the user has a trip plan.
          Once tripPlanDays exists (upcoming or recent/past) this is hidden
          to return vertical space to the actual plans + packing list. */}
      {tripPlanDays.length === 0 && (
      <View style={styles.tripFlowList}>
        <View style={styles.tripFlowStep}>
          <View style={styles.tripFlowIconWrap}>
            <Ionicons name="calendar-outline" size={16} color={D.accent} />
          </View>
          <View style={styles.tripFlowCopy}>
            <ScaledText style={styles.tripFlowTitle}>{t('profile.tripFlowStep1')}</ScaledText>
            <ScaledText style={styles.tripFlowSubtitle}>
              {t('profile.tripFlowStep1Desc')}
            </ScaledText>
          </View>
        </View>

        <View style={styles.tripFlowStep}>
          <View style={styles.tripFlowIconWrap}>
            <Ionicons name="sparkles-outline" size={16} color={D.accent} />
          </View>
          <View style={styles.tripFlowCopy}>
            <ScaledText style={styles.tripFlowTitle}>{t('profile.tripFlowStep2')}</ScaledText>
            <ScaledText style={styles.tripFlowSubtitle}>
              {t('profile.tripFlowStep2Desc')}
            </ScaledText>
          </View>
        </View>

        <View style={styles.tripFlowStep}>
          <View style={styles.tripFlowIconWrap}>
            <Ionicons name="checkmark-done-outline" size={16} color={D.accent} />
          </View>
          <View style={styles.tripFlowCopy}>
            <ScaledText style={styles.tripFlowTitle}>{t('profile.tripFlowStep3')}</ScaledText>
            <ScaledText style={styles.tripFlowSubtitle}>
              {t('profile.tripFlowStep3Desc')}
            </ScaledText>
          </View>
        </View>
      </View>
      )}
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
            style={StyleSheet.absoluteFill}
          />
          <LinearGradient
            colors={D.buttonHighlight}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={StyleSheet.absoluteFill}
          />
          <View style={styles.stickyHeaderContent}>
            <ScaledText style={styles.stickyHeaderTitle}>{userName}</ScaledText>
            <View style={styles.stickyPlanChip}>
              <Ionicons name={isPro ? 'diamond-outline' : 'sparkles-outline'} size={11} color={D.textSub} />
              <ScaledText style={styles.stickyPlanChipText}>{planLabel}</ScaledText>
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
                <ScaledText style={styles.profileChipText}>{t('profile.account')}</ScaledText>
              </View>

              <SecondaryGlassButton label={t('common.edit')} icon="create-outline" onPress={openEditProfile} />
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
                    <CachedImage uri={userAvatar} requestHeaders={privateAvatarHeaders} cachePolicy="memory" style={styles.avatarImage} contentFit="cover" fadeIn={false} />
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
                  style={StyleSheet.absoluteFill}
                />
                <LinearGradient
                  colors={[D.accentStart, D.accentEnd]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={StyleSheet.absoluteFill}
                />
                <Ionicons name="camera" size={12} color={D.white} />
              </TouchableOpacity>
            </View>

            <View style={styles.identityBlock}>
              <ScaledText style={styles.heroName}>{userName}</ScaledText>
              {!!user?.handle && <ScaledText style={styles.heroEmail}>@{user.handle}</ScaledText>}
              {userEmail ? <ScaledText style={styles.heroEmail}>{userEmail}</ScaledText> : null}

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
                    <ScaledText style={styles.membershipBadgeText}>{topPlanLabel}</ScaledText>
                  </LinearGradient>
                ) : (
                  <View style={styles.membershipBadgeGlass}>
                    <ScaledText style={styles.membershipBadgeSubtleText}>{topPlanLabel}</ScaledText>
                  </View>
                )}
              </View>
            </View>

            <View style={styles.heroActions}>
              {isPro ? (
                <PrimaryGradientButton
                  label={t('common.upgrade')}
                  icon="diamond-outline"
                  onPress={() => navigation.navigate('Paywall')}
                  style={styles.heroActionPrimary}
                />
              ) : (
                <PrimaryGradientButton
                  label={t('profile.goPro')}
                  icon="rocket-outline"
                  onPress={() => navigation.navigate('Paywall')}
                  style={styles.heroActionPrimary}
                />
              )}
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
                    <ScaledText style={styles.statValue}>{item.value}</ScaledText>
                    <ScaledText style={styles.statLabel}>{item.label}</ScaledText>
                  </View>
                  {index < stats.length - 1 ? <View style={styles.statDivider} /> : null}
                </React.Fragment>
              ))}
            </View>
          </GlassPanel>
        </Animated.View>

        {/* ── Subscription upgrade card (free users only) ── */}
        {!isPro && (
          <Animated.View entering={FadeInDown.delay(58).duration(350)} style={styles.upgradeCardWrap}>
            <TouchableOpacity
              activeOpacity={0.88}
              onPress={() => {
                triggerLightHaptic();
                navigation.navigate('Paywall');
              }}
            >
              <LinearGradient
                colors={[D.accentStart, D.accentEnd]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.upgradeCard}
              >
                {/* Decorative orb */}
                <View style={styles.upgradeOrb} />

                <View style={styles.upgradeLeft}>
                  <View style={styles.upgradeBadge}>
                    <Ionicons name="diamond" size={11} color={D.accentEnd} />
                    <ScaledText style={styles.upgradeBadgeText}>{t('profile.proBadge')}</ScaledText>
                  </View>
                  <ScaledText style={styles.upgradeTitle}>{t('profile.goPro')}</ScaledText>
                  <ScaledText style={styles.upgradeSubtitle}>{t('profile.goProSubtitle')}</ScaledText>
                </View>

                <View style={styles.upgradeRight}>
                  <View style={styles.upgradePriceWrap}>
                    <ScaledText style={styles.upgradePriceFrom}>{t('profile.priceFrom')}</ScaledText>
                    <ScaledText style={styles.upgradePrice}>$9.99</ScaledText>
                    <ScaledText style={styles.upgradePricePer}>/mo</ScaledText>
                  </View>
                  <View style={styles.upgradeArrow}>
                    <Ionicons name="arrow-forward" size={16} color={D.white} />
                  </View>
                </View>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>
        )}

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
                        style={styles.segmentedActiveFill}
                      />
                    ) : null}
                    <ScaledText style={[styles.segmentedText, isActive && styles.segmentedTextActive]}>
                      {tab === 'looks' ? t('profile.savedProductsTab', { defaultValue: 'Saved' }) : titleCase(tab)}
                    </ScaledText>
                    <View style={[styles.segmentedBadge, isActive && styles.segmentedBadgeActive]}>
                      <ScaledText style={[styles.segmentedBadgeText, isActive && styles.segmentedBadgeTextActive]}>
                        {tab === 'looks' ? savedProducts.length : tripPlanDays.length}
                      </ScaledText>
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
              <ScaledText style={styles.sectionEyebrow}>{activeTab === 'looks' ? t('profile.style') : t('profile.travel')}</ScaledText>
              <ScaledText style={styles.sectionTitle}>{activeTab === 'looks' ? t('profile.savedProducts', { defaultValue: 'Saved Items' }) : t('profile.tripsAndPacking')}</ScaledText>
            </View>
            {activeTab === 'looks' ? (
              <SecondaryGlassButton
                label={t('profile.browseShop', { defaultValue: 'Browse shop' })}
                icon="heart-outline"
                onPress={() => navigation.navigate('Main', { screen: 'Inspo', params: { openShop: true } })}
              />
            ) : (
              <SecondaryGlassButton
                label={t('profile.openCalendar')}
                icon="calendar-outline"
                onPress={() => openTripPlanner()}
              />
            )}
          </View>

          {activeTab === 'looks' ? renderLooksContent() : renderTripsContent()}
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(130).duration(350)} style={styles.sectionBlock}>
          <ScaledText style={styles.groupHeading}>{t('profile.accountHeading')}</ScaledText>
          <GlassPanel radius={28}>
            <MenuRow
              icon="create-outline"
              label={t('profile.editProfile')}
              onPress={openEditProfile}
            />
            {isAdminUser && (
              <>
                <View style={styles.menuSeparator} />
                <MenuRow
                  icon="shield-checkmark"
                  label={t('admin.title', 'Admin Panel')}
                  onPress={() => navigation.navigate('AdminPanel')}
                />
              </>
            )}
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(155).duration(350)} style={styles.sectionBlock}>
          <ScaledText style={styles.groupHeading}>{t('profile.membershipHeading')}</ScaledText>
          <GlassPanel radius={28}>
            <MenuRow
              icon="diamond-outline"
              label={t('profile.subscription')}
              trailing={planLabel}
              onPress={() => navigation.navigate('Paywall')}
            />
            <View style={styles.menuSeparator} />
            {/* Apple Offer Code redemption — compliant with Guideline 3.1.1 */}
            <MenuRow
              icon="ticket-outline"
              label={t('profile.redeemOfferCode', 'Redeem Offer Code')}
              onPress={() => iapService.presentCodeRedemptionSheet()}
            />
            {effectiveTier !== 'free' && (
              <>
                <View style={styles.menuSeparator} />
                <MenuRow
                  icon="settings-outline"
                  label={t('paywall.manageSubscription')}
                  onPress={() => iapService.manageSubscriptions()}
                />
              </>
            )}
            <View style={styles.menuSeparator} />
            {/* Terms of Use link — required by Guideline 3.1.2(c) */}
            <MenuRow
              icon="document-text-outline"
              label={t('paywall.termsOfUse', 'Terms of Use')}
              onPress={() => navigation.navigate('TermsOfService')}
            />
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(180).duration(350)} style={styles.sectionBlock}>
          <ScaledText style={styles.groupHeading}>{t('profile.preferencesHeading')}</ScaledText>
          <GlassPanel radius={28}>
            <MenuRow
              icon="language-outline"
              label={t('language.selectLanguage')}
              trailing={LANGUAGE_NAMES[currentLanguage]}
              onPress={() => {
                triggerLightHaptic();
                setShowLanguageModal(true);
              }}
            />
            <View style={styles.menuSeparator} />
            <MenuRow
              icon="notifications-outline"
              label={t('profile.notifications')}
              onPress={() => Alert.alert(t('profile.comingSoon'), t('profile.notificationSettingsUnavailable'))}
            />
            <View style={styles.menuSeparator} />
            <MenuRow
              icon="analytics-outline"
              label={t('profile.analyticsSharing', 'Analytics sharing')}
              trailing={analyticsEnabled ? t('profile.analyticsOn', 'On') : t('profile.analyticsOff', 'Off')}
              onPress={async () => {
                const next = !analyticsEnabled;
                setAnalyticsEnabled(next);
                await analyticsService.setEnabled(next);
              }}
            />
            <View style={styles.menuSeparator} />
            <MenuRow
              icon="shield-checkmark-outline"
              label={t('profile.privacyPolicy')}
              onPress={() => navigation.navigate('PrivacyPolicy')}
            />
          </GlassPanel>
        </Animated.View>

        {__DEV__ && isAdminUser && (
          <Animated.View entering={FadeInDown.delay(190).duration(350)} style={styles.sectionBlock}>
            <ScaledText style={styles.groupHeading}>{t('profile.developerTools')}</ScaledText>
            <GlassPanel radius={28}>
              <MenuRow
                icon="time-outline"
                label="Force Expire Trial (8 days ago)"
                onPress={handleForceExpireTrial}
              />
              <View style={styles.menuSeparator} />
              <MenuRow
                icon="refresh-outline"
                label="Reset Trial (Clear Storage)"
                onPress={handleResetTrial}
              />
            </GlassPanel>
          </Animated.View>
        )}

        <Animated.View entering={FadeInDown.delay(205).duration(350)} style={styles.sectionBlock}>
          <ScaledText style={styles.groupHeading}>{t('profile.session')}</ScaledText>
          <GlassPanel radius={28}>
            <MenuRow
              icon="log-out-outline"
              label={t('profile.signOut')}
              onPress={handleLogout}
              danger
            />
          </GlassPanel>
        </Animated.View>

        <Animated.View entering={FadeInDown.delay(230).duration(350)} style={styles.deleteAccountWrap}>
          <SecondaryGlassButton
            label={t('profile.deleteAccount')}
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
              <View style={styles.sheetHandle} />
              <ScaledText style={styles.sheetTitle}>{t('profile.editAccount')}</ScaledText>
              <ScaledText style={styles.sheetSubtitle}>{t('profile.updateDetails')}</ScaledText>

              <TouchableOpacity style={styles.sheetAvatarWrap} activeOpacity={0.85} onPress={pickAvatar}>
                <LinearGradient
                  colors={['rgba(255,255,255,0.98)', D.accentSoftStrong]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.sheetAvatarHalo}
                >
                  {editAvatar || userAvatar ? (
                    <CachedImage
                      uri={(editAvatar || userAvatar) as string}
                      requestHeaders={editAvatar ? undefined : privateAvatarHeaders}
                      cachePolicy="memory"
                      style={styles.sheetAvatar}
                      contentFit="cover"
                      fadeIn={false}
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
                    style={StyleSheet.absoluteFill}
                  />
                  <Ionicons name="camera" size={13} color={D.white} />
                </View>
              </TouchableOpacity>

              <ScaledText style={styles.fieldLabel}>{t('profile.name')}</ScaledText>
              <View style={styles.fieldCard}>
                <TextInput
                  style={styles.fieldInput}
                  value={editName}
                  onChangeText={setEditName}
                  placeholder={t('profile.displayName')}
                  maxLength={200}
                  accessibilityLabel={t('profile.displayName')}
                  placeholderTextColor={D.textMute}
                  autoCapitalize="none"
                  autoCorrect={false}
                />
              </View>

              <ScaledText style={styles.fieldLabel}>{t('profile.handle')}</ScaledText>
              <View style={styles.fieldCard}>
                <TextInput
                  style={styles.fieldInput}
                  value={editHandle}
                  onChangeText={setEditHandle}
                  placeholder={t('auth.username')}
                  placeholderTextColor={D.textMute}
                  autoCapitalize="none"
                  autoCorrect={false}
                  maxLength={30}
                  accessibilityLabel={t('profile.handle')}
                />
              </View>

              <ScaledText style={styles.fieldLabel}>{t('profile.email')}</ScaledText>
              <View style={styles.fieldCard}>
                <View style={styles.readonlyRow}>
                  <Ionicons name="mail-outline" size={16} color={D.textSub} />
                  <ScaledText style={styles.readonlyValue}>{userEmail || t('profile.noEmail')}</ScaledText>
                </View>
              </View>

              <View style={styles.sheetActions}>
                <PrimaryGradientButton
                  label={saving ? t('common.loading') : t('common.save')}
                  icon={saving ? 'time-outline' : 'checkmark-outline'}
                  onPress={handleSaveProfile}
                  style={styles.sheetPrimaryAction}
                />
                <SecondaryGlassButton
                  label={t('common.cancel')}
                  icon="close-outline"
                  onPress={() => setShowEditProfile(false)}
                  style={styles.sheetSecondaryAction}
                />
              </View>

              {saving ? (
                <View style={styles.savingRow}>
                  <ActivityIndicator size="small" color={D.accent} />
                  <ScaledText style={styles.savingText}>{t('profile.savingUpdates')}</ScaledText>
                </View>
              ) : null}
            </View>
          </Animated.View>
        </KeyboardAvoidingView>
      </Modal>

      {previewLook ? (
        <Modal transparent animationType="fade" visible onRequestClose={() => setPreviewLook(null)}>
          <View style={styles.previewOverlay}>
            <Pressable style={StyleSheet.absoluteFill} onPress={() => setPreviewLook(null)} />

            <Animated.View entering={FadeIn.duration(220)} style={styles.previewShell}>
              <View style={styles.previewCard}>
                {/* Header: title block + close */}
                <View style={styles.previewHeader}>
                  <View style={styles.previewHeaderText}>
                    <ScaledText style={styles.previewTitle} numberOfLines={1}>
                      {previewTitleText}
                    </ScaledText>
                    {previewLook.date ? (
                      <ScaledText style={styles.previewDate} numberOfLines={1}>
                        {previewLook.date}
                      </ScaledText>
                    ) : null}
                  </View>
                  <TouchableOpacity
                    style={styles.previewClose}
                    onPress={() => {
                      triggerLightHaptic();
                      setPreviewLook(null);
                    }}
                    accessibilityRole="button"
                    accessibilityLabel={t('common.close')}
                  >
                    <Ionicons name="close" size={16} color="#0A1931" />
                  </TouchableOpacity>
                </View>

                {/* Hero: full outfit collage or try-on result */}
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
                ) : (
                  <View style={styles.previewCollage}>
                    <OutfitCollagePreview
                      items={previewPieces}
                      backgroundColor="#F4F6FA"
                      placeholderIconColor="rgba(15,23,42,0.22)"
                      placeholderIconSize={38}
                      footerInset={16}
                    />
                  </View>
                )}

                {/* Piece breakdown strip */}
                {previewLook.image === undefined && previewPieces.length > 1 ? (
                  <View style={styles.previewPiecesWrap}>
                    <ScaledText style={styles.previewPiecesTitle}>
                      {t('profile.inThisLook')}
                    </ScaledText>
                    <ScrollView
                      horizontal
                      showsHorizontalScrollIndicator={false}
                      contentContainerStyle={styles.previewPiecesRow}
                    >
                      {previewPieces.map((piece, index) => {
                        const target = previewTargets[index];
                        const isWardrobe = !!target && target.kind === 'wardrobe';
                        const pieceId = String((piece as { id?: unknown }).id ?? '');
                        const navigable = pieceId.length > 0;
                        return (
                          <TouchableOpacity
                            key={String(piece.id ?? index)}
                            style={styles.previewPiece}
                            activeOpacity={navigable ? 0.7 : 1}
                            disabled={!navigable}
                            onPress={() => handlePiecePress(index)}
                            accessibilityRole="button"
                            accessibilityLabel={humanizePieceLabel(piece, index)}
                            accessibilityHint={t(
                              isWardrobe
                                ? 'profile.openInWardrobe'
                                : 'profile.openInShop',
                            )}
                          >
                            <View style={styles.previewPieceThumb}>
                              {typeof piece.image === 'number' ? (
                                <Image
                                  source={piece.image}
                                  style={styles.previewPieceImage}
                                  resizeMode="contain"
                                />
                              ) : (
                                <CachedImage
                                  uri={piece.image as string}
                                  style={styles.previewPieceImage}
                                  contentFit="contain"
                                  fadeIn={false}
                                />
                              )}
                              {navigable ? (
                                <View style={styles.previewPieceGo}>
                                  <Ionicons
                                    name={isWardrobe ? 'shirt-outline' : 'bag-outline'}
                                    size={10}
                                    color="#FFFFFF"
                                  />
                                </View>
                              ) : null}
                            </View>
                            <ScaledText style={styles.previewPieceLabel} numberOfLines={1}>
                              {humanizePieceLabel(piece, index)}
                            </ScaledText>
                          </TouchableOpacity>
                        );
                      })}
                    </ScrollView>
                  </View>
                ) : null}

                <View style={styles.previewActions}>
                  {previewLook.isShop ? (
                    <SecondaryGlassButton
                      label={t('profile.viewInShop', { defaultValue: 'View in Shop' })}
                      icon="bag-outline"
                      onPress={openShopPreviewInShop}
                      style={styles.previewActionButton}
                    />
                  ) : null}
                  {previewLook.isShop ? (
                    <PrimaryGradientButton
                      label={t('inspo.buyNow', { defaultValue: 'Buy' })}
                      icon="arrow-forward"
                      onPress={buyShopPreview}
                      style={styles.previewActionButton}
                    />
                  ) : null}
                  {previewLook.isWardrobe ? (
                    <SecondaryGlassButton
                      label={t('profile.openInWardrobe', { defaultValue: 'Open in Wardrobe' })}
                      icon="shirt-outline"
                      onPress={openWardrobePreviewInCloset}
                      style={styles.previewActionButton}
                    />
                  ) : null}

                  <SecondaryGlassButton
                    label={
                      previewLook.isShop
                        ? t('common.remove')
                        : t('common.cancel')
                    }
                    icon={
                      previewLook.isShop ? 'trash-outline' : 'close-outline'
                    }
                    onPress={() => {
                      if (previewLook.isShop) {
                        removeShopPreview();
                        return;
                      }

                      setPreviewLook(null);
                    }}
                    danger={previewLook.isShop}
                    style={styles.previewActionButton}
                  />
                </View>
              </View>
            </Animated.View>
          </View>
        </Modal>
      ) : null}

      <LanguageSwitcher
        visible={showLanguageModal}
        onClose={() => setShowLanguageModal(false)}
      />
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
      ...StyleSheet.absoluteFill,
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
      ...StyleSheet.absoluteFill,
      backgroundColor: D.glassStrong,
    },
    glassContent: {
      overflow: 'hidden',
      padding: 10,
      justifyContent: 'center',
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
    // ── Upgrade card (free users) ──
    upgradeCardWrap: {
      marginTop: 14,
      marginHorizontal: 20,
      borderRadius: 24,
      shadowColor: D.accentEnd,
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.38,
      shadowRadius: 20,
      elevation: 12,
    },
    upgradeCard: {
      borderRadius: 24,
      paddingVertical: 20,
      paddingHorizontal: 22,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      overflow: 'hidden',
    },
    upgradeOrb: {
      position: 'absolute',
      width: 160,
      height: 160,
      borderRadius: 80,
      backgroundColor: 'rgba(255,255,255,0.08)',
      top: -50,
      right: -30,
    },
    upgradeLeft: {
      flex: 1,
      gap: 4,
    },
    upgradeBadge: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 5,
      backgroundColor: 'rgba(255,255,255,0.92)',
      alignSelf: 'flex-start',
      paddingHorizontal: 9,
      paddingVertical: 3,
      borderRadius: 8,
      marginBottom: 4,
    },
    upgradeBadgeText: {
      fontSize: 10,
      fontWeight: '800',
      color: D.accentEnd,
      letterSpacing: 0.8,
    },
    upgradeTitle: {
      fontSize: 18,
      fontWeight: '800',
      color: '#FFFFFF',
      letterSpacing: -0.4,
    },
    upgradeSubtitle: {
      fontSize: 12,
      color: 'rgba(255,255,255,0.78)',
      fontWeight: '500',
    },
    upgradeRight: {
      alignItems: 'flex-end',
      gap: 10,
    },
    upgradePriceWrap: {
      alignItems: 'flex-end',
    },
    upgradePriceFrom: {
      fontSize: 10,
      color: 'rgba(255,255,255,0.72)',
      fontWeight: '500',
    },
    upgradePrice: {
      fontSize: 26,
      fontWeight: '800',
      color: '#FFFFFF',
      letterSpacing: -0.5,
    },
    upgradePricePer: {
      fontSize: 11,
      color: 'rgba(255,255,255,0.72)',
      fontWeight: '500',
    },
    upgradeArrow: {
      width: 34,
      height: 34,
      borderRadius: 17,
      backgroundColor: 'rgba(255,255,255,0.18)',
      alignItems: 'center',
      justifyContent: 'center',
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
      justifyContent: 'center',
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
      alignItems: 'center',
      padding: 6,
      gap: 8,
    },
    segmentedButton: {
      flex: 1,
      flexBasis: 0,
      flexGrow: 1,
      flexShrink: 1,
      minHeight: 48,
      borderRadius: 999,
      alignItems: 'center',
      justifyContent: 'center',
      flexDirection: 'row',
      gap: 8,
      overflow: 'hidden',
    },
    segmentedButtonActive: {
      // Solid navy under the gradient so the white active label stays
      // readable even if the gradient layer fails to paint.
      backgroundColor: D.accentEnd,
      shadowColor: D.accent,
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.24,
      shadowRadius: 18,
      elevation: 8,
    },
    segmentedActiveFill: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
      borderRadius: 999,
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
      marginTop: 28,
      paddingHorizontal: 24,
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
    // Connected trip plan (planned calendar days + packing list)
    tripPlanList: {
      marginTop: 6,
      gap: 12,
    },
    tripPlanSummary: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      gap: 12,
      padding: 14,
      borderRadius: 20,
      backgroundColor: D.accentSoft,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    tripDayCard: {
      padding: 14,
      borderRadius: 20,
      gap: 10,
      backgroundColor: D.isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.46)',
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    tripDayHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 8,
    },
    tripDayDate: {
      fontSize: 15,
      fontWeight: '800',
      color: D.text,
      letterSpacing: -0.2,
    },
    tripOccasionPill: {
      paddingHorizontal: 10,
      paddingVertical: 5,
      borderRadius: 14,
    },
    tripOccasionPillText: {
      fontSize: 12,
      fontWeight: '700',
    },
    tripDayThumbs: {
      flexDirection: 'row',
      gap: 8,
    },
    tripDayThumb: {
      width: 52,
      height: 52,
      borderRadius: 14,
      overflow: 'hidden',
    },
    tripDayMore: {
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: D.accentSoft,
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    tripDayMoreText: {
      fontSize: 13,
      fontWeight: '800',
      color: D.accent,
    },
    tripPackingCard: {
      padding: 14,
      borderRadius: 20,
      gap: 10,
      backgroundColor: D.isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.46)',
      borderWidth: 1,
      borderColor: D.glassBorderSoft,
    },
    packingTitle: {
      fontSize: 14,
      fontWeight: '800',
      color: D.text,
      letterSpacing: -0.2,
    },
    packingGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 10,
    },
    packingItem: {
      width: 64,
      alignItems: 'center',
      gap: 4,
    },
    packingThumb: {
      width: 56,
      height: 56,
      borderRadius: 14,
      overflow: 'hidden',
    },
    packingName: {
      fontSize: 10,
      fontWeight: '600',
      color: D.textSub,
      textAlign: 'center',
    },
    packingCount: {
      fontSize: 10,
      fontWeight: '800',
      color: D.accent,
    },
    packingMore: {
      fontSize: 12,
      fontWeight: '600',
      color: D.textSub,
    },
    lookGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: LOOK_CARD_GAP,
    },
    lookRow: {
      flexDirection: 'row',
      gap: LOOK_CARD_GAP,
      paddingRight: 20,
      paddingTop: 4,
      paddingBottom: 12,
    },
    lookCardShadow: {
      width: LOOK_ROW_CARD_WIDTH,
      borderRadius: 26,
      shadowColor: D.shadow,
      shadowOffset: { width: 0, height: 18 },
      shadowOpacity: D.isDark ? 0.35 : 0.14,
      shadowRadius: 24,
      elevation: 12,
    },
    lookCard: {
      width: LOOK_ROW_CARD_WIDTH,
      height: LOOK_ROW_CARD_WIDTH * 1.36,
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
    lookFooterBar: {
      position: 'absolute',
      left: 12,
      right: 12,
      bottom: 12,
      borderRadius: 18,
      overflow: 'hidden',
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.7)',
      shadowColor: '#0A1931',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.12,
      shadowRadius: 10,
      elevation: 4,
    },
    lookFooterScrim: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
      backgroundColor: 'rgba(255,255,255,0.55)',
    },
    lookFooterContent: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 10,
      paddingHorizontal: 12,
      paddingVertical: 10,
    },
    lookFooterText: {
      flex: 1,
      gap: 2,
    },
    lookFooterAction: {
      width: 32,
      height: 32,
      borderRadius: 16,
      backgroundColor: '#0A1931',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
    },
    lookTitle: {
      fontSize: 13,
      lineHeight: 18,
      fontWeight: '700',
      color: '#0A1931',
      letterSpacing: -0.2,
      textTransform: 'capitalize',
    },
    lookDate: {
      fontSize: 11,
      lineHeight: 15,
      color: 'rgba(10,25,49,0.55)',
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
      marginTop:7,
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
      borderBottomLeftRadius: 28,
      borderBottomRightRadius: 28,
      borderWidth: 1,
      borderColor: D.sheetBorder,
      backgroundColor: D.sheet,
      paddingHorizontal: 20,
      paddingTop: 12,
      paddingBottom: 6,
      shadowColor: '#000000',
      shadowOffset: { width: 0, height: -8 },
      shadowOpacity: D.isDark ? 0.4 : 0.16,
      shadowRadius: 24,
      elevation: 16,
    },
    sheetScrim: {
      ...StyleSheet.absoluteFill,
      backgroundColor: D.sheet,
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
      borderColor: D.fieldBorder,
      backgroundColor: D.field,
      marginBottom: 16,
    },
    fieldScrim: {
      ...StyleSheet.absoluteFill,
      backgroundColor: D.field,
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
    },
    previewHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 12,
      paddingHorizontal: 20,
      paddingTop: 18,
      paddingBottom: 12,
    },
    previewHeaderText: {
      flex: 1,
      gap: 4,
    },
    previewClose: {
      width: 36,
      height: 36,
      borderRadius: 18,
      backgroundColor: 'rgba(10,25,49,0.06)',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: 0,
    },
    previewPiecesWrap: {
      paddingTop: 14,
    },
    previewPiecesTitle: {
      fontSize: 11,
      fontWeight: '700',
      color: 'rgba(10,25,49,0.45)',
      letterSpacing: 1.2,
      textTransform: 'uppercase',
      paddingHorizontal: 20,
      marginBottom: 10,
    },
    previewPiecesRow: {
      paddingHorizontal: 20,
      paddingBottom: 4,
      gap: 12,
    },
    previewPiece: {
      width: 76,
      alignItems: 'center',
      gap: 6,
    },
    previewPieceThumb: {
      width: 76,
      height: 76,
      borderRadius: 18,
      backgroundColor: '#FFFFFF',
      borderWidth: 1,
      borderColor: 'rgba(10,25,49,0.08)',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
    },
    previewPieceGo: {
      position: 'absolute',
      right: 5,
      bottom: 5,
      width: 20,
      height: 20,
      borderRadius: 10,
      backgroundColor: '#0A1931',
      alignItems: 'center',
      justifyContent: 'center',
    },
    previewPieceImage: {
      width: 64,
      height: 64,
    },
    previewPieceLabel: {
      fontSize: 10.5,
      fontWeight: '600',
      color: 'rgba(10,25,49,0.55)',
      textAlign: 'center',
    },
    previewBadgeRow: {
      alignSelf: 'flex-start',
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderRadius: 999,
      backgroundColor: 'rgba(10,25,49,0.08)',
      borderWidth: 1,
      borderColor: 'rgba(10,25,49,0.12)',
      marginBottom: 8,
    },
    previewBadgeText: {
      fontSize: 12,
      lineHeight: 16,
      fontWeight: '700',
      color: '#0A1931',
      letterSpacing: 0.3,
    },
    previewTitle: {
      fontSize: 20,
      lineHeight: 26,
      fontWeight: '800',
      color: '#0A1931',
      letterSpacing: -0.4,
      textTransform: 'capitalize',
    },
    previewDate: {
      fontSize: 12,
      lineHeight: 16,
      color: 'rgba(10,25,49,0.55)',
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
