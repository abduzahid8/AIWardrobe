import { authEvents } from '../store/authEvents';
/**
 * InspoScreen — Inspiration page
 * Minimalist Liquid Glass design with Guide + Shop tabs
 */

import React, { useState, useCallback, useMemo, useEffect, useRef, useDeferredValue } from 'react';
import { View, TouchableOpacity, StyleSheet, Dimensions, ScrollView, FlatList, StatusBar, TextInput, ActivityIndicator, Linking, Platform,  } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRoute, type RouteProp } from '@react-navigation/native';
import { useAppNavigation } from '../hooks/useAppNavigation';
import type { TabParamList } from '../navigation/types';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import useSavedShopStore from '../store/savedShopStore';
import { supabase } from '../lib/supabase';
import Animated, {
    FadeInDown,
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withSequence,
    withTiming,
} from 'react-native-reanimated';

import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import { CachedImage } from '../components/ui/CachedImage';
import type { ShopCatalogItem } from '../src/types/shop';
import { INSPO_MENS_SHOP_ITEMS, CLASSIC_MENS_ITEMS } from '../data/inspoMensShopItems';
import { useShopCatalog } from '../hooks/useShopCatalog';
import { useFeaturedCapsules, type FeaturedCapsule } from '../hooks/useFeaturedCapsules';
import { useTranslation } from 'react-i18next';
import { formatMoney } from '../src/utils/outfitBudget';
import { inferShopRowMacro } from '../src/utils/shopCatalogOrder';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

// ── Data ──────────────────────────────────────

// Featured Capsules now come from Supabase (`featured_capsules` table) via
// `useFeaturedCapsules`. Rows are admin-editable from the Supabase dashboard.

const CAPSULE_CARD_WIDTH = 180;
const CAPSULE_CARD_HEIGHT = 250;
const PRODUCT_CARD_WIDTH = (SCREEN_WIDTH - spacing.screenPadding * 2 - spacing.sm) / 2;

// ── Shop price + spotlight helpers ──────────────────────────────────────
// shop_catalog rows are priced in UZS (median ~1.2M so'm) — never render
// them with a `$` prefix. formatMoney('UZS') groups as "2 232 000 UZS".
const formatShopPrice = (item: ShopCatalogItem): string => {
    const n = Number((item as unknown as { price?: unknown })?.price);
    if (!Number.isFinite(n) || n <= 0) return '';
    return formatMoney(n, 'UZS');
};

const getShopItemMacro = (item: ShopCatalogItem): string => {
    try {
        const g = String((item as unknown as { garmentType?: unknown })?.garmentType ?? '');
        const n = String((item as unknown as { name?: unknown })?.name ?? '');
        return inferShopRowMacro(g, n);
    } catch {
        const g = String((item as unknown as { garmentType?: unknown })?.garmentType ?? '');
        if (g === 'shoes') return 'shoes';
        if (g === 'lower_body') return 'bottom';
        return 'top';
    }
};

// Big (spotlight) cards are editorial covers — shoes or outerwear only,
// never pants/bottoms. Small pair cards can show anything.
const isSpotlightEligible = (item: ShopCatalogItem): boolean => {
    const macro = getShopItemMacro(item);
    return macro === 'shoes' || macro === 'outerwear';
};

const isPantsItem = (item: ShopCatalogItem): boolean =>
    getShopItemMacro(item) === 'bottom';

// ── Sub-Components ──────────────────────────────────────

const FeaturedCapsuleCard = React.memo(({ item, index, t }: { item: FeaturedCapsule; index: number; t: (key: string) => string }) => (
    <Animated.View entering={index < 8 ? FadeInDown.delay(100 + index * 80).duration(400) : undefined}>
        <TouchableOpacity
            style={styles.capsuleCard}
            activeOpacity={0.9}
            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
            accessibilityLabel={`${item.title} ${t('inspo.capsule')}`}
            accessibilityRole="button"
        >
            <CachedImage
                uri={item.imageUrl}
                style={styles.capsuleImage}
                contentFit="cover"
                fadeIn={false}
            />
            <LinearGradient
                colors={['transparent', 'rgba(0,0,0,0.65)']}
                style={styles.capsuleGradient}
            >
                <ScaledText style={styles.capsuleTitle}>{item.title}</ScaledText>
            </LinearGradient>
        </TouchableOpacity>
    </Animated.View>
), (prev, next) => prev.item.id === next.item.id && prev.item.imageUrl === next.item.imageUrl && prev.index === next.index);
FeaturedCapsuleCard.displayName = 'FeaturedCapsuleCard';

const trackBrandClick = async (item: ShopCatalogItem) => {
    try {
        await supabase.rpc('record_brand_click', {
            p_item_id: item.id,
            p_brand: item.brand,
            p_product_name: item.name,
            p_price: item.price,
            p_currency: item.currency || 'USD',
            p_source: 'app',
            p_device_type: Platform.OS,
        });
    } catch (_) {
        // Silent fail - don't block user experience
    }
};

const handleBuyPress = async (item: ShopCatalogItem) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    // Track the brand click in the background (fire-and-forget) to not block redirect
    trackBrandClick(item);

    // Open brand website (search for product)
    const searchQuery = encodeURIComponent(`${item.brand} ${item.name}`);
    const url = `https://www.google.com/search?q=${searchQuery}`;

    try {
        const canOpen = await Linking.canOpenURL(url);
        if (canOpen) {
            await Linking.openURL(url);
        }
    } catch (_) {
    }
};

// Memoized — previously every saveInspoSet change re-rendered ALL cards
// and re-ran the FadeInDown stagger per row (full grid reconcile on each heart tap).
const ProductCard = React.memo(({
    item,
    isSaved,
    onSave,
    index,
    t,
    spotlight = false,
    plate,
    highlighted = false,
}: {
    item: ShopCatalogItem;
    isSaved: boolean;
    onSave: (item: ShopCatalogItem) => void;
    index: number;
    t: (key: string) => string;
    spotlight?: boolean;
    plate?: number;
    highlighted?: boolean;
}) => {
    const handleSave = React.useCallback(() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onSave(item);
    }, [onSave, item]);
    const handleCardPress = React.useCallback(() => {
        handleBuyPress(item);
    }, [item]);
    // Only animate the first screenful — later pages previously waited up to
    // 600ms invisible (delay cap) making "Load more" look frozen.
    const entering = index < 8 ? FadeInDown.delay(Math.min(150 + (index % 12) * 40, 600)).duration(320) : undefined;
    return (
    // Cap the stagger so later pages (index 100+) don't wait many seconds
    // before their entering animation starts — otherwise pressing Load more
    // appears to do nothing because new cards are invisible until the delay elapses.
    <Animated.View entering={entering}>
        <TouchableOpacity
            style={[styles.productCard, spotlight && styles.productCardSpotlight]}
            onPress={handleCardPress}
            activeOpacity={0.9}
            accessibilityRole="button"
            accessibilityLabel={`${item.brand} ${item.name} ${formatShopPrice(item)}`}
            accessibilityHint={t('inspo.buyNow')}
        >
            <View
                style={[
                    styles.productImageBox,
                    spotlight && styles.productImageBoxSpotlight,
                    highlighted && styles.productImageBoxHighlighted,
                ]}
            >
                <CachedImage
                    uri={typeof item.imageUrl === 'string' ? item.imageUrl : ''}
                    style={styles.productImage}
                    contentFit={spotlight ? 'cover' : 'contain'}
                    contentPosition="center"
                    fadeIn={false}
                />

                {/* A lookbook plate number, quiet and typographic — the one
                    editorial flourish this flat design allows itself. */}
                {spotlight && typeof plate === 'number' && (
                    <ScaledText style={styles.plateNumber} pointerEvents="none">
                        {String(plate).padStart(2, '0')}
                    </ScaledText>
                )}

                <TouchableOpacity
                    style={styles.saveButton}
                    onPress={handleSave}
                    accessibilityLabel={isSaved ? t('inspo.removeFromSaved') : t('inspo.saveItem')}
                    accessibilityHint={isSaved ? t('inspo.unsavedToast') : t('inspo.savedToast')}
                    accessibilityRole="button"
                    accessibilityState={{ selected: isSaved }}
                    hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                >
                    <View style={styles.saveButtonCircle}>
                        <Ionicons
                            name={isSaved ? 'heart' : 'heart-outline'}
                            size={spotlight ? 20 : 18}
                            color={isSaved ? '#FF3B30' : '#0A1931'}
                        />
                    </View>
                </TouchableOpacity>
            </View>

            <View style={styles.caption}>
                <ScaledText style={[styles.captionBrand, spotlight && styles.captionBrandLarge]} numberOfLines={1}>
                    {item.brand}
                </ScaledText>
                <ScaledText style={[styles.captionPrice, spotlight && styles.captionPriceLarge]}>
                    {formatShopPrice(item)}
                </ScaledText>
            </View>
        </TouchableOpacity>
    </Animated.View>
    );
}, (prev, next) =>
    prev.item.id === next.item.id &&
    prev.isSaved === next.isSaved &&
    prev.highlighted === next.highlighted &&
    prev.spotlight === next.spotlight &&
    prev.plate === next.plate &&
    prev.index === next.index &&
    prev.onSave === next.onSave,
);
ProductCard.displayName = 'ProductCard';

// Memoized pair-row wrapper — previously renderShopRow built a fresh
// `<View style={styles.shopRow}>` tree on every data change, so a single
// heart toggle re-rendered ALL rows' wrappers (plus re-ran every card's
// entering animation check). With this memo, only rows whose card props
// actually changed re-render.
const ShopPairRow = React.memo(({
    left,
    right,
    leftIndex,
    t,
    isLeftSaved,
    isRightSaved,
    onSave,
    focusedShopItemId,
}: {
    left: ShopCatalogItem;
    right: ShopCatalogItem | null;
    leftIndex: number;
    t: (key: string) => string;
    isLeftSaved: boolean;
    isRightSaved: boolean;
    onSave: (item: ShopCatalogItem) => void;
    focusedShopItemId: string | null;
}) => (
    <View style={styles.shopRow}>
        <View style={styles.productCardWrap}>
            <ProductCard
                item={left}
                isSaved={isLeftSaved}
                onSave={onSave}
                index={leftIndex}
                t={t}
                highlighted={focusedShopItemId === String(left.id)}
            />
        </View>
        {right ? (
            <View style={styles.productCardWrap}>
                <ProductCard
                    item={right}
                    isSaved={isRightSaved}
                    onSave={onSave}
                    index={leftIndex + 1}
                    t={t}
                    highlighted={focusedShopItemId === String(right.id)}
                />
            </View>
        ) : (
            <View style={styles.productCardWrap} />
        )}
    </View>
));
ShopPairRow.displayName = 'ShopPairRow';

// A loading placeholder shaped like the real product tile (price tag +
// brand strip silhouettes) so the grid doesn't jump when real cards land.
const ShimmerProductCard = () => {
    const opacity = useSharedValue(0.55);

    useEffect(() => {
        opacity.value = withRepeat(
            withSequence(withTiming(1, { duration: 750 }), withTiming(0.55, { duration: 750 })),
            -1,
            true,
        );
    }, [opacity]);

    const animatedStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

    return (
        <Animated.View style={[styles.skeletonCard, animatedStyle]}>
            <View style={styles.skeletonImage} />
            <View style={styles.skeletonBrandLine} />
            <View style={styles.skeletonPriceLine} />
        </Animated.View>
    );
};

// A clean plate: the photo carries no overlay at all, every word lives in
// the flat caption beneath it — a printed lookbook index page rather than
// an Instagram-style gradient card. The italic plate number is the same
// device the Shop tab's spotlight cards use, so the two tabs read as one
// numbered catalogue.
const GuideGridCard = ({
    item,
    plate,
    onPress,
    t,
}: {
    item: any | null;
    plate: number;
    onPress: (item: any) => void;
    t: (key: string) => string;
}) => {
    if (!item) return <View style={styles.guideGridCardWrap} />;
    return (
        <View style={styles.guideGridCardWrap}>
            <TouchableOpacity
                style={styles.guideGridCard}
                activeOpacity={0.92}
                onPress={() => onPress(item)}
                accessibilityRole="button"
                accessibilityLabel={`${item.title}. ${t('inspo.shopThisLook')}`}
            >
                <CachedImage
                    uri={typeof item.image === 'string' ? item.image : ''}
                    style={styles.guideImage}
                    contentFit="cover"
                    fadeIn={false}
                />
            </TouchableOpacity>
            <View style={styles.guideGridCaption}>
                <ScaledText style={styles.guideGridPlate}>{String(plate).padStart(2, '0')}</ScaledText>
                <ScaledText style={styles.guideGridCardTitle} numberOfLines={1}>{item.title}</ScaledText>
                {!!item.subtitle && (
                    <ScaledText style={styles.guideGridCardSubtitle} numberOfLines={1}>{item.subtitle}</ScaledText>
                )}
            </View>
        </View>
    );
};

type GuideCategory = 'featured' | 'capsules' | 'outfits';

// ── Main Component ──────────────────────────────────────

type SegmentType = 'guide' | 'shop';

const InspoScreen = () => {
    const { t } = useTranslation();
    const navigation = useAppNavigation();

    const {
        items: syncedShopItems,
        loading: shopCatalogLoading,
        loadingMore: shopCatalogLoadingMore,
        error: shopCatalogError,
        hasMore: shopCatalogHasMore,
        loadMore: loadMoreShopCatalog,
        refresh: refreshShopCatalog,
    } = useShopCatalog({ source: 'all' });

    const {
        items: featuredCapsules,
        loading: featuredCapsulesLoading,
        refresh: refreshFeaturedCapsules,
    } = useFeaturedCapsules();

    const { items: guideOutfits } = useShopCatalog({
        category: 'outfits',
        source: 'all',
    });

    const [guideHero, setGuideHero] = useState<any>(null);

    useEffect(() => {
        let mounted = true;
        const fetchHero = async () => {
            try {
                // Prefer the idle-prewarmed row so first visit doesn't pay a
                // cold query mid-tab-switch (see tabPrewarmService).
                const { getPrewarmedGuideHero } = await import('../src/services/tabPrewarmService');
                const prewarmed = getPrewarmedGuideHero();
                if (mounted && prewarmed) {
                    setGuideHero(prewarmed);
                    return;
                }
                const { data } = await supabase.from('guide_page').select('*').eq('is_active', true).single();
                if (mounted && data) setGuideHero(data);
            } catch (_) {
            }
        };
        fetchHero();
        return () => { mounted = false; };
    }, []);

    const displayGuides = useMemo(() => {
        const results: any[] = [];

        // 1. Add static/dynamic Hero from guide_page — admin-set cta_url, if
        // any, is where "shop this look" should actually go.
        if (guideHero) {
            results.push({
                id: 'guide_hero',
                title: guideHero.title,
                subtitle: guideHero.subtitle,
                image: guideHero.hero_image_url,
                category: 'featured' as GuideCategory,
                linkUrl: guideHero.cta_url || undefined,
            });
        }

        // 2. Add Featured Capsules as big cards in the guide tab
        if (featuredCapsules && featuredCapsules.length > 0) {
            featuredCapsules.forEach(c => {
                results.push({
                    id: c.id,
                    title: c.title,
                    subtitle: c.subtitle || '',
                    image: c.imageUrl,
                    category: 'capsules' as GuideCategory,
                    linkUrl: c.linkUrl,
                });
            });
        }

        // 3. Add outfits if we have them — each is a real shop catalog item,
        // so tapping it should go straight to buying that item, not a link.
        if (guideOutfits && guideOutfits.length > 0) {
            guideOutfits.forEach((item) => {
                results.push({
                    id: item.id,
                    title: item.name,
                    subtitle: item.description || item.brand,
                    image: item.imageUrl,
                    category: 'outfits' as GuideCategory,
                    shopItem: item,
                });
            });
        }

        return results;
    }, [guideHero, featuredCapsules, guideOutfits]);

    const savedInspo = useSavedShopStore((s) => s.items);
    const hydrateSavedShop = useSavedShopStore((s) => s.hydrate);
    const [segment, setSegment] = useState<SegmentType>('guide');
    const [searchQuery, setSearchQuery] = useState('');
    const [showSavedOnly, setShowSavedOnly] = useState(false);
    const [saveToast, setSaveToast] = useState<string | null>(null);
    const saveToastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    // Deep link from an outfit piece ("In this look" → shop): the tapped
    // product is lifted to the top of the Shop grid and ringed until the
    // user searches or toggles Saved.
    const inspoRoute = useRoute<RouteProp<TabParamList, 'Inspo'>>();
    const [focusedShopItemId, setFocusedShopItemId] = useState<string | null>(null);

    const flashSaveToast = useCallback((message: string) => {
        setSaveToast(message);
        if (saveToastTimer.current) clearTimeout(saveToastTimer.current);
        saveToastTimer.current = setTimeout(() => setSaveToast(null), 2600);
    }, []);

    useEffect(() => () => {
        if (saveToastTimer.current) clearTimeout(saveToastTimer.current);
    }, []);

    const clearShopFocusParams = useCallback(() => {
        (navigation as unknown as { setParams?: (p: { focusShopItemId?: string; focusShopItemName?: string }) => void }).setParams?.({
            focusShopItemId: undefined,
            focusShopItemName: undefined,
        });
    }, [navigation]);

    // Consume one-shot deep links from outfit pieces. The params are cleared
    // after applying so plain tab switches never yank the grid around, and
    // tapping the same piece twice still works (undefined → id transition).
    // If the id isn't in the loaded catalog (later page / retired item) we
    // fall back to a name search so the tap still lands on shopping.
    const focusShopItemId = inspoRoute.params?.focusShopItemId;
    const focusShopItemName = inspoRoute.params?.focusShopItemName;
    useFocusEffect(
        useCallback(() => {
            if (focusShopItemId) {
                setSegment('shop');
                setShowSavedOnly(false);
                // Mirror the grid pool (live catalog + appended classics);
                // INSPO-fallback items are covered by the name search below.
                const found =
                    syncedShopItems.some((i) => i.id === focusShopItemId) ||
                    CLASSIC_MENS_ITEMS.some((i) => i.id === focusShopItemId);
                if (found || !focusShopItemName) {
                    setSearchQuery('');
                    setFocusedShopItemId(focusShopItemId);
                } else {
                    setFocusedShopItemId(null);
                    setSearchQuery(focusShopItemName);
                }
                clearShopFocusParams();
            }
        }, [focusShopItemId, focusShopItemName, syncedShopItems, clearShopFocusParams]),
    );

    const lastFetchRef = useRef<number>(Date.now());
    const INSPO_REFRESH_TTL_MS = 5 * 60 * 1000; // 5 minutes

    useFocusEffect(
        useCallback(() => {
            // Shared shop saves (also shown in Profile → Saved Looks).
            hydrateSavedShop();

            const now = Date.now();
            if (now - lastFetchRef.current > INSPO_REFRESH_TTL_MS) {
                lastFetchRef.current = now;
                refreshFeaturedCapsules();
                refreshShopCatalog();
            }
        }, [refreshFeaturedCapsules, refreshShopCatalog, hydrateSavedShop])
    );

    const saveInspo = useCallback(async (item: ShopCatalogItem) => {
        // toggleSave persists + syncs Profile saved outfits via the shared store.
        const revision = authEvents.getRevision();
        try {
            const nowSaved = await useSavedShopStore.getState().toggleSave(item);
            if (revision !== authEvents.getRevision() || !authEvents.getUserId()) return;
            Haptics.notificationAsync(
                nowSaved
                    ? Haptics.NotificationFeedbackType.Success
                    : Haptics.NotificationFeedbackType.Warning,
            ).catch(() => {});
            flashSaveToast(t(nowSaved ? 'inspo.savedToast' : 'inspo.unsavedToast'));
            if (nowSaved) setShowSavedOnly(false);
        } catch {
            if (revision === authEvents.getRevision()) flashSaveToast(t('inspo.savedItemsUnavailable'));
        }
    }, [flashSaveToast, t]);

    const isShopEmpty = syncedShopItems.length === 0;
    const isInitialShopLoad = isShopEmpty && shopCatalogLoading;
    const showingFallbackCatalog = isShopEmpty && !shopCatalogLoading;

    // Deferred so typing stays responsive: the TextInput keeps the immediate
    // value while the 100+ item filter + full grid rebuild runs at a lower
    // priority. Previously every keystroke synchronously rebuilt shopItems →
    // shopRows → the whole VirtualizedList, blocking the JS thread (~2s with
    // images/animations in flight) and dropping keystrokes.
    const deferredSearchQuery = useDeferredValue(searchQuery);

    const shopItems = useMemo(() => {
        const baseItems = showingFallbackCatalog ? INSPO_MENS_SHOP_ITEMS : syncedShopItems;

        // Always append curated men's classics so they surface alongside the
        // live Zara feed. Dedupe by id in case fallback already includes them.
        const seenIds = new Set(baseItems.map((item) => item.id));
        const classicsToAppend = CLASSIC_MENS_ITEMS.filter((item) => !seenIds.has(item.id));
        let sourceItems = [...baseItems, ...classicsToAppend];

        if (showSavedOnly) {
            const savedIds = new Set(savedInspo.map((s) => String(s.id)));
            sourceItems = sourceItems.filter((item) => savedIds.has(String(item.id)));
        }

        const query = deferredSearchQuery.trim().toLowerCase();
        if (!query) {
            // Deep-linked piece from an outfit: lift it to the top so the
            // user lands exactly on what they tapped.
            if (!showSavedOnly && focusedShopItemId) {
                const idx = sourceItems.findIndex((item) => item.id === focusedShopItemId);
                if (idx > 0) {
                    sourceItems = [sourceItems[idx], ...sourceItems.slice(0, idx), ...sourceItems.slice(idx + 1)];
                }
            }
            return sourceItems;
        }

        return sourceItems.filter((item) => {
            const haystack = `${item.brand} ${item.name} ${item.description ?? ''}`.toLowerCase();
            return haystack.includes(query);
        });
    }, [deferredSearchQuery, showingFallbackCatalog, syncedShopItems, showSavedOnly, savedInspo, focusedShopItemId]);

    // ── FlatList helpers for the shop grid ────────────────────────────────────
    const savedInspoSet = useMemo(
        () => new Set(savedInspo.map((s) => String(s.id))),
        [savedInspo],
    );

    // Break the grid's rhythm on purpose: every 5th slot is a full-width
    // spotlight before the next pair of small cards, so the shop reads like
    // a curated editorial spread instead of a flat uniform tile wall.
    // Spotlight covers are shoes or outerwear ONLY — never pants/bottoms.
    // Every item still renders exactly once; pants displaced from a cover
    // slot simply flow into the next small pair row.
    // startIndex = the card's position in display order — ProductCard only
    // animates `index < 8`, so this must be the real position, not a hardcoded
    // 0/1 (which previously made EVERY card animate on every data change).
    type ShopRow =
        | { type: 'spotlight'; item: ShopCatalogItem; plate: number; startIndex: number }
        | { type: 'pair'; items: [ShopCatalogItem, ShopCatalogItem | null]; startIndex: number };

    const shopRows = useMemo<ShopRow[]>(() => {
        if (shopItems.length === 0) return [];
        const rows: ShopRow[] = [];
        // Track by array index (not id) so every element renders exactly
        // once even if two rows ever share an id.
        const used = new Set<number>();
        const keyOf = (it: ShopCatalogItem) => String((it as unknown as { id?: unknown })?.id);
        let displayIndex = 0;
        let plate = 1;

        // Deep-linked piece from an outfit stays pinned as the first cover
        // even if it's pants — the user explicitly tapped it.
        const focusedIdx = focusedShopItemId
            ? shopItems.findIndex((it) => keyOf(it) === String(focusedShopItemId))
            : -1;

        const pickSpotlightIdx = (): number => {
            let fallbackNonPants = -1;
            for (let k = 0; k < shopItems.length; k++) {
                if (used.has(k)) continue;
                if (isSpotlightEligible(shopItems[k])) return k;
                if (fallbackNonPants < 0 && !isPantsItem(shopItems[k])) fallbackNonPants = k;
            }
            if (fallbackNonPants >= 0) return fallbackNonPants;
            for (let k = 0; k < shopItems.length; k++) {
                if (!used.has(k)) return k;
            }
            return -1;
        };

        const nextUnusedIdx = (): number => {
            for (let k = 0; k < shopItems.length; k++) {
                if (!used.has(k)) return k;
            }
            return -1;
        };

        if (focusedIdx >= 0) {
            rows.push({ type: 'spotlight', item: shopItems[focusedIdx], plate: plate++, startIndex: displayIndex++ });
            used.add(focusedIdx);
        }

        while (used.size < shopItems.length) {
            const coverIdx = pickSpotlightIdx();
            if (coverIdx < 0) break;
            rows.push({ type: 'spotlight', item: shopItems[coverIdx], plate: plate++, startIndex: displayIndex++ });
            used.add(coverIdx);

            for (let pairCount = 0; pairCount < 2; pairCount++) {
                const leftIdx = nextUnusedIdx();
                if (leftIdx < 0) break;
                used.add(leftIdx);
                const rightIdx = nextUnusedIdx();
                const right = rightIdx >= 0 ? shopItems[rightIdx] : null;
                if (rightIdx >= 0) used.add(rightIdx);
                rows.push({ type: 'pair', items: [shopItems[leftIdx], right], startIndex: displayIndex });
                displayIndex += right ? 2 : 1;
            }
        }
        return rows;
    }, [shopItems, focusedShopItemId]);

    const renderShopRow = useCallback(
        ({ item: row }: { item: ShopRow }) => {
            if (row.type === 'spotlight') {
                return (
                    <ProductCard
                        item={row.item}
                        isSaved={savedInspoSet.has(String(row.item.id))}
                        onSave={saveInspo}
                        index={row.startIndex}
                        t={t}
                        spotlight
                        plate={row.plate}
                        highlighted={focusedShopItemId === String(row.item.id) || focusedShopItemId === (row.item.id as unknown as string)}
                    />
                );
            }

            const [left, right] = row.items;
            return (
                <ShopPairRow
                    left={left}
                    right={right}
                    leftIndex={row.startIndex}
                    t={t}
                    isLeftSaved={savedInspoSet.has(String(left.id))}
                    isRightSaved={right ? savedInspoSet.has(String(right.id)) : false}
                    onSave={saveInspo}
                    focusedShopItemId={focusedShopItemId}
                />
            );
        },
        [savedInspoSet, saveInspo, t, focusedShopItemId],
    );

    const shopListFooter = useCallback(() => {
        if (showingFallbackCatalog || !shopCatalogHasMore) return null;
        return (
            <TouchableOpacity
                style={styles.loadMoreButton}
                onPress={loadMoreShopCatalog}
                disabled={shopCatalogLoadingMore}
                accessibilityRole="button"
                accessibilityLabel={t('inspo.loadMoreShopProducts')}
                activeOpacity={0.85}
            >
                {shopCatalogLoadingMore ? (
                    <ActivityIndicator size="small" color="#0A1931" />
                ) : (
                    <>
                        <ScaledText style={styles.loadMoreButtonText}>{t('inspo.loadMoreProducts')}</ScaledText>
                        <Ionicons name="add" size={16} color="#0A1931" />
                    </>
                )}
            </TouchableOpacity>
        );
    }, [showingFallbackCatalog, shopCatalogHasMore, shopCatalogLoadingMore, loadMoreShopCatalog, t, colors.text.primary]);

    const shopListKeyExtractor = useCallback(
        (row: ShopRow) => (row.type === 'spotlight' ? `spotlight-${row.item.id}` : `pair-${row.items[0].id}`),
        [],
    );

    const renderShopHeader = useCallback(() => (
        <>
            {/* Personal Stylist Button */}
            <Animated.View entering={FadeInDown.delay(80).duration(400)}>
                <TouchableOpacity
                    style={styles.personalStylistButton}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        navigation.navigate('AIOutfit', { source: 'shop' });
                    }}
                    accessibilityLabel={t('inspo.personalStylist')}
                    accessibilityRole="button"
                    activeOpacity={0.85}
                >
                    <View style={styles.personalStylistFlat}>
                        <Ionicons name="sparkles-outline" size={18} color="#FFF" />
                        <ScaledText style={styles.personalStylistText}>{t('inspo.personalStylist')}</ScaledText>
                        <View style={styles.personalStylistArrowWrap}>
                            <Ionicons name="arrow-forward" size={14} color="#FFF" />
                        </View>
                    </View>
                </TouchableOpacity>
            </Animated.View>

            {/* Featured Capsules */}
            {(featuredCapsulesLoading || featuredCapsules.length > 0) && (
                <View style={styles.section}>
                    <ScaledText style={styles.sectionTitle} accessibilityRole="header">{t('inspo.featuredCapsules')}</ScaledText>
                    {featuredCapsulesLoading && featuredCapsules.length === 0 ? (
                        <View style={styles.capsulesLoadingRow}>
                            <ActivityIndicator size="small" color={colors.text.primary} />
                        </View>
                    ) : (
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.capsulesScroll}
                        >
                            {featuredCapsules.map((item, index) => (
                                <FeaturedCapsuleCard key={item.id} item={item} index={index} t={t} />
                            ))}
                        </ScrollView>
                    )}
                </View>
            )}

            <Animated.View entering={FadeInDown.delay(140).duration(400)}>
                <View style={styles.searchContainer}>
                    <Ionicons name="search" size={20} color={colors.text.tertiary} style={styles.searchIcon} />
                    <TextInput
                        placeholder={t('inspo.searchPlaceholder')}
                        placeholderTextColor={colors.text.tertiary}
                        value={searchQuery}
                        onChangeText={(value) => {
                            setSearchQuery(value);
                            if (value.trim().length > 0) setFocusedShopItemId(null);
                        }}
                        style={styles.searchInput}
                        returnKeyType="search"
                        accessibilityLabel={t('inspo.searchForClothingItems')}
                        maxLength={200}
                    />
                </View>
            </Animated.View>

            {(shopCatalogError || showingFallbackCatalog) && (
                <Animated.View entering={FadeInDown.delay(160).duration(300)}>
                    <View style={styles.catalogStatusBanner}>
                        <ScaledText style={styles.catalogStatusText}>
                            {showingFallbackCatalog
                                ? t('inspo.catalogEmpty')
                                : t('inspo.catalogRefreshFailed')}
                        </ScaledText>
                        <TouchableOpacity onPress={refreshShopCatalog} accessibilityRole="button">
                            <ScaledText style={styles.catalogStatusAction}>{t('common.retry')}</ScaledText>
                        </TouchableOpacity>
                    </View>
                </Animated.View>
            )}

            <View style={styles.shopHeaderRow}>
                <ScaledText style={[styles.sectionTitle, styles.shopHeaderTitle]} accessibilityRole="header">
                    {t('inspo.shop')}
                </ScaledText>
                <View style={styles.shopHeaderRight}>
                    <TouchableOpacity
                        style={[styles.savedFilterChip, showSavedOnly && styles.savedFilterChipActive]}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            setShowSavedOnly((v) => !v);
                        }}
                        accessibilityRole="tab"
                        accessibilityState={{ selected: showSavedOnly }}
                        accessibilityLabel={`${t('inspo.saved')} (${savedInspo.length})`}
                        activeOpacity={0.85}
                    >
                        <Ionicons
                            name={showSavedOnly ? 'heart' : 'heart-outline'}
                            size={14}
                            color={showSavedOnly ? '#FFFFFF' : '#0A1931'}
                        />
                        <ScaledText style={[styles.savedFilterText, showSavedOnly && styles.savedFilterTextActive]}>
                            {`${t('inspo.saved')} (${savedInspo.length})`}
                        </ScaledText>
                    </TouchableOpacity>
                    <ScaledText style={styles.shopCountText}>
                        {t('inspo.itemsCount', { count: shopItems.length })}
                    </ScaledText>
                </View>
            </View>
        </>
    ), [searchQuery, setSearchQuery, shopCatalogError, showingFallbackCatalog, refreshShopCatalog, navigation, featuredCapsulesLoading, featuredCapsules, shopItems.length, showSavedOnly, savedInspo.length, t, colors.text.primary, colors.text.tertiary]);

    // Memoized elements — passing `renderShopHeader()` / `shopListFooter()`
    // inline creates a brand-new header/footer tree on EVERY parent render
    // (heart toggle, toast, focus refresh), forcing VirtualizedList to
    // reconcile its entire content (header + all visible rows + images) in
    // one JS-thread block (~2s, the "large list slow to update" warning).
    // These keep identity unless the underlying callback changes.
    const shopHeaderElement = useMemo(() => renderShopHeader(), [renderShopHeader]);
    const shopFooterElement = useMemo(() => shopListFooter(), [shopListFooter]);

    // Goal of the Guide tab: see a look, shop it — immediately, not by
    // dropping the user into the unrelated full catalog. A capsule/hero with
    // an admin-set link goes straight there; an outfit that's a real shop
    // item goes straight to buying it; only a look with neither falls back
    // to the Shop tab, since there's nowhere more specific to send it.
    const handleGuideCardPress = useCallback((item: any) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

        if (item?.linkUrl) {
            Linking.canOpenURL(item.linkUrl)
                .then((canOpen) => {
                    if (canOpen) Linking.openURL(item.linkUrl);
                })
                .catch(() => {});
            return;
        }

        if (item?.shopItem) {
            handleBuyPress(item.shopItem);
            return;
        }

        setSegment('shop');
    }, []);

    type GuideRow =
        | { type: 'hero'; item: any }
        | { type: 'pair'; items: [any, any | null]; plates: [number, number | null] };

    // The first "Featured" item spans full width as the one editorial cover
    // moment; everything else packs into a 2-column grid of numbered plates,
    // like the index pages of a printed lookbook.
    const guideRows = useMemo<GuideRow[]>(() => {
        if (displayGuides.length === 0) return [];
        const rows: GuideRow[] = [];
        let startIdx = 0;
        if (displayGuides[0]?.category === 'featured') {
            rows.push({ type: 'hero', item: displayGuides[0] });
            startIdx = 1;
        }
        let plate = 1;
        for (let i = startIdx; i < displayGuides.length; i += 2) {
            const left = displayGuides[i];
            const right = displayGuides[i + 1] ?? null;
            const leftPlate = plate++;
            const rightPlate = right ? plate++ : null;
            rows.push({ type: 'pair', items: [left, right], plates: [leftPlate, rightPlate] });
        }
        return rows;
    }, [displayGuides]);

    const guideKeyExtractor = useCallback((row: GuideRow) => (
        row.type === 'hero' ? `hero-${row.item.id}` : `pair-${row.items[0].id}`
    ), []);

    const renderGuideRow = useCallback(({ item: row, index }: { item: GuideRow; index: number }) => {
        if (row.type === 'hero') {
            const { item } = row;
            return (
                <Animated.View entering={FadeInDown.delay(100).duration(400)}>
                    <TouchableOpacity
                        style={styles.guideCardContainer}
                        activeOpacity={0.92}
                        onPress={() => handleGuideCardPress(item)}
                        accessibilityRole="button"
                        accessibilityLabel={`${item.title}. ${t('inspo.shopThisLook')}`}
                    >
                        <View style={styles.guideCard}>
                            <CachedImage
                                uri={typeof item.image === 'string' ? item.image : ''}
                                style={styles.guideImage}
                                contentFit="cover"
                                fadeIn={false}
                            />
                            <View style={styles.guideHeroArrow} pointerEvents="none">
                                <Ionicons name="arrow-forward" size={16} color="#FFFFFF" />
                            </View>
                        </View>
                        <View style={styles.guideHeroCaption}>
                            <ScaledText style={styles.guideEyebrow}>{t('inspo.featuredLook')}</ScaledText>
                            <ScaledText style={styles.guideHeroTitle}>{item.title}</ScaledText>
                            {!!item.subtitle && (
                                <ScaledText style={styles.guideHeroSubtitle} numberOfLines={2}>{item.subtitle}</ScaledText>
                            )}
                        </View>
                    </TouchableOpacity>
                </Animated.View>
            );
        }

        const [left, right] = row.items;
        const [leftPlate, rightPlate] = row.plates;
        const delay = Math.min(120 + (index % 6) * 60, 420);
        return (
            <Animated.View entering={FadeInDown.delay(delay).duration(380)}>
                <View style={styles.guideGridRow}>
                    <GuideGridCard item={left} plate={leftPlate} onPress={handleGuideCardPress} t={t} />
                    {right ? (
                        <GuideGridCard item={right} plate={rightPlate as number} onPress={handleGuideCardPress} t={t} />
                    ) : (
                        <View style={styles.guideGridCardWrap} />
                    )}
                </View>
            </Animated.View>
        );
    }, [handleGuideCardPress, t]);

    const renderGuideFooter = useCallback(() => (
        <>
            <View style={{ height: 120 }} />
        </>
    ), []);

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
                pointerEvents="none"
            />
            <View pointerEvents="none" style={styles.backgroundOrbTop} />
            <View pointerEvents="none" style={styles.backgroundOrbBottom} />
            <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
            <SafeAreaView style={styles.safeArea} edges={['top']}>
                {/* Header */}
                <View style={styles.header}>
                    <View style={[StyleSheet.absoluteFill, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                        <ScaledText style={styles.headerTitle} accessibilityRole="header">{t('inspo.title')}</ScaledText>
                    </View>
                </View>

                {/* Segmented Control */}
                <View style={styles.segmentContainer}>
                    <View style={styles.segmentBackground}>
                        {(['guide', 'shop'] as const).map((seg) => (
                            <TouchableOpacity
                                key={seg}
                                style={[styles.segmentButton, segment === seg && styles.segmentButtonActive]}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                    setSegment(seg);
                                }}
                                activeOpacity={1}
                                accessibilityLabel={seg === 'guide' ? t('inspo.guide') : t('inspo.shop')}
                                accessibilityRole="tab"
                                accessibilityState={{ selected: segment === seg }}
                            >
                                <ScaledText style={[styles.segmentText, segment === seg && styles.segmentTextActive]}>
                                    {seg === 'guide' ? t('inspo.guide') : `${t('inspo.shop')}${savedInspo.length > 0 ? ` (${savedInspo.length})` : ''}`}
                                </ScaledText>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>

                {segment === 'guide' ? (
                    <>
                        <ScaledText style={styles.guideTagline}>{t('inspo.guideTagline')}</ScaledText>
                        {displayGuides.length === 0 && featuredCapsulesLoading ? (
                            <View style={styles.guideLoadingContainer}>
                                <ActivityIndicator size="large" color={colors.text.primary} />
                            </View>
                        ) : guideRows.length === 0 ? (
                            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
                                <View style={styles.emptyState}>
                                    <ScaledText style={styles.emptyStateText}>{t('inspo.noGuideMatches')}</ScaledText>
                                </View>
                                {renderGuideFooter()}
                            </ScrollView>
                        ) : (
                            <FlatList
                                data={guideRows}
                                keyExtractor={guideKeyExtractor}
                                renderItem={renderGuideRow}
                                ListFooterComponent={renderGuideFooter()}
                                contentContainerStyle={styles.scrollContent}
                                showsVerticalScrollIndicator={false}
                                initialNumToRender={4}
                                maxToRenderPerBatch={4}
                                windowSize={3}
                                removeClippedSubviews={Platform.OS === 'android'}
                            />
                        )}
                    </>
                ) : (
                    <>
                        {isInitialShopLoad ? (
                            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
                                {shopHeaderElement}
                                <View style={styles.loadingRow}>
                                    {[0, 1, 2, 3].map((idx) => (
                                        <View key={`shop-skeleton-${idx}`} style={styles.productCardWrap}>
                                            <ShimmerProductCard />
                                        </View>
                                    ))}
                                </View>
                            </ScrollView>
                        ) : shopItems.length === 0 ? (
                            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
                                {shopHeaderElement}
                                <View style={styles.emptyState}>
                                    <ScaledText style={styles.emptyStateText}>
                                        {showSavedOnly ? t('inspo.savedEmpty') : t('inspo.noMenswearMatches')}
                                    </ScaledText>
                                    {showSavedOnly && (
                                        <ScaledText style={styles.emptyStateHint}>
                                            {t('inspo.savedEmptyHint')}
                                        </ScaledText>
                                    )}
                                </View>
                            </ScrollView>
                        ) : (
                            <FlatList
                                data={shopRows}
                                keyExtractor={shopListKeyExtractor}
                                renderItem={renderShopRow}
                                ListHeaderComponent={shopHeaderElement}
                                ListFooterComponent={shopFooterElement}
                                contentContainerStyle={[styles.scrollContent, styles.productsGrid]}
                                showsVerticalScrollIndicator={false}
                                keyboardShouldPersistTaps="handled"
                                initialNumToRender={4}
                                maxToRenderPerBatch={4}
                                windowSize={5}
                                updateCellsBatchingPeriod={50}
                                removeClippedSubviews={Platform.OS === 'android'}
                            />
                        )}
                    </>
                )}
            </SafeAreaView>
            {saveToast && (
                <View style={styles.saveToast} pointerEvents="box-none" accessibilityLiveRegion="polite">
                    <Ionicons name="heart" size={16} color="#FFFFFF" />
                    <ScaledText style={styles.saveToastText} numberOfLines={2}>
                        {saveToast}
                    </ScaledText>
                    {!showSavedOnly && savedInspo.length > 0 && saveToast === t('inspo.savedToast') && (
                        <TouchableOpacity
                            onPress={() => {
                                setShowSavedOnly(true);
                                setSaveToast(null);
                            }}
                            accessibilityRole="button"
                            accessibilityLabel={t('inspo.viewSaved')}
                        >
                            <ScaledText style={styles.saveToastAction}>{t('inspo.viewSaved')}</ScaledText>
                        </TouchableOpacity>
                    )}
                </View>
            )}
        </View>
    );
};

// ── Styles ──────────────────────────────────────

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
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
    safeArea: {
        flex: 1,
    },

    // Header (consistent with all tabs)
    header: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: spacing.screenPadding,
        paddingVertical: 14,
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        fontWeight: '700',
        color: colors.text.primary,
        letterSpacing: 0.3,
    },

    // Segmented Control (consistent with MyCloset)
    segmentContainer: {
        alignItems: 'center',
        paddingTop: 15,
        paddingBottom: 20,
    },
    segmentBackground: {
        flexDirection: 'row',
        backgroundColor: 'rgba(255,255,255,0.85)',
        borderRadius: 14,
        padding: 4,
        width: 280,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.07,
        shadowRadius: 14,
        elevation: 4,
    },
    segmentButton: {
        flex: 1,
        paddingVertical: 9,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 10,
    },
    segmentButtonActive: {
        backgroundColor: '#0A1931',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.2,
        shadowRadius: 8,
        elevation: 3,
    },
    segmentText: {
        fontSize: 14,
        fontWeight: '500',
        color: colors.text.tertiary,
        letterSpacing: 0.2,
    },
    segmentTextActive: {
        color: '#FFFFFF',
        fontWeight: '600',
    },

    // Guide tab tagline — a quiet italic masthead line, the one caption
    // explaining the whole page's job.
    guideTagline: {
        fontSize: 12.5,
        fontWeight: '500',
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        fontStyle: 'italic',
        color: colors.text.tertiary,
        textAlign: 'center',
        letterSpacing: 0.3,
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.lg,
    },

    // Shop section header — count reads as a visual chip, not a sentence.
    shopHeaderRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.md,
    },
    shopHeaderTitle: {
        paddingHorizontal: 0,
        marginBottom: 0,
    },
    shopHeaderRight: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    savedFilterChip: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderWidth: 1,
        borderColor: 'rgba(10,25,49,0.12)',
    },
    savedFilterChipActive: {
        backgroundColor: '#0A1931',
        borderColor: '#0A1931',
    },
    savedFilterText: {
        fontSize: 13,
        fontWeight: '600',
        color: '#0A1931',
    },
    savedFilterTextActive: {
        color: '#FFFFFF',
    },
    shopCountText: {
        fontSize: 13,
        fontWeight: '500',
        color: colors.text.tertiary,
    },

    // Scroll — 140 clears the 68pt floating tab bar + 34pt iOS home inset.
    scrollContent: {
        paddingTop: spacing.sm,
        paddingBottom: 140,
    },

    // Search
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.88)',
        marginHorizontal: spacing.screenPadding,
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.sm + 4,
        borderRadius: 14,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        marginBottom: spacing.lg,
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 14,
        elevation: 3,
    },
    searchIcon: {
        marginRight: spacing.sm,
    },
    searchInput: {
        flex: 1,
        ...typography.scale.bodyMedium,
        color: colors.text.primary,
        paddingVertical: 0,
    },

    // Sections
    section: {
        marginBottom: spacing.xl,
    },
    sectionTitle: {
        ...typography.scale.titleMedium,
        fontSize: 20,
        fontWeight: '700',
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        color: colors.text.primary,
        marginBottom: spacing.md,
        paddingHorizontal: spacing.screenPadding,
    },

    // Guide Hero — the one full-bleed cover moment, editorial proportions
    // (taller than square, like a lookbook plate rather than an app tile).
    guideCardContainer: {
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.xl,
    },
    guideCard: {
        width: '100%',
        aspectRatio: 4 / 5,
        borderRadius: 20,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        position: 'relative',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.72)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.08,
        shadowRadius: 18,
        elevation: 5,
    },
    guideImage: {
        width: '100%',
        height: '100%',
    },
    // The one quiet on-image affordance in the whole tab — a plain circle,
    // not a labeled pill, so the photo stays uninterrupted.
    guideHeroArrow: {
        position: 'absolute',
        right: spacing.md,
        bottom: spacing.md,
        width: 38,
        height: 38,
        borderRadius: 19,
        backgroundColor: 'rgba(10,25,49,0.55)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    guideHeroCaption: {
        paddingTop: spacing.md,
        paddingHorizontal: 2,
    },
    guideEyebrow: {
        fontSize: 11,
        fontWeight: '600',
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        fontStyle: 'italic',
        color: colors.text.tertiary,
        textTransform: 'uppercase',
        letterSpacing: 1.4,
        marginBottom: 6,
    },
    guideHeroTitle: {
        fontSize: 24,
        fontWeight: '700',
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        color: colors.text.primary,
        letterSpacing: -0.3,
        marginBottom: 4,
    },
    guideHeroSubtitle: {
        fontSize: 14,
        fontWeight: '400',
        color: colors.text.secondary,
        lineHeight: 20,
    },

    // Guide Grid — numbered lookbook plates. No overlay on the photo; every
    // word lives in the flat caption below it, same language as the Shop
    // tab's product tiles.
    guideGridRow: {
        flexDirection: 'row',
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.lg,
    },
    guideGridCardWrap: {
        flex: 1,
        marginHorizontal: spacing.xs,
    },
    guideGridCard: {
        width: '100%',
        aspectRatio: 3 / 4,
        borderRadius: 14,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.72)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.07,
        shadowRadius: 12,
        elevation: 3,
    },
    guideGridCaption: {
        paddingTop: spacing.sm,
        paddingHorizontal: 2,
    },
    guideGridPlate: {
        fontSize: 10.5,
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        fontStyle: 'italic',
        color: colors.text.tertiary,
        letterSpacing: 0.5,
        marginBottom: 3,
    },
    guideGridCardTitle: {
        fontSize: 13,
        fontWeight: '700',
        color: colors.text.primary,
        letterSpacing: -0.1,
    },
    guideGridCardSubtitle: {
        fontSize: 11,
        fontWeight: '500',
        color: colors.text.tertiary,
        marginTop: 2,
    },

    // Capsule Cards
    capsulesScroll: {
        paddingHorizontal: spacing.screenPadding,
        gap: spacing.md,
    },
    capsulesLoadingRow: {
        height: CAPSULE_CARD_HEIGHT,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: spacing.screenPadding,
    },
    capsuleCard: {
        width: CAPSULE_CARD_WIDTH,
        height: CAPSULE_CARD_HEIGHT,
        borderRadius: 16,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.72)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.07,
        shadowRadius: 16,
        elevation: 4,
    },
    capsuleImage: {
        width: '100%',
        height: '100%',
    },
    shopRow: {
        flexDirection: 'row',
        marginBottom: spacing.lg,
    },
    capsuleGradient: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '45%',
        justifyContent: 'flex-end',
        padding: spacing.sm,
    },
    capsuleTitle: {
        ...typography.scale.labelMedium,
        fontSize: 14,
        fontWeight: '600',
        color: '#FFF',
        textShadowColor: 'rgba(0,0,0,0.4)',
        textShadowOffset: { width: 0, height: 1 },
        textShadowRadius: 2,
    },

    // Product Grid (2-col via FlatList rows)
    productsGrid: {
        paddingHorizontal: spacing.screenPadding,
    },
    loadingRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        paddingHorizontal: spacing.screenPadding,
    },
    skeletonCard: {
        flex: 1,
        marginHorizontal: spacing.xs,
        marginBottom: spacing.lg,
    },
    skeletonImage: {
        width: '100%',
        aspectRatio: 3 / 4,
        borderRadius: 8,
        backgroundColor: '#F4F5F7',
    },
    skeletonBrandLine: {
        marginTop: spacing.sm,
        width: '50%',
        height: 10,
        borderRadius: 3,
        backgroundColor: 'rgba(10,25,49,0.1)',
        alignSelf: 'flex-start',
    },
    skeletonPriceLine: {
        marginTop: 8,
        width: 50,
        height: 16,
        borderRadius: 3,
        backgroundColor: 'rgba(10,25,49,0.16)',
        alignSelf: 'flex-start',
    },
    productCardWrap: {
        flex: 1,
        marginHorizontal: spacing.xs,
    },
    productCardWrapLeft: {
        // No longer needed with marginHorizontal, but keep for skeleton backwards compat
        marginRight: spacing.sm,
    },
    // Flat tile — no card background, no shadow, no border. The photo and
    // the caption typography below it carry the whole design.
    // No explicit width — a plain View already stretches to fill its
    // parent's cross-axis. Setting width:'100%' here AND marginHorizontal
    // on the spotlight variant below made the box 100% + 2*margin wide,
    // overflowing past the right edge while the left margin still pushed
    // the box inward — a left gap with no matching gap on the right.
    productCard: {
    },
    // Editorial lead card — full width, breaks the grid every 5 items.
    productCardSpotlight: {
        marginHorizontal: spacing.screenPadding,
        marginBottom: spacing.lg,
    },
    // A soft neutral fill, not pure white — product photos carry their own
    // near-white studio backdrop, and matching it against true #FFF makes
    // that backdrop's edge show up as a visible seam. Light gray hides it.
    productImageBox: {
        width: '100%',
        aspectRatio: 3 / 4,
        backgroundColor: '#F6F6F4',
        borderRadius: 12,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.07,
        shadowRadius: 12,
        elevation: 3,
    },
    productImageBoxSpotlight: {
        // Portrait, not landscape — garment photos are tall, and a wide box
        // forced heavy contain-fit letterboxing that made any asymmetry in
        // the source photo's own framing read as an off-center product.
        aspectRatio: 4 / 5,
        borderRadius: 16,
        shadowOpacity: 0.09,
        shadowRadius: 18,
        shadowOffset: { width: 0, height: 10 },
        elevation: 5,
    },
    // Deep-link highlight: navy ring marks the product jumped to from an outfit.
    productImageBoxHighlighted: {
        borderWidth: 2,
        borderColor: '#0A1931',
    },
    productImage: {
        width: '100%',
        height: '100%',
    },
    plateNumber: {
        position: 'absolute',
        left: spacing.sm,
        bottom: spacing.sm,
        fontSize: 13,
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        fontStyle: 'italic',
        color: 'rgba(10,25,49,0.38)',
        letterSpacing: 0.5,
    },
    saveButton: {
        position: 'absolute',
        top: spacing.sm,
        right: spacing.sm,
        zIndex: 10,
    },
    saveButtonCircle: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: '#FFFFFF',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(10,25,49,0.08)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.12,
        shadowRadius: 4,
        elevation: 2,
    },
    // Caption — brand and price as plain typography, no badges or chips.
    caption: {
        paddingTop: spacing.sm,
        paddingHorizontal: 2,
    },
    captionBrand: {
        fontSize: 10.5,
        fontWeight: '600',
        color: colors.text.tertiary,
        textTransform: 'uppercase',
        letterSpacing: 1,
        marginBottom: 4,
    },
    captionBrandLarge: {
        fontSize: 12,
        letterSpacing: 1.3,
        marginBottom: 8,
    },
    captionPrice: {
        fontSize: 15,
        fontWeight: '700',
        color: '#0A1931',
        letterSpacing: -0.2,
    },
    captionPriceLarge: {
        fontSize: 36,
        fontWeight: '700',
        fontFamily: Platform.select({ ios: 'Georgia', android: 'serif' }),
        letterSpacing: -0.5,
    },
    emptyState: {
        paddingHorizontal: spacing.screenPadding,
        marginHorizontal: spacing.screenPadding,
        paddingVertical: 20,
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.05,
        shadowRadius: 12,
        elevation: 2,
    },
    emptyStateText: {
        ...typography.scale.bodyMedium,
        color: colors.text.tertiary,
    },
    emptyStateHint: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        marginTop: 6,
    },
    saveToast: {
        position: 'absolute',
        left: spacing.screenPadding,
        right: spacing.screenPadding,
        bottom: 110,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        backgroundColor: '#0A1931',
        borderRadius: 16,
        paddingHorizontal: 16,
        paddingVertical: 14,
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.25,
        shadowRadius: 16,
        elevation: 8,
        zIndex: 50,
    },
    saveToastText: {
        flex: 1,
        color: '#FFFFFF',
        fontSize: 14,
        fontWeight: '500',
    },
    saveToastAction: {
        color: '#FFFFFF',
        fontSize: 14,
        fontWeight: '700',
        textDecorationLine: 'underline',
    },
    catalogStatusBanner: {
        marginHorizontal: spacing.screenPadding,
        marginBottom: spacing.lg,
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.sm + 2,
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.88)',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: spacing.md,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.05,
        shadowRadius: 12,
        elevation: 2,
    },
    catalogStatusText: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        flex: 1,
    },
    catalogStatusAction: {
        ...typography.scale.labelMedium,
        color: colors.text.primary,
        fontWeight: '700',
    },
    loadMoreButton: {
        marginTop: spacing.sm,
        marginHorizontal: spacing.screenPadding,
        paddingVertical: spacing.md,
        borderRadius: 14,
        backgroundColor: 'rgba(255,255,255,0.88)',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
        elevation: 3,
    },
    loadMoreButtonText: {
        ...typography.scale.bodyMedium,
        color: '#0A1931',
        fontWeight: '600',
    },

    // Header chat button — Liquid Glass
    headerChatButton: {
        position: 'absolute',
        right: spacing.screenPadding,
        top: 10,
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: 'rgba(255,255,255,0.72)',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.5)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 6,
        elevation: 4,
    },

    // Personal Stylist Button
    personalStylistButton: {
        marginHorizontal: spacing.screenPadding,
        marginBottom: spacing.xl,
        borderRadius: 16,
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.22,
        shadowRadius: 18,
        elevation: 6,
    },
    personalStylistFlat: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#0A1931',
        borderRadius: 16,
        overflow: 'hidden',
        paddingHorizontal: spacing.lg,
        paddingVertical: spacing.md + 2,
    },
    personalStylistText: {
        fontSize: 15,
        fontWeight: '600',
        color: '#FFF',
        flex: 1,
        marginLeft: spacing.sm + 2,
    },
    personalStylistArrowWrap: {
        width: 28,
        height: 28,
        borderRadius: 14,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.4)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    guideLoadingContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 100,
    },
});

export default InspoScreen;
