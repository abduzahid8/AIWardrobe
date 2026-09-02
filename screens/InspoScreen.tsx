/**
 * InspoScreen — Inspiration page
 * Minimalist Liquid Glass design with Guide + Shop tabs
 */

import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { View, TouchableOpacity, StyleSheet, Dimensions, ScrollView, FlatList, StatusBar, TextInput, ActivityIndicator, Linking, Platform, InteractionManager,  } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
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
import useWardrobeStore from '../store/wardrobeStore';
import { CachedImage } from '../components/ui/CachedImage';
import { generateVarietyOutfits, OCCASION_LABEL } from '../src/services/suggestionEngine';
import type { ScoredOutfit } from '../src/services/suggestionEngine';
import type { ShopCatalogItem } from '../features/try-on/types';
import { INSPO_MENS_SHOP_ITEMS, CLASSIC_MENS_ITEMS } from '../data/inspoMensShopItems';
import { useShopCatalog } from '../hooks/useShopCatalog';
import { useFeaturedCapsules, type FeaturedCapsule } from '../hooks/useFeaturedCapsules';
import { useTranslation } from 'react-i18next';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

// ── Data ──────────────────────────────────────

// Featured Capsules now come from Supabase (`featured_capsules` table) via
// `useFeaturedCapsules`. Rows are admin-editable from the Supabase dashboard.

const CAPSULE_CARD_WIDTH = 180;
const CAPSULE_CARD_HEIGHT = 250;
const PRODUCT_CARD_WIDTH = (SCREEN_WIDTH - spacing.screenPadding * 2 - spacing.sm) / 2;

// ── Sub-Components ──────────────────────────────────────

const FeaturedCapsuleCard = ({ item, index, t }: { item: FeaturedCapsule; index: number; t: (key: string) => string }) => (
    <Animated.View entering={FadeInDown.delay(100 + index * 80).duration(400)}>
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
);

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

// A flat, typographic tile: full-bleed photo, no chrome, price set as a
// graphic element rather than a labeled control. The spotlight variant is
// the same component at a different scale — the size shift alone marks it
// as featured, no ribbon or caption needed to say so.
const ProductCard = ({
    item,
    isSaved,
    onSave,
    index,
    t,
    spotlight = false,
    plate,
}: {
    item: ShopCatalogItem;
    isSaved: boolean;
    onSave: () => void;
    index: number;
    t: (key: string) => string;
    spotlight?: boolean;
    plate?: number;
}) => (
    // Cap the stagger so later pages (index 100+) don't wait many seconds
    // before their entering animation starts — otherwise pressing Load more
    // appears to do nothing because new cards are invisible until the delay elapses.
    <Animated.View entering={FadeInDown.delay(Math.min(150 + (index % 12) * 40, 600)).duration(320)}>
        <View style={[styles.productCard, spotlight && styles.productCardSpotlight]}>
            <View style={[styles.productImageBox, spotlight && styles.productImageBoxSpotlight]}>
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
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onSave();
                    }}
                    accessibilityLabel={isSaved ? t('inspo.removeFromSaved') : t('inspo.saveItem')}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                >
                    <View style={styles.saveButtonCircle}>
                        <Ionicons
                            name={isSaved ? 'heart' : 'heart-outline'}
                            size={spotlight ? 19 : 16}
                            color={isSaved ? '#FF3B5C' : '#0A1931'}
                        />
                    </View>
                </TouchableOpacity>
            </View>

            <View style={styles.caption}>
                <ScaledText style={[styles.captionBrand, spotlight && styles.captionBrandLarge]} numberOfLines={1}>
                    {item.brand}
                </ScaledText>
                <View style={styles.captionRow}>
                    <ScaledText style={[styles.captionPrice, spotlight && styles.captionPriceLarge]}>
                        ${item.price.toFixed(0)}
                    </ScaledText>
                    <TouchableOpacity
                        style={[styles.captionBuy, spotlight && styles.captionBuyLarge]}
                        onPress={() => handleBuyPress(item)}
                        activeOpacity={0.6}
                        accessibilityLabel={t('inspo.buyNow')}
                        accessibilityRole="button"
                        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    >
                        <Ionicons name="arrow-forward" size={spotlight ? 16 : 13} color="#0A1931" />
                    </TouchableOpacity>
                </View>
            </View>
        </View>
    </Animated.View>
);

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

// ── Outfit Variation Mini Card ─────────────────────────

interface VariationCardProps {
    outfit: ScoredOutfit;
    items: import('../src/types/domain').ClothingItem[];
    onPress: () => void;
}

const VariationCard = ({ outfit, items, onPress }: VariationCardProps) => {
    const cardItems = outfit.outfit.itemIds
        .map((id) => items.find((i) => i.id === id))
        .filter(Boolean)
        .slice(0, 4) as import('../src/types/domain').ClothingItem[];

    return (
        <TouchableOpacity style={styles.variationCard} onPress={onPress} activeOpacity={0.88}>
            <View style={styles.variationGrid}>
                {cardItems.map((item, idx) => (
                    <View key={item.id} style={styles.variationCell}>
                        {item.imageUrl ? (
                            <CachedImage
                                uri={item.imageUrl}
                                style={styles.variationImage}
                                contentFit="contain"
                                fadeIn={false}
                            />
                        ) : (
                            <Ionicons name="shirt-outline" size={22} color={colors.text.tertiary} />
                        )}
                    </View>
                ))}
            </View>
            <View style={styles.variationFooter}>
                <ScaledText style={styles.variationOccasion} numberOfLines={2}>
                    {OCCASION_LABEL[outfit.outfit.occasion] ?? outfit.outfit.occasion}
                </ScaledText>
                <ScaledText style={styles.variationScore}>{Math.round(outfit.score * 100)}%</ScaledText>
            </View>
        </TouchableOpacity>
    );
};

type GuideCategory = 'featured' | 'capsules' | 'outfits';

// ── Main Component ──────────────────────────────────────

type SegmentType = 'guide' | 'shop';

const InspoScreen = () => {
    const { t } = useTranslation();
    const navigation = useAppNavigation();
    const items = useWardrobeStore((s) => s.items);
    const wearLogs = useWardrobeStore((s) => s.wearLogs);

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

    const [savedInspo, setSavedInspo] = useState<ShopCatalogItem[]>([]);
    const [segment, setSegment] = useState<SegmentType>('guide');
    const [searchQuery, setSearchQuery] = useState('');

    const [computedVarietyOutfits, setComputedVarietyOutfits] = useState<ScoredOutfit[]>([]);

    useEffect(() => {
        const task = InteractionManager.runAfterInteractions(() => {
            if (items.length < 3) {
                setComputedVarietyOutfits([]);
                return;
            }
            const outfits = generateVarietyOutfits(items, wearLogs);
            // Deduplicate by itemIds combination
            const seen = new Set<string>();
            const unique = outfits.filter((o) => {
                const key = [...o.outfit.itemIds].sort().join(',');
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            });
            setComputedVarietyOutfits(unique.slice(0, 6));
        });
        return () => task.cancel();
    }, [items, wearLogs]);

    const lastFetchRef = useRef<number>(Date.now());
    const INSPO_REFRESH_TTL_MS = 5 * 60 * 1000; // 5 minutes

    useFocusEffect(
        useCallback(() => {
            let mounted = true;
            const load = async () => {
                try {
                    const raw = await AsyncStorage.getItem('savedInspo');
                    if (raw && mounted) setSavedInspo(JSON.parse(raw));
                } catch (_) { }
            };

            const now = Date.now();
            if (now - lastFetchRef.current > INSPO_REFRESH_TTL_MS) {
                lastFetchRef.current = now;
                refreshFeaturedCapsules();
                refreshShopCatalog();
            }

            load();
            return () => { mounted = false; };
        }, [refreshFeaturedCapsules, refreshShopCatalog])
    );

    const saveInspo = useCallback(async (item: ShopCatalogItem) => {
        setSavedInspo((prev) => {
            const has = prev.some((s) => s.id === item.id);
            const next = has ? prev.filter((s) => s.id !== item.id) : [...prev, item];
            AsyncStorage.setItem('savedInspo', JSON.stringify(next));
            return next;
        });
    }, []);

    const isShopEmpty = syncedShopItems.length === 0;
    const isInitialShopLoad = isShopEmpty && shopCatalogLoading;
    const showingFallbackCatalog = isShopEmpty && !shopCatalogLoading;

    const shopItems = useMemo(() => {
        const baseItems = showingFallbackCatalog ? INSPO_MENS_SHOP_ITEMS : syncedShopItems;

        // Always append curated men's classics so they surface alongside the
        // live Zara feed. Dedupe by id in case fallback already includes them.
        const seenIds = new Set(baseItems.map((item) => item.id));
        const classicsToAppend = CLASSIC_MENS_ITEMS.filter((item) => !seenIds.has(item.id));
        const sourceItems = [...baseItems, ...classicsToAppend];

        const query = searchQuery.trim().toLowerCase();
        if (!query) return sourceItems;

        return sourceItems.filter((item) => {
            const haystack = `${item.brand} ${item.name} ${item.description ?? ''}`.toLowerCase();
            return haystack.includes(query);
        });
    }, [searchQuery, showingFallbackCatalog, syncedShopItems]);

    // ── FlatList helpers for the shop grid ────────────────────────────────────
    const savedInspoSet = useMemo(
        () => new Set(savedInspo.map((s) => s.id)),
        [savedInspo],
    );

    // Break the grid's rhythm on purpose: every 5th item leads a fresh
    // block as a full-width spotlight before the next pair of small cards,
    // so the shop reads like a curated editorial spread instead of a flat
    // uniform tile wall.
    type ShopRow =
        | { type: 'spotlight'; item: ShopCatalogItem; plate: number }
        | { type: 'pair'; items: [ShopCatalogItem, ShopCatalogItem | null] };

    const shopRows = useMemo<ShopRow[]>(() => {
        const rows: ShopRow[] = [];
        let i = 0;
        let plate = 1;
        while (i < shopItems.length) {
            rows.push({ type: 'spotlight', item: shopItems[i], plate: plate++ });
            i += 1;
            for (let pairCount = 0; pairCount < 2 && i < shopItems.length; pairCount++) {
                const left = shopItems[i];
                const right = shopItems[i + 1] ?? null;
                rows.push({ type: 'pair', items: [left, right] });
                i += right ? 2 : 1;
            }
        }
        return rows;
    }, [shopItems]);

    const renderShopRow = useCallback(
        ({ item: row }: { item: ShopRow }) => {
            if (row.type === 'spotlight') {
                return (
                    <ProductCard
                        item={row.item}
                        isSaved={savedInspoSet.has(row.item.id)}
                        onSave={() => saveInspo(row.item)}
                        index={0}
                        t={t}
                        spotlight
                        plate={row.plate}
                    />
                );
            }

            const [left, right] = row.items;
            return (
                <View style={styles.shopRow}>
                    <View style={styles.productCardWrap}>
                        <ProductCard
                            item={left}
                            isSaved={savedInspoSet.has(left.id)}
                            onSave={() => saveInspo(left)}
                            index={0}
                            t={t}
                        />
                    </View>
                    {right ? (
                        <View style={styles.productCardWrap}>
                            <ProductCard
                                item={right}
                                isSaved={savedInspoSet.has(right.id)}
                                onSave={() => saveInspo(right)}
                                index={1}
                                t={t}
                            />
                        </View>
                    ) : (
                        <View style={styles.productCardWrap} />
                    )}
                </View>
            );
        },
        [savedInspoSet, saveInspo, t],
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
                        onChangeText={setSearchQuery}
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
                <ScaledText style={styles.shopCountText}>{shopItems.length}</ScaledText>
            </View>
        </>
    ), [searchQuery, setSearchQuery, shopCatalogError, showingFallbackCatalog, refreshShopCatalog, navigation, featuredCapsulesLoading, featuredCapsules, shopItems.length, t, colors.text.primary, colors.text.tertiary]);

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
            {/* From Your Closet — variety outfits */}
            {computedVarietyOutfits.length > 0 && (
                <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.section}>
                    <ScaledText style={styles.sectionTitle} accessibilityRole="header">{t('inspo.fromYourCloset')}</ScaledText>
                    <FlatList
                        data={computedVarietyOutfits}
                        horizontal
                        showsHorizontalScrollIndicator={false}
                        keyExtractor={(o, idx) => `${o.outfit.itemIds.join(',')}_${idx}`}
                        contentContainerStyle={styles.variationsScroll}
                        renderItem={({ item: outfit }) => (
                            <VariationCard
                                outfit={outfit}
                                items={items}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                    navigation.navigate('Main', { screen: 'Home' });
                                }}
                            />
                        )}
                    />
                </Animated.View>
            )}
            <View style={{ height: 120 }} />
        </>
    ), [computedVarietyOutfits, items, navigation, t]);

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
                    <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
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
                                accessibilityLabel={seg === 'guide' ? t('inspo.guide') : t('inspo.shop')}
                                accessibilityRole="tab"
                                accessibilityState={{ selected: segment === seg }}
                            >
                                <ScaledText style={[styles.segmentText, segment === seg && styles.segmentTextActive]}>
                                    {seg === 'guide' ? t('inspo.guide') : t('inspo.shop')}
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
                                {renderShopHeader()}
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
                                {renderShopHeader()}
                                <View style={styles.emptyState}>
                                    <ScaledText style={styles.emptyStateText}>{t('inspo.noMenswearMatches')}</ScaledText>
                                </View>
                            </ScrollView>
                        ) : (
                            <FlatList
                                data={shopRows}
                                keyExtractor={shopListKeyExtractor}
                                renderItem={renderShopRow}
                                ListHeaderComponent={renderShopHeader()}
                                ListFooterComponent={shopListFooter()}
                                contentContainerStyle={[styles.scrollContent, styles.productsGrid]}
                                showsVerticalScrollIndicator={false}
                                initialNumToRender={4}
                                maxToRenderPerBatch={4}
                                windowSize={3}
                                removeClippedSubviews={true}
                            />
                        )}
                    </>
                )}
            </SafeAreaView>
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
    shopCountText: {
        fontSize: 13,
        fontWeight: '500',
        color: colors.text.tertiary,
    },

    // Scroll
    scrollContent: {
        paddingTop: spacing.sm,
        paddingBottom: 100,
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
        width: 30,
        height: 30,
        borderRadius: 15,
        backgroundColor: '#FFFFFF',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(10,25,49,0.06)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
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
    captionRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
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
    captionBuy: {
        width: 26,
        height: 26,
        borderRadius: 13,
        borderWidth: 1,
        borderColor: 'rgba(10,25,49,0.16)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    captionBuyLarge: {
        width: 40,
        height: 40,
        borderRadius: 20,
        borderColor: '#0A1931',
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

    // Variation cards — From Your Closet
    variationsScroll: {
        paddingHorizontal: spacing.screenPadding,
        gap: 10,
    },
    variationCard: {
        width: 140,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderRadius: 16,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.7)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
        elevation: 3,
    },
    variationGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        width: 140,
        height: 140,
    },
    variationCell: {
        width: 70,
        height: 70,
        backgroundColor: '#F4F7FD',
        alignItems: 'center',
        justifyContent: 'center',
    },
    variationImage: {
        width: '100%',
        height: '100%',
    },
    variationFooter: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 8,
        paddingVertical: 7,
    },
    variationOccasion: {
        fontSize: 11,
        fontWeight: '600',
        color: colors.text.primary,
        flex: 1,
        textTransform: 'capitalize',
    },
    variationScore: {
        fontSize: 11,
        fontWeight: '700',
        color: colors.text.tertiary,
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
