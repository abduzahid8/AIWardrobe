/**
 * InspoScreen — Inspiration page
 * Minimalist Liquid Glass design with Guide + Shop tabs
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    FlatList,
    Image,
    StatusBar,
    TextInput,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Animated, { FadeInDown } from 'react-native-reanimated';

import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import { generateVarietyOutfits } from '../src/services/suggestionEngine';
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

const GUIDE_ITEMS = [
    { id: '1', title: 'Lewis Hamilton', subtitle: 'Street Style Icon', image: 'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=600&q=80' },
    { id: '2', title: 'A$AP Rocky', subtitle: 'Experimental Luxury', image: 'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=600&q=80' },
];

const CAPSULE_CARD_WIDTH = 180;
const CAPSULE_CARD_HEIGHT = 250;
const PRODUCT_CARD_WIDTH = (SCREEN_WIDTH - spacing.screenPadding * 2 - spacing.sm) / 2;

// ── Sub-Components ──────────────────────────────────────

const FeaturedCapsuleCard = ({ item, index }: { item: FeaturedCapsule; index: number }) => (
    <Animated.View entering={FadeInDown.delay(100 + index * 80).duration(400)}>
        <TouchableOpacity
            style={styles.capsuleCard}
            activeOpacity={0.9}
            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
            accessibilityLabel={`${item.title} capsule`}
            accessibilityRole="button"
        >
            <Image
                source={{ uri: item.imageUrl }}
                style={styles.capsuleImage}
                resizeMode="cover"
            />
            <LinearGradient
                colors={['transparent', 'rgba(0,0,0,0.65)']}
                style={styles.capsuleGradient}
            >
                <Text style={styles.capsuleTitle}>{item.title}</Text>
            </LinearGradient>
        </TouchableOpacity>
    </Animated.View>
);

const ProductCard = ({
    item,
    isSaved,
    onSave,
    index,
}: {
    item: ShopCatalogItem;
    isSaved: boolean;
    onSave: () => void;
    index: number;
}) => (
    // Cap the stagger so later pages (index 100+) don't wait many seconds
    // before their entering animation starts — otherwise pressing "Load more"
    // appears to do nothing because new cards are invisible until the delay elapses.
    <Animated.View entering={FadeInDown.delay(Math.min(150 + (index % 12) * 40, 600)).duration(320)}>
        <View style={styles.productCard}>
            <View style={styles.productImageBox}>
                <Image
                    source={typeof item.imageUrl === 'string' ? { uri: item.imageUrl } : item.imageUrl}
                    style={styles.productImage}
                    resizeMode="contain"
                />
                <TouchableOpacity
                    style={styles.saveButton}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onSave();
                    }}
                    accessibilityLabel={isSaved ? 'Remove from saved' : 'Save item'}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                >
                    <View style={styles.saveButtonCircle}>
                        <Ionicons
                            name={isSaved ? 'heart' : 'heart-outline'}
                            size={18}
                            color={isSaved ? '#FF3B5C' : colors.text.primary}
                        />
                    </View>
                </TouchableOpacity>
            </View>
            <Text style={styles.productBrand} numberOfLines={1}>
                {item.brand}
            </Text>
            <Text style={styles.productPrice}>${item.price.toFixed(2)}</Text>
        </View>
    </Animated.View>
);

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
                            <Image
                                source={{ uri: item.imageUrl }}
                                style={styles.variationImage}
                                resizeMode="contain"
                            />
                        ) : (
                            <Ionicons name="shirt-outline" size={22} color={colors.text.tertiary} />
                        )}
                    </View>
                ))}
            </View>
            <View style={styles.variationFooter}>
                <Text style={styles.variationOccasion} numberOfLines={1}>
                    {outfit.outfit.occasion}
                </Text>
                <Text style={styles.variationScore}>{Math.round(outfit.score * 100)}%</Text>
            </View>
        </TouchableOpacity>
    );
};

// ── Main Component ──────────────────────────────────────

type SegmentType = 'guide' | 'shop';

const InspoScreen = () => {
    const { t } = useTranslation();
    const navigation = useAppNavigation();
    const items    = useWardrobeStore((s) => s.items);
    const wearLogs = useWardrobeStore((s) => s.wearLogs);
    const {
        items: syncedShopItems,
        loading: shopCatalogLoading,
        loadingMore: shopCatalogLoadingMore,
        error: shopCatalogError,
        hasMore: shopCatalogHasMore,
        loadMore: loadMoreShopCatalog,
        refresh: refreshShopCatalog,
    } = useShopCatalog();

    const {
        items: featuredCapsules,
        loading: featuredCapsulesLoading,
    } = useFeaturedCapsules();

    const [savedInspo, setSavedInspo] = useState<ShopCatalogItem[]>([]);
    const [segment, setSegment] = useState<SegmentType>('guide');
    const [searchQuery, setSearchQuery] = useState('');

    const varietyOutfits = useMemo<ScoredOutfit[]>(() => {
        if (items.length < 3) return [];
        return generateVarietyOutfits(items, wearLogs).slice(0, 6);
    }, [items.length, wearLogs.length]);

    useFocusEffect(
        useCallback(() => {
            let mounted = true;
            const load = async () => {
                try {
                    const raw = await AsyncStorage.getItem('savedInspo');
                    if (raw && mounted) setSavedInspo(JSON.parse(raw));
                } catch (_) { }
            };
            load();
            return () => { mounted = false; };
        }, [])
    );

    const saveInspo = useCallback(async (item: ShopCatalogItem) => {
        setSavedInspo((prev) => {
            const has = prev.some((s) => s.id === item.id);
            const next = has ? prev.filter((s) => s.id !== item.id) : [...prev, item];
            AsyncStorage.setItem('savedInspo', JSON.stringify(next));
            return next;
        });
    }, []);

    const showingFallbackCatalog = syncedShopItems.length === 0;
    const isInitialShopLoad = !showingFallbackCatalog && shopCatalogLoading && syncedShopItems.length === 0;

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

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
                pointerEvents="none"
            />
            <View pointerEvents="none" style={styles.backgroundOrbTop} />
            <View pointerEvents="none" style={styles.backgroundOrbBottom} />
            <StatusBar barStyle="dark-content" backgroundColor={colors.background.primary} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>
                {/* Header */}
                <View style={styles.header}>
                    <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                        <Text style={styles.headerTitle} accessibilityRole="header">{t('inspo.title')}</Text>
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
                                accessibilityLabel={seg === 'guide' ? 'Guide' : 'Shop'}
                                accessibilityRole="tab"
                                accessibilityState={{ selected: segment === seg }}
                            >
                                <Text style={[styles.segmentText, segment === seg && styles.segmentTextActive]}>
                                    {seg === 'guide' ? 'Guide' : 'Shop'}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {/* ── Guide Tab ── */}
                    {segment === 'guide' && (
                        <>
                            <View style={styles.section}>
                                {GUIDE_ITEMS.map((item, index) => (
                                    <Animated.View key={item.id} entering={FadeInDown.delay(100 + index * 120).duration(500)}>
                                        <View style={styles.guideCardContainer}>
                                            <View style={styles.guideCard}>
                                                <Image
                                                    source={typeof item.image === 'string' ? { uri: item.image } : item.image}
                                                    style={styles.guideImage}
                                                    resizeMode="cover"
                                                />
                                                <LinearGradient
                                                    colors={['transparent', 'rgba(0,0,0,0.75)']}
                                                    style={styles.guideGradient}
                                                >
                                                    <Text style={styles.guideTitle}>{item.title}</Text>
                                                    <Text style={styles.guideSubtitle}>{item.subtitle}</Text>
                                                </LinearGradient>
                                            </View>
                                        </View>
                                    </Animated.View>
                                ))}
                            </View>

                            {/* From Your Closet — variety outfits */}
                            {varietyOutfits.length > 0 && (
                                <Animated.View entering={FadeInDown.delay(350).duration(400)} style={styles.section}>
                                    <Text style={styles.sectionTitle} accessibilityRole="header">{t('inspo.fromYourCloset')}</Text>
                                    <FlatList
                                        data={varietyOutfits}
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
                        </>
                    )}

                    {/* ── Shop Tab ── */}
                    {segment === 'shop' && (
                        <>
                            {/* Search */}
                            <Animated.View entering={FadeInDown.delay(80).duration(400)}>
                                <View style={styles.searchContainer}>
                                    <Ionicons name="search" size={20} color={colors.text.tertiary} style={styles.searchIcon} />
                                    <TextInput
                                        placeholder="Brown knit polo, tailored trousers, loafers..."
                                        placeholderTextColor={colors.text.tertiary}
                                        value={searchQuery}
                                        onChangeText={setSearchQuery}
                                        style={styles.searchInput}
                                        returnKeyType="search"
                                        accessibilityLabel="Search for clothing items"
                                        maxLength={200}
                                    />
                                </View>
                            </Animated.View>

                            {(shopCatalogError || showingFallbackCatalog) && (
                                <Animated.View entering={FadeInDown.delay(90).duration(300)}>
                                    <View style={styles.catalogStatusBanner}>
                                        <Text style={styles.catalogStatusText}>
                                            {showingFallbackCatalog
                                                ? 'Live Zara menswear is empty right now. Showing backup men products for now.'
                                                : 'Live catalog refresh failed. Showing the latest synced results.'}
                                        </Text>
                                        <TouchableOpacity onPress={refreshShopCatalog} accessibilityRole="button">
                                            <Text style={styles.catalogStatusAction}>{t('common.retry')}</Text>
                                        </TouchableOpacity>
                                    </View>
                                </Animated.View>
                            )}

                            {/* Personal Stylist Button */}
                            <Animated.View entering={FadeInDown.delay(120).duration(400)}>
                                <TouchableOpacity
                                    style={styles.personalStylistButton}
                                    onPress={() => {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                        navigation.navigate('AIOutfit', { source: 'shop' });
                                    }}
                                    accessibilityLabel="Personal Stylist"
                                    accessibilityRole="button"
                                >
                                    <LinearGradient
                                        colors={['#0A1931', '#1a3a5c']}
                                        start={{ x: 0, y: 0 }}
                                        end={{ x: 1, y: 1 }}
                                        style={styles.personalStylistGradient}
                                    >
                                        <Ionicons name="sparkles" size={20} color="#FFF" />
                                        <Text style={styles.personalStylistText}>{t('inspo.personalStylist')}</Text>
                                        <Ionicons name="arrow-forward" size={18} color="rgba(255,255,255,0.8)" />
                                    </LinearGradient>
                                </TouchableOpacity>
                            </Animated.View>

                            {/* Featured Capsules — sourced from Supabase (`featured_capsules`) */}
                            {(featuredCapsulesLoading || featuredCapsules.length > 0) && (
                                <View style={styles.section}>
                                    <Text style={styles.sectionTitle} accessibilityRole="header">{t('inspo.featuredCapsules')}</Text>
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
                                                <FeaturedCapsuleCard key={item.id} item={item} index={index} />
                                            ))}
                                        </ScrollView>
                                    )}
                                </View>
                            )}

                            {/* Product Grid */}
                            <View style={styles.section}>
                                <Text style={styles.sectionTitle} accessibilityRole="header">
                                    Shop
                                </Text>
                                {isInitialShopLoad ? (
                                    <View style={styles.loadingRow}>
                                        {[0, 1, 2, 3].map((idx) => (
                                            <View
                                                key={`shop-skeleton-${idx}`}
                                                style={[
                                                    styles.productCardWrap,
                                                    idx % 2 === 0 && styles.productCardWrapLeft,
                                                    styles.skeletonCard,
                                                ]}
                                            />
                                        ))}
                                    </View>
                                ) : shopItems.length === 0 ? (
                                    <View style={styles.emptyState}>
                                        <Text style={styles.emptyStateText}>{t('inspo.noMenswearMatches')}</Text>
                                    </View>
                                ) : (
                                    <>
                                        <View style={styles.productsGrid}>
                                            {shopItems.map((item, index) => (
                                                <View
                                                    key={item.id}
                                                    style={[
                                                        styles.productCardWrap,
                                                        index % 2 === 0 && styles.productCardWrapLeft,
                                                    ]}
                                                >
                                                    <ProductCard
                                                        item={item}
                                                        isSaved={savedInspo.some((s) => s.id === item.id)}
                                                        onSave={() => saveInspo(item)}
                                                        index={index}
                                                    />
                                                </View>
                                            ))}
                                        </View>

                                        {!showingFallbackCatalog && shopCatalogHasMore && (
                                            <TouchableOpacity
                                                style={styles.loadMoreButton}
                                                onPress={loadMoreShopCatalog}
                                                disabled={shopCatalogLoadingMore}
                                                accessibilityRole="button"
                                                accessibilityLabel="Load more shop products"
                                            >
                                                {shopCatalogLoadingMore ? (
                                                    <ActivityIndicator size="small" color={colors.text.primary} />
                                                ) : (
                                                    <Text style={styles.loadMoreButtonText}>{t('inspo.loadMoreProducts')}</Text>
                                                )}
                                            </TouchableOpacity>
                                        )}
                                    </>
                                )}
                            </View>
                        </>
                    )}

                    <View style={{ height: 120 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

// ── Styles ──────────────────────────────────────

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
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
        backgroundColor: 'rgba(255,255,255,0.84)',
        borderRadius: 26,
        padding: 4,
        width: 300,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
    },
    segmentButton: {
        flex: 1,
        paddingVertical: 10,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 20,
    },
    segmentButtonActive: {
        backgroundColor: colors.background.primary,
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 10,
        elevation: 2,
    },
    segmentText: {
        fontSize: 15,
        fontWeight: '500',
        color: colors.text.tertiary,
    },
    segmentTextActive: {
        color: colors.text.primary,
        fontWeight: '600',
    },

    // Scroll
    scrollContent: {
        paddingTop: spacing.sm,
    },

    // Search
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.88)',
        marginHorizontal: spacing.screenPadding,
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.sm + 2,
        borderRadius: 999,
        marginBottom: spacing.lg,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
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
        fontSize: 18,
        fontWeight: '700',
        color: colors.text.primary,
        marginBottom: spacing.md,
        paddingHorizontal: spacing.screenPadding,
    },

    // Guide Cards (full-width squares)
    guideCardContainer: {
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.lg,
    },
    guideCard: {
        width: '100%',
        aspectRatio: 1,
        borderRadius: 28,
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
    guideGradient: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: spacing.md,
        paddingBottom: spacing.lg,
    },
    guideTitle: {
        fontSize: 24,
        fontWeight: '700',
        color: '#FFF',
        marginBottom: 4,
    },
    guideSubtitle: {
        fontSize: 16,
        fontWeight: '500',
        color: 'rgba(255,255,255,0.8)',
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
        borderRadius: 24,
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

    // Product Grid (2-col)
    productsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        paddingHorizontal: spacing.screenPadding,
    },
    loadingRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        paddingHorizontal: spacing.screenPadding,
    },
    skeletonCard: {
        height: PRODUCT_CARD_WIDTH * (4 / 3) + 40,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        marginBottom: spacing.md,
    },
    productCardWrap: {
        width: PRODUCT_CARD_WIDTH,
        marginBottom: spacing.md,
    },
    productCardWrapLeft: {
        marginRight: spacing.sm,
    },
    productCard: {
        width: '100%',
    },
    productImageBox: {
        width: '100%',
        aspectRatio: 3 / 4,
        backgroundColor: 'rgba(255,255,255,0.92)',
        borderRadius: 22,
        overflow: 'hidden',
        marginBottom: spacing.xs,
        padding: spacing.sm,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.06)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.05,
        shadowRadius: 12,
        elevation: 3,
    },
    productImage: {
        width: '100%',
        height: '100%',
    },
    saveButton: {
        position: 'absolute',
        top: spacing.sm,
        right: spacing.sm,
        zIndex: 10,
    },
    saveButtonCircle: {
        width: 34,
        height: 34,
        borderRadius: 17,
        backgroundColor: 'rgba(255,255,255,0.9)',
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 4,
        elevation: 2,
    },
    productBrand: {
        fontSize: 13,
        fontWeight: '600',
        color: colors.text.secondary,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
        marginBottom: 2,
    },
    productPrice: {
        fontSize: 16,
        fontWeight: '700',
        color: colors.text.primary,
    },
    emptyState: {
        paddingHorizontal: spacing.screenPadding,
        marginHorizontal: spacing.screenPadding,
        paddingVertical: 20,
        borderRadius: 24,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.06)',
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
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.88)',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: spacing.md,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
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
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.9)',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
    },
    loadMoreButtonText: {
        ...typography.scale.bodyMedium,
        color: colors.text.primary,
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
        borderRadius: 22,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.06)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.05,
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
        marginBottom: spacing.lg,
        borderRadius: 24,
        overflow: 'hidden',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.14,
        shadowRadius: 18,
        elevation: 6,
    },
    personalStylistGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.lg,
        paddingVertical: spacing.md + 6,
    },
    personalStylistText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#FFF',
        flex: 1,
        marginLeft: spacing.sm,
    },
});

export default InspoScreen;
