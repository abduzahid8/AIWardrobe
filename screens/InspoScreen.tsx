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
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Animated, { FadeInDown } from 'react-native-reanimated';

import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import { generateVarietyOutfits } from '../src/services/suggestionEngine';
import type { ScoredOutfit } from '../src/services/suggestionEngine';
import { NavigationMenu } from '../src/components/NavigationMenu';
import { INSPO_SHOP_ITEMS } from '../data/inspoShopItems';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

// ── Data ──────────────────────────────────────
const FEATURED_CAPSULES = [
    { id: '1', title: 'Winter Dressing Guide', image: require('../pictures/image copy.png') },
    { id: '2', title: 'The Cozy Edit', image: require('../pictures/image.png') },
    { id: '3', title: 'Capsule Wardrobe Picks', image: 'https://images.unsplash.com/photo-1555069519-127aadedf1ee?w=400&q=80' },
];

const SHOPPING_ITEMS = INSPO_SHOP_ITEMS;

const GUIDE_ITEMS = [
    { id: '1', title: 'Lewis Hamilton', subtitle: 'Street Style Icon', image: require('../pictures/image.png') },
    { id: '2', title: 'A$AP Rocky', subtitle: 'Experimental Luxury', image: require('../pictures/image copy.png') },
];

const CAPSULE_CARD_WIDTH = 180;
const CAPSULE_CARD_HEIGHT = 250;
const PRODUCT_CARD_WIDTH = (SCREEN_WIDTH - spacing.screenPadding * 2 - spacing.sm) / 2;

// ── Sub-Components ──────────────────────────────────────

const FeaturedCapsuleCard = ({ item, index }: { item: (typeof FEATURED_CAPSULES)[0]; index: number }) => (
    <Animated.View entering={FadeInDown.delay(100 + index * 80).duration(400)}>
        <TouchableOpacity
            style={styles.capsuleCard}
            activeOpacity={0.9}
            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
            accessibilityLabel={`${item.title} capsule`}
            accessibilityRole="button"
        >
            <Image
                source={typeof item.image === 'string' ? { uri: item.image } : item.image}
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
    item: (typeof SHOPPING_ITEMS)[0];
    isSaved: boolean;
    onSave: () => void;
    index: number;
}) => (
    <Animated.View entering={FadeInDown.delay(150 + index * 60).duration(400)}>
        <View style={styles.productCard}>
            <View style={styles.productImageBox}>
                <Image
                    source={typeof item.image === 'string' ? { uri: item.image } : item.image}
                    style={styles.productImage}
                    resizeMode="cover"
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
    const navigation = useNavigation();
    const items    = useWardrobeStore((s) => s.items);
    const wearLogs = useWardrobeStore((s) => s.wearLogs);

    const [savedInspo, setSavedInspo] = useState<typeof SHOPPING_ITEMS>([]);
    const [segment, setSegment] = useState<SegmentType>('guide');
    const [searchQuery, setSearchQuery] = useState('');
    const [showNavMenu, setShowNavMenu] = useState(false);

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

    const saveInspo = useCallback(async (item: (typeof SHOPPING_ITEMS)[0]) => {
        setSavedInspo((prev) => {
            const has = prev.some((s) => s.id === item.id);
            const next = has ? prev.filter((s) => s.id !== item.id) : [...prev, item];
            AsyncStorage.setItem('savedInspo', JSON.stringify(next));
            return next;
        });
    }, []);

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={colors.background.primary} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>
                {/* Header */}
                <View style={styles.header}>
                    <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                        <Text style={styles.headerTitle} accessibilityRole="header">Inspiration</Text>
                    </View>
                    <TouchableOpacity
                        style={styles.headerChatButton}
                        onPress={() => setShowNavMenu(true)}
                        accessibilityLabel="Open navigation menu"
                        accessibilityRole="button"
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="menu" size={22} color={colors.text.primary} />
                    </TouchableOpacity>
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
                                    <Text style={styles.sectionTitle} accessibilityRole="header">From Your Closet</Text>
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
                                                    (navigation as any).navigate('Home');
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
                                        placeholder="Flared jeans in light wash, high-waisted..."
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

                            {/* Featured Capsules */}
                            <View style={styles.section}>
                                <Text style={styles.sectionTitle} accessibilityRole="header">Featured Capsules</Text>
                                <ScrollView
                                    horizontal
                                    showsHorizontalScrollIndicator={false}
                                    contentContainerStyle={styles.capsulesScroll}
                                >
                                    {FEATURED_CAPSULES.map((item, index) => (
                                        <FeaturedCapsuleCard key={item.id} item={item} index={index} />
                                    ))}
                                </ScrollView>
                            </View>

                            {/* Product Grid */}
                            <View style={styles.section}>
                                <Text style={styles.sectionTitle} accessibilityRole="header">Spring Transition</Text>
                                <View style={styles.productsGrid}>
                                    {SHOPPING_ITEMS.map((item, index) => (
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
                            </View>
                        </>
                    )}

                    <View style={{ height: 120 }} />
                </ScrollView>
            </SafeAreaView>

            {/* Navigation Menu */}
            <NavigationMenu visible={showNavMenu} onClose={() => setShowNavMenu(false)} />
        </View>
    );
};

// ── Styles ──────────────────────────────────────

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
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
        paddingVertical: 12,
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
        backgroundColor: colors.background.tertiary,
        borderRadius: 24,
        padding: 4,
        width: 300,
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
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 4,
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
        backgroundColor: colors.background.secondary,
        marginHorizontal: spacing.screenPadding,
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.sm + 2,
        borderRadius: 999,
        marginBottom: spacing.lg,
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
        borderRadius: radius.xl,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        position: 'relative',
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
    capsuleCard: {
        width: CAPSULE_CARD_WIDTH,
        height: CAPSULE_CARD_HEIGHT,
        borderRadius: radius.lg,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
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
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        overflow: 'hidden',
        marginBottom: spacing.xs,
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

    // Variation cards — From Your Closet
    variationsScroll: {
        paddingHorizontal: spacing.screenPadding,
        gap: 10,
    },
    variationCard: {
        width: 140,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        overflow: 'hidden',
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
        backgroundColor: colors.background.tertiary,
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
});

export default InspoScreen;
