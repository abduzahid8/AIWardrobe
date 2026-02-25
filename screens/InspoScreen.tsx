/**
 * InspoScreen - Inspiration page
 * Layout: Featured Capsules + Monnaie's Spring Transition (exact design)
 */

import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    StatusBar,
    TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

// Featured Capsules – collage-style cards with overlay text
const FEATURED_CAPSULES = [
    {
        id: '1',
        title: 'Winter Dressing Guide',
        image: require('../pictures/image copy.png'),
    },
    {
        id: '2',
        title: 'The Cozy Edit',
        image: require('../pictures/image.png'),
    },
    {
        id: '3',
        title: 'Capsule Wardrobe Picks',
        image: 'https://images.unsplash.com/photo-1555069519-127aadedf1ee?w=400&q=80',
    },
];

// Monnaie's Spring Transition – product cards (brand, price, heart)
const SHOPPING_ITEMS = [
    {
        id: '1',
        brand: 'ZARA',
        price: 129.00,
        image: require('../pictures/shop/image copy.png'),
    },
    {
        id: '2',
        brand: 'ZARA',
        price: 89.90,
        image: require('../pictures/shop/image copy 2.png'),
    },
    {
        id: '3',
        brand: 'ZARA',
        price: 69.90,
        image: require('../pictures/shop/image copy 3.png'),
    },
    {
        id: '4',
        brand: 'ZARA',
        price: 15.90,
        image: require('../pictures/shop/image copy 4.png'),
    },
];

const GUIDE_ITEMS = [
    {
        id: '1',
        title: 'Lewis Hamilton',
        subtitle: 'Street Style Icon',
        image: require('../pictures/image.png'),
    },
    {
        id: '2',
        title: 'A$AP Rocky',
        subtitle: 'Experimental Luxury',
        image: require('../pictures/image copy.png'),
    },
];

const CAPSULE_CARD_WIDTH = 180;
const CAPSULE_CARD_HEIGHT = 250;
const PRODUCT_CARD_WIDTH = (SCREEN_WIDTH - spacing.screenPadding * 2 - spacing.sm) / 2;

// Capsule card: image + bottom-left white text overlay
const FeaturedCapsuleCard = ({ item }: { item: (typeof FEATURED_CAPSULES)[0] }) => (
    <TouchableOpacity
        style={styles.capsuleCard}
        activeOpacity={0.9}
        onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
    >
        <Image
            source={typeof item.image === 'string' ? { uri: item.image } : item.image}
            style={styles.capsuleImage}
            resizeMode="cover"
        />
        <LinearGradient
            colors={['transparent', 'rgba(0,0,0,0.6)']}
            style={styles.capsuleGradient}
        >
            <Text style={styles.capsuleTitle}>{item.title}</Text>
        </LinearGradient>
    </TouchableOpacity>
);

// Product card: image, heart in white circle (top-right), brand, price
const ProductCard = ({
    item,
    isSaved,
    onSave,
}: {
    item: (typeof SHOPPING_ITEMS)[0];
    isSaved: boolean;
    onSave: () => void;
}) => (
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
            >
                <View style={styles.saveButtonCircle}>
                    <Ionicons
                        name={isSaved ? 'heart' : 'heart-outline'}
                        size={18}
                        color="#0A1931"
                    />
                </View>
            </TouchableOpacity>
        </View>
        <Text style={styles.productBrand} numberOfLines={1}>
            {item.brand}
        </Text>
        <Text style={styles.productPrice}>${item.price.toFixed(2)}</Text>
    </View>
);

type SegmentType = 'guide' | 'shop';

const InspoScreen = () => {
    const [savedInspo, setSavedInspo] = useState<typeof SHOPPING_ITEMS>([]);
    const [segment, setSegment] = useState<SegmentType>('guide');
    const [searchQuery, setSearchQuery] = useState('');

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
            return () => {
                mounted = false;
            };
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
                {/* Header Container exactly identical to MyClosetScreen */}
                <View style={[styles.header, { justifyContent: 'center' }]}>
                    <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                        <Text style={styles.headerTitle}>Inspiration</Text>
                    </View>
                </View>

                {/* Segmented Control */}
                <View style={styles.segmentContainer}>
                    <View style={styles.segmentBackground}>
                        <TouchableOpacity
                            style={[styles.segmentButton, segment === 'guide' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setSegment('guide');
                            }}
                        >
                            <Text style={[styles.segmentText, segment === 'guide' && styles.segmentTextActive]}>Guide</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.segmentButton, segment === 'shop' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setSegment('shop');
                            }}
                        >
                            <Text style={[styles.segmentText, segment === 'shop' && styles.segmentTextActive]}>Shop</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >



                    // ... (existing code)

                    {/* Guide Content - Big Square Star Photos */}
                    {segment === 'guide' && (
                        <View style={styles.section}>
                            {GUIDE_ITEMS.map((item) => (
                                <View key={item.id} style={styles.guideCardContainer}>
                                    <View style={styles.guideCard}>
                                        <Image
                                            source={typeof item.image === 'string' ? { uri: item.image } : item.image}
                                            style={styles.guideImage}
                                            resizeMode="cover"
                                        />
                                        <LinearGradient
                                            colors={['transparent', 'rgba(0,0,0,0.8)']}
                                            style={styles.guideGradient}
                                        >
                                            <Text style={styles.guideTitle}>{item.title}</Text>
                                            <Text style={styles.guideSubtitle}>{item.subtitle}</Text>
                                        </LinearGradient>
                                    </View>
                                </View>
                            ))}
                        </View>
                    )}

                    {/* Shop Content */}
                    {segment === 'shop' && (
                        <>
                            {/* Search input (Shop Only) */}
                            <View style={styles.searchContainer}>
                                <Ionicons name="search" size={20} color={colors.text.tertiary} style={styles.searchIcon} />
                                <TextInput
                                    placeholder="Flared jeans in light wash, high-waisted..."
                                    placeholderTextColor={colors.text.tertiary}
                                    value={searchQuery}
                                    onChangeText={setSearchQuery}
                                    style={styles.searchInput}
                                    returnKeyType="search"
                                />
                            </View>

                            {/* Featured Capsules (Shop Only now, as Guide has its own content) */}
                            <View style={styles.section}>
                                <Text style={styles.sectionTitle}>Featured Capsules</Text>
                                <ScrollView
                                    horizontal
                                    showsHorizontalScrollIndicator={false}
                                    contentContainerStyle={styles.capsulesScroll}
                                >
                                    {FEATURED_CAPSULES.map((item) => (
                                        <FeaturedCapsuleCard key={item.id} item={item} />
                                    ))}
                                </ScrollView>
                            </View>

                            {/* Monnaie's Spring Transition */}
                            <View style={styles.section}>
                                <Text style={styles.sectionTitle}>Monnaie's Spring Transition</Text>
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
                                            />
                                        </View>
                                    ))}
                                </View>
                            </View>
                        </>
                    )}

                    <View style={{ height: 100 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    // ... (existing styles)

    // Guide Styles
    guideCardContainer: {
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.lg,
    },
    guideCard: {
        width: '100%',
        aspectRatio: 1, // Big Square
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
        color: '#RGBA(255,255,255,0.8)',
    },

    // ... (rest of styles)

    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
    },
    safeArea: { flex: 1 },
    scrollContent: {
        paddingTop: spacing.md,
    },
    // Header
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingVertical: 12,
        backgroundColor: '#FFFFFF',
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        fontWeight: '700',
        color: '#0A1931',
        letterSpacing: 0.3,
    },
    headerSpacer: { width: 28 },

    // Segmented Control
    segmentContainer: {
        alignItems: 'center',
        paddingTop: 15,
        paddingBottom: 20,
        backgroundColor: '#FFFFFF',
    },
    segmentBackground: {
        flexDirection: 'row',
        backgroundColor: '#E5E5EA', // iOS System Gray 5
        borderRadius: 24, // Rounded
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
        backgroundColor: '#FFFFFF',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    segmentText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#8E8E93',
    },
    segmentTextActive: {
        color: '#0A1931',
        fontWeight: '600',
    },

    // Search input
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

    // Featured Capsules
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

    // Monnaie's Spring Transition – 2-column grid
    productsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        paddingHorizontal: spacing.screenPadding,
    },
    productCardWrap: {
        width: PRODUCT_CARD_WIDTH,
        marginBottom: spacing.sm,
    },
    productCardWrapLeft: {
        marginRight: spacing.sm,
    },
    productCard: {
        width: '100%',
    },
    productImageBox: {
        width: '100%',
        aspectRatio: 1,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.md,
        overflow: 'hidden',
        marginBottom: spacing.xs,
    },
    productImage: {
        width: '100%',
        height: '100%',
    },
    saveButton: {
        position: 'absolute',
        top: spacing.xs,
        right: spacing.xs,
        zIndex: 10,
    },
    saveButtonCircle: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#FFF',
        alignItems: 'center',
        justifyContent: 'center',
    },
    productBrand: {
        fontSize: 14,
        fontWeight: '700',
        color: colors.text.primary,
        textTransform: 'uppercase',
        marginBottom: 2,
    },
    productPrice: {
        fontSize: 15,
        fontWeight: '400',
        color: colors.text.primary,
    },
});

export default InspoScreen;
