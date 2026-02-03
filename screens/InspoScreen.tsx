/**
 * InspoScreen - 2026 Redesign
 * Inspiration Page with Liquid Glass aesthetics and Bento Grid products
 * Based on 2026 Digital Experience Report guidelines
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
    Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
    FadeInUp,
    FadeIn,
    useAnimatedStyle,
    useSharedValue,
    withSpring
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';

// 2026 Design System
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import {
    BentoGrid,
    BentoItem,
    FrostedGlassCard,
    PressableGlassCard,
} from '../components/ui';
import { useAccessibility } from '../hooks/useAccessibility';

const { width } = Dimensions.get('window');
const { colors, spacing, typography, radius, animation } = LiquidGlass2026Theme;

type TabType = 'community' | 'shopping' | 'saved';

// Sample trending data
const TRENDING_ITEMS = [
    { id: '1', image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400', aspectRatio: 1.2, likes: 234 },
    { id: '2', image: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400', aspectRatio: 1.5, likes: 189 },
    { id: '3', image: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400', aspectRatio: 1.3, likes: 312 },
    { id: '4', image: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400', aspectRatio: 1.4, likes: 156 },
];

// Sample shopping data
const SHOPPING_ITEMS = [
    { id: '1', brand: 'LORO PIANA', name: 'Cashmere Sweater', price: 1200, image: 'https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=300', discount: 0 },
    { id: '2', brand: 'Everlane', name: 'The No-Sweat Sweater', price: 98, image: 'https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=300', discount: 20 },
    { id: '3', brand: 'Todd Snyder', name: 'Silk-Cashmere Crewneck', price: 298, image: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300', discount: 0 },
    { id: '4', brand: 'Simkhai', name: 'Bennett Sweater', price: 365, image: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300', discount: 15 },
    { id: '5', brand: 'Zegna', name: 'Oasi Cashmere Jacket', price: 6090, image: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=300', discount: 0 },
    { id: '6', brand: 'ZEGNA', name: 'Riviera Slim-Fit Wool', price: 2730, image: 'https://images.unsplash.com/photo-1555069519-127aadedf1ee?w=300', discount: 10 },
];

// Tab Button with Liquid Glass
const TabButton = ({ title, isActive, onPress }: { title: string; isActive: boolean; onPress: () => void }) => (
    <TouchableOpacity
        style={[styles.tabButton, isActive && styles.tabButtonActive]}
        onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress();
        }}
        accessibilityRole="tab"
        accessibilityState={{ selected: isActive }}
    >
        <Text style={[styles.tabText, isActive && styles.tabTextActive]}>{title}</Text>
        {isActive && <View style={styles.tabIndicator} />}
    </TouchableOpacity>
);

// Trending Card with Glass overlay
const TrendingCard = ({ item, index }: { item: typeof TRENDING_ITEMS[0]; index: number }) => {
    const { isReducedMotionEnabled } = useAccessibility();

    return (
        <Animated.View
            entering={isReducedMotionEnabled ? undefined : FadeInUp.delay(index * 80).springify()}
            style={[styles.trendingCard, { aspectRatio: 1 / item.aspectRatio }]}
        >
            <Image source={{ uri: item.image }} style={styles.trendingImage} resizeMode="cover" />

            {/* Glass overlay with likes */}
            <BlurView intensity={60} tint="dark" style={styles.trendingOverlay}>
                <Ionicons name="heart" size={14} color="#FFF" />
                <Text style={styles.trendingLikes}>{item.likes}</Text>
            </BlurView>
        </Animated.View>
    );
};

// Shopping Product Card with Glass effect
const ProductCard = ({
    item,
    onSave,
    index,
    isSaved
}: {
    item: typeof SHOPPING_ITEMS[0];
    onSave: () => void;
    index: number;
    isSaved: boolean;
}) => {
    const { isReducedMotionEnabled } = useAccessibility();
    const [saved, setSaved] = useState(isSaved);
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, animation.spring.snappy) }]
    }));

    return (
        <BentoItem colSpan={1} aspectRatio="auto" index={index} animated={!isReducedMotionEnabled}>
            <Animated.View style={[styles.productCard, animatedStyle]}>
                <View style={styles.productImageBox}>
                    <Image source={{ uri: item.image }} style={styles.productImage} resizeMode="cover" />

                    {/* Discount badge */}
                    {item.discount > 0 && (
                        <View style={styles.discountBadge}>
                            <Text style={styles.discountText}>-{item.discount}%</Text>
                        </View>
                    )}

                    {/* Save button */}
                    <TouchableOpacity
                        style={styles.saveButton}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            setSaved(!saved);
                            if (!saved) onSave();
                        }}
                        accessibilityLabel={saved ? 'Remove from saved' : 'Save item'}
                    >
                        <BlurView intensity={80} tint="light" style={styles.saveButtonBlur}>
                            <Ionicons
                                name={saved ? "bookmark" : "bookmark-outline"}
                                size={16}
                                color={saved ? colors.accent.primary : colors.text.primary}
                            />
                        </BlurView>
                    </TouchableOpacity>
                </View>

                <View style={styles.productInfo}>
                    <Text style={styles.productBrand}>{item.brand}</Text>
                    <Text style={styles.productName} numberOfLines={2}>{item.name}</Text>
                    <View style={styles.priceRow}>
                        <Text style={styles.productPrice}>${item.price.toLocaleString()}</Text>
                        {item.discount > 0 && (
                            <Text style={styles.originalPrice}>
                                ${Math.round(item.price / (1 - item.discount / 100)).toLocaleString()}
                            </Text>
                        )}
                    </View>
                </View>
            </Animated.View>
        </BentoItem>
    );
};

const InspoScreen = () => {
    const navigation = useNavigation();
    const { isReducedMotionEnabled } = useAccessibility();
    const [activeTab, setActiveTab] = useState<TabType>('community');
    const [savedInspo, setSavedInspo] = useState<any[]>([]);

    useFocusEffect(useCallback(() => {
        loadSavedInspo();
    }, []));

    const loadSavedInspo = async () => {
        try {
            const saved = await AsyncStorage.getItem('savedInspo');
            if (saved) setSavedInspo(JSON.parse(saved));
        } catch (e) { }
    };

    const saveInspo = async (item: any) => {
        const updated = [...savedInspo, item];
        setSavedInspo(updated);
        await AsyncStorage.setItem('savedInspo', JSON.stringify(updated));
    };

    const handleAddInspo = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        (navigation as any).navigate('Camera');
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={colors.background.primary} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Header with Liquid Glass */}
                <BlurView intensity={Platform.OS === 'ios' ? 80 : 100} tint="light" style={styles.header}>
                    <Text style={styles.headerTitle}>Inspiration</Text>
                    <TouchableOpacity
                        style={styles.headerButton}
                        onPress={handleAddInspo}
                        accessibilityLabel="Add inspiration"
                    >
                        <Ionicons name="add-circle-outline" size={26} color={colors.text.primary} />
                    </TouchableOpacity>
                </BlurView>

                {/* Tabs */}
                <View style={styles.tabsContainer}>
                    <TabButton title="Community" isActive={activeTab === 'community'} onPress={() => setActiveTab('community')} />
                    <TabButton title="Shopping" isActive={activeTab === 'shopping'} onPress={() => setActiveTab('shopping')} />
                    <TabButton title="Saved" isActive={activeTab === 'saved'} onPress={() => setActiveTab('saved')} />
                </View>

                {/* Content */}
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {activeTab === 'community' && (
                        <>
                            {/* Trending Section */}
                            <View style={styles.trendingGrid}>
                                <View style={styles.trendingColumn}>
                                    {TRENDING_ITEMS.filter((_, i) => i % 2 === 0).map((item, idx) => (
                                        <TrendingCard key={item.id} item={item} index={idx * 2} />
                                    ))}
                                </View>
                                <View style={styles.trendingColumn}>
                                    {TRENDING_ITEMS.filter((_, i) => i % 2 === 1).map((item, idx) => (
                                        <TrendingCard key={item.id} item={item} index={idx * 2 + 1} />
                                    ))}
                                </View>
                            </View>

                            {/* Trending Info Card */}
                            <Animated.View
                                entering={isReducedMotionEnabled ? undefined : FadeIn.delay(300)}
                            >
                                <FrostedGlassCard style={styles.trendingInfoCard}>
                                    <View style={styles.updatedBadge}>
                                        <Text style={styles.updatedText}>UPDATED TODAY</Text>
                                        <View style={styles.liveDot} />
                                    </View>
                                    <Text style={styles.trendingTitle}>Trending Now</Text>
                                    <Text style={styles.trendingSubtitle}>
                                        See what's popular in the community
                                    </Text>
                                    <TouchableOpacity style={styles.discoverButton}>
                                        <Text style={styles.discoverText}>Discover trends</Text>
                                        <Ionicons name="arrow-forward" size={16} color={colors.accent.primary} />
                                    </TouchableOpacity>
                                </FrostedGlassCard>
                            </Animated.View>
                        </>
                    )}

                    {activeTab === 'shopping' && (
                        <>
                            <View style={styles.shopHeader}>
                                <Text style={styles.sectionTitle}>Curated For You</Text>
                                <TouchableOpacity>
                                    <Text style={styles.seeAllText}>See all</Text>
                                </TouchableOpacity>
                            </View>

                            <BentoGrid columns={2} gap={spacing.md} padding={0}>
                                {SHOPPING_ITEMS.map((item, index) => (
                                    <ProductCard
                                        key={item.id}
                                        item={item}
                                        index={index}
                                        isSaved={savedInspo.some(s => s.id === item.id)}
                                        onSave={() => saveInspo(item)}
                                    />
                                ))}
                            </BentoGrid>
                        </>
                    )}

                    {activeTab === 'saved' && (
                        savedInspo.length === 0 ? (
                            <FrostedGlassCard style={styles.emptyState} contentStyle={styles.emptyContent}>
                                <View style={styles.emptyIconContainer}>
                                    <Ionicons name="bookmark-outline" size={48} color={colors.text.tertiary} />
                                </View>
                                <Text style={styles.emptyTitle}>No saved inspo yet</Text>
                                <Text style={styles.emptySubtitle}>
                                    Get inspired — save something you like or share your own style.
                                </Text>
                                <TouchableOpacity
                                    style={styles.addInspoButton}
                                    onPress={handleAddInspo}
                                    accessibilityLabel="Add inspiration"
                                >
                                    <LinearGradient
                                        colors={colors.gradients.primaryAccent as [string, string]}
                                        start={{ x: 0, y: 0 }}
                                        end={{ x: 1, y: 0 }}
                                        style={styles.addInspoGradient}
                                    >
                                        <Ionicons name="add" size={20} color="#FFF" />
                                        <Text style={styles.addInspoText}>Add inspo</Text>
                                    </LinearGradient>
                                </TouchableOpacity>
                            </FrostedGlassCard>
                        ) : (
                            <BentoGrid columns={2} gap={spacing.md} padding={0}>
                                {savedInspo.map((item, index) => (
                                    <ProductCard
                                        key={index}
                                        item={item}
                                        index={index}
                                        isSaved={true}
                                        onSave={() => { }}
                                    />
                                ))}
                            </BentoGrid>
                        )
                    )}

                    <View style={{ height: 120 }} />
                </ScrollView>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    safeArea: { flex: 1 },

    // Header
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: spacing.screenPadding,
        paddingVertical: spacing.sm + 2,
        backgroundColor: colors.glass.frosted,
        borderBottomWidth: 0.5,
        borderBottomColor: colors.border.subtle,
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },
    headerButton: {
        width: spacing.touchTarget.minimum,
        height: spacing.touchTarget.minimum,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Tabs
    tabsContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: spacing.xl,
        paddingVertical: spacing.md,
        backgroundColor: colors.background.primary,
    },
    tabButton: {
        paddingVertical: spacing.xs,
        paddingHorizontal: spacing.xs,
        position: 'relative',
    },
    tabButtonActive: {},
    tabText: {
        ...typography.scale.bodyMedium,
        color: colors.text.tertiary,
    },
    tabTextActive: {
        color: colors.text.primary,
        fontWeight: '600',
    },
    tabIndicator: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: 2,
        backgroundColor: colors.accent.primary,
        borderRadius: 1,
    },

    // Content
    scrollContent: {
        paddingHorizontal: spacing.screenPadding,
        paddingTop: spacing.md,
    },

    // Trending Grid
    trendingGrid: {
        flexDirection: 'row',
        gap: spacing.md,
    },
    trendingColumn: {
        flex: 1,
        gap: spacing.md,
    },
    trendingCard: {
        borderRadius: radius.lg,
        overflow: 'hidden',
        backgroundColor: colors.background.secondary,
        minHeight: 180,
    },
    trendingImage: {
        width: '100%',
        height: '100%',
    },
    trendingOverlay: {
        position: 'absolute',
        bottom: spacing.sm,
        right: spacing.sm,
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.xs,
        paddingHorizontal: spacing.sm,
        paddingVertical: spacing.xs,
        borderRadius: radius.pill,
        overflow: 'hidden',
    },
    trendingLikes: {
        ...typography.scale.labelMedium,
        color: '#FFF',
        fontWeight: '600',
    },

    // Trending Info Card
    trendingInfoCard: {
        marginTop: spacing.lg,
    },
    updatedBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.xs,
        marginBottom: spacing.sm,
    },
    updatedText: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
    },
    liveDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: colors.accent.warning,
    },
    trendingTitle: {
        ...typography.scale.headlineMedium,
        color: colors.text.primary,
        marginBottom: spacing.xs,
    },
    trendingSubtitle: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        marginBottom: spacing.md,
    },
    discoverButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: spacing.xs,
        paddingTop: spacing.md,
        borderTopWidth: 0.5,
        borderTopColor: colors.border.subtle,
    },
    discoverText: {
        ...typography.scale.bodyMedium,
        color: colors.accent.primary,
        fontWeight: '600',
    },

    // Shopping
    shopHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: spacing.md,
    },
    sectionTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
    },
    seeAllText: {
        ...typography.scale.bodyMedium,
        color: colors.accent.primary,
        fontWeight: '600',
    },

    // Product Card
    productCard: {
        flex: 1,
    },
    productImageBox: {
        width: '100%',
        aspectRatio: 0.85,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        overflow: 'hidden',
        marginBottom: spacing.sm,
    },
    productImage: {
        width: '100%',
        height: '100%',
    },
    discountBadge: {
        position: 'absolute',
        top: spacing.sm,
        left: spacing.sm,
        backgroundColor: colors.accent.error,
        paddingHorizontal: spacing.sm,
        paddingVertical: spacing.xs,
        borderRadius: radius.sm,
    },
    discountText: {
        ...typography.scale.labelSmall,
        color: '#FFF',
        fontWeight: '700',
    },
    saveButton: {
        position: 'absolute',
        top: spacing.sm,
        right: spacing.sm,
        borderRadius: radius.full,
        overflow: 'hidden',
    },
    saveButtonBlur: {
        width: 32,
        height: 32,
        alignItems: 'center',
        justifyContent: 'center',
    },
    productInfo: {
        paddingHorizontal: spacing.xs,
    },
    productBrand: {
        ...typography.scale.labelSmall,
        color: colors.text.primary,
        marginBottom: spacing.xs,
    },
    productName: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        marginBottom: spacing.xs,
        lineHeight: 16,
    },
    priceRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.xs,
    },
    productPrice: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
    },
    originalPrice: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
        textDecorationLine: 'line-through',
    },

    // Empty State
    emptyState: {
        marginTop: spacing.xxl,
    },
    emptyContent: {
        alignItems: 'center',
        paddingVertical: spacing.xxl,
    },
    emptyIconContainer: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: colors.glass.frosted,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: spacing.lg,
    },
    emptyTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        marginBottom: spacing.sm,
    },
    emptySubtitle: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        textAlign: 'center',
        marginBottom: spacing.xl,
        paddingHorizontal: spacing.lg,
    },
    addInspoButton: {
        borderRadius: radius.pill,
        overflow: 'hidden',
        ...LiquidGlass2026Theme.elevation.getShadow(6),
    },
    addInspoGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: spacing.sm,
        paddingVertical: spacing.md,
        paddingHorizontal: spacing.xxl,
    },
    addInspoText: {
        ...typography.scale.titleMedium,
        color: '#FFF',
        fontWeight: '600',
    },
});

export default InspoScreen;
