/**
 * WardrobeAnalyticsScreen — Full wardrobe stats dashboard
 *
 * Features:
 *   - Closet utilization score (hero metric)
 *   - Most worn items (top 5)
 *   - Color palette distribution
 *   - Category breakdown (horizontal bar chart)
 *   - Unworn items nudge
 *   - Diversity score
 *
 * Uses existing store methods + new diversityEngine service.
 */

import React, { useMemo } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    Dimensions,
    Image,
    TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { BlurView } from 'expo-blur';
import Animated, {
    FadeInDown,
    FadeInRight,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';
import FeatureLockOverlay from '../components/paywall/FeatureLockOverlay';
import { scoreDiversity, getColorDistribution, getCategoryBreakdown } from '../src/services/diversityEngine';
import type { ColorDistEntry, CategoryBreakdownEntry } from '../src/services/diversityEngine';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// ============================================
// SUB-COMPONENTS
// ============================================

/** Circular progress indicator for utilization score */
const CircularProgress = ({ percentage, size = 120 }: { percentage: number; size?: number }) => {
    const strokeWidth = 10;
    const r = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * r;
    const fillLength = (percentage / 100) * circumference;

    const getColor = (pct: number) => {
        if (pct >= 70) return '#34D399'; // green
        if (pct >= 40) return '#FBBF24'; // amber
        return '#F87171'; // red
    };

    return (
        <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
            <View style={{
                width: size,
                height: size,
                borderRadius: size / 2,
                borderWidth: strokeWidth,
                borderColor: colors.border.glass,
                position: 'absolute',
            }} />
            <View style={{
                width: size,
                height: size,
                borderRadius: size / 2,
                borderWidth: strokeWidth,
                borderColor: getColor(percentage),
                borderTopColor: percentage >= 25 ? getColor(percentage) : 'transparent',
                borderRightColor: percentage >= 50 ? getColor(percentage) : 'transparent',
                borderBottomColor: percentage >= 75 ? getColor(percentage) : 'transparent',
                borderLeftColor: percentage < 100 ? 'transparent' : getColor(percentage),
                position: 'absolute',
                transform: [{ rotate: '-90deg' }],
            }} />
            <Text style={styles.circularValue}>{percentage}%</Text>
            <Text style={styles.circularLabel}>Utilized</Text>
        </View>
    );
};

/** Horizontal bar for category breakdown */
const CategoryBar = ({ label, count, total, color }: {
    label: string; count: number; total: number; color: string;
}) => {
    const pct = total > 0 ? (count / total) * 100 : 0;
    return (
        <View style={styles.categoryBarContainer}>
            <View style={styles.categoryBarLabelRow}>
                <Text style={styles.categoryBarLabel}>{label}</Text>
                <Text style={styles.categoryBarCount}>{count}</Text>
            </View>
            <View style={styles.categoryBarTrack}>
                <Animated.View
                    entering={FadeInRight.duration(600).delay(200)}
                    style={[styles.categoryBarFill, { width: `${Math.max(pct, 2)}%`, backgroundColor: color }]}
                />
            </View>
        </View>
    );
};

/** Color swatch in palette */
const ColorSwatch = ({ color, name, count, total }: {
    color: string; name: string; count: number; total: number;
}) => {
    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
    return (
        <View style={styles.swatchContainer}>
            <View style={[styles.swatchCircle, { backgroundColor: color }]} />
            <Text style={styles.swatchName} numberOfLines={1}>{name}</Text>
            <Text style={styles.swatchPct}>{pct}%</Text>
        </View>
    );
};

// ============================================
// MAIN COMPONENT
// ============================================

export default function WardrobeAnalyticsScreen({ navigation }: any) {
    const { canAccess } = useSubscriptionGate();
    const hasAccess = canAccess('analytics');

    const items = useWardrobeStore((s) => s.items);
    const wearLogs = useWardrobeStore((s) => s.wearLogs);

    // Compute derived values from store without creating infinite loops
    // by using the existing state snapshot
    const utilization = useMemo(() => useWardrobeStore.getState().getClosetUtilization(30), [items, wearLogs]);
    const unwornItems = useMemo(() => useWardrobeStore.getState().getUnwornItems(30), [items, wearLogs]);
    const streak = useMemo(() => useWardrobeStore.getState().getStreak(), [wearLogs]);

    // Derived analytics
    const analytics = useMemo(() => {
        const diversity = scoreDiversity(items, wearLogs);
        const colorDist = getColorDistribution(items);
        const categoryBreakdown = getCategoryBreakdown(items);

        // Most worn (top 5)
        const mostWorn = [...items]
            .sort((a, b) => b.wearCount - a.wearCount)
            .filter(i => i.wearCount > 0)
            .slice(0, 5);

        // Total wears
        const totalWears = wearLogs.length;

        // Avg wears per item
        const avgWears = items.length > 0
            ? Math.round((items.reduce((sum, i) => sum + i.wearCount, 0) / items.length) * 10) / 10
            : 0;

        return { diversity, colorDist, categoryBreakdown, mostWorn, totalWears, avgWears };
    }, [items, wearLogs]);

    // Analytics is gated for Free users. We render a tease overlay instead of
    // navigating away — seeing "locked, but your data is waiting" drives
    // meaningfully higher upgrade rates than a silent redirect.
    if (!hasAccess) {
        return (
            <FeatureLockOverlay
                requiredTier="Pro"
                featureName="Wardrobe Insights"
                tagline="See which pieces you actually wear and what's just sitting in your closet."
                icon="bar-chart"
                bullets={[
                    'Your real closet utilization score',
                    'Most-worn items and hidden gems',
                    'Color palette and category breakdown',
                    'Unworn items nudge so nothing goes to waste',
                ]}
            />
        );
    }

    const CATEGORY_COLORS: Record<string, string> = {
        top: '#60A5FA',
        bottom: '#A78BFA',
        shoes: '#F472B6',
        outerwear: '#34D399',
        accessory: '#FBBF24',
    };

    return (
        <SafeAreaView style={styles.container}>
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
                pointerEvents="none"
            />
            <View pointerEvents="none" style={styles.backgroundOrbTop} />
            <View pointerEvents="none" style={styles.backgroundOrbBottom} />
            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Wardrobe Analytics</Text>
                <View style={{ width: 32 }} />
            </View>

            <ScrollView
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
            >
                {/* Hero Stats Row */}
                <Animated.View entering={FadeInDown.duration(500)} style={styles.heroRow}>
                    <View style={styles.heroCard}>
                        <CircularProgress percentage={utilization} />
                    </View>
                    <View style={styles.heroStatsColumn}>
                        <View style={styles.heroStat}>
                            <Text style={styles.heroStatValue}>{items.length}</Text>
                            <Text style={styles.heroStatLabel}>Total Items</Text>
                        </View>
                        <View style={styles.heroStat}>
                            <Text style={styles.heroStatValue}>{analytics.totalWears}</Text>
                            <Text style={styles.heroStatLabel}>Total Wears</Text>
                        </View>
                        <View style={styles.heroStat}>
                            <Text style={styles.heroStatValue}>{analytics.avgWears}</Text>
                            <Text style={styles.heroStatLabel}>Avg/Item</Text>
                        </View>
                        <View style={styles.heroStat}>
                            <Text style={styles.heroStatValue}>🔥 {streak}</Text>
                            <Text style={styles.heroStatLabel}>Day Streak</Text>
                        </View>
                    </View>
                </Animated.View>

                {/* Diversity Score */}
                <Animated.View entering={FadeInDown.duration(500).delay(100)} style={styles.card}>
                    <View style={styles.cardHeader}>
                        <Text style={styles.cardTitle}>Style Diversity</Text>
                        <View style={[styles.scoreBadge, {
                            backgroundColor: analytics.diversity >= 70 ? '#DCFCE7' :
                                analytics.diversity >= 40 ? '#FEF9C3' : '#FEE2E2'
                        }]}>
                            <Text style={[styles.scoreBadgeText, {
                                color: analytics.diversity >= 70 ? '#166534' :
                                    analytics.diversity >= 40 ? '#854D0E' : '#991B1B'
                            }]}>
                                {analytics.diversity}/100
                            </Text>
                        </View>
                    </View>
                    <Text style={styles.cardSubtitle}>
                        {analytics.diversity >= 70
                            ? 'Great variety! You use your wardrobe well.'
                            : analytics.diversity >= 40
                                ? 'Good start. Try mixing in some unworn items!'
                                : 'You tend to repeat the same items. Try the Surprise Me feature!'}
                    </Text>
                    <View style={styles.diversityBar}>
                        <Animated.View
                            entering={FadeInRight.duration(800).delay(300)}
                            style={[styles.diversityFill, { width: `${analytics.diversity}%` }]}
                        />
                    </View>
                </Animated.View>

                {/* Most Worn Items */}
                {analytics.mostWorn.length > 0 ? (
                    <Animated.View entering={FadeInDown.duration(500).delay(200)} style={styles.card}>
                        <Text style={styles.cardTitle}>Most Worn</Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.mostWornScroll}>
                            {analytics.mostWorn.map((item, idx) => (
                                <View key={item.id} style={styles.mostWornItem}>
                                    <View style={styles.mostWornRank}>
                                        <Text style={styles.mostWornRankText}>#{idx + 1}</Text>
                                    </View>
                                    <Image
                                        source={{ uri: item.imageUrl || item.thumbnailUrl }}
                                        style={styles.mostWornImage}
                                    />
                                    <Text style={styles.mostWornCount}>{item.wearCount}×</Text>
                                    <Text style={styles.mostWornName} numberOfLines={1}>
                                        {item.name || item.subCategory || item.category}
                                    </Text>
                                </View>
                            ))}
                        </ScrollView>
                    </Animated.View>
                ) : null}

                {/* Color Palette */}
                {analytics.colorDist.length > 0 ? (
                    <Animated.View entering={FadeInDown.duration(500).delay(300)} style={styles.card}>
                        <Text style={styles.cardTitle}>Your Color Palette</Text>
                        <View style={styles.swatchRow}>
                            {analytics.colorDist.slice(0, 8).map(({ color, name, count }: ColorDistEntry) => (
                                <ColorSwatch
                                    key={name}
                                    color={color}
                                    name={name}
                                    count={count}
                                    total={items.length}
                                />
                            ))}
                        </View>
                    </Animated.View>
                ) : null}

                {/* Category Breakdown */}
                <Animated.View entering={FadeInDown.duration(500).delay(400)} style={styles.card}>
                    <Text style={styles.cardTitle}>Category Breakdown</Text>
                    {analytics.categoryBreakdown.map(({ category, count }: CategoryBreakdownEntry) => (
                        <CategoryBar
                            key={category}
                            label={category.charAt(0).toUpperCase() + category.slice(1)}
                            count={count}
                            total={items.length}
                            color={CATEGORY_COLORS[category] || '#94A3B8'}
                        />
                    ))}
                </Animated.View>

                {/* Unworn Alert */}
                {unwornItems.length > 0 ? (
                    <Animated.View entering={FadeInDown.duration(500).delay(500)} style={styles.unwornCard}>
                        <View style={styles.unwornHeader}>
                            <Ionicons name="alert-circle" size={20} color="#F59E0B" />
                            <Text style={styles.unwornTitle}>
                                {unwornItems.length} items unworn in 30 days
                            </Text>
                        </View>
                        <Text style={styles.unwornSubtitle}>
                            That's {Math.round((unwornItems.length / items.length) * 100)}% of your wardrobe sitting idle.
                        </Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.unwornScroll}>
                            {unwornItems.slice(0, 10).map((item) => (
                                <Image
                                    key={item.id}
                                    source={{ uri: item.imageUrl || item.thumbnailUrl }}
                                    style={styles.unwornImage}
                                />
                            ))}
                        </ScrollView>
                    </Animated.View>
                ) : null}

                {/* Empty state */}
                {items.length === 0 ? (
                    <View style={styles.emptyState}>
                        <Ionicons name="analytics-outline" size={64} color={colors.text.tertiary} />
                        <Text style={styles.emptyTitle}>No Data Yet</Text>
                        <Text style={styles.emptySubtitle}>
                            Add items to your wardrobe and log what you wear to see analytics.
                        </Text>
                    </View>
                ) : null}
            </ScrollView>
        </SafeAreaView>
    );
}

// ============================================
// STYLES
// ============================================

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
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.lg,
        paddingVertical: spacing.md,
    },
    backButton: {
        width: 32,
        height: 32,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.84)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },
    scrollContent: {
        paddingBottom: spacing.xxxl,
    },

    // Hero
    heroRow: {
        flexDirection: 'row',
        marginHorizontal: spacing.lg,
        marginBottom: spacing.lg,
        gap: spacing.lg,
    },
    heroCard: {
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderRadius: 28,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
    },
    heroStatsColumn: {
        flex: 1,
        gap: spacing.sm,
        justifyContent: 'center',
    },
    heroStat: {
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderRadius: 22,
        paddingVertical: spacing.sm,
        paddingHorizontal: spacing.md,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
    },
    heroStatValue: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '800',
    },
    heroStatLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
    },

    // Circular
    circularValue: {
        ...typography.scale.headlineMedium,
        color: colors.text.primary,
        fontWeight: '800',
    },
    circularLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        marginTop: -2,
    },

    // Card
    card: {
        marginHorizontal: spacing.lg,
        marginBottom: spacing.lg,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderRadius: 28,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 16,
        elevation: 4,
    },
    cardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: spacing.sm,
    },
    cardTitle: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '700',
        marginBottom: spacing.sm,
    },
    cardSubtitle: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
        marginBottom: spacing.md,
        lineHeight: 18,
    },

    // Score badge
    scoreBadge: {
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.xs,
        borderRadius: radius.pill,
    },
    scoreBadgeText: {
        ...typography.scale.labelMedium,
        fontWeight: '700',
    },

    // Diversity bar
    diversityBar: {
        height: 8,
        backgroundColor: colors.border.glass,
        borderRadius: 4,
        overflow: 'hidden',
    },
    diversityFill: {
        height: '100%',
        borderRadius: 4,
        backgroundColor: '#3B82F6',
    },

    // Most worn
    mostWornScroll: {
        marginHorizontal: -spacing.sm,
    },
    mostWornItem: {
        alignItems: 'center',
        marginHorizontal: spacing.sm,
        width: 72,
    },
    mostWornRank: {
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: 1,
        backgroundColor: 'rgba(0,0,0,0.7)',
        borderRadius: 8,
        paddingHorizontal: 4,
        paddingVertical: 1,
    },
    mostWornRankText: {
        fontSize: 10,
        color: '#FFF',
        fontWeight: '700',
    },
    mostWornImage: {
        width: 64,
        height: 64,
        borderRadius: radius.md,
        backgroundColor: colors.border.glass,
    },
    mostWornCount: {
        ...typography.scale.labelSmall,
        color: colors.text.primary,
        fontWeight: '700',
        marginTop: 4,
    },
    mostWornName: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        textAlign: 'center',
    },

    // Color swatches
    swatchRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.md,
    },
    swatchContainer: {
        alignItems: 'center',
        width: 56,
    },
    swatchCircle: {
        width: 32,
        height: 32,
        borderRadius: 16,
        borderWidth: 2,
        borderColor: 'rgba(255,255,255,0.5)',
        marginBottom: 4,
    },
    swatchName: {
        ...typography.scale.labelSmall,
        color: colors.text.secondary,
        fontSize: 9,
        textAlign: 'center',
    },
    swatchPct: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        fontSize: 9,
    },

    // Category bars
    categoryBarContainer: {
        marginBottom: spacing.sm,
    },
    categoryBarLabelRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 4,
    },
    categoryBarLabel: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
    },
    categoryBarCount: {
        ...typography.scale.labelMedium,
        color: colors.text.primary,
        fontWeight: '600',
    },
    categoryBarTrack: {
        height: 8,
        backgroundColor: colors.border.glass,
        borderRadius: 4,
        overflow: 'hidden',
    },
    categoryBarFill: {
        height: '100%',
        borderRadius: 4,
    },

    // Unworn
    unwornCard: {
        marginHorizontal: spacing.lg,
        marginBottom: spacing.lg,
        backgroundColor: 'rgba(245, 158, 11, 0.08)',
        borderRadius: radius.xl,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: 'rgba(245, 158, 11, 0.2)',
    },
    unwornHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.sm,
        marginBottom: spacing.xs,
    },
    unwornTitle: {
        ...typography.scale.titleSmall,
        color: '#D97706',
        fontWeight: '700',
    },
    unwornSubtitle: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
        marginBottom: spacing.md,
    },
    unwornScroll: {
        marginHorizontal: -spacing.xs,
    },
    unwornImage: {
        width: 56,
        height: 56,
        borderRadius: radius.md,
        marginHorizontal: spacing.xs,
        backgroundColor: colors.border.glass,
    },

    // Empty
    emptyState: {
        alignItems: 'center',
        paddingVertical: spacing.xxxl,
        gap: spacing.md,
    },
    emptyTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.secondary,
        fontWeight: '600',
    },
    emptySubtitle: {
        ...typography.scale.bodyMedium,
        color: colors.text.tertiary,
        textAlign: 'center',
        paddingHorizontal: spacing.xl,
    },
});
