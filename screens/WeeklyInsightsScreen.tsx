/**
 * WeeklyInsightsScreen — "Your week in style"
 *
 * Shows behavioral analytics from the retention service:
 *   - Closet utilization %
 *   - Streak progress
 *   - Color patterns
 *   - Unworn items nudge
 *   - Day-of-week habits
 *
 * Triggered by Sunday push notification or accessible from Profile.
 */

import React, { useMemo } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    SafeAreaView,
    TouchableOpacity,
    Image,
    Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import {
    calculateStreak,
    getClosetUtilization,
    getUnwornItems,
    generateStyleInsights,
} from '../src/services/retentionService';
import type { ClothingItem } from '../src/types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;
const SCREEN_WIDTH = Dimensions.get('window').width;

// ============================================
// COMPONENT
// ============================================

const WeeklyInsightsScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
    const items = useWardrobeStore((state) => state.items);
    const wearLogs = useWardrobeStore((state) => state.wearLogs);

    const streak = useMemo(() => calculateStreak(wearLogs), [wearLogs]);
    const utilization = useMemo(() => getClosetUtilization(items, wearLogs, 30), [items, wearLogs]);
    const unwornItems = useMemo(() => getUnwornItems(items, wearLogs, 30), [items, wearLogs]);
    const insights = useMemo(() => generateStyleInsights(items, wearLogs), [items, wearLogs]);

    // Most worn items
    const mostWorn = useMemo(() => {
        return [...items]
            .filter((i) => i.wearCount > 0)
            .sort((a, b) => b.wearCount - a.wearCount)
            .slice(0, 5);
    }, [items]);

    // Cost per wear (estimated)
    const totalWears = useMemo(() => {
        return items.reduce((sum, i) => sum + i.wearCount, 0);
    }, [items]);

    // Category breakdown
    const categoryBreakdown = useMemo(() => {
        const counts: Record<string, number> = {};
        items.forEach((item) => {
            counts[item.category] = (counts[item.category] || 0) + 1;
        });
        return Object.entries(counts).sort((a, b) => b[1] - a[1]);
    }, [items]);

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => navigation.goBack()}>
                        <Ionicons name="chevron-back" size={28} color={colors.text.primary} />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>Style Insights</Text>
                    <View style={{ width: 28 }} />
                </View>

                {/* Hero Stats */}
                <Animated.View entering={FadeInDown.delay(100).duration(400)} style={styles.heroRow}>
                    <View style={styles.heroCard}>
                        <Text style={styles.heroEmoji}>👗</Text>
                        <Text style={styles.heroValue}>{items.length}</Text>
                        <Text style={styles.heroLabel}>Total Items</Text>
                    </View>
                    <View style={styles.heroCard}>
                        <Text style={styles.heroEmoji}>🔥</Text>
                        <Text style={styles.heroValue}>{streak}</Text>
                        <Text style={styles.heroLabel}>Day Streak</Text>
                    </View>
                    <View style={styles.heroCard}>
                        <Text style={styles.heroEmoji}>📊</Text>
                        <Text style={styles.heroValue}>{totalWears}</Text>
                        <Text style={styles.heroLabel}>Total Wears</Text>
                    </View>
                </Animated.View>

                {/* Closet Utilization */}
                <Animated.View entering={FadeInDown.delay(200).duration(400)} style={styles.card}>
                    <View style={styles.cardHeader}>
                        <View style={[styles.iconBadge, { backgroundColor: 'rgba(59,130,246,0.12)' }]}>
                            <Ionicons name="pie-chart-outline" size={20} color="#3B82F6" />
                        </View>
                        <Text style={styles.cardTitle}>Closet Utilization</Text>
                    </View>
                    <Text style={styles.bigStat}>{utilization}%</Text>
                    <View style={styles.progressTrack}>
                        <View style={[styles.progressFill, { width: `${utilization}%` }]} />
                    </View>
                    <Text style={styles.cardSubtext}>
                        {utilization >= 80
                            ? 'Excellent! You\'re making great use of your wardrobe.'
                            : utilization >= 50
                                ? `You've worn ${utilization}% of your closet this month. Try mixing in forgotten pieces.`
                                : `Only ${utilization}% of your closet was worn this month. There's a lot to rediscover!`}
                    </Text>
                </Animated.View>

                {/* Unworn Items */}
                {unwornItems.length > 0 && (
                    <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.card}>
                        <View style={styles.cardHeader}>
                            <View style={[styles.iconBadge, { backgroundColor: 'rgba(249,115,22,0.12)' }]}>
                                <Ionicons name="sparkles-outline" size={20} color="#F97316" />
                            </View>
                            <Text style={styles.cardTitle}>
                                {unwornItems.length} Unworn Items
                            </Text>
                        </View>
                        <Text style={styles.cardSubtext}>
                            These items haven't been worn in 30 days. Try one this week!
                        </Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.unwornScroll}>
                            {unwornItems.slice(0, 8).map((item) => (
                                <UnwornItemCard key={item.id} item={item} />
                            ))}
                        </ScrollView>
                    </Animated.View>
                )}

                {/* Most Worn */}
                {mostWorn.length > 0 && (
                    <Animated.View entering={FadeInDown.delay(400).duration(400)} style={styles.card}>
                        <View style={styles.cardHeader}>
                            <View style={[styles.iconBadge, { backgroundColor: 'rgba(16,185,129,0.12)' }]}>
                                <Ionicons name="trending-up-outline" size={20} color="#10B981" />
                            </View>
                            <Text style={styles.cardTitle}>Most Worn</Text>
                        </View>
                        {mostWorn.map((item, index) => (
                            <View key={item.id} style={styles.rankRow}>
                                <Text style={styles.rankNumber}>#{index + 1}</Text>
                                {item.imageUrl ? (
                                    <Image
                                        source={{ uri: item.imageUrl }}
                                        style={styles.rankImage}
                                        resizeMode="cover"
                                    />
                                ) : (
                                    <View style={[styles.rankImage, styles.rankPlaceholder]}>
                                        <View style={[styles.rankColor, { backgroundColor: item.colorHex || '#CCC' }]} />
                                    </View>
                                )}
                                <View style={styles.rankInfo}>
                                    <Text style={styles.rankName} numberOfLines={1}>
                                        {item.subCategory || item.category}
                                    </Text>
                                    <Text style={styles.rankMeta}>{item.primaryColor}</Text>
                                </View>
                                <Text style={styles.rankCount}>{item.wearCount}×</Text>
                            </View>
                        ))}
                    </Animated.View>
                )}

                {/* Category Breakdown */}
                {categoryBreakdown.length > 0 && (
                    <Animated.View entering={FadeInDown.delay(500).duration(400)} style={styles.card}>
                        <View style={styles.cardHeader}>
                            <View style={[styles.iconBadge, { backgroundColor: 'rgba(139,92,246,0.12)' }]}>
                                <Ionicons name="grid-outline" size={20} color="#8B5CF6" />
                            </View>
                            <Text style={styles.cardTitle}>Wardrobe Breakdown</Text>
                        </View>
                        {categoryBreakdown.map(([category, count]) => {
                            const percentage = Math.round((count / items.length) * 100);
                            return (
                                <View key={category} style={styles.breakdownRow}>
                                    <Text style={styles.breakdownLabel}>
                                        {category.charAt(0).toUpperCase() + category.slice(1)}
                                    </Text>
                                    <View style={styles.breakdownTrack}>
                                        <View
                                            style={[
                                                styles.breakdownFill,
                                                { width: `${percentage}%` },
                                            ]}
                                        />
                                    </View>
                                    <Text style={styles.breakdownCount}>{count}</Text>
                                </View>
                            );
                        })}
                    </Animated.View>
                )}

                {/* Style Insights from retention service */}
                {insights.length > 0 && (
                    <Animated.View entering={FadeInDown.delay(600).duration(400)} style={styles.card}>
                        <View style={styles.cardHeader}>
                            <View style={[styles.iconBadge, { backgroundColor: 'rgba(239,68,68,0.12)' }]}>
                                <Ionicons name="bulb-outline" size={20} color="#EF4444" />
                            </View>
                            <Text style={styles.cardTitle}>Patterns & Habits</Text>
                        </View>
                        {insights
                            .filter((i) => i.type !== 'utilization' && i.type !== 'streak')
                            .slice(0, 4)
                            .map((insight, index) => (
                                <View key={`${insight.type}_${index}`} style={styles.insightRow}>
                                    <Text style={styles.insightTitle}>{insight.title}</Text>
                                    <Text style={styles.insightDesc}>{insight.description}</Text>
                                </View>
                            ))}
                    </Animated.View>
                )}

                {/* Empty state */}
                {items.length === 0 && (
                    <View style={styles.emptyState}>
                        <Ionicons name="analytics-outline" size={56} color={colors.text.tertiary} />
                        <Text style={styles.emptyTitle}>No data yet</Text>
                        <Text style={styles.emptySubtext}>
                            Start scanning and logging outfits to see your style insights
                        </Text>
                    </View>
                )}
            </ScrollView>
        </SafeAreaView>
    );
};

// ============================================
// SUB-COMPONENT
// ============================================

const UnwornItemCard: React.FC<{ item: ClothingItem }> = ({ item }) => (
    <View style={styles.unwornCard}>
        {item.imageUrl ? (
            <Image
                source={{ uri: item.imageUrl }}
                style={styles.unwornImage}
                resizeMode="cover"
            />
        ) : (
            <View style={[styles.unwornImage, styles.unwornPlaceholder]}>
                <View style={[styles.unwornColorDot, { backgroundColor: item.colorHex || '#CCC' }]} />
            </View>
        )}
        <Text style={styles.unwornLabel} numberOfLines={1}>
            {item.subCategory || item.category}
        </Text>
    </View>
);

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    scrollContent: {
        paddingBottom: spacing.xxl,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.lg,
        paddingTop: spacing.md,
        paddingBottom: spacing.sm,
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },

    // Hero
    heroRow: {
        flexDirection: 'row',
        paddingHorizontal: spacing.lg,
        gap: spacing.sm,
        marginTop: spacing.md,
    },
    heroCard: {
        flex: 1,
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.lg,
        padding: spacing.md,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: 4,
    },
    heroEmoji: { fontSize: 24 },
    heroValue: {
        ...typography.scale.headlineMedium,
        color: colors.text.primary,
        fontWeight: '800',
    },
    heroLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
    },

    // Cards
    card: {
        marginHorizontal: spacing.lg,
        marginTop: spacing.lg,
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.xl,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.md,
    },
    cardHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.sm,
    },
    iconBadge: {
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
    },
    cardTitle: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '700',
    },
    cardSubtext: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        lineHeight: 18,
    },

    // Big stat
    bigStat: {
        ...typography.scale.displayLarge,
        color: colors.text.primary,
        fontWeight: '800',
        textAlign: 'center',
    },

    // Progress
    progressTrack: {
        height: 8,
        backgroundColor: colors.background.secondary,
        borderRadius: 4,
        overflow: 'hidden',
    },
    progressFill: {
        height: '100%',
        backgroundColor: '#3B82F6',
        borderRadius: 4,
    },

    // Unworn
    unwornScroll: {
        marginHorizontal: -spacing.sm,
    },
    unwornCard: {
        width: 80,
        alignItems: 'center',
        marginHorizontal: spacing.xs,
        gap: spacing.xs,
    },
    unwornImage: {
        width: 64,
        height: 64,
        borderRadius: radius.md,
    },
    unwornPlaceholder: {
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    unwornColorDot: {
        width: 24,
        height: 24,
        borderRadius: 12,
    },
    unwornLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        textAlign: 'center',
    },

    // Most worn
    rankRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.sm,
    },
    rankNumber: {
        ...typography.scale.labelMedium,
        color: colors.text.tertiary,
        fontWeight: '700',
        width: 24,
    },
    rankImage: {
        width: 40,
        height: 40,
        borderRadius: radius.sm,
    },
    rankPlaceholder: {
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    rankColor: {
        width: 20,
        height: 20,
        borderRadius: 10,
    },
    rankInfo: {
        flex: 1,
    },
    rankName: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '600',
    },
    rankMeta: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
    },
    rankCount: {
        ...typography.scale.titleSmall,
        color: colors.text.secondary,
        fontWeight: '700',
    },

    // Category breakdown
    breakdownRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.sm,
    },
    breakdownLabel: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
        width: 80,
    },
    breakdownTrack: {
        flex: 1,
        height: 6,
        backgroundColor: colors.background.secondary,
        borderRadius: 3,
        overflow: 'hidden',
    },
    breakdownFill: {
        height: '100%',
        backgroundColor: '#8B5CF6',
        borderRadius: 3,
    },
    breakdownCount: {
        ...typography.scale.labelMedium,
        color: colors.text.tertiary,
        width: 28,
        textAlign: 'right',
    },

    // Insights
    insightRow: {
        gap: 2,
    },
    insightTitle: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '600',
    },
    insightDesc: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        lineHeight: 18,
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
    emptySubtext: {
        ...typography.scale.bodyMedium,
        color: colors.text.tertiary,
        textAlign: 'center',
        paddingHorizontal: spacing.xl,
    },
});

export default WeeklyInsightsScreen;
