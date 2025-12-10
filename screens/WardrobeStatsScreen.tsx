import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    RefreshControl,
    Dimensions,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
    FadeInDown,
    FadeInUp,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Header, Card, LoadingState, EmptyState } from '../components/ui';
import { colors, spacing } from '../src/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const API_URL = 'https://aiwardrobe-ivh4.onrender.com';

interface WardrobeStats {
    totalItems: number;
    totalValue: number;
    avgWearCount: number;
    neverWorn: number;
    favorites: number;
    byCategory: Record<string, number>;
    topColors: Array<{ color: string; count: number }>;
}

interface ClothingItemBasic {
    _id: string;
    type: string;
    color: string[];
    imageUrl: string;
    wearCount: number;
    costPerWear: number;
    price: number;
    lastWorn?: string;
}

const StatBox: React.FC<{
    label: string;
    value: string | number;
    icon: keyof typeof Ionicons.glyphMap;
    color: string;
    index: number;
}> = ({ label, value, icon, color, index }) => (
    <Animated.View
        entering={FadeInDown.delay(index * 100).springify()}
        style={styles.statBox}
    >
        <View style={[styles.statIconContainer, { backgroundColor: `${color}20` }]}>
            <Ionicons name={icon} size={24} color={color} />
        </View>
        <Text style={styles.statValue}>{value}</Text>
        <Text style={styles.statLabel}>{label}</Text>
    </Animated.View>
);

const CategoryBar: React.FC<{
    category: string;
    count: number;
    total: number;
    index: number;
}> = ({ category, count, total, index }) => {
    const percentage = total > 0 ? (count / total) * 100 : 0;

    return (
        <Animated.View
            entering={FadeInDown.delay(300 + index * 50).springify()}
            style={styles.categoryRow}
        >
            <View style={styles.categoryInfo}>
                <Text style={styles.categoryName}>{category}</Text>
                <Text style={styles.categoryCount}>{count} items</Text>
            </View>
            <View style={styles.categoryBarContainer}>
                <Animated.View
                    style={[
                        styles.categoryBar,
                        { width: `${percentage}%` },
                    ]}
                />
            </View>
        </Animated.View>
    );
};

const ColorDot: React.FC<{ color: string; count: number }> = ({ color, count }) => {
    // Map color names to actual colors
    const colorMap: Record<string, string> = {
        black: '#000000',
        white: '#FFFFFF',
        blue: '#3B82F6',
        red: '#EF4444',
        green: '#22C55E',
        yellow: '#EAB308',
        pink: '#EC4899',
        purple: '#A855F7',
        orange: '#F97316',
        brown: '#A16207',
        gray: '#6B7280',
        grey: '#6B7280',
        navy: '#1E3A5A',
        beige: '#F5F5DC',
    };

    const bgColor = colorMap[color.toLowerCase()] || '#9CA3AF';

    return (
        <View style={styles.colorDotContainer}>
            <View style={[styles.colorDot, { backgroundColor: bgColor }]} />
            <Text style={styles.colorName}>{color}</Text>
            <Text style={styles.colorCount}>{count}</Text>
        </View>
    );
};

export default function WardrobeStatsScreen() {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const insets = useSafeAreaInsets();

    const [stats, setStats] = useState<WardrobeStats | null>(null);
    const [mostWorn, setMostWorn] = useState<ClothingItemBasic[]>([]);
    const [leastWorn, setLeastWorn] = useState<ClothingItemBasic[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchStats = useCallback(async () => {
        try {
            const token = await AsyncStorage.getItem('token');
            if (!token) {
                setError('Please log in to view statistics');
                return;
            }

            const headers = { Authorization: `Bearer ${token}` };

            const [statsRes, mostWornRes, leastWornRes] = await Promise.all([
                axios.get(`${API_URL}/stats`, { headers }),
                axios.get(`${API_URL}/stats/most-worn?limit=5`, { headers }),
                axios.get(`${API_URL}/stats/least-worn?limit=5`, { headers }),
            ]);

            setStats(statsRes.data.data);
            setMostWorn(mostWornRes.data.data);
            setLeastWorn(leastWornRes.data.data);
            setError(null);
        } catch (err: any) {
            console.error('Stats fetch error:', err);
            setError(err.response?.data?.error || 'Failed to load statistics');
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, []);

    useEffect(() => {
        fetchStats();
    }, [fetchStats]);

    const onRefresh = useCallback(() => {
        setRefreshing(true);
        fetchStats();
    }, [fetchStats]);

    if (loading) {
        return (
            <View style={[styles.container, { paddingTop: insets.top }]}>
                <Header
                    title={t('stats.title', 'Wardrobe Stats')}
                    showBackButton
                    onBackPress={() => navigation.goBack()}
                />
                <LoadingState variant="profile" />
            </View>
        );
    }

    if (error || !stats) {
        return (
            <View style={[styles.container, { paddingTop: insets.top }]}>
                <Header
                    title={t('stats.title', 'Wardrobe Stats')}
                    showBackButton
                    onBackPress={() => navigation.goBack()}
                />
                <EmptyState
                    variant="error"
                    title={t('stats.error', 'Failed to load')}
                    description={error || 'Unable to fetch statistics'}
                    actionLabel={t('common.retry', 'Try Again')}
                    onActionPress={fetchStats}
                />
            </View>
        );
    }

    const totalByCategory = Object.values(stats.byCategory).reduce((a, b) => a + b, 0);

    return (
        <View style={[styles.container, { paddingTop: insets.top }]}>
            <Header
                title={t('stats.title', 'Wardrobe Stats')}
                showBackButton
                onBackPress={() => navigation.goBack()}
            />

            <ScrollView
                style={styles.scrollView}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                }
            >
                {/* Summary Cards */}
                <View style={styles.statsGrid}>
                    <StatBox
                        label={t('stats.totalItems', 'Total Items')}
                        value={stats.totalItems}
                        icon="shirt"
                        color="#3B82F6"
                        index={0}
                    />
                    <StatBox
                        label={t('stats.totalValue', 'Total Value')}
                        value={`$${stats.totalValue.toFixed(0)}`}
                        icon="cash"
                        color="#22C55E"
                        index={1}
                    />
                    <StatBox
                        label={t('stats.avgWears', 'Avg Wears')}
                        value={stats.avgWearCount.toFixed(1)}
                        icon="refresh"
                        color="#A855F7"
                        index={2}
                    />
                    <StatBox
                        label={t('stats.neverWorn', 'Never Worn')}
                        value={stats.neverWorn}
                        icon="alert-circle"
                        color="#EF4444"
                        index={3}
                    />
                </View>

                {/* Category Distribution */}
                <Animated.View entering={FadeInUp.delay(200).springify()}>
                    <Card style={styles.sectionCard}>
                        <Text style={styles.sectionTitle}>
                            {t('stats.byCategory', 'By Category')}
                        </Text>
                        {Object.entries(stats.byCategory).map(([category, count], index) => (
                            <CategoryBar
                                key={category}
                                category={category}
                                count={count}
                                total={totalByCategory}
                                index={index}
                            />
                        ))}
                    </Card>
                </Animated.View>

                {/* Color Distribution */}
                {stats.topColors.length > 0 && (
                    <Animated.View entering={FadeInUp.delay(400).springify()}>
                        <Card style={styles.sectionCard}>
                            <Text style={styles.sectionTitle}>
                                {t('stats.topColors', 'Top Colors')}
                            </Text>
                            <View style={styles.colorsGrid}>
                                {stats.topColors.map((item) => (
                                    <ColorDot
                                        key={item.color}
                                        color={item.color}
                                        count={item.count}
                                    />
                                ))}
                            </View>
                        </Card>
                    </Animated.View>
                )}

                {/* Most Worn */}
                {mostWorn.length > 0 && (
                    <Animated.View entering={FadeInUp.delay(500).springify()}>
                        <Card style={styles.sectionCard}>
                            <Text style={styles.sectionTitle}>
                                {t('stats.mostWorn', 'Most Worn')} 👑
                            </Text>
                            {mostWorn.map((item, index) => (
                                <View key={item._id} style={styles.itemRow}>
                                    <Text style={styles.itemRank}>#{index + 1}</Text>
                                    <Text style={styles.itemName}>{item.type}</Text>
                                    <View style={styles.itemStats}>
                                        <Text style={styles.itemWearCount}>
                                            {item.wearCount} {t('stats.wears', 'wears')}
                                        </Text>
                                        {item.costPerWear > 0 && (
                                            <Text style={styles.itemCostPerWear}>
                                                ${item.costPerWear}/wear
                                            </Text>
                                        )}
                                    </View>
                                </View>
                            ))}
                        </Card>
                    </Animated.View>
                )}

                {/* Least Worn - Sustainability */}
                {leastWorn.length > 0 && (
                    <Animated.View entering={FadeInUp.delay(600).springify()}>
                        <Card style={styles.sectionCard}>
                            <Text style={styles.sectionTitle}>
                                {t('stats.leastWorn', 'Wear More or Donate')} 🌱
                            </Text>
                            <Text style={styles.sectionSubtitle}>
                                {t('stats.sustainabilityTip', 'Wear items at least 30 times for sustainability')}
                            </Text>
                            {leastWorn.map((item, index) => (
                                <View key={item._id} style={styles.itemRow}>
                                    <Text style={styles.itemRank}>#{index + 1}</Text>
                                    <Text style={styles.itemName}>{item.type}</Text>
                                    <Text style={[styles.itemWearCount, styles.lowWearCount]}>
                                        {item.wearCount} {t('stats.wears', 'wears')}
                                    </Text>
                                </View>
                            ))}
                        </Card>
                    </Animated.View>
                )}

                <View style={{ height: spacing.xl * 2 }} />
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background,
    },
    scrollView: {
        flex: 1,
    },
    scrollContent: {
        padding: spacing.m,
    },
    statsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginHorizontal: -spacing.xs,
        marginBottom: spacing.m,
    },
    statBox: {
        width: '50%',
        padding: spacing.xs,
    },
    statIconContainer: {
        width: 48,
        height: 48,
        borderRadius: 24,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: spacing.s,
    },
    statValue: {
        fontSize: 28,
        fontWeight: '700',
        color: colors.text.primary,
    },
    statLabel: {
        fontSize: 14,
        color: colors.text.secondary,
        marginTop: 4,
    },
    sectionCard: {
        marginBottom: spacing.m,
        padding: spacing.m,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: colors.text.primary,
        marginBottom: spacing.m,
    },
    sectionSubtitle: {
        fontSize: 14,
        color: colors.text.secondary,
        marginTop: -spacing.s,
        marginBottom: spacing.m,
    },
    categoryRow: {
        marginBottom: spacing.m,
    },
    categoryInfo: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: spacing.xs,
    },
    categoryName: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.primary,
    },
    categoryCount: {
        fontSize: 14,
        color: colors.text.secondary,
    },
    categoryBarContainer: {
        height: 8,
        backgroundColor: colors.surfaceHighlight,
        borderRadius: 4,
        overflow: 'hidden',
    },
    categoryBar: {
        height: '100%',
        backgroundColor: colors.text.accent,
        borderRadius: 4,
    },
    colorsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
    },
    colorDotContainer: {
        alignItems: 'center',
        width: '20%',
        marginBottom: spacing.m,
    },
    colorDot: {
        width: 32,
        height: 32,
        borderRadius: 16,
        borderWidth: 2,
        borderColor: colors.border,
        marginBottom: spacing.xs,
    },
    colorName: {
        fontSize: 12,
        color: colors.text.primary,
        textTransform: 'capitalize',
    },
    colorCount: {
        fontSize: 10,
        color: colors.text.secondary,
    },
    itemRow: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: spacing.s,
        borderBottomWidth: StyleSheet.hairlineWidth,
        borderBottomColor: colors.border,
    },
    itemRank: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.secondary,
        width: 30,
    },
    itemName: {
        flex: 1,
        fontSize: 15,
        color: colors.text.primary,
    },
    itemStats: {
        alignItems: 'flex-end',
    },
    itemWearCount: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.accent,
    },
    itemCostPerWear: {
        fontSize: 12,
        color: colors.text.secondary,
    },
    lowWearCount: {
        color: '#F59E0B',
    },
});
