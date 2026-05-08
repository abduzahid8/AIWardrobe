/**
 * DailySuggestionScreen — "Your outfit for today"
 *
 * The morning entry point of the behavioral loop.
 * Shows 3 AI-scored outfit options. User picks one or skips.
 *
 * Flow:
 *   Push notification → opens this screen
 *   3 outfit cards (swipeable) → [Wear This] / [Show More] / [Skip]
 *   Selecting an outfit → sets dailySuggestion in store
 *   User can log wear now or later via WearLogScreen
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    SafeAreaView,
    ActivityIndicator,
    Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInDown, SlideInRight } from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import {
    generateSuggestions,
    type ScoredOutfit,
    type WeatherContext,
} from '../src/services/suggestionEngine';
import type { Occasion, DailySuggestion } from '../src/types/domain';
import { useTranslation } from 'react-i18next';
import { CachedImage } from '../components/ui/CachedImage';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;
const SCREEN_WIDTH = Dimensions.get('window').width;

// ============================================
// COMPONENT
// ============================================

const DailySuggestionScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
    const { t } = useTranslation();
    const items = useWardrobeStore((state) => state.items);
    const wearLogs = useWardrobeStore((state) => state.wearLogs);
    const streak = useWardrobeStore((state) => state.streak);
    const setDailySuggestion = useWardrobeStore((state) => state.setDailySuggestion);
    const logWear = useWardrobeStore((state) => state.logWear);

    const preferences = useStylePreferenceStore((state) => state.preferences);

    const [suggestions, setSuggestions] = useState<ScoredOutfit[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [isLogged, setIsLogged] = useState(false);

    // TODO: Replace with real weather fetch
    const weather: WeatherContext | undefined = useMemo(() => {
        // Placeholder — wire to weather API
        return { temp: 20, condition: 'clear' };
    }, []);

    // Generate suggestions on mount
    useEffect(() => {
        if (items.length === 0) {
            setIsLoading(false);
            return;
        }

        const results = generateSuggestions({
            items,
            wearLogs,
            occasion: 'casual',
            weather,
            preferences: {
                preferredColors: preferences.favoriteColors,
                avoidColors: preferences.avoidColors,
                preferredStyles: preferences.primaryOccasions,
                adventurousness: 0.5,
            },
        });

        setSuggestions(results);
        setIsLoading(false);
    }, [items, wearLogs, weather, preferences]);

    const currentSuggestion = suggestions[selectedIndex];

    const outfitItems = useMemo(() => {
        if (!currentSuggestion) return [];
        return currentSuggestion.outfit.itemIds
            .map((id) => items.find((i) => i.id === id))
            .filter(Boolean);
    }, [currentSuggestion, items]);

    const handleWearThis = useCallback(() => {
        if (!currentSuggestion) return;

        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        // Set as daily suggestion
        const dailySuggestion: DailySuggestion = {
            outfit: {
                id: `suggestion_${Date.now()}`,
                userId: '',
                itemIds: currentSuggestion.outfit.itemIds,
                occasion: currentSuggestion.outfit.occasion as Occasion,
                generatedBy: 'ai',
                saved: false,
                wornCount: 0,
                lastWornAt: null,
                reasoning: currentSuggestion.reasoning,
                createdAt: new Date().toISOString(),
            },
            reason: currentSuggestion.reasoning,
            weatherContext: weather ? {
                temp: weather.temp,
                condition: weather.condition,
            } : undefined,
            generatedAt: new Date().toISOString(),
        };

        setDailySuggestion(dailySuggestion);

        // Log the wear immediately
        logWear(
            currentSuggestion.outfit.itemIds,
            currentSuggestion.outfit.occasion,
            weather ? { temp: weather.temp, condition: weather.condition } : undefined
        );

        setIsLogged(true);
    }, [currentSuggestion, weather, setDailySuggestion, logWear]);

    const handleNext = useCallback(() => {
        Haptics.selectionAsync();
        setSelectedIndex((prev) => (prev + 1) % suggestions.length);
    }, [suggestions.length]);

    const handleSkip = useCallback(() => {
        navigation.goBack();
    }, [navigation]);

    // ── LOADING ──
    if (isLoading) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color={colors.text.primary} />
                    <Text style={styles.loadingText}>{t('dailySuggestion.loading')}</Text>
                </View>
            </SafeAreaView>
        );
    }

    // ── EMPTY STATE ──
    if (items.length === 0 || suggestions.length === 0) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.emptyContainer}>
                    <Ionicons name="shirt-outline" size={56} color={colors.text.tertiary} />
                    <Text style={styles.emptyTitle}>{t('dailySuggestion.noSuggestions')}</Text>
                    <Text style={styles.emptySubtext}>
                        {items.length === 0
                            ? t('dailySuggestion.scanWardrobeForSuggestions')
                            : t('dailySuggestion.addMoreItemsForSuggestions')}
                    </Text>
                    <TouchableOpacity
                        style={styles.emptyButton}
                        onPress={() => navigation.navigate('ScanWardrobe')}
                    >
                        <Text style={styles.emptyButtonText}>{t('dailySuggestion.scanWardrobe')}</Text>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        );
    }

    // ── LOGGED STATE ──
    if (isLogged) {
        return (
            <SafeAreaView style={styles.container}>
                <Animated.View entering={FadeIn.duration(400)} style={styles.loggedContainer}>
                    <Text style={styles.loggedEmoji}>
                        {streak + 1 >= 7 ? '🏆' : streak + 1 >= 3 ? '🔥' : '✅'}
                    </Text>
                    <Text style={styles.loggedTitle}>{t('dailySuggestion.outfitLogged')}</Text>
                    <Text style={styles.loggedStreak}>{streak + 1} {t('dailySuggestion.dayStreak')}</Text>
                    <TouchableOpacity
                        style={styles.loggedButton}
                        onPress={() => navigation.goBack()}
                    >
                        <Text style={styles.loggedButtonText}>{t('dailySuggestion.backToHome')}</Text>
                    </TouchableOpacity>
                </Animated.View>
            </SafeAreaView>
        );
    }

    // ── MAIN VIEW ──
    return (
        <SafeAreaView style={styles.container}>
            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity onPress={handleSkip}>
                    <Text style={styles.skipText}>{t('common.skip')}</Text>
                </TouchableOpacity>
                <View style={styles.headerCenter}>
                    <Text style={styles.headerTitle}>{t('dailySuggestion.todaysOutfit')}</Text>
                    {weather && (
                        <Text style={styles.weatherBadge}>
                            {Math.round(weather.temp)}° · {weather.condition}
                        </Text>
                    )}
                </View>
                <Text style={styles.pageIndicator}>
                    {selectedIndex + 1}/{suggestions.length}
                </Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>
                {/* Outfit Card */}
                <Animated.View
                    key={selectedIndex}
                    entering={SlideInRight.duration(300)}
                    style={styles.outfitCard}
                >
                    {/* Score */}
                    <View style={styles.scoreRow}>
                        <View style={styles.scoreBadge}>
                            <Text style={styles.scoreText}>
                                {Math.round(currentSuggestion.score * 100)}% match
                            </Text>
                        </View>
                    </View>

                    {/* Items grid */}
                    <View style={styles.itemsGrid}>
                        {outfitItems.map((item, index) => (
                            <Animated.View
                                key={item!.id}
                                entering={FadeInDown.delay(index * 80).duration(300)}
                                style={styles.outfitItem}
                            >
                                {item!.imageUrl ? (
                                    <CachedImage
                                        uri={item!.imageUrl}
                                        style={styles.outfitImage}
                                        contentFit="cover"
                                        fadeIn={false}
                                    />
                                ) : (
                                    <View style={[styles.outfitImage, styles.placeholderImage]}>
                                        <View style={[styles.colorDot, { backgroundColor: item!.colorHex || '#CCC' }]} />
                                    </View>
                                )}
                                <Text style={styles.outfitItemCategory}>
                                    {item!.category}
                                </Text>
                                <Text style={styles.outfitItemName} numberOfLines={1}>
                                    {item!.subCategory || item!.primaryColor}
                                </Text>
                            </Animated.View>
                        ))}
                    </View>

                    {/* Reasoning */}
                    <View style={styles.reasoningBox}>
                        <Ionicons name="sparkles" size={16} color={colors.text.secondary} />
                        <Text style={styles.reasoningText}>{currentSuggestion.reasoning}</Text>
                    </View>

                    {/* Score breakdown */}
                    <View style={styles.breakdownRow}>
                        <ScoreBar label={t('dailySuggestion.style')} value={currentSuggestion.breakdown.preferenceScore} />
                        <ScoreBar label={t('dailySuggestion.weather')} value={currentSuggestion.breakdown.weatherScore} />
                        <ScoreBar label={t('dailySuggestion.fresh')} value={currentSuggestion.breakdown.noveltyScore} />
                        <ScoreBar label={t('dailySuggestion.colors')} value={currentSuggestion.breakdown.harmonyScore} />
                    </View>
                </Animated.View>
            </ScrollView>

            {/* Bottom actions */}
            <View style={styles.bottomBar}>
                {suggestions.length > 1 && (
                    <TouchableOpacity
                        style={styles.nextButton}
                        onPress={handleNext}
                        activeOpacity={0.7}
                    >
                        <Ionicons name="shuffle-outline" size={20} color={colors.text.primary} />
                        <Text style={styles.nextButtonText}>{t('dailySuggestion.tryAnother')}</Text>
                    </TouchableOpacity>
                )}
                <TouchableOpacity
                    style={styles.wearButton}
                    onPress={handleWearThis}
                    activeOpacity={0.8}
                >
                    <Ionicons name="checkmark-circle" size={22} color="#FFF" />
                    <Text style={styles.wearButtonText}>{t('dailySuggestion.wearToday')}</Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
};

// ============================================
// SUB-COMPONENTS
// ============================================

const ScoreBar: React.FC<{ label: string; value: number }> = ({ label, value }) => (
    <View style={styles.scoreBarContainer}>
        <Text style={styles.scoreBarLabel}>{label}</Text>
        <View style={styles.scoreBarTrack}>
            <View style={[styles.scoreBarFill, { width: `${Math.round(value * 100)}%` }]} />
        </View>
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
        paddingBottom: 140,
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
    skipText: {
        ...typography.scale.labelLarge,
        color: colors.text.secondary,
    },
    headerCenter: {
        alignItems: 'center',
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },
    weatherBadge: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        marginTop: 2,
    },
    pageIndicator: {
        ...typography.scale.labelMedium,
        color: colors.text.tertiary,
    },

    // Outfit card
    outfitCard: {
        margin: spacing.lg,
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.xl,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.lg,
    },
    scoreRow: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
    },
    scoreBadge: {
        backgroundColor: 'rgba(34, 197, 94, 0.12)',
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.xs,
        borderRadius: radius.pill,
    },
    scoreText: {
        ...typography.scale.labelMedium,
        color: '#16A34A',
        fontWeight: '700',
    },

    // Items grid
    itemsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.md,
        justifyContent: 'center',
    },
    outfitItem: {
        alignItems: 'center',
        width: (SCREEN_WIDTH - spacing.lg * 2 - spacing.lg * 2 - spacing.md * 2) / 3,
        gap: spacing.xs,
    },
    outfitImage: {
        width: '100%',
        aspectRatio: 1,
        borderRadius: radius.lg,
    },
    placeholderImage: {
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    colorDot: {
        width: 32,
        height: 32,
        borderRadius: 16,
    },
    outfitItemCategory: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
    },
    outfitItemName: {
        ...typography.scale.labelMedium,
        color: colors.text.primary,
        fontWeight: '600',
        textAlign: 'center',
    },

    // Reasoning
    reasoningBox: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: spacing.sm,
        backgroundColor: colors.background.secondary,
        padding: spacing.md,
        borderRadius: radius.md,
    },
    reasoningText: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        flex: 1,
        lineHeight: 18,
    },

    // Score breakdown
    breakdownRow: {
        flexDirection: 'row',
        gap: spacing.sm,
    },
    scoreBarContainer: {
        flex: 1,
        gap: 4,
    },
    scoreBarLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        textAlign: 'center',
    },
    scoreBarTrack: {
        height: 4,
        backgroundColor: colors.background.secondary,
        borderRadius: 2,
        overflow: 'hidden',
    },
    scoreBarFill: {
        height: '100%',
        backgroundColor: '#22C55E',
        borderRadius: 2,
    },

    // Bottom bar
    bottomBar: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingHorizontal: spacing.lg,
        paddingBottom: spacing.xl,
        paddingTop: spacing.md,
        backgroundColor: colors.background.primary,
        borderTopWidth: 1,
        borderTopColor: colors.border.subtle,
        gap: spacing.sm,
    },
    nextButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.sm + 2,
        borderRadius: radius.pill,
        backgroundColor: colors.background.secondary,
        gap: spacing.sm,
    },
    nextButtonText: {
        ...typography.scale.labelLarge,
        color: colors.text.primary,
        fontWeight: '600',
    },
    wearButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#22C55E',
        paddingVertical: spacing.md + 2,
        borderRadius: radius.pill,
        gap: spacing.sm,
    },
    wearButtonText: {
        ...typography.scale.labelLarge,
        color: '#FFF',
        fontWeight: '700',
    },

    // Loading
    loadingContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        gap: spacing.md,
    },
    loadingText: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
    },

    // Empty
    emptyContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: spacing.xl,
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
    },
    emptyButton: {
        marginTop: spacing.md,
        backgroundColor: colors.text.primary,
        paddingVertical: spacing.md,
        paddingHorizontal: spacing.xl,
        borderRadius: radius.pill,
    },
    emptyButtonText: {
        ...typography.scale.labelLarge,
        color: '#FFF',
        fontWeight: '700',
    },

    // Logged
    loggedContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        gap: spacing.md,
    },
    loggedEmoji: {
        fontSize: 64,
        marginBottom: spacing.sm,
    },
    loggedTitle: {
        ...typography.scale.displaySmall,
        color: colors.text.primary,
        fontWeight: '700',
    },
    loggedStreak: {
        ...typography.scale.headlineMedium,
        color: '#F97316',
        fontWeight: '700',
    },
    loggedButton: {
        marginTop: spacing.lg,
        backgroundColor: colors.text.primary,
        paddingVertical: spacing.md,
        paddingHorizontal: spacing.xxl,
        borderRadius: radius.pill,
    },
    loggedButtonText: {
        ...typography.scale.labelLarge,
        color: '#FFF',
        fontWeight: '700',
    },
});

export default DailySuggestionScreen;
