/**
 * WearLogScreen — "What did you wear today?"
 *
 * The missing behavioral anchor of the Suggest → Wear → Log → Learn loop.
 * Triggered by evening push notification or manual entry.
 *
 * Flow:
 *   1. Show today's suggestion (if any) → one-tap confirm
 *   2. Or pick items manually from closet grid
 *   3. Optional: tag occasion + mood
 *   4. Log → streak update → celebration animation
 */

import React, { useState, useCallback, useMemo } from 'react';
import { View, StyleSheet, TouchableOpacity, ScrollView, FlatList, SafeAreaView,  } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withSequence,
    withTiming,
    FadeIn,
    SlideInUp,
} from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import { CachedImage } from '../components/ui/CachedImage';
import useWardrobeStore from '../store/wardrobeStore';
import type { ClothingItem, Occasion } from '../src/types/domain';
import { getMacroCategory } from '../src/utils/categoryMapper';
import { useTranslation } from 'react-i18next';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// ============================================
// CONSTANTS
// ============================================

const OCCASIONS: { value: Occasion; label: string; icon: string }[] = [
    { value: 'casual', label: 'Casual', icon: 'cafe-outline' },
    { value: 'work', label: 'Work', icon: 'briefcase-outline' },
    { value: 'formal', label: 'Formal', icon: 'diamond-outline' },
    { value: 'sport', label: 'Sport', icon: 'fitness-outline' },
    { value: 'date', label: 'Date', icon: 'heart-outline' },
    { value: 'travel', label: 'Travel', icon: 'airplane-outline' },
];

// ============================================
// COMPONENT
// ============================================

const WearLogScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
    const { t } = useTranslation();
    const items = useWardrobeStore((state) => state.items);
    const dailySuggestion = useWardrobeStore((state) => state.dailySuggestion);
    const logWear = useWardrobeStore((state) => state.logWear);
    const streak = useWardrobeStore((state) => state.streak);

    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [occasion, setOccasion] = useState<Occasion>('casual');
    const [isLogged, setIsLogged] = useState(false);
    const [mode, setMode] = useState<'suggestion' | 'manual'>(
        dailySuggestion ? 'suggestion' : 'manual'
    );

    const celebrationScale = useSharedValue(1);

    // Pre-select suggestion items
    const suggestionItems = useMemo(() => {
        if (!dailySuggestion) return [];
        return dailySuggestion.outfit.itemIds
            .map((id: string) => items.find((i) => i.id === id))
            .filter(Boolean) as ClothingItem[];
    }, [dailySuggestion, items]);

    // Group items by category for manual selection
    const groupedItems = useMemo(() => {
        const groups: Record<string, ClothingItem[]> = {};
        items.forEach((item) => {
            const cat = item.category;
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(item);
        });
        return groups;
    }, [items]);

    const toggleItem = useCallback(
        (id: string) => {
            Haptics.selectionAsync();
            setSelectedIds((prev) => {
                if (prev.includes(id)) {
                    return prev.filter((i) => i !== id);
                }
                const item = items.find(i => i.id === id);
                if (!item) return [...prev, id];
                const macro = getMacroCategory(item.category);
                const hasExistingInCategory = prev.some(existingId => {
                    const existing = items.find(i => i.id === existingId);
                    return existing && getMacroCategory(existing.category) === macro;
                });
                if (hasExistingInCategory) {
                    return prev.map(existingId => {
                        const existing = items.find(i => i.id === existingId);
                        return existing && getMacroCategory(existing.category) === macro
                            ? id
                            : existingId;
                    });
                }
                return [...prev, id];
            });
        },
        [items]
    );

    const handleLogWear = useCallback(async () => {
        const idsToLog = mode === 'suggestion' && dailySuggestion
            ? dailySuggestion.outfit.itemIds
            : selectedIds;

        if (idsToLog.length === 0) return;

        // Haptic + animation
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        celebrationScale.value = withSequence(
            withSpring(0.9, { damping: 15, stiffness: 400 }),
            withSpring(1.2, { damping: 12, stiffness: 300 }),
            withSpring(1, { damping: 15, stiffness: 400 })
        );

        logWear(idsToLog, occasion);
        setIsLogged(true);
    }, [mode, dailySuggestion, selectedIds, occasion, logWear, celebrationScale]);

    const celebrationStyle = useAnimatedStyle(() => ({
        transform: [{ scale: celebrationScale.value }],
    }));

    // ── LOGGED STATE ──
    if (isLogged) {
        const newStreak = streak + 1;
        return (
            <SafeAreaView style={styles.container}>
                <Animated.View
                    entering={FadeIn.duration(400)}
                    style={styles.celebrationContainer}
                >
                    <Animated.Text style={[styles.celebrationEmoji, celebrationStyle]}>
                        {newStreak >= 7 ? '🏆' : newStreak >= 3 ? '🔥' : '✅'}
                    </Animated.Text>
                    <ScaledText style={styles.celebrationTitle}>{t('wearLog.outfitLogged')}</ScaledText>
                    <ScaledText style={styles.celebrationStreak}>
                        {t('wearLog.dayStreak', { count: newStreak })}
                    </ScaledText>
                    <ScaledText style={styles.celebrationSubtext}>
                        {newStreak >= 30
                            ? t('wearLog.streak30')
                            : newStreak >= 14
                                ? t('wearLog.streak14')
                                : newStreak >= 7
                                    ? t('wearLog.streak7')
                                    : newStreak >= 3
                                        ? t('wearLog.streak3')
                                        : t('wearLog.streak1')}
                    </ScaledText>

                    <TouchableOpacity
                        style={styles.doneButton}
                        onPress={() => navigation.goBack()}
                        activeOpacity={0.8}
                    >
                        <ScaledText style={styles.doneButtonText}>{t('common.done')}</ScaledText>
                    </TouchableOpacity>
                </Animated.View>
            </SafeAreaView>
        );
    }

    // ── MAIN LOG SCREEN ──
    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => navigation.goBack()}>
                        <Ionicons name="close" size={28} color={colors.text.primary} />
                    </TouchableOpacity>
                    <ScaledText style={styles.headerTitle}>{t('wearLog.logTodaysOutfit')}</ScaledText>
                    <View style={{ width: 28 }} />
                </View>

                {/* Mode Switcher */}
                {dailySuggestion && (
                    <View style={styles.modeSwitch}>
                        <TouchableOpacity
                            style={[styles.modeButton, mode === 'suggestion' && styles.modeButtonActive]}
                            onPress={() => setMode('suggestion')}
                        >
                            <ScaledText style={[styles.modeText, mode === 'suggestion' && styles.modeTextActive]}>
                                {t('wearLog.suggestedOutfit')}
                            </ScaledText>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeButton, mode === 'manual' && styles.modeButtonActive]}
                            onPress={() => setMode('manual')}
                        >
                            <ScaledText style={[styles.modeText, mode === 'manual' && styles.modeTextActive]}>
                                {t('wearLog.pickItems')}
                            </ScaledText>
                        </TouchableOpacity>
                    </View>
                )}

                {/* Suggestion Mode */}
                {mode === 'suggestion' && dailySuggestion && (
                    <Animated.View entering={FadeIn.duration(300)} style={styles.suggestionCard}>
                        <ScaledText style={styles.suggestionLabel}>{t('wearLog.todaysSuggestion')}</ScaledText>
                        <View style={styles.suggestionItems}>
                            {suggestionItems.map((item) => (
                                <View key={item.id} style={styles.suggestionItem}>
                                    {item.imageUrl ? (
                                        <CachedImage
                                            uri={item.imageUrl}
                                            style={styles.suggestionImage}
                                            contentFit="cover"
                                            fadeIn={false}
                                        />
                                    ) : (
                                        <View style={[styles.suggestionImage, styles.placeholderImage]}>
                                            <Ionicons name="shirt-outline" size={24} color={colors.text.tertiary} />
                                        </View>
                                    )}
                                    <ScaledText style={styles.suggestionItemLabel} numberOfLines={1}>
                                        {item.subCategory || item.category}
                                    </ScaledText>
                                </View>
                            ))}
                        </View>
                        {dailySuggestion.reason && (
                            <ScaledText style={styles.suggestionReason}>{dailySuggestion.reason}</ScaledText>
                        )}
                    </Animated.View>
                )}

                {/* Manual Mode */}
                {mode === 'manual' && (
                    <Animated.View entering={FadeIn.duration(300)}>
                        {Object.entries(groupedItems).map(([category, categoryItems]) => (
                            <View key={category} style={styles.categorySection}>
                                <ScaledText style={styles.categoryTitle}>
                                    {category.charAt(0).toUpperCase() + category.slice(1)}
                                </ScaledText>
                                <FlatList
                                    data={categoryItems}
                                    horizontal
                                    showsHorizontalScrollIndicator={false}
                                    keyExtractor={(item) => item.id}
                                    contentContainerStyle={styles.itemList}
                                    renderItem={({ item }) => {
                                        const isSelected = selectedIds.includes(item.id);
                                        return (
                                            <TouchableOpacity
                                                style={[styles.itemCard, isSelected && styles.itemCardSelected]}
                                                onPress={() => toggleItem(item.id)}
                                                activeOpacity={0.7}
                                            >
                                                {item.imageUrl ? (
                                                    <CachedImage
                                                        uri={item.imageUrl}
                                                        style={styles.itemImage}
                                                        contentFit="cover"
                                                        fadeIn={false}
                                                    />
                                                ) : (
                                                    <View style={[styles.itemImage, styles.placeholderImage]}>
                                                        <Ionicons
                                                            name="shirt-outline"
                                                            size={20}
                                                            color={colors.text.tertiary}
                                                        />
                                                    </View>
                                                )}
                                                {isSelected && (
                                                    <View style={styles.checkBadge}>
                                                        <Ionicons name="checkmark" size={14} color="#FFF" />
                                                    </View>
                                                )}
                                                <ScaledText style={styles.itemLabel} numberOfLines={1}>
                                                    {item.subCategory || item.primaryColor || item.category}
                                                </ScaledText>
                                            </TouchableOpacity>
                                        );
                                    }}
                                />
                            </View>
                        ))}

                        {items.length === 0 && (
                            <View style={styles.emptyState}>
                                <Ionicons name="shirt-outline" size={48} color={colors.text.tertiary} />
                                <ScaledText style={styles.emptyText}>{t('wearLog.noItemsYet')}</ScaledText>
                                <ScaledText style={styles.emptySubtext}>{t('wearLog.scanWardrobe')}</ScaledText>
                            </View>
                        )}
                    </Animated.View>
                )}

                {/* Occasion Picker */}
                <View style={styles.occasionSection}>
                    <ScaledText style={styles.sectionLabel}>{t('wearLog.occasion')}</ScaledText>
                    <View style={styles.occasionRow}>
                        {OCCASIONS.map((occ) => (
                            <TouchableOpacity
                                key={occ.value}
                                style={[
                                    styles.occasionChip,
                                    occasion === occ.value && styles.occasionChipActive,
                                ]}
                                onPress={() => {
                                    setOccasion(occ.value);
                                    Haptics.selectionAsync();
                                }}
                            >
                                <Ionicons
                                    name={occ.icon as any}
                                    size={16}
                                    color={occasion === occ.value ? '#FFF' : colors.text.secondary}
                                />
                                <ScaledText
                                    style={[
                                        styles.occasionLabel,
                                        occasion === occ.value && styles.occasionLabelActive,
                                    ]}
                                >
                                    {occ.label}
                                </ScaledText>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>
            </ScrollView>

            {/* Log Button */}
            <View style={styles.bottomBar}>
                <TouchableOpacity
                    style={[
                        styles.logButton,
                        (mode === 'manual' && selectedIds.length === 0) && styles.logButtonDisabled,
                    ]}
                    onPress={handleLogWear}
                    disabled={mode === 'manual' && selectedIds.length === 0}
                    activeOpacity={0.8}
                >
                    <Ionicons name="checkmark-circle" size={22} color="#FFF" />
                    <ScaledText style={styles.logButtonText}>
                        {mode === 'suggestion'
                            ? t('wearLog.iWoreThisToday')
                            : selectedIds.length > 0
                                ? t('wearLog.logItems', { 
                                    count: selectedIds.length, 
                                    plural: selectedIds.length > 1 ? 's' : '' 
                                })
                                : t('wearLog.selectItemsToLog')}
                    </ScaledText>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    scrollContent: {
        paddingBottom: 120,
    },
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

    // Mode switch
    modeSwitch: {
        flexDirection: 'row',
        marginHorizontal: spacing.lg,
        marginTop: spacing.md,
        backgroundColor: colors.background.secondary,
        borderRadius: radius.pill,
        padding: 4,
    },
    modeButton: {
        flex: 1,
        paddingVertical: spacing.sm + 2,
        alignItems: 'center',
        borderRadius: radius.pill,
    },
    modeButtonActive: {
        backgroundColor: colors.text.primary,
    },
    modeText: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
        fontWeight: '600',
    },
    modeTextActive: {
        color: '#FFF',
    },

    // Suggestion mode
    suggestionCard: {
        margin: spacing.lg,
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.xl,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.md,
    },
    suggestionLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
    },
    suggestionItems: {
        flexDirection: 'row',
        gap: spacing.md,
    },
    suggestionItem: {
        alignItems: 'center',
        gap: spacing.xs,
        flex: 1,
    },
    suggestionImage: {
        width: 72,
        height: 72,
        borderRadius: radius.md,
    },
    placeholderImage: {
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    suggestionItemLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.secondary,
        textAlign: 'center',
    },
    suggestionReason: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        fontStyle: 'italic',
    },

    // Manual mode
    categorySection: {
        marginTop: spacing.lg,
    },
    categoryTitle: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '700',
        paddingHorizontal: spacing.lg,
        marginBottom: spacing.sm,
    },
    itemList: {
        paddingHorizontal: spacing.lg,
        gap: spacing.sm,
    },
    itemCard: {
        width: 80,
        alignItems: 'center',
        gap: spacing.xs,
    },
    itemCardSelected: {
        opacity: 1,
    },
    itemImage: {
        width: 72,
        height: 72,
        borderRadius: radius.md,
        borderWidth: 2,
        borderColor: 'transparent',
    },
    checkBadge: {
        position: 'absolute',
        top: 4,
        right: 8,
        width: 22,
        height: 22,
        borderRadius: 11,
        backgroundColor: '#22C55E',
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        textAlign: 'center',
        maxWidth: 72,
    },

    // Empty
    emptyState: {
        alignItems: 'center',
        paddingVertical: spacing.xxl,
        gap: spacing.sm,
    },
    emptyText: {
        ...typography.scale.titleSmall,
        color: colors.text.secondary,
        fontWeight: '600',
    },
    emptySubtext: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
    },

    // Occasion
    occasionSection: {
        marginTop: spacing.xl,
        paddingHorizontal: spacing.lg,
        gap: spacing.sm,
    },
    sectionLabel: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '700',
    },
    occasionRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.sm,
    },
    occasionChip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: spacing.sm,
        paddingHorizontal: spacing.md,
        borderRadius: radius.pill,
        backgroundColor: colors.glass.frosted,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.xs,
    },
    occasionChipActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    occasionLabel: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
    },
    occasionLabelActive: {
        color: '#FFF',
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
    },
    logButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#22C55E',
        paddingVertical: spacing.md + 2,
        borderRadius: radius.pill,
        gap: spacing.sm,
    },
    logButtonDisabled: {
        backgroundColor: colors.border.glass,
    },
    logButtonText: {
        ...typography.scale.labelLarge,
        color: '#FFF',
        fontWeight: '700',
    },

    // Celebration
    celebrationContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: spacing.xl,
        gap: spacing.md,
    },
    celebrationEmoji: {
        fontSize: 64,
        marginBottom: spacing.md,
    },
    celebrationTitle: {
        ...typography.scale.displaySmall,
        color: colors.text.primary,
        fontWeight: '700',
    },
    celebrationStreak: {
        ...typography.scale.headlineMedium,
        color: '#F97316',
        fontWeight: '700',
    },
    celebrationSubtext: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        textAlign: 'center',
        lineHeight: 22,
    },
    doneButton: {
        marginTop: spacing.xl,
        backgroundColor: colors.text.primary,
        paddingVertical: spacing.md,
        paddingHorizontal: spacing.xxl,
        borderRadius: radius.pill,
    },
    doneButtonText: {
        ...typography.scale.labelLarge,
        color: '#FFF',
        fontWeight: '700',
    },
});

export default WearLogScreen;
