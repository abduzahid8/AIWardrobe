/**
 * DailySuggestionCard — Today's outfit recommendation
 *
 * Weather-aware outfit suggestion card with WearLogButton.
 * Replaces generic "Today's Look" on HomeScreen.
 */

import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import WearLogButton from './WearLogButton';
import StreakBadge from './StreakBadge';
import type { DailySuggestion, ClothingItem } from '../src/types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

interface DailySuggestionCardProps {
    suggestion?: DailySuggestion | null;
    onRefresh?: () => void;
}

const DailySuggestionCard: React.FC<DailySuggestionCardProps> = ({ suggestion, onRefresh }) => {
    const { t } = useTranslation();
    const items = useWardrobeStore((state) => state.items);
    const streak = useWardrobeStore((state) => state.streak);

    // Get the suggestion from store if not provided as prop
    const storeSuggestion = useWardrobeStore((state) => state.dailySuggestion);
    const activeSuggestion = suggestion || storeSuggestion;

    if (!activeSuggestion) {
        return (
            <View style={styles.emptyCard}>
                <Ionicons name="sparkles-outline" size={32} color={colors.text.tertiary} />
                <Text style={styles.emptyTitle}>{t('dailySuggestion.noSuggestionYet')}</Text>
                <Text style={styles.emptySubtext}>
                    {t('dailySuggestion.addItemToCloset')}
                </Text>
            </View>
        );
    }

    const outfitItems: ClothingItem[] = activeSuggestion.outfit.itemIds
        .map((id: string) => items.find((item: ClothingItem) => item.id === id))
        .filter((item: ClothingItem | undefined): item is ClothingItem => Boolean(item));

    return (
        <View style={styles.card}>
            {/* Header */}
            <View style={styles.header}>
                <View>
                    <Text style={styles.title}>{t('dailySuggestion.todaysOutfit')}</Text>
                    {activeSuggestion.weatherContext && (
                        <Text style={styles.weatherText}>
                            {Math.round(activeSuggestion.weatherContext.temp)}°
                            {' · '}
                            {activeSuggestion.weatherContext.condition}
                        </Text>
                    )}
                </View>
                <StreakBadge variant="inline" />
            </View>

            {/* Reason */}
            <Text style={styles.reason}>{activeSuggestion.reason}</Text>

            {/* Item thumbnails */}
            <View style={styles.itemsRow}>
                {outfitItems.map((item) => (
                    <View key={item!.id} style={styles.itemThumb}>
                        {item!.imageUrl ? (
                            <Image
                                source={{ uri: item!.imageUrl }}
                                style={styles.itemImage}
                                resizeMode="cover"
                            />
                        ) : (
                            <View style={[styles.itemImage, styles.itemPlaceholder]}>
                                <Ionicons name="shirt-outline" size={20} color={colors.text.tertiary} />
                            </View>
                        )}
                        <Text style={styles.itemLabel} numberOfLines={1}>
                            {item!.subCategory || item!.category}
                        </Text>
                    </View>
                ))}
            </View>

            {/* Wear log button */}
            <WearLogButton
                itemIds={activeSuggestion.outfit.itemIds}
                outfitId={activeSuggestion.outfit.id}
                occasion={activeSuggestion.outfit.occasion}
                weather={activeSuggestion.weatherContext
                    ? { temp: activeSuggestion.weatherContext.temp, condition: activeSuggestion.weatherContext.condition }
                    : undefined
                }
            />
        </View>
    );
};

const styles = StyleSheet.create({
    card: {
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.xl,
        padding: spacing.lg,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.md,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
    },
    title: {
        ...typography.scale.titleLarge,
        color: colors.text.primary,
        fontWeight: '700',
    },
    weatherText: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        marginTop: 2,
    },
    reason: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        lineHeight: 20,
    },
    itemsRow: {
        flexDirection: 'row',
        gap: spacing.md,
    },
    itemThumb: {
        alignItems: 'center',
        gap: spacing.xs,
        flex: 1,
    },
    itemImage: {
        width: 64,
        height: 64,
        borderRadius: radius.md,
    },
    itemPlaceholder: {
        backgroundColor: colors.background.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemLabel: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        textAlign: 'center',
    },
    emptyCard: {
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.xl,
        padding: spacing.xl,
        borderWidth: 1,
        borderColor: colors.border.glass,
        alignItems: 'center',
        gap: spacing.sm,
    },
    emptyTitle: {
        ...typography.scale.titleSmall,
        color: colors.text.secondary,
        fontWeight: '600',
    },
    emptySubtext: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
        textAlign: 'center',
    },
});

export default DailySuggestionCard;
