/**
 * StreakBadge — Visual streak counter
 *
 * Shows the user's consecutive-day wear logging streak.
 * Displays on Home and Profile screens.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTranslation } from 'react-i18next';
import Animated, {
    useAnimatedStyle,
    withRepeat,
    withTiming,
    useSharedValue,
    withSequence,
} from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

interface StreakBadgeProps {
    variant?: 'inline' | 'card';
}

const StreakBadge: React.FC<StreakBadgeProps> = ({ variant = 'inline' }) => {
    const { t } = useTranslation();
    const streak = useWardrobeStore((state) => state.streak);
    const flame = useSharedValue(1);

    React.useEffect(() => {
        if (streak > 0) {
            flame.value = withRepeat(
                withSequence(
                    withTiming(1.15, { duration: 600 }),
                    withTiming(1, { duration: 600 })
                ),
                -1,
                true
            );
        }
    }, [streak, flame]);

    const flameStyle = useAnimatedStyle(() => ({
        transform: [{ scale: flame.value }],
    }));

    if (streak === 0) return null;

    if (variant === 'card') {
        return (
            <View style={styles.card}>
                <Animated.Text style={[styles.flameIcon, flameStyle]}>🔥</Animated.Text>
                <View style={styles.cardContent}>
                    <Text style={styles.cardStreak}>{streak} {t('streak.dayStreak')}</Text>
                    <Text style={styles.cardSubtext}>
                        {streak >= 7
                            ? t('streak.amazingConsistency')
                            : streak >= 3
                                ? t('streak.keepItGoing')
                                : t('streak.greatStart')}
                    </Text>
                </View>
            </View>
        );
    }

    return (
        <View style={styles.inline}>
            <Animated.Text style={[styles.inlineFlame, flameStyle]}>🔥</Animated.Text>
            <Text style={styles.inlineText}>{streak}</Text>
        </View>
    );
};

const styles = StyleSheet.create({
    inline: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255, 165, 0, 0.12)',
        paddingHorizontal: spacing.sm,
        paddingVertical: spacing.xs,
        borderRadius: radius.pill,
        gap: 4,
    },
    inlineFlame: {
        fontSize: 16,
    },
    inlineText: {
        ...typography.scale.labelMedium,
        color: '#F97316',
        fontWeight: '700',
    },
    card: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.glass.frosted,
        padding: spacing.md,
        borderRadius: radius.lg,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.md,
    },
    flameIcon: {
        fontSize: 32,
    },
    cardContent: {
        flex: 1,
    },
    cardStreak: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '700',
    },
    cardSubtext: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        marginTop: 2,
    },
});

export default StreakBadge;
