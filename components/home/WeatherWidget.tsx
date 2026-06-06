import React from 'react';
import { View, Image, ActivityIndicator, StyleSheet } from 'react-native'
import { ScaledText } from '../ui/ScaledText';;
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { FrostedGlassCard } from '../ui';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';

const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

interface WeatherData {
    temp: number;
    description: string;
    icon: string;
    city: string;
}

interface Props {
    weather: WeatherData | null;
    loading: boolean;
    isReducedMotionEnabled: boolean;
}

export default function WeatherWidget({ weather, loading, isReducedMotionEnabled }: Props) {
    const { t } = useTranslation();

    if (loading) {
        return (
            <FrostedGlassCard style={styles.container}>
                <ActivityIndicator size="small" color={colors.accent.primary} />
            </FrostedGlassCard>
        );
    }

    if (!weather) return null;

    const suggestion = weather.temp > 25 ? t('weatherWidget.suggestions.wearLight') : weather.temp > 15 ? t('weatherWidget.suggestions.useLayers') : t('weatherWidget.suggestions.dressWarm');

    return (
        <Animated.View entering={isReducedMotionEnabled ? undefined : FadeInDown.delay(100).duration(400)}>
            <View style={styles.container}>
                <View style={styles.content}>
                    <Image
                        source={{ uri: `https://openweathermap.org/img/wn/${weather.icon}@2x.png` }}
                        style={styles.icon as any}
                        accessibilityLabel={`Weather: ${weather.description}`}
                    />
                    <View style={styles.info}>
                        <ScaledText style={styles.temp}>{weather.temp}°C</ScaledText>
                        <ScaledText style={styles.desc}>{weather.description}</ScaledText>
                    </View>
                    <View style={styles.suggestion}>
                        <Ionicons name="shirt-outline" size={16} color={colors.text.secondary} />
                        <ScaledText style={styles.suggestionText}>{suggestion}</ScaledText>
                    </View>
                </View>
            </View>
        </Animated.View>
    );
}

const styles = StyleSheet.create({
    container: {
        marginHorizontal: spacing.screenPadding,
        marginBottom: spacing.xxl,
    },
    content: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    icon: { width: 56, height: 56 },
    info: { marginLeft: spacing.sm, flex: 1 },
    temp: {
        ...typography.scale.headlineMedium,
        color: colors.text.primary,
        fontWeight: '700',
    },
    desc: {
        ...typography.scale.bodyMedium,
        color: colors.text.secondary,
        textTransform: 'capitalize',
    },
    suggestion: {
        backgroundColor: colors.accent.primary + '15',
        paddingHorizontal: spacing.sm + 4,
        paddingVertical: spacing.sm,
        borderRadius: radius.md,
    },
    suggestionText: {
        ...typography.scale.bodySmall,
        color: colors.text.primary,
        fontWeight: '600',
    },
});
