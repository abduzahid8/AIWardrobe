import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
    FadeInRight,
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withSequence,
    withTiming,
    Easing,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../src/theme/ThemeContext';

const { width } = Dimensions.get('window');

interface QuickStatProps {
    icon: keyof typeof Ionicons.glyphMap;
    value: string | number;
    label: string;
    color?: string;
    trend?: 'up' | 'down' | 'neutral';
    trendValue?: string;
    index?: number;
    variant?: 'default' | 'gradient' | 'compact';
}

/**
 * QuickStat - Premium statistics display component
 * 
 * Features:
 * - Multiple variants (default, gradient, compact)
 * - Trend indicators
 * - Entrance animations
 * - Themed colors
 */
export const QuickStat: React.FC<QuickStatProps> = ({
    icon,
    value,
    label,
    color,
    trend,
    trendValue,
    index = 0,
    variant = 'default',
}) => {
    const { colors, isDark } = useTheme();
    const accentColor = color || colors.primary;

    const getTrendColor = () => {
        switch (trend) {
            case 'up': return '#22C55E';
            case 'down': return '#EF4444';
            default: return colors.text.muted;
        }
    };

    const getTrendIcon = () => {
        switch (trend) {
            case 'up': return 'trending-up';
            case 'down': return 'trending-down';
            default: return 'remove';
        }
    };

    if (variant === 'gradient') {
        return (
            <Animated.View
                entering={FadeInRight.delay(index * 80).springify()}
            >
                <LinearGradient
                    colors={isDark
                        ? [colors.surface, colors.surfaceHighlight]
                        : ['#FFFFFF', '#F8FAFC']
                    }
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[
                        styles.gradientCard,
                        { borderColor: isDark ? colors.border : '#E2E8F0' }
                    ]}
                >
                    <View style={[styles.iconCircle, { backgroundColor: `${accentColor}15` }]}>
                        <Ionicons name={icon} size={20} color={accentColor} />
                    </View>

                    <Text style={[styles.gradientValue, { color: colors.text.primary }]}>
                        {value}
                    </Text>
                    <Text style={[styles.gradientLabel, { color: colors.text.secondary }]}>
                        {label}
                    </Text>

                    {trend && trendValue && (
                        <View style={styles.trendContainer}>
                            <Ionicons
                                name={getTrendIcon() as any}
                                size={12}
                                color={getTrendColor()}
                            />
                            <Text style={[styles.trendText, { color: getTrendColor() }]}>
                                {trendValue}
                            </Text>
                        </View>
                    )}
                </LinearGradient>
            </Animated.View>
        );
    }

    if (variant === 'compact') {
        return (
            <Animated.View
                entering={FadeInRight.delay(index * 80).springify()}
                style={[
                    styles.compactCard,
                    { backgroundColor: isDark ? colors.surface : '#F8FAFC' }
                ]}
            >
                <Ionicons name={icon} size={18} color={accentColor} />
                <Text style={[styles.compactValue, { color: colors.text.primary }]}>
                    {value}
                </Text>
                <Text style={[styles.compactLabel, { color: colors.text.secondary }]}>
                    {label}
                </Text>
            </Animated.View>
        );
    }

    // Default variant
    return (
        <Animated.View
            entering={FadeInRight.delay(index * 80).springify()}
            style={[
                styles.defaultCard,
                {
                    backgroundColor: isDark ? colors.surface : '#FFFFFF',
                    borderColor: isDark ? colors.border : '#F1F5F9',
                }
            ]}
        >
            <View style={[styles.iconBox, { backgroundColor: `${accentColor}15` }]}>
                <Ionicons name={icon} size={18} color={accentColor} />
            </View>

            <Text style={[styles.defaultValue, { color: colors.text.primary }]}>
                {value}
            </Text>
            <Text style={[styles.defaultLabel, { color: colors.text.secondary }]}>
                {label}
            </Text>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    // Default variant
    defaultCard: {
        width: (width - 60) / 3,
        padding: 12,
        borderRadius: 14,
        borderWidth: 1,
        alignItems: 'center',
    },
    iconBox: {
        width: 36,
        height: 36,
        borderRadius: 10,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    defaultValue: {
        fontSize: 18,
        fontWeight: '700',
        marginBottom: 2,
    },
    defaultLabel: {
        fontSize: 11,
        fontWeight: '500',
        textAlign: 'center',
    },

    // Gradient variant
    gradientCard: {
        width: (width - 52) / 2,
        padding: 16,
        borderRadius: 18,
        borderWidth: 1,
        marginRight: 12,
        marginBottom: 12,
    },
    iconCircle: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    gradientValue: {
        fontSize: 24,
        fontWeight: '700',
        marginBottom: 2,
    },
    gradientLabel: {
        fontSize: 13,
        fontWeight: '500',
    },
    trendContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 8,
        gap: 4,
    },
    trendText: {
        fontSize: 12,
        fontWeight: '600',
    },

    // Compact variant
    compactCard: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 10,
        gap: 8,
    },
    compactValue: {
        fontSize: 15,
        fontWeight: '700',
    },
    compactLabel: {
        fontSize: 12,
    },
});

export default QuickStat;
