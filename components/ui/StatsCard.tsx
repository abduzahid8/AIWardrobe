// dormi/Wardrobe 2.0 inspired Stats Card Component
// Features: Animated counters, progress rings, trend indicators, gradient backgrounds

import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    useAnimatedProps,
    withTiming,
    Easing,
    interpolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Circle } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../src/hooks/useTheme';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

interface StatsCardProps {
    /** Stat title */
    title: string;
    /** Stat value */
    value: number | string;
    /** Optional subtitle/description */
    subtitle?: string;
    /** Icon name */
    icon?: keyof typeof Ionicons.glyphMap;
    /** Progress value (0-1) for progress ring */
    progress?: number;
    /** Trend indicator (up/down/neutral) */
    trend?: 'up' | 'down' | 'neutral';
    /** Trend value */
    trendValue?: string;
    /** Use gradient background */
    gradient?: boolean;
    /** Gradient colors (if gradient is true) */
    gradientColors?: string[];
    /** Animate the counter */
    animateCounter?: boolean;
    /** Custom style */
    style?: ViewStyle;
}

export const StatsCard: React.FC<StatsCardProps> = ({
    title,
    value,
    subtitle,
    icon,
    progress,
    trend,
    trendValue,
    gradient = false,
    gradientColors,
    animateCounter = true,
    style,
}) => {
    const { colors, spacing, borderRadius, shadows, typography } = useTheme();

    const animatedValue = useSharedValue(0);
    const progressAnim = useSharedValue(0);

    useEffect(() => {
        if (animateCounter && typeof value === 'number') {
            animatedValue.value = withTiming(value, {
                duration: 1000,
                easing: Easing.out(Easing.cubic),
            });
        }

        if (progress !== undefined) {
            progressAnim.value = withTiming(progress, {
                duration: 1200,
                easing: Easing.out(Easing.cubic),
            });
        }
    }, [value, progress]);

    const animatedTextProps = useAnimatedProps(() => {
        if (animateCounter && typeof value === 'number') {
            return {
                text: Math.round(animatedValue.value).toString(),
            };
        }
        return { text: value.toString() };
    });

    const gradientColorsArray = (gradientColors || [colors.primary, colors.primaryDark]) as [string, string, ...string[]];

    // Progress ring
    const size = 60;
    const strokeWidth = 6;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;

    const animatedCircleProps = useAnimatedProps(() => {
        const strokeDashoffset = circumference * (1 - progressAnim.value);
        return {
            strokeDashoffset,
        };
    });

    const getTrendIcon = () => {
        switch (trend) {
            case 'up':
                return 'trending-up';
            case 'down':
                return 'trending-down';
            default:
                return 'remove';
        }
    };

    const getTrendColor = () => {
        switch (trend) {
            case 'up':
                return colors.success;
            case 'down':
                return colors.error;
            default:
                return colors.text.secondary;
        }
    };

    const cardStyle = [
        styles.card,
        {
            borderRadius: borderRadius.l,
            padding: spacing.l,
            ...shadows.medium,
            backgroundColor: gradient ? undefined : colors.surface,
        },
        style,
    ];


    const cardContent = (
        <View style={styles.content}>
            {/* Left side - Text content */}
            <View style={styles.textContainer}>
                <Text
                    style={[
                        styles.title,
                        {
                            color: gradient ? colors.button.primaryText : colors.text.secondary,
                            fontSize: typography.bodySmall.fontSize,
                        },
                    ]}
                >
                    {title.toUpperCase()}
                </Text>

                <Text
                    style={[
                        styles.value,
                        {
                            color: gradient ? colors.button.primaryText : colors.text.primary,
                            fontSize: typography.h2.fontSize,
                            fontWeight: typography.h2.fontWeight,
                        },
                    ]}
                >
                    {typeof value === 'number' && animateCounter ? animatedValue.value : value}
                </Text>

                {subtitle && (
                    <Text
                        style={[
                            styles.subtitle,
                            {
                                color: gradient
                                    ? `${colors.button.primaryText}CC`
                                    : colors.text.secondary,
                                fontSize: typography.caption.fontSize,
                            },
                        ]}
                    >
                        {subtitle}
                    </Text>
                )}

                {/* Trend indicator */}
                {trend && (
                    <View style={styles.trendContainer}>
                        <Ionicons name={getTrendIcon()} size={16} color={getTrendColor()} />
                        {trendValue && (
                            <Text style={[styles.trendText, { color: getTrendColor(), fontSize: 13 }]}>
                                {trendValue}
                            </Text>
                        )}
                    </View>
                )}
            </View>

            {/* Right side - Icon or Progress Ring */}
            {progress !== undefined ? (
                <View style={styles.progressContainer}>
                    <Svg width={size} height={size}>
                        {/* Background circle */}
                        <Circle
                            cx={size / 2}
                            cy={size / 2}
                            r={radius}
                            stroke={gradient ? 'rgba(255,255,255,0.3)' : colors.border}
                            strokeWidth={strokeWidth}
                            fill="none"
                        />
                        {/* Progress circle */}
                        <AnimatedCircle
                            cx={size / 2}
                            cy={size / 2}
                            r={radius}
                            stroke={gradient ? colors.button.primaryText : colors.primary}
                            strokeWidth={strokeWidth}
                            fill="none"
                            strokeDasharray={circumference}
                            animatedProps={animatedCircleProps}
                            strokeLinecap="round"
                            rotation="-90"
                            origin={`${size / 2}, ${size / 2}`}
                        />
                    </Svg>
                    <View style={styles.progressTextContainer}>
                        <Text
                            style={[
                                styles.progressText,
                                {
                                    color: gradient ? colors.button.primaryText : colors.text.primary,
                                    fontSize: 14,
                                    fontWeight: '700',
                                },
                            ]}
                        >
                            {Math.round((progress || 0) * 100)}%
                        </Text>
                    </View>
                </View>
            ) : icon ? (
                <View
                    style={[
                        styles.iconContainer,
                        {
                            backgroundColor: gradient
                                ? 'rgba(255,255,255,0.2)'
                                : colors.primaryLight,
                            borderRadius: borderRadius.m,
                        },
                    ]}
                >
                    <Ionicons
                        name={icon}
                        size={28}
                        color={gradient ? colors.button.primaryText : colors.primary}
                    />
                </View>
            ) : null}
        </View>
    );

    if (gradient) {
        return (
            <LinearGradient
                colors={gradientColorsArray}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={cardStyle}
            >
                {cardContent}
            </LinearGradient>
        );
    }

    return (
        <View style={cardStyle}>
            {cardContent}
        </View>
    );
};

const styles = StyleSheet.create({
    card: {
        overflow: 'hidden',
    },
    content: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    textContainer: {
        flex: 1,
        gap: 4,
    },
    title: {
        fontWeight: '600',
        letterSpacing: 0.5,
    },
    value: {
        marginTop: 4,
    },
    subtitle: {
        fontWeight: '400',
    },
    trendContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 4,
        gap: 4,
    },
    trendText: {
        fontWeight: '600',
    },
    iconContainer: {
        width: 56,
        height: 56,
        justifyContent: 'center',
        alignItems: 'center',
    },
    progressContainer: {
        position: 'relative',
    },
    progressTextContainer: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        justifyContent: 'center',
        alignItems: 'center',
    },
    progressText: {},
});
