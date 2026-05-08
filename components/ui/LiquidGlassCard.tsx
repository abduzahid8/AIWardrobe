/**
 * LiquidGlassCard Component
 * Premium glass card with dynamic refraction effects
 * Features: variable blur, light sensitivity, accessibility fallbacks
 */

import React, { useMemo } from 'react';
import {
    View,
    StyleSheet,
    Platform,
    StyleProp,
    ViewStyle,
    Pressable,
    GestureResponderEvent,
} from 'react-native';
import { BlurView } from 'expo-blur';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    interpolate,
    Extrapolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';
import { useAccessibility } from '../../hooks/useAccessibility';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// ============================================
// TYPES
// ============================================

export type GlassVariant = 'clear' | 'light' | 'frosted' | 'opaque' | 'dark';
export type GlassElevation = 'surface' | 'raised' | 'card' | 'floating' | 'modal';

export interface LiquidGlassCardProps {
    /** Glass transparency variant */
    variant?: GlassVariant;
    /** Elevation level for shadow depth */
    elevation?: GlassElevation;
    /** Border radius preset or custom value */
    radius?: keyof typeof LiquidGlass2026Theme.radius | number;
    /** Blur intensity (20-100) */
    blurIntensity?: number;
    /** Show glass border */
    showBorder?: boolean;
    /** Enable press interaction */
    pressable?: boolean;
    /** Press callback */
    onPress?: (event: GestureResponderEvent) => void;
    /** Long press callback */
    onLongPress?: (event: GestureResponderEvent) => void;
    /** Scale on press (0.95-1.0) */
    pressScale?: number;
    /** Enable haptic feedback */
    haptic?: boolean | 'light' | 'medium' | 'heavy';
    /** Children content */
    children: React.ReactNode;
    /** Additional styles */
    style?: StyleProp<ViewStyle>;
    /** Content container styles */
    contentStyle?: StyleProp<ViewStyle>;
    /** Accessibility label */
    accessibilityLabel?: string;
    /** Accessibility hint */
    accessibilityHint?: string;
}

// ============================================
// HELPER FUNCTIONS
// ============================================

const getGlassBackground = (variant: GlassVariant): string => {
    const { glass } = LiquidGlass2026Theme.colors;
    switch (variant) {
        case 'clear': return glass.clear;
        case 'light': return glass.light;
        case 'frosted': return glass.frosted;
        case 'opaque': return glass.opaque;
        case 'dark': return glass.dark;
        default: return glass.frosted;
    }
};

const getElevationLevel = (elevation: GlassElevation): number => {
    return LiquidGlass2026Theme.elevation.levels[elevation];
};

const getBorderRadius = (
    radius: keyof typeof LiquidGlass2026Theme.radius | number
): number => {
    if (typeof radius === 'number') return radius;
    return LiquidGlass2026Theme.radius[radius];
};

const getBlurTint = (variant: GlassVariant): 'light' | 'dark' | 'default' => {
    if (variant === 'dark') return 'dark';
    return 'light';
};

// ============================================
// LIQUID GLASS CARD COMPONENT
// ============================================

export const LiquidGlassCard: React.FC<LiquidGlassCardProps> = ({
    variant = 'frosted',
    elevation = 'card',
    radius = 'card',
    blurIntensity = 60,
    showBorder = true,
    pressable = false,
    onPress,
    onLongPress,
    pressScale = 0.97,
    haptic = true,
    children,
    style,
    contentStyle,
    accessibilityLabel,
    accessibilityHint,
}) => {
    // Accessibility context
    const {
        isHighContrastEnabled,
        isReducedMotionEnabled,
        getElevationStyle,
        getGlassColor,
    } = useAccessibility();

    // Animation values
    const scale = useSharedValue(1);
    const pressed = useSharedValue(0);

    // Calculate values
    const borderRadiusValue = getBorderRadius(radius);
    const elevationLevel = getElevationLevel(elevation);

    // Get accessible styles
    const shadowStyle = useMemo(() => {
        return getElevationStyle(elevationLevel);
    }, [getElevationStyle, elevationLevel]);

    // Animated styles
    const animatedContainerStyle = useAnimatedStyle(() => {
        const animatedScale = isReducedMotionEnabled
            ? 1
            : interpolate(
                pressed.value,
                [0, 1],
                [1, pressScale],
                Extrapolate.CLAMP
            );

        return {
            transform: [{ scale: animatedScale }],
        };
    });

    // Handle press interactions
    const handlePressIn = () => {
        pressed.value = withSpring(1, LiquidGlass2026Theme.animation.spring.snappy);
    };

    const handlePressOut = () => {
        pressed.value = withSpring(0, LiquidGlass2026Theme.animation.spring.snappy);
    };

    const handlePress = (event: GestureResponderEvent) => {
        if (haptic) {
            const hapticStyle = typeof haptic === 'string' ? haptic : 'light';
            const hapticMap = {
                light: Haptics.ImpactFeedbackStyle.Light,
                medium: Haptics.ImpactFeedbackStyle.Medium,
                heavy: Haptics.ImpactFeedbackStyle.Heavy,
            };
            Haptics.impactAsync(hapticMap[hapticStyle]);
        }
        onPress?.(event);
    };

    // High contrast mode: Use solid background instead of blur
    if (isHighContrastEnabled) {
        const Container = pressable ? AnimatedPressable : Animated.View;

        return (
            <Container
                style={[
                    styles.container,
                    {
                        backgroundColor: '#FFFFFF',
                        borderRadius: borderRadiusValue,
                        borderWidth: 2,
                        borderColor: '#0A1931',
                    },
                    style,
                    pressable ? animatedContainerStyle : undefined,
                ]}
                onPress={pressable ? handlePress : undefined}
                onLongPress={pressable ? onLongPress : undefined}
                onPressIn={pressable ? handlePressIn : undefined}
                onPressOut={pressable ? handlePressOut : undefined}
                accessibilityRole={pressable ? 'button' : undefined}
                accessibilityLabel={accessibilityLabel}
                accessibilityHint={accessibilityHint}
            >
                <View style={[styles.content, contentStyle]}>
                    {children}
                </View>
            </Container>
        );
    }

    // Standard glass rendering
    const Container = pressable ? AnimatedPressable : Animated.View;

    // Glass gradient overlay for light reflection effect
    const GlassOverlay = () => (
        <LinearGradient
            colors={[
                'rgba(255, 255, 255, 0.15)',
                'rgba(255, 255, 255, 0.05)',
                'rgba(255, 255, 255, 0)',
            ]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[StyleSheet.absoluteFill, { borderRadius: borderRadiusValue }]}
            pointerEvents="none"
        />
    );

    return (
        <Container
            style={[
                styles.container,
                shadowStyle,
                {
                    borderRadius: borderRadiusValue,
                },
                style,
                pressable ? animatedContainerStyle : undefined,
            ]}
            onPress={pressable ? handlePress : undefined}
            onLongPress={pressable ? onLongPress : undefined}
            onPressIn={pressable ? handlePressIn : undefined}
            onPressOut={pressable ? handlePressOut : undefined}
            accessibilityRole={pressable ? 'button' : undefined}
            accessibilityLabel={accessibilityLabel}
            accessibilityHint={accessibilityHint}
        >
            {/* Blur background */}
            <BlurView
                intensity={blurIntensity}
                tint={getBlurTint(variant)}
                style={[
                    StyleSheet.absoluteFill,
                    {
                        borderRadius: borderRadiusValue,
                        overflow: 'hidden',
                    },
                ]}
                pointerEvents="none"
            />

            {/* Colored glass layer */}
            <View
                style={[
                    StyleSheet.absoluteFill,
                    {
                        backgroundColor: getGlassBackground(variant),
                        borderRadius: borderRadiusValue,
                    },
                ]}
                pointerEvents="none"
            />

            {/* Light reflection overlay */}
            <GlassOverlay />

            {/* Border */}
            {showBorder && (
                <View
                    style={[
                        StyleSheet.absoluteFill,
                        {
                            borderRadius: borderRadiusValue,
                            borderWidth: 1,
                            borderColor: LiquidGlass2026Theme.colors.border.glass,
                        },
                    ]}
                    pointerEvents="none"
                />
            )}

            {/* Content */}
            <View style={[styles.content, contentStyle]}>
                {children}
            </View>
        </Container>
    );
};

// ============================================
// PRESET VARIANTS
// ============================================

interface PresetCardProps extends Omit<LiquidGlassCardProps, 'variant' | 'elevation'> { }

/** Clear glass - very transparent, for subtle overlays */
export const ClearGlassCard: React.FC<PresetCardProps> = (props) => (
    <LiquidGlassCard variant="clear" elevation="raised" blurIntensity={40} {...props} />
);

/** Light glass - semi-transparent, for content cards */
export const LightGlassCard: React.FC<PresetCardProps> = (props) => (
    <LiquidGlassCard variant="light" elevation="card" blurIntensity={50} {...props} />
);

/** Frosted glass - default, balanced transparency */
export const FrostedGlassCard: React.FC<PresetCardProps> = (props) => (
    <LiquidGlassCard variant="frosted" elevation="card" blurIntensity={60} {...props} />
);

/** Opaque glass - high opacity, for important content */
export const OpaqueGlassCard: React.FC<PresetCardProps> = (props) => (
    <LiquidGlassCard variant="opaque" elevation="floating" blurIntensity={80} {...props} />
);

/** Dark glass - dark overlay, for modals and alerts */
export const DarkGlassCard: React.FC<PresetCardProps> = (props) => (
    <LiquidGlassCard variant="dark" elevation="modal" blurIntensity={70} {...props} />
);

/** Pressable frosted card - interactive button-like card */
export const PressableGlassCard: React.FC<PresetCardProps> = (props) => (
    <LiquidGlassCard
        variant="frosted"
        elevation="card"
        pressable
        haptic="light"
        {...props}
    />
);

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        overflow: 'hidden',
    },
    content: {
        padding: LiquidGlass2026Theme.spacing.md,
    },
});

// ============================================
// EXPORTS
// ============================================

export default LiquidGlassCard;
