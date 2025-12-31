import React from 'react';
import {
    View,
    ViewStyle,
    StyleSheet,
    TouchableOpacity,
    TouchableOpacityProps,
} from 'react-native';
import { BlurView } from 'expo-blur';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { colors, shadows, spacing, animations } from '../../src/theme';

type CardVariant = 'default' | 'glass' | 'elevated' | 'outline';

interface CardProps {
    children: React.ReactNode;
    style?: ViewStyle;
    variant?: CardVariant;
    onPress?: () => void;
    onLongPress?: () => void;
    disabled?: boolean;
    hapticFeedback?: boolean;
}

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

/**
 * Reusable Card component with multiple variants
 * 
 * @example
 * // Default card
 * <Card>
 *   <Text>Content here</Text>
 * </Card>
 * 
 * @example
 * // Glassmorphism card with press action
 * <Card variant="glass" onPress={() => console.log('pressed')}>
 *   <Text>Glassmorphism</Text>
 * </Card>
 */
export const Card: React.FC<CardProps> = ({
    children,
    style,
    variant = 'default',
    onPress,
    onLongPress,
    disabled = false,
    hapticFeedback = true,
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const handlePressIn = () => {
        if (onPress || onLongPress) {
            scale.value = withSpring(animations.scale.pressed, animations.springFast);
        }
    };

    const handlePressOut = () => {
        scale.value = withSpring(animations.scale.normal, animations.springFast);
    };

    const handlePress = () => {
        if (hapticFeedback) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }
        onPress?.();
    };

    const handleLongPress = () => {
        if (hapticFeedback) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        }
        onLongPress?.();
    };

    const variantStyles = getVariantStyles(variant);

    // Glass variant uses BlurView
    if (variant === 'glass') {
        const content = (
            <BlurView intensity={80} tint="light" style={[styles.glassContainer, style]}>
                <View style={styles.glassInner}>{children}</View>
            </BlurView>
        );

        if (onPress || onLongPress) {
            return (
                <AnimatedTouchable
                    onPress={handlePress}
                    onLongPress={handleLongPress}
                    onPressIn={handlePressIn}
                    onPressOut={handlePressOut}
                    disabled={disabled}
                    activeOpacity={1}
                    style={animatedStyle}
                >
                    {content}
                </AnimatedTouchable>
            );
        }

        return <Animated.View style={animatedStyle}>{content}</Animated.View>;
    }

    // Regular variants
    const content = (
        <View style={[styles.base, variantStyles, style]}>{children}</View>
    );

    if (onPress || onLongPress) {
        return (
            <AnimatedTouchable
                onPress={handlePress}
                onLongPress={handleLongPress}
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                disabled={disabled}
                activeOpacity={1}
                style={animatedStyle}
            >
                {content}
            </AnimatedTouchable>
        );
    }

    return <Animated.View style={animatedStyle}>{content}</Animated.View>;
};

const getVariantStyles = (variant: CardVariant): ViewStyle => {
    switch (variant) {
        case 'elevated':
            return {
                backgroundColor: colors.surface,
                ...shadows.medium,
            };
        case 'outline':
            return {
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderColor: colors.border,
            };
        case 'glass':
            return {}; // Handled separately with BlurView
        case 'default':
        default:
            return {
                backgroundColor: colors.surface,
                ...shadows.soft,
            };
    }
};

const styles = StyleSheet.create({
    base: {
        borderRadius: 16,
        padding: spacing.m,
        overflow: 'hidden',
    },
    glassContainer: {
        borderRadius: 16,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: colors.glass.border,
    },
    glassInner: {
        padding: spacing.m,
        backgroundColor: colors.glass.background,
    },
});

export default Card;
