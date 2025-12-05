import React from 'react';
import {
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
    Platform,
    ViewStyle,
    TextStyle,
} from 'react-native';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
    interpolate,
    Extrapolate,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { colors, spacing, shadows } from '../src/theme';
import { BlurView } from 'expo-blur';

interface AnimatedButtonProps {
    onPress: () => void;
    title?: string;
    icon?: keyof typeof Ionicons.glyphMap;
    iconSize?: number;
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'glass';
    size?: 'small' | 'medium' | 'large';
    disabled?: boolean;
    fullWidth?: boolean;
    style?: ViewStyle;
    textStyle?: TextStyle;
    haptic?: 'light' | 'medium' | 'heavy' | 'none';
    children?: React.ReactNode;
}

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

export const AnimatedButton: React.FC<AnimatedButtonProps> = ({
    onPress,
    title,
    icon,
    iconSize = 20,
    variant = 'primary',
    size = 'medium',
    disabled = false,
    fullWidth = false,
    style,
    textStyle,
    haptic = 'light',
    children,
}) => {
    const scale = useSharedValue(1);
    const opacity = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => {
        return {
            transform: [
                { scale: withSpring(scale.value, { damping: 15, stiffness: 400 }) },
            ],
            opacity: opacity.value,
        };
    });

    const handlePressIn = () => {
        scale.value = 0.96;
        opacity.value = withTiming(0.85, { duration: 80 });
    };

    const handlePressOut = () => {
        scale.value = 1;
        opacity.value = withTiming(1, { duration: 120 });
    };

    const handlePress = () => {
        if (disabled) return;

        // Haptic feedback
        if (haptic !== 'none') {
            switch (haptic) {
                case 'light':
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    break;
                case 'medium':
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                    break;
                case 'heavy':
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
                    break;
            }
        }

        onPress();
    };

    const getButtonStyle = () => {
        const baseStyle: ViewStyle[] = [styles.button, styles[`${size}Button`]];

        switch (variant) {
            case 'primary':
                baseStyle.push(styles.primaryButton);
                break;
            case 'secondary':
                baseStyle.push(styles.secondaryButton);
                break;
            case 'ghost':
                baseStyle.push(styles.ghostButton);
                break;
            case 'danger':
                baseStyle.push(styles.dangerButton);
                break;
            case 'glass':
                baseStyle.push(styles.glassButton);
                break;
        }

        if (fullWidth) {
            baseStyle.push(styles.fullWidth);
        }

        if (disabled) {
            baseStyle.push(styles.disabled);
        }

        return baseStyle;
    };

    const getTextColor = () => {
        if (disabled) return colors.text.secondary;

        switch (variant) {
            case 'primary':
            case 'danger':
                return '#FFFFFF';
            case 'secondary':
            case 'ghost':
                return colors.text.primary;
            case 'glass':
                return colors.text.primary;
            default:
                return colors.text.primary;
        }
    };

    const content = (
        <>
            {icon && (
                <Ionicons
                    name={icon}
                    size={iconSize}
                    color={getTextColor()}
                    style={title ? styles.iconWithText : undefined}
                />
            )}
            {title && (
                <Text style={[styles.buttonText, styles[`${size}Text`], { color: getTextColor() }, textStyle]}>
                    {title}
                </Text>
            )}
            {children}
        </>
    );

    if (variant === 'glass') {
        return (
            <AnimatedTouchable
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                onPress={handlePress}
                activeOpacity={1}
                disabled={disabled}
                style={[animatedStyle, style]}
            >
                <BlurView intensity={80} tint="light" style={[...getButtonStyle(), styles.glassBlur]}>
                    <View style={styles.glassInner}>
                        {content}
                    </View>
                </BlurView>
            </AnimatedTouchable>
        );
    }

    return (
        <AnimatedTouchable
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={handlePress}
            activeOpacity={1}
            disabled={disabled}
            style={[animatedStyle, ...getButtonStyle(), style]}
        >
            {content}
        </AnimatedTouchable>
    );
};

// Icon-only button for headers, etc.
export const IconButton: React.FC<{
    icon: keyof typeof Ionicons.glyphMap;
    onPress: () => void;
    size?: number;
    color?: string;
    style?: ViewStyle;
    haptic?: 'light' | 'medium' | 'none';
}> = ({ icon, onPress, size = 24, color = colors.text.primary, style, haptic = 'light' }) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 20, stiffness: 500 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.85;
    };

    const handlePressOut = () => {
        scale.value = 1;
    };

    const handlePress = () => {
        if (haptic !== 'none') {
            Haptics.impactAsync(
                haptic === 'medium'
                    ? Haptics.ImpactFeedbackStyle.Medium
                    : Haptics.ImpactFeedbackStyle.Light
            );
        }
        onPress();
    };

    return (
        <AnimatedTouchable
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={handlePress}
            activeOpacity={1}
            style={[styles.iconButton, animatedStyle, style]}
        >
            <Ionicons name={icon} size={size} color={color} />
        </AnimatedTouchable>
    );
};

// Floating Action Button
export const FAB: React.FC<{
    icon: keyof typeof Ionicons.glyphMap;
    onPress: () => void;
    style?: ViewStyle;
}> = ({ icon, onPress, style }) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 12, stiffness: 300 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.9;
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    };

    const handlePressOut = () => {
        scale.value = 1;
    };

    return (
        <AnimatedTouchable
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={onPress}
            activeOpacity={1}
            style={[styles.fab, shadows.strong, animatedStyle, style]}
        >
            <Ionicons name={icon} size={28} color="#FFFFFF" />
        </AnimatedTouchable>
    );
};

// Chip/Tag button
export const ChipButton: React.FC<{
    title: string;
    isActive?: boolean;
    onPress: () => void;
    style?: ViewStyle;
}> = ({ title, isActive = false, onPress, style }) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 20, stiffness: 400 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.95;
    };

    const handlePressOut = () => {
        scale.value = 1;
    };

    const handlePress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
    };

    return (
        <AnimatedTouchable
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={handlePress}
            activeOpacity={1}
            style={[
                styles.chip,
                isActive && styles.chipActive,
                animatedStyle,
                style,
            ]}
        >
            <Text style={[styles.chipText, isActive && styles.chipTextActive]}>
                {title}
            </Text>
        </AnimatedTouchable>
    );
};

const styles = StyleSheet.create({
    button: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 14,
        ...shadows.soft,
    },

    // Sizes
    smallButton: {
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
    },
    mediumButton: {
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.l,
    },
    largeButton: {
        paddingVertical: spacing.l,
        paddingHorizontal: spacing.xl,
    },

    // Variants
    primaryButton: {
        backgroundColor: colors.text.primary,
    },
    secondaryButton: {
        backgroundColor: colors.surfaceHighlight,
        borderWidth: 1,
        borderColor: colors.border,
    },
    ghostButton: {
        backgroundColor: 'transparent',
    },
    dangerButton: {
        backgroundColor: colors.delete,
    },
    glassButton: {
        overflow: 'hidden',
    },
    glassBlur: {
        borderRadius: 14,
    },
    glassInner: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.l,
    },

    fullWidth: {
        width: '100%',
    },
    disabled: {
        opacity: 0.5,
    },

    // Text
    buttonText: {
        fontWeight: '600',
    },
    smallText: {
        fontSize: 13,
    },
    mediumText: {
        fontSize: 15,
    },
    largeText: {
        fontSize: 17,
    },

    iconWithText: {
        marginRight: spacing.s,
    },

    // Icon Button
    iconButton: {
        padding: spacing.s,
        borderRadius: 50,
    },

    // FAB
    fab: {
        position: 'absolute',
        bottom: 30,
        right: 20,
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: colors.text.primary,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Chip
    chip: {
        paddingVertical: spacing.s,
        paddingHorizontal: spacing.m,
        borderRadius: 100,
        borderWidth: 1,
        borderColor: colors.border,
        backgroundColor: colors.surface,
    },
    chipActive: {
        backgroundColor: colors.text.primary,
        borderColor: colors.text.primary,
    },
    chipText: {
        fontSize: 14,
        color: colors.text.primary,
        fontWeight: '500',
    },
    chipTextActive: {
        color: colors.surface,
    },
});

export default AnimatedButton;
