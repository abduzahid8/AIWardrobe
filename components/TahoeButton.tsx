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
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';

// iOS 26 Tahoe Colors
const TAHOE = {
    glass: 'rgba(255, 255, 255, 0.18)',
    glassBorder: 'rgba(255, 255, 255, 0.3)',
    glassDark: 'rgba(0, 0, 0, 0.06)',
    primary: '#007AFF',
    secondary: '#8E8E93',
    success: '#34C759',
    danger: '#FF3B30',
    warning: '#FF9500',
    text: '#1C1C1E',
    textSecondary: '#8E8E93',
    textLight: '#FFFFFF',
    gradientStart: '#007AFF',
    gradientEnd: '#5856D6',
};

interface TahoeButtonProps {
    onPress: () => void;
    title?: string;
    icon?: keyof typeof Ionicons.glyphMap;
    iconPosition?: 'left' | 'right';
    iconSize?: number;
    variant?: 'glass' | 'solid' | 'outline' | 'gradient' | 'ghost';
    size?: 'small' | 'medium' | 'large';
    color?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning';
    disabled?: boolean;
    fullWidth?: boolean;
    style?: ViewStyle;
    textStyle?: TextStyle;
    haptic?: 'light' | 'medium' | 'heavy' | 'none';
    children?: React.ReactNode;
}

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

export const TahoeButton: React.FC<TahoeButtonProps> = ({
    onPress,
    title,
    icon,
    iconPosition = 'left',
    iconSize = 20,
    variant = 'glass',
    size = 'medium',
    color = 'primary',
    disabled = false,
    fullWidth = false,
    style,
    textStyle,
    haptic = 'light',
    children,
}) => {
    const scale = useSharedValue(1);
    const opacity = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [
            { scale: withSpring(scale.value, { damping: 20, stiffness: 400 }) },
        ],
        opacity: opacity.value,
    }));

    const handlePressIn = () => {
        scale.value = 0.96;
        opacity.value = withTiming(0.85, { duration: 60 });
    };

    const handlePressOut = () => {
        scale.value = 1;
        opacity.value = withTiming(1, { duration: 100 });
    };

    const handlePress = () => {
        if (disabled) return;

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

    const getColorValue = () => {
        switch (color) {
            case 'primary': return TAHOE.primary;
            case 'secondary': return TAHOE.secondary;
            case 'success': return TAHOE.success;
            case 'danger': return TAHOE.danger;
            case 'warning': return TAHOE.warning;
            default: return TAHOE.primary;
        }
    };

    const getSizeStyles = (): ViewStyle => {
        switch (size) {
            case 'small':
                return { paddingVertical: 8, paddingHorizontal: 14 };
            case 'large':
                return { paddingVertical: 18, paddingHorizontal: 28 };
            default:
                return { paddingVertical: 14, paddingHorizontal: 22 };
        }
    };

    const getTextSize = () => {
        switch (size) {
            case 'small': return 14;
            case 'large': return 18;
            default: return 16;
        }
    };

    const getTextColor = () => {
        if (disabled) return TAHOE.textSecondary;

        switch (variant) {
            case 'solid':
            case 'gradient':
                return TAHOE.textLight;
            case 'glass':
            case 'outline':
            case 'ghost':
                return getColorValue();
            default:
                return TAHOE.text;
        }
    };

    const iconComponent = icon && (
        <Ionicons
            name={icon}
            size={iconSize}
            color={getTextColor()}
            style={title ? (iconPosition === 'left' ? styles.iconLeft : styles.iconRight) : undefined}
        />
    );

    const content = (
        <View style={styles.contentRow}>
            {iconPosition === 'left' && iconComponent}
            {title && (
                <Text style={[
                    styles.buttonText,
                    { fontSize: getTextSize(), color: getTextColor() },
                    textStyle
                ]}>
                    {title}
                </Text>
            )}
            {iconPosition === 'right' && iconComponent}
            {children}
        </View>
    );

    // Glass variant with blur
    if (variant === 'glass') {
        return (
            <AnimatedTouchable
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                onPress={handlePress}
                activeOpacity={1}
                disabled={disabled}
                style={[
                    animatedStyle,
                    fullWidth && styles.fullWidth,
                    disabled && styles.disabled,
                    style
                ]}
            >
                <BlurView
                    intensity={Platform.OS === 'ios' ? 60 : 100}
                    tint="light"
                    style={[
                        styles.glassContainer,
                        getSizeStyles(),
                        { borderColor: TAHOE.glassBorder }
                    ]}
                >
                    {content}
                </BlurView>
            </AnimatedTouchable>
        );
    }

    // Gradient variant
    if (variant === 'gradient') {
        return (
            <AnimatedTouchable
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                onPress={handlePress}
                activeOpacity={1}
                disabled={disabled}
                style={[
                    animatedStyle,
                    fullWidth && styles.fullWidth,
                    disabled && styles.disabled,
                    style
                ]}
            >
                <LinearGradient
                    colors={[TAHOE.gradientStart, TAHOE.gradientEnd]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[styles.gradientContainer, getSizeStyles()]}
                >
                    {content}
                </LinearGradient>
            </AnimatedTouchable>
        );
    }

    // Other variants (solid, outline, ghost)
    const getVariantStyles = (): ViewStyle => {
        switch (variant) {
            case 'solid':
                return {
                    backgroundColor: getColorValue(),
                    borderWidth: 0,
                };
            case 'outline':
                return {
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderColor: getColorValue(),
                };
            case 'ghost':
                return {
                    backgroundColor: 'transparent',
                    borderWidth: 0,
                };
            default:
                return {};
        }
    };

    return (
        <AnimatedTouchable
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={handlePress}
            activeOpacity={1}
            disabled={disabled}
            style={[
                styles.button,
                getSizeStyles(),
                getVariantStyles(),
                fullWidth && styles.fullWidth,
                disabled && styles.disabled,
                animatedStyle,
                style
            ]}
        >
            {content}
        </AnimatedTouchable>
    );
};

// Tahoe Icon Button for headers
interface TahoeIconButtonProps {
    icon: keyof typeof Ionicons.glyphMap;
    onPress: () => void;
    size?: number;
    color?: string;
    variant?: 'glass' | 'ghost';
    style?: ViewStyle;
}

export const TahoeIconButton: React.FC<TahoeIconButtonProps> = ({
    icon,
    onPress,
    size = 24,
    color = TAHOE.text,
    variant = 'glass',
    style,
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 18, stiffness: 450 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.88;
    };

    const handlePressOut = () => {
        scale.value = 1;
    };

    const handlePress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
    };

    if (variant === 'glass') {
        return (
            <AnimatedTouchable
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                onPress={handlePress}
                activeOpacity={1}
                style={[styles.iconButtonContainer, animatedStyle, style]}
            >
                <BlurView
                    intensity={Platform.OS === 'ios' ? 50 : 80}
                    tint="light"
                    style={styles.iconButtonGlass}
                >
                    <Ionicons name={icon} size={size} color={color} />
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
            style={[styles.iconButtonGhost, animatedStyle, style]}
        >
            <Ionicons name={icon} size={size} color={color} />
        </AnimatedTouchable>
    );
};

// Tahoe Action Card for quick actions
interface TahoeActionCardProps {
    icon: keyof typeof Ionicons.glyphMap;
    title: string;
    subtitle?: string;
    iconColor?: string;
    onPress: () => void;
    style?: ViewStyle;
}

export const TahoeActionCard: React.FC<TahoeActionCardProps> = ({
    icon,
    title,
    subtitle,
    iconColor = TAHOE.primary,
    onPress,
    style,
}) => {
    const scale = useSharedValue(1);
    const shadowOpacity = useSharedValue(0.08);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 18, stiffness: 380 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.97;
        shadowOpacity.value = withTiming(0.15, { duration: 100 });
    };

    const handlePressOut = () => {
        scale.value = 1;
        shadowOpacity.value = withTiming(0.08, { duration: 150 });
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
            style={[styles.actionCardContainer, animatedStyle, style]}
        >
            <BlurView
                intensity={Platform.OS === 'ios' ? 40 : 60}
                tint="light"
                style={styles.actionCardBlur}
            >
                <View style={[styles.actionIconContainer, { backgroundColor: `${iconColor}15` }]}>
                    <Ionicons name={icon} size={24} color={iconColor} />
                </View>
                <Text style={styles.actionTitle}>{title}</Text>
                {subtitle && <Text style={styles.actionSubtitle}>{subtitle}</Text>}
            </BlurView>
        </AnimatedTouchable>
    );
};

// Tahoe Chip Button
interface TahoeChipProps {
    title: string;
    isActive?: boolean;
    onPress: () => void;
    style?: ViewStyle;
}

export const TahoeChip: React.FC<TahoeChipProps> = ({
    title,
    isActive = false,
    onPress,
    style,
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 20, stiffness: 400 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.94;
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
                style
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
        borderRadius: 16,
        overflow: 'hidden',
    },
    contentRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
    },
    buttonText: {
        fontWeight: '600',
        letterSpacing: -0.2,
    },
    iconLeft: {
        marginRight: 8,
    },
    iconRight: {
        marginLeft: 8,
    },
    fullWidth: {
        width: '100%',
    },
    disabled: {
        opacity: 0.5,
    },

    // Glass variant
    glassContainer: {
        borderRadius: 16,
        borderWidth: 1,
        backgroundColor: TAHOE.glass,
        overflow: 'hidden',
    },

    // Gradient variant
    gradientContainer: {
        borderRadius: 16,
        overflow: 'hidden',
    },

    // Icon Button
    iconButtonContainer: {
        borderRadius: 12,
        overflow: 'hidden',
    },
    iconButtonGlass: {
        width: 44,
        height: 44,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: TAHOE.glass,
        borderWidth: 1,
        borderColor: TAHOE.glassBorder,
        overflow: 'hidden',
    },
    iconButtonGhost: {
        width: 44,
        height: 44,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Action Card
    actionCardContainer: {
        borderRadius: 20,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 12,
        elevation: 4,
    },
    actionCardBlur: {
        padding: 20,
        backgroundColor: TAHOE.glass,
        borderWidth: 1,
        borderColor: TAHOE.glassBorder,
        borderRadius: 20,
        overflow: 'hidden',
    },
    actionIconContainer: {
        width: 48,
        height: 48,
        borderRadius: 14,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    actionTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: TAHOE.text,
        marginBottom: 4,
    },
    actionSubtitle: {
        fontSize: 13,
        color: TAHOE.textSecondary,
    },

    // Chip
    chip: {
        paddingVertical: 10,
        paddingHorizontal: 18,
        borderRadius: 100,
        backgroundColor: TAHOE.glass,
        borderWidth: 1,
        borderColor: TAHOE.glassBorder,
        marginRight: 10,
    },
    chipActive: {
        backgroundColor: TAHOE.primary,
        borderColor: TAHOE.primary,
    },
    chipText: {
        fontSize: 14,
        fontWeight: '500',
        color: TAHOE.text,
    },
    chipTextActive: {
        color: TAHOE.textLight,
    },
});

export default TahoeButton;
