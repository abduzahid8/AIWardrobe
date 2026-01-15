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
import AppColors from '../constants/AppColors';

// Strict Black & White Theme
const BW = {
    black: '#000000',
    white: '#FFFFFF',
    gray: '#8E8E93',
    lightGray: '#F5F5F7',
    border: '#EBEBEB',
    glass: 'rgba(255, 255, 255, 0.9)',
    glassBorder: 'rgba(0, 0, 0, 0.08)',
};

interface TahoeButtonProps {
    onPress: () => void;
    title?: string;
    icon?: keyof typeof Ionicons.glyphMap;
    iconPosition?: 'left' | 'right';
    iconSize?: number;
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'glass';
    size?: 'small' | 'medium' | 'large';
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

    const getSizeStyles = (): ViewStyle => {
        switch (size) {
            case 'small':
                return { paddingVertical: 10, paddingHorizontal: 16 };
            case 'large':
                return { paddingVertical: 18, paddingHorizontal: 28 };
            default:
                return { paddingVertical: 14, paddingHorizontal: 22 };
        }
    };

    const getTextSize = () => {
        switch (size) {
            case 'small': return 14;
            case 'large': return 17;
            default: return 16;
        }
    };

    const getTextColor = () => {
        if (disabled) return BW.gray;

        switch (variant) {
            case 'primary':
                return BW.white;
            case 'secondary':
                return BW.black;
            case 'outline':
            case 'ghost':
            case 'glass':
                return BW.black;
            default:
                return BW.black;
        }
    };

    const getVariantStyles = (): ViewStyle => {
        switch (variant) {
            case 'primary':
                return {
                    backgroundColor: BW.black,
                    borderWidth: 0,
                };
            case 'secondary':
                return {
                    backgroundColor: BW.lightGray,
                    borderWidth: 0,
                };
            case 'outline':
                return {
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderColor: BW.black,
                };
            case 'ghost':
                return {
                    backgroundColor: 'transparent',
                    borderWidth: 0,
                };
            case 'glass':
                return {
                    backgroundColor: BW.glass,
                    borderWidth: 1,
                    borderColor: BW.glassBorder,
                };
            default:
                return {};
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
                        { borderColor: BW.glassBorder }
                    ]}
                >
                    {content}
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

// Tahoe Icon Button for headers - Black & White
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
    color = BW.black,
    variant = 'ghost',
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

// Tahoe Action Card - Black & White theme
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
    iconColor = BW.black,
    onPress,
    style,
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 18, stiffness: 380 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.97;
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
            style={[styles.actionCardContainer, animatedStyle, style]}
        >
            <View style={styles.actionCardInner}>
                <View style={styles.actionIconContainer}>
                    <Ionicons name={icon} size={24} color={BW.black} />
                </View>
                <Text style={styles.actionTitle}>{title}</Text>
                {subtitle && <Text style={styles.actionSubtitle}>{subtitle}</Text>}
            </View>
        </AnimatedTouchable>
    );
};

// Tahoe Chip Button - Black & White
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
        borderRadius: 14,
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
        opacity: 0.4,
    },

    // Glass variant
    glassContainer: {
        borderRadius: 14,
        borderWidth: 1,
        backgroundColor: BW.glass,
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
        backgroundColor: BW.glass,
        borderWidth: 1,
        borderColor: BW.glassBorder,
        overflow: 'hidden',
    },
    iconButtonGhost: {
        width: 44,
        height: 44,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Action Card - Clean B&W
    actionCardContainer: {
        borderRadius: 18,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 8,
        elevation: 3,
    },
    actionCardInner: {
        padding: 18,
        backgroundColor: BW.white,
        borderWidth: 1,
        borderColor: BW.border,
        borderRadius: 18,
    },
    actionIconContainer: {
        width: 44,
        height: 44,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
        backgroundColor: BW.lightGray,
    },
    actionTitle: {
        fontSize: 15,
        fontWeight: '600',
        color: BW.black,
        marginBottom: 4,
    },
    actionSubtitle: {
        fontSize: 13,
        color: BW.gray,
    },

    // Chip - Black & White
    chip: {
        paddingVertical: 10,
        paddingHorizontal: 18,
        borderRadius: 100,
        backgroundColor: BW.white,
        borderWidth: 1,
        borderColor: BW.border,
        marginRight: 10,
    },
    chipActive: {
        backgroundColor: BW.black,
        borderColor: BW.black,
    },
    chipText: {
        fontSize: 14,
        fontWeight: '500',
        color: BW.black,
    },
    chipTextActive: {
        color: BW.white,
    },
});

export default TahoeButton;
