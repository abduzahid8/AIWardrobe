import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    interpolateColor,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { useTheme } from '../../src/theme/ThemeContext';

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

interface ActionCardProps {
    icon: keyof typeof Ionicons.glyphMap;
    title: string;
    subtitle?: string;
    onPress: () => void;
    color?: string;
    variant?: 'default' | 'gradient' | 'outline';
    rightElement?: React.ReactNode;
}

/**
 * ActionCard - Premium action card with animations
 * 
 * Features:
 * - Spring press animation
 * - Icon with color tint
 * - Subtitle support
 * - Multiple variants
 */
export const ActionCard: React.FC<ActionCardProps> = ({
    icon,
    title,
    subtitle,
    onPress,
    color,
    variant = 'default',
    rightElement,
}) => {
    const { colors, isDark } = useTheme();
    const scale = useSharedValue(1);
    const pressed = useSharedValue(0);

    const iconColor = color || colors.primary;

    const animStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const handlePressIn = () => {
        scale.value = withSpring(0.97);
        pressed.value = withSpring(1);
    };

    const handlePressOut = () => {
        scale.value = withSpring(1);
        pressed.value = withSpring(0);
    };

    const handlePress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
    };

    return (
        <AnimatedTouchable
            onPress={handlePress}
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            activeOpacity={1}
            style={[animStyle]}
        >
            <View
                style={[
                    styles.card,
                    {
                        backgroundColor: variant === 'outline'
                            ? 'transparent'
                            : (isDark ? colors.surface : '#FFFFFF'),
                        borderColor: variant === 'outline'
                            ? colors.border
                            : 'transparent',
                        borderWidth: variant === 'outline' ? 1.5 : 0,
                    }
                ]}
            >
                {/* Icon */}
                <View
                    style={[
                        styles.iconContainer,
                        { backgroundColor: `${iconColor}15` }
                    ]}
                >
                    <Ionicons name={icon} size={22} color={iconColor} />
                </View>

                {/* Content */}
                <View style={styles.content}>
                    <Text
                        style={[styles.title, { color: colors.text.primary }]}
                        numberOfLines={1}
                    >
                        {title}
                    </Text>
                    {subtitle && (
                        <Text
                            style={[styles.subtitle, { color: colors.text.secondary }]}
                            numberOfLines={1}
                        >
                            {subtitle}
                        </Text>
                    )}
                </View>

                {/* Right Element or Arrow */}
                {rightElement || (
                    <Ionicons
                        name="chevron-forward"
                        size={20}
                        color={colors.text.muted}
                    />
                )}
            </View>
        </AnimatedTouchable>
    );
};

const styles = StyleSheet.create({
    card: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        borderRadius: 16,
        marginBottom: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.04,
        shadowRadius: 4,
        elevation: 1,
    },
    iconContainer: {
        width: 44,
        height: 44,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 14,
    },
    content: {
        flex: 1,
    },
    title: {
        fontSize: 16,
        fontWeight: '600',
        marginBottom: 2,
    },
    subtitle: {
        fontSize: 13,
    },
});

export default ActionCard;
