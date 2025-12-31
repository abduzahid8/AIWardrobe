import React from 'react';
import { Text, StyleSheet, Pressable, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../src/theme/ThemeContext';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

interface SuggestionChipProps {
    icon?: keyof typeof Ionicons.glyphMap;
    text: string;
    onPress: () => void;
    variant?: 'primary' | 'secondary' | 'outline';
    disabled?: boolean;
}

export const SuggestionChip: React.FC<SuggestionChipProps> = ({
    icon,
    text,
    onPress,
    variant = 'outline',
    disabled = false,
}) => {
    const { colors, isDark } = useTheme();
    const scale = useSharedValue(1);

    const handlePressIn = () => {
        scale.value = withSpring(0.95);
    };

    const handlePressOut = () => {
        scale.value = withSpring(1);
    };

    const handlePress = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
    };

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const getVariantStyles = () => {
        switch (variant) {
            case 'primary':
                return {
                    backgroundColor: colors.primary,
                    borderColor: colors.primary,
                    textColor: '#FFFFFF',
                };
            case 'secondary':
                return {
                    backgroundColor: colors.surfaceHighlight,
                    borderColor: colors.surfaceHighlight,
                    textColor: colors.text.primary,
                };
            case 'outline':
            default:
                return {
                    backgroundColor: 'transparent',
                    borderColor: colors.border,
                    textColor: colors.text.secondary,
                };
        }
    };

    const variantStyles = getVariantStyles();

    return (
        <AnimatedPressable
            onPress={handlePress}
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            disabled={disabled}
            style={[animatedStyle]}
        >
            <View
                style={[
                    styles.chip,
                    {
                        backgroundColor: variantStyles.backgroundColor,
                        borderColor: variantStyles.borderColor,
                        opacity: disabled ? 0.5 : 1,
                    },
                ]}
            >
                {icon && (
                    <Ionicons
                        name={icon}
                        size={16}
                        color={variantStyles.textColor}
                        style={styles.icon}
                    />
                )}
                <Text
                    style={[
                        styles.text,
                        { color: variantStyles.textColor },
                    ]}
                    numberOfLines={1}
                >
                    {text}
                </Text>
            </View>
        </AnimatedPressable>
    );
};

const styles = StyleSheet.create({
    chip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1.5,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,
    },
    icon: {
        marginRight: 6,
    },
    text: {
        fontSize: 14,
        fontWeight: '600',
        letterSpacing: 0.2,
    },
});
