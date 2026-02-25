/**
 * WearLogButton — One-tap "Wearing this today" button
 *
 * Part of the behavioral loop: Suggest → Wear → Log → Learn
 * Logs the wear, increments streak, and provides haptic feedback.
 */

import React, { useState, useCallback } from 'react';
import { TouchableOpacity, Text, StyleSheet, View, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withSequence,
    withTiming,
} from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import useWardrobeStore from '../store/wardrobeStore';
import type { Occasion } from '../src/types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

interface WearLogButtonProps {
    itemIds: string[];
    outfitId?: string;
    occasion?: Occasion | string;
    weather?: { temp: number; condition: string };
    variant?: 'full' | 'compact';
    onLogged?: () => void;
}

const WearLogButton: React.FC<WearLogButtonProps> = ({
    itemIds,
    outfitId,
    occasion,
    weather,
    variant = 'full',
    onLogged,
}) => {
    const [isLogged, setIsLogged] = useState(false);
    const [isLogging, setIsLogging] = useState(false);
    const scale = useSharedValue(1);

    const logWear = useWardrobeStore((state) => state.logWear);
    const streak = useWardrobeStore((state) => state.streak);

    const handlePress = useCallback(async () => {
        if (isLogged || isLogging) return;

        setIsLogging(true);

        // Haptic feedback
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        // Bounce animation
        scale.value = withSequence(
            withSpring(0.9, { damping: 15, stiffness: 400 }),
            withSpring(1.1, { damping: 15, stiffness: 400 }),
            withSpring(1, { damping: 15, stiffness: 400 })
        );

        // Log the wear
        logWear(itemIds, occasion, weather);

        setIsLogging(false);
        setIsLogged(true);
        onLogged?.();
    }, [itemIds, occasion, weather, isLogged, isLogging, logWear, onLogged, scale]);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    if (variant === 'compact') {
        return (
            <Animated.View style={animatedStyle}>
                <TouchableOpacity
                    style={[styles.compactButton, isLogged && styles.loggedButton]}
                    onPress={handlePress}
                    disabled={isLogged}
                    activeOpacity={0.7}
                >
                    <Ionicons
                        name={isLogged ? 'checkmark-circle' : 'shirt-outline'}
                        size={20}
                        color={isLogged ? '#FFF' : colors.text.primary}
                    />
                </TouchableOpacity>
            </Animated.View>
        );
    }

    return (
        <Animated.View style={animatedStyle}>
            <TouchableOpacity
                style={[styles.fullButton, isLogged && styles.loggedButton]}
                onPress={handlePress}
                disabled={isLogged}
                activeOpacity={0.8}
            >
                {isLogging ? (
                    <ActivityIndicator size="small" color="#FFF" />
                ) : (
                    <>
                        <Ionicons
                            name={isLogged ? 'checkmark-circle' : 'shirt-outline'}
                            size={22}
                            color={isLogged ? '#FFF' : colors.text.primary}
                        />
                        <Text style={[styles.buttonText, isLogged && styles.loggedText]}>
                            {isLogged
                                ? `Logged! 🔥 ${streak + 1} day streak`
                                : 'Wearing this today'}
                        </Text>
                    </>
                )}
            </TouchableOpacity>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    fullButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.glass.frosted,
        paddingVertical: spacing.md,
        paddingHorizontal: spacing.lg,
        borderRadius: radius.pill,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.sm,
    },
    compactButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: colors.glass.frosted,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: colors.border.glass,
    },
    loggedButton: {
        backgroundColor: '#22C55E',
        borderColor: '#22C55E',
    },
    buttonText: {
        ...typography.scale.labelLarge,
        color: colors.text.primary,
        fontWeight: '600',
    },
    loggedText: {
        color: '#FFF',
    },
});

export default WearLogButton;
