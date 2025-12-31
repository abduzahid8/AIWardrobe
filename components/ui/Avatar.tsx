// Alta Daily-inspired Floating Avatar Component
// Features: 3D floating animation, glow effect, personalization

import React, { useEffect } from 'react';
import { View, Image, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withSequence,
    withTiming,
    Easing,
    interpolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../src/hooks/useTheme';

interface AvatarProps {
    /** Avatar image URI */
    imageUri?: string;
    /** Size variant */
    size?: 'small' | 'medium' | 'large' | 'xlarge';
    /** Enable floating animation */
    floating?: boolean;
    /** Enable glow effect */
    glow?: boolean;
    /** Fallback icon/text */
    fallback?: React.ReactNode;
    /** Custom style */
    style?: ViewStyle;
}

const SIZES = {
    small: 32,
    medium: 48,
    large: 64,
    xlarge: 96,
};

export const Avatar: React.FC<AvatarProps> = ({
    imageUri,
    size = 'medium',
    floating = false,
    glow = false,
    fallback,
    style,
}) => {
    const { colors } = useTheme();

    // Floating animation
    const floatAnim = useSharedValue(0);

    // Glow animation
    const glowAnim = useSharedValue(0);

    useEffect(() => {
        if (floating) {
            floatAnim.value = withRepeat(
                withSequence(
                    withTiming(1, {
                        duration: 3000,
                        easing: Easing.inOut(Easing.sin),
                    }),
                    withTiming(0, {
                        duration: 3000,
                        easing: Easing.inOut(Easing.sin),
                    })
                ),
                -1, // infinite
                false
            );
        }

        if (glow) {
            glowAnim.value = withRepeat(
                withSequence(
                    withTiming(1, {
                        duration: 2000,
                        easing: Easing.inOut(Easing.sin),
                    }),
                    withTiming(0, {
                        duration: 2000,
                        easing: Easing.inOut(Easing.sin),
                    })
                ),
                -1,
                false
            );
        }
    }, [floating, glow]);

    const animatedFloatStyle = useAnimatedStyle(() => {
        const translateY = interpolate(
            floatAnim.value,
            [0, 1],
            [0, -6]
        );

        return {
            transform: [{ translateY }],
        };
    });

    const animatedGlowStyle = useAnimatedStyle(() => {
        const opacity = interpolate(
            glowAnim.value,
            [0, 1],
            [0.2, 0.6]
        );

        return {
            opacity,
        };
    });

    const avatarSize = SIZES[size];

    return (
        <Animated.View
            style={[
                styles.container,
                { width: avatarSize, height: avatarSize, borderRadius: avatarSize / 2 },
                floating && animatedFloatStyle,
                style,
            ]}
        >
            {glow && (
                <Animated.View
                    style={[
                        StyleSheet.absoluteFill,
                        styles.glowContainer,
                        { borderRadius: avatarSize / 2 },
                        animatedGlowStyle,
                    ]}
                >
                    <LinearGradient
                        colors={[colors.primary, colors.primaryDark]}
                        style={[StyleSheet.absoluteFill, { borderRadius: avatarSize / 2 }]}
                    />
                </Animated.View>
            )}

            <View
                style={[
                    styles.imageContainer,
                    {
                        width: avatarSize,
                        height: avatarSize,
                        borderRadius: avatarSize / 2,
                        backgroundColor: colors.surfaceHighlight,
                    },
                ]}
            >
                {imageUri ? (
                    <Image
                        source={{ uri: imageUri }}
                        style={[
                            styles.image,
                            { width: avatarSize, height: avatarSize, borderRadius: avatarSize / 2 },
                        ]}
                        resizeMode="cover"
                    />
                ) : (
                    <View style={styles.fallbackContainer}>{fallback}</View>
                )}
            </View>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'relative',
    },
    glowContainer: {
        position: 'absolute',
        top: -4,
        left: -4,
        right: -4,
        bottom: -4,
    },
    imageContainer: {
        overflow: 'hidden',
        justifyContent: 'center',
        alignItems: 'center',
    },
    image: {
        width: '100%',
        height: '100%',
    },
    fallbackContainer: {
        width: '100%',
        height: '100%',
        justifyContent: 'center',
        alignItems: 'center',
    },
});
