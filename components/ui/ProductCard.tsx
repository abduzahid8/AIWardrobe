// SurfPad-inspired Product Card Component
// Features: 3D hover effect, professional image display, quick actions, glass variant

import React from 'react';
import {
    View,
    Image,
    Text,
    StyleSheet,
    Pressable,
    ViewStyle,
    Dimensions,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
    Easing,
    interpolate,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../src/hooks/useTheme';
import { springFastConfig } from '../../src/theme/animations';

const { width } = Dimensions.get('window');

interface ProductCardProps {
    /** Product image URI */
    imageUri: string;
    /** Product name/title */
    title?: string;
    /** Product category */
    category?: string;
    /** Brand name */
    brand?: string;
    /** Whether this item is favorited */
    isFavorite?: boolean;
    /** Callback when card is pressed */
    onPress?: () => void;
    /** Callback when favorite button is pressed */
    onFavorite?: () => void;
    /** Callback when long pressed */
    onLongPress?: () => void;
    /** Enable 3D hover effect */
    enable3D?: boolean;
    /** Use glass variant */
    glass?: boolean;
    /** Card width (defaults to 1/3 screen minus padding) */
    width?: number;
    /** Custom style */
    style?: ViewStyle;
}

export const ProductCard: React.FC<ProductCardProps> = ({
    imageUri,
    title,
    category,
    brand,
    isFavorite = false,
    onPress,
    onFavorite,
    onLongPress,
    enable3D = true,
    glass = false,
    width: customWidth,
    style,
}) => {
    const { colors, spacing, borderRadius, shadows } = useTheme();

    const pressed = useSharedValue(false);
    const hovered = useSharedValue(0);

    const handlePressIn = () => {
        pressed.value = true;
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    };

    const handlePressOut = () => {
        pressed.value = false;
    };

    const handleFavorite = () => {
        if (onFavorite) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            onFavorite();
        }
    };

    // Animated styles for press/hover
    const animatedCardStyle = useAnimatedStyle(() => {
        const scale = pressed.value ? 0.97 : 1;
        const elevation = interpolate(hovered.value, [0, 1], [0, 4]);

        return {
            transform: [{ scale: withSpring(scale, springFastConfig) }],
            shadowOpacity: withTiming(pressed.value ? 0.04 : 0.08, {
                duration: 150,
            }),
        };
    });

    // 3D tilt effect
    const animated3DStyle = useAnimatedStyle(() => {
        if (!enable3D) return {};

        const rotateX = interpolate(hovered.value, [0, 1], [0, -2]);
        const rotateY = interpolate(hovered.value, [0, 1], [0, 2]);

        return {
            transform: [
                { perspective: 1000 },
                { rotateX: `${rotateX}deg` },
                { rotateY: `${rotateY}deg` },
            ],
        };
    });

    const cardWidth = customWidth || (width - 48) / 3; // 3 columns with padding

    return (
        <Pressable
            onPress={onPress}
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onLongPress={onLongPress}
            style={[{ width: cardWidth, padding: spacing.xs }, style]}
        >
            <Animated.View
                style={[
                    styles.card,
                    {
                        backgroundColor: glass ? 'rgba(255,255,255,0.8)' : colors.surface,
                        borderRadius: borderRadius.m,
                        ...shadows.card,
                    },
                    animatedCardStyle,
                    animated3DStyle,
                ]}
            >
                {/* Image Container */}
                <View
                    style={[
                        styles.imageContainer,
                        { borderRadius: borderRadius.m, backgroundColor: colors.surfaceHighlight },
                    ]}
                >
                    <Image
                        source={{ uri: imageUri }}
                        style={styles.image}
                        resizeMode="cover"
                    />

                    {/* Favorite Button */}
                    {onFavorite && (
                        <Pressable
                            onPress={handleFavorite}
                            style={[styles.favoriteButton, { backgroundColor: colors.surface }]}
                            hitSlop={{ top: 10, right: 10, bottom: 10, left: 10 }}
                        >
                            <Ionicons
                                name={isFavorite ? 'heart' : 'heart-outline'}
                                size={18}
                                color={isFavorite ? colors.favorite : colors.text.secondary}
                            />
                        </Pressable>
                    )}
                </View>

                {/* Info Section */}
                {(title || category || brand) && (
                    <View style={[styles.info, { padding: spacing.s }]}>
                        {brand && (
                            <Text
                                style={[styles.brand, { color: colors.text.muted, fontSize: 10 }]}
                                numberOfLines={1}
                            >
                                {brand.toUpperCase()}
                            </Text>
                        )}
                        {title && (
                            <Text
                                style={[styles.title, { color: colors.text.primary, fontSize: 13 }]}
                                numberOfLines={1}
                            >
                                {title}
                            </Text>
                        )}
                        {category && (
                            <Text
                                style={[styles.category, { color: colors.text.secondary, fontSize: 11 }]}
                                numberOfLines={1}
                            >
                                {category}
                            </Text>
                        )}
                    </View>
                )}
            </Animated.View>
        </Pressable>
    );
};

const styles = StyleSheet.create({
    card: {
        overflow: 'hidden',
    },
    imageContainer: {
        width: '100%',
        aspectRatio: 0.75, // Portrait ratio for clothing
        overflow: 'hidden',
        position: 'relative',
    },
    image: {
        width: '100%',
        height: '100%',
    },
    favoriteButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 32,
        height: 32,
        borderRadius: 16,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    info: {
        gap: 2,
    },
    brand: {
        fontWeight: '600',
        letterSpacing: 0.5,
    },
    title: {
        fontWeight: '600',
    },
    category: {
        fontWeight: '400',
    },
});
