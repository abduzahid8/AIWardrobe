import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    ViewStyle,
} from 'react-native';
import Animated, {
    useAnimatedStyle,
    interpolate,
    SharedValue,
} from 'react-native-reanimated';
import { ClosetlyTheme } from '../../constants/ClosetlyTheme';
import { CachedImage } from './CachedImage';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface ClothingItem {
    _id: string;
    id?: string;
    type?: string;
    itemType?: string;
    category?: string;
    color?: string;
    colorHex?: string;
    imageUrl?: string;
    image?: string;
}

interface ClothingCardProps {
    item: ClothingItem;
    index: number;
    scrollX: SharedValue<number>;
    itemWidth: number;
    matchScore?: number;
    isSelected?: boolean;
    style?: ViewStyle;
}

/**
 * ClothingCard - Premium styled clothing item card
 * Follows Closetly "Invisible UI" aesthetic with:
 * - #F1F1F1 background
 * - 32px rounded corners
 * - Soft diffused shadows
 * - Parallax inner image movement
 * - Optional match percentage badge
 */
const ClothingCard: React.FC<ClothingCardProps> = ({
    item,
    index,
    scrollX,
    itemWidth,
    matchScore,
    isSelected = false,
    style,
}) => {
    const imageUrl = item.imageUrl || item.image;
    const itemLabel = item.type || item.itemType || item.category || 'Item';

    // Parallax animation - image moves faster than card
    const parallaxStyle = useAnimatedStyle(() => {
        const inputRange = [
            (index - 1) * itemWidth,
            index * itemWidth,
            (index + 1) * itemWidth,
        ];

        // Image moves 20% faster than the scroll
        const translateX = interpolate(
            scrollX.value,
            inputRange,
            [-15, 0, 15],
            'clamp'
        );

        // Subtle scale effect for depth
        const scale = interpolate(
            scrollX.value,
            inputRange,
            [0.95, 1, 0.95],
            'clamp'
        );

        return {
            transform: [{ translateX }, { scale }],
        };
    });

    // Card opacity based on distance from center
    const cardStyle = useAnimatedStyle(() => {
        const inputRange = [
            (index - 1) * itemWidth,
            index * itemWidth,
            (index + 1) * itemWidth,
        ];

        const opacity = interpolate(
            scrollX.value,
            inputRange,
            [0.7, 1, 0.7],
            'clamp'
        );

        return {
            opacity,
        };
    });

    return (
        <Animated.View style={[styles.cardContainer, { width: itemWidth }, cardStyle, style]}>
            <View style={[styles.card, isSelected && styles.cardSelected]}>
                {/* Parallax Image Container */}
                <View style={styles.imageContainer}>
                    <Animated.View style={[styles.imageWrapper, parallaxStyle]}>
                        {imageUrl ? (
                            <CachedImage
                                uri={imageUrl}
                                style={styles.image}
                                contentFit="cover"
                                fadeIn={false}
                            />
                        ) : (
                            <View style={styles.placeholderImage}>
                                <Text style={styles.placeholderText}>👕</Text>
                            </View>
                        )}
                    </Animated.View>
                </View>

                {/* Item Label */}
                <View style={styles.labelContainer}>
                    <Text style={styles.label} numberOfLines={1}>
                        {itemLabel}
                    </Text>
                    {item.color && (
                        <View style={styles.colorRow}>
                            {item.colorHex && (
                                <View style={[styles.colorDot, { backgroundColor: item.colorHex }]} />
                            )}
                            <Text style={styles.colorText}>{item.color}</Text>
                        </View>
                    )}
                </View>

                {/* Match Score Badge */}
                {matchScore !== undefined && matchScore > 0 && (
                    <View style={styles.matchBadge}>
                        <Text style={styles.matchText}>{matchScore}%</Text>
                    </View>
                )}

                {/* Selected Indicator */}
                {isSelected && (
                    <View style={styles.selectedIndicator}>
                        <Text style={styles.selectedCheck}>✓</Text>
                    </View>
                )}
            </View>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    cardContainer: {
        paddingHorizontal: 8,
        paddingVertical: 12,
    },
    card: {
        backgroundColor: ClosetlyTheme.colors.card,
        borderRadius: ClosetlyTheme.borderRadius.card,
        overflow: 'hidden',
        ...ClosetlyTheme.shadows.card,
    },
    cardSelected: {
        borderWidth: 2,
        borderColor: ClosetlyTheme.colors.text,
    },
    imageContainer: {
        width: '100%',
        height: 140,
        overflow: 'hidden',
        backgroundColor: ClosetlyTheme.colors.background,
    },
    imageWrapper: {
        width: '120%',  // Wider for parallax movement
        height: '100%',
        marginLeft: '-10%',
    },
    image: {
        width: '100%',
        height: '100%',
    },
    placeholderImage: {
        width: '100%',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: ClosetlyTheme.colors.card,
    },
    placeholderText: {
        fontSize: 40,
    },
    labelContainer: {
        padding: 12,
        gap: 4,
    },
    label: {
        ...ClosetlyTheme.typography.body,
        fontWeight: '600',
        color: ClosetlyTheme.colors.text,
    },
    colorRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
    },
    colorDot: {
        width: 10,
        height: 10,
        borderRadius: 5,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.1)',
    },
    colorText: {
        ...ClosetlyTheme.typography.caption,
    },
    matchBadge: {
        position: 'absolute',
        top: 10,
        right: 10,
        backgroundColor: ClosetlyTheme.colors.background,
        borderRadius: ClosetlyTheme.borderRadius.sm,
        paddingVertical: 4,
        paddingHorizontal: 8,
        ...ClosetlyTheme.shadows.cardSmall,
    },
    matchText: {
        ...ClosetlyTheme.typography.matchScore,
    },
    selectedIndicator: {
        position: 'absolute',
        top: 10,
        left: 10,
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: ClosetlyTheme.colors.text,
        alignItems: 'center',
        justifyContent: 'center',
    },
    selectedCheck: {
        color: ClosetlyTheme.colors.background,
        fontSize: 14,
        fontWeight: '700',
    },
});

export default ClothingCard;
