import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    TouchableOpacity,
    Image,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
    runOnJS,
    interpolate,
    Extrapolate,
    FadeIn,
} from 'react-native-reanimated';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import AppColors from '../constants/AppColors';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { useWardrobeItems } from '../src/hooks';

const { width, height } = Dimensions.get('window');

const SWIPE_THRESHOLD = width * 0.3;
const CARD_WIDTH = width - 48;
const CARD_HEIGHT = height * 0.55;

const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    like: '#4CAF50',
    dislike: '#FF5252',
    superLike: '#2196F3',
};

// Mock outfit data - in production, this would come from AI
const generateMockOutfits = (wardrobeItems: any[]) => {
    // Generate outfit combinations from wardrobe items
    const outfits: any[] = [];
    const tops = wardrobeItems.filter(i =>
        (i.type || i.itemType || '').toLowerCase().includes('shirt') ||
        (i.type || i.itemType || '').toLowerCase().includes('top') ||
        (i.type || i.itemType || '').toLowerCase().includes('jacket')
    );
    const bottoms = wardrobeItems.filter(i =>
        (i.type || i.itemType || '').toLowerCase().includes('pant') ||
        (i.type || i.itemType || '').toLowerCase().includes('jean') ||
        (i.type || i.itemType || '').toLowerCase().includes('skirt')
    );

    // Create combinations
    for (let i = 0; i < Math.min(tops.length, 5); i++) {
        for (let j = 0; j < Math.min(bottoms.length, 2); j++) {
            outfits.push({
                id: `outfit-${i}-${j}`,
                items: [tops[i], bottoms[j]],
                occasion: ['work', 'casual', 'date'][Math.floor(Math.random() * 3)],
                style: ['classic', 'trendy', 'minimalist'][Math.floor(Math.random() * 3)],
                matchScore: 70 + Math.floor(Math.random() * 30),
            });
        }
    }

    // If no wardrobe items, show sample outfits
    if (outfits.length === 0) {
        return [
            {
                id: 'sample-1',
                items: [
                    { type: 'White Shirt', color: 'White', image: null },
                    { type: 'Dark Jeans', color: 'Navy', image: null },
                ],
                occasion: 'casual',
                style: 'classic',
                matchScore: 85,
            },
            {
                id: 'sample-2',
                items: [
                    { type: 'Black Blazer', color: 'Black', image: null },
                    { type: 'Gray Trousers', color: 'Gray', image: null },
                ],
                occasion: 'work',
                style: 'classic',
                matchScore: 92,
            },
            {
                id: 'sample-3',
                items: [
                    { type: 'Floral Blouse', color: 'Pink', image: null },
                    { type: 'White Skirt', color: 'White', image: null },
                ],
                occasion: 'date',
                style: 'romantic',
                matchScore: 78,
            },
        ];
    }

    return outfits.slice(0, 10);
};

// ============================================
// OUTFIT CARD COMPONENT
// ============================================

interface OutfitCardProps {
    outfit: any;
    index: number;
    onSwipe: (direction: 'left' | 'right' | 'up') => void;
    isTop: boolean;
}

const OutfitCard = ({ outfit, index, onSwipe, isTop }: OutfitCardProps) => {
    const translateX = useSharedValue(0);
    const translateY = useSharedValue(0);
    const scale = useSharedValue(isTop ? 1 : 0.95);

    useEffect(() => {
        scale.value = withSpring(isTop ? 1 : 0.95 - index * 0.02);
    }, [isTop, index]);

    const panGesture = Gesture.Pan()
        .enabled(isTop)
        .onUpdate((event) => {
            translateX.value = event.translationX;
            translateY.value = event.translationY;
        })
        .onEnd(() => {
            if (Math.abs(translateX.value) > SWIPE_THRESHOLD) {
                // Horizontal swipe - like or dislike
                const direction = translateX.value > 0 ? 'right' : 'left';
                translateX.value = withTiming(
                    translateX.value > 0 ? width : -width,
                    { duration: 200 },
                    () => runOnJS(onSwipe)(direction)
                );
            } else if (translateY.value < -SWIPE_THRESHOLD) {
                // Upward swipe - super like
                translateY.value = withTiming(
                    -height,
                    { duration: 200 },
                    () => runOnJS(onSwipe)('up')
                );
            } else {
                // Return to center
                translateX.value = withSpring(0);
                translateY.value = withSpring(0);
            }
        });

    const cardStyle = useAnimatedStyle(() => {
        const rotate = interpolate(
            translateX.value,
            [-width / 2, 0, width / 2],
            [-15, 0, 15],
            Extrapolate.CLAMP
        );

        return {
            transform: [
                { translateX: translateX.value },
                { translateY: translateY.value },
                { rotate: `${rotate}deg` },
                { scale: scale.value },
            ],
        };
    });

    const likeOverlayStyle = useAnimatedStyle(() => ({
        opacity: interpolate(
            translateX.value,
            [0, SWIPE_THRESHOLD],
            [0, 1],
            Extrapolate.CLAMP
        ),
    }));

    const dislikeOverlayStyle = useAnimatedStyle(() => ({
        opacity: interpolate(
            translateX.value,
            [-SWIPE_THRESHOLD, 0],
            [1, 0],
            Extrapolate.CLAMP
        ),
    }));

    const superLikeOverlayStyle = useAnimatedStyle(() => ({
        opacity: interpolate(
            translateY.value,
            [-SWIPE_THRESHOLD, 0],
            [1, 0],
            Extrapolate.CLAMP
        ),
    }));

    return (
        <GestureDetector gesture={panGesture}>
            <Animated.View style={[styles.card, cardStyle, { zIndex: 10 - index }]}>
                {/* Card Content */}
                <View style={styles.cardInner}>
                    {/* Outfit Preview */}
                    <View style={styles.outfitPreview}>
                        {outfit.items.map((item: any, idx: number) => (
                            <View key={idx} style={styles.itemPreview}>
                                {item.image || item.imageUrl ? (
                                    <Image
                                        source={{ uri: item.image || item.imageUrl }}
                                        style={styles.itemImage}
                                        resizeMode="cover"
                                    />
                                ) : (
                                    <View style={styles.itemPlaceholder}>
                                        <Ionicons name="shirt-outline" size={40} color={COLORS.textSecondary} />
                                        <Text style={styles.itemPlaceholderText}>{item.type}</Text>
                                    </View>
                                )}
                            </View>
                        ))}
                    </View>

                    {/* Outfit Info */}
                    <View style={styles.outfitInfo}>
                        <View style={styles.outfitMeta}>
                            <View style={styles.occasionBadge}>
                                <Ionicons name="pricetag-outline" size={14} color={COLORS.primary} />
                                <Text style={styles.occasionText}>
                                    {outfit.occasion.charAt(0).toUpperCase() + outfit.occasion.slice(1)}
                                </Text>
                            </View>
                            <View style={styles.matchBadge}>
                                <Ionicons name="sparkles" size={14} color={COLORS.accent} />
                                <Text style={styles.matchText}>{outfit.matchScore}% Match</Text>
                            </View>
                        </View>

                        <Text style={styles.outfitItems}>
                            {outfit.items.map((i: any) => i.type || i.itemType).join(' + ')}
                        </Text>
                    </View>
                </View>

                {/* Swipe Overlays */}
                <Animated.View style={[styles.overlay, styles.likeOverlay, likeOverlayStyle]}>
                    <View style={styles.overlayBadge}>
                        <Ionicons name="heart" size={48} color="#FFF" />
                        <Text style={styles.overlayText}>LIKE</Text>
                    </View>
                </Animated.View>

                <Animated.View style={[styles.overlay, styles.dislikeOverlay, dislikeOverlayStyle]}>
                    <View style={styles.overlayBadge}>
                        <Ionicons name="close" size={48} color="#FFF" />
                        <Text style={styles.overlayText}>NOPE</Text>
                    </View>
                </Animated.View>

                <Animated.View style={[styles.overlay, styles.superLikeOverlay, superLikeOverlayStyle]}>
                    <View style={[styles.overlayBadge, { backgroundColor: COLORS.superLike }]}>
                        <Ionicons name="star" size={48} color="#FFF" />
                        <Text style={styles.overlayText}>LOVE IT!</Text>
                    </View>
                </Animated.View>
            </Animated.View>
        </GestureDetector>
    );
};


// ============================================
// MAIN SCREEN
// ============================================

const OutfitSwipeScreen = () => {
    const navigation = useNavigation();
    const { items: wardrobeItems } = useWardrobeItems();
    const { likeOutfit, dislikeOutfit, superLikeOutfit, totalLikes, totalDislikes } = useStylePreferenceStore();

    const [outfits, setOutfits] = useState<any[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Generate outfits
        const generated = generateMockOutfits(wardrobeItems);
        setOutfits(generated);
        setIsLoading(false);
    }, [wardrobeItems]);

    const handleSwipe = useCallback((direction: 'left' | 'right' | 'up') => {
        const outfit = outfits[currentIndex];
        if (!outfit) return;

        if (direction === 'right') {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            likeOutfit(outfit.id, outfit.items?.map((i: any) => i.id), outfit.occasion);
        } else if (direction === 'left') {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            dislikeOutfit(outfit.id, outfit.items?.map((i: any) => i.id), outfit.occasion);
        } else if (direction === 'up') {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            superLikeOutfit(outfit.id, outfit.items?.map((i: any) => i.id), outfit.occasion);
        }

        setCurrentIndex(prev => prev + 1);
    }, [currentIndex, outfits, likeOutfit, dislikeOutfit, superLikeOutfit]);

    const handleButtonPress = (type: 'dislike' | 'like' | 'superlike') => {
        if (type === 'dislike') handleSwipe('left');
        else if (type === 'like') handleSwipe('right');
        else if (type === 'superlike') handleSwipe('up');
    };

    const hasMoreOutfits = currentIndex < outfits.length;

    if (isLoading) {
        return (
            <View style={[styles.container, styles.centered]}>
                <ActivityIndicator size="large" color={COLORS.primary} />
                <Text style={styles.loadingText}>Generating outfit ideas...</Text>
            </View>
        );
    }

    return (
        <GestureHandlerRootView style={{ flex: 1 }}>
            <View style={styles.container}>
                <SafeAreaView style={styles.safeArea}>
                    {/* Header */}
                    <View style={styles.header}>
                        <TouchableOpacity
                            style={styles.headerButton}
                            onPress={() => navigation.goBack()}
                        >
                            <Ionicons name="arrow-back" size={24} color={COLORS.text} />
                        </TouchableOpacity>

                        <View style={styles.headerCenter}>
                            <Text style={styles.headerTitle}>Style Discovery</Text>
                            <Text style={styles.headerSubtitle}>
                                Swipe to teach AI your style
                            </Text>
                        </View>

                        <View style={styles.statsContainer}>
                            <View style={styles.statItem}>
                                <Ionicons name="heart" size={16} color={COLORS.like} />
                                <Text style={styles.statText}>{totalLikes}</Text>
                            </View>
                        </View>
                    </View>

                    {/* Cards Container */}
                    <View style={styles.cardsContainer}>
                        {hasMoreOutfits ? (
                            outfits.slice(currentIndex, currentIndex + 3).map((outfit, idx) => (
                                <OutfitCard
                                    key={outfit.id}
                                    outfit={outfit}
                                    index={idx}
                                    onSwipe={handleSwipe}
                                    isTop={idx === 0}
                                />
                            )).reverse()
                        ) : (
                            <Animated.View
                                style={styles.emptyState}
                                entering={FadeIn}
                            >
                                <Text style={styles.emptyEmoji}>🎉</Text>
                                <Text style={styles.emptyTitle}>All Done!</Text>
                                <Text style={styles.emptySubtitle}>
                                    You've rated all outfits. Your AI stylist is now smarter!
                                </Text>
                                <TouchableOpacity
                                    style={styles.refreshButton}
                                    onPress={() => {
                                        setCurrentIndex(0);
                                        const generated = generateMockOutfits(wardrobeItems);
                                        setOutfits(generated);
                                    }}
                                >
                                    <Ionicons name="refresh" size={20} color="#FFF" />
                                    <Text style={styles.refreshButtonText}>Get More Outfits</Text>
                                </TouchableOpacity>
                            </Animated.View>
                        )}
                    </View>

                    {/* Action Buttons */}
                    {hasMoreOutfits && (
                        <View style={styles.actionsContainer}>
                            <TouchableOpacity
                                style={[styles.actionButton, styles.dislikeButton]}
                                onPress={() => handleButtonPress('dislike')}
                            >
                                <Ionicons name="close" size={32} color={COLORS.dislike} />
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={[styles.actionButton, styles.superLikeButton]}
                                onPress={() => handleButtonPress('superlike')}
                            >
                                <Ionicons name="star" size={28} color={COLORS.superLike} />
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={[styles.actionButton, styles.likeButton]}
                                onPress={() => handleButtonPress('like')}
                            >
                                <Ionicons name="heart" size={32} color={COLORS.like} />
                            </TouchableOpacity>
                        </View>
                    )}

                    {/* Hint */}
                    {hasMoreOutfits && currentIndex === 0 && (
                        <Animated.View
                            style={styles.hintContainer}
                            entering={FadeIn.delay(500)}
                        >
                            <Text style={styles.hintText}>
                                👈 Swipe left to skip • Swipe right to like 👉
                            </Text>
                            <Text style={styles.hintText}>
                                ⬆️ Swipe up to super like
                            </Text>
                        </Animated.View>
                    )}
                </SafeAreaView>
            </View>
        </GestureHandlerRootView>
    );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    safeArea: {
        flex: 1,
    },
    centered: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        marginTop: 16,
        fontSize: 16,
        color: COLORS.textSecondary,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 12,
    },
    headerButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: COLORS.surface,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerCenter: {
        flex: 1,
        alignItems: 'center',
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
    },
    headerSubtitle: {
        fontSize: 13,
        color: COLORS.textSecondary,
    },
    statsContainer: {
        flexDirection: 'row',
        gap: 8,
    },
    statItem: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surface,
        paddingHorizontal: 10,
        paddingVertical: 6,
        borderRadius: 16,
        gap: 4,
    },
    statText: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.text,
    },

    // Cards
    cardsContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 24,
    },
    card: {
        position: 'absolute',
        width: CARD_WIDTH,
        height: CARD_HEIGHT,
        borderRadius: 24,
        backgroundColor: COLORS.surface,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.15,
        shadowRadius: 16,
        elevation: 8,
        overflow: 'hidden',
    },
    cardInner: {
        flex: 1,
    },
    outfitPreview: {
        flex: 1,
        flexDirection: 'row',
        padding: 16,
        gap: 12,
    },
    itemPreview: {
        flex: 1,
        backgroundColor: COLORS.background,
        borderRadius: 16,
        overflow: 'hidden',
    },
    itemImage: {
        width: '100%',
        height: '100%',
    },
    itemPlaceholder: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
    },
    itemPlaceholderText: {
        marginTop: 8,
        fontSize: 14,
        color: COLORS.textSecondary,
        textAlign: 'center',
    },
    outfitInfo: {
        padding: 16,
        borderTopWidth: 1,
        borderTopColor: COLORS.background,
    },
    outfitMeta: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 8,
    },
    occasionBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: `${COLORS.primary}15`,
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
        gap: 4,
    },
    occasionText: {
        fontSize: 13,
        color: COLORS.primary,
        fontWeight: '500',
    },
    matchBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: `${COLORS.accent}15`,
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
        gap: 4,
    },
    matchText: {
        fontSize: 13,
        color: COLORS.accent,
        fontWeight: '500',
    },
    outfitItems: {
        fontSize: 16,
        fontWeight: '600',
        color: COLORS.text,
    },

    // Overlays
    overlay: {
        ...StyleSheet.absoluteFillObject,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 24,
    },
    likeOverlay: {
        backgroundColor: 'rgba(76, 175, 80, 0.8)',
    },
    dislikeOverlay: {
        backgroundColor: 'rgba(255, 82, 82, 0.8)',
    },
    superLikeOverlay: {
        backgroundColor: 'rgba(33, 150, 243, 0.8)',
    },
    overlayBadge: {
        alignItems: 'center',
        padding: 20,
    },
    overlayText: {
        marginTop: 8,
        fontSize: 24,
        fontWeight: '800',
        color: '#FFF',
        letterSpacing: 2,
    },

    // Actions
    actionsContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        paddingVertical: 20,
        gap: 20,
    },
    actionButton: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: COLORS.surface,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 8,
        elevation: 4,
    },
    dislikeButton: {
        borderWidth: 2,
        borderColor: COLORS.dislike,
    },
    likeButton: {
        borderWidth: 2,
        borderColor: COLORS.like,
    },
    superLikeButton: {
        width: 52,
        height: 52,
        borderRadius: 26,
        borderWidth: 2,
        borderColor: COLORS.superLike,
    },

    // Empty state
    emptyState: {
        alignItems: 'center',
        padding: 24,
    },
    emptyEmoji: {
        fontSize: 64,
        marginBottom: 16,
    },
    emptyTitle: {
        fontSize: 24,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 8,
    },
    emptySubtitle: {
        fontSize: 16,
        color: COLORS.textSecondary,
        textAlign: 'center',
        marginBottom: 24,
    },
    refreshButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.primary,
        paddingHorizontal: 24,
        paddingVertical: 14,
        borderRadius: 16,
        gap: 8,
    },
    refreshButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: '#FFF',
    },

    // Hint
    hintContainer: {
        paddingBottom: 20,
        alignItems: 'center',
    },
    hintText: {
        fontSize: 13,
        color: COLORS.textSecondary,
    },
});

export default OutfitSwipeScreen;
