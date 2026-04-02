/**
 * src/components/SkeletonLoader.tsx — Skeleton placeholder UI components
 *
 * Provides animated shimmer skeleton shapes for use while data loads.
 * Never show a spinner — always show skeleton shapes that match the
 * actual content layout (outfit cards, closet grid, chat bubbles).
 *
 * Dependencies:
 *   - react-native-reanimated (shimmer animation)
 *   - LiquidGlass2026Theme (colors)
 */

import React, { useEffect } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withTiming,
    interpolate,
    Extrapolation,
} from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';

const { colors } = LiquidGlass2026Theme;
const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ============================================
// BASE SHIMMER HOOK
// ============================================

/** Returns an animated shimmer style. Used by all skeleton shapes. */
function useShimmer() {
    const progress = useSharedValue(0);

    useEffect(() => {
        progress.value = withRepeat(
            withTiming(1, { duration: 1200 }),
            -1,
            false
        );
    }, []);

    const shimmerStyle = useAnimatedStyle(() => ({
        opacity: interpolate(progress.value, [0, 0.5, 1], [0.4, 0.8, 0.4], Extrapolation.CLAMP),
    }));

    return shimmerStyle;
}

// ============================================
// BASE SKELETON BLOCK
// ============================================

interface SkeletonBlockProps {
    width: number | string;
    height: number;
    borderRadius?: number;
    style?: object;
}

/**
 * Base skeleton block with shimmer animation.
 * Use this to build custom skeleton layouts.
 */
export const SkeletonBlock = ({ width, height, borderRadius = 8, style }: SkeletonBlockProps) => {
    const shimmerStyle = useShimmer();
    return (
        <Animated.View
            style={[
                {
                    width: width as number,
                    height,
                    borderRadius,
                    backgroundColor: colors.background.tertiary,
                },
                shimmerStyle,
                style,
            ]}
        />
    );
};

// ============================================
// OUTFIT CARD SKELETON
// ============================================

/**
 * Skeleton placeholder for a single outfit suggestion card.
 * Matches the exact layout of OutfitCard.tsx.
 */
export const OutfitCardSkeleton = () => {
    const shimmerStyle = useShimmer();
    return (
        <Animated.View style={[styles.outfitCard, shimmerStyle]}>
            {/* Header row */}
            <View style={styles.outfitCardHeader}>
                <SkeletonBlock width={120} height={16} borderRadius={6} />
                <SkeletonBlock width={80} height={22} borderRadius={11} />
            </View>

            {/* 2×2 item grid */}
            <View style={styles.outfitItemGrid}>
                {[0, 1, 2, 3].map((i) => (
                    <SkeletonBlock key={i} width='47%' height={130} borderRadius={12} />
                ))}
            </View>

            {/* Footer row */}
            <View style={styles.outfitCardFooter}>
                <SkeletonBlock width={140} height={14} borderRadius={5} />
                <SkeletonBlock width={100} height={36} borderRadius={18} />
            </View>
        </Animated.View>
    );
};

/**
 * Two skeleton outfit cards shown side by side while suggestions generate.
 * Used by the Home screen carousel while generateDailyOutfits runs.
 */
export const OutfitCarouselSkeleton = () => (
    <View style={styles.carouselSkeleton}>
        <OutfitCardSkeleton />
    </View>
);

// ============================================
// CLOSET GRID SKELETON
// ============================================

/**
 * Skeleton grid for the closet/wardrobe screen.
 * Shows 6 placeholder item cards matching the 2-column grid layout.
 */
export const ClosetGridSkeleton = () => {
    const shimmerStyle = useShimmer();
    const items = Array.from({ length: 6 });
    return (
        <View style={styles.closetGrid}>
            {items.map((_, i) => (
                <Animated.View key={i} style={[styles.closetItem, shimmerStyle]}>
                    <SkeletonBlock width='100%' height={160} borderRadius={14} />
                    <SkeletonBlock width='60%' height={12} borderRadius={5} style={{ marginTop: 8 }} />
                </Animated.View>
            ))}
        </View>
    );
};

// ============================================
// CHAT BUBBLE SKELETON
// ============================================

/**
 * Skeleton placeholder for the chat screen while loading history.
 * Alternates left (AI) and right (user) bubbles.
 */
export const ChatBubbleSkeleton = () => {
    const shimmerStyle = useShimmer();
    return (
        <View style={styles.chatContainer}>
            {[80, 60, 90, 50, 70].map((width, i) => (
                <Animated.View
                    key={i}
                    style={[
                        styles.chatBubble,
                        i % 2 === 0 ? styles.chatBubbleLeft : styles.chatBubbleRight,
                        shimmerStyle,
                    ]}
                >
                    <SkeletonBlock
                        width={`${width}%`}
                        height={i === 2 ? 56 : 36}
                        borderRadius={16}
                    />
                </Animated.View>
            ))}
        </View>
    );
};

// ============================================
// ANALYTICS SKELETON
// ============================================

/**
 * Skeleton for the analytics screen.
 * Shows circular progress placeholder + 3 metric bars.
 */
export const AnalyticsSkeleton = () => {
    const shimmerStyle = useShimmer();
    return (
        <View style={styles.analyticsContainer}>
            <Animated.View style={[styles.analyticsHero, shimmerStyle]}>
                <SkeletonBlock width={120} height={120} borderRadius={60} />
            </Animated.View>
            <View style={styles.analyticsBars}>
                {[100, 80, 65].map((width, i) => (
                    <View key={i} style={styles.analyticsBarRow}>
                        <SkeletonBlock width={80} height={14} borderRadius={5} />
                        <SkeletonBlock width={`${width}%`} height={10} borderRadius={5} />
                    </View>
                ))}
            </View>
        </View>
    );
};

// ============================================
// INSPO GRID SKELETON
// ============================================

/**
 * Skeleton for the Inspo tab masonry grid.
 * Shows 6 cards in 2 columns with varying heights.
 */
export const InspoGridSkeleton = () => {
    const shimmerStyle = useShimmer();
    const heights = [200, 160, 180, 200, 170, 190];
    return (
        <View style={styles.inspoGrid}>
            <View style={styles.inspoColumn}>
                {[0, 2, 4].map((i) => (
                    <Animated.View key={i} style={[styles.inspoCard, { height: heights[i] }, shimmerStyle]} />
                ))}
            </View>
            <View style={styles.inspoColumn}>
                {[1, 3, 5].map((i) => (
                    <Animated.View key={i} style={[styles.inspoCard, { height: heights[i] }, shimmerStyle]} />
                ))}
            </View>
        </View>
    );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    outfitCard: {
        width: SCREEN_WIDTH - 32,
        backgroundColor: colors.background.secondary,
        borderRadius: 20,
        padding: 16,
        marginHorizontal: 16,
    },
    outfitCardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 14,
    },
    outfitItemGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
        marginBottom: 14,
    },
    outfitCardFooter: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    carouselSkeleton: {
        paddingVertical: 8,
    },
    closetGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        padding: 16,
        gap: 12,
    },
    closetItem: {
        width: (SCREEN_WIDTH - 44) / 2,
    },
    chatContainer: {
        padding: 16,
        gap: 12,
    },
    chatBubble: {
        width: '100%',
    },
    chatBubbleLeft: {
        alignItems: 'flex-start',
    },
    chatBubbleRight: {
        alignItems: 'flex-end',
    },
    analyticsContainer: {
        padding: 24,
        alignItems: 'center',
        gap: 24,
    },
    analyticsHero: {
        alignItems: 'center',
    },
    analyticsBars: {
        width: '100%',
        gap: 16,
    },
    analyticsBarRow: {
        gap: 8,
    },
    inspoGrid: {
        flexDirection: 'row',
        padding: 16,
        gap: 8,
    },
    inspoColumn: {
        flex: 1,
        gap: 8,
    },
    inspoCard: {
        borderRadius: 14,
        backgroundColor: colors.background.tertiary,
    },
});
