import React, { useCallback, useRef } from 'react';
import {
    View,
    FlatList,
    StyleSheet,
    Dimensions,
    ViewToken,
    Text,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedScrollHandler,
    runOnJS,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import ClothingCard from './ClothingCard';
import { ClosetlyTheme } from '../../constants/ClosetlyTheme';

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

interface SwipeableClothingCarouselProps {
    items: ClothingItem[];
    onItemChange: (item: ClothingItem, index: number) => void;
    category: 'tops' | 'bottoms';
    matchScores?: Map<string, number>;  // Map of item ID to match score
    selectedId?: string;
    itemWidth?: number;
    containerHeight?: number;
}

/**
 * SwipeableClothingCarousel - Horizontal swipeable carousel for clothing items
 * Features:
 * - Snap-to-center behavior
 * - Parallax animation on cards
 * - Haptic feedback on selection
 * - Pre-caching adjacent items
 * - AI match highlighting
 */
const SwipeableClothingCarousel: React.FC<SwipeableClothingCarouselProps> = ({
    items,
    onItemChange,
    category,
    matchScores,
    selectedId,
    itemWidth = 156,
    containerHeight = 220,
}) => {
    const scrollX = useSharedValue(0);
    const flatListRef = useRef<FlatList<ClothingItem>>(null);
    const lastHapticIndex = useRef(-1);

    // Calculate snap interval and offsets
    const snapInterval = itemWidth;
    const contentPadding = (SCREEN_WIDTH - itemWidth) / 2;

    // Haptic feedback when item changes
    const triggerHaptic = useCallback(() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }, []);

    // Scroll handler for parallax effect
    const scrollHandler = useAnimatedScrollHandler({
        onScroll: (event) => {
            scrollX.value = event.contentOffset.x;

            // Calculate current index for haptic feedback
            const currentIndex = Math.round(event.contentOffset.x / itemWidth);
            if (currentIndex !== lastHapticIndex.current && currentIndex >= 0 && currentIndex < items.length) {
                lastHapticIndex.current = currentIndex;
                runOnJS(triggerHaptic)();
            }
        },
    });

    // Handle viewable items change for selection callback
    const onViewableItemsChanged = useCallback(
        ({ viewableItems }: { viewableItems: ViewToken[] }) => {
            if (viewableItems.length > 0) {
                const centerItem = viewableItems[Math.floor(viewableItems.length / 2)];
                if (centerItem?.item) {
                    onItemChange(centerItem.item as ClothingItem, centerItem.index ?? 0);
                }
            }
        },
        [onItemChange]
    );

    const viewabilityConfig = useRef({
        itemVisiblePercentThreshold: 50,
        minimumViewTime: 100,
    }).current;

    // Render individual clothing card
    const renderItem = useCallback(
        ({ item, index }: { item: ClothingItem; index: number }) => {
            const itemId = item._id || item.id || '';
            const matchScore = matchScores?.get(itemId);
            const isSelected = selectedId === itemId;

            return (
                <ClothingCard
                    item={item}
                    index={index}
                    scrollX={scrollX}
                    itemWidth={itemWidth}
                    matchScore={matchScore}
                    isSelected={isSelected}
                />
            );
        },
        [scrollX, itemWidth, matchScores, selectedId]
    );

    // Key extractor
    const keyExtractor = useCallback(
        (item: ClothingItem, index: number) => item._id || item.id || `${category}-${index}`,
        [category]
    );

    // Get item layout for optimized scrolling
    const getItemLayout = useCallback(
        (_data: ArrayLike<ClothingItem> | null | undefined, index: number) => ({
            length: itemWidth,
            offset: itemWidth * index,
            index,
        }),
        [itemWidth]
    );

    if (items.length === 0) {
        return (
            <View style={[styles.emptyContainer, { height: containerHeight }]}>
                <View style={styles.emptyCard}>
                    <Text style={styles.emptyIcon}>
                        {category === 'tops' ? '👕' : '👖'}
                    </Text>
                </View>
            </View>
        );
    }

    return (
        <View style={[styles.container, { height: containerHeight }]}>
            <Animated.FlatList
                ref={flatListRef as React.RefObject<Animated.FlatList<ClothingItem>>}
                data={items}
                renderItem={renderItem}
                keyExtractor={keyExtractor}
                horizontal
                showsHorizontalScrollIndicator={false}
                snapToInterval={snapInterval}
                snapToAlignment="center"
                decelerationRate="fast"
                contentContainerStyle={{
                    paddingHorizontal: contentPadding,
                }}
                onScroll={scrollHandler}
                scrollEventThrottle={16}
                onViewableItemsChanged={onViewableItemsChanged}
                viewabilityConfig={viewabilityConfig}
                getItemLayout={getItemLayout}
                // Performance optimizations
                removeClippedSubviews={true}
                maxToRenderPerBatch={5}
                windowSize={5}
                initialNumToRender={3}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        width: '100%',
    },
    emptyContainer: {
        width: '100%',
        alignItems: 'center',
        justifyContent: 'center',
    },
    emptyCard: {
        width: 140,
        height: 180,
        backgroundColor: ClosetlyTheme.colors.card,
        borderRadius: ClosetlyTheme.borderRadius.card,
        alignItems: 'center',
        justifyContent: 'center',
        ...ClosetlyTheme.shadows.card,
    },
    emptyIcon: {
        fontSize: 48,
    },
});

export default SwipeableClothingCarousel;
