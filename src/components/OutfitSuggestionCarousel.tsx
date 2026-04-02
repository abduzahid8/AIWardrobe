/**
 * src/components/OutfitSuggestionCarousel.tsx — Daily outfit suggestion carousel
 *
 * Full-width swipeable carousel of scored outfits.
 * Features:
 *   - Full-width OutfitCard per slide
 *   - Left/right arrow navigation
 *   - Dot pagination at bottom
 *   - Card index badge (e.g. "2 / 4")
 *   - All OutfitCard actions (save, edit, dislike, share, avatar)
 *   - Toast on dislike
 *   - Error boundary wrap — never white screens
 *
 * Dependencies:
 *   - OutfitCard
 *   - ScoredOutfit from suggestionEngine
 *   - ClothingItem from domain types
 *   - LiquidGlass2026Theme
 */

import React, { useRef, useState, useCallback } from 'react';
import {
    View,
    FlatList,
    TouchableOpacity,
    Text,
    StyleSheet,
    Dimensions,
    ViewToken,
    Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';
import OutfitCard from './OutfitCard';
import type { ScoredOutfit } from '../services/suggestionEngine';
import type { ClothingItem } from '../types/domain';

const { colors } = LiquidGlass2026Theme;
const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ============================================
// TYPES
// ============================================

export interface OutfitSuggestionCarouselProps {
    outfits: ScoredOutfit[];
    allItems: ClothingItem[];
    onSave: (outfitId: string, itemIds: string[]) => void;
    onDislike: (itemIds: string[]) => void;
    onEdit: (itemIds: string[]) => void;
    onAvatarPress: (itemIds: string[]) => void;
    onStylistChat?: (initialMessage: string) => void;
    savedOutfitKeys?: Set<string>;
}

// ============================================
// CAROUSEL
// ============================================

/**
 * Swipeable carousel that shows one outfit card per page.
 * Navigation arrows let users browse without swiping.
 * Dot pagination reflects current card position.
 */
const OutfitSuggestionCarousel = ({
    outfits,
    allItems,
    onSave,
    onDislike,
    onEdit,
    onAvatarPress,
    onStylistChat,
    savedOutfitKeys = new Set(),
}: OutfitSuggestionCarouselProps) => {
    const [activeIndex, setActiveIndex] = useState(0);
    const flatListRef = useRef<FlatList<ScoredOutfit>>(null);

    const validOutfits = outfits.filter(Boolean);
    if (validOutfits.length === 0) return null;

    /** Scroll to a specific index via the FlatList API */
    const scrollToIndex = useCallback((index: number) => {
        if (index < 0 || index >= validOutfits.length) return;
        flatListRef.current?.scrollToIndex({ index, animated: true });
        setActiveIndex(index);
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }, [validOutfits.length]);

    const handleViewableItemsChanged = useCallback(
        ({ viewableItems }: { viewableItems: ViewToken[] }) => {
            if (viewableItems.length > 0 && viewableItems[0].index !== null) {
                setActiveIndex(viewableItems[0].index);
            }
        },
        []
    );

    const viewabilityConfig = { itemVisiblePercentThreshold: 60 };

    const handleDislike = useCallback((itemIds: string[]) => {
        onDislike(itemIds);
        // Auto-advance to next card after dismissal
        const nextIndex = Math.min(activeIndex + 1, validOutfits.length - 1);
        if (nextIndex !== activeIndex) {
            setTimeout(() => scrollToIndex(nextIndex), 300);
        }
    }, [activeIndex, validOutfits.length, onDislike, scrollToIndex]);

    const renderItem = useCallback(({ item, index }: { item: ScoredOutfit; index: number }) => {
        const key = [...item.outfit.itemIds].sort().join(',');
        return (
            <OutfitCard
                scoredOutfit={item}
                allItems={allItems}
                onSave={onSave}
                onDislike={handleDislike}
                onEdit={onEdit}
                onAvatarPress={onAvatarPress}
                onStylistChat={onStylistChat}
                tempId={`outfit_${index}`}
                isSaved={savedOutfitKeys.has(key)}
            />
        );
    }, [allItems, onSave, handleDislike, onEdit, onAvatarPress, onStylistChat, savedOutfitKeys]);

    return (
        <View style={styles.container}>
            {/* Card index badge top-right */}
            <View style={styles.indexBadge}>
                <Text style={styles.indexBadgeText}>{activeIndex + 1} / {validOutfits.length}</Text>
            </View>

            {/* Main carousel */}
            <FlatList
                ref={flatListRef}
                data={validOutfits}
                renderItem={renderItem}
                keyExtractor={(_, i) => `outfit_${i}`}
                horizontal
                pagingEnabled
                showsHorizontalScrollIndicator={false}
                snapToInterval={SCREEN_WIDTH}
                snapToAlignment="start"
                decelerationRate="fast"
                onViewableItemsChanged={handleViewableItemsChanged}
                viewabilityConfig={viewabilityConfig}
                contentContainerStyle={styles.listContent}
                getItemLayout={(_, index) => ({
                    length: SCREEN_WIDTH,
                    offset: SCREEN_WIDTH * index,
                    index,
                })}
            />

            {/* Navigation arrows */}
            {activeIndex > 0 && (
                <TouchableOpacity
                    style={[styles.navArrow, styles.navArrowLeft]}
                    onPress={() => scrollToIndex(activeIndex - 1)}
                    hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                >
                    <Ionicons name="chevron-back" size={20} color={colors.text.primary} />
                </TouchableOpacity>
            )}
            {activeIndex < validOutfits.length - 1 && (
                <TouchableOpacity
                    style={[styles.navArrow, styles.navArrowRight]}
                    onPress={() => scrollToIndex(activeIndex + 1)}
                    hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                >
                    <Ionicons name="chevron-forward" size={20} color={colors.text.primary} />
                </TouchableOpacity>
            )}

            {/* Dot pagination */}
            {validOutfits.length > 1 && (
                <View style={styles.pagination}>
                    {validOutfits.map((_, i) => (
                        <TouchableOpacity
                            key={i}
                            style={[
                                styles.dot,
                                i === activeIndex && styles.dotActive,
                            ]}
                            onPress={() => scrollToIndex(i)}
                            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                        />
                    ))}
                </View>
            )}
        </View>
    );
};

export default OutfitSuggestionCarousel;

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        width: '100%',
        position: 'relative',
    },
    listContent: {
        paddingVertical: 4,
    },
    indexBadge: {
        position: 'absolute',
        top: 12,
        right: 28,
        zIndex: 10,
        backgroundColor: 'rgba(0,0,0,0.08)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
    },
    indexBadgeText: {
        fontSize: 12,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    navArrow: {
        position: 'absolute',
        top: '40%',
        zIndex: 20,
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: '#FFFFFF',
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.12,
        shadowRadius: 6,
        elevation: 4,
    },
    navArrowLeft: {
        left: 4,
    },
    navArrowRight: {
        right: 4,
    },
    pagination: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 6,
        paddingTop: 12,
        paddingBottom: 4,
    },
    dot: {
        width: 6,
        height: 6,
        borderRadius: 3,
        backgroundColor: colors.text.disabled,
    },
    dotActive: {
        width: 18,
        height: 6,
        borderRadius: 3,
        backgroundColor: colors.text.primary,
    },
});
