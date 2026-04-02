/**
 * src/components/AvatarDisplay.tsx — Layered avatar + clothing compositor
 *
 * Renders the user's avatar silhouette with clothing items layered on top
 * using 2D image composition. Each clothing category maps to a fixed
 * position/scale on the silhouette so items appear naturally "worn".
 *
 * Composition order (bottom-to-top):
 *   1. Silhouette background (SVG or PNG placeholder)
 *   2. Bottom layer (trousers / shoes)
 *   3. Top layer (shirt / jacket)
 *   4. Outerwear layer (coat / blazer)
 *   5. Accessory layer (hat / watch)
 *
 * Dependencies:
 *   - react-native-svg (Svg, Rect, Circle)
 *   - ClothingItem from domain types
 *   - LiquidGlass2026Theme
 */

import React, { useCallback } from 'react';
import {
    View,
    Image,
    StyleSheet,
    Dimensions,
    Text,
} from 'react-native';
import Svg, { Ellipse, Rect, Circle } from 'react-native-svg';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';
import type { ClothingItem } from '../types/domain';

const { colors } = LiquidGlass2026Theme;
const { width: SCREEN_WIDTH } = Dimensions.get('window');

const AVATAR_WIDTH  = Math.min(SCREEN_WIDTH * 0.65, 280);
const AVATAR_HEIGHT = AVATAR_WIDTH * 2.2;

// ============================================
// CATEGORY → LAYOUT MAP
// ============================================

interface ClothingLayer {
    top: number;
    left: number;
    width: number;
    height: number;
    zIndex: number;
}

/**
 * Positions and dimensions for each clothing category layer,
 * expressed as fractions of the avatar dimensions.
 */
const LAYER_MAP: Record<string, ClothingLayer> = {
    top: {
        top:    0.22,
        left:   0.10,
        width:  0.80,
        height: 0.35,
        zIndex: 3,
    },
    bottom: {
        top:    0.54,
        left:   0.12,
        width:  0.76,
        height: 0.35,
        zIndex: 2,
    },
    shoes: {
        top:    0.86,
        left:   0.10,
        width:  0.80,
        height: 0.12,
        zIndex: 2,
    },
    outerwear: {
        top:    0.18,
        left:   0.06,
        width:  0.88,
        height: 0.42,
        zIndex: 4,
    },
    accessory: {
        top:    0.01,
        left:   0.30,
        width:  0.40,
        height: 0.14,
        zIndex: 5,
    },
};

// ============================================
// SILHOUETTE SVG
// ============================================

/**
 * Simple SVG male silhouette — used when no photo has been set.
 * Proportioned to match the clothing layer positions above.
 */
const MaleSilhouette = () => (
    <Svg
        width={AVATAR_WIDTH}
        height={AVATAR_HEIGHT}
        viewBox={`0 0 100 220`}
        style={StyleSheet.absoluteFillObject}
    >
        {/* Head */}
        <Circle cx="50" cy="15" r="11" fill="#D4C5B2" />
        {/* Neck */}
        <Rect x="45" y="25" width="10" height="8" rx="3" fill="#D4C5B2" />
        {/* Torso */}
        <Rect x="28" y="32" width="44" height="74" rx="6" fill="#C8BAA8" />
        {/* Left arm */}
        <Rect x="10" y="34" width="18" height="56" rx="8" fill="#C8BAA8" />
        {/* Right arm */}
        <Rect x="72" y="34" width="18" height="56" rx="8" fill="#C8BAA8" />
        {/* Left leg */}
        <Rect x="28" y="104" width="18" height="72" rx="7" fill="#B8AB9A" />
        {/* Right leg */}
        <Rect x="54" y="104" width="18" height="72" rx="7" fill="#B8AB9A" />
        {/* Left foot */}
        <Ellipse cx="37" cy="182" rx="12" ry="5" fill="#A89880" />
        {/* Right foot */}
        <Ellipse cx="63" cy="182" rx="12" ry="5" fill="#A89880" />
    </Svg>
);

// ============================================
// AVATAR DISPLAY
// ============================================

interface AvatarDisplayProps {
    /** List of clothing items to overlay on the avatar */
    items: ClothingItem[];
    /** Optional user face photo URI (currently unused — future avatar generation) */
    facePhotoUri?: string;
    /** Dimensions to render the avatar at */
    width?: number;
    height?: number;
}

/**
 * Renders the user's avatar silhouette with clothing items layered on top.
 * Items are positioned absolutely over the silhouette using LAYER_MAP.
 * Items without an imageUrl show a category placeholder color block.
 */
const AvatarDisplay = ({
    items,
    width = AVATAR_WIDTH,
    height = AVATAR_HEIGHT,
}: AvatarDisplayProps) => {
    const scaleX = width  / AVATAR_WIDTH;
    const scaleY = height / AVATAR_HEIGHT;

    /** Compute the absolute position/size of a clothing layer. */
    const getLayerStyle = useCallback((layer: ClothingLayer) => ({
        position: 'absolute' as const,
        top:    layer.top    * height,
        left:   layer.left   * width,
        width:  layer.width  * width,
        height: layer.height * height,
        zIndex: layer.zIndex,
    }), [width, height]);

    // Resolve top item per category (outerwear > top, prefer higher zIndex)
    const layeredItems: Record<string, ClothingItem> = {};
    for (const item of items) {
        const cat = item.category;
        if (LAYER_MAP[cat]) {
            // Keep the one with higher z-index preference (e.g. outerwear over top)
            if (!layeredItems[cat] || LAYER_MAP[cat].zIndex > LAYER_MAP[layeredItems[cat].category].zIndex) {
                layeredItems[cat] = item;
            }
        }
    }

    return (
        <View style={[styles.container, { width, height }]}>
            {/* Base silhouette */}
            <MaleSilhouette />

            {/* Clothing layers */}
            {Object.entries(layeredItems).map(([cat, item]) => {
                const layer = LAYER_MAP[cat];
                if (!layer) return null;

                return (
                    <View key={cat} style={getLayerStyle(layer)} pointerEvents="none">
                        {item.imageUrl || item.thumbnailUrl ? (
                            <Image
                                source={{ uri: item.thumbnailUrl || item.imageUrl }}
                                style={styles.clothingImage}
                                resizeMode="contain"
                            />
                        ) : (
                            <View style={[styles.categoryPlaceholder, { backgroundColor: CATEGORY_COLORS[cat] ?? '#E0E0E0' }]}>
                                <Text style={styles.categoryPlaceholderText}>
                                    {cat}
                                </Text>
                            </View>
                        )}
                    </View>
                );
            })}

            {/* Empty state label */}
            {items.length === 0 && (
                <View style={styles.emptyLabel}>
                    <Text style={styles.emptyLabelText}>Tap items below to try on</Text>
                </View>
            )}
        </View>
    );
};

export default AvatarDisplay;

// ============================================
// CONSTANTS + STYLES
// ============================================

const CATEGORY_COLORS: Record<string, string> = {
    top:       '#D6EAF8',
    bottom:    '#D5DBDB',
    shoes:     '#E8DAEF',
    outerwear: '#D5E8D4',
    accessory: '#FAD7A0',
};

const styles = StyleSheet.create({
    container: {
        position: 'relative',
        overflow: 'hidden',
        backgroundColor: '#F8F6F2',
        borderRadius: 20,
    },
    clothingImage: {
        width: '100%',
        height: '100%',
    },
    categoryPlaceholder: {
        width: '100%',
        height: '100%',
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
        opacity: 0.75,
    },
    categoryPlaceholderText: {
        fontSize: 11,
        color: colors.text.secondary,
        textTransform: 'capitalize',
        fontWeight: '500',
    },
    emptyLabel: {
        position: 'absolute',
        bottom: 16,
        left: 0,
        right: 0,
        alignItems: 'center',
    },
    emptyLabelText: {
        fontSize: 12,
        color: colors.text.tertiary,
        fontStyle: 'italic',
    },
});
