/**
 * Cover photo for a saved outfit: a small flat-lay cascade — full,
 * uncropped garment photos layered diagonally (top piece back, bottom
 * piece overlapping it, shoes overlapping that), the way outfit pieces
 * are laid out in a styled flat-lay. No grid cells, no cropped tiles,
 * no circular thumbnails.
 */
import React, { useState } from 'react';
import { View, Image, StyleSheet, StyleProp, ImageStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export type CollageImageSrc = string | number;

export interface OutfitCollageItem {
    id?: string | number;
    image: CollageImageSrc;
    /** Macro category (top/bottom/shoes/outerwear/...) used to order pieces
     * the way an outfit is worn. Optional — falls back to the order given. */
    category?: string;
    name?: string;
    isShopItem?: boolean;
}

interface OutfitCollagePreviewProps {
    items: OutfitCollageItem[];
    backgroundColor?: string;
    placeholderIconColor?: string;
    placeholderIconSize?: number;
    /** Reserve this many px at the bottom (clear of the cascade) for a
     * caption footer drawn on top of this component. */
    footerInset?: number;
    /** Editorial, tappable piece layout used by the Home look deck. */
    variant?: 'cascade' | 'lookboard';
}

const CATEGORY_PRIORITY: Record<string, number> = {
    outerwear: 0,
    top: 1,
    dress: 1,
    bottom: 2,
    shoes: 3,
    accessory: 4,
    other: 5,
};

function orderByWorn(items: OutfitCollageItem[]): OutfitCollageItem[] {
    if (!items.some((i) => !!i.category)) return items;
    return [...items].sort(
        (a, b) => (CATEGORY_PRIORITY[a.category || 'other'] ?? 5) - (CATEGORY_PRIORITY[b.category || 'other'] ?? 5)
    );
}

// Diagonal cascade slots, back piece to front. Percent-based so it scales
// with whatever card size the caller uses.
const SLOTS_BY_COUNT: Record<number, StyleProp<ImageStyle>[]> = {
    1: [{ top: '6%', left: '8%', width: '84%', height: '86%' }],
    2: [
        { top: 0, left: '2%', width: '68%', height: '58%' },
        { top: '36%', left: '28%', width: '56%', height: '50%' },
    ],
    3: [
        { top: 0, left: 0, width: '64%', height: '54%' },
        { top: '32%', left: '30%', width: '54%', height: '48%' },
        { bottom: '4%', right: '2%', width: '36%', height: '24%' },
    ],
    4: [
        { top: 0, left: 0, width: '54%', height: '46%' },
        { top: '2%', right: 0, width: '40%', height: '36%' },
        { bottom: '22%', left: '12%', width: '50%', height: '44%' },
        { bottom: 0, right: '4%', width: '32%', height: '24%' },
    ],
};

const LOOKBOARD_SLOTS: Record<number, StyleProp<ImageStyle>[]> = {
    1: [{ top: '3%', left: '10%', width: '80%', height: '92%' }],
    2: [
        { top: '1%', left: '3%', width: '61%', height: '64%' },
        { bottom: '1%', right: '2%', width: '56%', height: '56%' },
    ],
    3: [
        { top: '0%', left: '2%', width: '59%', height: '56%' },
        { bottom: '0%', left: '20%', width: '55%', height: '55%' },
        { bottom: '1%', right: '0%', width: '38%', height: '27%' },
    ],
    4: [
        { top: '0%', left: '0%', width: '54%', height: '57%' },
        { top: '1%', right: '0%', width: '47%', height: '49%' },
        { bottom: '0%', left: '18%', width: '54%', height: '55%' },
        { bottom: '0%', right: '0%', width: '37%', height: '28%' },
    ],
};

const Piece: React.FC<{ src: CollageImageSrc; style: StyleProp<ImageStyle>; zIndex: number }> = ({
    src,
    style,
    zIndex,
}) => {
    const [failed, setFailed] = useState(false);
    if (failed) return null;
    return (
        <Image
            source={typeof src === 'number' ? src : { uri: src }}
            style={[styles.piece, style, { zIndex }]}
            resizeMode="contain"
            onError={() => setFailed(true)}
        />
    );
};

export const OutfitCollagePreview: React.FC<OutfitCollagePreviewProps> = ({
    items,
    backgroundColor = '#F9FBFF',
    placeholderIconColor = 'rgba(15,23,42,0.22)',
    placeholderIconSize = 32,
    footerInset = 0,
    variant = 'cascade',
}) => {
    const ordered = orderByWorn(items.filter((i) => i.image !== undefined && i.image !== null && i.image !== '')).slice(
        0,
        4
    );

    if (ordered.length === 0) {
        return (
            <View style={[styles.root, styles.centered, { backgroundColor }]}>
                <Ionicons name="shirt-outline" size={placeholderIconSize} color={placeholderIconColor} />
            </View>
        );
    }

    if (variant === 'lookboard') {
        const slots = LOOKBOARD_SLOTS[ordered.length];
        return (
            <View style={[styles.root, { backgroundColor }]}>
                <View style={[styles.lookboardArea, { bottom: footerInset }]}>
                    <View style={styles.lookboardHalo} />
                    <View style={styles.lookboardHaloSmall} />
                    <View style={styles.lookboardRule} />
                    {ordered.map((item, index) => (
                        <Image
                            key={item.id ?? index}
                            source={typeof item.image === 'number' ? item.image : { uri: item.image }}
                            style={[styles.lookboardPiece, slots[index], { zIndex: index + 1 }]}
                            resizeMode="contain"
                            accessibilityLabel={item.name || item.category || 'Outfit piece'}
                        />
                    ))}
                </View>
            </View>
        );
    }

    const slots = SLOTS_BY_COUNT[ordered.length];

    return (
        <View style={[styles.root, { backgroundColor }]}>
            <View style={[styles.cascadeArea, { bottom: footerInset }]}>
                {ordered.map((item, index) => (
                    <Piece key={item.id ?? index} src={item.image} style={slots[index]} zIndex={index + 1} />
                ))}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    root: {
        width: '100%',
        height: '100%',
        overflow: 'hidden',
    },
    centered: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    cascadeArea: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
    },
    piece: {
        position: 'absolute',
    },
    lookboardArea: {
        position: 'absolute',
        top: 42,
        left: 0,
        right: 0,
        overflow: 'hidden',
    },
    lookboardHalo: {
        position: 'absolute',
        width: 250,
        height: 250,
        borderRadius: 125,
        backgroundColor: 'rgba(160, 176, 153, 0.14)',
        top: '8%',
        left: '12%',
    },
    lookboardHaloSmall: {
        position: 'absolute',
        width: 120,
        height: 120,
        borderRadius: 60,
        backgroundColor: 'rgba(211, 198, 181, 0.14)',
        right: '-5%',
        bottom: '3%',
    },
    lookboardRule: {
        position: 'absolute',
        top: '49%',
        left: '8%',
        right: '8%',
        height: 1,
        backgroundColor: 'rgba(23,58,101,0.055)',
    },
    lookboardPiece: {
        position: 'absolute',
    },
});

export default OutfitCollagePreview;
