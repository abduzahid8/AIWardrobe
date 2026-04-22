/**
 * LayeredOutfitView — Displays an outfit organized by clothing layers
 *
 * Visual breakdown:
 *   🧥 Outer Layer (coats, jackets)
 *   👕 Mid Layer (sweaters, hoodies)
 *   👔 Base Layer (t-shirts, pants, shoes)
 *   💍 Accessories (bags, hats, jewelry)
 */

import React from 'react';
import {
    View,
    Text,
    Image,
    StyleSheet,
    ScrollView,
    TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import type { ClothingItem } from '../src/types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

type ClothingLayer = 'outer' | 'mid' | 'base' | 'accessory';

interface LayeredOutfitViewProps {
    items: ClothingItem[];
    onItemPress?: (item: ClothingItem) => void;
    onLayerChange?: (itemId: string, newLayer: ClothingLayer) => void;
    editable?: boolean;
    compact?: boolean;
}

const LAYER_CONFIG: { key: ClothingLayer; label: string; icon: string; color: string }[] = [
    { key: 'outer', label: 'Outer', icon: 'snow-outline', color: '#60A5FA' },
    { key: 'mid', label: 'Mid', icon: 'shirt-outline', color: '#A78BFA' },
    { key: 'base', label: 'Base', icon: 'body-outline', color: '#34D399' },
    { key: 'accessory', label: 'Accessories', icon: 'diamond-outline', color: '#FBBF24' },
];

/** Get default layer from category if item doesn't have one set */
function getEffectiveLayer(item: ClothingItem): ClothingLayer {
    switch (item.category) {
        case 'outerwear': return 'outer';
        case 'accessory': return 'accessory';
        case 'top': return 'base';
        case 'bottom': return 'base';
        case 'shoes': return 'base';
        default: return 'mid';
    }
}

export default function LayeredOutfitView({
    items,
    onItemPress,
    onLayerChange,
    editable = false,
    compact = false,
}: LayeredOutfitViewProps) {
    // Group items by layer
    const grouped: Record<ClothingLayer, ClothingItem[]> = {
        outer: [],
        mid: [],
        base: [],
        accessory: [],
    };

    for (const item of items) {
        const layer = getEffectiveLayer(item);
        grouped[layer].push(item);
    }

    if (items.length === 0) {
        return (
            <View style={styles.emptyContainer}>
                <Ionicons name="layers-outline" size={40} color={colors.text.tertiary} />
                <Text style={styles.emptyText}>No items in this outfit</Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            {LAYER_CONFIG.map(({ key, label, icon, color }) => {
                const layerItems = grouped[key];
                if (layerItems.length === 0 && !editable) return null;

                return (
                    <View key={key} style={styles.layerSection}>
                        <View style={styles.layerHeader}>
                            <Ionicons name={icon as any} size={16} color={color} />
                            <Text style={[styles.layerLabel, { color }]}>{label}</Text>
                            <View style={[styles.layerBadge, { backgroundColor: color + '20' }]}>
                                <Text style={[styles.layerBadgeText, { color }]}>
                                    {layerItems.length}
                                </Text>
                            </View>
                        </View>

                        {layerItems.length > 0 ? (
                            <ScrollView
                                horizontal
                                showsHorizontalScrollIndicator={false}
                                contentContainerStyle={styles.itemsRow}
                            >
                                {layerItems.map((item) => (
                                    <TouchableOpacity
                                        key={item.id}
                                        onPress={() => onItemPress?.(item)}
                                        activeOpacity={0.7}
                                        style={styles.itemCard}
                                    >
                                        <Image
                                            source={{ uri: item.imageUrl || item.thumbnailUrl }}
                                            style={[
                                                styles.itemImage,
                                                compact && styles.itemImageCompact,
                                            ]}
                                        />
                                        <Text style={styles.itemName} numberOfLines={1}>
                                            {item.name || item.subCategory || item.category}
                                        </Text>
                                        {editable && (
                                            <View style={[styles.layerDot, { backgroundColor: color }]} />
                                        )}
                                    </TouchableOpacity>
                                ))}
                            </ScrollView>
                        ) : (
                            <View style={styles.emptyLayer}>
                                <Text style={styles.emptyLayerText}>
                                    No {label.toLowerCase()} items
                                </Text>
                            </View>
                        )}

                        {/* Connector line between layers */}
                        {key !== 'accessory' && layerItems.length > 0 && (
                            <View style={[styles.connector, { backgroundColor: color + '30' }]} />
                        )}
                    </View>
                );
            })}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        gap: spacing.xs,
    },
    emptyContainer: {
        alignItems: 'center',
        paddingVertical: spacing.xl,
        gap: spacing.sm,
    },
    emptyText: {
        ...typography.scale.bodyMedium,
        color: colors.text.tertiary,
    },

    // Layer sections
    layerSection: {
        paddingHorizontal: spacing.lg,
    },
    layerHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.sm,
        marginBottom: spacing.sm,
    },
    layerLabel: {
        ...typography.scale.labelMedium,
        fontWeight: '600',
    },
    layerBadge: {
        paddingHorizontal: 6,
        paddingVertical: 1,
        borderRadius: 8,
    },
    layerBadgeText: {
        fontSize: 10,
        fontWeight: '700',
    },

    // Items
    itemsRow: {
        gap: spacing.sm,
        paddingBottom: spacing.sm,
    },
    itemCard: {
        alignItems: 'center',
        width: 72,
    },
    itemImage: {
        width: 64,
        height: 64,
        borderRadius: radius.md,
        backgroundColor: colors.border.glass,
    },
    itemImageCompact: {
        width: 48,
        height: 48,
    },
    itemName: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
        textAlign: 'center',
        marginTop: 4,
        fontSize: 10,
    },
    layerDot: {
        width: 6,
        height: 6,
        borderRadius: 3,
        marginTop: 2,
    },

    // Empty layer
    emptyLayer: {
        paddingVertical: spacing.sm,
        paddingHorizontal: spacing.md,
        borderWidth: 1,
        borderColor: colors.border.glass,
        borderRadius: radius.md,
        borderStyle: 'dashed',
        alignItems: 'center',
        marginBottom: spacing.sm,
    },
    emptyLayerText: {
        ...typography.scale.labelSmall,
        color: colors.text.tertiary,
    },

    // Connector
    connector: {
        width: 2,
        height: 12,
        marginLeft: 8,
        borderRadius: 1,
    },
});
