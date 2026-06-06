/**
 * src/components/OutfitCard.tsx — Single outfit suggestion card
 *
 * Renders one scored outfit as a polished card with:
 *   - Occasion label (top-left) + dress code badge (top-right)
 *   - 2×2 clothing item grid (top/bottom left, outer/shoes right)
 *   - Shopping suggestion indicator (black dot badge) on non-owned slots
 *   - Item count label with ⓘ icon
 *   - Action buttons: ♡ Save · ✏️ Edit · 👎 Dislike · ✈️ Share
 *   - "Create Avatar" black pill button
 *
 * Dependencies:
 *   - ScoredOutfit from suggestionEngine
 *   - ClothingItem from domain types
 *   - LiquidGlass2026Theme
 */

import React, { useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    Image,
    StyleSheet,
    Dimensions,
    Share,
    Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';
import type { ScoredOutfit, ShoppingSuggestion } from '../services/suggestionEngine';
import type { ClothingItem } from '../types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;
const { width: SCREEN_WIDTH } = Dimensions.get('window');

const CARD_WIDTH = SCREEN_WIDTH - 32;
const ITEM_CELL_SIZE = (CARD_WIDTH - 32 - 8) / 2;

// ============================================
// TYPES
// ============================================

export interface OutfitCardProps {
  scoredOutfit: ScoredOutfit;
  allItems: ClothingItem[];
  onSave: (outfitId: string, itemIds: string[]) => void;
  onDislike: (itemIds: string[]) => void;
  onEdit: (itemIds: string[]) => void;
  onAvatarPress: (itemIds: string[]) => void;
  /** Opens Gemini stylist chat pre-seeded with this outfit context */
  onStylistChat?: (initialMessage: string) => void;
  /** Transient ID used for save/dislike actions before outfit is persisted */
  tempId: string;
  isSaved?: boolean;
  /** Regenerate this outfit with different items */
  onRegenerate?: () => void;
}

// ============================================
// ITEM CELL
// ============================================

interface ItemCellProps {
    item: ClothingItem | null;
    suggestion: ShoppingSuggestion | null;
    darker?: boolean;
}

/** One clothing item cell in the 2×2 grid. */
const ItemCell = ({ item, suggestion, darker = false }: ItemCellProps) => (
    <View style={[styles.itemCell, darker && styles.itemCellDarker]}>
        {item?.imageUrl || item?.thumbnailUrl ? (
            <Image
                source={{ uri: item.thumbnailUrl || item.imageUrl }}
                style={styles.itemImage}
                resizeMode="contain"
            />
        ) : suggestion ? (
            <View style={styles.suggestionPlaceholder}>
                <Ionicons name="bag-add-outline" size={28} color={colors.text.tertiary} />
                <Text style={styles.suggestionLabel} numberOfLines={2}>
                    {suggestion.subCategory}
                </Text>
            </View>
        ) : (
            <View style={styles.emptyCell}>
                <Ionicons name="add" size={24} color={colors.text.disabled} />
            </View>
        )}

        {/* Black dot badge for shopping suggestions */}
        {suggestion && (
            <View style={styles.shoppingDot} />
        )}
    </View>
);

// ============================================
// OUTFIT CARD
// ============================================

/**
 * Full outfit card component matching the AltaDaily card layout.
 * Displays occasion label, dress code badge, 2-column item grid,
 * item count, and action bar.
 */
const OutfitCard = ({
  scoredOutfit,
  allItems,
  onSave,
  onDislike,
  onEdit,
  onAvatarPress,
  onStylistChat,
  onRegenerate,
  tempId,
  isSaved = false,
}: OutfitCardProps) => {
    const { outfit, occasionLabel, dressCode, shoppingSuggestions } = scoredOutfit;
    const { itemIds } = outfit;

    // Resolve clothing items from IDs
    const resolvedItems = itemIds.map((id) => allItems.find((i) => i.id === id) ?? null);

    // Map to grid slots: [top, bottom, outerwear, shoes]
    const topItem      = resolvedItems.find((i) => i?.category === 'top') ?? null;
    const bottomItem   = resolvedItems.find((i) => i?.category === 'bottom') ?? null;
    const outerItem    = resolvedItems.find((i) => i?.category === 'outerwear') ?? null;
    const shoesItem    = resolvedItems.find((i) => i?.category === 'shoes') ?? null;

    const topSuggestion      = shoppingSuggestions.find((s) => s.category === 'top') ?? null;
    const bottomSuggestion   = shoppingSuggestions.find((s) => s.category === 'bottom') ?? null;
    const outerSuggestion    = shoppingSuggestions.find((s) => s.category === 'outerwear') ?? null;
    const shoesSuggestion    = shoppingSuggestions.find((s) => s.category === 'shoes') ?? null;

    const shoppingSuggestionCount = shoppingSuggestions.length;

    /** Build a pre-seeded message describing this outfit for the chat */
    const handleAskStylist = useCallback(() => {
        if (!onStylistChat) return;
        const itemNames = resolvedItems
            .filter(Boolean)
            .map((i) => i!.name || i!.subCategory || i!.category)
            .join(', ');
        onStylistChat(`I'm considering this ${occasionLabel} outfit: ${itemNames}. What do you think?`);
    }, [resolvedItems, occasionLabel, onStylistChat]);

    /** Handle native share sheet with outfit description */
    const handleShare = useCallback(async () => {
        try {
            const itemNames = resolvedItems
                .filter(Boolean)
                .map((i) => i!.name || i!.subCategory || i!.category)
                .join(', ');
            await Share.share({
                message: `My ${occasionLabel} outfit: ${itemNames} — styled with AIWardrobe`,
                title: `${occasionLabel} outfit`,
            });
        } catch {
            // Share cancelled or unavailable — no action needed
        }
    }, [resolvedItems, occasionLabel]);

    return (
        <View style={styles.card}>
            {/* Header */}
            <View style={styles.header}>
                <Text style={styles.occasionLabel}>{occasionLabel}</Text>
                <View style={styles.dressCodeBadge}>
                    <Text style={styles.dressCodeText}>{dressCode}</Text>
                </View>
            </View>

            {/* 2-column item grid */}
            <View style={styles.itemGrid}>
                {/* Left column: Top + Bottom */}
                <View style={styles.itemColumn}>
                    <ItemCell item={topItem} suggestion={topSuggestion} />
                    <ItemCell item={bottomItem} suggestion={bottomSuggestion} />
                </View>
                {/* Right column: Outerwear + Shoes (slightly darker) */}
                <View style={styles.itemColumn}>
                    <ItemCell item={outerItem} suggestion={outerSuggestion} darker />
                    <ItemCell item={shoesItem} suggestion={shoesSuggestion} darker />
                </View>
            </View>

            {/* Item count info */}
            {shoppingSuggestionCount > 0 && (
                <View style={styles.itemCountRow}>
                    <Text style={styles.itemCountText}>
                        {shoppingSuggestionCount} item{shoppingSuggestionCount > 1 ? 's' : ''} suggested
                    </Text>
                    <Ionicons name="information-circle-outline" size={16} color={colors.text.tertiary} />
                </View>
            )}

            {/* Reasoning */}
            {outfit.reasoning ? (
                <Text style={styles.reasoning} numberOfLines={2}>{outfit.reasoning}</Text>
            ) : null}

      {/* Action bar */}
      <View style={styles.actionBar}>
        {/* Left actions */}
        <View style={styles.leftActions}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => onSave(tempId, itemIds)}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons
              name={isSaved ? 'heart' : 'heart-outline'}
              size={22}
              color={isSaved ? '#E05C5C' : colors.text.secondary}
            />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => onEdit(itemIds)}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="pencil-outline" size={22} color={colors.text.secondary} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => onDislike(itemIds)}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="thumbs-down-outline" size={22} color={colors.text.secondary} />
          </TouchableOpacity>
          {onRegenerate && (
            <TouchableOpacity
              style={styles.actionButton}
              onPress={onRegenerate}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Ionicons name="refresh-outline" size={22} color={colors.text.secondary} />
            </TouchableOpacity>
          )}
          <TouchableOpacity
            style={styles.actionButton}
            onPress={handleShare}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="share-outline" size={22} color={colors.text.secondary} />
          </TouchableOpacity>
        </View>

                {/* Right: Ask Stylist + Create Avatar */}
                <View style={styles.rightActions}>
                    {onStylistChat && (
                        <TouchableOpacity
                            style={styles.stylistPill}
                            onPress={handleAskStylist}
                            activeOpacity={0.85}
                        >
                            <Ionicons name="chatbubble-outline" size={14} color={colors.text.primary} />
                            <Text style={styles.stylistPillText}>Ask</Text>
                        </TouchableOpacity>
                    )}
                    <TouchableOpacity
                        style={styles.avatarPill}
                        onPress={() => onAvatarPress(itemIds)}
                        activeOpacity={0.85}
                    >
                        <Text style={styles.avatarPillText}>Create Avatar</Text>
                    </TouchableOpacity>
                </View>
            </View>
        </View>
    );
};

export default OutfitCard;

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    card: {
        width: CARD_WIDTH,
        backgroundColor: '#F2F2F2',
        borderRadius: 20,
        padding: 16,
        marginHorizontal: 16,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 14,
    },
    occasionLabel: {
        fontSize: 13,
        fontWeight: '600',
        color: colors.text.secondary,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    dressCodeBadge: {
        backgroundColor: colors.text.primary,
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 12,
    },
    dressCodeText: {
        fontSize: 11,
        fontWeight: '600',
        color: '#FFFFFF',
        letterSpacing: 0.3,
    },
    itemGrid: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 12,
    },
    itemColumn: {
        flex: 1,
        gap: 8,
    },
    itemCell: {
        height: ITEM_CELL_SIZE,
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
    },
    itemCellDarker: {
        backgroundColor: '#EBEBEB',
    },
    itemImage: {
        width: '85%',
        height: '85%',
    },
    suggestionPlaceholder: {
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        padding: 8,
    },
    suggestionLabel: {
        fontSize: 11,
        color: colors.text.tertiary,
        textAlign: 'center',
        fontWeight: '500',
    },
    emptyCell: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    shoppingDot: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: colors.text.primary,
    },
    itemCountRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        marginBottom: 6,
    },
    itemCountText: {
        fontSize: 12,
        color: colors.text.tertiary,
    },
    reasoning: {
        fontSize: 12,
        color: colors.text.tertiary,
        marginBottom: 12,
        fontStyle: 'italic',
    },
    actionBar: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: 4,
    },
    leftActions: {
        flexDirection: 'row',
        gap: 4,
    },
  actionButton: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.6)',
  },
    rightActions: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    stylistPill: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 5,
        backgroundColor: 'rgba(255,255,255,0.72)',
        paddingHorizontal: 12,
        paddingVertical: 9,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.5)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 4,
        elevation: 3,
    },
    stylistPillText: {
        color: colors.text.primary,
        fontSize: 13,
        fontWeight: '600',
    },
    avatarPill: {
        backgroundColor: colors.text.primary,
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
    },
    avatarPillText: {
        color: '#FFFFFF',
        fontSize: 13,
        fontWeight: '700',
        letterSpacing: 0.2,
    },
});
