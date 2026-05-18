/**
 * OutfitResultSheet — Bottom sheet displaying the generated outfit result.
 *
 * Shows the AI-generated image (or OutfitCollageDisplay fallback), a horizontal
 * scrollable list of outfit item cards, the styling description, and Save /
 * Share / Regenerate action buttons.
 *
 * Requirements: 3.1, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
 *               8.1, 8.2, 8.3
 */

import React, { useEffect, useCallback } from 'react';
import {
  View,
  Text,
  Image,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Linking,
  AccessibilityInfo,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';
import { useTranslation } from 'react-i18next';

import { BottomSheet } from '../../../components/ui/BottomSheet';
import OutfitCollageDisplay from './OutfitCollageDisplay';
import LiquidGlass2026Theme, { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';
import { BASIC_CLOTHING_ITEMS } from '../../../data/basicClothingItems';
import type { GeneratedOutfit, OutfitItem, ClosetClothingItem } from '../types';

const { colors, spacing, typography } = LiquidGlass2026Theme;

// ─── Helpers ─────────────────────────────────────────────────

function resolveItemImage(src: string | number | undefined | null): string | number | null {
  if (src === null || src === undefined) return null;
  if (typeof src === 'number') return src;
  if (typeof src === 'string' && src.startsWith('basic_clothing_')) {
    const id = src.replace('basic_clothing_', '');
    const found = BASIC_CLOTHING_ITEMS.find((b) => b.id === id);
    return found ? found.image : null;
  }
  if (typeof src === 'string') {
    const trimmed = src.trim();
    if (trimmed.length === 0) return null;
    if (/^(null|undefined|none|false|0)$/i.test(trimmed)) return null;
    return trimmed;
  }
  return null;
}

function getCategoryLabel(item: OutfitItem): string {
  const raw = item.macroCategory ?? item.type ?? item.name ?? '';
  const map: Record<string, string> = {
    top: 'Top',
    outerwear: 'Outerwear',
    bottom: 'Bottom',
    shoes: 'Shoes',
    accessory: 'Accessory',
  };
  return map[raw.toLowerCase()] ?? ((raw.charAt(0).toUpperCase() + raw.slice(1)) || 'Item');
}

// ─── Props ───────────────────────────────────────────────────

export interface OutfitResultSheetProps {
  visible: boolean;
  outfit: GeneratedOutfit | null;
  generatedImageUrl: string | null;
  isImageLoading: boolean;
  isFallbackActive: boolean;
  anchorItem: ClosetClothingItem | null;
  onDismiss: () => void;
  onSave: () => void;
  onShare: () => void;
  onRegenerate: () => void;
}

// ─── Item Card ───────────────────────────────────────────────

interface ItemCardProps {
  item: OutfitItem;
  index: number;
}

const ItemCard: React.FC<ItemCardProps> = ({ item, index }) => {
  const [imgError, setImgError] = React.useState(false);
  const resolved = resolveItemImage(item.image);
  const label = getCategoryLabel(item);
  const colorValue = item.color ?? '';

  const handleShopPress = useCallback(() => {
    if (item.shopUrl) {
      Linking.openURL(item.shopUrl).catch(() => {
        // Silently ignore if URL can't be opened
      });
    }
  }, [item.shopUrl]);

  return (
    <Animated.View
      entering={FadeInDown.delay(index * 60).duration(280)}
      style={styles.itemCard}
      accessibilityLabel={`${label}${colorValue ? `, ${colorValue}` : ''}${item.isShopItem ? ', shop item' : ''}`}
    >
      {/* Thumbnail */}
      {!imgError && resolved !== null ? (
        <Image
          source={typeof resolved === 'number' ? resolved : { uri: resolved }}
          style={styles.itemImage}
          resizeMode="contain"
          onError={() => setImgError(true)}
        />
      ) : (
        <View style={styles.itemImagePlaceholder}>
          <Ionicons name="shirt-outline" size={36} color="#C8D0DA" />
        </View>
      )}

      {/* Category label */}
      <Text style={styles.itemCategoryLabel} numberOfLines={1}>
        {label}
      </Text>

      {/* Color dot */}
      {colorValue.length > 0 && (
        <View style={styles.itemColorRow}>
          <View
            style={[
              styles.itemColorDot,
              { backgroundColor: colorValue },
            ]}
          />
          <Text style={styles.itemColorText} numberOfLines={1}>
            {colorValue}
          </Text>
        </View>
      )}

      {/* Shop badge */}
      {item.isShopItem && (
        <TouchableOpacity
          style={styles.shopBadge}
          onPress={handleShopPress}
          activeOpacity={0.75}
          accessibilityLabel={`Shop ${label}`}
          accessibilityRole="link"
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons name="bag-outline" size={10} color="#FFFFFF" />
          <Text style={styles.shopBadgeText}>Shop</Text>
        </TouchableOpacity>
      )}
    </Animated.View>
  );
};

// ─── Main Component ──────────────────────────────────────────

const OutfitResultSheet: React.FC<OutfitResultSheetProps> = ({
  visible,
  outfit,
  generatedImageUrl,
  isImageLoading,
  isFallbackActive,
  anchorItem,
  onDismiss,
  onSave,
  onShare,
  onRegenerate,
}) => {
  const { t } = useTranslation();

  // Announce to screen readers when the sheet becomes visible
  useEffect(() => {
    if (visible && outfit) {
      AccessibilityInfo.announceForAccessibility(
        t('outfitGenerator.generationComplete', 'Your outfit is ready')
      );
    }
  }, [visible, outfit, t]);

  if (!outfit) return null;

  // Build accessibility label for the generated image
  const imageA11yLabel = outfit.items.length > 0
    ? `AI-generated outfit: ${outfit.items.map((i) => i.type ?? i.name ?? 'item').join(', ')}`
    : 'AI-generated outfit';

  // Styling description — use description or first styling tip
  const stylingDescription = outfit.description
    ?? (Array.isArray(outfit.stylingTips) ? outfit.stylingTips[0] : outfit.stylingTips)
    ?? '';

  return (
    <BottomSheet
      visible={visible}
      onClose={onDismiss}
      snapPoint={0.92}
      enableBlur
      style={styles.sheet}
    >
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
        bounces
      >
        {/* ── Image / Collage area ─────────────────────────── */}
        <View style={styles.imageContainer}>
          {isImageLoading ? (
            <View style={styles.imageLoadingState}>
              <ActivityIndicator size="large" color={colors.text.primary} />
              <Text style={styles.imageLoadingText}>Composing the image…</Text>
            </View>
          ) : generatedImageUrl ? (
            <Image
              source={{ uri: generatedImageUrl }}
              style={styles.generatedImage}
              resizeMode="cover"
              accessibilityLabel={imageA11yLabel}
              accessibilityRole="image"
            />
          ) : (
            <View
              accessibilityLabel={imageA11yLabel}
              accessibilityRole="image"
              style={styles.collageWrapper}
            >
              <OutfitCollageDisplay
                items={outfit.items}
                height={360}
                needsOuterwear={outfit.layered ?? true}
              />
            </View>
          )}

          {/* Simplified result banner */}
          {isFallbackActive && (
            <Animated.View entering={FadeIn.duration(300)} style={styles.fallbackBanner}>
              <Ionicons name="information-circle-outline" size={14} color="#FFFFFF" />
              <Text style={styles.fallbackBannerText}>Simplified result</Text>
            </Animated.View>
          )}

          {/* AI badge when real image is shown */}
          {generatedImageUrl && !isImageLoading && !isFallbackActive && (
            <View style={styles.aiBadge}>
              <Ionicons name="sparkles" size={11} color="#FFFFFF" />
              <Text style={styles.aiBadgeText}>AI Generated</Text>
            </View>
          )}
        </View>

        {/* ── Styling description ──────────────────────────── */}
        {stylingDescription.length > 0 && (
          <Animated.View entering={FadeInDown.delay(80).duration(300)} style={styles.descriptionBox}>
            <Text style={styles.descriptionText}>{stylingDescription}</Text>
          </Animated.View>
        )}

        {/* ── Item list ────────────────────────────────────── */}
        {outfit.items.length > 0 && (
          <Animated.View entering={FadeInDown.delay(120).duration(300)}>
            <Text style={styles.itemsLabel}>Items in this outfit</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.itemsScrollContent}
            >
              {outfit.items.map((item, idx) => (
                <ItemCard
                  key={String(item.id ?? idx)}
                  item={item}
                  index={idx}
                />
              ))}
            </ScrollView>
          </Animated.View>
        )}

        {/* ── Action buttons ───────────────────────────────── */}
        <Animated.View entering={FadeInDown.delay(180).duration(300)} style={styles.actionsContainer}>
          {/* Save Outfit */}
          <TouchableOpacity
            style={[styles.actionButton, styles.actionButtonPrimary]}
            onPress={onSave}
            activeOpacity={0.8}
            accessibilityLabel="Save outfit to your wardrobe"
            accessibilityRole="button"
          >
            <Ionicons name="bookmark-outline" size={20} color="#FFFFFF" />
            <Text style={[styles.actionButtonText, styles.actionButtonTextPrimary]}>
              Save Outfit
            </Text>
          </TouchableOpacity>

          {/* Share + Regenerate row */}
          <View style={styles.actionRow}>
            <TouchableOpacity
              style={[styles.actionButton, styles.actionButtonSecondary, styles.actionButtonFlex]}
              onPress={onShare}
              activeOpacity={0.8}
              accessibilityLabel="Share this outfit"
              accessibilityRole="button"
            >
              <Ionicons name="share-outline" size={20} color={colors.text.primary} />
              <Text style={styles.actionButtonText}>Share</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.actionButton, styles.actionButtonSecondary, styles.actionButtonFlex]}
              onPress={onRegenerate}
              activeOpacity={0.8}
              accessibilityLabel="Regenerate outfit with the same anchor item"
              accessibilityRole="button"
            >
              <Ionicons name="refresh-outline" size={20} color={colors.text.primary} />
              <Text style={styles.actionButtonText}>Regenerate</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>
      </ScrollView>
    </BottomSheet>
  );
};

// ─── Styles ──────────────────────────────────────────────────

const styles = StyleSheet.create({
  sheet: {
    borderTopLeftRadius: LiquidGlass2026Theme.radius.bottomSheet,
    borderTopRightRadius: LiquidGlass2026Theme.radius.bottomSheet,
  },

  scrollContent: {
    paddingBottom: 40,
  },

  // ── Image area ──────────────────────────────────────────────
  imageContainer: {
    width: '100%',
    height: 360,
    borderRadius: LiquidGlass2026Theme.radius.lg,
    overflow: 'hidden',
    backgroundColor: '#F7F8FA',
    marginBottom: spacing.md,
    position: 'relative',
  },

  generatedImage: {
    width: '100%',
    height: 360,
  },

  collageWrapper: {
    width: '100%',
    height: 360,
  },

  imageLoadingState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },

  imageLoadingText: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
    marginTop: spacing.xs,
  },

  // ── Fallback banner ─────────────────────────────────────────
  fallbackBanner: {
    position: 'absolute',
    bottom: 12,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(10, 25, 49, 0.78)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },

  fallbackBannerText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.2,
  },

  // ── AI badge ────────────────────────────────────────────────
  aiBadge: {
    position: 'absolute',
    top: 12,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(10, 25, 49, 0.82)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },

  aiBadgeText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.2,
  },

  // ── Description ─────────────────────────────────────────────
  descriptionBox: {
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderWidth: 1,
    borderColor: colors.border.glass,
    padding: spacing.md,
    marginBottom: spacing.md,
  },

  descriptionText: {
    ...typography.scale.bodyMedium,
    color: colors.text.secondary,
    lineHeight: 22,
  },

  // ── Item list ───────────────────────────────────────────────
  itemsLabel: {
    ...typography.scale.titleSmall,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },

  itemsScrollContent: {
    paddingRight: spacing.md,
    paddingVertical: 4,
    marginBottom: spacing.md,
    gap: 12,
  },

  itemCard: {
    width: 120,
    backgroundColor: '#FFFFFF',
    borderRadius: LiquidGlass2026Theme.radius.lg,
    overflow: 'hidden',
    alignItems: 'center',
    paddingBottom: 12,
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },

  itemImage: {
    width: 120,
    height: 120,
    borderRadius: LiquidGlass2026Theme.radius.lg,
  },

  itemImagePlaceholder: {
    width: 120,
    height: 120,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F3F6FA',
  },

  itemCategoryLabel: {
    fontSize: 12,
    color: '#1a1a2e',
    fontWeight: '700',
    marginTop: 8,
    paddingHorizontal: 8,
    textAlign: 'center',
    textTransform: 'capitalize',
    letterSpacing: 0.1,
  },

  itemColorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    gap: 5,
  },

  itemColorDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.1)',
  },

  itemColorText: {
    fontSize: 11,
    color: '#6B7280',
    textTransform: 'capitalize',
    maxWidth: 80,
  },

  shopBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    marginTop: 6,
    backgroundColor: '#0A1931',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    minWidth: 44,
    minHeight: 24,
    justifyContent: 'center',
  },

  shopBadgeText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.3,
  },

  // ── Action buttons ──────────────────────────────────────────
  actionsContainer: {
    gap: spacing.sm,
    marginTop: spacing.xs,
  },

  actionRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },

  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    gap: 8,
    minHeight: 44,
  },

  actionButtonFlex: {
    flex: 1,
  },

  actionButtonPrimary: {
    backgroundColor: '#0F172A',
  },

  actionButtonSecondary: {
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    borderWidth: 1,
    borderColor: colors.border.subtle,
  },

  actionButtonText: {
    ...typography.scale.labelLarge,
    color: colors.text.primary,
    fontWeight: '600',
  },

  actionButtonTextPrimary: {
    color: '#FFFFFF',
  },
});

export default OutfitResultSheet;
