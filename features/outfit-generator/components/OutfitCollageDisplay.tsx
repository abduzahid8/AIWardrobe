import React from 'react';
import { View, Image, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { BASIC_CLOTHING_ITEMS } from '../../../data/basicClothingItems';
import type { OutfitItem } from '../types';
import { getOutfitPreviewSlots, getOutfitPreviewTitle } from '../utils/outfitPreview';

function resolveClothingImage(src: string | number | undefined | null): string | number | null {
  if (src === null || src === undefined) return null;
  if (typeof src === 'number') return src;
  if (typeof src === 'string' && src.startsWith('basic_clothing_')) {
    const id = src.replace('basic_clothing_', '');
    const found = BASIC_CLOTHING_ITEMS.find(b => b.id === id);
    return found ? found.image : null;
  }
  return src.length > 0 ? src : null;
}

interface OutfitCollageDisplayProps {
  items: OutfitItem[];
  height?: number;
  /**
   * When true (cold / rainy / windy weather) the collage shows 4 tiles in a
   * 2×2 grid: outerwear + top + bottom + shoes. When false it shows 3 tiles
   * (top + bottom on the first row, shoes alone on the second). Shoes always
   * appear. Defaults to `true` so callers that don't pass weather still get
   * the richer layout.
   */
  needsOuterwear?: boolean;
}

const CollageImage: React.FC<{ src: string | number | undefined | null }> = ({ src }) => {
  const resolved = resolveClothingImage(src);
  if (resolved !== null) {
    return (
      <Image
        source={typeof resolved === 'number' ? resolved : { uri: resolved }}
        style={styles.collageSlotImage}
        resizeMode="contain"
      />
    );
  }
  return (
    <View style={styles.collageSlotPlaceholder}>
      <Ionicons name="shirt-outline" size={44} color="#C8D0DA" />
    </View>
  );
};

const OutfitCollageDisplay: React.FC<OutfitCollageDisplayProps> = ({
  items,
  height = 400,
  needsOuterwear = true,
}) => {
  const previewSlots = getOutfitPreviewSlots(items);

  if (previewSlots.length === 0) {
    const first = items[0];
    const firstResolved = resolveClothingImage(first?.image);
    if (!firstResolved) return null;
    return (
      <Image
        source={typeof firstResolved === 'number' ? firstResolved : { uri: firstResolved }}
        style={[styles.outfitImage, { height }]}
        resizeMode="cover"
      />
    );
  }

  // Pick the first item per canonical slot so the same piece never appears
  // twice in the collage (was the cause of "3 identical t-shirts" on Home).
  // We intentionally DO NOT fall back to leftover items when a canonical slot
  // is empty — filling "Shoes" with a random shirt is what produced the
  // duplicate `lower_body` tile in production.
  const pickFirst = (cat: 'outerwear' | 'top' | 'bottom' | 'shoes') =>
    previewSlots.find((s) => s.macroCategory === cat);

  const outerSlot = pickFirst('outerwear');
  const topSlot = pickFirst('top');
  const bottomSlot = pickFirst('bottom');
  const shoesSlot = pickFirst('shoes');

  // Slots rendered in order. Shoes ALWAYS render — if we have no shoe item
  // we emit a `null` placeholder slot so the tile shows "Shoes" + placeholder
  // icon, honoring the "shoes are always present" UX rule.
  type RenderSlot = { item: OutfitItem | null; label: string; key: string };
  const orderedSlots: RenderSlot[] = [];

  if (needsOuterwear && outerSlot) {
    orderedSlots.push({ item: outerSlot.item, label: 'Outerwear', key: 'outerwear' });
  }
  if (topSlot) {
    orderedSlots.push({ item: topSlot.item, label: 'Top', key: 'top' });
  }
  if (bottomSlot) {
    orderedSlots.push({ item: bottomSlot.item, label: 'Bottom', key: 'bottom' });
  }
  orderedSlots.push({
    item: shoesSlot ? shoesSlot.item : null,
    label: 'Shoes',
    key: 'shoes',
  });

  // 2-per-row layout. Cold → 4 tiles in 2×2. Warm → 3 tiles (top+bottom on
  // row 1, shoes alone on row 2).
  const rows: (typeof orderedSlots)[] = [];
  for (let i = 0; i < orderedSlots.length; i += 2) {
    rows.push(orderedSlots.slice(i, i + 2));
  }

  return (
    <View style={[styles.outfitCollage, { height }] }>
      {rows.map((row, rowIndex) => (
        <View
          key={`row_${rowIndex}`}
          style={[
            styles.collageRow,
            rowIndex < rows.length - 1 ? styles.collageRowSpacing : null,
          ]}
        >
          {row.map((slot, columnIndex) => (
            <View
              key={`${slot.key}_${String(slot.item?.id ?? `${rowIndex}_${columnIndex}`)}`}
              style={[
                styles.collageSlot,
                columnIndex === 0 && row.length === 2 ? styles.collageSlotRightSpacing : null,
              ]}
            >
              <CollageImage src={slot.item ? slot.item.image : null} />
              <View style={styles.collageSlotLabel}>
                <Text style={styles.collageSlotLabelText} numberOfLines={1}>
                  {slot.item ? getOutfitPreviewTitle(slot.item) : slot.label}
                </Text>
              </View>
            </View>
          ))}
          {row.length === 1 ? <View style={styles.collageSlotSpacer} /> : null}
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  outfitImage: {
    width: '100%',
    height: 400,
  },
  outfitCollage: {
    width: '100%',
    height: 400,
    backgroundColor: '#F7F8FA',
    overflow: 'hidden',
    padding: 12,
  },
  collageRow: {
    flex: 1,
    flexDirection: 'row',
  },
  collageRowSpacing: {
    marginBottom: 12,
  },
  collageSlot: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    backgroundColor: '#F3F6FA',
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(15, 23, 42, 0.08)',
  },
  collageSlotRightSpacing: {
    marginRight: 12,
  },
  // Invisible filler keeping a lone tile (e.g. shoes on warm-weather days)
  // at the same width as the tiles in full rows.
  collageSlotSpacer: {
    flex: 1,
    marginLeft: 12,
  },
  collageSlotImage: {
    width: '78%',
    height: '78%',
  },
  collageSlotPlaceholder: {
    width: '85%',
    height: '85%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  collageSlotLabel: {
    position: 'absolute',
    bottom: 10,
    left: 12,
    backgroundColor: 'rgba(10,25,49,0.62)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
  },
  collageSlotLabelText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.4,
    maxWidth: 120,
  },
});

export default OutfitCollageDisplay;
