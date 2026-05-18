import React, { useState } from 'react';
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
    if (found) return found.image;
    // BASIC_CLOTHING_ITEMS is empty — fall through so the string itself
    // isn't lost.  The caller will treat it as a URI and the <Image>
    // onError handler will swap to a placeholder if it can't load.
  }
  // Reject clearly empty / sentinel values but keep any other string
  // (including non-http URIs) so the Image component can attempt to
  // load it.  onError will catch failures.
  if (typeof src === 'string') {
    const trimmed = src.trim();
    if (trimmed.length === 0) return null;
    if (/^(null|undefined|none|false|0)$/i.test(trimmed)) return null;
    return trimmed;
  }
  return null;
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
  const [imgError, setImgError] = useState(false);
  const resolved = resolveClothingImage(src);

  // If the image previously failed to load, show placeholder immediately.
  if (imgError || resolved === null) {
    return (
      <View style={styles.collageSlotPlaceholder}>
        <Ionicons name="shirt-outline" size={44} color="#C8D0DA" />
      </View>
    );
  }

  return (
    <Image
      source={typeof resolved === 'number' ? resolved : { uri: resolved }}
      style={styles.collageSlotImage}
      resizeMode="contain"
      onError={() => {
        // Silently fall back to placeholder — the component already shows a
        // shirt-icon placeholder via setImgError(true). No warn needed since
        // __DEV__ is always true in Expo Go, making the warning appear in all
        // development sessions even though it's a handled, expected case.
        setImgError(true);
      }}
      onLoad={() => setImgError(false)}
    />
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

  // A layered/cold-weather outfit always renders 4 slots (Outerwear, Top,
  // Bottom, Shoes). A warm-weather outfit renders 3 slots (Top, Bottom,
  // Shoes). Missing items display a labelled placeholder tile with the
  // slot name so the user can see what the outfit was supposed to include.
  // This is preferable to silently dropping tiles, which produced cards
  // with only 1-2 tiles when the AI response was incomplete.
  type RenderSlot = { item: OutfitItem | null; label: string; key: string };
  const orderedSlots: RenderSlot[] = [];

  if (needsOuterwear) {
    orderedSlots.push({ item: outerSlot ? outerSlot.item : null, label: 'Outerwear', key: 'outerwear' });
  }
  orderedSlots.push({ item: topSlot ? topSlot.item : null, label: 'Top', key: 'top' });
  orderedSlots.push({ item: bottomSlot ? bottomSlot.item : null, label: 'Bottom', key: 'bottom' });
  orderedSlots.push({ item: shoesSlot ? shoesSlot.item : null, label: 'Shoes', key: 'shoes' });

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
              {!slot.item && (
                <View style={styles.collageSlotLabel}>
                  <Text style={styles.collageSlotLabelText} numberOfLines={1}>
                    {slot.label}
                  </Text>
                </View>
              )}
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
    width: '92%',
    height: '92%',
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
