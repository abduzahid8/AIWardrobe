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

const OutfitCollageDisplay: React.FC<OutfitCollageDisplayProps> = ({ items }) => {
  const previewSlots = getOutfitPreviewSlots(items).slice(0, 4);

  if (previewSlots.length === 0) {
    const first = items[0];
    const firstResolved = resolveClothingImage(first?.image);
    if (!firstResolved) return null;
    return (
      <Image
        source={typeof firstResolved === 'number' ? firstResolved : { uri: firstResolved }}
        style={styles.outfitImage}
        resizeMode="cover"
      />
    );
  }

  const rows = [];
  for (let index = 0; index < previewSlots.length; index += 2) {
    rows.push(previewSlots.slice(index, index + 2));
  }

  return (
    <View style={styles.outfitCollage}>
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
              key={String(slot.item.id || `${rowIndex}_${columnIndex}`)}
              style={[
                styles.collageSlot,
                columnIndex === 0 ? styles.collageSlotRightSpacing : null,
              ]}
            >
              <CollageImage src={slot.item.image} />
              <View style={styles.collageSlotLabel}>
                <Text style={styles.collageSlotLabelText} numberOfLines={1}>
                  {getOutfitPreviewTitle(slot.item)}
                </Text>
              </View>
            </View>
          ))}
          {row.length === 1 ? <View style={styles.collageSlot} /> : null}
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
