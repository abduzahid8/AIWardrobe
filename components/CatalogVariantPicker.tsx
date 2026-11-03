import React from 'react';
import { FlatList, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { useTranslation } from 'react-i18next';
import type { ShopCatalogItem } from '../src/types/shop';
import { catalogTitleColor } from '../src/utils/catalogIdentity';
import { ScaledText } from './ui/ScaledText';

/** Exact variants and native observed prices. Unknown colors get thumbnails and
 * numbered labels, never invented color names or hexadecimal swatches. */
export function CatalogVariantPicker({
  variants,
  selectedId,
  onSelect,
}: {
  variants: readonly ShopCatalogItem[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  if (variants.length < 2) return null;
  return (
    <FlatList
      horizontal
      style={styles.list}
      data={variants}
      extraData={selectedId}
      initialNumToRender={4}
      maxToRenderPerBatch={4}
      windowSize={3}
      keyExtractor={(variant) => variant.id}
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.row}
      renderItem={({ item: variant, index }) => {
        const color = catalogTitleColor(variant);
        const caption =
          color ||
          t('inspo.variantNumber', { defaultValue: 'Variant {{number}}', number: index + 1 });
        const name = color || `${variant.name} · ${caption}`;
        const label = `${name} · ${variant.price} ${variant.currency || ''}`;
        return (
          <TouchableOpacity
            accessibilityRole="button"
            accessibilityLabel={label}
            accessibilityState={{ selected: variant.id === selectedId }}
            onPress={() => onSelect(variant.id)}
            style={[styles.chip, variant.id === selectedId && styles.selected]}
          >
            {typeof variant.imageUrl === 'string' && (
              <Image
                source={{ uri: variant.imageUrl }}
                style={styles.image}
                resizeMode="contain"
                accessible={false}
              />
            )}
            <ScaledText style={styles.label} numberOfLines={1}>
              {caption}
            </ScaledText>
            <ScaledText style={styles.price}>
              {variant.price} {variant.currency}
            </ScaledText>
          </TouchableOpacity>
        );
      }}
    />
  );
}
const styles = StyleSheet.create({
  list: { height: 108, flexGrow: 0 },
  price: { fontSize: 11, color: '#40506B' },
  row: { gap: 8, paddingVertical: 8 },
  chip: {
    padding: 8,
    borderWidth: 1,
    borderRadius: 8,
    borderColor: '#CDD1D8',
    alignItems: 'center',
    maxWidth: 180,
  },
  selected: { borderColor: '#0A1931' },
  label: { color: '#0A1931', fontSize: 12 },
  image: { width: 36, height: 36, marginBottom: 4 },
});
