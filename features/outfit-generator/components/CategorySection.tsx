import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, StyleSheet, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import LiquidGlass2026Theme, { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';
import type { WardrobeDisplayItem } from '../types';

const { width } = Dimensions.get('window');

interface CategorySectionProps {
  category: string;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  items: WardrobeDisplayItem[];
  selectedItemIds: Set<string>;
  onToggleItem: (id: string) => void;
}

const CategorySection: React.FC<CategorySectionProps> = ({
  label,
  icon,
  items,
  selectedItemIds,
  onToggleItem,
}) => {
  if (items.length === 0) return null;

  return (
    <View style={{ marginBottom: 24 }}>
      <View style={styles.categorySectionHeader}>
        <Ionicons name={icon} size={18} color={LiquidGlass2026Theme.colors.text.primary} />
        <Text style={styles.categorySectionTitle}>{label}</Text>
        <Text style={styles.categorySectionBadge}>Select 1</Text>
      </View>

      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={{ paddingHorizontal: 16, gap: 12 }}
      >
        {items.map((item) => {
          const isSelected = selectedItemIds.has(item.id);
          return (
            <TouchableOpacity
              key={item.id}
              onPress={() => onToggleItem(item.id)}
              activeOpacity={0.8}
            >
              <View style={[styles.categoryGridItem, isSelected && styles.gridItemActive]}>
                {item.image ? (
                  <Image
                    source={typeof item.image === 'number' ? item.image : { uri: item.image as string }}
                    style={styles.categoryGridItemImage}
                    resizeMode="contain"
                  />
                ) : (
                  <Ionicons name="shirt-outline" size={40} color={LiquidGlass2026Theme.colors.text.disabled} />
                )}
                {isSelected && (
                  <View style={styles.checkBadge}>
                    <Ionicons name="checkmark" size={16} color="#FFF" />
                  </View>
                )}
              </View>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  categorySectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    marginBottom: 12,
    gap: 8,
  },
  categorySectionTitle: {
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    flex: 1,
  },
  categorySectionBadge: {
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
    backgroundColor: 'rgba(255,255,255,0.7)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    overflow: 'hidden',
  },
  categoryGridItem: {
    width: (width - 80) / 3,
    height: (width - 80) / 3,
    backgroundColor: 'rgba(255,255,255,0.8)',
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  gridItemActive: {
    borderColor: LiquidGlass2026Theme.colors.accent.primary,
    borderWidth: 2,
    backgroundColor: 'rgba(20, 30, 50, 0.05)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  categoryGridItemImage: {
    width: '90%',
    height: '90%',
  },
  checkBadge: {
    position: 'absolute',
    top: 6,
    right: 6,
    backgroundColor: LiquidGlass2026Theme.colors.accent.primary,
    borderRadius: LiquidGlass2026Theme.radius.full,
    padding: 3,
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
});

export default CategorySection;
