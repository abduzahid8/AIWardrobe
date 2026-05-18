import React from 'react';
import { TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import LiquidGlass2026Theme, { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';
import type { ClosetClothingItem } from '../types';

interface GenerateOutfitButtonProps {
  item: ClosetClothingItem;
  onPress: (item: ClosetClothingItem) => void;
  accessibilityLabel: string;
}

/**
 * A small floating action button rendered as a sparkle icon overlay
 * in the bottom-right corner of a clothing item card.
 *
 * - 44×44 pt touch target (iOS HIG compliant)
 * - Triggers medium haptic feedback on press
 * - Does not own generation state; delegates to parent via onPress
 */
const GenerateOutfitButton: React.FC<GenerateOutfitButtonProps> = ({
  item,
  onPress,
  accessibilityLabel,
}) => {
  const handlePress = async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    onPress(item);
  };

  return (
    <TouchableOpacity
      style={styles.button}
      onPress={handlePress}
      activeOpacity={0.8}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
    >
      <Ionicons
        name="sparkles"
        size={18}
        color={LiquidGlass2026Theme.colors.text.onDark}
      />
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    position: 'absolute',
    bottom: 6,
    right: 6,
    width: 44,
    height: 44,
    borderRadius: LiquidGlass2026Theme.radius.full,
    backgroundColor: LiquidGlass2026Theme.colors.accent.primary,
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
});

export default GenerateOutfitButton;
