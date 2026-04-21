import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import LiquidGlass2026Theme, { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';

const { width } = Dimensions.get('window');

interface ModeToggleProps {
  activeMode: 'auto' | 'manual';
  onModeChange: (mode: 'auto' | 'manual') => void;
}

const ModeToggle: React.FC<ModeToggleProps> = ({ activeMode, onModeChange }) => (
  <View style={styles.viewToggleWrap}>
    <BlurView intensity={30} tint="light" style={StyleSheet.absoluteFill} />
    <TouchableOpacity
      style={[styles.viewToggleOption, activeMode === 'auto' && styles.viewToggleActive]}
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onModeChange('auto');
      }}
    >
      <Text style={[styles.viewToggleText, activeMode === 'auto' && styles.viewToggleTextActive]}>
        Auto-Stylist
      </Text>
    </TouchableOpacity>
  </View>
);

const styles = StyleSheet.create({
  viewToggleWrap: {
    flexDirection: 'row',
    alignSelf: 'center',
    borderRadius: 24,
    padding: 6,
    marginTop: 24,
    marginBottom: 8,
    overflow: 'hidden',
    backgroundColor: 'rgba(255,255,255,0.6)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.8)',
    zIndex: 10,
    width: width - 40,
  },
  viewToggleOption: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    borderRadius: 18,
  },
  viewToggleActive: {
    backgroundColor: '#fff',
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  viewToggleText: {
    fontSize: 15,
    fontWeight: '600',
    color: LiquidGlass2026Theme.colors.text.secondary,
  },
  viewToggleTextActive: {
    color: LiquidGlass2026Theme.colors.text.primary,
  },
});

export default ModeToggle;
