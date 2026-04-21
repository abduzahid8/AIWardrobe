import React from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { BlurView } from 'expo-blur';
import LiquidGlass2026Theme, { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';

interface OutfitFABProps {
  selectedCount: number;
  loading: boolean;
  onGenerate: () => void;
}

const OutfitFAB: React.FC<OutfitFABProps> = ({ selectedCount, loading, onGenerate }) => {
  if (selectedCount === 0) return null;

  return (
    <View style={styles.fabContainer}>
      <BlurView intensity={40} tint="light" style={styles.fabGlass}>
        <TouchableOpacity
          style={styles.generateBtn}
          onPress={onGenerate}
          disabled={loading || selectedCount === 0}
          activeOpacity={0.85}
        >
          {loading ? (
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <ActivityIndicator color="#fff" style={{ marginRight: 10 }} />
              <Text style={styles.generateBtnText}>AI is styling...</Text>
            </View>
          ) : (
            <>
              <Ionicons name="sparkles" size={20} color="#fff" style={{ marginRight: 8 }} />
              <Text style={styles.generateBtnText}>
                AI Generate ({selectedCount})
              </Text>
            </>
          )}
        </TouchableOpacity>
      </BlurView>
    </View>
  );
};

const styles = StyleSheet.create({
  fabContainer: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 34 : 24,
    left: 20,
    right: 20,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    overflow: 'hidden',
    ...SpatialElevation.getShadow(SpatialElevation.levels.floating),
  },
  fabGlass: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.4)',
  },
  generateBtn: {
    flexDirection: 'row',
    backgroundColor: LiquidGlass2026Theme.colors.accent.primary,
    height: 56,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    alignItems: 'center',
    justifyContent: 'center',
  },
  generateBtnText: {
    color: LiquidGlass2026Theme.colors.text.onDark,
    ...LiquidGlass2026Theme.typography.scale.titleMedium,
    letterSpacing: 0.2,
  },
});

export default OutfitFAB;
