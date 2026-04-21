import React from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { AIStyle } from '../types';

interface VibeCardProps {
  style: AIStyle;
  isLoading: boolean;
  isSelected: boolean;
  isAnyLoading: boolean;
  onPress: () => void;
}

const VibeCard: React.FC<VibeCardProps> = ({
  style: styleObj,
  isLoading,
  isSelected,
  isAnyLoading,
  onPress,
}) => {
  const isThisLoading = isLoading && isSelected;

  return (
    <TouchableOpacity
      activeOpacity={0.8}
      disabled={isAnyLoading}
      onPress={onPress}
      style={[
        styles.vibeCard,
        isThisLoading && styles.vibeCardActive,
        isAnyLoading && !isSelected && styles.vibeCardDisabled,
      ]}
    >
      <View style={styles.vibeCardInner}>
        <View style={[styles.vibeIconWrap, isThisLoading && { backgroundColor: '#1a3a6e' }]}>
          <Ionicons name={styleObj.icon as any} size={24} color="#FFFFFF" />
        </View>
        <View style={{ flex: 1, marginLeft: 20 }}>
          <Text style={styles.vibeTitle}>{styleObj.label}</Text>
          <Text style={styles.vibeDesc}>{styleObj.desc}</Text>
        </View>
        <View style={{ paddingLeft: 12 }}>
          {isThisLoading
            ? <ActivityIndicator color="#0A1931" size="small" />
            : <Ionicons name="chevron-forward" size={24} color={isAnyLoading ? '#C0C0C0' : '#4B5563'} />}
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  vibeCard: {
    marginBottom: 16,
    borderRadius: 24,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    shadowColor: '#0A1931',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 12,
    elevation: 2,
  },
  vibeCardActive: {
    borderColor: '#0A1931',
    borderWidth: 2,
    backgroundColor: '#F0F4FF',
  },
  vibeCardDisabled: {
    opacity: 0.45,
  },
  vibeCardInner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    paddingRight: 24,
  },
  vibeIconWrap: {
    width: 64,
    height: 64,
    borderRadius: 18,
    backgroundColor: '#0A1931',
    alignItems: 'center',
    justifyContent: 'center',
  },
  vibeTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0A1931',
    marginBottom: 8,
    letterSpacing: -0.3,
  },
  vibeDesc: {
    fontSize: 14,
    color: '#4B5563',
    lineHeight: 22,
  },
});

export default VibeCard;
