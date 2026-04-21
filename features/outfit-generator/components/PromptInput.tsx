import React from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';

interface PromptInputProps {
  value: string;
  onChangeText: (text: string) => void;
  onSubmit?: () => void;
}

const PromptInput: React.FC<PromptInputProps> = ({ value, onChangeText, onSubmit }) => (
  <View style={styles.promptContainer}>
    <Ionicons name="sparkles-outline" size={18} color="#6B7280" style={{ marginRight: 10 }} />
    <TextInput
      style={styles.promptInput}
      placeholder="Describe the vibe... (e.g. beach trip, business meeting)"
      placeholderTextColor="#9CA3AF"
      value={value}
      onChangeText={onChangeText}
      returnKeyType={onSubmit ? 'send' : 'done'}
      onSubmitEditing={onSubmit}
      multiline={false}
      maxLength={120}
    />
    {value.length > 0 && (
      <TouchableOpacity onPress={() => onChangeText('')} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }} style={{ marginRight: onSubmit ? 6 : 0 }}>
        <Ionicons name="close-circle" size={18} color="#9CA3AF" />
      </TouchableOpacity>
    )}
    {value.length > 0 && onSubmit && (
      <TouchableOpacity onPress={onSubmit} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
        <Ionicons name="arrow-up-circle" size={26} color="#0A1931" />
      </TouchableOpacity>
    )}
  </View>
);

const styles = StyleSheet.create({
  promptContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 20,
    marginBottom: 20,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: 'rgba(255,255,255,0.75)',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: 'rgba(200,200,210,0.6)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.surface),
  },
  promptInput: {
    flex: 1,
    fontSize: 14,
    color: '#1a1a2e',
    paddingVertical: 0,
  },
});

export default PromptInput;
