/**
 * SelectableChip — Multi-purpose selectable chip for quiz options.
 */

import React from 'react';
import { TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AppColors from '../../../constants/AppColors';

interface SelectableChipProps {
    item: { id: string; name: string; emoji?: string; icon?: string; color?: string };
    selected: boolean;
    onPress: () => void;
    showColor?: boolean;
}

export const SelectableChip: React.FC<SelectableChipProps> = ({
    item,
    selected,
    onPress,
    showColor,
}) => (
    <TouchableOpacity
        style={[
            styles.chip,
            selected && styles.chipSelected,
            showColor && { borderColor: item.color },
        ]}
        onPress={onPress}
        activeOpacity={0.7}
    >
        {showColor && (
            <View style={[styles.colorDot, { backgroundColor: item.color }]} />
        )}
        {item.emoji && <Text style={styles.emoji}>{item.emoji}</Text>}
        {item.icon && (
            <Ionicons
                name={item.icon as any}
                size={20}
                color={selected ? AppColors.primary : AppColors.textSecondary}
            />
        )}
        <Text style={[styles.text, selected && styles.textSelected]}>
            {item.name}
        </Text>
        {selected && (
            <Ionicons name="checkmark-circle" size={18} color={AppColors.primary} />
        )}
    </TouchableOpacity>
);

const styles = StyleSheet.create({
    chip: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 14,
        backgroundColor: AppColors.surface,
        borderRadius: 14,
        borderWidth: 2,
        borderColor: AppColors.border,
        gap: 10,
    },
    chipSelected: {
        borderColor: AppColors.primary,
        backgroundColor: `${AppColors.primary}10`,
    },
    colorDot: {
        width: 20,
        height: 20,
        borderRadius: 10,
    },
    emoji: { fontSize: 20 },
    text: {
        flex: 1,
        fontSize: 16,
        color: AppColors.text,
    },
    textSelected: {
        fontWeight: '600',
        color: AppColors.primary,
    },
});

export default SelectableChip;
