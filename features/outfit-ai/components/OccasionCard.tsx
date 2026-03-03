/**
 * OccasionCard — Selectable occasion picker for outfit generation.
 */

import React from 'react';
import { TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, spacing, borderRadius, shadows } from '../../../src/theme';

interface OccasionCardProps {
    id: string;
    label: string;
    icon: string;
    selected: boolean;
    onPress: (id: string) => void;
}

export const OccasionCard: React.FC<OccasionCardProps> = ({
    id,
    label,
    icon,
    selected,
    onPress,
}) => (
    <TouchableOpacity
        style={[styles.card, selected && styles.selected]}
        onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress(id);
        }}
        activeOpacity={0.7}
    >
        <Text style={styles.icon}>{icon}</Text>
        <Text style={[styles.label, selected && styles.labelSelected]}>{label}</Text>
    </TouchableOpacity>
);

const styles = StyleSheet.create({
    card: {
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.l,
        backgroundColor: colors.surface,
        alignItems: 'center',
        gap: 4,
        borderWidth: 2,
        borderColor: 'transparent',
        ...shadows.soft,
    },
    selected: {
        borderColor: colors.button.primary,
        backgroundColor: `${colors.button.primary}10`,
    },
    icon: { fontSize: 24 },
    label: {
        fontSize: 12,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    labelSelected: {
        color: colors.button.primary,
    },
});

export default OccasionCard;
