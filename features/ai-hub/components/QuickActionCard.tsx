/**
 * QuickActionCard — Hub quick action button with icon and label.
 */

import React from 'react';
import { TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows } from '../../../src/theme';

interface QuickActionCardProps {
    icon: string;
    label: string;
    description?: string;
    color: string;
    bgColor: string;
    onPress: () => void;
}

export const QuickActionCard: React.FC<QuickActionCardProps> = ({
    icon,
    label,
    description,
    color,
    bgColor,
    onPress,
}) => (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
        <View style={[styles.iconContainer, { backgroundColor: bgColor }]}>
            <Ionicons name={icon as any} size={24} color={color} />
        </View>
        <View style={styles.textContainer}>
            <Text style={styles.label}>{label}</Text>
            {description && <Text style={styles.description}>{description}</Text>}
        </View>
        <Ionicons name="chevron-forward" size={20} color={colors.text.muted} />
    </TouchableOpacity>
);

const styles = StyleSheet.create({
    card: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.surface,
        borderRadius: borderRadius.l,
        padding: spacing.m,
        gap: spacing.m,
        ...shadows.soft,
    },
    iconContainer: {
        width: 48,
        height: 48,
        borderRadius: 14,
        justifyContent: 'center',
        alignItems: 'center',
    },
    textContainer: {
        flex: 1,
    },
    label: {
        fontSize: 16,
        fontWeight: '600',
        color: colors.text.primary,
    },
    description: {
        fontSize: 13,
        color: colors.text.secondary,
        marginTop: 2,
    },
});

export default QuickActionCard;
