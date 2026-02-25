/**
 * StyleInsightCard — Weekly style analytics card
 *
 * Displays insights from the retention service.
 * "You've worn 60% of your closet" / "Blue is your go-to" etc.
 */

import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import type { StyleInsight } from '../src/types/domain';

const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// Icon map by insight type
const INSIGHT_ICONS: Record<string, { name: string; color: string; bg: string }> = {
    utilization: { name: 'pie-chart-outline', color: '#3B82F6', bg: 'rgba(59,130,246,0.12)' },
    unworn_nudge: { name: 'sparkles-outline', color: '#F97316', bg: 'rgba(249,115,22,0.12)' },
    color_pattern: { name: 'color-palette-outline', color: '#8B5CF6', bg: 'rgba(139,92,246,0.12)' },
    variety: { name: 'calendar-outline', color: '#10B981', bg: 'rgba(16,185,129,0.12)' },
    streak: { name: 'flame-outline', color: '#EF4444', bg: 'rgba(239,68,68,0.12)' },
};

interface StyleInsightCardProps {
    insight: StyleInsight;
    onPress?: () => void;
}

const StyleInsightCard: React.FC<StyleInsightCardProps> = ({ insight, onPress }) => {
    const iconConfig = INSIGHT_ICONS[insight.type] || INSIGHT_ICONS.utilization;

    return (
        <TouchableOpacity
            style={styles.card}
            activeOpacity={onPress ? 0.7 : 1}
            onPress={onPress}
            disabled={!onPress}
        >
            <View style={[styles.iconContainer, { backgroundColor: iconConfig.bg }]}>
                <Ionicons
                    name={iconConfig.name as any}
                    size={22}
                    color={iconConfig.color}
                />
            </View>
            <View style={styles.content}>
                <Text style={styles.title} numberOfLines={1}>{insight.title}</Text>
                <Text style={styles.description} numberOfLines={2}>{insight.description}</Text>
            </View>
            {onPress && (
                <Ionicons
                    name="chevron-forward"
                    size={18}
                    color={colors.text.tertiary}
                />
            )}
        </TouchableOpacity>
    );
};

/**
 * Renders a list of style insight cards.
 */
export const StyleInsightsList: React.FC<{
    insights: StyleInsight[];
    maxVisible?: number;
}> = ({ insights, maxVisible = 3 }) => {
    const visible = insights.slice(0, maxVisible);

    if (visible.length === 0) return null;

    return (
        <View style={styles.list}>
            <Text style={styles.listTitle}>Style Insights</Text>
            {visible.map((insight, index) => (
                <StyleInsightCard key={`${insight.type}_${index}`} insight={insight} />
            ))}
        </View>
    );
};

const styles = StyleSheet.create({
    card: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.glass.frosted,
        padding: spacing.md,
        borderRadius: radius.lg,
        borderWidth: 1,
        borderColor: colors.border.glass,
        gap: spacing.md,
    },
    iconContainer: {
        width: 44,
        height: 44,
        borderRadius: 22,
        alignItems: 'center',
        justifyContent: 'center',
    },
    content: {
        flex: 1,
        gap: 2,
    },
    title: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '600',
    },
    description: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        lineHeight: 18,
    },
    list: {
        gap: spacing.sm,
    },
    listTitle: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '700',
        marginBottom: spacing.xs,
    },
});

export default StyleInsightCard;
