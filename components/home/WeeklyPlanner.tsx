import React from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';

const { colors, spacing, typography } = LiquidGlass2026Theme;

interface Props {
    itemCount: number;
    isReducedMotionEnabled: boolean;
}

export default function WeeklyPlanner({ itemCount, isReducedMotionEnabled }: Props) {
    const { t } = useTranslation();

    if (itemCount < 8) return null;

    const DAY_NAMES = [
        t('weeklyPlanner.days.0'),
        t('weeklyPlanner.days.1'),
        t('weeklyPlanner.days.2'),
        t('weeklyPlanner.days.3'),
        t('weeklyPlanner.days.4'),
        t('weeklyPlanner.days.5'),
        t('weeklyPlanner.days.6'),
    ];

    const now = new Date();
    const startOfWeek = new Date(now);
    startOfWeek.setDate(now.getDate() - now.getDay());

    const days = Array.from({ length: 7 }, (_, i) => {
        const d = new Date(startOfWeek);
        d.setDate(startOfWeek.getDate() + i);
        return {
            name: DAY_NAMES[d.getDay()],
            date: d.getDate(),
            isToday: d.toDateString() === now.toDateString(),
        };
    });

    return (
        <Animated.View
            entering={isReducedMotionEnabled ? undefined : FadeInDown.delay(180).duration(400)}
            style={styles.section}
        >
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.scroll}>
                {days.map((day, idx) => (
                    <View key={idx} style={[styles.dayItem, day.isToday && styles.dayToday]}>
                        <Text style={styles.dayName}>{day.name}</Text>
                        <View style={[styles.dateCircle, day.isToday && styles.dateCircleToday]}>
                            <Text style={[styles.dateText, day.isToday && styles.dateTextToday]}>{day.date}</Text>
                        </View>
                        <View style={styles.outfitContainer}>
                            <MaterialCommunityIcons name={"tshirt-crew" as any} size={14} color={day.isToday ? colors.text.primary : colors.text.tertiary} style={{ opacity: 0.3 }} />
                            <MaterialCommunityIcons name={"hanger" as any} size={14} color={day.isToday ? colors.text.primary : colors.text.tertiary} style={{ opacity: 0.3, marginTop: -2 }} />
                            <MaterialCommunityIcons name={"shoe-sneaker" as any} size={14} color={day.isToday ? colors.text.primary : colors.text.tertiary} style={{ opacity: 0.3, marginTop: -2 }} />
                        </View>
                        {day.isToday && <View style={styles.indicator} />}
                    </View>
                ))}
            </ScrollView>
        </Animated.View>
    );
}

const styles = StyleSheet.create({
    section: { marginBottom: spacing.md },
    scroll: { paddingHorizontal: spacing.screenPadding, gap: 16 },
    dayItem: { alignItems: 'center', width: 48, position: 'relative' },
    dayToday: {},
    dayName: {
        ...typography.scale.labelSmall,
        fontSize: 10,
        color: colors.text.tertiary,
        marginBottom: 4,
    },
    dateCircle: {
        width: 28, height: 28, borderRadius: 14,
        justifyContent: 'center', alignItems: 'center', marginBottom: 8,
    },
    dateCircleToday: { backgroundColor: '#F5F5F5' },
    dateText: {
        ...typography.scale.bodySmall,
        fontSize: 14, fontWeight: '700', color: colors.text.primary,
    },
    dateTextToday: { color: colors.text.primary },
    outfitContainer: { marginTop: 4, height: 48, justifyContent: 'center', alignItems: 'center' },
    indicator: {
        position: 'absolute', bottom: -12, height: 3, width: '100%',
        backgroundColor: colors.text.primary, borderRadius: 1.5,
    },
});
