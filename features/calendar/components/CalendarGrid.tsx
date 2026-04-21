/**
 * CalendarGrid — Pure visual calendar grid with day press callback.
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { colors, spacing } from '../../../src/theme';
import {
    WEEKDAYS,
    MONTHS,
    getDaysInMonth,
    getFirstDayOfMonth,
    formatDate,
    getOccasionColor,
    type OutfitLog,
} from '../hooks/useOutfitCalendar';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const DAY_SIZE = (SCREEN_WIDTH - spacing.l * 2 - spacing.xs * 12) / 7;

interface CalendarGridProps {
    currentMonth: number;
    currentYear: number;
    todayStr: string;
    today: Date;
    outfitLogs: Record<string, OutfitLog>;
    onDayPress: (day: number) => void;
    onPrevMonth: () => void;
    onNextMonth: () => void;
    monthlyStats: { logged: number; total: number };
}

export const CalendarGrid: React.FC<CalendarGridProps> = ({
    currentMonth,
    currentYear,
    todayStr,
    today,
    outfitLogs,
    onDayPress,
    onPrevMonth,
    onNextMonth,
    monthlyStats,
}) => {
    const daysInMonth = getDaysInMonth(currentYear, currentMonth);
    const firstDay = getFirstDayOfMonth(currentYear, currentMonth);

    const renderDays = () => {
        const days = [];

        for (let i = 0; i < firstDay; i++) {
            days.push(<View key={`empty-${i}`} style={styles.dayCell} />);
        }

        for (let day = 1; day <= daysInMonth; day++) {
            const dateStr = formatDate(currentYear, currentMonth, day);
            const log = outfitLogs[dateStr];
            const isToday = dateStr === todayStr;
            const isFuture = new Date(dateStr) > today;

            days.push(
                <TouchableOpacity
                    key={day}
                    style={[styles.dayCell, isToday && styles.todayCell]}
                    onPress={() => onDayPress(day)}
                    activeOpacity={0.7}
                >
                    <Text style={[
                        styles.dayText,
                        isToday && styles.todayText,
                        isFuture && styles.futureDayText,
                    ]}>
                        {day}
                    </Text>
                    {log && (
                        <View style={[styles.outfitDot, { backgroundColor: getOccasionColor(log.occasion) }]} />
                    )}
                </TouchableOpacity>
            );
        }

        return days;
    };

    return (
        <View style={styles.calendarCard}>
            {/* Month Header */}
            <View style={styles.monthHeader}>
                <TouchableOpacity onPress={onPrevMonth} style={styles.navButton}>
                    <Text style={styles.navText}>‹</Text>
                </TouchableOpacity>
                <View style={styles.monthInfo}>
                    <Text style={styles.monthTitle}>
                        {MONTHS[currentMonth]} {currentYear}
                    </Text>
                    <Text style={styles.monthStats}>
                        {monthlyStats.logged}/{monthlyStats.total} days logged
                    </Text>
                </View>
                <TouchableOpacity onPress={onNextMonth} style={styles.navButton}>
                    <Text style={styles.navText}>›</Text>
                </TouchableOpacity>
            </View>

            {/* Weekday Headers */}
            <View style={styles.weekdayRow}>
                {WEEKDAYS.map((day, idx) => (
                    <View key={idx} style={styles.weekdayCell}>
                        <Text style={styles.weekdayText}>{day}</Text>
                    </View>
                ))}
            </View>

            {/* Calendar Grid */}
            <View style={styles.calendarGrid}>
                {renderDays()}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    calendarCard: {
        marginHorizontal: spacing.l,
        backgroundColor: colors.surface,
        borderRadius: 20,
        padding: spacing.m,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 8,
        elevation: 3,
    },
    monthHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: spacing.m,
    },
    navButton: { padding: spacing.xs },
    navText: { fontSize: 28, color: colors.text.primary },
    monthInfo: { alignItems: 'center' },
    monthTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary },
    monthStats: { fontSize: 12, color: colors.text.secondary, marginTop: 2 },
    weekdayRow: { flexDirection: 'row', marginBottom: spacing.xs },
    weekdayCell: { width: DAY_SIZE, alignItems: 'center', marginHorizontal: spacing.xs },
    weekdayText: { fontSize: 12, fontWeight: '600', color: colors.text.secondary },
    calendarGrid: { flexDirection: 'row', flexWrap: 'wrap' },
    dayCell: {
        width: DAY_SIZE,
        height: DAY_SIZE + 4,
        alignItems: 'center',
        justifyContent: 'center',
        marginHorizontal: spacing.xs,
        marginBottom: spacing.xs,
    },
    todayCell: {
        backgroundColor: colors.button.primary,
        borderRadius: DAY_SIZE / 2,
    },
    dayText: { fontSize: 14, fontWeight: '500', color: colors.text.primary },
    todayText: { fontWeight: '700', color: '#FFF' },
    futureDayText: { color: colors.text.muted },
    outfitDot: { width: 5, height: 5, borderRadius: 2.5, marginTop: 2 },
});

export default CalendarGrid;
