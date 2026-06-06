/**
 * CalendarGrid — Pure visual calendar grid with day press callback.
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { useTranslation } from 'react-i18next';
import { ScaledText } from '../../../components/ui/ScaledText';
import { colors, spacing } from '../../../src/theme';
import {
    getDaysInMonth,
    getFirstDayOfMonth,
    formatDate,
    getOccasionColor,
    type OutfitLog,
} from '../hooks/useOutfitCalendar';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const DAY_SIZE = Math.max(30, (SCREEN_WIDTH - spacing.l * 2 - spacing.xs * 12) / 7);

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
    const { t } = useTranslation();

    const MONTHS = [
        t('calendar.jan'), t('calendar.feb'), t('calendar.mar'), t('calendar.apr'),
        t('calendar.may'), t('calendar.jun'), t('calendar.jul'), t('calendar.aug'),
        t('calendar.sep'), t('calendar.oct'), t('calendar.nov'), t('calendar.dec'),
    ];
    const WEEKDAYS = [
        t('calendar.sun'), t('calendar.mon'), t('calendar.tue'),
        t('calendar.wed'), t('calendar.thu'), t('calendar.fri'), t('calendar.sat'),
    ];

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
                    <ScaledText style={styles.monthTitle} minScale={0.65}>
                        {MONTHS[currentMonth]} {currentYear}
                    </ScaledText>
                    <ScaledText style={styles.monthStats} minScale={0.6}>
                        {t('calendar.loggedDays', { logged: monthlyStats.logged, total: monthlyStats.total })}
                    </ScaledText>
                </View>
                <TouchableOpacity onPress={onNextMonth} style={styles.navButton}>
                    <Text style={styles.navText}>›</Text>
                </TouchableOpacity>
            </View>

            {/* Weekday Headers */}
            <View style={styles.weekdayRow}>
                {WEEKDAYS.map((day, idx) => (
                    <View key={idx} style={styles.weekdayCell}>
                        <ScaledText style={styles.weekdayText} minScale={0.55}>
                            {day}
                        </ScaledText>
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
        marginBottom: spacing.s,
    },
    navButton: { padding: spacing.xs, minWidth: 36, alignItems: 'center' },
    navText: { fontSize: 24, color: colors.text.primary, lineHeight: 28 },
    monthInfo: { alignItems: 'center', flex: 1, paddingHorizontal: spacing.xs },
    monthTitle: { fontSize: 16, fontWeight: '700', color: colors.text.primary },
    monthStats: { fontSize: 11, color: colors.text.secondary, marginTop: 1 },
    weekdayRow: { flexDirection: 'row', marginBottom: spacing.xs },
    weekdayCell: { width: DAY_SIZE, alignItems: 'center', marginHorizontal: spacing.xs },
    weekdayText: { fontSize: 10, fontWeight: '600', color: colors.text.secondary },
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
    dayText: { fontSize: 13, fontWeight: '500', color: colors.text.primary },
    todayText: { fontWeight: '700', color: '#FFF' },
    futureDayText: { color: colors.text.muted },
    outfitDot: { width: 4, height: 4, borderRadius: 2, marginTop: 1.5 },
});

export default CalendarGrid;
