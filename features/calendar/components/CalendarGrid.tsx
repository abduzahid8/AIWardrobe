/**
 * CalendarGrid — 2050 Futuristic Style
 * Glassmorphism, neon accents, day-by-day outfit visualization
 */

import React from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import {
    WEEKDAYS,
    getDaysInMonth,
    getFirstDayOfMonth,
    formatDate,
    type OutfitLog,
    matchesCategory,
} from '../types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const HORIZONTAL_PADDING = 16;
const DAY_GAP = 10;
const DAYS_PER_ROW = 7;
const DAY_CELL_WIDTH = (SCREEN_WIDTH - HORIZONTAL_PADDING * 2 - DAY_GAP * (DAYS_PER_ROW - 1)) / DAYS_PER_ROW;
const SLOT_HEIGHT = (DAY_CELL_WIDTH - 20) / 3;

interface CalendarGridProps {
    currentMonth: number;
    currentYear: number;
    todayStr: string;
    today: Date;
    outfitLogs: Record<string, OutfitLog>;
    selectedDate: string | null;
    streak: number;
    onDayPress: (day: number) => void;
    onPrevMonth: () => void;
    onNextMonth: () => void;
    onClose?: () => void;
    onShare?: () => void;
}

export const CalendarGrid: React.FC<CalendarGridProps> = ({
    currentMonth,
    currentYear,
    todayStr,
    today,
    outfitLogs,
    selectedDate,
    streak,
    onDayPress,
    onPrevMonth,
    onNextMonth,
    onClose,
    onShare,
}) => {
    const daysInMonth = getDaysInMonth(currentYear, currentMonth);
    const firstDay = getFirstDayOfMonth(currentYear, currentMonth);

    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const monthYearText = `${monthNames[currentMonth]} ${currentYear}`;

    const renderDays = () => {
        const days = [];

        for (let i = 0; i < firstDay; i++) {
            days.push(
                <View key={`empty-${i}`} style={[styles.dayCell, styles.emptyDay]} />
            );
        }

        for (let day = 1; day <= daysInMonth; day++) {
            const dateStr = formatDate(currentYear, currentMonth, day);
            const log = outfitLogs[dateStr];
            const isToday = dateStr === todayStr;
            const isFuture = new Date(dateStr) > today;
            const isSelected = selectedDate === dateStr;
            const hasOutfit = log && log.items.length > 0;

            days.push(
                <Animated.View
                    key={`day-${dateStr}`}
                    entering={FadeInUp.delay(day * 20).duration(300)}
                    style={[
                        styles.dayCell,
                        isSelected && styles.selectedDayCell,
                        hasOutfit && styles.dayCellWithOutfit,
                    ]}
                >
                    <TouchableOpacity
                        style={styles.dayButton}
                        onPress={() => onDayPress(day)}
                        activeOpacity={0.8}
                    >
                        <View style={[
                            styles.dayNumberContainer,
                            isToday && styles.todayNumberContainer,
                            isSelected && styles.selectedNumberContainer,
                        ]}>
                            <Text style={[
                                styles.dayText,
                                isToday && styles.todayText,
                                isSelected && styles.selectedText,
                                isFuture && styles.futureDayText,
                            ]}>
                                {day}
                            </Text>
                        </View>

                        <View style={styles.slotsContainer}>
                            {[
                                { type: 'top', icon: '👕', label: 'Top' },
                                { type: 'pants', icon: '👖', label: 'Pants' },
                                { type: 'shoes', icon: '👟', label: 'Shoes' },
                            ].map(({ type, icon, label }, idx) => {
                                const item = log?.items.find((i) =>
                                    matchesCategory(i.type, type as 'top' | 'pants' | 'shoes')
                                );

                                return (
                                    <View
                                        key={`${dateStr}-${type}-${item?.id || idx}`}
                                        style={[
                                            styles.slot,
                                            item && styles.slotFilled,
                                            isSelected && styles.slotSelected,
                                        ]}
                                    >
                                        {item ? (
                                            <Image
                                                source={{ uri: item.image }}
                                                style={styles.slotImage}
                                                resizeMode="cover"
                                            />
                                        ) : (
                                            <View style={styles.slotEmpty}>
                                                <Text style={styles.slotIcon}>{icon}</Text>
                                                <Text style={styles.slotLabel}>{label}</Text>
                                            </View>
                                        )}
                                    </View>
                                );
                            })}
                        </View>

                        {hasOutfit && (
                            <View style={styles.outfitIndicator}>
                                <View style={[styles.indicatorDot, isSelected && styles.indicatorDotSelected]} />
                            </View>
                        )}
                    </TouchableOpacity>
                </Animated.View>
            );
        }

        return days;
    };

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={onPrevMonth} style={styles.navButton}>
                    <Ionicons name="chevron-back" size={28} color="#64748B" />
                </TouchableOpacity>

                <Text style={styles.monthTitle}>{monthYearText}</Text>

                <TouchableOpacity onPress={onNextMonth} style={styles.navButton}>
                    <Ionicons name="chevron-forward" size={28} color="#64748B" />
                </TouchableOpacity>
            </View>

            <View style={styles.weekdayRow}>
                {WEEKDAYS.map((w) => (
                    <View key={`weekday-${w}`} style={styles.weekdayCell}>
                        <Text style={styles.weekdayText}>{w.charAt(0)}</Text>
                    </View>
                ))}
            </View>

            <View style={styles.calendarGrid}>
                {renderDays()}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
        paddingHorizontal: HORIZONTAL_PADDING,
        paddingTop: 20,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 24,
        gap: 32,
    },
    navButton: {
        width: 48,
        height: 48,
        borderRadius: 14,
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
    },
    monthTitle: {
        fontSize: 28,
        fontWeight: '800',
        color: '#94A3B8',
        letterSpacing: -0.5,
    },
    weekdayRow: {
        flexDirection: 'row',
        marginBottom: 12,
        paddingHorizontal: 4,
    },
    weekdayCell: {
        width: DAY_CELL_WIDTH,
        alignItems: 'center',
    },
    weekdayText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#CBD5E1',
    },
    calendarGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: DAY_GAP,
    },
    dayCell: {
        width: DAY_CELL_WIDTH,
        minHeight: DAY_CELL_WIDTH * 1.4,
        backgroundColor: '#FAFBFC',
        borderRadius: 24,
        padding: 10,
        borderWidth: 2,
        borderColor: 'transparent',
    },
    emptyDay: {
        backgroundColor: 'transparent',
    },
    selectedDayCell: {
        backgroundColor: '#0F172A',
        borderColor: '#0F172A',
        shadowColor: '#0F172A',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 12,
        elevation: 4,
    },
    dayCellWithOutfit: {
        borderColor: '#E2E8F0',
    },
    dayButton: {
        alignItems: 'center',
    },
    dayNumberContainer: {
        width: 50,
        height: 50,
        borderRadius: 25,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 10,
    },
    todayNumberContainer: {
        backgroundColor: '#0F172A',
    },
    selectedNumberContainer: {
        backgroundColor: 'transparent',
    },
    dayText: {
        fontSize: 20,
        fontWeight: '800',
        color: '#0F172A',
    },
    todayText: {
        color: '#FFFFFF',
    },
    selectedText: {
        color: '#FFFFFF',
    },
    futureDayText: {
        color: '#CBD5E1',
    },
    slotsContainer: {
        width: '100%',
        gap: 6,
        marginTop: 8,
    },
    slot: {
        width: '100%',
        height: SLOT_HEIGHT * 1.4,
        borderRadius: 14,
        overflow: 'hidden',
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: '#E2E8F0',
    },
    slotFilled: {
        backgroundColor: '#FFFFFF',
        borderColor: '#6366F1',
        borderWidth: 3,
    },
    slotSelected: {
        borderColor: 'rgba(255,255,255,0.8)',
    },
    slotEmpty: {
        width: '100%',
        height: '100%',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 2,
    },
    slotIcon: {
        fontSize: 20,
    },
    slotLabel: {
        fontSize: 10,
        fontWeight: '600',
        color: '#94A3B8',
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    slotImage: {
        width: '100%',
        height: '100%',
    },
    outfitIndicator: {
        marginTop: 4,
        alignItems: 'center',
    },
    indicatorDot: {
        width: 4,
        height: 4,
        borderRadius: 2,
        backgroundColor: '#6366F1',
    },
    indicatorDotSelected: {
        backgroundColor: '#FFFFFF',
    },
});

export default CalendarGrid;
