/**
 * useCalendarState — manages calendar navigation and date selection
 */

import { useState } from 'react';

export const getDaysInMonth = (year: number, month: number) =>
    new Date(year, month + 1, 0).getDate();

export const getFirstDayOfMonth = (year: number, month: number) =>
    new Date(year, month, 1).getDay();

export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

export const OCCASIONS = [
    { id: 'work', label: 'Work', icon: '💼', color: '#3B82F6' },
    { id: 'casual', label: 'Casual', icon: '👕', color: '#10B981' },
    { id: 'date', label: 'Date', icon: '❤️', color: '#EF4444' },
    { id: 'party', label: 'Party', icon: '🎉', color: '#F59E0B' },
    { id: 'sport', label: 'Sport', icon: '🏃', color: '#8B5CF6' },
    { id: 'formal', label: 'Formal', icon: '🎩', color: '#1A1A1A' },
];

export interface OutfitLog {
    date: string;
    items: Array<{ id: string; type: string; image: string; color?: string }>;
    occasion: string;
    note?: string;
    rating?: number;
}

export function useCalendarState() {
    const now = new Date();
    const [currentYear, setCurrentYear] = useState(now.getFullYear());
    const [currentMonth, setCurrentMonth] = useState(now.getMonth());
    const [selectedDay, setSelectedDay] = useState<number | null>(null);

    const formatDate = (year: number, month: number, day: number) =>
        `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

    const goToPrevMonth = () => {
        if (currentMonth === 0) {
            setCurrentMonth(11);
            setCurrentYear(currentYear - 1);
        } else {
            setCurrentMonth(currentMonth - 1);
        }
        setSelectedDay(null);
    };

    const goToNextMonth = () => {
        if (currentMonth === 11) {
            setCurrentMonth(0);
            setCurrentYear(currentYear + 1);
        } else {
            setCurrentMonth(currentMonth + 1);
        }
        setSelectedDay(null);
    };

    return {
        currentYear,
        currentMonth,
        selectedDay,
        setSelectedDay,
        formatDate,
        goToPrevMonth,
        goToNextMonth,
    };
}
