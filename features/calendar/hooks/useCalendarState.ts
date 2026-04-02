/**
 * useCalendarState — manages calendar navigation and date selection
 */

import { useState } from 'react';
import { formatDate as _formatDate } from '../types';

// Re-export from single source of truth
export {
    type OutfitLog,
    type OutfitItem,
    type OccasionId,
    OCCASIONS,
    MONTHS,
    WEEKDAYS,
    getDaysInMonth,
    getFirstDayOfMonth,
    formatDate,
} from '../types';

export function useCalendarState() {
    const now = new Date();
    const [currentYear, setCurrentYear] = useState(now.getFullYear());
    const [currentMonth, setCurrentMonth] = useState(now.getMonth());
    const [selectedDay, setSelectedDay] = useState<number | null>(null);


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
        formatDate: _formatDate,
        goToPrevMonth,
        goToNextMonth,
    };
}
